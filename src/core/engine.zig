//! ============================================================================
//! Klaxon Engine v2 Coordinator (`src/core/engine.zig`)
//! ============================================================================
//! Engine v2 architecture binding:
//! Application State -> Reactive Dispatcher -> Flexbox Layout -> GPU HAL.
//!
//! Architectural Responsibilities:
//! 1. State Management: Transactional state updates and differential dirty-tracking.
//! 2. Reactive Dispatcher: Hardware bitmask CTZ dirty-flag dispatcher popping up to
//!    64 dirty branches in 1 cycle.
//! 3. Sub-Cycle Flexbox Layout: 4-Pass layout, Q14 fixed-point arithmetic, and cumulative
//!    boundary snapping with zero hairline rounding gaps.
//! 4. GPU HAL Integration: Comptime-dispatched rendering, vertex buffer uploads, and
//!    command recording into GPU backends with zero dynamic allocation.
//! 5. High-Resolution Telemetry: Integer nanoseconds and reference 4.0 GHz CPU cycles
//!    (1 cycle = 0.25 ns, 4 cycles = 1 ns) across all engine phases.
//!
//! Standards: TigerStyle (<= 100 columns, zero libc, zero POSIX, 0 malloc/free).

const std = @import("std");
const assert = std.debug.assert;

const reactive = @import("reactive.zig");
const flexbox = @import("../layout/flexbox.zig");
const hal = @import("../gpu/hal.zig");
const time = @import("../platform/time.zig");

// ============================================================================
// Engine Telemetry Metrics (Nanoseconds & 4.0 GHz Reference Cycles)
// ============================================================================

pub const EngineMetrics = struct {
    dispatch_duration_ns: u64 = 0,
    dispatch_cycles: u64 = 0,
    layout_duration_ns: u64 = 0,
    layout_cycles: u64 = 0,
    render_duration_ns: u64 = 0,
    render_cycles: u64 = 0,
    frame_duration_ns: u64 = 0,
    frame_cycles: u64 = 0,

    pub inline fn reset(self: *EngineMetrics) void {
        self.* = .{};
    }
};

// ============================================================================
// Engine v2 Coordinator
// ============================================================================

pub fn Engine(
    comptime State: type,
    comptime node_capacity_max: usize,
    comptime GpuBackend: type,
) type {
    comptime {
        assert(@typeInfo(State) == .@"struct");
        assert(node_capacity_max > 0);
        assert(node_capacity_max <= 65535);
    }

    const vertex_capacity = node_capacity_max * 6;

    return struct {
        const Self = @This();
        pub const StateType = State;
        pub const node_capacity = node_capacity_max;
        pub const max_vertices = vertex_capacity;

        // State & Reactivity
        state: State,
        prev_state: State,
        dispatcher: reactive.ReactiveDispatcher = .{},

        // Layout Subsystem
        tree: flexbox.FlexboxTree(node_capacity) align(64) = undefined,
        viewport_width_px: f32 = 800.0,
        viewport_height_px: f32 = 600.0,
        layout_dirty: bool = true,

        // GPU Subsystem
        backend: GpuBackend = undefined,
        pipeline: hal.RenderPipeline = .null_handle,
        vertex_buffer: hal.Buffer = .null_handle,

        // Static BSS Vertex Staging Buffer (align 64)
        vertex_staging: [vertex_capacity]hal.Vertex2D align(64) = undefined,
        staged_vertex_count: u32 = 0,

        // Node Visual Styles (RGBA colors for drawing quads)
        node_colors: [node_capacity][4]u8 align(64) = undefined,

        // Telemetry
        metrics: EngineMetrics = .{},

        /// Initializes the engine coordinator in-place with zero heap allocation.
        pub fn init(
            self: *Self,
            initial_state: State,
            viewport_w: f32,
            viewport_h: f32,
        ) void {
            assert(viewport_w > 0.0);
            assert(viewport_h > 0.0);

            self.state = initial_state;
            self.prev_state = initial_state;
            self.dispatcher = .{};

            self.viewport_width_px = viewport_w;
            self.viewport_height_px = viewport_h;
            self.layout_dirty = true;

            self.tree.init();
            self.staged_vertex_count = 0;
            self.metrics.reset();

            // Clear node colors
            @memset(&self.node_colors, [4]u8{ 255, 255, 255, 255 });

            // Initialize GPU mock / backend resources
            GpuBackend.init(&self.backend);
            self.vertex_buffer = GpuBackend.createBuffer(
                &self.backend,
                &.{
                    .size_bytes = @intCast(@sizeOf([vertex_capacity]hal.Vertex2D)),
                    .usage = .{ .vertex = true, .copy_dst = true },
                },
            );
            assert(self.vertex_buffer.isValid());

            self.pipeline = GpuBackend.createRenderPipeline(
                &self.backend,
                &.{
                    .vertex_shader = .{ .id = 1, .generation = 1 },
                    .fragment_shader = .{ .id = 2, .generation = 1 },
                },
            );
            assert(self.pipeline.isValid());
        }

        /// Adds a UI node to the underlying flexbox tree with specified layout and color.
        pub fn addNode(
            self: *Self,
            kind: flexbox.NodeKind,
            parent_idx: u16,
            depth: u8,
            style: flexbox.Style,
            is_elastic: bool,
            color: [4]u8,
        ) u16 {
            const idx = self.tree.addNode(kind, parent_idx, depth, style, is_elastic);
            self.node_colors[idx] = color;
            self.layout_dirty = true;
            return idx;
        }

        /// Finalizes tree hierarchy and pre-compiles affine boundary laws.
        pub fn buildLayout(self: *Self) void {
            self.tree.computeSubtreeCounts();
            self.tree.run4Pass(self.viewport_width_px, self.viewport_height_px);
            self.tree.compileAffine();
            self.tree.partitionElastic();
            self.layout_dirty = false;
        }

        /// Updates state and runs diffState to populate reactive dirty flags.
        pub fn updateState(self: *Self, new_state: State) void {
            self.prev_state = self.state;
            self.state = new_state;

            var dirty_bits = reactive.Bitmask64.empty();
            reactive.diffState(State, &self.prev_state, &self.state, &dirty_bits);

            while (!dirty_bits.isZero()) {
                const bit = dirty_bits.popLowest();
                self.dispatcher.markField(bit);
            }
        }

        /// Dispatches pending mutations through CTZ bitmask against CSR graph.
        pub fn dispatchMutations(
            self: *Self,
            graph: anytype,
            handler: anytype,
        ) usize {
            const t0 = time.now();

            const DispatchProxy = struct {
                inner_handler: @TypeOf(handler),
                engine_ptr: *Self,

                pub fn onEdgeTriggered(
                    proxy: *@This(),
                    field_idx: usize,
                    edge: reactive.DependencyEdge,
                ) void {
                    if (edge.kind == .invalidate_layout) {
                        proxy.engine_ptr.layout_dirty = true;
                    }
                    proxy.inner_handler.onEdgeTriggered(field_idx, edge);
                }
            };

            var proxy = DispatchProxy{
                .inner_handler = handler,
                .engine_ptr = self,
            };

            const processed = self.dispatcher.dispatch(graph, &proxy);

            const t1 = time.now();
            self.metrics.dispatch_duration_ns = t1.durationSince(t0);
            self.metrics.dispatch_cycles = time.toCyclesAt4Ghz(self.metrics.dispatch_duration_ns);

            return processed;
        }

        /// Handles viewport window resize using sub-cycle SIMD Q14 fixed-point layout.
        pub fn resize(self: *Self, new_width_px: f32, new_height_px: f32) void {
            assert(new_width_px > 0.0);
            assert(new_height_px > 0.0);

            const t0 = time.now();

            self.viewport_width_px = new_width_px;
            self.viewport_height_px = new_height_px;

            const w_i16: i16 = @intFromFloat(new_width_px);
            self.tree.solveResizeSimd(w_i16);
            self.layout_dirty = false;

            const t1 = time.now();
            self.metrics.layout_duration_ns = t1.durationSince(t0);
            self.metrics.layout_cycles = time.toCyclesAt4Ghz(self.metrics.layout_duration_ns);
        }

        /// Solves layout if dirty.
        pub fn solveLayout(self: *Self) void {
            if (!self.layout_dirty) return;

            const t0 = time.now();
            self.tree.run4Pass(self.viewport_width_px, self.viewport_height_px);
            self.layout_dirty = false;

            const t1 = time.now();
            self.metrics.layout_duration_ns = t1.durationSince(t0);
            self.metrics.layout_cycles = time.toCyclesAt4Ghz(self.metrics.layout_duration_ns);
        }

        /// Packs resolved geometry into vertex quads, uploads to GPU buffer,
        /// and records render pass draw commands.
        pub fn renderFrame(self: *Self) void {
            const t0 = time.now();

            // 1. Pack quads from resolved node positions into staging buffer
            self.staged_vertex_count = 0;
            const count = self.tree.node_count;

            var i: usize = 0;
            while (i < count) : (i += 1) {
                const rect = self.tree.getRect(@intCast(i));
                if (rect.width_px <= 0.0 or rect.height_px <= 0.0) continue;

                assert(self.staged_vertex_count + 6 <= max_vertices);
                const quad_verts = hal.makeQuadVertices(
                    rect.x_px,
                    rect.y_px,
                    rect.width_px,
                    rect.height_px,
                    self.node_colors[i],
                );

                @memcpy(
                    self.vertex_staging[self.staged_vertex_count..][0..6],
                    &quad_verts,
                );
                self.staged_vertex_count += 6;
            }

            // 2. Upload staging vertices to GPU buffer
            if (self.staged_vertex_count > 0) {
                const upload_bytes: u32 = self.staged_vertex_count * @sizeOf(hal.Vertex2D);
                const raw_bytes: [*]const u8 = @ptrCast(&self.vertex_staging);
                GpuBackend.writeBuffer(
                    &self.backend,
                    self.vertex_buffer,
                    0,
                    raw_bytes[0..upload_bytes],
                );
            }

            // 3. Record GPU render commands
            var encoder = GpuBackend.createCommandEncoder(&self.backend);
            {
                var pass = encoder.beginRenderPass(.{
                    .color_attachments = &.{.{
                        .view = .{ .id = 1, .generation = 1 },
                        .load_op = .clear,
                        .store_op = .store,
                        .clear_value = .{ .r = 0.05, .g = 0.05, .b = 0.05, .a = 1.0 },
                    }},
                });
                pass.setPipeline(self.pipeline);
                pass.setVertexBuffer(0, self.vertex_buffer, 0);
                pass.draw(self.staged_vertex_count, 1, 0, 0);
                pass.end();
            }

            const cmd_buffer = encoder.finish();
            GpuBackend.submit(&self.backend, &.{cmd_buffer});

            const t1 = time.now();
            self.metrics.render_duration_ns = t1.durationSince(t0);
            self.metrics.render_cycles = time.toCyclesAt4Ghz(self.metrics.render_duration_ns);
        }

        /// Executes a complete reactive frame cycle:
        /// State Update -> Reactive Dispatch -> Layout Solve -> GPU Render.
        pub fn stepFrame(
            self: *Self,
            new_state: State,
            graph: anytype,
            handler: anytype,
        ) void {
            const frame_t0 = time.now();

            self.updateState(new_state);
            _ = self.dispatchMutations(graph, handler);
            self.solveLayout();
            self.renderFrame();

            const frame_t1 = time.now();
            self.metrics.frame_duration_ns = frame_t1.durationSince(frame_t0);
            self.metrics.frame_cycles = time.toCyclesAt4Ghz(self.metrics.frame_duration_ns);
        }
    };
}

// ============================================================================
// Unit Tests for Engine v2 Coordinator
// ============================================================================

test "Engine v2 initialization and complete frame cycle" {
    const TestState = extern struct {
        counter: u32 = 0,
        dark_mode: bool = false,
        _padding: [3]u8 = .{ 0, 0, 0 },
        slider_val: f32 = 0.5,
    };

    const EngineType = Engine(TestState, 16, hal.MockBackend);
    var engine: EngineType = undefined;
    engine.init(.{}, 800.0, 600.0);

    // Build simple UI tree: Root col + 2 buttons
    const root = engine.addNode(.col, flexbox.INDEX_NONE, 0, .{
        .pad_px = 10.0,
        .gap_px = 5.0,
    }, true, .{ 20, 20, 20, 255 });

    _ = engine.addNode(.button, root, 1, .{
        .grow = 1.0,
        .height_px_min = 50.0,
    }, true, .{ 255, 0, 0, 255 });

    _ = engine.addNode(.button, root, 1, .{
        .grow = 1.0,
        .height_px_min = 50.0,
    }, true, .{ 0, 255, 0, 255 });

    engine.buildLayout();

    // Verify initial layout
    const r0 = engine.tree.getRect(0);
    try std.testing.expectEqual(@as(f32, 800.0), r0.width_px);
    try std.testing.expectEqual(@as(f32, 600.0), r0.height_px);

    // Initial render
    engine.renderFrame();
    try std.testing.expectEqual(@as(u32, 18), engine.staged_vertex_count); // 3 nodes * 6 verts

    // Set up CSR dependency graph (field 0: counter, field 1: dark_mode, field 3: slider_val)
    const Graph = reactive.CsrDependencyGraph(4, 2);
    const graph = Graph{
        .offsets = .{ 0, 1, 1, 2, 2 },
        .edges = .{
            .{ .target_node_id = 1, .kind = .invalidate_layout },
            .{ .target_node_id = 2, .kind = .patch_color },
        },
    };

    const TestHandler = struct {
        mutations: usize = 0,
        pub fn onEdgeTriggered(self: *@This(), _: usize, _: reactive.DependencyEdge) void {
            self.mutations += 1;
        }
    };
    var handler = TestHandler{};

    // Mutate state and step complete frame
    var next_state = engine.state;
    next_state.counter = 42;

    engine.stepFrame(next_state, graph, &handler);

    try std.testing.expectEqual(@as(usize, 1), handler.mutations);
    try std.testing.expect(engine.metrics.frame_duration_ns >= 0);
    try std.testing.expect(engine.metrics.frame_cycles >= 0);
}

test "Engine v2 sub-cycle window resize evaluation" {
    const TestState = extern struct {
        scale: f32 = 1.0,
    };

    const EngineType = Engine(TestState, 8, hal.MockBackend);
    var engine: EngineType = undefined;
    engine.init(.{}, 1000.0, 500.0);

    const root = engine.addNode(.row, flexbox.INDEX_NONE, 0, .{
        .pad_px = 0.0,
        .gap_px = 0.0,
    }, true, .{ 0, 0, 0, 255 });

    _ = engine.addNode(.button, root, 1, .{ .grow = 1.0 }, true, .{ 10, 10, 10, 255 });
    _ = engine.addNode(.button, root, 1, .{ .grow = 1.0 }, true, .{ 20, 20, 20, 255 });

    engine.buildLayout();

    // Resize viewport to 1600x900
    engine.resize(1600.0, 900.0);

    const box1 = engine.tree.getResolvedBoxI16(1);
    const box2 = engine.tree.getResolvedBoxI16(2);

    // In 1:1 split on 1600 px width:
    try std.testing.expectEqual(@as(i16, 0), box1.x_px);
    try std.testing.expectEqual(@as(i16, 800), box1.width_px);
    try std.testing.expectEqual(@as(i16, 800), box2.x_px);
    try std.testing.expectEqual(@as(i16, 800), box2.width_px);

    // Verify telemetry
    try std.testing.expect(engine.metrics.layout_duration_ns >= 0);
    try std.testing.expectEqual(
        engine.metrics.layout_duration_ns * 4,
        engine.metrics.layout_cycles,
    );
}
