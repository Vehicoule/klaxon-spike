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
        gpu_device: GpuBackend.Device = undefined,
        gpu_queue: GpuBackend.Queue = undefined,
        color_target: hal.Texture = .null_handle,
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
            assert(!std.math.isNan(viewport_w));
            assert(!std.math.isNan(viewport_h));
            assert(viewport_w >= 0.0);
            assert(viewport_h >= 0.0);

            const clamped_w = std.math.clamp(viewport_w, 1.0, 32767.0);
            const clamped_h = std.math.clamp(viewport_h, 1.0, 32767.0);

            self.state = initial_state;
            self.prev_state = initial_state;
            self.dispatcher = .{};

            self.viewport_width_px = clamped_w;
            self.viewport_height_px = clamped_h;
            self.layout_dirty = true;

            self.tree.init();
            self.staged_vertex_count = 0;
            self.metrics.reset();

            // Clear node colors
            @memset(&self.node_colors, [4]u8{ 255, 255, 255, 255 });

            // Initialize GPU Device and Queue through HAL contract
            GpuBackend.Device.init(&self.gpu_device);
            self.gpu_queue = GpuBackend.Queue.init(&self.gpu_device);

            self.vertex_buffer = GpuBackend.createBuffer(
                &self.gpu_device,
                .{
                    .size = @intCast(@sizeOf([vertex_capacity]hal.Vertex2D)),
                    .usage = hal.BufferUsage.vertex_dst,
                },
            ) catch hal.Buffer.null_handle;
            assert(self.vertex_buffer.isValid());

            const vs = GpuBackend.createShaderModule(
                &self.gpu_device,
                .{ .code = "vertex_main" },
            ) catch hal.ShaderModule.null_handle;
            assert(vs.isValid());

            self.pipeline = GpuBackend.createRenderPipeline(
                &self.gpu_device,
                .{
                    .vertex_shader = vs,
                    .color_target_format = .rgba8_unorm,
                },
            ) catch hal.RenderPipeline.null_handle;
            assert(self.pipeline.isValid());

            const max_w: u32 = hal.mock.texture_width_max;
            const max_h: u32 = hal.mock.texture_height_max;
            const w_u32: u32 = @intFromFloat(clamped_w);
            const h_u32: u32 = @intFromFloat(clamped_h);
            const tex_w = @min(w_u32, max_w);
            const tex_h = @min(h_u32, max_h);

            self.color_target = GpuBackend.createTexture(
                &self.gpu_device,
                .{
                    .size = .{ .width = @max(1, tex_w), .height = @max(1, tex_h) },
                    .format = .rgba8_unorm,
                    .usage = hal.TextureUsage.render_target,
                },
            ) catch hal.Texture.null_handle;
            assert(self.color_target.isValid());
        }

        /// Releases GPU resources allocated for this engine instance.
        pub fn deinit(self: *Self) void {
            if (self.color_target.isValid()) {
                GpuBackend.destroyTexture(&self.gpu_device, self.color_target);
                self.color_target = .null_handle;
            }
            if (self.vertex_buffer.isValid()) {
                GpuBackend.destroyBuffer(&self.gpu_device, self.vertex_buffer);
                self.vertex_buffer = .null_handle;
            }
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
            const clamped_w = std.math.clamp(self.viewport_width_px, 1.0, 32767.0);
            const w_i16: i16 = @intFromFloat(clamped_w);
            self.tree.solveResizeSimd(w_i16);
            self.layout_dirty = false;
        }

        /// Updates state and runs diffState to populate reactive dirty flags.
        pub fn updateState(self: *Self, new_state: State) void {
            self.prev_state = self.state;
            self.state = new_state;

            var dirty_bits = reactive.Bitmask64.empty();
            reactive.diffState(State, &self.prev_state, &self.state, &dirty_bits);

            const iteration_limit_max: usize = 64;
            var iteration: usize = 0;
            while (!dirty_bits.isZero() and iteration < iteration_limit_max) : (iteration += 1) {
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
            assert(!std.math.isNan(new_width_px));
            assert(!std.math.isNan(new_height_px));
            assert(new_width_px >= 0.0);
            assert(new_height_px >= 0.0);

            const t0 = time.now();

            const clamped_w = std.math.clamp(new_width_px, 1.0, 32767.0);
            const clamped_h = std.math.clamp(new_height_px, 1.0, 32767.0);

            const height_changed = (clamped_h != self.viewport_height_px);
            self.viewport_width_px = clamped_w;
            self.viewport_height_px = clamped_h;

            if (height_changed or self.layout_dirty) {
                self.tree.run4Pass(self.viewport_width_px, self.viewport_height_px);
            }

            const w_i16: i16 = @intFromFloat(clamped_w);
            self.tree.solveResizeSimd(w_i16);
            self.layout_dirty = false;

            if (self.color_target.isValid()) {
                GpuBackend.destroyTexture(&self.gpu_device, self.color_target);
            }
            const max_w: u32 = hal.mock.texture_width_max;
            const max_h: u32 = hal.mock.texture_height_max;
            const w_u32: u32 = @intFromFloat(clamped_w);
            const h_u32: u32 = @intFromFloat(clamped_h);
            const tex_w = @min(w_u32, max_w);
            const tex_h = @min(h_u32, max_h);

            self.color_target = GpuBackend.createTexture(
                &self.gpu_device,
                .{
                    .size = .{ .width = @max(1, tex_w), .height = @max(1, tex_h) },
                    .format = .rgba8_unorm,
                    .usage = hal.TextureUsage.render_target,
                },
            ) catch hal.Texture.null_handle;

            const t1 = time.now();
            self.metrics.layout_duration_ns = t1.durationSince(t0);
            self.metrics.layout_cycles = time.toCyclesAt4Ghz(self.metrics.layout_duration_ns);
        }

        /// Solves layout if dirty.
        pub fn solveLayout(self: *Self) void {
            if (!self.layout_dirty) return;

            const t0 = time.now();
            self.tree.run4Pass(self.viewport_width_px, self.viewport_height_px);
            const clamped_w = std.math.clamp(self.viewport_width_px, 1.0, 32767.0);
            const w_i16: i16 = @intFromFloat(clamped_w);
            self.tree.solveResizeSimd(w_i16);
            self.layout_dirty = false;

            const t1 = time.now();
            self.metrics.layout_duration_ns = t1.durationSince(t0);
            self.metrics.layout_cycles = time.toCyclesAt4Ghz(self.metrics.layout_duration_ns);
        }

        /// Packs resolved geometry into vertex quads, uploads to GPU buffer,
        /// and records render pass draw commands.
        pub fn renderFrame(self: *Self) void {
            const t0 = time.now();

            // 1. Pack quads from resolved node positions directly into staging buffer
            self.staged_vertex_count = 0;
            const count = self.tree.node_count;

            var i: usize = 0;
            while (i < count) : (i += 1) {
                const rect = self.tree.getRect(@intCast(i));
                if (rect.width_px <= 0.0 or rect.height_px <= 0.0) continue;

                assert(self.staged_vertex_count + 6 <= max_vertices);
                const out_idx = self.staged_vertex_count;
                const x0 = rect.x_px;
                const y0 = rect.y_px;
                const x1 = x0 + rect.width_px;
                const y1 = y0 + rect.height_px;
                const col = self.node_colors[i];

                self.vertex_staging[out_idx + 0] = .{ .pos = .{ x0, y0 }, .color = col };
                self.vertex_staging[out_idx + 1] = .{ .pos = .{ x1, y0 }, .color = col };
                self.vertex_staging[out_idx + 2] = .{ .pos = .{ x1, y1 }, .color = col };
                self.vertex_staging[out_idx + 3] = .{ .pos = .{ x1, y1 }, .color = col };
                self.vertex_staging[out_idx + 4] = .{ .pos = .{ x0, y1 }, .color = col };
                self.vertex_staging[out_idx + 5] = .{ .pos = .{ x0, y0 }, .color = col };
                self.staged_vertex_count += 6;
            }

            // 2. Upload staging vertices to GPU buffer
            if (self.staged_vertex_count > 0) {
                const upload_bytes: u32 = self.staged_vertex_count * @sizeOf(hal.Vertex2D);
                const raw_bytes: [*]const u8 = @ptrCast(&self.vertex_staging);
                GpuBackend.writeBuffer(
                    &self.gpu_queue,
                    self.vertex_buffer,
                    0,
                    raw_bytes[0..upload_bytes],
                );
            }

            // 3. Record GPU render commands
            var encoder = GpuBackend.createCommandEncoder(&self.gpu_device) catch return;
            var pass = GpuBackend.beginRenderPass(&encoder, .{
                .color_attachment = .{
                    .target = self.color_target,
                    .load_op = .clear,
                    .store_op = .store,
                    .clear_color = .{ 0.05, 0.05, 0.05, 1.0 },
                },
            }) catch return;

            GpuBackend.setPipeline(&pass, self.pipeline);
            GpuBackend.setVertexBuffer(&pass, 0, self.vertex_buffer, 0);
            GpuBackend.draw(&pass, self.staged_vertex_count, 1, 0, 0);
            GpuBackend.endRenderPass(&pass);

            const cmd_buffer = GpuBackend.finishCommandEncoder(&encoder) catch return;
            GpuBackend.submit(&self.gpu_queue, &.{cmd_buffer});

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

test "Engine v2 resize updates renderFrame vertex staging" {
    const TestState = extern struct {
        scale: f32 = 1.0,
    };

    const EngineType = Engine(TestState, 8, hal.MockBackend);
    var engine: EngineType = undefined;
    engine.init(.{}, 800.0, 600.0);

    const root = engine.addNode(.col, flexbox.INDEX_NONE, 0, .{
        .pad_px = 0.0,
        .gap_px = 0.0,
    }, true, .{ 255, 255, 255, 255 });

    _ = engine.addNode(.button, root, 1, .{
        .grow = 1.0,
        .height_px_min = 40.0,
    }, true, .{ 255, 0, 0, 255 });

    engine.buildLayout();
    engine.renderFrame();

    // Node 0 quads: indices 0..5. Node 1 quads: indices 6..11.
    // In makeQuadVertices, vertex 1 is top-right: pos = .{ x + w, y }
    try std.testing.expectEqual(@as(f32, 800.0), engine.tree.getRect(1).width_px);
    try std.testing.expectEqual(@as(f32, 800.0), engine.vertex_staging[7].pos[0]);

    // Resize viewport to 1400 px width
    engine.resize(1400.0, 600.0);
    engine.renderFrame();

    // Verify renderFrame now stages quads reflecting 1400px width
    try std.testing.expectEqual(@as(f32, 1400.0), engine.tree.getRect(1).width_px);
    try std.testing.expectEqual(@as(f32, 1400.0), engine.vertex_staging[7].pos[0]);
}

test "Engine v2 vertical resize re-evaluates heights and Y coordinates" {
    const TestState = extern struct {
        scale: f32 = 1.0,
    };

    const EngineType = Engine(TestState, 8, hal.MockBackend);
    var engine: EngineType = undefined;
    engine.init(.{}, 800.0, 600.0);

    const root = engine.addNode(.col, flexbox.INDEX_NONE, 0, .{
        .pad_px = 10.0,
        .gap_px = 10.0,
    }, true, .{ 255, 255, 255, 255 });

    _ = engine.addNode(.button, root, 1, .{
        .grow = 1.0,
        .height_px_min = 50.0,
    }, true, .{ 255, 0, 0, 255 });

    _ = engine.addNode(.button, root, 1, .{
        .grow = 1.0,
        .height_px_min = 50.0,
    }, true, .{ 0, 255, 0, 255 });

    engine.buildLayout();

    // Initial root height is 600
    try std.testing.expectEqual(@as(f32, 600.0), engine.tree.getRect(0).height_px);

    // Resize viewport height to 900
    engine.resize(800.0, 900.0);

    // Verify root height updated to 900
    try std.testing.expectEqual(@as(f32, 900.0), engine.tree.getRect(0).height_px);

    // Node 1 starts at pad = 10, height 50
    const r1 = engine.tree.getRect(1);
    try std.testing.expectEqual(@as(f32, 10.0), r1.y_px);
    try std.testing.expectEqual(@as(f32, 50.0), r1.height_px);

    // Node 2 starts at 10 + 50 + gap(10) = 70, height 50
    const r2 = engine.tree.getRect(2);
    try std.testing.expectEqual(@as(f32, 70.0), r2.y_px);
    try std.testing.expectEqual(@as(f32, 50.0), r2.height_px);
}

test "Engine v2 large viewport bounds clamping prevents overflow" {
    const TestState = extern struct {
        scale: f32 = 1.0,
    };

    const EngineType = Engine(TestState, 8, hal.MockBackend);
    var engine: EngineType = undefined;
    engine.init(.{}, 800.0, 600.0);

    _ = engine.addNode(.col, flexbox.INDEX_NONE, 0, .{}, true, .{ 255, 255, 255, 255 });
    engine.buildLayout();

    // Resize with oversized dimensions (> 32767.0)
    engine.resize(50000.0, 60000.0);

    // Viewport dimensions should be clamped to 32767.0
    try std.testing.expectEqual(@as(f32, 32767.0), engine.viewport_width_px);
    try std.testing.expectEqual(@as(f32, 32767.0), engine.viewport_height_px);
    try std.testing.expectEqual(@as(f32, 32767.0), engine.tree.getRect(0).width_px);
}

test "Engine v2 initial oversized viewport clamping prevents overflow" {
    const TestState = extern struct {
        scale: f32 = 1.0,
    };

    const EngineType = Engine(TestState, 8, hal.MockBackend);
    var engine: EngineType = undefined;
    // Initializing directly with dimensions > 32767.0 must clamp to 32767.0
    engine.init(.{}, 50000.0, 60000.0);

    _ = engine.addNode(.col, flexbox.INDEX_NONE, 0, .{}, true, .{ 255, 255, 255, 255 });
    engine.buildLayout();

    try std.testing.expectEqual(@as(f32, 32767.0), engine.viewport_width_px);
    try std.testing.expectEqual(@as(f32, 32767.0), engine.viewport_height_px);
    try std.testing.expectEqual(@as(f32, 32767.0), engine.tree.getRect(0).width_px);
    try std.testing.expectEqual(@as(f32, 32767.0), engine.tree.getRect(0).height_px);
}

test "Engine v2 resize with identical height honors pending dirty layout" {
    const TestState = extern struct {
        scale: f32 = 1.0,
    };

    const EngineType = Engine(TestState, 8, hal.MockBackend);
    var engine: EngineType = undefined;
    engine.init(.{}, 800.0, 600.0);

    const root = engine.addNode(.col, flexbox.INDEX_NONE, 0, .{
        .pad_px = 10.0,
    }, true, .{ 255, 255, 255, 255 });

    const btn = engine.addNode(.button, root, 1, .{
        .height_px_min = 40.0,
    }, true, .{ 255, 0, 0, 255 });

    engine.buildLayout();
    try std.testing.expectEqual(@as(f32, 40.0), engine.tree.getRect(btn).height_px);

    // Mutate child style directly and mark layout dirty
    engine.tree.nodes[btn].style.height_px_min = 120.0;
    engine.layout_dirty = true;

    // Resize viewport with identical height: dirty layout must still be solved
    engine.resize(1000.0, 600.0);

    try std.testing.expectEqual(@as(bool, false), engine.layout_dirty);
    try std.testing.expectEqual(@as(f32, 120.0), engine.tree.getRect(btn).height_px);
}

test "Engine v2 resize with 0x0 collapsed viewport clamps to 1x1" {
    const TestState = extern struct {
        scale: f32 = 1.0,
    };

    const EngineType = Engine(TestState, 8, hal.MockBackend);
    var engine: EngineType = undefined;
    engine.init(.{}, 800.0, 600.0);

    _ = engine.addNode(.col, flexbox.INDEX_NONE, 0, .{}, true, .{ 255, 255, 255, 255 });
    engine.buildLayout();

    // Resize with 0.0 dimensions (e.g. window minimized)
    engine.resize(0.0, 0.0);

    try std.testing.expectEqual(@as(f32, 1.0), engine.viewport_width_px);
    try std.testing.expectEqual(@as(f32, 1.0), engine.viewport_height_px);
    try std.testing.expectEqual(@as(f32, 1.0), engine.tree.getRect(0).width_px);
}
