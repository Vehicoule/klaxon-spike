//! ============================================================================
//! Klaxon Hardware Abstraction Layer (`src/gpu/hal.zig`)
//! ============================================================================
//! Comptime-dispatched, zero-vtable GPU Hardware Abstraction Layer (HAL).
//! Pinned 19-function contract following WebGPU idioms and TigerStyle safety:
//! 1. Zero dynamic allocations (`0 malloc/free`) in steady-state rendering.
//! 2. Zero libc and zero POSIX dependencies (pure freestanding compliance).
//! 3. Comptime backend validation guaranteeing bit-exact interface conformity.
//! 4. Static BSS memory working sets for deterministic execution and replay.
//!
//! Standards: TigerStyle (< 100 columns, zero trailing whitespace, paired asserts).

const std = @import("std");
const assert = std.debug.assert;

pub const mock = @import("mock.zig");
pub const MockBackend = mock.MockBackend;
pub const MockWindow = mock.MockWindow;
pub const Vertex2D = mock.Vertex2D;
pub const makeQuadVertices = mock.makeQuadVertices;

// ============================================================================
// 1. Resource Handle Primitives (Generational Identifiers)
// ============================================================================

/// Typed generational handle identifying a GPU memory buffer.
pub const Buffer = extern struct {
    id: u32 = 0,
    generation: u32 = 0,

    pub const null_handle: Buffer = .{ .id = 0, .generation = 0 };

    pub inline fn isValid(self: Buffer) bool {
        return self.id != 0;
    }
};

comptime {
    assert(@sizeOf(Buffer) == 8);
    assert(@alignOf(Buffer) == 4);
}

/// Typed generational handle identifying a GPU texture image.
pub const Texture = extern struct {
    id: u32 = 0,
    generation: u32 = 0,

    pub const null_handle: Texture = .{ .id = 0, .generation = 0 };

    pub inline fn isValid(self: Texture) bool {
        return self.id != 0;
    }
};

comptime {
    assert(@sizeOf(Texture) == 8);
    assert(@alignOf(Texture) == 4);
}

/// Typed generational handle identifying a GPU texture sampler.
pub const Sampler = extern struct {
    id: u32 = 0,
    generation: u32 = 0,

    pub const null_handle: Sampler = .{ .id = 0, .generation = 0 };

    pub inline fn isValid(self: Sampler) bool {
        return self.id != 0;
    }
};

comptime {
    assert(@sizeOf(Sampler) == 8);
    assert(@alignOf(Sampler) == 4);
}

/// Typed generational handle identifying a compiled GPU shader module.
pub const ShaderModule = extern struct {
    id: u32 = 0,
    generation: u32 = 0,

    pub const null_handle: ShaderModule = .{ .id = 0, .generation = 0 };

    pub inline fn isValid(self: ShaderModule) bool {
        return self.id != 0;
    }
};

comptime {
    assert(@sizeOf(ShaderModule) == 8);
    assert(@alignOf(ShaderModule) == 4);
}

/// Typed generational handle identifying an immutable render pipeline state.
pub const RenderPipeline = extern struct {
    id: u32 = 0,
    generation: u32 = 0,

    pub const null_handle: RenderPipeline = .{ .id = 0, .generation = 0 };

    pub inline fn isValid(self: RenderPipeline) bool {
        return self.id != 0;
    }
};

comptime {
    assert(@sizeOf(RenderPipeline) == 8);
    assert(@alignOf(RenderPipeline) == 4);
}

/// Typed generational handle identifying an executable recorded command buffer.
pub const CommandBuffer = extern struct {
    id: u32 = 0,
    generation: u32 = 0,

    pub const null_handle: CommandBuffer = .{ .id = 0, .generation = 0 };

    pub inline fn isValid(self: CommandBuffer) bool {
        return self.id != 0;
    }
};

comptime {
    assert(@sizeOf(CommandBuffer) == 8);
    assert(@alignOf(CommandBuffer) == 4);
}

// ============================================================================
// 2. Buffer Descriptors & Bitflags
// ============================================================================

/// Intended usage bitflags for GPU memory buffers.
pub const BufferUsage = packed struct(u32) {
    vertex: bool = false,
    index: bool = false,
    uniform: bool = false,
    storage: bool = false,
    copy_src: bool = false,
    copy_dst: bool = false,
    _padding: u26 = 0,

    pub const vertex_dst: BufferUsage = .{ .vertex = true, .copy_dst = true };
    pub const uniform_dst: BufferUsage = .{ .uniform = true, .copy_dst = true };
    pub const transfer_src: BufferUsage = .{ .copy_src = true };
    pub const transfer_dst: BufferUsage = .{ .copy_dst = true };
};

comptime {
    assert(@sizeOf(BufferUsage) == 4);
    assert(@alignOf(BufferUsage) == 4);
}

/// Parameters for creating a GPU memory buffer.
pub const BufferDesc = struct {
    size: u64,
    usage: BufferUsage,
    label: []const u8 = "",
};

// ============================================================================
// 3. Texture Descriptors, Dimensions & Data Layouts
// ============================================================================

/// 3D spatial extents for textures and copy operations.
pub const Extent3D = extern struct {
    width: u32,
    height: u32 = 1,
    depth_or_array_layers: u32 = 1,
};

comptime {
    assert(@sizeOf(Extent3D) == 12);
    assert(@alignOf(Extent3D) == 4);
}

/// Pixel format classifications supported by the HAL.
pub const TextureFormat = enum(u32) {
    rgba8_unorm = 0,
    bgra8_unorm = 1,
    r8_unorm = 2,
    depth32_float = 3,

    pub inline fn bytesPerPixel(self: TextureFormat) u32 {
        return switch (self) {
            .rgba8_unorm, .bgra8_unorm, .depth32_float => 4,
            .r8_unorm => 1,
        };
    }
};

/// Dimensionality of the texture resource.
pub const TextureDimension = enum(u8) {
    dimension_1d = 0,
    dimension_2d = 1,
    dimension_3d = 2,
};

/// Intended usage bitflags for GPU texture images.
pub const TextureUsage = packed struct(u32) {
    render_attachment: bool = false,
    texture_binding: bool = false,
    copy_src: bool = false,
    copy_dst: bool = false,
    _padding: u28 = 0,

    pub const render_target: TextureUsage = .{
        .render_attachment = true,
        .copy_src = true,
    };
    pub const sampled_image: TextureUsage = .{
        .texture_binding = true,
        .copy_dst = true,
    };
};

comptime {
    assert(@sizeOf(TextureUsage) == 4);
    assert(@alignOf(TextureUsage) == 4);
}

/// Parameters for creating a GPU texture image.
pub const TextureDesc = struct {
    size: Extent3D,
    format: TextureFormat = .rgba8_unorm,
    usage: TextureUsage = .{
        .render_attachment = true,
        .copy_src = true,
        .copy_dst = true,
    },
    dimension: TextureDimension = .dimension_2d,
    mip_level_count: u32 = 1,
    sample_count: u32 = 1,
    label: []const u8 = "",
};

/// Memory layout of pixel data within a linear staging buffer.
pub const TextureDataLayout = extern struct {
    offset: u64 = 0,
    bytes_per_row: u32,
    rows_per_image: u32 = 0,
};

comptime {
    assert(@sizeOf(TextureDataLayout) == 16);
    assert(@alignOf(TextureDataLayout) == 8);
}

// ============================================================================
// 4. Pinned Sampler Invariants (Bilinear Filtering & Clamp-To-Edge)
// ============================================================================

/// Coordinate wrapping mode for texture sampling.
pub const AddressMode = enum(u8) {
    clamp_to_edge = 0,
    repeat = 1,
    mirror_repeat = 2,
};

/// Magnification and minification filtering modes.
pub const FilterMode = enum(u8) {
    nearest = 0,
    linear = 1,
};

/// Mipmap level transition filter mode.
pub const MipmapFilterMode = enum(u8) {
    none = 0,
    nearest = 1,
    linear = 2,
};

/// Pinned sampler descriptor: bilinear filtering, clamp-to-edge, zero mipmaps.
/// Per AGENTS.md invariant 6: desc.mipmap_filter must always equal .none.
pub const SamplerDesc = extern struct {
    address_mode_u: AddressMode = .clamp_to_edge,
    address_mode_v: AddressMode = .clamp_to_edge,
    address_mode_w: AddressMode = .clamp_to_edge,
    mag_filter: FilterMode = .linear,
    min_filter: FilterMode = .linear,
    mipmap_filter: MipmapFilterMode = .none,
    lod_min_clamp: f32 = 0.0,
    lod_max_clamp: f32 = 0.0,
    max_anisotropy: u16 = 1,
    _padding: u16 = 0,

    /// Pinned default configuration: bilinear filtering, clamp-to-edge, no mipmaps.
    pub const pinned_bilinear_clamp: SamplerDesc = .{};
};

comptime {
    assert(@sizeOf(SamplerDesc) == 20);
    assert(@alignOf(SamplerDesc) == 4);
    assert(SamplerDesc.pinned_bilinear_clamp.mipmap_filter == .none);
    assert(SamplerDesc.pinned_bilinear_clamp.mag_filter == .linear);
    assert(SamplerDesc.pinned_bilinear_clamp.min_filter == .linear);
    assert(SamplerDesc.pinned_bilinear_clamp.address_mode_u == .clamp_to_edge);
    assert(SamplerDesc.pinned_bilinear_clamp.address_mode_v == .clamp_to_edge);
}

// ============================================================================
// 5. Shaders & Pipeline Descriptors
// ============================================================================

/// Execution stage of a programmable shader module.
pub const ShaderStage = enum(u8) {
    vertex = 0,
    fragment = 1,
    compute = 2,
};

/// Parameters for creating a compiled shader module.
pub const ShaderModuleDesc = struct {
    code: []const u8,
    stage: ShaderStage = .vertex,
    label: []const u8 = "",
};

/// Geometric primitive assembly topology.
pub const PrimitiveTopology = enum(u8) {
    point_list = 0,
    line_list = 1,
    line_strip = 2,
    triangle_list = 3,
    triangle_strip = 4,
};

/// Data format of a vertex buffer attribute.
pub const VertexFormat = enum(u8) {
    float32x2 = 0,
    float32x4 = 1,
    unorm8x4 = 2,
    uint32 = 3,

    pub inline fn sizeOf(self: VertexFormat) u32 {
        return switch (self) {
            .float32x2 => 8,
            .float32x4 => 16,
            .unorm8x4, .uint32 => 4,
        };
    }
};

/// Single attribute description within a vertex buffer layout.
pub const VertexAttribute = extern struct {
    format: VertexFormat,
    offset: u64,
    shader_location: u32,
    _padding: u32 = 0,
};

comptime {
    assert(@sizeOf(VertexAttribute) == 24);
    assert(@alignOf(VertexAttribute) == 8);
}

/// Rate at which vertex buffer data advances during rendering.
pub const VertexStepMode = enum(u8) {
    vertex = 0,
    instance = 1,
};

/// Layout configuration for an input vertex buffer slot.
pub const VertexBufferLayout = struct {
    array_stride: u64,
    step_mode: VertexStepMode = .vertex,
    attributes: []const VertexAttribute = &.{},
};

/// Factor controlling color/alpha blending arithmetic.
pub const BlendFactor = enum(u8) {
    zero = 0,
    one = 1,
    src_alpha = 2,
    one_minus_src_alpha = 3,
    dst_alpha = 4,
    one_minus_dst_alpha = 5,
};

/// Arithmetic operation used when combining blended colors.
pub const BlendOperation = enum(u8) {
    add = 0,
    subtract = 1,
    reverse_subtract = 2,
    min = 3,
    max = 4,
};

/// Individual blend equation for color or alpha channels.
pub const BlendComponent = extern struct {
    src_factor: BlendFactor = .one,
    dst_factor: BlendFactor = .zero,
    operation: BlendOperation = .add,
};

comptime {
    assert(@sizeOf(BlendComponent) == 3);
}

/// Combined blend state for color attachments.
pub const BlendState = extern struct {
    color: BlendComponent = .{
        .src_factor = .one,
        .dst_factor = .one_minus_src_alpha,
        .operation = .add,
    },
    alpha: BlendComponent = .{
        .src_factor = .one,
        .dst_factor = .one_minus_src_alpha,
        .operation = .add,
    },

    pub const premultiplied: BlendState = .{
        .color = .{ .src_factor = .one, .dst_factor = .one_minus_src_alpha },
        .alpha = .{ .src_factor = .one, .dst_factor = .one_minus_src_alpha },
    };
};

/// Configuration for creating a GPU render pipeline.
pub const RenderPipelineDesc = struct {
    vertex_shader: ShaderModule,
    fragment_shader: ?ShaderModule = null,
    vertex_layout: ?VertexBufferLayout = null,
    color_target_format: TextureFormat = .rgba8_unorm,
    primitive_topology: PrimitiveTopology = .triangle_list,
    blend: ?BlendState = null,
    label: []const u8 = "",
};

// ============================================================================
// 6. 2D Geometry & Render Pass Descriptors
// ============================================================================

/// Integer 2D bounding rectangle used for scissor/clipping operations.
pub const RectU32 = extern struct {
    x: u32 = 0,
    y: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,

    pub inline fn isEmpty(self: RectU32) bool {
        return (self.width == 0 or self.height == 0);
    }

    /// Clamps this rectangle strictly within the spatial bounds [0, bound_w) x [0, bound_h).
    pub fn clampToBounds(self: RectU32, bound_w: u32, bound_h: u32) RectU32 {
        if (self.x >= bound_w or self.y >= bound_h) {
            return .{ .x = 0, .y = 0, .width = 0, .height = 0 };
        }
        const clamped_w = @min(self.width, bound_w - self.x);
        const clamped_h = @min(self.height, bound_h - self.y);
        return .{
            .x = self.x,
            .y = self.y,
            .width = clamped_w,
            .height = clamped_h,
        };
    }
};

comptime {
    assert(@sizeOf(RectU32) == 16);
    assert(@alignOf(RectU32) == 4);
}

/// Action performed on render target contents upon pass initialization.
pub const LoadOp = enum(u8) {
    load = 0,
    clear = 1,
};

/// Action performed on render target contents upon pass completion.
pub const StoreOp = enum(u8) {
    store = 0,
    discard = 1,
};

/// Attachment configuration for a render pass color target.
pub const ColorAttachment = struct {
    target: Texture,
    load_op: LoadOp = .clear,
    store_op: StoreOp = .store,
    clear_color: [4]f32 = .{ 0.0, 0.0, 0.0, 1.0 },
};

/// Descriptors for initiating an active GPU render pass.
pub const RenderPassDesc = struct {
    color_attachment: ColorAttachment,
    label: []const u8 = "",
};

// ============================================================================
// 7. HAL Error Sets
// ============================================================================

pub const GpuError = error{
    OutOfMemory,
    PoolExhausted,
    InvalidDesc,
    InvalidHandle,
    BufferTooSmall,
    TargetNotFound,
    DeviceLost,
    InvalidCommandSequence,
};

// ============================================================================
// 8. Compile-Time Backend Contract Validation (19 Pinned Functions)
// ============================================================================

/// Compile-time contract assertion verifying that `Backend` implements every required
/// type and all 19 pinned functions with exact matching signatures.
pub fn validateBackend(comptime Backend: type) void {
    comptime {
        // Assert required types
        const required_types = [_][]const u8{
            "Device",
            "Queue",
            "Buffer",
            "Texture",
            "Sampler",
            "ShaderModule",
            "RenderPipeline",
            "CommandEncoder",
            "RenderPassEncoder",
            "CommandBuffer",
        };
        for (required_types) |type_name| {
            if (!@hasDecl(Backend, type_name)) {
                @compileError("Backend missing required type: " ++ type_name);
            }
        }

        // Assert 19 pinned functions
        const required_functions = [_][]const u8{
            "createBuffer",
            "destroyBuffer",
            "writeBuffer",
            "createTexture",
            "destroyTexture",
            "writeTexture",
            "createSampler",
            "createShaderModule",
            "createRenderPipeline",
            "createCommandEncoder",
            "beginRenderPass",
            "setPipeline",
            "setVertexBuffer",
            "setClipRect",
            "draw",
            "endRenderPass",
            "finishCommandEncoder",
            "submit",
            "readbackTarget",
        };
        for (required_functions) |fn_name| {
            if (!@hasDecl(Backend, fn_name)) {
                @compileError("Backend missing required 19-function HAL entry: " ++ fn_name);
            }
        }

        // Statically verify signatures of all 19 pinned functions
        _ = struct {
            fn verifySignatures() void {
                const dev: *Backend.Device = undefined;
                const q: *Backend.Queue = undefined;
                const b_desc: BufferDesc = undefined;
                const b: Backend.Buffer = undefined;
                const t_desc: TextureDesc = undefined;
                const t: Backend.Texture = undefined;
                const s_desc: SamplerDesc = undefined;
                const sm_desc: ShaderModuleDesc = undefined;
                const rp_desc: RenderPipelineDesc = undefined;
                const rp: Backend.RenderPipeline = undefined;
                const enc: *Backend.CommandEncoder = undefined;
                const pass: *Backend.RenderPassEncoder = undefined;
                const cbs: []const Backend.CommandBuffer = undefined;
                const layout: TextureDataLayout = undefined;
                const extent: Extent3D = undefined;
                const rect: RectU32 = undefined;
                const bytes_const: []const u8 = undefined;
                const bytes_mut: []u8 = undefined;

                _ = @as(anyerror!Backend.Buffer, Backend.createBuffer(dev, b_desc));
                Backend.destroyBuffer(dev, b);
                Backend.writeBuffer(q, b, 0, bytes_const);
                _ = @as(anyerror!Backend.Texture, Backend.createTexture(dev, t_desc));
                Backend.destroyTexture(dev, t);
                Backend.writeTexture(q, t, layout, extent, bytes_const);
                _ = @as(anyerror!Backend.Sampler, Backend.createSampler(dev, s_desc));
                _ = @as(anyerror!Backend.ShaderModule, Backend.createShaderModule(dev, sm_desc));
                _ = @as(
                    anyerror!Backend.RenderPipeline,
                    Backend.createRenderPipeline(dev, rp_desc),
                );
                _ = @as(anyerror!Backend.CommandEncoder, Backend.createCommandEncoder(dev));
                _ = @as(anyerror!Backend.RenderPassEncoder, Backend.beginRenderPass(enc, .{}));
                Backend.setPipeline(pass, rp);
                Backend.setVertexBuffer(pass, 0, b, 0);
                Backend.setClipRect(pass, rect);
                Backend.draw(pass, 0, 0, 0, 0);
                Backend.endRenderPass(pass);
                _ = @as(anyerror!Backend.CommandBuffer, Backend.finishCommandEncoder(enc));
                Backend.submit(q, cbs);
                _ = @as(anyerror!void, Backend.readbackTarget(dev, t, bytes_mut));
            }
        }.verifySignatures;
    }
}

// ============================================================================
// 9. Comptime-Dispatched Engine Pattern (Zero Dynamic Vtable Overhead)
// ============================================================================

/// Compile-time parameterized application engine coupling a GPU backend and window.
/// Dispatches all 19 HAL operations directly through static inlining with 0 vtable cost.
pub fn Engine(comptime GpuBackend: type, comptime Window: type) type {
    validateBackend(GpuBackend);

    return struct {
        const Self = @This();

        pub const Backend = GpuBackend;
        pub const WindowType = Window;

        device: GpuBackend.Device,
        queue: GpuBackend.Queue,
        window: Window,

        pub fn init(
            target: *Self,
            device: GpuBackend.Device,
            window: Window,
        ) void {
            target.device = device;
            target.queue = GpuBackend.Queue.init(&target.device);
            target.window = window;
        }

        pub inline fn createBuffer(self: *Self, desc: BufferDesc) !GpuBackend.Buffer {
            return GpuBackend.createBuffer(&self.device, desc);
        }

        pub inline fn destroyBuffer(self: *Self, buffer: GpuBackend.Buffer) void {
            GpuBackend.destroyBuffer(&self.device, buffer);
        }

        pub inline fn writeBuffer(
            self: *Self,
            buffer: GpuBackend.Buffer,
            offset: u64,
            data: []const u8,
        ) void {
            GpuBackend.writeBuffer(&self.queue, buffer, offset, data);
        }

        pub inline fn createTexture(self: *Self, desc: TextureDesc) !GpuBackend.Texture {
            return GpuBackend.createTexture(&self.device, desc);
        }

        pub inline fn destroyTexture(self: *Self, texture: GpuBackend.Texture) void {
            GpuBackend.destroyTexture(&self.device, texture);
        }

        pub inline fn writeTexture(
            self: *Self,
            texture: GpuBackend.Texture,
            layout: TextureDataLayout,
            size: Extent3D,
            data: []const u8,
        ) void {
            GpuBackend.writeTexture(&self.queue, texture, layout, size, data);
        }

        pub inline fn createSampler(self: *Self, desc: SamplerDesc) !GpuBackend.Sampler {
            return GpuBackend.createSampler(&self.device, desc);
        }

        pub inline fn createShaderModule(
            self: *Self,
            desc: ShaderModuleDesc,
        ) !GpuBackend.ShaderModule {
            return GpuBackend.createShaderModule(&self.device, desc);
        }

        pub inline fn createRenderPipeline(
            self: *Self,
            desc: RenderPipelineDesc,
        ) !GpuBackend.RenderPipeline {
            return GpuBackend.createRenderPipeline(&self.device, desc);
        }

        pub inline fn createCommandEncoder(self: *Self) !GpuBackend.CommandEncoder {
            return GpuBackend.createCommandEncoder(&self.device);
        }

        pub inline fn beginRenderPass(
            self: *Self,
            encoder: *GpuBackend.CommandEncoder,
            desc: RenderPassDesc,
        ) !GpuBackend.RenderPassEncoder {
            _ = self;
            return GpuBackend.beginRenderPass(encoder, desc);
        }

        pub inline fn setPipeline(
            self: *Self,
            pass: *GpuBackend.RenderPassEncoder,
            pipeline: GpuBackend.RenderPipeline,
        ) void {
            _ = self;
            GpuBackend.setPipeline(pass, pipeline);
        }

        pub inline fn setVertexBuffer(
            self: *Self,
            pass: *GpuBackend.RenderPassEncoder,
            slot: u32,
            buffer: GpuBackend.Buffer,
            offset: u64,
        ) void {
            _ = self;
            GpuBackend.setVertexBuffer(pass, slot, buffer, offset);
        }

        pub inline fn setClipRect(
            self: *Self,
            pass: *GpuBackend.RenderPassEncoder,
            rect: RectU32,
        ) void {
            _ = self;
            GpuBackend.setClipRect(pass, rect);
        }

        pub inline fn draw(
            self: *Self,
            pass: *GpuBackend.RenderPassEncoder,
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        ) void {
            _ = self;
            GpuBackend.draw(pass, vertex_count, instance_count, first_vertex, first_instance);
        }

        pub inline fn endRenderPass(
            self: *Self,
            pass: *GpuBackend.RenderPassEncoder,
        ) void {
            _ = self;
            GpuBackend.endRenderPass(pass);
        }

        pub inline fn finishCommandEncoder(
            self: *Self,
            encoder: *GpuBackend.CommandEncoder,
        ) !GpuBackend.CommandBuffer {
            _ = self;
            return GpuBackend.finishCommandEncoder(encoder);
        }

        pub inline fn submit(self: *Self, commands: []const GpuBackend.CommandBuffer) void {
            GpuBackend.submit(&self.queue, commands);
        }

        pub inline fn readbackTarget(
            self: *Self,
            target: GpuBackend.Texture,
            dest: []u8,
        ) !void {
            return GpuBackend.readbackTarget(&self.device, target, dest);
        }
    };
}

// ============================================================================
// 10. Unit Tests for HAL Invariants & Comptime Dispatch
// ============================================================================

test "HAL: Bit-exact structural layouts and size contracts" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Buffer));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Texture));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Sampler));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(ShaderModule));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(RenderPipeline));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(CommandBuffer));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(Extent3D));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(TextureDataLayout));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(SamplerDesc));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(RectU32));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(VertexAttribute));
}

test "HAL: Pinned SamplerDesc bilinear clamp invariants" {
    const sampler_desc = SamplerDesc.pinned_bilinear_clamp;
    try std.testing.expectEqual(MipmapFilterMode.none, sampler_desc.mipmap_filter);
    try std.testing.expectEqual(FilterMode.linear, sampler_desc.mag_filter);
    try std.testing.expectEqual(FilterMode.linear, sampler_desc.min_filter);
    try std.testing.expectEqual(AddressMode.clamp_to_edge, sampler_desc.address_mode_u);
    try std.testing.expectEqual(AddressMode.clamp_to_edge, sampler_desc.address_mode_v);
}

test "HAL: RectU32 boundary clamping arithmetic" {
    const rect = RectU32{ .x = 10, .y = 20, .width = 100, .height = 100 };
    const clamped = rect.clampToBounds(50, 60);
    try std.testing.expectEqual(@as(u32, 10), clamped.x);
    try std.testing.expectEqual(@as(u32, 20), clamped.y);
    try std.testing.expectEqual(@as(u32, 40), clamped.width);
    try std.testing.expectEqual(@as(u32, 40), clamped.height);

    const out_of_bounds = RectU32{ .x = 200, .y = 300, .width = 50, .height = 50 };
    const empty_clamped = out_of_bounds.clampToBounds(100, 100);
    try std.testing.expect(empty_clamped.isEmpty());
}

test "HAL: Comptime backend validation on MockBackend" {
    comptime validateBackend(MockBackend);
}

test "HAL: Engine instantiation with zero-cost comptime dispatch" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    const window = MockWindow{ .width_px = 640, .height_px = 480 };

    const MockEngine = Engine(MockBackend, MockWindow);
    var engine: MockEngine = undefined;
    MockEngine.init(&engine, device, window);

    const buf_desc = BufferDesc{
        .size = 256,
        .usage = BufferUsage.vertex_dst,
    };
    const buffer = try engine.createBuffer(buf_desc);
    try std.testing.expect(buffer.isValid());

    const test_bytes = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    engine.writeBuffer(buffer, 0, &test_bytes);

    // Verify bytes were actually written to engine.device
    const data = engine.device.getBufferData(buffer).?;
    try std.testing.expectEqualSlices(u8, &test_bytes, data[0..8]);

    engine.destroyBuffer(buffer);
}

test "HAL: All 19 functions fully operational through Engine comptime dispatch" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    const window = MockWindow{ .width_px = 32, .height_px = 32 };

    const MockEngine = Engine(MockBackend, MockWindow);
    var engine: MockEngine = undefined;
    MockEngine.init(&engine, device, window);

    // 1. createBuffer
    const buf = try engine.createBuffer(.{
        .size = 128,
        .usage = BufferUsage.vertex_dst,
    });
    try std.testing.expect(buf.isValid());

    // 2. writeBuffer
    const quad_verts = makeQuadVertices(2.0, 2.0, 8.0, 8.0, .{ 255, 0, 0, 255 });
    engine.writeBuffer(buf, 0, std.mem.sliceAsBytes(&quad_verts));

    // 3. createTexture
    const tex = try engine.createTexture(.{
        .size = .{ .width = 16, .height = 16 },
        .format = .rgba8_unorm,
    });
    try std.testing.expect(tex.isValid());

    // 4. writeTexture
    var initial_pixels: [16 * 16 * 4]u8 = @splat(0);
    engine.writeTexture(tex, .{ .bytes_per_row = 64 }, .{
        .width = 16,
        .height = 16,
    }, &initial_pixels);

    // 5. createSampler
    const sampler = try engine.createSampler(SamplerDesc.pinned_bilinear_clamp);
    try std.testing.expect(sampler.isValid());

    // 6. createShaderModule
    const vs = try engine.createShaderModule(.{ .code = "vertex_main" });
    try std.testing.expect(vs.isValid());

    // 7. createRenderPipeline
    const pipeline = try engine.createRenderPipeline(.{
        .vertex_shader = vs,
        .color_target_format = .rgba8_unorm,
    });
    try std.testing.expect(pipeline.isValid());

    // 8. createCommandEncoder
    var encoder = try engine.createCommandEncoder();

    // 9. beginRenderPass
    var pass = try engine.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = tex,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
        },
    });

    // 10. setPipeline
    engine.setPipeline(&pass, pipeline);

    // 11. setVertexBuffer
    engine.setVertexBuffer(&pass, 0, buf, 0);

    // 12. setClipRect
    engine.setClipRect(&pass, .{ .x = 0, .y = 0, .width = 16, .height = 16 });

    // 13. draw
    engine.draw(&pass, 6, 1, 0, 0);

    // 14. endRenderPass
    engine.endRenderPass(&pass);

    // 15. finishCommandEncoder
    const cmd_buf = try engine.finishCommandEncoder(&encoder);
    try std.testing.expect(cmd_buf.isValid());

    // 16. submit
    engine.submit(&[_]MockBackend.CommandBuffer{cmd_buf});

    // 17. readbackTarget
    var readback: [16 * 16 * 4]u8 = undefined;
    try engine.readbackTarget(tex, &readback);

    // Pixel at (4, 4) should be red {255, 0, 0, 255}
    const idx = (4 * 16 + 4) * 4;
    try std.testing.expectEqual(@as(u8, 255), readback[idx + 0]);
    try std.testing.expectEqual(@as(u8, 0), readback[idx + 1]);
    try std.testing.expectEqual(@as(u8, 0), readback[idx + 2]);
    try std.testing.expectEqual(@as(u8, 255), readback[idx + 3]);

    // 18. destroyBuffer
    engine.destroyBuffer(buf);

    // 19. destroyTexture
    engine.destroyTexture(tex);
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(mock);
    _ = mock;
}
