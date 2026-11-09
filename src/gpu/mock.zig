//! ============================================================================
//! Klaxon Headless & Mock GPU Backend (`src/gpu/mock.zig`)
//! ============================================================================
//! Deterministic software rasterizer and command recorder implementing the 19 HAL
//! functions in static BSS memory with zero heap allocations (`0 malloc/free`).
//!
//! Features:
//! 1. Complete implementation of all 19 pinned HAL contract functions.
//! 2. Static BSS memory working sets (64-byte cache-line aligned).
//! 3. Software rasterization with scissor clip-rect boundary clamping.
//! 4. Bit-exact pixel readback for deterministic CI golden oracle validation.
//! 5. Freestanding compiler-fence compliant (zero libc, zero POSIX dependencies).
//!
//! Standards: TigerStyle (< 100 columns, zero trailing whitespace, paired asserts).

const std = @import("std");
const assert = std.debug.assert;
const hal = @import("hal.zig");

// ============================================================================
// 1. Static Capacity Limits & Memory Pools
// ============================================================================

pub const buffer_pool_capacity: u32 = 16;
pub const buffer_bytes_max: u32 = 65536; // 64 KB per buffer

pub const texture_pool_capacity: u32 = 8;
pub const texture_width_max: u32 = 800;
pub const texture_height_max: u32 = 600;
pub const texture_bytes_max: u32 = texture_width_max * texture_height_max * 4; // 1.92 MB

pub const sampler_pool_capacity: u32 = 8;
pub const shader_pool_capacity: u32 = 8;
pub const pipeline_pool_capacity: u32 = 8;

pub const command_buffer_capacity: u32 = 16;
pub const commands_per_buffer_max: u32 = 128;

// Static BSS Working Sets (Align 64)
var mock_buffer_data: [buffer_pool_capacity][buffer_bytes_max]u8 align(64) = undefined;
var mock_texture_data: [texture_pool_capacity][texture_bytes_max]u8 align(64) = undefined;

// ============================================================================
// 2. Vertex Primitive for 2D UI Rendering
// ============================================================================

/// Compact 12-byte 2D vertex matching UI batching layouts.
pub const Vertex2D = extern struct {
    pos: [2]f32,
    color: [4]u8, // RGBA8 unorm
};

comptime {
    assert(@sizeOf(Vertex2D) == 12);
    assert(@alignOf(Vertex2D) == 4);
    assert(@offsetOf(Vertex2D, "pos") == 0);
    assert(@offsetOf(Vertex2D, "color") == 8);
}

/// Helper generating two triangles (6 vertices) forming a 2D axis-aligned rectangle.
pub fn makeQuadVertices(x: f32, y: f32, w: f32, h: f32, color: [4]u8) [6]Vertex2D {
    assert(w >= 0.0);
    assert(h >= 0.0);
    return [6]Vertex2D{
        .{ .pos = .{ x, y }, .color = color },
        .{ .pos = .{ x + w, y }, .color = color },
        .{ .pos = .{ x + w, y + h }, .color = color },
        .{ .pos = .{ x + w, y + h }, .color = color },
        .{ .pos = .{ x, y + h }, .color = color },
        .{ .pos = .{ x, y }, .color = color },
    };
}

/// Mock test window representation.
pub const MockWindow = struct {
    width_px: u32 = 800,
    height_px: u32 = 600,
    title: []const u8 = "Klaxon Mock Window",
};

// ============================================================================
// 3. Command Encoding Data Structures
// ============================================================================

pub const CommandTag = enum(u8) {
    begin_render_pass = 0,
    set_pipeline = 1,
    set_vertex_buffer = 2,
    set_clip_rect = 3,
    draw = 4,
    end_render_pass = 5,
};

pub const DrawArgs = extern struct {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

pub const SetVertexBufferArgs = extern struct {
    slot: u32,
    buffer: hal.Buffer,
    offset: u64,
};

pub const Command = union(CommandTag) {
    begin_render_pass: hal.RenderPassDesc,
    set_pipeline: hal.RenderPipeline,
    set_vertex_buffer: SetVertexBufferArgs,
    set_clip_rect: hal.RectU32,
    draw: DrawArgs,
    end_render_pass: void,
};

// ============================================================================
// 4. Resource Slots & Metadata
// ============================================================================

const BufferSlot = struct {
    active: bool = false,
    generation: u32 = 1,
    desc: hal.BufferDesc = .{ .size = 0, .usage = .{} },
};

const TextureSlot = struct {
    active: bool = false,
    generation: u32 = 1,
    desc: hal.TextureDesc = .{ .size = .{ .width = 0, .height = 0 } },
};

const SamplerSlot = struct {
    active: bool = false,
    generation: u32 = 1,
    desc: hal.SamplerDesc = .{},
};

const ShaderSlot = struct {
    active: bool = false,
    generation: u32 = 1,
    stage: hal.ShaderStage = .vertex,
    code_len: u32 = 0,
};

const PipelineSlot = struct {
    active: bool = false,
    generation: u32 = 1,
    desc: hal.RenderPipelineDesc = .{
        .vertex_shader = hal.ShaderModule.null_handle,
    },
};

const CommandBufferRecord = struct {
    active: bool = false,
    generation: u32 = 1,
    command_count: u32 = 0,
    commands: [commands_per_buffer_max]Command = undefined,
};

// ============================================================================
// 5. MockBackend Namespace & Context Types
// ============================================================================

pub const MockBackend = struct {
    pub const Buffer = hal.Buffer;
    pub const Texture = hal.Texture;
    pub const Sampler = hal.Sampler;
    pub const ShaderModule = hal.ShaderModule;
    pub const RenderPipeline = hal.RenderPipeline;
    pub const CommandBuffer = hal.CommandBuffer;

    /// Mock GPU Device holding resource allocation tables.
    pub const Device = struct {
        buffers: [buffer_pool_capacity]BufferSlot = @splat(.{}),
        textures: [texture_pool_capacity]TextureSlot = @splat(.{}),
        samplers: [sampler_pool_capacity]SamplerSlot = @splat(.{}),
        shaders: [shader_pool_capacity]ShaderSlot = @splat(.{}),
        pipelines: [pipeline_pool_capacity]PipelineSlot = @splat(.{}),
        command_buffers: [command_buffer_capacity]CommandBufferRecord = @splat(.{}),

        pub fn init(self: *Device) void {
            self.* = .{};
        }

        pub fn getBufferData(self: *const Device, handle: hal.Buffer) ?[]u8 {
            if (handle.id == 0 or handle.id > buffer_pool_capacity) return null;
            const idx = handle.id - 1;
            const slot = &self.buffers[idx];
            if (!slot.active or slot.generation != handle.generation) return null;
            const valid_len: usize = @intCast(@min(slot.desc.size, buffer_bytes_max));
            return mock_buffer_data[idx][0..valid_len];
        }

        pub fn getTextureData(self: *const Device, handle: hal.Texture) ?[]u8 {
            if (handle.id == 0 or handle.id > texture_pool_capacity) return null;
            const idx = handle.id - 1;
            const slot = &self.textures[idx];
            if (!slot.active or slot.generation != handle.generation) return null;
            const total_bytes = slot.desc.size.width * slot.desc.size.height *
                slot.desc.format.bytesPerPixel();
            if (total_bytes > texture_bytes_max) return null;
            return mock_texture_data[idx][0..total_bytes];
        }
    };

    /// Mock command submission queue.
    pub const Queue = struct {
        device: *Device,

        pub fn init(device: *Device) Queue {
            return .{ .device = device };
        }
    };

    /// Command recording state.
    pub const CommandEncoder = struct {
        device: *Device,
        command_count: u32 = 0,
        commands: [commands_per_buffer_max]Command = undefined,
        has_overflowed: bool = false,
        in_render_pass: bool = false,
        is_finished: bool = false,

        pub fn record(self: *CommandEncoder, cmd: Command) !void {
            if (self.is_finished) return hal.GpuError.InvalidCommandSequence;
            if (self.command_count >= commands_per_buffer_max) {
                self.has_overflowed = true;
                return hal.GpuError.PoolExhausted;
            }
            self.commands[self.command_count] = cmd;
            self.command_count += 1;
        }
    };

    /// Active render pass state and encoder wrapper.
    pub const RenderPassEncoder = struct {
        encoder: *CommandEncoder,
        pass_desc: hal.RenderPassDesc,
        current_pipeline: ?hal.RenderPipeline = null,
        bound_vertex_buffer: ?hal.Buffer = null,
        bound_vertex_offset: u64 = 0,
        current_clip_rect: ?hal.RectU32 = null,
        is_ended: bool = false,
    };

    // ========================================================================
    // 6. The 19 Pinned Contract Functions
    // ========================================================================

    /// 1. Allocates a GPU memory buffer within the static pool.
    pub fn createBuffer(device: *Device, desc: hal.BufferDesc) !hal.Buffer {
        assert(desc.size > 0);
        if (desc.size == 0) return hal.GpuError.InvalidDesc;
        if (desc.size > buffer_bytes_max) return hal.GpuError.BufferTooSmall;

        var i: u32 = 0;
        while (i < buffer_pool_capacity) : (i += 1) {
            if (!device.buffers[i].active) {
                device.buffers[i].active = true;
                device.buffers[i].desc = desc;
                const valid_len: usize = @intCast(@min(desc.size, buffer_bytes_max));
                @memset(mock_buffer_data[i][0..valid_len], 0);
                const handle = hal.Buffer{
                    .id = i + 1,
                    .generation = device.buffers[i].generation,
                };
                assert(handle.isValid());
                return handle;
            }
        }
        return hal.GpuError.PoolExhausted;
    }

    /// 2. Destroys an existing GPU buffer, returning its slot to the pool.
    pub fn destroyBuffer(device: *Device, buffer: hal.Buffer) void {
        if (buffer.id == 0 or buffer.id > buffer_pool_capacity) return;
        const idx = buffer.id - 1;
        if (device.buffers[idx].active and
            device.buffers[idx].generation == buffer.generation)
        {
            device.buffers[idx].active = false;
            device.buffers[idx].generation +%= 1;
            if (device.buffers[idx].generation == 0) device.buffers[idx].generation = 1;
        }
    }

    /// 3. Uploads streaming data into a GPU buffer at the specified byte offset.
    pub fn writeBuffer(
        queue: *Queue,
        buffer: hal.Buffer,
        offset: u64,
        data: []const u8,
    ) void {
        assert(buffer.isValid());
        if (data.len == 0) return;
        const target_slice = queue.device.getBufferData(buffer) orelse return;
        const end_offset = std.math.add(u64, offset, data.len) catch return;
        assert(end_offset <= target_slice.len);
        if (end_offset <= target_slice.len) {
            const start: usize = @intCast(offset);
            const end: usize = @intCast(end_offset);
            @memcpy(target_slice[start..end], data);
        }
    }

    /// 4. Allocates a GPU texture image within the static texture pool.
    pub fn createTexture(device: *Device, desc: hal.TextureDesc) !hal.Texture {
        assert(desc.size.width > 0);
        assert(desc.size.height > 0);
        if (desc.size.width == 0 or desc.size.height == 0) {
            return hal.GpuError.InvalidDesc;
        }
        if (desc.size.width > texture_width_max or desc.size.height > texture_height_max) {
            return hal.GpuError.InvalidDesc;
        }

        var i: u32 = 0;
        while (i < texture_pool_capacity) : (i += 1) {
            if (!device.textures[i].active) {
                device.textures[i].active = true;
                device.textures[i].desc = desc;
                const total_bytes = desc.size.width * desc.size.height *
                    desc.format.bytesPerPixel();
                @memset(mock_texture_data[i][0..total_bytes], 0);

                const handle = hal.Texture{
                    .id = i + 1,
                    .generation = device.textures[i].generation,
                };
                assert(handle.isValid());
                return handle;
            }
        }
        return hal.GpuError.PoolExhausted;
    }

    /// 5. Destroys an existing GPU texture image, returning its slot to the pool.
    pub fn destroyTexture(device: *Device, texture: hal.Texture) void {
        if (texture.id == 0 or texture.id > texture_pool_capacity) return;
        const idx = texture.id - 1;
        if (device.textures[idx].active and
            device.textures[idx].generation == texture.generation)
        {
            device.textures[idx].active = false;
            device.textures[idx].generation +%= 1;
            if (device.textures[idx].generation == 0) device.textures[idx].generation = 1;
        }
    }

    /// 6. Writes linear pixel data into a 2D texture respecting pitch and row bounds.
    pub fn writeTexture(
        queue: *Queue,
        texture: hal.Texture,
        layout: hal.TextureDataLayout,
        size: hal.Extent3D,
        data: []const u8,
    ) void {
        assert(texture.isValid());
        assert(size.width > 0);
        assert(size.height > 0);

        const dest_slice = queue.device.getTextureData(texture) orelse return;
        const idx = texture.id - 1;
        const tex_desc = queue.device.textures[idx].desc;
        const bpp = tex_desc.format.bytesPerPixel();
        const copy_row_bytes = size.width * bpp;

        if (size.width > tex_desc.size.width or size.height > tex_desc.size.height) return;

        var y: u32 = 0;
        while (y < size.height) : (y += 1) {
            const row_src_off = std.math.mul(u64, y, layout.bytes_per_row) catch continue;
            const src_offset_u64 = std.math.add(u64, layout.offset, row_src_off) catch continue;
            const src_end_u64 = std.math.add(u64, src_offset_u64, copy_row_bytes) catch continue;

            const row_dest_off = std.math.mul(
                u64,
                y,
                @as(u64, tex_desc.size.width) * bpp,
            ) catch continue;
            const dest_end_u64 = std.math.add(u64, row_dest_off, copy_row_bytes) catch continue;

            if (src_end_u64 <= data.len and dest_end_u64 <= dest_slice.len) {
                const src_start: usize = @intCast(src_offset_u64);
                const dest_start: usize = @intCast(row_dest_off);
                @memcpy(
                    dest_slice[dest_start .. dest_start + copy_row_bytes],
                    data[src_start .. src_start + copy_row_bytes],
                );
            }
        }
    }

    /// 7. Creates a texture sampler, strictly enforcing the pinned zero-mipmap invariant.
    pub fn createSampler(device: *Device, desc: hal.SamplerDesc) !hal.Sampler {
        // Pinned TigerStyle invariant per AGENTS.md: bilinear, clamp-to-edge, zero mipmaps
        assert(desc.mipmap_filter == .none);
        assert(desc.mag_filter == .linear);
        assert(desc.min_filter == .linear);
        assert(desc.address_mode_u == .clamp_to_edge);
        assert(desc.address_mode_v == .clamp_to_edge);
        assert(desc.address_mode_w == .clamp_to_edge);
        if (desc.mipmap_filter != .none or
            desc.mag_filter != .linear or
            desc.min_filter != .linear or
            desc.address_mode_u != .clamp_to_edge or
            desc.address_mode_v != .clamp_to_edge or
            desc.address_mode_w != .clamp_to_edge)
        {
            return hal.GpuError.InvalidDesc;
        }

        var i: u32 = 0;
        while (i < sampler_pool_capacity) : (i += 1) {
            if (!device.samplers[i].active) {
                device.samplers[i].active = true;
                device.samplers[i].desc = desc;
                const handle = hal.Sampler{
                    .id = i + 1,
                    .generation = device.samplers[i].generation,
                };
                assert(handle.isValid());
                return handle;
            }
        }
        return hal.GpuError.PoolExhausted;
    }

    /// 8. Creates an immutable compiled shader module.
    pub fn createShaderModule(device: *Device, desc: hal.ShaderModuleDesc) !hal.ShaderModule {
        assert(desc.code.len > 0);
        if (desc.code.len == 0) return hal.GpuError.InvalidDesc;

        var i: u32 = 0;
        while (i < shader_pool_capacity) : (i += 1) {
            if (!device.shaders[i].active) {
                device.shaders[i].active = true;
                device.shaders[i].stage = desc.stage;
                device.shaders[i].code_len = @intCast(@min(desc.code.len, 4096));
                const handle = hal.ShaderModule{
                    .id = i + 1,
                    .generation = device.shaders[i].generation,
                };
                assert(handle.isValid());
                return handle;
            }
        }
        return hal.GpuError.PoolExhausted;
    }

    /// 9. Creates an immutable render pipeline state.
    pub fn createRenderPipeline(
        device: *Device,
        desc: hal.RenderPipelineDesc,
    ) !hal.RenderPipeline {
        assert(desc.vertex_shader.isValid());
        if (desc.vertex_shader.id == 0 or desc.vertex_shader.id > shader_pool_capacity) {
            return hal.GpuError.InvalidHandle;
        }
        const vs_slot = &device.shaders[desc.vertex_shader.id - 1];
        if (!vs_slot.active or vs_slot.generation != desc.vertex_shader.generation) {
            return hal.GpuError.InvalidHandle;
        }
        if (desc.fragment_shader) |fs| {
            if (fs.id == 0 or fs.id > shader_pool_capacity) {
                return hal.GpuError.InvalidHandle;
            }
            const fs_slot = &device.shaders[fs.id - 1];
            if (!fs_slot.active or fs_slot.generation != fs.generation) {
                return hal.GpuError.InvalidHandle;
            }
        }

        var i: u32 = 0;
        while (i < pipeline_pool_capacity) : (i += 1) {
            if (!device.pipelines[i].active) {
                device.pipelines[i].active = true;
                device.pipelines[i].desc = desc;
                const handle = hal.RenderPipeline{
                    .id = i + 1,
                    .generation = device.pipelines[i].generation,
                };
                assert(handle.isValid());
                return handle;
            }
        }
        return hal.GpuError.PoolExhausted;
    }

    /// 10. Allocates a new command encoder for recording render commands.
    pub fn createCommandEncoder(device: *Device) !CommandEncoder {
        return CommandEncoder{
            .device = device,
            .command_count = 0,
            .has_overflowed = false,
        };
    }

    /// 11. Initiates an active render pass targeting a texture attachment.
    pub fn beginRenderPass(
        encoder: *CommandEncoder,
        desc: hal.RenderPassDesc,
    ) !RenderPassEncoder {
        assert(desc.color_attachment.target.isValid());
        if (encoder.is_finished or encoder.in_render_pass) {
            return hal.GpuError.InvalidCommandSequence;
        }
        const target = desc.color_attachment.target;
        if (target.id == 0 or target.id > texture_pool_capacity) {
            return hal.GpuError.InvalidHandle;
        }
        const slot = &encoder.device.textures[target.id - 1];
        if (!slot.active or slot.generation != target.generation) {
            return hal.GpuError.InvalidHandle;
        }

        try encoder.record(.{ .begin_render_pass = desc });
        encoder.in_render_pass = true;
        return RenderPassEncoder{
            .encoder = encoder,
            .pass_desc = desc,
        };
    }

    /// 12. Binds the active render pipeline for subsequent draw invocations.
    pub fn setPipeline(pass: *RenderPassEncoder, pipeline: hal.RenderPipeline) void {
        assert(pipeline.isValid());
        if (pass.is_ended) return;
        pass.current_pipeline = pipeline;
        pass.encoder.record(.{ .set_pipeline = pipeline }) catch return;
    }

    /// 13. Binds a vertex buffer slot and byte offset.
    pub fn setVertexBuffer(
        pass: *RenderPassEncoder,
        slot: u32,
        buffer: hal.Buffer,
        offset: u64,
    ) void {
        assert(buffer.isValid());
        if (pass.is_ended) return;
        pass.bound_vertex_buffer = buffer;
        pass.bound_vertex_offset = offset;
        pass.encoder.record(.{
            .set_vertex_buffer = .{
                .slot = slot,
                .buffer = buffer,
                .offset = offset,
            },
        }) catch return;
    }

    /// 14. Sets the active scissor clip rectangle for geometry clipping.
    pub fn setClipRect(pass: *RenderPassEncoder, rect: hal.RectU32) void {
        if (pass.is_ended) return;
        pass.current_clip_rect = rect;
        pass.encoder.record(.{ .set_clip_rect = rect }) catch return;
    }

    /// 15. Records a geometric draw command into the active render pass.
    pub fn draw(
        pass: *RenderPassEncoder,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) void {
        assert(!pass.is_ended);
        if (pass.is_ended) return;
        if (vertex_count == 0 or instance_count == 0) return;
        pass.encoder.record(.{
            .draw = .{
                .vertex_count = vertex_count,
                .instance_count = instance_count,
                .first_vertex = first_vertex,
                .first_instance = first_instance,
            },
        }) catch return;
    }

    /// 16. Concludes the active render pass.
    pub fn endRenderPass(pass: *RenderPassEncoder) void {
        assert(!pass.is_ended);
        if (pass.is_ended) return;
        pass.is_ended = true;
        pass.encoder.in_render_pass = false;
        pass.encoder.record(.{ .end_render_pass = {} }) catch return;
    }

    /// 17. Finalizes command recording and returns an executable CommandBuffer.
    pub fn finishCommandEncoder(encoder: *CommandEncoder) !hal.CommandBuffer {
        if (encoder.is_finished or encoder.in_render_pass) {
            return hal.GpuError.InvalidCommandSequence;
        }
        if (encoder.has_overflowed) {
            return hal.GpuError.PoolExhausted;
        }
        encoder.is_finished = true;

        var i: u32 = 0;
        while (i < command_buffer_capacity) : (i += 1) {
            if (!encoder.device.command_buffers[i].active) {
                encoder.device.command_buffers[i].active = true;
                encoder.device.command_buffers[i].command_count = encoder.command_count;
                @memcpy(
                    encoder.device.command_buffers[i].commands[0..encoder.command_count],
                    encoder.commands[0..encoder.command_count],
                );
                const handle = hal.CommandBuffer{
                    .id = i + 1,
                    .generation = encoder.device.command_buffers[i].generation,
                };
                assert(handle.isValid());
                return handle;
            }
        }
        return hal.GpuError.PoolExhausted;
    }

    /// 18. Submits recorded command buffers for deterministic software execution.
    pub fn submit(queue: *Queue, commands: []const hal.CommandBuffer) void {
        for (commands) |cb_handle| {
            if (!cb_handle.isValid() or cb_handle.id > command_buffer_capacity) continue;
            const cb_idx = cb_handle.id - 1;
            const rec = &queue.device.command_buffers[cb_idx];
            if (!rec.active or rec.generation != cb_handle.generation) continue;

            executeCommandBuffer(queue.device, rec);

            // Reclaim command buffer slot and bump generation
            rec.active = false;
            rec.generation +%= 1;
            if (rec.generation == 0) rec.generation = 1;
        }
    }

    /// 19. Synchronously reads back rendered pixel data from a target texture.
    pub fn readbackTarget(
        device: *Device,
        target: hal.Texture,
        dest: []u8,
    ) !void {
        assert(target.isValid());
        const pixels = device.getTextureData(target) orelse return hal.GpuError.TargetNotFound;
        if (dest.len < pixels.len) return hal.GpuError.BufferTooSmall;

        @memcpy(dest[0..pixels.len], pixels);
    }

    // ========================================================================
    // 7. Software Rasterization Execution Core
    // ========================================================================

    fn blendPixelRgba(dst: []u8, src_r: u8, src_g: u8, src_b: u8, alpha: u8) void {
        const inv_a: u32 = 255 - alpha;
        const a: u32 = alpha;
        const dst_r: u32 = dst[0];
        const dst_g: u32 = dst[1];
        const dst_b: u32 = dst[2];
        const dst_a: u32 = dst[3];

        const out_a: u32 = a + (dst_a * inv_a + 127) / 255;
        if (out_a == 0) return;

        if (dst_a == 255) {
            dst[0] = @intCast((@as(u32, src_r) * a + dst_r * inv_a) / 255);
            dst[1] = @intCast((@as(u32, src_g) * a + dst_g * inv_a) / 255);
            dst[2] = @intCast((@as(u32, src_b) * a + dst_b * inv_a) / 255);
            dst[3] = 255;
        } else {
            const r_num = @as(u32, src_r) * a + (dst_r * dst_a * inv_a) / 255;
            const g_num = @as(u32, src_g) * a + (dst_g * dst_a * inv_a) / 255;
            const b_num = @as(u32, src_b) * a + (dst_b * dst_a * inv_a) / 255;
            dst[0] = @intCast(@min(255, (r_num + out_a / 2) / out_a));
            dst[1] = @intCast(@min(255, (g_num + out_a / 2) / out_a));
            dst[2] = @intCast(@min(255, (b_num + out_a / 2) / out_a));
            dst[3] = @intCast(@min(255, out_a));
        }
    }

    fn blendPixelBgra(dst: []u8, src_r: u8, src_g: u8, src_b: u8, alpha: u8) void {
        const inv_a: u32 = 255 - alpha;
        const a: u32 = alpha;
        const dst_b: u32 = dst[0];
        const dst_g: u32 = dst[1];
        const dst_r: u32 = dst[2];
        const dst_a: u32 = dst[3];

        const out_a: u32 = a + (dst_a * inv_a + 127) / 255;
        if (out_a == 0) return;

        if (dst_a == 255) {
            dst[0] = @intCast((@as(u32, src_b) * a + dst_b * inv_a) / 255);
            dst[1] = @intCast((@as(u32, src_g) * a + dst_g * inv_a) / 255);
            dst[2] = @intCast((@as(u32, src_r) * a + dst_r * inv_a) / 255);
            dst[3] = 255;
        } else {
            const b_num = @as(u32, src_b) * a + (dst_b * dst_a * inv_a) / 255;
            const g_num = @as(u32, src_g) * a + (dst_g * dst_a * inv_a) / 255;
            const r_num = @as(u32, src_r) * a + (dst_r * dst_a * inv_a) / 255;
            dst[0] = @intCast(@min(255, (b_num + out_a / 2) / out_a));
            dst[1] = @intCast(@min(255, (g_num + out_a / 2) / out_a));
            dst[2] = @intCast(@min(255, (r_num + out_a / 2) / out_a));
            dst[3] = @intCast(@min(255, out_a));
        }
    }

    fn executeCommandBuffer(device: *Device, rec: *const CommandBufferRecord) void {
        var current_target: ?hal.Texture = null;
        var active_clip: ?hal.RectU32 = null;
        var bound_buffer: ?hal.Buffer = null;
        var bound_offset: u64 = 0;

        var cmd_i: u32 = 0;
        while (cmd_i < rec.command_count) : (cmd_i += 1) {
            switch (rec.commands[cmd_i]) {
                .begin_render_pass => |pass_desc| {
                    current_target = pass_desc.color_attachment.target;
                    const target = current_target.?;
                    if (target.id == 0 or target.id > texture_pool_capacity) {
                        current_target = null;
                        continue;
                    }
                    const target_desc = device.textures[target.id - 1].desc;
                    if (pass_desc.color_attachment.load_op == .clear) {
                        const pixels = device.getTextureData(target) orelse continue;
                        const c = pass_desc.color_attachment.clear_color;
                        const r: u8 = @intFromFloat(std.math.clamp(c[0], 0.0, 1.0) * 255.0);
                        const g: u8 = @intFromFloat(std.math.clamp(c[1], 0.0, 1.0) * 255.0);
                        const b: u8 = @intFromFloat(std.math.clamp(c[2], 0.0, 1.0) * 255.0);
                        const a: u8 = @intFromFloat(std.math.clamp(c[3], 0.0, 1.0) * 255.0);

                        switch (target_desc.format) {
                            .r8_unorm => {
                                @memset(pixels, r);
                            },
                            .rgba8_unorm => {
                                var px_i: usize = 0;
                                while (px_i + 4 <= pixels.len) : (px_i += 4) {
                                    pixels[px_i + 0] = r;
                                    pixels[px_i + 1] = g;
                                    pixels[px_i + 2] = b;
                                    pixels[px_i + 3] = a;
                                }
                            },
                            .bgra8_unorm => {
                                var px_i: usize = 0;
                                while (px_i + 4 <= pixels.len) : (px_i += 4) {
                                    pixels[px_i + 0] = b;
                                    pixels[px_i + 1] = g;
                                    pixels[px_i + 2] = r;
                                    pixels[px_i + 3] = a;
                                }
                            },
                            .depth32_float => {
                                @memset(pixels, 0);
                            },
                        }
                    }
                    active_clip = hal.RectU32{
                        .x = 0,
                        .y = 0,
                        .width = target_desc.size.width,
                        .height = target_desc.size.height,
                    };
                },
                .set_pipeline => {},
                .set_vertex_buffer => |args| {
                    bound_buffer = args.buffer;
                    bound_offset = args.offset;
                },
                .set_clip_rect => |rect| {
                    if (current_target) |target| {
                        if (target.id > 0 and target.id <= texture_pool_capacity) {
                            const t_desc = device.textures[target.id - 1].desc;
                            active_clip = rect.clampToBounds(t_desc.size.width, t_desc.size.height);
                        } else {
                            active_clip = rect;
                        }
                    } else {
                        active_clip = rect;
                    }
                },
                .draw => |args| {
                    if (current_target == null or bound_buffer == null) continue;
                    if (args.instance_count == 0 or args.vertex_count == 0) continue;
                    const target = current_target.?;
                    const buf_bytes = device.getBufferData(bound_buffer.?) orelse continue;
                    const target_pixels = device.getTextureData(target) orelse continue;
                    const t_desc = device.textures[target.id - 1].desc;

                    var inst_i: u32 = 0;
                    while (inst_i < args.instance_count) : (inst_i += 1) {
                        rasterizeGeometry(
                            t_desc.size.width,
                            t_desc.size.height,
                            t_desc.format,
                            target_pixels,
                            active_clip orelse hal.RectU32{
                                .x = 0,
                                .y = 0,
                                .width = t_desc.size.width,
                                .height = t_desc.size.height,
                            },
                            buf_bytes,
                            bound_offset,
                            args.vertex_count,
                            args.first_vertex,
                        );
                    }
                },
                .end_render_pass => {
                    current_target = null;
                    active_clip = null;
                },
            }
        }
    }

    /// Rasterizes a list of triangles in Vertex2D format, applying scissor clipping.
    fn rasterizeGeometry(
        width_px: u32,
        height_px: u32,
        format: hal.TextureFormat,
        dest_pixels: []u8,
        clip: hal.RectU32,
        vertex_bytes: []const u8,
        byte_offset: u64,
        vertex_count: u32,
        first_vertex: u32,
    ) void {
        const safe_clip = clip.clampToBounds(width_px, height_px);
        if (safe_clip.isEmpty()) return;
        const v_size: usize = @sizeOf(Vertex2D);
        if (byte_offset > vertex_bytes.len) return;
        const first_v_bytes = std.math.mul(usize, first_vertex, v_size) catch return;
        const start_offset = std.math.add(
            usize,
            @as(usize, @intCast(byte_offset)),
            first_v_bytes,
        ) catch return;
        const required_bytes = std.math.mul(usize, vertex_count, v_size) catch return;
        const total_needed = std.math.add(usize, start_offset, required_bytes) catch return;
        if (total_needed > vertex_bytes.len) return;

        const vertices_raw = vertex_bytes[start_offset .. start_offset + required_bytes];
        const triangle_count = vertex_count / 3;

        var tri_i: u32 = 0;
        while (tri_i < triangle_count) : (tri_i += 1) {
            const v0_off = tri_i * 3 * v_size;
            const v1_off = (tri_i * 3 + 1) * v_size;
            const v2_off = (tri_i * 3 + 2) * v_size;

            const v0_ptr: *align(1) const Vertex2D = @ptrCast(&vertices_raw[v0_off]);
            const v1_ptr: *align(1) const Vertex2D = @ptrCast(&vertices_raw[v1_off]);
            const v2_ptr: *align(1) const Vertex2D = @ptrCast(&vertices_raw[v2_off]);

            rasterizeTriangle(
                width_px,
                height_px,
                format,
                dest_pixels,
                safe_clip,
                v0_ptr.*,
                v1_ptr.*,
                v2_ptr.*,
            );
        }
    }

    /// Scissor-clipped barycentric rasterizer for a single 2D colored triangle.
    fn rasterizeTriangle(
        width_px: u32,
        height_px: u32,
        format: hal.TextureFormat,
        dest_pixels: []u8,
        clip: hal.RectU32,
        v0: Vertex2D,
        v1: Vertex2D,
        v2: Vertex2D,
    ) void {
        assert(width_px > 0);
        assert(height_px > 0);

        // Guard against NaN and infinite coordinates
        if (!std.math.isFinite(v0.pos[0]) or !std.math.isFinite(v0.pos[1]) or
            !std.math.isFinite(v1.pos[0]) or !std.math.isFinite(v1.pos[1]) or
            !std.math.isFinite(v2.pos[0]) or !std.math.isFinite(v2.pos[1]))
        {
            return;
        }

        // Signed 2D area (determinant). If zero or near-zero, triangle is degenerate.
        const det = (v1.pos[0] - v0.pos[0]) * (v2.pos[1] - v0.pos[1]) -
            (v1.pos[1] - v0.pos[1]) * (v2.pos[0] - v0.pos[0]);
        if (@abs(det) < 1e-6) return;

        const max_w_f = @as(f32, @floatFromInt(width_px));
        const max_h_f = @as(f32, @floatFromInt(height_px));

        const min_x_f = @min(v0.pos[0], @min(v1.pos[0], v2.pos[0]));
        const max_x_f = @max(v0.pos[0], @max(v1.pos[0], v2.pos[0]));
        const min_y_f = @min(v0.pos[1], @min(v1.pos[1], v2.pos[1]));
        const max_y_f = @max(v0.pos[1], @max(v1.pos[1], v2.pos[1]));

        if (max_x_f < 0.0 or min_x_f >= max_w_f or max_y_f < 0.0 or min_y_f >= max_h_f) {
            return;
        }

        const clamped_min_x: u32 = @intFromFloat(@max(0.0, @floor(min_x_f)));
        const clamped_max_x: u32 = @intFromFloat(@min(max_w_f, @ceil(max_x_f)));
        const clamped_min_y: u32 = @intFromFloat(@max(0.0, @floor(min_y_f)));
        const clamped_max_y: u32 = @intFromFloat(@min(max_h_f, @ceil(max_y_f)));

        const clip_x0 = clip.x;
        const clip_y0 = clip.y;
        const clip_x1 = clip.x + clip.width;
        const clip_y1 = clip.y + clip.height;

        const start_x = @max(clip_x0, clamped_min_x);
        const end_x = @min(clip_x1, clamped_max_x);
        const start_y = @max(clip_y0, clamped_min_y);
        const end_y = @min(clip_y1, clamped_max_y);

        if (start_x >= end_x or start_y >= end_y) return;

        const inv_det = 1.0 / det;
        const is_ccw = (det > 0.0);

        // Top-left rule edge classification for seamless shared boundary rendering
        const dx0 = v1.pos[0] - v0.pos[0];
        const dy0 = v1.pos[1] - v0.pos[1];
        const dx1 = v2.pos[0] - v1.pos[0];
        const dy1 = v2.pos[1] - v1.pos[1];
        const dx2 = v0.pos[0] - v2.pos[0];
        const dy2 = v0.pos[1] - v2.pos[1];

        const tl0 = if (is_ccw)
            (dy0 > 0.0 or (dy0 == 0.0 and dx0 < 0.0))
        else
            (dy0 < 0.0 or (dy0 == 0.0 and dx0 > 0.0));

        const tl1 = if (is_ccw)
            (dy1 > 0.0 or (dy1 == 0.0 and dx1 < 0.0))
        else
            (dy1 < 0.0 or (dy1 == 0.0 and dx1 > 0.0));

        const tl2 = if (is_ccw)
            (dy2 > 0.0 or (dy2 == 0.0 and dx2 < 0.0))
        else
            (dy2 < 0.0 or (dy2 == 0.0 and dx2 > 0.0));

        var y = start_y;
        while (y < end_y) : (y += 1) {
            const py = @as(f32, @floatFromInt(y)) + 0.5;
            var x = start_x;
            while (x < end_x) : (x += 1) {
                const px = @as(f32, @floatFromInt(x)) + 0.5;

                const e0 = (v1.pos[0] - v0.pos[0]) * (py - v0.pos[1]) -
                    (v1.pos[1] - v0.pos[1]) * (px - v0.pos[0]);
                const e1 = (v2.pos[0] - v1.pos[0]) * (py - v1.pos[1]) -
                    (v2.pos[1] - v1.pos[1]) * (px - v1.pos[0]);
                const e2 = (v0.pos[0] - v2.pos[0]) * (py - v2.pos[1]) -
                    (v0.pos[1] - v2.pos[1]) * (px - v2.pos[0]);

                const in0 = if (is_ccw)
                    (if (tl0) e0 >= 0.0 else e0 > 0.0)
                else
                    (if (tl0) e0 <= 0.0 else e0 < 0.0);

                const in1 = if (is_ccw)
                    (if (tl1) e1 >= 0.0 else e1 > 0.0)
                else
                    (if (tl1) e1 <= 0.0 else e1 < 0.0);

                const in2 = if (is_ccw)
                    (if (tl2) e2 >= 0.0 else e2 > 0.0)
                else
                    (if (tl2) e2 <= 0.0 else e2 < 0.0);

                const inside = in0 and in1 and in2;

                if (inside) {
                    const w0 = e1 * inv_det;
                    const w1 = e2 * inv_det;
                    const w2 = e0 * inv_det;

                    const r_f = w0 * @as(f32, @floatFromInt(v0.color[0])) +
                        w1 * @as(f32, @floatFromInt(v1.color[0])) +
                        w2 * @as(f32, @floatFromInt(v2.color[0]));
                    const g_f = w0 * @as(f32, @floatFromInt(v0.color[1])) +
                        w1 * @as(f32, @floatFromInt(v1.color[1])) +
                        w2 * @as(f32, @floatFromInt(v2.color[1]));
                    const b_f = w0 * @as(f32, @floatFromInt(v0.color[2])) +
                        w1 * @as(f32, @floatFromInt(v1.color[2])) +
                        w2 * @as(f32, @floatFromInt(v2.color[2]));
                    const a_f = w0 * @as(f32, @floatFromInt(v0.color[3])) +
                        w1 * @as(f32, @floatFromInt(v1.color[3])) +
                        w2 * @as(f32, @floatFromInt(v2.color[3]));

                    const cur_r: u8 = @intFromFloat(@round(std.math.clamp(r_f, 0.0, 255.0)));
                    const cur_g: u8 = @intFromFloat(@round(std.math.clamp(g_f, 0.0, 255.0)));
                    const cur_b: u8 = @intFromFloat(@round(std.math.clamp(b_f, 0.0, 255.0)));
                    const alpha: u8 = @intFromFloat(@round(std.math.clamp(a_f, 0.0, 255.0)));

                    switch (format) {
                        .rgba8_unorm => {
                            const px_idx = (@as(usize, y) * width_px + x) * 4;
                            if (px_idx + 4 <= dest_pixels.len) {
                                if (alpha == 255) {
                                    dest_pixels[px_idx + 0] = cur_r;
                                    dest_pixels[px_idx + 1] = cur_g;
                                    dest_pixels[px_idx + 2] = cur_b;
                                    dest_pixels[px_idx + 3] = 255;
                                } else if (alpha > 0) {
                                    blendPixelRgba(
                                        dest_pixels[px_idx .. px_idx + 4],
                                        cur_r,
                                        cur_g,
                                        cur_b,
                                        alpha,
                                    );
                                }
                            }
                        },
                        .bgra8_unorm => {
                            const px_idx = (@as(usize, y) * width_px + x) * 4;
                            if (px_idx + 4 <= dest_pixels.len) {
                                if (alpha == 255) {
                                    dest_pixels[px_idx + 0] = cur_b;
                                    dest_pixels[px_idx + 1] = cur_g;
                                    dest_pixels[px_idx + 2] = cur_r;
                                    dest_pixels[px_idx + 3] = 255;
                                } else if (alpha > 0) {
                                    blendPixelBgra(
                                        dest_pixels[px_idx .. px_idx + 4],
                                        cur_r,
                                        cur_g,
                                        cur_b,
                                        alpha,
                                    );
                                }
                            }
                        },
                        .r8_unorm => {
                            const px_idx = @as(usize, y) * width_px + x;
                            if (px_idx < dest_pixels.len) {
                                dest_pixels[px_idx] = cur_r;
                            }
                        },
                        .depth32_float => {},
                    }
                }
            }
        }
    }
};

// ============================================================================
// 8. Comprehensive Unit Tests & Validation Suite
// ============================================================================

test "MockBackend: Buffer lifecycle and capacity fences" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    const desc = hal.BufferDesc{
        .size = 1024,
        .usage = hal.BufferUsage.vertex_dst,
    };
    const b1 = try MockBackend.createBuffer(&device, desc);
    try std.testing.expect(b1.isValid());

    MockBackend.destroyBuffer(&device, b1);

    // Reuse slot with incremented generation
    const b2 = try MockBackend.createBuffer(&device, desc);
    try std.testing.expectEqual(b1.id, b2.id);
    try std.testing.expectEqual(b1.generation + 1, b2.generation);

    MockBackend.destroyBuffer(&device, b2);
}

test "MockBackend: Streaming buffer writes at offsets" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const b = try MockBackend.createBuffer(&device, .{
        .size = 256,
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, b);

    const chunk1 = [_]u8{ 0xAA, 0xBB, 0xCC, 0xDD };
    const chunk2 = [_]u8{ 0x11, 0x22, 0x33, 0x44 };

    MockBackend.writeBuffer(&queue, b, 0, &chunk1);
    MockBackend.writeBuffer(&queue, b, 4, &chunk2);

    const data = device.getBufferData(b).?;
    try std.testing.expectEqualSlices(u8, &chunk1, data[0..4]);
    try std.testing.expectEqualSlices(u8, &chunk2, data[4..8]);
}

test "MockBackend: Texture lifecycle, row-pitch writes, and destruction" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const tex = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 4, .height = 4 },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, tex);

    // 4x4 RGBA8 red texture
    var red_pixels: [4 * 4 * 4]u8 = undefined;
    var i: usize = 0;
    while (i < red_pixels.len) : (i += 4) {
        red_pixels[i + 0] = 255;
        red_pixels[i + 1] = 0;
        red_pixels[i + 2] = 0;
        red_pixels[i + 3] = 255;
    }

    MockBackend.writeTexture(&queue, tex, .{ .bytes_per_row = 16 }, .{
        .width = 4,
        .height = 4,
    }, &red_pixels);

    var readback_buf: [4 * 4 * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, tex, &readback_buf);
    try std.testing.expectEqualSlices(u8, &red_pixels, &readback_buf);
}

test "MockBackend: Sampler pinned invariant assertion" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    const valid_sampler = try MockBackend.createSampler(
        &device,
        hal.SamplerDesc.pinned_bilinear_clamp,
    );
    try std.testing.expect(valid_sampler.isValid());
}

test "MockBackend: Shader module and render pipeline creation" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    const vs = try MockBackend.createShaderModule(&device, .{
        .code = "void main() {}",
        .stage = .vertex,
    });
    const fs = try MockBackend.createShaderModule(&device, .{
        .code = "void main() {}",
        .stage = .fragment,
    });

    const pipeline = try MockBackend.createRenderPipeline(&device, .{
        .vertex_shader = vs,
        .fragment_shader = fs,
        .color_target_format = .rgba8_unorm,
    });
    try std.testing.expect(pipeline.isValid());
}

test "MockBackend: Quad draw recording, scissor clipping, and golden oracle readback" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const width_px: u32 = 20;
    const height_px: u32 = 20;

    const target = try MockBackend.createTexture(&device, .{
        .size = .{ .width = width_px, .height = height_px },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, target);

    // 10x10 green quad at (5, 5) with color {0, 255, 0, 255}
    const quad_verts = makeQuadVertices(5.0, 5.0, 10.0, 10.0, .{ 0, 255, 0, 255 });
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(quad_verts)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);

    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&quad_verts));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = target,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 }, // Clear black
        },
    });

    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);

    // Scissor clip: only allow rendering in [5, 10) x [5, 15) -> half of quad width!
    MockBackend.setClipRect(&pass, .{ .x = 5, .y = 5, .width = 5, .height = 10 });
    MockBackend.draw(&pass, 6, 1, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cmd_buf = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cmd_buf});

    var readback_pixels: [width_px * height_px * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, target, &readback_pixels);

    // Inside scissor: (7, 7) must be green {0, 255, 0, 255}
    const inside_idx = (7 * width_px + 7) * 4;
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[inside_idx + 0]);
    try std.testing.expectEqual(@as(u8, 255), readback_pixels[inside_idx + 1]);
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[inside_idx + 2]);
    try std.testing.expectEqual(@as(u8, 255), readback_pixels[inside_idx + 3]);

    // Outside scissor but inside quad: (12, 7) must remain black {0, 0, 0, 255}
    const clipped_idx = (7 * width_px + 12) * 4;
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[clipped_idx + 0]);
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[clipped_idx + 1]);
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[clipped_idx + 2]);
    try std.testing.expectEqual(@as(u8, 255), readback_pixels[clipped_idx + 3]);

    // Outside quad entirely: (2, 2) must remain black
    const outside_idx = (2 * width_px + 2) * 4;
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[outside_idx + 0]);
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[outside_idx + 1]);
    try std.testing.expectEqual(@as(u8, 0), readback_pixels[outside_idx + 2]);
    try std.testing.expectEqual(@as(u8, 255), readback_pixels[outside_idx + 3]);
}

test "MockBackend: Translucent quad has uniform pixel blending without diagonal seam" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const width_px: u32 = 10;
    const height_px: u32 = 10;
    const target = try MockBackend.createTexture(&device, .{
        .size = .{ .width = width_px, .height = height_px },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, target);

    // 10x10 quad from (0,0) with red color and alpha 128
    const quad_verts = makeQuadVertices(0.0, 0.0, 10.0, 10.0, .{ 200, 0, 0, 128 });
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(quad_verts)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);
    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&quad_verts));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = target,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
        },
    });
    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);
    MockBackend.draw(&pass, 6, 1, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cb = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    var readback: [width_px * height_px * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, target, &readback);

    // Non-diagonal pixel (1, 0): expected red channel value
    const off_diag_idx = (0 * width_px + 1) * 4;
    const expected_r = readback[off_diag_idx + 0];

    // Diagonal pixel (0, 0) and (2, 2) on the shared edge between the 2 triangles
    const diag_idx_0 = (0 * width_px + 0) * 4;
    const diag_idx_2 = (2 * width_px + 2) * 4;

    try std.testing.expectEqual(expected_r, readback[diag_idx_0 + 0]);
    try std.testing.expectEqual(expected_r, readback[diag_idx_2 + 0]);
}

test "MockBackend: Stale shader generation rejected by createRenderPipeline" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    const s1 = try MockBackend.createShaderModule(&device, .{ .code = "v1" });
    device.shaders[s1.id - 1].active = false;
    device.shaders[s1.id - 1].generation += 1;

    const s2 = try MockBackend.createShaderModule(&device, .{ .code = "v2" });
    try std.testing.expectEqual(s1.id, s2.id);
    try std.testing.expect(s1.generation != s2.generation);

    // Passing stale handle s1 to createRenderPipeline must fail with InvalidHandle
    try std.testing.expectError(
        hal.GpuError.InvalidHandle,
        MockBackend.createRenderPipeline(&device, .{ .vertex_shader = s1 }),
    );
}

test "MockBackend: All 19 contract functions exercised end-to-end" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    // 1. createBuffer
    const buf = try MockBackend.createBuffer(&device, .{
        .size = 128,
        .usage = hal.BufferUsage.vertex_dst,
    });

    // 2. writeBuffer
    const vertex_bytes = [_]u8{ 0, 1, 2, 3 };
    MockBackend.writeBuffer(&queue, buf, 0, &vertex_bytes);

    // 3. createTexture
    const tex = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 8, .height = 8 },
        .format = .rgba8_unorm,
    });

    // 4. writeTexture
    var tex_data: [8 * 8 * 4]u8 = @splat(128);
    MockBackend.writeTexture(&queue, tex, .{ .bytes_per_row = 32 }, .{
        .width = 8,
        .height = 8,
    }, &tex_data);

    // 5. createSampler
    const sampler = try MockBackend.createSampler(
        &device,
        hal.SamplerDesc.pinned_bilinear_clamp,
    );
    try std.testing.expect(sampler.isValid());

    // 6. createShaderModule
    const shader = try MockBackend.createShaderModule(&device, .{ .code = "src" });

    // 7. createRenderPipeline
    const pipeline = try MockBackend.createRenderPipeline(&device, .{
        .vertex_shader = shader,
    });

    // 8. createCommandEncoder
    var encoder = try MockBackend.createCommandEncoder(&device);

    // 9. beginRenderPass
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = tex,
            .load_op = .clear,
            .clear_color = .{ 0.1, 0.2, 0.3, 1.0 },
        },
    });

    // 10. setPipeline
    MockBackend.setPipeline(&pass, pipeline);

    // 11. setVertexBuffer
    MockBackend.setVertexBuffer(&pass, 0, buf, 0);

    // 12. setClipRect
    MockBackend.setClipRect(&pass, .{ .x = 0, .y = 0, .width = 8, .height = 8 });

    // 13. draw
    MockBackend.draw(&pass, 0, 1, 0, 0);

    // 14. endRenderPass
    MockBackend.endRenderPass(&pass);

    // 15. finishCommandEncoder
    const cb = try MockBackend.finishCommandEncoder(&encoder);

    // 16. submit
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    // 17. readbackTarget
    var readback: [8 * 8 * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, tex, &readback);

    // 18. destroyBuffer
    MockBackend.destroyBuffer(&device, buf);

    // 19. destroyTexture
    MockBackend.destroyTexture(&device, tex);
}

test "MockBackend: Buffer memory is zeroed upon creation" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const b1 = try MockBackend.createBuffer(&device, .{
        .size = 64,
        .usage = hal.BufferUsage.vertex_dst,
    });
    const dirty: [64]u8 = @splat(0xFF);
    MockBackend.writeBuffer(&queue, b1, 0, &dirty);
    MockBackend.destroyBuffer(&device, b1);

    // b2 reuses slot 0: must be zeroed!
    const b2 = try MockBackend.createBuffer(&device, .{
        .size = 64,
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, b2);

    const data = device.getBufferData(b2).?;
    for (data) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }
}

test "MockBackend: Destroy with invalid or null handles is safe no-op" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    // ID 0
    MockBackend.destroyBuffer(&device, hal.Buffer.null_handle);
    MockBackend.destroyTexture(&device, hal.Texture.null_handle);

    // Out of range ID
    MockBackend.destroyBuffer(&device, .{ .id = 999, .generation = 1 });
    MockBackend.destroyTexture(&device, .{ .id = 999, .generation = 1 });
}

test "MockBackend: Sampler pinned bilinear clamp contract" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    const desc = hal.SamplerDesc.pinned_bilinear_clamp;
    try std.testing.expectEqual(hal.MipmapFilterMode.none, desc.mipmap_filter);
    try std.testing.expectEqual(hal.FilterMode.linear, desc.mag_filter);
    try std.testing.expectEqual(hal.FilterMode.linear, desc.min_filter);
    try std.testing.expectEqual(hal.AddressMode.clamp_to_edge, desc.address_mode_u);
    try std.testing.expectEqual(hal.AddressMode.clamp_to_edge, desc.address_mode_v);

    const s = try MockBackend.createSampler(&device, desc);
    try std.testing.expect(s.isValid());
}

test "MockBackend: Degenerate collinear triangles render zero pixels" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const target = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 16, .height = 16 },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, target);

    // Collinear degenerate triangle: (2, 2), (6, 6), (10, 10)
    const degen_verts = [3]Vertex2D{
        .{ .pos = .{ 2.0, 2.0 }, .color = .{ 255, 255, 255, 255 } },
        .{ .pos = .{ 6.0, 6.0 }, .color = .{ 255, 255, 255, 255 } },
        .{ .pos = .{ 10.0, 10.0 }, .color = .{ 255, 255, 255, 255 } },
    };
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(degen_verts)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);
    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&degen_verts));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = target,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
        },
    });
    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);
    MockBackend.draw(&pass, 3, 1, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cb = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    var readback: [16 * 16 * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, target, &readback);

    // Entire target must remain clear black: no pixels drawn
    var px_i: usize = 0;
    while (px_i < readback.len) : (px_i += 4) {
        try std.testing.expectEqual(@as(u8, 0), readback[px_i + 0]);
        try std.testing.expectEqual(@as(u8, 0), readback[px_i + 1]);
        try std.testing.expectEqual(@as(u8, 0), readback[px_i + 2]);
        try std.testing.expectEqual(@as(u8, 255), readback[px_i + 3]);
    }
}

test "MockBackend: CommandEncoder overflow returns PoolExhausted" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    var encoder = try MockBackend.createCommandEncoder(&device);

    // Fill up to capacity
    var i: u32 = 0;
    while (i < commands_per_buffer_max) : (i += 1) {
        try encoder.record(.{ .set_clip_rect = .{} });
    }

    // Next record overflows
    try std.testing.expectError(
        hal.GpuError.PoolExhausted,
        encoder.record(.{ .set_clip_rect = .{} }),
    );

    // finishCommandEncoder must fail with PoolExhausted
    try std.testing.expectError(
        hal.GpuError.PoolExhausted,
        MockBackend.finishCommandEncoder(&encoder),
    );
}

test "MockBackend: R8 and BGRA format rasterization" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    // 1. Test R8 texture clearing and quad draw
    const r8_tex = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 8, .height = 8 },
        .format = .r8_unorm,
    });
    defer MockBackend.destroyTexture(&device, r8_tex);

    const quad_verts = makeQuadVertices(1.0, 1.0, 4.0, 4.0, .{ 200, 0, 0, 255 });
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(quad_verts)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);
    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&quad_verts));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = r8_tex,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
        },
    });
    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);
    MockBackend.draw(&pass, 6, 1, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cb = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    var r8_readback: [8 * 8]u8 = undefined;
    try MockBackend.readbackTarget(&device, r8_tex, &r8_readback);
    // (2, 2) inside quad: value 200
    try std.testing.expectEqual(@as(u8, 200), r8_readback[2 * 8 + 2]);
    // (7, 7) outside quad: value 0
    try std.testing.expectEqual(@as(u8, 0), r8_readback[7 * 8 + 7]);
}

test "MockBackend: Smooth barycentric vertex color interpolation" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const target = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 16, .height = 16 },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, target);

    // Triangle with red, green, blue corners
    const tri = [3]Vertex2D{
        .{ .pos = .{ 0.0, 0.0 }, .color = .{ 255, 0, 0, 255 } },
        .{ .pos = .{ 16.0, 0.0 }, .color = .{ 0, 255, 0, 255 } },
        .{ .pos = .{ 0.0, 16.0 }, .color = .{ 0, 0, 255, 255 } },
    };
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(tri)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);
    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&tri));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = target,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
        },
    });
    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);
    MockBackend.draw(&pass, 3, 1, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cb = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    var readback: [16 * 16 * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, target, &readback);

    // Near red vertex (1, 1): red channel should dominate
    const p1_idx = (1 * 16 + 1) * 4;
    try std.testing.expect(readback[p1_idx + 0] > 180);
}

test "MockBackend: CommandEncoder enforces proper render pass sequence" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);

    const tex = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 4, .height = 4 },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, tex);

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{ .target = tex },
    });

    // Finishing an encoder with an open render pass must fail
    try std.testing.expectError(
        hal.GpuError.InvalidCommandSequence,
        MockBackend.finishCommandEncoder(&encoder),
    );

    // Opening a nested render pass on the same encoder must fail
    try std.testing.expectError(
        hal.GpuError.InvalidCommandSequence,
        MockBackend.beginRenderPass(&encoder, .{
            .color_attachment = .{ .target = tex },
        }),
    );

    MockBackend.endRenderPass(&pass);

    // Now finishing succeeds
    const cb = try MockBackend.finishCommandEncoder(&encoder);
    try std.testing.expect(cb.isValid());

    // Finishing an already-finished encoder must fail
    try std.testing.expectError(
        hal.GpuError.InvalidCommandSequence,
        MockBackend.finishCommandEncoder(&encoder),
    );
}

test "MockBackend: Translucent rendering preserves and composites destination alpha" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const width_px: u32 = 4;
    const height_px: u32 = 4;
    const target = try MockBackend.createTexture(&device, .{
        .size = .{ .width = width_px, .height = height_px },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, target);

    const quad_verts = makeQuadVertices(0.0, 0.0, 4.0, 4.0, .{ 200, 50, 0, 100 });
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(quad_verts)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);
    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&quad_verts));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = target,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 0.0 },
        },
    });
    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);
    MockBackend.draw(&pass, 6, 1, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cb = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    var readback: [4 * 4 * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, target, &readback);

    // Alpha at (2, 2) must be 100 (composited over alpha 0), not forced to 255
    const idx = (2 * width_px + 2) * 4;
    try std.testing.expectEqual(@as(u8, 100), readback[idx + 3]);
}

test "MockBackend: Infinite vertex coordinates safely rejected without rasterization" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const target = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 8, .height = 8 },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, target);

    const inf_val = std.math.inf(f32);
    const inf_verts = [3]Vertex2D{
        .{ .pos = .{ inf_val, 0.0 }, .color = .{ 255, 0, 0, 255 } },
        .{ .pos = .{ 0.0, inf_val }, .color = .{ 0, 255, 0, 255 } },
        .{ .pos = .{ 0.0, 0.0 }, .color = .{ 0, 0, 255, 255 } },
    };
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(inf_verts)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);
    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&inf_verts));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = target,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
        },
    });
    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);
    MockBackend.draw(&pass, 3, 1, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cb = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    var readback: [8 * 8 * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, target, &readback);

    // Target pixels must remain clear black
    try std.testing.expectEqual(@as(u8, 0), readback[0]);
    try std.testing.expectEqual(@as(u8, 255), readback[3]);
}

test "MockBackend: Multiple instances rasterized when instance_count > 1" {
    var device: MockBackend.Device = undefined;
    MockBackend.Device.init(&device);
    var queue = MockBackend.Queue.init(&device);

    const target = try MockBackend.createTexture(&device, .{
        .size = .{ .width = 8, .height = 8 },
        .format = .rgba8_unorm,
    });
    defer MockBackend.destroyTexture(&device, target);

    const quad_verts = makeQuadVertices(1.0, 1.0, 4.0, 4.0, .{ 50, 0, 0, 128 });
    const vbuf = try MockBackend.createBuffer(&device, .{
        .size = @sizeOf(@TypeOf(quad_verts)),
        .usage = hal.BufferUsage.vertex_dst,
    });
    defer MockBackend.destroyBuffer(&device, vbuf);
    MockBackend.writeBuffer(&queue, vbuf, 0, std.mem.sliceAsBytes(&quad_verts));

    const vs = try MockBackend.createShaderModule(&device, .{ .code = "vs" });
    const pipeline = try MockBackend.createRenderPipeline(&device, .{ .vertex_shader = vs });

    var encoder = try MockBackend.createCommandEncoder(&device);
    var pass = try MockBackend.beginRenderPass(&encoder, .{
        .color_attachment = .{
            .target = target,
            .load_op = .clear,
            .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
        },
    });
    MockBackend.setPipeline(&pass, pipeline);
    MockBackend.setVertexBuffer(&pass, 0, vbuf, 0);
    // Draw 2 instances: pixel will be blended twice
    MockBackend.draw(&pass, 6, 2, 0, 0);
    MockBackend.endRenderPass(&pass);

    const cb = try MockBackend.finishCommandEncoder(&encoder);
    MockBackend.submit(&queue, &[_]hal.CommandBuffer{cb});

    var readback: [8 * 8 * 4]u8 = undefined;
    try MockBackend.readbackTarget(&device, target, &readback);

    // Pixel at (2, 2) blended twice: 1st blend -> 25, 2nd blend -> 37
    const idx = (2 * 8 + 2) * 4;
    try std.testing.expect(readback[idx + 0] > 25);
}
