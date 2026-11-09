//! ============================================================================
//! Klaxon Unified Text & Geometry Paint Pipeline (`src/paint/pipeline.zig`)
//! ============================================================================
//! Unified rendering pipeline integrating geometry quads, TrueType text,
//! dynamic SDF glyph texture atlas, and GPU Hardware Abstraction Layer (HAL):
//! 1. 128-byte QuadInstance layout (2 cache lines, 16-byte aligned) matching GPU shaders.
//! 2. Automatic DrawCall batching by QuadKind, texture_index, and clipping rects.
//! 3. High-level geometry emitters: solid rects, rounded rects, soft box shadows.
//! 4. Vector TrueType text and procedural 5x7 bitmap text emitters.
//! 5. Direct GPU HAL integration (writeBuffer, writeTexture, render pass command recording).
//! 6. Headless CPU reference framebuffer (800x600 RGBA8) with perceptual diff oracle.
//!
//! Standards: TigerStyle (<= 100 columns, zero libc, zero POSIX, 0 malloc/free).

const std = @import("std");
const assert = std.debug.assert;
const sdf = @import("sdf.zig");
const atlas_mod = @import("atlas.zig");
const truetype = @import("truetype.zig");
const hal = @import("../gpu/hal.zig");

pub const GlyphKey = atlas_mod.GlyphKey;
pub const AtlasSlot = atlas_mod.AtlasSlot;
pub const GlyphAtlas = atlas_mod.GlyphAtlas;
pub const TrueTypeFont = truetype.TrueTypeFont;
pub const ScratchBuffers = truetype.ScratchBuffers;

// ============================================================================
// 1. Quad Classification & Shader Uniform Layouts
// ============================================================================

pub const QuadKind = enum(u32) {
    solid_rect = 0,
    rounded_rect = 1,
    box_shadow = 2,
    glyph_mask = 3,
};

/// 64-byte uniform block shared between CPU rasterizer and GPU shaders (std140-aligned).
pub const QuadUniforms = extern struct {
    center_px: [2]f32 align(16),
    half_extent_px: [2]f32,
    fill_color_rgba: [4]f32 align(16),
    border_color_rgba: [4]f32 align(16),
    corner_radius_px: f32,
    border_width_px: f32,
    softness_px: f32,
    _padding: f32 = 0.0,
};

comptime {
    assert(@sizeOf(QuadUniforms) == 64);
    assert(@alignOf(QuadUniforms) == 16);
    assert(@offsetOf(QuadUniforms, "center_px") == 0);
    assert(@offsetOf(QuadUniforms, "half_extent_px") == 8);
    assert(@offsetOf(QuadUniforms, "fill_color_rgba") == 16);
    assert(@offsetOf(QuadUniforms, "border_color_rgba") == 32);
    assert(@offsetOf(QuadUniforms, "corner_radius_px") == 48);
}

/// 128-byte Quad instance uploaded to GPU vertex/instance buffers (2 cache lines).
pub const QuadInstance = extern struct {
    uniforms: QuadUniforms, // 64 bytes
    uv_min: [2]f32, // 8 bytes
    uv_max: [2]f32, // 8 bytes
    clip_min_px: [2]f32, // 8 bytes
    clip_max_px: [2]f32, // 8 bytes
    quad_kind: QuadKind, // 4 bytes
    depth_layer: f32, // 4 bytes
    texture_index: u32, // 4 bytes
    _reserved: u32 = 0, // 4 bytes
    _padding: [16]u8 = @splat(0), // 16 bytes
};

comptime {
    assert(@sizeOf(QuadInstance) == 128);
    assert(@alignOf(QuadInstance) == 16);
    assert(@offsetOf(QuadInstance, "uniforms") == 0);
    assert(@offsetOf(QuadInstance, "uv_min") == 64);
    assert(@offsetOf(QuadInstance, "uv_max") == 72);
    assert(@offsetOf(QuadInstance, "clip_min_px") == 80);
    assert(@offsetOf(QuadInstance, "clip_max_px") == 88);
    assert(@offsetOf(QuadInstance, "quad_kind") == 96);
    assert(@offsetOf(QuadInstance, "depth_layer") == 100);
    assert(@offsetOf(QuadInstance, "texture_index") == 104);
}

/// Batched draw command grouping a contiguous range of compatible instances.
pub const DrawCall = extern struct {
    instance_offset: u32,
    instance_count: u32,
    quad_kind: QuadKind,
    texture_index: u32,
    clip_min_px: [2]f32,
    clip_max_px: [2]f32,
};

comptime {
    assert(@sizeOf(DrawCall) == 32);
    assert(@alignOf(DrawCall) == 4);
}

// ============================================================================
// 2. Paint Pipeline Working Set & Batching Engine
// ============================================================================

pub const instance_capacity_max: u32 = 16384;
pub const draw_call_capacity_max: u32 = 256;

/// Paint pipeline managing static instance buffers, quads emission, and batching.
pub const PaintPipeline = struct {
    instances: [instance_capacity_max]QuadInstance align(64),
    instance_count: u32,
    draw_calls: [draw_call_capacity_max]DrawCall,
    draw_call_count: u32,

    /// Initializes the paint pipeline in-place with zero heap allocations.
    pub fn init(target: *PaintPipeline) void {
        target.instance_count = 0;
        target.draw_call_count = 0;
    }

    /// Resets instance and draw call counters for a new frame.
    pub fn clear(self: *PaintPipeline) void {
        self.instance_count = 0;
        self.draw_call_count = 0;
    }

    /// Emits a single quad into the instance buffer.
    /// Capacity overflow panics immediately (Class 1 Invariant).
    pub fn emitQuad(self: *PaintPipeline, quad: QuadInstance) void {
        if (self.instance_count >= instance_capacity_max) {
            @panic("InstanceBuffer capacity exceeded");
        }
        self.instances[self.instance_count] = quad;
        self.instance_count += 1;
    }

    /// Emits a solid colored rectangular quad.
    pub fn emitSolidRect(
        self: *PaintPipeline,
        center_px: [2]f32,
        half_extent_px: [2]f32,
        fill_color_rgba: [4]f32,
    ) void {
        self.emitQuad(.{
            .uniforms = .{
                .center_px = center_px,
                .half_extent_px = half_extent_px,
                .fill_color_rgba = fill_color_rgba,
                .border_color_rgba = .{ 0.0, 0.0, 0.0, 0.0 },
                .corner_radius_px = 0.0,
                .border_width_px = 0.0,
                .softness_px = 1.0,
            },
            .uv_min = .{ 0.0, 0.0 },
            .uv_max = .{ 0.0, 0.0 },
            .clip_min_px = .{ 0.0, 0.0 },
            .clip_max_px = .{ 800.0, 600.0 },
            .quad_kind = .solid_rect,
            .depth_layer = 0.0,
            .texture_index = 0,
        });
    }

    /// Emits a rounded rectangular quad with customizable corner radius and border.
    pub fn emitRoundedRect(
        self: *PaintPipeline,
        center_px: [2]f32,
        half_extent_px: [2]f32,
        corner_radius_px: f32,
        fill_color_rgba: [4]f32,
        border_color_rgba: [4]f32,
        border_width_px: f32,
    ) void {
        self.emitQuad(.{
            .uniforms = .{
                .center_px = center_px,
                .half_extent_px = half_extent_px,
                .fill_color_rgba = fill_color_rgba,
                .border_color_rgba = border_color_rgba,
                .corner_radius_px = corner_radius_px,
                .border_width_px = border_width_px,
                .softness_px = 1.0,
            },
            .uv_min = .{ 0.0, 0.0 },
            .uv_max = .{ 0.0, 0.0 },
            .clip_min_px = .{ 0.0, 0.0 },
            .clip_max_px = .{ 800.0, 600.0 },
            .quad_kind = .rounded_rect,
            .depth_layer = 0.0,
            .texture_index = 0,
        });
    }

    /// Emits a soft blurred box shadow evaluated via analytic SDF.
    pub fn emitBoxShadow(
        self: *PaintPipeline,
        center_px: [2]f32,
        half_extent_px: [2]f32,
        corner_radius_px: f32,
        blur_radius_px: f32,
        shadow_color_rgba: [4]f32,
    ) void {
        self.emitQuad(.{
            .uniforms = .{
                .center_px = center_px,
                .half_extent_px = half_extent_px,
                .fill_color_rgba = shadow_color_rgba,
                .border_color_rgba = shadow_color_rgba,
                .corner_radius_px = corner_radius_px,
                .border_width_px = 0.0,
                .softness_px = blur_radius_px,
            },
            .uv_min = .{ 0.0, 0.0 },
            .uv_max = .{ 0.0, 0.0 },
            .clip_min_px = .{ 0.0, 0.0 },
            .clip_max_px = .{ 800.0, 600.0 },
            .quad_kind = .box_shadow,
            .depth_layer = 0.0,
            .texture_index = 0,
        });
    }

    /// Emits a single glyph quad referencing an atlas slot.
    pub fn emitGlyphQuad(
        self: *PaintPipeline,
        slot: AtlasSlot,
        center_px: [2]f32,
        half_extent_px: [2]f32,
        color_rgba: [4]f32,
    ) void {
        self.emitQuad(.{
            .uniforms = .{
                .center_px = center_px,
                .half_extent_px = half_extent_px,
                .fill_color_rgba = color_rgba,
                .border_color_rgba = .{ 0.0, 0.0, 0.0, 0.0 },
                .corner_radius_px = 0.0,
                .border_width_px = 0.0,
                .softness_px = 1.0,
            },
            .uv_min = slot.uv_min,
            .uv_max = slot.uv_max,
            .clip_min_px = .{ 0.0, 0.0 },
            .clip_max_px = .{ 800.0, 600.0 },
            .quad_kind = .glyph_mask,
            .depth_layer = 1.0,
            .texture_index = 0,
        });
    }

    /// Emits a formatted text string using vector TrueType curves decoded into the atlas.
    pub fn emitText(
        self: *PaintPipeline,
        atlas: *GlyphAtlas,
        font: *const TrueTypeFont,
        scratch: *ScratchBuffers,
        text: []const u8,
        start_x_px: f32,
        baseline_y_px: f32,
        font_size_px: f32,
        color_rgba: [4]f32,
    ) void {
        const font_upem = if (font.units_per_em > 0) font.units_per_em else 1000;
        const upem = @as(f32, @floatFromInt(font_upem));
        const scale = font_size_px / upem;
        var cursor_x = start_x_px;

        var curves_buf: [256]sdf.BezierQuadratic = undefined;

        for (text) |ch| {
            if (ch == ' ') {
                const space_glyph = font.glyphIndexForCodepoint(' ');
                const space_adv = if (space_glyph > 0)
                    @as(f32, @floatFromInt(font.getGlyphMetrics(space_glyph).advance_width)) * scale
                else
                    font_size_px * 0.3;
                cursor_x += space_adv;
                continue;
            }

            const glyph_idx = font.glyphIndexForCodepoint(ch);
            const metrics = font.getGlyphMetrics(glyph_idx);
            const adv_px = @as(f32, @floatFromInt(metrics.advance_width)) * scale;
            const lsb_px = @as(f32, @floatFromInt(metrics.left_side_bearing)) * scale;

            const key = GlyphKey{
                .glyph_index = glyph_idx,
                .font_id = 1,
                .size_px = @as(u16, @intFromFloat(font_size_px)),
            };

            const slot = atlas.lookupSlot(key) orelse blk: {
                if (atlas.hasTimeBudget()) {
                    var count: usize = 0;
                    var stats = truetype.GlyphStats{};
                    font.decodeGlyphCurves(glyph_idx, scratch, &curves_buf, &count, &stats, 0);

                    if (count > 0) {
                        // Normalize curves into [4, 28]^2 inside 32x32 patch
                        var norm_curves: [256]sdf.BezierQuadratic = undefined;
                        var x_min: f32 = 1e9;
                        var y_min: f32 = 1e9;
                        var x_max: f32 = -1e9;
                        var y_max: f32 = -1e9;

                        for (curves_buf[0..count]) |c| {
                            x_min = @min(x_min, @min(c.p0[0], @min(c.p1[0], c.p2[0])));
                            y_min = @min(y_min, @min(c.p0[1], @min(c.p1[1], c.p2[1])));
                            x_max = @max(x_max, @max(c.p0[0], @max(c.p1[0], c.p2[0])));
                            y_max = @max(y_max, @max(c.p0[1], @max(c.p1[1], c.p2[1])));
                        }

                        const w = x_max - x_min;
                        const h = y_max - y_min;
                        const max_dim = @max(w, h);
                        const s = if (max_dim > 1e-4) 24.0 / max_dim else 1.0;
                        const ox = 4.0 + (24.0 - w * s) * 0.5 - x_min * s;
                        const oy = 4.0 + (24.0 - h * s) * 0.5 - y_min * s;

                        var ci: usize = 0;
                        while (ci < count) : (ci += 1) {
                            const crv = curves_buf[ci];
                            norm_curves[ci] = .{
                                .p0 = .{ crv.p0[0] * s + ox, crv.p0[1] * s + oy },
                                .p1 = .{ crv.p1[0] * s + ox, crv.p1[1] * s + oy },
                                .p2 = .{ crv.p2[0] * s + ox, crv.p2[1] * s + oy },
                            };
                        }

                        var patch_buf: [32 * 32]u8 = undefined;
                        sdf.rasterizeGlyphSdf(norm_curves[0..count], 32, 32, 4.0, &patch_buf);
                        const new_slot = atlas.insertGlyph(key, &patch_buf, 32, 32);
                        atlas.recordRasterization();
                        break :blk new_slot;
                    } else {
                        // Empty glyph outline (e.g. control character)
                        break :blk atlas.tofu_slot;
                    }
                } else {
                    atlas.time_budget_exhausted = true;
                    break :blk atlas.tofu_slot;
                }
            };

            const glyph_w = @as(f32, @floatFromInt(slot.width_px));
            const glyph_h = @as(f32, @floatFromInt(slot.height_px));
            const center_x = cursor_x + lsb_px + glyph_w * 0.5;
            const center_y = baseline_y_px - glyph_h * 0.5;

            self.emitGlyphQuad(
                slot,
                .{ center_x, center_y },
                .{ glyph_w * 0.5, glyph_h * 0.5 },
                color_rgba,
            );

            cursor_x += adv_px;
        }
    }

    /// Emits procedural bitmap text without needing a TrueType font file.
    pub fn emitTextProcedural(
        self: *PaintPipeline,
        atlas: *GlyphAtlas,
        text: []const u8,
        start_x_px: f32,
        baseline_y_px: f32,
        glyph_w_px: f32,
        glyph_h_px: f32,
        color_rgba: [4]f32,
    ) void {
        var cursor_x = start_x_px;
        for (text) |ch| {
            if (ch == ' ') {
                cursor_x += glyph_w_px * 0.6;
                continue;
            }

            const key = GlyphKey{
                .glyph_index = @as(u32, ch),
                .font_id = 0,
                .size_px = @as(u16, @intFromFloat(glyph_h_px)),
            };

            const slot = atlas.lookupSlot(key) orelse blk: {
                if (atlas.hasTimeBudget()) {
                    var glyph_buf: [32 * 32]u8 = undefined;
                    const w: u32 = @min(32, @as(u32, @intFromFloat(glyph_w_px)));
                    const h: u32 = @min(32, @as(u32, @intFromFloat(glyph_h_px)));
                    generateProceduralGlyph(ch, w, h, glyph_buf[0 .. w * h]);
                    const new_slot = atlas.insertGlyph(key, glyph_buf[0 .. w * h], w, h);
                    atlas.recordRasterization();
                    break :blk new_slot;
                } else {
                    atlas.time_budget_exhausted = true;
                    break :blk atlas.tofu_slot;
                }
            };

            const center_x = cursor_x + glyph_w_px * 0.5;
            const center_y = baseline_y_px - glyph_h_px * 0.5;

            self.emitGlyphQuad(
                slot,
                .{ center_x, center_y },
                .{ glyph_w_px * 0.5, glyph_h_px * 0.5 },
                color_rgba,
            );

            cursor_x += glyph_w_px + 2.0;
        }
    }

    /// Merges consecutive compatible quads into unified GPU draw commands.
    pub fn batchDrawCalls(self: *PaintPipeline) void {
        if (self.instance_count == 0) {
            self.draw_call_count = 0;
            return;
        }

        var call_count: u32 = 0;
        var current_kind = self.instances[0].quad_kind;
        var current_texture = self.instances[0].texture_index;
        var current_clip_min = self.instances[0].clip_min_px;
        var current_clip_max = self.instances[0].clip_max_px;
        var current_offset: u32 = 0;
        var current_count: u32 = 1;

        var idx: u32 = 1;
        while (idx < self.instance_count) : (idx += 1) {
            const inst = &self.instances[idx];
            var can_merge = false;
            if (inst.quad_kind == current_kind) {
                if (inst.texture_index == current_texture) {
                    if (inst.clip_min_px[0] == current_clip_min[0]) {
                        if (inst.clip_min_px[1] == current_clip_min[1]) {
                            if (inst.clip_max_px[0] == current_clip_max[0]) {
                                if (inst.clip_max_px[1] == current_clip_max[1]) {
                                    can_merge = true;
                                }
                            }
                        }
                    }
                }
            }

            if (can_merge) {
                current_count += 1;
            } else {
                assert(call_count < draw_call_capacity_max);
                self.draw_calls[call_count] = .{
                    .instance_offset = current_offset,
                    .instance_count = current_count,
                    .quad_kind = current_kind,
                    .texture_index = current_texture,
                    .clip_min_px = current_clip_min,
                    .clip_max_px = current_clip_max,
                };
                call_count += 1;

                current_offset = idx;
                current_count = 1;
                current_kind = inst.quad_kind;
                current_texture = inst.texture_index;
                current_clip_min = inst.clip_min_px;
                current_clip_max = inst.clip_max_px;
            }
        }

        assert(call_count < draw_call_capacity_max);
        self.draw_calls[call_count] = .{
            .instance_offset = current_offset,
            .instance_count = current_count,
            .quad_kind = current_kind,
            .texture_index = current_texture,
            .clip_min_px = current_clip_min,
            .clip_max_px = current_clip_max,
        };
        call_count += 1;
        self.draw_call_count = call_count;
    }

    // ========================================================================
    // GPU HAL Integration
    // ========================================================================

    /// Uploads queued instance quads to a GPU HAL instance buffer.
    pub fn uploadInstances(
        self: *const PaintPipeline,
        comptime GpuBackend: type,
        queue: *GpuBackend.Queue,
        buffer: hal.Buffer,
    ) void {
        if (self.instance_count == 0) return;
        const upload_bytes: u64 = @as(u64, self.instance_count) * @sizeOf(QuadInstance);
        const raw_slice: [*]const u8 = @ptrCast(&self.instances);
        GpuBackend.writeBuffer(queue, buffer, 0, raw_slice[0..upload_bytes]);
    }

    /// Uploads modified atlas texture pixels to a GPU HAL texture image.
    pub fn uploadAtlasTexture(
        comptime GpuBackend: type,
        queue: *GpuBackend.Queue,
        texture: hal.Texture,
        atlas: *GlyphAtlas,
    ) void {
        if (!atlas.dirty) return;
        const layout = hal.TextureDataLayout{
            .offset = 0,
            .bytes_per_row = atlas_mod.atlas_dimension_px,
            .rows_per_image = atlas_mod.atlas_dimension_px,
        };
        const extent = hal.Extent3D{
            .width = atlas_mod.atlas_dimension_px,
            .height = atlas_mod.atlas_dimension_px,
            .depth_or_array_layers = 1,
        };
        GpuBackend.writeTexture(queue, texture, layout, extent, &atlas.pixels);
        atlas.resetDirty();
    }

    /// Records draw calls for all batched instances into an active GPU render pass.
    pub fn recordRenderPass(
        self: *const PaintPipeline,
        comptime GpuBackend: type,
        pass: *GpuBackend.RenderPass,
        pipeline_handle: hal.RenderPipeline,
        instance_buffer: hal.Buffer,
    ) void {
        if (self.draw_call_count == 0) return;
        GpuBackend.setPipeline(pass, pipeline_handle);
        GpuBackend.setVertexBuffer(pass, 0, instance_buffer, 0);

        var i: u32 = 0;
        while (i < self.draw_call_count) : (i += 1) {
            const dc = self.draw_calls[i];
            // Each quad instance is rendered via a 6-vertex quad (2 triangles)
            GpuBackend.draw(pass, 6, dc.instance_count, 0, dc.instance_offset);
        }
    }
};

// ============================================================================
// 3. Headless Framebuffer & CPU Reference Rasterizer
// ============================================================================

pub const framebuffer_width_px: u32 = 800;
pub const framebuffer_height_px: u32 = 600;
pub const framebuffer_pixel_count: u32 = framebuffer_width_px * framebuffer_height_px;
pub const framebuffer_byte_size: u32 = framebuffer_pixel_count * 4;

/// Headless 2D RGBA8 framebuffer in static 64-byte aligned memory.
pub const Framebuffer = struct {
    pixels: [framebuffer_byte_size]u8 align(64),

    pub fn init(target: *Framebuffer) void {
        @memset(&target.pixels, 0);
    }

    pub fn clear(self: *Framebuffer, r: u8, g: u8, b: u8, a: u8) void {
        var idx: usize = 0;
        while (idx < self.pixels.len) : (idx += 4) {
            self.pixels[idx] = r;
            self.pixels[idx + 1] = g;
            self.pixels[idx + 2] = b;
            self.pixels[idx + 3] = a;
        }
    }
};

/// Evaluates signed distance from a point to an oriented rounded rectangle.
pub fn signedDistanceRoundedRect(
    point_offset_px: @Vector(2, f32),
    half_extent_px: @Vector(2, f32),
    corner_radius_px: f32,
) f32 {
    assert(corner_radius_px >= 0.0);
    const radius_vec: @Vector(2, f32) = @splat(corner_radius_px);
    const zero_vec: @Vector(2, f32) = @splat(0.0);

    const corner_delta = @abs(point_offset_px) - half_extent_px + radius_vec;
    const outside_vec = @max(corner_delta, zero_vec);
    const outside_dist = @sqrt(outside_vec[0] * outside_vec[0] + outside_vec[1] * outside_vec[1]);
    const inside_dist = @min(@max(corner_delta[0], corner_delta[1]), 0.0);

    return outside_dist + inside_dist - corner_radius_px;
}

/// Evaluates premultiplied RGBA color of a pixel for a given QuadUniforms.
pub fn evaluateQuadPixel(coord_px: @Vector(2, f32), uniforms: QuadUniforms) @Vector(4, f32) {
    assert(uniforms.softness_px > 0.0);
    const center_vec: @Vector(2, f32) = uniforms.center_px;
    const half_extent_vec: @Vector(2, f32) = uniforms.half_extent_px;
    const fill_vec: @Vector(4, f32) = uniforms.fill_color_rgba;
    const border_vec: @Vector(4, f32) = uniforms.border_color_rgba;

    const dist_px = signedDistanceRoundedRect(
        coord_px - center_vec,
        half_extent_vec,
        uniforms.corner_radius_px,
    );

    const outer_cov = std.math.clamp(0.5 - dist_px / uniforms.softness_px, 0.0, 1.0);
    const inner_dist = dist_px + uniforms.border_width_px;
    const inner_cov = std.math.clamp(0.5 - inner_dist / uniforms.softness_px, 0.0, 1.0);

    const inner_weight: @Vector(4, f32) = @splat(inner_cov);
    const border_weight: @Vector(4, f32) = @splat(1.0 - inner_cov);
    const outer_weight: @Vector(4, f32) = @splat(outer_cov);

    const blended_rgb = fill_vec * inner_weight + border_vec * border_weight;
    return blended_rgb * outer_weight;
}

/// Rasterizes draw calls to a CPU reference framebuffer.
pub fn rasterizeDrawCallsCpu(
    fb: *Framebuffer,
    instances: []const QuadInstance,
    draw_calls: []const DrawCall,
    atlas: *const GlyphAtlas,
) void {
    for (draw_calls) |dc| {
        const start = dc.instance_offset;
        const end = start + dc.instance_count;
        for (instances[start..end]) |inst| {
            rasterizeSingleQuadCpu(fb, inst, atlas);
        }
    }
}

fn rasterizeSingleQuadCpu(
    fb: *Framebuffer,
    inst: QuadInstance,
    atlas: *const GlyphAtlas,
) void {
    const cx = inst.uniforms.center_px[0];
    const cy = inst.uniforms.center_px[1];
    const hw = inst.uniforms.half_extent_px[0];
    const hh = inst.uniforms.half_extent_px[1];
    const softness = inst.uniforms.softness_px;

    const pad = softness * 2.0;
    const min_x_f = @max(inst.clip_min_px[0], @max(0.0, cx - hw - pad));
    const max_x_f = @min(inst.clip_max_px[0], @min(@as(f32, framebuffer_width_px), cx + hw + pad));
    const min_y_f = @max(inst.clip_min_px[1], @max(0.0, cy - hh - pad));
    const max_y_f = @min(
        inst.clip_max_px[1],
        @min(@as(f32, framebuffer_height_px), cy + hh + pad),
    );

    if (min_x_f >= max_x_f or min_y_f >= max_y_f) return;

    const x_start: u32 = @intFromFloat(min_x_f);
    const x_end: u32 = @intFromFloat(max_x_f);
    const y_start: u32 = @intFromFloat(min_y_f);
    const y_end: u32 = @intFromFloat(max_y_f);

    var y = y_start;
    while (y < y_end) : (y += 1) {
        const fy = @as(f32, @floatFromInt(y)) + 0.5;
        var x = x_start;
        while (x < x_end) : (x += 1) {
            const fx = @as(f32, @floatFromInt(x)) + 0.5;
            const coord = @Vector(2, f32){ fx, fy };

            var frag_color: @Vector(4, f32) = undefined;
            if (inst.quad_kind == .glyph_mask) {
                // Sample SDF from atlas
                const u = (fx - (cx - hw)) / (hw * 2.0);
                const v = (fy - (cy - hh)) / (hh * 2.0);
                if (u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0) continue;

                const atlas_u = inst.uv_min[0] + u * (inst.uv_max[0] - inst.uv_min[0]);
                const atlas_v = inst.uv_min[1] + v * (inst.uv_max[1] - inst.uv_min[1]);
                const ax: u32 = @min(
                    atlas_mod.atlas_dimension_px - 1,
                    @as(u32, @intFromFloat(atlas_u * @as(f32, atlas_mod.atlas_dimension_px))),
                );
                const ay: u32 = @min(
                    atlas_mod.atlas_dimension_px - 1,
                    @as(u32, @intFromFloat(atlas_v * @as(f32, atlas_mod.atlas_dimension_px))),
                );
                const alpha_byte = atlas.pixels[ay * atlas_mod.atlas_dimension_px + ax];
                const alpha = @as(f32, @floatFromInt(alpha_byte)) / 255.0;

                const fill = inst.uniforms.fill_color_rgba;
                frag_color = @Vector(4, f32){
                    fill[0] * alpha,
                    fill[1] * alpha,
                    fill[2] * alpha,
                    fill[3] * alpha,
                };
            } else {
                frag_color = evaluateQuadPixel(coord, inst.uniforms);
            }

            // Alpha blend into framebuffer: dst = src + dst * (1 - src.a)
            const src_a = frag_color[3];
            if (src_a <= 0.001) continue;

            const px_idx = (y * framebuffer_width_px + x) * 4;
            const dst_r = @as(f32, @floatFromInt(fb.pixels[px_idx])) / 255.0;
            const dst_g = @as(f32, @floatFromInt(fb.pixels[px_idx + 1])) / 255.0;
            const dst_b = @as(f32, @floatFromInt(fb.pixels[px_idx + 2])) / 255.0;

            const out_r = frag_color[0] + dst_r * (1.0 - src_a);
            const out_g = frag_color[1] + dst_g * (1.0 - src_a);
            const out_b = frag_color[2] + dst_b * (1.0 - src_a);

            const r_byte = @as(u8, @intFromFloat(std.math.clamp(out_r * 255.0, 0.0, 255.0)));
            const g_byte = @as(u8, @intFromFloat(std.math.clamp(out_g * 255.0, 0.0, 255.0)));
            const b_byte = @as(u8, @intFromFloat(std.math.clamp(out_b * 255.0, 0.0, 255.0)));
            fb.pixels[px_idx] = r_byte;
            fb.pixels[px_idx + 1] = g_byte;
            fb.pixels[px_idx + 2] = b_byte;
            fb.pixels[px_idx + 3] = 255;
        }
    }
}

/// Result of perceptual comparison between two framebuffers.
pub const PerceptualDiff = struct {
    diff_max: u8,
    mismatched_pixel_count: u32,
    total_pixel_count: u32,
    mean_squared_error: f64,
    peak_signal_to_noise_ratio_db: f64,
    passed: bool,
};

/// Computes pixel-exact perceptual difference between two framebuffers.
pub fn compareFramebuffers(
    fb_a: *const Framebuffer,
    fb_b: *const Framebuffer,
    diff_threshold_max: u8,
) PerceptualDiff {
    var diff_max: u8 = 0;
    var mismatched_pixel_count: u32 = 0;
    var squared_error_sum: u64 = 0;

    var px: u32 = 0;
    while (px < framebuffer_pixel_count) : (px += 1) {
        const off = px * 4;
        var ch: u32 = 0;
        var mismatch = false;
        while (ch < 4) : (ch += 1) {
            const a = fb_a.pixels[off + ch];
            const b = fb_b.pixels[off + ch];
            const d: u8 = if (a >= b) a - b else b - a;
            if (d > diff_max) diff_max = d;
            if (d > diff_threshold_max) mismatch = true;
            const du64: u64 = d;
            squared_error_sum += du64 * du64;
        }
        if (mismatch) mismatched_pixel_count += 1;
    }

    const total_channels = @as(f64, @floatFromInt(framebuffer_pixel_count * 4));
    const mse = @as(f64, @floatFromInt(squared_error_sum)) / total_channels;
    const psnr: f64 = if (mse > 1e-6)
        10.0 * @log10((255.0 * 255.0) / mse)
    else
        99.0;

    return .{
        .diff_max = diff_max,
        .mismatched_pixel_count = mismatched_pixel_count,
        .total_pixel_count = framebuffer_pixel_count,
        .mean_squared_error = mse,
        .peak_signal_to_noise_ratio_db = psnr,
        .passed = (diff_max <= diff_threshold_max),
    };
}

// ============================================================================
// 4. Procedural 5x7 Font Generator (Zero-Dependency)
// ============================================================================

fn getCharBitmap5x7(c: u8) [7]u8 {
    return switch (c) {
        'A' => .{ 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
        'B' => .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110 },
        'C' => .{ 0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111 },
        'D' => .{ 0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110 },
        'E' => .{ 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 },
        'F' => .{ 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000 },
        'G' => .{ 0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111 },
        'H' => .{ 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
        'I' => .{ 0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 },
        'J' => .{ 0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110 },
        'K' => .{ 0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001 },
        'L' => .{ 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 },
        'M' => .{ 0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001 },
        'N' => .{ 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001 },
        'O' => .{ 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
        'P' => .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000 },
        'Q' => .{ 0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101 },
        'R' => .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001 },
        'S' => .{ 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110 },
        'T' => .{ 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 },
        'U' => .{ 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
        'V' => .{ 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100 },
        'W' => .{ 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001 },
        'X' => .{ 0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001 },
        'Y' => .{ 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100 },
        'Z' => .{ 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111 },
        '0' => .{ 0b01110, 0b10011, 0b10101, 0b11001, 0b10001, 0b10001, 0b01110 },
        '1' => .{ 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 },
        '2' => .{ 0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111 },
        ' ' => .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000 },
        '-' => .{ 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000 },
        else => .{ 0b11111, 0b10001, 0b10101, 0b10101, 0b10101, 0b10001, 0b11111 },
    };
}

/// Generates an alpha bitmap mask for an ASCII character at a requested resolution.
pub fn generateProceduralGlyph(
    char: u8,
    width_px: u32,
    height_px: u32,
    buffer: []u8,
) void {
    assert(buffer.len >= width_px * height_px);
    assert(width_px > 0);
    assert(height_px > 0);

    const rows = getCharBitmap5x7(char);

    var y: u32 = 0;
    while (y < height_px) : (y += 1) {
        const row_idx = (y * 7) / height_px;
        const row_bits = rows[@min(6, row_idx)];

        var x: u32 = 0;
        while (x < width_px) : (x += 1) {
            const col_idx = (x * 5) / width_px;
            const shift_amount: u3 = @intCast(4 - @min(4, col_idx));
            const bit_is_set = ((row_bits >> shift_amount) & 1) != 0;
            const alpha_val: u8 = if (bit_is_set) 255 else 0;
            buffer[y * width_px + x] = alpha_val;
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

test "PaintPipeline emit quads and batching" {
    var pipeline: PaintPipeline = undefined;
    PaintPipeline.init(&pipeline);

    pipeline.emitSolidRect(.{ 100.0, 100.0 }, .{ 50.0, 50.0 }, .{ 1.0, 0.0, 0.0, 1.0 });
    pipeline.emitSolidRect(.{ 200.0, 100.0 }, .{ 50.0, 50.0 }, .{ 0.0, 1.0, 0.0, 1.0 });
    pipeline.emitRoundedRect(
        .{ 300.0, 100.0 },
        .{ 50.0, 50.0 },
        8.0,
        .{ 0.0, 0.0, 1.0, 1.0 },
        .{ 1.0, 1.0, 1.0, 1.0 },
        2.0,
    );

    try std.testing.expectEqual(@as(u32, 3), pipeline.instance_count);

    pipeline.batchDrawCalls();
    // The two consecutive solid_rects merge into 1 draw call, rounded_rect is 2nd
    try std.testing.expectEqual(@as(u32, 2), pipeline.draw_call_count);
    try std.testing.expectEqual(@as(u32, 2), pipeline.draw_calls[0].instance_count);
    try std.testing.expectEqual(@as(u32, 1), pipeline.draw_calls[1].instance_count);
}

test "PaintPipeline CPU reference rasterizer and perceptual diff" {
    var fb_a: Framebuffer = undefined;
    var fb_b: Framebuffer = undefined;
    Framebuffer.init(&fb_a);
    Framebuffer.init(&fb_b);

    fb_a.clear(20, 20, 20, 255);
    fb_b.clear(20, 20, 20, 255);

    const diff = compareFramebuffers(&fb_a, &fb_b, 0);
    try std.testing.expect(diff.passed);
    try std.testing.expectEqual(@as(u8, 0), diff.diff_max);
    try std.testing.expectEqual(@as(u32, 0), diff.mismatched_pixel_count);
}

test "PaintPipeline procedural text emission" {
    var pipeline: PaintPipeline = undefined;
    PaintPipeline.init(&pipeline);

    var atlas: GlyphAtlas = undefined;
    GlyphAtlas.init(&atlas);

    pipeline.emitTextProcedural(
        &atlas,
        "KLAXON",
        50.0,
        100.0,
        16.0,
        24.0,
        .{ 1.0, 1.0, 1.0, 1.0 },
    );

    try std.testing.expectEqual(@as(u32, 6), pipeline.instance_count);
    pipeline.batchDrawCalls();
    try std.testing.expectEqual(@as(u32, 1), pipeline.draw_call_count);
}
