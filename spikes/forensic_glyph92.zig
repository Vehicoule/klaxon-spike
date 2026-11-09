const std = @import("std");
const ttf = @import("truetype_real.zig");
const paint = @import("paint_pipeline.zig");

var font_buffer: [1024 * 1024]u8 align(64) = undefined;
var raw_curves: [1024]paint.BezierQuadratic align(64) = undefined;
var norm_curves: [1024]paint.BezierQuadratic align(64) = undefined;
var fast_patch: [32 * 32]u8 align(64) = undefined;
var base_patch: [32 * 32]u8 align(64) = undefined;

fn loadFont(path: []const u8) !usize {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0);
    defer _ = std.posix.system.close(fd);
    var total_read: usize = 0;
    while (total_read < font_buffer.len) {
        const n = try std.posix.read(fd, font_buffer[total_read..]);
        if (n == 0) break;
        total_read += n;
    }
    return total_read;
}

pub fn main() !void {
    const file_len = try loadFont("/System/Library/Fonts/Supplemental/Arial.ttf");
    const font = try ttf.TrueTypeFont.init(font_buffer[0..file_len]);

    const g: u16 = 92;
    const offsets = font.getGlyphOffset(g);
    const start = font.glyf_offset + offsets.o1;
    const end = font.glyf_offset + offsets.o2;
    const g_data = font.data[start..end];

    const num_contours_i = std.mem.readInt(i16, g_data[0..2], .big);
    const x_min_raw = std.mem.readInt(i16, g_data[2..4], .big);
    const y_min_raw = std.mem.readInt(i16, g_data[4..6], .big);
    const x_max_raw = std.mem.readInt(i16, g_data[6..8], .big);
    const y_max_raw = std.mem.readInt(i16, g_data[8..10], .big);

    std.debug.print("Glyph 92:\n", .{});
    std.debug.print("  Offset in file: {d}..{d} (size: {d} bytes)\n", .{ start, end, g_data.len });
    std.debug.print("  num_contours: {d}\n", .{num_contours_i});
    std.debug.print("  BBox raw: [{d}, {d}] -> [{d}, {d}]\n", .{
        x_min_raw, y_min_raw, x_max_raw, y_max_raw
    });

    var count: usize = 0;
    var stats: ttf.GlyphStats = .{};
    font.decodeGlyphCurves(g, &raw_curves, &count, &stats, 0);

    std.debug.print("  Decoded curves: total={d}, lines={d}, quads={d}, implicit_midpoints={d}, composites={d}\n", .{
        count, stats.lines_count, stats.quads_count, stats.implicit_midpoints, stats.composites_count
    });

    // Compute normalization
    var x_min: f32 = 1e9;
    var y_min: f32 = 1e9;
    var x_max: f32 = -1e9;
    var y_max: f32 = -1e9;

    for (raw_curves[0..count]) |c| {
        x_min = @min(x_min, @min(c.p0[0], @min(c.p1[0], c.p2[0])));
        y_min = @min(y_min, @min(c.p0[1], @min(c.p1[1], c.p2[1])));
        x_max = @max(x_max, @max(c.p0[0], @max(c.p1[0], c.p2[0])));
        y_max = @max(y_max, @max(c.p0[1], @max(c.p1[1], c.p2[1])));
    }

    const w = x_max - x_min;
    const h = y_max - y_min;
    const max_dim = @max(w, h);
    const s = 24.0 / max_dim;
    const ox = 4.0 + (24.0 - w * s) * 0.5 - x_min * s;
    const oy = 4.0 + (24.0 - h * s) * 0.5 - y_min * s;

    std.debug.print("  Normalized scale s={e}, ox={e}, oy={e}\n", .{ s, ox, oy });

    var ci: usize = 0;
    while (ci < count) : (ci += 1) {
        const crv = raw_curves[ci];
        norm_curves[ci] = .{
            .p0 = .{ crv.p0[0] * s + ox, crv.p0[1] * s + oy },
            .p1 = .{ crv.p1[0] * s + ox, crv.p1[1] * s + oy },
            .p2 = .{ crv.p2[0] * s + ox, crv.p2[1] * s + oy },
        };
    }

    const valid_curves = norm_curves[0..count];
    paint.rasterizeGlyphSdfBaseline(valid_curves, 32, 32, 4.0, &base_patch);
    paint.rasterizeGlyphSdfFast(valid_curves, 32, 32, 4.0, &fast_patch);

    // Analyze pixel (6, 30)
    const px: u32 = 6;
    const py: u32 = 30;
    const fx: f32 = 6.5;
    const fy: f32 = 30.5;

    std.debug.print("\nPixel ({d}, {d}) center=({d:.1}, {d:.1}):\n", .{ px, py, fx, fy });
    std.debug.print("  base_patch[30, 6] = {d}\n", .{base_patch[py * 32 + px]});
    std.debug.print("  fast_patch[30, 6] = {d}\n", .{fast_patch[py * 32 + px]});

    // Baseline details
    var min_dist_base: f32 = 1e6;
    var best_curve_base: usize = 0;
    var winding_base: i32 = 0;
    for (valid_curves, 0..) |crv, idx| {
        const d = crv.distanceToPoint(fx, fy);
        if (d < min_dist_base) {
            min_dist_base = d;
            best_curve_base = idx;
        }
        winding_base += crv.windingContribution(fx, fy);
    }

    const is_inside_base = (winding_base != 0);
    const signed_dist_base = if (is_inside_base) -min_dist_base else min_dist_base;
    const norm_base_f32 = std.math.clamp(0.5 - signed_dist_base / 8.0, 0.0, 1.0);
    const byte_f32_base = norm_base_f32 * 255.0 + 0.5;

    std.debug.print("\nBaseline detailed calculation:\n", .{});
    std.debug.print("  min_dist_base = {d:.9} (curve #{d})\n", .{ min_dist_base, best_curve_base });
    std.debug.print("  winding_base  = {d} (is_inside={})\n", .{ winding_base, is_inside_base });
    std.debug.print("  signed_dist   = {d:.9}\n", .{signed_dist_base});
    std.debug.print("  norm_base     = {d:.9}\n", .{norm_base_f32});
    std.debug.print("  norm*255+0.5  = {d:.9}\n", .{byte_f32_base});
    std.debug.print("  @as(u8)       = {d}\n", .{@as(u8, @intFromFloat(byte_f32_base))});

    // Check best curve in baseline
    const crv_b = valid_curves[best_curve_base];
    std.debug.print("  Curve #{d}: p0=({d:.4}, {d:.4}), p1=({d:.4}, {d:.4}), p2=({d:.4}, {d:.4})\n", .{
        best_curve_base,
        crv_b.p0[0], crv_b.p0[1],
        crv_b.p1[0], crv_b.p1[1],
        crv_b.p2[0], crv_b.p2[1],
    });

    // High precision brute force distance to that curve and to all curves
    var bf_min: f32 = 1e6;
    for (valid_curves) |crv| {
        const d = crv.bruteForceBezierDistance(fx, fy);
        bf_min = @min(bf_min, d);
    }
    std.debug.print("  bruteForceBezierDistance (1024 segs) = {d:.9}\n", .{bf_min});

    // Fast details: let's reproduce exactly what Fast did for pixel (6, 30)
    // Note: chunk for x=6 is chunk c=1 (x in [4..7], so x=6 is index 2 in chunk 1).
    // Let's trace Fast step by step for y=30, chunk 1, lane 2 (fx=6.5)
}
