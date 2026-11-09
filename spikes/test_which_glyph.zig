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

    var g: u16 = 0;
    while (g < 200) : (g += 1) {
        var count: usize = 0;
        var stats: ttf.GlyphStats = .{};
        font.decodeGlyphCurves(g, &raw_curves, &count, &stats, 0);
        if (count == 0) continue;

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
        if (w <= 1e-3 or h <= 1e-3) continue;

        const max_dim = @max(w, h);
        const s = 24.0 / max_dim;
        const ox = 4.0 + (24.0 - w * s) * 0.5 - x_min * s;
        const oy = 4.0 + (24.0 - h * s) * 0.5 - y_min * s;

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

        var max_d: u32 = 0;
        var diff_px: usize = 0;
        for (0..32) |py| {
            for (0..32) |px| {
                const b = base_patch[py * 32 + px];
                const f = fast_patch[py * 32 + px];
                const d: u32 = @intCast(@abs(@as(i32, b) - @as(i32, f)));
                if (d > 0) {
                    diff_px += 1;
                    max_d = @max(max_d, d);
                    std.debug.print("Diff at glyph {d}, (px,py)=({d},{d}): base={d}, fast={d}\n", .{
                        g, px, py, b, f
                    });
                }
            }
        }

        if (max_d > 0) {
            std.debug.print("FOUND DIVERGENT GLYPH: index={d}, max_delta={d}, diff_count={d}\n", .{
                g, max_d, diff_px
            });
        }
    }
}
