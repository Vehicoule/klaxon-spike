const std = @import("std");
const ttf = @import("truetype_real.zig");
const paint = @import("paint_pipeline.zig");

var font_buffer: [1024 * 1024]u8 align(64) = undefined;
var raw_curves: [1024]paint.BezierQuadratic align(64) = undefined;
var norm_curves: [1024]paint.BezierQuadratic align(64) = undefined;

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
    var count: usize = 0;
    var stats: ttf.GlyphStats = .{};
    font.decodeGlyphCurves(g, &raw_curves, &count, &stats, 0);

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

    const fx: f32 = 6.5;
    const fy: f32 = 30.5;
    const spread_px: f32 = 4.0;

    // Prepared curves
    var prep_storage: [64]paint.PreparedCurve4 = undefined;
    for (valid_curves, 0..) |crv, i| {
        prep_storage[i] = paint.PreparedCurve4.init(crv, spread_px);
    }

    // Let's check curve 6: is it a line?
    const prep6 = prep_storage[6];
    std.debug.print("prep6.is_line = {}\n", .{prep6.is_line});
    std.debug.print("prep6.ax={e}, ay={e}\n", .{prep6.ax, prep6.ay});
    std.debug.print("prep6.bx={e}, by={e}\n", .{prep6.bx, prep6.by});

    // In Fast:
    // How chunk_dist_sq was updated for chunk 1, lane 2:
    // fx = 6.5, fy = 30.5
    // Let's compute exactly what Fast does:
    const cy_scalar = prep6.curve.p0[1] - fy;
    const cx_scalar = prep6.curve.p0[0] - fx;
    const two: f32 = 2.0;
    const c1 = two * (prep6.ax * cx_scalar + prep6.ay * cy_scalar) + prep6.b_len_sq;
    const c0 = prep6.bx * cx_scalar + prep6.by * cy_scalar;

    std.debug.print("c1 = {d:.9}, c0 = {d:.9}\n", .{ c1, c0 });

    // Endpoint t=0:
    const dist_sq_0 = cx_scalar * cx_scalar + cy_scalar * cy_scalar;
    // Endpoint t=1:
    const dx_1 = prep6.curve.p2[0] - fx;
    const dy_1 = prep6.curve.p2[1] - fy;
    const dist_sq_1 = dx_1 * dx_1 + dy_1 * dy_1;
    var cand_dist_sq = @min(dist_sq_0, dist_sq_1);
    std.debug.print("t=0 dist_sq={d:.9}, t=1 dist_sq={d:.9} (dist_1={d:.9})\n", .{
        dist_sq_0, dist_sq_1, @sqrt(dist_sq_1)
    });

    if (prep6.is_line) {
        const t_line = std.math.clamp(-c0 / c1, 0.0, 1.0);
        const dx_l = prep6.bx * t_line + cx_scalar;
        const dy_l = prep6.by * t_line + cy_scalar;
        const dist_sq_l = dx_l * dx_l + dy_l * dy_l;
        cand_dist_sq = @min(cand_dist_sq, dist_sq_l);
        std.debug.print("Line: t_line={d:.9}, dist_sq_l={d:.9}, dist={d:.9}\n", .{
            t_line, dist_sq_l, @sqrt(dist_sq_l)
        });
    } else {
        std.debug.print("Quad Newton-Raphson:\n", .{});
        const seeds = [2]f32{ 0.25, 0.75 };
        for (seeds) |seed| {
            var t: f32 = seed;
            var iter: u32 = 0;
            while (iter < 4) : (iter += 1) {
                const val = ((prep6.c3 * t + prep6.c2) * t + c1) * t + c0;
                const deriv = (3.0 * prep6.c3 * t + 2.0 * prep6.c2) * t + c1;
                const delta = val / deriv;
                t = std.math.clamp(t - delta, 0.0, 1.0);
            }
            const dx_t = (prep6.ax * t + prep6.bx) * t + cx_scalar;
            const dy_t = (prep6.ay * t + prep6.by) * t + cy_scalar;
            const dist_sq_t = dx_t * dx_t + dy_t * dy_t;
            cand_dist_sq = @min(cand_dist_sq, dist_sq_t);
            std.debug.print("  seed {d:.2} -> t={d:.9}, dist_sq={d:.9}, dist={d:.9}\n", .{
                seed, t, dist_sq_t, @sqrt(dist_sq_t)
            });
        }
    }

    std.debug.print("Fast final cand_dist_sq = {d:.9}\n", .{cand_dist_sq});
    const d_fast = @sqrt(cand_dist_sq);
    std.debug.print("Fast d_fast = {d:.9}\n", .{d_fast});

    // Now check Fast norm and byte computation:
    const inv_two_spread: f32 = 1.0 / (2.0 * spread_px);
    const norm_fast = std.math.clamp(0.5 - d_fast * inv_two_spread, 0.0, 1.0);
    const byte_f32_fast = norm_fast * 255.0 + 0.5;
    const byte_fast = @as(u8, @intFromFloat(byte_f32_fast));
    std.debug.print("Fast norm = {d:.9}\n", .{norm_fast});
    std.debug.print("Fast byte_f32 = {d:.9}\n", .{byte_f32_fast});
    std.debug.print("Fast byte = {d}\n", .{byte_fast});

    // Compare with Baseline:
    const d_base: f32 = 3.011766200;
    const norm_base = std.math.clamp(0.5 - d_base / (2.0 * spread_px), 0.0, 1.0);
    const byte_f32_base = norm_base * 255.0 + 0.5;
    const byte_base = @as(u8, @intFromFloat(byte_f32_base));
    std.debug.print("\nComparison:\n", .{});
    std.debug.print("d_base = {d:.9}, norm_base = {d:.9}, byte_f32_base = {d:.9} -> {d}\n", .{
        d_base, norm_base, byte_f32_base, byte_base
    });
    std.debug.print("d_fast = {d:.9}, norm_fast = {d:.9}, byte_f32_fast = {d:.9} -> {d}\n", .{
        d_fast, norm_fast, byte_f32_fast, byte_fast
    });
    std.debug.print("diff d = {e}\n", .{@abs(d_base - d_fast)});
    std.debug.print("diff norm = {e}\n", .{@abs(norm_base - norm_fast)});
    std.debug.print("diff byte_f32 = {e}\n", .{@abs(byte_f32_base - byte_f32_fast)});
}
