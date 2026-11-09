//! Prototype Radical de Rasterisation SDF pour Diviser les Cycles par 100x
//!
//! Explore et benchmarque 5 architectures algorithmiques disruptives :
//! - Baseline Oracle : Rasteriseur analytique scalaire continu (vérité terrain)
//! - Baseline SIMD Fast : Rasteriseur scanline SIMD4 actuel (~5.95 us / glyphe)
//! - Option 1 : Élagage hiérarchique par tuiles 4x4 (Tile-based coarse-fine culling)
//! - Option 1B : Élagage tuiles 4x4 + SIMD16 ultra-vectorisé (16 pixels/instruction)
//! - Option 2 : Rasterisation contour 1-bit + Transformée EDT séparable de Meijster O(N)
//! - Option 3 : Résolution analytique cubique forme fermée trigonométrique (Viète / Cardano)
//! - Option 4 : Jump Flooding Algorithm (JFA) 2D vectorisé en 5 passes (log2 N)
//!
//! Conforme à TigerStyle : 0 allocation dynamique, align(64), <= 100 colonnes, zig fmt.

const std = @import("std");
const assert = std.debug.assert;
const paint_pipeline = @import("paint_pipeline.zig");
const bench_fast_sdf = @import("bench_fast_sdf.zig");

const BezierQuadratic = paint_pipeline.BezierQuadratic;
const glyph_o_curves = paint_pipeline.glyph_o_curves;
const glyph_w_curves = paint_pipeline.glyph_w_curves;
const glyph_e_curves = paint_pipeline.glyph_e_curves;

const patch_w: u32 = 32;
const patch_h: u32 = 32;
const patch_pixels: u32 = patch_w * patch_h;
const default_spread_px: f32 = 4.0;
const curves_max: u32 = 64;

comptime {
    assert(patch_w == 32);
    assert(patch_h == 32);
    assert(patch_pixels == 1024);
}

const builtin = @import("builtin");

const is_windows = (builtin.target.os.tag == .windows);
const is_wasi = (builtin.target.os.tag == .wasi);
const is_posix = (builtin.target.os.tag != .windows and
    builtin.target.os.tag != .freestanding and
    builtin.target.os.tag != .wasi);

extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;

/// Horloge monotone haute résolution (nanosecondes) compatible multi-OS.
fn readTimestampNs() u64 {
    if (is_windows) {
        var count: i64 = 0;
        var freq: i64 = 0;
        _ = QueryPerformanceCounter(&count);
        _ = QueryPerformanceFrequency(&freq);
        if (freq == 0) return 0;
        return @intCast(@divFloor(@as(i128, count) * 1_000_000_000, freq));
    } else if (is_wasi) {
        var ts: std.os.wasi.timestamp_t = 0;
        _ = std.os.wasi.clock_time_get(.MONOTONIC, 1, &ts);
        return ts;
    } else if (is_posix) {
        var ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    } else {
        return 0;
    }
}

/// Métriques statistiques rigoureuses pour l'évaluation de performance.
pub const BenchmarkStats = struct {
    mean_ns: f64,
    min_ns: f64,
    max_ns: f64,
    p99_ns: f64,
    cycles_4ghz: u64,
    max_delta: u32,
    mse: f64,
    speedup_vs_baseline: f64,
};

// ============================================================================
// Événement d'intersection de scanline pour le winding number exact
// ============================================================================

pub const ScanlineHit = struct {
    x_hit: f32,
    winding_delta: i32,
};

/// Précalcul des intersections analytiques horizontales pour les 32 scanlines.
pub fn computeAllScanlineHits(
    curves: []const BezierQuadratic,
    out_counts: *[32]u8,
    out_hits: *[32][16]ScanlineHit,
) void {
    for (0..32) |y| {
        const fy = @as(f32, @floatFromInt(y)) + 0.5;
        var count: u8 = 0;

        for (curves) |c| {
            const y_min = @min(c.p0[1], @min(c.p1[1], c.p2[1]));
            const y_max = @max(c.p0[1], @max(c.p1[1], c.p2[1]));
            if (fy < y_min or fy >= y_max) continue;

            const ay = c.p0[1] - 2.0 * c.p1[1] + c.p2[1];
            const by = 2.0 * (c.p1[1] - c.p0[1]);
            const cy = c.p0[1] - fy;

            if (@abs(ay) < 1e-6) {
                if (@abs(by) > 1e-6) {
                    const t = -cy / by;
                    if (t >= 0.0 and t < 1.0) {
                        const pt = c.evaluate(t);
                        if (count < 16) {
                            out_hits[y][count] = .{
                                .x_hit = pt[0],
                                .winding_delta = if (by > 0.0) 1 else -1,
                            };
                            count += 1;
                        }
                    }
                }
                continue;
            }

            const discr = by * by - 4.0 * ay * cy;
            if (discr < 0.0) continue;
            const sqrt_d = @sqrt(discr);
            const t1 = (-by - sqrt_d) / (2.0 * ay);
            const t2 = (-by + sqrt_d) / (2.0 * ay);

            if (t1 >= 0.0 and t1 < 1.0) {
                const pt1 = c.evaluate(t1);
                const dy1 = 2.0 * ay * t1 + by;
                if (count < 16) {
                    out_hits[y][count] = .{
                        .x_hit = pt1[0],
                        .winding_delta = if (dy1 > 0.0) 1 else -1,
                    };
                    count += 1;
                }
            }

            if (t2 >= 0.0 and t2 < 1.0) {
                const pt2 = c.evaluate(t2);
                const dy2 = 2.0 * ay * t2 + by;
                if (count < 16) {
                    out_hits[y][count] = .{
                        .x_hit = pt2[0],
                        .winding_delta = if (dy2 > 0.0) 1 else -1,
                    };
                    count += 1;
                }
            }
        }
        out_counts[y] = count;
    }
}

// ============================================================================
// Option 1 : Élagage Hiérarchique par Tuiles 4x4 (Tile-Based Coarse-Fine Culling)
// ============================================================================

pub const PreparedCurveSIMD4 = struct {
    curve: BezierQuadratic,
    aabb: bench_fast_sdf.CurveAabb,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    c3: f32,
    c2: f32,
    b_len_sq: f32,
    p0x_vec: @Vector(4, f32),
    p2x_vec: @Vector(4, f32),
    ax_vec: @Vector(4, f32),
    ay_vec: @Vector(4, f32),
    bx_vec: @Vector(4, f32),
    by_vec: @Vector(4, f32),
    c3_vec: @Vector(4, f32),
    c2_vec: @Vector(4, f32),
    c3_x3_vec: @Vector(4, f32),
    c2_x2_vec: @Vector(4, f32),
    b_sq_vec: @Vector(4, f32),
    x_min_vec: @Vector(4, f32),
    x_max_vec: @Vector(4, f32),
    is_line: bool,

    pub fn init(curve: BezierQuadratic) PreparedCurveSIMD4 {
        const ax = curve.p0[0] - 2.0 * curve.p1[0] + curve.p2[0];
        const ay = curve.p0[1] - 2.0 * curve.p1[1] + curve.p2[1];
        const bx = 2.0 * (curve.p1[0] - curve.p0[0]);
        const by = 2.0 * (curve.p1[1] - curve.p0[1]);
        const c3 = 2.0 * (ax * ax + ay * ay);
        const c2 = 3.0 * (ax * bx + ay * by);
        const b_len_sq = bx * bx + by * by;
        const is_line = (@abs(ax) < 1e-6 and @abs(ay) < 1e-6);
        const aabb = bench_fast_sdf.CurveAabb.computeTight(curve);

        return .{
            .curve = curve,
            .aabb = aabb,
            .ax = ax,
            .ay = ay,
            .bx = bx,
            .by = by,
            .c3 = c3,
            .c2 = c2,
            .b_len_sq = b_len_sq,
            .p0x_vec = @splat(curve.p0[0]),
            .p2x_vec = @splat(curve.p2[0]),
            .ax_vec = @splat(ax),
            .ay_vec = @splat(ay),
            .bx_vec = @splat(bx),
            .by_vec = @splat(by),
            .c3_vec = @splat(c3),
            .c2_vec = @splat(c2),
            .c3_x3_vec = @splat(3.0 * c3),
            .c2_x2_vec = @splat(2.0 * c2),
            .b_sq_vec = @splat(b_len_sq),
            .x_min_vec = @splat(aabb.x_min),
            .x_max_vec = @splat(aabb.x_max),
            .is_line = is_line,
        };
    }
};

pub fn rasterizeOption1TileCulling4x4(
    curves: []const BezierQuadratic,
    spread_px: f32,
    buffer: []u8,
) void {
    assert(curves.len <= curves_max);
    assert(buffer.len >= patch_pixels);

    const spread_sq = spread_px * spread_px;
    var prep_storage: [curves_max]PreparedCurveSIMD4 = undefined;
    const prep_curves = prep_storage[0..curves.len];
    for (curves, 0..) |c, i| {
        prep_curves[i] = PreparedCurveSIMD4.init(c);
    }

    var scanline_hit_counts: [32]u8 = undefined;
    var scanline_hits: [32][16]ScanlineHit = undefined;
    computeAllScanlineHits(curves, &scanline_hit_counts, &scanline_hits);

    const zero_vec: @Vector(4, f32) = @splat(0.0);
    const one_vec: @Vector(4, f32) = @splat(1.0);
    const eps_vec: @Vector(4, f32) = @splat(1e-7);
    const spread_sq_vec: @Vector(4, f32) = @splat(spread_sq);
    const inv_two_spread: @Vector(4, f32) = @splat(1.0 / (2.0 * spread_px));
    const half_vec: @Vector(4, f32) = @splat(0.5);
    const byte_scale_vec: @Vector(4, f32) = @splat(255.0);

    for (0..8) |ty| {
        const ty0 = @as(f32, @floatFromInt(ty * 4));
        const ty1 = ty0 + 4.0;

        for (0..8) |tx| {
            const tx0 = @as(f32, @floatFromInt(tx * 4));
            const tx1 = tx0 + 4.0;

            var active_indices: [64]u8 = undefined;
            var active_count: u8 = 0;

            for (prep_curves, 0..) |prep, c_idx| {
                const dx = @max(0.0, @max(prep.aabb.x_min - tx1, tx0 - prep.aabb.x_max));
                const dy = @max(0.0, @max(prep.aabb.y_min - ty1, ty0 - prep.aabb.y_max));
                if (dx * dx + dy * dy < spread_sq) {
                    active_indices[active_count] = @intCast(c_idx);
                    active_count += 1;
                }
            }

            // Cas Homogène : toute la tuile est extérieure ou intérieure
            if (active_count == 0) {
                const cx = tx0 + 2.0;
                const cy_idx = ty * 4 + 2;
                var winding: i32 = 0;
                const count = scanline_hit_counts[cy_idx];
                for (scanline_hits[cy_idx][0..count]) |hit| {
                    if (hit.x_hit > cx) winding += hit.winding_delta;
                }

                const byte_val: u8 = if (winding != 0) 255 else 0;
                const word_val: u32 = if (byte_val == 255) 0xFFFFFFFF else 0x00000000;

                inline for (0..4) |row| {
                    const offset = (ty * 4 + row) * 32 + tx * 4;
                    const ptr: *align(1) u32 = @ptrCast(&buffer[offset]);
                    ptr.* = word_val;
                }
                continue;
            }

            // Cas Frontière : évaluation vectorielle 4x4
            const fx_vec = @Vector(4, f32){
                tx0 + 0.5,
                tx0 + 1.5,
                tx0 + 2.5,
                tx0 + 3.5,
            };

            var row_dist_sq = [4]@Vector(4, f32){
                spread_sq_vec,
                spread_sq_vec,
                spread_sq_vec,
                spread_sq_vec,
            };

            for (active_indices[0..active_count]) |c_idx| {
                const prep = prep_curves[c_idx];
                const dx_box = @max(
                    zero_vec,
                    @max(prep.x_min_vec - fx_vec, fx_vec - prep.x_max_vec),
                );
                const dx_box_sq = dx_box * dx_box;

                inline for (0..4) |ly| {
                    const y = ty * 4 + ly;
                    const fy = @as(f32, @floatFromInt(y)) + 0.5;

                    const dy_box = @max(0.0, @max(prep.aabb.y_min - fy, fy - prep.aabb.y_max));
                    const dy_box_sq = dy_box * dy_box;
                    if (dy_box_sq < spread_sq) {
                        const d_box_sq = dx_box_sq + @as(@Vector(4, f32), @splat(dy_box_sq));
                        if (@reduce(.Or, d_box_sq < row_dist_sq[ly])) {
                            const cy_scalar = prep.curve.p0[1] - fy;
                            const cy = @as(@Vector(4, f32), @splat(cy_scalar));
                            const cy_sq_vec = @as(@Vector(4, f32), @splat(cy_scalar * cy_scalar));

                            const dy1_scalar = prep.curve.p2[1] - fy;
                            const dy1_sq = dy1_scalar * dy1_scalar;
                            const dy1_sq_vec = @as(@Vector(4, f32), @splat(dy1_sq));

                            const cx = prep.p0x_vec - fx_vec;
                            const two_vec: @Vector(4, f32) = @splat(2.0);
                            const c1 = two_vec * (prep.ax_vec * cx + prep.ay_vec * cy) +
                                prep.b_sq_vec;
                            const c0 = prep.bx_vec * cx + prep.by_vec * cy;

                            var cand_dist_sq = cx * cx + cy_sq_vec;
                            const dx_1 = prep.p2x_vec - fx_vec;
                            cand_dist_sq = @min(cand_dist_sq, dx_1 * dx_1 + dy1_sq_vec);

                            if (prep.is_line) {
                                const safe_c1 = @select(f32, @abs(c1) < eps_vec, one_vec, c1);
                                const t_line = std.math.clamp(-c0 / safe_c1, zero_vec, one_vec);
                                const dx_l = prep.bx_vec * t_line + cx;
                                const dy_l = prep.by_vec * t_line + cy;
                                cand_dist_sq = @min(cand_dist_sq, dx_l * dx_l + dy_l * dy_l);
                            } else {
                                var t: @Vector(4, f32) = @splat(0.5);
                                inline for (0..3) |_| {
                                    const val = ((prep.c3_vec * t + prep.c2_vec) * t + c1) * t + c0;
                                    const deriv = (prep.c3_x3_vec * t + prep.c2_x2_vec) * t + c1;
                                    const safe_deriv = @select(
                                        f32,
                                        @abs(deriv) < eps_vec,
                                        @as(@Vector(4, f32), @splat(1e-7)),
                                        deriv,
                                    );
                                    t = std.math.clamp(t - val / safe_deriv, zero_vec, one_vec);
                                }

                                const dx_t = (prep.ax_vec * t + prep.bx_vec) * t + cx;
                                const dy_t = (prep.ay_vec * t + prep.by_vec) * t + cy;
                                cand_dist_sq = @min(cand_dist_sq, dx_t * dx_t + dy_t * dy_t);
                            }

                            row_dist_sq[ly] = @min(row_dist_sq[ly], cand_dist_sq);
                        }
                    }
                }
            }

            inline for (0..4) |ly| {
                const y = ty * 4 + ly;
                var winding_vec: @Vector(4, i32) = @splat(0);
                const count = scanline_hit_counts[y];
                for (scanline_hits[y][0..count]) |hit| {
                    const hit_x_vec: @Vector(4, f32) = @splat(hit.x_hit);
                    const hit_mask = hit_x_vec > fx_vec;
                    const delta_vec: @Vector(4, i32) = @splat(hit.winding_delta);
                    const zero_delta: @Vector(4, i32) = @splat(0);
                    winding_vec += @select(i32, hit_mask, delta_vec, zero_delta);
                }

                const min_dist_vec = @sqrt(row_dist_sq[ly]);
                const is_inside = winding_vec != @as(@Vector(4, i32), @splat(0));
                const signed_dist = @select(f32, is_inside, -min_dist_vec, min_dist_vec);

                const norm = std.math.clamp(
                    half_vec - signed_dist * inv_two_spread,
                    zero_vec,
                    one_vec,
                );
                const byte_f32 = norm * byte_scale_vec + half_vec;
                const bytes: [4]u8 = @as(@Vector(4, u8), @intFromFloat(byte_f32));

                const offset = y * 32 + tx * 4;
                buffer[offset] = bytes[0];
                buffer[offset + 1] = bytes[1];
                buffer[offset + 2] = bytes[2];
                buffer[offset + 3] = bytes[3];
            }
        }
    }
}

// ============================================================================
// Option 1B : Élagage Tuiles 4x4 + SIMD16 (16 pixels/instruction)
// ============================================================================

pub const PreparedCurveSIMD16 = struct {
    curve: BezierQuadratic,
    aabb: bench_fast_sdf.CurveAabb,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    c3: f32,
    c2: f32,
    b_len_sq: f32,
    p0x_vec: @Vector(16, f32),
    p0y_vec: @Vector(16, f32),
    p2x_vec: @Vector(16, f32),
    p2y_vec: @Vector(16, f32),
    ax_vec: @Vector(16, f32),
    ay_vec: @Vector(16, f32),
    bx_vec: @Vector(16, f32),
    by_vec: @Vector(16, f32),
    c3_vec: @Vector(16, f32),
    c2_vec: @Vector(16, f32),
    c3_x3_vec: @Vector(16, f32),
    c2_x2_vec: @Vector(16, f32),
    b_sq_vec: @Vector(16, f32),
    x_min_vec: @Vector(16, f32),
    x_max_vec: @Vector(16, f32),
    y_min_vec: @Vector(16, f32),
    y_max_vec: @Vector(16, f32),
    is_line: bool,

    pub fn init(curve: BezierQuadratic) PreparedCurveSIMD16 {
        const ax = curve.p0[0] - 2.0 * curve.p1[0] + curve.p2[0];
        const ay = curve.p0[1] - 2.0 * curve.p1[1] + curve.p2[1];
        const bx = 2.0 * (curve.p1[0] - curve.p0[0]);
        const by = 2.0 * (curve.p1[1] - curve.p0[1]);
        const c3 = 2.0 * (ax * ax + ay * ay);
        const c2 = 3.0 * (ax * bx + ay * by);
        const b_len_sq = bx * bx + by * by;
        const is_line = (@abs(ax) < 1e-6 and @abs(ay) < 1e-6);
        const aabb = bench_fast_sdf.CurveAabb.computeTight(curve);

        return .{
            .curve = curve,
            .aabb = aabb,
            .ax = ax,
            .ay = ay,
            .bx = bx,
            .by = by,
            .c3 = c3,
            .c2 = c2,
            .b_len_sq = b_len_sq,
            .p0x_vec = @splat(curve.p0[0]),
            .p0y_vec = @splat(curve.p0[1]),
            .p2x_vec = @splat(curve.p2[0]),
            .p2y_vec = @splat(curve.p2[1]),
            .ax_vec = @splat(ax),
            .ay_vec = @splat(ay),
            .bx_vec = @splat(bx),
            .by_vec = @splat(by),
            .c3_vec = @splat(c3),
            .c2_vec = @splat(c2),
            .c3_x3_vec = @splat(3.0 * c3),
            .c2_x2_vec = @splat(2.0 * c2),
            .b_sq_vec = @splat(b_len_sq),
            .x_min_vec = @splat(aabb.x_min),
            .x_max_vec = @splat(aabb.x_max),
            .y_min_vec = @splat(aabb.y_min),
            .y_max_vec = @splat(aabb.y_max),
            .is_line = is_line,
        };
    }
};

pub fn rasterizeOption1bTileSIMD16(
    curves: []const BezierQuadratic,
    spread_px: f32,
    buffer: []u8,
) void {
    assert(curves.len <= curves_max);
    assert(buffer.len >= patch_pixels);

    const spread_sq = spread_px * spread_px;
    var prep_storage: [curves_max]PreparedCurveSIMD16 = undefined;
    const prep_curves = prep_storage[0..curves.len];
    for (curves, 0..) |c, i| {
        prep_curves[i] = PreparedCurveSIMD16.init(c);
    }

    var scanline_hit_counts: [32]u8 = undefined;
    var scanline_hits: [32][16]ScanlineHit = undefined;
    computeAllScanlineHits(curves, &scanline_hit_counts, &scanline_hits);

    var tile_mask: [8][8]u64 = @splat(@splat(0));
    for (prep_curves, 0..) |prep, c_idx| {
        const bit = @as(u64, 1) << @intCast(c_idx);
        const x_lo = prep.aabb.x_min - spread_px;
        const x_hi = prep.aabb.x_max + spread_px;
        const y_lo = prep.aabb.y_min - spread_px;
        const y_hi = prep.aabb.y_max + spread_px;

        const tx_start: usize = if (x_lo <= 0.0)
            0
        else
            @min(7, @as(usize, @intFromFloat(x_lo * 0.25)));
        const tx_end: usize = if (x_hi >= 32.0)
            7
        else
            @min(7, @as(usize, @intFromFloat(x_hi * 0.25)));
        const ty_start: usize = if (y_lo <= 0.0)
            0
        else
            @min(7, @as(usize, @intFromFloat(y_lo * 0.25)));
        const ty_end: usize = if (y_hi >= 32.0)
            7
        else
            @min(7, @as(usize, @intFromFloat(y_hi * 0.25)));

        var ty = ty_start;
        while (ty <= ty_end) : (ty += 1) {
            var tx = tx_start;
            while (tx <= tx_end) : (tx += 1) {
                tile_mask[ty][tx] |= bit;
            }
        }
    }

    const zero_vec: @Vector(16, f32) = @splat(0.0);
    const one_vec: @Vector(16, f32) = @splat(1.0);
    const eps_vec: @Vector(16, f32) = @splat(1e-7);
    const spread_sq_vec: @Vector(16, f32) = @splat(spread_sq);
    const inv_two_spread: @Vector(16, f32) = @splat(1.0 / (2.0 * spread_px));
    const half_vec: @Vector(16, f32) = @splat(0.5);
    const byte_scale_vec: @Vector(16, f32) = @splat(255.0);

    const rel_x = @Vector(16, f32){
        0.5, 1.5, 2.5, 3.5,
        0.5, 1.5, 2.5, 3.5,
        0.5, 1.5, 2.5, 3.5,
        0.5, 1.5, 2.5, 3.5,
    };
    const rel_y = @Vector(16, f32){
        0.5, 0.5, 0.5, 0.5,
        1.5, 1.5, 1.5, 1.5,
        2.5, 2.5, 2.5, 2.5,
        3.5, 3.5, 3.5, 3.5,
    };

    for (0..8) |ty| {
        const ty_base = @as(f32, @floatFromInt(ty * 4));
        const fy_16 = rel_y + @as(@Vector(16, f32), @splat(ty_base));

        for (0..8) |tx| {
            const mask = tile_mask[ty][tx];
            const tx_base = @as(f32, @floatFromInt(tx * 4));

            if (mask == 0) {
                const cx = tx_base + 2.0;
                const cy_idx = ty * 4 + 2;
                var winding: i32 = 0;
                const count = scanline_hit_counts[cy_idx];
                for (scanline_hits[cy_idx][0..count]) |hit| {
                    if (hit.x_hit > cx) winding += hit.winding_delta;
                }

                const byte_val: u8 = if (winding != 0) 255 else 0;
                const word_val: u32 = if (byte_val == 255) 0xFFFFFFFF else 0x00000000;

                inline for (0..4) |row| {
                    const offset = (ty * 4 + row) * 32 + tx * 4;
                    const ptr: *align(1) u32 = @ptrCast(&buffer[offset]);
                    ptr.* = word_val;
                }
                continue;
            }

            const fx_16 = rel_x + @as(@Vector(16, f32), @splat(tx_base));
            var min_dist_sq = spread_sq_vec;

            var m = mask;
            while (m != 0) {
                const c_idx = @ctz(m);
                m &= m - 1;
                const prep = prep_curves[c_idx];

                const dx_box = @max(zero_vec, @max(prep.x_min_vec - fx_16, fx_16 - prep.x_max_vec));
                const dy_box = @max(zero_vec, @max(prep.y_min_vec - fy_16, fy_16 - prep.y_max_vec));
                const d_box_sq = dx_box * dx_box + dy_box * dy_box;

                if (!@reduce(.Or, d_box_sq < min_dist_sq)) {
                    continue;
                }

                const cx = prep.p0x_vec - fx_16;
                const cy = prep.p0y_vec - fy_16;
                const two_vec: @Vector(16, f32) = @splat(2.0);
                const c1 = two_vec * (prep.ax_vec * cx + prep.ay_vec * cy) + prep.b_sq_vec;
                const c0 = prep.bx_vec * cx + prep.by_vec * cy;

                var cand_dist_sq = cx * cx + cy * cy;
                const dx_1 = prep.p2x_vec - fx_16;
                const dy_1 = prep.p2y_vec - fy_16;
                cand_dist_sq = @min(cand_dist_sq, dx_1 * dx_1 + dy_1 * dy_1);

                if (prep.is_line) {
                    const safe_c1 = @select(f32, @abs(c1) < eps_vec, one_vec, c1);
                    const t_line = std.math.clamp(-c0 / safe_c1, zero_vec, one_vec);
                    const dx_l = prep.bx_vec * t_line + cx;
                    const dy_l = prep.by_vec * t_line + cy;
                    cand_dist_sq = @min(cand_dist_sq, dx_l * dx_l + dy_l * dy_l);
                } else {
                    var t: @Vector(16, f32) = @splat(0.5);
                    inline for (0..3) |_| {
                        const val = ((prep.c3_vec * t + prep.c2_vec) * t + c1) * t + c0;
                        const deriv = (prep.c3_x3_vec * t + prep.c2_x2_vec) * t + c1;
                        const safe_deriv = @select(
                            f32,
                            @abs(deriv) < eps_vec,
                            @as(@Vector(16, f32), @splat(1e-7)),
                            deriv,
                        );
                        t = std.math.clamp(t - val / safe_deriv, zero_vec, one_vec);
                    }

                    const dx_t = (prep.ax_vec * t + prep.bx_vec) * t + cx;
                    const dy_t = (prep.ay_vec * t + prep.by_vec) * t + cy;
                    cand_dist_sq = @min(cand_dist_sq, dx_t * dx_t + dy_t * dy_t);
                }

                min_dist_sq = @min(min_dist_sq, cand_dist_sq);
            }

            var winding_vec: @Vector(16, i32) = @splat(0);
            inline for (0..4) |row| {
                const y = ty * 4 + row;
                const count = scanline_hit_counts[y];
                for (scanline_hits[y][0..count]) |hit| {
                    const hit_x = hit.x_hit;
                    const hit_delta = hit.winding_delta;

                    inline for (0..4) |col| {
                        const idx = row * 4 + col;
                        const px = tx_base + @as(f32, @floatFromInt(col)) + 0.5;
                        if (hit_x > px) {
                            winding_vec[idx] += hit_delta;
                        }
                    }
                }
            }

            const min_dist_vec = @sqrt(min_dist_sq);
            const is_inside = winding_vec != @as(@Vector(16, i32), @splat(0));
            const signed_dist = @select(f32, is_inside, -min_dist_vec, min_dist_vec);

            const norm = std.math.clamp(
                half_vec - signed_dist * inv_two_spread,
                zero_vec,
                one_vec,
            );
            const byte_f32 = norm * byte_scale_vec + half_vec;
            const bytes: [16]u8 = @as(@Vector(16, u8), @intFromFloat(byte_f32));

            inline for (0..4) |row| {
                const offset = (ty * 4 + row) * 32 + tx * 4;
                const b_row: *align(1) [4]u8 = @ptrCast(&buffer[offset]);
                b_row.* = bytes[row * 4 .. row * 4 + 4].*;
            }
        }
    }
}

// ============================================================================
// Option 2 : Rasterisation Contour + Meijster Separable 2D EDT (Linear Time)
// ============================================================================

const edt_infinity: i32 = 1000;

fn sepMeijster(i: i32, u: i32, fi: i32, fu: i32) i32 {
    const num = (u * u - i * i + fu - fi);
    const den = 2 * (u - i);
    return @divFloor(num, den);
}

pub fn rasterizeOption2MeijsterEDT(
    curves: []const BezierQuadratic,
    spread_px: f32,
    buffer: []u8,
) void {
    assert(curves.len <= curves_max);
    assert(buffer.len >= patch_pixels);

    var b_grid: [32][32]bool = @splat(@splat(false));

    // 1. Rasterisation des contours vectoriels en pixels semences 1-bit
    for (curves) |c| {
        const dx01 = c.p1[0] - c.p0[0];
        const dy01 = c.p1[1] - c.p0[1];
        const dx12 = c.p2[0] - c.p1[0];
        const dy12 = c.p2[1] - c.p1[1];
        const len_est = @sqrt(dx01 * dx01 + dy01 * dy01) + @sqrt(dx12 * dx12 + dy12 * dy12);
        const steps: u32 = @max(6, @as(u32, @intFromFloat(len_est * 2.5)));
        const inv_steps = 1.0 / @as(f32, @floatFromInt(steps));

        var s_idx: u32 = 0;
        while (s_idx <= steps) : (s_idx += 1) {
            const t = @as(f32, @floatFromInt(s_idx)) * inv_steps;
            const pt = c.evaluate(t);
            const ix = @as(i32, @intFromFloat(pt[0]));
            const iy = @as(i32, @intFromFloat(pt[1]));
            if (ix >= 0 and ix < 32 and iy >= 0 and iy < 32) {
                b_grid[@intCast(iy)][@intCast(ix)] = true;
            }
        }
    }

    // 2. Meijster Phase 1 : Transformée 1D verticale sur les colonnes
    var d1: [32][32]i32 = undefined;
    for (0..32) |x| {
        d1[0][x] = if (b_grid[0][x]) 0 else edt_infinity;
        for (1..32) |y| {
            if (b_grid[y][x]) {
                d1[y][x] = 0;
            } else if (d1[y - 1][x] < edt_infinity) {
                d1[y][x] = d1[y - 1][x] + 1;
            } else {
                d1[y][x] = edt_infinity;
            }
        }
        var y: i32 = 30;
        while (y >= 0) : (y -= 1) {
            const uy: usize = @intCast(y);
            if (d1[uy + 1][x] < edt_infinity) {
                if (d1[uy + 1][x] + 1 < d1[uy][x]) {
                    d1[uy][x] = d1[uy + 1][x] + 1;
                }
            }
        }
    }

    // 3. Meijster Phase 2 : Enveloppe parabolique 1D horizontale sur les lignes
    var d2_sq: [32][32]i32 = undefined;
    var s_buf: [32]i32 = undefined;
    var t_buf: [32]i32 = undefined;

    for (0..32) |y| {
        var f: [32]i32 = undefined;
        for (0..32) |x| {
            const v = d1[y][x];
            f[x] = if (v >= edt_infinity) edt_infinity * edt_infinity else v * v;
        }

        var q: i32 = 0;
        s_buf[0] = 0;
        t_buf[0] = 0;

        for (1..32) |u_u| {
            const u: i32 = @intCast(u_u);
            const fu = f[u_u];
            while (q >= 0) {
                const sq = s_buf[@intCast(q)];
                const fsq = f[@intCast(sq)];
                const w = sepMeijster(sq, u, fsq, fu);
                if (w <= t_buf[@intCast(q)]) {
                    q -= 1;
                } else {
                    break;
                }
            }
            if (q < 0) {
                q = 0;
                s_buf[0] = u;
                t_buf[0] = 0;
            } else {
                const sq = s_buf[@intCast(q)];
                const fsq = f[@intCast(sq)];
                const w = 1 + sepMeijster(sq, u, fsq, fu);
                if (w < 32) {
                    q += 1;
                    s_buf[@intCast(q)] = u;
                    t_buf[@intCast(q)] = w;
                }
            }
        }

        var k: usize = 0;
        for (0..32) |x| {
            while (k < @as(usize, @intCast(q)) and t_buf[k + 1] <= @as(i32, @intCast(x))) {
                k += 1;
            }
            const dx = @as(i32, @intCast(x)) - s_buf[k];
            d2_sq[y][x] = dx * dx + f[@as(usize, @intCast(s_buf[k]))];
        }
    }

    // 4. Scanline winding et quantification R8 SDF
    var scanline_hit_counts: [32]u8 = undefined;
    var scanline_hits: [32][16]ScanlineHit = undefined;
    computeAllScanlineHits(curves, &scanline_hit_counts, &scanline_hits);

    for (0..32) |y| {
        const count = scanline_hit_counts[y];
        const hits = scanline_hits[y][0..count];

        for (0..32) |x| {
            const fx = @as(f32, @floatFromInt(x)) + 0.5;
            var winding: i32 = 0;
            for (hits) |hit| {
                if (hit.x_hit > fx) winding += hit.winding_delta;
            }

            const dist = @sqrt(@as(f32, @floatFromInt(d2_sq[y][x])));
            const signed_dist = if (winding != 0) -dist else dist;
            const norm = std.math.clamp(0.5 - signed_dist / (2.0 * spread_px), 0.0, 1.0);
            buffer[y * 32 + x] = @as(u8, @intFromFloat(@min(255.0, norm * 255.0 + 0.5)));
        }
    }
}

// ============================================================================
// Option 3 : Résolution Analytique Cubique Forme Fermée (Cardano / Viète)
// ============================================================================

/// Résolution exacte trigonométrique de l'équation cubique c3*t^3 + c2*t^2 + c1*t + c0 = 0.
pub fn solveCubicVieteExact(c3: f32, c2: f32, c1: f32, c0: f32, out_roots: *[3]f32) u8 {
    if (@abs(c3) < 1e-7) {
        if (@abs(c2) < 1e-7) {
            if (@abs(c1) < 1e-7) return 0;
            out_roots[0] = -c0 / c1;
            return 1;
        }
        const discr = c1 * c1 - 4.0 * c2 * c0;
        if (discr < 0.0) return 0;
        const sqrt_d = @sqrt(discr);
        out_roots[0] = (-c1 - sqrt_d) / (2.0 * c2);
        out_roots[1] = (-c1 + sqrt_d) / (2.0 * c2);
        return 2;
    }

    const inv_c3 = 1.0 / c3;
    const a = c2 * inv_c3;
    const b = c1 * inv_c3;
    const c = c0 * inv_c3;

    const a_div_3 = a / 3.0;
    const p = b - a * a_div_3;
    const q = (2.0 * a_div_3 * a_div_3 - b) * a_div_3 + c;

    const p_div_3 = p / 3.0;
    const q_div_2 = q / 2.0;
    const discr = q_div_2 * q_div_2 + p_div_3 * p_div_3 * p_div_3;

    if (discr <= 0.0) {
        const r = @sqrt(-p_div_3);
        const phi = std.math.acos(std.math.clamp(-q_div_2 / (r * r * r), -1.0, 1.0)) / 3.0;
        const cos_phi = @cos(phi);
        const sin_phi = @sin(phi);
        const sqrt3_div_2: f32 = 0.8660254037844386;

        out_roots[0] = 2.0 * r * cos_phi - a_div_3;
        out_roots[1] = 2.0 * r * (-0.5 * cos_phi + sqrt3_div_2 * sin_phi) - a_div_3;
        out_roots[2] = 2.0 * r * (-0.5 * cos_phi - sqrt3_div_2 * sin_phi) - a_div_3;
        return 3;
    } else {
        const sqrt_discr = @sqrt(discr);
        const u = std.math.cbrt(-q_div_2 + sqrt_discr);
        const v = std.math.cbrt(-q_div_2 - sqrt_discr);
        out_roots[0] = u + v - a_div_3;
        return 1;
    }
}

pub fn rasterizeOption3VieteCardano(
    curves: []const BezierQuadratic,
    spread_px: f32,
    buffer: []u8,
) void {
    assert(curves.len <= curves_max);
    assert(buffer.len >= patch_pixels);

    var scanline_hit_counts: [32]u8 = undefined;
    var scanline_hits: [32][16]ScanlineHit = undefined;
    computeAllScanlineHits(curves, &scanline_hit_counts, &scanline_hits);

    for (0..32) |y| {
        const fy = @as(f32, @floatFromInt(y)) + 0.5;
        const count = scanline_hit_counts[y];
        const hits = scanline_hits[y][0..count];

        for (0..32) |x| {
            const fx = @as(f32, @floatFromInt(x)) + 0.5;
            var min_dist_sq: f32 = spread_px * spread_px;

            for (curves) |c| {
                const d0 = (c.p0[0] - fx) * (c.p0[0] - fx) + (c.p0[1] - fy) * (c.p0[1] - fy);
                const d1 = (c.p2[0] - fx) * (c.p2[0] - fx) + (c.p2[1] - fy) * (c.p2[1] - fy);
                var cand = @min(d0, d1);

                const ax = c.p0[0] - 2.0 * c.p1[0] + c.p2[0];
                const ay = c.p0[1] - 2.0 * c.p1[1] + c.p2[1];
                const bx = 2.0 * (c.p1[0] - c.p0[0]);
                const by = 2.0 * (c.p1[1] - c.p0[1]);
                const cx = c.p0[0] - fx;
                const cy = c.p0[1] - fy;

                const c3 = 2.0 * (ax * ax + ay * ay);
                const c2 = 3.0 * (ax * bx + ay * by);
                const b_sq = bx * bx + by * by;
                const c1 = 2.0 * (ax * cx + ay * cy) + b_sq;
                const c0 = bx * cx + by * cy;

                var roots: [3]f32 = undefined;
                const root_count = solveCubicVieteExact(c3, c2, c1, c0, &roots);
                var r_idx: u8 = 0;
                while (r_idx < root_count) : (r_idx += 1) {
                    const t = roots[r_idx];
                    if (t >= 0.0 and t <= 1.0) {
                        const dx_t = (ax * t + bx) * t + cx;
                        const dy_t = (ay * t + by) * t + cy;
                        cand = @min(cand, dx_t * dx_t + dy_t * dy_t);
                    }
                }

                min_dist_sq = @min(min_dist_sq, cand);
            }

            var winding: i32 = 0;
            for (hits) |hit| {
                if (hit.x_hit > fx) winding += hit.winding_delta;
            }

            const dist = @sqrt(min_dist_sq);
            const signed_dist = if (winding != 0) -dist else dist;
            const norm = std.math.clamp(0.5 - signed_dist / (2.0 * spread_px), 0.0, 1.0);
            buffer[y * 32 + x] = @as(u8, @intFromFloat(@min(255.0, norm * 255.0 + 0.5)));
        }
    }
}

// ============================================================================
// Option 4 : Jump Flooding Algorithm (JFA) 2D Vectorisé en 5 Passes
// ============================================================================

const JfaPoint = struct { x: i8, y: i8 };

pub fn rasterizeOption4JumpFlooding(
    curves: []const BezierQuadratic,
    spread_px: f32,
    buffer: []u8,
) void {
    assert(curves.len <= curves_max);
    assert(buffer.len >= patch_pixels);

    var cur_grid: [32][32]JfaPoint = @splat(@splat(JfaPoint{ .x = -1, .y = -1 }));

    // 1. Initialisation des semences de contours
    for (curves) |c| {
        const dx01 = c.p1[0] - c.p0[0];
        const dy01 = c.p1[1] - c.p0[1];
        const dx12 = c.p2[0] - c.p1[0];
        const dy12 = c.p2[1] - c.p1[1];
        const len_est = @sqrt(dx01 * dx01 + dy01 * dy01) + @sqrt(dx12 * dx12 + dy12 * dy12);
        const steps: u32 = @max(8, @as(u32, @intFromFloat(len_est * 3.0)));
        const inv_steps = 1.0 / @as(f32, @floatFromInt(steps));

        var s_idx: u32 = 0;
        while (s_idx <= steps) : (s_idx += 1) {
            const t = @as(f32, @floatFromInt(s_idx)) * inv_steps;
            const pt = c.evaluate(t);
            const ix = @as(i32, @intFromFloat(pt[0]));
            const iy = @as(i32, @intFromFloat(pt[1]));
            if (ix >= 0 and ix < 32 and iy >= 0 and iy < 32) {
                cur_grid[@intCast(iy)][@intCast(ix)] = .{ .x = @intCast(ix), .y = @intCast(iy) };
            }
        }
    }

    // 2. Propagation par saut en 5 passes (step = 16, 8, 4, 2, 1)
    var nxt_grid: [32][32]JfaPoint = undefined;
    const jfa_steps = [_]i32{ 16, 8, 4, 2, 1 };

    inline for (jfa_steps) |step| {
        for (0..32) |y| {
            const iy: i32 = @intCast(y);
            for (0..32) |x| {
                const ix: i32 = @intCast(x);
                var best_pt = cur_grid[y][x];
                var best_dist_sq: i32 = if (best_pt.x >= 0)
                    (ix - best_pt.x) * (ix - best_pt.x) + (iy - best_pt.y) * (iy - best_pt.y)
                else
                    100000;

                inline for ([_]i32{ -1, 0, 1 }) |dy| {
                    inline for ([_]i32{ -1, 0, 1 }) |dx| {
                        if (dx != 0 or dy != 0) {
                            const ny = iy + dy * step;
                            const nx = ix + dx * step;
                            if (nx >= 0 and nx < 32 and ny >= 0 and ny < 32) {
                                const cand = cur_grid[@intCast(ny)][@intCast(nx)];
                                if (cand.x >= 0) {
                                    const d_sq = (ix - cand.x) * (ix - cand.x) +
                                        (iy - cand.y) * (iy - cand.y);
                                    if (d_sq < best_dist_sq) {
                                        best_dist_sq = d_sq;
                                        best_pt = cand;
                                    }
                                }
                            }
                        }
                    }
                }
                nxt_grid[y][x] = best_pt;
            }
        }
        cur_grid = nxt_grid;
    }

    // 3. Scanline winding et normalisation SDF R8
    var scanline_hit_counts: [32]u8 = undefined;
    var scanline_hits: [32][16]ScanlineHit = undefined;
    computeAllScanlineHits(curves, &scanline_hit_counts, &scanline_hits);

    for (0..32) |y| {
        const count = scanline_hit_counts[y];
        const hits = scanline_hits[y][0..count];
        const iy: i32 = @intCast(y);

        for (0..32) |x| {
            const fx = @as(f32, @floatFromInt(x)) + 0.5;
            const ix: i32 = @intCast(x);

            var winding: i32 = 0;
            for (hits) |hit| {
                if (hit.x_hit > fx) winding += hit.winding_delta;
            }

            const pt = cur_grid[y][x];
            const dist: f32 = if (pt.x >= 0) blk: {
                const dx_pt = ix - pt.x;
                const dy_pt = iy - pt.y;
                break :blk @sqrt(@as(f32, @floatFromInt(dx_pt * dx_pt + dy_pt * dy_pt)));
            } else spread_px;

            const signed_dist = if (winding != 0) -dist else dist;
            const norm = std.math.clamp(0.5 - signed_dist / (2.0 * spread_px), 0.0, 1.0);
            buffer[y * 32 + x] = @as(u8, @intFromFloat(@min(255.0, norm * 255.0 + 0.5)));
        }
    }
}

// ============================================================================
// Banc d'Essai & Mesures Statistiques Comparatives
// ============================================================================

const BenchmarkKind = enum {
    baseline_oracle,
    baseline_simd_fast,
    option1_tile_culling_4x4,
    option1b_tile_simd16,
    option2_meijster_edt,
    option3_viete_cardano,
    option4_jump_flooding,
};

fn runRasterizer(
    kind: BenchmarkKind,
    curves: []const BezierQuadratic,
    spread_px: f32,
    buffer: []u8,
) void {
    switch (kind) {
        .baseline_oracle => {
            paint_pipeline.rasterizeGlyphSdfBaseline(curves, 32, 32, spread_px, buffer);
        },
        .baseline_simd_fast => {
            bench_fast_sdf.rasterizeGlyphSdfFast(curves, 32, 32, spread_px, buffer);
        },
        .option1_tile_culling_4x4 => rasterizeOption1TileCulling4x4(curves, spread_px, buffer),
        .option1b_tile_simd16 => rasterizeOption1bTileSIMD16(curves, spread_px, buffer),
        .option2_meijster_edt => rasterizeOption2MeijsterEDT(curves, spread_px, buffer),
        .option3_viete_cardano => rasterizeOption3VieteCardano(curves, spread_px, buffer),
        .option4_jump_flooding => rasterizeOption4JumpFlooding(curves, spread_px, buffer),
    }
}

fn measureStats(
    kind: BenchmarkKind,
    curves: []const BezierQuadratic,
    spread_px: f32,
    oracle_patch: []const u8,
    samples_count: u32,
    baseline_mean_ns: f64,
) BenchmarkStats {
    assert(samples_count > 0);
    assert(samples_count <= 20_000);

    var patch_buf: [patch_pixels]u8 = undefined;

    // Warmup
    var w: u32 = 0;
    while (w < 100) : (w += 1) {
        runRasterizer(kind, curves, spread_px, &patch_buf);
    }

    // Mesure de fidélité par rapport à l'oracle
    var max_delta: u32 = 0;
    var sum_sq_err: f64 = 0.0;
    for (oracle_patch, patch_buf) |o, p| {
        const delta: u32 = @intCast(@abs(@as(i32, o) - @as(i32, p)));
        max_delta = @max(max_delta, delta);
        sum_sq_err += @as(f64, @floatFromInt(delta * delta));
    }
    const mse = sum_sq_err / @as(f64, @floatFromInt(patch_pixels));

    // Échantillonnage statistique
    var durations: [20_000]u64 = undefined;
    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;

    var idx: u32 = 0;
    while (idx < samples_count) : (idx += 1) {
        const t0 = readTimestampNs();
        runRasterizer(kind, curves, spread_px, &patch_buf);
        const t1 = readTimestampNs();
        std.mem.doNotOptimizeAway(&patch_buf);

        const dt = t1 - t0;
        durations[idx] = dt;
        total_ns += dt;
        min_ns = @min(min_ns, dt);
        max_ns = @max(max_ns, dt);
    }

    std.mem.sort(u64, durations[0..samples_count], {}, std.sort.asc(u64));
    const p99_idx = @min(samples_count - 1, (samples_count * 99) / 100);
    const p99_ns = @as(f64, @floatFromInt(durations[p99_idx]));

    const mean_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(samples_count));
    const cycles_4ghz = @as(u64, @intFromFloat(mean_ns * 4.0));
    const speedup = if (baseline_mean_ns > 0.0) baseline_mean_ns / mean_ns else 1.0;

    return .{
        .mean_ns = mean_ns,
        .min_ns = @as(f64, @floatFromInt(min_ns)),
        .max_ns = @as(f64, @floatFromInt(max_ns)),
        .p99_ns = p99_ns,
        .cycles_4ghz = cycles_4ghz,
        .max_delta = max_delta,
        .mse = mse,
        .speedup_vs_baseline = speedup,
    };
}

pub fn main() void {
    std.debug.print("\n============================================================\n", .{});
    std.debug.print("  BENCHMARK RADICAL : PROTOTYPE SDF 100X (Apple Silicon M5)\n", .{});
    std.debug.print("============================================================\n\n", .{});

    var oracle_patch: [patch_pixels]u8 = undefined;

    const glyph_targets = [_]struct {
        name: []const u8,
        desc: []const u8,
        curves: []const BezierQuadratic,
    }{
        .{ .name = "O", .desc = "circulaire (8 arcs quadratiques)", .curves = &glyph_o_curves },
        .{ .name = "W", .desc = "pointes aiguës (10 segments droits)", .curves = &glyph_w_curves },
        .{ .name = "e", .desc = "non-convexe avec boucle (12 arcs)", .curves = &glyph_e_curves },
    };

    const options = [_]struct { kind: BenchmarkKind, label: []const u8, samples: u32 }{
        .{ .kind = .baseline_oracle, .label = "Baseline Oracle (Scalaire)", .samples = 150 },
        .{ .kind = .baseline_simd_fast, .label = "Baseline SIMD Fast (Scanline)", .samples = 2000 },
        .{ .kind = .option1_tile_culling_4x4, .label = "Option 1 : Tuiles 4x4", .samples = 3000 },
        .{ .kind = .option1b_tile_simd16, .label = "Option 1B: Tuiles SIMD16", .samples = 3000 },
        .{ .kind = .option2_meijster_edt, .label = "Option 2 : Meijster EDT", .samples = 2000 },
        .{ .kind = .option3_viete_cardano, .label = "Option 3 : Viète/Cardano", .samples = 100 },
        .{ .kind = .option4_jump_flooding, .label = "Option 4 : JFA 5-pass", .samples = 300 },
    };

    for (glyph_targets) |target| {
        std.debug.print("------------------------------------------------------------\n", .{});
        std.debug.print(">>> Glyphe '{s}' : {s} ({d} courbes)\n", .{
            target.name,
            target.desc,
            target.curves.len,
        });
        std.debug.print("------------------------------------------------------------\n", .{});

        paint_pipeline.rasterizeGlyphSdfBaseline(
            target.curves,
            32,
            32,
            default_spread_px,
            &oracle_patch,
        );

        const fast_ref = measureStats(
            .baseline_simd_fast,
            target.curves,
            default_spread_px,
            &oracle_patch,
            2000,
            0.0,
        );
        const baseline_simd_mean_ns = fast_ref.mean_ns;

        std.debug.print(
            "| {s: <20} | {s: >8} | {s: >7} | {s: >9} | {s: >5} | {s: >6} | {s: >5} |\n",
            .{
                "Approche",
                "Moyenne",
                "Min/P99",
                "Cycles@4G",
                "Delta",
                "MSE",
                "Gain",
            },
        );
        std.debug.print(
            "|:---------------------|:--------:|:-------:|:---------:|:-----:|:------:|:-----:|\n",
            .{},
        );

        for (options) |opt| {
            const stats = measureStats(
                opt.kind,
                target.curves,
                default_spread_px,
                &oracle_patch,
                opt.samples,
                baseline_simd_mean_ns,
            );

            var mean_buf: [32]u8 = undefined;
            const mean_str = if (stats.mean_ns >= 1000.0)
                std.fmt.bufPrint(&mean_buf, "{d:6.2} us", .{stats.mean_ns / 1000.0}) catch "err"
            else
                std.fmt.bufPrint(&mean_buf, "{d:6.1} ns", .{stats.mean_ns}) catch "err";

            var min_p99_buf: [32]u8 = undefined;
            const min_p99_str = if (stats.min_ns >= 1000.0)
                std.fmt.bufPrint(&min_p99_buf, "{d:3.0}/{d:3.0}u", .{
                    stats.min_ns / 1000.0,
                    stats.p99_ns / 1000.0,
                }) catch "err"
            else
                std.fmt.bufPrint(&min_p99_buf, "{d:3.0}/{d:3.0}n", .{
                    stats.min_ns,
                    stats.p99_ns,
                }) catch "err";

            std.debug.print(
                "| {s: <20} | {s: >8} | {s: >7} | {d:9} | {d:5} | {d:6.2} | {d:4.1}x |\n",
                .{
                    opt.label,
                    mean_str,
                    min_p99_str,
                    stats.cycles_4ghz,
                    stats.max_delta,
                    stats.mse,
                    stats.speedup_vs_baseline,
                },
            );
        }
        std.debug.print("\n", .{});
    }
}

// ============================================================================
// Tests Unitaires de Non-Régression & Validation Formelle
// ============================================================================

test "Option 1 Tile Culling: bit-exact match with baseline oracle (Delta = 0)" {
    var base_patch: [patch_pixels]u8 = undefined;
    var tile_patch: [patch_pixels]u8 = undefined;
    const test_glyphs = [_][]const BezierQuadratic{
        &glyph_o_curves,
        &glyph_w_curves,
        &glyph_e_curves,
    };
    for (test_glyphs) |curves| {
        paint_pipeline.rasterizeGlyphSdfBaseline(curves, 32, 32, 4.0, &base_patch);
        rasterizeOption1TileCulling4x4(curves, 4.0, &tile_patch);
        try std.testing.expectEqualSlices(u8, &base_patch, &tile_patch);
    }
}

test "Option 1B Tile SIMD16: bit-exact match with baseline oracle (Delta = 0)" {
    var base_patch: [patch_pixels]u8 = undefined;
    var simd16_patch: [patch_pixels]u8 = undefined;
    const test_glyphs = [_][]const BezierQuadratic{
        &glyph_o_curves,
        &glyph_w_curves,
        &glyph_e_curves,
    };
    for (test_glyphs) |curves| {
        paint_pipeline.rasterizeGlyphSdfBaseline(curves, 32, 32, 4.0, &base_patch);
        rasterizeOption1bTileSIMD16(curves, 4.0, &simd16_patch);
        try std.testing.expectEqualSlices(u8, &base_patch, &simd16_patch);
    }
}

test "Option 3 Viète/Cardano: bit-exact match with baseline oracle (Delta = 0)" {
    var base_patch: [patch_pixels]u8 = undefined;
    var viete_patch: [patch_pixels]u8 = undefined;
    const test_glyphs = [_][]const BezierQuadratic{
        &glyph_o_curves,
        &glyph_w_curves,
        &glyph_e_curves,
    };
    for (test_glyphs) |curves| {
        paint_pipeline.rasterizeGlyphSdfBaseline(curves, 32, 32, 4.0, &base_patch);
        rasterizeOption3VieteCardano(curves, 4.0, &viete_patch);
        try std.testing.expectEqualSlices(u8, &base_patch, &viete_patch);
    }
}

test "Option 2 Meijster EDT & Option 4 JFA: valid range and boundary contrast" {
    var edt_patch: [patch_pixels]u8 = undefined;
    var jfa_patch: [patch_pixels]u8 = undefined;
    rasterizeOption2MeijsterEDT(&glyph_o_curves, 4.0, &edt_patch);
    rasterizeOption4JumpFlooding(&glyph_o_curves, 4.0, &jfa_patch);

    var edt_ink: u32 = 0;
    var jfa_ink: u32 = 0;
    for (0..patch_pixels) |i| {
        if (edt_patch[i] > 128) edt_ink += 1;
        if (jfa_patch[i] > 128) jfa_ink += 1;
    }
    try std.testing.expect(edt_ink > 50);
    try std.testing.expect(jfa_ink > 50);
}
