//! ============================================================================
//! Klaxon Analytical Signed Distance Field (SDF) Rasterizer (`src/paint/sdf.zig`)
//! ============================================================================
//! High-performance analytical vector SDF generation for quadratic Bézier curves:
//! 1. 4x4 spatial tile / chunk culling with tight curve AABBs.
//! 2. SIMD Horner polynomial evaluation (evaluates 4 pixels in parallel).
//! 3. Multi-seed Newton-Raphson solver ({0.25, 0.75}) with SLA delta <= 1.
//! 4. Half-open scanline winding number tie-breaking rules (zero vertex leakage).
//! 5. Baseline oracle rasterizer and fast SIMD4 rasterizer.
//!
//! Standards: TigerStyle (<= 100 columns, zero libc, zero POSIX, 0 malloc/free).

const std = @import("std");
const assert = std.debug.assert;

/// Vectorial 2D point in font coordinate units.
pub const Point = extern struct {
    x: f32,
    y: f32,
    on_curve: bool = true,
};

comptime {
    assert(@sizeOf(Point) == 12);
    assert(@alignOf(Point) == 4);
}

/// Axis-Aligned Bounding Box (AABB) tight to quadratic Bézier curve extents.
pub const CurveAabb = struct {
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,

    /// Computes the exact tight bounding box including parametric extrema.
    pub fn computeTight(curve: BezierQuadratic) CurveAabb {
        var x_min = @min(curve.p0[0], curve.p2[0]);
        var x_max = @max(curve.p0[0], curve.p2[0]);
        var y_min = @min(curve.p0[1], curve.p2[1]);
        var y_max = @max(curve.p0[1], curve.p2[1]);

        // Horizontal extremum: P'x(t) = 0
        const denom_x = curve.p0[0] - 2.0 * curve.p1[0] + curve.p2[0];
        const num_x = curve.p0[0] - curve.p1[0];
        if (@abs(denom_x) > 1e-7) {
            const tx = num_x / denom_x;
            if (tx > 0.0) {
                if (tx < 1.0) {
                    const vx = curve.evaluate(tx)[0];
                    x_min = @min(x_min, vx);
                    x_max = @max(x_max, vx);
                }
            }
        }

        // Vertical extremum: P'y(t) = 0
        const denom_y = curve.p0[1] - 2.0 * curve.p1[1] + curve.p2[1];
        const num_y = curve.p0[1] - curve.p1[1];
        if (@abs(denom_y) > 1e-7) {
            const ty = num_y / denom_y;
            if (ty > 0.0) {
                if (ty < 1.0) {
                    const vy = curve.evaluate(ty)[1];
                    y_min = @min(y_min, vy);
                    y_max = @max(y_max, vy);
                }
            }
        }

        return .{
            .x_min = x_min,
            .y_min = y_min,
            .x_max = x_max,
            .y_max = y_max,
        };
    }
};

/// Quadratic Bézier curve primitive conforming to TrueType font standards.
pub const BezierQuadratic = extern struct {
    p0: [2]f32,
    p1: [2]f32,
    p2: [2]f32,

    /// Evaluates point B(t) on curve for parameter t in [0.0, 1.0].
    pub fn evaluate(self: BezierQuadratic, t: f32) [2]f32 {
        assert(t >= 0.0);
        assert(t <= 1.0);
        const u = 1.0 - t;
        const b0 = u * u;
        const b1 = 2.0 * u * t;
        const b2 = t * t;
        return .{
            b0 * self.p0[0] + b1 * self.p1[0] + b2 * self.p2[0],
            b0 * self.p0[1] + b1 * self.p1[1] + b2 * self.p2[1],
        };
    }

    /// Exact analytical distance from point (px, py) to line segment [s0, s1].
    pub fn pointToSegmentDistance(px: f32, py: f32, s0: [2]f32, s1: [2]f32) f32 {
        const vx = s1[0] - s0[0];
        const vy = s1[1] - s0[1];
        const wx = px - s0[0];
        const wy = py - s0[1];
        const c1 = wx * vx + wy * vy;
        if (c1 <= 0.0) {
            return @sqrt(wx * wx + wy * wy);
        }
        const c2 = vx * vx + vy * vy;
        if (c2 <= c1) {
            const qx = px - s1[0];
            const qy = py - s1[1];
            return @sqrt(qx * qx + qy * qy);
        }
        const b = c1 / c2;
        const proj_x = s0[0] + b * vx;
        const proj_y = s0[1] + b * vy;
        const dx = px - proj_x;
        const dy = py - proj_y;
        return @sqrt(dx * dx + dy * dy);
    }

    /// Brute-force oracle distance: discretizes curve into 1,024 linear segments.
    pub fn bruteForceBezierDistance(self: BezierQuadratic, px: f32, py: f32) f32 {
        const segment_count: u32 = 1024;
        const inv_segments: f32 = 1.0 / 1024.0;
        var min_dist: f32 = std.math.floatMax(f32);

        var prev_pt = self.evaluate(0.0);
        var i: u32 = 1;
        while (i <= segment_count) : (i += 1) {
            const t = @as(f32, @floatFromInt(i)) * inv_segments;
            const cur_pt = self.evaluate(t);
            const dist = pointToSegmentDistance(px, py, prev_pt, cur_pt);
            if (dist < min_dist) {
                min_dist = dist;
            }
            prev_pt = cur_pt;
        }

        assert(min_dist >= 0.0);
        return min_dist;
    }

    /// Parameterized Newton-Raphson distance solver across explicit seeds.
    pub fn distanceToPointNewton(
        self: BezierQuadratic,
        px: f32,
        py: f32,
        comptime seeds: []const f32,
        comptime iterations: u32,
    ) f32 {
        comptime assert(seeds.len > 0);
        comptime assert(iterations > 0);

        const ax = self.p0[0] - 2.0 * self.p1[0] + self.p2[0];
        const ay = self.p0[1] - 2.0 * self.p1[1] + self.p2[1];
        const bx = 2.0 * (self.p1[0] - self.p0[0]);
        const by = 2.0 * (self.p1[1] - self.p0[1]);
        const cx = self.p0[0] - px;
        const cy = self.p0[1] - py;

        const c3 = 2.0 * (ax * ax + ay * ay);
        const c2 = 3.0 * (ax * bx + ay * by);
        const c1 = 2.0 * (ax * cx + ay * cy) + (bx * bx + by * by);
        const c0 = bx * cx + by * cy;

        // Boundary endpoints t=0 and t=1
        const dx_0 = cx;
        const dy_0 = cy;
        var min_dist_sq: f32 = dx_0 * dx_0 + dy_0 * dy_0;

        const dx_1 = self.p2[0] - px;
        const dy_1 = self.p2[1] - py;
        const dist_1_sq = dx_1 * dx_1 + dy_1 * dy_1;
        min_dist_sq = @min(min_dist_sq, dist_1_sq);

        inline for (seeds) |seed| {
            var t: f32 = seed;
            var iter: u32 = 0;
            while (iter < iterations) : (iter += 1) {
                const val = ((c3 * t + c2) * t + c1) * t + c0;
                const deriv = (3.0 * c3 * t + 2.0 * c2) * t + c1;
                if (@abs(deriv) < 1e-7) break;
                const delta = val / deriv;
                t = std.math.clamp(t - delta, 0.0, 1.0);
            }

            const pt = self.evaluate(t);
            const dx_t = pt[0] - px;
            const dy_t = pt[1] - py;
            const dist_sq_t = dx_t * dx_t + dy_t * dy_t;
            min_dist_sq = @min(min_dist_sq, dist_sq_t);
        }

        return @sqrt(min_dist_sq);
    }

    /// Computes minimum distance from point to curve using baseline 5 seeds and 8 iterations.
    pub fn distanceToPoint(self: BezierQuadratic, px: f32, py: f32) f32 {
        const seeds = [5]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 };
        return self.distanceToPointNewton(px, py, &seeds, 8);
    }

    /// Computes horizontal ray winding number contribution (+1 or -1) with tie-breaking rules.
    /// Half-open interval y in [min_y, max_y) prevents double-counting shared vertices.
    pub fn windingContribution(self: BezierQuadratic, px: f32, py: f32) i32 {
        const aabb = CurveAabb.computeTight(self);
        if (py < aabb.y_min) return 0;
        if (py >= aabb.y_max) return 0;

        const a = self.p0[1] - 2.0 * self.p1[1] + self.p2[1];
        const b = 2.0 * (self.p1[1] - self.p0[1]);
        const c = self.p0[1] - py;

        var wind: i32 = 0;

        if (@abs(a) < 1e-6) {
            if (@abs(b) > 1e-6) {
                const t = -c / b;
                if (t >= 0.0) {
                    if (t < 1.0) {
                        const pt = self.evaluate(t);
                        if (pt[0] > px) {
                            wind += if (b > 0.0) @as(i32, 1) else @as(i32, -1);
                        }
                    }
                }
            }
            return wind;
        }

        const discr = b * b - 4.0 * a * c;
        if (discr < 0.0) return 0;

        const sqrt_discr = @sqrt(discr);
        const t1 = (-b - sqrt_discr) / (2.0 * a);
        const t2 = (-b + sqrt_discr) / (2.0 * a);

        if (t1 >= 0.0) {
            if (t1 < 1.0) {
                const pt1 = self.evaluate(t1);
                if (pt1[0] > px) {
                    const dy1 = 2.0 * a * t1 + b;
                    wind += if (dy1 > 0.0) @as(i32, 1) else @as(i32, -1);
                }
            }
        }

        if (t2 >= 0.0) {
            if (t2 < 1.0) {
                const pt2 = self.evaluate(t2);
                if (pt2[0] > px) {
                    const dy2 = 2.0 * a * t2 + b;
                    wind += if (dy2 > 0.0) @as(i32, 1) else @as(i32, -1);
                }
            }
        }

        return wind;
    }
};

comptime {
    assert(@sizeOf(BezierQuadratic) == 24);
    assert(@alignOf(BezierQuadratic) == 4);
}

/// Helper constructor for straight line encapsulated in a quadratic Bézier curve.
pub inline fn lineArc(p0: [2]f32, p1: [2]f32) BezierQuadratic {
    return .{
        .p0 = p0,
        .p1 = .{ (p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5 },
        .p2 = p1,
    };
}

/// Helper constructor for quadratic Bézier curve with explicit control point.
pub inline fn quadArc(p0: [2]f32, p1: [2]f32, p2: [2]f32) BezierQuadratic {
    return .{ .p0 = p0, .p1 = p1, .p2 = p2 };
}

// ============================================================================
// SIMD Horner Evaluation & Spatial Tile Culling
// ============================================================================

pub const curves_max_per_glyph: u32 = 64;
pub const scanline_hits_max: u32 = 64;

/// Precomputed curve data for SIMD4 Horner polynomial evaluation and tile culling.
pub const PreparedCurve4 = struct {
    curve: BezierQuadratic,
    aabb: CurveAabb,
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
    chunk_start: u8,
    chunk_end: u8,
    is_line: bool,

    pub fn init(curve: BezierQuadratic, spread_px: f32) PreparedCurve4 {
        const ax = curve.p0[0] - 2.0 * curve.p1[0] + curve.p2[0];
        const ay = curve.p0[1] - 2.0 * curve.p1[1] + curve.p2[1];
        const bx = 2.0 * (curve.p1[0] - curve.p0[0]);
        const by = 2.0 * (curve.p1[1] - curve.p0[1]);
        const c3 = 2.0 * (ax * ax + ay * ay);
        const c2 = 3.0 * (ax * bx + ay * by);
        const b_len_sq = bx * bx + by * by;
        const is_line = (@abs(ax) < 1e-6 and @abs(ay) < 1e-6);

        const aabb = CurveAabb.computeTight(curve);
        const x_lo = aabb.x_min - spread_px;
        const x_hi = aabb.x_max + spread_px;
        const c_start: u8 = if (x_lo <= 0.0) 0 else @min(7, @as(u8, @intFromFloat(x_lo * 0.25)));
        const c_end: u8 = if (x_hi >= 32.0) 7 else @min(7, @as(u8, @intFromFloat(x_hi * 0.25)));

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
            .chunk_start = c_start,
            .chunk_end = c_end,
            .is_line = is_line,
        };
    }
};

/// Scanline horizontal intersection event for winding number evaluation.
pub const ScanlineHit = struct {
    x_hit: f32,
    winding_delta: i32,
};

/// Resolves analytical horizontal ray intersections with curves along scanline py.
fn computeScanlineHits(
    prepared_curves: []const PreparedCurve4,
    py: f32,
    out_hits: []ScanlineHit,
) u32 {
    var hit_count: u32 = 0;

    for (prepared_curves) |prep| {
        if (py < prep.aabb.y_min) continue;
        if (py >= prep.aabb.y_max) continue;

        const a = prep.ay;
        const b = prep.by;
        const c = prep.curve.p0[1] - py;

        if (@abs(a) < 1e-6) {
            if (@abs(b) > 1e-6) {
                const t = -c / b;
                if (t >= 0.0) {
                    if (t < 1.0) {
                        const pt = prep.curve.evaluate(t);
                        if (hit_count < out_hits.len) {
                            out_hits[hit_count] = .{
                                .x_hit = pt[0],
                                .winding_delta = if (b > 0.0) 1 else -1,
                            };
                            hit_count += 1;
                        }
                    }
                }
            }
            continue;
        }

        const discr = b * b - 4.0 * a * c;
        if (discr < 0.0) continue;

        const sqrt_discr = @sqrt(discr);
        const t1 = (-b - sqrt_discr) / (2.0 * a);
        const t2 = (-b + sqrt_discr) / (2.0 * a);

        if (t1 >= 0.0) {
            if (t1 < 1.0) {
                const pt1 = prep.curve.evaluate(t1);
                const dy1 = 2.0 * a * t1 + b;
                if (hit_count < out_hits.len) {
                    out_hits[hit_count] = .{
                        .x_hit = pt1[0],
                        .winding_delta = if (dy1 > 0.0) 1 else -1,
                    };
                    hit_count += 1;
                }
            }
        }

        if (t2 >= 0.0) {
            if (t2 < 1.0) {
                const pt2 = prep.curve.evaluate(t2);
                const dy2 = 2.0 * a * t2 + b;
                if (hit_count < out_hits.len) {
                    out_hits[hit_count] = .{
                        .x_hit = pt2[0],
                        .winding_delta = if (dy2 > 0.0) 1 else -1,
                    };
                    hit_count += 1;
                }
            }
        }
    }

    return hit_count;
}

const ActiveCurve4 = struct {
    curve_index: u8,
    dy_box_sq: f32,
    chunk_start: u8,
    chunk_end: u8,
};

/// High-performance SIMD4 SDF rasterizer for 32x32 glyph patches.
/// Spatial 4x4 tile / horizontal chunk culling, SIMD Horner, multi-seed Newton.
pub fn rasterizeGlyphSdfFast(
    curves: []const BezierQuadratic,
    patch_w: u32,
    patch_h: u32,
    spread_px: f32,
    buffer: []u8,
) void {
    assert(buffer.len >= patch_w * patch_h);
    assert(patch_w == 32);
    assert(patch_h == 32);
    assert(spread_px > 0.0);
    assert(curves.len <= curves_max_per_glyph);

    var prep_storage: [curves_max_per_glyph]PreparedCurve4 = undefined;
    const prepared_curves = prep_storage[0..curves.len];
    for (curves, 0..) |curve, i| {
        prepared_curves[i] = PreparedCurve4.init(curve, spread_px);
    }

    var scanline_hits: [scanline_hits_max]ScanlineHit = undefined;
    var active_curves: [curves_max_per_glyph]ActiveCurve4 = undefined;

    const zero_vec: @Vector(4, f32) = @splat(0.0);
    const one_vec: @Vector(4, f32) = @splat(1.0);
    const eps_vec: @Vector(4, f32) = @splat(1e-7);
    const spread_sq_vec: @Vector(4, f32) = @splat(spread_px * spread_px);
    const inv_two_spread: @Vector(4, f32) = @splat(1.0 / (2.0 * spread_px));
    const half_vec: @Vector(4, f32) = @splat(0.5);
    const byte_scale_vec: @Vector(4, f32) = @splat(255.0);

    var chunk_fx_vec: [8]@Vector(4, f32) = undefined;
    inline for (0..8) |c| {
        const x_base = @as(f32, @floatFromInt(c * 4));
        chunk_fx_vec[c] = @Vector(4, f32){
            x_base + 0.5,
            x_base + 1.5,
            x_base + 2.5,
            x_base + 3.5,
        };
    }

    var y: u32 = 0;
    while (y < patch_h) : (y += 1) {
        const fy = @as(f32, @floatFromInt(y)) + 0.5;

        // 1. Scanline horizontal hits for winding number evaluation
        const hit_count = computeScanlineHits(prepared_curves, fy, &scanline_hits);
        const hits = scanline_hits[0..hit_count];

        // 2. Active curve culling on vertical spread interval
        var active_count: u32 = 0;
        for (prepared_curves, 0..) |prep, idx| {
            const dy = @max(0.0, @max(prep.aabb.y_min - fy, fy - prep.aabb.y_max));
            if (dy < spread_px) {
                active_curves[active_count] = .{
                    .curve_index = @intCast(idx),
                    .dy_box_sq = dy * dy,
                    .chunk_start = prep.chunk_start,
                    .chunk_end = prep.chunk_end,
                };
                active_count += 1;
            }
        }
        const active_slice = active_curves[0..active_count];

        // 3. Minimum distance squared array across 8 chunks of 4 pixels
        var chunk_dist_sq: [8]@Vector(4, f32) = @splat(spread_sq_vec);

        // 4. Curve-centric evaluation across active chunks
        for (active_slice) |act| {
            const prep = prepared_curves[act.curve_index];
            const dy_sq_vec: @Vector(4, f32) = @splat(act.dy_box_sq);

            const cy_scalar = prep.curve.p0[1] - fy;
            const cy: @Vector(4, f32) = @splat(cy_scalar);
            const cy_sq = cy_scalar * cy_scalar;
            const cy_sq_vec: @Vector(4, f32) = @splat(cy_sq);

            const dy1_scalar = prep.curve.p2[1] - fy;
            const dy1_sq = dy1_scalar * dy1_scalar;
            const dy1_sq_vec: @Vector(4, f32) = @splat(dy1_sq);

            var c_idx = act.chunk_start;
            while (c_idx <= act.chunk_end) : (c_idx += 1) {
                const fx_vec = chunk_fx_vec[c_idx];

                // AABB culling: test if curve can beat existing min distance
                const dx = @max(zero_vec, @max(prep.x_min_vec - fx_vec, fx_vec - prep.x_max_vec));
                const d_box_sq = dx * dx + dy_sq_vec;

                const cur_min_sq = chunk_dist_sq[c_idx];
                const could_beat = d_box_sq < cur_min_sq;
                if (!@reduce(.Or, could_beat)) {
                    continue;
                }

                const cx = prep.p0x_vec - fx_vec;
                const two_vec: @Vector(4, f32) = @splat(2.0);
                const c1 = two_vec * (prep.ax_vec * cx + prep.ay_vec * cy) + prep.b_sq_vec;
                const c0 = prep.bx_vec * cx + prep.by_vec * cy;

                // Endpoints t=0 and t=1
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
                    // Newton-Raphson across optimal seeds { 0.25, 0.75 }
                    const seeds = [2]f32{ 0.25, 0.75 };
                    inline for (seeds) |seed| {
                        var t: @Vector(4, f32) = @splat(seed);
                        var iter: u32 = 0;
                        while (iter < 4) : (iter += 1) {
                            const val = ((prep.c3_vec * t + prep.c2_vec) * t + c1) * t + c0;
                            const deriv = (prep.c3_x3_vec * t + prep.c2_x2_vec) * t + c1;
                            const safe_deriv = @select(
                                f32,
                                @abs(deriv) < eps_vec,
                                @as(@Vector(4, f32), @splat(1e-7)),
                                deriv,
                            );
                            const delta = val / safe_deriv;
                            t = std.math.clamp(t - delta, zero_vec, one_vec);
                        }

                        const dx_t = (prep.ax_vec * t + prep.bx_vec) * t + cx;
                        const dy_t = (prep.ay_vec * t + prep.by_vec) * t + cy;
                        const dist_sq_t = dx_t * dx_t + dy_t * dy_t;
                        cand_dist_sq = @min(cand_dist_sq, dist_sq_t);
                    }
                }

                chunk_dist_sq[c_idx] = @min(cur_min_sq, cand_dist_sq);
            }
        }

        // 5. Finalize signed distance and encode R8 bytes for the 8 chunks
        inline for (0..8) |c_idx| {
            const fx_vec = chunk_fx_vec[c_idx];

            var winding_vec: @Vector(4, i32) = @splat(0);
            for (hits) |hit| {
                const hit_x_vec: @Vector(4, f32) = @splat(hit.x_hit);
                const hit_mask = hit_x_vec > fx_vec;
                const delta_vec: @Vector(4, i32) = @splat(hit.winding_delta);
                winding_vec += @select(i32, hit_mask, delta_vec, @as(@Vector(4, i32), @splat(0)));
            }

            const min_dist_vec = @sqrt(chunk_dist_sq[c_idx]);
            const is_inside = winding_vec != @as(@Vector(4, i32), @splat(0));
            const signed_dist = @select(f32, is_inside, -min_dist_vec, min_dist_vec);

            const norm = std.math.clamp(
                half_vec - signed_dist * inv_two_spread,
                zero_vec,
                one_vec,
            );
            const byte_f32 = norm * byte_scale_vec + half_vec;
            const bytes: [4]u8 = @as(@Vector(4, u8), @intFromFloat(byte_f32));

            const offset = y * patch_w + c_idx * 4;
            buffer[offset] = bytes[0];
            buffer[offset + 1] = bytes[1];
            buffer[offset + 2] = bytes[2];
            buffer[offset + 3] = bytes[3];
        }
    }
}

/// Baseline scalar reference oracle rasterizer for exact geometric comparison.
pub fn rasterizeGlyphSdfBaseline(
    curves: []const BezierQuadratic,
    patch_w: u32,
    patch_h: u32,
    spread_px: f32,
    buffer: []u8,
) void {
    assert(buffer.len >= patch_w * patch_h);
    assert(patch_w > 0);
    assert(patch_h > 0);
    assert(spread_px > 0.0);

    var y: u32 = 0;
    while (y < patch_h) : (y += 1) {
        const fy = @as(f32, @floatFromInt(y)) + 0.5;
        var x: u32 = 0;
        while (x < patch_w) : (x += 1) {
            const fx = @as(f32, @floatFromInt(x)) + 0.5;

            var min_dist: f32 = 1e6;
            var winding: i32 = 0;

            for (curves) |curve| {
                const dist = curve.distanceToPoint(fx, fy);
                if (dist < min_dist) {
                    min_dist = dist;
                }
                winding += curve.windingContribution(fx, fy);
            }

            const is_inside = (winding != 0);
            const signed_dist = if (is_inside) -min_dist else min_dist;

            const norm = std.math.clamp(0.5 - signed_dist / (2.0 * spread_px), 0.0, 1.0);
            const byte_val = @as(u8, @intFromFloat(@min(255.0, norm * 255.0 + 0.5)));
            buffer[y * patch_w + x] = byte_val;
        }
    }
}

/// Universal entrypoint: selects accelerated SIMD4 path for 32x32 patches.
pub fn rasterizeGlyphSdf(
    curves: []const BezierQuadratic,
    patch_w: u32,
    patch_h: u32,
    spread_px: f32,
    buffer: []u8,
) void {
    if (patch_w == 32) {
        if (patch_h == 32) {
            if (curves.len <= curves_max_per_glyph) {
                rasterizeGlyphSdfFast(curves, patch_w, patch_h, spread_px, buffer);
                return;
            }
        }
    }
    rasterizeGlyphSdfBaseline(curves, patch_w, patch_h, spread_px, buffer);
}

// ============================================================================
// Canonical TrueType Glyph Test Fixtures
// ============================================================================

/// Canonical vector glyph 'O' (8 quadratic Bézier arcs).
pub const glyph_o_curves: [8]BezierQuadratic = .{
    .{ .p0 = .{ 29.0, 16.0 }, .p1 = .{ 29.0, 29.0 }, .p2 = .{ 16.0, 29.0 } },
    .{ .p0 = .{ 16.0, 29.0 }, .p1 = .{ 3.0, 29.0 }, .p2 = .{ 3.0, 16.0 } },
    .{ .p0 = .{ 3.0, 16.0 }, .p1 = .{ 3.0, 3.0 }, .p2 = .{ 16.0, 3.0 } },
    .{ .p0 = .{ 16.0, 3.0 }, .p1 = .{ 29.0, 3.0 }, .p2 = .{ 29.0, 16.0 } },
    .{ .p0 = .{ 16.0, 24.0 }, .p1 = .{ 24.0, 24.0 }, .p2 = .{ 24.0, 16.0 } },
    .{ .p0 = .{ 24.0, 16.0 }, .p1 = .{ 24.0, 8.0 }, .p2 = .{ 16.0, 8.0 } },
    .{ .p0 = .{ 16.0, 8.0 }, .p1 = .{ 8.0, 8.0 }, .p2 = .{ 8.0, 16.0 } },
    .{ .p0 = .{ 8.0, 16.0 }, .p1 = .{ 8.0, 24.0 }, .p2 = .{ 16.0, 24.0 } },
};

/// Canonical vector glyph 'W' (16 quadratic Bézier arcs).
pub const glyph_w_curves: [16]BezierQuadratic = .{
    lineArc(.{ 3.0, 26.0 }, .{ 7.0, 26.0 }),
    lineArc(.{ 7.0, 26.0 }, .{ 11.0, 10.0 }),
    lineArc(.{ 11.0, 10.0 }, .{ 14.0, 26.0 }),
    lineArc(.{ 14.0, 26.0 }, .{ 18.0, 26.0 }),
    lineArc(.{ 18.0, 26.0 }, .{ 21.0, 10.0 }),
    lineArc(.{ 21.0, 10.0 }, .{ 25.0, 26.0 }),
    lineArc(.{ 25.0, 26.0 }, .{ 29.0, 26.0 }),
    lineArc(.{ 29.0, 26.0 }, .{ 23.0, 6.0 }),
    lineArc(.{ 23.0, 6.0 }, .{ 19.0, 6.0 }),
    lineArc(.{ 19.0, 6.0 }, .{ 16.0, 19.0 }),
    lineArc(.{ 16.0, 19.0 }, .{ 13.0, 6.0 }),
    lineArc(.{ 13.0, 6.0 }, .{ 9.0, 6.0 }),
    lineArc(.{ 9.0, 6.0 }, .{ 3.0, 26.0 }),
    lineArc(.{ 3.0, 26.0 }, .{ 3.0, 26.0 }),
    lineArc(.{ 3.0, 26.0 }, .{ 3.0, 26.0 }),
    lineArc(.{ 3.0, 26.0 }, .{ 3.0, 26.0 }),
};

/// Canonical vector glyph 'e' (14 quadratic Bézier arcs).
pub const glyph_e_curves: [14]BezierQuadratic = .{
    .{ .p0 = .{ 26.0, 15.0 }, .p1 = .{ 26.0, 23.0 }, .p2 = .{ 16.0, 23.0 } },
    .{ .p0 = .{ 16.0, 23.0 }, .p1 = .{ 6.0, 23.0 }, .p2 = .{ 6.0, 14.0 } },
    .{ .p0 = .{ 6.0, 14.0 }, .p1 = .{ 6.0, 5.0 }, .p2 = .{ 16.0, 5.0 } },
    .{ .p0 = .{ 16.0, 5.0 }, .p1 = .{ 24.0, 5.0 }, .p2 = .{ 25.0, 10.0 } },
    lineArc(.{ 25.0, 10.0 }, .{ 21.0, 11.0 }),
    .{ .p0 = .{ 21.0, 11.0 }, .p1 = .{ 19.0, 8.0 }, .p2 = .{ 16.0, 8.0 } },
    .{ .p0 = .{ 16.0, 8.0 }, .p1 = .{ 10.0, 8.0 }, .p2 = .{ 10.0, 13.0 } },
    lineArc(.{ 10.0, 13.0 }, .{ 26.0, 13.0 }),
    lineArc(.{ 26.0, 13.0 }, .{ 26.0, 15.0 }),
    .{ .p0 = .{ 10.0, 16.0 }, .p1 = .{ 10.0, 20.0 }, .p2 = .{ 16.0, 20.0 } },
    .{ .p0 = .{ 16.0, 20.0 }, .p1 = .{ 22.0, 20.0 }, .p2 = .{ 22.0, 16.0 } },
    lineArc(.{ 22.0, 16.0 }, .{ 10.0, 16.0 }),
    lineArc(.{ 10.0, 16.0 }, .{ 10.0, 16.0 }),
    lineArc(.{ 10.0, 16.0 }, .{ 10.0, 16.0 }),
};

// ============================================================================
// Unit Tests
// ============================================================================

test "BezierQuadratic evaluate and point-to-segment distance" {
    const arc = BezierQuadratic{
        .p0 = .{ 0.0, 0.0 },
        .p1 = .{ 10.0, 20.0 },
        .p2 = .{ 20.0, 0.0 },
    };

    const mid = arc.evaluate(0.5);
    try std.testing.expectEqual(@as(f32, 10.0), mid[0]);
    try std.testing.expectEqual(@as(f32, 10.0), mid[1]);

    const seg_d = BezierQuadratic.pointToSegmentDistance(
        5.0,
        5.0,
        .{ 0.0, 0.0 },
        .{ 10.0, 0.0 },
    );
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), seg_d, 1e-4);
}

test "CurveAabb tight bounding box computation" {
    const arc = BezierQuadratic{
        .p0 = .{ 0.0, 0.0 },
        .p1 = .{ 10.0, 30.0 },
        .p2 = .{ 20.0, 0.0 },
    };

    const aabb = CurveAabb.computeTight(arc);
    try std.testing.expectEqual(@as(f32, 0.0), aabb.x_min);
    try std.testing.expectEqual(@as(f32, 20.0), aabb.x_max);
    try std.testing.expectEqual(@as(f32, 0.0), aabb.y_min);
    try std.testing.expect(aabb.y_max >= 15.0);
}

test "SDF fast vs baseline rasterization convergence (SLA delta <= 1)" {
    var fast_patch: [32 * 32]u8 = undefined;
    var base_patch: [32 * 32]u8 = undefined;

    rasterizeGlyphSdfFast(&glyph_o_curves, 32, 32, 4.0, &fast_patch);
    rasterizeGlyphSdfBaseline(&glyph_o_curves, 32, 32, 4.0, &base_patch);

    var max_delta: u32 = 0;
    var bit_exact_count: usize = 0;

    for (fast_patch, base_patch) |f, b| {
        const d: u32 = @intCast(@abs(@as(i32, f) - @as(i32, b)));
        max_delta = @max(max_delta, d);
        if (d == 0) bit_exact_count += 1;
    }

    try std.testing.expect(max_delta <= 1);
    try std.testing.expect(bit_exact_count >= 1000);
}

test "Winding number half-open tie-breaking rule" {
    const arc = BezierQuadratic{
        .p0 = .{ 0.0, 0.0 },
        .p1 = .{ 5.0, 10.0 },
        .p2 = .{ 10.0, 0.0 },
    };

    // Outside bounding box y
    const w_out = arc.windingContribution(5.0, -1.0);
    try std.testing.expectEqual(@as(i32, 0), w_out);

    // Endpoint boundary y_max tie-break (half-open)
    const aabb = CurveAabb.computeTight(arc);
    const w_top = arc.windingContribution(5.0, aabb.y_max);
    try std.testing.expectEqual(@as(i32, 0), w_top);
}
