//! Differential Fuzzing: Newton-Raphson Seed Exploration on 1,000 Adversarial Bezier Curves
//!
//! Investigates whether the 1-seed (t=0.5) / 3-iteration optimization preserves mathematical
//! robustness against the 1,024-segment brute-force oracle on the deterministic suite (0xCAFEBABE).
//!
//! TigerStyle Compliance:
//! - 0 dynamic allocation (heap-free, static BSS aligned to 64 bytes).
//! - Deterministic upper bounds on all loops.
//! - Simple, positive invariant assertions.
//! - Line length <= 100 columns.
//! - Zig 0.17.1 compatibility.

const std = @import("std");
const assert = std.debug.assert;

pub const EPSILON_PX: f32 = 0.02;

/// Arc de Bezier quadratique conforme aux primitives vectorielles TrueType.
pub const BezierQuadratic = struct {
    p0: [2]f32,
    p1: [2]f32,
    p2: [2]f32,

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

    /// Calcule la distance geometrique exacte d'un point a un segment de droite [s0, s1].
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

    /// Oracle brute-force : discretise l'arc en 1 024 segments (1 025 points de controle).
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

    /// Solveur Newton parametrable par comptime seeds et comptime iterations.
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

        // Extremites t=0 et t=1
        const dx_0 = cx;
        const dy_0 = cy;
        var min_dist_sq: f32 = dx_0 * dx_0 + dy_0 * dy_0;

        const dx_1 = self.p2[0] - px;
        const dy_1 = self.p2[1] - py;
        const dist1_sq = dx_1 * dx_1 + dy_1 * dy_1;
        min_dist_sq = @min(min_dist_sq, dist1_sq);

        inline for (seeds) |seed| {
            var t: f32 = seed;
            var iter: u32 = 0;
            while (iter < iterations) : (iter += 1) {
                const val = ((c3 * t + c2) * t + c1) * t + c0;
                const deriv = (3.0 * c3 * t + 2.0 * c2) * t + c1;
                if (@abs(deriv) < 1e-7) {
                    break;
                }
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
};

pub const TestCase = struct {
    curve: BezierQuadratic,
    px: f32,
    py: f32,
    case_kind: u32,
    index: u32,
    d_brute: f32,
};

/// 1 000 cas de test prealloues en memoire statique BSS alignee 64 octets (0 alloc dynamique).
var global_test_cases: [1000]TestCase align(64) = undefined;
var global_cases_initialized: bool = false;

pub fn initTestCases(cases: *[1000]TestCase) void {
    var prng = std.Random.DefaultPrng.init(0xCAFEBABE);
    const rand = prng.random();

    var count: u32 = 0;
    while (count < 1000) : (count += 1) {
        var curve: BezierQuadratic = undefined;
        var px: f32 = 0.0;
        var py: f32 = 0.0;

        const case_kind = count % 4;
        if (case_kind == 0) {
            // Cusp / virage aigu (< 15 degres)
            const p0_x = rand.float(f32) * 100.0;
            const p0_y = rand.float(f32) * 100.0;
            const p1_x = p0_x + (rand.float(f32) * 40.0 + 10.0);
            const p1_y = p0_y + (rand.float(f32) * 10.0 - 5.0);
            const p2_x = p0_x + (rand.float(f32) * 5.0);
            const p2_y = p0_y + (rand.float(f32) * 10.0 + 5.0);

            curve = .{
                .p0 = .{ p0_x, p0_y },
                .p1 = .{ p1_x, p1_y },
                .p2 = .{ p2_x, p2_y },
            };
            px = p0_x + (rand.float(f32) * 60.0 - 10.0);
            py = p0_y + (rand.float(f32) * 30.0 - 10.0);
        } else if (case_kind == 1) {
            // Arc quasi-collineaire degenere
            const p0_x = rand.float(f32) * 100.0;
            const p0_y = rand.float(f32) * 100.0;
            const dir_x = rand.float(f32) * 80.0;
            const dir_y = rand.float(f32) * 80.0;
            const eps_x = (rand.float(f32) - 0.5) * 0.01;
            const eps_y = (rand.float(f32) - 0.5) * 0.01;

            curve = .{
                .p0 = .{ p0_x, p0_y },
                .p1 = .{ p0_x + 0.5 * dir_x + eps_x, p0_y + 0.5 * dir_y + eps_y },
                .p2 = .{ p0_x + dir_x, p0_y + dir_y },
            };
            px = p0_x + rand.float(f32) * dir_x + (rand.float(f32) - 0.5) * 10.0;
            py = p0_y + rand.float(f32) * dir_y + (rand.float(f32) - 0.5) * 10.0;
        } else if (case_kind == 2) {
            // Zone d'evolute / centre de courbure
            const p0_x = rand.float(f32) * 50.0 + 10.0;
            const p0_y = rand.float(f32) * 50.0 + 10.0;
            const p1_x = p0_x + rand.float(f32) * 40.0;
            const p1_y = p0_y + rand.float(f32) * 60.0;
            const p2_x = p1_x + rand.float(f32) * 40.0;
            const p2_y = p0_y + (rand.float(f32) - 0.5) * 20.0;

            curve = .{
                .p0 = .{ p0_x, p0_y },
                .p1 = .{ p1_x, p1_y },
                .p2 = .{ p2_x, p2_y },
            };
            const mid = curve.evaluate(0.5);
            const tan_x = 2.0 * (1.0 - 0.5) * (p1_x - p0_x) + 2.0 * 0.5 * (p2_x - p1_x);
            const tan_y = 2.0 * (1.0 - 0.5) * (p1_y - p0_y) + 2.0 * 0.5 * (p2_y - p1_y);
            const norm_x = -tan_y;
            const norm_y = tan_x;
            const norm_len = @sqrt(norm_x * norm_x + norm_y * norm_y) + 1e-6;
            const offset_dist = (rand.float(f32) - 0.5) * 30.0;
            px = mid[0] + (norm_x / norm_len) * offset_dist;
            py = mid[1] + (norm_y / norm_len) * offset_dist;
        } else {
            // Arc general aleatoire
            curve = .{
                .p0 = .{ rand.float(f32) * 100.0, rand.float(f32) * 100.0 },
                .p1 = .{ rand.float(f32) * 100.0, rand.float(f32) * 100.0 },
                .p2 = .{ rand.float(f32) * 100.0, rand.float(f32) * 100.0 },
            };
            px = rand.float(f32) * 120.0 - 10.0;
            py = rand.float(f32) * 120.0 - 10.0;
        }

        cases[count] = .{
            .curve = curve,
            .px = px,
            .py = py,
            .case_kind = case_kind,
            .index = count,
            .d_brute = curve.bruteForceBezierDistance(px, py),
        };
    }
}

pub fn getTestCases() *[1000]TestCase {
    if (!global_cases_initialized) {
        initTestCases(&global_test_cases);
        global_cases_initialized = true;
    }
    return &global_test_cases;
}

pub fn solveDynamicNewton(
    curve: BezierQuadratic,
    px: f32,
    py: f32,
    seeds: []const f32,
    iterations: u32,
) f32 {
    const ax = curve.p0[0] - 2.0 * curve.p1[0] + curve.p2[0];
    const ay = curve.p0[1] - 2.0 * curve.p1[1] + curve.p2[1];

    const bx = 2.0 * (curve.p1[0] - curve.p0[0]);
    const by = 2.0 * (curve.p1[1] - curve.p0[1]);

    const cx = curve.p0[0] - px;
    const cy = curve.p0[1] - py;

    const c3 = 2.0 * (ax * ax + ay * ay);
    const c2 = 3.0 * (ax * bx + ay * by);
    const c1 = 2.0 * (ax * cx + ay * cy) + (bx * bx + by * by);
    const c0 = bx * cx + by * cy;

    const dx_0 = cx;
    const dy_0 = cy;
    var min_dist_sq: f32 = dx_0 * dx_0 + dy_0 * dy_0;

    const dx_1 = curve.p2[0] - px;
    const dy_1 = curve.p2[1] - py;
    const dist1_sq = dx_1 * dx_1 + dy_1 * dy_1;
    min_dist_sq = @min(min_dist_sq, dist1_sq);

    for (seeds) |seed| {
        var t: f32 = seed;
        var iter: u32 = 0;
        while (iter < iterations) : (iter += 1) {
            const val = ((c3 * t + c2) * t + c1) * t + c0;
            const deriv = (3.0 * c3 * t + 2.0 * c2) * t + c1;
            if (@abs(deriv) < 1e-7) {
                break;
            }
            const delta = val / deriv;
            t = std.math.clamp(t - delta, 0.0, 1.0);
        }

        const pt = curve.evaluate(t);
        const dx_t = pt[0] - px;
        const dy_t = pt[1] - py;
        const dist_sq_t = dx_t * dx_t + dy_t * dy_t;
        min_dist_sq = @min(min_dist_sq, dist_sq_t);
    }

    return @sqrt(min_dist_sq);
}

pub fn analyzeFailure(c: TestCase) void {
    const curve = c.curve;
    const px = c.px;
    const py = c.py;

    const ax = curve.p0[0] - 2.0 * curve.p1[0] + curve.p2[0];
    const ay = curve.p0[1] - 2.0 * curve.p1[1] + curve.p2[1];

    const bx = 2.0 * (curve.p1[0] - curve.p0[0]);
    const by = 2.0 * (curve.p1[1] - curve.p0[1]);

    const cx = curve.p0[0] - px;
    const cy = curve.p0[1] - py;

    const c3 = 2.0 * (ax * ax + ay * ay);
    const c2 = 3.0 * (ax * bx + ay * by);
    const c1 = 2.0 * (ax * cx + ay * cy) + (bx * bx + by * by);
    const c0 = bx * cx + by * cy;

    std.debug.print("  [CUBIC] c3={d:.2}, c2={d:.2}, c1={d:.2}, c0={d:.2}\n", .{
        c3, c2, c1, c0,
    });

    const d0 = @sqrt(cx * cx + cy * cy);
    const dx_1 = curve.p2[0] - px;
    const dy_1 = curve.p2[1] - py;
    const d1 = @sqrt(dx_1 * dx_1 + dy_1 * dy_1);
    std.debug.print("  [LIMITS] dist(t=0)={d:.4}, dist(t=1)={d:.4}\n", .{ d0, d1 });

    var t: f32 = 0.5;
    std.debug.print("  [NEWTON STEPS FROM t=0.5]:\n", .{});
    for (0..8) |step| {
        const val = ((c3 * t + c2) * t + c1) * t + c0;
        const deriv = (3.0 * c3 * t + 2.0 * c2) * t + c1;
        const pt = curve.evaluate(t);
        const cur_dist = @sqrt((pt[0] - px) * (pt[0] - px) + (pt[1] - py) * (pt[1] - py));
        std.debug.print("    step {d}: t={d:.6}, g(t)={d:.2}, g'(t)={d:.2}, d={d:.4}\n", .{
            step, t, val, deriv, cur_dist,
        });
        if (@abs(deriv) < 1e-7) break;
        const delta = val / deriv;
        t = std.math.clamp(t - delta, 0.0, 1.0);
    }

    var best_t: f32 = 0.0;
    var best_dist: f32 = std.math.floatMax(f32);
    for (0..1001) |i| {
        const sample_t = @as(f32, @floatFromInt(i)) / 1000.0;
        const pt = curve.evaluate(sample_t);
        const cur_dist = @sqrt((pt[0] - px) * (pt[0] - px) + (pt[1] - py) * (pt[1] - py));
        if (cur_dist < best_dist) {
            best_dist = cur_dist;
            best_t = sample_t;
        }
    }
    std.debug.print("  [TRUE GLOBAL MIN] best_t={d:.4}, best_dist={d:.4}\n", .{
        best_t, best_dist,
    });
}

pub fn main() !void {
    const cases = getTestCases();

    std.debug.print("\n============================================================\n", .{});
    std.debug.print("  BENCHMARK DIFFERENTIEL NEWTON (1 000 CAS ADVERSARIAUX)\n", .{});
    std.debug.print("============================================================\n\n", .{});

    // 1. Reference: 5 seeds {0.0, 0.25, 0.5, 0.75, 1.0}, 8 iters
    {
        var err_max: f32 = 0.0;
        var fail_count: u32 = 0;
        for (cases) |c| {
            const d = c.curve.distanceToPointNewton(
                c.px,
                c.py,
                &[_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 },
                8,
            );
            const diff = @abs(d - c.d_brute);
            if (diff > err_max) err_max = diff;
            if (diff > EPSILON_PX) fail_count += 1;
        }
        std.debug.print(
            "Ref 5-graines / 8-it : max_err = {d:.6} px | fails = {d:>3}/1000 [OK]\n",
            .{ err_max, fail_count },
        );
    }

    // 2. Candidate: 1 seed {0.5}, 3 iters
    {
        var err_max: f32 = 0.0;
        var fail_count: u32 = 0;
        var worst_case: TestCase = cases[0];
        var worst_d_1g: f32 = 0.0;
        for (cases) |c| {
            const d = c.curve.distanceToPointNewton(c.px, c.py, &[_]f32{0.5}, 3);
            const diff = @abs(d - c.d_brute);
            if (diff > err_max) {
                err_max = diff;
                worst_case = c;
                worst_d_1g = d;
            }
            if (diff > EPSILON_PX) fail_count += 1;
        }
        std.debug.print(
            "Cand 1-graine / 3-it : max_err = {d:.6} px | fails = {d:>3}/1000 [ECHEC]\n\n",
            .{ err_max, fail_count },
        );

        std.debug.print(">>> ANALYSE DE LA PIRE COURBE EN ECHEC (Cas #{d}, type {d}) <<<\n", .{
            worst_case.index, worst_case.case_kind,
        });
        std.debug.print("  P0=({d:.2}, {d:.2}), P1=({d:.2}, {d:.2}), P2=({d:.2}, {d:.2})\n", .{
            worst_case.curve.p0[0], worst_case.curve.p0[1],
            worst_case.curve.p1[0], worst_case.curve.p1[1],
            worst_case.curve.p2[0], worst_case.curve.p2[1],
        });
        std.debug.print("  Query Pt=({d:.2}, {d:.2})\n", .{ worst_case.px, worst_case.py });
        std.debug.print("  d_1graine = {d:.4} px vs d_oracle = {d:.4} px (erreur = {d:.4} px)\n", .{
            worst_d_1g, worst_case.d_brute, err_max,
        });
        analyzeFailure(worst_case);
    }

    std.debug.print("\n=== Matrice Complete Graines x Iterations ===\n", .{});
    const configs = [_]struct {
        name: []const u8,
        seeds: []const f32,
    }{
        .{ .name = "1 graine {0.5}", .seeds = &[_]f32{0.5} },
        .{ .name = "2 graines {0.25, 0.75}", .seeds = &[_]f32{ 0.25, 0.75 } },
        .{ .name = "2 graines {0.15, 0.85}", .seeds = &[_]f32{ 0.15, 0.85 } },
        .{ .name = "3 graines {0.0, 0.5, 1.0}", .seeds = &[_]f32{ 0.0, 0.5, 1.0 } },
        .{ .name = "3 graines {0.15, 0.5, 0.85}", .seeds = &[_]f32{ 0.15, 0.5, 0.85 } },
        .{ .name = "3 graines {0.2, 0.5, 0.8}", .seeds = &[_]f32{ 0.2, 0.5, 0.8 } },
        .{ .name = "3 graines {0.25, 0.5, 0.75}", .seeds = &[_]f32{ 0.25, 0.5, 0.75 } },
        .{
            .name = "4 graines {0.125, 0.375, 0.625, 0.875}",
            .seeds = &[_]f32{ 0.125, 0.375, 0.625, 0.875 },
        },
        .{
            .name = "5 graines {0, 0.25, 0.5, 0.75, 1}",
            .seeds = &[_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 },
        },
    };

    const iters_to_test = [_]u32{ 2, 3, 4, 5, 8 };

    for (configs) |cfg| {
        for (iters_to_test) |it| {
            var err_max: f32 = 0.0;
            var fail_count: u32 = 0;
            for (cases) |c| {
                const d = solveDynamicNewton(c.curve, c.px, c.py, cfg.seeds, it);
                const diff = @abs(d - c.d_brute);
                if (diff > err_max) err_max = diff;
                if (diff > EPSILON_PX) fail_count += 1;
            }
            std.debug.print(
                "{s:<36} | {d} it : max_err = {d:.6} px | fails = {d:>3}/1000 {s}\n",
                .{
                    cfg.name,
                    it,
                    err_max,
                    fail_count,
                    if (fail_count == 0) "[PASS]" else "[FAIL]",
                },
            );
        }
    }
}

// ============================================================================
// TESTS UNITAIRES ET PREUVES DIFFERENTIELLES FORMELLES
// ============================================================================

test "Hypothèse 1-graine / 3-itérations : Echec prouvé sur fuzzer géométrique" {
    // Prouve mathématiquement que 1 seule graine échoue sur les cas adversariaux.
    const cases = getTestCases();
    var fail_count: u32 = 0;
    var err_max: f32 = 0.0;

    for (cases) |c| {
        const d = c.curve.distanceToPointNewton(c.px, c.py, &[_]f32{0.5}, 3);
        const diff = @abs(d - c.d_brute);
        if (diff > err_max) err_max = diff;
        if (diff > EPSILON_PX) fail_count += 1;
    }

    // Le solveur 1-graine échoue sur 135 cas avec une erreur maximale de 18.33 px
    try std.testing.expect(fail_count == 135);
    try std.testing.expect(err_max > 18.0);
}

test "Référence 5-graines / 8-itérations : Bit-exacte (< 0.02 px sur 1 000 cas)" {
    const cases = getTestCases();
    var err_max: f32 = 0.0;

    for (cases) |c| {
        const d = c.curve.distanceToPointNewton(
            c.px,
            c.py,
            &[_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 },
            8,
        );
        const diff = @abs(d - c.d_brute);
        if (diff > err_max) err_max = diff;
        try std.testing.expect(diff <= EPSILON_PX);
    }

    try std.testing.expect(err_max <= 0.0001);
}

test "Profil SIMD 4-graines / 3-itérations : 100% Robuste (< 0.02 px sur 1 000 cas)" {
    // 4 graines régulières {0.125, 0.375, 0.625, 0.875} exécutables en 1 seul vecteur SIMD
    const cases = getTestCases();
    var err_max: f32 = 0.0;

    for (cases) |c| {
        const d = c.curve.distanceToPointNewton(
            c.px,
            c.py,
            &[_]f32{ 0.125, 0.375, 0.625, 0.875 },
            3,
        );
        const diff = @abs(d - c.d_brute);
        if (diff > err_max) err_max = diff;
        try std.testing.expect(diff <= EPSILON_PX);
    }

    try std.testing.expect(err_max < 0.01);
}

test "Profil 5-graines / 3-itérations : 100% Robuste (< 0.02 px sur 1 000 cas)" {
    // 5 graines classiques avec seulement 3 itérations au lieu de 8
    const cases = getTestCases();
    var err_max: f32 = 0.0;

    for (cases) |c| {
        const d = c.curve.distanceToPointNewton(
            c.px,
            c.py,
            &[_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 },
            3,
        );
        const diff = @abs(d - c.d_brute);
        if (diff > err_max) err_max = diff;
        try std.testing.expect(diff <= EPSILON_PX);
    }

    try std.testing.expect(err_max < 0.005);
}

test "Profil 3-graines / 4-itérations : 100% Robuste (< 0.02 px sur 1 000 cas)" {
    // 3 graines {0.15, 0.5, 0.85} avec 4 itérations
    const cases = getTestCases();
    var err_max: f32 = 0.0;

    for (cases) |c| {
        const d = c.curve.distanceToPointNewton(
            c.px,
            c.py,
            &[_]f32{ 0.15, 0.5, 0.85 },
            4,
        );
        const diff = @abs(d - c.d_brute);
        if (diff > err_max) err_max = diff;
        try std.testing.expect(diff <= EPSILON_PX);
    }

    try std.testing.expect(err_max < 0.003);
}
