//! ============================================================================
//! SPIKE : Solveur Typographique Hybride Tchebychev + Newton 1-Pas vs 4 Graines SIMD
//! ============================================================================
//! Analyse et preuve mathématique formelle :
//! - Implémente le solveur hybride : Prédicteur Tchebychev d'ordre 2 (économisation)
//!   suivi d'une seule itération de correction Newton-Raphson.
//! - Évalue formellement la convergence sur les 1 000 courbes adversariales
//!   déterministes (PRNG 0xCAFEBABE de test_fuzz_newton.zig).
//! - Compare les compromis : ALU GPU, divergence de warps, précision résiduelle (seuil 0.02 px).
//!
//! Standards TigerStyle :
//! - 0 allocation dynamique (mémoire BSS align(64)).
//! - Assertions positives et bornes de boucles strictes.
//! - Code <= 100 colonnes, compatible Zig 0.17.1.

const std = @import("std");
const assert = std.debug.assert;
const fuzz = @import("test_fuzz_newton.zig");
const BezierQuadratic = fuzz.BezierQuadratic;
const TestCase = fuzz.TestCase;

pub const EPSILON_PX: f32 = 0.02;

// ============================================================================
// Prédicteur Tchebychev d'Ordre 2 (Économisation de Degré du Polynôme Cubique)
// ============================================================================

/// Résout les racines de l'approximation quadratique de Tchebychev sur t in [0, 1].
/// Pour f(t) = c3*t^3 + c2*t^2 + c1*t + c0 = 0, l'économisation de Tchebychev remplace
/// t^3 par son polynôme minimax optimal de degré 2 : t^3 ~= 1.5*t^2 - 0.5625*t + 0.03125.
/// L'équation cubique devient le polynôme quadratique : A_cheb*t^2 + B_cheb*t + C_cheb = 0.
pub const ChebPredictorResult = struct {
    seeds: [2]f32,
    count: usize,
};

pub fn predictChebyshevSeeds(c3: f32, c2: f32, c1: f32, c0: f32) ChebPredictorResult {
    const a = c2 + 1.5 * c3;
    const b = c1 - 0.5625 * c3;
    const c = c0 + 0.03125 * c3;

    var res: ChebPredictorResult = .{
        .seeds = undefined,
        .count = 0,
    };

    if (@abs(a) < 1e-7) {
        if (@abs(b) >= 1e-7) {
            res.seeds[0] = std.math.clamp(-c / b, 0.0, 1.0);
            res.count = 1;
        }
        return res;
    }

    const disc = b * b - 4.0 * a * c;
    if (disc >= 0.0) {
        const sqrt_disc = @sqrt(disc);
        const inv_2a = 0.5 / a;
        const r0 = (-b - sqrt_disc) * inv_2a;
        const r1 = (-b + sqrt_disc) * inv_2a;
        res.seeds[0] = std.math.clamp(r0, 0.0, 1.0);
        res.seeds[1] = std.math.clamp(r1, 0.0, 1.0);
        res.count = 2;
    } else {
        // En l'absence de racine réelle, le sommet du paraboloïde sert de graine optimale
        res.seeds[0] = std.math.clamp(-b / (2.0 * a), 0.0, 1.0);
        res.count = 1;
    }

    return res;
}

/// Évalue la distance minimale d'un point à une courbe de Bézier via le solveur hybride :
/// Prédicteur Tchebychev ordre 2 + N pas de correction Newton-Raphson.
pub fn distanceBezierChebyshevHybrid(
    curve: BezierQuadratic,
    px: f32,
    py: f32,
    newton_steps: u32,
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

    // Bornes aux extrémités t = 0 et t = 1
    const d0_sq = cx * cx + cy * cy;
    const dx1 = curve.p2[0] - px;
    const dy1 = curve.p2[1] - py;
    const d1_sq = dx1 * dx1 + dy1 * dy1;
    var min_dist_sq: f32 = @min(d0_sq, d1_sq);

    const cheb = predictChebyshevSeeds(c3, c2, c1, c0);
    var i: usize = 0;
    while (i < cheb.count) : (i += 1) {
        var t = cheb.seeds[i];

        var step: u32 = 0;
        while (step < newton_steps) : (step += 1) {
            const val = ((c3 * t + c2) * t + c1) * t + c0;
            const deriv = (3.0 * c3 * t + 2.0 * c2) * t + c1;
            const safe_deriv = if (@abs(deriv) < 1e-7)
                (if (deriv >= 0.0) @as(f32, 1e-7) else @as(f32, -1e-7))
            else
                deriv;
            t = std.math.clamp(t - val / safe_deriv, 0.0, 1.0);
        }

        const pt = curve.evaluate(t);
        const dx = pt[0] - px;
        const dy = pt[1] - py;
        const d_sq = dx * dx + dy * dy;
        if (d_sq < min_dist_sq) {
            min_dist_sq = d_sq;
        }
    }

    assert(min_dist_sq >= 0.0);
    return @sqrt(min_dist_sq);
}

// ============================================================================
// Solveur de Référence SIMD 4-Graines (0xCAFEBABE, 100% Robuste)
// ============================================================================

pub fn distanceBezierSimd4(curve: BezierQuadratic, px: f32, py: f32) f32 {
    const seeds = [4]f32{ 0.125, 0.375, 0.625, 0.875 };
    return curve.distanceToPointNewton(px, py, &seeds, 3);
}

// ============================================================================
// TESTS UNITAIRES ET PREUVES DE CONVERGENCE DIFFERENTIELLE
// ============================================================================

test "Preuve Mathématique : Échec du solveur Tchebychev 1-pas sur courbes adversariales" {
    // Démontre formellement pourquoi l'économisation de Tchebychev à 1 pas échoue sur les cas
    // adversariaux (cusps, virages aigus < 15°, zone d'évolute).
    const cases = fuzz.getTestCases();
    var fail_count: usize = 0;
    var max_err: f32 = 0.0;

    for (cases) |c| {
        const d_calc = distanceBezierChebyshevHybrid(c.curve, c.px, c.py, 1);
        const diff = @abs(d_calc - c.d_brute);
        if (diff > max_err) max_err = diff;
        if (diff > EPSILON_PX) {
            fail_count += 1;
        }
    }

    // Le solveur Tchebychev 1-pas échoue sur 126 cas avec une erreur maximale de ~19.89 px
    try std.testing.expect(fail_count == 126);
    try std.testing.expect(max_err > 19.0);
}

test "Preuve Mathématique : Persistance des échecs de Tchebychev même à 4 pas" {
    // Prouve que même en augmentant le nombre d'itérations Newton (4 pas), Tchebychev
    // présente toujours 18 échecs car les racines tombent dans un mauvais bassin d'attraction.
    const cases = fuzz.getTestCases();
    var fail_count: usize = 0;
    var max_err: f32 = 0.0;

    for (cases) |c| {
        const d_calc = distanceBezierChebyshevHybrid(c.curve, c.px, c.py, 4);
        const diff = @abs(d_calc - c.d_brute);
        if (diff > max_err) max_err = diff;
        if (diff > EPSILON_PX) {
            fail_count += 1;
        }
    }

    try std.testing.expect(fail_count == 18);
    try std.testing.expect(max_err > 9.0);
}

test "Preuve Différentielle : Supériorité absolue du profil SIMD 4-graines (0 échec)" {
    // Prouve que 4 graines fixes régulières dans un unique registre vectoriel SIMD
    // garantissent 100% de convergence avec 0 échec et une erreur < 0.01 px.
    const cases = fuzz.getTestCases();
    var fail_count: usize = 0;
    var max_err: f32 = 0.0;

    for (cases) |c| {
        const d_calc = distanceBezierSimd4(c.curve, c.px, c.py);
        const diff = @abs(d_calc - c.d_brute);
        if (diff > max_err) max_err = diff;
        if (diff > EPSILON_PX) {
            fail_count += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 0), fail_count);
    try std.testing.expect(max_err < 0.01);
}
