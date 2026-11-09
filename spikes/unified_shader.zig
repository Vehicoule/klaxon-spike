//! Spike 2 : Shader Unifié Zig -> SPIR-V Vulkan + Rasterizer CPU + Génération Metal MSL (TigerStyle).
//!
//! Ce module prouve qu'un seul fichier Zig sert simultanément :
//! 1. De shader GPU compilé nativement en SPIR-V Vulkan v1.5 via le backend self-hosted Zig :
//!    `zig build-obj unified_shader.zig -target spirv32-vulkan -mcpu vulkan_v1_2 -fno-llvm -O ReleaseFast -femit-bin=quad.spv`
//! 2. De fonction de référence CPU exacte (`signedDistanceRoundedRect`, `evaluateQuadPixel`)
//!    importée directement par le moteur Headless M0 pour les tests déterministes.
//! 3. De source de vérité à `comptime` pour générer les définitions C++/MSL (Metal Shading Language)
//!    avec garantie d'alignement et d'offsets au bit près.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

/// Bloc d'uniformes partagé entre CPU (Zig), Vulkan (SPIR-V) et Metal (MSL).
/// Aligné sur 16 octets pour respecter `std140` / `float4` GPU sans padding implicite.
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

/// Calcule la distance signée (SDF) d'un point `point_offset_px` (relatif au centre)
/// vers un rectangle de demi-taille `half_extent_px` aux coins arrondis de rayon `corner_radius_px`.
/// Retourne une valeur négative à l'intérieur, zéro sur le bord, et positive à l'extérieur.
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
    const outside_distance = @sqrt(outside_vec[0] * outside_vec[0] + outside_vec[1] * outside_vec[1]);
    const inside_distance = @min(@max(corner_delta[0], corner_delta[1]), 0.0);

    return outside_distance + inside_distance - corner_radius_px;
}

/// Évalue la couleur RGBA prémultipliée d'un pixel pour un `QuadUniforms` donné.
/// Exécuté à l'identique sur GPU (SPIR-V) et sur CPU (Headless M0).
pub fn evaluateQuadPixel(fragment_coord_px: @Vector(2, f32), uniforms: QuadUniforms) @Vector(4, f32) {
    assert(uniforms.softness_px > 0.0);
    const center_vec: @Vector(2, f32) = uniforms.center_px;
    const half_extent_vec: @Vector(2, f32) = uniforms.half_extent_px;
    const fill_vec: @Vector(4, f32) = uniforms.fill_color_rgba;
    const border_vec: @Vector(4, f32) = uniforms.border_color_rgba;

    const distance_px = signedDistanceRoundedRect(
        fragment_coord_px - center_vec,
        half_extent_vec,
        uniforms.corner_radius_px,
    );

    const outer_coverage = std.math.clamp(0.5 - distance_px / uniforms.softness_px, 0.0, 1.0);
    const inner_distance_px = distance_px + uniforms.border_width_px;
    const inner_coverage = std.math.clamp(0.5 - inner_distance_px / uniforms.softness_px, 0.0, 1.0);

    const inner_weight: @Vector(4, f32) = @splat(inner_coverage);
    const border_weight: @Vector(4, f32) = @splat(1.0 - inner_coverage);
    const outer_weight: @Vector(4, f32) = @splat(outer_coverage);

    const blended_rgb = fill_vec * inner_weight + border_vec * border_weight;
    return blended_rgb * outer_weight;
}

// ============================================================================
// Point d'entrée GPU (activé uniquement lors de la compilation vers SPIR-V)
// ============================================================================

extern const gpu_vertex_pos: @Vector(2, f32) addrspace(.input);
extern const gpu_frag_coord: @Vector(4, f32) addrspace(.input);
extern var gpu_frag_color: @Vector(4, f32) addrspace(.output);
extern const gpu_uniforms: QuadUniforms addrspace(.uniform);

pub fn vertexShaderMain() callconv(.spirv_vertex) void {
    std.spirv.position_out.* = .{ gpu_vertex_pos[0], gpu_vertex_pos[1], 0.0, 1.0 };
}

pub fn fragmentShaderMain() callconv(.{ .spirv_fragment = .{} }) void {
    const coord_xy = @Vector(2, f32){ gpu_frag_coord[0], gpu_frag_coord[1] };
    gpu_frag_color = evaluateQuadPixel(coord_xy, gpu_uniforms);
}

comptime {
    if (builtin.target.cpu.arch.isSpirV()) {
        @export(&vertexShaderMain, .{ .name = "vertexShaderMain" });
        @export(&fragmentShaderMain, .{ .name = "fragmentShaderMain" });
    }
}

// ============================================================================
// Générateur Comptime Metal Shading Language (MSL) via Réflexion SoA Zig 0.17
// ============================================================================

pub fn generateMetalStruct(comptime StructType: type) []const u8 {
    comptime {
        const info = @typeInfo(StructType).@"struct";
        assert(info.layout == .@"extern");

        var source: []const u8 = "struct " ++ @typeName(StructType) ++ " {\n";
        for (info.field_names, info.field_types) |field_name, field_type| {
            const msl_type_name = switch (field_type) {
                f32 => "float",
                u32 => "uint",
                [2]f32 => "packed_float2",
                [4]f32 => "float4",
                else => @compileError("Type non supporté pour l'export Metal MSL : " ++ @typeName(field_type)),
            };
            source = source ++ "    " ++ msl_type_name ++ " " ++ field_name ++ ";\n";
        }
        source = source ++ "};\n";
        return source;
    }
}

test "Unified Shader: CPU SDF evaluation and Metal MSL comptime codegen" {
    const uniforms = QuadUniforms{
        .center_px = .{ 50.0, 50.0 },
        .half_extent_px = .{ 40.0, 20.0 },
        .fill_color_rgba = .{ 0.2, 0.6, 1.0, 1.0 },
        .border_color_rgba = .{ 1.0, 1.0, 1.0, 1.0 },
        .corner_radius_px = 6.0,
        .border_width_px = 2.0,
        .softness_px = 1.0,
    };

    // Au centre exact (50, 50) : l'intérieur est plein et la distance signée est négative.
    const center_distance = signedDistanceRoundedRect(.{ 0.0, 0.0 }, uniforms.half_extent_px, uniforms.corner_radius_px);
    try std.testing.expectEqual(@as(f32, -20.0), center_distance);

    const center_pixel = evaluateQuadPixel(.{ 50.0, 50.0 }, uniforms);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), center_pixel[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), center_pixel[3], 0.001);

    // Loin à l'extérieur (0, 0) : la couverture alpha est strictement nulle.
    const outside_pixel = evaluateQuadPixel(.{ 0.0, 0.0 }, uniforms);
    try std.testing.expectEqual(@as(f32, 0.0), outside_pixel[3]);

    // Vérifie la génération du struct Metal MSL à comptime.
    const msl_header = comptime generateMetalStruct(QuadUniforms);
    try std.testing.expect(std.mem.indexOf(u8, msl_header, "float4 fill_color_rgba;") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl_header, "float corner_radius_px;") != null);
}
