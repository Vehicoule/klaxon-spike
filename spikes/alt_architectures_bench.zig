//! Prototype et Benchmarks pour les 5 Pistes d'Optimisation Radicale Klaxon
//! 1. Comptime-Folded Layout Tree
//! 2. Hardware Bitmask CTZ Dispatch
//! 3. Chebyshev Polynomial vs Newton Bezier
//! 4. Ring Buffer Zero-Copy Streamer

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const is_windows = (builtin.target.os.tag == .windows);
const is_wasi = (builtin.target.os.tag == .wasi);
const is_posix = (builtin.target.os.tag != .windows and
    builtin.target.os.tag != .freestanding and
    builtin.target.os.tag != .wasi);

extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;

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

// ============================================================================
// 1. COMPTIME-FOLDED LAYOUT TREE PROTOTYPE
// ============================================================================

pub const NodeKind = enum(u8) {
    fixed_box,
    elastic_container,
    text_leaf,
};

pub const ComptimeNode = struct {
    kind: NodeKind,
    width_px: f32 = 0.0,
    height_px: f32 = 0.0,
    pad_px: f32 = 0.0,
    gap_px: f32 = 0.0,
    grow: f32 = 0.0,
    children: []const ComptimeNode = &.{},
};

/// Représentation cuite (baked) à comptime d'un sous-arbre statique :
/// Tous les décalages internes relatifs (dx, dy, w, h) sont précalculés.
pub const BakedQuad = struct {
    rel_x: f32,
    rel_y: f32,
    width: f32,
    height: f32,
};

pub const BakedSubtree = struct {
    total_width: f32,
    total_height: f32,
    quad_count: usize,
    quads: [32]BakedQuad, // borne statique pour le prototype
};

pub fn bakeStaticSubtree(comptime node: ComptimeNode) BakedSubtree {
    var result: BakedSubtree = .{
        .total_width = 0.0,
        .total_height = 0.0,
        .quad_count = 0,
        .quads = undefined,
    };
    bakeNodeInternal(node, 0.0, 0.0, &result);
    return result;
}

fn bakeNodeInternal(
    comptime node: ComptimeNode,
    cur_x: f32,
    cur_y: f32,
    result: *BakedSubtree,
) void {
    const idx = result.quad_count;
    result.quads[idx] = .{
        .rel_x = cur_x,
        .rel_y = cur_y,
        .width = node.width_px,
        .height = node.height_px,
    };
    result.quad_count += 1;

    var child_x = cur_x + node.pad_px;
    const child_y = cur_y + node.pad_px;
    var max_child_h: f32 = 0.0;

    for (node.children) |child| {
        bakeNodeInternal(child, child_x, child_y, result);
        child_x += child.width_px + node.gap_px;
        if (child.height_px > max_child_h) max_child_h = child.height_px;
    }

    const subtree_w = if (node.width_px > 0) node.width_px else (child_x - node.pad_px + node.gap_px);
    const subtree_h = if (node.height_px > 0) node.height_px else (child_y + max_child_h + node.pad_px);

    if (subtree_w > result.total_width) result.total_width = subtree_w;
    if (subtree_h > result.total_height) result.total_height = subtree_h;
}

// ============================================================================
// 2. HARDWARE BITMASK CTZ DISPATCH PROTOTYPE
// ============================================================================

pub const AppFields = enum(u6) {
    scroll_offset_y = 0,
    viewport_width = 1,
    counter_value = 2,
    theme_accent = 3,
    user_name = 4,
    avatar_loaded = 5,
};

pub const FieldDispatcher = struct {
    dirty_mask: u64 = 0,

    pub inline fn markDirty(self: *FieldDispatcher, field: AppFields) void {
        self.dirty_mask |= (@as(u64, 1) << @backingInt(field));
    }

    pub inline fn dispatchCtz(self: *FieldDispatcher, handler: anytype) usize {
        var mask = self.dirty_mask;
        self.dirty_mask = 0;
        var dispatched_count: usize = 0;

        while (mask != 0) {
            const bit_idx = @ctz(mask);
            mask &= mask - 1; // BLSR hardware: 1 cycle CPU
            handler.onFieldDirty(bit_idx);
            dispatched_count += 1;
        }
        return dispatched_count;
    }
};

// ============================================================================
// 3. MAIN BENCHMARK EXECUTION
// ============================================================================

pub fn main() !void {
    std.debug.print("\n=== BENCHMARK EXPLORATEUR ARCHITECTURES KLAXON (Apple Silicon M5) ===\n\n", .{});

    // 1. Test Comptime Fold
    comptime {
        const static_card = ComptimeNode{
            .kind = .fixed_box,
            .width_px = 300.0,
            .height_px = 80.0,
            .pad_px = 12.0,
            .gap_px = 8.0,
            .children = &.{
                ComptimeNode{ .kind = .fixed_box, .width_px = 48.0, .height_px = 48.0 },
                ComptimeNode{ .kind = .text_leaf, .width_px = 120.0, .height_px = 20.0 },
                ComptimeNode{ .kind = .text_leaf, .width_px = 80.0, .height_px = 16.0 },
            },
        };
        const baked = bakeStaticSubtree(static_card);
        assert(baked.quad_count == 4);
        assert(baked.quads[0].rel_x == 0.0);
        assert(baked.quads[1].rel_x == 12.0); // icon
        assert(baked.quads[2].rel_x == 68.0); // 12 + 48 + 8
        assert(baked.quads[3].rel_x == 196.0); // 68 + 120 + 8
    }
    std.debug.print("[1. Comptime Fold] Vérification comptime réussie : 4 quads pliés en décalages constants purs.\n", .{});

    // Mesure de layout avec sous-arbres pliés vs layout déplié
    // Simuler 20 000 nœuds totaux avec 90% statique (18 000 pliés en 2 000 conteneurs)
    // Seuls les 2 000 conteneurs élastiques sont évalués au resize
    const elastic_containers = 2000;
    var container_x: [elastic_containers]i16 = undefined;
    var container_w: [elastic_containers]i16 = undefined;
    const alpha_l: [elastic_containers]i16 = @splat(1638); // 0.1 in Q14
    const beta_l: [elastic_containers]i16 = @splat(20);
    const alpha_r: [elastic_containers]i16 = @splat(8192); // 0.5 in Q14
    const beta_r: [elastic_containers]i16 = @splat(100);

    const iters = 50_000;
    const t0 = readTimestampNs();
    for (0..iters) |iter| {
        const w_i32: i32 = @intCast(1200 + (iter % 400));
        var i: usize = 0;
        const V = @Vector(16, i32);
        const w_vec: V = @splat(w_i32);
        const shift: @Vector(16, u5) = @splat(14);
        const bias: V = @splat(8192);

        while (i < elastic_containers) : (i += 16) {
            const al_16: @Vector(16, i16) = alpha_l[i..][0..16].*;
            const bl_16: @Vector(16, i16) = beta_l[i..][0..16].*;
            const ar_16: @Vector(16, i16) = alpha_r[i..][0..16].*;
            const br_16: @Vector(16, i16) = beta_r[i..][0..16].*;

            const al_32: V = al_16;
            const bl_32: V = bl_16;
            const ar_32: V = ar_16;
            const br_32: V = br_16;

            const x_32 = (al_32 * w_vec + (bl_32 << shift) + bias) >> shift;
            const r_32 = (ar_32 * w_vec + (br_32 << shift) + bias) >> shift;

            const x_i16: @Vector(16, i16) = @intCast(x_32);
            const w_i16: @Vector(16, i16) = @intCast(r_32 - x_32);
            container_x[i..][0..16].* = x_i16;
            container_w[i..][0..16].* = w_i16;
        }
        std.mem.doNotOptimizeAway(&container_x);
        std.mem.doNotOptimizeAway(&container_w);
    }
    const t1 = readTimestampNs();
    const elapsed_ns = t1 - t0;
    const ns_per_resize = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters));
    const cycles_total = ns_per_resize * 4.0;
    const cycles_per_original_node = cycles_total / 20000.0;

    std.debug.print("   • Comptime-Folded Tree (2 000 élastiques / 18 000 statiques pliés) :\n", .{});
    std.debug.print("     - Temps total par resize (20k nœuds UI) : {d:.2} ns\n", .{ns_per_resize});
    std.debug.print("     - Cycles CPU totaux @ 4.0 GHz          : {d:.1} cycles\n", .{cycles_total});
    std.debug.print("     - Cycles amortis par nœud d'UI         : {d:.4} cycle/nœud\n", .{cycles_per_original_node});
    std.debug.print("     - Accélération vs Option C (470 ns)    : {d:.1}x\n\n", .{470.0 / ns_per_resize});

    // 2. Test Hardware Bitmask CTZ Dispatch
    var dispatcher = FieldDispatcher{};
    var dummy_accumulator: u64 = 0;
    const Handler = struct {
        acc: *u64,
        pub inline fn onFieldDirty(self: @This(), bit: u32) void {
            self.acc.* +%= bit;
        }
    };
    const handler = Handler{ .acc = &dummy_accumulator };

    const ctz_iters = 1_000_000;
    const ctz_t0 = readTimestampNs();
    for (0..ctz_iters) |iter| {
        // Simuler 2 champs modifiés aléatoirement
        dispatcher.markDirty(@fromBackingInt(@intCast(iter % 6)));
        dispatcher.markDirty(@fromBackingInt(@intCast((iter + 2) % 6)));
        _ = dispatcher.dispatchCtz(handler);
    }
    const ctz_t1 = readTimestampNs();
    const ctz_elapsed_ns = ctz_t1 - ctz_t0;
    const ctz_ns_per_event = @as(f64, @floatFromInt(ctz_elapsed_ns)) / @as(f64, @floatFromInt(ctz_iters));
    const ctz_cycles = ctz_ns_per_event * 4.0;

    std.debug.print("[2. Hardware Bitmask CTZ Dispatch]\n", .{});
    std.debug.print("   • Temps par événement (2 champs invalidés) : {d:.3} ns\n", .{ctz_ns_per_event});
    std.debug.print("   • Cycles CPU par événement @ 4.0 GHz     : {d:.2} cycles\n", .{ctz_cycles});
    std.debug.print("   • Cycles par champ invalidé              : {d:.2} cycles / mutation\n", .{ctz_cycles / 2.0});
    std.debug.print("   • Amorti sur 20 000 nœuds d'UI           : {d:.6} cycle / nœud\n\n", .{ctz_cycles / 20000.0});

    std.mem.doNotOptimizeAway(&dummy_accumulator);
}
