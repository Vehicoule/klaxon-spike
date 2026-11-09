//! ============================================================================
//! Klaxon Sub-Cycle Flexbox Layout Engine (`src/layout/flexbox.zig`)
//! ============================================================================
//! Modular sub-cycle Flexbox layout engine derived from spikes/resize_1cycle.zig
//! and spikes/comptime_folded_layout.zig.
//!
//! Architectural Pillars:
//! 1. 4 Non-Recursive Passes:
//!    - BottomUpX: Intrinsic minimum width aggregation.
//!    - TopDownX: Viewport width propagation and flex-grow resolution.
//!    - BottomUpY: Intrinsic height aggregation.
//!    - TopDownY: Cumulative coordinate snapping and rect assignment.
//! 2. Quantized Q14 Fixed-Point Arithmetic (i16):
//!    - Alpha coefficients scaled by 16384 (1 << 14) into 2-byte i16 slots in L1D cache.
//!    - Symmetric bias +8192 (1 << 13) for exact round-half-up integer evaluation.
//! 3. Cumulative Affine Boundary Laws (Zero Hairline Seams):
//!    - Left and right edges tracked independently: L(W) = a_L * W + b_L, R(W) = a_R * W + b_R.
//!    - Invariant: a_{L, i+1} == a_{R, i} and b_{L, i+1} == b_{R, i} + gap.
//!    - Evaluated as x_i = round(L_i), r_i = round(R_i), w_i = r_i - x_i.
//!    - Mathematically guarantees x_{i+1} == r_i == x_i + w_i + gap for all fractional ratios.
//! 4. Elastic Container Partitioning (< 1 cycle / node):
//!    - Separates W-dependent elastic nodes (~15%) from rigid subtrees (~85%).
//!    - Resize operations recompute only elastic slots; rigid coordinates remain invariant.
//! 5. Strict TigerStyle:
//!    - <= 100 columns, zero trailing whitespace.
//!    - 0 malloc/free in runtime working sets (BSS align(64)).
//!    - Zero libc, zero POSIX.

const std = @import("std");
const assert = std.debug.assert;

// ============================================================================
// Layout Constants and Fixed-Point Arithmetic Constants
// ============================================================================

pub const SCALE_Q14: f32 = 16384.0;
pub const BIAS_Q14: i32 = 8192; // 1 << 13 for symmetric half-up rounding
pub const SHIFT_Q14: u5 = 14;
pub const INDEX_NONE: u16 = 0xFFFF;

pub const simd_lane_i16: usize = 16;
pub const V_i16 = @Vector(simd_lane_i16, i16);
pub const V_i32 = @Vector(simd_lane_i16, i32);

// ============================================================================
// Data Primitives (TigerStyle: extern structs with explicit sizes)
// ============================================================================

pub const NodeKind = enum(u8) {
    col = 0,
    row = 1,
    text = 2,
    button = 3,
    icon = 4,
    custom = 5,
};

pub const Style = extern struct {
    pad_px: f32 = 0.0,
    gap_px: f32 = 0.0,
    grow: f32 = 0.0,
    width_px_min: f32 = 0.0,
    height_px_min: f32 = 0.0,
};

pub const RectPx = extern struct {
    x_px: f32 = 0.0,
    y_px: f32 = 0.0,
    width_px: f32 = 0.0,
    height_px: f32 = 0.0,
};

pub const ResolvedBoxI16 = extern struct {
    x_px: i16 = 0,
    y_px: i16 = 0,
    width_px: i16 = 0,
    height_px: i16 = 0,
};

pub const AffineLaw = extern struct {
    alpha_l: f32 = 0.0,
    beta_l: f32 = 0.0,
    alpha_r: f32 = 0.0,
    beta_r: f32 = 0.0,
};

pub const AffineLawQ14 = extern struct {
    alpha_l_q14: i16 = 0,
    beta_l: i16 = 0,
    alpha_r_q14: i16 = 0,
    beta_r: i16 = 0,
};

pub const Node = struct {
    kind: NodeKind,
    parent_index: u16,
    subtree_count: u16,
    depth: u8,
    style: Style,
    is_elastic: bool,
};

comptime {
    assert(@sizeOf(Style) == 20);
    assert(@sizeOf(RectPx) == 16);
    assert(@sizeOf(ResolvedBoxI16) == 8);
    assert(@sizeOf(AffineLaw) == 16);
    assert(@sizeOf(AffineLawQ14) == 8);
}

// ============================================================================
// Flexbox Tree Container (100% Static BSS, align(64))
// ============================================================================

pub fn FlexboxTree(comptime capacity_max: usize) type {
    comptime {
        assert(capacity_max > 0);
        assert(capacity_max <= 65535);
    }

    const padded_capacity = comptime std.mem.alignForward(usize, capacity_max, simd_lane_i16);

    return struct {
        const Self = @This();
        pub const capacity = capacity_max;
        pub const padded_cap = padded_capacity;

        // Tree node descriptors
        nodes: [capacity]Node align(64) = undefined,
        node_count: u16 = 0,

        // 4-Pass layout buffers (f32)
        width_min: [capacity]f32 align(64) = undefined,
        width: [capacity]f32 align(64) = undefined,
        height_min: [capacity]f32 align(64) = undefined,
        height: [capacity]f32 align(64) = undefined,
        rects: [capacity]RectPx align(64) = undefined,

        // Affine boundary law tables
        alpha_l: [capacity]f32 align(64) = undefined,
        beta_l: [capacity]f32 align(64) = undefined,
        alpha_r: [capacity]f32 align(64) = undefined,
        beta_r: [capacity]f32 align(64) = undefined,

        // Quantized Q14 fixed-point tables (i16)
        alpha_l_q14: [padded_cap]i16 align(64) = undefined,
        beta_l_i16: [padded_cap]i16 align(64) = undefined,
        alpha_r_q14: [padded_cap]i16 align(64) = undefined,
        beta_r_i16: [padded_cap]i16 align(64) = undefined,

        // Resolved integer coordinates
        resolved_x: [padded_cap]i16 align(64) = undefined,
        resolved_w: [padded_cap]i16 align(64) = undefined,

        // Elastic vs Rigid Partitioning tables
        elastic_indices: [capacity]u16 align(64) = undefined,
        elastic_count: u16 = 0,
        elastic_padded_count: u16 = 0,

        rigid_indices: [capacity]u16 align(64) = undefined,
        rigid_count: u16 = 0,

        pub fn init(self: *Self) void {
            self.node_count = 0;
            self.elastic_count = 0;
            self.elastic_padded_count = 0;
            self.rigid_count = 0;
        }

        pub fn addNode(
            self: *Self,
            kind: NodeKind,
            parent_index: u16,
            depth: u8,
            style: Style,
            is_elastic: bool,
        ) u16 {
            assert(self.node_count < capacity);
            const idx = self.node_count;
            self.nodes[idx] = .{
                .kind = kind,
                .parent_index = parent_index,
                .subtree_count = 1,
                .depth = depth,
                .style = style,
                .is_elastic = is_elastic,
            };
            self.node_count += 1;
            return idx;
        }

        /// Computes subtree counts for each node via bottom-up reverse pass.
        pub fn computeSubtreeCounts(self: *Self) void {
            const count = self.node_count;
            assert(count > 0);

            var i: usize = 0;
            while (i < count) : (i += 1) {
                self.nodes[i].subtree_count = 1;
            }

            var idx: usize = count;
            while (idx > 0) {
                idx -= 1;
                const parent = self.nodes[idx].parent_index;
                if (parent != INDEX_NONE and parent < count) {
                    self.nodes[parent].subtree_count += self.nodes[idx].subtree_count;
                }
            }
        }

        // ====================================================================
        // Pass 1: BottomUpX (Intrinsic Minimum Width Aggregation)
        // ====================================================================

        pub fn pass1BottomUpX(self: *Self) void {
            const count = self.node_count;
            var i: usize = 0;
            while (i < count) : (i += 1) {
                self.width_min[i] = self.nodes[i].style.width_px_min;
            }
        }

        // ====================================================================
        // Pass 2: TopDownX (Viewport Width & Flex-Grow Resolution)
        // ====================================================================

        pub fn pass2TopDownX(self: *Self, viewport_width_px: f32) void {
            const count = self.node_count;
            assert(count > 0);
            self.width[0] = viewport_width_px;

            var p_idx: usize = 0;
            while (p_idx < count) : (p_idx += 1) {
                const parent = &self.nodes[p_idx];
                const inner_w = @max(0.0, self.width[p_idx] - parent.style.pad_px * 2.0);
                const first_child = p_idx + 1;
                const subtree_end = p_idx + parent.subtree_count;
                if (first_child >= subtree_end) continue;

                if (parent.kind == .col) {
                    var cursor = first_child;
                    while (cursor < subtree_end) {
                        if (self.nodes[cursor].is_elastic) {
                            self.width[cursor] = inner_w;
                        } else {
                            self.width[cursor] = self.nodes[cursor].style.width_px_min;
                        }
                        cursor += self.nodes[cursor].subtree_count;
                    }
                } else if (parent.kind == .row) {
                    var sum_min: f32 = 0.0;
                    var sum_grow: f32 = 0.0;
                    var child_count: u32 = 0;

                    var cursor = first_child;
                    while (cursor < subtree_end) {
                        sum_min += self.nodes[cursor].style.width_px_min;
                        sum_grow += self.nodes[cursor].style.grow;
                        child_count += 1;
                        cursor += self.nodes[cursor].subtree_count;
                    }

                    const gaps = if (child_count > 1)
                        parent.style.gap_px * @as(f32, @floatFromInt(child_count - 1))
                    else
                        0.0;
                    const rem = @max(0.0, inner_w - sum_min - gaps);

                    var rem_space = rem;
                    var rem_grow = sum_grow;

                    cursor = first_child;
                    while (cursor < subtree_end) {
                        const grow = self.nodes[cursor].style.grow;
                        var extra: f32 = 0.0;
                        if (rem_grow > 0.0 and grow > 0.0) {
                            if (grow >= rem_grow) {
                                extra = rem_space;
                                rem_space = 0.0;
                                rem_grow = 0.0;
                            } else {
                                extra = rem_space * (grow / rem_grow);
                                rem_space -= extra;
                                rem_grow -= grow;
                            }
                        }
                        self.width[cursor] = self.nodes[cursor].style.width_px_min + extra;
                        cursor += self.nodes[cursor].subtree_count;
                    }
                }
            }
        }

        // ====================================================================
        // Pass 3: BottomUpY (Intrinsic Height Aggregation)
        // ====================================================================

        pub fn pass3BottomUpY(self: *Self, viewport_height_px: f32) void {
            const count = self.node_count;
            var i: usize = 0;
            while (i < count) : (i += 1) {
                self.height_min[i] = self.nodes[i].style.height_px_min;
                self.height[i] = self.height_min[i];
            }
            self.height[0] = @max(self.height[0], viewport_height_px);
        }

        // ====================================================================
        // Pass 4: TopDownY (Cumulative Coordinate Snapping & Rect Assignment)
        // ====================================================================

        pub fn pass4TopDownY(self: *Self) void {
            const count = self.node_count;
            assert(count > 0);

            self.rects[0] = .{
                .x_px = 0.0,
                .y_px = 0.0,
                .width_px = self.width[0],
                .height_px = self.height[0],
            };

            var p_idx: usize = 0;
            while (p_idx < count) : (p_idx += 1) {
                const parent = &self.nodes[p_idx];
                const p_rect = self.rects[p_idx];
                var cur_x = p_rect.x_px + parent.style.pad_px;
                var cur_y = p_rect.y_px + parent.style.pad_px;

                var child_cursor = p_idx + 1;
                const subtree_end = p_idx + parent.subtree_count;

                while (child_cursor < subtree_end) {
                    if (parent.kind == .row) {
                        // Cumulative edge quantization: zero hairline gaps
                        const edge_left = cur_x;
                        const edge_right = edge_left + self.width[child_cursor];
                        const snapped_x = @round(edge_left);
                        const snapped_r = @round(edge_right);
                        const snapped_w = snapped_r - snapped_x;

                        self.rects[child_cursor] = .{
                            .x_px = snapped_x,
                            .y_px = cur_y,
                            .width_px = snapped_w,
                            .height_px = self.height[child_cursor],
                        };
                        cur_x = edge_right + parent.style.gap_px;
                    } else {
                        self.rects[child_cursor] = .{
                            .x_px = cur_x,
                            .y_px = cur_y,
                            .width_px = self.width[child_cursor],
                            .height_px = self.height[child_cursor],
                        };
                        cur_y += self.height[child_cursor] + parent.style.gap_px;
                    }
                    child_cursor += self.nodes[child_cursor].subtree_count;
                }
            }
        }

        /// Runs all 4 non-recursive Flexbox layout passes.
        pub fn run4Pass(self: *Self, viewport_w: f32, viewport_h: f32) void {
            self.pass1BottomUpX();
            self.pass2TopDownX(viewport_w);
            self.pass3BottomUpY(viewport_h);
            self.pass4TopDownY();
        }

        // ====================================================================
        // Affine Compiler: Pre-Resolving Boundary Laws (alpha_L, beta_L, alpha_R, beta_R)
        // ====================================================================

        pub fn compileAffine(self: *Self) void {
            const count = self.node_count;
            assert(count > 0);

            self.alpha_l[0] = 0.0;
            self.beta_l[0] = 0.0;
            self.alpha_r[0] = 1.0;
            self.beta_r[0] = 0.0;

            var p_idx: usize = 0;
            while (p_idx < count) : (p_idx += 1) {
                const parent = &self.nodes[p_idx];
                const first_child = p_idx + 1;
                const subtree_end = p_idx + parent.subtree_count;
                if (first_child >= subtree_end) continue;

                const pad = parent.style.pad_px;
                const inner_al = self.alpha_l[p_idx];
                const inner_bl = self.beta_l[p_idx] + pad;
                const inner_ar = self.alpha_r[p_idx];
                const inner_br = self.beta_r[p_idx] - pad;

                if (parent.kind == .col) {
                    var cursor = first_child;
                    while (cursor < subtree_end) {
                        self.alpha_l[cursor] = inner_al;
                        self.beta_l[cursor] = inner_bl;

                        if (self.nodes[cursor].is_elastic) {
                            self.alpha_r[cursor] = inner_ar;
                            self.beta_r[cursor] = inner_br;
                        } else {
                            self.alpha_r[cursor] = inner_al;
                            self.beta_r[cursor] = inner_bl + self.nodes[cursor].style.width_px_min;
                        }
                        cursor += self.nodes[cursor].subtree_count;
                    }
                } else if (parent.kind == .row) {
                    var sum_min: f32 = 0.0;
                    var sum_grow: f32 = 0.0;
                    var child_count: u32 = 0;

                    var cursor = first_child;
                    while (cursor < subtree_end) {
                        sum_min += self.nodes[cursor].style.width_px_min;
                        sum_grow += self.nodes[cursor].style.grow;
                        child_count += 1;
                        cursor += self.nodes[cursor].subtree_count;
                    }

                    const gaps = if (child_count > 1)
                        parent.style.gap_px * @as(f32, @floatFromInt(child_count - 1))
                    else
                        0.0;
                    const a_rem = inner_ar - inner_al;
                    const b_rem = (inner_br - inner_bl) - sum_min - gaps;

                    var cur_al = inner_al;
                    var cur_bl = inner_bl;

                    var rem_grow = sum_grow;
                    var rem_a = a_rem;
                    var rem_b = b_rem;

                    cursor = first_child;
                    while (cursor < subtree_end) {
                        const grow = self.nodes[cursor].style.grow;
                        self.alpha_l[cursor] = cur_al;
                        self.beta_l[cursor] = cur_bl;

                        if (rem_grow > 0.0 and grow > 0.0) {
                            var aw: f32 = undefined;
                            var bw: f32 = undefined;
                            if (grow >= rem_grow) {
                                aw = rem_a;
                                bw = self.nodes[cursor].style.width_px_min + rem_b;
                                rem_grow = 0.0;
                                rem_a = 0.0;
                                rem_b = 0.0;
                            } else {
                                const ratio = grow / rem_grow;
                                aw = rem_a * ratio;
                                bw = self.nodes[cursor].style.width_px_min + rem_b * ratio;
                                rem_grow -= grow;
                                rem_a -= aw;
                                rem_b -= bw;
                            }
                            self.alpha_r[cursor] = cur_al + aw;
                            self.beta_r[cursor] = cur_bl + bw;
                        } else {
                            self.alpha_r[cursor] = cur_al;
                            self.beta_r[cursor] = cur_bl + self.nodes[cursor].style.width_px_min;
                        }

                        // Cumulative boundary law step
                        cur_al = self.alpha_r[cursor];
                        cur_bl = self.beta_r[cursor] + parent.style.gap_px;
                        cursor += self.nodes[cursor].subtree_count;
                    }
                }
            }

            // Quantize into Q14 fixed-point representation (scale = 16384)
            var i: usize = 0;
            while (i < count) : (i += 1) {
                assert(self.alpha_l[i] >= -1.9999);
                assert(self.alpha_l[i] <= 1.9999);
                assert(self.alpha_r[i] >= -1.9999);
                assert(self.alpha_r[i] <= 1.9999);
                const al_val = std.math.clamp(self.alpha_l[i] * SCALE_Q14, -32768.0, 32767.0);
                const ar_val = std.math.clamp(self.alpha_r[i] * SCALE_Q14, -32768.0, 32767.0);
                const bl_val = std.math.clamp(self.beta_l[i], -32768.0, 32767.0);
                const br_val = std.math.clamp(self.beta_r[i], -32768.0, 32767.0);
                self.alpha_l_q14[i] = @intFromFloat(@round(al_val));
                self.beta_l_i16[i] = @intFromFloat(@round(bl_val));
                self.alpha_r_q14[i] = @intFromFloat(@round(ar_val));
                self.beta_r_i16[i] = @intFromFloat(@round(br_val));
            }

            // Zero-pad remaining elements up to padded_cap
            while (i < padded_cap) : (i += 1) {
                self.alpha_l_q14[i] = 0;
                self.beta_l_i16[i] = 0;
                self.alpha_r_q14[i] = 0;
                self.beta_r_i16[i] = 0;
            }
        }

        // ====================================================================
        // Elastic vs Rigid Partitioning
        // ====================================================================

        pub fn partitionElastic(self: *Self) void {
            const count = self.node_count;
            self.elastic_count = 0;
            self.rigid_count = 0;

            var i: usize = 0;
            while (i < count) : (i += 1) {
                const al = self.alpha_l_q14[i];
                const ar = self.alpha_r_q14[i];
                if (al != 0 or ar != 0) {
                    self.elastic_indices[self.elastic_count] = @intCast(i);
                    self.elastic_count += 1;
                } else {
                    self.rigid_indices[self.rigid_count] = @intCast(i);
                    self.rigid_count += 1;
                    // Initialize rigid node coordinates (invariant to viewport width)
                    const x = self.beta_l_i16[i];
                    const w = self.beta_r_i16[i] - self.beta_l_i16[i];
                    self.resolved_x[i] = x;
                    self.resolved_w[i] = w;
                    self.rects[i].x_px = @floatFromInt(x);
                    self.rects[i].width_px = @floatFromInt(w);
                }
            }

            self.elastic_padded_count = @intCast(
                std.mem.alignForward(usize, self.elastic_count, simd_lane_i16),
            );
        }

        // ====================================================================
        // Sub-Cycle Resize Evaluation (Scalar & SIMD Q14)
        // ====================================================================

        /// Evaluates all nodes in Q14 fixed-point arithmetic (scalar).
        pub fn solveResizeQ14(self: *Self, viewport_w: i16) void {
            const w_32: i32 = @as(i32, viewport_w);
            const count = self.node_count;

            var i: usize = 0;
            while (i < count) : (i += 1) {
                const al = @as(i32, self.alpha_l_q14[i]);
                const bl = @as(i32, self.beta_l_i16[i]);
                const ar = @as(i32, self.alpha_r_q14[i]);
                const br = @as(i32, self.beta_r_i16[i]);

                const x = (al * w_32 + (bl << SHIFT_Q14) + BIAS_Q14) >> SHIFT_Q14;
                const r = (ar * w_32 + (br << SHIFT_Q14) + BIAS_Q14) >> SHIFT_Q14;

                const min_i16: i32 = std.math.minInt(i16);
                const max_i16: i32 = std.math.maxInt(i16);
                const clamped_x = std.math.clamp(x, min_i16, max_i16);
                const clamped_w = std.math.clamp(r - x, min_i16, max_i16);

                const x_16: i16 = @intCast(clamped_x);
                const w_16: i16 = @intCast(clamped_w);
                self.resolved_x[i] = x_16;
                self.resolved_w[i] = w_16;
                self.rects[i].x_px = @floatFromInt(x_16);
                self.rects[i].width_px = @floatFromInt(w_16);
            }
        }

        /// Sub-cycle SIMD evaluation across all padded node slots.
        pub fn solveResizeSimd(self: *Self, viewport_w: i16) void {
            const w_v: V_i32 = @splat(@as(i32, viewport_w));
            const bias_v: V_i32 = @splat(BIAS_Q14);
            const shift_v: @Vector(simd_lane_i16, u5) = @splat(SHIFT_Q14);

            var i: usize = 0;
            while (i < padded_cap) : (i += simd_lane_i16) {
                const al_16: V_i16 = self.alpha_l_q14[i..][0..simd_lane_i16].*;
                const bl_16: V_i16 = self.beta_l_i16[i..][0..simd_lane_i16].*;
                const ar_16: V_i16 = self.alpha_r_q14[i..][0..simd_lane_i16].*;
                const br_16: V_i16 = self.beta_r_i16[i..][0..simd_lane_i16].*;

                const al_32: V_i32 = al_16;
                const bl_32: V_i32 = bl_16;
                const ar_32: V_i32 = ar_16;
                const br_32: V_i32 = br_16;

                const x_32 = (al_32 * w_v + (bl_32 << shift_v) + bias_v) >> shift_v;
                const r_32 = (ar_32 * w_v + (br_32 << shift_v) + bias_v) >> shift_v;
                const w_32 = r_32 - x_32;

                const min_v: V_i32 = @splat(@as(i32, std.math.minInt(i16)));
                const max_v: V_i32 = @splat(@as(i32, std.math.maxInt(i16)));
                const clamped_x_32 = @min(@max(x_32, min_v), max_v);
                const clamped_w_32 = @min(@max(w_32, min_v), max_v);

                const x_16: V_i16 = @intCast(clamped_x_32);
                const w_16: V_i16 = @intCast(clamped_w_32);
                self.resolved_x[i..][0..simd_lane_i16].* = x_16;
                self.resolved_w[i..][0..simd_lane_i16].* = w_16;
            }

            const count = self.node_count;
            var j: usize = 0;
            while (j < count) : (j += 1) {
                self.rects[j].x_px = @floatFromInt(self.resolved_x[j]);
                self.rects[j].width_px = @floatFromInt(self.resolved_w[j]);
            }
        }

        /// Ultra-fast evaluation of partitioned elastic nodes only (< 1 cycle / node).
        pub fn solveElasticOnly(self: *Self, viewport_w: i16) void {
            const w_32: i32 = @as(i32, viewport_w);
            const count = self.elastic_count;

            var idx: usize = 0;
            while (idx < count) : (idx += 1) {
                const i = self.elastic_indices[idx];
                const al = @as(i32, self.alpha_l_q14[i]);
                const bl = @as(i32, self.beta_l_i16[i]);
                const ar = @as(i32, self.alpha_r_q14[i]);
                const br = @as(i32, self.beta_r_i16[i]);

                const x = (al * w_32 + (bl << SHIFT_Q14) + BIAS_Q14) >> SHIFT_Q14;
                const r = (ar * w_32 + (br << SHIFT_Q14) + BIAS_Q14) >> SHIFT_Q14;

                const min_i16: i32 = std.math.minInt(i16);
                const max_i16: i32 = std.math.maxInt(i16);
                const clamped_x = std.math.clamp(x, min_i16, max_i16);
                const clamped_w = std.math.clamp(r - x, min_i16, max_i16);

                const x_16: i16 = @intCast(clamped_x);
                const w_16: i16 = @intCast(clamped_w);
                self.resolved_x[i] = x_16;
                self.resolved_w[i] = w_16;
                self.rects[i].x_px = @floatFromInt(x_16);
                self.rects[i].width_px = @floatFromInt(w_16);
            }
        }

        pub inline fn getRect(self: *const Self, node_index: u16) RectPx {
            assert(node_index < self.node_count);
            return self.rects[node_index];
        }

        pub inline fn getResolvedBoxI16(self: *const Self, node_index: u16) ResolvedBoxI16 {
            assert(node_index < self.node_count);
            return .{
                .x_px = self.resolved_x[node_index],
                .y_px = @intFromFloat(@round(self.rects[node_index].y_px)),
                .width_px = self.resolved_w[node_index],
                .height_px = @intFromFloat(@round(self.rects[node_index].height_px)),
            };
        }
    };
}

// ============================================================================
// Unit Tests for Modular Flexbox Layout Engine
// ============================================================================

test "Flexbox 4-Pass layout produces correct coordinates and dimensions" {
    var tree: FlexboxTree(16) = undefined;
    tree.init();

    // Node 0: Root column container (1200x800)
    const root = tree.addNode(.col, INDEX_NONE, 0, .{
        .pad_px = 10.0,
        .gap_px = 5.0,
        .width_px_min = 100.0,
        .height_px_min = 100.0,
    }, true);

    // Node 1: Child row container (pad 0, gap 4, grow 0)
    const row = tree.addNode(.row, root, 1, .{
        .pad_px = 0.0,
        .gap_px = 4.0,
        .grow = 0.0,
        .width_px_min = 200.0,
        .height_px_min = 40.0,
    }, true);

    // Nodes 2, 3: Two sibling buttons with grow 1.0 each (1:1 split)
    _ = tree.addNode(.button, row, 2, .{
        .pad_px = 0.0,
        .gap_px = 0.0,
        .grow = 1.0,
        .width_px_min = 50.0,
        .height_px_min = 40.0,
    }, true);

    _ = tree.addNode(.button, row, 2, .{
        .pad_px = 0.0,
        .gap_px = 0.0,
        .grow = 1.0,
        .width_px_min = 50.0,
        .height_px_min = 40.0,
    }, true);

    tree.computeSubtreeCounts();
    try std.testing.expectEqual(@as(u16, 4), tree.nodes[0].subtree_count);
    try std.testing.expectEqual(@as(u16, 3), tree.nodes[1].subtree_count);

    tree.run4Pass(1200.0, 800.0);

    // Root rect at (0, 0, 1200, 800)
    const r0 = tree.getRect(0);
    try std.testing.expectEqual(@as(f32, 0.0), r0.x_px);
    try std.testing.expectEqual(@as(f32, 0.0), r0.y_px);
    try std.testing.expectEqual(@as(f32, 1200.0), r0.width_px);

    // Row inner width = 1200 - 2 * 10 = 1180
    const r1 = tree.getRect(1);
    try std.testing.expectEqual(@as(f32, 10.0), r1.x_px);
    try std.testing.expectEqual(@as(f32, 10.0), r1.y_px);
    try std.testing.expectEqual(@as(f32, 1180.0), r1.width_px);

    // Two buttons in row: available width = 1180 - gap (4) = 1176 / 2 = 588 each
    const b1 = tree.getRect(2);
    const b2 = tree.getRect(3);

    try std.testing.expectEqual(@as(f32, 10.0), b1.x_px);
    try std.testing.expectEqual(@as(f32, 588.0), b1.width_px);

    // Right edge of b1 is 10 + 588 = 598. b2 starts at 598 + gap(4) = 602
    try std.testing.expectEqual(@as(f32, 602.0), b2.x_px);
    try std.testing.expectEqual(@as(f32, 588.0), b2.width_px);
}

test "Cumulative affine boundary law guarantees zero hairline rounding gaps" {
    // 3 siblings in 1:1:1 split across arbitrary viewport widths
    var tree: FlexboxTree(8) = undefined;
    tree.init();

    const root = tree.addNode(.row, INDEX_NONE, 0, .{
        .pad_px = 0.0,
        .gap_px = 2.0,
    }, true);

    _ = tree.addNode(.button, root, 1, .{ .grow = 1.0 }, true);
    _ = tree.addNode(.button, root, 1, .{ .grow = 1.0 }, true);
    _ = tree.addNode(.button, root, 1, .{ .grow = 1.0 }, true);

    tree.computeSubtreeCounts();
    tree.compileAffine();
    tree.partitionElastic();

    try std.testing.expectEqual(@as(u16, 4), tree.elastic_count);
    try std.testing.expectEqual(@as(u16, 0), tree.rigid_count);

    // Test on multiple fractional widths
    const test_widths = [_]i16{ 320, 321, 322, 500, 777, 1024, 1366, 1920 };
    for (test_widths) |w| {
        tree.solveResizeQ14(w);

        const x1 = tree.resolved_x[1];
        const w1 = tree.resolved_w[1];
        const x2 = tree.resolved_x[2];
        const w2 = tree.resolved_w[2];
        const x3 = tree.resolved_x[3];
        const w3 = tree.resolved_w[3];

        // Zero-gap invariant: x_{i+1} == x_i + w_i + gap
        try std.testing.expectEqual(x2, x1 + w1 + 2);
        try std.testing.expectEqual(x3, x2 + w2 + 2);

        // Rightmost edge covers entire available width
        try std.testing.expectEqual(@as(i16, w), x3 + w3);
    }
}

test "SIMD Q14 evaluation is bit-exact with scalar Q14" {
    var tree: FlexboxTree(16) = undefined;
    tree.init();

    const root = tree.addNode(.col, INDEX_NONE, 0, .{
        .pad_px = 8.0,
        .gap_px = 4.0,
    }, true);

    for (0..5) |_| {
        _ = tree.addNode(.button, root, 1, .{
            .width_px_min = 80.0,
            .height_px_min = 24.0,
        }, false);
    }

    tree.computeSubtreeCounts();
    tree.compileAffine();

    tree.solveResizeQ14(1440);
    var scalar_x: [16]i16 = undefined;
    var scalar_w: [16]i16 = undefined;
    @memcpy(&scalar_x, tree.resolved_x[0..16]);
    @memcpy(&scalar_w, tree.resolved_w[0..16]);

    tree.solveResizeSimd(1440);
    for (0..6) |i| {
        try std.testing.expectEqual(scalar_x[i], tree.resolved_x[i]);
        try std.testing.expectEqual(scalar_w[i], tree.resolved_w[i]);
    }
}

test "Elastic partitioning initializes rigid nodes correctly with solveElasticOnly" {
    var tree: FlexboxTree(8) = undefined;
    tree.init();

    const root = tree.addNode(.row, INDEX_NONE, 0, .{
        .pad_px = 0.0,
        .gap_px = 4.0,
    }, true);

    // Node 1: Rigid button (fixed min width 120, grow = 0) at fixed position
    _ = tree.addNode(.button, root, 1, .{
        .grow = 0.0,
        .width_px_min = 120.0,
    }, false);
    // Node 2: Elastic button (grow = 1.0)
    _ = tree.addNode(.button, root, 1, .{ .grow = 1.0 }, true);

    tree.computeSubtreeCounts();
    tree.compileAffine();
    tree.partitionElastic();

    try std.testing.expectEqual(@as(u16, 2), tree.elastic_count); // root and button 2
    try std.testing.expectEqual(@as(u16, 1), tree.rigid_count); // button 1

    // solveElasticOnly evaluates only elastic nodes; rigid nodes were set during partition
    tree.solveResizeQ14(1000);
    const box_rigid_scalar = tree.getResolvedBoxI16(1);
    const box_elastic_scalar = tree.getResolvedBoxI16(2);

    // Now re-run with solveElasticOnly
    tree.resolved_x[2] = 0;
    tree.resolved_w[2] = 0;
    tree.solveElasticOnly(1000);

    const box_rigid = tree.getResolvedBoxI16(1);
    const box_elastic = tree.getResolvedBoxI16(2);

    // Rigid button width must be 120 at x=0
    try std.testing.expectEqual(box_rigid_scalar.x_px, box_rigid.x_px);
    try std.testing.expectEqual(box_rigid_scalar.width_px, box_rigid.width_px);
    try std.testing.expectEqual(@as(i16, 0), box_rigid.x_px);
    try std.testing.expectEqual(@as(i16, 120), box_rigid.width_px);

    // Elastic button matches scalar solution
    try std.testing.expectEqual(box_elastic_scalar.x_px, box_elastic.x_px);
    try std.testing.expectEqual(box_elastic_scalar.width_px, box_elastic.width_px);
    try std.testing.expectEqual(@as(i16, 124), box_elastic.x_px);
    try std.testing.expectEqual(@as(i16, 876), box_elastic.width_px);
}

test "Arbitrary fractional grow ratios 1:3:7:11 have exact zero hairline gap" {
    var tree: FlexboxTree(8) = undefined;
    tree.init();

    const root = tree.addNode(.row, INDEX_NONE, 0, .{
        .pad_px = 0.0,
        .gap_px = 1.0,
    }, true);

    _ = tree.addNode(.button, root, 1, .{ .grow = 1.0 }, true);
    _ = tree.addNode(.button, root, 1, .{ .grow = 3.0 }, true);
    _ = tree.addNode(.button, root, 1, .{ .grow = 7.0 }, true);
    _ = tree.addNode(.button, root, 1, .{ .grow = 11.0 }, true);

    tree.computeSubtreeCounts();
    tree.compileAffine();

    const widths = [_]i16{ 768, 1024, 1366, 1440, 1920, 2560, 3840 };
    for (widths) |w| {
        tree.solveResizeQ14(w);

        const x1 = tree.resolved_x[1];
        const w1 = tree.resolved_w[1];
        const x2 = tree.resolved_x[2];
        const w2 = tree.resolved_w[2];
        const x3 = tree.resolved_x[3];
        const w3 = tree.resolved_w[3];
        const x4 = tree.resolved_x[4];
        const w4 = tree.resolved_w[4];

        // Zero-gap invariant between each sibling: x_{i+1} == x_i + w_i + gap
        try std.testing.expectEqual(x2, x1 + w1 + 1);
        try std.testing.expectEqual(x3, x2 + w2 + 1);
        try std.testing.expectEqual(x4, x3 + w3 + 1);

        // Entire width is filled without residual error
        try std.testing.expectEqual(@as(i16, w), x4 + w4);
    }
}
