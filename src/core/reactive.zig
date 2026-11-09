//! ============================================================================
//! Klaxon Hardware Bitmask CTZ Reactive Dispatcher (`src/core/reactive.zig`)
//! ============================================================================
//! Deterministic, zero-allocation reactive dispatch engine via CPU bitmask instructions.
//!
//! Features:
//! 1. Hardware Bitmask64: Checks and pops up to 64 dirty branches in 1 CPU cycle
//!    using trailing zero count (`@ctz` -> x86: TZCNT, ARM64: RBIT+CLZ) and reset lowest
//!    set bit (`BLSR` -> `x & (x - 1)`).
//! 2. Hardware Bitmask256: Extended 4x64-bit bitmask for complex forms and dashboards.
//! 3. Static CSR Dependency Graph: Compressed Sparse Row dependency table stored in `.rodata`.
//! 4. Safe State Diffing (`diffState`): Field-by-field comparison immune to Zig struct padding.
//! 5. Zero dynamic allocation: 100% static BSS / stack storage, 0 malloc/free.
//! 6. Strict TigerStyle: <= 100 columns, zero libc, zero POSIX.

const std = @import("std");
const assert = std.debug.assert;

// ============================================================================
// 1. Hardware Bitmask Primitives (Bitmask64 and Bitmask256)
// ============================================================================

/// Single 64-bit integer bitmask checking up to 64 dirty branches in 1 cycle via `@ctz`.
pub const Bitmask64 = struct {
    mask: u64 = 0,

    pub inline fn empty() Bitmask64 {
        return .{ .mask = 0 };
    }

    pub inline fn all() Bitmask64 {
        return .{ .mask = ~@as(u64, 0) };
    }

    pub inline fn set(self: *Bitmask64, bit_idx: u6) void {
        self.mask |= (@as(u64, 1) << bit_idx);
    }

    pub inline fn clearBit(self: *Bitmask64, bit_idx: u6) void {
        self.mask &= ~(@as(u64, 1) << bit_idx);
    }

    pub inline fn testBit(self: Bitmask64, bit_idx: u6) bool {
        return (self.mask & (@as(u64, 1) << bit_idx)) != 0;
    }

    pub inline fn isZero(self: Bitmask64) bool {
        return self.mask == 0;
    }

    pub inline fn clear(self: *Bitmask64) void {
        self.mask = 0;
    }

    pub inline fn setAll(self: *Bitmask64) void {
        self.mask = ~@as(u64, 0);
    }

    /// Pops the lowest active bit in 1 CPU cycle via @ctz and clears it via BLSR.
    pub inline fn popLowest(self: *Bitmask64) u6 {
        assert(self.mask != 0);
        const idx: u6 = @intCast(@ctz(self.mask));
        self.mask &= self.mask - 1;
        return idx;
    }

    pub inline fn countSet(self: Bitmask64) usize {
        return @popCount(self.mask);
    }

    pub inline fn unionWith(self: Bitmask64, other: Bitmask64) Bitmask64 {
        return .{ .mask = self.mask | other.mask };
    }

    pub inline fn intersectWith(self: Bitmask64, other: Bitmask64) Bitmask64 {
        return .{ .mask = self.mask & other.mask };
    }
};

/// 256-bit bitmask primitive for large screens, forms, and complex component trees.
pub const Bitmask256 = struct {
    words: [4]u64 = .{ 0, 0, 0, 0 },

    pub inline fn empty() Bitmask256 {
        return .{ .words = .{ 0, 0, 0, 0 } };
    }

    pub inline fn all() Bitmask256 {
        return .{ .words = .{ ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0) } };
    }

    pub inline fn set(self: *Bitmask256, bit_idx: u8) void {
        const word_idx = bit_idx >> 6;
        const local_bit: u6 = @intCast(bit_idx & 0x3F);
        self.words[word_idx] |= (@as(u64, 1) << local_bit);
    }

    pub inline fn clearBit(self: *Bitmask256, bit_idx: u8) void {
        const word_idx = bit_idx >> 6;
        const local_bit: u6 = @intCast(bit_idx & 0x3F);
        self.words[word_idx] &= ~(@as(u64, 1) << local_bit);
    }

    pub inline fn testBit(self: Bitmask256, bit_idx: u8) bool {
        const word_idx = bit_idx >> 6;
        const local_bit: u6 = @intCast(bit_idx & 0x3F);
        return (self.words[word_idx] & (@as(u64, 1) << local_bit)) != 0;
    }

    pub inline fn isZero(self: Bitmask256) bool {
        return (self.words[0] | self.words[1] | self.words[2] | self.words[3]) == 0;
    }

    pub inline fn clear(self: *Bitmask256) void {
        self.words = .{ 0, 0, 0, 0 };
    }

    pub inline fn setAll(self: *Bitmask256) void {
        self.words = .{ ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0) };
    }

    pub inline fn unionWith(self: Bitmask256, other: Bitmask256) Bitmask256 {
        return .{
            .words = .{
                self.words[0] | other.words[0],
                self.words[1] | other.words[1],
                self.words[2] | other.words[2],
                self.words[3] | other.words[3],
            },
        };
    }

    pub inline fn intersectWith(self: Bitmask256, other: Bitmask256) Bitmask256 {
        return .{
            .words = .{
                self.words[0] & other.words[0],
                self.words[1] & other.words[1],
                self.words[2] & other.words[2],
                self.words[3] & other.words[3],
            },
        };
    }

    /// Pops the lowest active bit across words. Asserts that the bitmask is non-zero.
    pub inline fn popLowest(self: *Bitmask256) u8 {
        assert(!self.isZero());
        inline for (0..4) |w_idx| {
            if (self.words[w_idx] != 0) {
                const local_bit: u6 = @intCast(@ctz(self.words[w_idx]));
                self.words[w_idx] &= self.words[w_idx] - 1;
                return @intCast(w_idx * 64 + local_bit);
            }
        }
        unreachable;
    }

    pub inline fn countSet(self: Bitmask256) usize {
        return @as(usize, @popCount(self.words[0])) +
            @as(usize, @popCount(self.words[1])) +
            @as(usize, @popCount(self.words[2])) +
            @as(usize, @popCount(self.words[3]));
    }
};

// ============================================================================
// 2. Static CSR Dependency Graph (Compressed Sparse Row in .rodata)
// ============================================================================

pub const TargetKind = enum(u8) {
    patch_color = 0,
    patch_text = 1,
    invalidate_layout = 2,
    trigger_effect = 3,
    custom = 4,
};

pub const DependencyEdge = struct {
    target_node_id: u16,
    kind: TargetKind,
};

/// Type generator for compile-time Compressed Sparse Row (CSR) dependency graph.
pub fn CsrDependencyGraph(comptime field_count: usize, comptime total_edges: usize) type {
    return struct {
        const Self = @This();
        pub const fields = field_count;
        pub const edges_total = total_edges;

        offsets: [field_count + 1]u16,
        edges: [total_edges]DependencyEdge,

        pub inline fn getEdges(self: Self, field_idx: usize) []const DependencyEdge {
            if (field_idx >= field_count) return &.{};
            const start = self.offsets[field_idx];
            const end = self.offsets[field_idx + 1];
            assert(start <= end);
            assert(end <= total_edges);
            return self.edges[start..end];
        }
    };
}

// ============================================================================
// 3. State Diffing (Immune to Undefined Struct Padding)
// ============================================================================

/// Compares `old_state` and `new_state` field-by-field to identify modified fields.
/// Avoids raw memory comparison (`memcmp`), preventing false positives from undefined
/// padding bytes inserted by the compiler.
pub fn diffState(
    comptime State: type,
    old_state: *const State,
    new_state: *const State,
    dirty_mask: *Bitmask64,
) void {
    const field_names = comptime @typeInfo(State).@"struct".field_names;
    comptime assert(field_names.len <= 64);

    inline for (field_names, 0..) |fname, i| {
        const old_val = @field(old_state, fname);
        const new_val = @field(new_state, fname);
        if (!std.meta.eql(old_val, new_val)) {
            dirty_mask.set(@intCast(i));
        }
    }
}

// ============================================================================
// 4. Reactive Dispatcher
// ============================================================================

/// Zero-allocation reactive event dispatcher.
pub const ReactiveDispatcher = struct {
    dirty: Bitmask64 = Bitmask64.empty(),
    dispatched_events: usize = 0,
    dispatched_edges: usize = 0,

    /// Marks a specific state field as dirty.
    pub inline fn markField(self: *ReactiveDispatcher, field_idx: u6) void {
        self.dirty.set(field_idx);
    }

    /// Marks all 64 fields as dirty.
    pub inline fn markAll(self: *ReactiveDispatcher) void {
        self.dirty.setAll();
    }

    /// Checks if any field is dirty.
    pub inline fn isDirty(self: ReactiveDispatcher) bool {
        return !self.dirty.isZero();
    }

    /// Clears all dirty flags.
    pub inline fn clear(self: *ReactiveDispatcher) void {
        self.dirty.clear();
    }

    /// Dispatches all pending dirty flags against a CSR graph and event handler.
    /// Returns the number of dirty mutations processed.
    pub fn dispatch(
        self: *ReactiveDispatcher,
        graph: anytype,
        handler: anytype,
    ) usize {
        var mutations_processed: usize = 0;
        const iteration_limit_max: usize = 64;
        var iteration: usize = 0;

        while (!self.dirty.isZero() and iteration < iteration_limit_max) : (iteration += 1) {
            const field = self.dirty.popLowest();
            const edges = graph.getEdges(field);

            for (edges) |edge| {
                handler.onEdgeTriggered(field, edge);
                self.dispatched_edges += 1;
            }

            mutations_processed += 1;
            self.dispatched_events += 1;
        }

        return mutations_processed;
    }
};

// ============================================================================
// Unit Tests for Reactive Dispatcher
// ============================================================================

test "Bitmask64 operations and popLowest ordering" {
    var b = Bitmask64.empty();
    try std.testing.expect(b.isZero());
    try std.testing.expectEqual(@as(usize, 0), b.countSet());

    b.set(0);
    b.set(15);
    b.set(63);

    try std.testing.expect(!b.isZero());
    try std.testing.expectEqual(@as(usize, 3), b.countSet());
    try std.testing.expect(b.testBit(0));
    try std.testing.expect(b.testBit(15));
    try std.testing.expect(b.testBit(63));
    try std.testing.expect(!b.testBit(1));

    // Pop in strictly ascending order
    try std.testing.expectEqual(@as(u6, 0), b.popLowest());
    try std.testing.expectEqual(@as(u6, 15), b.popLowest());
    try std.testing.expectEqual(@as(u6, 63), b.popLowest());
    try std.testing.expect(b.isZero());
}

test "Bitmask64 union and intersect" {
    var b1 = Bitmask64.empty();
    var b2 = Bitmask64.empty();
    b1.set(3);
    b1.set(7);
    b2.set(7);
    b2.set(10);

    const u = b1.unionWith(b2);
    try std.testing.expectEqual(@as(usize, 3), u.countSet());
    try std.testing.expect(u.testBit(3));
    try std.testing.expect(u.testBit(7));
    try std.testing.expect(u.testBit(10));

    const inter = b1.intersectWith(b2);
    try std.testing.expectEqual(@as(usize, 1), inter.countSet());
    try std.testing.expect(inter.testBit(7));
}

test "Bitmask256 cross-word bit manipulation" {
    var b = Bitmask256.empty();
    try std.testing.expect(b.isZero());

    const bits = [_]u8{ 0, 63, 64, 127, 128, 191, 192, 255 };
    for (bits) |bit| {
        b.set(bit);
    }

    try std.testing.expectEqual(@as(usize, 8), b.countSet());
    for (bits) |bit| {
        try std.testing.expect(b.testBit(bit));
    }
    try std.testing.expect(!b.testBit(1));
    try std.testing.expect(!b.testBit(65));

    for (bits) |expected| {
        try std.testing.expectEqual(expected, b.popLowest());
    }
    try std.testing.expect(b.isZero());
}

test "diffState ignores padding and captures changed fields" {
    const TestState = extern struct {
        counter: u32 = 10,
        active: bool = false,
        _pad: [3]u8 = .{ 0, 0, 0 },
        scale: f32 = 1.0,
    };

    var s0 = TestState{};
    var s1 = TestState{};
    var dirty = Bitmask64.empty();

    diffState(TestState, &s0, &s1, &dirty);
    try std.testing.expect(dirty.isZero());

    s1.scale = 2.5;
    diffState(TestState, &s0, &s1, &dirty);
    try std.testing.expect(!dirty.isZero());
    try std.testing.expectEqual(@as(usize, 1), dirty.countSet());
    try std.testing.expectEqual(@as(u6, 3), dirty.popLowest());
}

test "ReactiveDispatcher with CSR dependency graph" {
    const field_count = 3;
    const total_edges = 4;
    const Graph = CsrDependencyGraph(field_count, total_edges);

    const graph = Graph{
        .offsets = .{ 0, 2, 3, 4 },
        .edges = .{
            .{ .target_node_id = 10, .kind = .invalidate_layout },
            .{ .target_node_id = 11, .kind = .patch_color },
            .{ .target_node_id = 20, .kind = .patch_text },
            .{ .target_node_id = 30, .kind = .trigger_effect },
        },
    };

    const MockHandler = struct {
        events: [8]u16 = undefined,
        count: usize = 0,

        pub fn onEdgeTriggered(self: *@This(), _: usize, edge: DependencyEdge) void {
            self.events[self.count] = edge.target_node_id;
            self.count += 1;
        }
    };

    var handler = MockHandler{};
    var dispatcher = ReactiveDispatcher{};

    // Mark field 0 (has 2 edges: 10, 11) and field 2 (has 1 edge: 30)
    dispatcher.markField(0);
    dispatcher.markField(2);

    try std.testing.expect(dispatcher.isDirty());

    const processed = dispatcher.dispatch(graph, &handler);
    try std.testing.expectEqual(@as(usize, 2), processed);
    try std.testing.expectEqual(@as(usize, 3), handler.count);
    try std.testing.expectEqual(@as(u16, 10), handler.events[0]);
    try std.testing.expectEqual(@as(u16, 11), handler.events[1]);
    try std.testing.expectEqual(@as(u16, 30), handler.events[2]);
    try std.testing.expect(!dispatcher.isDirty());
}

test "Bitmask256 setAll, unionWith, and intersectWith" {
    var b1 = Bitmask256.empty();
    var b2 = Bitmask256.empty();

    b1.set(10);
    b1.set(100);
    b2.set(100);
    b2.set(200);

    const u = b1.unionWith(b2);
    try std.testing.expectEqual(@as(usize, 3), u.countSet());
    try std.testing.expect(u.testBit(10));
    try std.testing.expect(u.testBit(100));
    try std.testing.expect(u.testBit(200));

    const inter = b1.intersectWith(b2);
    try std.testing.expectEqual(@as(usize, 1), inter.countSet());
    try std.testing.expect(inter.testBit(100));

    var b_all = Bitmask256.empty();
    b_all.setAll();
    try std.testing.expectEqual(@as(usize, 256), b_all.countSet());
}

test "CsrDependencyGraph safely handles unmonitored fields" {
    const field_count = 2;
    const total_edges = 1;
    const Graph = CsrDependencyGraph(field_count, total_edges);
    const graph = Graph{
        .offsets = .{ 0, 1, 1 },
        .edges = .{.{ .target_node_id = 99, .kind = .custom }},
    };

    // Field 0 has 1 edge
    try std.testing.expectEqual(@as(usize, 1), graph.getEdges(0).len);
    // Field 1 has 0 edges
    try std.testing.expectEqual(@as(usize, 0), graph.getEdges(1).len);
    // Field 5 (unmonitored) safely returns empty slice without panicking
    try std.testing.expectEqual(@as(usize, 0), graph.getEdges(5).len);
}
