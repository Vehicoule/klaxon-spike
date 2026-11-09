//! ============================================================================
//! Klaxon Structure of Arrays (SoA) Container (`src/core/soa.zig`)
//! ============================================================================
//! Generic zero-padding `align(64)` Structure of Arrays (SoA) for SIMD vectorization.
//!
//! Features:
//! 1. Memory alignment: Every parallel array is cache-line aligned to 64 bytes (`align(64)`).
//! 2. Zero-padding: Storage capacity is rounded up to a multiple of SIMD lanes (default 32)
//!    and tail elements are guaranteed zeroed (`zeroPad()`), eliminating boundary branches
//!    and out-of-bounds page faults during full-width SIMD loads/stores.
//! 3. Zero dynamic allocation: 100% static BSS storage, 0 malloc/free.
//! 4. Strict TigerStyle: <= 100 columns, zero libc, explicit assertions.

const std = @import("std");
const assert = std.debug.assert;

pub const default_simd_lane_alignment: usize = 32;

/// Generic zero-padding `align(64)` Structure of Arrays container.
pub fn SoA(comptime Item: type, comptime capacity_max: usize) type {
    return SoAWithSimdLen(Item, capacity_max, default_simd_lane_alignment);
}

pub fn SoAWithSimdLen(
    comptime Item: type,
    comptime capacity_max: usize,
    comptime simd_lanes: usize,
) type {
    comptime {
        assert(@typeInfo(Item) == .@"struct");
        assert(capacity_max > 0);
        assert(simd_lanes > 0);
    }

    const field_names = comptime @typeInfo(Item).@"struct".field_names;
    const field_types = comptime @typeInfo(Item).@"struct".field_types;
    const field_count = field_names.len;

    const padded_cap = comptime std.mem.alignForward(usize, capacity_max, simd_lanes);

    // Build the container struct holding parallel arrays
    comptime var array_types: [field_count]type = undefined;
    inline for (0..field_count) |i| {
        array_types[i] = [padded_cap]field_types[i];
    }

    const ArraysStruct = @Struct(
        .auto,
        null,
        field_names,
        &array_types,
        &@splat(.{ .@"align" = 64 }),
    );

    return struct {
        const Self = @This();

        pub const ItemType = Item;
        pub const capacity = capacity_max;
        pub const padded_capacity = padded_cap;
        pub const lane_alignment = simd_lanes;

        arrays: ArraysStruct align(64) = undefined,
        len: usize = 0,

        /// Initializes the container in-place and zeros all padded storage.
        pub fn init(self: *Self) void {
            self.len = 0;
            self.zeroAll();
        }

        /// Zeros all parallel storage across all fields.
        pub fn zeroAll(self: *Self) void {
            inline for (field_names) |name| {
                const slice_ptr = &@field(self.arrays, name);
                @memset(slice_ptr, std.mem.zeroes(field_types[fieldIndex(name)]));
            }
        }

        /// Zeros out the padding elements between `len` and `padded_capacity`.
        pub fn zeroPad(self: *Self) void {
            assert(self.len <= capacity);
            if (self.len >= padded_capacity) return;

            inline for (field_names) |name| {
                const slice_ptr = &@field(self.arrays, name);
                const zero_val = std.mem.zeroes(field_types[fieldIndex(name)]);
                @memset(slice_ptr[self.len..padded_capacity], zero_val);
            }
        }

        /// Returns the number of active items.
        pub inline fn count(self: Self) usize {
            return self.len;
        }

        /// Returns maximum logical capacity.
        pub inline fn capacityMax(self: Self) usize {
            _ = self;
            return capacity;
        }

        /// Returns padded capacity (aligned for SIMD lanes).
        pub inline fn paddedCapacity(self: Self) usize {
            _ = self;
            return padded_capacity;
        }

        /// Appends an Item to the SoA container.
        pub fn append(self: *Self, item: Item) void {
            assert(self.len < capacity);
            const idx = self.len;
            inline for (field_names) |name| {
                @field(self.arrays, name)[idx] = @field(item, name);
            }
            self.len += 1;
        }

        /// Sets an Item at a given index.
        pub fn set(self: *Self, index: usize, item: Item) void {
            assert(index < self.len);
            inline for (field_names) |name| {
                @field(self.arrays, name)[index] = @field(item, name);
            }
        }

        /// Reconstructs an Item at a given index.
        pub fn get(self: *const Self, index: usize) Item {
            assert(index < self.len);
            var result: Item = undefined;
            inline for (field_names) |name| {
                @field(result, name) = @field(self.arrays, name)[index];
            }
            return result;
        }

        /// Returns a slice of the active elements for a given field name.
        pub inline fn slice(
            self: *const Self,
            comptime field_name: []const u8,
        ) []const field_types[fieldIndex(field_name)] {
            assert(self.len <= capacity);
            return @field(self.arrays, field_name)[0..self.len];
        }

        /// Returns a mutable slice of the active elements for a given field name.
        pub inline fn sliceMut(
            self: *Self,
            comptime field_name: []const u8,
        ) []field_types[fieldIndex(field_name)] {
            assert(self.len <= capacity);
            return @field(self.arrays, field_name)[0..self.len];
        }

        /// Returns a slice spanning up to `padded_capacity` for SIMD operations.
        pub inline fn paddedSlice(
            self: *Self,
            comptime field_name: []const u8,
        ) []field_types[fieldIndex(field_name)] {
            return &@field(self.arrays, field_name);
        }

        /// Returns a const slice spanning up to `padded_capacity` for SIMD operations.
        pub inline fn paddedSliceConst(
            self: *const Self,
            comptime field_name: []const u8,
        ) []const field_types[fieldIndex(field_name)] {
            return &@field(self.arrays, field_name);
        }

        /// Loads a SIMD vector from the specified field at offset.
        pub inline fn vector(
            self: *const Self,
            comptime field_name: []const u8,
            comptime lanes: usize,
            offset: usize,
        ) @Vector(lanes, field_types[fieldIndex(field_name)]) {
            assert(offset + lanes <= padded_capacity);
            const array_ptr = &@field(self.arrays, field_name);
            const sub: *const [lanes]field_types[fieldIndex(field_name)] =
                array_ptr[offset..][0..lanes];
            return sub.*;
        }

        /// Stores a SIMD vector to the specified field at offset.
        pub inline fn setVector(
            self: *Self,
            comptime field_name: []const u8,
            comptime lanes: usize,
            offset: usize,
            vec: @Vector(lanes, field_types[fieldIndex(field_name)]),
        ) void {
            assert(offset + lanes <= padded_capacity);
            const array_ptr = &@field(self.arrays, field_name);
            const sub: *[lanes]field_types[fieldIndex(field_name)] = array_ptr[offset..][0..lanes];
            sub.* = vec;
        }

        fn fieldIndex(comptime name: []const u8) usize {
            inline for (field_names, 0..) |fn_name, i| {
                if (std.mem.eql(u8, fn_name, name)) return i;
            }
            @compileError("Field not found: " ++ name);
        }
    };
}

// ============================================================================
// Unit Tests for Structure of Arrays (SoA)
// ============================================================================

const TestParticle = struct {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    id: u32,
};

test "SoA basic append, get, and set" {
    var particles: SoA(TestParticle, 50) = undefined;
    particles.init();

    try std.testing.expectEqual(@as(usize, 0), particles.count());
    // 50 capacity rounded up to multiple of 32 -> 64
    try std.testing.expectEqual(@as(usize, 64), particles.paddedCapacity());

    particles.append(.{ .x = 10.0, .y = 20.0, .vx = 1.0, .vy = 2.0, .id = 100 });
    particles.append(.{ .x = 30.0, .y = 40.0, .vx = 3.0, .vy = 4.0, .id = 101 });

    try std.testing.expectEqual(@as(usize, 2), particles.count());

    const p0 = particles.get(0);
    try std.testing.expectEqual(@as(f32, 10.0), p0.x);
    try std.testing.expectEqual(@as(u32, 100), p0.id);

    particles.set(1, .{ .x = 99.0, .y = 88.0, .vx = 0.0, .vy = 0.0, .id = 999 });
    const p1 = particles.get(1);
    try std.testing.expectEqual(@as(f32, 99.0), p1.x);
    try std.testing.expectEqual(@as(u32, 999), p1.id);
}

test "SoA cache line align(64) verification" {
    var particles: SoA(TestParticle, 16) = undefined;
    particles.init();

    // Verify 64-byte alignment of each parallel array
    const addr_x = @intFromPtr(&particles.arrays.x);
    const addr_y = @intFromPtr(&particles.arrays.y);
    const addr_vx = @intFromPtr(&particles.arrays.vx);
    const addr_vy = @intFromPtr(&particles.arrays.vy);
    const addr_id = @intFromPtr(&particles.arrays.id);

    try std.testing.expectEqual(@as(usize, 0), addr_x % 64);
    try std.testing.expectEqual(@as(usize, 0), addr_y % 64);
    try std.testing.expectEqual(@as(usize, 0), addr_vx % 64);
    try std.testing.expectEqual(@as(usize, 0), addr_vy % 64);
    try std.testing.expectEqual(@as(usize, 0), addr_id % 64);
}

test "SoA zeroPad clears tail elements up to padded_capacity" {
    var particles: SoA(TestParticle, 10) = undefined;
    particles.init();

    // Fill first 2 elements
    particles.append(.{ .x = 1.0, .y = 2.0, .vx = 3.0, .vy = 4.0, .id = 1 });
    particles.append(.{ .x = 5.0, .y = 6.0, .vx = 7.0, .vy = 8.0, .id = 2 });

    // Dirty the padding area manually
    particles.arrays.x[15] = 999.0;
    particles.arrays.id[31] = 777;

    particles.zeroPad();

    try std.testing.expectEqual(@as(f32, 0.0), particles.arrays.x[15]);
    try std.testing.expectEqual(@as(u32, 0), particles.arrays.id[31]);
    try std.testing.expectEqual(@as(f32, 1.0), particles.arrays.x[0]);
    try std.testing.expectEqual(@as(u32, 2), particles.arrays.id[1]);
}

test "SoA SIMD vector load and store" {
    var particles: SoA(TestParticle, 32) = undefined;
    particles.init();

    const V4 = @Vector(4, f32);
    const input_v: V4 = .{ 1.5, 2.5, 3.5, 4.5 };

    particles.setVector("x", 4, 0, input_v);
    const read_v = particles.vector("x", 4, 0);

    try std.testing.expectEqual(input_v[0], read_v[0]);
    try std.testing.expectEqual(input_v[1], read_v[1]);
    try std.testing.expectEqual(input_v[2], read_v[2]);
    try std.testing.expectEqual(input_v[3], read_v[3]);
}

test "SoA slice and sliceMut access" {
    var particles: SoA(TestParticle, 16) = undefined;
    particles.init();

    particles.append(.{ .x = 10.0, .y = 20.0, .vx = 0.0, .vy = 0.0, .id = 1 });
    particles.append(.{ .x = 30.0, .y = 40.0, .vx = 0.0, .vy = 0.0, .id = 2 });

    const x_slice = particles.slice("x");
    try std.testing.expectEqual(@as(usize, 2), x_slice.len);
    try std.testing.expectEqual(@as(f32, 10.0), x_slice[0]);
    try std.testing.expectEqual(@as(f32, 30.0), x_slice[1]);

    const y_mut = particles.sliceMut("y");
    y_mut[1] = 99.0;
    try std.testing.expectEqual(@as(f32, 99.0), particles.get(1).y);

    const pad_slice = particles.paddedSlice("id");
    try std.testing.expectEqual(@as(usize, 32), pad_slice.len);
}
