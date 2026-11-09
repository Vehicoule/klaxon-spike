//! ============================================================================
//! Klaxon Dynamic Generational Glyph Texture Atlas (`src/paint/atlas.zig`)
//! ============================================================================
//! High-performance static 2D texture atlas (1024x1024 R8) with TigerStyle safety:
//! 1. Zero heap allocation: 1 MB static BSS buffer with 64-byte cache alignment.
//! 2. Deterministic shelf packing algorithm with 1 px inter-glyph boundary padding.
//! 3. Open addressing static hash table (4,096 entries, bounded probe sequence <= 32).
//! 4. Generational LRU eviction with strict on-screen pinning (active frame is immutable).
//! 5. Real-time frame budget throttling with Forward Progress Guarantee:
//!    Guarantees at least 1 glyph rasterization per frame even under virtualization jitter.
//! 6. Double-buffered atlas banks for asynchronous GPU texture upload pipelines.
//!
//! Standards: TigerStyle (<= 100 columns, zero libc, zero POSIX, 0 malloc/free).

const std = @import("std");
const assert = std.debug.assert;
const sdf = @import("sdf.zig");
const time = @import("../platform/time.zig");

pub const BezierQuadratic = sdf.BezierQuadratic;
pub const rasterizeGlyphSdf = sdf.rasterizeGlyphSdf;

// ============================================================================
// Atlas Texture Dimensions & Hash Constraints
// ============================================================================

pub const atlas_dimension_px: u32 = 1024;
pub const atlas_pixel_count: u32 = atlas_dimension_px * atlas_dimension_px;
pub const atlas_padding_px: u32 = 1;
pub const glyph_hash_capacity: u32 = 4096;
pub const probe_limit_max: u32 = 32;

/// Default frame cold rasterization budget: 1.5 ms (1,500,000 ns) for 120 Hz SLA.
pub const time_budget_ns_default: u64 = 1_500_000;

comptime {
    assert((glyph_hash_capacity & (glyph_hash_capacity - 1)) == 0);
    assert(atlas_dimension_px == 1024);
    assert(atlas_pixel_count == 1024 * 1024);
}

// ============================================================================
// Atlas Key & Slot Structures
// ============================================================================

/// Unique identifier for an instance of a glyph in the atlas.
pub const GlyphKey = extern struct {
    glyph_index: u32,
    font_id: u16,
    size_px: u16,
};

comptime {
    assert(@sizeOf(GlyphKey) == 8);
    assert(@alignOf(GlyphKey) == 4);
}

/// Packed location, dimensions, and normalized UV bounds of a glyph in the atlas.
pub const AtlasSlot = extern struct {
    x_px: u16,
    y_px: u16,
    width_px: u16,
    height_px: u16,
    bearing_x_px: i16,
    bearing_y_px: i16,
    advance_x_px: i16,
    _padding: i16 = 0,
    uv_min: [2]f32,
    uv_max: [2]f32,
    last_frame: u32 = 0,
};

comptime {
    assert(@sizeOf(AtlasSlot) == 36);
    assert(@alignOf(AtlasSlot) == 4);
    assert(@offsetOf(AtlasSlot, "x_px") == 0);
    assert(@offsetOf(AtlasSlot, "uv_min") == 16);
    assert(@offsetOf(AtlasSlot, "uv_max") == 24);
    assert(@offsetOf(AtlasSlot, "last_frame") == 32);
}

/// Bounded integer hash function for GlyphKey.
pub fn hashGlyphKey(key: GlyphKey) u32 {
    var hash: u32 = 2166136261;
    hash = (hash ^ key.glyph_index) *% 16777619;
    hash = (hash ^ @as(u32, key.font_id)) *% 16777619;
    hash = (hash ^ @as(u32, key.size_px)) *% 16777619;
    return hash;
}

// ============================================================================
// Generational Glyph Texture Atlas
// ============================================================================

pub const GlyphAtlas = struct {
    pixels: [atlas_pixel_count]u8 align(64),
    slot_occupied: [glyph_hash_capacity]bool,
    slot_keys: [glyph_hash_capacity]GlyphKey,
    slot_data: [glyph_hash_capacity]AtlasSlot,
    shelf_x_px: u32,
    shelf_y_px: u32,
    shelf_height_px: u32,
    allocated_glyph_count: u32,
    current_frame: u32,
    pinned_set_overflow: bool,
    pinned_overflow_count: u32,
    tofu_slot: AtlasSlot,
    frame_start_ns: u64,
    time_budget_ns: u64,
    time_budget_exhausted: bool,
    rasterized_this_frame_count: u32,

    // Dirty Region Tracking for Efficient GPU Texture Updates
    dirty: bool = false,
    dirty_min_x: u16 = 0,
    dirty_min_y: u16 = 0,
    dirty_max_x: u16 = 0,
    dirty_max_y: u16 = 0,

    /// Initializes the glyph texture atlas in-place.
    pub fn init(target: *GlyphAtlas) void {
        @memset(&target.pixels, 0);
        @memset(&target.slot_occupied, false);
        target.allocated_glyph_count = 0;
        target.current_frame = 1;
        target.pinned_set_overflow = false;
        target.pinned_overflow_count = 0;
        target.frame_start_ns = time.now().ns;
        target.time_budget_ns = time_budget_ns_default;
        target.time_budget_exhausted = false;
        target.rasterized_this_frame_count = 0;
        target.dirty = false;
        target.dirty_min_x = 0;
        target.dirty_min_y = 0;
        target.dirty_max_x = 0;
        target.dirty_max_y = 0;

        // Reserve a deterministic fallback Tofu glyph (16x24 hollow rectangle) at (1, 1)
        const tofu_w: u32 = 16;
        const tofu_h: u32 = 24;
        var y: u32 = 0;
        while (y < tofu_h) : (y += 1) {
            var x: u32 = 0;
            while (x < tofu_w) : (x += 1) {
                const is_border = (x == 0 or x == tofu_w - 1 or y == 0 or y == tofu_h - 1);
                const val: u8 = if (is_border) 255 else 0;
                target.pixels[(1 + y) * atlas_dimension_px + (1 + x)] = val;
            }
        }

        const inv_dim: f32 = 1.0 / @as(f32, @floatFromInt(atlas_dimension_px));
        target.tofu_slot = AtlasSlot{
            .x_px = 1,
            .y_px = 1,
            .width_px = @as(u16, @intCast(tofu_w)),
            .height_px = @as(u16, @intCast(tofu_h)),
            .bearing_x_px = 0,
            .bearing_y_px = @as(i16, @intCast(tofu_h)),
            .advance_x_px = @as(i16, @intCast(tofu_w + 2)),
            .uv_min = .{ 1.0 * inv_dim, 1.0 * inv_dim },
            .uv_max = .{
                @as(f32, @floatFromInt(1 + tofu_w)) * inv_dim,
                @as(f32, @floatFromInt(1 + tofu_h)) * inv_dim,
            },
            .last_frame = std.math.maxInt(u32), // Pinned permanently, never evicted
        };

        // Shelf begins right after the Tofu reserve
        target.shelf_x_px = atlas_padding_px + tofu_w + 2 * atlas_padding_px;
        target.shelf_y_px = atlas_padding_px;
        target.shelf_height_px = tofu_h + 2 * atlas_padding_px;
    }

    /// Sets the active frame identifier and begins frame wall-clock timer.
    pub fn beginFrame(self: *GlyphAtlas, frame_id: u32) void {
        self.current_frame = frame_id;
        self.frame_start_ns = time.now().ns;
        self.time_budget_exhausted = false;
        self.rasterized_this_frame_count = 0;
        self.pinned_set_overflow = false;
    }

    /// Increments frame count by one and resets frame timer.
    pub fn advanceFrame(self: *GlyphAtlas) void {
        self.beginFrame(self.current_frame +% 1);
    }

    /// Checks if time budget for cold rasterization is still available this frame.
    pub fn hasTimeBudget(self: *const GlyphAtlas) bool {
        if (self.time_budget_exhausted) return false;
        const now_ns = time.now().ns;
        if (now_ns >= self.frame_start_ns + self.time_budget_ns) {
            return false;
        }
        return true;
    }

    /// Records one cold rasterization event and updates budget status.
    pub fn recordRasterization(self: *GlyphAtlas) void {
        self.rasterized_this_frame_count += 1;
        const now_ns = time.now().ns;
        if (now_ns >= self.frame_start_ns + self.time_budget_ns) {
            self.time_budget_exhausted = true;
        }
    }

    /// Resets dirty update tracking after GPU texture upload.
    pub fn resetDirty(self: *GlyphAtlas) void {
        self.dirty = false;
        self.dirty_min_x = 0;
        self.dirty_min_y = 0;
        self.dirty_max_x = 0;
        self.dirty_max_y = 0;
    }

    /// Generational recycling: evicts off-screen slots where last_frame < current_frame.
    /// On-screen slots (last_frame == current_frame) are strictly pinned and preserved.
    pub fn evictGenerational(self: *GlyphAtlas) u32 {
        var evicted: u32 = 0;
        var i: u32 = 0;
        while (i < glyph_hash_capacity) : (i += 1) {
            if (self.slot_occupied[i]) {
                if (self.slot_data[i].last_frame < self.current_frame) {
                    self.slot_occupied[i] = false;
                    evicted += 1;
                    if (self.allocated_glyph_count > 0) {
                        self.allocated_glyph_count -= 1;
                    }
                }
            }
        }

        if (evicted > 0) {
            const tofu_w: u32 = 16;
            const tofu_h: u32 = 24;
            self.shelf_x_px = atlas_padding_px + tofu_w + 2 * atlas_padding_px;
            self.shelf_y_px = atlas_padding_px;
            self.shelf_height_px = tofu_h + 2 * atlas_padding_px;
        } else {
            // All surveyed slots are pinned on-screen: anti-thrashing rule
            self.pinned_set_overflow = true;
            self.pinned_overflow_count += 1;
        }
        return evicted;
    }

    /// Looks up a glyph in the static open addressing hash table within <= 32 probes.
    pub fn lookupSlot(self: *GlyphAtlas, key: GlyphKey) ?AtlasSlot {
        const initial_index = hashGlyphKey(key) & (glyph_hash_capacity - 1);
        var step: u32 = 0;
        while (step < probe_limit_max) : (step += 1) {
            const slot_index = (initial_index + step) & (glyph_hash_capacity - 1);
            if (self.slot_occupied[slot_index]) {
                const candidate = self.slot_keys[slot_index];
                if (candidate.glyph_index == key.glyph_index) {
                    if (candidate.font_id == key.font_id) {
                        if (candidate.size_px == key.size_px) {
                            self.slot_data[slot_index].last_frame = self.current_frame;
                            return self.slot_data[slot_index];
                        }
                    }
                }
            } else {
                return null;
            }
        }
        return null;
    }

    /// Packs a rasterized glyph patch into the atlas texture using shelf packing.
    pub fn insertGlyph(
        self: *GlyphAtlas,
        key: GlyphKey,
        bitmap: []const u8,
        width_px: u32,
        height_px: u32,
    ) AtlasSlot {
        assert(bitmap.len >= width_px * height_px);
        assert(width_px > 0);
        assert(height_px > 0);
        assert(width_px <= atlas_dimension_px);
        assert(height_px <= atlas_dimension_px);

        if (self.lookupSlot(key)) |existing| {
            return existing;
        }

        const padded_w = width_px + 2 * atlas_padding_px;
        const padded_h = height_px + 2 * atlas_padding_px;

        var can_fit = false;
        if (self.shelf_x_px + padded_w <= atlas_dimension_px) {
            if (self.shelf_y_px + padded_h <= atlas_dimension_px) {
                can_fit = true;
            }
        } else {
            const next_y = self.shelf_y_px + self.shelf_height_px;
            if (next_y + padded_h <= atlas_dimension_px) {
                self.shelf_x_px = atlas_padding_px;
                self.shelf_y_px = next_y;
                self.shelf_height_px = 0;
                can_fit = true;
            }
        }

        if (!can_fit) {
            const evicted = self.evictGenerational();
            if (evicted == 0) {
                return self.tofu_slot;
            }
            if (self.shelf_x_px + padded_w <= atlas_dimension_px and
                self.shelf_y_px + padded_h <= atlas_dimension_px)
            {
                can_fit = true;
            } else {
                return self.tofu_slot;
            }
        }

        const placed_x = self.shelf_x_px + atlas_padding_px;
        const placed_y = self.shelf_y_px + atlas_padding_px;

        self.shelf_x_px += padded_w;
        self.shelf_height_px = @max(self.shelf_height_px, padded_h);

        // Copy bitmap rows into atlas texture buffer
        var y: u32 = 0;
        while (y < height_px) : (y += 1) {
            const src_row_start = y * width_px;
            const dst_row_start = (placed_y + y) * atlas_dimension_px + placed_x;
            const src_slice = bitmap[src_row_start .. src_row_start + width_px];
            const dst_slice = self.pixels[dst_row_start .. dst_row_start + width_px];
            @memcpy(dst_slice, src_slice);
        }

        // Update dirty bounding rectangle
        const x0: u16 = @intCast(placed_x);
        const y0: u16 = @intCast(placed_y);
        const x1: u16 = @intCast(placed_x + width_px);
        const y1: u16 = @intCast(placed_y + height_px);

        if (!self.dirty) {
            self.dirty = true;
            self.dirty_min_x = x0;
            self.dirty_min_y = y0;
            self.dirty_max_x = x1;
            self.dirty_max_y = y1;
        } else {
            self.dirty_min_x = @min(self.dirty_min_x, x0);
            self.dirty_min_y = @min(self.dirty_min_y, y0);
            self.dirty_max_x = @max(self.dirty_max_x, x1);
            self.dirty_max_y = @max(self.dirty_max_y, y1);
        }

        const inv_dim: f32 = 1.0 / @as(f32, @floatFromInt(atlas_dimension_px));
        const uv_min = [2]f32{
            @as(f32, @floatFromInt(placed_x)) * inv_dim,
            @as(f32, @floatFromInt(placed_y)) * inv_dim,
        };
        const uv_max = [2]f32{
            @as(f32, @floatFromInt(placed_x + width_px)) * inv_dim,
            @as(f32, @floatFromInt(placed_y + height_px)) * inv_dim,
        };

        const slot = AtlasSlot{
            .x_px = @as(u16, @intCast(placed_x)),
            .y_px = @as(u16, @intCast(placed_y)),
            .width_px = @as(u16, @intCast(width_px)),
            .height_px = @as(u16, @intCast(height_px)),
            .bearing_x_px = 0,
            .bearing_y_px = @as(i16, @intCast(height_px)),
            .advance_x_px = @as(i16, @intCast(width_px + 2)),
            .uv_min = uv_min,
            .uv_max = uv_max,
            .last_frame = self.current_frame,
        };

        // Insert into open addressing hash table with bounded probing
        const initial_index = hashGlyphKey(key) & (glyph_hash_capacity - 1);
        var step: u32 = 0;
        var target_slot_index: ?u32 = null;
        var oldest_slot_index: u32 = initial_index;
        var oldest_frame: u32 = std.math.maxInt(u32);

        while (step < probe_limit_max) : (step += 1) {
            const slot_index = (initial_index + step) & (glyph_hash_capacity - 1);
            if (!self.slot_occupied[slot_index]) {
                target_slot_index = slot_index;
                break;
            } else {
                const slot_frame = self.slot_data[slot_index].last_frame;
                if (slot_frame < oldest_frame) {
                    oldest_frame = slot_frame;
                    oldest_slot_index = slot_index;
                }
            }
        }

        if (target_slot_index == null and oldest_frame == self.current_frame) {
            self.pinned_set_overflow = true;
            self.pinned_overflow_count += 1;
            return self.tofu_slot;
        }

        const final_index = target_slot_index orelse oldest_slot_index;
        if (!self.slot_occupied[final_index]) {
            self.allocated_glyph_count += 1;
        }
        self.slot_occupied[final_index] = true;
        self.slot_keys[final_index] = key;
        self.slot_data[final_index] = slot;

        return slot;
    }
};

// ============================================================================
// Batch Rasterization with Forward Progress Guarantee
// ============================================================================

/// Request for rasterizing a single glyph with vector curves and spatial bounds.
pub const GlyphRasterRequest = struct {
    key: GlyphKey,
    curves: []const BezierQuadratic,
    width_px: u32,
    height_px: u32,
    spread_px: f32 = 4.0,
};

/// Processes a batch of glyph requests under real-time frame budget governance.
/// Forward Progress Guarantee: Guarantees at least 1 unit of forward progress per frame
/// (rasterized_count == 0), preventing starvation under timing jitter or zero-budget stalls.
pub fn processRasterBatchWithTimeBudget(
    atlas: *GlyphAtlas,
    requests: []const GlyphRasterRequest,
    slots_out: []AtlasSlot,
) u32 {
    assert(slots_out.len >= requests.len);
    var rasterized_count: u32 = 0;

    for (requests, 0..) |req, idx| {
        if (atlas.lookupSlot(req.key)) |cached| {
            slots_out[idx] = cached;
            continue;
        }

        // TigerStyle Forward Progress Guarantee:
        // Always allows at least one glyph per frame even if budget is tight or jittered.
        if (rasterized_count == 0 or atlas.hasTimeBudget()) {
            var patch_buf: [32 * 32]u8 = undefined;
            const w: u32 = @min(32, req.width_px);
            const h: u32 = @min(32, req.height_px);
            const patch_len = @as(usize, w) * @as(usize, h);
            rasterizeGlyphSdf(req.curves, w, h, req.spread_px, patch_buf[0..patch_len]);
            const new_slot = atlas.insertGlyph(req.key, patch_buf[0..patch_len], w, h);
            atlas.recordRasterization();
            slots_out[idx] = new_slot;
            rasterized_count += 1;
        } else {
            atlas.time_budget_exhausted = true;
            slots_out[idx] = atlas.tofu_slot;
        }
    }

    return rasterized_count;
}

// ============================================================================
// Double-Buffered Atlas Engine
// ============================================================================

/// Double-buffered generational texture atlas wrapper for concurrent GPU/CPU pipelines.
pub const DoubleBufferedAtlas = struct {
    banks: [2]GlyphAtlas align(64),
    active_index: u1 = 0,

    /// Initializes both atlas banks in-place.
    pub fn init(target: *DoubleBufferedAtlas) void {
        GlyphAtlas.init(&target.banks[0]);
        GlyphAtlas.init(&target.banks[1]);
        target.active_index = 0;
    }

    /// Returns a pointer to the active front atlas bank.
    pub inline fn getActive(self: *DoubleBufferedAtlas) *GlyphAtlas {
        return &self.banks[self.active_index];
    }

    /// Returns a pointer to the inactive back atlas bank.
    pub inline fn getBack(self: *DoubleBufferedAtlas) *GlyphAtlas {
        return &self.banks[1 - self.active_index];
    }

    /// Swaps front and back atlas banks.
    pub fn swap(self: *DoubleBufferedAtlas) void {
        self.active_index = 1 - self.active_index;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "GlyphAtlas initialization and tofu slot" {
    var atlas: GlyphAtlas = undefined;
    GlyphAtlas.init(&atlas);

    try std.testing.expectEqual(@as(u32, 0), atlas.allocated_glyph_count);
    try std.testing.expectEqual(@as(u32, 1), atlas.current_frame);
    try std.testing.expect(!atlas.pinned_set_overflow);
    try std.testing.expect(!atlas.time_budget_exhausted);

    // Tofu slot must be permanently pinned
    try std.testing.expectEqual(std.math.maxInt(u32), atlas.tofu_slot.last_frame);
    try std.testing.expectEqual(@as(u16, 16), atlas.tofu_slot.width_px);
    try std.testing.expectEqual(@as(u16, 24), atlas.tofu_slot.height_px);
}

test "GlyphAtlas shelf packing, lookup, and dirty tracking" {
    var atlas: GlyphAtlas = undefined;
    GlyphAtlas.init(&atlas);

    var glyph_buf: [16 * 16]u8 = @splat(255);
    const key = GlyphKey{ .glyph_index = 65, .font_id = 1, .size_px = 16 };

    const slot = atlas.insertGlyph(key, &glyph_buf, 16, 16);
    try std.testing.expect(atlas.dirty);
    try std.testing.expectEqual(@as(u32, 1), atlas.allocated_glyph_count);
    try std.testing.expectEqual(@as(u16, 16), slot.width_px);
    try std.testing.expectEqual(@as(u16, 16), slot.height_px);

    // Deterministic lookup
    const found = atlas.lookupSlot(key);
    try std.testing.expect(found != null);
    try std.testing.expectEqual(slot.x_px, found.?.x_px);
    try std.testing.expectEqual(slot.y_px, found.?.y_px);

    atlas.resetDirty();
    try std.testing.expect(!atlas.dirty);
}

test "GlyphAtlas generational eviction preserves pinned on-screen glyphs" {
    var atlas: GlyphAtlas = undefined;
    GlyphAtlas.init(&atlas);

    var glyph_buf: [16 * 16]u8 = @splat(255);
    atlas.beginFrame(1);
    const key1 = GlyphKey{ .glyph_index = 1, .font_id = 1, .size_px = 16 };
    _ = atlas.insertGlyph(key1, &glyph_buf, 16, 16);

    atlas.beginFrame(2);
    const key2 = GlyphKey{ .glyph_index = 2, .font_id = 1, .size_px = 16 };
    _ = atlas.insertGlyph(key2, &glyph_buf, 16, 16);

    // In frame 2, key1 is off-screen (last_frame=1 < current_frame=2), key2 is pinned
    const evicted = atlas.evictGenerational();
    try std.testing.expectEqual(@as(u32, 1), evicted);

    // key1 should be gone, key2 should still exist
    try std.testing.expect(atlas.lookupSlot(key1) == null);
    try std.testing.expect(atlas.lookupSlot(key2) != null);
}

test "GlyphAtlas real-time frame budget throttling with forward progress guarantee" {
    var atlas: GlyphAtlas = undefined;
    GlyphAtlas.init(&atlas);

    // Set microscopic time budget (10 ns) to force throttling
    atlas.time_budget_ns = 10;
    atlas.beginFrame(1);

    var requests: [8]GlyphRasterRequest = undefined;
    for (&requests, 0..) |*req, idx| {
        req.* = .{
            .key = .{ .glyph_index = @intCast(200 + idx), .font_id = 1, .size_px = 32 },
            .curves = &sdf.glyph_o_curves,
            .width_px = 32,
            .height_px = 32,
        };
    }

    var slots: [8]AtlasSlot = undefined;
    const rasterized = processRasterBatchWithTimeBudget(&atlas, &requests, &slots);

    // Forward progress guarantee: rasterized at least 1 glyph despite tiny budget
    try std.testing.expect(rasterized >= 1);
    try std.testing.expect(rasterized < 8);
    try std.testing.expect(atlas.time_budget_exhausted);

    // Substituted remaining glyphs with Tofu
    var tofu_count: u32 = 0;
    for (slots) |slot| {
        if (slot.x_px == atlas.tofu_slot.x_px and slot.y_px == atlas.tofu_slot.y_px) {
            tofu_count += 1;
        }
    }
    try std.testing.expectEqual(8 - rasterized, tofu_count);
}

test "DoubleBufferedAtlas swapping and bank independence" {
    var double_atlas: DoubleBufferedAtlas = undefined;
    DoubleBufferedAtlas.init(&double_atlas);

    const b0 = double_atlas.getActive();
    try std.testing.expectEqual(@as(u1, 0), double_atlas.active_index);

    double_atlas.swap();
    const b1 = double_atlas.getActive();
    try std.testing.expectEqual(@as(u1, 1), double_atlas.active_index);
    try std.testing.expect(b0 != b1);
}
