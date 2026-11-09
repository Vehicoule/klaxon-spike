//! ============================================================================
//! Klaxon Platform Time & Monotonic Benchmarking Layer (`src/platform/time.zig`)
//! ============================================================================
//! High-resolution time primitives, CPU cycle conversions, and zero-allocation
//! benchmark statistics accumulators.
//!
//! Standards: TigerStyle (< 100 columns, zero dynamic allocations, explicit assertions).

const std = @import("std");
const assert = std.debug.assert;
const os = @import("os.zig");

// ============================================================================
// Monotonic Timestamp Primitive
// ============================================================================

/// High-resolution monotonic timestamp in nanoseconds.
pub const Timestamp = struct {
    ns: u64,

    /// Samples current monotonic time from the underlying platform clock.
    pub inline fn now() Timestamp {
        return .{ .ns = os.readMonotonicClockNs() };
    }

    /// Computes the elapsed nanoseconds since an earlier timestamp.
    /// Asserts that this timestamp is chronologically at or after `earlier`.
    pub fn durationSince(self: Timestamp, earlier: Timestamp) u64 {
        assert(self.ns >= earlier.ns);
        if (self.ns < earlier.ns) return 0;
        return self.ns - earlier.ns;
    }

    /// Converts this timestamp duration into CPU cycles at 4.0 GHz reference clock.
    pub inline fn toCycles(self: Timestamp) u64 {
        return toCyclesAt4Ghz(self.ns);
    }
};

// ============================================================================
// Global Conversion Helpers (4.0 GHz Reference Clock)
// ============================================================================

/// Samples the current timestamp from the platform monotonic clock.
pub inline fn now() Timestamp {
    return Timestamp.now();
}

/// Converts a nanosecond count to reference CPU cycles at 4.0 GHz (4 cycles per ns).
pub inline fn toCyclesAt4Ghz(ns: u64) u64 {
    return ns * 4;
}

/// Converts reference CPU cycles at 4.0 GHz back into nanoseconds (integer division).
pub inline fn cyclesToNs(cycles: u64) u64 {
    return @divFloor(cycles, 4);
}

/// Converts reference CPU cycles at 4.0 GHz into high-precision fractional nanoseconds.
pub inline fn cyclesToNsFloat(cycles: u64) f64 {
    return @as(f64, @floatFromInt(cycles)) / 4.0;
}

// ============================================================================
// Zero-Allocation Benchmark Statistics Accumulator
// ============================================================================

/// Statistical summary and streaming accumulator for micro-benchmarking.
/// Guarantees zero heap allocations: samples are accumulated in O(1) or sorted in-place.
pub const BenchmarkStats = struct {
    count: u64 = 0,
    total_ns: u64 = 0,
    min_ns: u64 = 0,
    max_ns: u64 = 0,
    p50_ns: u64 = 0,
    p90_ns: u64 = 0,
    p99_ns: u64 = 0,

    pub const empty: BenchmarkStats = .{};

    /// Records a single duration measurement into the streaming accumulator.
    pub fn record(self: *BenchmarkStats, duration_ns: u64) void {
        if (self.count == 0) {
            self.min_ns = duration_ns;
            self.max_ns = duration_ns;
        } else {
            self.min_ns = @min(self.min_ns, duration_ns);
            self.max_ns = @max(self.max_ns, duration_ns);
        }
        self.count += 1;
        self.total_ns += duration_ns;
    }

    /// Computes the arithmetic mean duration in nanoseconds.
    pub fn meanNs(self: BenchmarkStats) f64 {
        if (self.count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_ns)) / @as(f64, @floatFromInt(self.count));
    }

    /// Computes reference CPU cycles at 4.0 GHz corresponding to the mean duration.
    pub fn cyclesAt4Ghz(self: BenchmarkStats) u64 {
        if (self.count == 0) return 0;
        return toCyclesAt4Ghz(@intFromFloat(self.meanNs()));
    }

    /// Resets all accumulated values to zero.
    pub fn reset(self: *BenchmarkStats) void {
        self.* = .{};
    }

    /// Computes comprehensive benchmark statistics over a caller-provided slice.
    /// Sorts `samples` in-place to calculate percentiles with zero dynamic allocation.
    pub fn compute(samples: []u64) BenchmarkStats {
        if (samples.len == 0) return .{};

        std.mem.sort(u64, samples, {}, std.sort.asc(u64));

        var sum: u64 = 0;
        for (samples) |sample| {
            sum += sample;
        }

        const len = samples.len;
        const p50_idx = len / 2;
        const p90_idx = @min(len - 1, (len * 90) / 100);
        const p99_idx = @min(len - 1, (len * 99) / 100);

        return .{
            .count = @intCast(len),
            .total_ns = sum,
            .min_ns = samples[0],
            .max_ns = samples[len - 1],
            .p50_ns = samples[p50_idx],
            .p90_ns = samples[p90_idx],
            .p99_ns = samples[p99_idx],
        };
    }
};

// ============================================================================
// Unit Tests & Invariant Verification
// ============================================================================

test "Timestamp monotonicity and duration calculation" {
    const t0 = now();
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        sum +%= @intCast(i);
        std.mem.doNotOptimizeAway(sum);
    }
    const t1 = now();

    const dt = t1.durationSince(t0);
    try std.testing.expect(t1.ns >= t0.ns);
    try std.testing.expectEqual(t1.ns - t0.ns, dt);

    const cycles = t1.toCycles();
    try std.testing.expectEqual(t1.ns * 4, cycles);
}

test "Time and cycle conversions at 4.0 GHz" {
    // 1 ns = 4 cycles
    try std.testing.expectEqual(@as(u64, 4_000), toCyclesAt4Ghz(1_000));
    try std.testing.expectEqual(@as(u64, 1_000), cyclesToNs(4_000));
    try std.testing.expectEqual(@as(f64, 0.25), cyclesToNsFloat(1));
    try std.testing.expectEqual(@as(f64, 2.5), cyclesToNsFloat(10));
}

test "BenchmarkStats streaming accumulation" {
    var stats = BenchmarkStats.empty;
    try std.testing.expectEqual(@as(u64, 0), stats.count);
    try std.testing.expectEqual(@as(f64, 0.0), stats.meanNs());

    stats.record(100);
    stats.record(200);
    stats.record(300);

    try std.testing.expectEqual(@as(u64, 3), stats.count);
    try std.testing.expectEqual(@as(u64, 600), stats.total_ns);
    try std.testing.expectEqual(@as(u64, 100), stats.min_ns);
    try std.testing.expectEqual(@as(u64, 300), stats.max_ns);
    try std.testing.expectEqual(@as(f64, 200.0), stats.meanNs());
    try std.testing.expectEqual(@as(u64, 800), stats.cyclesAt4Ghz());

    stats.reset();
    try std.testing.expectEqual(@as(u64, 0), stats.count);
    try std.testing.expectEqual(@as(u64, 0), stats.total_ns);
}

test "BenchmarkStats in-place percentiles calculation" {
    var samples = [_]u64{ 500, 100, 900, 300, 200, 800, 700, 1000, 400, 600 };
    const stats = BenchmarkStats.compute(&samples);

    try std.testing.expectEqual(@as(u64, 10), stats.count);
    try std.testing.expectEqual(@as(u64, 5500), stats.total_ns);
    try std.testing.expectEqual(@as(u64, 100), stats.min_ns);
    try std.testing.expectEqual(@as(u64, 1000), stats.max_ns);
    try std.testing.expectEqual(@as(u64, 600), stats.p50_ns);
    try std.testing.expectEqual(@as(u64, 1000), stats.p90_ns);
    try std.testing.expectEqual(@as(u64, 1000), stats.p99_ns);
    try std.testing.expectEqual(@as(f64, 550.0), stats.meanNs());
}

test "BenchmarkStats empty slice handling" {
    var empty_samples = [_]u64{};
    const stats = BenchmarkStats.compute(&empty_samples);
    try std.testing.expectEqual(@as(u64, 0), stats.count);
    try std.testing.expectEqual(@as(u64, 0), stats.total_ns);
    try std.testing.expectEqual(@as(f64, 0.0), stats.meanNs());
}
