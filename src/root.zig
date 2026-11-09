//! ============================================================================
//! Klaxon Framework Core Root (`src/root.zig`)
//! ============================================================================
//! Unified export interface for Klaxon engine subsystems:
//! - `platform`: Zero-libc OS syscalls, monotonic clock, and direct file I/O.
//! - `time`: Monotonic timestamps, 4.0 GHz cycle conversions, and benchmark stats.
//!
//! Standards: TigerStyle (< 100 columns, zero dynamic allocations, zero libc).

pub const platform = @import("platform/os.zig");
pub const time = @import("platform/time.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
