//! ============================================================================
//! Klaxon Framework Core Root (`src/root.zig`)
//! ============================================================================
//! Unified export interface for Klaxon engine subsystems:
//! - `platform`: Zero-libc OS syscalls, monotonic clock, and direct file I/O.
//! - `time`: Monotonic timestamps, 4.0 GHz cycle conversions, and benchmark stats.
//! - `gpu`: Comptime-dispatched 19-function GPU Hardware Abstraction Layer.
//! - `core`: Engine v2 coordinator, hardware bitmask CTZ reactive dispatcher,
//!   and generic zero-padding align(64) Structure of Arrays (SoA).
//! - `layout`: Sub-cycle Flexbox layout with cumulative boundary laws (zero hairline seams).
//!
//! Standards: TigerStyle (< 100 columns, zero dynamic allocations, zero libc).

pub const platform = @import("platform/os.zig");
pub const time = @import("platform/time.zig");
pub const gpu = @import("gpu/hal.zig");

pub const core = struct {
    pub const soa = @import("core/soa.zig");
    pub const reactive = @import("core/reactive.zig");
    pub const engine = @import("core/engine.zig");
};

pub const layout = struct {
    pub const flexbox = @import("layout/flexbox.zig");
};

test {
    const testing = @import("std").testing;
    testing.refAllDecls(@This());
    testing.refAllDecls(platform);
    testing.refAllDecls(time);
    testing.refAllDecls(gpu);
    testing.refAllDecls(core.soa);
    testing.refAllDecls(core.reactive);
    testing.refAllDecls(core.engine);
    testing.refAllDecls(layout.flexbox);
}
