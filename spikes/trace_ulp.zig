const std = @import("std");

pub fn main() void {
    const p0x: f32 = 14.657732;
    const p1x: f32 = 11.418612;
    const p2x: f32 = 8.179493;

    const p0y: f32 = 10.896245;
    const p1y: f32 = 19.448122;
    const p2y: f32 = 28.0;

    const fx: f32 = 6.5;
    const fy: f32 = 30.5;

    // Endpoint directly:
    const dx_end = p2x - fx;
    const dy_end = p2y - fy;
    const dsq_end = dx_end * dx_end + dy_end * dy_end;
    const d_end = @sqrt(dsq_end);

    // Fast formula with t_line = 1.0:
    const bx = 2.0 * (p1x - p0x);
    const by = 2.0 * (p1y - p0y);
    const cx = p0x - fx;
    const cy = p0y - fy;
    const dx_fast = bx * 1.0 + cx;
    const dy_fast = by * 1.0 + cy;
    const dsq_fast = dx_fast * dx_fast + dy_fast * dy_fast;
    const d_fast = @sqrt(dsq_fast);

    std.debug.print("dx_end = {d:.9}, dx_fast = {d:.9}, diff = {e}\n", .{ dx_end, dx_fast, dx_end - dx_fast });
    std.debug.print("dy_end = {d:.9}, dy_fast = {d:.9}, diff = {e}\n", .{ dy_end, dy_fast, dy_end - dy_fast });
    std.debug.print("dsq_end = {d:.9}, dsq_fast = {d:.9}, diff = {e}\n", .{ dsq_end, dsq_fast, dsq_end - dsq_fast });
    std.debug.print("d_end = {d:.9}, d_fast = {d:.9}, diff = {e}\n", .{ d_end, d_fast, d_end - d_fast });
}
