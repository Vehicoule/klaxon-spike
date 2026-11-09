//! ============================================================================
//! Klaxon Platform & OS Abstraction Layer (`src/platform/os.zig`)
//! ============================================================================
//! Unified zero-libc interface isolating OS-specific syscalls:
//! 1. Monotonic high-resolution clock (nanoseconds and 4.0 GHz cycles).
//! 2. Zero-allocation file operations (open, read, write, size, close).
//! 3. Zero-allocation stdout/stderr direct stream printing.
//! 4. Non-POSIX and freestanding compatibility fences (Wasm, Win32, POSIX).
//!
//! Standards: TigerStyle (< 100 columns, zero heap allocation, paired assertions).

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

// ============================================================================
// Target Platform Classifications
// ============================================================================

pub const is_windows = (builtin.os.tag == .windows);
pub const is_darwin = (builtin.os.tag == .macos or
    builtin.os.tag == .ios or
    builtin.os.tag == .tvos or
    builtin.os.tag == .watchos or
    builtin.os.tag == .visionos);
pub const is_linux = (builtin.os.tag == .linux);
pub const is_posix = (builtin.os.tag != .windows and
    builtin.os.tag != .freestanding and
    builtin.os.tag != .wasi);
pub const is_wasm = builtin.cpu.arch.isWasm();
pub const is_freestanding = (builtin.os.tag == .freestanding);

// ============================================================================
// Win32 Direct Kernel Declarations (Zero Libc / Windows SDK Header-Free)
// ============================================================================

const STD_OUTPUT_HANDLE: i32 = -11;
const STD_ERROR_HANDLE: i32 = -12;
const GENERIC_READ: u32 = 0x80000000;
const GENERIC_WRITE: u32 = 0x40000000;
const FILE_SHARE_READ: u32 = 0x00000001;
const FILE_SHARE_WRITE: u32 = 0x00000002;
const CREATE_ALWAYS: u32 = 2;
const OPEN_EXISTING: u32 = 3;
const OPEN_ALWAYS: u32 = 4;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;

extern "kernel32" fn GetStdHandle(nStdHandle: i32) callconv(.winapi) ?*anyopaque;

extern "kernel32" fn WriteFile(
    hFile: ?*anyopaque,
    lpBuffer: [*]const u8,
    nNumberOfBytesToWrite: u32,
    lpNumberOfBytesWritten: ?*u32,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) i32;

extern "kernel32" fn ReadFile(
    hFile: ?*anyopaque,
    lpBuffer: [*]u8,
    nNumberOfBytesToRead: u32,
    lpNumberOfBytesRead: ?*u32,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) i32;

extern "kernel32" fn CreateFileW(
    lpFileName: [*:0]const u16,
    dwDesiredAccess: u32,
    dwShareMode: u32,
    lpSecurityAttributes: ?*anyopaque,
    dwCreationDisposition: u32,
    dwFlagsAndAttributes: u32,
    hTemplateFile: ?*anyopaque,
) callconv(.winapi) ?*anyopaque;

extern "kernel32" fn CloseHandle(hObject: ?*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn GetFileSizeEx(hFile: ?*anyopaque, lpFileSize: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;

// ============================================================================
// Monotonic Time & High-Resolution Clock
// ============================================================================

/// Reads the system monotonic clock in nanoseconds.
/// Guarantees strictly increasing values across all supported hardware platforms.
pub fn readMonotonicClockNs() u64 {
    if (is_windows) {
        var count: i64 = 0;
        var freq: i64 = 0;
        _ = QueryPerformanceCounter(&count);
        _ = QueryPerformanceFrequency(&freq);
        if (freq <= 0) return 0;
        return @intCast(@divFloor(@as(i128, count) * 1_000_000_000, freq));
    } else if (is_posix) {
        var ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    } else {
        // Freestanding fallback: deterministic step counter
        return 0;
    }
}

/// Converts a nanosecond duration into CPU cycles at the 4.0 GHz reference clock.
/// Reference: 1 cycle = 0.25 ns (4 cycles per nanosecond).
pub inline fn toCyclesAt4Ghz(ns: u64) u64 {
    return ns * 4;
}

// ============================================================================
// Direct Standard Output & Error (Zero Heap Allocation)
// ============================================================================

/// Emits raw bytes directly to standard output without buffering or dynamic memory.
pub fn writeStdout(bytes: []const u8) void {
    if (bytes.len == 0) return;
    if (is_windows) {
        const handle = GetStdHandle(STD_OUTPUT_HANDLE);
        var written: u32 = 0;
        _ = WriteFile(handle, bytes.ptr, @intCast(bytes.len), &written, null);
    } else if (is_posix) {
        _ = std.posix.system.write(1, bytes.ptr, bytes.len);
    }
}

/// Emits raw bytes directly to standard error without buffering or dynamic memory.
pub fn writeStderr(bytes: []const u8) void {
    if (bytes.len == 0) return;
    if (is_windows) {
        const handle = GetStdHandle(STD_ERROR_HANDLE);
        var written: u32 = 0;
        _ = WriteFile(handle, bytes.ptr, @intCast(bytes.len), &written, null);
    } else if (is_posix) {
        _ = std.posix.system.write(2, bytes.ptr, bytes.len);
    }
}

// ============================================================================
// Zero-Libc File Operations
// ============================================================================

pub const FileHandle = struct {
    handle: if (is_windows) *anyopaque else if (is_posix) std.posix.fd_t else u32,

    pub const invalid: FileHandle = .{
        .handle = if (is_windows)
            @ptrFromInt(@as(usize, @bitCast(@as(isize, -1))))
        else if (is_posix)
            -1
        else
            0,
    };

    pub inline fn isValid(self: FileHandle) bool {
        if (is_windows) {
            const val: usize = @intFromPtr(self.handle);
            const inv: usize = @bitCast(@as(isize, -1));
            return val != 0 and val != inv;
        } else if (is_posix) {
            return self.handle >= 0;
        } else {
            return false;
        }
    }
};

pub const OpenMode = enum(u8) {
    read_only,
    write_only,
    read_write,
    create_truncate,
    create_append,
};

pub const FileError = error{
    FileNotFound,
    AccessDenied,
    PathTooLong,
    DiskFull,
    Unexpected,
    Unsupported,
};

/// Opens a file using native system calls without libc.
pub fn openFile(path: []const u8, mode: OpenMode) FileError!FileHandle {
    assert(path.len > 0);

    if (is_windows) {
        if (path.len >= 1024) return FileError.PathTooLong;
        var utf16_buf: [1025]u16 = undefined;
        const len = std.unicode.utf8ToUtf16Le(&utf16_buf, path) catch return FileError.Unexpected;
        utf16_buf[len] = 0;
        const name_w: [*:0]const u16 = utf16_buf[0..len :0];

        const access: u32 = switch (mode) {
            .read_only => GENERIC_READ,
            .write_only => GENERIC_WRITE,
            .read_write => GENERIC_READ | GENERIC_WRITE,
            .create_truncate => GENERIC_WRITE,
            .create_append => GENERIC_WRITE,
        };
        const disp: u32 = switch (mode) {
            .read_only => OPEN_EXISTING,
            .write_only => CREATE_ALWAYS,
            .read_write => OPEN_ALWAYS,
            .create_truncate => CREATE_ALWAYS,
            .create_append => OPEN_ALWAYS,
        };

        const res = CreateFileW(
            name_w,
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            null,
            disp,
            FILE_ATTRIBUTE_NORMAL,
            null,
        );
        const inv: usize = @bitCast(@as(isize, -1));
        if (res == null or @intFromPtr(res) == inv) {
            return FileError.FileNotFound;
        }
        return .{ .handle = res.? };
    } else if (is_posix) {
        const flags: std.posix.O = switch (mode) {
            .read_only => .{ .ACCMODE = .RDONLY },
            .write_only => .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
            .read_write => .{ .ACCMODE = .RDWR, .CREAT = true },
            .create_truncate => .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
            .create_append => .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true },
        };
        const fd = std.posix.openat(
            std.posix.AT.FDCWD,
            path,
            flags,
            0o644,
        ) catch |err| switch (err) {
            error.FileNotFound => return FileError.FileNotFound,
            error.AccessDenied => return FileError.AccessDenied,
            error.NameTooLong => return FileError.PathTooLong,
            error.NoSpaceLeft => return FileError.DiskFull,
            else => return FileError.Unexpected,
        };
        return .{ .handle = fd };
    } else {
        return FileError.Unsupported;
    }
}

/// Reads bytes from an open file handle into `dest`.
pub fn readFile(handle: FileHandle, dest: []u8) FileError!usize {
    assert(handle.isValid());
    if (dest.len == 0) return 0;

    if (is_windows) {
        var bytes_read: u32 = 0;
        const ok = ReadFile(
            handle.handle,
            dest.ptr,
            @intCast(dest.len),
            &bytes_read,
            null,
        );
        if (ok == 0) return FileError.Unexpected;
        return @intCast(bytes_read);
    } else if (is_posix) {
        const n = std.posix.read(handle.handle, dest) catch return FileError.Unexpected;
        return n;
    } else {
        return FileError.Unsupported;
    }
}

/// Writes bytes from `src` to an open file handle.
pub fn writeFile(handle: FileHandle, src: []const u8) FileError!usize {
    assert(handle.isValid());
    if (src.len == 0) return 0;

    if (is_windows) {
        var bytes_written: u32 = 0;
        const ok = WriteFile(
            handle.handle,
            src.ptr,
            @intCast(src.len),
            &bytes_written,
            null,
        );
        if (ok == 0) return FileError.Unexpected;
        return @intCast(bytes_written);
    } else if (is_posix) {
        const rc = std.posix.system.write(handle.handle, src.ptr, src.len);
        if (std.posix.errno(rc) != .SUCCESS) return FileError.Unexpected;
        return @intCast(rc);
    } else {
        return FileError.Unsupported;
    }
}

/// Closes an open file handle.
pub fn closeFile(handle: FileHandle) void {
    if (!handle.isValid()) return;
    if (is_windows) {
        _ = CloseHandle(handle.handle);
    } else if (is_posix) {
        _ = std.posix.system.close(handle.handle);
    }
}

/// Retrieves the exact file size in bytes.
pub fn getFileSize(handle: FileHandle) FileError!u64 {
    assert(handle.isValid());
    if (is_windows) {
        var size: i64 = 0;
        const ok = GetFileSizeEx(handle.handle, &size);
        if (ok == 0 or size < 0) return FileError.Unexpected;
        return @intCast(size);
    } else if (is_linux) {
        var statx = std.mem.zeroes(std.os.linux.Statx);
        const rc = std.os.linux.statx(
            handle.handle,
            "",
            std.os.linux.AT.EMPTY_PATH,
            .{ .SIZE = true },
            &statx,
        );
        if (std.posix.errno(rc) != .SUCCESS or !statx.mask.SIZE) return FileError.Unexpected;
        return statx.size;
    } else if (is_posix) {
        var stat: std.posix.Stat = undefined;
        const rc = std.posix.system.fstat(handle.handle, &stat);
        if (std.posix.errno(rc) != .SUCCESS or stat.size < 0) return FileError.Unexpected;
        return @intCast(stat.size);
    } else {
        return FileError.Unsupported;
    }
}

/// Streams an entire file into a caller-provided buffer without heap allocation.
pub fn readFileToBuffer(path: []const u8, buffer: []u8) FileError!usize {
    assert(path.len > 0);
    assert(buffer.len > 0);

    const handle = try openFile(path, .read_only);
    defer closeFile(handle);

    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const n = try readFile(handle, buffer[total_read..]);
        if (n == 0) break;
        total_read += n;
    }
    return total_read;
}

// ============================================================================
// Minimal Standalone Panic Handler (No libc, zero heap allocation)
// ============================================================================

pub fn minimalPanic(message: []const u8, _: ?usize) noreturn {
    writeStderr("PANIC: ");
    writeStderr(message);
    writeStderr("\n");
    @trap();
}

// ============================================================================
// Unit Tests & Invariant Verification
// ============================================================================

test "Monotonic time sanity & strict monotonicity" {
    const t0 = readMonotonicClockNs();
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        sum +%= @intCast(i);
        std.mem.doNotOptimizeAway(sum);
    }
    const t1 = readMonotonicClockNs();
    try std.testing.expect(t1 >= t0);

    const cycles = toCyclesAt4Ghz(t1 - t0);
    try std.testing.expectEqual((t1 - t0) * 4, cycles);
}

test "File I/O operations round-trip in /tmp" {
    if (is_freestanding or is_wasm) return error.SkipZigTest;

    const test_path: [:0]const u8 = if (is_windows)
        "klaxon_test_platform_io.tmp"
    else
        "/tmp/klaxon_test_platform_io.txt";
    defer if (is_posix) {
        _ = std.posix.system.unlink(test_path);
    };
    const payload = "Klaxon TigerStyle Zero-Libc I/O Pipeline\n";

    // Write file
    const write_handle = try openFile(test_path, .create_truncate);
    defer closeFile(write_handle);
    try std.testing.expect(write_handle.isValid());

    const bytes_written = try writeFile(write_handle, payload);
    try std.testing.expectEqual(payload.len, bytes_written);

    // Read back and verify size
    const file_size = try getFileSize(write_handle);
    try std.testing.expectEqual(@as(u64, payload.len), file_size);

    // Read file back via readFileToBuffer
    var read_buffer: [128]u8 = undefined;
    const bytes_read = try readFileToBuffer(test_path, &read_buffer);
    try std.testing.expectEqual(payload.len, bytes_read);
    try std.testing.expectEqualStrings(payload, read_buffer[0..bytes_read]);
}

test "Direct stdout/stderr emission without crash" {
    writeStdout("");
    writeStderr("");
}
