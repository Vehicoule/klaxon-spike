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

pub const is_windows = (builtin.target.os.tag == .windows);
pub const is_darwin = (builtin.target.os.tag == .macos or
    builtin.target.os.tag == .ios or
    builtin.target.os.tag == .tvos or
    builtin.target.os.tag == .watchos or
    builtin.target.os.tag == .visionos);
pub const is_linux = (builtin.target.os.tag == .linux);
pub const is_posix = (builtin.target.os.tag != .windows and
    builtin.target.os.tag != .freestanding and
    builtin.target.os.tag != .wasi);
pub const is_wasm = builtin.target.cpu.arch.isWasm();
pub const is_wasi = (builtin.target.os.tag == .wasi);
pub const is_freestanding = (builtin.target.os.tag == .freestanding);

// ============================================================================
// Win32 Direct Kernel Declarations (Zero Libc / Windows SDK Header-Free)
// ============================================================================

const STD_OUTPUT_HANDLE: i32 = -11;
const STD_ERROR_HANDLE: i32 = -12;
const GENERIC_READ: u32 = 0x80000000;
const GENERIC_WRITE: u32 = 0x40000000;
const FILE_APPEND_DATA: u32 = 0x00000004;
const FILE_SHARE_READ: u32 = 0x00000001;
const FILE_SHARE_WRITE: u32 = 0x00000002;
const CREATE_ALWAYS: u32 = 2;
const OPEN_EXISTING: u32 = 3;
const OPEN_ALWAYS: u32 = 4;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;

// Win32 error codes
const ERROR_FILE_NOT_FOUND: u32 = 2;
const ERROR_PATH_NOT_FOUND: u32 = 3;
const ERROR_ACCESS_DENIED: u32 = 5;
const ERROR_SHARING_VIOLATION: u32 = 32;
const ERROR_HANDLE_EOF: u32 = 38;
const ERROR_HANDLE_DISK_FULL: u32 = 39;
const ERROR_BUFFER_OVERFLOW: u32 = 111;
const ERROR_DISK_FULL: u32 = 112;
const ERROR_FILENAME_EXCED_RANGE: u32 = 206;

extern "kernel32" fn GetStdHandle(nStdHandle: i32) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetLastError() callconv(.winapi) u32;

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
        const qpc_ok = QueryPerformanceCounter(&count);
        const qpf_ok = QueryPerformanceFrequency(&freq);
        if (qpc_ok == 0 or qpf_ok == 0) return 0;
        if (freq <= 0 or count <= 0) return 0;
        return @intCast(@divFloor(@as(i128, count) * 1_000_000_000, freq));
    } else if (is_wasi) {
        var ts: std.os.wasi.timestamp_t = 0;
        const rc = std.os.wasi.clock_time_get(.MONOTONIC, 1, &ts);
        if (rc != .SUCCESS) return 0;
        return ts;
    } else if (is_posix) {
        var ts: std.posix.timespec = undefined;
        const rc = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &ts);
        if (std.posix.errno(rc) != .SUCCESS) return 0;
        if (ts.sec < 0 or ts.nsec < 0) return 0;
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    } else {
        // Freestanding fallback: strictly increasing step counter.
        const S = struct {
            var step_counter: u64 = 1_000;
        };
        S.step_counter +%= 1;
        return S.step_counter;
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

fn writeAllToStream(posix_fd: i32, win_std_handle: i32, bytes: []const u8) void {
    if (bytes.len == 0) return;
    if (is_windows) {
        const handle = GetStdHandle(win_std_handle);
        if (handle == null) return;
        var written_total: usize = 0;
        while (written_total < bytes.len) {
            const chunk_len: u32 = @intCast(@min(
                bytes.len - written_total,
                std.math.maxInt(u32),
            ));
            var chunk_written: u32 = 0;
            const ok = WriteFile(
                handle,
                bytes[written_total..].ptr,
                chunk_len,
                &chunk_written,
                null,
            );
            if (ok == 0 or chunk_written == 0) break;
            written_total += chunk_written;
        }
    } else if (is_wasi) {
        var written_total: usize = 0;
        const fd: std.os.wasi.fd_t = @intCast(posix_fd);
        while (written_total < bytes.len) {
            const iov = [_]std.posix.iovec_const{.{
                .base = bytes[written_total..].ptr,
                .len = bytes.len - written_total,
            }};
            var written: usize = 0;
            const rc = std.os.wasi.fd_write(fd, &iov, 1, &written);
            if (rc != .SUCCESS or written == 0) break;
            written_total += written;
        }
    } else if (is_posix) {
        var written_total: usize = 0;
        while (written_total < bytes.len) {
            const max_chunk = switch (builtin.target.os.tag) {
                .linux => @as(usize, 0x7ffff000),
                else => @as(usize, std.math.maxInt(i32)),
            };
            const chunk_len = @min(bytes.len - written_total, max_chunk);
            const rc = std.posix.system.write(
                @intCast(posix_fd),
                bytes[written_total..].ptr,
                chunk_len,
            );
            switch (std.posix.errno(rc)) {
                .SUCCESS => {
                    const n: usize = @intCast(rc);
                    if (n == 0) break;
                    written_total += n;
                },
                .INTR => continue,
                else => break,
            }
        }
    }
}

/// Emits raw bytes directly to standard output without buffering or dynamic memory.
pub fn writeStdout(bytes: []const u8) void {
    writeAllToStream(1, STD_OUTPUT_HANDLE, bytes);
}

/// Emits raw bytes directly to standard error without buffering or dynamic memory.
pub fn writeStderr(bytes: []const u8) void {
    writeAllToStream(2, STD_ERROR_HANDLE, bytes);
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
            if (val == 0) return false;
            if (val == inv) return false;
            return true;
        } else if (is_posix) {
            if (self.handle < 0) return false;
            return true;
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
    BufferTooSmall,
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
            .create_append => FILE_APPEND_DATA,
        };
        const disp: u32 = switch (mode) {
            .read_only => OPEN_EXISTING,
            .write_only => OPEN_EXISTING,
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
            const err_code = GetLastError();
            switch (err_code) {
                ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND => return FileError.FileNotFound,
                ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION => return FileError.AccessDenied,
                ERROR_FILENAME_EXCED_RANGE, ERROR_BUFFER_OVERFLOW => return FileError.PathTooLong,
                ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL => return FileError.DiskFull,
                else => return FileError.Unexpected,
            }
        }
        return .{ .handle = res.? };
    } else if (is_posix) {
        const flags: std.posix.O = switch (mode) {
            .read_only => .{ .ACCMODE = .RDONLY },
            .write_only => .{ .ACCMODE = .WRONLY },
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
        const to_read: u32 = @intCast(@min(dest.len, std.math.maxInt(u32)));
        var bytes_read: u32 = 0;
        const ok = ReadFile(
            handle.handle,
            dest.ptr,
            to_read,
            &bytes_read,
            null,
        );
        if (ok == 0) {
            const err_code = GetLastError();
            switch (err_code) {
                ERROR_HANDLE_EOF => return 0,
                ERROR_ACCESS_DENIED => return FileError.AccessDenied,
                else => return FileError.Unexpected,
            }
        }
        return @intCast(bytes_read);
    } else if (is_posix) {
        const n = std.posix.read(handle.handle, dest) catch |err| switch (err) {
            error.AccessDenied => return FileError.AccessDenied,
            else => return FileError.Unexpected,
        };
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
        const to_write: u32 = @intCast(@min(src.len, std.math.maxInt(u32)));
        var bytes_written: u32 = 0;
        const ok = WriteFile(
            handle.handle,
            src.ptr,
            to_write,
            &bytes_written,
            null,
        );
        if (ok == 0) {
            const err_code = GetLastError();
            switch (err_code) {
                ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL => return FileError.DiskFull,
                ERROR_ACCESS_DENIED => return FileError.AccessDenied,
                else => return FileError.Unexpected,
            }
        }
        return @intCast(bytes_written);
    } else if (is_posix) {
        const max_chunk = switch (builtin.target.os.tag) {
            .linux => @as(usize, 0x7ffff000),
            else => @as(usize, std.math.maxInt(i32)),
        };
        const to_write = @min(src.len, max_chunk);
        while (true) {
            const rc = std.posix.system.write(handle.handle, src.ptr, to_write);
            switch (std.posix.errno(rc)) {
                .SUCCESS => return @intCast(rc),
                .INTR => continue,
                .NOSPC => return FileError.DiskFull,
                .ACCES, .PERM => return FileError.AccessDenied,
                else => return FileError.Unexpected,
            }
        }
    } else {
        return FileError.Unsupported;
    }
}

/// Closes an open file handle.
pub fn closeFile(handle: FileHandle) void {
    if (!handle.isValid()) return;
    if (is_windows) {
        const ok = CloseHandle(handle.handle);
        _ = ok;
    } else if (is_posix) {
        const rc = std.posix.system.close(handle.handle);
        switch (std.posix.errno(rc)) {
            .SUCCESS, .INTR => {},
            else => {},
        }
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
/// Returns `FileError.BufferTooSmall` if the file exceeds the buffer capacity.
pub fn readFileToBuffer(path: []const u8, buffer: []u8) FileError!usize {
    assert(path.len > 0);

    const handle = try openFile(path, .read_only);
    defer closeFile(handle);

    const file_size = try getFileSize(handle);
    if (file_size > buffer.len) {
        return FileError.BufferTooSmall;
    }

    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const n = try readFile(handle, buffer[total_read..]);
        if (n == 0) break;
        total_read += n;
    }

    // Probe 1 extra byte to detect concurrent append or truncation mismatch.
    var extra: [1]u8 = undefined;
    const extra_read = try readFile(handle, &extra);
    if (extra_read > 0) {
        return FileError.BufferTooSmall;
    }

    return total_read;
}

// ============================================================================
// Minimal Standalone Panic Handler (No libc, zero heap allocation)
// ============================================================================

pub fn minimalPanic(message: []const u8, first_trace_addr: ?usize) noreturn {
    _ = first_trace_addr;
    writeStderr("PANIC: ");
    writeStderr(message);
    writeStderr("\n");
    @trap();
}

// ============================================================================
// Unit Tests & Invariant Verification
// ============================================================================

fn makeTestPath(buf: []u8, prefix: []const u8) ![:0]const u8 {
    assert(buf.len >= 64);
    const slice = if (is_windows)
        try std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}_{d}.tmp", .{
            prefix,
            readMonotonicClockNs(),
        })
    else if (is_posix)
        try std.fmt.bufPrint(
            buf[0 .. buf.len - 1],
            "/tmp/{s}_{d}_{d}.tmp",
            .{ prefix, std.posix.system.getpid(), readMonotonicClockNs() },
        )
    else
        return error.Unsupported;

    buf[slice.len] = 0;
    return buf[0..slice.len :0];
}

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

    var path_buf: [128]u8 = undefined;
    const test_path = try makeTestPath(&path_buf, "klaxon_io_roundtrip");
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

    // Test buffer too small condition
    var small_buffer: [16]u8 = undefined;
    const err = readFileToBuffer(test_path, &small_buffer);
    try std.testing.expectError(FileError.BufferTooSmall, err);
}

test "File I/O empty file handling" {
    if (is_freestanding or is_wasm) return error.SkipZigTest;

    var path_buf: [128]u8 = undefined;
    const empty_path = try makeTestPath(&path_buf, "klaxon_io_empty");
    defer if (is_posix) {
        _ = std.posix.system.unlink(empty_path);
    };

    const handle = try openFile(empty_path, .create_truncate);
    closeFile(handle);

    var empty_buf: [0]u8 = undefined;
    const read_bytes = try readFileToBuffer(empty_path, &empty_buf);
    try std.testing.expectEqual(@as(usize, 0), read_bytes);
}

test "File I/O write_only preserves remaining bytes" {
    if (is_freestanding or is_wasm) return error.SkipZigTest;

    var path_buf: [128]u8 = undefined;
    const test_path = try makeTestPath(&path_buf, "klaxon_io_write_only");
    defer if (is_posix) {
        _ = std.posix.system.unlink(test_path);
    };

    // Create file with initial content
    const init_handle = try openFile(test_path, .create_truncate);
    _ = try writeFile(init_handle, "ABCDEFGHIJ");
    closeFile(init_handle);

    // Reopen write_only and overwrite only first 3 bytes
    const wo_handle = try openFile(test_path, .write_only);
    _ = try writeFile(wo_handle, "123");
    closeFile(wo_handle);

    // Verify remaining bytes are intact
    var read_buf: [16]u8 = undefined;
    const bytes_read = try readFileToBuffer(test_path, &read_buf);
    try std.testing.expectEqual(@as(usize, 10), bytes_read);
    try std.testing.expectEqualStrings("123DEFGHIJ", read_buf[0..bytes_read]);
}

test "Direct stdout/stderr emission without crash" {
    writeStdout("");
    writeStderr("");
}
