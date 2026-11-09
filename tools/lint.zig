//! ============================================================================
//! Klaxon Quality Gate: TigerStyle Static Analysis Linter
//! ============================================================================
//! Enforces non-negotiable TigerStyle architectural rules across production sources:
//! 1. Line length <= 100 characters/columns (hard limit, no exceptions).
//! 2. Zero trailing whitespace.
//! 3. Zero dynamic allocation (`page_allocator`, `c_allocator`, `GeneralPurposeAllocator`)
//!    in `core/`, `layout/`, and `paint/`.
//! 4. Zero POSIX leakage (`std.posix`) in `core/`, `layout/`, and `paint/`.
//!
//! Scope:
//! By default, scans production code: `src/`, `tools/`, and `build.zig`.
//! Historical exploratory research prototypes in `spikes/` are excluded by default
//! and can be explicitly scanned via the `--all` CLI flag.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

/// 2 MB static BSS buffer for zero-allocation file reading.
var file_buffer: [2 * 1024 * 1024]u8 align(64) = undefined;

/// Maximum line length allowed by TigerStyle.
const column_limit_max: usize = 100;

/// Statistics gathered during the lint pass.
pub const LintStats = struct {
    files_scanned: usize = 0,
    files_with_errors: usize = 0,
    total_errors: usize = 0,
};

/// Check if a relative path belongs to the core, layout, or paint modules.
pub fn isCoreLayoutPaint(path: []const u8) bool {
    if (std.mem.startsWith(u8, path, "core/") or std.mem.startsWith(u8, path, "src/core/")) {
        return true;
    }
    if (std.mem.startsWith(u8, path, "layout/") or std.mem.startsWith(u8, path, "src/layout/")) {
        return true;
    }
    if (std.mem.startsWith(u8, path, "paint/") or std.mem.startsWith(u8, path, "src/paint/")) {
        return true;
    }
    if (std.mem.startsWith(u8, path, "gpu/") or std.mem.startsWith(u8, path, "src/gpu/")) {
        return true;
    }
    if (std.mem.indexOf(u8, path, "/core/") != null) return true;
    if (std.mem.indexOf(u8, path, "/layout/") != null) return true;
    if (std.mem.indexOf(u8, path, "/paint/") != null) return true;
    if (std.mem.indexOf(u8, path, "/gpu/") != null) return true;
    return false;
}

/// Checks whether a path should be skipped from linting.
pub fn shouldSkipPath(path: []const u8, scan_all: bool) bool {
    if (std.mem.indexOf(u8, path, ".zig-cache") != null) return true;
    if (std.mem.indexOf(u8, path, "zig-cache") != null) return true;
    if (std.mem.indexOf(u8, path, "zig-out") != null) return true;
    if (std.mem.indexOf(u8, path, ".git") != null) return true;
    if (std.mem.indexOf(u8, path, ".agents") != null) return true;

    // Spikes are historical research prototypes, skipped unless --all is passed.
    if (!scan_all) {
        if (std.mem.startsWith(u8, path, "spikes/") or
            std.mem.indexOf(u8, path, "/spikes/") != null)
        {
            return true;
        }
    }
    return false;
}

/// Reads an entire file into the static BSS buffer using direct POSIX calls.
fn readFileContents(path: []const u8) ![]const u8 {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0);
    defer _ = std.posix.system.close(fd);

    var total_read: usize = 0;
    while (total_read < file_buffer.len) {
        const n = try std.posix.read(fd, file_buffer[total_read..]);
        if (n == 0) break;
        total_read += n;
    }
    if (total_read == file_buffer.len) {
        return error.FileTooLarge;
    }
    return file_buffer[0..total_read];
}

/// Computes visible column count, handling UTF-8 code points correctly.
fn computeColumnCount(line: []const u8) usize {
    return std.unicode.utf8CountCodepoints(line) catch line.len;
}

/// Lints a single file buffer in memory and reports violations.
pub fn lintFile(path: []const u8, content: []const u8) usize {
    var error_count: usize = 0;
    const is_core = isCoreLayoutPaint(path);

    var line_number: usize = 1;
    var line_start: usize = 0;

    while (line_start < content.len) {
        var line_end = line_start;
        while (line_end < content.len and content[line_end] != '\n') {
            line_end += 1;
        }

        var line_len = line_end - line_start;
        // Strip carriage return if present
        if (line_len > 0 and content[line_start + line_len - 1] == '\r') {
            line_len -= 1;
        }

        const line = content[line_start .. line_start + line_len];
        const col_count = computeColumnCount(line);

        // Rule 1: Line length <= 100 columns
        if (col_count > column_limit_max) {
            if (!builtin.is_test) {
                std.debug.print(
                    "{s}:{d}:1: error: line length {d} exceeds limit of {d} columns\n",
                    .{ path, line_number, col_count, column_limit_max },
                );
            }
            error_count += 1;
        }

        // Rule 2: Zero trailing whitespace
        if (line_len > 0) {
            const last_char = line[line_len - 1];
            if (last_char == ' ' or last_char == '\t') {
                if (!builtin.is_test) {
                    std.debug.print(
                        "{s}:{d}:{d}: error: trailing whitespace detected\n",
                        .{ path, line_number, col_count },
                    );
                }
                error_count += 1;
            }
        }

        // Rules 3 & 4: Core module invariant fences
        if (is_core) {
            if (std.mem.indexOf(u8, line, "page_allocator") != null or
                std.mem.indexOf(u8, line, "c_allocator") != null or
                std.mem.indexOf(u8, line, "GeneralPurposeAllocator") != null)
            {
                if (!builtin.is_test) {
                    std.debug.print(
                        "{s}:{d}:1: error: forbidden dynamic allocator in core/layout/paint\n",
                        .{ path, line_number },
                    );
                }
                error_count += 1;
            }

            if (std.mem.indexOf(u8, line, "std.posix") != null or
                std.mem.indexOf(u8, line, ".posix") != null)
            {
                if (!builtin.is_test) {
                    std.debug.print(
                        "{s}:{d}:1: error: forbidden std.posix import in core/layout/paint\n",
                        .{ path, line_number },
                    );
                }
                error_count += 1;
            }
        }

        line_start = if (line_end < content.len) line_end + 1 else content.len;
        line_number += 1;
    }

    return error_count;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    var stats = LintStats{};
    var scan_all: bool = false;

    const args = try init.minimal.args.toSlice(arena);

    var explicit_files = false;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--all")) {
            scan_all = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            explicit_files = true;
        }
    }

    if (explicit_files) {
        // Explicit file list passed via arguments
        for (args[1..]) |path| {
            if (std.mem.startsWith(u8, path, "-")) continue;
            if (shouldSkipPath(path, scan_all)) continue;
            if (!std.mem.endsWith(u8, path, ".zig")) continue;

            const content = readFileContents(path) catch |err| {
                std.debug.print("{s}: error: cannot read file: {s}\n", .{ path, @errorName(err) });
                stats.total_errors += 1;
                continue;
            };

            stats.files_scanned += 1;
            const errs = lintFile(path, content);
            if (errs > 0) {
                stats.files_with_errors += 1;
                stats.total_errors += errs;
            }
        }
    } else {
        // Walk repository tree starting at current directory
        var dir = try std.Io.Dir.openDir(.cwd(), io, ".", .{ .iterate = true });
        defer dir.close(io);

        var walker = try dir.walk(gpa);
        defer walker.deinit();

        while (try walker.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;
            if (shouldSkipPath(entry.path, scan_all)) continue;

            const content = readFileContents(entry.path) catch |err| {
                std.debug.print(
                    "{s}: error: cannot read file: {s}\n",
                    .{ entry.path, @errorName(err) },
                );
                stats.total_errors += 1;
                continue;
            };

            stats.files_scanned += 1;
            const errs = lintFile(entry.path, content);
            if (errs > 0) {
                stats.files_with_errors += 1;
                stats.total_errors += errs;
            }
        }
    }

    std.debug.print(
        "[lint] {d} files scanned, {d} errors found across {d} files\n",
        .{ stats.files_scanned, stats.total_errors, stats.files_with_errors },
    );

    if (stats.total_errors > 0) {
        std.process.exit(1);
    }
}

// ============================================================================
// Unit Tests for TigerStyle Quality Gates
// ============================================================================

test "lintFile detects line exceeding 100 columns" {
    var long_buf: [102]u8 = @splat('a');
    long_buf[101] = '\n';
    const errs = lintFile("src/test.zig", &long_buf);
    try std.testing.expectEqual(@as(usize, 1), errs);
}

test "lintFile accepts line of exactly 100 columns" {
    var ok_buf: [101]u8 = @splat('a');
    ok_buf[100] = '\n';
    const errs = lintFile("src/test.zig", &ok_buf);
    try std.testing.expectEqual(@as(usize, 0), errs);
}

test "lintFile detects trailing whitespace" {
    const trailing_space = "const x = 1; \n";
    const trailing_tab = "const y = 2;\t\n";
    try std.testing.expectEqual(@as(usize, 1), lintFile("src/test.zig", trailing_space));
    try std.testing.expectEqual(@as(usize, 1), lintFile("src/test.zig", trailing_tab));
}

test "lintFile enforces 0 dynamic allocation in core modules" {
    const code_page_alloc = "const alloc = std.heap.page_allocator;\n";
    const code_c_alloc = "const alloc = c_allocator;\n";
    const code_gpa = "var gpa = std.heap.GeneralPurposeAllocator;\n";

    try std.testing.expectEqual(@as(usize, 1), lintFile("src/core/engine.zig", code_page_alloc));
    try std.testing.expectEqual(@as(usize, 1), lintFile("layout/box.zig", code_c_alloc));
    try std.testing.expectEqual(@as(usize, 1), lintFile("src/paint/atlas.zig", code_gpa));

    // Non-core modules are exempt from allocator check
    try std.testing.expectEqual(@as(usize, 0), lintFile("tools/tool.zig", code_page_alloc));
}

test "lintFile enforces no std.posix in core modules" {
    const posix_code = "const posix = @import(\"std\").posix;\n";
    try std.testing.expectEqual(@as(usize, 1), lintFile("src/core/sys.zig", posix_code));
    try std.testing.expectEqual(@as(usize, 1), lintFile("src/layout/sys.zig", posix_code));
    try std.testing.expectEqual(@as(usize, 1), lintFile("src/paint/sys.zig", posix_code));

    // Non-core modules may use posix
    try std.testing.expectEqual(@as(usize, 0), lintFile("tools/tool.zig", posix_code));
}

test "isCoreLayoutPaint path matching" {
    try std.testing.expect(isCoreLayoutPaint("core/foo.zig"));
    try std.testing.expect(isCoreLayoutPaint("src/core/foo.zig"));
    try std.testing.expect(isCoreLayoutPaint("layout/box.zig"));
    try std.testing.expect(isCoreLayoutPaint("src/layout/box.zig"));
    try std.testing.expect(isCoreLayoutPaint("paint/atlas.zig"));
    try std.testing.expect(isCoreLayoutPaint("src/paint/atlas.zig"));
    try std.testing.expect(isCoreLayoutPaint("gpu/hal.zig"));
    try std.testing.expect(isCoreLayoutPaint("src/gpu/hal.zig"));
    try std.testing.expect(!isCoreLayoutPaint("spikes/paint_pipeline.zig"));
    try std.testing.expect(!isCoreLayoutPaint("spikes/engine_m0.zig"));
    try std.testing.expect(!isCoreLayoutPaint("tools/lint.zig"));
}
