//! ============================================================================
//! Klaxon Build System (Zig 0.17.1)
//! ============================================================================
//! Exposes primary developer quality gates and spike targets:
//! - `zig build lint`  : executes tools/lint.zig static analysis
//! - `zig build test`  : executes all test suites & fuzzer oracles
//! - `zig build run-*` : executes specified spike benchmarks

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ------------------------------------------------------------------------
    // Quality Gate: TigerStyle Linter
    // ------------------------------------------------------------------------
    const lint_exe = b.addExecutable(.{
        .name = "lint",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/lint.zig"),
            .target = b.graph.host,
            .optimize = .debug,
        }),
    });
    const run_lint = b.addRunArtifact(lint_exe);
    run_lint.addPassthruArgs();
    const lint_step = b.step("lint", "Run TigerStyle static analysis linter");
    lint_step.dependOn(&run_lint.step);

    // ------------------------------------------------------------------------
    // Quality Gate: Unit Tests & Spike Oracles
    // ------------------------------------------------------------------------
    const test_step = b.step("test", "Run all unit tests and fuzzer spikes");

    const tests = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "root", .path = "src/root.zig" },
        .{ .name = "platform-os", .path = "src/platform/os.zig" },
        .{ .name = "platform-time", .path = "src/platform/time.zig" },
        .{ .name = "gpu-tests", .path = "src/gpu/hal.zig" },
        .{ .name = "gpu-mock", .path = "src/gpu/mock.zig" },
        .{ .name = "lint-tests", .path = "tools/lint.zig" },
        .{ .name = "fuzz-newton", .path = "spikes/test_fuzz_newton.zig" },
        .{ .name = "autopsy-glyph", .path = "spikes/autopsy_glyph_delta1.zig" },
        .{ .name = "paint-pipeline", .path = "spikes/paint_pipeline.zig" },
        .{ .name = "resize-1cycle", .path = "spikes/resize_1cycle.zig" },
        .{ .name = "truetype-real", .path = "spikes/truetype_real.zig" },
        .{ .name = "comptime-graph", .path = "spikes/comptime_graph.zig" },
    };

    inline for (tests) |t| {
        const is_spike = std.mem.startsWith(u8, t.path, "spikes/");
        const test_opt = if (is_spike and optimize == .debug) .fast else optimize;
        const test_exe = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(t.path),
                .target = target,
                .optimize = test_opt,
            }),
        });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }

    // ------------------------------------------------------------------------
    // Spikes Execution Runners & Benchmark Step
    // ------------------------------------------------------------------------
    const bench_step = b.step("bench", "Run all flagship performance benchmarks");

    const spikes = [_]struct { name: []const u8, path: []const u8, is_bench: bool }{
        .{ .name = "engine_m0", .path = "spikes/engine_m0.zig", .is_bench = false },
        .{ .name = "paint_pipeline", .path = "spikes/paint_pipeline.zig", .is_bench = false },
        .{ .name = "resize_1cycle", .path = "spikes/resize_1cycle.zig", .is_bench = true },
        .{ .name = "truetype_real", .path = "spikes/truetype_real.zig", .is_bench = true },
        .{
            .name = "perf_100x_prototype",
            .path = "spikes/perf_100x_prototype.zig",
            .is_bench = true,
        },
        .{
            .name = "alt_architectures_bench",
            .path = "spikes/alt_architectures_bench.zig",
            .is_bench = true,
        },
    };

    inline for (spikes) |s| {
        const exe = b.addExecutable(.{
            .name = s.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(s.path),
                .target = target,
                .optimize = optimize,
            }),
        });
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.addPassthruArgs();
        const run_step = b.step(b.fmt("run-{s}", .{s.name}), b.fmt("Run spike {s}", .{s.name}));
        run_step.dependOn(&run_cmd.step);
        if (s.is_bench) {
            bench_step.dependOn(&run_cmd.step);
        }
    }
}
