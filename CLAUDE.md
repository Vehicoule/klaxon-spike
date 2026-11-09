# CLAUDE.md — Quick Reference Guide for Klaxon (`klaxon-spike`)

Klaxon is an experimental high-performance, deterministic, zero-allocation reactive UI engine in Zig 0.17+ following TigerStyle engineering.

---

## Quick Commands

### Build & Quality Gates
```bash
# Run TigerStyle static linter
zig build lint
# or directly:
zig run tools/lint.zig

# Check code formatting (always run before committing)
zig fmt --check spikes/*.zig tools/*.zig build.zig

# Format all Zig files
zig fmt spikes/*.zig tools/*.zig build.zig

# Run all test suites
zig build test
```

### Spikes Execution
```bash
# 1-cycle resize layout benchmark (heterogeneous 20,000 nodes)
zig run spikes/resize_1cycle.zig -O ReleaseFast

# TrueType font parsing & vector SDF typography benchmark (Arial.ttf)
zig run spikes/truetype_real.zig -O ReleaseFast

# Adversarial 1,000 curves Newton solver geometry fuzzer
zig test spikes/test_fuzz_newton.zig -O ReleaseFast

# Glyph 92 autopsy & half-step quantization proof
zig test spikes/autopsy_glyph_delta1.zig -O ReleaseFast

# Paint pipeline & atlas allocation tests
zig test spikes/paint_pipeline.zig -O ReleaseFast

# End-to-end M0 engine spike
zig run spikes/engine_m0.zig -O ReleaseFast
```

### Multi-Target & Cross-Compilation Checks
```bash
# Non-POSIX freestanding fence (must compile without libc or POSIX)
zig build-obj spikes/resize_1cycle.zig -target wasm32-freestanding -O ReleaseSmall
zig build-obj spikes/resize_1cycle.zig -target x86_64-windows-gnu -O ReleaseSmall

# Size-budget verification
zig build-exe spikes/engine_m0.zig -target x86_64-linux-musl -O ReleaseSmall -fstrip
# Must be <= 15,360 bytes (actual ~13,032 bytes)

# Rosetta 2 test run on macOS Apple Silicon (2-step method: cross-compile then run)
zig test spikes/paint_pipeline.zig -target x86_64-macos -O ReleaseFast --test-no-exec -femit-bin=paint_pipeline_x86
arch -x86_64 ./paint_pipeline_x86
rm ./paint_pipeline_x86
```

---

## Code Style & Invariants (TigerStyle)

- **Line length**: maximum **100 columns**. Use trailing commas to let `zig fmt` wrap nicely.
- **Indentation**: 4 spaces, no tabs.
- **Zero dynamic allocation**: steady-state runtime working sets use static BSS `align(64)`.
  No `page_allocator`, `c_allocator`, or `GeneralPurposeAllocator` in core/layout/paint.
- **No POSIX leakage**: core/layout/paint modules must never import `std.posix`.
- **Assertions**: assert inputs, assert invariants, pair assertions (write vs read).
- **Control flow**: simple, explicit, bounded `while` loops with upper iteration limits.
- **Metrics**: measure in nanoseconds (`ns`) and CPU cycles at reference 4.0 GHz ($1\text{ ns} = 4.0\text{ cycles}$).

Consult `.agents/skills/zig-0-17/SKILL.md` and `.agents/skills/zig-tiger-style/SKILL.md` for full details.
