# AGENTS.md — AI Engineering Guide for Klaxon (`klaxon-spike`)

> **Target Audience**: AI coding assistants (Claude, Antigravity, Copilot, etc.) and human contributors.
> **Language**: Zig 0.17.1 (stable).
> **Engineering Methodology**: Strict **TigerStyle** (distilled from TigerBeetle safety-critical systems).

---

## 1. Prime Directive & Core Invariants

Design goal priority is non-negotiable: **Safety > Performance > Developer Experience**.

1. **Zero Dynamic Allocation in Steady-State (`0 malloc / free`)**
   - All runtime working sets (node graphs, render lists, text caches, quad instances, replay logs)
     reside in static BSS memory with strict 64-byte cache-line alignment: `align(64)`.
   - Never use `std.heap.page_allocator`, `c_allocator`, or `GeneralPurposeAllocator` in
     `core/`, `layout/`, or `paint/`.
   - Dynamic allocations are strictly quarantined to build scripts and developer CLI tools (`tools/`).

2. **Sub-Cycle Layout Execution (< 1 CPU Cycle / Node)**
   - Flexbox layout runs non-recursively in 4 separate passes: BottomUpX, TopDownX, BottomUpY, TopDownY.
   - Layout arithmetic uses quantized Q14 fixed-point numbers in `i16` with elastic container
     partitioning, achieving **0.094 cycle / node** on ARM64 and **0.352 cycle / node** on x86_64.

3. **Cumulative Left/Right Boundary Law (Zero Hairline Gaps)**
   - Rounding cracks and 1-pixel seams between sibling containers are mathematically eliminated by
     tracking cumulative affine boundary laws $(\alpha_L, \beta_L)$ and $(\alpha_R, \beta_R)$ in vector registers:
     $$\alpha_{L, i+1} \equiv \alpha_{R, i}, \quad \beta_{L, i+1} \equiv \beta_{R, i} + \text{gap}$$
   - Both boundaries are rounded independently in SIMD lanes: $w_i = \text{round}(R_i) - \text{round}(L_i)$.

4. **Compile-Time Topological Graph (Zero Virtual DOM)**
   - UI trees compile directly into flat Compressed Sparse Row (CSR) tables in `.rodata`.
   - State mutations bypass runtime tree diffing and update target memory slots directly via comptime dispatch.

5. **Analytical TrueType Vector SDF Typography**
   - Decodes TrueType `.ttf` geometry directly without libc or FreeType.
   - Rasterizes into an R8 distance field atlas via $4 \times 4$ coarse-fine culling tiles and SIMD4 Horner evaluation.
   - Multi-seed Newton-Raphson solver converges with 0 failures on 1,000 adversarial curves.

6. **Comptime-Dispatched GPU HAL (19-Function Contract)**
   - Narrow Hardware Abstraction Layer parameterized at compile time: `Engine(GpuBackend, Window)`.
   - 19 pinned functions with zero vtable overhead, including:
     - `setClipRect(encoder: *CommandEncoder, rect: RectU32) void`
     - `readbackTarget(device: *Device, target: RenderTarget, dest: []u8) !void`
     - `SamplerDesc`: pinned to bilinear filtering, clamp-to-edge, zero mipmaps.

7. **Deterministic Replay & Time-Travel**
   - State is stored as flat Plain Old Data (POD).
   - Instant state restoration via sub-50 ns `memcpy`.
   - Automated CPU golden oracle enforces visual parity against GPU output ($\Delta E_{00} < 2.0$, $\text{DSSIM} < 0.01$).

8. **Zero Libc Dependency & Non-POSIX Architecture**
   - Core modules (`core/`, `layout/`, `paint/`) must compile seamlessly on freestanding targets
     (`wasm32-freestanding`, `x86_64-windows-gnu`) with zero references to `std.posix`.

---

## 2. Formatting & Code Conventions (TigerStyle)

- **Hard column limit**: maximum **100 characters per line**, no exceptions.
- **Indentation**: 4 spaces (no tabs).
- **Trailing commas**: always include trailing commas in multi-line argument lists, struct literals,
  and enum declarations to allow `zig fmt` to preserve readable line wrapping.
- **Assertions**: assert all function arguments at entry. Pair assertions (assert upon write and
  assert upon read). Assert compile-time constants: `comptime assert(...)`.
- **Bounded loops**: every `while` loop must have an explicit iteration counter and upper bound.
- **Explicit types**: use `u32`, `i16`, `f32`; avoid architecture-dependent `usize` in data structures.
- **Out pointers**: initialize large structs in-place via an out pointer: `fn init(target: *T) void`.
- **Comments**: full sentences explaining *why* decisions were made, not just *what*.

---

## 3. Metrics Conventions & Size Budgets

- **Time units**: all latencies must be reported in **nanoseconds (ns)** and **CPU clock cycles**.
- **Reference clock**: **4.0 GHz** ($1\text{ cycle} = 0.25\text{ ns}$, $1\text{ ns} = 4.0\text{ cycles}$).
- **Binary size budgets (stripped ReleaseSmall)**:
  - `x86_64-linux-musl`: **<= 15 KB** (15 360 bytes). Baseline: 13.03 KB.
  - `aarch64-macos` (Mach-O): **<= 52 KB** (52 000 bytes, accounting for 16 KB page padding). Baseline: 49.9 KB.

---

## 4. Key Commands

```bash
# Static analysis & TigerStyle quality gate
zig run tools/lint.zig
zig build lint

# Format check
zig fmt --check spikes/*.zig tools/*.zig build.zig

# Run all test suites
zig build test

# Run specific spikes in ReleaseFast
zig run spikes/resize_1cycle.zig -O ReleaseFast
zig run spikes/truetype_real.zig -O ReleaseFast
zig test spikes/test_fuzz_newton.zig -O ReleaseFast
zig test spikes/autopsy_glyph_delta1.zig -O ReleaseFast
zig test spikes/paint_pipeline.zig -O ReleaseFast

# Rosetta 2 execution on macOS ARM64 (2-step cross-compile + arch invocation)
zig test spikes/paint_pipeline.zig -target x86_64-macos -O ReleaseFast --test-no-exec -femit-bin=paint_pipeline_x86
arch -x86_64 ./paint_pipeline_x86
rm ./paint_pipeline_x86

# Non-POSIX compiler fence validation
zig build-obj spikes/resize_1cycle.zig -target wasm32-freestanding -O ReleaseSmall
zig build-obj spikes/resize_1cycle.zig -target x86_64-windows-gnu -O ReleaseSmall
```

---

## 5. Skills & Deep Documentation

Agents must consult the repository skills located in `.agents/skills/`:
- `.agents/skills/zig-0-17/SKILL.md`: Zig 0.17 API updates, `@backingInt`/`@fromBackingInt`,
  `@splat`, `SafeAllocator`, and build system patterns.
- `.agents/skills/zig-tiger-style/SKILL.md`: complete TigerStyle guidelines, assertion pairing,
  memory layouts, and naming rules.
