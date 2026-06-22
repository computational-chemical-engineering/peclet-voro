# Tessellation performance — analysis, current state, and the road past voro++

This note is the durable record of the device tessellator's performance work: how the
serial/parallel costs were measured, what was optimised, where we stand against
**voro++**, and whether the remaining serial gap can be closed by memory layout while
keeping CPU + GPU parallelism and the physics-extension surface intact.

All numbers below are on a **Threadripper PRO 5965WX (24 cores)** + **RTX 5080**, random
uniform seeds, `double`. Benchmarks: `tests/kokkos/bench_device.cpp` (vs voro++ and the
legacy CPU builder) and `tests/kokkos/bench_cutter.cpp` (per-cell sub-phase profiler,
counters gated by `-DVORFLOW_CUTTER_PROFILE`). Reproduce a comparison with
`OMP_NUM_THREADS=1 ./bench_device 1000000` (env knobs: `VORFLOW_NOFORCEGEOM`,
`VORFLOW_DENS`, `VORFLOW_SW`, `VORFLOW_DEVONLY`).

## Where the time goes (Phase-0 profile, serial)

Per **cell build** (cut on pre-sorted neighbours): cut **57%**, force-gradient geometry
(`fdV`) **37%**, volume 5%. But the *standalone* cutter runs at 209 kcells/s (4.8 µs/cell)
while the full tessellator is ~44 kcells/s (22.7 µs/cell) doing the **same** 42.9
`cutCell2`/cell — so of the whole serial tessellation:

| phase | share | notes |
|---|---|---|
| **neighbour gather** | **~60%** | the dominant cost; memory-latency bound |
| cut (`cutCell2`) | ~21% | a faithful port of the legacy half-edge cutter |
| force-gradient geometry | ~15% | `computeGeometry` → `fdV`; voro++ does **not** compute this |
| volume + CSR compaction | ~5% | |

The headline: **the cutter is not the bottleneck — the gather is.** A cutter rewrite
was therefore shelved (1.5× on 21% ≈ 7% overall).

## Optimisations applied (all bit-exact vs the legacy oracle, volErr 1e-15)

1. **Adaptive expanding search** — grow the gathered sphere shell-by-shell, process each
   shell closest-first via a min-heap, stop at the security radius. Most cells close at
   the innermost shell.
2. **Temp-slab fix** — `WithoutInitializing` + a tight facet cap + dropping the
   recomputable connecting vector removed a ~1.2 GB/build zero-fill that throttled
   scaling and made large N infeasible.
3. **Grid-order reorder** — build cells in grid-sorted order so the gather reads
   contiguous memory (fixed the large-N cache cliff).
4. **sw-independent wrap** — a per-offset single-step grid wrap + a compare-based minimal
   image (bit-identical to `round(r/L)` in the single-image range) replaced a per-cell
   "interior" flag, so a large safety `sw` stays fast.
5. **Backend-aware grid density** — `kSeedsPerCell = (host && !Power) ? 2 : 1`. ~2
   seeds/cell on **CPU** packs neighbours into fewer, contiguous cells (cache win); the
   **GPU** keeps 1/cell (its gather is bandwidth/latency-hidden, so coarser only adds
   per-thread work); **Power** keeps 1/cell (its no-early-out full-sphere gather would
   exceed the candidate cap at 2/cell).
6. **Force geometry opt-in** — `buildTessellation(..., withForceGeom)`; pure tessellation
   skips the `fdV` gradient (the apples-to-apples comparison with voro++, which never
   computes it).

## Current standing vs voro++

| regime | device | voro++ | ratio |
|---|---:|---:|---:|
| CPU serial, pure tessellation, N=1M | 67.7 k/s | 74.7 k/s | **0.91×** |
| CPU serial, pure tessellation, N=100k | 71.0 k/s | 75.4 k/s | **0.94×** |
| CPU serial, **with** force gradients | 52 k/s | 75 k/s | 0.74× (device does strictly more) |
| **CPU 24-core**, with forces, N=1M | **892 k/s** | 75 k/s (serial) | **11.9×** |
| **GPU (RTX 5080)**, pure tess, N=1M | **2258 k/s** | 75 k/s (serial) | **30×** |

So in the regimes that matter for production — multicore and GPU — the device is **12–30×
voro++**. The only place voro++ still leads is **single-thread CPU pure tessellation, by
~10%**, and that gap is memory layout / cache misses, not arithmetic.

## Why voro++ is still ~10% faster serial

voro++ packs ~8 particles per **block** (contiguous in memory) and processes one cell at a
time with a tiny, reused working set that stays hot in L1/L2. Our gather, even at 2
seeds/cell, still indexes the grid **linearly** (`gc = x + y·dimx + z·dimx·dimy`), so the
z-neighbours of a cell are `dimx·dimy` entries apart: the 3-D gather sphere is scattered
across `cellStart[]` and the position array, and `cellStart[]` (~N/2 entries) spills L2 →
cache misses on every cell. That scatter is the residual ~10%.

## Can we reach/pass voro++ on memory layout — and keep parallelism + physics? Yes (feasible)

The key structural fact: **all of these levers live below the `TessellationView` CSR
boundary.** The build reorders internally and writes results back at the original cell
index, so physics consumers (which read the published CSR: `cellVolume`, `facetArea`,
`facetConnect`=dV, `facetConnVec`, `facetNeighbor`) and the parallel structure are
untouched. Memory-layout work and physics extensions are orthogonal.

In expected-impact order:

1. **Morton / Z-order the grid (highest impact, low risk).** Replace the linear cell
   index with a Morton code so spatially-near cells are near in memory; the gather
   sphere's `cellStart`/position reads become near-contiguous instead of scattered.
   We already pay for a counting-sort reorder — switching the sort key to Morton is
   essentially free, and the suite already ships a fast BMI2/AVX-512 encoder in
   **`morton/`** (a sibling repo). Morton ordering helps **both** backends: CPU cache
   locality *and* GPU memory coalescing (it is the standard layout for GPU spatial codes),
   so it does not trade one for the other. This alone should close most of the ~10%.
2. **Shrink the per-cell scratch.** `ScratchCell` is ~30 KB, dominated by
   `edgeInv[128][3][3]` (~9 KB). A smaller scratch stays L1-resident across cells (voro++'s
   real advantage) and stops evicting the gather's working set. Compute `edgeInv`
   transiently / lower the cap. Helps the cut and reduces cache pollution; on GPU it cuts
   local-memory/register pressure → better occupancy.
3. **Fuse the temp-slab round-trip.** Facets are written to a temp slab then compacted to
   the CSR (a full read+write of facet data) — voro++ has no such round-trip. Emitting the
   CSR directly (atomic facet allocation, or count-then-fill) removes that traffic. Bigger
   change; the lever most likely to push **past** voro++ rather than just match it.

**Feasibility verdict:** reaching voro++ parity on serial CPU is feasible and low-risk
(Morton + smaller scratch, both internal, both validated bit-exact against the legacy
oracle); exceeding it is plausible but needs the temp-slab fusion too. None of it
conflicts with CPU/GPU parallelism (Morton and a smaller scratch *help* both) or with
physics extensions (they sit above the CSR). An optional Morton-ordered *output* could
additionally speed the physics force gather's neighbour access, at the cost of the
`cell i == particle i` contract — so it would be opt-in.

**Caveat worth stating:** single-thread CPU is the least important regime — real runs use
all cores or the GPU, where the device is already 12–30× voro++. So closing the last 10%
serial is a completeness goal, not a throughput necessity; the layout work is worth doing
mainly because Morton ordering *also* benefits the multicore and GPU paths.
