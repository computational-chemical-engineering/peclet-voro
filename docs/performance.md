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
7. **Morton (Z-order) grid indexing on GPU** — clusters the gather's spatial neighbourhood
   for better memory coalescing (+18% with forces). GPU-only; it regresses the CPU (see
   below). Backend-gated, with a linear fallback for pathological grids.

## Current standing vs voro++

| regime | device | voro++ | ratio |
|---|---:|---:|---:|
| CPU serial, pure tessellation, N=1M | 67.7 k/s | 74.7 k/s | **0.91×** |
| CPU serial, pure tessellation, N=100k | 71.0 k/s | 75.4 k/s | **0.94×** |
| CPU serial, **with** force gradients | 52 k/s | 75 k/s | 0.74× (device does strictly more) |
| **CPU 24-core**, with forces, N=1M | **892 k/s** | 75 k/s (serial) | **11.9×** |
| **GPU (RTX 5080)**, pure tess, N=1M | **2316 k/s** | 75 k/s (serial) | **31×** |
| **GPU (RTX 5080)**, with forces, N=1M | **1876 k/s** | 75 k/s (serial) | **25×** |

So in the regimes that matter for production — multicore and GPU — the device is **12–31×
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

### Morton / Z-order the grid — DONE, and it split by backend (measured)

Replacing the linear cell index with a 3D Morton (Z-order) code clusters a cell's
spatial neighbourhood in memory. We tried it (device-portable magic-bits `morton3`;
the `morton/` library's fast encoders are CPU-BMI2 and not callable from a GPU kernel,
and the encode is a tiny O(N) cost anyway). **Measured, it does NOT help both backends —
it splits them** (RTX 5080 / Threadripper 5965WX, N=1M):

| | linear | Morton |
|---|---:|---:|
| GPU pure-tess | 2258 k/s (30×) | **2316 k/s (31×)** |
| GPU with forces | 1584 k/s (21×) | **1876 k/s (25×, +18%)** |
| CPU serial pure-tess | **0.91×** | 0.87× (regresses) |

On the **GPU** the spatial order improves the gather's **memory coalescing** (the large
with-forces gain is the geometry reads + facet writes), and the per-cell encode is hidden
by spare ALU. On the **CPU** it is a net loss: Morton codes are **not additive**, so the
gather must re-encode every neighbour cell in the hot loop (~18 ops vs ~5 for the linear
index) and that cost is not hidden, while the ~2-seeds/cell density already supplies the
cache locality. So it shipped **GPU-only** (`useMorton = !kHostBackend && …`, with a
linear fallback when the power-of-two padding would inflate the cell array), mirroring the
seeds-per-cell density knob. **Lesson: the CPU serial gap is not closed by reordering the
*grid* — that locality is already captured by the density; the remaining cost is the
per-cell working set and the temp round-trip (below).**

The remaining levers (and what shrinking the scratch actually bought):

### Shrink the per-cell scratch — DONE, marginal (measured)

`ScratchCell` was ~30 KB, dominated by the stored `edgeInv[128][3][3]` (~9 KB). That
tensor is purely transient (filled and consumed inside `computeGeometry`/`gradFacetAreaSq`,
never touched during the gather/cut), so it was replaced by a `vertexEdgeInv` helper that
recomputes a vertex's 3×3 block on demand — bit-identical, no capacity change. **Measured
N=1M: small and only where geometry runs** — CPU with-forces 0.74→0.75×, GPU with-forces
1876→1906 kcells/s (+1.6%); pure tessellation flat on both. The reason it is *not* the
hoped-for CPU lever: `edgeInv` was never on the gather's hot path, so removing it doesn't
relieve the gather's cache; and on the GPU the binding local-memory item is the build
kernel's **candidate arrays** (`ckey[1024]`/`cjid[1024]` ≈ 12 KB/thread), not the cell.
Still a genuine 9 KB reduction with no downside, and it tidies the geometry.

### Shrink the candidate arrays — TRIED, doesn't pay off (measured)

`ckey[1024]`/`cjid[1024]` (12 KB/thread) is the binding GPU local-memory item, so a
smaller cap should raise occupancy. It can't be done: **the arrays must hold the
full-sphere gather.** Any Voronoi cell that expands to `sw`, and *every* Power cell (no
early-out), gathers ~`nOff` candidates — the worklist size, **667 at sw=4** plus Poisson
fluctuation (~750–800). Measured on the GPU: a cap of 512 **overflows** (all Power cells +
~1/4000 Voronoi → wrong cells); the safe minimum (~896) gives **no** gain (2300/1898 vs
2324/1906 kcells/s, within noise) — 1.5 KB of ~33 KB doesn't move occupancy. Reverted to
1024. (The typical Voronoi `nc` is ~60 thanks to the early-out, but the array must be sized
for the rare full-expansion cell, so the *cap* can't shrink.)

The lever that remains:

1. **Fuse the temp-slab round-trip.** Facets are written to a temp slab then compacted to
   the CSR (a full read+write of facet data) — voro++ has no such round-trip. Emitting the
   CSR directly (atomic facet allocation, or count-then-fill) removes that traffic. The
   lever most likely to push **past** voro++ on CPU rather than just match it.

**Net so far:** three of the four layout levers are now spent — grid Morton (GPU-only win),
per-cell scratch (marginal), candidate-array cap (bounded by the full-sphere gather, no
gain). The CPU serial gap to voro++ (~10%) and further GPU gains both now rest on the
**temp-slab fusion**, the one structural change left.

**Feasibility verdict (updated after measuring Morton):** the Morton lever turned out to
be a **GPU** win (now shipped, +18% with forces), not the CPU lever it was expected to be —
on CPU the ~2-seeds/cell density already captured the grid locality. Closing the last ~10%
on serial CPU therefore rests on the per-cell working set (smaller `ScratchCell`) and
removing the temp-slab round-trip; both are internal, bit-exact-validatable, and compatible
with CPU/GPU parallelism and physics (they sit below the CSR). Parity is still plausible
that way but is no longer a single easy change. An optional Morton-ordered *output* could
additionally speed the physics force gather's neighbour access on GPU, at the cost of the
`cell i == particle i` contract — so it would be opt-in.

**Caveat worth stating:** single-thread CPU is the least important regime — real runs use
all cores or the GPU, where the device is already 12–30× voro++. So closing the last 10%
serial is a completeness goal, not a throughput necessity; the layout work is worth doing
mainly because Morton ordering *also* benefits the multicore and GPU paths.
