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
8. **One-pass CSR fusion** — atomic-packed CSR replaces the temp slab + scan + compaction
   (see the dedicated section). The single biggest serial win (+18% pure-tess CPU); pushed
   serial CPU past voro++.

## Current standing vs voro++

| regime | device | voro++ | ratio |
|---|---:|---:|---:|
| CPU serial, pure tessellation, N=1M | 76.8 k/s | 74.7 k/s | **1.03×** |
| CPU serial, **with** force gradients | 62.7 k/s | 75 k/s | 0.84× (device does strictly more) |
| **CPU 24-core**, with forces, N=1M | **982 k/s** | 75 k/s (serial) | **13×** |
| **GPU (RTX 5080)**, pure tess, N=1M | **2435 k/s** | 75 k/s (serial) | **32×** |
| **GPU (RTX 5080)**, with forces, N=1M | **1994 k/s** | 75 k/s (serial) | **26×** |

After the one-pass CSR fusion (below), the device **matches/beats voro++ on serial CPU
pure tessellation (1.03×)** and is **13–32×** in the production regimes (multicore, GPU).
The "~10% serial gap" that the rest of this note analyses is now **closed** — it turned
out not to be the gather/cache floor but the temp-slab→CSR staging.

## Why voro++ led on serial (the gap, now closed by the CSR fusion above)

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

### Fuse the temp-slab round-trip — DONE, the big win (closed the gap)

The build wrote each cell's facets to a fixed-stride temp slab; a separate exclusive scan
then a compaction copied the temp into the packed CSR, recomputing the connecting vector
with `round()` per facet. This was replaced by a **one-pass fusion**: each cell reserves a
contiguous CSR range with a single `atomic_fetch_add` and writes its facets straight into
an over-buffer (the connecting vector is the cut's own plane vector — no minimal-image
recompute); the used prefix is then copied into a right-sized view in one **contiguous**
pass (no scan, no strided gather). The atomic was first verified **free** on both backends
(a per-cell `atomic_fetch_add` probe left CPU scaling and GPU throughput unchanged), so
there is no contention regression. Because the CSR is now packed in cell-finish order,
`cellFacetOffset` is a per-cell base and the view carries an explicit `cellFacetCount`
(`facetEnd(i)=base(i)+count(i)`); consumers use `facetBegin`/`facetEnd` and are unchanged.

**Measured (N=1M, bit-exact):** CPU serial pure tessellation **64.8 → 76.8 kcells/s
(0.91 → 1.03× voro++ — now beats it)**, with forces 52 → 62.7 (0.74 → 0.84×); CPU 24-core
**improved** 915 → 982 (no parallel regression); GPU pure 2316 → 2435 (31→32×), with forces
1876 → 1994 (25→26×). The strided temp write + scan + strided compaction + per-facet
`round()` cost **~18%** of serial pure-tess — far more than the `csr` phase timer showed,
because it hid the temp's ~2.6 GB/build allocation + first-touch.

**Net:** the "~10% serial gap" was **not** the gather/cache floor after all — it was the
temp-slab→CSR staging. Of the layout sweep: density (CPU win), Morton (GPU-only win),
per-cell scratch (marginal), candidate cap (bounded, dead), **temp-slab fusion (the win —
serial now ≥ voro++)**. There is no obvious next layout lever; further gains would need the
block-processing redesign (analysed and judged not worth it for a CPU+GPU-portable engine)
or GPU team-per-cell.

**Feasibility verdict (updated after measuring Morton):** the Morton lever turned out to
be a **GPU** win (now shipped, +18% with forces), not the CPU lever it was expected to be —
on CPU the ~2-seeds/cell density already captured the grid locality. The serial CPU gap was
ultimately closed by the **CSR fusion** above (now ≥ voro++).

## GPU is occupancy-limited — the next big lever (a separate, major effort)

The GPU (RTX 5080, ~960 GB/s, ~10⁴ cores) does **~2435 kcells/s pure-tess — only ~2.5× the
24-core CPU** (982 kcells/s). That ratio is low for this hardware and the cause is **not**
compute or bandwidth: it is **occupancy**.

Evidence — `cuobjdump --dump-resource-usage` on the build kernel (RTX 5080, sm_120):
**142 registers/thread and a 32 736-byte (32 KB) stack frame/thread.**
- The **142 registers** alone cap it to ~14 of 48 warps/SM → **~29% theoretical occupancy**
  (register-limited). (Achieved occupancy via Nsight Compute needs GPU perf-counter
  permission — `ERR_NVGPUCTRPERM` without root — but the static ceiling is the point.)
- The **32 KB stack/thread** is the per-cell state: `ScratchCell<double>` (~20 KB:
  `vpos`/`vlab`/`rsq`/`dist`/`pvec`/`fArea`/`fdV`/… at CAP=128) plus the build kernel's
  candidate arrays `ckey`/`cjid` (~12 KB). At 32 KB/thread it lives entirely in **local
  memory** (global, L1/L2-cached) and starves occupancy — GPU threads want ≪1 KB of state.
- **Throughput rises with N** (2.0 → 2.2 → 2.4 Mcells/s at N = 50k → 200k → 1M, still
  climbing) — the signature of an under-occupied kernel that needs more cells to hide
  local-memory latency, rather than a saturated one.
- The earlier small shrinks confirm the regime: removing `edgeInv` (−9 KB) gave only +1.6%
  and the candidate cap can't shrink (full-sphere bound) — incremental per-thread trims
  don't cross the occupancy threshold.

**The fix is a different GPU parallelisation: team/warp-per-cell.** Build one cell per warp
(or team), keep the cell in **shared memory** (on-chip, fast), and make the per-thread state
tiny → high occupancy. The neighbour gather and the per-facet geometry parallelise cleanly
across a warp; the half-edge cut (`cutCell2`) is inherently sequential, so it stays on the
warp leader (or needs a genuinely parallel cut) — Amdahl on the cut bounds the win, so expect
**a few-fold**, not 10×, unless the cut itself is parallelised. This is the roadmap's
"at-scale GPU granularity" item and a major redesign of `cell_cutter.hpp` + the build kernel;
it is **orthogonal to the CSR/physics layer** (still publishes the same `TessellationView`).

Related GPU items to fold in:
- **Over-buffer over-allocation caps N — DONE.** The CSR fusion's over-buffer was
  `N×MAXF_TMP` (50), ~3× the real need, and was held alive alongside the full compact CSR
  during the pack (~2.5× the CSR) → OOM on a 16 GB GPU at N≈4M. Fixed: size from a mean-facet
  estimate (`N×18`; mean faces/cell ~15.5, the *sum* has negligible relative variance) with an
  atomic overflow guard, and pack each over-buffer component into its compact view then free it
  immediately (peak ≈ over-buffer + one compact array, not over-buffer + full CSR). N=4M/5M/6M
  now build on a 16 GB card (was OOM past ~3M); N=1M unchanged; bit-exact. See the team-per-cell
  section below.
- **CAP/precision**: a smaller cell representation (lower CAP for facet-indexed arrays;
  mixed-precision storage) directly buys occupancy and pairs naturally with shared-memory
  cells. **This is now the binding lever** — see the measured team-per-cell results below.

An optional Morton-ordered *output* could also speed the physics force gather's neighbour
access on GPU, at the cost of the `cell i == particle i` contract (opt-in).

**Caveat worth stating:** single-thread CPU is the least important regime — real runs use
all cores or the GPU, where the device is already 12–30× voro++. So closing the last 10%
serial is a completeness goal, not a throughput necessity; the layout work is worth doing
mainly because Morton ordering *also* benefits the multicore and GPU paths.

## Team/warp-per-cell — implemented, measured: occupancy-bound by the shared cell

The team-per-cell redesign above was implemented incrementally and gated behind
`VORFLOW_TEAM=<teamSize>` (the default GPU path stays the per-thread RangePolicy build, so
production throughput is unchanged). The per-cell numerics are factored into one
`CellBuilder` functor used by both paths, so they are **bit-exact** (validated on the device
+ MPI ctests, Voronoi and Power, volErr 1e-15).

What was done and measured (RTX 5080, sm_120, N=1M, `double`):

| build path | pure-tess | with forces | note |
|---|---:|---:|---:|
| per-thread default | **2459 k/s** | **2024 k/s** | cell in 32 KB/thread local frame |
| team, leader-only | 439 | 175 | cell in shared; cuobjdump stack 32 736 → **200 B** |
| team, **parallel gather** | **753** | 208 | gather across the warp (Voronoi) |

- **The 32 KB/thread local frame is eliminated** — cuobjdump confirms the team build kernel's
  stack drops from 32 736 to 200 bytes (the `ScratchCell` + candidate arrays now live in team
  shared memory). That was the stated goal of moving the cell on-chip.
- **But occupancy is now shared-memory-limited.** The cell + candidate arrays are ~32 KB of
  *shared* per team, so only ~3 teams/SM are resident (a ~100 KB shared SM). With the cut (and,
  with forces, the geometry) still on the warp leader, that is far fewer concurrent cuts than
  the per-thread path's many in-flight cells — so even a perfectly parallel gather cannot reach
  the default. Team size 32 (one warp) is the measured sweet spot (8/16/64 are worse).
- **Parallel gather is bit-exact for Voronoi** because the per-shell min-heap re-sorts each
  shell's candidates by key — the cut order is independent of the (now racy, parallel) gather
  order. Power keeps the serial leader gather (no re-sort; parallelising it would reorder the
  cuts → needs a deterministic sort first).

**Verdict (measured, not predicted):** warp-per-cell at `CAP=128` **cannot beat the per-thread
path** — the binding constraint is the **shared-memory footprint of the cell**, which caps
teams/SM at ~3. The gather parallelisation is a genuine 1.7× over leader-only but the path is
~3× below the default. **The next lever is therefore cell-shrinking, not more parallelism**:
a smaller cell representation (lower `CAP` for the facet/vertex-indexed arrays; mixed-precision
storage for the geometry arrays) shrinks the shared footprint so many more teams fit per SM —
*then* the team path's parallel gather/geometry can overtake the per-thread build. Remaining
team-path work, all gated and lower-value until the cell shrinks: parallelise `computeGeometry`
across the team with a **deterministic** (bit-exact) per-facet reduction — naïve vertex-parallel
atomic accumulation reorders the FP sums and is *not* bit-exact — and a deterministic Power
gather sort. The half-edge `cutCell2` stays leader-sequential (Amdahl) unless a cooperative cut
is written.
