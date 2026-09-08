# Cold Voronoi tessellation — throughput vs N (serial / multicore / GPU)

**Date:** 2026-06-26 · **GPU:** RTX 5080 (sm_120), CUDA 13.2 · **CPU:** AMD Threadripper PRO 5965WX
(24 cores / 48 threads)

Cold (from-scratch) Voronoi build — neighbour **gather + clip** — of N uniform random sites, vs N. Our
code vs **voro++** (serial reference) and the **SOTA GPU code** (Liu/Ma/Guo/Yan, TVCG 2020;
`extern_bench/voro_gpu`). Precision matches each reference: **CPU = FP64** (voro++ is FP64), **GPU = FP32**
(the SOTA GPU code is `float`). "Ours" = `bench_convexcell`'s Morton-sorted path at its per-backend optimal
grid density (CPU `CC_DENS=0.3`, GPU `CC_DENS=0.5`). Domain: **ours periodic**; the SOTA code is a clipped,
non-periodic tessellation in a fixed `[0,1000]³` tet domain.

All "ours" numbers below are **machine-exact** (space-filling identity Σvol=box to ~1e-14 FP64 / ~3e-8
FP32). A long-standing gather bug (block radius compared to `secR2` instead of `2·secR2`, a √2 under-reach)
had made the build silently inexact (~3.8e-5); fixing it made the build both exact **and** faster.

**A worklist gather is now the default on BOTH backends — it wins on each.** Each grid cell is subdivided
into `S³` sub-regions (`CC_WLS`); for the sub-region a query lands in, a precomputed list of block offsets
sorted by nearest-corner dist² (`rmin`) replaces the runtime per-block distance arithmetic with a table
lookup, tightening the radius break (it fires sooner → fewer blocks walked → less work *and* less warp
divergence). The same branchless inner loop as the old sorted-offset walk; only the offset sequence is new.
- **Serial CPU** (S=3): worklist ≈ **0.074–0.075 M/s vs voro++ ≈ 0.071–0.076 (≈1.0–1.05×)**, up from the
  sorted-offset walk's 0.89×.
- **GPU FP32** (S=4, dens≈1.0): **7.88 M/s vs sorted-offset 6.99 (+13%)**, now **1.28× the Liu-2020 SOTA**
  (was 1.13×). The earlier expectation that worklist branching would hurt the GPU was wrong — that concern
  was about whole-block-accept (off); the gather itself just walks fewer blocks. The block offsets are
  packed `(dx,dy,dz)`→one int (1 table load + bit-unpack per offset), worth a further ~1%.

Both machine-exact, clip untouched. Whole-block-accept via the farthest-corner dist² (`rmax`, `CC_WBA=1`)
was tried and is a **net loss** on this fine grid (≤~1 seed/block ⇒ nothing to amortise); kept opt-in, off.

## Full build (gather + clip), Mcells/s

"ours" columns = the worklist gather (the default on both backends); the sorted-offset walk it replaces is
in parentheses. Serial/24-core S=3 FP64 `dens=0.3`; GPU S=4 FP32 `dens=1.0`.

| N | ours **serial** worklist (sorted) | **voro++** (serial, FP64) | ours **24-core** (48 thr, FP64) | ours **GPU** worklist (sorted) | **SOTA** Liu GPU, full (FP32) |
|------:|------:|------:|------:|------:|------:|
| 10 k  | 0.074 (0.066) | 0.071 | 0.66 | 1.59 (1.45) | — *(SOTA `n>14000` branch only)* |
| 100 k | 0.075 (0.067) | 0.075 | 1.28 | 5.07 (4.62) | 5.82 |
| 1 M   | 0.075 (0.068) | 0.074 | 1.40 | **7.88** (6.99) | 6.16 |

SOTA "full" = `N/(gather_ms+construct_ms)`; its two phases at 1 M are gather 17.5 Msites/s, construct
9.5 Mcells/s. Our GPU at 1 M (**7.88**) now **beats the SOTA full build (1.28×)** — the correctness fix took
it from 5.95 (inexact) to 6.99 (exact, sorted-offset), and the worklist gather (packed offsets) lifts it
again to 7.88.

## Construct kernel only — clip from cached neighbours (GPU, FP32, 1 M)

| ours (G1, volume) | ours (G2, + derivatives) | SOTA construct |
|------:|------:|------:|
| **16.9** | 16.0 | **9.5** |

→ our clip kernel is **~1.8× the SOTA clip kernel**.

## Gather vs clip decomposition (serial, FP64, 1 M, µs/cell)

Each code's *own* clip-only (from cached neighbours) subtracted from its *own* full build, so the volume
method cancels — `bench_geogram` times all three clip kernels on the same precomputed neighbours; voro++'s
clip is its `voronoicell::plane` (no container/gather).

| | **clip** | **gather** (= full − clip) | **full** |
|---|------:|------:|------:|
| **ours** (worklist) | **8.7** (115 k/s) | 4.6 | 13.3 (75.4 k/s) |
| **ours** (sorted) | **8.7** (115 k/s) | 6.3 | 14.8 (67.6 k/s) |
| **voro++** | 9.5 (105 k/s) | **3.8** | 13.2 (75.7 k/s) |
| geogram | 10.6 (94 k/s) | — | — |

The worklist cuts the gather **6.3 → 4.6 µs/cell** (−27%), bringing the full build to voro++ parity. The
residual 4.6-vs-3.8 µs gather gap is voro++'s coarse-block container walking fewer, fuller blocks; our clip
(8.7 µs, SOTA) more than covers it. **Completeness is now self-checked:** the worklist reports `exhausted=K`
— cells that drained the offset list without the `rmin` radius break (potentially incomplete). `K=0` over
all N at the operating density ⇒ provably exact; the fixed-window sorted-offset walk has no such guard and
goes silently inexact if its window is too small (e.g. at coarse `dens`).

## Reading

- **Our clip is SOTA** on every platform — fastest of the three on CPU (8.7 µs, beats voro++ 9.5 and
  geogram 10.6) and ~1.8× the Liu-2020 clip kernel on GPU.
- **Serial CPU full build: ≈ voro++ parity (≈1.0–1.05×)** with the worklist gather (`CC_GATHER=1`),
  up from 0.89× with the sorted-offset walk. The old deficit was **entirely the gather** (6.3 vs voro++
  3.8 µs/cell); the worklist's per-sub-region `rmin` table (no runtime per-block distance arithmetic)
  tightens the radius break and drops the gather to 4.6 µs — closing the gap. Whole-block-accept (`rmax`,
  `CC_WBA=1`) was the obvious next lever but is a *net loss* here: our fine grid holds ≤~1 seed/block, so
  the accept amortises over nothing while adding a per-block branch (voro++'s win needs its coarse ~8-seed
  regions). The cell data structure and clip are untouched. **The win over voro++ is multicore (~18×) and
  GPU (~92×); serial CPU is now level.**
- **GPU: the worklist wins here too — 7.88 M/s at 1 M, 1.28× the SOTA full build** (was 6.99 / 1.13× on
  the sorted-offset path) and 1.8× the SOTA clip kernel. The worklist keeps the same branchless inner loop
  (per-block ±L periodic shift), just walks a tighter offset list — so it cuts work *and* warp divergence;
  the gain grows with density (its GPU optimum sits at a coarser `dens≈1.0, S=4`). Whole-block-accept's
  branches *would* diverge, which is why that stays off. (`CC_GATHER` selects the gather on either backend.)
- **Scaling:** CPU throughput is flat in N above ~100 k; the GPU climbs with N (1.59 → 5.07 → 7.88) as
  occupancy fills, needing ≥~300 k to near its ceiling.
- **Multicore CPU (5965WX, 24c/48t, measured):** near-**linear to ~12 cores** (75 → 900 k/s, 99.6%
  efficiency), then rolls off — 24 cores ≈ 1.29 M/s (71% eff), 48 threads (SMT) ≈ 1.66 M/s (+28%). The
  rolloff is **NOT clock throttling** (cores hold 4.54 GHz at 1 *and* 48 threads) and **NOT byte-bandwidth**
  (FP32 ≡ FP64 at every thread count — halving the position bytes changes nothing). Per-core it's
  **latency / cache-line-transaction-bound** (each cell pulls ~13–27 scattered 64 B neighbour lines;
  precision doesn't shrink the line count), and the 12→24 rolloff is the **shared uncore** (L3 / Infinity-
  Fabric / memory-controller transaction throughput across the 4 CCDs) saturating — not the algorithm, which
  is embarrassingly parallel. SMT adds only ~28% (latency-bound ⇒ the 2nd thread hides some stalls). Lever:
  SIMD *cells-as-lanes* (more compute per fetched line) raises the per-core ceiling; FP32 helps only once
  vectorised. Count **cores, not the 48 threads** — it's 17× on 24 physical cores.

**Headline:** our clip is SOTA everywhere; the worklist gather is now the default on both backends and wins
on each — **serial CPU reaches voro++ parity** (was 0.89×) and **GPU reaches 1.28× the Liu-2020 SOTA** (7.88
M/s, was 1.13×) — all machine-exact with a self-checked completeness guard, while we win decisively on
multicore too.

## Reproduce

```bash
cd voro
NS="10000 100000 1000000"
OMP_NUM_THREADS=1  CC_DENS=0.3 ./build/host-openmp/tests/kokkos/bench_convexcell      $NS  # serial worklist + voro++
OMP_NUM_THREADS=48 CC_NOVORO=1 CC_DENS=0.3 ./build/host-openmp/tests/kokkos/bench_convexcell $NS  # 24-core
CC_NOVORO=1 CC_DENS=1.0 ./build/nvidia-cuda/tests/kokkos/bench_convexcell_f32          $NS  # GPU FP32 (worklist S=4)
./build/nvidia-cuda/tests/kokkos/bench_construct_f32                                        # construct-only
extern_bench/build.sh && OMP_NUM_THREADS=1 extern_bench/bench_geogram 1000000               # clip: all 3
# SOTA (extern_bench/voro_gpu; see voro_gpu_bench.md for build + timing patch):
#   ./build/bin/VolumeVoronoiGPU box.tet box_sites_<N>.xyz 1 | grep '@@'
```
`bench_convexcell` reports `convexcell` (unsorted) and `spatial-sort` (Morton sorted-offset) k/s plus the
timed voro++ reference. The `worklist` line (the headline on both backends, with its `exhausted=K`
completeness guard) is **on by default everywhere**; `CC_GATHER=0/1` overrides. `CC_NOVORO=1` skips voro++;
`CC_DENS` sets the grid density (CPU optimum 0.3, GPU 1.0); `CC_WLS` the worklist sub-grid (default: host 3,
device 4); `CC_WBA=1` enables (opt-in) whole-block-accept; `CC_SW` overrides the search window.
