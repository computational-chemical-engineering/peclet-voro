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

**Serial CPU now matches/edges voro++ via a worklist gather (`CC_GATHER=1`).** The fine-grid sorted-offset
walk (default, and the GPU path) trailed voro++ by 0.89× purely on neighbour-search efficiency. A
voro++-style **worklist** — each grid cell subdivided into `S³` sub-regions (`CC_WLS`, default 3), with a
per-sub-region list of block offsets sorted by nearest-corner dist² (`rmin`) — replaces the runtime
per-block distance arithmetic with a table lookup, tightening the radius break. That alone closes the gap:
worklist ≈ **0.074–0.075 M/s vs voro++ ≈ 0.071–0.076 M/s (≈1.0–1.05×)**, still machine-exact, with the clip
and the GPU path untouched. Whole-block-accept via the farthest-corner dist² (`rmax`, `CC_WBA=1`) was tried
and is a **net loss** on this fine grid (≤~1 seed/block ⇒ nothing to amortise); kept opt-in, default off.

## Full build (gather + clip), Mcells/s

"ours **serial**" = the worklist gather (`CC_GATHER=1`, CPU's best); the sorted-offset walk it replaces is
in parentheses. 24-core/GPU use the sorted-offset path (the worklist is a CPU option).

| N | ours **serial** worklist (sorted) | **voro++** (serial, FP64) | ours **24-core** (48 thr, FP64) | ours **GPU** (FP32) | **SOTA** Liu GPU, full (FP32) |
|------:|------:|------:|------:|------:|------:|
| 10 k  | 0.074 (0.066) | 0.071 | 0.66 | 1.45 | — *(SOTA `n>14000` branch only)* |
| 100 k | 0.075 (0.067) | 0.075 | 1.28 | 4.62 | 5.82 |
| 1 M   | 0.075 (0.068) | 0.074 | 1.40 | **6.99** | 6.16 |

SOTA "full" = `N/(gather_ms+construct_ms)`; its two phases at 1 M are gather 17.5 Msites/s, construct
9.5 Mcells/s. Our GPU at 1 M (6.99) now **beats the SOTA full build (1.13×)** — the correctness fix lifted
it from 5.95 (inexact, tied) to 6.99 (exact, ahead).

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
- **GPU: ours beats the SOTA full build (1.13× at 1 M) and clip kernel (1.8×).** The branchless inner
  gather (per-block ±L periodic shift) helps the GPU; whole-block-accept's branches would diverge, so the
  CPU and GPU use different gathers (`#ifdef __CUDA_ARCH__`).
- **Scaling:** CPU throughput is flat in N (memory-bandwidth-saturated above ~100 k); the GPU climbs with
  N (1.45 → 6.99) as occupancy fills, needing ≥~300 k to near its ceiling. 24-core is bandwidth-bound
  (~1.4 M/s, well under the ~2.75 M/s clip-only ceiling).

**Headline:** our clip is SOTA everywhere; on GPU the full cold build now beats the Liu-2020 SOTA; on serial
CPU the worklist gather (`CC_GATHER=1`) brings us to **voro++ parity** (was 0.89×) at machine-exact accuracy
with a self-checked completeness guard, while we win decisively on multicore and GPU.

## Reproduce

```bash
cd vorflow
NS="10000 100000 1000000"
OMP_NUM_THREADS=1  CC_DENS=0.3 CC_GATHER=1 ./build/host-openmp/tests/kokkos/bench_convexcell $NS  # serial worklist + voro++
OMP_NUM_THREADS=48 CC_NOVORO=1 CC_DENS=0.3 ./build/host-openmp/tests/kokkos/bench_convexcell $NS  # 24-core (sorted path)
CC_NOVORO=1 CC_DENS=0.5 ./build/nvidia-cuda/tests/kokkos/bench_convexcell_f32          $NS  # GPU FP32
./build/nvidia-cuda/tests/kokkos/bench_construct_f32                                        # construct-only
extern_bench/build.sh && OMP_NUM_THREADS=1 extern_bench/bench_geogram 1000000               # clip: all 3
# SOTA (extern_bench/voro_gpu; see voro_gpu_bench.md for build + timing patch):
#   ./build/bin/VolumeVoronoiGPU box.tet box_sites_<N>.xyz 1 | grep '@@'
```
`bench_convexcell` reports `convexcell` (unsorted) and `spatial-sort` (Morton sorted-offset) k/s plus the
timed voro++ reference; with `CC_GATHER=1` it also reports the `worklist` line (the serial-CPU headline,
with its `exhausted=K` completeness guard). `CC_NOVORO=1` skips voro++; `CC_DENS` sets the grid density
(CPU optimum 0.3, GPU 0.5); `CC_WLS` the worklist sub-grid (default 3); `CC_WBA=1` enables (opt-in)
whole-block-accept; `CC_SW` overrides the search window.
