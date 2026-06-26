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

## Full build (gather + clip), Mcells/s

| N | ours **serial** (1 core, FP64) | **voro++** (serial, FP64) | ours **24-core** (48 thr, FP64) | ours **GPU** (FP32) | **SOTA** Liu GPU, full (FP32) |
|------:|------:|------:|------:|------:|------:|
| 10 k  | 0.066 | 0.073 | 0.66 | 1.45 | — *(SOTA `n>14000` branch only)* |
| 100 k | 0.067 | 0.076 | 1.28 | 4.62 | 5.82 |
| 1 M   | 0.068 | 0.076 | 1.40 | **6.99** | 6.16 |

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
| **ours** | **8.7** (115 k/s) | 6.3 | 14.8 (67.6 k/s) |
| **voro++** | 9.5 (105 k/s) | **3.8** | 13.2 (75.7 k/s) |
| geogram | 10.6 (94 k/s) | — | — |

## Reading

- **Our clip is SOTA** on every platform — fastest of the three on CPU (8.7 µs, beats voro++ 9.5 and
  geogram 10.6) and ~1.8× the Liu-2020 clip kernel on GPU.
- **Serial CPU full build: 0.89× voro++** (67.6 vs 75.7 k/s). The deficit is **entirely the gather**
  (ours 6.3 vs voro++ 3.8 µs/cell, 1.66×) — voro++'s coarse-block container + precomputed sub-position
  worklist + whole-block acceptance is a more efficient neighbour search than our fine-grid sorted-offset
  walk (which at the optimal `dens=0.3` walks ~217 mostly-empty blocks). Porting whole-block-accept alone
  onto our fine grid is a *net loss* (the per-block distance arithmetic costs more than it saves without
  the worklist); matching voro++ would need its full coarse-block+worklist structure (no cell-data-
  structure change — the clip stays as-is). **The win over voro++ is multicore (~18×) and GPU (~92×).**
- **GPU: ours beats the SOTA full build (1.13× at 1 M) and clip kernel (1.8×).** The branchless inner
  gather (per-block ±L periodic shift) helps the GPU; whole-block-accept's branches would diverge, so the
  CPU and GPU use different gathers (`#ifdef __CUDA_ARCH__`).
- **Scaling:** CPU throughput is flat in N (memory-bandwidth-saturated above ~100 k); the GPU climbs with
  N (1.45 → 6.99) as occupancy fills, needing ≥~300 k to near its ceiling. 24-core is bandwidth-bound
  (~1.4 M/s, well under the ~2.75 M/s clip-only ceiling).

**Headline:** our clip is SOTA everywhere; on GPU the full cold build now beats the Liu-2020 SOTA; on serial
CPU we trail voro++ by 0.89× purely on neighbour-search efficiency (structural — voro++'s worklist), while
winning decisively on multicore and GPU.

## Reproduce

```bash
cd vorflow
NS="10000 100000 1000000"
OMP_NUM_THREADS=1  CC_DENS=0.3 ./build/host-openmp/tests/kokkos/bench_convexcell      $NS  # serial + voro++
OMP_NUM_THREADS=48 CC_NOVORO=1 CC_DENS=0.3 ./build/host-openmp/tests/kokkos/bench_convexcell $NS  # 24-core
CC_NOVORO=1 CC_DENS=0.5 ./build/nvidia-cuda/tests/kokkos/bench_convexcell_f32          $NS  # GPU FP32
./build/nvidia-cuda/tests/kokkos/bench_construct_f32                                        # construct-only
extern_bench/build.sh && OMP_NUM_THREADS=1 extern_bench/bench_geogram 1000000               # clip: all 3
# SOTA (extern_bench/voro_gpu; see voro_gpu_bench.md for build + timing patch):
#   ./build/bin/VolumeVoronoiGPU box.tet box_sites_<N>.xyz 1 | grep '@@'
```
`bench_convexcell` reports `convexcell` (unsorted) and `spatial-sort` (Morton, the headline) k/s plus the
timed voro++ reference; `CC_NOVORO=1` skips voro++; `CC_DENS` sets the grid density (CPU optimum 0.3, GPU
0.5).
