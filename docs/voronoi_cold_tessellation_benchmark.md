# Cold Voronoi tessellation — throughput vs N (serial / multicore / GPU)

**Date:** 2026-06-25 · **GPU:** RTX 5080 (sm_120), CUDA 13.2 · **CPU:** AMD Threadripper PRO 5965WX
(24 cores / 48 threads)

Cold (from-scratch) Voronoi build — neighbour **gather + clip** — of N uniform random sites, as a
function of N. Our code vs **voro++** (serial reference) and the **SOTA GPU code** (Liu/Ma/Guo/Yan,
"Parallel Computation of 3D Clipped Voronoi Diagrams", TVCG 2020; `extern_bench/voro_gpu`).

Precision is chosen to match each reference: **CPU = FP64** (voro++ is FP64), **GPU = FP32** (the SOTA
GPU code is `float`; FP64 is 1/64 on consumer GPUs). "Ours" is the Morton-sorted path (`bench_convexcell`'s
`spatial-sort`), our representative cold build. Domain: **ours periodic** (`container_periodic`); the SOTA
code is a **clipped, non-periodic** tessellation in a fixed `[0,1000]³` tet domain — comparable as
throughput, different boundary handling.

## Full build (gather + clip), Mcells/s

| N | ours **serial** (1 core, FP64) | **voro++** (serial, FP64) | ours **24-core** (48 thr, FP64) | ours **GPU** (FP32) | **SOTA** Liu GPU, full (FP32) |
|------:|------:|------:|------:|------:|------:|
| 10 k  | 0.047 | 0.072 | 1.55 | 1.12 | — *(SOTA `n>14000` branch only)* |
| 30 k  | 0.049 | 0.074 | 1.65 | 2.71 | 5.27 |
| 100 k | 0.048 | 0.075 | 1.65 | 3.94 | 5.82 |
| 300 k | 0.048 | 0.075 | 1.63 | 5.16 | 6.10 |
| 1 M   | 0.049 | 0.075 | 1.68 | **5.95** | **6.16** |

SOTA "full" = `N / (gather_ms + construct_ms)` (its two phases run as separate kernels; ours is fused).
SOTA phases at 1 M: **gather 17.5 Msites/s**, **construct 9.5 Mcells/s**. The SOTA timing patch lives in
its `n>14000` branch, so 10 k has no comparable number.

## Construct kernel only — clip from cached neighbours (GPU, FP32, 1 M)

The direct apples-to-apples kernel comparison (no gather), `bench_construct_f32` vs SOTA's isolated
construct:

| ours (G1, volume) | ours (G2, + derivatives) | SOTA construct |
|------:|------:|------:|
| **16.9** | 16.0 | **9.5** |

→ our clip kernel is **~1.8× the SOTA clip kernel** (Mcells/s).

## Reading

- **vs voro++ (serial):** voro++ wins single-thread (~1.5×) — a mature gather-optimized serial library,
  and our full build pays a heavy neighbour gather on one core. But voro++ has no parallel mode: on 24
  cores we are **~22×** voro++, and on the GPU **~79×**.
- **vs SOTA (GPU):** our **clip kernel beats it ~1.8×**, yet the **full cold build is gather-bound** and
  ends up *matched* (5.95 vs 6.16 at 1 M, 0.97×). Both codes spend most of the wall clock in the kNN
  gather — the great equalizer (cf. `voronoi_construct_ledger.md`). Below ~100 k the GPU is underutilized
  (crossover with the 24-core CPU ≈ 30–50 k).
- **Scaling:** CPU throughput is flat in N (memory-bandwidth-saturated; 24 physical cores give ~34× over
  serial). The GPU climbs with N (1.1 → 5.95) as occupancy fills, needing ≥~300 k to near its ceiling.

**Headline:** we own the construct kernel (~1.8× SOTA); the full cold build is gather-limited, so it ties
the SOTA on GPU and trails voro++ on a single core — the win over voro++ is multicore (22×) and GPU (79×).
The neighbour gather is the thing to optimize next if the *full* cold build is what matters.

## Reproduce

```bash
cd vorflow
NS="10000 30000 100000 300000 1000000"
# ours: serial (+ voro++), multicore, GPU
OMP_NUM_THREADS=1  ./build/host-openmp/tests/kokkos/bench_convexcell      $NS   # serial FP64 + voro++
OMP_NUM_THREADS=48 CC_NOVORO=1 ./build/host-openmp/tests/kokkos/bench_convexcell $NS   # 24-core FP64
CC_NOVORO=1 ./build/nvidia-cuda/tests/kokkos/bench_convexcell_f32         $NS   # GPU FP32
./build/nvidia-cuda/tests/kokkos/bench_construct_f32                            # construct-only kernel
# SOTA (extern_bench/voro_gpu; see voro_gpu_bench.md for the build + timing patch):
for N in 30000 100000 300000 1000000; do
  python3 -c "import random;L=1000.;random.seed(7);f=open(f'box_sites_$N.xyz','w');f.write('$N\n');[f.write(f'{random.random()*L} {random.random()*L} {random.random()*L}\n') for _ in range($N)]"
  ./build/bin/VolumeVoronoiGPU box.tet box_sites_$N.xyz 1 | grep '@@'
done
```
`bench_convexcell` reports `convexcell` (unsorted) and `spatial-sort` (Morton, the headline) k/s, plus the
timed voro++ reference; `CC_NOVORO=1` skips voro++ for the multicore/GPU sweeps.
