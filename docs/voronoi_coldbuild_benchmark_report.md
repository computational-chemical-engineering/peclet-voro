# Cold Voronoi build — benchmark report (serial / multicore / GPU / MPI + external)

**Date:** 2026-06-27 · **GPU:** RTX 5080 (sm_120), CUDA 13.2 · **CPU:** AMD Threadripper PRO 5965WX
(24c / 48t, 1 NUMA, AVX2, all-core 4.54 GHz) · build: `host-openmp` (FP64) / `nvidia-cuda` (FP32).

Cold (from-scratch) Voronoi build of N uniform random sites in a periodic unit cube: neighbour **gather +
clip**. "Ours" = the `bench_convexcell` ConvexCell path with the worklist gather (default both backends).
CPU/serial FP64; GPU FP32 (matching each external reference's precision).

> **Measurement caveat.** During this run the machine carried **concurrent external load** — a second session
> running `test_amr_distri` (~4 of 48 cores at 100%, intermittent). It depresses **serial / low-thread**
> points the most (memory-subsystem contention): the unpinned 1-thread cold build read 52 k/s vs **72.9 k/s
> when pinned to a free core** (`taskset`), and the multi-thread sweep was erratic (e.g. 12t=631 but 36t=1873
> k/s). Where a point was contention-sensitive, the table below uses the **pinned / clean** value (and the
> clean multicore curve measured earlier this session on a quiet machine). GPU and MPI (≥2 ranks) are
> largely unaffected (they out-mass the 4 external cores or run on the GPU).

---

## 1. Serial CPU (1 core, FP64, cold build)

| N | ours (worklist) k/s | voro++ k/s | ours / voro++ |
|------:|------:|------:|------:|
| 10 k  | 74.2 | 70.6 | 1.05× |
| 100 k | 74.8 | 75.2 | 0.99× |
| 1 M   | 75.4 (pinned 72.9) | 75.4 | ~1.0× |

**Serial CPU is at voro++ parity** (≈1.0×), machine-exact (Σvol/box err ~1e-14). voro++ (Rycroft, single-
threaded library) is the standard serial reference. Throughput is flat in N (memory-bandwidth-saturated
above ~100 k).

## 2. Multicore CPU (FP64, N=1 M, worklist) — clean curve

| cores | k/s | speedup | efficiency |
|---:|---:|---:|---:|
| 1  | 75.3 | 1.0× | 100% |
| 12 | 899.9 | 12.0× | **99.6%** |
| 24 | 1291 | 17.1× | 71% |
| 48 (SMT) | 1656 | 22.0× | — |

**Embarrassingly parallel and near-linear to ~12 cores (99.6%)**, then rolls off — not clock throttling
(cores hold 4.54 GHz at 1 *and* 48 threads) and not byte-bandwidth (FP32 ≡ FP64); the 12→24 rolloff is the
shared **uncore** (L3 / Infinity-Fabric / memory controllers) saturating. SMT adds ~28%. Count **cores, not
the 48 threads**: it's 17× on 24 physical cores. Peak ≈ **1.66 Mcell/s**.

## 3. GPU (FP32, dens≈1.0, S=4, worklist) — measured today

| N | ours GPU Mcell/s |
|------:|------:|
| 10 k  | 1.64 |
| 100 k | 5.06 |
| 1 M   | **7.91** |

Machine-exact for FP32 (Σvol/box err 3.1e-8), climbs with N as occupancy fills.

## 4. Comparison with external codes

**Full cold build, Mcell/s:**

| | ours **serial** (1 core, FP64) | **voro++** serial | ours **24-core** | ours **48-thr** | ours **GPU** (FP32) | **Liu-2020 GPU SOTA** (FP32) |
|---|---:|---:|---:|---:|---:|---:|
| 1 M | 0.075 | 0.075 | 1.29 | 1.66 | **7.91** | 6.16 |

- **Serial CPU = voro++ parity.**
- **GPU = 1.28× the Liu-2020 SOTA** (7.91 vs 6.16 Mcell/s full build, same RTX 5080).
  *Note:* the SOTA binary (`extern_bench/voro_gpu`) **errors on re-run today** ("Unable to execute kernel" on
  Blackwell/CUDA 13); its 6.16 / construct 9.5 / gather 17.5 Mcell/s are the previously-recorded values from
  the validated run on this same GPU (`docs/voronoi_construct_ledger.md`).

**Clip-only kernel head-to-head** (construct from cached neighbours, 1 M, FP64, µs/cell — clean values from
`voronoi_cold_tessellation_benchmark.md`; today's contended re-run read ours 72.4 / voro++ 79 / geogram 65
k/s, all depressed proportionally):

| | clip µs/cell | kcells/s |
|---|---:|---:|
| **ours** (ConvexCell) | **8.7** | **115** |
| voro++ (`voronoicell::plane`) | 9.5 | 105 |
| geogram (`clip_by_plane`) | 10.6 | 94 |

**Our clip kernel is the fastest of the three on CPU**, and ~1.8× the Liu-2020 clip on GPU. The serial full-
build parity (not a win) is the *gather*, where voro++'s coarse-block container slightly edges our worklist.

---

## 5. MPI — distributed cold build

**There was no distributed cold-build benchmark, so one was written**: `tests/kokkos_mpi/bench_voronoi_mpi.cpp`
(the timing sibling of the validated `test_voronoi_mpi`). Each rank owns a block (core ORB via
`VoronoiHalo`), gathers ghost seeds within `rcut` (the MPI halo exchange), and tessellates its owned+ghost
subset with the **device tessellator** (`vor::device::buildTessellation`). One MPI rank per core
(`OMP_NUM_THREADS=1`, `--bind-to core`).

> **Important:** the distributed path uses the device tessellator's **own grid gather** and emits **full
> facet/neighbour connectivity** — a heavier, different kernel than `bench_convexcell`'s volume-only worklist.
> So its per-core rate (~30–63 kcell/s) is **not** comparable to §1–2; it measures the distributed path as it
> stands. Correctness is exact at every np: **Σ owned volumes = 1.000000, owned cells partition the global
> set exactly** (totOwned = N).

### 5a. Strong scaling (fixed N = 1 M)

| np | aggregate Mcell/s | per-core kcell/s | build s | gather ms | ghost frac |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.048 | 48.5 | 20.6 | 42 | 0% |
| 2  | 0.076 | 38.2 | 13.1 | 39 | ~35% |
| 4  | 0.122 | 30.6 | 8.2 | 40 | — |
| 8  | 0.203 | 25.4 | 4.9 | 38 | — |
| 16 | 0.337 | 21.1 | 3.0 | 32 | — |
| 24 | **0.403** | 16.8 | 2.5 | 32 | high |
| 48 | FAILED (oversubscribe/crash under the concurrent external load) | | | | |

**8.3× on 24 ranks (35 % parallel efficiency).** Per-core falls 48.5 → 16.8 kcell/s. The drop is dominated by
**ghost-cell redundancy**: as blocks shrink, the `rcut` halo grows relative to owned volume (surface/volume),
so each rank rebuilds a growing fraction of cells its neighbours also build (np=2 already 35 % ghost). The
**MPI communication is cheap** — the gather (halo exchange) is 32–42 ms against multi-second builds.

### 5b. Weak scaling (~200 k cells/rank — the "MPI process per core" question)

| np | N | per-core kcell/s | aggregate Mcell/s | build s | gather ms |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.2 M | 63.0 | 0.063 | 3.2 | 6 |
| 2  | 0.4 M | 45.5 | 0.091 | 4.4 | 14 |
| 4  | 0.8 M | 38.0 | 0.152 | 5.3 | 26 |
| 8  | 1.6 M | 34.1 | 0.273 | 5.9 | 46 |
| 16 | 3.2 M | 32.6 | 0.521 | 6.1 | 74 |
| 24 | 4.8 M | **29.3** | 0.704 | 6.8 | 117 |

**Per-MPI-process-per-core throughput holds far better under weak scaling: 63 → ~30 kcell/s, and levels off
around 30** (np 8→24: 34 → 33 → 29). The remaining ~2× drop from 1→many ranks is the **same shared-memory /
uncore saturation** that caps the OpenMP multicore run (§2) — not communication (gather stays ≤117 ms), and
not the algorithm (it's embarrassingly parallel). So: **a busy MPI rank delivers ≈ 30 kcell/s on its core
once the memory subsystem is saturated**, vs ≈ 54 kcell/s/core for OpenMP threads at 24 cores — the gap being
the heavier distributed tessellator kernel, not MPI overhead.

---

## Update — worklist port + ghost-skip (2026-06-27)

The two distributed levers identified below were applied.

1. **Worklist gather ported into `buildTessellation`.** The per-sub-position rmin worklist (proven in
   `bench_convexcell`) replaced the adaptive shell-offset walk; the serial path now clips on the fly (no
   candidate buffer). Serial cold build **48.8 → 66.7 kcell/s (1.37×)** at N=200k, single-thread — matching
   the standalone worklist's 1.31–1.32× win. test_tessellator / test_voronoi_mpi / the FP32 guard all PASS.
2. **Ghost cells are no longer built** (`nBuild` = nOwned). Profiling showed the grid is negligible (~0.2%)
   and ~25% of the build was tessellating ghost cells that are then discarded — ghosts are only needed as
   cutting candidates. Skipping them (still using every seed as a candidate) removes that cost; the bench
   ghost radius was also tightened 5·sp → 3·sp (the worklist closes Poisson cells by ~2.6·sp).

Result (weak, 100k owned/rank, host-OpenMP, 1 thread/rank): per-core is now **~83 kcell/s and FLAT** across
np=1,2,4 (82.9 / 83.1 / 81.2) — the ghost-bound strong-scaling penalty is gone. np=2 N=200k: the old
**42.8 → 83.1 kcell/s = 1.94×**. (The "local-density grid" lever was dropped: the grid is already ~0.2% of
the build, so it buys nothing.)

**GPU note (worklist is host-only).** **[SUPERSEDED — retiring the half-edge ScratchCell for the compact
ConvexCell removed the occupancy penalty, so the worklist gather now runs on the GPU too and is the default on
BOTH backends; see `voronoi_cold_tessellation_benchmark.md`.]** An A/B on the GPU (RTX 5080, CUDA 13.2, FP64) found the worklist gather
REGRESSES the production force-geometry path: pure tessellation sped up (geom-off 2.5 → 3.8 Mcell/s) but the
geom-on build dropped ~15–35%. The device cut is fused with the register/occupancy-bound geometry kernel
(cells are register-resident — the geometry can't be split into a second pass), so the worklist's heavier
gather steals occupancy from geometry, a pure-tess win the geom path can't use. The GPU therefore keeps the
legacy expanding-shell gather (geom-on 2.12 Mcell/s, bit-identical to pre-port — zero regression); the
worklist is applied on the CPU only. `nBuild` + the tighter `rcut` are backend-agnostic and apply everywhere.

## 6. Findings & summary

| platform | cold-build throughput (1 M) | vs reference |
|---|---:|---|
| **Serial CPU** (1 core) | 0.075 Mcell/s | = voro++ (parity) |
| **Multicore CPU** (48 thr) | 1.66 Mcell/s | 22× serial; uncore-bound past ~12 cores |
| **GPU** (FP32) | 7.91 Mcell/s | **1.28× Liu-2020 SOTA** |
| **MPI distributed** (24 ranks, strong) | 0.40 Mcell/s | correct; 35 % strong-scaling eff |
| **MPI distributed** (weak) | 0.70 Mcell/s @ 24 ranks | ~30 kcell/s/core, levels off |

1. **Single-process is solved**: serial = voro++ parity, multicore near-linear to ~12 cores, GPU beats the
   SOTA (1.28×). All machine-exact.
2. **The distributed path is correct but under-optimised.** It scales (8.3× strong / clean weak per-core
   plateau) and communication is cheap, but two things hold it back, both fixable:
   - **[SUPERSEDED — the worklist gather is now the default on both backends.]** it uses the **device
     tessellator's older grid gather**, not the worklist — porting the worklist gather
     into `buildTessellation` should lift the per-core rate toward the §1–2 numbers;
   - **strong-scaling ghost overhead** (surface/volume) — expected; weak scaling shows the real per-core
     story (~30 kcell/s, memory-bound).
3. **MPI overhead is not the bottleneck** — the halo gather is 1–2 orders of magnitude below the build time;
   the per-core ceiling is the shared memory subsystem (same uncore wall as OpenMP) plus the heavier
   connectivity-emitting kernel.

### Reproduce
```bash
cd vorflow
# serial + voro++, multicore, clip head-to-head (pin away from any background load):
OMP_NUM_THREADS=1 taskset -c 44 CC_DENS=0.3 build/host-openmp/tests/kokkos/bench_convexcell 1000000
OMP_NUM_THREADS=24 CC_DENS=0.3 CC_NOVORO=1 build/host-openmp/tests/kokkos/bench_convexcell 1000000
OMP_NUM_THREADS=1 extern_bench/bench_geogram 1000000
# GPU FP32:
CC_DENS=1.0 CC_NOVORO=1 build/nvidia-cuda/tests/kokkos/bench_convexcell_f32 1000000
# distributed MPI (built via the standalone kokkos_mpi project):
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"
cmake --build build_kmpi -j --target bench_voronoi_mpi
OMP_NUM_THREADS=1 mpirun -np 24 --bind-to core build_kmpi/bench_voronoi_mpi 1000000   # strong
OMP_NUM_THREADS=1 mpirun -np 24 --bind-to core build_kmpi/bench_voronoi_mpi 4800000   # weak (~200k/rank)
```
