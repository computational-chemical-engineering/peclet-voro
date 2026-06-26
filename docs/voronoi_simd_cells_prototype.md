# SIMD cells-as-lanes prototype — per-vertex geometry kernel (CPU)

**Date:** 2026-06-26 · **CPU:** AMD Threadripper PRO 5965WX (Zen3, AVX2, 24c/48t) · **Compiler:** gcc 14

A bounded prototype answering: *can multicore CPU throughput of the ConvexCell engine be improved with
SIMD cells-as-lanes?* Bench: `tests/kokkos/bench_simd_cells.cpp` — a standalone microbench that faithfully
reproduces the per-triangle arithmetic of `convex_cell.hpp::volumePerVertex` (xprod / dot3 / det3 /
edgeFoot + the `D<0` canonical swap, here a branchless blend), comparing the current **scalar one-cell-at-
a-time** form against **cells-as-lanes** (SoA, one cell per SIMD lane, `#pragma omp simd`). Plane normals
are pre-resolved per triangle (the per-cell `n[t0[t]]` indirection hoisted out — the layout cells-as-lanes
needs, no per-lane gather). Standalone because the bootstrapped Kokkos was built without `KOKKOS_ARCH_AVX2`
(its `simd` falls back to scalar); compiled with `-march=native` for real AVX2.

## Headline finding: the kernel is DIVIDE-BOUND, and that — not SIMD width — caps the speedup

The volume kernel does **3 reciprocals per triangle** (`edgeFoot` = `v − (v·ck/ck·ck) ck`). `vdivpd`/`vdivps`
barely pipeline at vector width, so they serialize the wide ALU. As written, cells-as-lanes gives almost
nothing; remove or cheapen the reciprocal and it delivers **near-ideal lane-width speedup**.

**Single thread, N=8192, nt=24 (Mcell/s, scalar → cells-as-lanes):**

| | scalar | cells-as-lanes | speedup |
|---|---:|---:|---:|
| FP64, exact divide | 2.0 | 2.3 | **1.13×** |
| FP32, exact divide | 2.0 | 2.9 | **1.40×** |
| FP64, no divide *(diagnostic)* | 6.2 | 7.7 | 1.23× |
| FP32, no divide *(diagnostic)* | 8.6 | 23.7 | **2.74×** |
| FP64, fast reciprocal (`-ffast-math`) | 2.2 | 5.9 | **2.67×** |
| FP32, fast reciprocal (`-ffast-math`) | 2.2 | 19.3 | **8.68×** |

- Removing the divide makes even the **scalar** path 3× faster (FP64 2.0 → 6.2) — the reciprocals are ~⅔ of
  the work.
- With a fast reciprocal the SIMD speedup reaches the **theoretical width**: 2.67× (FP64, 4-wide AVX2) and
  **8.68× (FP32, 8-wide)** — i.e. cells-as-lanes is genuinely "trivially SIMD-able" *once the divide is
  cheap*. The exact `vdivpd`/`vdivps` was the whole bottleneck.

## It compounds with cores (compute-bound — unlike the cold-build gather)

**24 threads, N=131072 (Mcell/s):**

| | scalar | cells-as-lanes | speedup |
|---|---:|---:|---:|
| FP64, exact divide | 43.1 | 47.6 | 1.10× |
| FP32, exact divide | 49.2 | 67.1 | 1.36× |
| FP64, fast reciprocal | 49.3 | 75.6 | 1.53× |
| FP32, fast reciprocal | 53.0 | **353.8** | **6.67×** |

**FP32 + fast-reciprocal + cells-as-lanes + 24 threads = 353.8 Mcell/s — 7.2× over scalar-exact (49.2).**
It scales 18.3× from 1→24 cores (19.3 → 353.8), ~76% efficiency: this kernel is **compute-bound** and
resident (no scattered neighbour gather), so SIMD and threads multiply cleanly — the opposite of the cold
construct, which is gather/latency-bound (see the multicore note in `voronoi_cold_tessellation_benchmark.md`:
there FP32 ≡ FP64 and SIMD-width is irrelevant).

## What this means for the CPU migration

- **The geometry / re-eval kernels (Part II, moving points) are the place SIMD pays.** They are
  compute-bound and dominate the moving-point workload (re-eval is 11.4× a rebuild on GPU; on CPU the
  topology-reuse path is where the throughput lives). Adopt **cells-as-lanes SoA + FP32 + fast reciprocal**
  there → ~7× per the matrix above, and it scales across cores.
- **The reciprocal is THE lever.** Three options, increasing effort: (1) compile the geometry kernels with
  `-ffast-math` / `-freciprocal-math -mrecip` (relaxes the FP64 machine-exactness, fine for FP32 physics —
  the GPU already runs FP32); (2) hand-code `vrcpps` + one Newton step for FP32 (near-full precision, no
  global fast-math); (3) **reformulate `edgeFoot`** so the per-edge `1/|ck|²` cancels in the divergence sum
  (algebraic; removes the divide entirely and would speed the *scalar* path 3× too — the highest-value fix).
- **The cold-construct gather is a different problem** — latency/transaction-bound; SIMD cells-as-lanes
  won't help it (the per-lane neighbour walks diverge and the loads are scattered). Its lever is the
  worklist (already done) and locality, not vectorisation.

## Reproduce
```bash
cd vorflow/tests/kokkos
g++ -O3 -march=native            -fopenmp -std=c++20 bench_simd_cells.cpp -o bsc        # exact divide
g++ -O3 -march=native -ffast-math -fopenmp -std=c++20 bench_simd_cells.cpp -o bsc_fast   # fast reciprocal
g++ -O3 -march=native -DNODIV    -fopenmp -std=c++20 bench_simd_cells.cpp -o bsc_nodiv   # divide-cost diagnostic
OMP_NUM_THREADS=1  ./bsc 8192 24 400
OMP_NUM_THREADS=24 OMP_PROC_BIND=spread OMP_PLACES=cores ./bsc_fast 131072 24 60
# args: N cells, nt triangles/cell, reps
```

## Implemented: the 1-divide reformulation in `convex_cell.hpp` (2026-06-27)

Applied option 3 to the real `volumePerVertex` — replaced `edgeFoot` + `det3` with the exact identity
`det3(e, edgeFoot(v,ck), v) = (v·ck)(e·(v×ck))/|ck|²` and folded the three `1/|ck|²` into **one divide**
over the common denominator `g1·g2·g3` (3 divides/triangle → 1). **Machine-exact:** `test_pervertex_geometry`
passes all 8 criteria on both batches (isotropic + obtuse), `Vpv vs ordered` 3.8e-15 / 4.0e-15 — unchanged.

Measured on the isolated kernel (`bench_simd_cells`, build with `-DV1DIV`; 1 thread, scalar = the CPU path):

| | FP64 scalar | FP32 scalar |
|---|---:|---:|
| 3-divide (old) | 2.0 | 2.0 |
| **1-divide (the fix)** | **2.2 (+10%)** | **2.3 (+15%)** |
| 1-divide **+ fast reciprocal** (`-ffast-math`) | **6.2 (3.1×)** | **9.1 (4.5×)** |

The exact fix alone is a modest +10–15% (one true `vdiv` still partly serialises), but it's the **enabler**:
with a single reciprocal left, a fast-reciprocal lands the scalar kernel at **3.1× / 4.5×** — the divide is
no longer the wall. Next step for the full win is a fast reciprocal scoped to the geometry kernels (option 2
`vrcpps`+Newton for FP32, or `-freciprocal-math` on those TUs), now that there is only one per triangle.

**End-to-end in the cold construct it's ~invisible** (`bench_construct` G1−G0: the volume kernel is only ~5%
of the clip-dominated construct, so the change is within run-to-run noise). The win is on the
geometry-dominated **Part II re-eval / moving-points** path, where this kernel is ~100% of the work.

## Caveats (it's a prototype)
- Synthetic cells (random non-degenerate triangles) — measures *arithmetic* throughput, not a real
  tessellation; the per-triangle op mix matches `volumePerVertex` exactly, which is what sets the ceiling.
- Fixed `nt` per cell; real cells vary (~24–28 triangles), so a production kernel pays some masking overhead
  for the `alive`/variable-`nt` tails — modest, doesn't change the divide conclusion.
- Only the **volume** kernel; `facetAreasPerVertex`/`geometryPerVertex` share the same divide-heavy
  edgeFoot pattern, so the same conclusion applies.
- `-ffast-math` here stands in for "a fast reciprocal"; production would scope it to the kernel (option 2/3
  above), not the whole TU.
