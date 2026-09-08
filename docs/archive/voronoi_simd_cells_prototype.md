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
- **The reciprocal looked like THE lever — but the obvious fixes don't survive contact (see the section
  below).** Options considered: (1) `-ffast-math`/`-freciprocal-math` on the geometry TUs — relaxes the FP64
  machine-exactness *and* its FP32 gain is mostly reassociation/vectorisation, not the reciprocal; (2) a
  scoped `vrcpps`+Newton fast reciprocal — measured negligible once divides are already low; (3) reformulate
  `edgeFoot` to fold 3 reciprocals → 1 — FP64-exact but **NaN in FP32** (Montgomery product overflows). So
  the real path is the cells-as-lanes SIMD kernel itself, at FP32 accuracy, not a scalar reformulation.
- **The cold-construct gather is a different problem** — latency/transaction-bound; SIMD cells-as-lanes
  won't help it (the per-lane neighbour walks diverge and the loads are scattered). Its lever is the
  worklist (already done) and locality, not vectorisation.

## Reproduce
```bash
cd voro/tests/kokkos
g++ -O3 -march=native            -fopenmp -std=c++20 bench_simd_cells.cpp -o bsc        # exact divide
g++ -O3 -march=native -ffast-math -fopenmp -std=c++20 bench_simd_cells.cpp -o bsc_fast   # fast reciprocal
g++ -O3 -march=native -DNODIV    -fopenmp -std=c++20 bench_simd_cells.cpp -o bsc_nodiv   # divide-cost diagnostic
OMP_NUM_THREADS=1  ./bsc 8192 24 400
OMP_NUM_THREADS=24 OMP_PROC_BIND=spread OMP_PLACES=cores ./bsc_fast 131072 24 60
# args: N cells, nt triangles/cell, reps
```

## ATTEMPTED the 1-divide reformulation in `convex_cell.hpp` — REVERTED (FP32-unsafe)

Option 3 was implemented across `volumePerVertex` and (via a shared `edgeFeet3` helper)
`facetAreasPerVertex` / `facetMomentsPerVertex` / `geometryPerVertex` / `geomVolumeGrad` / `geomVolumeArea`:
the identity `det3(e, edgeFoot(v,ck), v) = (v·ck)(e·(v×ck))/|ck|²` lets the three `1/|ck|²` be folded into
**one divide** over the common denominator `g1·g2·g3` (3 divides/triangle → 1). It was **FP64 machine-exact**
— `test_pervertex_geometry` passed all criteria, `Vpv` 3.8e-15 unchanged — and the synthetic `bench_simd_cells`
(`-DV1DIV`) showed +10–15% scalar.

**But it is NOT FP32-safe and was reverted (commit `a65d430`).** Reducing 3 reciprocals to 1 is a Montgomery
batch inversion: it needs the product `g1·g2·g3` of the three `|c_k|²` (and numerator terms scaled by the
other two `g`). Across cells the per-triangle scale of `|c_k|²` varies enough that those products leave FP32's
narrow range → `bench_convexcell_f32` cold build produced **`Σvol err = NaN`**. The FP64-only acceptance test
stayed exact and hid it (the deferred GPU-FP32 re-validation would have caught it). Since **FP32 is the
GPU/production precision**, the fold is not viable; the divide stays at **3/triangle**.

**The `-ffast-math` numbers below are a ceiling, not a shippable win.** On the synthetic kernel `-ffast-math`
gave FP32 scalar 9.1 / SIMD 22 Mcell/s — but that is a *package* (reassociation + vectorisation + vectorised
reciprocal), not the reciprocal alone: a *scoped* scalar fast-reciprocal (`vrcpps`+Newton) on the already-low-
divide kernel measured **negligible**. And `-ffast-math` relaxes the very FP64 machine-exactness the per-vertex
design exists to provide. So neither the fold nor a scoped fast-reciprocal is the path.

| (synthetic, 1 thread) | FP64 scalar | FP32 scalar |
|---|---:|---:|
| 3-divide (shipped) | 2.0 | 2.0 |
| 1-divide fold (reverted) | 2.2 | 2.3 |
| `-ffast-math` (ceiling, not shippable) | 6.2 | 9.1 |

**What remains true:** the per-vertex geometry kernel is divide-bound, and on the geometry-dominated **Part II
re-eval** path that matters (it's ~invisible in the clip-dominated cold construct). The viable lever is the
genuine **cells-as-lanes SIMD kernel** (FP32, compiler-vectorised reciprocal accepted *within* the FP32 path),
not a scalar reformulation — i.e. the real CPU-migration work, with FP32 (not FP64) accuracy as the bar.

## On the sqrts (area vectors are sqrt-free)
`facetAreasPerVertex` carries a per-facet `sqrt(|n_i|)` because it returns scalar area **magnitudes**. The
area **VECTOR** (direction + magnitude) is sqrt-free: `A_k = S_a·n_k/(2·nn[k])` with `nn[k]=|n_k|²` cached —
a pure reciprocal, no sqrt. `geomVolumeArea` / `geomVolumeGrad` already compute exactly this (the comment in
the header says "fully SQRT-FREE"). So for physics (fluxes, momentum, forces) use the sqrt-free area-vector
kernel; the sqrt only appears when a scalar magnitude is explicitly wanted.

## Caveats (it's a prototype)
- Synthetic cells (random non-degenerate triangles) — measures *arithmetic* throughput, not a real
  tessellation; the per-triangle op mix matches `volumePerVertex` exactly, which is what sets the ceiling.
- Fixed `nt` per cell; real cells vary (~24–28 triangles), so a production kernel pays some masking overhead
  for the `alive`/variable-`nt` tails — modest, doesn't change the divide conclusion.
- Only the **volume** kernel; `facetAreasPerVertex`/`geometryPerVertex` share the same divide-heavy
  edgeFoot pattern, so the same conclusion applies.
- `-ffast-math` here stands in for "a fast reciprocal"; production would scope it to the kernel (option 2/3
  above), not the whole TU.
