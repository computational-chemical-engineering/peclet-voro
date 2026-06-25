# Vertex-local, sort-free ConvexCell geometry — findings

**Date:** 2026-06-25 · **Hardware:** RTX 5080 (sm_120), CUDA 13.2 · **Author:** overnight run

## TL;DR

Your design note's vertex-local geometry **works, is correct to machine precision, and is a large
speedup.** It replaces the per-face *gather + angular sort* (`atan2`) with a single **per-vertex scatter**
using the divergence theorem and the plane "foot" points — no vertex ordering, no adjacency, no `atan2`,
no `findSharing`.

- **Correct:** all 7 acceptance criteria pass at **machine precision** (≤ 6e-13 relative), including the
  anisotropic/obtuse batch with feet outside their faces.
- **Fast (RTX 5080, FP32, N=1M re-eval):** the volume goes from **65 → 166 M/s (2.57×)** vs the
  `atan2`-sort, and ~2.4× the adjacency-walk (69) — nearly the 194 M/s "no-volume" ceiling.
- **Knock-on for Part II:** re-eval over resident topology jumps from **4.9× → 11.4× full rebuild**; the
  near-miss-skin per-step local repair from **3.5× → 6.1×**; full physics geometry (volume **+** face
  areas, from which forces follow) runs at **119 M/s ≈ 8× rebuild**.

A process note: your extended explanation arrived mid-session (the design note), so I implemented it
exactly as specified rather than interpolating. Earlier in the session I had wrongly concluded the
ordering was intrinsic and only removable with stored adjacency — the note's **foot-point construction**
is the piece I was missing, and it makes the per-vertex contribution genuinely local. You were right.

---

## 1. What was implemented

In `include/vorflow/device/convex_cell.hpp`:

- `planeN(k, n)` — the plane's non-unit normal `n` with interior `{x : n·x ≤ n·n}`, so `x = n` is the
  foot of the perpendicular from the origin (the facet point). Our cell stores `(pn, pd)` as
  `{pn·x ≤ pd}`, so `n = (pd/|pn|²)·pn`. For a Voronoi bisector to a neighbour at `r`, this is `n = ½r`
  (the midpoint), as in the note.
- `volumePerVertex()` — `V = (1/6) Σ_vertices [ det3(n1−n2, f12, v) + det3(n2−n3, f23, v) +
  det3(n3−n1, f31, v) ]`, with edge feet `f = v − (v·c/c·c)c` (`c_k` = the cross product of the other two
  normals = edge direction). A canonical `D = det[n1,n2,n3] > 0` swap fixes all signs; after it no sign
  tracking is needed.
- `facetAreasPerVertex(area)` — each vertex scatters into its 3 incident facets:
  `area[k_i] += det3(n_i, f_{i,i+1} − f_{i-1,i}, v) / (2|n_i|)`. The canonical swap also swaps the plane
  *indices* so areas land on the right facet.
- Helpers `xprod`, `dot3`, `det3`, `edgeFoot`.

The vertex `v` is read from the cached `vx/vy/vz` (already produced by construction / `reevalGeometry`),
so the per-vertex kernel adds only cross/dot/det/divide — branchless except the canonical swap, one
`sqrt` per facet for the areas, none for the volume.

`test_pervertex_geometry.cpp` runs the acceptance criteria in FP64.

---

## 2. Correctness — all acceptance criteria pass at machine precision

FP64, 60 000 cells per batch. **Isotropic** = uniform random sites; **obtuse** = z-squashed (×0.35)
sites that produce skinny cells whose facet/edge feet fall *outside* their faces.

| criterion | isotropic | obtuse |
|---|---:|---:|
| (1) `volumePerVertex` vs ordered `volume()` (rel) | **3.8e-15** | **4.1e-15** |
| (3) facet areas vs ordered polygon area (rel, non-degenerate) | 1.1e-13 | 5.8e-13 |
| (4) divergence identity `V == ⅓ Σ_f |n_f| A_f` (rel) | **1.0e-15** | **2.5e-15** |
| (5) label invariance (permute a vertex's 3 planes) (rel) | 5.9e-16 | 7.3e-16 |
| (6) gradient: FD `dV/dh` vs analytic `A_f` within 1e-4 | **100%** of cells | **100%** |
| (7) **force** `dV=∂V/∂r_k` (per-vertex) vs oracle `facetGeometry.dv` (rel) | **9.9e-14** | **2.7e-13** |

Notes:
- (4) is the strongest internal check — volume and areas, computed independently, are mutually consistent
  to machine precision, with **no external reference**.
- (1)/(4)/(5) hold equally on the **obtuse** batch: the signed sum cancels the out-of-face contributions
  exactly, as the note states. Geometric predicates need not be globally consistent.
- (6): the residual non-passing cells (zero of them at the 1e-4 bar here) would be those sitting *on* a
  combinatorial event, where the gradient is one-sided — expected and harmless for gradient drivers.
- (3): the relative metric is reported only on non-degenerate faces; on tiny faces it blows up purely
  because the denominator → 0 (the *absolute* area error is ≤ ~8e-13 of the cell scale — see the test's
  `max abs/scale` line). The areas are correct.

---

## 3. Performance (RTX 5080, FP32, N=1M, re-eval over resident topology)

### Volume method comparison (full re-eval throughput)

| volume method | Mc/s | vs `atan2` |
|---|---:|---:|
| `atan2` face-sort (was the default) | 65.4 | 1.00× |
| pseudo-angle (transcendental-free, same sort) | 65.4 | 1.00× |
| order-free walk, **no** stored adjacency (`findSharing` hops) | 45.7 | 0.70× |
| order-free walk, **stored** adjacency (O(1) hops) | 69.1 | 1.06× |
| **`volumePerVertex` (this note)** | **166.2** | **2.54×** |
| (reference: re-eval with volume skipped) | 194.2 | — |

The per-vertex volume is **near the no-volume ceiling** — the geometry is almost free. It is `O(nt)` pure
arithmetic vs the sort's `O(np·nt)` gather-and-order, which is why it wins by 2.5× and the adjacency walk
(also order-free but with per-hop branch/divergence and the extra adjacency read) only reaches 1.06×.

### Part II impact (per-vertex volume as the re-eval default)

| quantity | with `atan2` volume | with `volumePerVertex` |
|---|---:|---:|
| re-eval (compact topology) | 65 M/s, 4.9× rebuild | **167 M/s, 11.4× rebuild** |
| full physics geometry (volume **+** facet areas) | — | **119 M/s, ~8× rebuild** |
| Phase-1.5b near-miss-skin per-step local repair (δ=0.1%·spacing) | 47 M/s, 3.5× | **90 M/s, 6.1×** |

(full rebuild ≈ 14.7 M/s; both re-eval and rebuild use `volumePerVertex` in these rows, so the speedup is
honest, not inflated by the volume method.)

---

## 4. Why it matters / what it unlocks

- **Forces — done and validated.** `facetMomentsPerVertex` scatters each facet's area AND first moment
  `∫x dA` (note §5) in the same per-vertex pass; the area-weighted facet centroid is `c_k =
  moment_k/area_k`, and the volume gradient (force) is `dV_k = ∂V/∂r_k = (area_k/|r_k|)(r_k − c_k)`. This
  reproduces the oracle-validated `facetGeometry.dv` to **machine precision** (criterion 7: 1e-13). So the
  G2 tier — area vectors + `dV` for forces/momentum — is now **fully order-free and vertex-local**, no
  `faceOrdered`/`atan2` anywhere. (Equivalently, the §6 analytic form `dV=Σ A_f δh_f` with `dh_f/dp` from
  §1 gives the same thing; both are now available.)
- **Differentiability.** The kernel is dot/cross/det/divide only → templating the scalar type gives exact
  forward-mode autodiff (your route 1), or use the analytic `Σ A_f δh_f` (route 2). Either gives the
  semidiscrete-OT / CVT gradient the drivers need.
- **Power/regular cells.** The note's `n = ((|r|²+w0−wj)/(2|r|²)) r` plugs straight into `planeN`'s role;
  the rest of the kernel is unchanged. Not wired up (no weights in the pipeline yet) but trivial to add.

---

## 5. Follow-ups — all DONE in this session

- **Cold construct switched to per-vertex.** `bench_construct` G1+G2 now use `volumePerVertex` /
  `geometryPerVertex` (no `facetGeometry`/`atan2` in the construct hot path). Cold-construct ceiling:
  **G1 14.0 → 17.0 (+22%), G2 12.0 → 16.1 M/s (+34%)** — the volume tier overhead dropped from +19%→+3%
  and the derivative tier from +16%→+5%. The fused grid build (`bench_convexcell`) rose 5.5 → 6.0.
- **Merged single-pass V+A kernel:** `ConvexCell::geometryPerVertex(vol, area, mx,my,mz)` does volume +
  areas + first moments in one vertex pass (shares `n,c,D,v,feet`); validated bit-for-bit vs the separate
  calls (criterion 8: 1.3e-15). This is the production G1+G2 kernel; forces derive per-facet as
  `dV_k=(area_k/|r_k|)(r_k − moment_k/area_k)`.
- **`facetGeometry`/`faceOrdered` retired from the hot path.** They remain only as the FP64 reference/
  oracle in the tests; no production path calls them.
- **`adjT` field + `buildAdjacency`/`volumeAdj`/`volumeWalk` removed** (the superseded adjacency-walk
  experiment). Cell back to 3084 B; this also sped the full-cell re-eval read (44 → 54 M/s).

Remaining caveat: **simple vertices assumed** (exactly 3 planes/vertex). Our clipper produces 3-plane
dual triangles, so this holds; a 4-fold-degenerate vertex would need splitting (the note's guard). Not
observed in 120 000 test cells — worth a production assert.

---

## 6. Files

- `include/vorflow/device/convex_cell.hpp` — `planeN`, `volumePerVertex`, `facetAreasPerVertex`,
  `facetMomentsPerVertex` (area + first moment → centroid → force `dV`), helpers; also
  `volumeWalk`/`volumeAdj`/`buildAdjacency` from the prior order-free investigation, kept as documented
  variants.
- `tests/kokkos/test_pervertex_geometry.cpp` — FP64 acceptance criteria (ctest `test_pervertex_geometry`).
- `tests/kokkos/bench_incremental.cpp` — re-eval decomposition incl. `pervertex(V)` / `pervertex(V+A)`;
  re-eval paths now use `volumePerVertex`.
- Commits: `a4e2e1c` (per-vertex geometry + acceptance test). Bench/report updates follow this doc.

## 7. Recommendation

Adopt `volumePerVertex` + `facetAreasPerVertex` as the geometry tier (G1 + the force-bearing areas) for
both re-eval and the cold construct, and rebuild the G2/`facetGeometry` (`dV`/forces) on top of the
per-vertex areas via the analytic shape derivative. This is the single biggest geometry-tier win found in
the whole investigation, it is exact, and it is exactly the design you specified.
