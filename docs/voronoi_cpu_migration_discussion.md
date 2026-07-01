# GPU → multicore-CPU migration of the ConvexCell Voronoi engine — discussion starting point

Purpose: a self-contained brief so a fresh session can discuss **adapting the GPU per-vertex
ConvexCell Voronoi engine to run efficiently on (multicore) CPU** without re-deriving anything.
The user has **one specific technical item** to raise (TBD at session start); this doc is the shared
context, not a plan.

---

## 1. What exists today (the thing to migrate)

A GPU Voronoi/power-tessellation engine for physics (volumes, face areas, force derivatives `dV`),
header-only C++ + Kokkos, in `voro/include/peclet/voro/`. Two phases:

**Cold build (Part I).** `ConvexCell<Real,MAXP,MAXT>` (`convex_cell.hpp`) — dual-triangle cell:
each vertex = a triple of plane indices, planes = neighbour bisectors. Built by clipping the K≈64
nearest neighbours (security-radius early-out), horizon retriangulation via an O(n) `findSharing`
scan (no stored adjacency — fastest on GPU). Neighbour gather = ArborX BVH kNN or a counting-sort
grid. Status: **at SOTA** — our construct ≈ 14–17 M/s on an RTX 5080, matched/beat a real SOTA GPU
code (Liu et al. 2020) built and run on the same hardware. Cold-build full number is gather-limited
(~6.7 M/s, inherent ~70 distance-tests/cell). See `voronoi_build_plan.md`, `voronoi_construct_ledger.md`.

**Per-vertex geometry (the key kernel).** `volumePerVertex` / `facetAreasPerVertex` /
`facetMomentsPerVertex` / `geometryPerVertex` (merged) — the **vertex-local, sort-free** geometry from
the user's design note (Ray/Sokolov/Lefebvre/Lévy TOG 2018; geogram). Plane stored as a non-unit
normal `n` with interior `{n·x ≤ n·n}` so `x=n` is the facet foot; volume by the divergence theorem
coning each boundary flag `(0, n_i, f, v)` to the origin — **no ordering, no adjacency, no atan2, no
findSharing**, a pure per-vertex scatter. Forces follow from the areas (`dV = Σ_f A_f δh_f`). The
kernel is dot/cross/det/divide → trivially SIMD-able and autodiff-able.

**Part II (moving points).** Reuse resident topology when seeds move a little: `reevalGeometry` recomputes
planes+vertices over the stored topology (compact 712 B/cell: `pnbr` + packed triangles), then
`geometryPerVertex`. A **near-miss skin** (precomputed per-cell watch list) gives a cheap per-step
needs-reclip flag → local re-clip only the flipped cells; **Verlet full rebuild** when the displacement
budget is spent.

### Validated (FP64, `test_pervertex_geometry`, 60k cells × isotropic + obtuse) — machine precision
volume vs ordered 3.8e-15 · divergence identity `V=⅓Σ|n|A` 1.0e-15 · areas 1e-13 · label-invariance
6e-16 · force `dV` vs oracle 1e-13 · gradient FD 100% · merged kernel 1.3e-15. All 8 criteria pass.

### Performance (RTX 5080, FP32, N=1M)
- Cold construct ceiling: **G1 17.0 / G2 16.1 M/s** (per-vertex; was 14/12 with atan2-sort).
- Re-eval over resident topology: **167 M/s = 11.4× full rebuild**.
- Full physics geometry (volume + areas → forces): **~120 M/s ≈ 8× rebuild**.
- Near-miss-skin per-step local repair: **90 M/s = 6.1× rebuild** at realistic (0.1%-spacing) displacement.

### Finalized vs open (the "is the GPU impl finalized?" precondition)
- **Finalized:** the geometry kernel (machine-exact, all criteria), the cold construct, re-eval + skin.
- **Open (GPU side):** production assert for simple vertices (exactly 3 planes/vertex — holds in 120k
  test cells but unguarded); power-cell `planeN` variant (`n=((|r|²+w0−wj)/2|r|²)r`) when weights land;
  wiring into an actual time-stepping driver (re-eval + skin + Verlet + displacement accumulator).

---

## 2. Why CPU, and what "migrate" really means

It's **already Kokkos** and already runs on the OpenMP backend — all benches here were also run on
`host-openmp`. So migration = **make it efficient on multicore CPU**, not port it. The CPU build dir is
`build/host-openmp`. Reference measured CPU numbers (FP64, geogram head-to-head): our cold construct ≈
82 k cells/s/core (single thread), geogram's actual `ConvexCell` ≈ 95 (we're 0.87×); at 48 threads both
~2.75 M/s (memory-bandwidth-saturated). So per-core CPU construct is ~150× slower than the GPU per-core —
the CPU win has to come from SIMD + threads + the Part II topology reuse, not raw per-cell speed.

---

## 3. The technical axes likely in play (framing, not decisions)

1. **SIMD strategy** — the per-vertex geometry is branchless arithmetic, ideal for vectorization. Two
   axes: *cells-as-lanes* (SoA, W cells per AVX vector, each lane a scalar cell — the usual CPU-Voronoi
   answer) vs *within-cell* (vectorize across a cell's vertices/planes). Cells-as-lanes composes cleanly
   with the per-vertex scatter; within-cell has variable trip counts (np, nt differ per cell).
2. **Clip: findSharing vs stored adjacency.** GPU prefers the flat O(n) `findSharing` (divergence kills
   the adjacency walk, measured 0.46×). On CPU the geogram head-to-head showed the **adjacency walk is
   ~1.15× faster per core** (branch prediction, cache, no divergence). So the CPU build may keep the
   ConvexCell but use adjacency where the GPU uses findSharing.
3. **Precision / SIMD width.** GPU forced FP32 (FP64 1/64). On CPU FP64 is ~free, but FP32 doubles SIMD
   width (16-wide AVX-512). FP32-vs-FP64 becomes a throughput-vs-simplicity choice, not forced.
4. **Cache & threading.** Spatial (Morton/grid) ordering of cells for L1/L2 reuse and coalesced-ish
   neighbour reads; Kokkos `RangePolicy` over cells; the 3 KB cell fits L1 fine (no GPU local-memory
   penalty), so a richer representation is affordable if useful.
5. **Part II carries over** unchanged and matters *more* on CPU (per-cell work is relatively dearer →
   topology reuse + skin repair pay even better). The Verlet driver is shared.

---

## 4. Key files / entry points
- `include/peclet/voro/convex_cell.hpp` — the cell + clip + `*PerVertex` geometry (planeN,
  volumePerVertex, facetAreasPerVertex, facetMomentsPerVertex, geometryPerVertex; ordered `volume()`/
  `facetGeometry()`/`faceOrdered` kept only as the FP64 oracle).
- `tests/kokkos/test_pervertex_geometry.cpp` — the 8 acceptance criteria (FP64).
- `tests/kokkos/bench_incremental.cpp` — re-eval / skin / Verlet decomposition + per-vertex timing.
- `tests/kokkos/bench_construct.cpp` — cold construct ceiling (G0/G1/G2).
- `extern_bench/` — geogram CPU head-to-head harness (`build.sh`), Liu-2020 GPU SOTA repro.
- Docs: `voronoi_pervertex_geometry_report.md` (the geometry win), `voronoi_build_plan.md` (Part I/II +
  all studies), `voronoi_construct_ledger.md` (every construct attempt + verdict).
- Build (CPU): `cmake --build build/host-openmp`; (GPU): `build/nvidia-cuda`, needs CUDA 13.2 on PATH.
- Deprecated/disabled: the legacy half-edge CPU voronoi (`voronoi.hpp`) + its tests
  (`test_build_trial/droplet/interface`, `test_sdf_boundary_device`) — being retired *for the engine in
  this doc*; the migration TODO is to re-implement those cases on the new ConvexCell (CPU).
