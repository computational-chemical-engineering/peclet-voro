# Plan — SDF-defined geometry for the tessellation (device + suite SDF)

> **Status:** the device clip now lives in `include/vorflow/device/sdf.hpp` and operates on the
> **ConvexCell** tessellator (`clipCellAgainstSdf(ConvexCell&, …)`). Device providers
> `SdfSphere`/`SdfBox`/`SdfHollowCylinder` (analytic) and `SdfGrid` (VTI / sampled) port the
> legacy `SignedDistanceBoundary` build. Originally validated to machine precision against the
> half-edge oracle, but the **ConvexCell SDF re-port is still pending re-validation**: the boundary
> device test (`tests/kokkos/test_sdf_boundary_device`) is **currently disabled** while it is
> re-implemented on ConvexCell (see `voronoi_cpu_migration_discussion.md`). The distributed clip
> test (`tests/kokkos_mpi/test_sdf_mpi`) follows the same path. Python exposure (Phase 3) is on the
> device `vorflow` module surface but does not yet expose SDF geometry — a separate binding effort.

**Goal.** Embed a solid geometry defined by a signed-distance field into the Voronoi/power
tessellation: a cell that would extend into the solid is clipped by a plane located at the
**sdf = 0** surface, with the plane **normal taken from the SDF gradient** ∇sdf. Seeds inside the
solid get no cell; cells fully in the fluid are untouched. Suite sign convention: **sdf < 0 inside
solid, sdf > 0 in fluid, ∇sdf points outward (into the fluid).**

## What already exists (don't rebuild)

This mechanism is **already implemented on the legacy CPU path** and tested:

- `SignedDistanceBoundary<real_t>` (`include/vorflow/voronoi.hpp:1211`) — abstract `value(x)`,
  `gradient(x)`, and `closestPoint(x, c, normal)` which projects `x` onto sdf=0:
  `c = x − φ·∇φ/|∇φ|²`, `normal = ∇φ/|∇φ|` — exactly "surface point at sdf=0, normal from the SDF".
- `CellComplex::clipCellAgainstBoundary` (`voronoi.hpp:3905`) — per cell: if the seed has φ ≤ 0
  (inside solid) the cell is emptied; if φ > cell radius the cell is untouched; otherwise it
  iterates (up to `m_boundaryMaxCuts`): find the most-violating vertex, project it to the surface,
  and `clipByPlane(planeVec = −normal, offset, boundaryNbr)` — a plane through the surface point
  with the SDF normal. Iterating lets a **curved** surface be approximated by several planar cuts.
- `tests/test_sdf_boundary.cpp` validates it for slab, spherical-hole, and cylinder boundaries.

The suite also already has the shared SDF the rest of this migration reused: **`tpx::geom`**
(`core/include/tpx/geom/sdf.hpp`) — the `Sdf` concept (`eval(p)`), analytic `Sphere`,
`Box`, `HollowCylinder`, `Complement`, a central-difference `gradient`, plus `GridSdf` (trilinear
sampling) and VTI read/write.

**So the work is not the algorithm — it is (1) porting that clip to the device tessellator (the
migration gap: the Phase-3 device path does Voronoi cuts but no SDF boundary yet), and (2)
unifying on `tpx::geom::Sdf` instead of vorflow's bespoke `SignedDistanceBoundary`, so geometry is
shared with sdflow/dem and can come from analytic shapes or a VTI grid.**

## Design

### A. Device-callable SDF (reuse core)
The cutter needs `sdf(x)` and `∇sdf(x)` inside a `KOKKOS_FUNCTION`. Two interchangeable providers
behind one tiny policy (mirrors the `Weighting` policy):
- **GridSdf on device (primary, universal).** Sample any geometry once into a `tpx::Field3D<float>`
  device view (`tpx::geom::sample(shape, dims, origin, spacing)` → upload), and evaluate in-kernel by
  trilinear interpolation; ∇sdf by central differences on the grid. Works for analytic shapes *and*
  VTI-loaded geometry (`tpx::geom::readVti`), matching how sdflow/dem carry geometry.
- **Analytic SDF (fast path).** `KOKKOS_INLINE_FUNCTION` `eval`/`grad` for sphere/box/cylinder for
  exact, allocation-free boundaries (the analytic `tpx::geom` shapes are trivial to mark device-callable).

### B. Device boundary clip (faithful port of `clipCellAgainstBoundary`)
Add an optional SDF-clip stage to the per-cell build in `device/tessellator.hpp`, after the Voronoi
cuts, operating on the same `ConvexCell`:
1. φ_center = sdf(seed); if φ_center ≤ 0 → mark cell empty (seed in solid), done.
2. if φ_center > cell circumradius + tol → no boundary interaction, done.
3. iterate ≤ `maxCuts`: scan the cell's vertices (in world coords = seed + vertexPos) for the most
   negative φ; if none < −tol, stop; else project that vertex to the surface (closest-point) to get
   `surfacePoint`, `normal`; orient the normal into the solid; `cutCell2(pv = −normal, off =
   pv·(surfacePoint − seed), nbr = kBoundary)` on the scratch cell.
Boundary facets carry the **`kBoundary` sentinel** (negative id, already understood by the published
view and the reciprocal map). This reproduces the legacy result; for a smooth surface the cell gains
a few planar facets approximating the curve.

### C. Wiring + published view
- `buildTessellation` gains an optional SDF provider argument; when present, the clip stage runs and
  cells flag `kEmpty` for in-solid seeds (skipped in the CSR, like buried Power cells).
- The `TessellationView` already represents boundary facets (`facetNbr < 0`); physics consumers
  treat a `kBoundary` facet as a wall (no neighbour term), consistent with the Phase-4 gather force.

### D. Distributed + Python
- **Distributed:** the SDF is read-only geometry replicated on every rank (analytic params, or the
  same VTI grid), so the Phase-6 path needs **no extra exchange** — each rank clips its owned+ghost
  cells against the same field. Seeds inside the solid simply have no owned cell.
- **Python:** expose geometry on the `vorflow` surface (set an analytic shape, or load a VTI) so the
  SDF-bounded tessellation is drivable from Python, matching the suite's geometry-from-Python pattern.

## Phases

1. **Device SDF provider.** `include/vorflow/device/sdf.hpp`: `DeviceGridSdf` (trilinear eval +
   central-diff grad over a `tpx::Field3D`) + analytic `KOKKOS_INLINE_FUNCTION` shapes; host helper
   to sample/upload a `tpx::geom` shape or a VTI. *Accept:* device eval/grad match `tpx::geom` on a
   point cloud to interpolation tolerance.
2. **Device boundary clip.** Port `clipCellAgainstBoundary` into a `ConvexCell` method + a clip
   stage in `buildTessellation` (sentinel `kBoundary`, empty-on-φ≤0). *Accept:* device SDF-clipped
   cells match the legacy `SignedDistanceBoundary` build (volume + boundary-facet count) on the
   `test_sdf_boundary` geometries (slab / sphere-hole / cylinder); space-filling over the fluid
   region; seed-in-solid ⇒ empty.
3. **Distributed + Python.** Replicated SDF across ranks (owned cells == serial with the boundary);
   `vorflow` geometry setters (analytic + VTI). *Accept:* distributed SDF tessellation == single-rank;
   a Python smoke test bounds a packing by a sphere/box.

## Critical files
- Reuse: `core/include/tpx/geom/{sdf,grid_sdf,vti_io}.hpp`, `tpx/common/view.hpp` (`Field3D`).
- Port from: `include/vorflow/voronoi.hpp` (`SignedDistanceBoundary`, `clipCellAgainstBoundary`),
  `tests/test_sdf_boundary.cpp` (golden geometries + tolerances).
- New: `include/vorflow/device/sdf.hpp`, clip stage in `include/vorflow/device/{cell_cutter,tessellator}.hpp`,
  `tests/kokkos/test_sdf_boundary_device.cpp`.

## Verification
Build `-DVORFLOW_KOKKOS=ON`; new ctest diffs device SDF-clipped cells against the legacy
`test_sdf_boundary` reference (slab/sphere/cylinder) for volume, boundary-facet count, and
space-filling of the fluid region; `mpirun` np=1,2,4 confirms distributed == single-rank with the
boundary; Python smoke test bounds a seed set by an analytic shape and a VTI.

## Open design points
- **Curved-surface fidelity:** the iterative multi-plane clip (legacy) approximates a curve by a few
  tangent planes per cell — keep that faithful port, or expose `maxCuts`/tol as accuracy knobs.
- **Grid vs analytic** crossover (resolution vs exactness) — default to GridSdf for generality, analytic
  for exact spheres/boxes.
- **sdf sign source:** use the suite convention (negative-inside) end-to-end; vorflow's legacy
  `SignedDistanceBoundary` uses the same (φ ≤ 0 ⇒ solid), so no flip is needed when moving to `tpx::geom`.
