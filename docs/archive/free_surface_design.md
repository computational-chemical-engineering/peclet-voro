# Free surfaces from the power diagram — design (Effort 3)

> **Status: design only.** No engine code yet. This document specifies how the free-surface layer
> sits on the **power-cell** machinery built in Efforts 1–2 (`plane_policy.hpp` `Power`, the
> radical-plane forward build + weight-aware gate in `tessellator.hpp`, `chainToDofs<Power>`, the
> dynamic repair path, and the differentiable SDF wall in `sdf.hpp`). It exists so the free-surface
> spec is *optimally connected* to the rest of the code rather than a parallel geometry engine.

## 1. What the free surface is

For a set of balls `B_i = ball(p_i, r_i)` with `r_i = √w_i` (the power weights *are* the squared
radii), the free surface is

    Surface  =  ∂U,   U = ⋃_i B_i
             =  boundary of the weighted alpha-shape
             =  dual of the POWER diagram restricted to U.

So it is **not** new geometry — it is a *reading* of the power tessellation already produced by
Effort 1. The radical plane between `B_i` and `B_j` is exactly the power-cell face plane
(`n_ij = α r`, offset `d_ij = ½(|r|²+w_i−w_j)`), and the surface arc `∂B_i ∩ ∂B_j` is the circle
that plane cuts out of the sphere `∂B_i`. Every primitive below lives on data the power `ConvexCell`
already holds.

**Primitives**
- **vertex (triple point)** `p = ∂B_i ∩ ∂B_j ∩ ∂B_k`, kept iff it lies outside all other balls
  (equivalently, iff the corresponding power-cell dual vertex is real — it is already in the cell's
  triangle list).
- **edge (arc)**: an arc of the circle `C_ij = ∂B_i ∩ ∂B_j`. That circle lies in the `i,j` radical
  plane (the cell face), centred at `p_i + h_ij n̂_ij` with `h_ij = d_ij/|r_ij|` (signed distance
  `p_i`→plane) and radius `ρ_ij = √(r_i² − h_ij²)`. Bounded by two consecutive triple points; shared
  by exactly patches `i` and `j` ⇒ watertight seam.
- **face (patch)**: the spherical polygon on `∂B_i` bounded by its arcs. Outward normal at direction
  `u` is `u`; area `A_i` from a Girard / spherical-excess routine (new — see §3).

**Gate.** Face `(i,j)` contributes an arc iff `h_ij < r_i` (the sphere pokes through the plane) —
a scalar test on quantities the power face already stores. Particle `i` is a surface particle iff
its patch is non-empty, i.e. the signed cap depth `e_i = r_i − (distance from p_i to the nearest
covering configuration)` is `> 0`.

## 2. The seam to Efforts 1–2 (the load-bearing decision)

The free-surface layer is **one post-geometry pass, structurally identical to
`reeval_tessellation.hpp::reevalPublish`** and to `sdf.hpp::addSdfWallForce`: it reads the resident
power geometry, computes a scalar/area functional, and routes its gradient back to the DOFs through
the *existing* chain. It needs **zero** change to the clip, the security gate, or the repair path.

Three responsibilities, nothing more:

1. **Read** the power `ConvexCell` `V_i` (its radical-plane faces `pnbr[k]≥0` with cached
   `n_k, nn_k`, and its dual vertices `vx/vy/vz`).
2. **Compute** the cap area `A_i` (and centroid, energy `γ A_i`) with a new spherical primitive (§3).
3. **Supply** the per-plane geometry gradient `g_k = ∂A_i/∂n_k` and route it through the existing
   `chainToDofs<Power>` to reach the `(x, w)` DOFs — the same combiner the volume/interface forces
   use. Boundary/SDF walls on the surface reuse `addSdfWallForce`.

This is the fix for "the free-surface spec was maybe not optimally connected": the arcs live *in the
power-cell faces*, the triple points *are* the power-cell dual vertices, and the force *is*
`chainToDofs<Power>` applied to a cap-area gradient. The layer is a consumer, not an engine.

### Why the power cell, specifically
Effort 1 established (and `test_power_cells` validates) that the device reproduces the min-image
power diagram exactly, including **buried cells** emptied via `d_ij ≤ 0`. A buried cell is precisely
a ball fully covered by its neighbours ⇒ it contributes **no** surface patch. So the buried-cell
detection already computed by the power build is exactly the "interior particle" filter the free
surface needs — no separate coverage test.

## 3. New geometry primitives (none exist today — every facet is currently planar)

The power `ConvexCell` geometry (`facetGeometry`, `geomVolumeArea`, `geomVolumeAreaGrad`) computes
**planar** facet areas and their `dA_k/dn_l`. The free surface needs **spherical** quantities:

- **Spherical-cap / spherical-polygon area** `A_i = r_i² Ω_i`, where `Ω_i = Σ_m θ_m − (n−2)π` is the
  solid angle of the exposed spherical polygon (Girard's theorem; `θ_m` the interior angles at the
  triple points). Decompose into orthoschemes (right spherical triangles from the patch centroid)
  for robustness at few-vertex caps and slivers.
- **Arc primitive**: circle centre `c_ij = p_i + h_ij n̂_ij`, radius `ρ_ij = √(r_i² − h_ij²)`,
  endpoints = the two bounding triple points; arc length `= ρ_ij Δθ`.
- **Spherical vertex angle** at a triple point: the angle between the two incident arcs measured on
  `∂B_i` (a dihedral-style computation from the two arc tangents).

Suggested location: a new header `include/peclet/voro/free_surface.hpp` with a
`SurfacePatch`/`SurfaceView` SoA and a `buildSurface(power TessellationView + weights) → SurfaceView`
pass, plus the Girard kernels as `KOKKOS_INLINE_FUNCTION` on the cell (mirroring the per-vertex
planar kernels in `convex_cell.hpp`).

## 4. Output (SoA, GPU-friendly, mirrors `TessellationView`)

- `patch[i]`   : owner id, cap area `A_i`, solid angle `Ω_i`, centroid, energy `γ A_i`,
                 `∂A_i/∂x_i`, `∂A_i/∂w_i` (the surface-tension force + its weight/radius response).
- `arc[e]`     : `(i, j)`, axis `n̂_ij`, radius `ρ_ij`, `θ0, θ1`, endpoint vertex ids `v_a, v_b` —
                 ONE record, referenced by both patches with opposite orientation (watertight seam).
- `vertex[v]`  : position, triple `{i,j,k}`, incident arcs.
- CSR maps `patch → arcs`, `arc → endpoints`.
- Global: `E = γ Σ_i A_i`; force on `p_i` from `∂A/∂n` (§5) plus the volume-constraint multiplier
  (the same `w`-space pressure the power solver carries).

## 5. Forces — reuse vs new

- **The chain `n_k → (x, w)` is REUSED unchanged**: `chainToDofs<Power>` maps `g_k = ∂A_i/∂n_k` to
  `∂A_i/∂x` and `∂A_i/∂w` exactly as it does for the volume gradient. This is the whole point of the
  seam — the surface force uses the same `dn/dDOF` Jacobians validated in P3.
- **The geometry gradient `g_k = ∂A_i/∂n_k` is NEW** (spherical, not `geomVolumeAreaGrad`, which is
  planar). The cap area varies with a bounding radical plane only through its arc: by the spherical
  Gauss–Bonnet variation,

      ∂A_i/∂h_k  ∝  (geodesic length of arc k),

  and `h_k = √(nn_k)` relates the plane offset to the cap, so `∂A_i/∂n_k` is a rank-1 expression in
  `n_k` times the arc length. This is the one new derivative to derive and validate; feed it as
  `g_k` into the existing chain.
- **SDF walls on the surface** (a free surface meeting a solid boundary) reuse `addSdfWallForce`
  from Effort 2 for their self-DOF contribution.

**Mirror hook.** Spawn a mirror patch for `i` when `e_i` crosses `0⁺` (its first arc is born at zero
length) and despawn at `0⁻`; the energy change `= γ ΔA = 0` by criterion 5 below. This is the same
face gain/loss event the dynamic repair already detects (arc length crossing zero ⇔ a power-cell
face gained/lost), so it reuses the certificate machinery, not a new event system.

**Flat-mesh fallback (rendering only).** Replace each spherical patch by the planar polygon through
its triple points (fan-triangulated from the centroid) — this is exactly the existing planar facet
polygon, so it is free. Do **not** feed chord-polygon area into the energy: it under-counts `γ`; the
dynamics must use the spherical cap area `A_i`.

## 6. Acceptance criteria (machine-checkable test scaffold, FP64)

A `test_free_surface.cpp` would assert, mirroring the `test_power_cells` / `test_pervertex_geometry`
culture (analytic/closed-form oracles + FD gradients + host/device parity):

1. **Closure.** Every arc referenced by exactly two patches; every vertex by ≥3 arcs;
   `V − E + F = 2·(#components)` (Euler characteristic of the surface).
2. **Volume.** `Σ_i ⅓ ∮_{patch_i} (x·n) dA` equals `Σ_i vol(V_i ∩ B_i)` within tol (the surface
   encloses the union volume).
3. **Orientation.** `(centroid_i − p_i)·n > 0` for every patch (outward).
4. **Seam.** For a shared arc `(i,j)`, endpoints computed from side `i` and side `j` agree to tol;
   sampled arc points satisfy `|π(x,i) − π(x,j)| < tol` (the arc lies in the shared radical plane).
5. **Continuity (load-bearing).** As any `h_ij → r_i⁻`, arc length → 0 and `A_i` varies continuously
   — no jump. This is the zero-area spawn/despawn invariant the mirror hook relies on.
6. **Regression.** Equal weights (`w_i ≡ const ⇒ r_i ≡ const`) reproduce the ordinary (unweighted)
   alpha-shape boundary.

Plus the two checks the rest of the suite always adds:
7. **Closed-form spot checks.** Two overlapping equal balls → cap area vs the analytic lens area,
   rel < 1e-12; an isolated ball → `A = 4πr²`, no arcs.
8. **Gauss–Bonnet (the strong, oracle-free one).** Over a closed cluster,
   `Σ(caps Gaussian curvature) + Σ(arc geodesic curvature) + Σ(vertex exterior angles) = 4π·χ` —
   the machine-precision internal consistency check that needs no external reference, and the
   backstop for the new `∂A_i/∂n_k` derivative (FD-vs-analytic `∂A/∂x`, `∂A/∂w` < 1e-4).

## 7. Build order (when implemented)

1. `free_surface.hpp` primitives: arc from a power face, triple point from a dual vertex, Girard cap
   area. Gate `h_ij < r_i`. *Accept:* criteria 1–4, 6, 7 on static power tessellations.
2. `∂A_i/∂n_k` spherical derivative + route through `chainToDofs<Power>`. *Accept:* criterion 8
   (Gauss–Bonnet + FD gradients).
3. SoA `SurfaceView` + CSR publish + flat-mesh render fallback; host/device parity + np=1,2,4 MPI.
4. Mirror spawn/despawn wired to the dynamic repair's face gain/loss (criterion 5).

## 8. Dependencies and caveats carried from Efforts 1–2

- **Small-weight regime.** Efforts 1–2 are exact for `d_ij > 0` (live faces); the free surface
  inherits this. Radii `r_i = √w_i` with `w_i` in the power-solver's small range keep faces `d>0`.
- **Min-image space-filling.** The periodic min-image power diagram is not a perfect partition at
  large weights (`test_power_cells` diag `oracleFill`); a periodic free surface inherits the same
  `O(weight)` caveat. Non-periodic / open-boundary free surfaces (the usual case) are unaffected.
- **`ConvexCell` half-space limit.** A `d_ij ≤ 0` face is unrepresentable (buried cell) — correct for
  the free surface (a buried ball has no patch). A *live* `d<0` face would need the generalized
  origin-excluding half-space noted in Effort 1; out of scope here.
