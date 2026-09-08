# Power diagrams beyond the small-weight regime — rung A2 design note

> Status: design note, 2026-09-03 (Voronoi methods plan, rung A2 — "the load-bearing rung").
> Measured facts first, then what the engine cannot do today, then the plan. Written after A0/A3/A1
> landed, so the moving-point SDF path, the energy layer and the second-order walls are assumed.

## 1. What the engine does today (re-grep before acting)

The `Power` plane policy (`plane_policy.hpp`) builds the radical plane between seeds i, j as the
foot-point half-space `{x : n·x ≤ n·n}` with `n = α r`, `α = d/|r|²`, `d = ½(|r|² + w_i − w_j)`.
The foot-point form always contains the origin (the seed), so a plane with `d ≤ 0` — the seed on
the far side of the radical plane, i.e. neighbour j has the lower power AT the seed's own position
— cannot be represented. `CellBuilder::buildCell` (`tessellator.hpp`) detects `d ≤ 0`, calls the
cell **buried**, and emits it EMPTY (`kEmpty`). The gather is min-image over a worklist bounded by
the weight-aware reach `blockReachSq` (`√rSqMax + √(rSqMax + W_max − w_self)`).

Both are exact for the small-weight regime (`test_power_cells`: brute radical oracle 1e-15). Two
known gaps: (a) a cell whose seed lies outside it is NOT empty in general — the power cell is
just elsewhere — so "buried ⇒ empty" drops real volume; (b) at large weight spread a face can
need a non-nearest periodic image, which the min-image gather never sees (`oracleFill` 0.97–0.99
in the test's diagnostics).

## 2. Measured: which applications actually hit the gaps

`Tessellation.set_weights(r²)` on the unit periodic box, Σ V / L³ − 1 (an exact partition gives 0):

| configuration | ratio r_max/r_min | Σ V − 1 | empty cells |
|---|---|---|---|
| **uniform random positions** (overlapping balls), N = 4k | 1 | 0 | 0 |
| | 2 | −2.5e-2 | 377 |
| | 5 | −5.6e-2 | 1483 |
| | 10 | −8.4e-2 | 2677 |
| same, N = 32k | 2 / 5 / 10 | −2.7e-2 / −5.9e-2 / −9.1e-2 | 3176 / 12080 / 21219 |
| **non-overlapping packing** (RSA, φ = 0.25), N ≈ 2k | 1 / 2 / 5 / 10 | **0 / 0 / 0 / 0** | **0** |

The reason is algebra, not luck: for two spheres that do not overlap, `|r| ≥ r_i + r_j`, so

    d_ij = ½ (|r|² + r_i² − r_j²) ≥ ½ ((r_i + r_j)² + r_i² − r_j²) = r_i (r_i + r_j) > 0.

**A power diagram with `w = r²` of non-overlapping spheres never has a buried cell, at any
polydispersity.** Overlaps of DEM size (δ ≪ r) keep `d > 0` too (`d ≥ r_i(r_i + r_j) − ½δ(2r_i +
2r_j − δ) > 0` for δ < r_i). Buried cells appear only when a centre sits INSIDE another sphere
(`|r|² ≤ r_j² − r_i²`), which the uniform-random rows above are full of.

Consequences for the plan:
- **Tracks F (power-cell contacts) and G (cell-network CFD-DEM)** run on `w = r²` of a packing:
  the small-weight engine is exact for them as it stands. No engine change needed.
- **Track D (moving-cell fluid)**: the weights are the pressure DOFs of the volume projection,
  small by construction (`|w_i − w_j| ≪ |r|²`): exact as it stands.
- **Track E (droplets, union of balls)**: liquid particles overlap on purpose, but with moderate
  polydispersity (`r_j² − r_i² < |r|²` for neighbours) `d > 0` still holds; the gap opens only for
  a small ball deep inside a large one — a configuration the droplet model never needs.
- The remaining exposure is (b): a reach larger than half the box (few particles, or W_max ≫ h²).

So A2 is **re-scoped**: not a prerequisite for D/E/F/G. What remains is a correctness item for
exotic weight fields and small boxes, plus the diagnostics that make the boundary of validity
visible.

## 3. The plan (re-scoped)

| rung | deliverable | gate | size |
|---|---|---|---|
| **A2a Validity diagnostics** | `buildTessellation` reports (status bits already exist) the number of buried cells and whether any cell's reach exceeded L/2 (min-image invalid); Python `Tessellation.build` raises or warns on either unless `allow_buried=True` | the RSA rows above stay silent; the uniform-random rows warn with the counts | S |
| **A2b Buried cells are cells** | a general half-space form for the ConvexCell: keep the dual-triangle representation (vertices = plane triples, unchanged) but store each plane as `(û, d)` with the test `û·x ≤ d`, `d` signed; `clip`, `computeVertex`, the certificate and the per-vertex geometry kernels use `(û, d)` instead of `(n, nn)`; the derivative kernels (`geomVolumeGrad`, `geomVolumeAreaGrad`, `chainToDofs`) re-derived for the `(û, d)` parametrisation (∂/∂n → ∂/∂û, ∂/∂d). A buried seed's cell is then built by starting the clip from a point INSIDE the cell (the power-nearest neighbour's centre works: the seed's own "cell" is the set of points whose power to i is lowest — start the cuboid at the point of minimum power difference found on the worklist) | uniform-random rows: Σ V − 1 → 1e-12, empty cells → 0; `test_power_cells` brute oracle 1e-15 unchanged; the Voronoi path byte-identical (policy `if constexpr`) | L |
| **A2c Multi-image gather** | when the reach exceeds L/2, the worklist walks periodic images beyond the nearest one and the clip accepts the same neighbour id from several images (the published CSR then carries an image tag per facet, or the reciprocal map keys on (id, image)) | a 2-cell periodic box with unequal weights partitions exactly; `oracleFill` → 1 for the test's large-spread case | M |

A2a is cheap and should land with the next voro session; A2b is the engine surgery and is only
justified by an application that needs overlapping-ball weights (none of the current tracks);
A2c matters for small periodic boxes (benchmarks with few particles), not production sizes.

## 4. Gates carried from the plan

The plan's A2 gate ("`oracleFill` = box to 1e-12 at radius ratio 1…10 on LS packings") is met
today by the RSA rows for the packing case; the new gates above replace it for the overlapping
case.
