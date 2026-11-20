# Design note — power cells whose seed lies outside the cell (voro cold build + repair)

*Architect pass, 2026-11-17, from `briefs/ARCH_OUTSIDE_SEED.md` (Desk `voro-outside-seed` = A, verified). Code read on
voro `origin/repair-det` (d59d30f; every `file:line` below is that branch) and on `origin/main` b93f0e5 (Power
`reevalPublish`, `facetGeometryFrom`). Defect evidence: brief §4, logs `~/Codes/suite/voro-det/gate_logs/` (untracked).
Falsifier of this pass: `pilots/outside_seed/{flag.py,dvdn.py,falsifier_output.txt}`. Copied into voro `docs/` at landing.
Claims are tagged [derived], [measured] or [est]. Stands alone. **Addendum A (§10, 2026-11-17) supersedes §5.7 "Empty
cells under motion" and G2′(iii) of `briefs/WO_OS2.md` amendment 6.***

## 0. Verdict

1. **Keep the seed frame and the `(n, nn)` storage; make the plane sign-safe.** A plane `{x : p·x ≤ d}` (p = generator
   connector r_ij, d = ½(|r|² + w_i − w_j)) is stored as `n = (|d|/|p|²)·p`, `nn = sign(d)·(n·n)`. `n` is then always the
   OUTWARD normal with |n| = distance seed→plane, and `n·x ≤ nn` is the intended half-space for either sign of d. For
   d > 0 this is the legacy representation bit for bit; for d < 0 it is exactly `(−n_old, −nn_old)`.
2. **The clip algorithm never needed the seed inside its cell.** Cramer vertices, the kill test, the horizon walk,
   `inconsistentFan`'s orientation signs and `growDisk` read only half-space equations and outward normals. The seed's
   position enters in exactly two places: the representation (fixed by 1) and formulas that use `n` as a *point* or
   `nn` as |n|² (fixed by the role rule of §5.2). [derived; falsified numerically, §0.7]
3. **An empty power cell is the clip outcome "every live vertex strictly outside"**, detected in the clip's existing
   "killed but no horizon" branch (today `overflow = true`, "cannot happen"). The cell sets `nt = 0` and the gather
   stops. No LP, no Chebyshev radius, no threshold: the same sign decisions that drive every cut decide emptiness.
4. **d = 0** (seed exactly on its own facet plane) is replaced by `d = ε·|p|²` (one unit of roundoff of |p|², smaller
   than the roundoff d itself carries); the plane stays on the legacy path.
5. **No new state anywhere.** TopologyStore is unchanged: the repair rebuilds every plane from positions and weights
   each step, and the sign comes with it, continuously in d. No per-cell frame shift, no LP, no persisted origin.
6. **Bitwise:** Voronoi and all-positive power cells take only expressions that reduce to the legacy ones when
   `nn > 0`; gate G6 proves it on host and CUDA.
7. **Falsifier (this pass, `pilots/outside_seed/falsifier_output.txt`)** [measured]: three random 14-plane polytopes with the
   origin outside (5–7 planes with d < 0): the per-vertex volume with foot points `q = sign(nn)·n` matches the qhull
   volume to ≤ 8.9e-16 relative (with `n` instead of `q`: 0.63 → 1.20, wrong); the area sum with `det3(n,·,·)/(2|n|)`
   matches the hull area to ≤ 2.2e-16; facet centroids lie on their planes; dV/dq with the signed fold `1/(2·nn)`
   matches central differences to ≤ 1.1e-7 (FD noise); ∂v/∂n_j = cof_j (2q_j − v)ᵀ/D matches FD to 2.0e-8 over 200
   random sign patterns.
8. **Status bits:** `kNegOffset` (8, defined, unused today) = "seed outside its own non-empty cell; geometry exact";
   `kBuried` (16) narrows to "power cell genuinely empty" (always with `kEmpty`). Neither is a defect; the Python strict
   mode stops raising on `buried` (default, §8 Q1).

## 1. Problem and scope

**Built.** voro's ConvexCell, cold build (`tessellator.hpp`) and MovingTessellation repair (`repair.hpp`,
`reeval_tessellation.hpp`) publish a power (Laguerre) cell whose generator lies outside it with its true geometry, and
an empty power cell as genuinely empty (zero volume, no facets), on device, deterministic, with no change to Voronoi
or all-positive power results. Today such a cell is emptied at the first candidate with d ≤ 0 (`tessellator.hpp:656-659`,
status `kEmpty|kBuried`); Fable's 32³ run lost up to 1.891e-3 of the box volume in 349 of 500 frames and produced up to
1805 one-sided facets this way (frame 152, seed 17238: true volume 0.6866 h³, 13 facets) [measured, brief §4].

**In scope:** ConvexCell plane representation and every ConvexCell member that reads `n`/`nn`; `gatherCell`,
`buildCell`, `finishCell`; status bits and the validity report (C++ and Python); certify/certifyList/reevalGeometry/
reevalPublish behaviour (verified, little code); tests and gate tooling.

**Out of scope:** NHV-W `lagrangian/*` (untouched; its seeds stay inside by the binding gate, its connectors come from
positions); SDF wall clipping beyond keeping it working (wall planes are committed only with offset > 0,
`sdf.hpp:569,658`, so they stay on the legacy path); MPI changes; performance work beyond §2's envelope; landing on
voro main; the voro decision register edit (the orchestrator files §9).

## 2. Constraints and invariants

- **Convention:** the seed is the origin of its cell's frame; `pdir = r = x_j − x_i` (min-image), unnormalized;
  `d = Power::offsetFromRel = ½(|r|² + w_self − w_nbr)` (`plane_policy.hpp:121`); half-space `{x : r·x ≤ d}`; box planes
  `n = ±(L/2) e_ax`, `nn = (L/2)²` (`convex_cell.hpp:179-195`). Plane index ≤ 255; MAXP 64, MAXT 112 unchanged.
- **Bit-exact:** Voronoi cells (cold build, repair, publish) and power cells whose every plane has d > 0, on host and
  CUDA, at the configurations of G6. Determinism across thread counts (repair-det gate (a)) holds.
- **May move:** results of cells with some d ≤ 0 (they are wrong today). Correct means: volume within 1e-12·h³ of an
  independent oracle, identical facet-neighbour set, Σ V = L³ to 1e-12 relative.
- **Device-first, MPI-distributable:** every change is per-cell device code; no host path except test oracles; halo
  semantics unchanged. **Sugihara:** combinatorial decisions by sign tests plus topology rules; no exact or adaptive
  predicates, no new tolerance that decides topology.
- **Performance:** all-positive cold build, repair step and publish within 2 % of the baseline (G8); outside-seed
  cells may cost a normal cell's build (they are ≤ 156/32 768 per frame at W = 0.15 h²).
- **Interfaces kept:** `clip(pdir, d, nbr, gapOut)`, TopologyStore layout, `facetGeometryFrom`, `Policy::chain`
  (consumes derivatives with respect to the foot point `q = α r`, α signed).

## 3. Mechanism

**3.1 Why the current cell is wrong.** `clip` stores `n = (d/|p|²) p`, `nn = n·n` (`convex_cell.hpp:820-824`). The stored
set `{n·x ≤ n·n}` always contains the origin. For d < 0 it is `{p·x ≥ d}`, the complement of the intended half-space
(1-D: p = e_x, d = −0.1 → stored x ≥ −0.1, intended x ≤ −0.1). For d = 0 it is all of space. The cold build avoids
this by emptying the cell; reeval of a stored plane whose d turns negative silently inverts it (`:276-283`), and
`isSelfConsistent` then flags the cell and the rebuild empties it.

**3.2 What actually depends on the seed being inside.** [derived] Reading every `n`/`nn` use:
- the representation itself (3.1);
- `nn` used as |n|²: `isSelfConsistent(:325)`, `isSelfConsistentPartners(:363)`, `isLocallyConvex(:449)`,
  `isLocallyConvexPartners(:476)`, `inconsistentFan`'s sine threshold (`:704`), clip's `gapOut` (`:844`),
  `faceMoment2(:506)`, `volume()`'s `support = sqrt(nn)` (`:1444`), the `1/(2nn)` folds (`:1721,1806`), `den`/self term
  in `geomVolumeAreaGrad` (`:1900` ff.);
- `n` used as a *point* (the foot of the perpendicular, which is `−n` when d < 0): cone volumes `det3(n_a − n_b, f, v)`
  in `volumePerVertex` and its copies, triangle centroids `(n_i + f + v)/3` in the moment scatters, the reference
  point of `faceMoment2`, the `2·da·n` numerator of the volume gradient, `u_j = 2n_j − v` in the vertex derivative;
- the clip's "no horizon ⇒ cannot happen" branch (`:868-870`, `:886-889`): with the seed inside a plane can never take
  every vertex.

**3.3 What does not.** Cramer (`computeVertex :213`) solves `n_i·v = nn_i` — sign-agnostic. The kill test
`n·v − nn > 0`, `growDisk`'s slack, the horizon walks and `compactPlanes` read half-space equations or combinatorics.
`inconsistentFan`'s `det(x,y,p)·det(x,y,z) > 0` and the canonical `det(n1,n2,n3) > 0` order of the per-vertex kernels
need the normals to be the OUTWARD normals of the half-spaces; the sign-safe form keeps that true. `faceOrderedIdx`
sorts about the facet centroid around `n/|n|` (outward). `facetGeometryFrom` takes its orientation and connector from
the generator vector r and the vertices. The security reach `blockReachSq(maxVertexRsq, …)` bounds |v| from the seed,
which is valid wherever the seed is. The area numerator `det3(n_k, g, v)` is `2|n_k|·A_signed` for either sign
(the `q_k ∥ n_k` term drops out), so area formulas need no change. All of this is confirmed by §0.7.

## 4. Decision and rejected alternatives

**Chosen — R1, sign-safe foot-point planes in the seed frame** (§5.1). It touches no state, keeps the hot kill loop and
Cramer verbatim, and its d > 0 branch is the legacy arithmetic, so bit identity is by construction.

Rejected:
- **Per-cell interior origin (LP / Chebyshev centre) and frame shift, persisted in TopologyStore.** The frame can only
  be chosen once the cell is known, so the cold build would still need either a sign-safe clip or a device LP — a
  second geometric algorithm whose empty/non-empty decision is a tolerance on a Chebyshev radius (a numerical
  threshold deciding topology, against the suite's topology-oriented rule). The shift is state that motion
  invalidates (the point leaves the cell), so certify would need a new detection and re-solve path; and every
  position-valued output and every consumer of `n` (`sdf.hpp`, `fv/polymesh.hpp`, the Power chain `n = α r`,
  gradients) would have to apply and undo it. More code, more state, more failure modes, for an audit that R1 reduces
  to one rule.
- **Unit-normal signed form `(n̂, δ)` with an interior reference point.** Changes every plane's rounding (Cramer on unit
  normals), so Voronoi and all-positive power cannot stay bit-exact; adds a scalar per plane in registers.
- **R2: keep `n` = foot point, add a per-plane sign mask.** Equivalent mathematically, but puts the sign into the hot
  kill test, both certificates and the orientation tests, and the mask must follow `compactPlanes`. R1 carries the
  sign in `sign(nn)` for free.
- **Status quo / empty every d ≤ 0 cell:** refuted by §1's measurement.
- **A host fallback for outside-seed cells:** violates device-first.
- **d = 0 as a fourth plane kind (unit normal, nn = 0):** breaks the invariant |nn| = |n|² that §5.2 relies on in ten
  places. **d = 0 → −ε|p|²** is as valid as +; + keeps the plane on the legacy path.

## 5. Design

### 5.1 Representation and helpers (`convex_cell.hpp`)

Invariant for every plane k after this change: `n_k` is the outward normal of the half-space `{n_k·x ≤ nn_k}`,
`|nn_k| = n_k·n_k` (as computed), and `σ_k := sign(nn_k)` is +1 iff the seed is strictly inside the half-space.
Box planes and SDF wall planes always have σ = +1. Derived quantities:

| name | definition | for σ = +1 |
|---|---|---|
| foot point `q_k` | `σ_k·n_k` (implemented `nn_k < 0 ? −n_k : n_k`, exact negation) | `n_k` (same bits) |
| plane distance | `sqrt(fabs(nn_k))` | `sqrt(nn_k)` (same bits) |
| signed support `h_k` | `nn_k >= 0 ? sqrt(nn_k) : −sqrt(−nn_k)` (= seed's signed distance, + inside) | `sqrt(nn_k)` |

New members (internal C++ names, not public API): `setHalfSpace(k, p, d)`, `footPoint(k, out[3])`,
`planeDistance(k)`, `signedSupport(k)`. `setHalfSpace` is the ONLY place a neighbour plane is written:

```
setHalfSpace(k, p, d):                       // plane {x : p·x <= d}
  l2 = p0*p0 + p1*p1 + p2*p2                 // same expression as today's l2 / rho
  if !(l2 > 0):  a = 0; n = a*p; nn = n·n; return        // coincident seeds: legacy bits (no constraint)
  if d == 0:     d = eps<Real> * l2                      // §5.1.1
  if d > 0:      a = d / l2;    n = a*p; nn = n·n        // legacy, verbatim
  else:          a = (-d) / l2; n = a*p; nn = -(n·n)     // seed outside this half-space
```

Callers: `clip` (replacing `:820-824`) and `reevalGeometry<Power>` (replacing `:276-283`; it already computes
`rho` and `off` with the same expressions). The Voronoi branch of `reevalGeometry` (`:284-291`), `initBox`,
`initBoxPlanes`, `WallStore::load` and `fv/polymesh.hpp:288` stay as they are (σ = +1 by construction).
`Power::buildNormal` (`plane_policy.hpp`) keeps returning the signed foot point `q = α r`; it is the chain's variable.

**5.1.1 d = 0.** The computed d carries an absolute rounding error of order ε·max(|r|², |w_i|, |w_j|) ≥ ε|r|²; replacing
an exact 0 by ε|r|² moves the plane by ε|r|, inside that uncertainty. It is the only "perturbation" in the design, it is
symbolic in spirit (a fixed, consistent choice), and the partner's copy (d_ji = |r|² > 0) is unaffected. Reachable only
for Power (Voronoi d = |r|²/2 > 0).

### 5.2 The role rule (applies to every formula that reads `n` or `nn`)

| role of `n_k` / `nn_k` in a formula | use |
|---|---|
| half-space evaluation (`n·v − nn`), Cramer, slacks | `n`, `nn` unchanged |
| orientation (canonical `det(n1,n2,n3) > 0` swap, `inconsistentFan` det signs, `faceOrdered` CCW, edge directions `n_a×n_b`) | `n` unchanged |
| area numerator `det3(n_k, ·, ·)` and its `1/(2|n_k|)` | `n` and plane distance |
| a point in the facet plane (cone apex, triangle centroid, moment arm, `faceMoment2` reference) | foot point `q_k` |
| `|n|²` normalisation (`1/sqrt(nn)`, sine threshold, area-vector fold) | `fabs(nn)` |
| derivative of a plane parameter (volume-gradient fold, vertex derivative) | signed `nn` / `q`, see 5.5 |

Every gradient output is a derivative with respect to the **foot point q_k** — what `n_k` meant before this change and
what `Policy::chain` (`n = α r`, α signed) consumes. For σ = +1 every entry of the table reduces to today's expression. `q` and σ always follow their plane
through a kernel's canonical swap (swap them with the normals and cofactors).

### 5.3 `clip` (`convex_cell.hpp:803-947`)

1. Capacity handling unchanged. `pi = np`; `setHalfSpace(pi, pdir, d)`; `pnbr[pi] = nbr`.
2. Kill loop unchanged (`s = n·v − nn > 0`). No live-count is added to the loop.
3. `gapOut`: `nl = planeDistance(pi)` (was `sqrt(nn)`); `*gapOut = nl > 0 ? smax/nl : −1e30`.
4. `nk == 0 → return false` unchanged.
5. **Empty outcome.** Lean path: after `collectHorizon` (and `growDisk` if taken), in the existing branch
   `nnew == 0 && anyKilled(kill)` (today: commit then `overflow = true`), first test `allLiveKilled(kill)` (a new O(nt)
   scan: every alive triangle has `kill[t]`). If true: `nt = 0`; do not commit the plane; `return true`. If false, keep
   today's `overflow = true` (a broken triangulation must never publish as empty). TrackAdj path: identical, in the
   `H == 0 && anyKilled(kill)` branch after `walkHorizon` (which returns 0 when no killed triangle has a surviving
   neighbour, `:1014`). The cost is paid only in the clip that empties the cell.
6. Comments: replace "(no horizon <=> nothing killed: the seed is inside every half-space …)" and the clip doc
   (`:796-802`) with the R1 statement.

An empty cell is `nt == 0`: `empty()` is true, every loop over triangles is empty, a later `clip` returns false at
step 4 (smax = −1e30), `maxVertexRsq() = 0`, `planeGap = −1e30`.

### 5.4 Certificates and the topology rule

- `isSelfConsistent`, `isSelfConsistentPartners`, `isLocallyConvex`, `isLocallyConvexPartners`:
  `invlen = (nn[k] != 0) ? 1/sqrt(fabs(nn[k])) : 0` (was `nn > 0 ? 1/sqrt(nn) : 0`). The slack test itself is unchanged.
  For a power cell the slack test is the empty-orthosphere test (`π_i(v) ≤ π_k(v)`) wherever the seed is; replace the
  stale "WEIGHTED/POWER (deferred)" paragraph (`:344-349`) by that sentence.
- `computePoke4` (`:421`): unchanged (slack `nn − n·v` is sign-correct). Its 4th-face pick compares raw slacks across
  planes (a pre-existing scale mix); left as is for bit identity, guarded by gate G2 and risk R4.
- `inconsistentFan` (`:704`): `dp*dp <= kFanSine2 * fabs(nn[x]) * fabs(nn[y]) * fabs(nn[p])` (same multiplication
  order). Orientation part unchanged.
- `growDisk`, `collectHorizon`, `walkHorizon`, `diskHorizon`, `simpleHorizon*`, `planeGap`, `computeVertex`,
  `compactPlanes`, `maxVertexRsq`: unchanged.

### 5.5 Geometry kernels (`convex_cell.hpp`)

| function | change (σ = +1: identical) |
|---|---|
| `volume()` `:1434` | `support = signedSupport(k)` |
| `faceMoment2(k)` `:500` | `nl = planeDistance(k)`; unit normal `u = n/nl`; reference point `q_k` in `a = f_q − q_k`, `b = f_r − q_k` |
| `volumePerVertex()` `:1489` | load `n_i` and `q_i` for the three planes; the canonical swap exchanges `(n2,c2,q2)` with `(n3,c3,q3)`; cone terms `e = q_a − q_b` |
| `facetAreasPerVertex` `:1538` | none |
| `scatterFacetMoment` / `facetMomentsPerVertex` `:1589,1605` | scatter takes `n_i` and `q_i`: `sa, sb` from `det3(n_i, …)·0.5/|n_i|`; centroids `(q_i + ff + v)/3`, `(q_i + v + fl)/3` |
| `scatterGradRaw` / `geomVolumeGrad` `:1649,1671` | V as `volumePerVertex`; `sa, sb` from `n_i`; centroids with `q_i`; numerator `2·da·q_i − mom`; fold `inv = (nn != 0) ? 1/(2·nn) : 0` (signed `nn`) |
| `scatterAreaMomentRaw` / `geomVolumeArea` `:1733,1754` | V via `q`; `S_a` via `n`; `S_m` via `q`; fold: `areaVec = S_a·n/(2·fabs(nn))`, `dV/dq = (2·S_a·q − S_m)/(2·nn)` |
| `geomVolumeAreaGrad` `:1847` | `u_j = 2·q_j − v`; `den_i = 2·sqrt(fabs(nn_i))`; self term `f = contrib_i/fabs(nn_i)`, `grad[i][i] −= f·n_i`; finally `grad[i][j] *= σ_{pl[j]}` (derivative w.r.t. `q`) |
| `geomVolumeAreaGradAD` `:1953` | duals seeded on `n` as now; dual plane offset `σ_k·(n·n)`; outputs `grad[i][j] *= σ_{pl[j]}`; stays the oracle of the analytic form |
| `geometryPerVertex` `:2024` | V via `q`; areas via `n`; moments via `q`; doc: area vector `area·n/|n|` is outward = along r |
| `facetGeometry` (Voronoi, `r = 2n`) | none (σ = +1 for Voronoi); doc says Voronoi only |
| `facetGeometryFrom` (main) | none |

Derivations [derived, §0.7]: with `S_a = Σ det3(n_k, ·, v) = 2|n_k| A_k` and `S_m = 2|n_k| m_k` (moments about
q-apex triangles), `(2 S_a q_k − S_m)/(2 nn_k) = σ_k (A_k/|n_k|)(2 q_k − x̄_k) = ∂V/∂q_k`; `∂v/∂n_j = cof_j (2q_j − v)ᵀ/D`
from `n_j·v = σ_j |n_j|²`; `∂/∂q_j = σ_j ∂/∂n_j` because `n_j = σ_j q_j`.

### 5.6 Cold build (`tessellator.hpp`)

`gatherCell` (`:556-688`):
1. Delete the `off <= 0 → buried` block (`:653-660`) and the `buried` variable; the loops run on `!c.overflow && !emptied`.
2. Power cull (`:662-663`): `if (!emitNear && off > Real(0) && rho * c.maxVertexRsq() <= off * off) continue;` — for
   d < 0 the old test would skip planes that kill the whole cell.
3. After a successful Power clip: `if (c.nt == 0) { emptied = true; break; }` (O(1)), before the `reachSq` update.
4. Return `!emptied` (true for Voronoi). `pore_cells.hpp:185` ignores the return (Voronoi).

`buildCell` (`:696-718`): the existing buried block becomes the empty block: `if (Weighted && !nonEmpty)`:
`status = kEmpty | kBuried`, `facetCount = 0`, `cellFacetBase = 0`, `cellVol = 0`, `emitEmptyTopo(i)`, and — new —
`oCandCnt(i) = ncRec` when `emitCand`, `oNearCnt(i) = 0` when `emitNear` (today they keep stale values; a stale
overflow marker would re-gather the cell every step). No SDF clip, no completeness test (a subset of constraints that
is already empty certifies emptiness).

`finishCell` (`:407`): after `empty` is known, for Weighted, `!empty && !c.overflow`: if any alive triangle references a
plane with `nn < 0` → `st |= kNegOffset` (an O(nt) scan; a seed outside a convex cell violates some facet inequality,
so this is "seed outside", independent of gather order). Everything else unchanged; the Power facet emission already
uses `facetGeometryFrom` with the generator connector (main b93f0e5); box/wall planes keep `2n` (σ = +1).

Status bits (`:50-69`, doc rewrite, values unchanged):
- `kNegOffset = 8` — Power, informational: the seed lies outside its own non-empty cell; geometry exact.
- `kBuried = 16` — Power: the power cell is empty (every point is dominated in power distance); published with zero
  volume and no facets; exact; always together with `kEmpty`.
- `kEmpty = 2` — zero-volume cell: SDF seed in the solid, or `kBuried`.

### 5.7 Repair (`repair.hpp`, `topology_store.hpp`, `reeval_tessellation.hpp`)

- **TopologyStore: unchanged.** Planes are rebuilt from `pnbr`, positions and weights by `reevalGeometry<Power>`
  through `setHalfSpace`; a plane whose d changes sign between steps changes representation continuously
  (|n| → 0 → |n|, nn through 0). Empty cells are stored by `emitEmptyTopo` (np 6, nt 0) as today.
- **certify / certifyList:** no code change. `Vv(i) = volumePerVertex()` is correct by 5.5; the consistency tests by
  5.4; `planeGap` (near-miss) takes the raw `(pdir, off)`. An outside-seed cell is now a normal stored cell; the
  inverted-certificate rebuild loop disappears.
- **reevalGeometry:** 5.1. **reevalPublish (main):** Power re-evaluation and generator connectors already on main;
  `faceMoment2` and the area-gradient publish (`areaGradFill`, via the 5.5 kernels) become correct with WO-1/WO-4.
- **Empty cells under motion** [derived; **SUPERSEDED by §10** — premise measured false in WO-OS3]: a stored non-empty cell that collapses inverts on its fixed topology; each
  triple point then violates a remaining plane, so the self-consistency (and edge-opposite local) test flags it and the
  re-gather empties it; its neighbours lose the facet the same way. An empty cell has no vertices and never flags
  itself; its re-appearance is a face GAIN for some neighbour j, i.e. the plane ji starts to cut j. That is covered
  exactly like any non-neighbour candidate: by j's near-miss list (gap > −margin at j's build; the certificate then
  flags j and, mutually, i), or, beyond the margin, by the skin-mover trigger on i or j and Pass 2's expansion of a
  mover's new neighbours. No new mechanism.
- **Known cost, accepted** (since §10.3 rule R-a: every empty power cell, SDF or not)**:** in an SDF domain, an empty power cell (`nt == 0`, seed in the fluid) trips the certify wall
  branch "seed back in the fluid" (`repair.hpp:~452`) and is re-gathered every step. Rare; R6.

### 5.8 Public surface

- `BuildReport::buried` doc (`repair.hpp:78`): "power cells that are empty (exact)". `empty` doc: in-solid seeds +
  empty power cells.
- `src/voro_bindings_tess.hpp:154-158`: drop `r.buried > 0` from the warn/raise condition and from the message (exact
  results are not validity failures). `src/voro_tessellation.cpp:36` docstring likewise. Report keys unchanged.
- `python/test_voro.py:264-273`: assert that a configuration with empty power cells neither warns nor raises and
  reports `buried > 0`; build that configuration deterministically (one seed with w = 0 in a lattice whose other
  weights are 2h², or whatever makes the cell empty — check by the C++ oracle).

### 5.9 Robustness argument

Combinatorics: the clip's validity (the conflict region is a disk; the fan corners inherit the killed corners'
orientation) is a statement about half-space equations and outward normals on a closed dual triangulation; R1
preserves both for every sign of d. The only new outcome is "conflict region = the whole surface", which leaves no
horizon; it is decided by the same sign tests, yields a valid (empty) structure, and is accepted only after an explicit
all-alive-killed check (so a broken triangulation still reports overflow). Near-degenerate inputs (the plane touches
the cell; a vanishing cell) are handled by the existing topology rules; a cell judged empty while exact arithmetic
would keep it has all its vertices within roundoff of one plane, i.e. a volume at roundoff level. Numerics: vertex
error ∝ ε·|v| with |v| ≤ √maxVertexRsq measured from the seed, the bound the reach already uses; the volume sum
`Σ h_k A_k/3` with mixed-sign h_k loses about log10(dist(seed, cell)/size(cell)) digits — O(1) for |w| ≲ h²
(frame 152: inscribed radius 0.362 h, seed adjacent). Tiny cells far from their seed lose more; recorded by G5, not
gated (R7). No exact or adaptive predicate, no new topology tolerance; d = 0 is a fixed symbolic choice.

### 5.10 Cost estimate [est]

All-positive path: clip +1 compare and a sign select per clip (kill loop untouched); cull +1 compare per candidate;
`volumePerVertex` +3 selects per vertex against ~170 flops; certificate changes only inside the violation branch;
`finishCell` +1 scan of ~40 alive triangles × 3 plane-sign loads per cell against ~10⁴ flops of building it.
Expected ≤ 1 % on host and CUDA (gate ≤ 2 %, G8). Outside-seed cells: a full build instead of an early abort
(+≤ 0.5 % of cold-build time at W = 0.15 h², where they are ≤ 0.5 % of cells); empty cells stop at the emptying clip.

## 6. Work orders (dependency order; one commit each; who: `engineer`)

Branch `m7-outside-seed`, worktree `suite/voro-outside-seed` (own `build*/`), based on the rebased `repair-det` tip
(main b93f0e5 + determinism commits). Bounded OMP (`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`), CUDA only via vof's
`heavy.lock`, logs in `gate_logs/` named per gate. Commit trailers: as given in the brief.

**WO-0 — gate tooling and baseline (test code only).** In `tests/kokkos/test_determinism.cpp`: `--wmax W` (weights
uniform in [0, W h²], default 0.15, so default output is unchanged); `--dump` accepts a comma list of frames and also
writes the cold build's per-cell status, volume and sorted facet-neighbour list; the guard prints per frame the counts
of `kOverflow`, `kIncomplete`, `kNegOffset`, `kBuried`, and the repair-vs-cold metrics `repVolMax/h³ =
max_i |V_pub,i − V_cold,i|/h³` and `repFacetMismatch` (cells whose published facet-neighbour set, facets with |A| > 0,
differs from the cold build's). Check that the power publish passes `Policy = Power` and the weights to
`reevalPublish` (main 434adb8 signature); fix if not. Build the baseline (branch base) on host and CUDA. Measure W0 =
the largest of {0.10, 0.05, 0.03, 0.02, 0.01} with zero buried cells (old semantics) in all 500 frames at n = 32.
Record baseline HASH/FRAME files (Voronoi and power-W0, n = 32, 500 frames, OMP 8; CUDA cold hashes), `--time`
medians (5 runs), and the G1 baseline at W = 0.15. *Accept:* default-argument hash lines identical before/after the
test edit; baseline reproduces brief §4 (349 frames, 1.891e-3, 1805) within the logs' digits; W0 recorded.

**WO-1 — R1 representation in ConvexCell.** §5.1, §5.3, §5.4, and from §5.5 `volume`, `faceMoment2`,
`volumePerVertex`; doc comments (struct `n`/`nn` doc `:147-150`, block comment `:1449-1455`, `volume()` doc, clip doc).
New ctest `test_outside_seed` (host + device), with exact expectations:
(a) `initBox(2,2,2)`, clip `p = e_x, d = −0.25`: `nn < 0`; `volume() = volumePerVertex() = 3` (rel ≤ 1e-15); areas
4, 4, 1.5 ×4; facet centroid of the cut plane `(−0.25,0,0)`; `faceMoment2 = 8/3`; `signedSupport = −0.25`;
`isSelfConsistent(0)`; (b) `d = 0`: `|V − 4| ≤ 1e-14`; (c) then `p = −e_x, d = −0.25`: clip returns true, `nt == 0`,
`!overflow`, `empty()`; a further clip returns false; (d) two seeds in L = 1, `x1 − x0 = 0.25 e_x`, w0 = 0: build with
w1 = 0.0225 (d = +0.02), then set w1 = 0.1025 (d = −0.02) and `reevalGeometry<Power>` on the same topology: volume
0.42 (rel ≤ 1e-14) equal to a fresh clip; (e) after (a), touching planes `p = (1,1,0), d = 0.75` and
`p = (5,1,0), d = −0.25` (both touch the edge x = −0.25, y = 1; the second has d < 0): no overflow, volume 3 (rel ≤ 1e-14), self-consistent. *Accept:* T1 passes; all existing
voro ctests pass; G6 bitwise.

**WO-2 — cold build, status, report.** §5.6, §5.8. Rewrite the `test_power_cells` host oracle (`:52-90`): sign-safe
brute build over all N−1 neighbours (sorted, min-image) PLUS an independent check that never executes a σ = −1 code
path — the **translation oracle**: y = mean of the oracle cell's live vertices; require `d_j − p_j·y > 0` for every
plane (else FAIL "not interior"); rebuild the cell in the frame of y (`initBox(2L)` centred at y, planes
`(p_j, d_j − p_j·y)`, same order); volume, facet-neighbour set and facet areas must agree (|ΔV| ≤ 1e-12 h³, sets
equal, |ΔA| ≤ 1e-12 h²; no box face). Empty certificate: rebuild the oracle cell with the candidates before the emptying one (the snapshot),
validate the snapshot by the translation oracle, and require every vertex of the validated snapshot to violate the emptying plane.
Configurations: 8³ jittered lattice ±0.3h, weights uniform in [0, W h²], W ∈ {0.15, 0.5, 1.0}, 3 RNG seeds, plus the
existing moved-positions case. Device vs oracle per cell: |ΔV| ≤ 1e-12 h³, empty ⟺ empty, facet sets equal,
`kNegOffset` ⟺ oracle has a face with `nn < 0`. Non-vacuity: ≥ 50 outside-seed non-empty and ≥ 10 empty cells in
total (if not reached, add W = 2.0 and record the counts). *Accept:* G4 (host), G1, G6, G10.

**WO-3 — repair verification.** No library change expected. Run G2, G3, G5, G7. If G2 fails only for cells with a
`nn < 0` plane under the default local certificate, **WO-3b**: in `certify`/`certifyList`, a cell with any alive plane
of `nn < 0` uses `isSelfConsistentPartners` instead of `isLocallyConvexPartners` (one branch; all-positive cells
untouched), rerun G2/G3/G6/G7. Any other G2 failure: stop and report. *Accept:* G2, G3, G5, G7. Then WO-3c (and WO-3d if
triggered), §10.5.

**WO-4 — per-vertex moments and gradients.** The rest of §5.5. Extend `test_pervertex_geometry` (T3) on ≥ 20
outside-seed cells from a W = 1.0, 8³ configuration: translation oracle for every value kernel (volumes ≤ 1e-12 h³,
areas ≤ 1e-12 h², first moments `m_k = m'_k + A_k y` ≤ 1e-12 h³, `faceMoment2` via the parallel-axis identity
`M2 − A|x̄ − q|²` equal in both frames ≤ 1e-12 h⁴); `geomVolumeGrad` and `geomVolumeArea`'s `dV/dq` vs central
differences of `volumePerVertex` under `q_k → q_k ± δ e_a` (δ = 1e-6 h, plane rewritten as `n = σ q`, `nn = σ|q|²`,
live vertices recomputed): rel ≤ 1e-6; `geomVolumeAreaGrad` vs `geomVolumeAreaGradAD` ≤ 1e-12 rel (max-norm per
triangle) and both vs FD of `facetAreasPerVertex` ≤ 1e-6. *Accept:* G9, G6.

**WO-5 — device and performance gates.** G4/G6/G11 on CUDA, G8 on host and CUDA. No code unless a gate fails; a perf
miss is reported with the per-kernel times, not optimized beyond §2. *Accept:* G8, G11.

## 7. Verification gates (criteria fixed before the runs; a miss is recorded as a miss)

| gate | configuration | pass criterion | baseline / note |
|---|---|---|---|
| G1 cold correctness | test_determinism power, n = 32, 500 frames, W = 0.15, OMP 8 | every frame: volRel ≤ 1e-12, nonRecip = 0, emptyBad = 0, overflow = 0, incomplete = 0 | main: 349 frames miss, 1.891e-3, 1805 [measured §4] |
| G2 repair = cold | same run | every frame: repVolMax ≤ 1e-12 h³, repFacetMismatch = 0 | unreachable for any change (base: 369/500 frames); replaced by G2′ (`briefs/WO_OS2.md` amendment 6), its (iii) restated in §10.4 |
| G3 stress | W = 1.0, n = 32, 200 frames | G1 + G2 criteria; non-vacuous: ≥ 1 frame with an empty cell, ≥ 1000 kNegOffset cell-frames (else rerun at W = 2.0, declared now) | — |
| G4 oracle ctest | `test_power_cells` (WO-2) host + CUDA | per cell \|ΔV\| ≤ 1e-12 h³, empty ⟺ empty, facet sets equal, translation oracle agrees, counts ≥ 50 / ≥ 10 | — |
| G5 scipy cross-check | G1 run, frames 50, 100, 152, 200, …, 500 | every kNegOffset/kBuried cell vs `HalfspaceIntersection`: rel volume ≤ 1e-10, facet set equal; empty ⟺ LP max-slack ≤ 0; frame 152 seed 17238: 0.6866 h³, 13 facets | Fable's `oracle_buried.py` |
| G6 bitwise | Voronoi and power W0, n = 32, 500 frames, OMP 8; CUDA cold hashes | all HASH and FRAME lines identical to the WO-0 baseline | — |
| G7 determinism | G1 run and G3 run at OMP 1/4/8 | hash lines identical across thread counts | repair-det (a): 1108 lines [measured] |
| G8 cost | `--time`, Voronoi and power W0, n = 32 and 64, OMP 8 and CUDA, 5 interleaved runs | median new/base ≤ 1.02 for cold build, repair step, publish; if base-vs-base max/min > 1.02, the bound is that ratio | — |
| G9 kernels | T3 (WO-4) | as WO-4 | — |
| G10 Python | `python/test_voro.py` | passes with 5.8 semantics | — |
| G11 battery | all voro ctests (incl. MPI, cospherical, SDF) host + CUDA; `tools/clang_format_check.sh` | pass | — |

## 8. Risks and open questions (each with a default)

- **Q1 [preference, the user] Strict mode and warning on `buried`.** Default: neither warns nor raises (exact results);
  the count stays in the report. Alternative: keep a warning for empty power cells (often a sign of too-large
  weights). Reversible by one line in `voro_bindings_tess.hpp`.
- **Q2 [preference, the user, NAMING] Names.** `buried` now means "empty power cell"; no report key for outside-seed
  cells. Default: keep names, add no key (status bit `kNegOffset` only). A new key (e.g. outside-seed count) is a
  public name and waits for the user.
- **R3 [decided] d = 0 → +ε|p|².** Reversible in `setHalfSpace`.
- **R4 [fact] Local certificate completeness for outside-seed cells.** The Lawson edge test is origin-independent;
  the poke4 4th-face pick compares raw slacks. Measured by G2/G3 under the default `localCert = true`. Default if it
  misses: WO-3b.
- **R5 [fact] Weights changed between steps.** The near-miss/skin argument bounds position motion only; a caller that
  changes weights in place without a rebuild is outside today's contract, unchanged here. Default: document in
  `setWeights`; no code.
- **R6 [fact] Empty power cells in SDF domains** are re-gathered every step (5.7); since §10.3 (R-a) every empty power cell is. Default: accept; revisit if a wall
  + power run measures it.
- **R7 [fact] Accuracy of tiny cells far from their seed** (digits lost ∝ distance/size). Default: record the worst
  ratio and its volume error in G5; no gate.
- **R8 [fact] One-sided facets of roundoff area from a vanishing cell.** Expected never at G1's statistics. If
  nonRecip > 0 with area ≤ 1e-20 h², it is still a miss; the orchestrator analyses it before any waiver.
- **R9 [fact] MAXP/MAXT for outside-seed cells.** G1/G3 require overflow = 0. Default if they overflow: report counts;
  a capacity change is a separate decision.
- **R10 [fact] The repair-det rebase.** Gates assume the rebased tip with main's Power `reevalPublish`. Default: WO-0
  verifies; if the base is not pushed when WO-0 starts, base on `origin/repair-det` and rebase with it.
- **R11 [fact] MPI halo width vs the reach of outside-seed cells.** Not changed here; `kIncomplete` would flag it.
  Default: the MPI ctests in G11; a power MPI campaign is out of scope.
- **R12 [derived; falsified in §0.7] Gradient conventions** (5.2, derivative w.r.t. q). A load-bearing derivation for
  WO-4 only; default: G9's FD and AD comparisons are its falsifier; the orchestrator may add a `derivation-checker`
  pass on §5.5 before WO-4.
- **Float builds:** the design is precision-generic (ε of `Real`); bitwise gates run in double. Default: compile check
  only.

## 9. Register entries (for the voro decision register and M7 `DECISIONS_LOG.md`)

1. Power planes are stored sign-safe: `n = (|d|/|p|²) p` (outward), `nn = sign(d)|n|²`; foot point `q = sign(nn) n`.
   Rejected: per-cell interior-origin frame shift (state invalidated by motion, LP threshold decides emptiness, every
   consumer must undo it); unit-normal `(n̂, δ)` form (breaks bit identity); sign mask (sign in the hot loop).
2. An empty power cell is the clip outcome "all live vertices strictly outside" (`nt = 0`), checked explicitly before it
   replaces the "no horizon" overflow. Rejected: an LP or Chebyshev-radius emptiness test.
3. d = 0 is evaluated as d = ε|p|². Rejected: a fourth plane kind with nn = 0.
4. `kNegOffset` = seed outside its non-empty cell (informational); `kBuried` = empty power cell (with `kEmpty`); neither
   is a validity failure.
5. Geometry gradients are derivatives with respect to the foot point q (the chain's variable). Rejected: derivatives
   with respect to the stored n (would need σ in `Policy::chain`).
6. (§10) Vanishing power cells in the repair: an empty power cell is re-gathered every repair step (it has nothing to
   certify), and a stale power cell whose re-evaluated volume is not positive is rebuilt with its live neighbours in the
   same step; emptiness then matches the cold build except at roundoff (|V| ≤ 1e-24 h³), and no stale topology
   publishes a negative volume. Rejected: accepting emptiness flips below the certificate tolerance (an absolute
   tolerance bounds nothing for a cell smaller than it; measured 20 flips at W = 1.0 h², 3 with V < 0); near-listing
   committed planes; an LP/Chebyshev emptiness test; publishing max(V, 0).

## 10. Addendum A — vanishing power cells in the repair (2026-11-17)

*Architect pass from `briefs/ARCH_OS_Q2.md`. Code: voro `repair-det` 1ef9994 (WO-0…4 + WO-3b); every `file:line` in
this section is that commit. Evidence: `~/Codes/suite/voro-det/gate_logs/os/` (`g3_w1_3b.det.log`,
`g1_w015_3b.det.log`, `diag_w1.log`, `diag2_w1.log`, `diag/lp_w1.log`); handoff `proto3d/WO_OS_HANDOFF.md`. Supersedes
§5.7 "Empty cells under motion" and G2′(iii) of `briefs/WO_OS2.md` amendment 6. Stands alone.*

### 10.1 Problem: what was measured, and why §5.7 is wrong

Definitions. *tol*: the repair certificate's absolute tolerance, 1e-4 h in FP64 (`tests/kokkos/test_determinism.cpp:217`).
*Flip*: a cell published empty (`nt = 0`) by exactly one of the repair and the cold build of the same frame. *t\**: LP
Chebyshev radius of the true cell (> 0 non-empty, < 0 empty; `diag/lp_flips.py`). Out of scope: the base's
tolerance-level stale facets (amendment 6 stands), MPI power repair, CUDA (WO-5).

Measured (WO-OS3, power, W = 1.0 h², n = 32, 200 frames, after WO-3b) [measured]:
- 20 flips, 20 distinct cells, one frame each; cold = LP oracle in all 20.
- 17 gains (cold non-empty, repair empty): 16 four-facet slivers and 1 five-facet, V 1.1e-18 … 7.6e-11 h³, t* 3.7e-7 …
  9.05e-5 h, V/t*³ = 17 … 392 (regular tetrahedron: 13.9).
- 3 losses (cold empty, t* −6.0e-6 … −1.2e-5 h): the repair publishes the inverted tetrahedron, V = −3.0e-14 … −7.2e-14 h³.
- 81/81 neighbour rows: plane ij is a committed plane of j (54 redundant, 27 live), never on j's near-miss list; no skin
  mover (displacement ≤ 0.12 h).
- Empty power cells per frame: 223 … 497 at W 1.0 (`buried=` in `g3_w1_3b.det.log`), 0 at W 0.15
  (`g1_w015_3b.det.log`). The repair rebuilds ≈ 15 080 cells per step at W 1.0 (totalRebuilt 3 015 485 / 200 frames).

Mechanism [derived]:
- **A gain is a 1→4 flip**: the hidden point i enters the regular tetrahedron (j1…j4) around their common vertex v. No
  existing facet changes; the only witness is v lying beyond a plane (j_k, i). Those depths are the distances of v to
  the faces of the new cell, whose area-weighted mean is 3V/S (the inradius t* for a tetrahedron), so they are O(t*) <
  tol. j tests that plane, if at all, with tol (the brute test for an outside-seed j; the local test only when it is an
  edge-opposite or poke4 plane). The zero-tolerance near-miss watch (`planeGap > 0`, `repair.hpp:463-492`) holds only
  candidates that never cut during j's build; plane (j, i) cut an intermediate polytope and was committed, so it is
  never listed. i itself has no vertices: no certificate of i can fail, and its `planeGap` is −1e30.
- **A loss is a 4→1 flip.** The four stored planes of the tetrahedron stay positively spanning; its signed volume is
  +vol(T) while their half-space intersection is non-empty and −vol(T) once it is empty (the realised tetrahedron
  inverts) — exact for a four-plane cell. The measured inverted tetrahedra violate their planes by about their size,
  (|V|/13.9)^{1/3} ≈ 1.3e-5 … 1.7e-5 h < tol, so the certificate passes them.
- **Consequence:** an absolute tolerance bounds errors relative to the cells it certifies. For a cell smaller than tol —
  every cell at its appearance and disappearance — it bounds nothing: a flip is a 100 % error of that cell, and a sign
  error for a loss. Amendment 6's acceptance of stale facets (≤ 2.25e-9 h³ on cells of ~h³) does not transfer.

### 10.2 Decisions

**D-A1 (brief Q1): emptiness must match the cold build, except at roundoff.** Roundoff class: a flip whose cell has
max(|V_cold|, |V_rep|) ≤ V_round = 1e-24 h³ (t* ≈ 4e-9 h for a regular tetrahedron; six orders below the smallest
measured flip). Rejected: accepting flips with |t*| < tol, as amendment 6 does for facets. t* < tol bounds the inradius
only, not V (V/t*³ up to 392 measured), so acceptance would have to borrow (ii)'s 1e-8 h³ for V, would keep negative
volumes, and would publish cells empty that a cold build of the same positions publishes non-empty.

**D-A2 (brief Q2): no negative volume from a stale topology.** Rule R-b (§10.3): a power cell whose re-evaluated volume
is not positive is rebuilt, with its live neighbours, in the same step. A freshly clipped cell may still compute V < 0
at roundoff (valid topology, inexact geometry: Sugihara); the gate floor is −V_round. Rejected: publishing such a cell
empty by fiat (a topology decision without the clip; leaves the neighbours' facets one-sided); `V_pub := max(V, 0)`
(hides a stale topology instead of repairing it; repair ≠ cold). Cost: one compare per power cell in the Pass-1
certify, on a value it already computes; per event, the cell plus its ~4 live neighbours are gathered. Deterministic:
a per-cell predicate on per-cell data; the marks are idempotent writes of 1.

**D-A3 (brief Q3): an empty power cell is re-gathered every step** (rule R-a): it is made a mover, so it takes the full
gather (the cold algorithm, so its emptiness equals cold by construction) and, if it comes back, Pass 2 rebuilds its
face neighbours. Cost [derived from the measured counts]: +223 … 497 gathers per step on ≈ 15 080 at W 1.0
(+1.5 … 3.3 %), each ending at its emptying clip; zero at W 0.15 and on the all-positive path. Rejected:
(b) near-listing committed planes with a small miss distance — blind to the 27/81 rows where j already has a live
facet toward the empty i, active only while near lists are fresh (`repair.hpp:382`, `:884`), a new margin and more
`kNearCap` pressure; (c) certifying empty cells against stored neighbours — an empty cell stores only its 6 box planes,
so this needs new state (the emptying snapshot), excluded by §0.5; (d) an LP/Chebyshev emptiness test — a threshold
deciding topology (§9 entry 2); (e) a zero certificate tolerance — rebuild flip-flop on cospherical input, and a
verify loop that cannot converge on tol-degenerate cells.

**D-A4: compound losses are left to measurement.** A stale cell can empty without its own planes inverting (swept by a
fifth stored plane, or an intermediate flip and the 4→1 within one frame); its V then stays > 0 and every violation
stays < tol only if the cell is smaller than ~tol. Not observed (0 of 3 losses). The remedy is pre-specified as WO-3d
(rule R-c) and executed only if the gate shows such a flip.

### 10.3 Design

Placement: `MovingTessellation::certify` (`repair.hpp:346-520`), inside the per-cell lambda, after the skin-mover test
(`:430-434`) and before the flag line (`:494`); compiled only for `Weighted`; active only when `skinOn` (the Pass-1
certify of `step`, `:854`). **Not** in `certifyList` (verify): a freshly gathered empty cell, or a fresh cell with a
roundoff-negative V, must pass verify, else the bounded verify loop (`:932-951`) ends in a cold rebuild (`:952`) every
step.

```
if constexpr (Weighted) {
  if (skinOn) {
    bool vanish = false;
    if (c.nt == 0) {                          // R-a: empty power cell (an overflowed cell is stored the same way)
      vanish = true;
      if constexpr (kHasSdf)                  //   a seed in the solid is empty by the wall, not by power:
        vanish = SdC.eval(seed) > Real(0);    //   same predicate as the wall branch (:456); evaluate it once
    } else if (!(Vv(i) > Real(0))) {          // R-b: folded stale topology (NaN included)
      vanish = true;
      for (int t = 0; t < c.nt; ++t) {        //   flag every owned live-plane partner
        if (!c.alive[t]) continue;
        for (int k : {c.t0[t], c.t1[t], c.t2[t]}) {
          const int j = c.pnbr[k];
          if (k >= 6 && j >= 0 && j < nP_) Kokkos::atomic_exchange(&M(j), 1);
        }
      }
    }
    if (vanish) { mover = true; Mv(i) = 1; }  // full gather + Pass 2; the flag line (:494) sets M(i)
  }
}
```

- i must be a **mover**, not only flagged: movers take the full gather (`:913`) and Pass 2 (`collectNewNbrs`, `:668`,
  which rebuilds every owned face neighbour not yet rebuilt). The surgical path re-clips stored and partner-scattered
  candidates only, and an empty cell has none beyond its box planes.
- R-b's partners are flagged, not movers: in a 4→1 flip the four neighbours are pairwise adjacent before and after.
- Keep the file's nvcc rule: a capture first used inside an `if constexpr` is forced outside it (`:399-404`).
- Overflowed cells are also stored with `nt = 0` (`tessellator.hpp:413`); R-a retries them every step (an improvement;
  G1/G3 require overflow = 0 anyway).
- **Item 0 (orchestrator's decision):** `emitEmptyTopo` (`tessellator.hpp:395`) writes −1 into `pnbr` slots 0…5 of
  cell i, so an empty cell's store is fully defined. Inert: `load()` reads `[6, np)`.
- Doc: certify's comment block (`:336-344`) gains R-a and R-b in two sentences ("an empty power cell has nothing to
  certify and is re-gathered each step; a power cell with V ≤ 0 is folded and is rebuilt with its live neighbours").
- Invariance [derived]: an all-positive power cell contains its seed, so it is never empty, and at G6's configurations
  (W0 = 0.02 h², seed-to-plane distances ≫ tol) its re-evaluated V is positive. R-a and R-b never fire there; Voronoi
  does not compile them. G6 proves it.

### 10.4 G2′(iii), restated (written before any WO-3c run)

Runs: power, W 0.15 (n = 32, 500 frames) and W 1.0 (n = 32, 200 frames), OMP 8. O = the cells with kNegOffset or
kBuried in cold or repair, plus their face neighbours in either.
- **(a)** On O, the bounds of (ii): per frame repVolMax ≤ 1e-8 h³; median over frames of cells with differing facet
  sets ≤ 22; max one-view-only facet area ≤ 1.845e-4 h².
- **(b) Emptiness**, over all cells: every frame emptyFlip = 0. A flip with max(|V_cold|, |V_rep|) ≤ 1e-24 h³ is
  roundoff class: counted and reported, not a miss. Any other flip is a MISS.
- **(c) Sign:** every frame min_i V_pub,i ≥ −1e-24 h³; the count of V_pub < 0 is reported.
- **Non-vacuity** (W 1.0): over the 200 frames the repair's own published sequence has ≥ 10 empty→non-empty and ≥ 3
  non-empty→empty transitions; else rerun at W 2.0 (declared now) and judge (b), (c) there.
- Reported, no threshold: per frame `oneSidedToEmpty` (facets a non-empty cell publishes toward a cell published empty)
  and its max area; totalRebuilt and the median repair-step time (`--time`, 5 runs) against 1ef9994 at W 1.0.
- Expected [derived]: no gain flip (the repair runs the cold gather on every empty cell) and no loss flip from a
  four-plane cell (its volume changes sign exactly when it empties).

### 10.5 Work orders

**WO-3c — vanishing cells (one commit on `repair-det`; who: `design-implementer`).**
1. *Test first.* New ctest `test_repair_vanishing` (`tests/kokkos/`, host now, CUDA in WO-5). Construction, fully fixed:
   - L = 1, 4³ jittered lattice (h = 0.25, offsets uniform in ±0.2 h, `std::mt19937_64` seed 20261117).
   - Build cell 0 with the host brute oracle of `test_power_cells` (all N − 1 neighbours, min image, equal weights).
     Each live triangle that touches no box plane gives a vertex v = x_0 + (vx, vy, vz) and the tetrahedron
     (j1 = 0, j2…j4 = its three `pnbr` in stored order), R² = |v − x_0|². Margin m = min over the other 60 seeds of
     (|v − x_k|² − R²) (min image). Take the triangle with the largest m; require m ≥ 1e-3 h².
   - Seed i (N = 65): x_i = 0.4 x_j1 + 0.3 x_j2 + 0.2 x_j3 + 0.1 x_j4 (coordinates unwrapped about v), u = x_i − v;
     require |u| ≥ 0.01 h. Weights: R² for every lattice seed, w_i = |u|² + δ_A. Then i owns v iff
     δ(x_i) := w_i − |x_i − v|² > 0, and near v its cell is {v + y : (x_j − x_i)·y < δ/2, j = j1…j4}: a 1↔4 flip at δ = 0.
   - Frame A: x_i as above. Frame B moves only i: x_i,B = v + (|u| + s) û, s = −|u| + sqrt(|u|² + 2δ_A), so δ_B = −δ_A.
   - Gain case δ_A = −2e-6 h²; loss case δ_A = +2e-6 h² (cell size ~1e-5 h, t* ~1e-6 h ≪ tol).
   - `MovingTessellation` as at `test_determinism.cpp:217` (tol 1e-4 h, skin 0.25 h); `rebuild` at A, one `step` at B;
     reference = a cold build at B. Variants: `localCert` true/false × `useNearMiss` true (default)/false.
   - Preconditions (assert): cold at the δ > 0 frame has i non-empty with exactly the facet neighbours {j1…j4}; cold at
     the δ < 0 frame has i empty; the step has no skin mover.
   - Assertions at B: (1) i empty in the repair ⟺ empty in cold; (2) i and j1…j4 have cold's facet-neighbour sets,
     |V_rep − V_cold| ≤ 1e-12·V_cold for a non-empty i and ≤ 1e-15 h³ for the j's; (3) every V_pub ≥ −1e-24 h³.
   - Non-vacuity: on 1ef9994's library both cases FAIL in the `useNearMiss = false` variants (log
     `gate_logs/os/wo3c_pre.log`); after step 3 every variant PASSES.
2. *Gate tooling* (`test_determinism`): GUARD lines gain `flipRound`, `flipMiss` (emptyFlip split at 1e-24 h³),
   `minVpub/h3`, `negV`, `oneSidedToEmpty`, `maxOneSidedToEmptyA/h2`; RESULT gains `transGain`, `transLoss` (the
   repair's published emptiness against its previous frame, summed). HASH and FRAME lines unchanged.
3. *Library:* item 0, R-a, R-b (§10.3), doc comments.

*Accept:* item 1 as stated; G6 IDENTICAL 1108/1108 (Voronoi + power W0 = 0.02); G7 at W 0.15 and W 1.0 identical at
OMP 1/4/8 with store slots [0, np) hashed; G1 at W 0.15 and W 1.0; G2′(ii); G2′(iii) of §10.4; host voro ctest battery.
*On a miss:* a (iii)(b) loss flip (repair non-empty, cold empty) outside the roundoff class → WO-3d. A (iii)(b) gain
flip outside the roundoff class → stop and report (the repair's gather disagrees with the cold build's: a bug, not a
design gap). A (iii)(c) miss → stop and report the cell, its previous-frame state and whether it was gathered this
step. An existing ctest that asserts the fast path on a configuration with empty power cells contradicts R-a: change
its expectation in the WO-3c commit and say so in the message.

**WO-3d — conditional: only after a (iii)(b) loss MISS in WO-3c.** Rule R-c, a third branch of the §10.3 block:
`else if (c.anyLiveNegOffset() && Vv(i) < Vwatch) vanish = true;` (no partner flagging), with Vwatch = (10·tol)³ from
the certificate's own tol (1e-9 h³ at tol = 1e-4 h; no new parameter, no public name). Report the R-c cells per frame.
*Accept:* as WO-3c. Still a miss → stop and report (architect).

**WO-5** then runs as written, plus `test_repair_vanishing` on CUDA; G8 ≤ 2 % on the all-positive path (R-a/R-b add one
integer and one floating-point compare per power cell in the Pass-1 certify).

### 10.6 Risks and open questions (each with a default)

- **A1 [fact] Compound losses** (D-A4): measured by (iii)(b). Default: WO-3d on a miss.
- **A2 [fact] The repair's gather equals the cold build's** for a re-gathered cell (same grid, same candidate order):
  the premise of "no gain flip". Measured by ctest assertion (2) and by (iii)(b). Default: roundoff-class flips are
  reported, not chased.
- **A3 [fact] Cost of R-a:** +1.5 … 3.3 % gathers at W 1.0 [derived]; the no-gather fast path (`n1 == 0`, `:868`) is
  never taken while an empty power cell exists. Default: accept on the outside-seed path (§2 lets those cells cost a
  normal build); the step time is reported, not gated.
- **A4 [fact] Thin-slab blind spot:** a cell thinner than tol with V ≥ Vwatch, emptied by a near-parallel stored plane,
  passes every rule. Not observed. Default: accept; it would appear as a (iii)(b) loss MISS → stop and report.
- **A5 [fact] One-sided facets toward an empty cell** (27/81 rows were live): after R-a/R-b they remain only where a
  tol-flag rebuild emptied a cell whose unviolated live neighbours were not rebuilt; tolerance class, inside (ii)'s
  facet-area bound. Default: report `oneSidedToEmpty`; no rule.
- **A6 [fact, future] MPI:** the distributed repair is Voronoi-only (`mpi/distributed_moving.hpp`:
  `MovingTessellation<Real, MAXP, MAXT, false, Sdf>`) and Pass 2 expands owned neighbours only (`repair.hpp:683`). A
  power MPI repair must give the owner of j a trigger when a ghost empty cell i re-appears (gather empty ghost cells,
  or return Pass-2 marks to owners). Default: a recorded requirement of that future design; nothing now.
- **A7 [preference; the default needs no sign-off] Clamping V_pub ≥ 0** to protect consumers (e.g. `pow(V, 1/3)` of
  −1e-30). Default: no clamp (D-A2); a consumer that needs V ≥ 0 clamps at use. Reversible by one `max` at publish.
