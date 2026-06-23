# Research note — a GPU Voronoi engine for physical simulation

*Goal (from the brief): a Voronoi/power tessellation that (i) is suited to physical simulation —
volumes, face areas, **and their derivatives** w.r.t. moving points; (ii) runs **near-optimally on
GPU**; (iii) is **beyond voro++ on serial CPU**; (iv) **scales linearly** with threads; and whose
**dominant real workload is the per-step update of moving points**, not the cold rebuild.*

This note reframes the problem around that workload, records what we have measured, answers the
specific design questions raised, and proposes a phased experimental program. It builds on three
sources: the OpenMP/CPU work (the half-edge cutter + its `host/incremental.hpp` Verlet-skin update),
our GPU findings in [performance.md](performance.md) (the half-edge per-thread path, the team-per-cell
dead end, the ConvexCell prototype), and the literature (Ray et al. 2018; Basselin et al. 2021; the
2026 "Scalable GPU Construction of 3D Voronoi and Power Diagrams"; gDel3D).

---

## 1. The reframing: optimise the *update*, not the rebuild

In a physical simulation the points move a little every step. Two things must be distinguished:

- **Topology** — which seeds are neighbours, and the face/edge connectivity. Changes **rarely** (a
  face appears/disappears only when a seed crosses a bisector). This is the output of the *clip*, the
  expensive part.
- **Geometry** — vertex coordinates, face areas, cell volumes, **and their derivatives**. Changes
  **every step** (any neighbour move moves the bisector planes), but given fixed topology it is a
  *closed-form re-evaluation*, no clipping.

The OpenMP incremental win (`host/incremental.hpp`, Verlet skin, bit-exact to rebuild) is exactly:
**skip the clip for topology-stable cells; recompute only geometry.** For small time steps the
topology-stable fraction is very high, so the per-step cost collapses to the geometry re-evaluation
plus a re-clip of the few unstable cells.

**Consequence for the design.** The figure of merit is **per-step update throughput**, dominated by:

1. a **cheap, parallel "needs-reclip?" test** (a security/skin criterion per cell);
2. a **geometry+derivative re-evaluation** kernel over fixed topology (the common case — must be
   embarrassingly parallel and cache-light);
3. a **fast re-clip** for the small unstable set (the rare case — where rebuild throughput matters).

Cold-rebuild throughput (what performance.md has been chasing) is really *case 3 plus the first
build*. It matters, but it is **not** the steady-state cost. **This single reframing reorders the
priorities: the geometry/derivative re-evaluation and the needs-reclip test deserve as much attention
as the clip.** It is also the natural place a GPU wins, because both are data-parallel and branch-light.

---

## 2. Requirements vs. what we have (status)

| requirement | half-edge (production) | ConvexCell prototype | gap to close |
|---|---|---|---|
| GPU rebuild, pure tess | 2.49 M cells/s (occupancy-bound, 32 KB frame) | **5.4 M/s FP32** (3.5 KB frame, 2.18× half-edge) | neighbour query (BVH+directional) → SOTA ~7–12 M/s |
| serial CPU vs voro++ | **≈ 1.0×** (matches) | 0.4–0.6× (more arithmetic) | ConvexCell loses serial; half-edge wins |
| linear thread scaling | ~16× / 24 cores (mem-bound at top) | similar | gather memory traffic |
| volumes + areas | **yes, validated** | yes (volume; areas straightforward) | areas/derivatives unproven in ConvexCell |
| **derivatives** (dV, dA²) | **yes, validated** (`computeGeometry`, `gradFacetAreaSq`) | **not yet** (feasible — see §4.3) | re-derive + re-validate |
| topological robustness | **yes** (round-off fallback in `cutCell2`) | fragile at degeneracies (§4.4) | robust predicates |
| incremental update | **yes, CPU** (Verlet skin) | none | port to GPU; needs-reclip test |
| power/weighted | yes | trivial (radical plane) | — |

The honest reading: **the half-edge is the only artifact that meets the *physics + robustness +
incremental* requirements today**, but it is occupancy-bound on GPU. The ConvexCell is the only
artifact that is GPU-fast, but it is unproven on derivatives, robustness, and serial CPU. Neither is
a finished answer; the program below is about converging them.

---

## 3. The design space

Two nearly-orthogonal axes — **representation** and **parallel granularity** — plus the neighbour
query. Decoupling them is the key move.

**Representation (what is stored resident between steps):**

- **R1 full half-edge** — positions + explicit edge links + geometry. ~21–32 KB. Validated physics.
- **R2 compact half-edge** — *connectivity only* (the labels / neighbour list), recompute vertex
  coords + geometry on demand. **This is the user's "split into parts" idea, and it converges onto the
  ConvexCell footprint** (store topology, recompute coordinates).
- **R3 ConvexCell (dual)** — each vertex = a triple of plane indices; recompute coords. ~0.4–3.5 KB.
- **R4 face-list** — no cell topology at all; each Voronoi face stored/derived as an independent
  clipped polygon. Cell quantities are reductions over faces.

**Parallel granularity (who computes a cell):**

- **G1 thread-per-cell, sequential clip** — production + ConvexCell. *Full occupancy with a compact
  cell; not idle.* Sequential per cell, hidden by many cells in flight.
- **G2 team-per-cell, leader clips** — **measured dead end** (31/32 idle; the cut is not
  warp-parallelisable; per-cut barriers cost more than they save).
- **G3 warp-per-cell, lane-per-face** — the **face-based** idea: each lane owns one candidate face and
  clips it independently (a 2-D problem in the face plane). No sequential cut, tiny per-lane state,
  full warp utilisation. See §4.2.

**Neighbour query:** uniform grid (have it; loose isotropic radius) **vs. BVH best-first + directional
culling** (the SOTA lever; **ArborX gives us BVH on GPU for free** since the packing/DEM code already
depends on it).

---

## 4. Answering the specific questions

### 4.1 Is the GPU "massively under-utilised because the cut is sequential"?

Partly a misconception, partly real. In **G1 (production / ConvexCell)** every thread builds a whole
cell, so there are no idle lanes and a compact cell reaches high occupancy — the GPU is *not* idle.
The "one working thread per team" picture is **G2**, which we already rejected. What *is* true: (a)
each cell's clip is a **sequential critical path**, which bounds single-cell latency (irrelevant when
millions of cells are in flight, relevant for a *small* unstable set in the incremental case); and (b)
~85 % of candidate clips are **no-ops** (measured), pure wasted work. So the lever is **work
reduction and finer-grained parallelism**, not "filling idle lanes." That makes the face idea
attractive for the right reason.

### 4.2 Faces as the primitive (G3) — embarrassingly parallel, robust

A Voronoi face `(i,j)` is the polygon `{ x on bisector(i,j) : closer to i,j than to all other
seeds }`. Computing it = take the bisector plane and **clip it (a 2-D convex polygon clip) by the
other candidate half-spaces** — *independent of every other face*. So:

- **Parallelism:** one lane per candidate face → a warp does ~32 faces at once, no sequential cut, no
  idle leader. Per-lane state is one small 2-D polygon (a few registers). Most candidates yield an
  empty polygon (not a real face) — that *is* the filter, no separate pass. Work is O(K²) per cell
  (K candidates) but K-way parallel; critical path drops from ~50 sequential clips to ~one 2-D clip.
- **Quantities & derivatives:** `V_i = (1/3) Σ_j support_ij · area_ij`, a **reduction over faces**; no
  cell topology object is needed. Face area + its derivative w.r.t. the defining seeds is a standard
  planar-polygon computation.
- **Robustness:** each face is an independent **convex** 2-D clip — always valid. The classic worry
  (an edge present in face `(i,j)` but not exactly in `(j,i)` under round-off) does **not** corrupt a
  convex face; it perturbs an area by O(ε).
- **The real risk — momentum conservation.** Physics forces need `area_ij == area_ji` and consistent
  derivatives, or momentum drifts. Independently computed faces can differ by round-off.
  **Mitigation:** compute each face **once at a canonical owner** (e.g. `min(i,j)`) and scatter to
  both cells — exact `A_ij = A_ji`, parallel, no global topology. (This is the GPU analogue of the
  half-edge's `buildReciprocalMap`.) Whether the non-canonical, fully-independent variant conserves
  momentum *well enough* over a long run is a **make-or-break experiment** (§5, Phase 1).

### 4.3 Derivatives for ConvexCell / face-list

**Feasible, and not new.** Each dual vertex is the intersection of three planes (`computeVertex`,
Cramer), each plane is a bisector = a differentiable function of two seed positions; the chain rule
gives `∂vertex/∂x`, hence `∂area/∂x` and `∂V/∂x`. This is precisely the derivative machinery of
**semi-discrete optimal transport / Laguerre (power) cells** (Lévy and collaborators — the same
ConvexCell lineage), which compute cell-volume and area gradients w.r.t. seed positions and weights
for Lloyd/CVT/OT. So the literature already does this on exactly this representation. **But** our
half-edge code *already has it, validated* (`computeGeometry` → `fdV`, `gradFacetAreaSq`); switching
representations means re-deriving and re-validating — a real cost, not a free lunch.

### 4.4 Topological robustness

The half-edge `cutCell2` has an explicit **round-off fallback** (exhaustive sign-change search) and
maintains half-edge-link consistency — demonstrated robust. The convex-clip family is **always
convex-valid** (intersecting half-spaces cannot produce a non-convex cell), but the *dual
triangulation* (which 3 planes meet at a vertex) is **fragile at degeneracies** (≥4 cospherical
seeds): a mis-classified vertex corrupts the horizon — we *hit* this (the dead-triangle cascade).
A production convex-clip needs **robust predicates** (adaptive/exact orientation à la Shewchuk, or
simulation-of-simplicity symbolic perturbation), as the SOTA papers use. The **face-list (R4) is the
most robust** of the candidates for *physics quantities*, because each face is an isolated convex
2-D problem and topological inconsistency only perturbs areas by O(ε) rather than breaking a shared
mesh.

### 4.5 Compacting the half-edge ("split into parts")

Yes — and it converges on the same place. The half-edge stores **redundant** data: `vpos` and `pvec`
are recomputable from connectivity + neighbour positions. Keep only the **connectivity** (labels /
neighbour-id list) resident and recompute coordinates/geometry each step, and the footprint drops to
~the ConvexCell size (a few KB) — solving the occupancy problem **without changing the clip algorithm
or the validated derivatives.** So "compact half-edge (R2)" is a *low-risk* path to the GPU-occupancy
win that **keeps the proven physics + robustness**. It is probably the single most under-rated option
in the table and deserves a first-class experiment.

### 4.6 The serial-CPU ↔ GPU representation tension

Measured: the half-edge **matches voro++ on serial CPU**; the ConvexCell is **0.4–0.6×** (it trades
arithmetic for a tiny footprint, which only pays off on a high-occupancy GPU, and in FP32). So "beyond
voro++ on serial CPU" and "near-optimal on GPU" may want **different representations**, exactly as the
codebase already keeps backend-specific knobs (Morton GPU-only, seeds/cell host-only). The escape is
the §1 reframing: in steady state both backends spend most time in the **geometry/derivative
re-evaluation**, which is cheap and bandwidth-bound on *both* — so the representation choice matters
most for the *rare* re-clip, where we can afford a backend-specific code path.

---

## 5. The proposed program (phased, measurement-gated)

Each phase is gated on a measurement; later phases only run if the earlier numbers justify them.
Reuse the existing harness: `bench_device`, `bench_convexcell[_f32]`, the voro++ reference, the
Monte-Carlo cell validator (`test_convexcell_unit`), the OpenMP incremental test
(`test_incremental_device`), and ArborX (already a dependency) for BVH queries.

**Phase 0 — characterise the real workload (do this first; it sets every target).**
Drive a representative moving-point simulation (the vorflow Euler/NS driver, or a Lennard-Jones / SPH
proxy) and **measure, per step:** the fraction of cells whose **topology** changes vs. only geometry;
the cost split *clip : geometry : neighbour-query*; and how the topology-stable fraction varies with
time step / skin. **Deliverable:** the actual figure of merit (e.g. "98 % geometry-only at typical
dt"). *If topology is almost always stable, the whole problem becomes the geometry kernel and the
clip is a side-show — invert the priorities accordingly.*

**Phase 1 — geometry + derivatives over fixed topology (the common case).**
Implement a GPU kernel that, given cached topology + new positions, recomputes `V, A, dV, dA²` and
validate to the half-edge oracle **and** check **momentum conservation** (Σ forces = 0, drift over
10⁴ steps). Do it for **R2 compact-half-edge** (reuse the validated formulas) **and** the **R4
face-list / canonical-face** variant. **Key experiment:** does the *fully independent* face variant
conserve momentum acceptably, or is the canonical-owner scatter required? **Deliverable:** the
steady-state per-step throughput and the verdict on face independence — this decides R2 vs R4.

**Phase 2 — the re-clip (rare case), parallel + BVH.**
Replace the grid + isotropic-radius gather with an **ArborX BVH best-first / radius query +
directional culling** (the SOTA lever; closes our 2–3× gap to Ray/Basselin). Compare two clip
granularities on the unstable set: **G1 thread-per-cell ConvexCell** (have it) vs. **G3
warp-per-cell lane-per-face**. **Deliverable:** re-clip throughput vs. published SOTA, and whether
G3 beats G1 once the gather is BVH-based.

**Phase 3 — robustness.**
Add adaptive/exact orientation predicates (or SoS) to whichever clip wins Phase 2; stress on
cospherical/lattice/degenerate inputs and near-coincident points; confirm no cascade and bounded
volume error. **Deliverable:** a robustness suite the convex-clip passes that the half-edge already
passes.

**Phase 4 — incremental driver, end-to-end.**
A GPU "needs-reclip?" test (Verlet skin / per-cell security radius cached at last clip), stream-compact
the unstable cells, re-clip them (Phase 2), geometry-update everything (Phase 1). Compare per-step
cost to full rebuild and to the OpenMP incremental speedup. **Deliverable:** the end-to-end per-step
number that the whole effort is really about.

**Metrics throughout:** per-step update time (primary); topology-stable fraction; cold-rebuild
throughput; derivative accuracy vs. half-edge; momentum drift; robustness pass/fail on degenerate
sets; strong/weak scaling on CPU cores and GPU; footprint (cuobjdump reg/stack). Reference points:
voro++ serial, half-edge GPU 2.49 M/s, ConvexCell FP32 5.4 M/s, Ray-et-al ~12.5 M/s (V100).

---

## 6. A first recommendation (to be falsified by Phase 0–1)

If forced to bet before the measurements:

1. **Decouple topology from geometry.** Keep the **validated half-edge geometry+derivative engine** as
   the physics core, but feed it from a **compact, connectivity-only resident representation (R2)** so
   the GPU footprint drops to ConvexCell scale **without re-deriving the physics or losing robustness.**
   This is the lowest-risk way to get the occupancy win *and* keep everything that already works.
2. **Make the geometry/derivative re-evaluation the optimised hot kernel** (Phase 1), warp-parallel
   over faces with **canonical faces** for exact momentum conservation. This is the steady-state cost.
3. **Use ArborX BVH** for the neighbour query everywhere (Phase 2) — it is the measured 2–3× gap to
   SOTA and it is already a dependency.
4. **Treat the ConvexCell + lane-per-face clip (G3+R4) as the high-upside contender for the re-clip
   only**, adopted if Phase 2–3 show it beats the per-thread half-edge clip *and* the face variant
   passes the momentum/robustness tests. Its FP32 GPU speed is real; its physics and robustness are
   the open risks.

The non-negotiables that order everything: **derivatives + momentum conservation + robustness** (these
are why the half-edge exists and must not regress), and **the per-step update — not the cold rebuild —
is the number to minimise.**

---

## Phase 0 results (measured) — the workload, characterised

`tests/kokkos/phase0_incremental.cpp` (half-edge device tessellator; build at P0, displace every
point by a random vector of scale `frac·spacing` periodic, rebuild, compare each cell's sorted
neighbour set). RTX 5080, random uniform seeds. **N-independent** (identical at N=2e5 and N=1e6):

| rms displacement / spacing | topology-stable cells | mean faces flipped (changed cells) |
|---:|---:|---:|
| 0.002 | **94 %** | 1.04 |
| 0.005 | **86 %** | 1.09 |
| 0.010 | **73 %** | 1.18 |
| 0.020 | 54 % | 1.37 |
| 0.030 | 40 % | 1.57 |
| 0.100 | 6 % | 3.2 |

Mean faces/cell = 15.53; cold rebuild ≈ 2.4 M cells/s.

**Two decisive findings:**

1. **In the realistic regime the majority of cells are geometry-only.** Stable explicit dynamics keep
   the per-step displacement to a small fraction of the spacing (~0.001–0.01), where **73–94 % of cells
   keep their exact neighbour set** — they need only a *geometry+derivative re-evaluation*, no gather
   and no clip. This **confirms the §1 reframe** and quantifies it: the geometry kernel + a cheap
   needs-reclip test are the steady-state cost; the clip is the minority case (6–27 %).

2. **Cells that *do* change topology change by ~one face** (symmetric-difference ≈ 1.0–1.2 in the
   realistic regime — a single neighbour swap). So the unstable set does **not** need a full rebuild:
   a **local incremental repair** (drop one face, insert one face) suffices — far cheaper than
   re-clipping from the box. This is the GPU analogue of the legacy vorflow "incremental cell repair,"
   and it adds a **third tier** to the per-step model.

**Revised per-step cost model (and the speedup it implies).** Three tiers, not two:

```
per-step  =  (stable fraction)        × geometry-only update        [no gather, no clip]
           + (1-face-flip fraction)   × local repair                [drop/insert ~1 face]
           + (rare large change)      × full re-clip                [box clip + BVH gather]
```

With geometry-only ≈ 0.1× a full per-cell build (gather+clip dominate the build) and a single-face
repair ≈ 0.2×, the per-step cost at disp/spacing = 0.01 (73 % stable) is ≈ `0.73·0.1 + 0.27·0.2 ≈
0.13×` a full rebuild → **~8× speedup**, rising toward ~15–20× at disp/spacing = 0.002. This matches
the "significant speedup vs full rebuild" seen for the OpenMP incremental path, and says the GPU
target is the **geometry kernel and the repair**, not faster cold clipping.

**Consequences for the program:**
- **Phase 1 (geometry+derivative kernel) is now clearly the top priority** — it serves 73–94 % of
  cells every step. Optimise it hard (warp-parallel over canonical faces; bandwidth-bound).
- **Add a Phase 1.5: single-face local repair** on the resident topology (remove the lost face,
  insert the gained one, fix the local connectivity). This likely dominates the *clip* budget,
  demoting full re-clip (Phase 2) to a rare fallback — which in turn **lowers the stakes on matching
  cold-rebuild SOTA**: if full clips are <5 % of steps, a 2× rather than 5× clip is acceptable.
- **The needs-reclip test** is a per-cell skin criterion (cache each cell's security radius at last
  build; flag when a neighbour move could cross a face) — cheap, parallel, stream-compactable.

Net: Phase 0 inverts the priority list exactly as anticipated. The cold-rebuild contest (half-edge vs
ConvexCell vs SOTA) matters far less than the **geometry/derivative re-evaluation + single-face repair
on a resident, compact topology** — which also happens to be the part where the validated half-edge
physics can be reused directly (Recommendation 1 in §6).
