# Neighbour finding & moving-point cell updating in vorflow — method, legacy, literature, avenues

**Purpose & audience.** This is a **self-contained handoff** for someone (human or LLM) picking up the problem
of *efficiently maintaining a 3D Voronoi/Power tessellation of moving points* in vorflow. It documents (1) the
**currently implemented method** — the ConvexCell, the worklist gather, and essentially every optimisation
already in the code; (2) the **correct framing** of the moving-point problem (the gather / neighbour set is the
core, in the Verlet sense); (3) the **legacy pre-Kokkos CPU strategies** and what was promising; (4) a
**literature survey**; and (5) an **overview of avenues to investigate**. It is deliberately broad and
cross-references the other `docs/` notes (which hold the detailed measurements).

Companion docs (read for detail): `performance.md` (the optimisation ledger + profiles), `voronoi_build_plan.md`
(design), `voronoi_gpu_research_program.md` (the reframing + phased program + R/G design space),
`voronoi_worklist_gather_project.md` (the gather), `voronoi_construct_ledger.md` (every clip/cell-rep attempt),
`voronoi_pervertex_geometry_report.md` (sort-free geometry), `voronoi_cold_tessellation_benchmark.md` /
`voronoi_coldbuild_benchmark_report.md` (cold-build numbers), `voronoi_dynamic_update_study.md` (the recent
update-strategy study), `update_and_repair_redesign.md` (the legacy wave-repair design).

---

## 1. Problem & the correct framing

A Voronoi cell of seed *i* is the intersection of half-spaces, one per neighbour *j* (the bisector of *i,j*),
plus the box/SDF boundary. Two quantities, **very different update rates**:

- **Topology** — *which* seeds are neighbours and the face/edge connectivity. Changes **rarely** (only when a
  seed crosses a bisector — a "flip"). It is the output of the **clip**, the expensive part.
- **Geometry** — vertex coordinates, face areas, cell volumes, **and their derivatives** w.r.t. seed positions.
  Changes **every step** (any neighbour move shifts a bisector), but for *fixed topology* it is a closed-form
  re-evaluation — no clip, no neighbour search.

**The figure of merit is the per-step update, not the cold rebuild.** Phase-0 measurement
(`phase0_incremental.cpp`, RTX 5080, N-independent): at realistic per-step displacement (0.002–0.01 of the mean
spacing) **73–94 % of cells keep their exact neighbour set**, and cells that *do* change flip ≈ **one** face.
So the steady state is dominated by (a) a cheap "needs-reclip?" test, (b) a geometry+derivative re-evaluation
over fixed topology, and (c) a *local* repair of the few changed cells.

**The neighbour set / gather is the core problem — in the Verlet sense.** The hard, dominant cost is *finding
all the relevant neighbours*. In the cold build the gather is **~60 % of serial time** (`performance.md`), and
the gap to GPU SOTA is *entirely* the neighbour query (§5.5). The right way to think about the moving-point case
is the **Verlet-list view**: keep, per cell, a candidate set **guaranteed to contain every seed that could
become a neighbour** until a validity criterion is violated; while it holds you never re-gather (you re-clip /
re-evaluate from the stored set); when it breaks you re-gather. The open design question is **what that
guaranteed candidate set is and how to maintain it** — a geometric skin (distance), a topological neighbourhood
(k-ring), or a kinetic certificate (event time). This document is mostly about that question.

---

## 2. The currently implemented method (essentials)

Header-only Kokkos (CUDA/HIP/OpenMP from one source). Key files:
`include/vorflow/device/convex_cell.hpp` (the cell), `…/tessellator.hpp` (the build/gather + CSR),
`…/topology_store.hpp` (resident topology, new), `…/transpose.hpp` (reciprocal-facet maps),
`include/vorflow/tessellation_view.hpp` (the published CSR consumers read).

### 2.1 ConvexCell — the cell representation (`convex_cell.hpp`)

The cell is stored in the **dual** (Ray/Sokolov/Lefebvre/Lévy, geogram `VBW::ConvexCell`): each primal **vertex
is a triple of plane indices** (`t0,t1,t2`, one byte each), with the vertex coordinate **cached** per triangle
(`vx,vy,vz`). A plane is stored as its **foot-point normal** `n` with interior half-space `{x : n·x ≤ n·n}`, so
`x=n` is the foot of the perpendicular from the seed (origin) and `|n|` the seed→plane distance; for Voronoi
`n = ½(p_j − p_i)` and the connector `r = 2n`. `pnbr[k]` is the neighbour seed id of plane *k* (`<0` = box).
Frame: ~3.1 KB FP32 (vs the retired half-edge ScratchCell's ~20–32 KB) — this collapse is the GPU-occupancy
lever (51 reg / 3.5 KB → high occupancy; the half-edge was 142 reg / 32 KB ≈ 29 % occupancy).

### 2.2 The clip (`ConvexCell::clip`)

Clipping by a new plane *p* (geogram/Ray convex-cell clip): mark dual triangles whose cached vertex is outside
*p* (`n_p·v > n_p·n_p`); collect the **horizon** (edges shared between a killed and a kept triangle) by a small
linear scan (`findSharing` — no stored adjacency; the cell is tiny); add one new triangle `(x,y,p)` per horizon
edge. **Half-space intersection is order-independent**, so cuts are applied **on the fly with no candidate
buffer** — the cell is the only per-thread state. The plane is only *committed* if it actually cuts (redundant
candidates don't grow `np`). A per-clip security radius `secR2 = 2·maxVertexRsq` shrinks as the cell tightens.
**This clip is SOTA** (serial 8.7 µs/cell, beats voro++ 9.5 and geogram 10.6 on the same neighbours).

### 2.3 Sort-free, adjacency-free per-vertex geometry + derivatives

Volume, face-area vectors, centroids/first-moments, the volume gradient `dV/dn`, and the full area-Jacobian
`dA_k/dn_l` are all computed by a **pure per-vertex scatter** (divergence theorem, coning each boundary flag to
the origin) — **no vertex ordering, no adjacency, no atan2, no findSharing** (Ray TOG 2018 trick; details in
`voronoi_pervertex_geometry_report.md`). Production kernels: `volumePerVertex`, `geometryPerVertex` (V + area +
moment in one pass), `geomVolumeGrad` (V + dV/dn, sqrt-free, 3-array), `geomVolumeArea` (V + area-vectors +
dV/dn), `geomVolumeAreaGrad` (the coupled Hessian, analytic, with an `…AD` forward-AD oracle). All validated to
machine precision against an ordered/AD oracle. The forces follow: `dV_k = (area_k/|r_k|)(r_k − centroid_k)`.
A `plane_policy.hpp` layer separates the geometry (a function of `{n_k}`) from the plane-from-DOFs map + its
Jacobian, so Power/Spheres/SDF become *policies*, not new geometry kernels (Phase 3, partial).

### 2.4 `reevalGeometry` — the fixed-topology fast path

`ConvexCell::reevalGeometry(seed, pos, L)`: after the seeds move, **reuse the resident topology**
(`pnbr` + the dual-triangle structure), rebuild each neighbour plane from the neighbour's new position
(`n = ½r`), recompute every live vertex (Cramer), then geometry. No gather, no clip. Benchmarked ~**85–190×
cheaper than a full rebuild** (the geometry is nearly free). This is the engine of every incremental strategy.
`isSelfConsistent(tol)` is the **D2 "convexity" certificate**: flag the cell iff a re-evaluated vertex now pokes
outside one of the cell's *other* planes (a lost-face signature; its 3 defining planes are excluded).

### 2.5 The worklist gather — the neighbour query (`tessellator.hpp`)

This is the part the user's reframing centres on. A **voro++-style precomputed worklist** (the work that took
the cold build to SOTA on both backends; `voronoi_worklist_gather_project.md`):

- **Counting-sort grid** (`kSeedsPerCell`: 1 on GPU, 2 on host) bins seeds; outputs are written back at the
  original index so `cell i == seed i`.
- **Per-sub-position worklist**: each grid cell is split into `wlS³` sub-positions; for the sub-region a seed
  lands in, a **table of block offsets `(dx,dy,dz)` sorted by nearest-corner distance² (`wlRmin`)**, packed
  into one int. The gather walks this presorted list and **breaks once `wlRmin > 2·secR2`** (sorted ⇒ all
  remaining blocks are too far) — a *table lookup, no per-block geometry at runtime*. A per-candidate cull
  (`off ≥ secR2 ⇒ skip`) removes no-op clips before the O(#tri) side-test.
- **Periodic ±L shift folded per block** (branchless), minimal-image relative vectors.
- **Completeness guard**: `exhausted=K` counts cells that drained the worklist without the radius break;
  measured `K=0` over operating densities ⇒ provably complete (no fixed-window blind spot).

**Why this is the SOTA-getting gather:** the radius break + sorted worklist is exactly voro++'s "no per-block
distance math" trick. It made serial CPU reach voro++ parity and GPU FP32 1.28× the Liu-2020 code. **But it is
still a uniform-grid, isotropic-radius query** — and that is the remaining gap to the published GPU SOTA (§5.5):
~85 % of the clips it issues are no-ops (candidate within the isotropic radius but in a direction where the cell
is already tight), which a *directional* cull inside a BVH traversal removes.

### 2.6 Full pipeline & the optimisation ledger (all bit-exact unless noted; `performance.md`)

`buildTessellation<Real,Weighted>(pos, …)` → grid → worklist gather + clip per cell (one thread per cell,
RangePolicy) → per-facet CSR (`TessellationView`: `cellVolume`, `facetNeighbor`, `facetArea`, `facetConnect`=dV,
`facetConnVec`=r). Optimisations in place:

1. **Adaptive expanding / radius-break gather** — most cells close at the inner shells.
2. **Grid-order reorder** — build in grid-sorted order; gather reads contiguous memory (large-N cache win).
3. **Backend-aware density** — 2 seeds/cell CPU (cache), 1/cell GPU (latency-hidden), 1/cell Power.
4. **Morton (Z-order) grid indexing — GPU only** (+18 % with forces via coalescing; *regresses* CPU because
   Morton codes are non-additive → re-encode in the hot loop; linear index kept on CPU).
5. **One-pass CSR fusion** — atomic-reserved contiguous CSR range per cell (no temp slab, no scan, no
   compaction); the single biggest serial win (closed the last ~10 % vs voro++). `cellFacetOffset`=per-cell
   base + explicit `cellFacetCount`.
6. **Force-geometry opt-in** (`withForceGeom`) — pure tessellation skips dV (apples-to-apples with voro++).
7. **Per-candidate cull** — halved clips/cell ~109→~50, FP32 4.37→5.42 M/s.
8. **Sort-free per-vertex geometry** (§2.3) — retired the atan2/face-ordering hot path; G1 14→17, G2 12→16 M/s.
9. **Over-buffer sizing + interleaved pack/free** — N×18 facet estimate + atomic overflow guard; N=4–6M now
   fit a 16 GB GPU (was OOM past ~3M).
10. **Topology-store + skin-list emission (new, opt-in, additive)** — `buildTessellation` can now write the
    `TopologyStore` (np/nt/pnbr/tri) and a per-cell candidate (skin) list, the persistence layer for the
    moving-point loop. Default-off so existing callers are unaffected (production tests green).

**Dead ends already measured (do not repeat):** team/warp-per-cell (~0.6×; the clip is sequential, 31/32 idle);
cooperative cut (~1 % slower; per-cut barriers cost the saved dot-products); whole-block-accept on the fine grid
(net loss); coarse-grid worklist (slower + inexact); triangle adjacency (CPU +25 %, **GPU −36 %**); incremental
security radius (GPU flat); candidate-cap shrink (bounded by full-sphere need); packed/recompute cell (0.72×);
best-first traversal *for the cold build* (0.22× — over-gather is cheap, fused divergent priority queue is not);
no-op-clip AABB cull as a separate pass (≈0). The clear lesson: **on GPU only *reducing work* helps; making each
op cheaper does not** (spare ALU, bound by frame/memory/divergence).

### 2.7 Current standing

- **Cold build:** serial CPU ≈ voro++ (1.0–1.05×); 24-core ≈ 13× voro++; **GPU FP32 ≈ 6–7 M cells/s** (≈ 1.1–
  1.3× the Liu-2020 GPU code; ~2–3× below Ray/Basselin SOTA — the gap is the neighbour query, §5.5).
- **Moving-point update** (`voronoi_dynamic_update_study.md`, vs the *production* `buildTessellation` rebuild,
  disp 0.001 of spacing): pure re-eval 26× (GPU)/90× (host) but inexact; convexity+independent local repair
  3.7× (GPU)/9× (host) near-exact; convexity+propagating repair 1.6×/4.2× exact; **reclip-all (the textbook
  Verlet usage, no detection) only 1.05× (GPU)/3× (host)** — because the production worklist is *clip-bound*, so
  reclipping every cell costs ~a rebuild. **Conclusion: the win comes from re-clipping only the *changed* cells,
  i.e. the detector + a maintained candidate set earn their keep; blanket reclip does not.** The speed-up scales
  with rebuild cost, so it is modest on the (fast) GPU and larger on host.

---

## 3. The Verlet-list view of neighbour maintenance

The canonical Verlet/cell-list pattern (MD, §5.4): store per particle a neighbour list gathered within
`r_cut + skin`; **rebuild the list only when some particle has moved > skin/2** from its list-build position
(then no neighbour can have entered `r_cut` undetected). Between rebuilds you iterate the stored list.

For Voronoi the analogue is: store per cell a **candidate set guaranteed to contain every future Voronoi
neighbour** until a criterion breaks; between rebuilds, re-clip / re-evaluate from it; on break, re-gather.
Three concrete realisations of "guaranteed candidate set", in increasing sophistication:

- **(V1) Geometric skin** — the seeds within `2·R_secure + skin` of the cell (R_secure = max vertex distance).
  Simple; rebuild when max displacement > skin/2. This is what the current study harness emits (the worklist's
  examined security ball), and its measured **exact budget is small (~0.03 of the spacing)** before a gained
  neighbour falls outside — to extend it, record a wider security reach. The skin trades rebuild frequency vs
  gather cost; on a *clip-bound* build (GPU) it buys little because reclip dominates regardless.
- **(V2) Topological k-ring** — the candidate set is the **1-ring (current faces) + 2-ring (neighbours of
  neighbours)**. By the Delaunay property a *new* Voronoi neighbour is (almost always) a neighbour-of-a-current-
  neighbour, so the 2-ring is the natural guaranteed set, and it is **maintained by the repair itself** (when a
  cell gains/loses a face, promote/demote and append the new neighbour's 1-ring). The spatial grid is queried
  **only as a fallback** (empty candidate list / large jump). This is the legacy `ConnectivityArena` design
  (§4.3) and is likely the *right* Verlet structure here — it never re-gathers from the grid in the common case
  and is displacement-magnitude-independent (a flip is a flip regardless of how far it moved).
- **(V3) Kinetic certificate** — write the bisector-crossing predicate as a function of time and compute the
  **event time** at which each cell's topology next changes; process events in order (§5.3). No skin, no polling;
  exact. Heavier machinery, classically CPU/serial, but the most principled.

The current code uses (V1). The legacy CPU work pointed at (V2). (V3) is unexplored here.

---

## 4. Legacy (pre-Kokkos) CPU strategies & profiling

The pre-Kokkos vorflow was an OpenMP, half-edge, data-oriented-design (DOD) CPU code. Several pieces survive or
are documented; they were **promising on CPU** and encode ideas worth porting.

### 4.1 `NbrList` — cell-linked-list neighbour finder (`include/vorflow/nbrlist.hpp`, still present)

A classic **cell-linked list** over a uniform `Grid` (+`Box`/`BoxLE` for periodic & Lees–Edwards minimal image):
`setup(pos, rcut)` bins particles by a counting sort (serial and an OpenMP per-thread-counts variant), storing
`PosAndId` contiguously per cell (`m_cell2Pos` + `m_headCell` CSR). Queries: `getNbrs` (27-cell 1-ring),
`getDirectNbrs` (6-face), `getGridNbrs` (8-cell octant). Notable extras: **`setupSubset(pos, ids, rcut)`** —
build the grid over *only a subset* of particles (the natural primitive for re-gathering just the cells that need
it), and **`setupCurrentTeam`** — build the grid *inside* an existing OpenMP parallel region (team-local scratch,
barriers) to avoid nested-parallel overhead. The `rcut`-sized grid (`n = floor(L/rcut)`) is the cell-list with
the cut radius baked into the cell size — the Verlet/cell-list idea in its CPU form.

### 4.2 `host/incremental.hpp` — the Verlet-skin incremental update (DELETED, documented)

The OpenMP incremental path (retired with the half-edge ScratchCell, but it is the proven precedent): **skip the
clip for topology-stable cells, recompute only geometry**, with a Verlet-skin "needs-reclip?" test; bit-exact to
a full rebuild. This is exactly the §2.4 `reevalGeometry` idea, and it delivered the "significant speedup vs
rebuild" on CPU that motivated the whole moving-point reframe. (Its device re-implementation is the current
study; the CPU version's reports are the precedent that it works.)

### 4.3 Wave-based BFS repair + `ConnectivityArena` (1-ring/2-ring) — `docs/update_and_repair_redesign.md`

The legacy **dynamic-update design** (a DOD plan; the most relevant artifact for the user's reframing). Two-path
pipeline:

- **Fast path:** recompute geometry for every cell on the new positions; if the cell stays **convex**, commit
  geometry; if it inverts/goes non-convex, push its id to an `active_queue` (this is the D2 convexity certificate
  again).
- **Slow path — iterative wave (BFS) repair:** while `active_queue` not empty, in parallel per active cell: fetch
  its **1-ring + 2-ring candidates from a `ConnectivityArena`** (only query the spatial grid if the candidate
  list is empty), re-clip, then apply **topological wave rules** — *Rule A (lost neighbour):* an old 1-ring
  neighbour no longer in the cut → enqueue it; *Rule B (promoted neighbour):* a 2-ring candidate became a 1-ring
  neighbour → enqueue it and append *its* 1-ring to this cell's 2-ring pool. Merge thread-local next-queues under
  an OMP barrier, dedup, swap. Memory hygiene: `ChunkedPool` free-list (recycle overflow chunks, no leaks) and
  **generation counters** for "visited" (O(1) reset, no `memset` per cell).

This is **(V2) — the topological Verlet list** done properly: the 2-ring *is* the guaranteed candidate set, it is
maintained incrementally by the wave (promotion appends the new neighbour's ring), and the grid is a fallback
only. It is the GPU analogue of "process only the cells whose natural-neighbour identity changed" from the
kinetic-Delaunay literature (§5.3). The current device study's S4 "propagating repair" is a *cruder* version of
this (it re-gathers a geometric skin and propagates by neighbour-set asymmetry, not a maintained 2-ring).

### 4.4 Legacy DOD-storage precedent

The retired data-oriented-design (DOD) storage experiments (separate Topology/Connectivity/Geometry arenas, a
chunked/arena pool, and per-cell capacity tuning around vertex≈28/facet≈20) are the structural precedent for
keeping a *resident, compact, connectivity-only* representation between steps (the R2 idea in §6). Those
standalone reports have since been removed; the idea now lives in the ConvexCell `TopologyStore`
(`include/vorflow/device/topology_store.hpp`).

---

## 5. Literature survey

### 5.1 Convex-cell clipping family (the method vorflow implements)

- **Ray, Sokolov, Lefebvre, Lévy, "Meshless Voronoi on the GPU," ACM TOG 37(6), 2018** — the dual ConvexCell +
  per-vertex geometry this code uses; ~12.5 M cells/s on a V100 (volume-only). https://doi.org/10.1145/3272127.3275092
- **Basselin, Alonso, Ray, Sokolov, Lefebvre, Lévy, "Restricted Power Diagrams on the GPU," CGF (Eurographics)
  2021** — same clipping, power/restricted diagrams. https://doi.org/10.1111/cgf.142610
- **"Scalable GPU Construction of 3D Voronoi and Power Diagrams," arXiv:2605.06408 (2026)** — *the current SOTA
  and the key reference for the gather*: replaces the isotropic-radius criterion with a **directional culling
  criterion from the cell's evolving axis-aligned bounds**, combined with a **best-first BVH traversal**
  augmented for power-distance queries; on par with SOTA Delaunay at small/moderate N and scales to large/diverse
  point sets. https://arxiv.org/abs/2605.06408
- **Liu et al., "Parallel Computation of 3D Clipped Voronoi Diagrams," 2020** — the runnable GPU
  ConvexCell+grid-kNN code we benchmark against (built on the RTX 5080). https://github.com/xh-liu-tech/3D-Voronoi-GPU
- **"In Search of Empty Spheres: 3D Apollonius Diagrams on the GPU," ACM TOG 2025** — GPU additively-weighted
  (sphere) Voronoi, relevant if polydisperse/sphere cells matter. https://doi.org/10.1145/3730868
- **geogram / Lévy** — the CPU `VBW::ConvexCell` reference (we benchmark against it); also the semi-discrete
  optimal-transport lineage that already computes cell volume/area **gradients** w.r.t. seeds (our derivatives).

### 5.2 GPU Delaunay (the alternative, generally slower here)

- **Nanjappa, gDel3D / gStar4D; Cao, Nanjappa, Gao, Tan, "A GPU accelerated algorithm for 3D Delaunay
  triangulation," I3D 2014** — massively-parallel insertion + flipping, then **star-splaying** repair (Shewchuk)
  of the nearly-Delaunay result. https://www.comp.nus.edu.sg/~tants/gdel3d.html — `gReg3D` does the regular
  (weighted) triangulation. The 2026 paper beats gDel3D at ≥5 M points, so clipping > Delaunay for our use.
- **Shewchuk, "Star splaying: an algorithm for repairing Delaunay triangulations and convex hulls"** — the
  canonical *repair-after-vertices-moved* primitive; linear-time when input is nearly-Delaunay. The model for a
  topological wave repair.

### 5.3 Kinetic / dynamic Delaunay & topology-change detection (moving points)

- **Lulli et al., "GPU Based Detection of Topological Changes in Voronoi Diagrams," Comp. Phys. Comm. 2017
  (arXiv:1607.00908)** — *directly on point*: CUDA kernels that detect topological changes of moving-point
  Voronoi diagrams (built for plastic events in LBM/soft-glassy sims). The per-vertex predicate is the rigorous
  version of our convexity certificate (empty-orthosphere / in-sphere).
- **Russo, Lambrakos, et al., "Kinetic and dynamic Delaunay tetrahedralizations in 3D," CPC 2004
  (arXiv:physics/0302018)** — simplex-flip algorithms, visibility-walk point location, vertex insert/delete;
  *"only sample points whose natural-neighbour identity changes need an update"* — the topology event.
- **Guibas–Russel, kinetic data structures**; **"An Empirical Comparison of Techniques for Updating Delaunay
  Triangulations" (SoCG 2004)** — survey of update techniques (rebuild vs local flips vs KDS). KDS = rewrite the
  geometric predicate as a function of time, solve for the event time when it flips — the (V3) certificate.
- **arXiv:1304.3671** (topological changes of moving-point DT, bounds), **arXiv:1312.2194** (near-quadratic
  bound). Delaunay's lemma (locally regular ⟹ globally regular) is the theoretical basis for *local* certificates.

### 5.4 Verlet / cell lists for moving particles (the maintenance pattern)

- The **Verlet list + skin** pattern and the **move > skin/2 rebuild criterion** (measured w.r.t. the
  list-build positions, not the previous step) — standard MD; tradeoff = excess pairs vs rebuild frequency.
- **GPU/parallel Verlet lists:** "Parallel Verlet Neighbor List Algorithm for GPU-Optimized MD"
  (https://users.wfu.edu/choss/docs/papers/21.pdf); **LAMMPS** cell-list-builds-Verlet; **HOOMD-blue**.
- **Awile et al., "Fast neighbor lists for adaptive-resolution particle simulations," 2012**
  (https://publications.mpi-cbg.de/Awile_2012_5148.pdf) — cell lists with variable cut radius.
- **"GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data Layout," arXiv:2602.19873** — recent,
  SFC layout for GPU neighbor lists (relevant to our Morton ordering of the gather).

### 5.5 The SOTA neighbour query — what we are missing

The measured 2–3× gap of our 6–7 M/s to Ray/Basselin is **the neighbour query**: we use a uniform grid +
**isotropic** radius (so ~85 % of issued clips are no-ops); the SOTA uses a **BVH best-first traversal +
directional culling from the cell's evolving AABB** (arXiv:2605.06408). **ArborX (already a vorflow/DEM
dependency) provides GPU BVH for free.** Our own earlier best-first attempt failed *for the cold build* (fused
divergent priority queue), but that was without directional bounds and as a cold-build fusion — the SOTA result
says the directional cull is the missing ingredient, and best-first is most natural for the *incremental*
small-candidate-set case anyway.

---

## 6. Avenues to investigate (the alleys)

Ordered roughly by expected value × certainty. Each is independent; most are gated by a measurement.

**A. Directional culling + BVH best-first gather (close the cold-build SOTA gap, and speed every re-gather).**
Replace the grid+isotropic-radius query with an **ArborX BVH** traversed best-first with the **directional AABB
cull** of arXiv:2605.06408. This is the single measured 2–3× cold-build lever, and because the gather dominates
(~60 %), it also speeds every Verlet rebuild and every local re-gather. Risk: BVH build cost per step for moving
points (amortise via refit, or only rebuild the BVH on the Verlet trip). *This is the most defensible big win.*

**B. Topological Verlet list (V2 / `ConnectivityArena`) instead of a geometric skin.** Maintain per cell the
**1-ring + 2-ring** candidate set; re-clip a changed cell from it (no grid); propagate via promote/lost wave
rules; query the grid/BVH only as a fallback. This is displacement-independent (a flip is local regardless of
step size), avoids the small geometric-skin budget (~0.03 spacing) entirely, and is the legacy CPU design that
"seemed promising." On GPU it needs a stream-compacted wavefront BFS (the current S4 is a crude proxy). *This is
the right Verlet structure for this problem and the most promising neighbour-maintenance idea.*

**C. Kinetic / event-driven updates (V3).** Compute, per cell or per bisector, the **time of the next topology
event** (predicate-as-function-of-time) and process events in order — no polling, no skin, exact. Heaviest
machinery and classically serial; worth a literature-deep feasibility note for GPU before committing. Could be
hybridised: kinetic certificates as the exact "needs-reclip?" test feeding B's wavefront.

**D. Three-tier incremental driver (the per-step model from `voronoi_gpu_research_program.md`).**
`geometry-only (stable) | single-face local repair (1-flip) | full re-clip (rare)`. Phase-0 says changed cells
flip ~1 face, so a *single-face* repair (drop one face, insert one) is far cheaper than re-clipping from the box;
this demotes full re-clip to a rare fallback and **lowers the stakes on cold-build SOTA**. Pair with B's candidate
set. *Highest leverage on the actual steady-state cost.*

**E. Compact half-edge (R2) — keep the validated physics, get the occupancy.** Keep the **connectivity only**
resident (labels/neighbour ids), recompute coordinates+geometry each step using the *already-validated* half-edge
derivative formulas. Footprint drops to ConvexCell scale **without re-deriving/​re-validating physics or losing
robustness**. The lowest-risk way to get GPU occupancy + proven derivatives + momentum conservation. (Currently
the ConvexCell path re-derived geometry from scratch; R2 is the conservative alternative.)

**F. Face-list (R4) + lane-per-face granularity (G3).** Represent each Voronoi face `(i,j)` as an independent 2-D
convex polygon (bisector clipped by other half-spaces); a warp does ~32 faces at once — no sequential cut, tiny
per-lane state, robust (each face is an isolated convex clip). Cell quantities are reductions over faces.
**Open risk: momentum conservation** needs `A_ij = A_ji` exactly — compute each face once at a canonical owner
`min(i,j)` and scatter (the GPU analogue of `buildReciprocalMap`). High upside for the re-clip throughput; the
face-independence/momentum experiment is make-or-break.

**G. Robust predicates.** The convex-clip dual triangulation is fragile at ≥4 cospherical seeds (dead-triangle
cascade). For a production moving-point engine add adaptive/exact orientation (Shewchuk) or simulation-of-
simplicity, as the SOTA papers do. The half-edge had an explicit round-off fallback; ConvexCell needs an
equivalent. Topology-oriented (Sugihara) valid-by-construction is the chosen philosophy.

**H. Distributed / MPI.** The cold path already supports it (`mpi/voronoi_halo.hpp`, ghost gather within the
security radius). The moving-point loop composes: rebuild = migrate + gatherGhosts + build; between rebuilds,
halo-update ghost positions only and re-evaluate (the Verlet skin also bounds migration). Not yet wired to a
device stepper.

**Cross-cutting:** the value of *any* incremental scheme **scales with the cold-rebuild cost** — modest on the
fast GPU (~6 M/s), larger on host and for expensive cells (polydisperse, with force-geometry). And **on GPU only
work-reduction helps** (the directional cull, fewer reclipped cells), not cheaper per-op. The non-negotiables
that must not regress: **derivatives + momentum conservation + robustness**, and the figure of merit is the
**per-step update**, not the cold rebuild.

---

## 7. Quick reference

- **Cell / geometry:** `include/vorflow/device/convex_cell.hpp` (clip, reevalGeometry, isSelfConsistent,
  geometryPerVertex/geomVolume*); `plane_policy.hpp` (Voronoi/Power/SDF policies + chain-to-DOFs).
- **Build / gather / CSR:** `include/vorflow/device/tessellator.hpp` (worklist gather, CSR, opt-in store+skin
  emission); `tessellation_view.hpp` (consumer CSR + `buildReciprocalMap`); `transpose.hpp` (device aux maps).
- **Resident topology:** `include/vorflow/device/topology_store.hpp`.
- **Legacy:** `include/vorflow/nbrlist.hpp` (cell list), `include/vorflow/voronoi.hpp` (legacy CPU half-edge
  CellMaker oracle), `docs/update_and_repair_redesign.md` (wave-BFS + ConnectivityArena).
- **Benches/tests (`tests/kokkos/`):** `bench_convexcell[_f32]` (cold build + voro++), `bench_incremental`
  (re-eval vs rebuild), `bench_update_strategies[_f32]` (the strategy study S0–S6), `phase0_incremental`
  (topology-stability vs displacement), `test_pervertex_geometry` / `test_device_geometry` / `test_tessellator`.
  `extern_bench/` builds geogram + Liu-2020 for head-to-head. Build: `cmake -S . -B build/<cfg> -DVORFLOW_KOKKOS=ON
  -DCMAKE_PREFIX_PATH=…/extern/install/<backend>`; CUDA on PATH for the GPU build.
- **Key knobs (env):** `CC_DENS`, `CC_GATHER`, `CC_WLS`, `VORFLOW_TEAM`, `VORFLOW_PROFILE`, `VORFLOW_NOFORCEGEOM`.

## 8. Open questions for whoever picks this up

1. Which Verlet structure: geometric skin (V1, simple, small budget), **topological 2-ring (V2, recommended)**,
   or kinetic (V3, exact, heavy)?
2. Is the BVH+directional-cull gather (avenue A) worth the per-step BVH cost for moving points (refit vs rebuild)?
3. R2 compact-half-edge (keep validated physics) vs R3 ConvexCell (GPU-fast, re-derive) vs R4 face-list — decide
   per the momentum-conservation + robustness experiments, not on cold-build speed alone.
4. Does the fully-independent face variant (R4/G3) conserve momentum, or is canonical-owner scatter required?
5. The steady-state target is the **three-tier per-step update** (D); cold-build SOTA matters only for the rare
   full re-clip.
