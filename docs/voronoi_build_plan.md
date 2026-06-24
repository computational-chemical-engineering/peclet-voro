# Plan — GPU Voronoi for physics: build engine first, then moving particles

Companion to [voronoi_gpu_research_program.md](voronoi_gpu_research_program.md) (the analysis +
Phase-0 findings) and [performance.md](performance.md) (the measured GPU/CPU numbers). This is the
**actionable plan**. It reflects two decisions:

1. Phase 0 showed the steady-state workload is the *per-step update* (73–94 % of cells geometry-only,
   the rest a ~1-face local repair). **But** the cell-construction engine is **shared** by the initial
   build, the local repair, and the rare full re-clip — so optimising it pays everywhere, and it is
   the "always relevant" core. We do it **first**.
2. The engine is **two separable stages**, and the second one factors further. Getting these
   separations right *is* the architecture.

---

## Architecture: three separated concerns

```
   positions ──▶ (N) NEIGHBOUR QUERY ──▶ candidate planes ──▶ (B) CELL CONSTRUCTION
                                                                   │
                                                          (T) TOPOLOGY  ──┐  resident, compact
                                                                   │      │
                                                          (G) GEOMETRY ◀──┘  recomputed on demand,
                                                              tiered           at the tier needed
```

**(N) Neighbour query — find candidate neighbours of a seed.**
- *Most relevant for the initial build*, and **different for repairs** (a repair only re-queries near
  the moved seed; the existing neighbour list is mostly reused).
- Pluggable + independently benchmarkable: uniform grid (have it; loose isotropic radius) vs **ArborX
  BVH** best-first + directional/security pruning (already a suite dependency; the SOTA lever).
- Output: candidate seed ids + their bisector planes, ideally distance-ordered/pruned.

**(B) Cell construction — build the cell from a set of candidate planes. The shared core.**
- Key insight (the user's): this is **always an update**. The initial build is "update an empty cell
  seeded with the bounding cuboid"; a repair is "update the existing cell with the few changed
  planes." **Same algorithm.** So the engine must be written as *update(topology, planes)*, never
  assuming a cuboid start. This is the single most important design constraint — it is the bridge from
  cold build to repair (Phase 1.5).
- **More important than (N)** because it runs in every mode (build, repair) and is the part we can
  optimise by data-structure choice.

Inside (B), separate **topology** from **geometry**:

**(T) Topology — the connectivity** (which faces/edges/vertices, how they link). Built/mutated by the
clip; stored **compact and resident** between steps. Candidates:
- *Half-edge* — explicit links; good for **local repair** (drop/insert a face needs traversal) and the
  validated CPU robustness (round-off fallback). Larger.
- *Dual triangle* (ConvexCell) — vertex = plane-index triple; tiny footprint; good for the **parallel
  GPU clip**; weaker for local edits + degeneracies.
- These are the two ends; the choice may differ by backend (the codebase already keeps backend knobs).

**(G) Geometry — derived from (T) + positions, computed on demand and TIERED** (compute only what the
application needs; not all data is always wanted):
- **G0 — vertex coordinates** (intersection of each face/vertex's defining planes). Simplest apps.
- **G1 — face areas + cell volume** (from the vertices).
- **G2 — derivatives** `∂V/∂x`, `∂(area²)/∂x` w.r.t. seed positions (chain rule through the planes).
  The physics tier; this is what the half-edge `computeGeometry`/`gradFacetAreaSq` already produce.

Why tier (G): for moving particles (G) is re-run every step at the needed tier while (T) is reused;
for a cold build only the requested tier is paid. SoA layout, one pass per tier.

---

## Part I — the cell-construction engine (DO NOW, before moving particles)

Optimise the cold build with the separation above. Each step is measurement-gated.

**F1 — Separate the stages; make (B) an `update(topology, planes)`.**
Refactor the device build into three composable pieces: a neighbour-query producing candidate planes,
a topology clip that *updates* a cell (cuboid-seeded for the cold build), and geometry passes over the
topology. Verify the cold build = "update an empty cell from the cuboid" — bit-exact to today.
*Deliverable:* clean interfaces + per-stage timing (today they are fused).

**F2 — Topology representation A/B (the clip).**
Benchmark **half-edge** vs **dual-triangle** topology on the cold clip: footprint (cuobjdump
reg/stack), throughput, and — critically for Phase 1.5 — **how cheaply each supports a single-face
local edit**. We already have endpoints (half-edge 2.49 M/s @32 KB; ConvexCell FP32 5.4 M/s @3.5 KB);
this step adds the *repair-friendliness* axis and a decision (possibly per backend).
*Deliverable:* a topology engine with the best speed/footprint/repair trade-off, and the verdict.

**F3 — Tiered geometry passes (G0/G1/G2) over the topology.**
Implement geometry as separate passes reading (T) + positions: G0 vertices, G1 areas+volume, G2
derivatives. **Measure each tier's cost** (we have hints: ConvexCell volume ~8 %; half-edge geometry
15–37 %). Validate G1 vs voro++ and G2 vs the half-edge oracle. This module is reused verbatim by
Phase 1.
*Deliverable:* a tiered geometry kernel + the cost of each tier (so callers pay only for what they use).

**F4 — Neighbour query, separated + BVH.**
Pull (N) out as its own benchmark; compare grid + isotropic radius vs **ArborX BVH** best-first +
directional culling (closes the measured 2–3× gap to Ray/Basselin and is reusable by the repair's
local re-query). Feed both into the same (B).
*Deliverable:* a pluggable neighbour query + the build-vs-query cost split, and which wins.

*Exit criterion for Part I:* a cold-build pipeline, `N → B(update from cuboid) → G(tier)`, that is
(a) faster / smaller-footprint than today on GPU, (b) ≥ voro++ on serial CPU, (c) with topology and
tiered geometry cleanly separated so Part II can reuse them.

---

## Part II — moving particles (builds on Part I)

- **Phase 1 — geometry + derivative re-evaluation over fixed topology.** = re-run F3 (G2 tier) each
  step on the resident (T); validate vs the half-edge oracle **and momentum conservation** (Σforces=0,
  drift over 10⁴ steps); compact-half-edge reuse vs canonical-face variant. *Serves 73–94 % of cells.*
- **Phase 1.5 — single-face local repair + needs-reclip test.** Per-cell skin criterion (cache the
  security radius at last build) → stream-compact flagged cells → `update(topology, changed planes)`
  via F1/F2 (drop the lost face, insert the gained one). Phase 0: ~1 face/cell, so this absorbs most
  of the clip budget.
- **Phase 2 — full re-clip (rare fallback).** F4 BVH gather + F2 clip from the cuboid, for the few
  cells with large change. Lower stakes: if <5 % of steps, a 2× (not 5×) clip is acceptable.
- **Phase 3 — topology-oriented (Sugihara) robustness.** Valid-by-construction, FP as guide — **not**
  exact/Shewchuk predicates; no requirement that the Delaunay dual be consistent (see the
  `robustness-topology-oriented-sugihara` memory). Stress degenerate/cospherical/lattice inputs.
- **Phase 4 — end-to-end incremental driver.** Wire the three tiers; per-step cost vs full rebuild and
  vs the OpenMP incremental.

---

## Metrics & harness (throughout)

Reuse `bench_device`, `bench_convexcell[_f32]`, `phase0_incremental`, the voro++ reference, the
Monte-Carlo validator (`test_convexcell_unit`), the incremental test (`test_incremental_device`), and
ArborX for BVH. Track: cold-build throughput + per-stage split (N/B/G); per-tier geometry cost;
footprint (reg/stack); serial-CPU vs voro++; thread scaling; (Part II) per-step update time,
topology-stable fraction, derivative accuracy, momentum drift, robustness on degenerate sets.
Reference points: voro++ ≈ 75 k/s serial; half-edge GPU 2.49 M/s; ConvexCell FP32 5.4 M/s;
Ray-et-al ≈ 12.5 M/s (V100, the SOTA family — clip + BVH + directional culling).

**Non-negotiables that order everything:** derivatives + momentum conservation + topology-oriented
robustness must not regress (this is why the half-edge exists), and the per-step update — not the cold
rebuild — is the ultimate number to minimise. Part I optimises the shared engine that both depend on.

---

## F1 results — stages separated, per-stage split measured

**The structure was already there.** Both engines are `init(cuboid) → update(plane)* → geometry`:
the half-edge is `initCuboid` + a sequence of `cutCell2` (each clips by one plane — *already an
update*); the ConvexCell is `initBox` + `clip`. So the **"always an update" formulation is native** —
a cold build *is* "update an empty cell from the cuboid," and a repair will be "update the existing
cell with the changed planes," same code. **(N) and (B) are deliberately coupled** by the security
early-out (you must clip to shrink the cell to know when to stop gathering); **(G) is cleanly
separable** (skip it and the build still produces correct topology + vertices).

**Per-stage split (cold build, N=1M, measured):**

| stage | half-edge GPU | ConvexCell FP32 GPU | half-edge serial CPU |
|---|---:|---:|---:|
| **N** gather | (N+B = 82%) | (N+B = 91%) | **≈ 49 %** |
| **B** clip/topology | | | ≈ 30 % |
| **G** geometry | **18 %** (incl. derivatives) | **9 %** (volume only) | ≈ 21 % |

(GPU: half-edge 2015→2455 k/s with G off ⇒ G=18 %; ConvexCell 5428→5959 ⇒ G1=9 %. Serial: standalone
cutter 124 k/s vs full 62.7 ⇒ gather ≈ 49 %; within the cutter, cut ≈ 59 % / geom ≈ 36 %. N and B can't
be split cleanly on GPU because the early-out couples them; the serial split is the guide.)

**Three findings that order F2/F3/F4:**

1. **The neighbour query (N) is the single largest stage of the *cold* build (~49 % serial; dominant
   on GPU).** So the biggest *cold-build* throughput lever is **F4 (BVH/directional gather)**, not the
   clip. *But* in the steady-state incremental loop (Part II) N is mostly **avoided** (the topology and
   its neighbour list are resident), so there B (repair) + G dominate. This is the key tension: **N
   matters for the first build; B + G matter every step.** It justifies the user's instinct to optimise
   the *cell-building (B) + geometry (G) data structure* first — that is the always-relevant part — and
   to treat the gather (F4) as a separable, build-only concern.
2. **G is cheap and cleanly separable — and tiering pays.** Volume-only (G1) is ~9 %; adding the
   *derivatives* (G2) roughly doubles the geometry cost (half-edge's geometry, which includes `fdV`, is
   ~18 % vs ConvexCell's volume-only 9 %). So computing only the needed tier saves ~half the geometry
   time when derivatives aren't required — exactly the tiering motivation. (G0 vertices are a *free
   byproduct* of the clip in both engines — already cached.)
3. **B (the clip) is ~30 % serial / a large part of the GPU N+B.** This is the part the user wants to
   attack via the topology data structure (F2) and where the half-edge↔ConvexCell choice + the
   single-face-repair cost (Phase 1.5) live.

**F1 exit:** stages confirmed separable with the native update-from-cuboid form; G isolated and
measured; the split says **optimise B + tiered-G data structure now (F2, F3 — always relevant), keep N
(F4) as the build-only lever.** No refactor was forced — the engines already compose as
`N → B(update) → G(tier)`; the explicit reusable interface will be cut when F2 picks the topology
representation (so the interface is shaped by the winner, not guessed).

---

## F2 + F3 (geometry) results — topology decision + tiered geometry, measured

Two pieces landed together: **G2 derivatives on the ConvexCell** (the open "can derivatives be done
on the dual?" question) and the **isolated construction cost** (B+G, no gather — the always-relevant
core + the repair fast path).

**G2 on the ConvexCell — YES, and machine-exact.** `facetGeometry(k)` returns the face area vector,
the volume gradient `dV = ∂V/∂r_k = (area/|r_k|)(r_k − centroid_k)` (derived directly — moving plane k
sweeps area·displacement; no edge-tensor machinery), and the connecting vector. Validated vs the
half-edge oracle per (cell, neighbour), 23278 facets: **maxArea 3.8e-17, maxDV 3.1e-17, maxConn 0** —
the simple per-face formula reproduces the half-edge's intricate `computeGeometry`/`fdV` exactly. The
ConvexCell is therefore a **complete physics representation** (volumes + areas + derivatives).

**Isolated construction (RTX 5080, no gather, kcells/s):**

| tier | ConvexCell FP32 | ConvexCell FP64 |
|---|---:|---:|
| G0 topology only (vertices free) | **17 200** | 8 880 |
| G1 + volume | 13 900 (+19 %) | 7 170 |
| G2 + per-facet area/dV/connVec | **12 000** (+13 %) | 5 300 |

vs full build *with* gather: ConvexCell FP32 5 420, half-edge 2 490.

**Findings.** (1) **Construction is not the bottleneck — the gather is**: removing it triples
throughput (5.4 → 17.2 M/s). (2) **Construction-from-cached-neighbours with full physics is 12 M/s
FP32** — ~5× the half-edge full build, so the Phase-1.5 repair (re-clip a cell from its cached
neighbour list, no gather) is cheap, confirming the Phase-0 speedup model. (3) **Tiering pays**:
+volume ≈ 19 %, +derivatives ≈ 13–26 % — callers that only need volumes skip the derivative cost.

**F2 decision — topology representation.**

| axis | half-edge | ConvexCell (dual) |
|---|---|---|
| GPU full build | 2.49 M/s | **5.4 M/s** (2.2×) |
| construct-from-cache, G2 | (not isolated on GPU) | **12 M/s** |
| frame footprint | 32 KB | **3.5 KB** (9×) |
| physics (V, A, derivatives) | validated | **validated to machine eps** |
| serial CPU vs voro++ | **≈ 1.0×** | 0.4–0.6× |
| robustness (today) | round-off fallback | needs Phase-3 (topology-oriented) |
| repair (re-clip cached list) | supported | supported, **faster** |

**Verdict: the ConvexCell (dual-triangle) is the GPU build-engine topology** — it wins build speed,
footprint, and (now) full physics, and its construction core is very fast. Keep the **half-edge as the
validated oracle + the serial-CPU representation** (it matches voro++ serially; the ConvexCell trades
that for GPU-occupancy + FP32). Robustness will be **topology-oriented (Sugihara, Phase 3)**, which
suits the ConvexCell's convex-by-construction property — not exact predicates. The remaining
cold-build lever is the **gather (F4, ArborX BVH)**, since construction is now 12–17 M/s and the gather
caps the full build at 5.4.

---

## F4 results — ArborX BVH (kNN) gather beats the grid

The gather was the cold-build bottleneck (construction alone is 12–17 M/s; the grid full build caps at
5.4). `bench_bvh_gather.cpp` replaces the uniform grid with an **ArborX 2.1 BVH k-nearest-neighbour
query** (ArborX is already a suite dependency via dem/packing) feeding the ConvexCell construction;
periodicity via boundary ghosts (~33 % extra points at the chosen band).

**Measured (RTX 5080, FP32, N=1M, fair = volume in both):**

| K | Σvol err | faces/cell | BVH-build | kNN-query | construct (G1) | **total** |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 3.7e-4 (incomplete) | 15.44 | ~negligible | 21.5 M/s | 13.6 M/s | 8.25 M/s |
| **64** | **3.2e-5 (complete)** | **15.52** | ~negligible | 15.3 M/s | 12.2 M/s | **6.73 M/s** |
| 80 | 3.0e-6 | 15.53 | — | 11.5 M/s | 12.9 M/s | 6.0 M/s |

vs **grid full build 5.42 M/s**.

**Findings:**
- **At full completeness (K=64), the BVH gather is 1.24× the grid** (6.73 vs 5.42 M/s) on *uniform*
  data — and the kNN query (15.3 M/s) is markedly cheaper than the grid block-gather (which made the
  grid build the bottleneck). K=64 is the sweet spot: K<64 misses neighbours (incomplete cells), K>64
  over-gathers. The construct from the kNN set (12.2 M/s) matches the cached-construct number, i.e. the
  BVH delivers neighbours efficiently enough that construction is again the floor.
- **On clustered/non-uniform point sets the BVH advantage is larger** (a uniform grid degrades badly
  where density varies — the reason Ray/Basselin and the 2026 paper use a BVH; here we only measured
  uniform, where the grid is at its best, and the BVH still won).
- **The kNN query is now the gather floor.** The SOTA literature uses a best-first traversal + a
  cell-dependent (security-radius) stop rather than fixed-K kNN — measured next (Option 2). It did
  **not** pay off here (see below); fixed-K=64 remains the cold-build engine.
- **ArborX is reusable for the incremental repair's local re-query** (Phase 1.5): a moved seed re-queries
  its small neighbourhood, not the whole grid block.

## Option 2 results — best-first BVH traversal (custom, cell-dependent stop) does NOT beat fixed-K kNN

`bench_bvh_bestfirst.cpp` drives a **custom best-first traversal over ArborX's BVH nodes** and fuses
the gather with the clip: per cell, a per-thread min-heap of pending nodes keyed by distance²-to-seed;
pop the closest, clip the ConvexCell, and **stop the instant the closest pending node is beyond the
security radius** `2·√maxVertexRsq` — no fixed K, no over/under-gather, exact and robust to clustering.
This is the thing ArborX's public API *cannot* express (its `nearest` is fixed-K, callback post-hoc), so
node access goes through a thin `BvhAccess` adapter over `ArborX::Details::HappyTreeFriends` (the one
file that would change if ArborX's internals move). Two enabling fixes were needed: seed a **tight
initial box** (~10·spacing, not the full domain) so the security radius starts ~100× smaller and prunes
from the first node; and **prune children at push time** (`dist² ≥ secR`). A stack-DFS variant with the
same radius was measured *slower* (more nodes visited despite cheaper per-node cost), so the heap stays.

**Measured (RTX 5080 / 48-thread OpenMP, FP32, N=1M, faces/cell 15.53 = complete):**

| backend | best-first (traverse+clip, fused) | fixed-K=64 kNN (F4) | ratio |
|---|---:|---:|---:|
| **GPU (Cuda)** | 1.49 M/s | 6.73 M/s | **0.22×** |
| **CPU (OpenMP)** | 1.30 M/s | 1.44 M/s | 0.90× |

Stable across N (0.2M/1M/4M GPU: 1.51/1.49/1.47 M/s).

**Findings:**
- **Best-first is correct but loses — decisively on GPU (4.5×), marginally on CPU (1.1×).** The exact,
  K-free stop saves clip work, but that saving is dwarfed by the cost of the traversal itself.
- **Why fixed-K wins:** the fixed-K over-gather is cheap, and F4's **two-pass pipeline** (ArborX-tuned,
  warp-coherent kNN query → construct from a *contiguous* candidate buffer) keeps each pass coalesced.
  The fused best-first is a **divergent per-thread priority queue** with BVH-node fetches throughout —
  exactly the access pattern GPUs punish, which is why its deficit is 4.5× on GPU but only 1.1× on CPU.
- **Conclusion: fusion hurts here; fixed-K=64 kNN (F4) is the cold-build engine on both backends.**
  Best-first would only win on point sets clustered enough that fixed-K badly mis-sizes — not the
  uniform/Poisson workload. Its real home is the **incremental/repair path** (Phase 1.5), where the
  candidate set per moved seed is a handful of nodes and a small heap is cheap and exact.
- The `HappyTreeFriends` coupling is contained to `BvhAccess`; the bench stays as the characterisation
  that justifies *not* taking the SOTA best-first route for the cold build.

## Compact-cell redesign — tested, does NOT help: the construct is not memory-bound

Hypothesis (good one): the cell is 3536 B of local memory (ptxas: `3536 bytes stack frame, 0 spill,
56 reg` — local because the arrays are dynamically indexed, *not* because of size), so shrink the
footprint and the memory-bound construct should speed up. `convex_cell_compact.hpp` does exactly that:
**packed 32-bit triangle** (3×10-bit plane indices + alive bit, 16 B→4 B) and **no vertex cache** —
the dual vertex is recomputed from its 3 planes (Cramer) wherever needed, trading ~17 KB/cell of local
vertex traffic for FP32 on the ~idle units. Cell shrinks 3084→1484 B. `bench_construct_compact` A/Bs
it against the cached cell on identical candidates.

**Result (RTX 5080, FP32, N=1M):** compact = **0.72× the cached cell** (G1 10.0 vs 14.0 M/s), same
volumes. Smaller AND fewer registers (52 vs 56) yet *slower*. So the construct is **not** footprint- or
register-occupancy-bound — the recompute's division in the dependent kill-scan cost more than the memory
it saved. The cached-vertex design is already the right call.

**Occupancy sweep (LaunchBounds<MaxThreads,MinBlocksPerSM>):** default 13.2 → `<256,4>` **14.5** (+10%)
→ `<256,6>`/`<128,12>` 12.2 (register spill). So the construct is only *mildly* occupancy-sensitive:
+10% at the sweet spot, then negative. `LaunchBounds<256,4>` is a small free win worth keeping on the
construct-heavy kernels.

**Conclusion:** at 0.1% of compute and ~4.5% of bandwidth the construct *looks* latency-bound, but it is
bound by the **intrinsic serial work of the clip chain** (per-candidate `maxVertexRsq` O(nt) + per-clip
O(nt) kill-scan + O(nt)-per-edge `findSharing`), and every standard lever to cut that has now been tried
and lost: footprint/packing (here, 0.72×), explicit triangle adjacency (earlier, GPU-negative),
incremental security radius (earlier, GPU-negative), cell-cap shrink (neutral at overflow=0). **14.5 M/s
is the algorithm's ceiling on this hardware, and it already matches/beats Ray-et-al's V100 12.5.** The
cold-build redesign target was the wrong component: ConvexCell is at its ceiling and is *not* the
full-build bottleneck — the **gather** is (next section). Warp-cooperative is also contraindicated: the
occupancy result says we want *more* cells in flight, and one-cell-per-warp gives 32× *fewer*.

## Where the performance is, and why the cold build sits at ~6.7 not ~14 M/s (full accounting)

Prompted by "reach SOTA — you expected 15–20 M/s and moved away," the whole pipeline was re-measured
and every locality/occupancy hypothesis tested to destruction (RTX 5080, FP32, N=1M). The result is an
honest decomposition, not a knob left unturned.

**The construct (cell building) is already at SOTA.** Construct-from-cache (candidates pre-supplied, no
gather; `bench_construct`): **G0 17.2 / G1 14.0 / G2 12.0 M/s**. That *is* the Ray-et-al class (12.5 M/s,
V100, 2018). The hard part — clip + dual-triangle retriangulation + volume + derivatives — is not behind.

**The entire gap is the neighbour gather.** Full cold build: F4 fixed-K=64 kNN **6.73 M/s**; fused
grid-walk **5.5 M/s** (tuned 6.45, below). So neighbour-finding roughly *doubles* the time vs the 14 M/s
construct ceiling. Diagnosis of the fused grid (`bench_convexcell`, instrumented):

- **It examines ~108 candidate distances to build a 15.5-face cell** (`examined/cell mean=107.6`). This
  is the cost, and it is *not* wasteful: a cell's security ball (radius 2·Rmax, Rmax≈1.3·spacing) holds
  ⁴⁄₃π(2.6)³≈**70 points** — you genuinely must test ~70 neighbours to *safely* close a 15-face cell.
  Computing ~70 periodic neighbour-distances per cell costs about as much as building the cell. Inherent.
- **Things that do NOT move it (each measured, each ~0):** Morton/Z-order point ordering vs row-major
  (gather is *not* memory-latency-bound — it's instruction/throughput-bound on the distance count);
  branchless periodic wrap vs integer-modulo; caching `secR2` between clips; shrinking the cell caps
  MAXP/MAXT (with overflow held at 0; peak `nt` is only mean 28.8 / max 66, so 112 was head-room, and
  68 runs identically — the apparent "+47% at 24/44" was **overflow cells bailing early**, an artifact).
- **What DOES help (kept):** processing seeds in spatial (binned) order so a warp hits shared neighbour
  cells — a steady **1.16–1.20×**; and a **finer grid + matched search window** (CC_DENS≈0.3, CC_SW=4)
  that tightens the per-cell stop from 108→~65 examined → **6.45 M/s** (err 8e-5, complete). Both land
  the fused path at F4-kNN parity (~6.5 M/s), up from the original 4.75.

**Why we cannot just be 4× faster than V100 despite 4× its FP32.** The construct (14 M/s) is only ~1.1×
Ray's V100 number, not ~4×. So the kernel is **not FP-bound** — it is memory/latency-bound on the
per-thread cell state (the ConvexCell is hundreds of bytes of `pn/pd/pnbr` + `t*/v*/alive`, partly
local-memory-resident). RTX 5080 and V100 have similar bandwidth, so they land in the same place. **This
is the real ceiling, and SOTA shares it** — Ray's 12.5 M/s is construct-ceiling-dominated, with the
gather folded in. The best-first detour (1.5 M/s) violated exactly this: it replaced a coalesced stream
with a divergent per-thread BVH heap — maximal scatter, the opposite of what the hardware wants.

**What it would actually take to exceed it** (none a quick win, all logged for Part III):
1. **Warp-cooperative cell building** — distribute one cell across 32 threads' registers (cell leaves
   local memory → register-resident → unlocks the 5080's FP headroom). The earlier warp-per-cell attempt
   was occupancy/cut-bound (~0.6×); doing it *right* (vertex-parallel clip, no atomics) is the lever.
2. **A more compact cell** (fewer live triangles / bytes) so one-thread-per-cell stops spilling.
3. **Cooperative tiled gather** — a thread-block stages a grid tile + halo into shared memory once and
   all its cells read neighbours from there, amortising the ~70 distance-tests across the block.

**Bottom line for the cold build: construct = SOTA (14 M/s); full build = ~6.7 M/s, gather-limited by an
irreducible ~70 distance-tests/cell; beating it needs warp-cooperative register-resident cells, not
another gather tweak.** The moving-point workload (Part II) is the bigger win anyway — it *reuses*
topology and re-runs only G (12–18 M/s) over a resident cell, skipping the gather entirely.

**Part I exit:** `N(BVH) → B(ConvexCell update from cuboid) → G(tier)` — BVH gather 1.24× the grid
(more on clustered), ConvexCell build 2.2× the half-edge with a 9× smaller frame, physics (G2)
validated to machine precision, geometry tiered. The pieces are separated and each beats the
prior baseline. Next: Part II (moving particles) — Phase 1 geometry/derivative re-eval over resident
topology + momentum, then Phase 1.5 single-face repair.
