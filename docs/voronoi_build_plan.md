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
