# Dynamic Voronoi/Power Tessellation Update — Decision & Implementation Plan (vorflow)

**Status:** decision + phased plan. **Phase 0 + Phase 1 IMPLEMENTED 2026-06-29** (not pushed) — see the
"Implementation status" box below; Phase 2 (the two-pass repair loop) is the next prompt (§8 templates).
**Audience:** the implementer (human or LLM) picking up the moving-point update loop.
**Companion context:** `voronoi_neighbor_update_overview.md` (method/legacy/literature/avenues), `voronoi_dynamic_update_study.md` (the S0–S6 numbers), `performance.md`, `update_and_repair_redesign.md` (legacy wave/ConnectivityArena design).

> **Implementation status (Phase 0 + Phase 1, 2026-06-29).** Done and gated on OpenMP + CUDA (RTX 5080);
> MPI np=1,2,4 green. The cold-build grid/worklist was factored into `include/vorflow/device/tess_grid.hpp`
> (`TessGrid`/`buildTessGrid`, + a `slotOf` inverse map) so one grid backs both the cold build and the
> subset gather — proved byte-identical (single-thread fingerprint; the build is not bit-deterministic
> across thread counts because atomic binning order changes the candidate-clip order). New headers:
> `dynamic_validate.hpp` (invariants + oracle diff — compare FACE neighbours, ≥3 incident live triangles),
> `subset_gather.hpp` (`subsetGather`, GATE 1a bit-for-bit vs same-grid full gather), `verlet_skin.hpp`
> (`flagSkinMovers`/`maxDisplacement`, extensible `SkinTrigger`). `ConvexCell::isSelfConsistent` gained a
> partner-emitting overload (violated-plane `pnbr` seeds). Driver `tests/kokkos/bench_dynamic_update.cpp`
> (`--gates`/`--sweep`/`--phase0`) consolidates + replaces `bench_update_strategies.cpp` +
> `phase0_incremental.cpp`. **No repair loop / single-face / arena / SDF-boundary trigger / BVH yet.**
> Sweep highlights (RTX 5080, uniform, disp 0.001): S0 reeval ~17×, S3 local-repair ~3.6× vs rebuild; S4
> propagating ≈1× (launch overhead — confirms the flat two-pass plan). `clustered`/`near-wall` expose grid
> under-resolution at `sw=4` (the oracle itself goes nondeterministic on incomplete cells — a real finding,
> bump `sw`/local-density grid). Power still `static_assert(!Weighted)` — deferred, hooks documented (Risk 3).

---

## 0. Decision in one paragraph

Implement an **adaptive hybrid updater** with a **gather-based two-pass repair** (no maintained connectivity). Every step: re-evaluate **all** cells at fixed topology (`reevalGeometry`); run the all-cells convexity/in-sphere certificate (complete by Delaunay's lemma — §1); then repair in **two bulk-parallel gather passes** using the cold-build worklist restricted to a subset, with a **per-backend adaptive fallback to the existing cold build**:
- **Pass 1 — rebuild by gather:** all flagged (non-convex) cells **∪** their violated-plane partner seeds **∪** any particle that moved beyond a per-particle Verlet skin. The certificate catches the flip/deletion side; the skin catches the insertion side of far-movers (an insertion is *invisible* to the neighbours' certificate — §1c).
- **Pass 2 — rebuild by gather:** the new neighbours that Pass 1's gathers revealed and that were not already rebuilt (chiefly a far-mover's new neighbours).
- **Verify:** re-run the certificate; if clean (the common case) publish; else one more pass or fall back to rebuild.

Each pass is the cold-build kernel over an index list — embarrassingly parallel, no wavefront BFS, no maintained 1-ring/2-ring. The cold build is not a failure mode — it is the competing algorithm and the correctness oracle. This realizes your overview's **D (three-tier driver)** with a gather-based repair that also handles far jumps, and **demotes B (topological Verlet/`ConnectivityArena`)** to an optional later micro-optimization (a candidate hint for the Pass-1/2 gathers), since the worklist gather finds *any* new neighbour — including far insertions a 2-ring cannot reach. Defer single-face surgical repair to a later phase; the BVH/directional gather is explicitly **parked** (the cold build is already settled — see Parked alleys).

This direction is forced by your own data: fixed-topology `reevalGeometry` is **85–190× cheaper** than rebuild, but **reclip-all is only 1.05× on GPU** — so the win cannot come from reclipping a stored candidate set; it must come from **not reclipping most cells**. At realistic displacement, **73–94 % of cells keep their exact neighbour set** and changed cells flip **≈ one face**. Selective repair is therefore the only lever that pays on a clip-bound GPU build.

---

## 1. The algorithm is settled — these three things are the real risk

The two external analyses (and your overview) agree on the loop. Where they are thin is exactly where this project will succeed or fail. The plan below is sequenced to attack these, not to chase features.

### Risk 1 — Detector exactness (RESOLVED: the convexity certificate is complete by Delaunay's lemma)

> **Correction.** An earlier draft claimed the per-cell convexity certificate could "silently miss a pure intrusion." That was wrong. Detection over **all** cells is **complete**; the residual risks are predicate *robustness* and *repair* candidate-locality, not detection completeness.

`isSelfConsistent` flags cell *i* when a re-evaluated vertex *v* (the circumcentre of the Delaunay tet dual to that vertex) pokes outside a stored plane (*i*,*m*). Geometrically *v* outside (*i*,*m*) means *m* lies inside the circumsphere of that tet — i.e. **`isSelfConsistent` is exactly the in-sphere predicate restricted to stored neighbours.** By **Delaunay's lemma** (a triangulation is globally Delaunay iff every facet is locally Delaunay), if the old topology on the new positions is no longer Voronoi then some shared facet is locally non-Delaunay, and that failure always pokes a stored vertex outside a *stored* plane of some cell — so **some cell always flags.** Proof sketch: for the failing facet shared by tets τ₁, τ₂, let *s* be the intruding apex of τ₂ and *p* any vertex of the shared facet; *p* and *s* are both vertices of τ₂, so (*p*,*s*) ∈ `pnbr[p]`, and the circumcentre of τ₁ (a vertex of cell *p*) now sits closer to *s* than to *p* → it violates stored plane (*p*,*s*) → cell *p* flags.

The non-obvious part is *which* cell flags. For a **pure-gain (2→3) flip** among {*i*,*a*,*b*,*c*,*m*} creating new face (*i*,*m*), the gaining cells *i* and *m* do **not** flag (neither has the other in `pnbr`); the **common neighbours** *a*,*b*,*c* flag, and the planes they report as violated are precisely (*·*,*i*) and (*·*,*m*) — so **the violated-plane partners are exactly the gaining cells.** A **loss (3→2) flip** flags the losing cells directly (their face collapses to zero area). Consequently the seed (flagged cells ∪ their violated-plane partners) already covers the full neighbourhood of an isolated flip in **one** pass.

**The violated-plane partners are necessary, not optional.** In a 2→3 flip the common neighbours *a*,*b*,*c* flag but their *neighbour set does not change* (each still faces *i*,*b*,*c*,*m*) — so rebuilding them by gather yields correct cells that report **no new neighbours**, and the actual gain (the *i*–*m* face) is repaired *only* if *i*,*m* are themselves rebuilt. Since *i*,*m* never flag and never appear as new neighbours of *a*,*b*,*c*, the **only** cheap signal that reaches them is the violated-plane partner read off the certificate. Without it the gather-from-flagged-cells repair silently leaves the *i*–*m* face missing — this is exactly why the menu's "independent local repair" measured *near-exact*, not exact. Reading which plane a poking vertex violates is local certificate output, not maintained connectivity.

**Resolution — the detector is just the convexity certificate over all cells, plus a power-correct predicate and a robustness fallback:**
1. **Convexity / in-sphere certificate** (cheap, all cells) — complete for both gains and losses (above).
2. **Power-diagram form:** use the **empty-orthosphere** (power-distance) predicate for weighted cells so completeness carries over to the regular triangulation (Risk 3).
3. **Robustness fallback (avenue G):** near 5-cosphere degeneracy a new face can appear with sub-`tol` area; this is the same exact-predicate concern as the cold build, handled by adaptive/exact orientation or a widened-tolerance re-test, **not** by an extra detector layer.
4. **Periodic full-rebuild oracle** (every *k* steps, and on any threshold trip) — the ultimate guarantee and the source of truth for tuning.

There is **no need** for a separate "orthosphere-over-2-ring gain detector" as a completeness backstop — the all-cells certificate already is complete. Finding the new neighbours of flagged/moved cells is a **repair** concern, handled by the subset gather (§3), not by detection.

### Risk 1b — Two passes is exact for separated events, not a universal bound

By insertion/deletion locality, an isolated event closes in **two gather passes**: a far jump is a deletion (its old neighbours flag → Pass 1) plus an insertion (the mover is rebuilt via the skin trigger in Pass 1; its new neighbours are rebuilt in Pass 2). Inserting a point creates new adjacencies only of the form (mover, x) — never between two pre-existing seeds — so Pass-2 cells introduce nothing outside the set, and there is no Pass 3. A small flip closes in **Pass 1 alone** (flagged common-neighbours ∪ their violated-plane partners = the whole flip; Pass 2 empty). This is exact for spatially **separated** events, the small-displacement CFD common case. It is **not** a universal guarantee: dense simultaneous large moves can couple events so a Pass-1 rebuild shifts geometry enough to make an untouched cell wrong — a genuine third pass. Hence the mandatory **verify** (re-run the certificate after Pass 2): clean → publish; dirty → one more pass or rebuild. A verify, not an unbounded iterate.

### Risk 1c — Insertion is invisible to the certificate (why the skin trigger is required)

The certificate detects that a particle *left* a region (its old neighbours flag), never that one *arrived*. When mover p lands among a cleanly-tessellated `{e,f,g}`, those cells did not move and do not have p in their stored planes, so they never flag, and no common neighbour links p to them — the gain is undetectable by convexity or by any stored-connectivity propagation. The only way to find `N_new(p)` is to **rebuild p's own cell by gather**. So Pass 1 must rebuild every particle that moved beyond a per-particle **Verlet skin** (moved > skin/2 from its last rebuild position), independently of the certificate. In pure small-displacement CFD this trigger fires for almost no one; it activates exactly for the far-movers the certificate cannot see. This is the Verlet skin in its proper role here — an *insertion* detector, not a topology-stability test.

### Risk 1d — SDF boundary interaction is invisible to the certificate (DEFERRED, but required for SDF walls)

> **Deliberately out of scope for the initial build; must be added before production use with SDF boundaries.** Distinct from the parked BVH alley (which is "settled, don't touch") — this is a genuine completeness gap, just one we are choosing not to address yet.

Cells are also clipped by the domain boundary (box planes and SDF-derived wall planes; `pnbr[k] < 0`). A particle that **newly contacts** an SDF wall should gain a boundary face, but — exactly like the insertion case (§1c) — the cell stays convex against its current planes, so the certificate never flags it. A particle that **loses contact** changes (or drops) a wall face, which also alters the contact-angle / SDF wall-energy term — physics-relevant for the droplet application, not cosmetic. So a change in boundary-contact status is a rebuild trigger that the convexity certificate cannot supply.

The fix, when taken up, parallels the Verlet skin and is nearly free because the immersed-boundary method already evaluates the SDF per particle: track each particle's boundary-contact status via `|SDF(p)| < security radius` ("a wall could clip this cell"), and flag the particle for rebuild whenever that status changes across a step (gained or lost contact), or when it stays in contact but the relevant wall facet/normal changes near a curved boundary or corner. This is the boundary analogue of §1c's insertion trigger and feeds the same Pass-1 set. **Not implemented in Phase 1 below;** flagged here so it is not silently forgotten.

### Risk 2 — GPU launch overhead (largely dissolved by the flat two-pass)

The earlier plan used a stream-compacted frontier BFS (worklist of active cells, atomic-appended frontier, `visited` generation counters, iterated launches to closure), whose shrinking tail pays kernel-launch latency to repair a handful of cells. **The gather-based two-pass removes this:** each pass is the cold-build kernel over a precomputed index list — two (occasionally three) bulk launches with no inter-cell dependency, no atomic frontier, no generation counters. The residual overhead is just compaction of the Pass-1 trigger set and the Pass-2 new-neighbour set (a stream compaction each), plus the per-step grid rebuild (cheap, memory-bound) and the all-cell re-eval + certificate sweep (≈ 1/26 of a rebuild). This is the architecture that should hit the menu's *independent-local-repair* speed (≈ 3.7× GPU / 9× host) while being *exact* — beating both measured points, since the slow exact option paid the BFS tax.

**Resolution:**
- **Correctness-first (Phase 2):** Pass 1 → collect new neighbours → Pass 2 → verify. Prove EXACT-vs-oracle. No fusion, no tuning yet.
- **Phase 3:** fuse the re-eval + certificate + Pass-1-trigger build into one launch; tune per-backend thresholds (when to give up and rebuild). On AMD/HIP the wavefront width (64) and occupancy differ from CUDA (32) — divergence cost in the gather is backend-specific, which feeds the per-backend tuning. There is no shrinking-frontier tail to special-case.

### Risk 3 — Power diagrams need a power-aware gather radius (this is *your* application)

Detection stays complete in the weighted case: Delaunay's lemma generalizes to the **regular (weighted Delaunay)** triangulation via the empty-**orthosphere** predicate, so "some cell always flags" still holds, and `isSelfConsistent` must use the orthosphere form for weighted cells. What weights change is the **gather**: the repair rebuilds each subset cell with the same cold-build worklist, and under power/Laguerre a large-weight seed can contribute a face from far away, the security radius must be **power-distance-aware**, and a cell can be **empty/subsumed** entirely. Your droplet application (mirror-particle surface energy, topology changes, power/Laguerre cells) lives exactly here. Because the repair already gathers from the grid (not a fixed topological ring), this is the *same* power-distance-aware security-radius requirement the cold build has — there is no separate 2-ring locality assumption to break.

**Resolution:**
- Reuse the cold build's **power-distance-aware security radius** in the subset gather (the `Weighting` policy already parameterizes this); no special repair-only candidate structure.
- Use the **empty-orthosphere** predicate in `isSelfConsistent` for weighted cells.
- Handle **empty/subsumed cells** explicitly: a zero-face cell cannot flag, and a far-mover that becomes subsumed (or a subsumed cell that re-emerges) is caught by the Verlet-skin trigger (Risk 1c) and the periodic oracle. Power diagrams are the strongest argument for keeping the oracle frequent during bring-up.

---

## 2. Device strategy: one path, per-backend thresholds (not forked algorithms)

The optimal *algorithm* is the same on every device; the optimal *decision thresholds* are not, because the cost ratios differ:

- **GPU (CUDA/HIP):** rebuild is cheap (6–7 M cells/s) and clip-bound. Incremental wins are modest (1.6–3.7×) and come **only from work reduction**. The "give up and rebuild" threshold should be **lower** (rebuild sooner once the active fraction grows).
- **Host (OpenMP):** rebuild is expensive relative to re-eval (90× re-eval, 9× local repair). Incremental wins are large over a **wider** active-fraction range; the threshold should be **higher** (try harder to repair before rebuilding).
- **AMD MI250X (LUMI-G):** qualitatively tracks CUDA, but wavefront width (64) and occupancy shift gather divergence cost — tune its own constants.

So: build **one** Kokkos code path; express the policy (rebuilt-fraction → rebuild threshold, skin width, verify-pass cap) as **per-backend constants** selected at compile time or from a tuned table. The whole point of the hybrid is that on each device it is *allowed to say* "for this step, full rebuild is cheaper." Do not maintain separate algorithms per device.

---

## 3. Architecture & interfaces

Keep the layered, one-way-dependency structure. New/extended pieces:

- **Subset gather (new, device).** The cold-build worklist gather + clip, run over an **arbitrary index list** instead of all cells (the device analogue of the legacy `setupSubset`). This is the single primitive both repair passes call; it produces correct cells with no dependence on prior connectivity. Reuses the existing grid (rebuilt each step, cheap) and `buildTessellation`'s clip.
- **Certificate with partner output (extend `isSelfConsistent`).** Per flagged cell, also emit the **violated-plane partner seed(s)** (the neighbour id of the plane a poking vertex crosses). Empty-orthosphere form for weighted cells. This is the cheap topological signal that reaches the gaining cells of a flip (§1) — no maintained connectivity.
- **Verlet-skin tracker (new).** Per particle, store its last-rebuild position; flag movers exceeding `skin/2`. This is the *insertion* trigger (§1c).
- **Two-pass repair driver (new, device).** `Pass1 = compact(flagged ∪ partners ∪ skin-movers)` → subset-gather → collect each repaired cell's new neighbours not already rebuilt → `Pass2` subset-gather → re-run certificate to **verify** → publish, or fall back to `buildTessellation`. Two stream compactions, two subset-gather launches, no frontier BFS, no generation counters.
- **Adaptive policy (upfront three-way gate, decided after the free certificate, before any gather).** The certificate + skin tracker run every step regardless, so the route is chosen before paying for a single gather — a rebuild step then wastes nothing. Using only signals already computed:
  - **Sparse events** → two-pass repair (the common path).
  - **Dense local cluster** (high affected-cell density in a grid region; secondary: cells with several violated planes) → **dilate the Pass-1 set** by a 1–2 grid-cell buffer around the cluster and gather it in one regional pass, over-covering any local cascade without a global rebuild. The same subset-gather primitive; just a bigger index list.
  - **High global churn** (`|flagged ∪ skin-movers| / N` over a per-backend threshold) → skip straight to full `buildTessellation`.
  - The cascade signal is event **density/clustering**, *not* violation depth — a single far-mover is deep but a clean 2-pass insertion. Bias the gate **permissive** (rebuild only on clear signals): a failed repair attempt costs only the repair-fraction of a gather on top of the eventual rebuild, so the `verify` net (above) is a cheap residual, not the primary guard.
- **Validator (new, first-class).** Per-step cheap invariants + periodic oracle diff. See Phase 0 — build this *before* the repair.
- **Reuse unchanged:** `reevalGeometry`, `TopologyStore`, `TessellationView` CSR, `buildReciprocalMap`/`transpose`, the worklist gather and cold `buildTessellation` (oracle + fallback), the `Weighting` policy.
- **Optional, deferred:** a `ConnectivityArena` (1-ring/2-ring CSR) could later seed the subset gather with topological candidate hints to shave gather cost in the small-flip regime. It is **not** on the critical path — the worklist gather already finds any new neighbour, including far insertions a ring cannot reach — so it is a parked micro-optimization, not a prerequisite.

Data flow per step: `move → reevalGeometry(all) → certificate(all)+skin → GATE{sparse | dense-cluster | high-churn} → [two-pass subset-gather | dilated regional gather | buildTessellation] → verify → publish CSR`. The gate is evaluated before any gather, so a rebuild-routed step pays only the (mandatory) re-eval + certificate.

---

## 4. Phased, measurement-gated plan

Each phase has an **exit gate**. Do not advance until the gate is green. Everything is normalized to the current `buildTessellation` time.

### Phase 0 — Foundation: validators + normalized benchmark (build this first)
You cannot trust any incremental result without an oracle and invariants; both analyses underweight this.
- **Per-step invariants:** Σ volumes = box volume; reciprocal face areas `A_ij = A_ji`; Σ forces = 0; reciprocal `pnbr` symmetry. Cheap, run every step in debug.
- **Oracle diff:** full rebuild + neighbour-set / volume / force comparison, sampled every step and full every *k* steps.
- **Benchmark driver:** consolidate the existing S0–S6 (`bench_update_strategies`, `phase0_incremental`) to report **update-time / rebuild-time** across displacement amplitudes `δ/h ∈ {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}` and distributions (uniform Poisson, lattice+jitter, clustered, near-wall, polydisperse/power), on each backend.
- **Gate:** invariants and oracle catch a deliberately corrupted update; benchmark emits normalized cost + active-fraction + wave-iteration + missed-neighbour-rate columns.

### Phase 1 — Repair primitives: subset gather + certificate partners + skin
- **Subset gather:** run the worklist gather + clip over an arbitrary device index list, producing cells identical to the cold build for those indices.
- **Certificate partner output:** extend `isSelfConsistent` to emit the violated-plane partner seed(s) per flagged cell; empty-orthosphere form for weighted cells.
- **Verlet-skin tracker:** per-particle last-rebuild position + `skin/2` test.
- **Deferred (Risk 1d):** the SDF boundary-contact trigger is **not** built here. Particle-particle insertion (skin) only for now. Leave the trigger set extensible so the boundary trigger can be added later without reworking the driver.
- **Gate:** subset-gather over a random index subset reproduces the oracle's cells bit-for-bit for those indices; partner extraction is correct on synthetic 2→3/3→2 flips; the skin test fires exactly for movers beyond the threshold. No repair loop yet. (Boundary-interaction correctness is explicitly *not* gated here — see Risk 1d.)

### Phase 2 — Two-pass gather repair, correctness-first
- Wire the driver: `Pass1 = flagged ∪ partners ∪ skin-movers` → subset-gather → collect new neighbours → `Pass2` subset-gather → certificate **verify** → publish or rebuild-fallback. **No fusion, no per-backend tuning, no single-face repair, no arena.**
- **Gate:** EXACT vs oracle (bit-for-bit neighbour sets, machine-precision volumes/forces) over all distributions and `δ/h` up to where the rebuilt fraction is moderate, **including the power cases and including a far-jump stress test** (teleport a fraction of particles to verify the insertion path). The verify pass must come back clean within ≤1 extra pass on these cases. This is the correctness milestone.

### Phase 3 — Adaptive hybrid + per-backend tuning + fusion
- Add the **upfront three-way gate** (§3): after the certificate, route sparse → two-pass, dense-cluster → dilated regional gather, high-churn → full rebuild. Fuse re-eval + certificate + gate-signal computation into one launch. Instrument Phase-0 to log, per step, the cheap signals (global affected fraction, per-region affected density, max violated-planes-per-cell, skin-mover count) against the actual outcome (did two passes verify clean?), and tune the thresholds per backend from that — biased permissive, leaning on the verify net.
- **Gate:** updater ≥ rebuild on every (`δ/h`, distribution, backend) cell of the matrix — i.e. the adaptive path is never slower than just rebuilding — and **beats the menu's exact propagating-repair (1.6×/4.2×)**, landing near the independent-local-repair speed (3.7×/9×) while being exact. p99 latency must not exceed a full rebuild (the gate should pre-empt failed-repair spikes).

### Phase 4 — Single-face surgical repair (the steady-state ceiling)
- Since changed cells flip ≈ 1 face, replace whole-cell reclip with a surgical drop-one/insert-one on the dual triangles for the one-flip case. **Delicate:** a wrong flip silently corrupts topology, so gate it behind the Phase-0 validator and fall back to full reclip on any certificate failure.
- **Gate:** EXACT vs oracle with single-face active; measurable steady-state gain over Phase-3 on coherent cases.

**Why this order:** Phases 0–2 buy *correctness you can trust*; Phase 3 buys *never-worse-than-rebuild on every device*; Phase 4 is the optional steady-state ceiling.

### Parked alleys (deliberately not now)
- **BVH / directional-cull gather (avenue A).** Not in scope. The cold-build gather is already heavily optimized (voro++-derived worklist, SOTA-parity serial, ~1.1–1.3× the Liu-2020 GPU code) and a prior best-first BVH attempt did not improve it. The two-pass repair is built on this *existing* gather (the subset gather is the same kernel over an index list), so the dynamic loop does not need it, and a faster cold build would *shrink* the GPU incremental win (which scales with rebuild cost), not grow it. The 2026 paper (arXiv:2605.06408) argues the missing ingredient is the directional cull the earlier attempt lacked, so the question is *open*, not *closed* — but it is a possible **future** finetuning of the rare rebuild fallback, not a present task. Do not re-enter the gather.
- **`ConnectivityArena` 1-ring/2-ring hints.** Optional gather-cost micro-optimization for the small-flip regime; not on the critical path (above).

---

## 5. Benchmark matrix (the study that finds the actual optimum)

| Variant | Purpose |
|---|---|
| S0 cold build | baseline + oracle |
| S1 re-eval only | lower bound (inexact) |
| S2 re-eval + convexity + independent local reclip | simple near-exact baseline |
| **S3 re-eval + certificate + two-pass gather repair** | **main candidate (Phase 2)** |
| S4 S3 + single-face one-flip repair | steady-state ceiling (Phase 4) |
| S5 S3 + ConnectivityArena candidate hints | gather-cost micro-opt (parked) |

Sweep `δ/h ∈ {0.001…0.05}` × {uniform, lattice+jitter, clustered, near-wall, polydisperse/power} × {CUDA, HIP, OpenMP}, plus a **far-jump stress case** (teleport a fraction of particles) to exercise the insertion path. Record: update/rebuild ratio, changed-neighbour fraction, Pass-1 / Pass-2 rebuilt fractions, repair-pass count (should be 2, occasionally 3), skin-trigger count, clips/repaired cell, **missed-neighbour rate vs oracle**, reciprocal asymmetry, Σvolume error, force-pair symmetry, memory overhead, p95/p99 latency. **Oracle = your own full rebuild**, not an external library — the question is "can we beat our own cold build," and the non-negotiables (derivatives, momentum conservation, robustness) must never regress.

---

## 6. Resolving the overview's open questions (§8)

1. **Which Verlet structure?** → **None as a load-bearing structure.** Detection is the all-cells convexity certificate (complete by Delaunay's lemma — §1); repair is a **two-pass subset gather** from the cold-build worklist, triggered by the certificate (flip/deletion) plus a per-particle Verlet **skin** (far-mover insertion — §1c). The skin is retained only as that insertion trigger; the topological 2-ring (`ConnectivityArena`) is demoted to an optional gather-cost hint; kinetic (V3) only as a certificate idea, not a data structure.
2. **BVH+directional gather worth the per-step cost?** → **Parked.** The cold build is settled and a prior BVH attempt didn't improve it; the repair runs on the existing gather and a faster cold build would shrink the GPU incremental win. Revisit only as future finetuning of the rare fallback (see Parked alleys).
3. **R2 half-edge vs R3 ConvexCell vs R4 face-list?** → Stay on **R3 ConvexCell** (you already have validated sort-free geometry + derivatives on it and the occupancy win). Decide on momentum/robustness experiments, not cold-build speed; R4 face-list is a Phase-4+ throughput experiment gated on the `A_ij = A_ji` momentum check.
4. **Fully-independent faces conserve momentum?** → Treat as an open experiment behind the Phase-0 force-symmetry invariant; assume **canonical-owner `min(i,j)` scatter** is required until proven otherwise.
5. **Three-tier per-step the steady-state target?** → **Yes** (Phases 2→4). The existing cold build is the rebuild fallback and oracle; it is already good enough and is not being re-optimized.

---

## 7. Kickoff prompt for Claude Code (Phase 0 + Phase 1)

Paste the block in §8 into Claude Code running in the vorflow repo. Scope is deliberately limited to the foundation + repair primitives, with hard correctness gates and no premature optimization. Subsequent phases get their own short prompts (templates below).

### Subsequent-phase prompt templates
- **Phase 2:** "Wire the two-pass gather repair (S3) per §1/§3/§4 of `dynamic_update_decision_and_plan.md`: Pass 1 = subset-gather over (flagged ∪ violated-plane partners ∪ Verlet-skin movers); collect each repaired cell's new neighbours not already rebuilt; Pass 2 = subset-gather over those; re-run the certificate to **verify** (one more pass or full rebuild if still dirty). No fusion, no tuning, no single-face repair, no arena. Gate: EXACT vs the Phase-0 oracle across all distributions including power, plus a far-jump (teleport) stress case; verify must be clean within ≤1 extra pass. Add an `S3` path to `bench_update_strategies`."
- **Phase 3:** "Add the upfront three-way gate (sparse → two-pass; dense local cluster → dilated regional subset-gather; high global churn → full rebuild), decided after the certificate and before any gather; fuse re-eval + certificate + gate-signal computation into one launch; tune per-backend thresholds against Phase-0 logs of (signals vs did-two-passes-verify-clean), biased permissive. Gate: never slower than always-rebuild on any matrix cell, p99 ≤ a full rebuild, and beats the exact propagating-repair baseline."
- **Phase 4:** "Add single-face one-flip surgical repair (S4) gated behind the Phase-0 validator with full-reclip fallback on certificate failure."
- **(Parked — not now)** BVH/directional gather: the cold build is settled; do not re-enter the gather.

---

## 8. The exact text to paste into Claude Code

```
Read these first and confirm you understand the interfaces before editing anything:
- docs/dynamic_update_decision_and_plan.md (this plan)
- docs/voronoi_neighbor_update_overview.md
- include/vorflow/device/convex_cell.hpp (reevalGeometry, isSelfConsistent, clip,
  geometryPerVertex/geomVolume*)
- include/vorflow/device/tessellator.hpp (buildTessellation, worklist gather, CSR,
  topology-store + skin emission)
- include/vorflow/device/topology_store.hpp (pnbr/tri/np/nt)
- include/vorflow/tessellation_view.hpp + transpose.hpp (CSR, buildReciprocalMap)
- tests/kokkos/bench_update_strategies*, phase0_incremental, bench_incremental

Non-negotiables that must not regress: analytic derivatives (dV/dn, dA/dn),
momentum conservation (A_ij = A_ji exactly), robustness. One Kokkos path for
CUDA/HIP/OpenMP — no backend-only code. Figure of merit is the per-step update,
not the cold rebuild. The cold build stays as oracle and fallback.

Do PHASE 0 then PHASE 1 from the plan. Propose a short file-level plan and wait for
my confirmation before large edits.

PHASE 0 — validation + benchmark foundation:
1. Add per-step invariant checks (debug-gated): sum(cellVolume) == box volume;
   reciprocal facet areas A_ij == A_ji; sum of forces == 0; pnbr reciprocity.
2. Add an oracle diff: full buildTessellation rebuild + neighbour-set / volume /
   force comparison, sampled every step and full every k steps.
3. Consolidate bench_update_strategies + phase0_incremental into one driver that
   reports update_time/rebuild_time plus changed-neighbour fraction, rebuilt
   fraction, repair-pass count, skin-trigger count, and missed-neighbour-rate vs
   oracle, swept over displacement delta/h in {0.001,0.002,0.005,0.01,0.02,0.05}
   and over distributions {uniform Poisson, lattice+jitter, clustered, near-wall,
   polydisperse/power}, plus a far-jump (teleport a fraction of particles) case.
GATE 0: deliberately corrupt one cell's update and show the invariants + oracle catch
it; benchmark emits all columns on the available backend.

PHASE 1 — repair primitives (no repair loop yet):
1. Subset gather: run the existing worklist gather + clip over an arbitrary device
   index list (cold-build kernel restricted to a subset; the device analogue of the
   legacy setupSubset), producing cells identical to buildTessellation for those
   indices. Reuse the per-step grid.
2. Extend isSelfConsistent to also output, per flagged cell, the violated-plane
   partner seed(s) (the pnbr id of the plane a poking vertex crosses). Use the
   empty-orthosphere form for the weighted (power) policy.
3. Verlet-skin tracker: store each particle's last-rebuild position; flag movers
   that exceed skin/2.
NOTE (deferred, Risk 1d): do NOT implement the SDF boundary-contact trigger now, but
keep the Pass-1 trigger set extensible so a |SDF(p)| < security-radius status-change
trigger can be added later. Particle-particle insertion (skin) only for this phase.
GATE 1: subset-gather over a random index subset reproduces the oracle cells
bit-for-bit for those indices; partner extraction is correct on synthetic 2->3 and
3->2 flips; the skin test fires exactly for movers beyond the threshold.

Do NOT yet wire the two-pass repair loop, single-face repair, a ConnectivityArena,
the SDF boundary trigger, or BVH. Keep everything behind the existing opt-in flags so
production callers are unaffected and the existing tests stay green.
```