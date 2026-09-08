# Dynamic updater — Phase 3 (adaptive gate) + Phase 4 (surgical) results (2026-06-29)

## Phase 3 — adaptive three-way gate + per-backend thresholds + fusion (PRODUCTION)

`MovingTessellation::step` now decides its route **after the (mandatory) certificate and before any
gather**, using only the free signal `churn = flagged/nProc` (the first compact count):

- **high global churn** (`churn > churnThresh`) → full cold rebuild (a rebuild is cheaper than
  re-clipping that many cells). This is the "never much slower than rebuild" guard.
- **dense local cluster** (global churn low but a 27-grid-cell neighbourhood holds `> clusterNbhd`
  flagged seeds) → dilate the Pass-1 set by that regional buffer and gather it in one pass (over-covers
  a cascade without iterating the verify loop). **Default OFF** — see note.
- **sparse** → the Phase-2 two-pass gather repair.

**Per-backend thresholds** (set in `alloc` from the cross-device sweep crossovers; the plan's tuning
would fit these from logged signal-vs-outcome): `churnThresh = 0.50` on GPU (cheap, clip-bound rebuild
→ give up sooner), `0.70` on the CPU/host paths. "Fusion": the gate signal is the existing compact
count (no extra kernel); the mover trigger is already folded into the one certify launch.

### Effect (repair vs cold build, speedup; the gate engages at the high-δ/h end)

| backend | δ/h=0.001 | 0.005 | 0.010 | 0.020 | 0.050 |
|---|---|---|---|---|---|
| CUDA (RTX 5080) — Phase 2 | 1.96× | 1.12× | 0.86× | **0.74×** | **0.71×** |
| CUDA — **Phase 3** | 1.96× | 1.12× | 0.85× (gate) | **0.85×** (gate) | **0.82×** (gate) |
| OpenMP (8c) — **Phase 3** | 2.98× | 1.42× | 0.82× (gate) | 0.91× (gate) | 0.89× (gate) |

The gate lifts the worst case (GPU δ/h=0.05: 0.71×→0.82×) by routing high-churn steps straight to a
rebuild instead of re-clipping ~all cells and then converging. The residual ~0.82–0.91 at gated steps
is the cost of the mandatory re-eval+certificate (≈1 reeval + the grid build) paid *before* the gate
can know the churn — an inherent floor (you must detect to decide), not catastrophic. Sparse-regime
speed and exactness are unchanged from Phase 2 (the gate doesn't fire); gates + exactness **PASS** on
OpenMP and CUDA, and the distributed path composes (the MPI driver's own skin-trip re-gather and the
internal gate both route to rebuild at high δ/h).

**Dense-cluster dilation is implemented but default OFF.** On Poisson/lattice/polydisperse inputs there
are no genuine clusters; a uniformly-moderate field trivially trips a local-neighbourhood threshold, so
the dilation over-fires and its per-step count/mark scan is pure overhead (it dropped δ/h=0.001 from
2.98×→2.60× when left on). The verify loop already closes the rare cascade (extra-passes were 0 across
the whole Phase-2 sweep). It is gated behind `useDilation` + a low-global-churn guard for genuinely
clustered workloads, where over-covering a region in one pass is the intended win.

## Phase 4 — single-face surgical repair: NEGATIVE RESULT (implemented, default OFF)

Surgical repair rebuilds a flip cell by re-clipping its **known** candidate set (stored neighbours ∪
partner-discovered new neighbours, scattered to both gaining endpoints in the certificate) from the
box, with **no grid gather**. Implemented behind `surgical` (env `VORF_SURGICAL`). The Phase-4 gate
(EXACT vs oracle **and** faster than Phase 3) is **not met on this engine**, for two structural reasons:

1. **Not faster.** The cold gather clips closest-first with the security-radius early-out, so the cell
   tightens fast and each clip touches few live triangles; re-clipping the candidates *unsorted* keeps
   the cell large and costs more O(#triangles) horizon scans. Measured OpenMP δ/h=0.001: **2.08×
   surgical vs 2.98× gated gather** — surgical is *slower*.
2. **Not exact (errors compound).** A surgically-rebuilt cell that *gains* a neighbour not in its
   candidate set stays convex, so the certificate can't see the gain (§1c) and the verify loop can't
   catch it. The small per-step error then persists in the resident store and **compounds**:
   `maxRelV` grew from ~7e-2 (δ/h=0.001) to ~0.45 (δ/h=0.005) over a 10-step sweep. The partner-pair
   scatter covers flips where a common neighbour reports both gainers, but cannot guarantee completeness
   for every gain.

### Phase 4 retry — with a ConnectivityArena 2-ring candidate set (also a negative result)

The natural fix is to feed surgical the **2-ring** (a cell's 1-ring ∪ its neighbours' 1-rings, read on
demand from the resident store) instead of the stored∪partner set, since by the Delaunay property a
flip's new neighbour is a neighbour-of-a-neighbour. Implemented and measured (`buildRing` +
ring-based `surgicalRepair`). It does **not** rescue Phase 4 — three independent obstacles, now
understood:

1. **Gains are invisible to the certificate, so any *almost*-complete candidate set still compounds.**
   The 2-ring contains the new neighbour *almost* always, not always (the Delaunay 2-ring property has
   exceptions, and 2-ring membership is read from the *pre-step* store). A surgical cell that misses its
   gain stays convex → the verify certificate can't flag it → the error persists in the store and
   **compounds** (measured maxRelV grew to ~4e4 over a 10-step sweep — *worse* than the stored∪partner
   set, since more cells are surgically touched). The only candidate source that is *guaranteed*
   complete is the grid gather itself — which is the cold build. So an exact repair fundamentally needs
   the gather; a candidate-re-clip cannot be made exact via the (gain-blind) verify.
2. **Unsorted re-clip overflows the cell.** Re-clipping the ~60-candidate 2-ring in arbitrary order
   commits *transient* planes (a plane that cuts when the cell is still large stays committed even after
   later cuts make it redundant — `clip` never removes planes), so `ConvexCell`'s plane cap overflows
   for ~2% of cells *at step 1* from an exact store. The cold build avoids this entirely with its
   **precomputed distance-sorted worklist + security-radius early-out** (each committed plane is a real
   contributor); the 2-ring has no such order.
3. **Fixing (2) is GPU-hostile and still not faster.** Overflow-safety needs per-cell closest-first
   ordering of ~60–128 candidates; a per-thread sort of that many entries blows up GPU local memory
   (the grid gather sidesteps this with a *global* precomputed sorted worklist), and the 2-ring is not
   smaller than the gather's candidate set — so even a correct sorted 2-ring re-clip would not beat the
   SOTA grid gather. Measured (overflow-forced-to-gather variant): 2.05× vs the gated gather's 2.98× at
   δ/h=0.001 — slower.

**Conclusion (both attempts).** On this clip-bound `ConvexCell` engine, candidate-set surgical re-clip
— with stored∪partner *or* the 2-ring — is a dead end: exactness requires the gather (the only complete
+ ordered candidate source), and re-clip is not faster than the heavily-optimized grid gather anyway.
The only Phase-4 variant that could win is **true O(1) dual-triangle single-face surgery**
(drop-one/insert-one on the dual triangulation, not re-clip-from-candidates) — which needs robust
exact predicates (avenue G) to not silently corrupt topology, and is high-risk/out-of-scope. Phase 4
stays parked; the production path is the Phase-3 gated two-pass gather (exact via the gather, never much
slower than rebuild).

## Bottom line

- **Production: Phase 3 gated two-pass gather.** Exact (to tolerance; machine-exact at `VORF_TOL=1e-7`),
  never much slower than rebuild on any backend, 2–3× on the CPU paths and ~2× on GPU at small δ/h.
- **Phase 4 surgical: parked** pending the ConnectivityArena 2-ring (negative result documented above).
- Knobs: `MovingTessellation::{churnThresh, useDilation, clusterNbhd, surgical, verifyCap}`; bench
  `--repair` (env `VORF_TOL`, `VORF_SURGICAL`).
