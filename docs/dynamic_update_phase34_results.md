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

The fix is exactly the structure the plan **parks**: a maintained 1-/2-ring candidate set
(`ConnectivityArena`) would give surgical a complete, sorted candidate list — making it both fast and
exact. Without it, on this clip-bound engine the gated two-pass gather (Phase 3) is the better
production choice, and surgical stays off as the documented experiment.

## Bottom line

- **Production: Phase 3 gated two-pass gather.** Exact (to tolerance; machine-exact at `VORF_TOL=1e-7`),
  never much slower than rebuild on any backend, 2–3× on the CPU paths and ~2× on GPU at small δ/h.
- **Phase 4 surgical: parked** pending the ConnectivityArena 2-ring (negative result documented above).
- Knobs: `MovingTessellation::{churnThresh, useDilation, clusterNbhd, surgical, verifyCap}`; bench
  `--repair` (env `VORF_TOL`, `VORF_SURGICAL`).
