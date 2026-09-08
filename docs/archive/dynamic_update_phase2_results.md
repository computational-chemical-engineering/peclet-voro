# Dynamic updater — Phase 2 results: two-pass repair vs cold build (2026-06-29)

Phase 2 wires the **two-pass gather repair** (`include/peclet/voro/repair.hpp`,
`MovingTessellation`) and benchmarks it against the production cold build as a function of the
dimensionless per-step displacement δ/h (h = mean spacing), across the four requested device
configurations. All FP64.

## Algorithm (per `step(pos)`)

1. build the per-step grid (`buildTessGrid`);
2. re-eval every owned cell over its stored topology + run the convexity/in-sphere certificate,
   collecting: flagged (lost-face) cells, their violated-plane **partners** (the gaining cells of a
   flip), and the **Verlet-skin movers** (the insertion trigger, §1c);
3. **Pass 1** = `subsetGather(flagged ∪ partners ∪ movers)` — exact freshly-clipped cells;
4. **Pass 2** = `subsetGather(new face-neighbours of the MOVERS only)` — the insertion side. Flips are
   fully covered by Pass 1, so expanding *every* Pass-1 cell (an early bug) was pure waste; restricting
   Pass 2 to movers dropped it from ~55% of cells to ~0 and ~3×'d the repair;
5. **verify** the certificate; bounded extra passes; else cold-rebuild fallback (always exact).

The per-cell Verlet reference resets when a cell is gathered (its neighbourhood is fresh).

## Exactness

The certificate is **tolerance-limited**: at the production tol (1e-4·h) the worst-cell volume error
tracks the tol (sub-tol marginal faces on unflagged cells, growing with δ); at `VORF_TOL=1e-7` the
repair is **machine-exact** — volume error ~1e-15, `missedNbr=0`, and the far-jump teleport (1% of
seeds relocated, exercising the delete+insert path) is exact — at the **same speed** (flagging is
dominated by genuine pokes, not tol noise). The lone residual (a single near-degenerate cell at
δ/h≥0.02, ~1e-3) is the exact-predicate/robustness concern (Risk 1 / avenue G), not a repair-logic
bug. The benchmark's exactness gate (no row > 1e-2) **PASSes** on every backend.

## Repair vs cold build vs δ/h (speedup = cold_ms / repair_ms)

| device (config) | δ/h=0.001 | 0.002 | 0.005 | 0.010 | crossover |
|---|---|---|---|---|---|
| **serial** (Kokkos Serial) | 2.71× | 2.09× | 1.36× | 1.00× | ~0.010 |
| **multicore** (OpenMP, 8 threads) | 2.94× | 2.34× | 1.52× | 1.09× | ~0.012 |
| **GPU** (CUDA, RTX 5080) | 1.96× | 1.74× | 1.12× | 0.86× | ~0.005 |
| **serial+MPI** np=1 (1 proc/core) | 2.74× | 2.08× | 1.36× | 1.01× | ~0.010 |
| **serial+MPI** np=2 | 2.74× | 2.12× | 1.42× | 1.03× | ~0.010 |
| **serial+MPI** np=4 | 2.60× | 2.09× | 1.38× | 1.02× | ~0.010 |

(single-domain serial/OpenMP/CUDA: N=40k–200k, nSteps 8–10, uniform; MPI: N=48k, nSteps 5.)

### Reading the numbers

- **Per-backend, exactly as the plan's §2 predicts.** The incremental win scales with the cold-rebuild
  cost: largest on the slow CPU paths (serial/host/MPI ~2.7–2.9× at δ/h=0.001), smallest on the fast,
  clip-bound GPU (~2×). Every config crosses cold-build near δ/h≈0.01, where the certificate flags
  ~70% of cells (the re-clipped fraction `p1%` approaches 1 and the two paths converge).
- **MPI: the repair speedup is robust to domain decomposition.** Both cold and repair times scale ~1/np
  (cold 832→426→214 ms; repair 303→155→83 ms at np=1→2→4), so each rank keeps ~the single-process
  serial speedup (2.7×). The MPI halo exchange is common to both paths (it is *not* the bottleneck at
  these sizes), so it does not erode the ratio. The distributed repair refreshes ghost positions on the
  established halo topology (`VoronoiHalo::refreshPositions` — stable combined ordering, resident
  neighbour indices stay valid) and re-gathers only on a Verlet-skin trip (the `regath%` column: 0 below
  δ/h≈0.01, ~20% at 0.01).

### Why repair is "only" ~2–3×, not the 17–24× of pure re-eval

The exact gather-based repair re-clips the **flagged** cells off the full grid (the cold-build gather,
~50–100 candidates/cell), and the convexity certificate conservatively over-flags (~5× the truly-changed
cells) plus their partners. So `p1%` is ~12% at δ/h=0.001 even though only ~2.7% of cells truly change,
and each costs a full re-clip. Pure re-eval (S0, inexact) and the stored-skin-list reclip (S3,
near-exact) are cheaper per cell but not exact for far insertions; this is the exact-vs-fast trade-off
the plan chose deliberately (`bench_dynamic_update --sweep` has the S0–S4 comparison). Tightening the
flag set (the over-flagging) is the Phase-3 adaptive-gate lever.

## Files

- `include/peclet/voro/repair.hpp` — `MovingTessellation` (single-domain + a distributed `nProc`
  mode where cells [0,nProc) are maintained and [nProc,N) are ghost cut-candidates).
- `include/peclet/voro/mpi/voronoi_halo.hpp` — `refreshPositions` (position-only halo refresh) + `numGhost`.
- `tests/kokkos/bench_dynamic_update.cpp --repair` — single-domain sweep (serial/OpenMP/CUDA).
- `tests/kokkos_mpi/bench_repair_mpi.cpp` — distributed sweep (`mpirun -np R`, OMP_NUM_THREADS=1).

## Next (Phase 3)

The upfront adaptive three-way gate (sparse → two-pass; dense cluster → dilated regional gather; high
churn → full rebuild), fused re-eval+certificate+gate, per-backend thresholds — to push the crossover
out and tighten the over-flagging. The GPU number (~2×) is the one most worth improving (work-reduction
only, per the plan's GPU lesson).
