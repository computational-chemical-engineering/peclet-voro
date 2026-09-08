# Dynamic updater — Phase 0 + Phase 1 results (2026-06-29)

Implementation + measurement record for Phase 0 (validators + benchmark foundation) and Phase 1
(repair primitives) of `dynamic_update_decision_and_plan.md`. Backends: **OpenMP** (host, 8 threads)
and **CUDA (RTX 5080)**. Not pushed.

## What was built

- **`include/peclet/voro/tess_grid.hpp`** — `TessGrid<Real>` + `buildTessGrid`: the counting-sort
  grid + presorted worklist, factored out of `buildTessellation` so one grid backs both the cold build
  and the subset gather. Adds `slotOf` (original-index → grid-slot, inverse of `binned`). Pure code
  motion: the cold build is **byte-identical** (verified by an order-independent fingerprint at
  `OMP_NUM_THREADS=1`; the build is not bit-deterministic across thread counts because atomic binning
  order changes the candidate-clip order → FP differences on marginal vertices).
- **`include/peclet/voro/dynamic_validate.hpp`** — `checkInvariants` (Σvol = box vol; reciprocal
  area antisymmetry `A_ij=−A_ji`; Σarea; Σ dV; pnbr-reciprocity) and `compareVolumes` /
  `compareNeighbours` (oracle diff on **face** neighbours = planes with ≥3 incident live triangles).
- **`include/peclet/voro/subset_gather.hpp`** — `subsetGather(grid, indices, …)`: the cold-build
  `CellBuilder::buildCell` over an arbitrary index list off the shared grid.
- **`ConvexCell::isSelfConsistent(tol, partners[], maxP, &nP)`** — the convexity certificate, now also
  emitting the violated-plane partner seeds (the `pnbr` of each poked plane).
- **`include/peclet/voro/verlet_skin.hpp`** — `flagSkinMovers` / `maxDisplacement`, with an
  extensible `SkinTrigger` bitmask (the deferred SDF boundary trigger, Risk 1d, slots in here).
- **`tests/kokkos/bench_dynamic_update.cpp`** (+`_f32`) — consolidated driver
  (`--gates`/`--sweep`/`--phase0`), replacing `bench_update_strategies.cpp` + `phase0_incremental.cpp`.

## Gates (FP64 + FP32, OpenMP + CUDA — all PASS)

- **GATE 0** — invariants + oracle catch a deliberately corrupted cell (collapse topology to box +
  inflate volume): clean state is exact (`changedNbrFrac=0`, `missedNbr=0`, `volRelErr≈1e-16`); the
  corrupted cell is caught (`missedNbr≥1`, `maxVolRelErr=0.1`).
- **GATE 1a** — `subsetGather` over a random index subset is **bit-for-bit** identical (np/nt/pnbr/tri
  /volume) to a full gather off the same grid.
- **GATE 1b** — partner extraction is **sound** (every emitted partner is a stored neighbour of the
  flagging cell; `bad=0`) and the seed `flagged ∪ partners` covers the changed-cell set to ≈99.8%
  (FP64) at disp 0.01 — the residual is coupled/near-degenerate flips the Phase-2 verify pass handles.
  FP32 leaves ~3% uncovered (marginal-face flicker; voro does topology in FP64), so the coverage
  tolerance is precision-aware while soundness stays strict.
- **GATE 1c** — the Verlet skin fires for *exactly* the movers beyond skin/2.

## Benchmark foundation (strategy sweep, normalized to the production cold build)

Per-step update time vs cold rebuild, uniform Poisson, the regime where the Verlet skin (0.1 spacing)
has not yet forced a rebuild (disp ≤ 0.002). S0 = pure re-eval (inexact); S3 = re-eval + convexity +
independent local repair; S4 = + propagating repair.

| backend | disp | rebuild ms | S0 reeval | S3 local | S4 propagating |
|---|---|---|---|---|---|
| CUDA (RTX 5080), N=120k | 0.001 | 44.9 | 2.6 ms (**17×**) | 12.6 ms (**3.6×**) | 45.9 ms (~1×) |
| OpenMP (host), N=60k | 0.001 | 160 | 6.8 ms (**23.7×**) | 27.9 ms (**5.7×**) | 47.7 ms (**3.4×**) |

This reproduces the prior study and confirms the plan's **per-backend** thesis (§2): host rebuild is
expensive, so incremental wins are larger and the *propagating* repair pays (3.4×); GPU rebuild is
cheap and clip-bound, so the propagating repair's kernel-launch overhead erases its win (~1×) — which
is exactly why the Phase-2 plan uses a **flat two-pass gather** instead of a wavefront BFS.

## Phase-0 characterization (CUDA, N=500k, uniform)

| disp/spacing | stable cells | mean flipped faces (changed cells) |
|---|---|---|
| 0.001 | 0.73 | 1.21 |
| 0.003 | 0.66 | 1.30 |
| 0.01 | 0.48 | 1.56 |
| 0.03 | 0.21 | 2.26 |
| 0.1 | 0.017 | 4.72 |

Matches "73–94% stable at realistic displacement, ≈1 face flips" — selective repair is the only lever
that pays on the clip-bound GPU build.

## Findings worth carrying forward

- **Build profile unchanged by the refactor**: grid 1.3%, build 97%, CSR 1.4% (CUDA, N=500k, FP64,
  3.25 Mcell/s). The grid extraction added no measurable overhead.
- **`clustered` / `near-wall` distributions under-resolve at `sw=4`**: locally dense regions produce
  incomplete cells, and the oracle itself becomes nondeterministic on them (`maxRelV=1.0`, `missedNbr>0`
  *even for rebuild-each*). This is the Phase-0 validator doing its job — surfacing a grid-resolution
  limit. Fix when those regimes matter: raise `sw` or use a local-density grid. Not an updater bug.
- **`polydisperse` ≡ `uniform`** on the Voronoi path (weights are inert — Power/Laguerre is deferred,
  `static_assert(!Weighted)`; see Risk 3 and the `voro-power-cells-deferred` memory).
- **`A_ij = −A_ji`** holds to ~1e-5..1e-6 (geometric agreement between two independently clipped cells,
  not machine-zero); **Σarea ≈ 5e-17** (the exact global conservation). `Σ dV` is *not* zero (dV is a
  volume gradient, not an antisymmetric force) — reported but not asserted.

## Next (Phase 2)

Wire the two-pass gather repair: Pass 1 = `subsetGather` over (flagged ∪ partners ∪ skin-movers) →
collect each repaired cell's new neighbours → Pass 2 = `subsetGather` over those → re-run the
certificate to verify (≤1 extra pass, else full-rebuild fallback). Gate: EXACT vs the Phase-0 oracle
across all distributions + a far-jump teleport case.
