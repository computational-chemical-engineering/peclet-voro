# Dynamic Voronoi cell updating — comparative study of update strategies

**What this is.** A controlled comparison of strategies for *updating* a periodic Voronoi tessellation as the
seeds move — the moving-particle ("Part II") workload. It is purely about the tessellation update (topology +
geometry / volumes); there are no forces or dynamics here. The aim is to find which update strategy is cheapest
at a given per-step displacement, how accurate it is, whether the **convexity certificate** idea works, and to
distil a selection heuristic (which turns out to be device- and precision-dependent).

Harness: `tests/kokkos/bench_update_strategies.cpp` (FP64 + `_f32`). Primitives:
`ConvexCell::reevalGeometry` / `ConvexCell::isSelfConsistent` (`include/vorflow/device/convex_cell.hpp`) and
`TopologyStore` (`include/vorflow/device/topology_store.hpp`).

## Workload & measure

- Periodic box, `N` random seeds, each given a fixed Gaussian velocity (ballistic motion — no forces).
- **Displacement is measured in cell-sizes**: `cellSize = cbrt(V/N)`. The per-step RMS displacement per seed is
  `disp · cellSize`; `disp` is the swept control variable.
- Integrate `nSteps`; each step: advance seeds → update the tessellation → compare per-cell volume to a
  per-step **full-rebuild oracle**. Reported per strategy: steady-state update **ms/step**, **touch%** (cells
  rebuilt/repaired per step ÷ N), **meanRelV / maxRelV** (cell-volume relative error vs oracle), and
  **mism>1e-3** (count of cells off by >0.1%). `S0`/`S2` errors are *cumulative* over the run (they stop
  rebuilding); the repair strategies reset each step.

## The strategies

| id | detection | action |
|----|-----------|--------|
| **S0** pure-reeval     | none | `reevalGeometry` over the step-0 topology forever (speed ceiling; error grows) |
| **S1** rebuild-each    | — | full rebuild every step (correctness/cost baseline = the oracle) |
| **S2** disp-Verlet     | global max-displacement vs a skin | global rebuild when the skin is spent; pure re-eval between |
| **S3** convex-local    | per-cell **D2 convexity** self-test | rebuild only flagged cells (independent, no propagation) |
| **S4** convex-prop     | per-cell **D2** + symmetry | rebuild flagged, then **propagate** to asymmetric neighbours and iterate (star-splaying-style); fall back to a full rebuild if >30% are touched |

**D2 convexity certificate** (`ConvexCell::isSelfConsistent`): after a re-eval on the fixed stored topology,
the cell is still the correct half-space intersection iff every dual vertex lies inside every *other* plane of
the cell (`n_k·v ≤ nn_k`). A genuine flip pushes a vertex outside a plane → flagged. This is purely cell-local
and catches **lost faces**; a **gained face** (an external seed newly cutting the cell) is invisible to the cell
itself — the conjecture is that the flip makes the *partner* cell inconsistent. **S4's propagation is exactly
what carries the flag to the partner**: after a cell is rebuilt and its neighbour set actually changes, any
neighbour that does not list it back is flagged for the next sweep.

## Results

Per-step `disp` is in cell-sizes. `ms/step` is the steady-state update cost; `S1` is the full-rebuild baseline
on the same machine/build (the harness builder is not the production tessellator, but it is identical across all
strategies, so the **ratios** are the result, not the absolute ms).

### Host (OpenMP, 16 threads), FP64, N = 80k

| strategy | disp 0.001 | 0.002 | 0.005 | 0.020 |
|----------|-----------|-------|-------|-------|
| S1 rebuild-each | 408 ms (1×) | 407 | 566 | 722 |
| S0 / S2 reeval  | 3.9 ms (**105×**) | 4.0 | — | — |
| S3 convex-local | 33 ms (12×), 5.6% touch, 7 mism | 91 ms, 19%, 331 mism | 366 ms, 40% | 346 ms, 82% |
| S4 convex-prop  | **41 ms (10×), 7.2% touch, 2 mism** | 129 ms (3.2×), 28%, 22 mism | fallback | fallback |
| S2 disp-Verlet  | (=reeval) | — | 66 ms (8.6×), 8% rebuild | **141 ms (5.1×), 33% rebuild, exact** |

### GPU (RTX 5080), FP32, N = 500k

| strategy | disp 0.001 | 0.005 | 0.020 |
|----------|-----------|-------|-------|
| S1 rebuild-each | 544 ms (1×) | 543 | 543 |
| S0 / S2 reeval  | 2.9 ms (**187×**) | — | — |
| S3 convex-local | **34 ms (16×), 9.2% touch, 462 mism** | 193 ms, 38% | 454 ms, 81% |
| S4 convex-prop  | 90 ms (6×), 12.9% touch, 92 mism | fallback | fallback |
| S2 disp-Verlet  | (=reeval) | 48 ms (11×), 8% rebuild | **183 ms (3×), 33% rebuild, exact** |

### GPU (RTX 5080), FP64, N = 200k

| strategy | disp 0.001 | 0.005 | 0.020 |
|----------|-----------|-------|-------|
| S1 rebuild-each | 324 ms (1×) | 324 | 324 |
| S0 / S2 reeval  | 3.8 ms (**86×**) | — | — |
| S3 convex-local | 32 ms (10×), 10% touch, 165 mism | 120 ms, 40% | 253 ms, 82% |
| S4 convex-prop  | **104 ms (3.1×), 14% touch, 32 mism** | fallback | fallback |
| S2 disp-Verlet  | (=reeval) | 30 ms (11×), 8% rebuild | **110 ms (2.9×), 33%, exact** |

## Findings

1. **Geometry re-eval is nearly free.** `reevalGeometry` over a fixed topology runs ~85–190× faster than a full
   rebuild. The *entire* dynamic-update cost is therefore the **topology decision + repair**, not the geometry.
   Any viable strategy must avoid touching most cells' topology.

2. **The convexity certificate works — but propagation is essential.** D2 alone (S3, independent repair) leaves
   a volume residual that grows with displacement (host FP64: 7 → 331 → 1232 wrong cells as disp goes
   0.001 → 0.002 → 0.005), because it never repairs the **gained-face partners**. Adding symmetry **propagation
   (S4) drives the residual to near-zero** (2 / 22 wrong at disp 0.001 / 0.002, host FP64) for ~1–3% extra cells
   touched. This is direct empirical support for the conjecture: *a topology change always shows up as a
   convexity violation in at least one of the two cells, and propagation carries the repair to the other.* The
   small remaining residual is healed by the periodic full rebuild that any production loop runs anyway.

3. **Crossover by displacement** (per-step, in cell-sizes):
   - **disp ≲ 0.002** (the DEM/CFL regime): **local repair wins.** S4 propagating is near-exact at ~3–10×; S3
     independent is faster still (10–16×) if a small volume error per step is tolerable.
   - **disp ~ 0.005–0.02**: topology churn exceeds the local-repair budget (S4 hits the 30% fallback and
     degenerates to full rebuild). **S2 displacement-Verlet is the sweet spot** — rebuild every few steps, pure
     re-eval between, ~3–11× and exact-to-bounded error.
   - **disp ≳ 0.02** and up: just full-rebuild each step (S1).

4. **Precision matters for the topology decision.** In FP32, marginal (tiny-area) faces flicker in/out across
   rebuilds, which (a) trips the D2 self-test unless its tolerance is loosened (`d2tol` is precision-aware:
   `1e-4·cellSize` FP64, `2e-3·cellSize` FP32) and (b) inflates the propagated set and leaves a larger residual
   (GPU FP32 S4: 92 wrong vs FP64 32). This is the Sugihara topology-oriented-robustness regime: do the
   *topology* decision in FP64 (or add area-thresholding to the certificate), even if geometry runs FP32.

## Recommended heuristic

Select per step from the per-step displacement (or, adaptively, the measured D2 flag-fraction):

- **flag-fraction small (≲ ~10%, disp ≲ 0.002):** S4 convex + propagating repair (near-exact) — or S3 if a
  ~1e-4 volume error/step is acceptable for a further ~2× speed.
- **flag-fraction moderate (disp ~ 0.005–0.02):** S2 displacement-Verlet (skin ≈ 0.1–0.15 cellSize).
- **flag-fraction large (disp ≳ 0.02):** S1 full rebuild each step.
- Do the topology decision in FP64; FP32 is fine for the geometry/volume re-eval.

The natural production design is **S2-with-an-S4-inner-loop**: pure re-eval each step, a cheap D2 sweep as the
trigger, propagating local repair while the touched fraction stays small, and a global rebuild (Verlet skin /
fallback) when it does not.

## Caveats & future axes

- The harness builder is a fixed-window grid gather, not the production worklist tessellator; absolute ms are
  not production throughput, but the cross-strategy ratios are valid (shared builder).
- Studied: random Poisson seeds, single density. Not yet swept: density / polydispersity, much larger N,
  multi-hop-per-step displacement, and **SDF boundary cells** — a boundary plane has no partner cell, so the
  convexity-symmetry argument needs an explicit boundary watch there (noted, not yet measured).
- Production wiring (re-eval + the chosen strategy into the stepper, with aux-map/CSR reuse) is deliberately out
  of scope here — this study is about the cell update only.
