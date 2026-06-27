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
| **S4** convex-prop     | per-cell **D2** + symmetry | rebuild flagged, then **propagate** to asymmetric neighbours and iterate (star-splaying-style); fall back to a full rebuild if >30% are touched. *Re-gathers candidates from the grid every step (suboptimal — see S5).* |
| **S5** combined        | **Verlet** skin + per-cell **D2** + symmetry | **the production design.** Each cell keeps the candidate (skin) list from the last full rebuild; while max-displacement < skin/2 the list is guaranteed complete, so local repair clips the **stored** list (no grid, no re-gather) — the S4 inner loop. When the Verlet criterion trips: rebuild grid, re-gather skin lists, full rebuild, reset reference. |

**Verlet list (S5).** S0–S4 do *not* keep a guaranteed neighbour list — S3/S4 re-gather the k-NN from a freshly
rebuilt grid every step, which is wasteful. S5 fixes this: the per-cell candidate list (with skin) is gathered
once per full rebuild and is provably complete while no seed has moved more than skin/2 (standard Verlet). Then
local repair just re-clips the stored list. Because that list is distance-sorted and the clip keeps its
security-radius early-out, the extra skin candidates sit past the security radius and are **never examined** — so
a larger skin does not slow the repair clip; it only enlarges the (rarer) full-rebuild gather. The skin therefore
trades full-rebuild *frequency* (∝ 1/skin) against full-rebuild *gather cost*, with an optimum near skin ≈ O(cell
size), confirmed below.

**D2 convexity certificate** (`ConvexCell::isSelfConsistent`): after a re-eval on the fixed stored topology,
the cell is still the correct half-space intersection iff every dual vertex lies inside every *other* plane of
the cell (`n_k·v ≤ nn_k`). A genuine flip pushes a vertex outside a plane → flagged. A vertex is the meet of its
own 3 planes and so lies *on* them (`n·v = nn`); those three are excluded from the test (`isSelfConsistent`
skips `k ∈ {t0,t1,t2}`), so a valid cell never self-flags from round-off on its defining planes. This is purely cell-local
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

### S5 combined (persistent skin-list + Verlet), displacement × skin

Skin in cell-sizes; `rebuild%` = fraction of steps doing a full rebuild (Verlet trip or >30% fallback);
`wrong` = cells with >0.1% volume error vs the per-step rebuild oracle. The persistent list makes S5 markedly
faster than S4's re-gather-every-step (disp 0.001: S5 **23×** vs S4 6×).

GPU RTX 5080, FP32, N = 500k, baseline S1 = 544 ms/step (ms/step, speedup, wrong/500k):

| disp \ skin | 0.05 | 0.10 | 0.20 | 0.40 |
|-------------|------|------|------|------|
| 0.0005 | 73 (7.5×), 11 | **20 (27×), 41** | 20 (27×), 41 | 20 (27×), 41 |
| 0.0010 | 183 (3.0×), 0 | 76 (7.2×), 31 | **24 (23×), 165** | 24 (23×), 165 |
| 0.0020 | 294 (1.8×), 1 | 188 (2.9×), 1 | 82 (6.6×), 97 | 30 (18×), 333 |
| 0.0030 | 457 (1.2×), 0 | 245 (2.2×), 0 | 140 (3.9×), 35 | 87 (6.2×), 70 |
| 0.0050 | ~885 (0.6×) all-fallback | — | — | — |

Host FP64, N = 80k (timings noise-prone on the shared box; the accuracy/diagnostic columns are deterministic):
at disp 0.001, `wrong` = 0 / 7 / 34 / 34 for skin 0.05 / 0.10 / 0.20 / 0.40 — the same shape as FP32 but ~5×
fewer volume-significant misses (FP32 165 → FP64 34 at skin 0.20).

Reading the table: **as displacement grows you need a smaller skin** (more frequent rebuilds) to hold accuracy,
and the speedup achievable at a given accuracy falls; by disp ≈ 0.005 every step hits the fallback and local
repair stops paying (hand off to S2). **skin 0.20 ≡ skin 0.40** wherever rebuild% = 0 (identical ms/touch/wrong):
once the skin is large enough that no Verlet trip occurs in the window, extra skin is free — the sorted list +
security-radius early-out never examines it.

### Why are cells "missed"? (the `listMiss` / `topoMism` diagnostic)

The harness reports two extra columns to localise the error source:
- **`listMiss`** = cells where a true (oracle) neighbour is absent from the strategy's candidate list. **It is 0
  in every row, FP32 and FP64.** So the missed cells are *not* a neighbour-completeness problem — with K=128
  nearest vs ~15 actual neighbours, the Verlet list stays complete under the small drift. (A cell "not having all
  potential neighbours" would show here; it never does.)
- **`topoMism`** = cells whose stored topology differs from a fresh rebuild. This is *large* (~10–35%, and about
  the same fraction in FP32 and FP64) while the volume error is tiny — because most of those differences are
  **marginal, near-zero-area faces** that drift in/out under motion. D2 deliberately ignores them (its tolerance
  is a perpendicular distance), and they do not move the volume. So `topoMism` measures bit-level topology drift,
  not error; **volume error is the meaningful accuracy measure** (mean rel-V stays ~1e-6 throughout).

So the handful of volume-significant misses are **detection misses** — D2+propagation didn't rebuild a cell whose
topology change *did* matter — not missing neighbours. They are rare (≤0.03%), bounded, grow with skin (more
drift between rebuilds), shrink with precision (FP64 ~5× fewer than FP32), and are wiped whenever a full rebuild
is recent (skin 0.05). The periodic full rebuild any production loop runs caps this residual.

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

4. **Precision matters for the borderline flips, not for the marginal-face drift.** The D2 self-test needs a
   precision-aware tolerance (`d2tol` = `1e-4·cellSize` FP64, `2e-3·cellSize` FP32) or FP32 vertex noise
   over-flags. With that, the flagged/touched fraction is similar across precision, and so is `topoMism`
   (marginal-face drift is ~precision-independent, ~24%). What FP32 *does* worsen is the **volume-significant
   residual** on near-degenerate (borderline) flips — at disp 0.001 skin 0.20, FP64 leaves 34 wrong cells vs
   FP32's 165 (~5×). This is the Sugihara topology-oriented-robustness regime: do the *topology* decision in
   FP64 (or add area-thresholding to the certificate), even if the geometry/volume runs FP32.

## Recommended heuristic

**S5 is the production strategy** — a persistent Verlet skin-list with a propagating-repair inner loop. It
unifies the others: pure re-eval each step, a cheap D2 sweep as the trigger, propagating local repair off the
**stored** list while the touched fraction stays small, and a full rebuild (with worklist/skin-list regather)
when the Verlet criterion trips or too many cells are flagged. It dominates the per-step-re-gather variants
(GPU FP32 disp 0.001: 23× vs S4's 6×).

Tuning, per the measured behaviour:

- **disp ≲ 0.002 (the DEM/CFL regime):** S5 with **skin ≈ 0.1–0.2 cell-sizes**. skin 0.1 → ~6% of steps
  rebuild, near-exact; skin 0.2 → no rebuilds in-window, ~23×, with a ~0.03% volume residual healed at the next
  rebuild. Choose along that curve by the accuracy budget. (Larger skin never hurts the repair clip, only the
  rebuild gather and memory.)
- **disp ~ 0.005–0.02:** topology churn exceeds the local-repair budget (S5's fallback would fire every step);
  use **S2 displacement-Verlet** (rebuild every few steps, exact-to-bounded, ~3–11×).
- **disp ≳ 0.02:** S1 full rebuild each step.
- Do the **topology decision in FP64** (FP32 marginal-face flicker inflates the flagged set and the residual);
  FP32 is fine for the geometry/volume re-eval.

An adaptive controller can pick the regime live from the measured D2 flag-fraction (small → S5 local; moderate →
S2; large → S1) and auto-tune the skin toward the rebuild-fraction that minimises measured ms/step.

## Caveats & future axes

- The harness builder is a fixed-window grid gather, not the production worklist tessellator; absolute ms are
  not production throughput, but the cross-strategy ratios are valid (shared builder).
- Studied: random Poisson seeds, single density. Not yet swept: density / polydispersity, much larger N,
  multi-hop-per-step displacement, and **SDF boundary cells** — a boundary plane has no partner cell, so the
  convexity-symmetry argument needs an explicit boundary watch there (noted, not yet measured).
- Production wiring (re-eval + the chosen strategy into the stepper, with aux-map/CSR reuse) is deliberately out
  of scope here — this study is about the cell update only.
