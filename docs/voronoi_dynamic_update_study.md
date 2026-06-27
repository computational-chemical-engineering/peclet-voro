# Dynamic Voronoi cell updating — comparative study of update strategies

**What this is.** A controlled comparison of strategies for *updating* a periodic Voronoi tessellation as the
seeds move — the moving-particle ("Part II") workload. It is purely about the tessellation update (topology +
geometry / volumes); there are no forces or dynamics here. Goals: find which update strategy is cheapest at a
given per-step displacement, how accurate it is, whether the **convexity certificate** idea works, and a
selection heuristic (which turns out to be device-dependent).

All full rebuilds — and the ground-truth oracle — go through the **production worklist tessellator**
`vor::device::buildTessellation`, which now (opt-in) also emits the resident `TopologyStore` (np/nt/pnbr/tri)
and a per-cell candidate (skin) list. Re-eval runs `ConvexCell::reevalGeometry` over the store; local repair
re-clips the stored skin list (no re-gather). Harness: `tests/kokkos/bench_update_strategies.cpp`.

## Workload & measure

- Periodic box, `N` random seeds, each given a fixed Gaussian velocity (ballistic — no forces).
- **Displacement is in cell-sizes**: `cellSize = cbrt(V/N)`; per-step RMS displacement is `disp · cellSize`.
- Each step: advance seeds → update the tessellation → compare per-cell volume to a per-step
  `buildTessellation` oracle. Per strategy: steady-state **ms/step**, **rebuild%** / **touch%**, cell-volume
  **mean/max rel error**, **mism** (>0.1% vol error), and two diagnostics: **topoMism** (stored topology
  differs from a fresh rebuild) and **listMiss** (a TRUE oracle neighbour absent from the cell's candidate list
  = genuine list incompleteness).

## The strategies

| id | detection | action |
|----|-----------|--------|
| **S0** pure-reeval     | none | `reevalGeometry` over the store forever (speed ceiling; error grows) |
| **S1** rebuild-each    | — | `buildTessellation` every step (baseline = oracle path) |
| **S2** disp-Verlet     | global max-displacement vs skin | rebuild on trip; pure re-eval between |
| **S3** convex-local    | per-cell **D2 convexity** self-test | rebuild flagged cells off the stored skin list (no propagation), Verlet rebuild on trip/fallback |
| **S4** convex-prop     | **D2** + symmetry | rebuild flagged, **propagate** to asymmetric neighbours, iterate (star-splaying); Verlet rebuild on trip/fallback |
| **S5** | = S4 swept over skin | the optimal-skin study |

**D2 convexity certificate** (`ConvexCell::isSelfConsistent`): after re-eval on the fixed stored topology, the
cell is still correct iff every dual vertex lies inside every *other* plane (`n_k·v ≤ nn_k`); the vertex's own 3
defining planes are excluded (it lies on them). Catches **lost faces**; a **gained face** is invisible to the
cell itself — **S4's propagation** carries the flag to the partner (after a rebuilt cell's neighbour set
changes, any neighbour that doesn't list it back is re-flagged).

## Validation against ground truth

The oracle and all rebuilds are the production `buildTessellation` (validated: space-filling 1e-9, voro++
parity). So every number is against true Voronoi. The earlier self-contained harness builder was **bit-identical
in FP64** (round-off in FP32) but **2.5× (host) – 11× (GPU) slower** than `buildTessellation` (its gather
insertion-sorted a 128-NN list); routing rebuilds through the worklist both fixes the baseline and is the
production path. With the candidate cap at 256 (covers the tail-cell examined count), **`listMiss = 0`**
throughout: the candidate lists genuinely contain every true neighbour, so the residual is never list
incompleteness.

## Results — the rebuild baseline is the production worklist

The proper baseline is `buildTessellation` itself: **GPU RTX 5080 FP32 ~84 ms / 500k (~6 Mcell/s)**; host
OpenMP FP64 ~230–320 ms / 120k (host is far slower per cell here — allocation + CSR pack dominate at this N).
The speed-up of incremental updating is therefore **device-dependent: it scales with how expensive the full
rebuild is**, and is modest on the (fast) GPU.

### GPU RTX 5080, FP32, N=500k (baseline S1 = 84 ms/step)

| disp | S0 reeval | S2 Verlet | S3 independent | S4 propagating |
|------|-----------|-----------|----------------|----------------|
| 0.001 | 3.2 ms (26×), grows | 8.3 ms (**10×**), mism 1666 | 22.5 ms (**3.7×**), mism 56 | 52 ms (1.6×), mism 16 |
| 0.002 | 3.2 ms (26×) | 18.4 ms (4.6×), mism 48 | 35.6 ms (2.4×), mism 2 | 66.9 ms (1.3×), mism 0 |
| 0.005 | 3.2 ms | ~84 ms (1×) — all rebuild | ~92 ms (1×) | ~92 ms (1×) |

### host OpenMP FP64, N=120k (baseline S1 ≈ 234–318 ms/step; shared-box timing, ±)

| disp | S0 reeval | S2 Verlet | S3 independent | S4 propagating |
|------|-----------|-----------|----------------|----------------|
| 0.001 | 7.7 ms (30×) | 16.7 ms (**14×**), mism 403 | 37 ms (**6.3×**), mism 13 | 55 ms (4.2×), mism 2 |
| 0.002 | 10 ms | 38.6 ms (8.2×), mism 18 | 74 ms (4.3×), mism 0 | 98.5 ms (3.2×), mism 0 |

### S5 skin sweep (propagating), GPU FP32, disp 0.001

| skin | ms/step | rebuild% | mism |
|------|---------|----------|------|
| 0.05 | 51 | 18.8 | **0** |
| 0.10 | 53 | 6.2 | 14 |
| 0.20 | 54 | 0 | 47 |
| 0.40 | 54 | 0 | 45 |

On GPU the skin barely changes the time (~51–54 ms): the re-eval + detection + repair floor dominates and a full
rebuild is cheap, so **a *smaller* skin (more frequent but cheap rebuilds) is strictly better** — exact (mism 0)
at the same cost. On host (rebuild expensive) the opposite held: larger skin avoids the costly rebuild.

## Findings

1. **Geometry re-eval is nearly free, but it is not the per-step cost.** Pure `reevalGeometry` is ~26× (GPU) /
   ~30× (host) cheaper than a `buildTessellation` rebuild. But a usable strategy must also *detect* and *repair*
   topology changes, and that overhead (D2 sweep + compaction + propagation + the repair clips) dominates the
   per-step cost — so the realised whole-strategy speed-up is far below the raw re-eval ratio.
2. **The convexity certificate works; propagation makes it exact.** D2 alone (S3) leaves a small residual that
   grows with displacement; propagation (S4) drives it to ~0 (mism 0–16) with `listMiss = 0`. Empirical support
   for the conjecture: a topology flip is a convexity violation in ≥1 of the two cells, and propagation carries
   the repair to the partner.
3. **The net win is modest and device-dependent — this is the key correction.** Against the *fast production
   rebuild*, incremental updating buys, at disp 0.001: GPU **10× (S2) / 3.7× (S3) / 1.6× (S4)**; host **14× /
   6.3× / 4.2×**. It scales with rebuild cost, so the GPU (≈6 Mcell/s cold) gains least. By disp ≈ 0.005 every
   step rebuilds and the benefit is gone. (The earlier "23×" was an artifact of a too-slow harness baseline.)
4. **Independent vs propagating is a device-dependent trade-off.** Propagation buys exactness but costs ~2–3×
   the time of independent repair. On the GPU, where the rebuild is cheap, that cost isn't worth it —
   **independent repair (S3, near-exact, 3.7×)** or even just **Verlet (S2, 10×, moderate error)** dominate; and
   a small skin makes the cheap rebuilds frequent enough to stay exact. On host (expensive rebuild) propagation
   is worthwhile (4.2× exact).
5. **Precision:** D2 needs a precision-aware tolerance (`1e-4·cellSize` FP64, `2e-3·cellSize` FP32); with it,
   touch% and topoMism are ~precision-independent (topoMism ~10–35% is marginal-face drift D2 ignores by
   design — volume-irrelevant, not error). FP32 only worsens the borderline-flip residual (~5×). Do the topology
   decision in FP64; geometry/volume can be FP32.

## Recommended heuristic

The right strategy depends on the **rebuild cost** (device/N/density) and the **per-step displacement**:

- **Fast rebuild (GPU, ~6 Mcell/s) + small disp (≲0.002):** prefer **S3 independent repair** (near-exact,
  ~3–4×) or **S2 Verlet** (~10×) if a small volume error is acceptable; use a *small* skin so the cheap rebuilds
  keep it exact. Full propagation (S4) is rarely worth its overhead here.
- **Expensive rebuild (host / large cells / polydisperse / with force-geometry) + small disp:** **S4 propagating
  repair** pays off (exact, 4×+), and a *larger* skin avoids the costly rebuild.
- **disp ≳ 0.005 (any device):** just rebuild every step (`buildTessellation`); incremental updating no longer
  wins.
- Do the topology decision in FP64; FP32 for geometry/volume.

The production-faithful implementation is **S2/S3 with a Verlet skin + buildTessellation rebuilds** (now wired:
`buildTessellation` emits the topology store + skin list); S4 propagation is an opt-in exactness mode for the
expensive-rebuild regime.

## Caveats & future axes

- Host timings are noise-prone on the shared box (S1 varied 113–369 ms between runs); GPU numbers are the
  reliable quantitative ones, host is indicative.
- Studied: random Poisson seeds, single density, cells-only (no forces). Not swept: density / polydispersity,
  much larger N, multi-hop displacement, and **SDF boundary cells** — a boundary plane has no partner cell, so
  the convexity-symmetry argument needs an explicit boundary watch there.
- The per-step detection/repair overhead (compaction + per-sweep deep-copies) is not fully optimised; fusing it
  would raise the GPU speed-ups somewhat, but the qualitative device-dependence (cheap rebuild ⇒ modest win)
  stands.
