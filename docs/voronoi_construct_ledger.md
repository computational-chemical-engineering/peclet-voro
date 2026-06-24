# ConvexCell construct — optimization ledger (so we stop running in circles)

A running record of **every** attempt to push the single-thread ConvexCell construct toward SOTA, with
the measured result. RTX 5080, FP32, N=1M, construct-from-cache (no gather) unless noted. The construct
is the cell-building step (clip + retriangulation + volume/derivatives); the **gather** (neighbour
finding) is a separate concern tracked in `voronoi_build_plan.md` — do not conflate them.

## Target
SOTA = Ray, Sokolov, Lefebvre, Lévy, *Meshless Voronoi on the GPU* (TOG 2018): **10 M points in 800 ms
on a V100 = 12.5 M/s for the FULL build** (gather + construct). Code = geogram `VBW::ConvexCell`.

## Current standing
- construct-from-cache: **G0 17.3 / G1 14.0 / G2 12.0 M/s** (LaunchBounds<256,4>: G1 14.5).
- full cold build (gather + construct): **6.7 M/s** (F4 kNN) / 6.45 (tuned fused grid).
- So our *construct alone* already ≈ Ray's *full-build* number; the cold-build gap is the gather.

## What the kernel is bound on (measured, not assumed)
ptxas: 56 reg, **3536 B stack frame, 0 spill** — the cell is local-memory-resident because its arrays
are *dynamically indexed* (not because of size). At runtime it uses ~0.1% of compute and ~4.5% of
bandwidth. Conclusion: bound by the **intrinsic serial work of the clip chain** — per-candidate
`maxVertexRsq` (O(nt)), per-clip kill-scan (O(nt)), and especially `findSharing` (**O(nt) per horizon
edge**) + `allocTri` (**O(nt) per new triangle**). NOT memory/footprint/occupancy bound (see below).

## Attempts (chronological), each with its verdict

| # | Attempt | Result | Verdict |
|---|---------|--------|---------|
| 1 | **Best-first BVH traversal** (gather, Option 2): per-cell min-heap over ArborX nodes, exact security stop | GPU 1.5 vs 6.7 M/s | ✗ **0.22×** — divergent per-thread heap; reverted as char. |
| 2 | **Spatial (binned) seed order** in the fused build | +16–20% | ✓ small win, kept |
| 3 | **Morton/Z-order** point ordering vs row-major | ~0% | ✗ gather is not memory-latency-bound |
| 4 | **Branchless periodic wrap** vs integer modulo | ~0% | ✗ not modulo-bound |
| 5 | **Cache `secR2`** between clips (recompute only on cut) | ~0% | ✗ not the cost |
| 6 | **Grid density / search-window** sweep (fewer examined) | 5.5→6.45 | ✓ tuned win (gather), plateaus at kNN parity |
| 7 | **Shrink cell caps** MAXP/MAXT (overflow=0): 112→68 | ~0% | ✗ footprint-via-cap is not the bound (the "+47% at 24/44" was overflow cells bailing — artifact) |
| 8 | **Compact cell**: packed 32-bit triangle (16→4 B) + **no vertex cache** (recompute via Cramer); cell 3084→1484 B | **0.72×** | ✗ smaller AND fewer regs yet slower → **NOT footprint/occupancy bound**; recompute's division in the kill-scan costs more than the saved memory |
| 9 | **Occupancy** via LaunchBounds<256,4> | **+10%** (14.5 M/s) | ✓ small win; harder (>=6 blk) = register spill, negative |
| 10 | **Explicit triangle adjacency** (earlier, pre-ledger) | "GPU-negative" | ✗ — but suspected paired with the vertex cache (footprint bloat); see plan |
| 11 | **Incremental security radius** (earlier, pre-ledger) | "GPU-negative" | ✗ |
| 12 | **Warp-cooperative cell** (one cell across 32 lanes) | not built | ✗ contraindicated: occupancy test (#9) says we want MORE cells in flight; 1-cell-per-warp = 32× fewer; earlier naive try ~0.6× |

## The SOTA secret (read from geogram `VBW::ConvexCell`, convex_cell.{h,cpp})
Their triangle (14 B) = `{ushort i,j,k}` (3 plane indices) + `ushort flags` (linked-list chaining +
conflict bit) + **`t_adj_[t][3]`: three adjacency pointers** (neighbour triangle per edge). Vertices are
**recomputed** (`compute_triangle_point`), cached only on demand. The clip:
1. **conflict detection**: plain linear scan over the *linked list of valid triangles* (they explicitly
   prefer linear scan — "no more than a few tens of vertices" — so this is NOT the lever);
2. **horizon retriangulation**: walk the conflict-zone border **via adjacency pointers**, one new
   triangle per border edge, linking new triangles into a ring — **O(border)**, never an O(n) search;
3. **allocation** from a **free list** — O(1).

⇒ The three things we do as O(n) searches (`findSharing` per edge, `allocTri` per triangle, scanning
dead slots) SOTA does as O(1)/O(border) via **adjacency + free list + linked-list of valid triangles**.
These O(n) searches ARE our measured bottleneck. The untested-by-us combination is **adjacency-walked
horizon + recompute vertices** (attempt #8 tried recompute *without* adjacency → slower; the earlier
adjacency try (#10) was *with* the vertex cache). 

## RESULT — the SOTA construct structure was implemented and it LOSES on this GPU

`convex_cell_adj.hpp` (geogram-faithful: triangle = 3 plane indices + 3 adjacency pointers + free-list,
conflict zone walked via adjacency) was built and A/B'd both ways (vertices recomputed AND cached), plus
a packed-recompute variant (`convex_cell_compact.hpp`). Construct-from-cache, RTX 5080, FP32, N=1M, all
bit-identical volumes (Σvol err 3.5e-5):

| cell | G0 (topo) | G1 (+vol) | vs cached |
|---|---:|---:|---:|
| **ConvexCell — cached vertex + findSharing (current)** | **17.3** | **14.0** | **1.00×** |
| Compact — packed tri + recompute | 11.5 | 9.9 | 0.72× |
| Adj — adjacency + free list + recompute | 7.0 | 6.5 | 0.46× |
| Adj+cache — adjacency + free list + cached vertex | 6.9 | 6.5 | 0.46× |

**Verdict: every redesign is equal or slower; the simple cached `findSharing` cell is optimal.** Key
facts established:
- Adj+recompute and Adj+cache are **identical (0.46×)** ⇒ the slowdown is the **adjacency machinery
  itself** (ring-linking O(new²), free list, scatter writes to `tadj`, extra divergence), NOT the
  recompute. For a ~28-triangle cell the O(n) linear `findSharing` scan WINS — it's tiny, branch-
  predictable, low-state. **geogram says the same** ("we prefer complete linear scan… no more than a few
  tens of vertices"). The earlier "adjacency GPU-negative" was correct.
- Compact (recompute, no adjacency) = 0.72× ⇒ recompute's division in the per-triangle conflict scan
  costs more than the local-memory it saves ⇒ not footprint-bound (also shown by attempt #7/#8).
- Caching `maxVertexRsq` + `break` on sorted candidates = **0% change** (attempt #5 again).
- Occupancy `LaunchBounds<256,4>` = +5–10% (→14.5–14.7). The only positive, and small.

**CONCLUSION: the construct is at its ceiling, ~14.5 M/s, which already ≥ Ray's V100 full-build 12.5.**
The "SOTA secret" does not exist as a *construct* win on this GPU — SOTA's number is a FULL-BUILD figure
and is dominated by the gather + hardware, not a faster cell algorithm. **Do not reopen the construct.**
The full-build gap (6.7 vs 12.5) is the gather and the fused-pipeline overlap; the genuinely large,
likely win is Part II (moving points, topology reuse, no gather).

## Newer references to check
- *Efficient Computation of Voronoi Diagrams Using Point-in-Cell Tests* (arXiv 2509.07175, 2025) —
  possibly a different/faster construct; not yet read.
