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

## Head-to-head vs geogram's ACTUAL code (Ray et al.'s `VBW::ConvexCell`), same machine

Built geogram's ConvexCell standalone (`-DSTANDALONE_CONVEX_CELL`) and ran it against ours on the
identical workload (N=1M, K=64 sorted neighbours, clip all, + volume), FP64, CPU. Harness =
`extern_bench/` (`build.sh` clones geogram + builds; `bench_geogram.cpp`). Both give Σvol err 2.77e-5
(identical cells).

| threads | geogram `clip_by_plane` | geogram `_fast` | ours | ours / geogram |
|---|---:|---:|---:|---:|
| **1 (pure algorithm)** | **94.6 k/s** | 49.7 | **82.2 k/s** | **0.87×** |
| 8 | 758.8 | — | 655.6 | 0.86× |
| 48 | 2754.6 | — | 2754.8 | 1.00× (bandwidth-saturated) |

**Findings:**
- **Ours is 0.87× geogram per core on CPU — a real but modest ~13–15% gap, not a 2×.** geogram's
  adjacency-walked horizon (O(border)) beats our `findSharing` (O(n)) on CPU (no divergence, good cache).
- **This is the OPPOSITE of GPU**, where our GPU port of that same adjacency structure
  (`convex_cell_adj.hpp`) was **0.46×** — adjacency's branchy pointer-walk is punished by GPU divergence,
  while the flat `findSharing` scan is coalesced. ⇒ the right cell structure is *architecture-dependent*;
  our `findSharing` cell is correct for our GPU target, geogram's adjacency cell for CPU.
- `clip_by_plane_fast` is *slower* here (49.7) — not geogram's best path for this usage.
- At 48 threads both saturate memory bandwidth (reading the 1.5 GB candidate array) → identical 2754 k/s.
- Our GPU construct (FP32) is **14.5 M/s** — geogram's heap-backed standalone cell can't run on GPU, so
  no direct GPU number, but the CPU 0.87× + the GPU adjacency 0.46× bound the cell-algorithm difference.

**Net: our construct is within ~15% of SOTA per-core on CPU and uses the GPU-correct structure. There is
no large hidden construct win in the SOTA cell.** A ~15% CPU gap exists but closing it (adopt adjacency)
would regress the GPU, which is our target — so not worth it.

## The actually-promising lead: avoid the no-op clips (point-in-cell, arXiv 2509.07175, 2026)

Read the paper (Xiao, Cao, Chen, IEEE TVCG, Feb 2026). **Fig. 1: 5M cells bounded by a cube, RTX 3080,
their method 0.49 s (white noise) / 0.26 s (blue noise) = 10–19 M cells/s, ~6× faster than the kNN-based
methods [24],[27] (2.81–3.12 s).** They DO produce the clipped cell geometry (domain-cell intersection),
not just combinatorics — so volumes are available; derivatives are not addressed.

**Their secret is directly relevant and untested by us:** they **avoid invalid/no-op clippings**. We
measured our construct examines ~70 candidates but only ~15 actually cut a face — the other ~55 are
wasted `clip()` calls that scan all triangles to find they don't cut. Their **point-in-cell / edge-
crossing test** predicts which bisectors actually contribute (walk the current cell's edges; a bisector
contributes iff an edge has one endpoint inside and one outside it) and clips ONLY those. This attacks
the real waste — the ~55 no-op clips — which neither our cell nor geogram's avoids. This is the most
promising path to a genuinely faster construct AND a cheaper gather (fewer candidates fully tested).
Hardware: RTX 2080 Ti / 3080. Code: not located (no GitHub link in the paper).

**Recommended next direction (if pushing construct further): implement the edge-crossing / point-in-cell
bisector test to skip no-op clips, rather than any further cell-representation tweak.**

## REAL SOTA GPU benchmark on the RTX 5080 (Liu et al. 2020 CUDA), measured

Built and ran an actual published SOTA GPU Voronoi CUDA code — **Liu/Ma/Guo/Yan, "Parallel Computation of
3D Clipped Voronoi Diagrams", TVCG 2020** (`github.com/xh-liu-tech/3D-Voronoi-GPU`), the ConvexCell+kNN
follow-up to Ray et al. — **on this RTX 5080**. Ported to CUDA 13.2 / sm_120 (arch flag, cublas path,
3 removed `cudaDeviceProp` fields; the texture code uses the modern object API so it built). Harness =
`extern_bench/voro_gpu_bench.md` (reproduction steps + the box-mesh generator + the timing patch).
Input: refined unit-cube tet domain + **1M uniform (white-noise) sites**, K=90 (their white-noise preset).
Added `cudaEvent` timing around exactly the gather (`kn_solve`) and the pure construct
(`voro_cell_test_GPU_param`); bypassed the restricted/tet-clip path (it crashes on our coarse box, and is
not the comparable phase).

| phase (1M sites, K=90, RTX 5080, FP32) | SOTA (Liu et al.) | ours |
|---|---:|---:|
| **gather** (grid kNN) | **17.5 Msites/s** (57 ms) | ~15 (F4 kNN query) |
| **construct** (ConvexCell clip) | **9.5 Mcells/s** (105 ms) | **14.5** (G1) / 12.0 (G2) |

**Findings — this settles the "are we at SOTA" question empirically:**
- **Our construct (14.5 M/s) is FASTER than this SOTA code's construct (9.5 M/s)** on the same GPU. Caveat:
  their `voro_cell_test` also writes the full cell topology (`vc.tr[]`, `vc.clip[]`) to global memory,
  while our G1 writes only volume — but even our G2 (full geometry + derivatives, 12.0) beats their 9.5,
  and we compute dV which they don't. So we are at/above SOTA on the construct, confirmed against running code.
- **The "~12 M/s" is confirmed as the right ballpark** for the per-phase GPU throughput: construct 9.5–14.5,
  gather 17.5 — consistent with Ray's V100 12.5 (full, volume-only) and the 2026 paper's 10–19 M/s. The
  full restricted-Voronoi-with-topology pipeline here is ~6 M/s on the 5080 (gather 57 ms + construct
  105 ms), i.e. **the same order as our own full build (6.7)** — the full-build number is gather-limited
  for everyone, not a sign we're behind.
- **How the SOTA gather hits 17.5 M/s** (`knearests.cu::knearest`, the part specifically asked about):
  (1) a **grid counting-sort** reorders points so a cell's neighbours are contiguous; (2) a per-thread
  **shared-memory K-max-heap** — the heap root is the running K-th-nearest distance, so most candidates
  are killed by ONE O(1) compare and only closer ones pay O(log K); (3) a **distance-sorted shell walk**
  over precomputed cell offsets; (4) **early termination** when the K-th-nearest-so-far is closer than the
  nearest possible point in the next shell. That is the gather recipe — and it's essentially the kNN
  pattern we already use; the 17.5 vs our ~15 gap is small.

**Net (empirical, on real SOTA code): we are at/above SOTA on the construct, comparable on the gather and
full build. The remaining headroom is the no-op-clip avoidance (point-in-cell, above), not the cell rep
and not the gather.**
