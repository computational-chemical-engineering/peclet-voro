# Project brief: voro++-style worklist gather for the cold Voronoi build (CPU)

> **STATUS — DONE (2026-06-26).** Implemented in `bench_convexcell` behind `CC_GATHER=1` (commits
> Phase A `a87e3d1`, Phase B `04a98cd`, Phase C `7ecba1e`). The worklist gather reaches **voro++ parity
> (≈1.0–1.05×)** on serial CPU FP64 — worklist ≈0.074–0.075 M/s vs voro++ ≈0.071–0.076 — up from the
> 0.89× the sorted-offset walk trailed by, machine-exact (Σvol err ~1e-14), clip and GPU path untouched.
> - **Phase A** — per-sub-region worklist (`S³`, `CC_WLS`, default 3): block offsets sorted by nearest-corner
>   dist² (`rmin`), radius break is a table lookup (no runtime per-block geometry). Cut gather 6.3 → 4.6 µs/cell.
> - **Phase B** — `rmax` whole-block-accept (`CC_WBA=1`) tried; **net loss** on the fine grid (≤~1 seed/block,
>   nothing to amortise) → gated off by default. `S=3` is the config that beats voro++.
> - **Phase C** — completeness guard: `exhausted=K` counts cells that drained the worklist without the radius
>   break; `K=0` over all operating-density N ⇒ provably complete (the sorted-offset walk has no such guard).
>
> Sections below are the original pre-work brief, kept for context. The deferred GPU/CPU-migration items in
> the memory index are unaffected. **Not pushed** (per milestone-commit convention).

Self-contained starting point for a fresh session. Goal: add a **worklist-based neighbour gather** as an
**option** beside the current fine-grid sorted-offset gather in `bench_convexcell`, to beat voro++ on
**serial CPU** at exact accuracy. Nothing else changes — the cell data structure and the clip are SOTA and
stay untouched; only the spatial search is new.

## 1. Where things stand (all committed, working tree clean)

Cold full build (gather + clip), machine-exact. Serial CPU FP64 = **67.6 kcells/s** (`CC_DENS=0.3`),
GPU FP32 = **6.99 Mcells/s** (`CC_DENS=0.5`, already beats the Liu-2020 SOTA 6.16). voro++ serial =
**75.7 kcells/s**. So serial CPU is **0.89× voro++** — the one place we trail. Full numbers + methodology:
`docs/voronoi_cold_tessellation_benchmark.md`.

**Decomposition (serial FP64, 1 M, µs/cell)** — each code's own clip-only subtracted from its own full,
volume cancels (measured via `extern_bench/bench_geogram`, which times geogram / voro++ / ours clip-only
on the same precomputed neighbours; voro++ clip = `voronoicell::plane`):

| | clip | gather (= full − clip) | full |
|---|---:|---:|---:|
| ours | **8.7** (our clip is SOTA) | 6.3 | 14.8 (67.6 k/s) |
| voro++ | 9.5 | **3.8** | 13.2 (75.7 k/s) |
| geogram | 10.6 | — | — |

**The entire gap is the gather: ours 6.3 vs voro++ 3.8 µs/cell (1.66×).** Our clip is already faster, so
to beat voro++'s full 13.2 µs we need the gather **< ~4.5 µs** (8.7 + 4.5 = 13.2). examined/cell: ours ~65
at the optimal `dens=0.3` (vs ~15.5 real faces — ~4× over-examination).

## 2. What was already tried and FAILED (do not repeat)

- **Whole-block-accept (runtime per-block min/max distance → skip / accept-all) on the fine grid**: NET
  LOSS (66.5 → 53.9 k/s). At `dens=0.3` we walk ~217 mostly-EMPTY blocks; the runtime per-block distance
  arithmetic costs more than the extra pruning saves.
- **Same, on a coarse grid (`dens` 2–8) to amortise it**: slower at every density AND inexact at coarse
  (box-err 5e-2 at `dens=4`) — our cell-corner `offD` block bound loosens as cells coarsen.
- **Expanding-shell gather (Chebyshev order)**: a correct radius break in Chebyshev order interacts badly
  with the shrinking radius (the sorted-offset list is the right structure). Abandoned.
- **Per-block periodic ±L displacement (branchless inner loop)**: CPU-neutral, GPU +5% — KEPT (it's the
  current gather). voro++'s real per-candidate win.

**Lesson:** voro++'s speed is the full *trifecta* — coarse blocks **+** a precomputed sub-position
**worklist** (so there is ZERO per-block distance arithmetic at runtime, just a table lookup + compare)
**+** whole-block acceptance. Porting only the whole-block piece onto our fine-grid `offD` walk adds
runtime cost without the worklist that pays for it. This project is to build the actual worklist.

## 3. What voro++ does (the thing to replicate) — `build/host-openmp/_deps/voropp-src/src/`

Read `v_compute.cc::compute_cell` and `worklist.hh`:
- **Coarse blocks** (container regions, ~8 particles/region default).
- **Precomputed worklists** (`worklist.hh`, a generated table; `wl_hgrid`³ sub-positions per the algorithm):
  for each sub-cell position within a block, a *sorted sequence of block offsets* with a precomputed radius
  threshold `radp[g]`. Indexed at runtime by the query's sub-position — no distance math per block.
- **Radius cutoff** `if (con.r_ctest(radp[g], mrs)) return;` — `mrs = c.max_radius_squared()` (the cell's
  current max vertex dist², recomputed periodically as clipping shrinks it). Stop when no untested block
  can reach the cell.
- **Whole-block min/max** (`compute_min_max_radius`): per block, min > radius → skip; max < radius → accept
  every particle with no per-particle radius test; else per-particle.
- **Dynamic unbounded expansion**: when the worklist is exhausted, go block-by-block off a grown queue
  (`add_to_mask`, `add_list_memory`) — never a fixed window.
- **Periodic shift per block** (region offset `qx,qy,qz`), not per particle.

## 4. Proposed design (our version, CPU-only, an OPTION)

Select the gather at runtime, default = current (so GPU + the SOTA-beating path are untouched):
`CC_GATHER` env (or a template tag on `build_sorted`): `0` = current sorted-offset (default), `1` = worklist.

**Precompute (host, once per N, in `run_once`):** subdivide each grid cell into `S³` sub-positions
(start `S=2`, sweep). For each sub-position `p`, build the sorted list of block offsets `(dx,dy,dz)` with
**two precomputed thresholds**: `rmin[p][g]` = nearest-corner dist² from p's sub-region to that block, and
`rmax[p][g]` = farthest dist². Store flat (indexed by `p` then `g`). The key point: both the radius break
(`rmin > 2·secR2`) and whole-block-accept (`rmax < 2·secR2`) are then table lookups — **no runtime
per-block geometry**, which is what killed the naive attempt.

**Runtime (the gather lambda):** compute the query's sub-position (one extra floor on the in-cell offset
`fx,fy,fz`), index its worklist, walk it: break on `rmin[p][g] > 2·secR2` (sorted ⇒ complete); per block,
`rmax[p][g] < 2·secR2` → cull-free inner loop, else per-candidate cull; per-block ±L periodic shift folded
in (already have). Run at a **coarse `CC_DENS`** (4–8) so few full blocks. Keep the `offD`/auto-`sw`
fallback to cover the rare cell whose radius exceeds the worklist (voro++'s dynamic expansion — or just a
generous window since exactness is gated by Σvol).

## 5. Validation gate + target

- **Exactness is non-negotiable:** `Σvol/box err` must stay ~1e-14 (FP64) at all densities — the bench
  prints it; the naive coarse attempt failed here (5e-2). Also cross-check `Σvol/voro err`.
- **Target:** serial CPU full build **> 75.7 k/s** (beat voro++) at exact accuracy. Stretch: gather → ~3.8
  µs/cell (voro++ parity). Compare in-run (the bench times voro++ unless `CC_NOVORO=1`).

## 6. Files & how to run

- `tests/kokkos/bench_convexcell.cpp` — the bench. `run_once` builds the counting-sort grid + the sorted
  offset list (`offX/offY/offZ/offD`) and the gather lives in the `build_sorted` KOKKOS_LAMBDA (the
  `#ifndef __CUDA_ARCH__` CPU path / `#else` GPU path). Add the worklist precompute + a `CC_GATHER==1` path.
- `extern_bench/bench_geogram.cpp` (+ `build.sh`) — clip-only head-to-head (geogram/voro++/ours).
- voro++ reference: `build/host-openmp/_deps/voropp-src/src/v_compute.cc`, `worklist.hh`.
- `docs/voronoi_cold_tessellation_benchmark.md` — current exact numbers.
- Build: `cmake --build build/host-openmp -j --target bench_convexcell`. Run:
  `OMP_NUM_THREADS=1 CC_DENS=<d> CC_GATHER=1 ./build/host-openmp/tests/kokkos/bench_convexcell 1000000`.
  CUDA on PATH (`/usr/local/cuda-13.2/bin`) for the GPU build (which must stay on the current gather).

## 7. Constraints

- **Do NOT change `ConvexCell` or the clip** — they are SOTA (clip 8.7 µs beats voro++/geogram). Spatial
  search only.
- **Keep it an option**; default stays the current gather (exact, beats SOTA on GPU). GPU keeps the
  branchless fine-grid gather.
- Commit at validated milestones (exact + faster), don't push.
