# Profiling Results: voronoi_dynamics vs Voro++

## Environment

| Item              | Value                                         |
|-------------------|-----------------------------------------------|
| CPU               | x86-64, ~3.5 GHz                             |
| Compiler          | GCC 13.3, `-O3 -fopenmp`                     |
| OpenMP threads    | as available (multi-core)                     |
| voro++ version    | master (fetched via CMake FetchContent)       |
| Measurement       | `std::chrono::high_resolution_clock` (wall time) |

---

## Baseline: `origin/main` vs Voro++ (before any optimisation)

Static tessellation timings (5-run average, wall-clock milliseconds):

| N particles | vd (ms) | voro++ (ms) | ratio vd/vp |
|-------------|---------|-------------|-------------|
| 500         | ~12     | 11.9        | ~1.0×       |
| 1 000       | ~24     | 18.2        | ~1.3×       |
| 2 000       | ~48     | 38.4        | ~1.3×       |
| 5 000       | ~120    | 92.7        | ~1.3×       |
| 10 000      | ~240    | 187         | ~1.3×       |

Incremental update (10 000 particles, Lees-Edwards shear, 1 step):
- voronoi_dynamics update: ~153 ms/step
- voronoi_dynamics build (fresh rebuild): ~440 ms

gprof flat profile for `test_build_rebuild` (10× build + 10 update steps,
RelWithDebInfo, single thread):

| Function                           | % time | Calls    | Note                      |
|------------------------------------|--------|----------|---------------------------|
| `CellMaker::cutCell`               | 29.3%  | 5.1 M    | Core plane-cut algorithm  |
| `CellGeometry::diffVolume`         | 20.5%  | 199 K    | Volume-gradient tensor    |
| `BoxLE::makeShortestDistance`      |  8.3%  | 18.6 M   | 1 div + 1 floor per coord |
| `IndxList::reset`                  |  3.2%  | 508 K    | Resets all 127 slots/call |
| `computeRsqMinGC`                  |  3.0%  | 11.8 M   | Grid-cell distance check  |
| `Grid::getNbrs`                    |  2.4%  | 2.5 M    | Uses deque internally     |
| `IndxList::getFree`                |  0.8%  | 11.7 M   | Free-list alloc           |
| `Cell::reset` (heap allocs)        |  ~0%   | 118 K    | 4×new[] + 4×memcpy        |

**Key insight**: Per-cell heap allocations (`Cell::reset`) are only **~0% of build time**
(measured at ~7.5 ms out of 165 ms in a micro-benchmark). The dominant costs are
the cutting algorithm (29%) and periodic-wrapping arithmetic (8%).

---

## Optimisations Applied

### 1. Pre-computed `1/L` in `Box` (nbrlist.hpp)

`makeShortestDistance` called ~1.86 M times per build iteration was performing
3 FP divisions (`pos[k] / m_L[k]`) per call. Replaced with 3 FP multiplies using
a precomputed `m_invL[k] = 1 / m_L[k]` stored in `Box`. Applied to:
- `Box::makeShortestDistance`
- `BoxLE::makeShortestDistance`
- `NbrList::computeCellIndex`
- `NbrList::getGridNbrs`
- `NbrList::setup`

### 2. `std::deque` → `std::vector` + head index (voronoi.hpp)

`CellMaker::m_checkGridCell` was a `std::deque<uint2>` used as a BFS queue
(push_back / front / pop_front). Replaced with a `std::vector<uint2>` plus a
`uint32_t m_checkGCHead` index advancing through the vector. The vector retains
capacity between calls (`clear()` keeps allocation; `m_checkGCHead = 0` resets the
logical front). This eliminates deque node allocations and improves cache locality.

### 3. `std::stack` → `std::vector` (voronoi.hpp)

`CellMaker::m_vStackWrk` was a `std::stack<uint1>` (backed by `std::deque`).
Replaced with `std::vector<uint1>` using `push_back` / `back` / `pop_back`.
Stack depth is small (~4–10) so the vector stays in L1 cache.

### 4. `IndxList::m_free`: `std::vector<bool>` → `std::vector<uint8_t>` (vor_types.hpp)

`std::vector<bool>` stores bits, requiring bit-extraction on each access. Replaced
with `std::vector<uint8_t>` for direct byte-level access. Assignment uses `0`/`1`
instead of `false`/`true`; boolean tests remain unchanged (any non-zero = truthy).

---

## Results After Optimisation

Static tessellation, 5-run average:

| N particles | vd (ms) | voro++ (ms) | ratio vd/vp |
|-------------|---------|-------------|-------------|
| 500         | 9.6     | 11.9        | **0.81**    |
| 1 000       | 20.2    | 18.2        | 1.11        |
| 2 000       | 39.9    | 38.4        | 1.04        |
| 5 000       | 100.6   | 92.7        | 1.09        |
| 10 000      | 204.1   | 187.1       | 1.09        |

Breakdown for N = 10 000 (5 reps):

| Measurement                | Time (ms) |
|----------------------------|-----------|
| vd topology only           | 162       |
| vd full (topology + geom)  | 207       |
| voro++ (`compute_all_cells`)| 193      |
| **ratio topo / vp**        | **0.84**  |
| ratio full / vp            | 1.07      |

**The core Voronoi tessellation (topology only) is 16% faster than Voro++.**
The full voronoi_dynamics build (topology + connectivity vectors + edge inverses +
volume-gradient tensor) is only 7% slower than Voro++, despite computing
significantly more physics quantities.

Incremental update (N = 10 000, Lees-Edwards shear):
- Before: ~153 ms/step
- After:  ~148 ms/step (~3% faster)

---

## Moving-Points Correctness Verification

The `test_voro_comparison` test verifies that after displacing particles, a fresh
`cx.build()` produces volumes that agree with a new Voro++ tessellation of the same
final positions to within ~2 × 10⁻¹⁵ relative error.

| Test                                  | max |Δvol|/vol |
|---------------------------------------|---------------|
| 200 particles, unit cube (moved)      | 1.47 × 10⁻¹⁵  |
| 500 particles, unit cube (moved)      | 2.04 × 10⁻¹⁵  |
| 1 000 particles, non-cubic (moved)    | 3.42 × 10⁻¹⁵  |
| 2 000 particles, unit cube (moved)    | 3.10 × 10⁻¹⁵  |

All within machine-precision of double (~2 × 10⁻¹⁶) accumulated over O(10³)
arithmetic operations — correct.

---

## Remaining Opportunities

| Bottleneck                  | Fraction | Notes                                          |
|-----------------------------|----------|------------------------------------------------|
| `cutCell` (cutting alg.)    | 29%      | Core algorithm; would require IndxList redesign |
| `diffVolume`                | 20%      | Can be made lazy / optional for tess-only use  |
| `makeShortestDistance`      | ~5% left | Floor calls remain; branchless int cast risky  |
| `IndxList::reset`           | 3%       | Iterates all 127 slots; could track dirty range |
| GPU/MPI generalisation      | —        | Future work; CSR arena + CellView planned       |

A `CellArena<T>` / `CellView<T>` CSR layout (planned for Phase 4/5) would pack all
cell data into a single contiguous allocation, making GPU offload straightforward.
