# Voronoi Dynamics: Data Structure Redesign & GPU Roadmap

## Executive Summary

The current `voronoi_dynamics` library is roughly **2× slower** than Voro++ for
static tessellation builds.  After careful analysis of both codebases the main
culprit is **not** the cutting algorithm itself — the half-plane cutting method
is sound and efficient — but the way per-cell data is **stored, accessed, and
copied**.  This document diagnoses the bottlenecks, proposes a new data layout
that is both CPU-cache-friendly and GPU-mappable, and sketches a phased
implementation plan.  A central design decision is to use **CSR (Compressed
Sparse Row)** storage on the CPU for memory efficiency, and a **two-tier
fixed-size layout (32 compact / 128 overflow)** on the GPU for coalesced
memory access without excessive waste.

---

## 1  Diagnosis — Where the Factor 2 Comes From

### 1.1  Per-Cell Heap Fragmentation

Every `Cell` owns **four separate `new[]` allocations** (`m_vertexPos`,
`m_vertices`, `m_facets`, `m_nbr`).  For *N* particles this means **4 *N***
small, disjoint heap blocks.  The allocator overhead alone (bookkeeping,
alignment padding, potential `malloc` lock contention under OpenMP) is
significant.  Worse, when the OpenMP parallel build loop copies a finished
`CellMaker` into `m_cells[i]`, the assignment operator calls `reset()` which
**deallocates and re-allocates all four arrays** and then copies them
element-by-element in four separate loops:

```cpp
// voronoi.hpp, lines 509-523  (assignment operator)
this->reset(rhs.m_numVertices, rhs.m_numFacets);   // 4× delete[] + 4× new[]
for (uint0 i(0); i < m_numVertices; ++i)
  this->m_vertexPos[i] = rhs.m_vertexPos[i];       // loop 1
for (uint0 i(0); i < m_numVertices; ++i)
  this->m_vertices[i] = rhs.m_vertices[i];         // loop 2
for (uint0 i(0); i < m_numFacets; ++i)
  this->m_facets[i] = rhs.m_facets[i];             // loop 3
for (uint0 i(0); i < m_numFacets; ++i)
  this->m_nbr[i] = rhs.m_nbr[i];                   // loop 4
```

Voro++ avoids this entirely: it works in-place on a single pre-sized buffer
per thread and only ever stores the final result once.

**Estimated impact: 20-30% of the gap.**

### 1.2  Linked-List Vertex Iteration (`IndxList`)

`CellMaker` tracks which vertex/facet slots are in use via `IndxList`, which is
a **singly-linked list over an index array** (`m_next[]`).  Every
"for-all-vertices" loop — and there are many in `cutCell2`, `computeAllDistGC`,
`renumber` — chases pointers through `m_next[]` in essentially **random order**.
This defeats the hardware prefetcher and triggers one L1 cache miss per vertex
on average.

The `release()` method is even worse: it does a **backwards linear scan** of
`m_free[]` (a `std::vector<bool>`, i.e. a bit-packed array with extra
extraction overhead) to find the insertion point.

Voro++ keeps a simple dense counter (`p`, the current vertex count) and
compacts on the fly.  No linked-list overhead at all.

**Estimated impact: 25-35% of the gap.**

### 1.3  Scattered Memory Accesses During Cutting

During `cutCell2`, the hot inner loop walks along edges via:

```cpp
v1 = getVertex(m_vertices[v1][k]);   // read topology at v1
m_dist[v1] = ...;                     // read/write distance at new v1
m_vertexPos[v1] = ...;                // read/write position at new v1
```

Because `m_vertexPos`, `m_vertices`, `m_dist` and `m_isKnownDist` are
**separate arrays**, every vertex access touches **four different cache lines**
that are not co-located.  If the vertex indices are non-sequential (which they
are after several cuts — the free-list fills holes arbitrarily), this yields
3-4 cache misses per vertex step.

**Estimated impact: 15-20% of the gap.**

### 1.4  Minor But Additive Costs

| Issue | Location | Note |
|-------|----------|------|
| `floor()` in periodic wrapping | `Box::makeShortestDistance` | 3 transcendental calls per neighbour pair |
| Grid index expand/compress | `Grid::expand` | 2 integer divisions per call |
| `std::deque` for grid-cell queue | `CellMaker::m_checkGridCell` | Poor cache locality |
| `std::vector<bool>` bit-packing | `IndxList::m_free` | Bit extraction overhead |

Combined these add another ~10%.

---

## 2  Design Principles for the New Layout

Before proposing concrete structures, here are the guiding principles, chosen
to simultaneously optimise for CPU caches **and** future GPU execution:

1. **Contiguous, coalesced memory.**  All data for the same field across all
   cells should live in a single flat buffer (Structure-of-Arrays = SoA).  On a
   GPU every thread in a warp then reads consecutive addresses → coalesced
   global-memory transactions.  On a CPU the hardware prefetcher sees simple
   strides.

2. **CSR (Compressed Sparse Row) storage on CPU.**  Each cell stores only as
   many vertex/facet slots as it actually uses (plus a small headroom factor).
   An offset array `offsets[c]` gives the start of cell `c`'s data, so
   `offsets[c+1] - offsets[c]` is the capacity for that cell.  This reduces
   memory by 5-10× compared to a fixed-128 layout (typical cells have 10-20
   vertices, not 128).  On the CPU, sequential access within a cell is
   cache-friendly regardless of the offset-based start address.

   For the GPU a different strategy is used — see §4.5.

3. **Avoid per-cell heap allocations.**  A single arena per `CellComplex` (or
   per thread for the build phase) is allocated once.  Cells are views/slices
   into this arena.

4. **Minimise copy traffic.**  During the build, the `CellMaker` should write
   **directly** into the final arena slot for cell *i*, not into a private copy
   that is later assigned.

5. **Keep the cutting algorithm unchanged.**  The half-edge mesh topology and
   the incremental plane-cutting method are correct and algorithmically
   efficient.  Only the *container* around the data changes.

---

## 3  Proposed Data Structures

### 3.1  `CellArena` — The Global Cell Store (CSR on CPU)

The CPU arena uses Compressed Sparse Row storage.  All vertex data for all
cells is packed contiguously, with an offset array to locate each cell's slice.

```
CellArena<real_t>  (CPU / CSR layout)
│
├── numCells            : uint32_t
│
│   ── Offset arrays (length = numCells + 1) ──
├── vertexOffsets : uint32_t[numCells + 1]  // vertexOffsets[c] = start of cell c's vertices
├── facetOffsets  : uint32_t[numCells + 1]  // facetOffsets[c]  = start of cell c's facets
│   (total vertices = vertexOffsets[numCells], total facets = facetOffsets[numCells])
│
│   ── Per-vertex arrays (contiguous, length = totalVertices) ──
├── vertexPos  : real_t[totalVertices × 3]    // x,y,z interleaved per vertex
├── vertexTopo : uint16_t[totalVertices × 3]  // 3 half-edge labels per vertex
├── rSq        : real_t[totalVertices]         // |v|² cache
│
│   ── Per-facet arrays (contiguous, length = totalFacets) ──
├── facetLabel : uint16_t[totalFacets]        // starting half-edge label
├── facetNbr   : uint32_t[totalFacets]        // neighbour cell id
│
│   ── Per-cell scalars ──
├── numVertices: uint8_t[numCells]
├── numFacets  : uint8_t[numCells]
├── cellId     : uint32_t[numCells]
└── (optionally: volume, centroid, …)
```

**Indexing:**  vertex data for cell `c`, local vertex `v`, lives at global
offset `vertexOffsets[c] + v`.  Facet data for cell `c`, local facet `f`,
lives at `facetOffsets[c] + f`.

**Memory efficiency:**  for typical cells with ~15 vertices and ~12 facets,
the CSR layout uses roughly **15/128 ≈ 12%** of the memory that a fixed-128
layout would consume.  The offset arrays themselves are negligible (two
`uint32_t` per cell ≈ 8 bytes vs. hundreds of bytes of saved vertex data).

**Build-phase strategy:**  during the parallel build, each `CellMaker` thread
works into a thread-local scratch buffer (fixed-size 128, small enough for the
stack or a per-thread pre-allocation).  After all cells are built, a single
pass computes the prefix-sum offsets and packs the results into the CSR arena.
This two-step approach avoids the need to predict per-cell sizes in advance.

### 3.2  `CellView` — Lightweight Handle (CSR)

A `CellView` is a non-owning accessor into the CSR arena:

```cpp
template <typename real_t>
struct CellView {
  uint32_t cellIdx;         // which cell in the arena
  CellArena<real_t> *arena; // pointer to the arena

  // Accessors (inline, zero-overhead)
  uint32_t vOff() const { return arena->vertexOffsets[cellIdx]; }
  uint32_t fOff() const { return arena->facetOffsets[cellIdx]; }

  real_t   *vertexPos(uint8_t v)  { return &arena->vertexPos[(vOff() + v) * 3]; }
  uint16_t *vertexTopo(uint8_t v) { return &arena->vertexTopo[(vOff() + v) * 3]; }
  uint16_t &facetLabel(uint8_t f) { return arena->facetLabel[fOff() + f]; }
  uint32_t &facetNbr(uint8_t f)   { return arena->facetNbr[fOff() + f]; }
  uint8_t  &numVertices()         { return arena->numVertices[cellIdx]; }
  uint8_t  &numFacets()           { return arena->numFacets[cellIdx]; }
};
```

This replaces the current `Cell` class.  It costs **16 bytes** (index + pointer)
and is trivially copyable — no heap, no destructor, no copy overhead.

### 3.3  `CellMaker` Refactored — Thread-Local Build + CSR Pack

During the parallel build each thread works into a **thread-local scratch
buffer** of fixed size 128 (small enough for the stack).  After all cells are
built, a single sequential pass computes the CSR offsets and packs results:

```
// Phase A — parallel build (one thread per cell)
CellMaker::build(cellIdx, pos, nbrList, scratchBuffer)
  initFromCuboid(scratchBuffer);
  for each neighbour:
    cutCell(scratchBuffer, plane);
  compact(scratchBuffer);
  // scratchBuffer now holds the finished cell with numV vertices, numF facets
  cellSizes[cellIdx] = {numV, numF};

// Phase B — sequential prefix sum
vertexOffsets = exclusive_prefix_sum(cellSizes.numV)
facetOffsets  = exclusive_prefix_sum(cellSizes.numF)
allocate arena with total sizes

// Phase C — parallel pack
for each cell c (parallel):
  memcpy(arena.vertexPos + vertexOffsets[c], scratch[c].vertexPos, numV[c] * ...)
  memcpy(arena.facetLabel + facetOffsets[c], scratch[c].facetLabel, numF[c] * ...)
  ...
```

### 3.4  Replacing `IndxList` with a Dense Slot Array

The linked-list free-list (`IndxList`) should be replaced by a simple scheme:

```
struct SlotAllocator {
  uint8_t slots[maxV];   // dense permutation: used slots at front
  uint8_t numUsed;
  
  uint8_t alloc()            { return slots[numUsed++]; }
  void    free(uint8_t idx)  { swap to back; --numUsed; }
  // iteration: for (int i = 0; i < numUsed; ++i) use slots[i]
};
```

Iteration is now a **sequential scan of a small dense array** — ideal for the
hardware prefetcher.  `alloc()` and `free()` are O(1) via swap.  The only cost
is that vertex indices may change after `free()` (swap moves the last used
element), but the cutting algorithm already handles renumbering.

An even simpler alternative for the cutting phase: use a **dense count** and
only compact once at the end (as Voro++ does).  New vertices are appended at
position `numUsed++`; deleted vertices are marked with a sentinel and skipped;
a final linear-time compaction pass renumbers everything.  This keeps the loop
body branch-free and sequential.

### 3.5  Neighbour-List Improvements

| Current | Proposed | Benefit |
|---------|----------|---------|
| `std::deque<uint2>` for grid-cell queue | Fixed-size ring buffer or `std::vector` with head/tail indices | Cache-friendly, no node allocation |
| `Grid::expand()` with 2 integer divides | Store pre-computed `(ix, iy, iz)` per cell, or use Morton (Z-order) curve index | Remove expensive divides; Z-order gives spatial locality |
| `Box::makeShortestDistance` with `floor()` | Replace with branchless `round()` or integer truncation: `int n = (int)(x * invL + 0.5); x -= n * L;` | Avoid transcendental `floor()`; ~3× faster |

---

## 4  GPU-Readiness Analysis

### 4.1  Can the Cutting Algorithm Run on a GPU?

**Yes**, with the proposed SoA layout.  The key observations are:

1. **Each cell is independent during the build phase.**  Cell *i*'s cutting
   only reads neighbour *positions* (read-only shared data) and writes to its
   own arena slot.  There are no data races.  This maps to **one GPU thread (or
   one warp) per cell**.

2. **The per-cell work is modest.**  A typical cell undergoes 10-20 cuts, each
   visiting ~10-20 vertices.  Total work per cell: a few hundred FLOPs and a
   few hundred integer operations.  This is a comfortable workload for a single
   GPU thread or a small cooperative group.

3. **The SoA layout gives coalesced access.**  If threads 0..31 (a warp) are
   building cells 0..31, then reading `vertexPos[c*maxV + v]` for the same `v`
   across all threads accesses 32 consecutive `real_t` values — one 128-byte
   memory transaction.

4. **The neighbour list is read-only after setup.**  The grid + cell-linked
   list can be built on the CPU (or GPU via parallel prefix sum + scatter) and
   then used as a read-only texture/buffer.

### 4.2  Challenges and Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| **Divergent branch in `cutCell2`** (gradient-descent vs exhaustive search) | Keep gradient descent as primary path; exhaustive fallback is rare (<1% of cuts). Warp divergence cost is acceptable. |
| **Variable per-cell work** (some cells have many neighbours) | Assign cells to warps, not individual threads. Use warp-level cooperative processing for the inner loop. |
| **Dynamic vertex allocation** (free-list) | The dense slot allocator (§3.4) is trivially GPU-compatible: `atomicAdd` on a per-cell counter (no contention since one thread per cell). |
| **`std::sort` in `processNbrs`** | Replace with a small fixed-size insertion sort or bitonic sort (n < 40 elements). |
| **`std::stack` in DFS deletion** | Replace with a fixed-size array-based stack (max depth = maxV). |

### 4.3  Proposed GPU Execution Model

```
Phase 1 — Neighbour List Build (GPU)
  ├── Kernel 1: compute cell indices          [1 thread / particle]
  ├── Kernel 2: parallel prefix sum (CUB)     [grid-wide]
  └── Kernel 3: scatter particles to bins     [1 thread / particle]

Phase 2 — Cell Build (GPU)
  ├── Kernel 4: init cuboids                  [1 thread / cell]
  └── Kernel 5: cut cells                     [1 warp / cell]
       └── Each warp:
           ├── Load neighbour positions from grid (coalesced reads)
           ├── Sort by distance (warp-cooperative insertion sort)
           └── Apply plane cuts sequentially (single-thread per warp,
               or warp-parallel for distance computations)

Phase 3 — Geometry (GPU)
  └── Kernel 6: volumes, areas, derivatives   [1 thread / cell]
```

### 4.4  CSR on GPU — Quantitative Assessment

Pure CSR (variable-length rows, offset-array indirection) is a natural fit for
CPUs but introduces two costs on GPUs:

| Concern | Severity | Explanation |
|---------|----------|-------------|
| **One extra indirection per access** | Low (~2-5%) | Reading `offsets[c]` is a single global-memory load that hits L2 cache after the first access per cell.  Negligible. |
| **Loss of coalesced access in geometry kernels** | Moderate (~15-30%) | When a geometry kernel maps one thread to one cell and all threads read the same local vertex index `v`, with fixed-size layout the addresses `c * stride + v` form a regular stride that the memory controller serves in a few wide transactions.  With CSR the addresses `offsets[c] + v` are scattered — each thread's data is at an unpredictable location — causing many small transactions instead of a few wide ones.  On modern GPUs (Ampere / Hopper) the enlarged L2 cache (40-60 MB) partially mitigates this, but a 15-30% throughput loss on memory-bound kernels is realistic. |
| **Build kernel impact** | Negligible (~0-5%) | The build kernel assigns one *warp* to one cell.  The warp accesses only its own cell's data sequentially — there is no cross-cell coalescing to lose.  CSR vs. fixed-size is irrelevant here. |
| **Incremental-update kernel impact** | Low-Moderate (~5-15%) | Only changed cells are processed, so the working set is small and likely fits in L2.  Scattered offsets matter less when the total data volume is low. |

**Bottom line:** CSR is *moderately* detrimental on GPU — not catastrophic, but
enough to matter for memory-bound geometry kernels that run every time step.
The build kernel (the most complex part) is essentially unaffected.

### 4.5  Two-Tier GPU Layout — The Recommended Middle Ground

Because the CSR penalty is moderate (not severe), a **pure fixed-128** layout
would waste 6-10× memory for typical 10-20 vertex cells.  Conversely, pure CSR
leaves 15-30% geometry-kernel performance on the table.  The best trade-off is
a **two-tier fixed layout**:

```
GPU CellArena (two-tier)
│
│   ── Tier 1: compact slots (capacity = 32 per cell) ──
│   Covers ~95% of cells (those with ≤ 32 vertices/facets)
├── vertexPos_T1  : real_t[numCells × 32 × 3]
├── vertexTopo_T1 : uint16_t[numCells × 32 × 3]
├── rSq_T1        : real_t[numCells × 32]
├── facetLabel_T1 : uint16_t[numCells × 32]
├── facetNbr_T1   : uint32_t[numCells × 32]
│
│   ── Tier 2: overflow slots (capacity = 128 per cell) ──
│   Only allocated for the ~5% of cells that exceed 32 vertices/facets
├── numLargeCells  : uint32_t
├── largeCellMap   : uint32_t[numLargeCells]        // maps overflow slot → cell id
├── vertexPos_T2   : real_t[numLargeCells × 128 × 3]
├── vertexTopo_T2  : uint16_t[numLargeCells × 128 × 3]
├── rSq_T2         : real_t[numLargeCells × 128]
├── facetLabel_T2  : uint16_t[numLargeCells × 128]
├── facetNbr_T2    : uint32_t[numLargeCells × 128]
│
│   ── Per-cell metadata ──
├── isLarge        : bool[numCells]                 // false → Tier 1, true → Tier 2
├── largeCellIdx   : uint32_t[numCells]             // index into Tier 2 (valid if isLarge)
├── numVertices    : uint8_t[numCells]
├── numFacets      : uint8_t[numCells]
└── cellId         : uint32_t[numCells]
```

**How it works:**

- **Common path (~95% of cells):** `isLarge[c] == false`.  Vertex data lives at
  `vertexPos_T1[c * 32 + v]`.  All threads in a warp access addresses with a
  stride of 32 — small enough for efficient coalesced transactions (one 128-byte
  transaction serves 4 consecutive cells for the same vertex).

- **Overflow path (~5% of cells):** `isLarge[c] == true`.  Vertex data lives in
  the Tier 2 arena at `vertexPos_T2[largeCellIdx[c] * 128 + v]`.  These cells
  are processed in a separate kernel launch (or a separate warp-group) to avoid
  divergence within a warp.

**Why 32?**  Empirical observation: for well-equilibrated particle distributions
the number of Voronoi vertices per cell is typically 10-20, and almost never
exceeds 30.  A capacity of 32 provides ~60-100% headroom and covers the vast
majority of cells.  It is also a natural GPU number (warp width), which means
each cell's 32-slot block aligns with memory transaction boundaries.

**Advantages over pure CSR:**

| Metric | Pure CSR | Two-tier (32 / 128) |
|--------|----------|---------------------|
| Memory (1M cells) | ~0.6 GB (optimal) | ~0.9 GB (Tier 1) + ~0.1 GB (Tier 2) ≈ 1.0 GB |
| Geometry kernel throughput | ~70-85% of peak | ~95-100% of peak (Tier 1 path) |
| Indexing complexity | offset lookup per access | simple `c * 32 + v` for common path |
| Build kernel throughput | ~same | ~same |

**Advantages over pure fixed-128:**

| Metric | Fixed-128 | Two-tier (32 / 128) |
|--------|-----------|---------------------|
| Memory (1M cells) | ~5.6 GB | ~1.0 GB |
| Geometry kernel throughput | ~same (both coalesced) | ~same for Tier 1 |
| Fits on GPU? | 1M cells barely fit on 24 GB GPU | 5-6M cells fit on 24 GB GPU |

### 4.6  MPI Domain Decomposition

For very large point sets the standard approach is:

1. **Spatial domain decomposition** — each MPI rank owns a rectangular sub-box.
2. **Ghost layers** — particles within one cut-off distance of a sub-box
   boundary are communicated to the neighbouring rank.
3. **Local build** — each rank builds the tessellation for its own particles
   (including ghosts).
4. **Communication of cell data** — after the build, each rank may need facet
   areas and volumes for ghost particles.  With CSR on CPU, ghost-cell data is
   already packed tightly and can be sent with a single `MPI_Sendrecv` per
   neighbour rank (no padding waste).  On GPU (two-tier layout), the Tier 1
   blocks for ghost cells form a contiguous region if ghost cells are appended
   at the end of the cell array.

Both the CSR layout (CPU) and the two-tier layout (GPU) support this pattern.
With CSR the ghost data is already tightly packed — no wasted padding.  With
the two-tier GPU layout, ghost cells are appended at positions *N*..*N+G* in
the Tier 1 arena; since most ghost cells are also small, the Tier 1 region
for ghosts is contiguous and can be transferred with a single memcpy or
`MPI_Sendrecv`.

---

## 5  Memory Budget Estimates

### 5.1  CPU — CSR Layout

For *N* = 1,000,000 particles, assuming an average of 15 vertices and 12 facets
per cell (typical for well-equilibrated systems):

| Array | Element size | Total |
|-------|-------------|-------|
| `vertexPos` | 3 × 8 B × 15 × 10⁶ | 0.36 GB |
| `vertexTopo` | 3 × 2 B × 15 × 10⁶ | 0.09 GB |
| `rSq` | 8 B × 15 × 10⁶ | 0.12 GB |
| `facetLabel` | 2 B × 12 × 10⁶ | 0.02 GB |
| `facetNbr` | 4 B × 12 × 10⁶ | 0.05 GB |
| Offset arrays | 2 × 4 B × 10⁶ | 0.008 GB |
| **Total** | | **~0.65 GB** |

This is **~8.6× smaller** than the fixed-128 layout (5.6 GB).  Even 10 million
particles fit comfortably in 6.5 GB of CPU RAM.

### 5.2  GPU — Two-Tier Layout (32 / 128)

For *N* = 1,000,000 particles, with ~5% overflow cells (50,000 large cells):

| Component | Calculation | Total |
|-----------|------------|-------|
| **Tier 1** (all cells × 32 slots) | (3×8 + 3×2 + 8 + 2 + 4) B × 32 × 10⁶ | **1.41 GB** |
| **Tier 2** (50K cells × 128 slots) | (3×8 + 3×2 + 8 + 2 + 4) B × 128 × 5×10⁴ | **0.28 GB** |
| Per-cell metadata | ~16 B × 10⁶ | **0.02 GB** |
| **Total** | | **~1.7 GB** |

This is **~3.3× smaller** than a pure fixed-128 layout and comfortably fits on any
modern GPU.  On a 24 GB consumer GPU (RTX 4090), this leaves ample room for
the neighbour list, geometry arrays, and simulation state.

### 5.3  Comparison

| Layout | 1M cells | 10M cells | GPU-friendly? |
|--------|----------|-----------|---------------|
| Fixed-128 | 5.6 GB | 56 GB | ✓ (coalesced) |
| CSR (CPU) | 0.65 GB | 6.5 GB | ✗ (scattered) |
| Two-tier 32/128 (GPU) | 1.7 GB | 17 GB | ✓ (mostly coalesced) |

---

## 6  Relationship to the Incremental Update Path

The incremental update (`CellComplex::update`, `CellUpdater`) is one of the
main advantages of this library over Voro++.  The proposed changes **preserve
and improve** this path:

1. **`CellView`-based updates.**  The `CellUpdater` receives a `CellView` and
   modifies the arena in-place.  With CSR, if a cell grows beyond its allocated
   capacity during an update (e.g. a new neighbour is inserted), the cell's
   data is relocated to a free region at the end of the arena (amortised O(1)
   with a growth factor).  In practice, incremental updates rarely change the
   vertex count by more than 1-2, so re-allocation is infrequent.

2. **Change detection.**  `m_hasChanged[c]` remains a simple boolean array.
   On GPU this becomes a device-side flag buffer; only flagged cells are
   scheduled for re-cutting.

3. **Neighbour insertion.**  The `NbrInsert` requests can be collected in a
   device-side append buffer (using `atomicAdd` for the count), sorted with a
   parallel sort, and processed in a second kernel.

4. **Convergence iteration.**  The repair loop (`while (nbrInserts not empty)`)
   maps to repeated kernel launches until a device-side counter reaches zero.
   This is a standard GPU iterative pattern.

---

## 7  Implementation Plan

### Phase 1 — CSR Arena + CellView (CPU-only, no algorithm change)

**Goal:** replace `Cell` with a CSR `CellArena` + `CellView`; eliminate
per-cell heap allocations and copies.

| Step | Description | Files touched |
|------|-------------|---------------|
| 1a | Define CSR `CellArena` and `CellView` in a new header `cell_arena.hpp` | new file |
| 1b | Implement two-step build: thread-local scratch → prefix sum → CSR pack | `voronoi.hpp`, `cell_arena.hpp` |
| 1c | Refactor `CellMaker` to work on a scratch buffer, then pack into CSR | `voronoi.hpp` |
| 1d | Refactor `CellComplex` to own a CSR `CellArena` instead of `std::vector<Cell>` | `voronoi.hpp` |
| 1e | Update `CellGeometry` and `CellUpdater` to use `CellView` | `voronoi.hpp` |
| 1f | Handle CSR re-allocation for incremental updates (grow-at-end strategy) | `cell_arena.hpp` |
| 1g | Update all tests to use new API | `tests/*.cpp` |
| 1h | Benchmark: compare static-build time against current version and Voro++ | `tests/test_voro_comparison.cpp` |

**Expected speedup: 1.3-1.5×** (from eliminating allocations and copies).

### Phase 2 — Dense Slot Allocator

**Goal:** replace `IndxList` with a dense slot array; remove `std::vector<bool>`
overhead and linked-list pointer chasing.

| Step | Description | Files touched |
|------|-------------|---------------|
| 2a | Implement `SlotAllocator` (or simple dense count + sentinel) | `vor_types.hpp` or new header |
| 2b | Refactor `CellMaker::cutCell2` to use dense iteration | `voronoi.hpp` |
| 2c | Refactor renumbering/compaction to use linear scan | `voronoi.hpp` |
| 2d | Benchmark again | `tests/test_voro_comparison.cpp` |

**Expected speedup: 1.3-1.5×** (from sequential memory access in inner loops).

### Phase 3 — Minor Optimisations

| Step | Description |
|------|-------------|
| 3a | Replace `floor()` in periodic wrapping with branchless integer arithmetic |
| 3b | Replace `std::deque` grid-cell queue with ring buffer or vector |
| 3c | Pre-compute `(ix, iy, iz)` grid coordinates to avoid `expand()` divisions |
| 3d | Replace `std::stack` in DFS with fixed-size array |

**Expected speedup: 1.1-1.2×** (from removing minor overheads).

### Phase 4 — GPU Port (CUDA / HIP)

| Step | Description |
|------|-------------|
| 4a | Implement two-tier GPU `CellArena` (Tier 1: 32 slots, Tier 2: 128 slots) with device memory management |
| 4b | GPU neighbour-list build (prefix sum + scatter) |
| 4c | GPU cell-build kernel (one warp per cell), writing to thread-local scratch, then packing into two-tier arena |
| 4d | Tier classification kernel: after build, flag cells with >32 vertices as `isLarge` and pack into Tier 2 |
| 4e | GPU geometry kernels — separate kernels for Tier 1 (coalesced, stride-32) and Tier 2 (stride-128) |
| 4f | GPU incremental update (flagged cells only); promote/demote between tiers as needed |
| 4g | CPU↔GPU transfer: CSR (CPU) → two-tier (GPU) converter for hybrid workflows |

### Phase 5 — MPI Integration

| Step | Description |
|------|-------------|
| 5a | Spatial domain decomposition |
| 5b | Ghost-layer communication of positions |
| 5c | Ghost-layer communication of cell data (contiguous arena slices) |
| 5d | Load balancing (redistribute cells when imbalanced) |

---

## 8  Summary of Key Recommendations

| # | Recommendation | Rationale |
|---|----------------|-----------|
| 1 | **CSR arena on CPU** instead of per-cell heap arrays | Eliminates 4*N* allocations; 8-9× less memory than fixed-128; tight packing for MPI |
| 2 | **Two-tier GPU arena (32 / 128)** | Coalesced access for 95% of cells (stride-32); only 5% overflow to Tier 2; ~3× less memory than fixed-128 |
| 3 | **CellView** (non-owning handle) instead of `Cell` (owning class) | Zero-copy, trivially copyable, 16 bytes; works with both CSR and two-tier backends |
| 4 | **Thread-local scratch + pack** for the build phase | Avoids predicting cell sizes; trivially parallel; single prefix-sum to compute CSR offsets |
| 5 | **Dense slot allocator** instead of `IndxList` linked list | Sequential iteration, O(1) alloc/free, no `vector<bool>` |
| 6 | **Branchless periodic wrapping** | Remove `floor()` overhead |
| 7 | **Separate geometry kernels** for Tier 1 and Tier 2 on GPU | Avoids warp divergence; each kernel has uniform stride |
| 8 | **Keep the cutting algorithm** | It is correct and efficient; only the container changes |

With Phases 1-3 the library should reach performance **parity with or better
than Voro++** on static builds, while retaining the incremental-update
capability that Voro++ lacks.  Phase 4 then unlocks GPU acceleration for very
large systems (with the two-tier layout providing near-peak memory bandwidth),
and Phase 5 adds distributed-memory scalability.
