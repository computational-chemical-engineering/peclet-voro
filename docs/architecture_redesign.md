# Voronoi Dynamics: Data Structure Redesign & GPU Roadmap

## Executive Summary

The current `voronoi_dynamics` library is roughly **2× slower** than Voro++ for
static tessellation builds.  After careful analysis of both codebases the main
culprit is **not** the cutting algorithm itself — the half-plane cutting method
is sound and efficient — but the way per-cell data is **stored, accessed, and
copied**.  This document diagnoses the bottlenecks, proposes a new data layout
that is both CPU-cache-friendly and GPU-mappable, and sketches a phased
implementation plan.

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

2. **Fixed-capacity per cell with a dense count.**  Each cell gets a
   compile-time maximum number of vertex/facet slots (the existing
   `maxNumVertices = 128`, `maxNumFacets = 128` are fine).  Instead of a linked
   list, a simple `uint8_t numVertices` count says how many are active;
   compaction is done only when required.  This removes the `IndxList` entirely.

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

### 3.1  `CellArena` — The Global Cell Store (SoA)

```
CellArena<real_t>
│
├── numCells          : uint32_t
├── maxVerticesPerCell: uint8_t   (compile-time, e.g. 128)
├── maxFacetsPerCell  : uint8_t   (compile-time, e.g. 128)
│
│   ── Per-vertex arrays (contiguous, length = numCells × maxV) ──
├── vertexPos  : real_t[numCells × maxV × 3]   // x,y,z interleaved per vertex
├── vertexTopo : uint16_t[numCells × maxV × 3] // 3 half-edge labels per vertex
├── rSq        : real_t[numCells × maxV]        // |v|² cache
│
│   ── Per-facet arrays ──
├── facetLabel : uint16_t[numCells × maxF]      // starting half-edge label
├── facetNbr   : uint32_t[numCells × maxF]      // neighbour cell id
│
│   ── Per-cell scalars ──
├── numVertices: uint8_t[numCells]
├── numFacets  : uint8_t[numCells]
├── cellId     : uint32_t[numCells]
└── (optionally: volume, centroid, …)
```

**Indexing:**  vertex data for cell `c`, vertex `v` lives at
offset `c * maxV + v`.  This is a standard 2-D row-major layout that maps
trivially to GPU global memory.

**Alignment:**  each cell's vertex block starts at a
`maxV`-aligned boundary, which means the start of every cell's data is
naturally aligned for SIMD/warp-width access.

### 3.2  `CellView` — Lightweight Handle

A `CellView` is a non-owning accessor into the arena:

```cpp
template <typename real_t>
struct CellView {
  uint32_t cellIdx;         // which cell in the arena
  CellArena<real_t> *arena; // pointer to the arena

  // Accessors (inline, zero-overhead)
  real_t *vertexPos(uint8_t v)  { return &arena->vertexPos[(cellIdx*maxV + v)*3]; }
  uint16_t *vertexTopo(uint8_t v) { return &arena->vertexTopo[(cellIdx*maxV + v)*3]; }
  uint16_t &facetLabel(uint8_t f) { return arena->facetLabel[cellIdx*maxF + f]; }
  uint32_t &facetNbr(uint8_t f)   { return arena->facetNbr[cellIdx*maxF + f]; }
  uint8_t  &numVertices()         { return arena->numVertices[cellIdx]; }
  uint8_t  &numFacets()           { return arena->numFacets[cellIdx]; }
};
```

This replaces the current `Cell` class.  It costs **16 bytes** (index + pointer)
and is trivially copyable — no heap, no destructor, no copy overhead.

### 3.3  `CellMaker` Refactored — Direct-Write to Arena

Instead of maintaining its own private arrays, `CellMaker` receives a
`CellView` and writes directly into the arena:

```
CellMaker::build(cellIdx, pos, nbrList, arena)
  CellView view{cellIdx, &arena};
  initFromCuboid(view);          // write initial 8-vertex box into arena slot
  for each neighbour:
    cutCell(view, plane);         // modify the arena slot in-place
  compact(view);                  // remove free slots, update numVertices/numFacets
```

**Thread safety during the parallel build** is trivial: each cell index is
processed by exactly one thread, and cells do not share arena memory (each has
its own `maxV`-sized block).

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

### 4.4  MPI Domain Decomposition

For very large point sets the standard approach is:

1. **Spatial domain decomposition** — each MPI rank owns a rectangular sub-box.
2. **Ghost layers** — particles within one cut-off distance of a sub-box
   boundary are communicated to the neighbouring rank.
3. **Local build** — each rank builds the tessellation for its own particles
   (including ghosts).
4. **Communication of cell data** — after the build, each rank may need facet
   areas and volumes for ghost particles.  The SoA arena layout makes this
   trivial: contiguous slices of the arena can be sent with a single
   `MPI_Sendrecv`.

The SoA layout is again crucial here: if vertex data for cells 0..*N* is
contiguous in memory, and ghost cells are appended at positions *N*..*N+G*,
then the ghost data is a contiguous buffer that can be communicated without
packing.

---

## 5  Memory Budget Estimate

For *N* = 1,000,000 particles with `maxV = 128`, `maxF = 128`:

| Array | Element size | Total |
|-------|-------------|-------|
| `vertexPos` | 3 × 8 B × 128 × 10⁶ | 3.07 GB |
| `vertexTopo` | 3 × 2 B × 128 × 10⁶ | 0.77 GB |
| `rSq` | 8 B × 128 × 10⁶ | 1.02 GB |
| `facetLabel` | 2 B × 128 × 10⁶ | 0.26 GB |
| `facetNbr` | 4 B × 128 × 10⁶ | 0.51 GB |
| **Total** | | **~5.6 GB** |

This fits in a single modern GPU (e.g. A100 with 80 GB, or even an RTX 4090
with 24 GB).  For larger systems, MPI decomposition brings each rank below
this budget.

**Optimisation:** the actual number of vertices per cell is typically 10-20,
not 128.  A **variable-length** scheme (allocating only as many slots as
needed, plus some headroom) can reduce memory by 5-10×, at the cost of an
extra indirection (offset array).  This is the standard CSR (Compressed Sparse
Row) approach.  Decision: start with the fixed-capacity layout (simpler, more
GPU-friendly), and switch to CSR if memory becomes the bottleneck.

---

## 6  Relationship to the Incremental Update Path

The incremental update (`CellComplex::update`, `CellUpdater`) is one of the
main advantages of this library over Voro++.  The proposed changes **preserve
and improve** this path:

1. **`CellView`-based updates.**  The `CellUpdater` receives a `CellView` and
   modifies the arena in-place.  No copies needed.

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

### Phase 1 — Arena + CellView (CPU-only, no algorithm change)

**Goal:** replace `Cell` with `CellArena` + `CellView`; eliminate per-cell
heap allocations and copies.

| Step | Description | Files touched |
|------|-------------|---------------|
| 1a | Define `CellArena` and `CellView` in a new header `cell_arena.hpp` | new file |
| 1b | Refactor `CellMaker` to write directly into a `CellView` | `voronoi.hpp` |
| 1c | Refactor `CellComplex` to own a `CellArena` instead of `std::vector<Cell>` | `voronoi.hpp` |
| 1d | Update `CellGeometry` and `CellUpdater` to use `CellView` | `voronoi.hpp` |
| 1e | Update all tests to use new API | `tests/*.cpp` |
| 1f | Benchmark: compare static-build time against current version and Voro++ | `tests/test_voro_comparison.cpp` |

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
| 4a | GPU neighbour-list build (prefix sum + scatter) |
| 4b | GPU cell-build kernel (one warp per cell) |
| 4c | GPU geometry kernel (volumes, areas, forces) |
| 4d | GPU incremental update (flagged cells only) |
| 4e | CPU-GPU memory management (unified or explicit transfers) |

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
| 1 | **Single contiguous arena** instead of per-cell heap arrays | Eliminates 4*N* allocations, enables GPU coalesced access |
| 2 | **CellView** (non-owning handle) instead of `Cell` (owning class) | Zero-copy, trivially copyable, 16 bytes |
| 3 | **Direct write** from `CellMaker` into arena | Eliminates assignment-operator copy |
| 4 | **Dense slot allocator** instead of `IndxList` linked list | Sequential iteration, O(1) alloc/free, no `vector<bool>` |
| 5 | **Branchless periodic wrapping** | Remove `floor()` overhead |
| 6 | **SoA layout** for all per-vertex / per-facet data | Cache-line utilisation on CPU, coalesced reads on GPU |
| 7 | **Fixed-capacity per cell** (start simple, CSR later if needed) | GPU-friendly, deterministic memory, simple indexing |
| 8 | **Keep the cutting algorithm** | It is correct and efficient; only the container changes |

With Phases 1-3 the library should reach performance **parity with or better
than Voro++** on static builds, while retaining the incremental-update
capability that Voro++ lacks.  Phase 4 then unlocks GPU acceleration for very
large systems, and Phase 5 adds distributed-memory scalability.
