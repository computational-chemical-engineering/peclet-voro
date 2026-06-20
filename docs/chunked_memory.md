# Architecture Refactor Plan: Data-Oriented Memory Arenas

## Context & Objective
We are finalizing the transition to a Data-Oriented Design. The goal is to completely eradicate the `Cell` class and implement a **Primary Block + Overflow Pool** memory strategy for the global arenas (`TopologyArena` and `GeometryArena`). This guarantees zero race conditions on the CPU, ensures memory coalescence on the GPU, and prevents expensive global reallocations. 

We will use **Composition** to manage the Structure of Arrays (SoA).

---

## Phase 1: Core Memory Primitives

### 1. Implement `ChunkedPool<T>` (The Overflow Allocator)
Create a self-contained memory pool in `vor_types.hpp` that allocates memory in fixed-size blocks (chunks). This prevents pointer invalidation, which is crucial for thread safety.
* **Storage:** `std::vector<std::unique_ptr<T[]>> m_chunks;`
* **State:** `std::atomic<size_t> m_globalOffset;` and `size_t m_chunkSize;` (e.g., 1024).
* **Allocation Method:** `T* allocate(size_t count, uint2& out_overflow_idx)`. 
  * Threads use `m_globalOffset.fetch_add(count)` to claim space.
  * If the offset exceeds current chunk capacity, a thread-safe mechanism allocates a new chunk.
  * It returns the raw pointer to the start of the claimed space and an integer index (`out_overflow_idx`) that the View can use to locate it later.

### 2. Implement `PrimaryOverflowArray<T, PrimaryCap>`
Create this reusable templated data structure to manage a single property (e.g., vertices, facets, or areas).
* **Storage:** * `std::vector<T> m_primary;` (Sized exactly to `NumCells * PrimaryCap`).
  * `ChunkedPool<T> m_overflow;`
* **Registry:**
  * `std::vector<uint1> m_counts;` (How many items cell `i` actually has).
  * `std::vector<uint2> m_overflowIdx;` (Index in the `ChunkedPool`. Default to `~0` / `InvalidIdx`).
* **Interface:** Provide a method `void insert(uint2 cellId, const T* data, uint1 count)` that writes directly into `m_primary` up to `PrimaryCap`, and routes the rest to `m_overflow`.

---

## Phase 2: The Specific Arenas

### 3. Update `ConstructionArena` (Thread-Local Scratchpad)
This arena is thread-local and processes one cell at a time. It does **not** need the Primary/Overflow split.
* Use standard `std::vector<T>` for the buffers (`vertexPos`, `vertices`, `facets`, `nbrs`).
* Implement a `reset()` method that sets the logical size to 0 but *keeps the capacity*.
* Implement `ensureCapacity()` to grow the vectors only if a specific edge-case cell exceeds the current capacity.

### 4. Build `TopologyArena`
Replace the old `CellArena` implementation. Use the new generic components.
* Set constants: `static constexpr uint1 PrimaryV = 28;` and `static constexpr uint1 PrimaryF = 18;`
* **Members:**
  * `PrimaryOverflowArray<std::array<real_t, 3>, PrimaryV> m_vertexPos;`
  * `PrimaryOverflowArray<Vertex, PrimaryV> m_vertices;`
  * `PrimaryOverflowArray<uint1, PrimaryF> m_facets;`
  * `PrimaryOverflowArray<uint2, PrimaryF> m_nbrs;`

### 5. Build `GeometryArena` (SoA Layout)
This arena holds computed physics data. Note that some data maps 1:1 with cells, while other data maps 1:N with facets.
* **1:1 Data:** `std::vector<real_t> m_volumes;` (Flat array, sized to `NumCells`).
* **1:N Data (Facet properties):**
  * `PrimaryOverflowArray<std::array<real_t, 3>, TopologyArena::PrimaryF> m_dV;`
  * `PrimaryOverflowArray<std::array<real_t, 3>, TopologyArena::PrimaryF> m_areas;`

---

## Phase 3: The Views and Eradication of `Cell`

### 6. Implement `CellView` and `GeometryView`
These are lightweight structs (handles) instantiated by physics kernels to read data seamlessly, abstracting the Primary vs. Overflow logic.
* **`CellView` Example:**
  ```cpp
  template <typename real_t>
  struct CellView {
      uint2 id;
      TopologyArena<real_t>* arena; // Pointer to the parent arena

      inline uint1 numFacets() const { return arena->m_facets.m_counts[id]; }
      
      inline uint2 getNbr(uint1 i) const {
          if (i < TopologyArena::PrimaryF) {
              return arena->m_nbrs.m_primary[id * TopologyArena::PrimaryF + i];
          } else {
              uint2 overIdx = arena->m_nbrs.m_overflowIdx[id];
              return arena->m_nbrs.m_overflow.get(overIdx + (i - TopologyArena::PrimaryF));
          }
      }
  };
  ```

### Replacing `Cuboid`

**Context:** The `Cuboid` class currently inherits from `Cell` and is passed to `CellMaker::build` to initialize the starting geometry. Since we are deleting `Cell`, `Cuboid` cannot exist in its current object-oriented form.

**The Solution: Direct Initialization Method**
Instead of passing a `Cuboid` object, we will give `CellMaker` a method to directly initialize its `ConstructionArena` as a cuboid.

1. **Delete the old `Cuboid` class:** Remove `template <typename real_t> class Cuboid : public Cell<real_t>` entirely from `voronoi.hpp`.
2. **Add `initAsCuboid` to `CellMaker`:**
   Add the following method to `CellMaker`. This method bypasses the need for a `Cell` object and directly writes the 8 vertices and 6 facets into the `ConstructionArena` (using the same math and `makeLabel` logic from the old `Cuboid` constructor).

   ```cpp
   template <typename real_t>
   void CellMaker<real_t>::initAsCuboid(const std::array<real_t, 3> &L) {
       // 1. Reset the slots for exactly 8 vertices and 6 facets
       m_slotsV.reset(8);
       m_slotsF.reset(6);
       
       // Ensure the ConstructionArena buffers are large enough
       m_arena->ensureCapacity(8, 6);
       
       // 2. Refresh raw pointers from the arena
       m_vertexPos = m_arena->m_vertexPos_buf.data();
       m_vertices  = m_arena->m_vertices_buf.data();
       m_facets    = m_arena->m_facets_buf.data();
       m_nbr       = m_arena->m_nbr_buf.data();

       // 3. Set all neighbors to noNbr
       for (uint0 i = 0; i < 6; ++i) {
           m_nbr[i] = noNbr;
       }

       // 4. Copy the exact coordinate math from the old Cuboid constructor
       m_vertexPos[0][0] = -0.5 * L[0];
       m_vertexPos[0][1] = -0.5 * L[1];
       // ... (Copy the rest of the 8 vertex positions here) ...

       // 5. Copy the topology labels from the old Cuboid constructor
       m_facets[0] = makeLabel(0, 0, 0);
       m_vertices[0][0] = makeLabel(2, 1, 2);
       // ... (Copy the rest of the makeLabel assignments here) ...
       
       // 6. Reset distances for the cutting algorithm
       computeAllRsq();
       resetDist();
   }
   ```

3. **Update `CellMaker::build`:**
   * Change the signature of `build` so it no longer takes `const Cell<real_t> &initCell`. Instead, it takes `const std::array<real_t, 3> &L`.
   * Inside `build`, replace `this->init(initCell);` with `this->initAsCuboid(L);`.

4. **Update `CellComplex::build`:**
   * Remove the line `Cuboid<real_t> cub(L);`.
   * Pass `L` directly into `maker.build(i, p, m_nbrList, L);`.

### 8. The Hard Deletion (Eradicate `Cell`)
* **DELETE** `class Cell` and `class Cuboid` from `voronoi.hpp`.
* **DELETE** `std::vector<Cell<real_t>> m_cells;` from `CellComplex`.
* Any code in `simulation.hpp` (e.g., `ExplicitEuler::computeForces`) that previously materialized a `Cell` object must now call `CellView<real_t> cell = m_complex.getTopologyArena().getView(i);`.

## Success Criteria for Agent
1. The codebase relies entirely on `TopologyArena` and `GeometryArena` for global storage.
2. The `Cell` class no longer exists.
3. The `PrimaryOverflowArray` transparently handles data insertion without requiring heavy global thread locks.