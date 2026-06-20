# Architecture Refactor Plan: Data-Oriented Voronoi Dynamics

## Context & Objective
The goal of this refactor is to transition the Voronoi dynamics library from an Object-Oriented, individually allocated model (`std::vector<Cell>`, `std::vector<CellGeometry>`) to a **Data-Oriented, Arena-based architecture**. 

This is a preparatory step for GPU portability (CUDA/HIP) and robust multi-threading. We must separate **Memory Ownership** from **Algorithmic Logic**. 
* **GPU Preparedness:** Logic classes (`CellMaker`, `CellGeometry`) will operate exclusively on raw pointers or lightweight "Views" rather than owning `std::vector`s. 
* **CPU Robustness:** Memory will be managed by "Arenas" using a block-allocation strategy to handle edge-case overflows without crashing, while avoiding the overhead of constant `std::vector` reallocations in tight OpenMP loops.

---

## Phase 1: Core Utilities and the "Scratchpad"

### 1. Replace `DenseSlots` with `DenseSlotsView` (`vor_types.hpp`)
Remove the template-capacity `DenseSlots` class. Replace it with `DenseSlotsView`. This class manages slot logic but **does not own memory**.
* Add a constant: `static constexpr UInt InvalidIdx = static_cast<UInt>(~0);`
* Replace arrays with pointers: `uint8_t* m_alive;` and `UInt* m_freeStack;`
* Add `void setStorage(uint8_t* alivePtr, UInt* stackPtr, UInt capacity);`
* Modify `getFree()` to return `InvalidIdx` if `m_numAllocated >= m_capacity` instead of crashing.

### 2. Create `ConstructionArena` (`voronoi.hpp`)
Create a new class `template <typename real_t> class ConstructionArena`. This acts as the thread-local "scratchpad" memory for `CellMaker`.
* **Strategy:** It should hold `std::vector` buffers for vertices (`std::array<real_t, 3>`), facets, neighbor tracking, and the `DenseSlotsView` control arrays (`uint8_t` alive flags, `uint1` free stacks).
* **Interface:** Provide a method `void ensureCapacity(uint1 minVertices, uint1 minFacets)` that resizes the internal vectors if the requested capacity exceeds current sizes.
* Provide getters that return `.data()` pointers to these buffers.

### 3. Refactor `CellMaker` (`voronoi.hpp`)
`CellMaker` must no longer own any `std::vector`s.
* **Remove:** All `std::vector` and `DenseSlots` members.
* **Add:** A pointer/reference to a `ConstructionArena<real_t>`.
* **Refactor Allocation:** In `allocVertexChecked` and `allocFacetChecked`, if `m_slotsV.getFree()` returns `InvalidIdx`, ask the `ConstructionArena` to double its capacity, re-bind the `DenseSlotsView` pointers via `setStorage()`, and retry. (This handles the overflow requirement).
* Update all algorithmic logic to read/write to the raw pointers provided by the `ConstructionArena`.

---

## Phase 2: Storage Arenas and Views

### 4. Create `TopologyArena` and `CellView` (`voronoi.hpp`)
Rename/upgrade `CellArena` to `TopologyArena`. This is the permanent packed storage for the entire tessellation.
* **Storage:** Maintain flat `std::vector`s for `vertexPos`, `vertices` (labels), `facets`, and `nbrs`. Maintain a `std::vector<CellSpan>` to track offsets and counts for each cell.
* **View:** Create a lightweight struct `template <typename real_t> struct CellView`.
  * Members: `uint2 id`, `uint1 numVertices`, `uint1 numFacets`, and raw `const` pointers to the cell's specific segment of the arrays in `TopologyArena`.
  * Add helper methods matching the old `Cell` class (e.g., `uint2 getNbr(uint1 i) const { return nbrs[i]; }`).

### 5. Create `GeometryArena` and `GeometryView` (`voronoi.hpp`)
Replace `std::vector<CellGeometry>` with `GeometryArena`.
* **Storage Strategy (SoA):** Use a Structure-of-Arrays approach. `GeometryArena` should contain flat `std::vector`s parallel to the number of cells: `m_volumes`, `m_dV` (flat array of facet volume derivatives), `m_areas`, etc.
* **View:** Create `template <typename real_t> struct GeometryView`.
  * Members: `uint2 id`, `real_t volume`, and pointers to the cell's `dV` and `areas` segments.
* Refactor the geometry computation functions (currently in `CellGeometry`) to be free functions or methods on `GeometryArena` that take a `CellView` and write directly into the `GeometryArena` buffers.

---

## Phase 3: Integration and Simulation Logic

### 6. Refactor `CellComplex` (`voronoi.hpp`)
* **Remove:** `std::vector<Cell> m_cells` and `std::vector<CellGeometry> m_geom`.
* **Add:** `TopologyArena<real_t> m_topology` and `GeometryArena<real_t> m_geometry`.
* **Update `build()`:** * Instantiate thread-local `ConstructionArena`s inside the `#pragma omp parallel` region.
  * Pass the thread-local arena to `CellMaker`.
  * After the OpenMP loop, "commit" the data from the scratchpads / temporary structures into the contiguous `TopologyArena`.
* **Update `update()` and `repair()`:** Adapt these to read from `TopologyArena` via `CellView`, use `ConstructionArena` for rebuilt cells, and update the Arenas.

### 7. Update Simulation Classes (`simulation.hpp`)
* Search for all instances where `Cell<real_t>` or `CellGeometry<real_t>` are materialized or accessed (e.g., `ExplicitEuler::computeForces`, `NavierStokes::computeForces`, `IntfDyn::computeIntfForces`).
* Replace calls to `m_complex.materializeCell(i, cell)` with requests for lightweight views: `CellView<real_t> cell = m_complex.getTopologyArena().getView(i);` and `GeometryView<real_t> geom = m_complex.getGeometryArena().getView(i);`.
* Update the physics logic to use the View interfaces. 

## Success Criteria for Agent
1. The code compiles without errors.
2. `std::vector` is completely removed from `CellMaker` and `DenseSlotsView`.
3. OpenMP parallel regions in `CellComplex::build` and `update` correctly utilize thread-local `ConstructionArena`s to prevent race conditions.
4. Out-of-bounds capacity errors in `CellMaker` dynamically trigger buffer resizing in `ConstructionArena` rather than aborting.