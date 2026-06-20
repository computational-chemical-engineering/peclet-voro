# Architecture Refactor Plan: Dynamic Pipeline & Wave-Based Graph Repair

## Context & Objective
We are implementing the dynamic update phase of a Data-Oriented Design (DOD) Voronoi simulation. Particles move, causing cells to stretch (Geometry Update) or break (Topology Repair). 

**The constraints are strict:**
1. Zero memory leaks during topology changes (must recycle overflow chunks).
2. No full spatial grid searches unless absolutely necessary (rely on temporal coherence).
3. No heavy memory zeroing (`memset` on `visited` arrays) in the inner loop.
4. Total eradication of the stateful `CellUpdater` class.

We will achieve this using a two-path pipeline: a **Fast Path** for convex cells, and a **Slow Path** using an iterative, wave-based Breadth-First Search (BFS) graph repair with OpenMP barriers.

---

## Phase 1: Memory Reclamation & Utility Upgrades

### 1. Upgrade `ChunkedPool` (Thread-Safe Free-List)
To prevent memory leaks when a cell drops an overflow chunk, implement a thread-safe LIFO free-list inside `ChunkedPool` (`vor_types.hpp`).
* **Add:** `std::vector<uint2> m_freeList;` and `std::mutex m_mutex;`.
* **Update `allocate`:** Lock mutex -> check `m_freeList`. If not empty, pop and return. If empty, use atomic `fetch_add` to allocate new space (lock only if vector resize is needed).
* **Add `releaseChunk(uint2 chunkIdx)`:** Lock mutex -> push `chunkIdx` onto `m_freeList`.

### 2. Implement Generation Counters in `ConstructionArena`
To track "visited" neighbors without the massive overhead of clearing a boolean array per cell, use a generation counter in the thread-local `ConstructionArena`.
* **Add Members:** `uint2 m_currentGeneration = 0;` and `std::vector<uint2> m_visitedGen;` (Sized to total number of particles `N`, initialized to 0).
* **Add Method:** `void startNewCell() { ++m_currentGeneration; }` (O(1) reset!).
* **Add Method:** `bool markAndCheckVisited(uint2 nbrId)`
  * If `m_visitedGen[nbrId] == m_currentGeneration`, return `true` (already visited).
  * Else, set `m_visitedGen[nbrId] = m_currentGeneration` and return `false`.

---

## Phase 2: Graph Separation & The `ConnectivityArena`

### 3. Create the `ConnectivityArena`
Extract neighbor tracking from the physical shape data. Create a new global storage class: `ConnectivityArena`.
* Uses the exact same `PrimaryOverflowArray` pattern as `TopologyArena`.
* **Storage:** `PrimaryOverflowArray<uint2, PrimaryF> m_nbrs;`
* **Purpose:** Holds the 1-ring neighbors for stable cells, and dynamically expands to hold 2-ring candidates during the update waves.
* **Overwrite Logic:** Implement `overwriteFromMaker(cellId, maker)` which checks for existing overflow chunks, calls `releaseChunk()` on them to recycle memory, and then inserts the new 1-ring neighbors. 
* *Note: Remove `m_nbrs` from `TopologyArena` to prevent duplication.*

### 4. Implement `TopologyArena::overwriteFromMaker`
Add the same overwrite logic here to recycle vertex/facet overflow chunks before inserting the new repaired topology.

---

## Phase 3: The Update Pipeline (Fast Path)

### 5. Create `CellComplex::update(...)` Fast Path
* Create a global `std::vector<uint2> active_queue;` (and a `next_queue`).
* `#pragma omp parallel`: Loop over all cells.
* Extract `CellView` and recompute geometry based on new particle positions.
* **Convexity Check:** If the cell remains convex, write new volumes/areas to `GeometryArena`.
* **Broken Topology:** If the cell inverts or is non-convex, do *not* update the geometry. Instead, push its `cellId` to a thread-local list, and merge these into the global `active_queue` at the end of the parallel block.

---

## Phase 4: The Wave Repair (Slow Path)

### 6. Implement the Iterative BFS Wave
Replace `CellUpdater` logic entirely with a `while (!active_queue.empty())` loop inside `CellComplex::update`.

Inside the loop:
1. `#pragma omp parallel for schedule(dynamic)` over `active_queue`.
2. **Fetch Candidates:** For `cellId`, read its current 1-ring and 2-ring candidates from `ConnectivityArena`.
3. **Fallback Check:** If candidate list is empty, query the spatial grid (`NbrList`) for candidates.
4. **Cut Planes:** Initialize `CellMaker`. Call `m_arena->startNewCell()`. Iterate over candidates, using `markAndCheckVisited()` to skip duplicates. Cut the cell.
5. **Evaluate Topological Rules:**
   * **Rule A (Lost Neighbor):** If an old 1-ring neighbor is *not* in the newly cut cell, add that neighbor to a `thread_local_next_queue`.
   * **Rule B (Promoted Neighbor):** If a 2-ring candidate becomes a direct 1-ring neighbor, add that new neighbor to `thread_local_next_queue`. Also, append this new neighbor's 1-ring to the current cell's 2-ring candidate pool for the next pass.
6. **Commit:** Call `m_topology.overwriteFromMaker(cellId, maker)` and `m_connectivity.overwriteFromMaker(cellId, maker)`. Recompute and commit geometry.

### 7. Synchronization Barriers
At the end of the `#pragma omp parallel for` block:
* Use `#pragma omp barrier` to ensure all threads have finished cutting and evaluating rules.
* `#pragma omp single`:
  * Merge all `thread_local_next_queue`s into the global `next_queue`.
  * Remove duplicates from `next_queue` (e.g., `std::sort` + `std::unique`).
  * `active_queue.swap(next_queue);`
  * `next_queue.clear();`

---

## Phase 5: Deletion & Cleanup

### 8. Eradicate Old Architecture
* **DELETE** `class CellUpdater` from `voronoi.hpp` and `simulation.hpp`.
* Ensure no instances of `std::vector<Cell>` remain.
* Ensure physics computations only query `CellView`, `GeometryView`, and now `ConnectivityView`.

## Success Criteria for Agent
1. `ChunkedPool` successfully recycles chunks via `m_freeList` without memory leaks.
2. `TopologyArena` and `ConnectivityArena` correctly release old chunks before allocating new ones during overwrites.
3. No `memset` or `std::fill` is used to clear visited neighbor arrays; the Generation Counter logic must be strictly utilized.
4. The simulation accurately resolves topological changes via iterative wavefronts, utilizing OpenMP barriers to avoid race conditions.