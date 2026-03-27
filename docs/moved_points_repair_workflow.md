# Workflow for Repairing the Tessellation After Points Move

## Scope

This document describes how the tessellation is updated after particle positions change, focusing on the incremental repair path used by `CellComplex::update` and `CellComplex::repair`.

The goal is to avoid a full rebuild when possible, while restoring a valid convex Voronoi tessellation and consistent neighbor connectivity.

## Entry Point

Typical simulation integration updates positions, then calls:

- `CellComplex::update(const std::vector<Array<real_t, 3>> &p)`

If the point count changed or the complex is not yet built, `update` falls back to `build(p)`.

## High-Level Strategy

The update path has three phases:

1. Detect potentially invalid cells after motion.
2. Locally rebuild those cells from previous neighbors.
3. Iteratively repair neighbor consistency by exchanging insertion proposals.

After repair, packed storage and geometry caches are regenerated.

## Detailed Step-by-Step Flow

### 1. Reset updater state

At the start of `update`:

- All `CellUpdater` instances are reset.
- `m_hasChanged[i]` is cleared for all cells.

This ensures repair starts from a clean per-step state.

### 2. Recompute geometry and detect invalid cells

For each cell `i` (parallel loop):

1. `computeConnectingVectors(p, box)`
2. `computeEdgeInv()`
3. `updateVertexPos()`
4. `isConvex()`

If convexity fails, the cell is flagged as changed:

- `m_hasChanged[i] = true`
- `numChanged++`

If `numChanged == 0`, `update` returns early.

### 3. Rebuild changed cells from their current neighbor set

For each changed cell:

1. Initialize `CellMaker` from stored cell topology (`maker = m_cells[i]`).
2. Call `maker.rebuild(p, box, cuboid)`.
   - `rebuild` gathers current neighbors from facets.
   - Reinitializes from a cuboid.
   - Re-cuts using only gathered neighbors.
3. Write result back: `m_cells[i] = maker`.

Cells that end up empty or neighborless are recorded in `emptyCells`.

### 4. Fallback full local rebuild for empty cells

If `emptyCells` is non-empty:

1. Reinitialize neighbor list grid via `initNbrList(p)`.
2. For each empty cell id, call full `maker.build(cellId, p, m_nbrList, cuboid)`.

This recovers cells that cannot be repaired by neighbor-only recutting.

### 5. Iterative topological repair

Call `repair(p)`.

Repair works as a propagation loop over candidate neighbor insertions.

#### 5.1 Seed insertion proposals

For each changed cell updater:

- `updateNbrInserts()` generates `NbrInsert` triplets from existing facets and facet cycles.

These triplets are proposals indicating which neighboring relationships should be tested next.

#### 5.2 Global gather and group by target cell

Inside each iteration of the while-loop:

1. Collect all updater insertion vectors in parallel.
2. Sort private vectors with `CompareNbrInsert`.
3. Merge into one globally sorted vector `nbrInserts`.
4. If no inserts were collected, loop terminates.

The sorted order groups work by target cell id (`NbrInsert[0]`).

#### 5.3 Apply insertions per target cell

In parallel, each worker repeatedly claims one contiguous group for a target cell and runs:

- `m_updaters[cellId].processNbrInserts(beginPriv, endPriv, maker, p, box)`

Inside `processNbrInserts`:

1. Skip already-known neighbors.
2. For direct insertion tests (`triplet[1] == triplet[2]`), attempt a plane cut using `maker.cutCell(...)`.
3. If insertion fails, propose near-neighbor alternatives via `maker.getCloseNbrs(...)` and push new inserts.
4. For indirect proposals (`triplet[1] != triplet[2]`), stage neighbor candidates and process with `maker.processNbrs(...)`.
5. If topology changed, commit updated cell and emit new insertions via `updateNbrInserts()`.

This creates wave-like propagation of local repairs until no new updates are produced.

### 6. Rebuild packed storage and geometry caches

After repair converges:

1. `m_cellArena.rebuildFromCells(m_cells)`
2. Materialize geometry cells from arena
3. Recompute geometric fields:
   - `computeConnectingVectors`
   - `computeEdgeInv`
   - `diffVolume`

At this point, tessellation topology and geometry are synchronized with moved points.

## Notes on Cut Methods During Repair

There are two cut implementations in `CellMaker`:

- `cutCell2` (default in main neighbor-processing path via `applyCut`)
- `cutCell` (currently used directly in `CellUpdater::processNbrInserts` direct insertion path)

So the moved-point workflow is hybrid today:

- General neighbor processing uses the configured `applyCut` mode.
- Direct repair insertion tests currently call `cutCell` explicitly.

## Practical Implications

- Most time steps are cheap if no cell loses convexity.
- Cost scales with number of changed cells and repair propagation depth.
- Empty-cell fallback prevents unrecoverable local failures.
- Final arena rebuild guarantees contiguous topology storage after incremental edits.

## Suggested Validation Checks

After significant algorithm or parameter changes, validate:

1. No empty cells remain after `update` for representative workloads.
2. Convexity is restored for all cells after repair.
3. Neighbor reciprocity is preserved (facet neighbor relationships are consistent).
4. Topology statistics (face count distributions, degree distributions) remain stable across modes and seeds.
