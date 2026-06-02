# Distributed Voronoi (block decomposition + ghosts) — design & status

Suite roadmap Phase 5: decompose the periodic domain into blocks across MPI ranks; each rank gathers
**ghost particles** one interaction radius deep so the Voronoi cells touching its block boundary close
correctly; validate the owned cells against the serial tessellation. The Lagrangian halo (migration +
ghost particles) is reused from `transport-core` via its `tpx_mpi` Python shim, exactly as in
`packing-gpu`. The per-cell observables (`vordyn.get_volumes()` / `get_num_neighbors()`) give the
serial-vs-distributed comparison.

**Status: implemented and validated** — `mpi/validate_voronoi.py`. Owned-cell volumes match the serial
full-box tessellation to **machine precision** (max ~1e-15) at np=1/2/4, and owned volumes sum exactly
to the box (perfect partition).

## The algorithm (per rank)

1. **Migrate** particles to their owning block (`tpx_mpi.Migrator.migrate`, periodic-aware).
2. **Gather ghosts** within `rcut` of the block (`gather_ghosts`); `rcut` must exceed the largest
   owned-cell circumradius (a perturbed lattice of spacing 1 needs `rcut ≈ 1`).
3. **Tessellate owned + ghost in the full periodic `[0,L]` box** (`vordyn`), `put_in_box` first.
4. **Keep the owned cells** (the first `n_owned`); discard ghost cells.
5. **Validate**: owned-cell volumes/neighbours equal the serial tessellation.

## Why this is exact (and why no special imaging / non-periodic mode is needed)

The library is periodic-native (the box uses minimal-image distances). That is exactly what makes the
distributed case trivial **for a periodic domain**: each rank tessellates its owned+ghost subset in
the *same* full `[0,L]` periodic box as the serial run.

- An owned cell's Voronoi neighbours all lie within `rcut`, so they are all present (owned or ghost) →
  the owned cell is identical to the serial cell.
- The far particles a rank omits are never neighbours of its owned cells, so omitting them changes
  nothing for the owned cells. The omitted region stays **bounded** between gathered ghosts, so no
  unbounded cells appear and `build()` does not hang.
- `gather_ghosts` returns periodic *images* placed near the block; `put_in_box` (and the box's
  minimal-image) wraps them back to canonical positions, so they tessellate identically to their
  originals. No shifted-image bookkeeping or non-periodic box is required.

## Approaches that do NOT work (avoid)

These were tried before the periodic-subset insight above; keep to step 3 instead:

- **Large pseudo-open box** (cluster in a box ≫ its extent): edge cells become unbounded → `build()`
  hangs; the cell grid is also sized for the huge empty box.
- **Axis-aligned box SDF boundary** to clip a sub-block: the boundary clipping is built for *smooth*
  SDFs (slab/cylinder/sphere); a box SDF has creases at edges/corners and hangs in the clip, even with
  all particles inside and via the `CellComplex` path.

## Notes / future work

- For a **non-periodic** domain (walls), step 3 would need genuine open/bounded tessellation; that is
  the only case requiring the smooth-SDF boundary or a polytope (multi-plane) clip via
  `Cell::clipByPlane`. The periodic case (above) needs none of it.
- Lees–Edwards (sheared) boxes: the migrator/halo would need the LE image shift (deferred; see the
  suite `docs/ROADMAP.md` Phase 1 note).
- Perf: each rank currently rebuilds a `vordyn.Simulation` per validation; a persistent tessellation
  with incremental `update()` is the natural next step for a running distributed simulation.
