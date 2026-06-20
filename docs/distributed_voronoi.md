# Distributed Voronoi (block decomposition + ghosts) — design & status

Suite roadmap Phase 5: decompose the periodic domain into blocks across MPI ranks; each rank gathers
**ghost particles** one interaction radius deep so the Voronoi cells touching its block boundary close
correctly; validate the owned cells against the serial tessellation. The Lagrangian halo (migration +
ghost particles) is reused from `transport-core` via its `tpx_mpi` Python shim, exactly as in
`packing-gpu`. The per-cell observables (`vordyn.get_volumes()` / `get_num_neighbors()`) give the
serial-vs-distributed comparison.

**Status: implemented and validated.**
- *Tessellation* (`mpi/validate_voronoi.py`): owned-cell **volumes and neighbour counts** match the
  serial full-box tessellation to **machine precision** (max ~1e-15, 0 neighbour mismatches) at
  np=1/2/4; owned volumes sum exactly to the box (perfect partition).
- *Dynamics* (`mpi/validate_vorflow.py`): 6 steps of compressible-Euler dynamics distributed
  vs serial match to **machine precision** (~4e-15) at np=1/2/4.

### Halo depth: 1 ring for the tessellation, 2 rings for the dynamics
The owned cells *close* with a **1-ring** halo (`rcut ≳ max owned-cell circumradius`). But the **force**
on an owned cell uses its neighbours' cell *pressures* (EOS of their volumes), so each neighbour cell
must itself be correct → *its* neighbours (the 2nd ring) must be present. So the **dynamics needs a
2-ring halo** (≈ 2× the tessellation `rcut`): `rcut=2.0` gives exact owned dynamics here, `rcut=1.2`
(~1.5 rings) leaves a ~5e-6 force error that grows with steps.

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

## Two communication schemes (`mpi/validate_voronoi_scheme_c.py`)

The dynamics above re-gathers full ghost **state** every step. A leaner alternative communicates only
**forces** over a persistent halo (the "scheme C" / conservative-flux exchange):

- **re-gather (replicate):** each step migrate + `gather_ghosts` (pos+vel) + rebuild the local
  tessellation from scratch; integrate owned; discard ghosts.
- **scheme C (force forward):** build a **persistent** owner↔ghost halo once per `rebuild_every`
  steps, hold one local Simulation over owned+ghost, and each step velocity-Verlet owned+ghost while
  **forwarding the owner forces onto their ghost copies** (`tpx_mpi.Halo.forward`) and integrating the
  ghosts locally with them — so ghosts track their owners with no per-step state re-gather, and the
  tessellation is updated incrementally instead of rebuilt. (For Voronoi the owned-cell force is
  already complete from the local closure, so the *reverse* contributes zero; only the *forward* is
  needed. The `vordyn` force/integrate split — `recompute_forces`/`get_forces`/`set_forces` — lets the
  Python driver insert the exchange between force computation and the Verlet kick.)

Both match the serial trajectory to **machine precision**. Profiling (np=2, N=512, 40 steps; single
shared dev box):

| scheme | MPI time / step | total / step |
|---|---|---|
| re-gather | 4.69 ms | 29.5 ms |
| scheme C, `rebuild_every=10` | **1.27 ms** | 10.3 ms |
| scheme C, `rebuild_every=20` | **0.71 ms** | 8.8 ms |

So **scheme C reduces the MPI overhead ~4–7×** (and total time ~3×). The win is **amortization**: at
`rebuild_every=1` scheme C is *slower* than re-gather (7.0 ms — its per-step rebuild does
migrate+build+forward, heavier than one gather), so the gain comes entirely from reusing the topology
across steps; the per-step exchange itself (one force vector) is cheap. Accuracy stays at machine
precision even at `rebuild_every=40` here because the skin (0.5) ≫ the motion over the interval
(~0.02); for faster dynamics `rebuild_every` must shrink so no Voronoi neighbour crosses the skin
between rebuilds (the usual neighbour-list-rebuild trade-off).

## Notes / future work

- For a **non-periodic** domain (walls), step 3 would need genuine open/bounded tessellation; that is
  the only case requiring the smooth-SDF boundary or a polytope (multi-plane) clip via
  `Cell::clipByPlane`. The periodic case (above) needs none of it.
- Lees–Edwards (sheared) boxes: the migrator/halo would need the LE image shift (deferred; see the
  suite `docs/ROADMAP.md` Phase 1 note).
- Perf: each rank currently rebuilds a `vordyn.Simulation` per validation; a persistent tessellation
  with incremental `update()` is the natural next step for a running distributed simulation.
