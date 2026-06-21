# Plan — full de-legacy port (no legacy code; one `vorflow`)

**Goal.** Remove all remaining use of the legacy header-only engine
(`voronoi.hpp`: `CellComplex` / `CellMaker` / `CellGeometry` / `Cell`) and
`simulation.hpp` (`Simulation` / `ExplicitEuler` / `NavierStokes` / `Incompressible`
/ `IntfDyn`). Everything runs on the Kokkos device path — **multicore CPU (OpenMP)
and GPU (CUDA/HIP) from one source** — published through `TessellationView`, with a
single Python module named **`vorflow`** (retiring `vordyn`).

**Method discipline (load-bearing).** Port the *current* numerical methods
**faithfully** first — each ported piece validated bit-for-bit / to machine
precision against the legacy, which is kept only as a test oracle during the
transition and deleted at the end. Methodological improvements (see
`power_cell_solver_spec.md`) come **after**, on the clean device base — never mix a
method change with a backend change.

## Where we are

**Already on Kokkos (CPU+GPU), validated:** device cutter (`ScratchCell`), full
tessellator (grid + CSR + SDF clip), `TessellationView`, reciprocal-facet
transpose, distributed halo (transport-core), and **one** reference physics — the
atomic-free Euler pressure force.

**Still legacy (to remove):**
- physics: viscous (NavierStokes), incompressible projection, interface tension;
- the time integrator + energies (`Simulation::step`, kinetic/internal/interfacial);
- the **CPU incremental update** (`CellComplex::update`) — still on the legacy
  `CellMaker`, not the new cutter;
- per-cell geometry the device view does not yet publish (vertex positions,
  velocity-gradient operator);
- the Python surface (`vordyn`) + the `mpi/validate_*.py` drivers and benchmarks.

## Phases

**1. Publish the remaining geometry.** Extend `TessellationView` (and the
tessellator write-out) with the **vertex CSR** (`cellVertexOffset`, `vertexPos`)
and a per-cell **velocity-gradient** operator (the least-squares facet stencil),
behind the existing "publish only what a consumer requests" field set. *Accept:*
vertex CSR + gradient reproduce legacy `CellGeometry` to machine precision.

**2. Viscous force (NavierStokes).** Port `NavierStokes::computeForces` as a
`TessellationView`-only kernel: per-cell velocity gradient → viscous stress →
atomic-free gather (same form as the Euler reference). *Accept:* matches legacy to
~1e-12; zero atomics; view-only dependency.

**3. Interface tension (IntfDyn).** Port the multiphase surface-tension force; it
couples a facet to the *other* facets of the same cell and lands on neighbours, so
use the **`NbrsToFacets` transpose** already built (plan §2.5). *Accept:* matches
legacy; zero atomics.

**4. Integrator + energies + driver.** A device `Stepper` (velocity-Verlet, the
force/integrate split for the distributed scheme-C path) and the energies as device
reductions; a thin device `Simulation` facade orchestrating
tessellate → publish → force → integrate. *Accept:* a full trajectory matches the
legacy `Simulation::step` to machine precision (the `mpi/validate_voronoi_dynamics`
scenarios, run against the device path).

**5. Incompressible projection — DEFERRED (no faithful port possible).** The legacy
`Incompressible` is an **incomplete stub**: `buildConstraintMatrix()` only assembles
the pressure-Poisson operator `A = D M⁻¹ Dᵀ` and prints timings — it never solves the
system, there is no projection, and no `Incompressible::step()` (it falls through to the
explicit Euler step). So there is nothing functional to faithfully port. A real
device elliptic solver (assemble the Voronoi-graph Laplacian from `TessellationView`
+ CG/AMG via Kokkos-Kernels, reusing `sdflow`'s MG-PCG experience) is **new-method
work** and belongs to the later methodological-improvement effort, not this
faithful-port pass. The reusable foundation — assembling `A` from the view — can be
ported when that solver is written.

**6. CPU incremental update on the new cutter.** A host (OpenMP/Threads) worklist
driver that, each step, uses `SkinRefresh` to find cells whose neighbourhood moved
past the skin and **re-runs the `ScratchCell` cutter** on just those cells (it
already rebuilds a single cell), with the local-repair / full-rebuild fallback —
replacing legacy `CellComplex::update`. GPU keeps full rebuild. *Accept:*
incremental == full rebuild over a moving trajectory; measured incremental-vs-full
(CPU) and CPU-incremental-vs-GPU-full crossovers.

**7. `vorflow` Python module + rename.** A pybind11 module **`vorflow`** over the
device tessellator + ported physics (the `Stepper`/`Simulation` facade), exposing
the same verbs as `vordyn` plus geometry/SDF setters; **rename `vordyn` → `vorflow`**
and port `python/test_vordyn.py`, `mpi/validate_*.py`, and the benchmarks to it.
*Accept:* the Python smoke + distributed validation scripts pass on the device path.

**8. Retire legacy.** Once every consumer is on the device path: convert the test
oracles to **frozen golden data** (already captured in `test_golden`/`test_invariants`),
then delete the legacy `CellComplex`/`CellMaker`/`CellGeometry`/`Cell` from
`voronoi.hpp` and all of `simulation.hpp`. `vor_types.hpp` (label encoding,
constants) stays — the device cutter uses it. *Accept:* nothing includes
`simulation.hpp`; `voronoi.hpp` retains only shared types/utilities; full suite
green on the device path alone.

## GPU-efficiency outlook
Forces (pressure/viscous/interface) and the velocity gradient are per-cell/-facet
gathers — atomic-free, coalesced, GPU-efficient (proven for the Euler force).
Integration/energies are trivially parallel. The **only** non-trivial kernel is the
incompressible elliptic solve (unstructured CG/AMG); it is GPU-efficient and the
suite has the MG-PCG precedent in `sdflow`. So the full physics engine *is*
efficiently GPU-able; the elliptic/implicit linear solves are the real work.

## Sequencing
geometry (1) → explicit physics (2,3) → integrator/driver (4) → implicit/projection
(5) → CPU incremental (6) → Python `vorflow` + rename (7) → delete legacy (8).
Phases 1–4 already remove most legacy from the *explicit* dynamics; 5 is the elliptic
solver; 6 restores fast moving-point updates on CPU; 7–8 finish the de-legacy + rename.

## Open choices (defaults shown)
- **Faithful-port-first**, then improve per `power_cell_solver_spec.md` (consistent
  mass matrix, Lloyd regularization, implicit midpoint, proper near-incompressible
  projection) as a *separate* effort. (Recommended; matches "port, later improve".)
- **Keep legacy as an oracle through Phase 7, delete in Phase 8** (vs. hard cut now) —
  safest; every port is diffed against legacy before it's removed.
- **Incremental stays CPU-only** (the architecture's GPU=rebuild / CPU=update split);
  revisit a GPU incremental only if a regime demands it.
