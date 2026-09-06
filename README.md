# peclet.voro

[![PyPI version](https://img.shields.io/pypi/v/peclet-voro.svg)](https://pypi.org/project/peclet-voro/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/peclet-voro/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/computational-chemical-engineering/peclet-voro/blob/main/LICENSE)
[![CI](https://github.com/computational-chemical-engineering/peclet-voro/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/peclet-voro/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21132443.svg)](https://doi.org/10.5281/zenodo.21132443)

A **Kokkos** engine for dynamic Voronoi tessellation of moving particles in three
dimensions, part of the `peclet` suite. The same sources run on **CUDA / HIP / OpenMP**
(the backend is chosen by the bootstrapped Kokkos prefix the build is pointed at, not
hard-coded), and the distributed path is built on the shared `core` MPI halo.
Features:

- Periodic boundary conditions (cubic and Lees–Edwards shear boxes)
- Incremental cell updates under particle motion (persistent Verlet-skin worklist)
- Compressible and incompressible Euler / Navier–Stokes dynamics
- Multiphase interface-tension (surface-tension) forces
- GPU and multicore execution through Kokkos backends; MPI block decomposition via `core`

The compact **ConvexCell** dual-triangle tessellator is the engine. (The original header-only
half-edge CPU engine has been retired and removed; the device path is the whole library.)

---

## Repository layout

```
voro/
├── include/
│   └── peclet/voro/                 # the Kokkos tessellator engine (namespace peclet::voro)
│       ├── convex_cell.hpp          #   compact dual-triangle ConvexCell + per-vertex geometry
│       ├── tessellator.hpp          #   cold build: grid + worklist gather + clip + CSR publish
│       ├── repair.hpp               #   MovingTessellation: incremental two-pass repair update
│       ├── topology_store.hpp       #   resident compact topology (+ poke4 cert planes) between steps
│       ├── tess_grid.hpp            #   counting-sort grid + presorted worklist
│       ├── subset_gather.hpp        #   the cold-build kernel restricted to an index list (repair)
│       ├── dynamic_validate.hpp     #   geometric invariants + oracle diff (validators)
│       ├── verlet_skin.hpp          #   per-particle Verlet-skin (insertion) tracker
│       ├── sdf.hpp                  #   SDF half-space clipping (solid boundaries)
│       ├── plane_policy.hpp         #   Voronoi / Power / SDF plane-definition policies
│       ├── transpose.hpp            #   neighbour<->facet reciprocal map helpers
│       ├── energy/                  # rung A3: energy terms on the published view (+ area Jacobians)
│       │   ├── route.hpp            #   per-facet gradient -> seed DOFs (Voronoi/Power chain, wall chain)
│       │   ├── interface.hpp        #   Σ σ(t_i,t_j) A_ij  (surface tension between species)
│       │   ├── wall.hpp             #   Σ σ_s(t_i) A_wall   (wetting; Young's angle from σ_sg − σ_sl)
│       │   └── volume.hpp           #   Σ e_i(V_i)         (target / log-barrier / free energy)
│       ├── physics/                 # simulation + forces over the published view
│       │   ├── simulation.hpp       #   Euler / Navier-Stokes facade (ExplicitEuler)
│       │   ├── euler_pressure.hpp   #   EOS pressure force
│       │   ├── viscous.hpp          #   viscous Navier-Stokes term
│       │   └── interface.hpp        #   multiphase interface-tension force
│       ├── mpi/
│       │   └── voronoi_halo.hpp     #   distributed halo glue over core
│       └── tessellation_view.hpp    # published read-only CSR device view (engine<->consumer seam)
├── src/voro_bindings.cpp     # nanobind Python module (`peclet.voro`)
├── python/test_voro.py       # Python smoke test (Tessellation + Simulation)
├── tests/kokkos/                # device unit tests + benchmarks
├── tests/kokkos_mpi/            # distributed benchmarks
├── docs/                        # design notes, performance_report.md, Doxygen config
└── CMakeLists.txt               # build system (Kokkos device path)
```

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| C++ compiler | C++20 | GCC ≥ 11 recommended |
| CMake | ≥ 3.16 | |
| **Kokkos** | bootstrapped | `-DPECLET_VORO_KOKKOS=ON`; CUDA/HIP/OpenMP backend chosen by the prefix |
| **core** | sibling repo | Shared MPI halo + array bridge; required for the distributed path |
| **morton** | sibling repo | Z-order spatial-index primitive used by the device tessellator |
| MPI | any | Distributed path (`-DPECLET_VORO_MPI=ON`) |
| nanobind | ≥ 2.0 | Python module (`-DPECLET_VORO_BUILD_PYTHON=ON`); found via the active interpreter |
| Voro++ | master | Fetched by CMake FetchContent as the throughput reference for `bench_convexcell` |

The Kokkos/ArborX backend and target architecture come from the bootstrapped prefix
`../extern/install/<backend>` (built once by `../tools/bootstrap_deps.sh`), exactly as in
`sdflow` and `dem`. Put `nvcc` on `PATH` for the CUDA backend.

---

## Building with CMake

### Device (production) path

```bash
# Point at the bootstrapped Kokkos prefix; clone core and morton as siblings.
cmake -B build -DPECLET_VORO_KOKKOS=ON \
      -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build --parallel
ctest --test-dir build --output-on-failure        # device tests under tests/kokkos
```

Add `-DPECLET_VORO_MPI=ON` to link MPI + `core` for the distributed path.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `PECLET_VORO_KOKKOS` | `OFF` | Build the Kokkos device path (`find_package(Kokkos)`) |
| `PECLET_VORO_MPI` | `OFF` | Build the distributed path against MPI + core |
| `PECLET_VORO_BUILD_PYTHON` | `OFF` | Build the device-native nanobind module `peclet.voro` (under `PECLET_VORO_KOKKOS`) |
| `PECLET_VORO_BUILD_TESTS` | `ON` | Build the test executables |
| `PECLET_VORO_BUILD_BENCHMARKS` | `OFF` | Build the performance benchmarks |
| `PECLET_VORO_BUILD_DOCS` | `OFF` | Build Doxygen HTML documentation |

---

## Python bindings (`peclet.voro`)

The device tessellator is exposed to Python through a **nanobind** module that uses the
shared `core` **zero-copy** array bridge (numpy `(N,3)` / `(N,)` arrays alias the
device-staged buffers — no per-call copies). This is the same drive-from-Python pattern as
the rest of the suite (`sdflow`/`pnm`, `dem`). The module is **not** pybind11 and is **not**
fetched automatically: nanobind is located via the active interpreter through the suite's
`cmake/SuiteNanobind.cmake`. The built module is importable as `peclet.voro` (formerly
`vordyn`).

```bash
cmake -B build -DPECLET_VORO_KOKKOS=ON -DPECLET_VORO_BUILD_PYTHON=ON \
      -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build --target voro -j
PYTHONPATH=build python3 -c "import peclet.voro; print(peclet.voro.execution_space)"
```

The module exposes two surfaces — the bare **`Tessellation`** (cold build + incremental repair of a
moving point set) and the **`Simulation`** fluid solver:

```python
import numpy as np
import peclet.voro as voro

# bare moving-point Voronoi tessellation
t = voro.Tessellation()
t.set_box([1.0, 1.0, 1.0])
t.build(pos)                         # cold build, pos = (N,3) float64
vol = t.volumes()                    # (N,) cell volumes (sum ~= box volume)
nbr = t.neighbor_counts()            # (N,) Voronoi neighbours per cell
stats = t.step(pos_moved)            # incremental repair to new positions

# compressible-Euler / Navier-Stokes fluid on top of it
s = voro.Simulation()
s.set_box([6.0, 6.0, 6.0])
s.set_positions(pos)                 # (N,3) float64
s.set_velocities(vel)                # (N,3) float64
s.set_masses(masses)                 # (N,) float64
s.set_pressure(1.0)
s.set_viscosities(nu)                # (N,) float64 — enables the viscous Navier–Stokes term
s.init()                             # build the first tessellation + forces
s.step(num_steps=10, dt=1e-3)        # velocity-Verlet dynamics

pos  = s.get_positions()             # (N,3)
vol  = s.get_volumes()               # per-cell Voronoi volume (N,)
ke   = s.get_kinetic_energy()

# SDF solids + power weights on the moving-point path (Voronoi methods plan, rung A0)
scene = peclet.core.geom.SceneBuilder()
root = scene.add_leaf("sphere", [0.25], translation=(0.5, 0.5, 0.5))
node_ints, node_reals, _, _ = scene.encode()
t.set_geometry(node_ints, node_reals, root=root)   # any analytic core scene (CSG, transforms, ...)
t.set_weights(w)                     # (N,) power (Laguerre) weights — optional
t.build(pos)                         # cells clipped by the solid; in-solid seeds get volume 0
stats = t.step(pos_moved)            # wall planes are resident; stats['wall_flagged'] = re-clips
walls = t.wall_counts()              # (N,) wall planes per cell
s.set_geometry(node_ints, node_reals)  # the same walls for the fluid (pressure acts on them)
```

Array shapes follow the suite convention (`../docs/CONVENTIONS.md` §6): positions/velocities
`(N,3)` float64, masses/viscosities/volumes `(N,)`. Call `peclet.voro.finalize()` for
deterministic Kokkos teardown (also run from an `atexit` hook).

For the distributed (MPI) validation scripts see [`mpi/README.md`](mpi/README.md) and
[`docs/distributed_voronoi.md`](docs/distributed_voronoi.md).

---

## Code quality

### Formatting

The codebase follows the Google C++ Style Guide enforced by `clang-format`:

```bash
clang-format --dry-run --Werror include/peclet/voro/**/*.hpp tests/kokkos/*.cpp src/*.cpp
```

### Static analysis

```bash
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
clang-tidy -p build include/peclet/voro/*.hpp
```

### Documentation

```bash
cmake -B build -DPECLET_VORO_BUILD_DOCS=ON
cmake --build build --target docs
# HTML output: build/docs/html/index.html
```

---

## Data structure overview

The production engine stores a Voronoi *cell* as a compact **ConvexCell** in the **dual**
representation (`include/peclet/voro/convex_cell.hpp`). The cell is the intersection of
half-spaces `{x : n_k·x ≤ n_k·n_k}` — one plane per neighbour (`n_k` is the foot-point
normal: `x = n_k` is the foot of the perpendicular from the seed, so the connector to the
neighbour is `2·n_k`), plus the six bounding-box planes. Instead of explicit half-edge
topology, **each primal vertex is the intersection of three planes**, stored as a *triple of
plane indices* (one byte each):

```cpp
unsigned char t0[MAXT], t1[MAXT], t2[MAXT];  // triangle = triple of plane indices
Real vx[MAXT], vy[MAXT], vz[MAXT];           // cached dual-vertex position
```

so the whole cell is a small triangle list (a few hundred bytes) that lives in registers /
a tiny local frame — this is what lifts GPU occupancy far above the old ~32 KB half-edge
frame. Clipping by a new plane (GEOGRAM / Ray-et-al. convex-cell clip) marks the triangles
whose dual vertex falls outside the plane, finds the horizon, and adds one new triangle per
horizon edge; there is no stored adjacency (the cell is tiny, so the triangle sharing an
edge is found by a short scan). Cuts are applied closest-first with a security-radius
early-out.

**Validity diagnostics (rung A2a).** `Tessellation.build(positions, strict=False)` warns (raises
if strict) when the result is not a guaranteed-exact partition — buried power cells (a seed
outside its own cell, which the engine empties; never for w = r² of non-overlapping spheres),
a search reach beyond half the box (min-image invalid), overflowed cells — and
`build_report()` returns the counts (`StatusBit` kBuried / kReachExceeded).

**Certificate completeness (engine hardening, 2026-09-03).** The seed-local certificate of the
repair cannot see a face GAINED from a seed outside the stored topology (its bisector drifts into
the cell without either seed tripping the Verlet skin) — measured ~0.1 % of neighbour relations
missed per step and a 1.6e-3 volume error over 400 steps. Every (re)build now records the
candidates whose plane missed the cell by less than a margin (`MovingTessellation::nearMarginFrac`
× skin, default ½) and the certificate re-tests those planes each step
(`ConvexCell::planeGap`), flagging both cells of a new face. The repair is then exact to
1e-11 (default tolerance) / 1e-15 (tight) over 400 steps at ~80 % of the previous speedup;
`useNearMiss = false` restores the old certificate.

**Curved walls (rung A1).** `clipCellAgainstSdf` places each wall plane to SECOND order: after the
multi-plane tangent clip it re-clips the cell with every plane translated into the solid by the
sagitta of its final face (`½ tr(∇²φ · M)/|∇φ| / A`, M the face's second moments about the tangency
point). Measured on a sphere: the fluid-volume error drops 17–90× (1e-3 → 2e-5 at 12k seeds, seeds
in the fluid); flat walls are bit-identical to the tangent clip; `TangentOnly<Sdf>` restores the
first-order cut. The wall FORCE has two forms: the seed-foot chain (`sdfWallChain`, the tangent-plane model,
first-order on curved walls) and the exact in-kernel finite-difference wall part
(`buildTessellation(..., withWallFD=true)` publishes `cellWallDV`/`cellWallDA` per cell via
`sdfWallFD`; the energy layer's volume and wetting terms use it when present) — exact for whatever
the clip does on the wall plane itself (`test_sdf_policy` (D): sphere 1194/1200 at 1e-4). What
neither has yet is the sagitta placement's dependence on the NEIGHBOUR planes through the face
polygon (its second moments): with the sagitta clip a consistent gradient is still open
(`test_sdf_policy` (D2)); wrap the provider in `TangentOnly<Sdf>` where gradient consistency
matters more than second-order tiling (the mesh optimiser at curved walls).

**Grids and the covolume solver (tracks B and C).** `fv/mesh.hpp` turns a published tessellation
into a face mesh (one record per geometric face, owner/neighbour, centroid, both seed distances,
cell→faces CSR) and `fv/operators.hpp` provides the staggered covolume operators on it —
divergence of face fluxes, the two-point gradient/Laplacian (exact on Voronoi orthogonality; `L`
is the graph Laplacian `A_f/d_f`), Green–Gauss, Perot reconstruction, adjoint inner products and a
matrix-free Poisson CG. Measured: the Poisson solution converges at second order on a jittered
lattice while the cellwise residual of the two-point flux is skewness-limited; the centroidal
(Lloyd) energy `energy/lloyd.hpp` removes that skewness (random grid: skewness 0.23 → 0.04, volume
CV 0.42 → 0.06, residual consistency 0.17 → 0.04 in 40 steps; `tests/kokkos/test_grid_relax`).

**Covolume Navier–Stokes (rung C2a).** `fv/covolume.hpp` is the staggered solver on the face
mesh: face fluxes and cell pressures, face momentum as the exact transpose of the Perot
reconstruction (adjoint to round-off), cell-centred convection with the arithmetic face mean
(skew-symmetric for divergence-free fluxes), the two-point viscous term on the reconstructed
field, SSP-RK3 with a projection per stage, and pressure PCG with core's `GraphAMGDevice` on the
symmetric `−V L` (12 vs 76 CG iterations). Measured (`tests/kokkos/test_covolume_ns`): the inviscid
Taylor–Green energy drift is the RK3 time error only (dt-order 2.96), divergence 6e-14; the
viscous decay is second order on the cubic lattice (1.93/1.98) but first order on unstructured
Voronoi meshes (0.2h-jittered 0.96/0.82, Lloyd CVT 1.43/1.28) — the Perot reconstruction is only
first-order consistent on non-symmetric cells and the viscous term inherits it; the DEC
(Nicolaides) curl-curl viscous term on the Voronoi edges is the planned remedy.

**DEC viscous term (rung C2a′, measured, shelved).** `fv/dec.hpp` builds the Nicolaides
covolume Laplacian grad div − curl curl on the Voronoi–Delaunay pair (the view publishes the
Voronoi edge lengths, `edgeLength`, with the facet-edge CSR). It is symmetric and dissipative to
round-off (`tests/kokkos/test_covolume_dec`), but no accuracy remedy: first-order consistent on
skewed meshes like the Perot term (face-average flux vs connector-midpoint 1-form), inconsistent on
the degenerate cubic lattice, and ~8× stiffer explicitly. The covolume scheme's second order needs
centroidal meshes or the collocated scheme.

**Collocated Navier–Stokes (rung C2b).** `fv/collocated.hpp` is `peclet.flow`'s SolverColocated
structure on the Voronoi face mesh: incremental predictor whose cell pressure gradient is the
exact transpose of the centre→face constraint (flow's gauge-exact gradient), the constraint
interpolation `T`, an exact face projection with the same `L`, the transpose cell correction and
`P += φ`. The unstructured extension makes `T` second order on skewed faces (each cell is
extrapolated to the face centroid with the Green–Gauss gradient times the Gauss-exact factor
`(I − S)⁻¹`, `faceInterpTranspose` is its exact adjoint — linear fields are reproduced at the
centroid to 5e-16 on a random mesh of skewness 0.24). Measured
(`tests/kokkos/test_collocated_ns`): Taylor–Green order 1.97 / 2.11 / 2.08 on the cubic lattice /
a 0.2h-jittered lattice / a Lloyd CVT (plain pair 1.97 / 1.72 / 1.17; covolume flux 1.98 / 0.82
/ 1.29), energy drift O(dt·h²), face divergence 3e-14. The comparison page is
`suite/docs/studies/voro_covolume_vs_collocated.md`.

**Early wall clip.** The cell build clips first with the tangent plane at the seed's own wall
foot, retreated by the seed's wall distance into the solid, so a cell next to a curved wall never
extends through the solid before its neighbours are gathered: wall-adapted seed shells (nearly
cospherical seeds) no longer overflow the plane cap, and `kIncomplete` no longer fires on wall
cells; the cell capacities are template parameters of `buildTessellation<…, MAXP, MAXT>` (64/112
production). The PolyMesh's wall layer is still non-conforming (a topological mismatch of the
shared interface faces; `mergeWallVertices` measured it).

**Semi-implicit step (rung C2c).** `CollocatedNS::implicitDiffusion` is flow's step: explicit
convection, a backward-Euler viscous solve per component (two-point Laplacian and two-point wall
term implicit, the quadratic wall correction lagged; `PressureSolver::setupVelocity`, GraphAMG-PCG),
the approximate projection, and the optional rotational pressure update `P += φ − ν div u*`.
Stokes marches take Δt = 10–20 h²/ν (the sphere-array ladder: 260 steps at n = 32 instead of
3700, same K to four digits; Poiseuille exact to 4e-13 in 154 steps). Works under MPI through the
same hooks. Python: `FlowSolver.set_implicit_diffusion(True)`.

**Body-fitted walls (rung C3).** Both solvers take a prescribed wall velocity
(`setWallVelocity`, empty = no-slip) on the wall faces of the SDF-clipped cells: the constraint
returns `U_wall·n`, the viscous wall flux is the two-point `ν A (U_wall − U_i)/h_A`, the pressure
is Neumann. The viscous wall flux is the wall-anchored least-squares quadratic gradient (`wallGradientLS`,
flow's wall-anchored reconstruction on the unstructured mesh; `wallQuadratic`, default on) or the
two-point `(U_wall − U_i)/h_A`. Measured (`tests/kokkos/test_body_fitted`, Poiseuille between SDF
slabs): the parabola is the exact discrete steady state with the quadratic gradient (wall residual
5e-11, march error 7e-6); the two-point flux leaves a wall-row residual of f/4 and converges at
order 2.00. The face
mesh drops zero-area non-reciprocal facets of degenerate lattices (`nDropped`); the tessellator
needs every periodic box extent above twice its coverage radius.

**Sphere-array permeability (rung C4).** `tests/kokkos/test_permeability` marches Stokes flow
through a simple-cubic sphere array (φ = 0.216) on jittered-lattice seeds clipped by the sphere:
drag −2.4 % / −0.96 % / −0.37 % of Zick & Homsy at 16 / 24 / 32 cells per box edge (second
order; flow's cut-cell IBM −0.49 % at 32) with the quadratic wall gradient — the two-point wall
flux gave −13 % / −7.5 % on the same meshes (fat wall cells). A cubic (unjittered) lattice around
a sphere is degenerate and overflows the clipper — jitter the seeds; a seed shell at h/2 overflows
the 64-plane cell cap.

**Python `FlowSolver` (rung C5, Python half).** `peclet.voro.FlowSolver(tess, viscosity,
layout='collocated'|'covolume')` runs either static solver on the face mesh of a resident
`Tessellation` (walls from its SDF geometry): body force, Stokes switch, wall velocities, initial
velocity, `step`, cell velocity / pressure / volumes, kinetic energy, divergence. **Distributed (MPI).** `fv/distributed.hpp` runs the collocated solver over `VoronoiHalo`'s
decomposition: `buildFaceMesh(view, aux, nOwned)` keeps facets toward ghost seeds as interface
faces owned locally, cell fields are sized owned+ghost and refreshed through the halo's `forward`
at every stage, the pressure PCG exchanges its search direction, all-reduces its dot products and
preconditions with the per-rank block of the GraphAMG. `tests/kokkos_mpi/test_flow_mpi`
(`flow_mpi_np{1,2,4}`): np = 1 bit-exact to single rank on the host backends (on CUDA/HIP the
two runs differ at round-off, ~1e-15, and are not reproducible run-to-run — the tessellator's
facet CSR is assembled with atomics — so the device gate is 1e-13 / 1e-14), np = 2/4 within
3e-15 (velocity) and 2e-16 (energy) of it, divergence 2e-14. The isolated pressure-solve gate (true residual ==
recursive residual, K·1 = 0, symmetry) is what caught the rank-local total volume in the mean
deflation. The covolume solver carries the same hooks (`flow_mpi_covolume_np{1,2,4}`), and the
exchange packs on the device (only the send/receive buffers cross to the host for MPI, bitwise
equal to the host path).

**Pore-mesh redistribution (rung B2).** `peclet.voro.redistribute_pore_mesh(positions, centres,
radii, L, s_lo, s_hi, slope=…)` drives interstitial seeds to the graded target `V_ref = s(φ)³` by
the topological moves a position-only optimiser cannot make — split oversized cells (along the
wall for wall cells), remove undersized and dead ones, relax with a Lloyd blend plus the graded
volume descent, re-seed the wall layers by the graded-shell heuristic, keep the best state.
Measured from a 2× mismatched start: uniform target max |V/V_ref − 1| ≈ 0.1, rms ≈ 0.04, no dead
cells; graded (slope 0.3) rms ≈ 0.08, max ≈ 0.5 in the first wall shell. The search grid now clamps
the window `sw` to the grid (the optimiser's default `sw = 6` on a few hundred seeds segfaulted).

**PolyMesh (rung B3).** `fv/polymesh.hpp` assembles the internal polyhedral mesh of a resident
tessellation — shared vertices (periodic-aware), CCW face polygons, owner/neighbour, wall patches,
cell→faces CSR — and writes VTU polyhedra (`writeVtu`). Watertight and Euler-exact off walls;
along a CURVED wall each cell clips with its own tangent plane, so wall-adjacent faces of two cells
differ slightly (not watertight there — a conforming per-edge wall plane is the follow-up).

**Energies (rung A3 of the Voronoi methods plan).** `buildTessellation(..., withAreaGrad=true)`
additionally publishes a facet-edge CSR of area Jacobians `∂A_f/∂n_l` (a facet's area depends on
its own plane and on its edge-neighbours' planes; `TessellationView::edgeBegin/edgeEnd/edgePartner/
areaGrad`). The `energy/` headers evaluate interfacial, wetting and volume energies and their exact
gradients on that view — one kernel over the cells, no per-cell reconstruction — and route them to
the seed positions (and power weights) through the plane-policy chain; SDF wall planes go through
the one seed-foot wall chain `sdfWallChain`. `interfaceMinimize` runs on this path (gated against
the old reconstruction to round-off, `tests/kokkos/test_energy_layer`); the incremental path
publishes the same CSR (`reevalPublish(..., withAreaGrad=true)`).

Per-cell **geometry** (volume, per-facet area and first moment, volume gradients) is computed
by a **sort-free, adjacency-free per-vertex scatter** (`volumePerVertex` /
`geometryPerVertex` and the `facet*PerVertex` family): each dual vertex scatters signed
determinants into its three incident facets, so no facet polygon is ever assembled or
ordered. Consumers (physics, microstructure analysis) read the results through the published
read-only **facetGeometry CSR** in `tessellation_view.hpp` (`TessellationView`: a Kokkos
View CSR of per-cell / per-facet quantities) rather than touching the cell internals.

See `docs/mainpage.dox` for the architecture overview and `docs/performance_report.md` for the
cross-backend performance/memory/accuracy study.

---

## License

See [LICENSE](https://github.com/computational-chemical-engineering/peclet-voro/blob/main/LICENSE) for details.
</content>
</invoke>
