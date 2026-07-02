# peclet.voro

[![PyPI version](https://img.shields.io/pypi/v/peclet-voro.svg)](https://pypi.org/project/peclet-voro/)
[![Python versions](https://img.shields.io/pypi/pyversions/peclet-voro.svg)](https://pypi.org/project/peclet-voro/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/computational-chemical-engineering/peclet-voro/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/peclet-voro/actions/workflows/ci.yml)

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

See [LICENSE](LICENSE) for details.
</content>
</invoke>
