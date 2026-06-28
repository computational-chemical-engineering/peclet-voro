# vorflow

[![CI](https://github.com/computational-chemical-engineering/vorflow/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/vorflow/actions/workflows/ci.yml)

A **Kokkos** engine for dynamic Voronoi tessellation of moving particles in three
dimensions, part of the `peclet` suite. The same sources run on **CUDA / HIP / OpenMP**
(the backend is chosen by the bootstrapped Kokkos prefix the build is pointed at, not
hard-coded), and the distributed path is built on the shared `transport-core` MPI halo.
Features:

- Periodic boundary conditions (cubic and Lees–Edwards shear boxes)
- Incremental cell updates under particle motion (persistent Verlet-skin worklist)
- Compressible and incompressible Euler / Navier–Stokes dynamics
- Multiphase interface-tension (surface-tension) forces
- GPU and multicore execution through Kokkos backends; MPI block decomposition via `transport-core`

The compact **ConvexCell** dual-triangle tessellator is the production engine. The old
header-only half-edge engine (`include/vorflow/voronoi.hpp`) survives **only as a CPU
oracle** used to cross-check the device path in tests; it is no longer the production code.

---

## Repository layout

```
vorflow/
├── include/
│   └── vorflow/
│       ├── device/                  # Production device (Kokkos) tessellator
│       │   ├── convex_cell.hpp      #   compact dual-triangle ConvexCell + per-vertex geometry
│       │   ├── convex_cell_adj.hpp  #   adjacency-aware ConvexCell variant
│       │   ├── convex_cell_compact.hpp # packed compact-storage variant
│       │   ├── tessellator.hpp      #   cell construction / clipping driver (worklist gather)
│       │   ├── topology_store.hpp   #   resident, compact connectivity between steps
│       │   ├── sdf.hpp              #   SDF half-space clipping (solid boundaries)
│       │   ├── plane_policy.hpp     #   Voronoi / Power / SDF plane-definition policies
│       │   └── transpose.hpp        #   neighbour<->facet reciprocal map helpers
│       ├── physics/                 # Drive-from-Python device simulation + forces
│       │   ├── device_simulation.hpp #  device-native Euler / Navier–Stokes facade (Sim)
│       │   ├── euler_pressure.hpp   #   EOS pressure force
│       │   ├── viscous.hpp         #   viscous Navier–Stokes term
│       │   └── interface.hpp       #   multiphase interface-tension force
│       ├── mpi/
│       │   └── voronoi_halo.hpp    #   distributed halo glue over transport-core
│       ├── tessellation_view.hpp   # published read-only CSR device view (engine<->consumer seam)
│       ├── tessellation_build.hpp  # legacy CellComplex -> HostTessellation converter
│       ├── skin_refresh.hpp        # Verlet-skin / worklist refresh
│       ├── nbrlist.hpp            # neighbour-list / cell-linked grid
│       ├── vor_types.hpp          # integer type aliases and small utilities
│       ├── voronoi.hpp           # LEGACY half-edge engine — CPU oracle only
│       └── simulation.hpp       # LEGACY oracle-only simulation driver
├── src/vorflow_bindings.cpp     # nanobind device Python module (`vorflow`)
├── mpi/                          # distributed validation scripts (validate_voronoi*.py)
├── tests/                        # legacy tests + tests/kokkos device tests
├── data/                         # sample particle-position data files
├── docs/                         # design notes, benchmark reports, Doxygen config
└── CMakeLists.txt                # build system (legacy header path + Kokkos device path)
```

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| C++ compiler | C++20 (device) / C++17 (legacy) | GCC ≥ 11 recommended |
| CMake | ≥ 3.16 | |
| **Kokkos** | bootstrapped | Device path (`-DVORFLOW_KOKKOS=ON`); CUDA/HIP/OpenMP backend chosen by the prefix |
| **transport-core** | sibling repo | Shared MPI halo + array bridge; required for the distributed path |
| **morton** | sibling repo | Z-order spatial-index primitive used by the device tessellator |
| ArborX | optional | Strongly-polydisperse / Power neighbour search (needed from Phase 3 on) |
| MPI | any | Distributed path (`-DVORFLOW_MPI=ON`) |
| Boost | ≥ 1.65 | Header-only `random`/`container`, used by the legacy header path |
| Voro++ | master | Fetched by CMake FetchContent for the legacy comparison test only |
| OpenMP | any | Optional parallelism for the legacy oracle path |

The Kokkos/ArborX backend and target architecture come from the bootstrapped prefix
`../extern/install/<backend>` (built once by `../tools/bootstrap_deps.sh`), exactly as in
`sdflow` and `dem`. Put `nvcc` on `PATH` for the CUDA backend.

---

## Building with CMake

### Device (production) path

```bash
# Point at the bootstrapped Kokkos prefix; clone transport-core and morton as siblings.
cmake -B build -DVORFLOW_KOKKOS=ON \
      -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build --parallel
ctest --test-dir build --output-on-failure        # device tests under tests/kokkos
```

Add `-DVORFLOW_MPI=ON` to link MPI + `transport-core` for the distributed path.

### Legacy header-only / CPU-oracle path

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release          # Voro++ is fetched automatically here
cmake --build build --parallel
ctest --test-dir build -R "test_static_voronoi|test_voro_comparison" --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `VORFLOW_KOKKOS` | `OFF` | Build the Kokkos device path (`find_package(Kokkos)`) |
| `VORFLOW_MPI` | `OFF` | Build the distributed path against MPI + transport-core |
| `VORFLOW_BUILD_PYTHON` | `OFF` | Build the device-native nanobind module `vorflow` (under `VORFLOW_KOKKOS`) |
| `VORONOI_BUILD_TESTS` | `ON` | Build the test executables |
| `VORONOI_BUILD_BENCHMARKS` | `OFF` | Build the performance benchmarks |
| `VORONOI_BUILD_DOCS` | `OFF` | Build Doxygen HTML documentation |

---

## Python bindings (`vorflow`)

The device tessellator is exposed to Python through a **nanobind** module that uses the
shared `transport-core` **zero-copy** array bridge (numpy `(N,3)` / `(N,)` arrays alias the
device-staged buffers — no per-call copies). This is the same drive-from-Python pattern as
the rest of the suite (`sdflow`/`pnm`, `dem`). The module is **not** pybind11 and is **not**
fetched automatically: nanobind is located via the active interpreter through the suite's
`cmake/SuiteNanobind.cmake`. The built module is importable as `vorflow` (it was formerly
`vordyn`).

```bash
cmake -B build -DVORFLOW_KOKKOS=ON -DVORFLOW_BUILD_PYTHON=ON \
      -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build --target vorflow_device -j
PYTHONPATH=build python3 -c "import vorflow; print(vorflow.execution_space)"
```

```python
import numpy as np, vorflow

s = vorflow.Simulation()
s.set_l([6.0, 6.0, 6.0])
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
`(N,3)` float64, masses/viscosities/volumes `(N,)`. Call `vorflow.finalize()` for
deterministic Kokkos teardown (also run from an `atexit` hook).

For the distributed (MPI) validation scripts see [`mpi/README.md`](mpi/README.md) and
[`docs/distributed_voronoi.md`](docs/distributed_voronoi.md).

---

## Using the legacy header-only library

The legacy half-edge engine (CPU oracle) is header-only — add `include/` to your include path:

```cmake
find_package(vorflow REQUIRED)
target_link_libraries(my_app PRIVATE vorflow::vorflow)
```

```bash
g++ -std=c++17 -Ipath/to/vorflow/include my_app.cpp -o my_app
```

> The oracle-only C++ API (`vor::ExplicitEuler<double>` in `simulation.hpp`) is retained for
> cross-checking the device path; new work should drive the device engine from Python (above).

---

## Code quality

### Formatting

The codebase follows the Google C++ Style Guide enforced by `clang-format`:

```bash
clang-format --dry-run --Werror include/vorflow/*.hpp tests/*.cpp
```

### Static analysis

```bash
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
clang-tidy -p build include/vorflow/*.hpp
```

### Documentation

```bash
cmake -B build -DVORONOI_BUILD_DOCS=ON
cmake --build build --target docs
# HTML output: build/docs/html/index.html
```

---

## Data structure overview

The production engine stores a Voronoi *cell* as a compact **ConvexCell** in the **dual**
representation (`include/vorflow/device/convex_cell.hpp`). The cell is the intersection of
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

The legacy half-edge representation (`makeLabel(facet,vertex,edge)`) used by the CPU oracle
is documented in `docs/mainpage.dox`.

---

## License

See [LICENSE](LICENSE) for details.
</content>
</invoke>
