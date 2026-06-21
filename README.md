# vorflow

[![CI](https://github.com/computational-chemical-engineering/vorflow/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/vorflow/actions/workflows/ci.yml)

A header-only C++17 library for dynamic Voronoi tessellation of moving
particles in three dimensions with support for:

- Periodic boundary conditions (cubic and Lees-Edwards shear boxes)
- Incremental cell updates under particle motion
- Compressible and incompressible Euler / Navier-Stokes dynamics
- Multiphase interface-tension (surface-tension) forces
- OpenMP parallelisation for cell construction and force computation

---

## Repository layout

```
vorflow/
├── include/
│   └── vorflow/       # Public headers (header-only library)
│       ├── vor_types.hpp       # Integer type aliases and utility classes
│       ├── nbrlist.hpp         # Neighbour-list / cell-linked-list grid
│       ├── voronoi.hpp         # Voronoi cell construction and geometry
│       └── simulation.hpp      # Simulation classes (Euler, NS, multiphase)
├── tests/                      # Test programs
├── data/                       # Sample particle-position data files
├── docs/                       # Doxygen configuration and main page
├── cmake/                      # CMake helper files
├── .clang-format               # Google-style formatting rules
├── .clang-tidy                 # Static-analysis configuration
└── CMakeLists.txt              # Modern CMake build system
```

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| C++ compiler | C++17 | GCC ≥ 9, Clang ≥ 9 |
| CMake | ≥ 3.16 | |
| Boost | ≥ 1.65 | Header-only components (`random`, `container`) |
| OpenMP | any | Optional – enables parallel cell construction |
| Voro++ | master | **Fetched automatically** by CMake FetchContent; no manual install needed |

---

## Building with CMake

```bash
# Configure (Voro++ is fetched automatically at this step)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the test programs (including the Voro++ comparison test)
cmake --build build --parallel

# Run the fast test suite
ctest --test-dir build -R "test_static_voronoi|test_voro_comparison" --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `VORONOI_BUILD_TESTS` | `ON` | Build the test executables |
| `VORONOI_BUILD_PYTHON` | `OFF` | Build the `vorflow` pybind11 Python module |
| `VORONOI_BUILD_DOCS` | `OFF` | Build Doxygen HTML documentation |

---

## Python bindings (`vorflow`)

A pybind11 module gives a Python surface for the moving-particle Voronoi solvers — the same
drive-from-Python pattern as the rest of the suite (cfd-gpu `pnm_backend`, packing-gpu `demgpu`).
pybind11 is fetched automatically.

```bash
cmake -B build -DVORONOI_BUILD_PYTHON=ON && cmake --build build --target vorflow -j
PYTHONPATH=build/python python3 python/test_vorflow.py     # smoke test (needs numpy)
```

```python
import numpy as np, vorflow
s = vorflow.NavierStokes()          # or ExplicitEuler / Incompressible / IntfDyn
s.set_l([6.0, 6.0, 6.0]); s.set_mass_density(1.0)
s.set_positions(pos)               # (N,3) float64; set_velocities/set_masses likewise
s.set_viscosity(0.05); s.init()
s.step(num_steps=10, dt=1e-3)
pos = s.get_positions(); ke = s.get_kinetic_energy()
```

Particle arrays are numpy: positions/velocities `(N,3)` float64, masses/types `(N,)`. Verb names
(`set_positions`/`get_positions`/`step`/…) match the suite convention (`../docs/CONVENTIONS.md` §6).

---

## Using the library in your own project

Because the library is header-only you only need to add the `include/` directory
to your compiler's include path:

```cmake
# CMakeLists.txt
find_package(vorflow REQUIRED)
target_link_libraries(my_app PRIVATE vorflow::vorflow)
```

Or without CMake:

```bash
g++ -std=c++17 -Ipath/to/vorflow/include my_app.cpp -o my_app
```

---

## Quick example

```cpp
#include <vorflow/simulation.hpp>
#include <vector>

int main() {
  using vor::Array;

  vor::ExplicitEuler<double> sim;

  Array<double, 3> L;
  L[0] = L[1] = L[2] = 1.0;
  sim.setL(L);
  sim.setPressure(1.0);
  sim.setMassDensity(1.0);

  // Load or generate positions ...
  std::vector<Array<double, 3>> pos(1000);
  std::vector<Array<double, 3>> vel(1000, {});

  sim.init(pos, vel);
  for (int step = 0; step < 100; ++step)
    sim.step(0.01);

  return 0;
}
```

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

A Voronoi *cell* is stored as a half-edge mesh.  Each vertex has exactly three
outgoing edges, ordered counter-clockwise when viewed from outside the cell.
An edge is encoded as a single integer label:

```
label = makeLabel(facet_id, vertex_id, edge_id)
```

The topology is stored as `m_vertices[v][e] = makeLabel(f_in, v_in, e_in)`,
meaning "the edge `e` from vertex `v` leads to the incoming edge `e_in` of
vertex `v_in` on facet `f_in`."

Facet traversal starting from `m_facets[i]`:

```cpp
uint1 label = cell.m_facets[i];
do {
  uint1 v = vor::getVertex(label);
  uint1 e = vor::getEdge(label);
  label = cell.m_vertices[v][(e + 1) % 3];
} while (label != cell.m_facets[i]);
```

See `docs/mainpage.dox` and the Doxygen HTML output for full details.

---

## License

See [LICENSE](LICENSE) for details.
