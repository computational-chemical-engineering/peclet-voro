# CLAUDE.md — peclet.voro

Dynamic 3-D Voronoi tessellation of moving particles on Kokkos (CUDA/HIP/OpenMP), header-only C++20
under `include/peclet/voro/`, driven from Python as `peclet.voro` (`src/voro_*.cpp`, nanobind on
core's zero-copy bridge — ONE translation unit per subsystem, see "Why the module is seven
files" below). Names follow `../docs/NAMING.md`: `set_domain(extent=)` (origin (0,0,0) and
all-periodic are CHECKED), `set_dt` + `dt` + `step(n)`, scalars/counts bare, copied-out arrays `get_*`.
Two API tiers (QUALITY_PLAN D2, package F done 2026-09-10): the public surface is `Tessellation`,
`FlowSolver`, `Simulation`, `optimize_volume_mesh`, `minimize_interface`, the lazily imported
`peclet.voro.pore_mesh` / `peclet.voro.scenes` submodules (`packaging/voro_pore_mesh.py`,
`voro_scenes.py` — staged + installed by the `PECLET_VORO_PY_FILES` loop in CMakeLists.txt) and,
under MPI, `VoronoiHalo` + `DistributedTessellation`; every instrument/ablation is
`obj.diagnostics.<name>`. No environment variable reaches the library: the cold build's stderr
profile (timing, worklist, over-buffer rebuilds, max facets/cell) is `diagnostics.set_profile(True)`
on `Tessellation` / `Simulation` / `DistributedTessellation` (C++: `MovingTessellation::profile`,
`buildTessellation(..., profile)`; the old `PECLET_VORO_PROFILE` is gone), and the silent
over-buffer rebuild of a cold build is counted in `diagnostics.build_report()['over_buffer_rebuilds']`. Modes are validated strings, results are typed objects, triples are
3-sequences. `python/state_hash.py` prints the SHA-256 of every public path's final state — run it
before and after any API change (at `OMP_NUM_THREADS=1`); the hashes must not move.

## Build + test (the tests exist ONLY under `PECLET_VORO_KOKKOS=ON`)

```bash
source ../.venv/bin/activate                         # the one suite venv (nanobind, numpy)
cmake -B build_dev -DPECLET_VORO_KOKKOS=ON -DPECLET_VORO_BUILD_PYTHON=ON \
      -DPECLET_VORO_MPI=ON -DPECLET_VORO_BUILD_TESTS=ON \
      -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"    # or nvidia-cuda (nvcc on PATH)
cmake --build build_dev -j8                          # -> build_dev/peclet/voro/_voro.*.so
ctest --test-dir build_dev -N                        # 42 = 24 single-rank + 18 MPI (label `mpi`)
OMP_NUM_THREADS=4 OMP_PROC_BIND=false ctest --test-dir build_dev -LE bench --output-on-failure
PYTHONPATH=build_dev python python/test_voro.py      # the Python smoke test on its own
```

ONE tree per backend carries everything: `PECLET_VORO_BUILD_TESTS` (default OFF, so `pip install .`
builds only the module) registers the single-rank suite, and with `PECLET_VORO_MPI=ON` the MPI suite
(`tests/kokkos_mpi`, np = 1, 2, 4, `OMP_NUM_THREADS=1` set per test) joins the same tree.
A plain `cmake -B build` (no Kokkos) configures zero tests — "No tests were found!!!" is not a
failure signal. `core` and `morton` are found as sibling checkouts (`../core`, `../morton`) through
`cmake/PecletDeps.cmake` (`-DPECLET_VENDOR_SIBLINGS=ON` fetches them at the pinned tags instead — CI's
configure-only check). The version is read from `pyproject.toml` (do not edit `project(VERSION)`).

## Why the module is seven files

`src/` holds ONE translation unit per subsystem, not one for the whole module. **Read the "for the
dev loop, not for the wheel" part below before reasoning about this** — the split was originally
justified by the CUDA wheel taking 61-100 min against peclet-flow's ~17 for MORE device kernels
(242 vs 146), on the theory that flow has three TUs and voro had one. That theory was WRONG, and
measuring it is what showed so: the wheel's cost is the architecture count, not the file layout, and
`nvcc --threads` fixes it for one line. flow's three TUs removed *duplicate instantiation*; voro had
none to remove, so splitting could only add fixed cost to buy a parallelism `--threads` supplies more
cheaply. What survived that measurement is the dev-loop argument, which is a good one on its own.

The seams are the classes themselves:

| TU | owns |
|---|---|
| `voro_bindings.cpp` | `NB_MODULE` + the typed results. No engine. |
| `voro_tessellation.cpp` | `Tessellation` — the four `MovingTessellation<..,W,S>` variants |
| `voro_flow.cpp` | `FlowSolver` — collocated + covolume |
| `voro_simulation.cpp` | `Simulation` — the two `ExplicitEuler<S>` variants |
| `voro_optimizers.cpp` | `optimize_volume_mesh`, `minimize_interface` |
| `voro_pore.cpp` | the SDF-walled pore-space family + its host oracle |
| `voro_mpi.cpp` | `VoronoiHalo`, `DistributedTessellation` (in the source list only under MPI) |

`voro_bindings_common.hpp` carries the host-side glue (array conversion, the domain/mode contract
checks, the SDF scene holder, the typed results); `voro_bindings_optim.hpp` the two helpers the
optimiser and pore TUs share; `voro_bindings_tess.hpp` the `Tess` class, because `Flow` and the
distributed drivers name it.

**The one rule that keeps the split from being pointless**: `Tess::face_mesh()` is DECLARED in
`voro_bindings_tess.hpp` and DEFINED in `voro_tessellation.cpp`. It is the only `Tess` method a
second TU calls, and its body instantiates `reevalPublish` / `buildAuxMaps` / `buildFaceMesh` for
all four variants — inline, it would re-instantiate them in `voro_flow.cpp` and `voro_mpi.cpp` and
trade serial compilation for redundant compilation. This is suite QUALITY_PLAN G.8, the same
discipline as flow's `src/flow_solver_staggered.cpp` + `extern template`. **If you add a `Tess`
member that another TU needs, declare it in the header and define it in `voro_tessellation.cpp`.**

Adding a subsystem means adding a TU and a `bind*(nb::module_&)` entry in
`voro_bindings_common.hpp`, called from `NB_MODULE` — not growing an existing file.

**The split is for the DEV LOOP, not for the wheel**, and the two want opposite things. Measured,
one architecture, host backend, `-j8`: editing anything in the old single TU cost 41 s; in the split
tree, `voro_optimizers.cpp` costs 11 s, `voro_flow.cpp` 12 s, `voro_tessellation.cpp` 24 s. But a
FROM-SCRATCH build pays each TU's re-parse of Kokkos + nanobind + the Python headers once per
compilation pass, which is **2.03x the single TU's total CPU work** (922 s -> 1869 s at `-j1`,
2 arches), and four cores cannot divide that back. At the wheel's five architectures on a 4-vCPU
runner:

| | wall |
|---|---|
| single TU, as the wheel built it | 2268 s |
| **single TU + `nvcc --threads 4`** | **735 s** |
| split `-j4` + `--threads 2` (best split config) | 1227 s |
| split `-j4` | 1941 s |

So `pyproject.toml` sets `CMAKE_UNITY_BUILD=ON`: every from-source install re-concatenates the seven
TUs into one and gets the single-TU compilation, while a dev `cmake -B build_dev` keeps the split.
`python/state_hash.py` reproduces all 18 hashes under the unity build — **if you add a TU, check it
still unity-builds** (a name collision between two TUs only shows up there).

The real lever for the wheel was never the file layout: nvcc compiles the architectures serially, so
`--threads` in the wheel step (release.yml) is worth 3.09x on its own — **at four threads on a box
with memory to spare.** It is a MEMORY decision, not a core-count one: one nvcc compiler peaks at
**3.7 GB** on this TU and `--threads N` runs N at once, so four threads is ~15 GB against a hosted
runner's 16 GB. release.yml therefore derives the count from the runner's own `MemAvailable`
(2 on a standard runner, more on a larger one) and walks it down on failure rather than failing a
release build. Note the flag is set in the WHEEL STEP, not the job env: the Kokkos build in the same
job runs `cmake --build -j` over many TUs, where `-j` and `--threads` would multiply. The per-TU critical path,
if you ever do need to split further, is `voro_flow.cpp` (660 s at 2 arches) — the collocated +
covolume solvers, NOT the tessellator (647 s), which is where one would expect it.

## Settled decisions — do not reverse silently

Chosen *against* the obvious or textbook alternative, on measured evidence. Full entries with
verbatim quotes and provenance in [`../docs/decisions/voro.md`](../docs/decisions/voro.md); the index is
[`../docs/DECISIONS.md`](../docs/DECISIONS.md). Reversing one takes a new recorded decision, not a
judgement call in the moment.

- **The GPU topology is the dual-triangle ConvexCell**, not a half-edge cell representation.
- **Robustness is topology-oriented (Sugihara), valid-by-construction — NOT exact predicates.**
- **The collocated pressure coupling is the ABC approximate projection, NOT Rhie–Chow** (the same
  decision flow and amr hold; it has been re-proposed by mistake in all three).
- **The optimizer move direction is steepest descent (plain −g), not Newton** — the GN Hessian is
  rank-deficient here.
- **Physics uses the sqrt-free area-vector formula**, not `facetAreasPerVertex`'s magnitude.
- **The equivalence contract is a 1e-9 volume tolerance plus an exact neighbour set** — explicitly
  NOT bit-exactness.
- **Morton (Z-order) indexing is GPU-only**, not for the CPU backend.
- **Closed dead ends, do not re-attempt:** the cooperative/warp-parallel half-edge cut;
  `-ffast-math` reciprocal speedups (a ceiling, not shippable); order-free volume walk without
  stored adjacency (atan2 is not the bottleneck). Cold-build gather at ~70 distance tests/cell is
  **near-optimal — stop chasing it.**

## Header map (one-way layering, enforced by `tools/check_include_graph.sh` = ctest `test_include_graph`)

| layer | headers | what |
|---|---|---|
| 0 | `params`, `convex_cell`, `plane_policy`, `tessellation_view`, `topology_store`, `transpose`, `verlet_skin`, `tess_grid` | the named defaults (capacities, window, tolerance/skin — every template default names one), the dual-triangle cell, plane policies, the published CSR view, leaves |
| 1 | `sdf` | SDF half-space clipping (tangent + sagitta), wall store |
| 2 | `tessellator`, `subset_gather`, `pore_cells` | cold build: grid + worklist gather + clip + publish; the pore-space polyhedra / section export on the same gather |
| 3 | `repair`, `reeval_tessellation`, `dynamic_validate` | `MovingTessellation` two-pass repair, geometry re-publish, validators |
| 4 | `physics/`, `energy/`, `fv/`, `mpi/`, `mesh_optimizer`, `ot_optimizer` | moving-cell Euler/NS, energies + gradients, face-mesh FV solvers (covolume, collocated, distributed), halo, optimisers |

A header may include its own layer or a lower one, never a higher one. Adding a header means
assigning it a layer in the script, or the test fails.

## Tests vs benchmarks

`tests/kokkos/test_*.cpp` are the 21 device ctests (`add_voro_kokkos_test`); plus
`bench_dynamic_update_gates`, `test_include_graph` (script) and `test_voro_python`
(`python/test_voro.py`, needs `PECLET_VORO_BUILD_PYTHON=ON`) = 24.
`bench_dynamic_update --gates` (ctest `bench_dynamic_update_gates`) is a real gate — its FP64 binary is
always built with the tests. Every other `bench_*` (incl. the `_f32` precision variants and Voro++, the
`bench_convexcell` throughput reference, pinned by commit) is opt-in via `PECLET_VORO_BUILD_BENCHMARKS=ON`
and registered with the ctest label `bench` (`ctest -L bench` runs them at test sizes; CI runs
`-LE bench`). `bench_mesh_optimizer` needs a `packing.txt` and is built but not registered.

## MPI tests (`tests/kokkos_mpi/`)

Part of the main tree (above). `bench_voronoi_mpi` / `bench_repair_mpi` (+ `--sdf`) self-check owned-cell
exactness vs single rank; `test_flow_mpi` gates the distributed collocated / covolume / implicit
solvers (np=1 bit-exact on host). The directory is also a standalone project for a lean MPI-only
tree: `cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH=… [-DPECLET_VORO_MPIEXEC=…]`;
`-DMPIEXEC_PREFLAGS=--oversubscribe` for np=4 on a small box (what CI does).

## Style

Google style via `.clang-format` (+ `.clang-tidy`), checked BLOCKING in CI (`quality.yml`, clang-format
18.1.8 = the venv's) over `include/ src/ tests/`: `CLANG_FORMAT_BIN=../.venv/bin/clang-format bash
tools/clang_format_check.sh`. The tree was reformatted in one commit (2026-09-08); run
`clang-format -i` on what you touch. `-Wall -Wextra -Wpedantic` with no blanket `-Wno-*`: keep
`include/` warning-free (it hits every consumer's build). `docs/` holds only the live documents — `architecture.dox` (the Doxygen page),
`distributed_voronoi.md` (the MPI path) and `performance_report.md`; the dated design notes, phased
plans and benchmark records are in `docs/archive/` (indexed by its README, NOT maintained — several
describe the retired half-edge engine or the `vordyn` module name as current). A new document that
describes what the code does belongs in `docs/`; a plan or a campaign record belongs in
`docs/archive/`.
