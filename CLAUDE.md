# CLAUDE.md — peclet.voro

Dynamic 3-D Voronoi tessellation of moving particles on Kokkos (CUDA/HIP/OpenMP), header-only C++20
under `include/peclet/voro/`, driven from Python as `peclet.voro` (`src/voro_bindings.cpp`, nanobind on
core's zero-copy bridge). Names follow `../docs/NAMING.md`: `set_domain(extent=)` (origin (0,0,0) and
all-periodic are CHECKED), `set_dt` + `dt` + `step(n)`, scalars/counts bare, copied-out arrays `get_*`.

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

## Header map (one-way layering, enforced by `tools/check_include_graph.sh` = ctest `test_include_graph`)

| layer | headers | what |
|---|---|---|
| 0 | `convex_cell`, `plane_policy`, `tessellation_view`, `topology_store`, `transpose`, `verlet_skin`, `tess_grid` | the dual-triangle cell, plane policies, the published CSR view, leaves |
| 1 | `sdf` | SDF half-space clipping (tangent + sagitta), wall store |
| 2 | `tessellator`, `subset_gather` | cold build: grid + worklist gather + clip + publish |
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
