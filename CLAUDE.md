# CLAUDE.md — peclet.voro

Dynamic 3-D Voronoi tessellation of moving particles on Kokkos (CUDA/HIP/OpenMP), header-only C++20
under `include/peclet/voro/`, driven from Python as `peclet.voro` (`src/voro_bindings.cpp`, nanobind on
core's zero-copy bridge). Names follow `../docs/NAMING.md`: `set_domain(extent=)` (origin (0,0,0) and
all-periodic are CHECKED), `set_dt` + `dt` + `step(n)`, scalars/counts bare, copied-out arrays `get_*`.

## Build + test (the tests exist ONLY under `PECLET_VORO_KOKKOS=ON`)

```bash
source ../.venv/bin/activate                         # the one suite venv (nanobind, numpy)
cmake -B build_dev -DPECLET_VORO_KOKKOS=ON -DPECLET_VORO_BUILD_PYTHON=ON \
      -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"    # or nvidia-cuda (nvcc on PATH)
cmake --build build_dev -j8                          # -> build_dev/peclet/voro/_voro.*.so
OMP_NUM_THREADS=4 OMP_PROC_BIND=false ctest --test-dir build_dev --output-on-failure   # 24 tests
PYTHONPATH=build_dev python python/test_voro.py      # the Python smoke test on its own
```

A plain `cmake -B build` (no Kokkos) configures zero tests — "No tests were found!!!" is not a
failure signal. `-DPECLET_VORO_MPI=ON` adds `VoronoiHalo` to the module; the MPI tests are a separate
project (below). `core` and `morton` are found as sibling checkouts (`../core`, `../morton`) through
`cmake/PecletDeps.cmake`. The version is read from `pyproject.toml` (do not edit `project(VERSION)`).

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

`tests/kokkos/` holds both: `test_*.cpp` are ctests (22, via `add_voro_kokkos_test`); `bench_*.cpp` are
always built but not run, except `bench_dynamic_update --gates` (ctest `bench_dynamic_update_gates`).
Plus `test_include_graph` (script) and `test_voro_python` (`python/test_voro.py`, needs
`PECLET_VORO_BUILD_PYTHON=ON`) = 24. Voro++ is fetched only as `bench_convexcell`'s throughput reference.

## MPI tests (`tests/kokkos_mpi/`, project `peclet_voro_mpi_tests`)

```bash
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" \
      -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun && cmake --build build_kmpi -j8
OMP_NUM_THREADS=1 OMP_PROC_BIND=false ctest --test-dir build_kmpi --output-on-failure   # 18 tests, np = 1,2,4
```

`bench_voronoi_mpi` / `bench_repair_mpi` (+ `--sdf`) self-check owned-cell exactness vs single rank;
`test_flow_mpi` gates the distributed collocated / covolume / implicit solvers (np=1 bit-exact on host).

## Style

Google style via `.clang-format`; `CLANG_FORMAT_BIN=clang-format-18 bash tools/clang_format_check.sh` is
what CI runs, **informational only** (~570 pre-existing violations, mostly unicode-in-comment lines in
`repair.hpp`, `sdf.hpp`, `tessellator.hpp` + tests). Keep new code clean; a repo-wide reformat is a
separate deliberate commit. Design notes live in `docs/*.md`; `docs/architecture.dox` is the Doxygen page.
