# Kokkos device tests + benchmarks for peclet.voro (built only when -DPECLET_VORO_KOKKOS=ON).
#
#   test_*.cpp   ctests (PECLET_VORO_BUILD_TESTS) — every one runs in CI.
#   bench_*.cpp  timing instruments (PECLET_VORO_BUILD_BENCHMARKS), registered as ctests with the
#                label `bench` at test-sized arguments so `ctest -N` lists them and `ctest -LE bench`
#                excludes them. bench_dynamic_update is the one bench that is ALSO a gate
#                (`--gates`): its FP64 binary is built with the tests and the gate run is unlabelled.
#
# Include paths / defines come from the parent (PECLET_CORE_INCLUDE, PECLET_VORO_MORTON_INCLUDE from
# cmake/PecletDeps.cmake, Kokkos::kokkos from the prefix); this directory is not a standalone project.

set(_voro_test_includes
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${PECLET_CORE_INCLUDE}" "${PECLET_VORO_MORTON_INCLUDE}")

function(add_voro_kokkos_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE Kokkos::kokkos)
  target_include_directories(${name} PRIVATE ${_voro_test_includes})
  target_compile_definitions(${name} PRIVATE MORTON_ENABLE_KOKKOS=1 PECLET_CORE_HAVE_MORTON=1)
  target_compile_features(${name} PRIVATE cxx_std_20)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false")
endfunction()

# A bench executable `target` from `source` (+ extra compile definitions in ARGN).
function(add_voro_bench target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
  target_compile_definitions(${target} PRIVATE MORTON_ENABLE_KOKKOS=1 ${ARGN})
  target_include_directories(${target} PRIVATE ${_voro_test_includes})
  target_compile_features(${target} PRIVATE cxx_std_20)
endfunction()

# A `bench`-labelled ctest running `target` with test-sized ARGN (never run by `ctest -LE bench`).
function(add_voro_bench_test name target)
  add_test(NAME ${name} COMMAND ${target} ${ARGN})
  set_tests_properties(${name} PROPERTIES LABELS bench ENVIRONMENT "OMP_PROC_BIND=false")
endfunction()

# ── Tests ──────────────────────────────────────────────────────────────────────────────────────
if(PECLET_VORO_BUILD_TESTS)
  add_voro_kokkos_test(test_kokkos_smoke)

  # voro over the SHARED core geometry: SdfScene wraps peclet::core::geom::evalTree, so the cell
  # clipper and mesh optimiser reach the whole analytic vocabulary (capsule/torus/cone/ellipsoid/
  # superquadric + CSG + transforms + grids), not just the three legacy single-shape providers.
  # Also pins that a legacy provider and the same shape as a scene node agree bit-for-bit.
  add_voro_kokkos_test(test_sdf_scene)

  # Rung A0 (Voronoi methods plan): SDF solids on the MOVING-POINT path — persisted wall planes +
  # the boundary watch; repair vs cold SDF build through a sphere ∪ torus scene, cells entering /
  # leaving the solid, exact wall mode gated at 1e-9, skin mode reported.
  add_voro_kokkos_test(test_sdf_dynamic)

  # Rung A3: the facet-edge area-Jacobian CSR the build publishes + the energy layer on it
  # (interface / wall / volume terms, Voronoi + Power chains, the shared seed-foot wall chain):
  # published == reconstruction to round-off, FD-exact gradients incl. weights and a flat wall.
  add_voro_kokkos_test(test_energy_layer)

  # Rung A1: second-order (sagitta-shifted) wall placement on curved solids — fluid-volume error
  # vs the bare tangent clip on a sphere and a cavity at three resolutions (10x + order gate).
  add_voro_kokkos_test(test_sdf_curved)

  # Track C, rung C1: the face mesh + covolume operators (div / two-point grad / Laplacian /
  # Green–Gauss / Perot): adjointness + symmetry to round-off, exact uniform-field reconstruction,
  # manufactured-solution convergence on a jittered lattice.
  add_voro_kokkos_test(test_fv_operators)

  # Track B, rung B1: centroidal (Lloyd) relaxation of a random grid — skewness / volume spread /
  # Poisson residual consistency all drop (grid quality is solver quality).
  add_voro_kokkos_test(test_grid_relax)

  # Track B, rung B3: the internal PolyMesh (shared vertices, ordered polygons, owner/neighbour,
  # wall patches) — watertight, Euler-exact, volumes == engine to 1e-12, VTU written.
  add_voro_kokkos_test(test_polymesh)

  # Track B, rung B4: grid quality IS solver quality — Poisson residual/solution error vs skewness at
  # fixed h (Lloyd-relaxed jittered lattices) and the solution's convergence order on relaxed grids.
  add_voro_kokkos_test(test_grid_quality)
  add_voro_kokkos_test(test_covolume_ns)
  add_voro_kokkos_test(test_collocated_ns)
  add_voro_kokkos_test(test_body_fitted)
  add_voro_kokkos_test(test_covolume_dec)
  add_voro_kokkos_test(test_permeability)

  # Oracle-free acceptance for the device tessellation: geometric invariants (space-filling, positive
  # volumes, completeness, facet-area reciprocity A_ij = -A_ji, area closure, topology reciprocity) over
  # several sizes/seeds. Replaces the retired half-edge-oracle comparison tests (tessellator,
  # device_geometry, tessellation_view, sdf_boundary_device) and the device-vs-legacy-physics tests
  # (euler_pressure, viscous_device, interface_energy, device_step); the device physics path is now
  # exercised end-to-end by the Python smoke test (python/test_voro.py).
  add_voro_kokkos_test(test_tessellation_invariants)

  # P1 acceptance for the POWER (Laguerre) tessellation: equal-weights regression to Voronoi
  # (machine-exact), oracle-free power-diagram invariants (space-filling / radical-facet reciprocity),
  # and an independent host brute-force radical-plane oracle (per-cell volume + neighbour set).
  add_voro_kokkos_test(test_power_cells)

  # P5: differentiable SDF wall force (seed-foot J_wall). FD-vs-analytic — flat wall exact, sphere
  # first-order.
  add_voro_kokkos_test(test_sdf_policy)

  # Semi-discrete OT volume-control optimiser: Newton on the power weights (L δw = Vset−V) via the core
  # MomentumSolver, driving cells to target volumes (uniform + spatially-graded refinement).
  add_voro_kokkos_test(test_mesh_optimizer)

  # Option-A prototype: ConvexCell controlled unit tests (known cells, exact volumes).
  add_voro_kokkos_test(test_convexcell_unit)

  # TrackAdj axis: incremental edge-adjacency + Lawson local-convexity certificate foundation gates.
  add_voro_kokkos_test(test_convexcell_adj)

  # Vertex-local sort-free geometry (design note): acceptance criteria in FP64.
  add_voro_kokkos_test(test_pervertex_geometry)

  # Part II: consolidated moving-point UPDATE driver. Its Phase-1 gates double as a self-test (GATE 0
  # corruption catch, GATE 1 subset/partner/skin), so the FP64 binary is always built with the tests;
  # the timing modes (strategy sweep, phase-0 characterisation) are `bench`-labelled below.
  add_voro_bench(bench_dynamic_update bench_dynamic_update.cpp)
  add_test(NAME bench_dynamic_update_gates COMMAND bench_dynamic_update 20000 4 --gates)
  set_tests_properties(bench_dynamic_update_gates PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false")

  # Architectural invariant (pure-script ctest): the tessellation core must never include a physics
  # header, so the cutter stays reusable for non-physics consumers (packing/microstructure/meshing).
  add_test(NAME test_include_graph
    COMMAND ${CMAKE_COMMAND} -E env bash
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/check_include_graph.sh"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../include/peclet/voro")
endif()

# ── Benchmarks (opt-in, label `bench`) ─────────────────────────────────────────────────────────
if(PECLET_VORO_BUILD_BENCHMARKS)
  # voro++ reference (fetched once; the serial throughput target for bench_convexcell). Pinned to a
  # commit: the only release tag (v0.4.6, 2013) predates Voro++'s CMake support, so `master` was the
  # de-facto pin — this is the commit it resolved to on 2026-09-08.
  include(FetchContent)
  FetchContent_Declare(voropp
    GIT_REPOSITORY https://github.com/chr1shr/voro.git
    GIT_TAG b0dac575a47af0f90b5b100e6dc199a493c7cb83)
  set(VORO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(VORO_BUILD_CMD_LINE OFF CACHE BOOL "" FORCE)
  set(VORO_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(voropp)

  # Option-A prototype: compact ConvexCell (dual-triangle) one-cell-per-thread benchmark, FP64 +
  # FP32 (the precision-policy lever for the crippled consumer FP64).
  foreach(suffix "" "_f32")
    add_voro_bench(bench_convexcell${suffix} bench_convexcell.cpp)
    if(TARGET voro++)
      target_link_libraries(bench_convexcell${suffix} PRIVATE voro++)
      target_compile_definitions(bench_convexcell${suffix} PRIVATE PECLET_VORO_HAVE_VOROPP)
    endif()
  endforeach()
  target_compile_definitions(bench_convexcell_f32 PRIVATE CC_FLOAT=1)
  add_voro_bench_test(bench_convexcell_run bench_convexcell 20000)

  # F2/F3: isolate cell construction from cached candidate planes (B+G, no gather), per tier.
  foreach(suffix "" "_f32")
    add_voro_bench(bench_construct${suffix} bench_construct.cpp)
  endforeach()
  target_compile_definitions(bench_construct_f32 PRIVATE CC_FLOAT=1)
  add_voro_bench_test(bench_construct_run bench_construct 50000)

  # Part II / Phase 1: incremental geometry re-eval over resident topology vs full rebuild.
  add_voro_bench(bench_incremental bench_incremental.cpp)
  add_voro_bench_test(bench_incremental_run bench_incremental 50000)

  # bench_dynamic_update: the FP64 binary is a test-time target (above, or built here when the tests
  # are off); the FP32 variant and the full strategy sweep are benchmark-only.
  if(NOT TARGET bench_dynamic_update)
    add_voro_bench(bench_dynamic_update bench_dynamic_update.cpp)
  endif()
  add_voro_bench(bench_dynamic_update_f32 bench_dynamic_update.cpp CC_FLOAT=1)
  add_voro_bench_test(bench_dynamic_update_run bench_dynamic_update 20000 4)

  # Interstitial-packing volume-equalisation solver study (Newton preconditioners: CG / Jacobi /
  # colored-GS / Chebyshev / GraphAMG, plus first-order steepest-descent & nonlinear-CG). Needs a
  # packing.txt produced by ../pack_bed.py (peclet.dem), so it is built but not registered.
  add_voro_bench(bench_mesh_optimizer bench_mesh_optimizer.cpp PECLET_CORE_HAVE_MORTON=1)

  # CSV performance + memory + accuracy report (cold-build N-sweep + repair displacement-sweep).
  foreach(suffix "" "_f32")
    add_voro_bench(bench_report${suffix} bench_report.cpp)
  endforeach()
  target_compile_definitions(bench_report_f32 PRIVATE CC_FLOAT=1)
  add_voro_bench_test(bench_report_run bench_report --repair 20000 4)
endif()
