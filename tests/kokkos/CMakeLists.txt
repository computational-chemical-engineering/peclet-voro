# Kokkos device tests + benchmarks for vorflow (built only when -DVORFLOW_KOKKOS=ON).

function(add_vorflow_kokkos_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE Kokkos::kokkos)
  target_include_directories(${name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    "${VORFLOW_TPX_INCLUDE}"
    "${VORFLOW_MORTON_INCLUDE}")
  target_compile_definitions(${name} PRIVATE MORTON_ENABLE_KOKKOS=1)
  target_compile_features(${name} PRIVATE cxx_std_20)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false")
endfunction()

add_vorflow_kokkos_test(test_kokkos_smoke)

# Oracle-free acceptance for the device tessellation: geometric invariants (space-filling, positive
# volumes, completeness, facet-area reciprocity A_ij = -A_ji, area closure, topology reciprocity) over
# several sizes/seeds. Replaces the retired half-edge-oracle comparison tests (tessellator,
# device_geometry, tessellation_view, sdf_boundary_device) and the device-vs-legacy-physics tests
# (euler_pressure, viscous_device, interface_energy, device_step); the device physics path is now
# exercised end-to-end by the Python smoke test (python/test_vorflow.py).
add_vorflow_kokkos_test(test_tessellation_invariants)

# voro++ reference (fetched once; the serial throughput target for bench_convexcell).
include(FetchContent)
FetchContent_Declare(voropp
  GIT_REPOSITORY https://github.com/chr1shr/voro.git
  GIT_TAG master)
set(VORO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_CMD_LINE OFF CACHE BOOL "" FORCE)
set(VORO_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(voropp)

# Option-A prototype: ConvexCell controlled unit tests (known cells, exact volumes).
add_vorflow_kokkos_test(test_convexcell_unit)

# TrackAdj axis: incremental edge-adjacency + Lawson local-convexity certificate foundation gates.
add_vorflow_kokkos_test(test_convexcell_adj)

# Vertex-local sort-free geometry (design note): acceptance criteria in FP64.
add_vorflow_kokkos_test(test_pervertex_geometry)

# Option-A prototype: compact ConvexCell (dual-triangle) one-cell-per-thread benchmark.
add_executable(bench_convexcell bench_convexcell.cpp)
target_link_libraries(bench_convexcell PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
target_compile_definitions(bench_convexcell PRIVATE MORTON_ENABLE_KOKKOS=1)
target_include_directories(bench_convexcell PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
target_compile_features(bench_convexcell PRIVATE cxx_std_20)
if(TARGET voro++)
  target_link_libraries(bench_convexcell PRIVATE voro++)
  target_compile_definitions(bench_convexcell PRIVATE VORFLOW_HAVE_VOROPP)
endif()

# FP32 ConvexCell variant (precision-policy lever for the crippled consumer FP64).
add_executable(bench_convexcell_f32 bench_convexcell.cpp)
target_link_libraries(bench_convexcell_f32 PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
target_compile_definitions(bench_convexcell_f32 PRIVATE MORTON_ENABLE_KOKKOS=1 CC_FLOAT=1)
target_include_directories(bench_convexcell_f32 PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
target_compile_features(bench_convexcell_f32 PRIVATE cxx_std_20)
if(TARGET voro++)
  target_link_libraries(bench_convexcell_f32 PRIVATE voro++)
  target_compile_definitions(bench_convexcell_f32 PRIVATE VORFLOW_HAVE_VOROPP)
endif()

# F2/F3: isolate cell construction from cached candidate planes (B+G, no gather), per tier.
foreach(suffix "" "_f32")
  add_executable(bench_construct${suffix} bench_construct.cpp)
  target_link_libraries(bench_construct${suffix} PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_construct${suffix} PRIVATE MORTON_ENABLE_KOKKOS=1)
  target_include_directories(bench_construct${suffix} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
  target_compile_features(bench_construct${suffix} PRIVATE cxx_std_20)
endforeach()
target_compile_definitions(bench_construct_f32 PRIVATE CC_FLOAT=1)

# Part II / Phase 1: incremental geometry re-eval over resident topology vs full rebuild.
add_executable(bench_incremental bench_incremental.cpp)
target_link_libraries(bench_incremental PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
target_compile_definitions(bench_incremental PRIVATE MORTON_ENABLE_KOKKOS=1)
target_include_directories(bench_incremental PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
target_compile_features(bench_incremental PRIVATE cxx_std_20)

# Part II: consolidated moving-point UPDATE driver — Phase-0 validators (invariants + oracle diff) +
# benchmark foundation (strategy sweep, distributions, phase-0 characterization) AND the Phase-1
# repair-primitive gates (subset gather == oracle, certificate partner extraction, Verlet skin).
# Replaces bench_update_strategies.cpp + phase0_incremental.cpp. FP64 + FP32 variants.
foreach(suffix "" "_f32")
  add_executable(bench_dynamic_update${suffix} bench_dynamic_update.cpp)
  target_link_libraries(bench_dynamic_update${suffix} PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_dynamic_update${suffix} PRIVATE MORTON_ENABLE_KOKKOS=1)
  target_include_directories(bench_dynamic_update${suffix} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
  target_compile_features(bench_dynamic_update${suffix} PRIVATE cxx_std_20)
endforeach()
target_compile_definitions(bench_dynamic_update_f32 PRIVATE CC_FLOAT=1)

# CSV performance + memory + accuracy report benchmark (cold-build N-sweep + repair displacement-sweep).
foreach(suffix "" "_f32")
  add_executable(bench_report${suffix} bench_report.cpp)
  target_link_libraries(bench_report${suffix} PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_report${suffix} PRIVATE MORTON_ENABLE_KOKKOS=1)
  target_include_directories(bench_report${suffix} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
  target_compile_features(bench_report${suffix} PRIVATE cxx_std_20)
endforeach()
target_compile_definitions(bench_report_f32 PRIVATE CC_FLOAT=1)

# Phase-1 gates double as a self-test (GATE 0 corruption catch, GATE 1 subset/partner/skin).
add_test(NAME bench_dynamic_update_gates COMMAND bench_dynamic_update 20000 4 --gates)
set_tests_properties(bench_dynamic_update_gates PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false")


# Architectural invariant (pure-script ctest): the tessellation core must never include a physics
# header, so the cutter stays reusable for non-physics consumers (packing/microstructure/meshing).
add_test(NAME test_include_graph
  COMMAND ${CMAKE_COMMAND} -E env bash
          "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/check_include_graph.sh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../include/vorflow")
