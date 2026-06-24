# Kokkos device tests for vorflow (built only when -DVORFLOW_KOKKOS=ON).
#
# Phase 0: a smoke test proving the suite toolchain compiles+links a vorflow
# Kokkos target, runs a device kernel, exercises transport-core's tpx::View
# round-trip, and that vorflow's C++17 core types coexist under the C++20 Kokkos
# build. Device kernels and their parity tests are added in later phases.

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

# Phase 1: published data-layer round-trip (legacy <-> CSR <-> device view).
add_vorflow_kokkos_test(test_tessellation_view)

# Phase 2: device-callable cutter vs legacy topology (per-cell parallel_for).
add_vorflow_kokkos_test(test_cell_cutter)

# Phase 3: full-system device tessellation (grid + cutter + CSR) vs legacy.
add_vorflow_kokkos_test(test_tessellator)

# De-legacy Phase 3: device tessellator publishes full per-facet geometry (dV,
# area, connVec) ported from CellGeometry::computeEdgeInv + diffVolume.
add_vorflow_kokkos_test(test_device_geometry)

# SDF geometry: device boundary clip (cut planes at sdf=0, normal from grad) vs
# the legacy SignedDistanceBoundary build over the same geometry.
add_vorflow_kokkos_test(test_sdf_boundary_device)

# Phase 4: atomic-free pressure force over TessellationView vs legacy Euler.
add_vorflow_kokkos_test(test_euler_pressure)

# De-legacy Phase 2: atomic-free viscous force vs legacy NavierStokes.
add_vorflow_kokkos_test(test_viscous_device)

# De-legacy Phase 3: device interface-tension energy vs legacy IntfDyn.
add_vorflow_kokkos_test(test_interface_energy)
add_vorflow_kokkos_test(test_interface_force)

# De-legacy Phase 4: device-native Euler trajectory vs legacy ExplicitEuler.
add_vorflow_kokkos_test(test_device_step)

# De-legacy Phase 6: CPU incremental update on the new cutter vs full rebuild.
add_vorflow_kokkos_test(test_incremental_device)

# Throughput benchmark (device vs legacy) — a plain executable, not a ctest.
# The legacy side MUST be compiled with VORONOI_USE_OPENMP so its per-thread
# CellMaker parallel build is exercised; without it the legacy comparison runs
# serial and the device "wins" unfairly.
find_package(OpenMP REQUIRED)
add_executable(bench_device bench_device.cpp)
target_link_libraries(bench_device PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
# Add VORFLOW_CUTTER_PROFILE here to print the tessellator's per-cell cutCell2/cdist
# counts (serial); off by default so normal benchmarking has no extra work.
target_compile_definitions(bench_device PRIVATE VORONOI_USE_OPENMP MORTON_ENABLE_KOKKOS=1)
target_include_directories(bench_device PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
target_compile_features(bench_device PRIVATE cxx_std_20)

# Optional voro++ reference (the serial throughput target). FetchContent reuses the
# already-downloaded source; if it resolves, bench_device adds a voro++ column.
include(FetchContent)
FetchContent_Declare(voropp
  GIT_REPOSITORY https://github.com/chr1shr/voro.git
  GIT_TAG master)
set(VORO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_CMD_LINE OFF CACHE BOOL "" FORCE)
set(VORO_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(voropp)
if(TARGET voro++)
  target_link_libraries(bench_device PRIVATE voro++)
  target_compile_definitions(bench_device PRIVATE VORFLOW_HAVE_VOROPP)
endif()

# Option-A prototype: ConvexCell controlled unit tests (known cells, exact volumes).
add_vorflow_kokkos_test(test_convexcell_unit)

# G2 validation: ConvexCell per-facet geometry (area/dV/connVec) vs the half-edge oracle.
add_vorflow_kokkos_test(test_convexcell_geometry)

# Vertex-local sort-free geometry (design note): acceptance criteria in FP64.
add_vorflow_kokkos_test(test_pervertex_geometry)

# Phase 0 of the GPU-Voronoi-for-physics program: characterise the moving-point workload
# (topology-stable fraction vs per-step displacement). Uses the half-edge device tessellator.
add_executable(phase0_incremental phase0_incremental.cpp)
target_link_libraries(phase0_incremental PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
target_compile_definitions(phase0_incremental PRIVATE MORTON_ENABLE_KOKKOS=1)
target_include_directories(phase0_incremental PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
target_compile_features(phase0_incremental PRIVATE cxx_std_20)

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

# Per-cell cutter profiler (Phase 0): attributes cutCell2 time to its sub-phases.
# Built with VORFLOW_CUTTER_PROFILE so cell_cutter.hpp bumps the host-only counters.
add_executable(bench_cutter bench_cutter.cpp)
target_link_libraries(bench_cutter PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
target_compile_definitions(bench_cutter PRIVATE VORFLOW_CUTTER_PROFILE MORTON_ENABLE_KOKKOS=1)
target_include_directories(bench_cutter PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
target_compile_features(bench_cutter PRIVATE cxx_std_20)

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

# Compact-cell A/B: packed-triangle + recomputed-vertex ConvexCellCompact vs cached ConvexCell.
foreach(suffix "" "_f32")
  add_executable(bench_construct_compact${suffix} bench_construct_compact.cpp)
  target_link_libraries(bench_construct_compact${suffix} PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_construct_compact${suffix} PRIVATE MORTON_ENABLE_KOKKOS=1)
  target_include_directories(bench_construct_compact${suffix} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
  target_compile_features(bench_construct_compact${suffix} PRIVATE cxx_std_20)
endforeach()
target_compile_definitions(bench_construct_compact_f32 PRIVATE CC_FLOAT=1)

# Part II / Phase 1: incremental geometry re-eval over resident topology vs full rebuild.
add_executable(bench_incremental bench_incremental.cpp)
target_link_libraries(bench_incremental PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
target_compile_definitions(bench_incremental PRIVATE MORTON_ENABLE_KOKKOS=1)
target_include_directories(bench_incremental PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
target_compile_features(bench_incremental PRIVATE cxx_std_20)

# F4: ArborX BVH (kNN) neighbour query vs the grid gather. ArborX is in the same prefix as Kokkos.
find_package(ArborX CONFIG QUIET)
if(ArborX_FOUND)
  add_executable(bench_bvh_gather bench_bvh_gather.cpp)
  target_link_libraries(bench_bvh_gather PRIVATE Kokkos::kokkos ArborX::ArborX OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_bvh_gather PRIVATE MORTON_ENABLE_KOKKOS=1 CC_FLOAT=1)
  target_include_directories(bench_bvh_gather PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
  target_compile_features(bench_bvh_gather PRIVATE cxx_std_20)
  add_executable(bench_bvh_bestfirst bench_bvh_bestfirst.cpp)
  target_link_libraries(bench_bvh_bestfirst PRIVATE Kokkos::kokkos ArborX::ArborX OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_bvh_bestfirst PRIVATE MORTON_ENABLE_KOKKOS=1 CC_FLOAT=1)
  target_include_directories(bench_bvh_bestfirst PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${VORFLOW_TPX_INCLUDE}" "${VORFLOW_MORTON_INCLUDE}")
  target_compile_features(bench_bvh_bestfirst PRIVATE cxx_std_20)
  message(STATUS "vorflow: ArborX ${ArborX_VERSION} found -> bench_bvh_gather")
else()
  message(STATUS "vorflow: ArborX not found -> skipping bench_bvh_gather")
endif()
