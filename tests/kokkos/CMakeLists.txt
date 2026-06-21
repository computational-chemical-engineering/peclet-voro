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
    "${VORFLOW_TPX_INCLUDE}")
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

# De-legacy Phase 4: device-native Euler trajectory vs legacy ExplicitEuler.
add_vorflow_kokkos_test(test_device_step)

# De-legacy Phase 6: CPU incremental update on the new cutter vs full rebuild.
add_vorflow_kokkos_test(test_incremental_device)
