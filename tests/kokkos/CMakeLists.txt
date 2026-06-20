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
