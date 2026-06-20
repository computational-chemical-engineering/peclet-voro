# Kokkos device tests for vorflow (built only when -DVORFLOW_KOKKOS=ON).
#
# Phase 0: a smoke test proving the suite toolchain compiles+links a vorflow
# Kokkos target, runs a device kernel, exercises transport-core's tpx::View
# round-trip, and that vorflow's C++17 core types coexist under the C++20 Kokkos
# build. Device kernels and their parity tests are added in later phases.

add_executable(test_kokkos_smoke test_kokkos_smoke.cpp)
target_link_libraries(test_kokkos_smoke PRIVATE Kokkos::kokkos)
target_include_directories(test_kokkos_smoke PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  "${VORFLOW_TPX_INCLUDE}")
target_compile_features(test_kokkos_smoke PRIVATE cxx_std_20)
add_test(NAME test_kokkos_smoke COMMAND test_kokkos_smoke)
