# vorflow — distributed (MPI) Voronoi tessellation test (migration Phase 6).
#
# Standalone find_package project (mirrors dem/tests/kokkos_mpi): Kokkos + MPI +
# the header-only transport-core + the vorflow headers (include/). It validates
# that the owned-cell tessellation is identical single-rank vs multi-rank.
#   cmake -S tests/kokkos_mpi -B build_kmpi \
#         -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
#         -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
#   cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(vorflow_mpi_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
find_package(MPI REQUIRED)

set(TPX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../transport-core" CACHE PATH "transport-core repo")
# The device tessellator's Z-order grid ordering uses morton::Morton<3,21> (a sibling
# suite repo). Mirror the main build's include + MORTON_ENABLE_KOKKOS define so the
# software bit path is device-callable (MORTON_HD -> KOKKOS_FUNCTION).
set(MORTON_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../morton/include"
    CACHE PATH "morton repo include/")
if(NOT EXISTS "${MORTON_INCLUDE}/morton/morton.hpp")
  message(FATAL_ERROR
    "vorflow_mpi_tests: morton not found at ${MORTON_INCLUDE} (clone it as a sibling repo)")
endif()
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
endif()

enable_testing()

# Distributed benchmarks (run manually under mpirun -np <R> [N ...]).
foreach(b bench_voronoi_mpi bench_repair_mpi)
  add_executable(${b} ${b}.cpp)
  target_include_directories(${b} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${TPX_DIR}/include ${MORTON_INCLUDE})
  target_compile_definitions(${b} PRIVATE PECLET_VORO_MPI MORTON_ENABLE_KOKKOS=1)
  target_link_libraries(${b} PRIVATE Kokkos::kokkos MPI::MPI_CXX)
endforeach()
