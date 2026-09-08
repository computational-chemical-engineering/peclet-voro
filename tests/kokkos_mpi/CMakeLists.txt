# peclet.voro — the distributed (MPI) test suite: np = 1, 2, 4 ctests, label `mpi`.
#
# Two ways in:
#   * from the main build (the normal way):  -DPECLET_VORO_MPI=ON -DPECLET_VORO_BUILD_TESTS=ON adds this
#     directory to the same tree as the single-rank tests, reusing the parent's Kokkos, MPI and the
#     already-resolved core / morton include paths (cmake/PecletDeps.cmake).
#   * standalone (e.g. a lean MPI-only tree on a cluster):
#       cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH=<suite>/extern/install/<backend> \
#             [-DPECLET_CORE_INCLUDE=... -DPECLET_VORO_MORTON_INCLUDE=...] [-DPECLET_VORO_MPIEXEC=...]
#     (defaults: the sibling checkouts ../../../core and ../../../morton).
#
# Oversubscribed runs (np=4 on a 2-slot box): -DMPIEXEC_PREFLAGS=--oversubscribe.
#
# The binaries validate that the owned-cell tessellation / repair / distributed FV solvers are
# identical single-rank vs multi-rank; they exit nonzero on failure.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(peclet_voro_mpi_tests LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  find_package(Kokkos CONFIG REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(PECLET_CORE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../core/include" CACHE PATH "core include/")
  set(PECLET_VORO_MORTON_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../morton/include"
      CACHE PATH "morton include/")
  if(NOT EXISTS "${PECLET_VORO_MORTON_INCLUDE}/morton/morton.hpp")
    message(FATAL_ERROR "peclet_voro_mpi_tests: morton not found at ${PECLET_VORO_MORTON_INCLUDE}")
  endif()
  if(NOT EXISTS "${PECLET_CORE_INCLUDE}/peclet/core/halo/particle_halo.hpp")
    message(FATAL_ERROR "peclet_voro_mpi_tests: core not found at ${PECLET_CORE_INCLUDE}")
  endif()
  enable_testing()
endif()

# The launcher MUST come from the same MPI as the library we link: FindMPI takes the first mpiexec on
# PATH (a ParaView/MPICH hydra, say), which starts `-np N` independent OpenMPI singletons — each passes
# a multi-rank-vs-single-rank check trivially (seen 2026-09-08). So look next to MPI_CXX_COMPILER
# first; -DPECLET_VORO_MPIEXEC=... overrides. The binaries also check MPI_COMM_WORLD's size against
# PECLET_VORO_EXPECT_NP (set per test below) and fail on a mismatch.
get_filename_component(_voro_mpi_bindir "${MPI_CXX_COMPILER}" DIRECTORY)
find_program(PECLET_VORO_MPIEXEC NAMES mpirun mpiexec HINTS "${_voro_mpi_bindir}" NO_DEFAULT_PATH
             DOC "MPI launcher for the voro MPI tests (same MPI as MPI_CXX_COMPILER)")
if(NOT PECLET_VORO_MPIEXEC)
  set(PECLET_VORO_MPIEXEC "${MPIEXEC_EXECUTABLE}")
endif()
message(STATUS "voro MPI tests: launcher ${PECLET_VORO_MPIEXEC} ${MPIEXEC_PREFLAGS}")

# The MPI test binaries (also runnable by hand under mpirun -np <R> [N ...] for at-scale numbers).
foreach(b bench_voronoi_mpi bench_repair_mpi test_flow_mpi)
  add_executable(${b} ${b}.cpp)
  target_include_directories(${b} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${PECLET_CORE_INCLUDE}" "${PECLET_VORO_MORTON_INCLUDE}")
  target_compile_definitions(${b} PRIVATE PECLET_VORO_MPI MORTON_ENABLE_KOKKOS=1)
  target_link_libraries(${b} PRIVATE Kokkos::kokkos MPI::MPI_CXX)
  target_compile_features(${b} PRIVATE cxx_std_20)
endforeach()

# Both benches self-check (voronoi: completeness invariants — owned count, no bad cells,
# Σvol == |box|; repair: MPI repair-vs-cold-rebuild exactness) and exit nonzero on failure,
# so register test-sized runs as ctests at np=1,2,4. The mode switches (FLOW_MPI_IMPLICIT / _COVOLUME)
# reach the ranks through ctest's ENVIRONMENT: both OpenMPI and hydra hand local ranks the launcher's
# environment (no launcher-specific `-x`, which hydra rejects).
set(_mpirun ${PECLET_VORO_MPIEXEC} ${MPIEXEC_PREFLAGS})
foreach(np 1 2 4)
  add_test(NAME voronoi_mpi_np${np}
           COMMAND ${_mpirun} -np ${np} $<TARGET_FILE:bench_voronoi_mpi> 20000 1)
  add_test(NAME repair_mpi_np${np}
           COMMAND ${_mpirun} -np ${np} $<TARGET_FILE:bench_repair_mpi> 20000 4)
  # Rung A0: the same through a replicated SDF scene — repair == distributed cold == single-rank cold.
  add_test(NAME repair_sdf_mpi_np${np}
           COMMAND ${_mpirun} -np ${np} $<TARGET_FILE:bench_repair_mpi> 20000 4 --sdf)
  # The distributed FV solvers (collocated default / implicit / covolume): np=1 bit-exact on host.
  add_test(NAME flow_mpi_np${np}
           COMMAND ${_mpirun} -np ${np} $<TARGET_FILE:test_flow_mpi> 12 20)
  add_test(NAME flow_mpi_implicit_np${np}
           COMMAND ${_mpirun} -np ${np} $<TARGET_FILE:test_flow_mpi> 12 20)
  add_test(NAME flow_mpi_covolume_np${np}
           COMMAND ${_mpirun} -np ${np} $<TARGET_FILE:test_flow_mpi> 12 20)
  set_tests_properties(voronoi_mpi_np${np} repair_mpi_np${np} repair_sdf_mpi_np${np} flow_mpi_np${np}
                       flow_mpi_implicit_np${np} flow_mpi_covolume_np${np} PROPERTIES
    LABELS mpi PROCESSORS ${np}
    ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;PECLET_VORO_EXPECT_NP=${np}")
  set_property(TEST flow_mpi_implicit_np${np} APPEND PROPERTY ENVIRONMENT "FLOW_MPI_IMPLICIT=1")
  set_property(TEST flow_mpi_covolume_np${np} APPEND PROPERTY ENVIRONMENT "FLOW_MPI_COVOLUME=1")
endforeach()
