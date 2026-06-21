cmake_minimum_required(VERSION 3.16)

project(
  vorflow
  VERSION 1.0.0
  DESCRIPTION "Header-only Voronoi-dynamics simulation library for moving-particle methods"
  LANGUAGES CXX
)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ── Compile options ───────────────────────────────────────────────────────────
add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wno-unused-variable
  -Wno-unused-parameter
  -Wno-sign-compare
  -Wno-reorder
)

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(Boost 1.65 REQUIRED)
find_package(OpenMP)

# ── Header-only interface library ────────────────────────────────────────────
add_library(vorflow INTERFACE)
add_library(vorflow::vorflow ALIAS vorflow)

target_include_directories(vorflow
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vorflow
  INTERFACE
    Boost::boost
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vorflow INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(vorflow INTERFACE VORONOI_USE_OPENMP)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
option(VORONOI_BUILD_TESTS "Build the test programs" ON)
if(VORONOI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
option(VORONOI_BUILD_BENCHMARKS "Build the performance benchmarks" OFF)
if(VORONOI_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ── Python bindings ───────────────────────────────────────────────────────────
option(VORONOI_BUILD_PYTHON "Build the pybind11 Python module (vorflow)" OFF)
if(VORONOI_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# ── Kokkos / suite device build (migration) ───────────────────────────────────
# Suite-aligned build: find_package(Kokkos/ArborX) against the bootstrapped prefix
# (../extern/install/<backend>, built by ../tools/bootstrap_deps.sh), mirroring
# dem/sdflow. Device kernels are added across the migration phases; Phase 0 wires
# the toolchain and a smoke test. The legacy header-only path above is untouched.
#
#   cmake -S . -B build/host-openmp -DVORFLOW_KOKKOS=ON \
#     -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"
#   cmake --build build/host-openmp -j
#
# Add -DVORFLOW_MPI=ON to link MPI + transport-core for the distributed path.
option(VORFLOW_KOKKOS "Build the Kokkos device path (find_package(Kokkos))" OFF)
option(VORFLOW_MPI "Build the distributed path against MPI + transport-core" OFF)

if(VORFLOW_KOKKOS)
  find_package(Kokkos CONFIG REQUIRED)
  # ArborX is the strongly-polydisperse / Power neighbour-search policy; it is not
  # needed until Phase 3 (the cell-linked grid covers near-monodisperse seeds), so
  # it is optional here to keep the Phase 0-2 toolchain (and standalone CI) lean.
  find_package(ArborX CONFIG QUIET)
  if(ArborX_FOUND)
    message(STATUS "vorflow: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}), ArborX ${ArborX_VERSION}")
  else()
    message(STATUS "vorflow: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}); ArborX not found "
                   "(required from Phase 3 onward)")
  endif()

  set(VORFLOW_TPX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include"
      CACHE PATH "transport-core include dir")

  if(VORFLOW_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    if(NOT EXISTS "${VORFLOW_TPX_INCLUDE}/tpx/halo/particle_halo_kokkos.hpp")
      message(FATAL_ERROR
        "transport-core not found at ${VORFLOW_TPX_INCLUDE} (clone it as a sibling repo)")
    endif()
    message(STATUS "vorflow: distributed path ENABLED (MPI + transport-core)")
  endif()

  if(VORONOI_BUILD_TESTS)
    add_subdirectory(tests/kokkos)
  endif()

  # Device-native Python module `vorflow` (the de-legacy surface over the device
  # tessellator + ExplicitEulerDevice). Opt-in; shares the name with the legacy
  # python/ module (build only one). pybind11 is fetched, no install needed.
  option(VORFLOW_BUILD_PYTHON "Build the device-native vorflow Python module" OFF)
  if(VORFLOW_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.13.6)
    FetchContent_MakeAvailable(pybind11)
    pybind11_add_module(vorflow_device NO_EXTRAS src/vorflow_bindings.cpp)
    set_target_properties(vorflow_device PROPERTIES OUTPUT_NAME vorflow)
    target_include_directories(vorflow_device PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${VORFLOW_TPX_INCLUDE}")
    target_link_libraries(vorflow_device PRIVATE Kokkos::kokkos)
    target_compile_features(vorflow_device PRIVATE cxx_std_20)
  endif()
endif()

# ── Documentation ─────────────────────────────────────────────────────────────
option(VORONOI_BUILD_DOCS "Build Doxygen documentation" OFF)
if(VORONOI_BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "vorflow")
    set(DOXYGEN_PROJECT_BRIEF "Voronoi-dynamics simulation library")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "README.md")
    # power_cell_solver_spec.md uses LaTeX \dot{...}, which collides with Doxygen's
    # \dot graph command; exclude that design note from the API docs.
    set(DOXYGEN_EXCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/docs/power_cell_solver_spec.md")
    doxygen_add_docs(
      docs
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/python"
      "${CMAKE_CURRENT_SOURCE_DIR}/docs"
      "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
      COMMENT "Generating Doxygen documentation"
    )
  endif()
endif()

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
  TARGETS vorflow
  EXPORT vorflowTargets
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT vorflowTargets
  FILE vorflowTargets.cmake
  NAMESPACE vorflow::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vorflow
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vorflowConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vorflowConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vorflow"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/vorflowConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/vorflowConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/vorflowConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vorflow"
)
