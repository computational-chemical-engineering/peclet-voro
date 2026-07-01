cmake_minimum_required(VERSION 3.16)

project(
  peclet_voro
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
find_package(OpenMP)

# ── Header-only interface library ────────────────────────────────────────────
# The voro headers are header-only; this target just carries the include path for downstream
# consumers. The device path (tests/kokkos, src/) sets its own include dirs against core +
# morton, so it does not link this target.
add_library(vorflow INTERFACE)
add_library(vorflow::vorflow ALIAS vorflow)

target_include_directories(vorflow
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Tests ─────────────────────────────────────────────────────────────────────
# Only the Kokkos device tests remain (tests/kokkos, configured below). The legacy CPU half-edge
# test suite (tests/) was retired together with the half-edge engine.
option(PECLET_VORO_BUILD_TESTS "Build the test programs" ON)
if(PECLET_VORO_BUILD_TESTS)
  enable_testing()
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
# Device benchmarks live under tests/kokkos (bench_*); the legacy CPU benchmark suite was retired.

# ── Python bindings ───────────────────────────────────────────────────────────
# The Python surface is the device-native nanobind module `peclet.voro` (src/voro_bindings.cpp),
# configured below with -DVORFLOW_BUILD_PYTHON=ON. The legacy pybind11 module over the retired
# half-edge engine has been removed.

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
# Add -DVORFLOW_MPI=ON to link MPI + core for the distributed path.
option(PECLET_VORO_KOKKOS "Build the Kokkos device path (find_package(Kokkos))" OFF)
option(PECLET_VORO_MPI "Build the distributed path against MPI + core" OFF)

if(PECLET_VORO_KOKKOS)
  # Dependencies via the vendored PecletDeps helper: installed prefix + sibling checkouts for the
  # dev/suite build, or FetchContent-built Kokkos + fetched core/morton headers for a
  # self-contained sdist/wheel (cibuildwheel). See cmake/PecletDeps.cmake.
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(PecletDeps)
  peclet_require_kokkos()
  # ArborX is the strongly-polydisperse / Power neighbour-search policy; it is not needed until Phase 3
  # (the cell-linked grid covers near-monodisperse seeds), so it stays optional (found from a prefix if
  # present) to keep the Phase 0-2 toolchain and the lean CPU wheel small.
  find_package(ArborX CONFIG QUIET)
  if(ArborX_FOUND)
    message(STATUS "voro: ArborX ${ArborX_VERSION}")
  else()
    message(STATUS "voro: ArborX not found (required from Phase 3 onward)")
  endif()

  # core's zero-copy bridge + the morton spatial-index primitive (the device tessellator's
  # Z-order grid uses morton::Morton<3,21>; MORTON_ENABLE_KOKKOS -> MORTON_HD is KOKKOS_FUNCTION).
  peclet_sibling_include(peclet-core "${PECLET_TPX_TAG}" "../core" PECLET_VORO_TPX_INCLUDE)
  peclet_sibling_include(peclet-morton         "${PECLET_MORTON_TAG}" "../morton"      PECLET_VORO_MORTON_INCLUDE)

  if(PECLET_VORO_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    if(NOT EXISTS "${PECLET_VORO_TPX_INCLUDE}/peclet/core/halo/particle_halo_kokkos.hpp")
      message(FATAL_ERROR
        "core not found at ${PECLET_VORO_TPX_INCLUDE} (clone it as a sibling repo)")
    endif()
    message(STATUS "voro: distributed path ENABLED (MPI + core)")
  endif()

  if(PECLET_VORO_BUILD_TESTS)
    add_subdirectory(tests/kokkos)
  endif()

  # Device-native Python module `peclet.voro` (the de-legacy surface over the device tessellator +
  # ExplicitEulerDevice). Opt-in. nanobind is found via the active interpreter (PecletDeps helper);
  # arrays use the core zero-copy bridge.
  option(PECLET_VORO_BUILD_PYTHON "Build the peclet.voro nanobind Python module" OFF)
  if(PECLET_VORO_BUILD_PYTHON)
    peclet_require_nanobind()
    # NOMINSIZE: nanobind's default -Os size optimization is rejected by nvcc ("'s': expected a
    # number") since Kokkos device sources compile as CXX through the launch compiler.
    nanobind_add_module(voro NB_STATIC NOMINSIZE src/voro_bindings.cpp)
    # -> peclet.voro._voro (NB_MODULE(_voro)), re-exported by peclet/voro/__init__.py. The extension +
    # the staged __init__.py land under <build>/peclet/voro so `import peclet.voro` works both from the
    # build tree (dev loop, PYTHONPATH=<build>) and from the SKBUILD wheel install.
    set_target_properties(voro PROPERTIES
      OUTPUT_NAME _voro
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/voro")
    # The package __init__.py is kept as a plain file OUTSIDE any importable peclet/ dir (so an
    # incomplete source package can never shadow the installed one); staged into the build tree here.
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/packaging/voro_init.py"
                   "${CMAKE_CURRENT_BINARY_DIR}/peclet/voro/__init__.py" COPYONLY)
    target_include_directories(voro PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${PECLET_VORO_TPX_INCLUDE}" "${PECLET_VORO_MORTON_INCLUDE}")
    target_compile_definitions(voro PRIVATE MORTON_ENABLE_KOKKOS=1)
    target_link_libraries(voro PRIVATE Kokkos::kokkos)
    target_compile_features(voro PRIVATE cxx_std_20)
    # `pip install .` (scikit-build-core, via pyproject.toml) installs the module as peclet.voro. Inert
    # for a plain `cmake --build`, so the developer workflow is unchanged.
    if(DEFINED SKBUILD)
      install(TARGETS voro LIBRARY DESTINATION peclet/voro COMPONENT python)
      install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/voro_init.py"
              DESTINATION peclet/voro RENAME __init__.py COMPONENT python)
    endif()
    # Python smoke test (Tessellation + Simulation) — needs numpy in the active interpreter.
    if(PECLET_VORO_BUILD_TESTS AND NOT DEFINED SKBUILD)
      find_package(Python3 COMPONENTS Interpreter QUIET)
      if(Python3_Interpreter_FOUND)
        add_test(NAME test_voro_python
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_voro.py)
        set_tests_properties(test_voro_python PROPERTIES
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};OMP_PROC_BIND=false")
      endif()
    endif()
  endif()
endif()

# ── Documentation ─────────────────────────────────────────────────────────────
option(PECLET_VORO_BUILD_DOCS "Build Doxygen documentation" OFF)
if(PECLET_VORO_BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "peclet.voro")
    set(DOXYGEN_PROJECT_BRIEF "Device-native moving-particle Voronoi dynamics")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_RECURSIVE YES)
    # The \mainpage lives in docs/mainpage.dox (the architecture overview); the design notes under
    # docs/ are not part of the API reference. power_cell_solver_spec.md uses LaTeX \dot{...}, which
    # collides with Doxygen's \dot graph command.
    set(DOXYGEN_EXCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/docs/power_cell_solver_spec.md")
    doxygen_add_docs(
      docs
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/docs/mainpage.dox"
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
