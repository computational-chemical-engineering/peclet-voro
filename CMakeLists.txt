cmake_minimum_required(VERSION 3.16)

# One version source per repo (suite/docs/QUALITY_PLAN.md D4): pyproject.toml's `version = "x.y.z"`.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _peclet_voro_version_line
     REGEX "^version = \"[0-9]+\.[0-9]+\.[0-9]+")
string(REGEX REPLACE "^version = \"([0-9]+\.[0-9]+\.[0-9]+).*" "\\1"
       PECLET_VORO_VERSION "${_peclet_voro_version_line}")
if(NOT PECLET_VORO_VERSION MATCHES "^[0-9]+\.[0-9]+\.[0-9]+$")
  message(FATAL_ERROR "voro: could not read `version = \"x.y.z\"` from pyproject.toml")
endif()

project(
  peclet_voro
  VERSION ${PECLET_VORO_VERSION}
  DESCRIPTION "Dynamic Voronoi tessellation of moving particles (Kokkos)"
  LANGUAGES CXX
)

# ── C++ standard ─────────────────────────────────────────────────────────────
# C++20 everywhere (suite/docs/STYLE.md); every device target below also states cxx_std_20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ── Compile options ───────────────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic)

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(OpenMP)

# ── Header-only interface library ────────────────────────────────────────────
# The voro headers are header-only; this target just carries the include path for downstream
# consumers. The device path (tests/kokkos, src/) sets its own include dirs against core +
# morton, so it does not link this target.
add_library(voro_headers INTERFACE)
add_library(peclet::voro ALIAS voro_headers)

target_include_directories(voro_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Tests / benchmarks ────────────────────────────────────────────────────────
# Only the Kokkos device tests exist (tests/kokkos + tests/kokkos_mpi, configured below under
# PECLET_VORO_KOKKOS). OFF by default so a wheel build (`pip install .`) compiles nothing but the
# module; ON in CI and in the CLAUDE.md dev recipe. The MPI suite (tests/kokkos_mpi) joins the same
# tree when PECLET_VORO_MPI=ON, so one tree per backend carries everything.
option(PECLET_VORO_BUILD_TESTS "Build the test programs (ctest)" OFF)
# The bench_* timing instruments (incl. the *_f32 precision variants and the Voro++ throughput
# reference) are NOT tests: opt in. The one bench binary that doubles as a gate
# (bench_dynamic_update --gates) is always built with the tests.
option(PECLET_VORO_BUILD_BENCHMARKS "Build the bench_* timing instruments (ctest label `bench`)" OFF)
if(PECLET_VORO_BUILD_TESTS OR PECLET_VORO_BUILD_BENCHMARKS)
  enable_testing()
endif()

# ── Python bindings ───────────────────────────────────────────────────────────
# The Python surface is the device-native nanobind module `peclet.voro` (src/voro_bindings.cpp),
# configured below with -DPECLET_VORO_BUILD_PYTHON=ON. The legacy pybind11 module over the retired
# half-edge engine has been removed.

# ── Kokkos / suite device build (migration) ───────────────────────────────────
# Suite-aligned build: find_package(Kokkos/ArborX) against the bootstrapped prefix
# (../extern/install/<backend>, built by ../tools/bootstrap_deps.sh), mirroring
# dem/flow. Device kernels are added across the migration phases; Phase 0 wires
# the toolchain and a smoke test. The legacy header-only path above is untouched.
#
#   cmake -S . -B build/host-openmp -DPECLET_VORO_KOKKOS=ON \
#     -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"
#   cmake --build build/host-openmp -j
#
# Add -DPECLET_VORO_MPI=ON to link MPI + core for the distributed path.
option(PECLET_VORO_KOKKOS "Build the Kokkos device path (find_package(Kokkos))" OFF)
option(PECLET_VORO_MPI "Build the distributed path against MPI + core" OFF)
# Redistributable single-GPU CUDA wheel: libcudart is provided by the `nvidia-cuda-runtime` PyPI
# dependency (installed to site-packages/nvidia/cu13/lib), not the system. ON sets the module's RPATH
# to reach that wheel from its install location so `import peclet.voro` works with no system CUDA. Used
# by the peclet-voro-cu13 packaging (packaging/pyproject-cuda.toml + the release workflow's cuda job).
# Mirrors peclet-flow.
option(PECLET_CUDA_RUNTIME_WHEEL "RPATH the module to the nvidia-cuda-runtime wheel's libcudart" OFF)

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
  peclet_sibling_include(peclet-core "${PECLET_CORE_TAG}" "../core" PECLET_CORE_INCLUDE)
  peclet_sibling_include(peclet-morton         "${PECLET_MORTON_TAG}" "../morton"      PECLET_VORO_MORTON_INCLUDE)

  if(PECLET_VORO_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    if(NOT EXISTS "${PECLET_CORE_INCLUDE}/peclet/core/halo/particle_halo.hpp")
      message(FATAL_ERROR
        "core not found at ${PECLET_CORE_INCLUDE} (clone it as a sibling repo)")
    endif()
    message(STATUS "voro: distributed path ENABLED (MPI + core)")
  endif()

  if(PECLET_VORO_BUILD_TESTS OR PECLET_VORO_BUILD_BENCHMARKS)
    add_subdirectory(tests/kokkos)
  endif()
  if(PECLET_VORO_BUILD_TESTS AND PECLET_VORO_MPI)
    add_subdirectory(tests/kokkos_mpi)   # np = 1, 2, 4 ctests, label `mpi` (also a standalone project)
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
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${PECLET_CORE_INCLUDE}" "${PECLET_VORO_MORTON_INCLUDE}")
    target_compile_definitions(voro PRIVATE MORTON_ENABLE_KOKKOS=1 PECLET_CORE_HAVE_MORTON=1)
    target_link_libraries(voro PRIVATE Kokkos::kokkos)
    target_compile_features(voro PRIVATE cxx_std_20)
    # HIP/lld: keep sections so Kokkos SharedAllocationRecord<HIPSpace> vtables aren't gc'd (see flow).
    if(Kokkos_ENABLE_HIP)
      target_link_options(voro PRIVATE -Wl,--no-gc-sections)
  # HIP/lld experiment 1 (docs/RELEASE.md §8): nanobind sets CXX_VISIBILITY_PRESET hidden, and under hipcc
  # the host objects then reference the Kokkos SharedAllocationRecord / shared_ptr control-block vtables
  # as hidden symbols that nothing defines ("undefined hidden symbol: vtable for ..."). Default visibility
  # on the HIP path only; CUDA/OpenMP builds are untouched.
  set_target_properties(voro PROPERTIES CXX_VISIBILITY_PRESET default)
    endif()
    # Distributed surface: compile the VoronoiHalo binding (guarded by PECLET_VORO_MPI) and link MPI +
    # the core particle halo. Default (OFF) leaves the single-rank module untouched.
    if(PECLET_VORO_MPI)
      target_compile_definitions(voro PRIVATE PECLET_VORO_MPI)
      target_link_libraries(voro PRIVATE MPI::MPI_CXX)
    endif()
    # Redistributable CUDA wheel: point the module at the nvidia-cuda-runtime wheel's libcudart via a
    # relative $ORIGIN RPATH (peclet/voro -> site-packages -> nvidia/cu13/lib).
    if(PECLET_CUDA_RUNTIME_WHEEL)
      set_target_properties(voro PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib" INSTALL_RPATH_USE_LINK_PATH OFF)
    endif()
    # `pip install .` (scikit-build-core, via pyproject.toml) installs the module as peclet.voro. Inert
    # for a plain `cmake --build`, so the developer workflow is unchanged.
    if(DEFINED SKBUILD)
      install(TARGETS voro LIBRARY DESTINATION peclet/voro COMPONENT python)
      install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/voro_init.py"
              DESTINATION peclet/voro RENAME __init__.py COMPONENT python)
    endif()
    # Python smoke test (Tessellation + Simulation) on the interpreter the module was built for
    # (Python_EXECUTABLE, from peclet_require_nanobind); needs numpy there.
    if(PECLET_VORO_BUILD_TESTS AND NOT DEFINED SKBUILD)
      add_test(NAME test_voro_python
               COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_voro.py)
      set_tests_properties(test_voro_python PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};OMP_PROC_BIND=false")
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
    # Match the presentation the rest of the family pins (core, flow, pnm, dem).  These have
    # to be set HERE, not in docs/Doxyfile: doxygen_add_docs() writes its own Doxyfile from
    # the DOXYGEN_* variables and never reads that file, so the published site ignored it.
    # HTML_COLORSTYLE especially - its default AUTO_LIGHT follows the reader's browser theme,
    # which is why this page could come out dark while the others stayed light.
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_DISABLE_INDEX NO)
    set(DOXYGEN_FULL_SIDEBAR NO)
    set(DOXYGEN_HTML_COLORSTYLE LIGHT)
    # The README is the landing page, as in core/flow/pnm/dem; the architecture overview that
    # used to be the \mainpage is now a \page of its own (docs/architecture.dox), reachable
    # from the sidebar.  The notes under docs/ (and docs/archive/) are not part of the API
    # reference: the INPUT list below names the files that are.  The EXCLUDE is belt-and-braces
    # for docs/archive/power_cell_solver_spec.md, whose LaTeX \dot{...} collides with Doxygen's
    # \dot graph command.
    set(DOXYGEN_EXCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/docs/archive")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    doxygen_add_docs(
      docs
      "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/docs/architecture.dox"
      COMMENT "Generating Doxygen documentation"
    )
  endif()
endif()

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
  TARGETS voro_headers
  EXPORT voroTargets
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT voroTargets
  FILE voroTargets.cmake
  NAMESPACE peclet::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/voro
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/voroConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/voroConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/voro"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/voroConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/voroConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/voroConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/voro"
)
