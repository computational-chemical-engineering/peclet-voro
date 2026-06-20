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
option(VORONOI_BUILD_PYTHON "Build the pybind11 Python module (vordyn)" OFF)
if(VORONOI_BUILD_PYTHON)
  add_subdirectory(python)
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
    doxygen_add_docs(
      docs
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
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
