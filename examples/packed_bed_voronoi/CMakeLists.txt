# Standalone build for the packed-bed interstitial-Voronoi example.
#   cmake -B build -DCMAKE_PREFIX_PATH="<repo>/extern/install/host-openmp" -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
cmake_minimum_required(VERSION 3.20)
project(packed_bed_voronoi CXX)

find_package(Kokkos REQUIRED)

set(_pbv_inc
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include"        # peclet/voro
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../core/include" # peclet/core
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../morton/include")

add_executable(sdf_voronoi_vtu sdf_voronoi_vtu.cpp)
target_link_libraries(sdf_voronoi_vtu PRIVATE Kokkos::kokkos)
target_include_directories(sdf_voronoi_vtu PRIVATE ${_pbv_inc})
target_compile_definitions(sdf_voronoi_vtu PRIVATE MORTON_ENABLE_KOKKOS=1)
target_compile_features(sdf_voronoi_vtu PRIVATE cxx_std_20)

# Four-stage pore-mesh study (random / equalized / graded-seed / graded-relaxed) → VTUs.
add_executable(pore_mesh_stages pore_mesh_stages.cpp)
target_link_libraries(pore_mesh_stages PRIVATE Kokkos::kokkos)
target_include_directories(pore_mesh_stages PRIVATE ${_pbv_inc})
target_compile_definitions(pore_mesh_stages PRIVATE MORTON_ENABLE_KOKKOS=1 PECLET_CORE_HAVE_MORTON=1)
target_compile_features(pore_mesh_stages PRIVATE cxx_std_20)
