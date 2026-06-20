# Benchmarks for vorflow
#
# Active benchmark workflow:
#   - benchmark_cell_creation.cpp
#   - report_cell_creation_threads.py
#
# Older benchmark sources, reports, and raw result sets have been moved to
# benchmarks/archive_pending/ and build_bench/benchmarks/results/archive_pending/
# for later deletion if no longer needed.

cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  voropp
  GIT_REPOSITORY https://github.com/chr1shr/voro.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
)

set(VORO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_CMD_LINE OFF CACHE BOOL "" FORCE)
set(VORO_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(voropp)

get_target_property(_voropp_inc voro++ INTERFACE_INCLUDE_DIRECTORIES)
if(_voropp_inc)
  set_target_properties(voro++ PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_voropp_inc}"
  )
endif()

target_compile_options(voro++ PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
  >
)
target_compile_definitions(voro++ PRIVATE NDEBUG)

add_executable(benchmark_cell_creation benchmark_cell_creation.cpp)
add_executable(benchmark_full_build benchmark_full_build.cpp)
add_executable(benchmark_cell_creation_targeted benchmark_cell_creation_targeted.cpp)
add_executable(benchmark_nbrlist_setup_diagnostic benchmark_nbrlist_setup_diagnostic.cpp)
add_executable(benchmark_update_fast_study benchmark_update_fast_study.cpp)
add_executable(benchmark_update_strategy_study benchmark_update_strategy_study.cpp)
add_executable(benchmark_update_thread_scaling benchmark_update_thread_scaling.cpp)

target_link_libraries(benchmark_cell_creation
  PRIVATE
    vorflow::vorflow
    voro++
)

target_link_libraries(benchmark_full_build
  PRIVATE
    vorflow::vorflow
    voro++
)

target_link_libraries(benchmark_cell_creation_targeted
  PRIVATE
    vorflow::vorflow
    voro++
)

target_link_libraries(benchmark_nbrlist_setup_diagnostic
  PRIVATE
    vorflow::vorflow
    voro++
)

target_link_libraries(benchmark_update_fast_study
  PRIVATE
    vorflow::vorflow
    voro++
)

target_link_libraries(benchmark_update_strategy_study
  PRIVATE
    vorflow::vorflow
    voro++
)

target_link_libraries(benchmark_update_thread_scaling
  PRIVATE
    vorflow::vorflow
    voro++
)

target_compile_definitions(benchmark_cell_creation
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_definitions(benchmark_full_build
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_definitions(benchmark_cell_creation_targeted
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_definitions(benchmark_nbrlist_setup_diagnostic
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_definitions(benchmark_update_fast_study
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_definitions(benchmark_update_strategy_study
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_definitions(benchmark_update_thread_scaling
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_options(benchmark_cell_creation
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
      -Wno-unused-result
    >
)

target_compile_options(benchmark_full_build
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
      -Wno-unused-result
    >
)

target_compile_options(benchmark_cell_creation_targeted
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
      -Wno-unused-result
    >
)

target_compile_options(benchmark_nbrlist_setup_diagnostic
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
      -Wno-unused-result
    >
)

target_compile_options(benchmark_update_fast_study
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
      -Wno-unused-result
    >
)

target_compile_options(benchmark_update_strategy_study
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
      -Wno-unused-result
    >
)

target_compile_options(benchmark_update_thread_scaling
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
      -Wno-unused-result
    >
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/results")

message(STATUS "benchmark_cell_creation: run from ${CMAKE_BINARY_DIR}/benchmarks")
message(STATUS "  ./benchmark_cell_creation > results/benchmark_cell_creation.csv")
