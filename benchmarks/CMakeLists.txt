# Benchmarks for voronoi_dynamics
#
# Compares voronoi_dynamics against Voro++ for building a static periodic
# Voronoi tessellation.  All targets in this directory are compiled with
# maximum performance flags (-O3 -march=native -ffast-math -funroll-loops).
#
# Three point-set distributions are exercised:
#   random_uniform  — N uniform-random points in a unit cube
#   cubic_lattice   — simple-cubic lattice (N = n³) with tiny jitter
#   sphere_surface  — N points on a sphere of radius 0.4 inside the unit cube
#
# Build:
#   cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release -DVORONOI_BUILD_BENCHMARKS=ON
#   cmake --build ../build --target benchmark_voronoi -j$(nproc)
#
# Run:
#   ../build/benchmarks/benchmark_voronoi > results/benchmark_results.csv

cmake_minimum_required(VERSION 3.16)

# ── Voro++ (shared with tests/ via FetchContent name deduplication) ─────────
# If tests/ has already declared 'voropp', this declaration is silently
# ignored by CMake; FetchContent_MakeAvailable then reuses the existing target.
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

# Suppress third-party warnings from Voro++ headers
get_target_property(_voropp_inc voro++ INTERFACE_INCLUDE_DIRECTORIES)
if(_voropp_inc)
  set_target_properties(voro++ PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_voropp_inc}"
  )
endif()

# Apply maximum-performance flags to Voro++ itself so both libraries are
# compiled under the same optimisation level.
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

# ── benchmark_voronoi executable ────────────────────────────────────────────
add_executable(benchmark_voronoi benchmark_voronoi.cpp)

target_link_libraries(benchmark_voronoi
  PRIVATE
    voronoi_dynamics::voronoi_dynamics
    voro++
)

# Forward the data directory (mirrors the test helper, unused here but kept
# for consistency).
target_compile_definitions(benchmark_voronoi
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

# Maximum-performance compile flags.
# -O3              : enable all O2 optimisations + additional vectorisation
# -march=native    : tune for the current CPU (may enable AVX/AVX-512)
# -mtune=native    : micro-architectural scheduling tuned to this CPU
# -ffast-math      : allow reassociation, NaN/Inf elision, fused multiply-add
#                    (disable if strict IEEE compliance is required)
# -funroll-loops   : unroll small loops for throughput on the hot path
# -fno-omit-frame-pointer : preserve stack frames so perf/gprof can attribute
#                           timing to the correct call sites
target_compile_options(benchmark_voronoi
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
    >
)

# Disable unused-result warning that some compilers raise for voro++ internals
target_compile_options(benchmark_voronoi
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-unused-result>
)

# ── Results directory ────────────────────────────────────────────────────────
# Created at configure time so the user can run the benchmark from the build
# directory and the CSV lands in a predictable place.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/results")

message(STATUS "benchmark_voronoi: run from ${CMAKE_BINARY_DIR}/benchmarks")
message(STATUS "  ./benchmark_voronoi > results/benchmark_results.csv")

# ── benchmark_tess_single_thread executable ─────────────────────────────────
# Intended for a build configured with:
#   -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE
# so that voronoi_dynamics runs with OpenMP pragmas ignored.
add_executable(benchmark_tess_single_thread benchmark_tess_single_thread.cpp)

target_link_libraries(benchmark_tess_single_thread
  PRIVATE
    voronoi_dynamics::voronoi_dynamics
    voro++
)

target_compile_definitions(benchmark_tess_single_thread
  PRIVATE
    VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    NDEBUG
)

target_compile_options(benchmark_tess_single_thread
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
    >
)

message(STATUS "benchmark_tess_single_thread: run from ${CMAKE_BINARY_DIR}/benchmarks")
message(STATUS "  ./benchmark_tess_single_thread > results/benchmark_tess_single_thread.csv")

# ── benchmark_cell_creation executable ───────────────────────────────────────
# Cell-creation only benchmark, excluding sphere case.
# Default: vd_tess vs voropp_compute_cell. Use --vd-only for thread scaling.
add_executable(benchmark_cell_creation benchmark_cell_creation.cpp)

target_link_libraries(benchmark_cell_creation
  PRIVATE
    voronoi_dynamics::voronoi_dynamics
    voro++
)

target_compile_definitions(benchmark_cell_creation
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
    >
)

message(STATUS "benchmark_cell_creation: run from ${CMAKE_BINARY_DIR}/benchmarks")
message(STATUS "  ./benchmark_cell_creation > results/benchmark_cell_creation.csv")

# ── benchmark_cell_arena_blocks executable ───────────────────────────────────
# Focused N=1e4 random benchmark for CellArena reserve-capacity sweeps.
add_executable(benchmark_cell_arena_blocks benchmark_cell_arena_blocks.cpp)

target_link_libraries(benchmark_cell_arena_blocks
  PRIVATE
    voronoi_dynamics::voronoi_dynamics
)

target_compile_definitions(benchmark_cell_arena_blocks
  PRIVATE
    NDEBUG
)

target_compile_options(benchmark_cell_arena_blocks
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -O3
      -march=native
      -mtune=native
      -ffast-math
      -funroll-loops
      -fno-omit-frame-pointer
    >
)

message(STATUS "benchmark_cell_arena_blocks: run from ${CMAKE_BINARY_DIR}/benchmarks")
message(STATUS "  ./benchmark_cell_arena_blocks --n 10000 --reps 12")
