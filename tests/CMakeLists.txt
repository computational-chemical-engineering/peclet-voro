# Test executables for vorflow
# Each test is a standalone program that exercises a specific subsystem.

# ── Helper function ───────────────────────────────────────────────────────────
# Add a test executable, link it against the library, and register it with
# CTest.  The DATA_DIR variable is forwarded as a compile definition so that
# tests can locate the sample data files regardless of where they are run from.
function(add_voronoi_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE vorflow::vorflow)
  target_compile_definitions(${name}
    PRIVATE
      VORONOI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
  )
  add_test(NAME ${name} COMMAND ${name}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/data"
  )
endfunction()

# ── Test programs ─────────────────────────────────────────────────────────────

# Static Voronoi tessellation – verifies volume conservation for random point
# sets; does NOT need the sample data file.
add_voronoi_test(test_static_voronoi       test_static_voronoi.cpp)

# Geometric-invariant oracle (suite migration §5): space-filling, positive
# volumes, min facets, facet reciprocity, area negation, area closure, Euler.
# The cross-platform/cross-rank truth the device port is validated against.
add_voronoi_test(test_invariants           test_invariants.cpp)

# Golden-master capture (suite migration §5.6): order-invariant summaries on a
# frozen seed set, committed under tests/golden/. Device ports diff against this.
add_voronoi_test(test_golden               test_golden.cpp)
target_compile_definitions(test_golden
  PRIVATE VORONOI_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")

# Include-graph dependency check (suite migration §1): the tessellation core must
# never include a physics header. Pure-script ctest, no build needed.
add_test(NAME test_include_graph
  COMMAND ${CMAKE_COMMAND} -E env bash
          "${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_include_graph.sh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../include/vorflow")

# Incremental update under periodic random displacements, validated against a rebuild
add_voronoi_test(test_build_rebuild        test_build_rebuild.cpp)

# Particle insertion/deletion/renumbering against sparse and dense rebuilds
add_voronoi_test(test_particle_lifecycle   test_particle_lifecycle.cpp)

# Alternative build trials and stress testing
add_voronoi_test(test_build_trial          test_build_trial.cpp)

# Moving random points with periodic wrapping; report repair/convexity stats
# and validate the final incremental tessellation against a clean rebuild.
add_voronoi_test(test_moving_points_repair_stats test_moving_points_repair_stats.cpp)

# Power-cell regression tests: zero-weight equivalence, empty-cell handling,
# and inactive-particle reactivation.
add_voronoi_test(test_power_cells test_power_cells.cpp)

# SDF boundary clipping and boundary-driven reactivation.
add_voronoi_test(test_sdf_boundary test_sdf_boundary.cpp)

# Force tiny CellMaker scratch arenas to grow and verify growth is handled.
add_voronoi_test(test_cell_maker_growth test_cell_maker_growth.cpp)

# Compressible Euler equations (needs pos_eq_10000.dat)
add_voronoi_test(test_simulation           test_simulation.cpp)

# Sparse-cell simulation step after logical particle deletion
add_voronoi_test(test_simulation_sparse_particles test_simulation_sparse_particles.cpp)

# Droplet dynamics in viscous flow (needs pos_eq_10000.dat)
add_voronoi_test(test_droplet              test_droplet.cpp)

# Interface dynamics with surface tension (needs pos_eq_10000.dat)
add_voronoi_test(test_interface            test_interface.cpp)

# Viscous Navier-Stokes flow (needs pos_eq_10000.dat)
add_voronoi_test(test_viscous              test_viscous.cpp)

# ── Voro++ comparison test (fetched automatically) ───────────────────────────
# Voro++ is downloaded and compiled at configure time via CMake FetchContent.
# No manual installation is required; the test is always part of the suite.
include(FetchContent)

FetchContent_Declare(
  voropp
  GIT_REPOSITORY https://github.com/chr1shr/voro.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
)

# Disable Voro++ components that are not needed for the comparison test so
# that only the library itself is compiled.
set(VORO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_CMD_LINE OFF CACHE BOOL "" FORCE)
set(VORO_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(voropp)

# Suppress third-party compiler warnings originating from Voro++ headers.
get_target_property(_voropp_inc voro++ INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(voro++ PROPERTIES
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_voropp_inc}"
)

# Register the comparison test – uses the same helper as all other tests.
add_voronoi_test(test_voro_comparison test_voro_comparison.cpp)
target_link_libraries(test_voro_comparison PRIVATE voro++)
