# Test executables for voronoi_dynamics
# Each test is a standalone program that exercises a specific subsystem.

# ── Helper function ───────────────────────────────────────────────────────────
# Add a test executable, link it against the library, and register it with
# CTest.  The DATA_DIR variable is forwarded as a compile definition so that
# tests can locate the sample data files regardless of where they are run from.
function(add_voronoi_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE voronoi_dynamics::voronoi_dynamics)
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

# Build and rebuild under Lees-Edwards shear deformation
add_voronoi_test(test_build_rebuild        test_build_rebuild.cpp)

# Alternative build trials and stress testing
add_voronoi_test(test_build_trial          test_build_trial.cpp)

# Compressible Euler equations (needs pos_eq_10000.dat)
add_voronoi_test(test_simulation           test_simulation.cpp)

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
