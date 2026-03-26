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

# ── Optional Voro++ comparison test ──────────────────────────────────────────
option(VORONOI_BUILD_VORO_COMPARISON "Build Voro++ comparison test (requires Voro++ library)" OFF)
if(VORONOI_BUILD_VORO_COMPARISON)
  find_library(VORO_LIB voro++ HINTS voro/src)
  find_path(VORO_INC voro++.hh HINTS voro/src)
  if(VORO_LIB AND VORO_INC)
    add_voronoi_test(test_voro_comparison test_voro_comparison.cpp)
    target_include_directories(test_voro_comparison PRIVATE ${VORO_INC})
    target_link_libraries(test_voro_comparison PRIVATE ${VORO_LIB})
    message(STATUS "Voro++ comparison test enabled")
  else()
    message(WARNING "Voro++ library not found; skipping test_voro_comparison")
  endif()
endif()
