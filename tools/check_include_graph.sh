#!/usr/bin/env bash
# check_include_graph.sh — enforce the migration's one-way dependency rule (plan §1):
#   physics -> TessellationView -> engine -> core
# The tessellation *core* must never include a *physics* header, so the cutter
# stays reusable for non-physics consumers (packing/microstructure/meshing).
#
# Core headers may include each other and transport-core (tpx/...); physics
# headers may include core. A core header that includes a physics header fails.
#
# Usage: check_include_graph.sh [include/vorflow]
set -euo pipefail

INC_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/include/vorflow}"

# Physics (downstream) headers — extend as physics modules are added.
PHYSICS_HEADERS=(simulation.hpp)

# Core/engine headers — everything else under include/vorflow that is not physics.
CORE_HEADERS=(vor_types.hpp nbrlist.hpp voronoi.hpp tessellation_view.hpp device/cell_cutter.hpp device/sdf.hpp device/tessellator.hpp)

status=0
for core in "${CORE_HEADERS[@]}"; do
  f="$INC_DIR/$core"
  [ -f "$f" ] || continue  # not all exist yet (added across phases)
  for phys in "${PHYSICS_HEADERS[@]}"; do
    # Match an actual #include directive (not a comment/word in prose).
    if grep -Eq "^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"][^>\"]*${phys}" "$f"; then
      echo "VIOLATION: core header '$core' includes physics header '$phys'"
      status=1
    fi
  done
done

if [ "$status" -eq 0 ]; then
  echo "include-graph OK: no physics header is included by a core header"
fi

# Reference physics modules must compile against the published view ONLY — they
# must not reach into the legacy engine (voronoi.hpp) or other physics
# (simulation.hpp). This is the plan §4 "compiles against TessellationView only".
VIEW_ONLY=(physics/euler_pressure.hpp physics/viscous.hpp physics/device_simulation.hpp)
FORBIDDEN=(voronoi.hpp simulation.hpp)
for mod in "${VIEW_ONLY[@]}"; do
  f="$INC_DIR/$mod"
  [ -f "$f" ] || continue
  for bad in "${FORBIDDEN[@]}"; do
    if grep -Eq "^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"][^>\"]*${bad}" "$f"; then
      echo "VIOLATION: view-only physics '$mod' includes engine header '$bad'"
      status=1
    fi
  done
done
if [ "$status" -eq 0 ]; then
  echo "include-graph OK: view-only physics modules depend on the view alone"
fi
exit "$status"
