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
VIEW_ONLY=(physics/euler_pressure.hpp physics/viscous.hpp physics/interface.hpp physics/simulation.hpp)
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

# De-legacy invariant: the PRODUCTION device path (shipped library + device Python
# bindings) must not include the legacy engine (voronoi.hpp / simulation.hpp). The
# legacy engine is retained only as a TEST ORACLE and as the legacy<->view bridge
# tessellation_build.hpp, until the oracle is converted to golden data and deleted.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for g in "$INC_DIR/device" "$INC_DIR/physics" "$INC_DIR/host" \
         "$INC_DIR/tessellation_view.hpp" "$ROOT/src"; do
  [ -e "$g" ] || continue
  while IFS= read -r f; do
    # Anchor to the exact legacy paths so simulation.hpp is not a false hit.
    if grep -Eq "^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"]vorflow/(voronoi|simulation)\.hpp" "$f"; then
      echo "VIOLATION: production file '${f#"$ROOT"/}' includes the legacy engine"
      status=1
    fi
  done < <(find "$g" -type f \( -name '*.hpp' -o -name '*.cpp' \) 2>/dev/null)
done
if [ "$status" -eq 0 ]; then
  echo "include-graph OK: the production device path is legacy-free"
fi
exit "$status"
