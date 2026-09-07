#!/usr/bin/env bash
# check_include_graph.sh — enforce the acyclic layering of include/peclet/voro (voro/CLAUDE.md):
#
#   L0  convex_cell + leaves      convex_cell.hpp plane_policy.hpp tessellation_view.hpp
#                                 topology_store.hpp transpose.hpp verlet_skin.hpp tess_grid.hpp
#   L1  sdf                       sdf.hpp
#   L2  tessellator               tessellator.hpp subset_gather.hpp
#   L3  repair / reeval           repair.hpp reeval_tessellation.hpp dynamic_validate.hpp
#   L4  consumers                 physics/ energy/ fv/ mpi/ mesh_optimizer.hpp ot_optimizer.hpp
#
# A header may include headers of its own layer or of a LOWER layer, never a higher one — so the
# cutter (L0-L2) stays reusable without the physics, and a physics header can never be pulled
# into the engine. Every listed header must exist: a missing file is a FAILURE (the old version
# of this script `continue`d over files that had been renamed away and checked nothing).
#
# Usage: check_include_graph.sh [include/peclet/voro]
set -euo pipefail

INC_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/include/peclet/voro}"

layer_of() {  # header path relative to INC_DIR -> layer number
  case "$1" in
    convex_cell.hpp|plane_policy.hpp|tessellation_view.hpp|topology_store.hpp|transpose.hpp|verlet_skin.hpp|tess_grid.hpp) echo 0 ;;
    sdf.hpp) echo 1 ;;
    tessellator.hpp|subset_gather.hpp) echo 2 ;;
    repair.hpp|reeval_tessellation.hpp|dynamic_validate.hpp) echo 3 ;;
    physics/*|energy/*|fv/*|mpi/*|mesh_optimizer.hpp|ot_optimizer.hpp) echo 4 ;;
    *) echo "" ;;
  esac
}

status=0
n_headers=0
n_edges=0
while IFS= read -r f; do
  rel="${f#"$INC_DIR"/}"
  lf="$(layer_of "$rel")"
  if [ -z "$lf" ]; then
    echo "VIOLATION: '$rel' is not assigned to a layer — add it to layer_of() in $0"
    status=1
    continue
  fi
  n_headers=$((n_headers + 1))
  # Every `#include "peclet/voro/<x>"` directive of this header (not comments / prose).
  while IFS= read -r inc; do
    [ -n "$inc" ] || continue
    n_edges=$((n_edges + 1))
    if [ ! -f "$INC_DIR/$inc" ]; then
      echo "VIOLATION: '$rel' includes '$inc', which does not exist"
      status=1
      continue
    fi
    li="$(layer_of "$inc")"
    if [ -z "$li" ]; then
      echo "VIOLATION: '$rel' includes '$inc', which is not assigned to a layer"
      status=1
    elif [ "$li" -gt "$lf" ]; then
      echo "VIOLATION: '$rel' (layer $lf) includes '$inc' (layer $li) — upward include"
      status=1
    fi
  done < <(sed -nE 's|^[[:space:]]*#[[:space:]]*include[[:space:]]*"peclet/voro/([^"]+)".*|\1|p' "$f")
done < <(find "$INC_DIR" -type f -name '*.hpp' | sort)

if [ "$n_headers" -eq 0 ]; then
  echo "VIOLATION: no headers found under $INC_DIR"
  status=1
fi

# The header set must not have silently shrunk below the engine's known core: these must exist.
for must in convex_cell.hpp sdf.hpp tessellator.hpp repair.hpp tessellation_view.hpp physics/simulation.hpp fv/mesh.hpp; do
  if [ ! -f "$INC_DIR/$must" ]; then
    echo "VIOLATION: expected header '$must' is missing"
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "include-graph OK: $n_headers headers, $n_edges intra-voro includes, all downward or same-layer"
fi
exit "$status"
