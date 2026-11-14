#!/usr/bin/env bash
# Gate G0.3 of the moving-grid design note (docs/ale_design.md §7 WO-S0.3; criteria fixed 2026-11-15 in the
# brief ALE_S0_3): the host-only Python reference of the bulk operators of §A.3 (tests/python/ale_reference.py)
# on three exported 8^3 power meshes M1, M2, M3.
#
#   tools/ale_gates/G0_3.sh <build dir>
#
# Builds test_ale_frame (-j8), exports the meshes (`test_ale_frame --export`, one thread), then runs
# tests/python/test_ale_reference.py (R0-R4, R5', R6'; criteria in its docstring). Logs in <build dir>/gate_logs/:
#   G0_3.build.log, G0_3.export.log (+ the arrays in G0_3.export/), G0_3.py.log (one line per check and mesh).
# Prints ONE line `G0.3' PASS|FAIL R0=... R6'=...`; exit 1 on FAIL.
set -u
B="$(cd "${1:?usage: G0_3.sh <build dir>}" && pwd)"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
L="$B/gate_logs"
mkdir -p "$L"
cd "$SRC"
if [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
fi
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if ! cmake --build "$B" -j8 --target test_ale_frame > "$L/G0_3.build.log" 2>&1; then
  echo "G0.3' FAIL build (log $L/G0_3.build.log)"
  exit 1
fi
D="$L/G0_3.export"
rm -rf "$D" && mkdir -p "$D"
if ! "$B/tests/kokkos/test_ale_frame" --export "$D" > "$L/G0_3.export.log" 2>&1; then
  echo "G0.3' FAIL export (log $L/G0_3.export.log)"
  exit 1
fi
python tests/python/test_ale_reference.py "$D" > "$L/G0_3.py.log" 2>&1
rc=$?
line=$(grep "^G0.3' " "$L/G0_3.py.log")
if [[ -z "$line" ]]; then
  echo "G0.3' FAIL reference did not finish (log $L/G0_3.py.log)"
  exit 1
fi
echo "$line"
exit $rc
