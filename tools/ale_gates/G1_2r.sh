#!/usr/bin/env bash
# Gate G1.2r of the moving-grid design note (docs/ale_design.md §4.1-§4.4, §4.9, §11 A7, A13; work order S1.2a,
# criteria R0-R7 fixed 2026-11-17 in the brief ALE_S1_2A): the host reference of the time step for one phase on
# a static mesh (tests/python/ale_reference_stepper.py).
#
#   tools/ale_gates/G1_2r.sh <host build dir> [--jobs J] [--cases "c1 c2 ..."] [--no-runs]
#
#   export   test_ale_frame --export <build>/gate_logs/S1_2a.export --cells n --mesh M1|M2, n = 8, 16, 32, one
#            thread (Z = M1: jitter ±0.3h, zero weights; W = M2: the same seeds, weights uniform in [0, 0.15h²])
#   pytest   tests/python/test_ale_stepper_reference.py (operators; R0-R2 at 8^3)
#   runs     one process per case (`ale_reference_stepper.py cases`), one thread each, at most J at once
#            (default 4), longest first; log <build>/gate_logs/S1_2a.<case>.log, last line `RESULT case=...`
#   judge    ale_reference_stepper.py judge: one line per criterion R0-R7
# --no-runs judges the logs already present. PID of this script in <build>/gate_logs/S1_2a.pid.
# Prints one line per criterion, then `G1.2r PASS|FAIL`; exit 1 on FAIL.
set -u
B="$(cd "${1:?usage: G1_2r.sh <host build dir> [--jobs J] [--cases LIST] [--no-runs]}" && pwd)"
shift
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
JOBS=4
CASES=""
RUNS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift ;;
    --cases) CASES="$2"; shift ;;
    --no-runs) RUNS=0 ;;
  esac
  shift
done
L="$B/gate_logs"
mkdir -p "$L"
echo $$ > "$L/S1_2a.pid"
cd "$SRC"
if [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
fi
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=tests/python/ale_reference_stepper.py
E="$L/S1_2a.export"
fail=0

if [[ $RUNS -eq 1 ]]; then
  if ! cmake --build "$B" -j8 --target test_ale_frame > "$L/S1_2a.build.log" 2>&1; then
    echo "G1.2r FAIL build (log $L/S1_2a.build.log)"
    exit 1
  fi
  rm -rf "$E" && mkdir -p "$E"
  : > "$L/S1_2a.export.log"
  for n in 8 16 32; do
    for mesh in M1 M2; do
      if ! "$B/tests/kokkos/test_ale_frame" --export "$E" --cells "$n" --mesh "$mesh" >> "$L/S1_2a.export.log" 2>&1
      then
        echo "G1.2r FAIL export $mesh $n (log $L/S1_2a.export.log)"
        exit 1
      fi
    done
  done
  ALE_EXPORT_DIR="$E" python -m pytest -q tests/python/test_ale_stepper_reference.py > "$L/S1_2a.pytest.log" 2>&1
  prc=$?
  if [[ -z "$CASES" ]]; then
    # longest first: the 32^3 runs and the 2000-step run, then 16^3, then 8^3
    CASES="tg_Z32_F tg_Z32_R abc_Z32_F tg_W32_F abc_W32_F r3_tg_W16_F tg_Z16_F tg_Z16_R tg_W16_F tg_W16_R
           abc_Z16_F abc_W16_F tg_Z16_F_rtol14 tg_Z16_R_rtol14 tg_W16_F_rtol14 tg_W16_R_rtol14
           r6i_W16 r6iii_W16 r6ii_rest_W16 r6ii_tg_W16 tg_Z8_F tg_W8_F abc_Z8_F abc_W8_F
           r0_zero_F r0_zero_R r0_unif_F r0_unif_R"
  fi
  run_one() {
    python "$PY" run "$E" "$1" --log "$L/S1_2a.$1.log" > /dev/null 2> "$L/S1_2a.$1.err" \
      || echo "RESULT case=$1 status=CRASH" >> "$L/S1_2a.$1.log"
  }
  export -f run_one
  export PY E L
  # shellcheck disable=SC2086
  printf '%s\n' $CASES | xargs -P "$JOBS" -I{} bash -c 'run_one "$1"' _ {}
else
  prc=0
  [[ -f "$L/S1_2a.pytest.log" ]] && ! grep -qE '[0-9]+ passed' "$L/S1_2a.pytest.log" && prc=1
fi
pline=$(tail -n 1 "$L/S1_2a.pytest.log" 2>/dev/null)
if [[ $prc -eq 0 ]]; then
  echo "pytest PASS ($pline; log $L/S1_2a.pytest.log)"
else
  echo "pytest FAIL ($pline; log $L/S1_2a.pytest.log)"
  fail=1
fi
python "$PY" judge "$L" > "$L/S1_2a.judge.log" 2>&1
jrc=$?
cat "$L/S1_2a.judge.log"
if [[ $jrc -ne 0 || $fail -ne 0 ]]; then
  [[ $fail -ne 0 ]] && grep -q '^G1.2r PASS' "$L/S1_2a.judge.log" && echo "G1.2r FAIL (pytest)"
  exit 1
fi
exit 0
