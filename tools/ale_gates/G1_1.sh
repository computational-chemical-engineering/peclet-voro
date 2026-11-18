#!/usr/bin/env bash
# Gate G1.1 of the moving-grid design note (docs/ale_design.md §7 WO-S1.1, §11 A7, A9; criteria K0-K8 fixed
# 2026-11-15 in the brief ALE_S1_1): the bulk operators of include/peclet/voro/ale/operators.hpp.
#
#   tools/ale_gates/G1_1.sh <host build dir> [--cuda-dir DIR] [--no-cuda] [--no-battery]
#
#   K0, K1  export M1-M3 (test_ale_frame --export, one thread) -> tests/python/test_ale_operators.py inputs (K0, the
#           reference checked first) -> test_ale_operators --apply (8 threads) -> test_ale_operators.py compare (K1)
#   K2-K6   test_ale_operators (the ctest binary, 8 threads)
#   K7      --apply M2 at OMP_NUM_THREADS 1, 8, 8: SHA-256 of every output file equal
#   base    every ctest of the host tree passes (ctest -LE bench, 4 threads; count printed) and python/state_hash.py
#           at one thread equals gate_logs/G0_1.state_hash
#   K8      CUDA tree (default <src>/build_s1_cuda, prefix ../extern/install/nvidia-cuda; only test_ale_frame and
#           test_ale_operators): build (no lock); then, under `flock -w 1800 <heavy.lock>`: both ctests on the GPU,
#           two GPU exports bitwise equal, test_ale_reference.py on the GPU export (the G0.3' line), K1 on the GPU's
#           outputs against the reference on its own export. Lock not obtained in 30 min: run part NOT RUN.
#   timing  test_ale_operators --time 32 (8 threads; and on the GPU), reported without threshold.
# Logs in <host build dir>/gate_logs/G1_1.*. Prints ONE line `G1.1 PASS|FAIL K0=... K8=...`; exit 1 on FAIL.
set -u
B="$(cd "${1:?usage: G1_1.sh <host build dir> [--cuda-dir DIR] [--no-cuda] [--no-battery]}" && pwd)"
shift
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
C="$SRC/build_s1_cuda"
LOCK=/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock
DO_CUDA=1
DO_BATTERY=1
GPU_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-dir) C="$2"; shift ;;
    --no-cuda) DO_CUDA=0 ;;
    --no-battery) DO_BATTERY=0 ;;
    --gpu-run) GPU_RUN=1 ;;   # internal: the GPU part, run under the lock
  esac
  shift
done
L="$B/gate_logs"
mkdir -p "$L"
cd "$SRC"
if [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
fi
export OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYT=tests/python/test_ale_operators.py

# ---------------------------------------------------------------- the GPU part (invoked under the lock)
if [[ $GPU_RUN -eq 1 ]]; then
  export OMP_NUM_THREADS=8
  r="ctest=FAIL"
  if ctest --test-dir "$C" -R '^test_ale_(frame|operators)$' --output-on-failure > "$L/G1_1.gpu_ctest.log" 2>&1; then
    r="ctest=PASS"
  fi
  r="$r($(grep -cE 'Test +#[0-9]+: .* Passed' "$L/G1_1.gpu_ctest.log")/2)"
  for t in a b; do
    rm -rf "$L/G1_1.gpu_export.$t" && mkdir -p "$L/G1_1.gpu_export.$t"
    "$C/tests/kokkos/test_ale_frame" --export "$L/G1_1.gpu_export.$t" > "$L/G1_1.gpu_export.$t.log" 2>&1
  done
  if diff -rq "$L/G1_1.gpu_export.a" "$L/G1_1.gpu_export.b" > "$L/G1_1.gpu_export.diff" 2>&1 \
      && grep -q '=== S0.3 export PASS ===' "$L/G1_1.gpu_export.a.log"; then
    r="$r,export=bitwise"
  else
    r="$r,export=FAIL(see G1_1.gpu_export.diff)"
  fi
  python tests/python/test_ale_reference.py "$L/G1_1.gpu_export.a" > "$L/G1_1.gpu_g03.log" 2>&1
  g03=$(grep "^G0.3' " "$L/G1_1.gpu_g03.log" | awk '{print $2}')
  r="$r,G0.3'=${g03:-FAIL}"
  python "$PYT" inputs "$L/G1_1.gpu_export.a" > "$L/G1_1.gpu_k0.log" 2>&1
  "$C/tests/kokkos/test_ale_operators" --apply "$L/G1_1.gpu_export.a" > "$L/G1_1.gpu_apply.log" 2>&1
  grep -q '=== S1.1 apply PASS ===' "$L/G1_1.gpu_apply.log" || r="$r,apply=FAIL($(grep -m1 FAIL "$L/G1_1.gpu_apply.log"))"
  python "$PYT" compare "$L/G1_1.gpu_export.a" "$L/G1_1.gpu_export.a" > "$L/G1_1.gpu_k1.log" 2>&1
  k1=$(grep -E '^K1 (PASS|FAIL)' "$L/G1_1.gpu_k1.log" | awk '{print $2"("$3")"}')
  r="$r,K1=${k1:-FAIL}"
  "$C/tests/kokkos/test_ale_operators" --time 32 > "$L/G1_1.time.gpu.log" 2>&1
  echo "GPURESULT $r"
  exit 0
fi

fail=0
judge() { [[ "$1" == PASS* ]] || fail=1; }

# ---------------------------------------------------------------- host build
if ! cmake --build "$B" -j8 > "$L/G1_1.build.log" 2>&1; then
  echo "G1.1 FAIL build (log $L/G1_1.build.log)"
  exit 1
fi
T="$B/tests/kokkos/test_ale_operators"

# ---------------------------------------------------------------- K0, K1
D="$L/G1_1.export"
rm -rf "$D" && mkdir -p "$D"
if ! OMP_NUM_THREADS=1 "$B/tests/kokkos/test_ale_frame" --export "$D" > "$L/G1_1.export.log" 2>&1; then
  echo "G1.1 FAIL export (log $L/G1_1.export.log)"
  exit 1
fi
OMP_NUM_THREADS=1 python "$PYT" inputs "$D" > "$L/G1_1.k0.log" 2>&1
k0=$(grep -E '^K0 (PASS|FAIL)' "$L/G1_1.k0.log" | sed -E 's/^K0 ([A-Z]+) /\1(/; s/ /,/g; s/$/)/')
k0=${k0:-FAIL(nofinish)}
judge "$k0"
for t in t8a t1 t8b; do mkdir -p "$L/G1_1.out.$t"; done
OMP_NUM_THREADS=8 "$T" --apply "$D" --out "$L/G1_1.out.t8a" > "$L/G1_1.apply.t8a.log" 2>&1
OMP_NUM_THREADS=1 python "$PYT" compare "$D" "$L/G1_1.out.t8a" > "$L/G1_1.k1.log" 2>&1
k1=$(grep -E '^K1 (PASS|FAIL)' "$L/G1_1.k1.log" | awk '{print $2"("$3","$4")"}')
if ! grep -q '=== S1.1 apply PASS ===' "$L/G1_1.apply.t8a.log"; then k1="FAIL(apply;$(grep -m1 FAIL "$L/G1_1.apply.t8a.log"))"; fi
k1=${k1:-FAIL(nofinish)}
judge "$k1"

# ---------------------------------------------------------------- K2-K6
OMP_NUM_THREADS=8 "$T" > "$L/G1_1.ident.log" 2>&1
declare -A K
for k in K2 K3 K4 K5 K6; do
  v=$(grep -E "^$k (PASS|FAIL) " "$L/G1_1.ident.log" | sed -E "s/^$k ([A-Z]+) /\1(/; s/ /,/g; s/\$/)/")
  K[$k]=${v:-FAIL(nofinish)}
  judge "${K[$k]}"
done

# ---------------------------------------------------------------- K7
OMP_NUM_THREADS=1 "$T" --apply "$D" --mesh M2 --out "$L/G1_1.out.t1" > "$L/G1_1.apply.t1.log" 2>&1
OMP_NUM_THREADS=8 "$T" --apply "$D" --mesh M2 --out "$L/G1_1.out.t8b" > "$L/G1_1.apply.t8b.log" 2>&1
: > "$L/G1_1.k7.sha"
for t in t1 t8a t8b; do
  (cd "$L/G1_1.out.$t" && sha256sum M2.out_* > "../G1_1.k7.sha.$t")
  cat "$L/G1_1.k7.sha.$t" >> "$L/G1_1.k7.sha"
done
nOut=$(wc -l < "$L/G1_1.k7.sha.t1")
if [[ $nOut -ge 17 ]] && cmp -s "$L/G1_1.k7.sha.t1" "$L/G1_1.k7.sha.t8a" && cmp -s "$L/G1_1.k7.sha.t8a" "$L/G1_1.k7.sha.t8b"; then
  k7="PASS(files=$nOut,sha:t1=t8a=t8b)"
else
  k7="FAIL(files=$nOut;see G1_1.k7.sha)"
fi
judge "$k7"

# ---------------------------------------------------------------- base: all ctests, state hash
base="SKIPPED"
if [[ $DO_BATTERY -eq 1 ]]; then
  OMP_NUM_THREADS=4 ctest --test-dir "$B" -LE bench > "$L/G1_1.ctest.log" 2>&1
  n_all=$(grep -cE 'Test +#[0-9]+: ' "$L/G1_1.ctest.log")
  n_pass=$(grep -cE 'Test +#[0-9]+: .* Passed' "$L/G1_1.ctest.log")
  OMP_NUM_THREADS=1 PYTHONPATH="$B" python python/state_hash.py > "$L/G1_1.state_hash" 2> "$L/G1_1.state_hash.err"
  hash="differs"
  if [[ -s "$L/G1_1.state_hash" ]] && cmp -s "$L/G0_1.state_hash" "$L/G1_1.state_hash"; then hash="same"; fi
  if [[ $n_all -eq 47 && $n_pass -eq 47 && $hash == same ]]; then
    base="PASS(ctest=$n_pass/$n_all,hash=$hash)"
  else
    base="FAIL(ctest=$n_pass/$n_all,hash=$hash)"
  fi
  judge "$base"
fi

# ---------------------------------------------------------------- timing (host)
OMP_NUM_THREADS=8 "$T" --time 32 > "$L/G1_1.time.host.log" 2>&1

# ---------------------------------------------------------------- K8
k8="SKIPPED"
if [[ $DO_CUDA -eq 1 ]]; then
  export PATH=/usr/local/cuda/bin:$PATH
  # (re)configure every time: a tree configured before a test was added does not know its target
  cmake -B "$C" -DCMAKE_BUILD_TYPE=Release -DPECLET_VORO_KOKKOS=ON -DPECLET_VORO_BUILD_PYTHON=ON \
    -DPECLET_VORO_MPI=ON -DPECLET_VORO_BUILD_TESTS=ON "-DMPIEXEC_PREFLAGS=--bind-to;none" \
    -DCMAKE_PREFIX_PATH="$SRC/../extern/install/nvidia-cuda" > "$L/G1_1.cuda_configure.log" 2>&1
  if cmake --build "$C" -j8 --target test_ale_frame test_ale_operators > "$L/G1_1.cuda_build.log" 2>&1; then
    nwarn=$(grep -c 'warning' "$L/G1_1.cuda_build.log")
    flock -w 1800 "$LOCK" bash "$0" "$B" --cuda-dir "$C" --gpu-run > "$L/G1_1.gpu.log" 2>&1
    g=$(grep '^GPURESULT ' "$L/G1_1.gpu.log" | cut -d' ' -f2-)
    if [[ -z "$g" ]]; then
      k8="FAIL(build=ok,warnings=$nwarn,run=NOT_RUN:lock not obtained in 30 min or run aborted, log G1_1.gpu.log)"
    elif [[ "$g" != *FAIL* ]]; then
      k8="PASS(build=ok,warnings=$nwarn,$g)"
    else
      k8="FAIL(build=ok,warnings=$nwarn,$g)"
    fi
  else
    k8="FAIL(build;log G1_1.cuda_build.log)"
  fi
  judge "$k8"
fi

verdict=PASS
[[ $fail -eq 0 ]] || verdict=FAIL
echo "G1.1 $verdict K0=$k0 K1=$k1 K2=${K[K2]} K3=${K[K3]} K4=${K[K4]} K5=${K[K5]} K6=${K[K6]} K7=$k7 K8=$k8 base=$base (logs $L/G1_1.*)"
[[ $fail -eq 0 ]]
