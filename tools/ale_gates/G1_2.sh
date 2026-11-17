#!/usr/bin/env bash
# Gate G1.2 of the moving-grid design note (docs/ale_design.md §4.1-§4.5, §4.9, §11 A15; work order S1.2b,
# criteria B, L, C0-C3, R0-R6, T fixed 2026-11-17 in the brief ALE_S1_2B): the device time step of
# include/peclet/voro/ale/stepper.hpp (one phase, static mesh) on the host (OpenMP, 8 threads) and the GPU (CUDA).
#
#   tools/ale_gates/G1_2.sh [--configure] [--base] [--judge-only] [--cases "LIST"] [--no-gpu]
#
# Trees (core headers from ../core-ale-amg, the amgpar branch; morton from ../morton; nothing downloaded):
#   build_s2       prefix ../extern/install/host-openmp     build_s2_cuda  prefix ../extern/install/nvidia-cuda
#   --configure re-runs the two configure lines below (the trees are configured once by hand otherwise).
# Parts (logs in build_s2/gate_logs/S1_2b.*):
#   B   G1_1.sh build_s2 --cuda-dir build_s2_cuda, G0_2.sh and G0_3.sh on both trees (--base runs them; by
#       default their outputs S1_2b.base_*.out are read, as written by the base run before test_ale_stepper was
#       registered: G1_1.sh's battery counts exactly 46 ctests) + every ALE ctest (ctest -R ^test_ale_) on the host
#       and the GPU. K1-K6 must equal the record build_s0/gate_logs/S1_1c.G1_1.out (the G1.1 PASS of A11) field by field.
#   C0  test_ale_frame --export (16^3, M1-M3) -> test_ale_stepper.py inputs -> test_ale_stepper --apply (host, GPU)
#       -> test_ale_stepper.py compare.
#   C1, C2  ale_reference_stepper.py trace (tg_Z16_F, tg_W16_R, 10 steps, predictor rtol 1e-13) -> test_ale_stepper
#       --trace (tight tolerances) on the host (8 threads, twice; 1 thread) and the GPU (twice) -> test_ale_stepper.py
#       trace.
#   C3  GPU against host (tg_Z16_F, U after 10 steps, <= 1e-9); host 8 threads run twice and GPU run twice bitwise;
#       1 thread against 8 threads reported.
#   L   test_ale_stepper --laplacian 32 / 64 on the host and the GPU.
#   R0-R6, T  test_ale_stepper --run CASE on the host and the GPU (default tolerances), judged by
#       test_ale_stepper.py judge against the reference's logs build_s0/gate_logs/S1_2a.*.
# GPU work runs under `flock -w 3600` on the shared heavy.lock, one process at a time; the host runs one at a time
# beside it (at most two runs at once). Prints one line per criterion and ONE final line `G1.2 PASS|FAIL`; exit 1
# on FAIL.
set -u
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
B="$SRC/build_s2"
C="$SRC/build_s2_cuda"
L="$B/gate_logs"
REF="$SRC/build_s0/gate_logs"
LOCK=/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock
CONFIGURE=0
BASE=0
JUDGE_ONLY=0
GPU=1
CASES="r0_zero_F r0_zero_R r0_unif_F r0_unif_R tg_Z16_F tg_Z16_R tg_W16_F tg_W16_R r3_tg_W16_F r3_tg_W32_F
       tg_Z32_R tg_Z8_F tg_W8_F abc_Z16_F abc_W16_F tg_Z32_F tg_W32_F abc_Z32_F abc_W32_F
       r6i_W16 r6ii_rest_W16 r6ii_tg_W16 tg_Z64_F tg_W64_F abc_Z64_F abc_W64_F"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure) CONFIGURE=1 ;;
    --base) BASE=1 ;;
    --judge-only) JUDGE_ONLY=1 ;;
    --cases) CASES="$2"; shift ;;
    --no-gpu) GPU=0 ;;
  esac
  shift
done
mkdir -p "$L"
echo $$ > "$L/S1_2b.gate.pid"
cd "$SRC"
if [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
fi
export PATH=/usr/local/cuda/bin:$PATH
export OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=tests/python/test_ale_stepper.py
RPY=tests/python/ale_reference_stepper.py
fail=0
out=()
say() {  # one criterion line "<id> PASS|FAIL|REPORT ..."; the second word decides
  out+=("$1")
  [[ "$(awk '{print $2}' <<< "$1")" == PASS || "$(awk '{print $2}' <<< "$1")" == REPORT ]] || fail=1
  echo "$1" >> "$L/S1_2b.gate.lines"
}
: > "$L/S1_2b.gate.lines"

configure() {  # $1 tree, $2 prefix
  cmake -S "$SRC" -B "$1" -DCMAKE_BUILD_TYPE=Release -DPECLET_VORO_KOKKOS=ON -DPECLET_VORO_BUILD_PYTHON=ON \
    -DPECLET_VORO_MPI=ON -DPECLET_VORO_BUILD_TESTS=ON "-DMPIEXEC_PREFLAGS=--bind-to;none" \
    -DCMAKE_PREFIX_PATH="$SRC/../extern/install/$2" -DPECLET_VENDOR_SIBLINGS=ON \
    "-DFETCHCONTENT_SOURCE_DIR_PECLET_SIB_PECLET-CORE=$SRC/../core-ale-amg" \
    "-DFETCHCONTENT_SOURCE_DIR_PECLET_SIB_PECLET-MORTON=$SRC/../morton" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
}
gpu() { flock -w 3600 "$LOCK" "$@"; }   # one GPU process at a time on the shared GPU

if [[ $JUDGE_ONLY -eq 0 ]]; then
  if [[ $CONFIGURE -eq 1 ]]; then
    configure "$B" host-openmp > "$L/S1_2b.configure.log" 2>&1 || { echo "G1.2 FAIL configure host"; exit 1; }
    configure "$C" nvidia-cuda > "$L/S1_2b.configure_cuda.log" 2>&1 || { echo "G1.2 FAIL configure cuda"; exit 1; }
  fi
  # the include paths must be the local siblings (core-ale-amg, morton) and nothing fetched
  inc=$(grep -rhoE '(-I|-isystem )[^ ]*(core|morton)[^ ]*' "$B" "$C" --include=flags.make | sort -u | tr '\n' ' ')
  echo "include paths: $inc" > "$L/S1_2b.includes.log"
  if [[ "$inc" != *core-ale-amg/include* || "$inc" == *"/core/include"* || -d "$B/_deps" || -d "$C/_deps" ]]; then
    echo "G1.2 FAIL include paths ($inc)"; exit 1
  fi
  cmake --build "$B" --parallel 8 --target test_ale_stepper test_ale_frame test_ale_operators \
    > "$L/S1_2b.build.log" 2>&1 || { echo "G1.2 FAIL host build (log $L/S1_2b.build.log)"; exit 1; }
  if [[ $GPU -eq 1 ]]; then
    cmake -B "$C" > "$L/S1_2b.reconfigure_cuda.log" 2>&1
    cmake --build "$C" --parallel 8 --target test_ale_stepper test_ale_frame test_ale_operators \
      > "$L/S1_2b.build_cuda.log" 2>&1 || { echo "G1.2 FAIL cuda build (log $L/S1_2b.build_cuda.log)"; exit 1; }
  fi
  nwarn=$(grep -c 'warning' "$L/S1_2b.build.log" "$L/S1_2b.build_cuda.log" 2>/dev/null | awk -F: '{s+=$2} END{print s}')
  echo "build ok, warnings=$nwarn" > "$L/S1_2b.build.summary"
fi
TH="$B/tests/kokkos/test_ale_stepper"
TG="$C/tests/kokkos/test_ale_stepper"

# ------------------------------------------------------------------------------------------------ B
if [[ $JUDGE_ONLY -eq 0 && $BASE -eq 1 ]]; then
  bash tools/ale_gates/G1_1.sh "$B" --cuda-dir "$C" > "$L/S1_2b.base_G1_1.out" 2>&1
  bash tools/ale_gates/G0_3.sh "$B" > "$L/S1_2b.base_G0_3.out" 2>&1
  gpu bash tools/ale_gates/G0_3.sh "$C" > "$L/S1_2b.base_G0_3_cuda.out" 2>&1
  bash tools/ale_gates/G0_2.sh "$B" > "$L/S1_2b.base_G0_2.out" 2>&1
  gpu bash tools/ale_gates/G0_2.sh "$C" > "$L/S1_2b.base_G0_2_cuda.out" 2>&1
fi
if [[ $JUDGE_ONLY -eq 0 ]]; then
  OMP_NUM_THREADS=4 ctest --test-dir "$B" -R '^test_ale_' --output-on-failure > "$L/S1_2b.ale_ctest.host.log" 2>&1
  if [[ $GPU -eq 1 ]]; then
    OMP_NUM_THREADS=8 gpu ctest --test-dir "$C" -R '^test_ale_(frame|operators|stepper)$' --output-on-failure \
      > "$L/S1_2b.ale_ctest.gpu.log" 2>&1
  fi
fi
g11=$(grep '^G1.1 ' "$L/S1_2b.base_G1_1.out" 2>/dev/null)
rec=$(grep -h '^G1.1 PASS' "$REF/S1_1c.G1_1.out" 2>/dev/null | tail -n 1)
kdiff=""
for k in K1 K2 K3 K4 K5 K6; do
  a=$(grep -oE " $k=[^ ]+" <<< "$g11"); b=$(grep -oE " $k=[^ ]+" <<< "$rec")
  [[ -n "$a" && "$a" == "$b" ]] || kdiff="$kdiff $k"
done
kok=1
for k in K0 K1 K2 K3 K4 K5 K6 K7 K8; do grep -qE " $k=PASS" <<< "$g11" || kok=0; done
bfield=$(grep -oE ' base=[^ ]+' <<< "$g11")
bok=0
if [[ "$bfield" =~ ctest=([0-9]+)/([0-9]+),hash=same ]] && [[ ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]]; then bok=1; fi
g02=$(grep -oE "^S0.2' [A-Z]+" "$L/S1_2b.base_G0_2.out" 2>/dev/null | head -n1)
g02c=$(grep -oE "^S0.2' [A-Z]+( build)?" "$L/S1_2b.base_G0_2_cuda.out" 2>/dev/null | head -n1)
g03=$(grep -oE "^G0.3' [A-Z]+" "$L/S1_2b.base_G0_3.out" 2>/dev/null | head -n1)
g03c=$(grep -oE "^G0.3' [A-Z]+" "$L/S1_2b.base_G0_3_cuda.out" 2>/dev/null | head -n1)
cth=$(grep -oE '[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+' "$L/S1_2b.ale_ctest.host.log" 2>/dev/null)
ctg=$(grep -oE '[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+' "$L/S1_2b.ale_ctest.gpu.log" 2>/dev/null)
bline="K0-K8 PASS=$kok K1-K6 equal to the record=$([[ -z $kdiff ]] && echo 1 || echo "0(${kdiff# })") battery(all pass,hash same)=$bok [$bfield] G0.2' host=${g02:-missing} gpu=${g02c:-missing} G0.3' host=${g03:-missing} gpu=${g03c:-missing} ALE ctests host: ${cth:-missing}; gpu: ${ctg:-missing}"
if [[ $kok -eq 1 && -z $kdiff && $bok -eq 1 && "$g02" == "S0.2' PASS" && "$g02c" == "S0.2' PASS" \
      && "$g03" == "G0.3' PASS" \
      && "$g03c" == "G0.3' PASS" && "$cth" == 100%* && ( $GPU -eq 0 || "$ctg" == 100%* ) ]]; then
  say "B PASS $bline"
else
  say "B FAIL $bline"
fi

# ------------------------------------------------------------------------------------------------ C0
E="$L/S1_2b.export16"
if [[ $JUDGE_ONLY -eq 0 ]]; then
  rm -rf "$E" && mkdir -p "$E" "$E/host" "$E/gpu"
  OMP_NUM_THREADS=1 "$B/tests/kokkos/test_ale_frame" --export "$E" --cells 16 --mesh all > "$L/S1_2b.export16.log" 2>&1
  OMP_NUM_THREADS=1 python "$PY" inputs "$E" --cells 16 > "$L/S1_2b.c0_inputs.log" 2>&1
  OMP_NUM_THREADS=8 "$TH" --apply "$E" --cells 16 --out "$E/host" > "$L/S1_2b.c0_apply.host.log" 2>&1
  [[ $GPU -eq 1 ]] && OMP_NUM_THREADS=8 gpu "$TG" --apply "$E" --cells 16 --out "$E/gpu" > "$L/S1_2b.c0_apply.gpu.log" 2>&1
fi
c0ref=$(grep -E '^C0ref (PASS|FAIL)' "$L/S1_2b.c0_inputs.log" 2>/dev/null)
for be in host gpu; do
  [[ $be == gpu && $GPU -eq 0 ]] && continue
  python "$PY" compare "$E" "$E/$be" --cells 16 > "$L/S1_2b.c0_compare.$be.log" 2>&1
  l=$(grep -E '^C0 (PASS|FAIL)' "$L/S1_2b.c0_compare.$be.log")
  grep -q '=== S1.2b apply PASS ===' "$L/S1_2b.c0_apply.$be.log" 2>/dev/null || l="C0 FAIL apply ($be)"
  say "${l:-C0 FAIL nofinish} [$be] [${c0ref:-C0ref missing}]"
done

# ------------------------------------------------------------------------------------------------ C1-C3
T="$L/S1_2b.trace"
if [[ $JUDGE_ONLY -eq 0 ]]; then
  rm -rf "$T" && mkdir -p "$T"/{h8a,h8b,h1,ga,gb}
  for m in M1 M2; do cp "$E/${m}_16."* "$T/"; done
  for cs in tg_Z16_F tg_W16_R; do
    OMP_NUM_THREADS=1 python "$RPY" trace "$T" "$cs" "$T" --steps 10 --pred-rtol 1e-13 >> "$L/S1_2b.trace_ref.log" 2>&1
    OMP_NUM_THREADS=8 "$TH" --trace "$T" "$cs" --out "$T/h8a" >> "$L/S1_2b.trace.h8a.log" 2>&1
    OMP_NUM_THREADS=8 "$TH" --trace "$T" "$cs" --out "$T/h8b" >> "$L/S1_2b.trace.h8b.log" 2>&1
    OMP_NUM_THREADS=1 "$TH" --trace "$T" "$cs" --out "$T/h1" >> "$L/S1_2b.trace.h1.log" 2>&1
    if [[ $GPU -eq 1 ]]; then
      OMP_NUM_THREADS=8 gpu "$TG" --trace "$T" "$cs" --out "$T/ga" >> "$L/S1_2b.trace.ga.log" 2>&1
      OMP_NUM_THREADS=8 gpu "$TG" --trace "$T" "$cs" --out "$T/gb" >> "$L/S1_2b.trace.gb.log" 2>&1
    fi
  done
fi
for be in h8a ga; do
  [[ $be == ga && $GPU -eq 0 ]] && continue
  python "$PY" trace "$T" "$T/$be" > "$L/S1_2b.c12.$be.log" 2>&1
  for c in C1 C2; do
    l=$(grep -E "^$c (PASS|FAIL)" "$L/S1_2b.c12.$be.log")
    say "${l:-$c FAIL nofinish} [$([[ $be == h8a ]] && echo host || echo gpu)]"
  done
done
pair() { python "$PY" c3 "$T/$1" "$T/$2" ${3:-} 2>&1 | grep -E '^C3pair'; }
if [[ $GPU -eq 1 ]]; then
  gh=$(pair ga h8a); hh=$(pair h8a h8b); gg=$(pair ga gb); h1=$(pair h1 h8a)
  ok=1
  [[ "$gh" == "C3pair PASS"* ]] || ok=0
  [[ "$hh" == *"bitwise=1"* && "$gg" == *"bitwise=1"* ]] || ok=0
  say "C3 $([[ $ok == 1 ]] && echo PASS || echo FAIL) GPU vs host: ${gh#C3pair } | host 8 vs 8: ${hh#C3pair } | GPU vs GPU: ${gg#C3pair } | reported host 1 vs 8: ${h1#C3pair }"
else
  hh=$(pair h8a h8b); h1=$(pair h1 h8a)
  say "C3 FAIL GPU not run | host 8 vs 8: ${hh#C3pair } | host 1 vs 8: ${h1#C3pair }"
fi

# ------------------------------------------------------------------------------------------------ L, runs
if [[ $JUDGE_ONLY -eq 0 ]]; then
  host_jobs() {
    for n in 32 64; do OMP_NUM_THREADS=8 "$TH" --laplacian $n > "$L/S1_2b.lap.$n.host.log" 2>&1; done
    for cs in $CASES; do
      OMP_NUM_THREADS=8 "$TH" --run "$cs" > "$L/S1_2b.run.$cs.host.log" 2>&1 \
        || echo "RESULT case=$cs status=CRASH" >> "$L/S1_2b.run.$cs.host.log"
    done
  }
  gpu_jobs() {
    for n in 32 64; do OMP_NUM_THREADS=8 gpu "$TG" --laplacian $n > "$L/S1_2b.lap.$n.gpu.log" 2>&1; done
    for cs in $CASES; do
      OMP_NUM_THREADS=8 gpu "$TG" --run "$cs" > "$L/S1_2b.run.$cs.gpu.log" 2>&1 \
        || echo "RESULT case=$cs status=CRASH" >> "$L/S1_2b.run.$cs.gpu.log"
    done
  }
  { uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; } \
    > "$L/S1_2b.load_start.log" 2>&1
  host_jobs &
  hp=$!
  [[ $GPU -eq 1 ]] && gpu_jobs
  wait $hp
  { uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; } \
    > "$L/S1_2b.load_end.log" 2>&1
fi
python "$PY" judge "$L" "$REF" > "$L/S1_2b.judge.log" 2>&1
while IFS= read -r ln; do
  [[ "$ln" == G1.2runs* ]] && continue
  say "$ln"
done < <(grep -E '^(R[0-9]|R2|L |T )' "$L/S1_2b.judge.log")
grep -q '^G1.2runs' "$L/S1_2b.judge.log" || say "R FAIL judge did not finish (log $L/S1_2b.judge.log)"

for ln in "${out[@]}"; do echo "$ln"; done
verdict=PASS
[[ $fail -eq 0 ]] || verdict=FAIL
echo "G1.2 $verdict (logs $L/S1_2b.*)"
[[ $fail -eq 0 ]]
