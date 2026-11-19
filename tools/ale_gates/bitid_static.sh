#!/usr/bin/env bash
# Bit-identity of the static path against reference binaries (criterion N of the brief ALE_S1_3A_R2):
# test_ale_stepper --trace (the C1/C2 traces of G1_2.sh, inputs build_s2/gate_logs/S1_2b.trace{,_gpu}) and
# --run (static, no --motion) of tg_Z16_F and tg_W16_R, on the host (8 threads) and the GPU (under the heavy lock),
# with the reference binaries REF/{host,gpu}/test_ale_stepper and the trees' own. Every trace output array equal
# (np.array_equal of the raw float64 files, the text files byte for byte), every RESULT key equal except the
# wall times.
#
#   tools/ale_gates/bitid_static.sh REF OUT [--no-gpu]
#
# Prints one line `N-static PASS|FAIL ...`; exit 1 on FAIL. Logs and outputs in OUT.
set -u
REF=$1
OUT=$2
GPU=1
[[ ${3:-} == --no-gpu ]] && GPU=0
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
L="$SRC/build_s2/gate_logs"
LOCK=/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock
export OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$OUT"
declare -A NEW=([host]="$SRC/build_s2/tests/kokkos/test_ale_stepper" [gpu]="$SRC/build_s2_cuda/tests/kokkos/test_ale_stepper")
declare -A TIN=([host]="$L/S1_2b.trace" [gpu]="$L/S1_2b.trace_gpu")
bes=host
[[ $GPU -eq 1 ]] && bes="host gpu"
for be in $bes; do
  for v in ref new; do
    bin=$([[ $v == ref ]] && echo "$REF/$be/test_ale_stepper" || echo "${NEW[$be]}")
    pre=$([[ $be == gpu ]] && echo "flock -w 3600 $LOCK" || echo "")
    for cs in tg_Z16_F tg_W16_R; do
      d="$OUT/$be.$v.trace"
      mkdir -p "$d"
      $pre "$bin" --trace "${TIN[$be]}" "$cs" --out "$d" > "$OUT/$be.$v.trace.$cs.log" 2>&1
      $pre "$bin" --run "$cs" > "$OUT/$be.$v.run.$cs.log" 2>&1
    done
  done
done
source "$SRC/../.venv/bin/activate" 2>/dev/null
python - "$OUT" "$bes" <<'EOF'
import os, sys
import numpy as np
out, bes = sys.argv[1], sys.argv[2].split()
skip = ('wall_per_step', 'setup', 'frame_setup')
bad, n_arr, n_keys = [], 0, 0
def result(p):
    rs = [x for x in open(p) if x.startswith('RESULT')]
    return dict(kv.split('=', 1) for kv in rs[-1].split()[1:] if '=' in kv) if rs else None
for be in bes:
    a, b = os.path.join(out, f'{be}.ref.trace'), os.path.join(out, f'{be}.new.trace')
    fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
    if fa != fb or not fa:
        bad.append(f'{be}: trace files differ ({len(fa)}/{len(fb)})')
    for f in fa:
        if f not in fb:
            continue
        ra, rb = open(os.path.join(a, f), 'rb').read(), open(os.path.join(b, f), 'rb').read()
        if f.endswith('scalars'):
            ok = ra == rb
        else:
            ok = len(ra) == len(rb) and np.array_equal(np.frombuffer(ra, np.float64), np.frombuffer(rb, np.float64))
        n_arr += 1
        if not ok:
            bad.append(f'{be}:{f}')
    for cs in ('tg_Z16_F', 'tg_W16_R'):
        for kind in ('trace', 'run'):
            ra = result(os.path.join(out, f'{be}.ref.{kind}.{cs}.log'))
            rb = result(os.path.join(out, f'{be}.new.{kind}.{cs}.log'))
            if kind == 'trace' and ra is None and rb is None:
                continue
            if ra is None or rb is None or ra.get('status', 'DONE') != 'DONE' and kind == 'run':
                bad.append(f'{be}:{kind}.{cs} RESULT missing/not DONE')
                continue
            for k in ra:
                if k in skip or k.startswith('t_'):
                    continue
                n_keys += 1
                if ra[k] != rb.get(k):
                    bad.append(f'{be}:{kind}.{cs}:{k} {ra[k]}!={rb.get(k)}')
print(f"N-static {'FAIL' if bad else 'PASS'} backends={'+'.join(bes)} trace_files_equal={n_arr - sum(1 for x in bad if ':' in x and '.' in x.split(':')[1] and 'RESULT' not in x)}/{n_arr} "
      f"result_keys_compared={n_keys} differences={bad[:8]}")
sys.exit(1 if bad else 0)
EOF
