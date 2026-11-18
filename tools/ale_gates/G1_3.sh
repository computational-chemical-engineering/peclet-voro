#!/usr/bin/env bash
# Gate G1.3a of the moving-grid design note (docs/ale_design.md §11 A18; work order S1.3a, criteria V0.2, T, a-d,
# F, S fixed 2026-11-18 in the brief ALE_S1_3A before any run): the time step on a moving mesh, one phase
# (include/peclet/voro/ale/moving_stepper.hpp), on the host (OpenMP, 8 threads) and the GPU (CUDA), 16^3.
#
#   tools/ale_gates/G1_3.sh [--no-build] [--no-gpu] [--judge-only] [--no-g12]
#
# Trees build_s2 (host) and build_s2_cuda, configured as in G1_2.sh (core from ../core-ale-amg). Logs in
# build_s2/gate_logs/S1_3a.*:
#   V0.2  test_ale_stepper --run tg_Z16_F|tg_W16_F --motion rest --steps 100: the static path beside it in the same
#         process; U and p bit for bit after every step (v02_maxdiff = 0, v02_bitwise_steps = 100).
#   T     test_ale_stepper --trace-moving (tg_Z16_F, tg_Z16_R, follow, 5 steps, the tight tolerances of --trace)
#         -> test_ale_stepper.py trace-moving: F, U*, U1, p per step against ale_reference_stepper.moving_step,
#         relative <= 1e-12. Reported beside it (INFO T'): the same at the predictor tolerance 1e-14.
#   a     gcl <= 1e-14 in every moving run (V0.2, b, c, and the T traces).
#   b     unif_Z16_F|R --motion prescribed (tau = 1/400, 400 steps): unif_dev <= 1e-12, rebuilt >= 1000.
#   c     tg_Z16_F --motion follow to t = 1: closure <= 1e-13 K per step; B1 != 0 reported.
#   d     the runs of b and c: mom_booked <= 1e-13 sum M|U| (R2'); the unbooked drift (mom) and mom_def reported.
#   F     every run status=DONE (no frame error); ctest -R '^test_ale_' on both trees; G1_2.sh --no-runs: the
#         verdicts of its B and C lines equal to the record build_s2/gate_logs/S1_2c.G1_2.out.
#   S     --run tg_Z16_F (no --motion), host: tau, steps, err character for character those of
#         build_s2/gate_logs/S1_2b.run.tg_Z16_F.host.log.
#   INFO  (GPU) err of tg_Z16_F, tg_Z32_F with --motion follow, prescribed and static at --tend 0.5 and 1; the wall
#         time per step and its split at 32^3 (follow, t = 0.5); uptime and GPU load at the start and the end.
# GPU work runs under `flock -w 3600` on the shared heavy.lock; the host runs beside it. Prints one line per
# criterion and ONE final line `G1.3a PASS|FAIL`; exit 1 on FAIL.
set -u
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
B="$SRC/build_s2"
C="$SRC/build_s2_cuda"
L="$B/gate_logs"
LOCK=/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock
BUILD=1
GPU=1
JUDGE_ONLY=0
G12=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) BUILD=0 ;;
    --no-gpu) GPU=0 ;;
    --judge-only) JUDGE_ONLY=1 ;;
    --no-g12) G12=0 ;;
  esac
  shift
done
mkdir -p "$L"
echo $$ > "$L/S1_3a.gate.pid"
cd "$SRC"
if [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
fi
export PATH=/usr/local/cuda/bin:$PATH
export OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=tests/python/test_ale_stepper.py
TH="$B/tests/kokkos/test_ale_stepper"
TG="$C/tests/kokkos/test_ale_stepper"
gpu() { flock -w 3600 "$LOCK" "$@"; }
TARGETS="test_ale_stepper test_ale_frame test_ale_operators test_ale_facemesh"

if [[ $JUDGE_ONLY -eq 0 ]]; then
  if [[ $BUILD -eq 1 ]]; then
    # shellcheck disable=SC2086
    cmake --build "$B" --parallel 8 --target $TARGETS > "$L/S1_3a.build.log" 2>&1 \
      || { echo "G1.3a FAIL host build (log $L/S1_3a.build.log)"; exit 1; }
    if [[ $GPU -eq 1 ]]; then
      # shellcheck disable=SC2086
      cmake --build "$C" --parallel 8 --target $TARGETS > "$L/S1_3a.build_cuda.log" 2>&1 \
        || { echo "G1.3a FAIL cuda build (log $L/S1_3a.build_cuda.log)"; exit 1; }
    fi
  fi
  { uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; } \
    > "$L/S1_3a.load_start.log" 2>&1
  # one run: $1 backend, $2 log tag, the rest the arguments of test_ale_stepper
  run() {
    local be=$1 tag=$2
    shift 2
    local log="$L/S1_3a.$tag.$be.log"
    if [[ $be == host ]]; then
      OMP_NUM_THREADS=8 "$TH" "$@" > "$log" 2>&1
    else
      OMP_NUM_THREADS=8 gpu "$TG" "$@" > "$log" 2>&1
    fi
    local rc=$?
    [[ $rc -eq 0 ]] || grep -q '^RESULT' "$log" || echo "RESULT status=CRASH rc=$rc" >> "$log"
  }
  jobs_of() {  # $1 backend
    local be=$1
    for cs in tg_Z16_F tg_W16_F; do run "$be" "v02.$cs" --run "$cs" --motion rest --steps 100; done
    for cs in unif_Z16_F unif_Z16_R; do run "$be" "b.$cs" --run "$cs" --motion prescribed; done
    run "$be" c.tg_Z16_F --run tg_Z16_F --motion follow
    for x in tm tmx; do rm -rf "$L/S1_3a.$x.$be" && mkdir -p "$L/S1_3a.$x.$be"; done
    for cs in tg_Z16_F tg_Z16_R; do
      run "$be" "tm.$cs" --trace-moving "$L/S1_3a.tm.$be" "$cs"
      run "$be" "tmx.$cs" --trace-moving "$L/S1_3a.tmx.$be" "$cs" --pred-rtol 1e-14
    done
    if [[ $be == host ]]; then
      run host S.tg_Z16_F --run tg_Z16_F
    else
      for cs in tg_Z16_F tg_Z32_F; do
        for te in 0.5 1; do
          run gpu "info.$cs.none.$te" --run "$cs" --tend "$te"
          for mo in follow prescribed; do run gpu "info.$cs.$mo.$te" --run "$cs" --motion "$mo" --tend "$te"; done
        done
      done
    fi
  }
  jobs_of host &
  hp=$!
  [[ $GPU -eq 1 ]] && jobs_of gpu
  wait $hp
  { uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; } \
    > "$L/S1_3a.load_end.log" 2>&1
  for be in host gpu; do
    [[ $be == gpu && $GPU -eq 0 ]] && continue
    for x in tm tmx; do
      OMP_NUM_THREADS=4 python "$PY" trace-moving "$L/S1_3a.$x.$be" tg_Z16_F tg_Z16_R > "$L/S1_3a.$x.judge.$be.log" 2>&1
    done
  done
  # F: the ALE ctests on both trees, then G1_2.sh --no-runs (B and C)
  OMP_NUM_THREADS=8 ctest --test-dir "$B" -R '^test_ale_' --output-on-failure > "$L/S1_3a.ctest.host.log" 2>&1
  if [[ $GPU -eq 1 ]]; then
    OMP_NUM_THREADS=8 gpu ctest --test-dir "$C" -R '^test_ale_' --output-on-failure > "$L/S1_3a.ctest.gpu.log" 2>&1
  fi
  if [[ $G12 -eq 1 ]]; then
    bash tools/ale_gates/G1_2.sh --no-runs > "$L/S1_3a.G1_2.out" 2>&1
  fi
fi

# ------------------------------------------------------------------------------------------------ judge
python - "$L" "$GPU" > "$L/S1_3a.gate.lines" <<'EOF'
import os
import re
import sys

L, gpu = sys.argv[1], sys.argv[2] == '1'
bes = ['host', 'gpu'] if gpu else ['host']
fail = False


def say(line):
    global fail
    w = line.split()[1]
    if w not in ('PASS', 'REPORT', 'INFO'):
        fail = True
    print(line)


def result(tag, be):
    p = os.path.join(L, f'S1_3a.{tag}.{be}.log')
    if not os.path.exists(p):
        return None
    rs = [x for x in open(p) if x.startswith('RESULT')]
    if not rs:
        return None
    return dict(kv.split('=', 1) for kv in rs[-1].split()[1:] if '=' in kv)


def num(r, k, default=float('nan')):
    try:
        return float(r[k])
    except (TypeError, KeyError, ValueError):
        return default


runs = {be: {} for be in bes}
for be in bes:
    for f in sorted(os.listdir(L)):
        m = re.match(rf'S1_3a\.((?:v02|b|c|S|info|tm|tmx)\..+)\.{be}\.log$', f)
        if m:
            runs[be][m.group(1)] = result(m.group(1), be)

# V0.2
txt, ok = [], True
for be in bes:
    for cs in ('tg_Z16_F', 'tg_W16_F'):
        r = runs[be].get(f'v02.{cs}')
        d, s = num(r, 'v02_maxdiff'), num(r, 'v02_bitwise_steps')
        good = r is not None and r.get('status') == 'DONE' and d == 0.0 and s == 100.0
        ok = ok and good
        txt.append(f'{be} {cs} maxdiff={d:.1e} bitwise_steps={s:.0f}/100')
say(f"V0.2 {'PASS' if ok else 'FAIL'} {'; '.join(txt)} (U and p after every step, against the static path)")

# T (and T' as INFO)
for x, cid in (('tm', 'T'), ('tmx', "T'")):
    txt, ok = [], True
    for be in bes:
        p = os.path.join(L, f'S1_3a.{x}.judge.{be}.log')
        lines = open(p).read().splitlines() if os.path.exists(p) else []
        cases = [ln for ln in lines if ln.startswith('Tcase')]
        fin = [ln for ln in lines if ln.startswith('T ')]
        good = bool(fin) and fin[-1].split()[1] == 'PASS'
        ok = ok and good
        txt.append(f"[{be}] " + ' | '.join(c[6:] for c in cases) if cases else f'[{be}] missing')
    if cid == 'T':
        say(f"T {'PASS' if ok else 'FAIL'} {' '.join(txt)}")
    else:
        say(f"T' INFO predictor rtol 1e-14: {'pass' if ok else 'miss'} {' '.join(txt)}")

# a: gcl of every moving run, and of the trace steps
txt, ok, worst = [], True, 0.0
for be in bes:
    for tag in ('v02.tg_Z16_F', 'v02.tg_W16_F', 'b.unif_Z16_F', 'b.unif_Z16_R', 'c.tg_Z16_F'):
        g = num(runs[be].get(tag), 'gcl')
        ok = ok and g <= 1e-14
        worst = max(worst, g) if g == g else worst
        txt.append(f'{be} {tag.split(".", 1)[1]}{"(rest)" if tag.startswith("v02") else ""}={g:.2e}')
    for x in ('tm',):
        for cs in ('tg_Z16_F', 'tg_Z16_R'):
            p = os.path.join(L, f'S1_3a.{x}.{be}', f'{cs}.dev_scalars')
            gs = [float(ln.split()[3]) for ln in open(p)] if os.path.exists(p) else [float('nan')]
            g = max(gs)
            ok = ok and g <= 1e-14
            txt.append(f'{be} trace {cs}={g:.2e}')
say(f"a {'PASS' if ok else 'FAIL'} gcl max {worst:.2e} (<=1e-14): {', '.join(txt)}")

# b
txt, ok = [], True
for be in bes:
    for cs in ('unif_Z16_F', 'unif_Z16_R'):
        r = runs[be].get(f'b.{cs}')
        u, rb = num(r, 'unif_dev'), num(r, 'rebuilt')
        ok = ok and u <= 1e-12 and rb >= 1000
        txt.append(f'{be} {cs} unif_dev={u:.2e} rebuilt={rb:.0f} steps={r.get("steps") if r else None}')
say(f"b {'PASS' if ok else 'FAIL'} {'; '.join(txt)} (<=1e-12, rebuilt>=1000)")

# c
txt, ok = [], True
for be in bes:
    r = runs[be].get('c.tg_Z16_F')
    cl, b1 = num(r, 'closure'), num(r, 'B1_absmax')
    ok = ok and cl <= 1e-13 and b1 > 0
    txt.append(f'{be} closure={cl:.2e} B1_absmax/K={b1:.2e} steps={r.get("steps") if r else None} '
               f'rebuilt={num(r, "rebuilt"):.0f}')
say(f"c {'PASS' if ok else 'FAIL'} {'; '.join(txt)} (closure<=1e-13 K, B1 != 0)")

# d
txt, ok = [], True
for be in bes:
    for tag in ('b.unif_Z16_F', 'b.unif_Z16_R', 'c.tg_Z16_F'):
        r = runs[be].get(tag)
        mb = num(r, 'mom_booked')
        ok = ok and mb <= 1e-13
        txt.append(f'{be} {tag.split(".", 1)[1]} booked={mb:.2e} unbooked={num(r, "mom"):.2e} '
                   f'defect={num(r, "mom_def"):.2e}')
say(f"d {'PASS' if ok else 'FAIL'} {'; '.join(txt)} (mom_booked<=1e-13 sum M|U|)")

# F
bad = [f'{be}:{t}={(r or {}).get("status")}' for be in bes for t, r in runs[be].items() if not t.startswith('tm')
       and (r is None or r.get('status') != 'DONE')]
for be in bes:
    for x in ('tm', 'tmx'):
        for cs in ('tg_Z16_F', 'tg_Z16_R'):
            r = runs[be].get(f'{x}.{cs}')
            if r is None or r.get('status') != 'DONE':
                bad.append(f'{be}:{x}.{cs}={(r or {}).get("status")}')
ct = {}
for be in bes:
    p = os.path.join(L, f'S1_3a.ctest.{be}.log')
    m = re.search(r'\d+% tests passed, (\d+) tests failed out of (\d+)', open(p).read()) if os.path.exists(p) else None
    ct[be] = f'{int(m.group(2)) - int(m.group(1))}/{m.group(2)}' if m else 'missing'
ctok = all(v != 'missing' and v.split('/')[0] == v.split('/')[1] for v in ct.values())


def verdicts(path):
    if not os.path.exists(path):
        return None
    out = []
    for ln in open(path):
        m = re.match(r'^(B|C0|C1|C2|C3) (PASS|FAIL)', ln)
        if m:
            tag = re.search(r'\[(host|gpu)\]', ln)
            out.append(f'{m.group(1)}{"[" + tag.group(1) + "]" if tag and m.group(1) != "B" else ""}={m.group(2)}')
    return out


now, rec = verdicts(os.path.join(L, 'S1_3a.G1_2.out')), verdicts(os.path.join(L, 'S1_2c.G1_2.out'))
g12ok = now is not None and now == rec
bline = ''
if now is not None:
    bl = [ln for ln in open(os.path.join(L, 'S1_3a.G1_2.out')) if ln.startswith('B ')]
    bline = bl[0].strip()[:240] if bl else ''
ok = not bad and ctok and g12ok
say(f"F {'PASS' if ok else 'FAIL'} runs not DONE: {bad if bad else 'none'}; ctest ^test_ale_ passed "
    f"{', '.join(f'{k} {v}' for k, v in ct.items())}; G1_2.sh --no-runs verdicts {now} "
    f"{'= record' if g12ok else '!= record ' + str(rec)} (S1_2c.G1_2.out) [{bline}]")

# S
r = runs['host'].get('S.tg_Z16_F')
ref = None
p = os.path.join(L, 'S1_2b.run.tg_Z16_F.host.log')
if os.path.exists(p):
    rs = [x for x in open(p) if x.startswith('RESULT')]
    ref = dict(kv.split('=', 1) for kv in rs[-1].split()[1:] if '=' in kv) if rs else None
ok = r is not None and ref is not None and all(r.get(k) == ref.get(k) for k in ('tau', 'steps', 'err'))
say(f"S {'PASS' if ok else 'FAIL'} tg_Z16_F static, host: " +
    ' '.join(f"{k}={(r or {}).get(k)}/{(ref or {}).get(k)}" for k in ('tau', 'steps', 'err')) + ' (now/record)')

# INFO
if gpu:
    rows = []
    for cs in ('tg_Z16_F', 'tg_Z32_F'):
        for te in ('0.5', '1'):
            e = {mo: num(runs['gpu'].get(f'info.{cs}.{mo}.{te}'), 'err') for mo in ('none', 'follow', 'prescribed')}
            st = runs['gpu'].get(f'info.{cs}.none.{te}') or {}
            rows.append(f"{cs} t={st.get('t')}: static {e['none']:.4e} follow {e['follow']:.4e} "
                        f"({e['follow'] / e['none']:.2f}x) prescribed {e['prescribed']:.4e} "
                        f"({e['prescribed'] / e['none']:.2f}x)")
    say('INFO err (GPU): ' + '; '.join(rows))
    r = runs['gpu'].get('info.tg_Z32_F.follow.0.5') or {}
    keys = ('wall_per_step', 't_frames', 't_setup', 't_s3', 't_pred', 't_mu', 't_s5', 't_s6', 't_budget')
    load = ' | '.join(open(os.path.join(L, f'S1_3a.load_{w}.log')).read().replace('\n', ' ').strip()
                      for w in ('start', 'end') if os.path.exists(os.path.join(L, f'S1_3a.load_{w}.log')))
    say('INFO 32^3 follow GPU (t = 0.5) per step [s]: ' + ' '.join(f'{k}={r.get(k)}' for k in keys) +
        f' rebuilt={r.get("rebuilt")} frames={r.get("frames")} [load start | end: {load}] (shared host: not a benchmark)')
print(f"G1.3a {'FAIL' if fail else 'PASS'} (logs {L}/S1_3a.*)")
EOF
cat "$L/S1_3a.gate.lines"
grep -q '^G1.3a PASS' "$L/S1_3a.gate.lines"
