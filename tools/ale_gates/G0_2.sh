#!/usr/bin/env bash
# Gate S0.2' of the moving-grid design note (docs/ale_design.md §7 WO-S0.2; criteria fixed
# 2026-11-14 in the brief ALE_S0_2): the frame pipeline (ale/frame.hpp), the forced worklist and
# the changed set (repair.hpp).
#
#   tools/ale_gates/G0_2.sh <build dir>
#
# Needs, from the baseline tree (be2d67d) in the same build dir:
#   <build dir>/gate_logs/ctest.S0_2base.log  (OMP_NUM_THREADS=4 ctest -LE bench)
#   <build dir>/gate_logs/state_hash.base     (OMP_NUM_THREADS=1 python python/state_hash.py)
# Builds the tree (-j8), then
#   base  every ctest that passed in the baseline passes; state_hash.py output == state_hash.base
#   V0.1a test_ale_frame --part a, 16³ and 32³, Voronoi + power, 100 frames at rest: every FaceMesh
#         array bitwise equal to frame 0, changedMask zero; OMP 1 default builder, OMP 8 frame
#         options
#   V0.1b test_ale_frame --part b, 16³, CollocatedNS (B) RK3 and (D) implicit, 100 steps: U, p, uf
#         with the mesh replaced every step bitwise equal to the fixed-mesh run (in the test, and
#         sha256sum of the dumps); OMP 1 default builder, OMP 8 deterministicOrder
#   F     test_ale_frame --part f, 16³ power at rest, 50 forced slots
#   G0.2' test_ale_frame --part g, 32³ power, 500 frames, enforcement off and on, variant Z (all
#         weights zero) and W (weights in [0, 0.15 h²], evaluated before the first empty cell), four
#         runs: OMP 1, 1, 8, 8. (i)-(iv') in the test (docs/ale_design.md §11 A5); (v') here: the two
#         OMP-1 runs have identical SHA-256 of every dumped array of the last evaluated frame, and
#         its cell volumes agree to 1e-9 relative between the first OMP-1 run and each OMP-8 run
#         (W: if the runs evaluated different frames, the three compared runs are repeated capped at
#         the smallest evaluated frame count); the first frame at which the rebuilt counts differ
#         (1 v 8, 8 v 8) is read from the logs. PASS needs V0.1a, V0.1b, F, base and G0.2'Z;
#         G0.2'W is printed with its own PASS/MISS and does not decide the line.
# Logs in <build dir>/gate_logs/S0_2p.*. Prints ONE line `S0.2' PASS|FAIL V0.1a=… V0.1b=… G0.2'Z=…
# G0.2'W=… F=… base=…`; exit 1 on FAIL.
set -u
B="$(cd "${1:?usage: G0_2.sh <build dir>}" && pwd)"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
L="$B/gate_logs"
mkdir -p "$L"
cd "$SRC"
if [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
fi
export OMP_PROC_BIND=false
for f in "$L/ctest.S0_2base.log" "$L/state_hash.base"; do
  [[ -f "$f" ]] || { echo "S0.2' FAIL missing baseline $f"; exit 1; }
done

if ! cmake --build "$B" -j8 > "$L/S0_2p.build.log" 2>&1; then
  echo "S0.2' FAIL build (log $L/S0_2p.build.log)"
  exit 1
fi
T="$B/tests/kokkos/test_ale_frame"

# ---- base ----
OMP_NUM_THREADS=4 ctest --test-dir "$B" -LE bench > "$L/S0_2p.ctest.log" 2>&1
passed_base=$(grep -E 'Test +#[0-9]+: .* Passed' "$L/ctest.S0_2base.log" | sed -E 's/.*: ([^ ]+) .*/\1/' | sort)
passed_now=$(grep -E 'Test +#[0-9]+: .* Passed' "$L/S0_2p.ctest.log" | sed -E 's/.*: ([^ ]+) .*/\1/' | sort)
lost=$(comm -23 <(echo "$passed_base") <(echo "$passed_now") | tr '\n' ' ')
n_base=$(echo "$passed_base" | grep -c .)
n_now=$(echo "$passed_now" | grep -c .)
n_all=$(grep -cE 'Test +#[0-9]+: ' "$L/S0_2p.ctest.log")
OMP_NUM_THREADS=1 PYTHONPATH="$B" python python/state_hash.py > "$L/S0_2p.state_hash" 2> "$L/S0_2p.state_hash.err"
if cmp -s "$L/state_hash.base" "$L/S0_2p.state_hash" && [[ -s "$L/S0_2p.state_hash" ]]; then
  hash_ok=1
else
  hash_ok=0
fi
if [[ -z "${lost// /}" && $hash_ok == 1 ]]; then
  base="PASS(ctest ${n_now}/${n_all}pass,${n_base}base,hash=same)"
else
  base="FAIL(lost:${lost:-none},hash_ok=${hash_ok})"
fi

# ---- V0.1 (a) ----
na=0
for n in 16 32; do
  OMP_NUM_THREADS=1 "$T" $n --part a --full > "$L/S0_2p.a$n.t1.log" 2>&1
  OMP_NUM_THREADS=8 "$T" $n --part a --full > "$L/S0_2p.a$n.t8.log" 2>&1
  na=$((na + $(grep -c '^RESULT a .* PASS' "$L/S0_2p.a$n.t1.log" "$L/S0_2p.a$n.t8.log" | awk -F: '{s+=$2} END {print s}')))
done
if [[ $na == 8 ]]; then a="PASS(8/8)"; else a="FAIL($na/8)"; fi

# ---- V0.1 (b) ----
D="$L/S0_2p.dump"
rm -rf "$D" && mkdir -p "$D/b1" "$D/b8"
OMP_NUM_THREADS=1 "$T" 16 --part b --full --dump "$D/b1" > "$L/S0_2p.b.t1.log" 2>&1
OMP_NUM_THREADS=8 "$T" 16 --part b --full --dump "$D/b8" > "$L/S0_2p.b.t8.log" 2>&1
nb=$(cat "$L/S0_2p.b.t1.log" "$L/S0_2p.b.t8.log" | grep -c '^RESULT b .* PASS')
shadiff=0
: > "$L/S0_2p.b.sha"
for r in b1 b8; do
  for c in B_rk3 D_implicit; do
    for x in U p uf; do
      h1=$(sha256sum < "$D/$r/b_${c}_run1.$x" | cut -c1-64)
      h2=$(sha256sum < "$D/$r/b_${c}_run2.$x" | cut -c1-64)
      echo "$r $c $x $h1 $h2" >> "$L/S0_2p.b.sha"
      [[ "$h1" == "$h2" ]] || shadiff=$((shadiff + 1))
    done
  done
done
if [[ $nb == 4 && $shadiff == 0 ]]; then b="PASS(4/4,sha=12/12)"; else b="FAIL($nb/4,shadiff=$shadiff)"; fi

# ---- F ----
OMP_NUM_THREADS=8 "$T" 16 --part f > "$L/S0_2p.f.log" 2>&1
if grep -q '^RESULT f .* PASS' "$L/S0_2p.f.log"; then f="PASS"; else f="FAIL"; fi

# ---- G0.2' ----
runG() {  # runG <dir> <threads> <log> [extra args]
  local d=$1 t=$2 lg=$3
  shift 3
  mkdir -p "$d"
  OMP_NUM_THREADS=$t "$T" 32 --part g --full --dump "$d" "$@" > "$lg" 2>&1
}
runG "$D/g1a" 1 "$L/S0_2p.g.t1a.log"
runG "$D/g1b" 1 "$L/S0_2p.g.t1b.log"
runG "$D/g8a" 8 "$L/S0_2p.g.t8a.log" --time
runG "$D/g8b" 8 "$L/S0_2p.g.t8b.log"
glogs="$L/S0_2p.g.t1a.log $L/S0_2p.g.t1b.log $L/S0_2p.g.t8a.log $L/S0_2p.g.t8b.log"
# variant W: the compared runs must have evaluated the same frames; else repeat them capped at the
# smallest evaluated count (frames before the first empty cell)
cap=$(cat $glogs | grep '^RESULT g variant=W ' | sed -E 's/.* evaluated=([0-9]+) .*/\1/' | sort -n | head -1)
same=$(cat "$L/S0_2p.g.t1a.log" "$L/S0_2p.g.t8a.log" "$L/S0_2p.g.t8b.log" | grep '^RESULT g variant=W ' | sed -E 's/.* (enforce=[0-9]) .* evaluated=([0-9]+) .*/\1 \2/' | sort -u | wc -l)
wdir1="$D/g1a"; wdir8a="$D/g8a"; wdir8b="$D/g8b"
wlog1="$L/S0_2p.g.t1a.log"; wlog8a="$L/S0_2p.g.t8a.log"; wlog8b="$L/S0_2p.g.t8b.log"
if [[ "$same" != 2 ]]; then  # one distinct evaluated count per enforce setting
  runG "$D/c1a" 1 "$L/S0_2p.gc.t1a.log" --gframes "$cap"
  runG "$D/c8a" 8 "$L/S0_2p.gc.t8a.log" --gframes "$cap"
  runG "$D/c8b" 8 "$L/S0_2p.gc.t8b.log" --gframes "$cap"
  wdir1="$D/c1a"; wdir8a="$D/c8a"; wdir8b="$D/c8b"
  wlog1="$L/S0_2p.gc.t1a.log"; wlog8a="$L/S0_2p.gc.t8a.log"; wlog8b="$L/S0_2p.gc.t8b.log"
fi
crit() {  # crit <variant> <name>: PASS if the field is PASS|INFO in every RESULT line (4 runs x 2)
  local tot ok
  tot=$(cat $glogs | grep -c "^RESULT g variant=$1 ")
  ok=$(cat $glogs | grep "^RESULT g variant=$1 " | grep -cE " $2=(PASS|INFO)")
  if [[ $tot == 8 && $ok == 8 ]]; then echo PASS; else echo "FAIL($ok/$tot)"; fi
}
vprime() {  # vprime <variant> <dir1a> <dir1b> <dir1 for volumes> <dir8a> <dir8b> <log1> <log8a> <log8b>
  python3 -I - "$@" << 'PYEOF'
import hashlib, os, re, sys
import numpy as np
v, d1a, d1b, d1v, d8a, d8b, l1, l8a, l8b = sys.argv[1:10]
def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()
def rebuilt(log):
    r = {}
    for ln in open(log):
        m = re.match(r'FRAME variant=%s enforce=(\d) k=(\d+) rebuilt=(\d+) ' % v, ln)
        if m:
            r[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    return r
def firstdiff(a, b):
    ks = sorted(k for k in a if k in b)
    for k in ks:
        if a[k] != b[k]:
            return 'e%d:k%d' % k
    return 'none'
shaok, worst, ok = True, 0.0, True
for e in (0, 1):
    tag = 'g_%s_enforce%d.' % (v, e)
    names = sorted(n for n in os.listdir(d1a) if n.startswith(tag))
    if not names:
        ok = False
    for n in names:
        if sha(os.path.join(d1a, n)) != sha(os.path.join(d1b, n)):
            shaok = False
    vol = tag + 'cellVolume'
    if vol not in names:
        ok = False
        continue
    a = np.fromfile(os.path.join(d1v, vol), dtype=np.float64)
    for d8 in (d8a, d8b):
        b = np.fromfile(os.path.join(d8, vol), dtype=np.float64)
        if a.shape != b.shape or a.size == 0:
            ok = False
            continue
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))))
R1, R8a, R8b = rebuilt(l1), rebuilt(l8a), rebuilt(l8b)
fd18, fd88 = firstdiff(R1, R8a), firstdiff(R8a, R8b)
ok = ok and shaok and worst <= 1e-9
print('%s(sha1v1=%s,maxRelDvol=%.2e,firstDiffRebuilt:1v8=%s,8v8=%s)' %
      ('PASS' if ok else 'FAIL', 'same' if shaok else 'DIFF', worst, fd18, fd88))
PYEOF
}
gz="i=$(crit Z i),ii=$(crit Z ii),iii=$(crit Z iii),iv=$(crit Z iv),empty=$(crit Z empty)"
gw="i=$(crit W i),ii=$(crit W ii),iii=$(crit W iii),iv=$(crit W iv)"
vz=$(vprime Z "$D/g1a" "$D/g1b" "$D/g1a" "$D/g8a" "$D/g8b" "$L/S0_2p.g.t1a.log" "$L/S0_2p.g.t8a.log" "$L/S0_2p.g.t8b.log")
vw=$(vprime W "$D/g1a" "$D/g1b" "$wdir1" "$wdir8a" "$wdir8b" "$wlog1" "$wlog8a" "$wlog8b")
fe=$(cat $glogs | grep '^RESULT g variant=W ' | sed -E 's/.* firstEmpty=([0-9a-z]+).*/\1/' | sort -u | tr '\n' ',' | sed 's/,$//')
gzverdict=PASS
for x in $(echo "$gz" | tr ',' ' ' | sed 's/[a-z]*=//g') "$vz"; do
  [[ $x == PASS* ]] || gzverdict=FAIL
done
gwverdict=PASS
for x in $(echo "$gw" | tr ',' ' ' | sed 's/[a-z]*=//g') "$vw"; do
  [[ $x == PASS* ]] || gwverdict=MISS
done
GZ="$gzverdict($gz,v'=$vz)"
GW="$gwverdict($gw,v'=$vw,firstEmpty=$fe,evaluated<=$cap)"

verdict=PASS
for x in "$gi" "$gii" "$giii" "$giv" "$gv"; do
  [[ $x == PASS* ]] || gverdict=FAIL
done

verdict=PASS
for x in "$a" "$b" "$f" "$gzverdict" "$base"; do
  [[ $x == PASS* ]] || verdict=FAIL
done
echo "S0.2' $verdict V0.1a=$a V0.1b=$b G0.2'Z=$GZ G0.2'W=$GW F=$f base=$base (logs $L/S0_2p.*)"
[[ $verdict == PASS ]]
