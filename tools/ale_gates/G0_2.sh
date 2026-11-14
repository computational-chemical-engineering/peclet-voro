#!/usr/bin/env bash
# Gate S0.2 of the moving-grid design note (docs/ale_design.md §7 WO-S0.2; criteria fixed
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
#   G0.2  test_ale_frame --part g, 32³ power, 500 frames, enforcement off and on: (i)-(iv) in the
#         test at OMP 1, 8, 8; (v) the frame-500 FaceMesh arrays bitwise equal across the three runs
# Logs in <build dir>/gate_logs/S0_2.*. Prints ONE line `S0.2 PASS|FAIL V0.1a=… V0.1b=… G0.2=… F=…
# base=…`; exit 1 on FAIL.
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
  [[ -f "$f" ]] || { echo "S0.2 FAIL missing baseline $f"; exit 1; }
done

if ! cmake --build "$B" -j8 > "$L/S0_2.build.log" 2>&1; then
  echo "S0.2 FAIL build (log $L/S0_2.build.log)"
  exit 1
fi
T="$B/tests/kokkos/test_ale_frame"

# ---- base ----
OMP_NUM_THREADS=4 ctest --test-dir "$B" -LE bench > "$L/S0_2.ctest.log" 2>&1
passed_base=$(grep -E 'Test +#[0-9]+: .* Passed' "$L/ctest.S0_2base.log" | sed -E 's/.*: ([^ ]+) .*/\1/' | sort)
passed_now=$(grep -E 'Test +#[0-9]+: .* Passed' "$L/S0_2.ctest.log" | sed -E 's/.*: ([^ ]+) .*/\1/' | sort)
lost=$(comm -23 <(echo "$passed_base") <(echo "$passed_now") | tr '\n' ' ')
n_base=$(echo "$passed_base" | grep -c .)
n_now=$(echo "$passed_now" | grep -c .)
n_all=$(grep -cE 'Test +#[0-9]+: ' "$L/S0_2.ctest.log")
OMP_NUM_THREADS=1 PYTHONPATH="$B" python python/state_hash.py > "$L/S0_2.state_hash" 2> "$L/S0_2.state_hash.err"
if cmp -s "$L/state_hash.base" "$L/S0_2.state_hash" && [[ -s "$L/S0_2.state_hash" ]]; then
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
  OMP_NUM_THREADS=1 "$T" $n --part a --full > "$L/S0_2.a$n.t1.log" 2>&1
  OMP_NUM_THREADS=8 "$T" $n --part a --full > "$L/S0_2.a$n.t8.log" 2>&1
  na=$((na + $(grep -c '^RESULT a .* PASS' "$L/S0_2.a$n.t1.log" "$L/S0_2.a$n.t8.log" | awk -F: '{s+=$2} END {print s}')))
done
if [[ $na == 8 ]]; then a="PASS(8/8)"; else a="FAIL($na/8)"; fi

# ---- V0.1 (b) ----
D="$L/S0_2.dump"
rm -rf "$D" && mkdir -p "$D/b1" "$D/b8" "$D/g1" "$D/g8a" "$D/g8b"
OMP_NUM_THREADS=1 "$T" 16 --part b --full --dump "$D/b1" > "$L/S0_2.b.t1.log" 2>&1
OMP_NUM_THREADS=8 "$T" 16 --part b --full --dump "$D/b8" > "$L/S0_2.b.t8.log" 2>&1
nb=$(cat "$L/S0_2.b.t1.log" "$L/S0_2.b.t8.log" | grep -c '^RESULT b .* PASS')
shadiff=0
: > "$L/S0_2.b.sha"
for r in b1 b8; do
  for c in B_rk3 D_implicit; do
    for x in U p uf; do
      h1=$(sha256sum < "$D/$r/b_${c}_run1.$x" | cut -c1-64)
      h2=$(sha256sum < "$D/$r/b_${c}_run2.$x" | cut -c1-64)
      echo "$r $c $x $h1 $h2" >> "$L/S0_2.b.sha"
      [[ "$h1" == "$h2" ]] || shadiff=$((shadiff + 1))
    done
  done
done
if [[ $nb == 4 && $shadiff == 0 ]]; then b="PASS(4/4,sha=12/12)"; else b="FAIL($nb/4,shadiff=$shadiff)"; fi

# ---- F ----
OMP_NUM_THREADS=8 "$T" 16 --part f > "$L/S0_2.f.log" 2>&1
if grep -q '^RESULT f .* PASS' "$L/S0_2.f.log"; then f="PASS"; else f="FAIL"; fi

# ---- G0.2 ----
OMP_NUM_THREADS=1 "$T" 32 --part g --full --dump "$D/g1" > "$L/S0_2.g.t1.log" 2>&1
OMP_NUM_THREADS=8 "$T" 32 --part g --full --dump "$D/g8a" --time > "$L/S0_2.g.t8a.log" 2>&1
OMP_NUM_THREADS=8 "$T" 32 --part g --full --dump "$D/g8b" > "$L/S0_2.g.t8b.log" 2>&1
gl="$L/S0_2.g.t1.log $L/S0_2.g.t8a.log $L/S0_2.g.t8b.log"
crit() {  # crit <name>: PASS if every RESULT g line of the three runs has <name>=PASS|INFO
  local tot ok
  tot=$(cat $gl | grep -c '^RESULT g ')
  ok=$(cat $gl | grep '^RESULT g ' | grep -cE " $1=(PASS|INFO)")
  if [[ $tot == 6 && $ok == 6 ]]; then echo PASS; else echo "FAIL($ok/6)"; fi
}
gi=$(crit i); gii=$(crit ii); giii=$(crit iii); giv=$(crit iv)
for r in g1 g8a g8b; do
  (cd "$D/$r" && sha256sum ./g_enforce*) > "$L/S0_2.g.sha.$r"
done
nfiles=$(grep -c . "$L/S0_2.g.sha.g1")
d18=$(diff "$L/S0_2.g.sha.g1" "$L/S0_2.g.sha.g8a" | grep -c '^<')
d88=$(diff "$L/S0_2.g.sha.g8a" "$L/S0_2.g.sha.g8b" | grep -c '^<')
if [[ $nfiles -gt 0 && $d18 == 0 && $d88 == 0 ]]; then gv="PASS(${nfiles}arrays)"; else gv="FAIL(1v8:${d18},8v8:${d88}/${nfiles})"; fi
g="i=$gi,ii=$gii,iii=$giii,iv=$giv,v=$gv"
gverdict=PASS
for x in "$gi" "$gii" "$giii" "$giv" "$gv"; do
  [[ $x == PASS* ]] || gverdict=FAIL
done

verdict=PASS
for x in "$a" "$b" "$f" "$gverdict" "$base"; do
  [[ $x == PASS* ]] || verdict=FAIL
done
echo "S0.2 $verdict V0.1a=$a V0.1b=$b G0.2=$gverdict($g) F=$f base=$base (logs $L/S0_2.*)"
[[ $verdict == PASS ]]
