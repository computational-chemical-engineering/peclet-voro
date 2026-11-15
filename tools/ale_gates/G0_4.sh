#!/usr/bin/env bash
# Gate G0.4 of the moving-grid branch (work order WO-S0.4; criteria fixed 2026-11-15 in the brief ALE_S0_4): the
# node-label option of core's parallel GraphAMG setup on the pressure operator L_rho (ratio 1 and 1000).
#
#   tools/ale_gates/G0_4.sh <core build dir> <export dir> [<core CUDA build dir>]
#
# <export dir> holds M1_{32,64}.{r1,r1000,drop1000}.{csr,label} (tools/ale_gates/export_lrho.py). The core build
# dir must have test_graph_amg_parallel built; criterion a also reads the reference recorded at core b21d4bd in
# build_s0/gate_logs/G0_4.a_base_{hash.txt,ctest.log} and runs the core battery (`ctest -LE bench`, 2 threads).
#   a  default path: --hash 12/16 lines at 1 thread identical to the reference; ctest pass count >= reference's
#      and no failure
#   b  every labeled run (in-process ctest case, the 32^3/64^3 files): mixed aggregates = cross-label P entries =
#      bad coarse labels = 0 and a uniform label inert
#   c1 iters(r1000) <= 1.5 iters(r1) at 32^3 and 64^3, labels on;  c2 iters(64^3) <= 1.2 iters(32^3), r1 and r1000
#   d  setup <= one solve, 64^3 r1000 labels, 8 threads if the 1-min load + 8 <= 36, else the largest count that fits
#   e  hierarchy hash + iterations identical at 1, 2, 4, 8 threads and twice at 8 (32^3 files, in-process case)
#   f  (CUDA dir given) the in-process ctest gate and the 32^3 files pass b on the GPU; iterations reported
# Logs: build_s0/gate_logs/G0_4.*. Prints ONE line `G0.4 PASS|FAIL a=... b=... c1=... c2=... d=... e=... f=...`.
set -u
CB="$(cd "${1:?usage: G0_4.sh <core build dir> <export dir> [cuda build dir]}" && pwd)"
D="$(cd "${2:?export dir}" && pwd)"
CU="${3:-}"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
L="$SRC/build_s0/gate_logs"
mkdir -p "$L"
T="$CB/tests/test_graph_amg_parallel"
export OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fail=0
iters() { sed -n 's/.* iters=\([0-9]*\) .*/\1/p' "$1"; }

# ---- a: default path
for m in 12 16; do OMP_NUM_THREADS=1 "$T" --hash "$m"; done 2> "$L/G0_4.a_hash.err" | grep -v twophase > "$L/G0_4.a_hash.txt"
if cmp -s "$L/G0_4.a_hash.txt" "$L/G0_4.a_base_hash.txt"; then ah=identical; else ah=DIFF; fi
(cd "$CB" && OMP_NUM_THREADS=2 ctest -LE bench --output-on-failure > "$L/G0_4.a_ctest.log" 2>&1)
pc() { grep -oE '[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+' "$1"; }
passed() { grep -E '^ *[0-9]+/[0-9]+ Test +#[0-9]+: .* Passed' "$1" | awk '{print $4}' | sort; }
np=$(pc "$L/G0_4.a_ctest.log" | awk '{print $NF - $4}'); nt=$(pc "$L/G0_4.a_ctest.log" | awk '{print $NF}')
np0=$(pc "$L/G0_4.a_base_ctest.log" | awk '{print $NF - $4}'); nt0=$(pc "$L/G0_4.a_base_ctest.log" | awk '{print $NF}')
# every test that passed at the reference passes now (a reference failure, e.g. an np8 timeout under load, is named)
lost=$(comm -23 <(passed "$L/G0_4.a_base_ctest.log") <(passed "$L/G0_4.a_ctest.log") | tr '\n' ' ')
a="hash:$ah,ctest:${np:-?}/${nt:-?}(ref ${np0:-?}/${nt0:-?})"
[[ "$ah" == identical && -n "${np:-}" && -z "$lost" ]] || { a="FAIL($a,lost:${lost:-none})"; fail=1; }

# ---- b, c: labeled runs at 4 threads (the in-process case is the ctest gate)
OMP_NUM_THREADS=4 "$T" > "$L/G0_4.b_inproc.log" 2>&1
bin=$?
for n in 32 64; do
  for r in r1 r1000; do OMP_NUM_THREADS=4 "$T" --file "$D/M1_$n.$r" > "$L/G0_4.bc_${n}_$r.log" 2>&1; done
  OMP_NUM_THREADS=4 "$T" --file "$D/M1_$n.r1000" --nolabel > "$L/G0_4.ctl_${n}_r1000_nolabel.log" 2>&1
done
bok=1
[[ $bin -eq 0 ]] || bok=0
for f in "$L"/G0_4.bc_*.log; do grep -q 'mixedAgg=0 crossP=0 badCoarse=0 uniformInert=1 ok' "$f" || bok=0; done
b=$([[ $bok == 1 ]] && echo "0/0/0,inert" || { fail=1; echo FAIL; })
i32a=$(iters "$L/G0_4.bc_32_r1.log"); i32b=$(iters "$L/G0_4.bc_32_r1000.log")
i64a=$(iters "$L/G0_4.bc_64_r1.log"); i64b=$(iters "$L/G0_4.bc_64_r1000.log")
c1=$(awk -v a="$i32a" -v b="$i32b" -v c="$i64a" -v d="$i64b" 'BEGIN{ok=(b<=1.5*a && d<=1.5*c); printf "%s(32:%d/%d,64:%d/%d)", ok?"ok":"FAIL", b,a,d,c}')
c2=$(awk -v a="$i32a" -v b="$i32b" -v c="$i64a" -v d="$i64b" 'BEGIN{ok=(c<=1.2*a && d<=1.2*b); printf "%s(r1:%d->%d=%.2f,r1000:%d->%d=%.2f)", ok?"ok":"FAIL", a,c,c/a,b,d,d/b}')
[[ $c1 == ok* ]] || fail=1
[[ $c2 == ok* ]] || fail=1

# ---- d: setup vs one solve, 64^3 r1000 labels
load=$(awk '{printf "%d", $1 + 0.999}' /proc/loadavg)
thr=8
while (( thr > 1 && load + thr > 36 )); do thr=$((thr / 2)); done
OMP_NUM_THREADS=$thr "$T" --file "$D/M1_64.r1000" > "$L/G0_4.d.log" 2>&1
ts=$(sed -n 's/.* setup_s=\([0-9.]*\) .*/\1/p' "$L/G0_4.d.log"); tv=$(sed -n 's/.* solve_s=\([0-9.]*\) .*/\1/p' "$L/G0_4.d.log")
d=$(awk -v s="$ts" -v v="$tv" -v t="$thr" -v l="$load" 'BEGIN{printf "%s(setup %.3fs/solve %.3fs,thr %d,load %d)", (s<=v)?"ok":"FAIL", s, v, t, l}')
[[ $d == ok* ]] || fail=1

# ---- e: thread-count identity
: > "$L/G0_4.e.log"
for t in 1 2 4 8 8; do
  for r in r1 r1000; do
    OMP_NUM_THREADS=$t "$T" --file "$D/M1_32.$r" | grep '^file' | sed -E 's/.* (iters=[0-9]+) .* (hash=[0-9a-f]+) .*/'"$r"' \1 \2/' >> "$L/G0_4.e.log"
  done
  OMP_NUM_THREADS=$t "$T" --hash 20 | grep twophase | sed 's/^/inproc /' >> "$L/G0_4.e.log"
  OMP_NUM_THREADS=$t "$T" | grep -E '^  twophase m=20' | sed -E 's/.*ratio ([0-9]+), labels: .*iters ([0-9]+).*/inproc ratio\1 iters=\2/' >> "$L/G0_4.e.log"
done
distinct=$(sort "$L/G0_4.e.log" | uniq -c | awk '$1 != 5' | wc -l)
nkeys=$(sort -u "$L/G0_4.e.log" | wc -l)
e=$([[ $distinct == 0 && $nkeys == 5 ]] && echo "identical(1,2,4,8,8)" || { fail=1; echo "FAIL(lines not 5x identical: $distinct)"; })

# ---- f: device
f=skipped
if [[ -n "$CU" ]]; then
  TC="$CU/tests/test_graph_amg_parallel"
  "$TC" > "$L/G0_4.f_inproc.log" 2>&1
  fin=$?
  fok=1
  [[ $fin -eq 0 ]] || fok=0
  fwhy=""
  for r in r1 r1000; do
    "$TC" --file "$D/M1_32.$r" > "$L/G0_4.f_32_$r.log" 2>&1
    grep -q 'mixedAgg=0 crossP=0 badCoarse=0 uniformInert=1 ok' "$L/G0_4.f_32_$r.log" || fok=0
    grep -q 'failed to allocate' "$L/G0_4.f_32_$r.log" && fwhy="32^3 setup: device allocation failed"
  done
  grep -q 'steps 5-7 bitwise vs host-order reference: FAIL' "$L/G0_4.f_inproc.log" &&
    fwhy="${fwhy:+$fwhy;}steps 5-7 not bitwise on device"
  gi=$(grep -E '^  twophase m=20 ratio 1000' "$L/G0_4.f_inproc.log" | sed -E 's/.*iters ([0-9]+).*/\1/')
  f=$([[ $fok == 1 ]] && echo "ok" || echo "FAIL")
  bcl=$(grep -cE '^  twophase m=20 .*mixed aggregates 0, cross-label P entries 0, bad coarse labels 0' "$L/G0_4.f_inproc.log")
  f="$f(inproc b1-b3 clean ${bcl}/2,inproc r1000 iters ${gi:-?},32^3 iters r1 $(iters "$L/G0_4.f_32_r1.log")/r1000 $(iters "$L/G0_4.f_32_r1000.log")${fwhy:+;$fwhy})"
  [[ $fok == 1 ]] || fail=1
fi

line="G0.4 $([[ $fail == 0 ]] && echo PASS || echo FAIL) a=$a b=$b c1=$c1 c2=$c2 d=$d e=$e f=$f"
echo "$line" | tee "$L/G0_4.out"
exit $fail
