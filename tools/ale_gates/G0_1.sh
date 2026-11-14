#!/usr/bin/env bash
# Gate G0.1 of the moving-grid design note (docs/ale_design.md §7 WO-S0.1; criteria fixed
# 2026-11-14 in the brief ALE_S0_1): the opt-in FaceMeshOptions of fv::buildFaceMesh.
#
#   tools/ale_gates/G0_1.sh <build dir>
#
# Needs, from the baseline (unmodified tree) in the same build dir:
#   <build dir>/gate_logs/ctest.base.log   (OMP_NUM_THREADS=4 ctest -LE bench)
#   <build dir>/gate_logs/state_hash.base  (OMP_NUM_THREADS=1 python python/state_hash.py)
# Builds the tree (-j8), then
#   (a) every ctest that passed in the baseline passes; state_hash.py output == state_hash.base
#   (b') test_ale_facemesh 32: identities and oracles with the face normal = conn/d, power and
#       Voronoi (RESULT b lines); published-normal sine <= 1e-12 on A_f > 1e-2 h^2 (RESULT b_sine)
#   (c') two builds from the same view bitwise equal at OMP 8, both meshes (RESULT c_sameview);
#       every power-mesh FaceMesh array of the deterministic configurations bitwise identical at
#       OMP_NUM_THREADS 1, 8, 8. The Voronoi comparison across runs is information only
#       (voro_runs=<differing>/<arrays>): the published Voronoi view differs between runs.
#   (d) deterministicOrder alone == the default builder bitwise at OMP_NUM_THREADS=1
# Logs in <build dir>/gate_logs/. Prints ONE line `G0.1' PASS|FAIL a=… b=… c=… d=…`; exit 1 on FAIL.
set -u
B="$(cd "${1:?usage: G0_1.sh <build dir>}" && pwd)"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
L="$B/gate_logs"
mkdir -p "$L"
cd "$SRC"
if [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
fi
export OMP_PROC_BIND=false
for f in "$L/ctest.base.log" "$L/state_hash.base"; do
  [[ -f "$f" ]] || { echo "G0.1' FAIL missing baseline $f"; exit 1; }
done

if ! cmake --build "$B" -j8 > "$L/G0_1.build.log" 2>&1; then
  echo "G0.1' FAIL build (log $L/G0_1.build.log)"
  exit 1
fi

# ---- (a) ----
OMP_NUM_THREADS=4 ctest --test-dir "$B" -LE bench > "$L/G0_1.ctest.log" 2>&1
passed_base=$(grep -E 'Test +#[0-9]+: .* Passed' "$L/ctest.base.log" | sed -E 's/.*: ([^ ]+) .*/\1/' | sort)
passed_now=$(grep -E 'Test +#[0-9]+: .* Passed' "$L/G0_1.ctest.log" | sed -E 's/.*: ([^ ]+) .*/\1/' | sort)
lost=$(comm -23 <(echo "$passed_base") <(echo "$passed_now") | tr '\n' ' ')
n_base=$(echo "$passed_base" | grep -c .)
n_now=$(echo "$passed_now" | grep -c .)
OMP_NUM_THREADS=1 PYTHONPATH="$B" python python/state_hash.py > "$L/G0_1.state_hash" 2> "$L/G0_1.state_hash.err"
if cmp -s "$L/state_hash.base" "$L/G0_1.state_hash" && [[ -s "$L/G0_1.state_hash" ]]; then
  hash_ok=1
else
  hash_ok=0
fi
if [[ -z "${lost// /}" && $hash_ok == 1 ]]; then
  a="PASS(ctest ${n_now}/${n_base}base,hash=same)"
else
  a="FAIL(lost:${lost:-none},hash_ok=${hash_ok})"
fi

# ---- (b), (d) at one thread; (c) dumps at 1, 8, 8 ----
T="$B/tests/kokkos/test_ale_facemesh"
D="$L/G0_1.dump"
rm -rf "$D" && mkdir -p "$D/r1" "$D/r8a" "$D/r8b"
OMP_NUM_THREADS=1 "$T" 32 --dump "$D/r1" > "$L/G0_1.t1.log" 2>&1
rc1=$?
OMP_NUM_THREADS=8 "$T" 32 --dump "$D/r8a" --time > "$L/G0_1.t8a.log" 2>&1
OMP_NUM_THREADS=8 "$T" 32 --dump "$D/r8b" > "$L/G0_1.t8b.log" 2>&1

nb_pass=$(grep -c '^RESULT b .* PASS' "$L/G0_1.t1.log")
nb_sine=$(grep -c '^RESULT b_sine .* PASS' "$L/G0_1.t1.log")
sines=$(grep '^RESULT b_sine' "$L/G0_1.t1.log" | sed -E 's/.*= ([0-9.e+-]+) on.*/\1/' | tr '\n' ',')
nc_same=$(grep -c '^RESULT c_sameview .* PASS' "$L/G0_1.t8a.log")
nd_pass=$(grep -c '^RESULT d .* PASS .*(bitwise CSR)' "$L/G0_1.t1.log")
if [[ $nb_pass == 2 && $nb_sine == 2 ]]; then
  b="PASS"
elif [[ $nb_pass == 2 ]]; then
  b="FAIL(sine:${sines%,})"
else
  b="FAIL"
fi
if [[ $nd_pass == 2 ]]; then d="PASS"; else d="FAIL"; fi
[[ $rc1 == 0 ]] || { b="${b}(rc=$rc1)"; }

# (c'): FaceMesh arrays of the deterministic configurations (*.det.*, *.all.*); the power mesh
# decides, the Voronoi mesh is information
for r in r1 r8a r8b; do
  (cd "$D/$r" && sha256sum ./power.det.* ./power.all.*) > "$L/G0_1.sha.$r"
  (cd "$D/$r" && sha256sum ./voronoi.det.* ./voronoi.all.*) > "$L/G0_1.shav.$r"
done
nfiles=$(grep -c . "$L/G0_1.sha.r1")
diff18=$(diff "$L/G0_1.sha.r1" "$L/G0_1.sha.r8a" | grep -c '^<')
diff88=$(diff "$L/G0_1.sha.r8a" "$L/G0_1.sha.r8b" | grep -c '^<')
nvf=$(grep -c . "$L/G0_1.shav.r1")
vdiff=$( (diff "$L/G0_1.shav.r1" "$L/G0_1.shav.r8a"; diff "$L/G0_1.shav.r8a" "$L/G0_1.shav.r8b") | grep -c '^<')
if [[ $nfiles -gt 0 && $diff18 == 0 && $diff88 == 0 && $nc_same == 2 ]]; then
  c="PASS(${nfiles}arrays,sameview=2/2)"
else
  c="FAIL(1v8:${diff18},8v8:${diff88}/${nfiles},sameview=${nc_same}/2)"
fi

verdict=PASS
for x in "$a" "$b" "$c" "$d"; do
  [[ $x == PASS* ]] || verdict=FAIL
done
echo "G0.1' $verdict a=$a b=$b c=$c d=$d voro_runs=${vdiff}/${nvf} (logs $L/G0_1.*)"
[[ $verdict == PASS ]]
