#!/bin/bash
# Run test_determinism at each OpenMP thread count in THREADS and require every HASH / FRAME line
# to be identical to the first count's. One summary line; exit 1 on a hash mismatch or a failed
# run (the binary exits nonzero when a partition guard check fails).
#   determinism_check.sh <binary> "<threads...>" [binary args...]
set -u
bin=$1; threads=$2; shift 2
dir=$(mktemp -d); trap 'rm -rf "$dir"' EXIT
rc=0; ref=""; first=""
for t in $threads; do
  out=$dir/omp$t.log
  OMP_NUM_THREADS=$t OMP_PROC_BIND=false "$bin" "$@" > "$out" 2>&1 || rc=1
  grep -E '^(HASH|FRAME) ' "$out" > "$dir/omp$t.hash"
  grep -E '^(RESULT|GUARD) ' "$out" | sed "s/^/[omp $t] /"
  if [ -z "$ref" ]; then ref=$dir/omp$t.hash; first=$t; continue; fi
  nd=$(diff "$ref" "$dir/omp$t.hash" | grep -c '^>')
  if [ "$nd" -ne 0 ] || [ ! -s "$ref" ]; then
    rc=1
    echo "MISMATCH omp $first vs $t: $nd lines differ, e.g.:"
    diff "$ref" "$dir/omp$t.hash" | grep '^[<>]' | head -6
  fi
done
echo "DETERMINISM $([ $rc -eq 0 ] && echo PASS || echo FAIL) threads=[$threads] lines=$(wc -l < "$ref")"
exit $rc
