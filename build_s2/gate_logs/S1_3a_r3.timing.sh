#!/bin/bash
# S1.3a-r3 item 5: time per frame of test_ale_frame --part g --time (host, 8 threads) at gather
# windows 4, 5, 6; 32^3 and 64^3; uptime before and after each (shared host: not a benchmark)
cd /home/frankp/Codes/suite/voro-ale-s0
while kill -0 $(cat /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.dumps32.pid) 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false
for n in 32 64; do for w in 4 5 6 4; do
  log=/home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.time.n$n.w$w.host.log
  { uptime; build_s2/tests/kokkos/test_ale_frame $n --part g --time --window $w; echo "rc=$?"; uptime; } >> $log 2>&1
  echo "n=$n w=$w $(grep '^TIME g' $log | tail -n 2 | sed 's/.*per-frame ms: //' | tr '\n' '|') load=$(grep -o 'load average: [0-9.]*' $log | tail -n 2 | tr '\n' ' ') $(grep -c 'RESULT.*FAIL' $log) FAIL lines"
done; done
echo EXIT
