#!/bin/bash
# S1.3a-r3 item 4: final-frame dumps of tg_Z32_F --motion follow at t = 0.45 and t = 1 (host, 8 threads)
cd /home/frankp/Codes/suite/voro-ale-s0
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false
for te in 0.45 1; do
  mkdir -p /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.final.tg_Z32_F.t$te.dump
  build_s2/tests/kokkos/test_ale_stepper --run tg_Z32_F --motion follow --tend $te --dump-final /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.final.tg_Z32_F.t$te.dump > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.final.tg_Z32_F.t$te.host.log 2>&1
  echo "t=$te rc=$? $(grep ^RESULT /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.final.tg_Z32_F.t$te.host.log | cut -c1-200)"
done
echo EXIT
