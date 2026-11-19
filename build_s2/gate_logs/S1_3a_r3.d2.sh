#!/bin/bash
# S1.3a-r3 item 5 / D2: rebuild the CUDA tests (--window), rerun the one failing run on the GPU with
# --window 6 under the heavy lock; beside it the host ALE ctests (tests changed, include/ did not)
cd /home/frankp/Codes/suite/voro-ale-s0
export PATH=/usr/local/cuda/bin:$PATH OMP_PROC_BIND=false
cmake --build build_s2_cuda --parallel 8 --target test_ale_stepper test_ale_frame test_ale_operators test_ale_facemesh > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.build_cuda.log 2>&1 || { echo "BUILD FAIL"; echo EXIT; exit 1; }
cmake --build build_s2 --parallel 8 --target test_ale_stepper test_ale_frame test_ale_operators test_ale_facemesh > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.build.log 2>&1 || { echo "HOST BUILD FAIL"; echo EXIT; exit 1; }
( OMP_NUM_THREADS=8 ctest --test-dir build_s2 -R '^test_ale_' > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.ctest.host.log 2>&1 ) &
( OMP_NUM_THREADS=8 ctest --test-dir build_s2_cuda -R '^test_ale_' > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.ctest.gpu.log 2>&1 ) &
mkdir -p /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.d2.tg_Z64_F.dump
{ uptime; OMP_NUM_THREADS=8 flock -w 3600 /home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock build_s2_cuda/tests/kokkos/test_ale_stepper --run tg_Z64_F --motion follow --tend 1 --window 6 --dump-on-error /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.d2.tg_Z64_F.dump; echo "rc=$?"; uptime; } > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.d2.tg_Z64_F.gpu.log 2>&1
rmdir /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.d2.tg_Z64_F.dump 2>/dev/null
wait
echo "D2 $(grep '^RESULT' /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.d2.tg_Z64_F.gpu.log | cut -c1-400)"
echo "ctest host: $(grep 'tests passed' /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.ctest.host.log) gpu: $(grep 'tests passed' /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.ctest.gpu.log)"
echo EXIT
