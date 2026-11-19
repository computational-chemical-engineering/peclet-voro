#!/bin/bash
# S1.3a-r3 item 1: GPU replay (with --facets) of the failing frame, under the shared heavy lock
cd /home/frankp/Codes/suite/voro-ale-s0
export PATH=/usr/local/cuda/bin:$PATH OMP_NUM_THREADS=8 OMP_PROC_BIND=false
cmake --build build_s2_cuda --parallel 8 --target test_ale_frame > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.build_cuda.log 2>&1 || { echo "BUILD FAIL"; echo EXIT; exit 1; }
flock -w 3600 /home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock build_s2_cuda/tests/kokkos/test_ale_frame --replay /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r2.f2.tg_Z64_F.1.dump --repeat 1 --facets > /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.facets.gpu.log 2>&1
echo "rc=$? $(grep -E '^REPLAY |REPLAYREPORT' /home/frankp/Codes/suite/voro-ale-s0/build_s2/gate_logs/S1_3a_r3.facets.gpu.log | tr '\n' ' ')"
echo EXIT
