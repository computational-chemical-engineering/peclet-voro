# Running the SOTA GPU Voronoi (Liu et al. 2020) on current hardware

Reproduces the SOTA per-phase GPU numbers in `docs/voronoi_construct_ledger.md`
(RTX 5080: gather 17.5 Msites/s, construct 9.5 Mcells/s, 1M sites, K=90).

Code: Liu/Ma/Guo/Yan, "Parallel Computation of 3D Clipped Voronoi Diagrams", TVCG 2020.
Repo: https://github.com/xh-liu-tech/3D-Voronoi-GPU  (ConvexCell + grid-kNN, CUDA).

## Build on CUDA 13.2 / sm_120 (Blackwell)
```bash
git clone --depth 1 https://github.com/xh-liu-tech/3D-Voronoi-GPU.git voro_gpu
cd voro_gpu
# 1) arch:  CMakeLists.txt  OPTIONS "-arch sm_61 -lineinfo"  ->  "-arch=sm_120 -lineinfo"
# 2) cublas: link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
#            -> link_directories(/usr/local/cuda-13.2/targets/x86_64-linux/lib)
# 3) test_voronoi.cu printDevProp(): comment the 3 lines using devProp.clockRate /
#    .deviceOverlap / .kernelExecTimeoutEnabled  (removed in CUDA 13)
mkdir build && cd build && cmake .. && make
```

## Timing patch (isolate gather vs construct)
In `voronoi.cu`, the non-transposed (n>14000) branch, wrap `kn_solve(kn)` and add an
isolated `voro_cell_test_GPU_param` run, each with cudaEvent timing + a `return` after
the construct (the restricted/tet-clip path crashes on a coarse box domain and is not the
comparable phase). See the ledger for the exact prints (`@@ GATHER` / `@@ CONSTRUCT`).

## Input: 1M uniform sites in a unit-cube tet domain
```bash
python3 ../gen_box.py        # writes box.tet (unit cube -> [0,1000]^3, 48k tets) + box_sites.xyz (1M)
./bin/VolumeVoronoiGPU box.tet box_sites.xyz 1   # K from params.h PRESET 0 (white noise) = 90
```
