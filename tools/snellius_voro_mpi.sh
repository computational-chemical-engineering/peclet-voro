#!/usr/bin/env bash
#SBATCH --job-name=voro-mpi
#SBATCH --partition=gpu_h100
#SBATCH --account=tes24005
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=voro-mpi-%j.out
# Voronoi methods plan, rung A4/C5 on Snellius (billed per allocated GPU — this job asks for the 4
# H100 of one node and uses 1, 2, 4 of them in turn): builds voro's MPI tests against the suite's
# nvidia-cuda prefix ON THE NODE, then runs
#   * bench_repair_mpi weak scaling (400k seeds per GPU, 8 repair steps) without and with an SDF
#     scene — the moving-point update path at scale (docs/studies/voro_update_throughput.md),
#   * test_flow_mpi at 4 ranks (RK3, implicit, covolume) — the distributed flow solver on H100s
#     with the device-packed exchange.
# Submit from the suite's voro/ directory: `sbatch tools/snellius_voro_mpi.sh` (no leading VAR=
# assignments — SURF's sbatch drops them; export SUITE beforehand to override).
set -euo pipefail
SUITE="${SUITE:-/projects/0/prjs1022/peclet/suite}"
ENV_SH="$SUITE/../peclet-examples/examples/wall-bounded-turbulence/snellius_env.sh"
source "$ENV_SH"
cd "$SUITE/voro"
echo "== voro $(git rev-parse --short HEAD), core $(git -C ../core rev-parse --short HEAD), morton $(git -C ../morton rev-parse --short HEAD)"
BUILD=build_snellius_mpi
if [[ "${FRESH:-0}" == "1" ]]; then rm -rf "$BUILD"; fi
cmake -S tests/kokkos_mpi -B "$BUILD" -DCMAKE_PREFIX_PATH="$SUITE/extern/install/nvidia-cuda" \
      -DMPIEXEC_EXECUTABLE=/usr/bin/false -DCMAKE_BUILD_TYPE=Release > "$BUILD.cfg.log" 2>&1 || { tail -30 "$BUILD.cfg.log"; exit 1; }
cmake --build "$BUILD" -j 16 > "$BUILD.build.log" 2>&1 || { grep -E "error" "$BUILD.build.log" | head -20; exit 1; }
echo "== built"
export OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores
run() {  # run <ntasks> <binary> <args...>
  local n=$1; shift
  echo "---- np=$n: $*"
  srun --mpi=pmix --ntasks="$n" --gpus-per-task=1 --gpu-bind=per_task:1 "$@"
}
for n in 1 2 4; do
  run "$n" "$BUILD/bench_repair_mpi" $((400000 * n)) 8
  run "$n" "$BUILD/bench_repair_mpi" $((400000 * n)) 8 --sdf
done
run 4 "$BUILD/test_flow_mpi" 24 20
FLOW_MPI_IMPLICIT=1 run 4 "$BUILD/test_flow_mpi" 24 20
FLOW_MPI_COVOLUME=1 run 4 "$BUILD/test_flow_mpi" 24 20
echo "== done"
