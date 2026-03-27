# CellArena Block Capacity Study (Single-Thread, -O3)

## Setup

- Case: random uniform points in unit cube
- Particles: 10,000
- Repetitions per configuration: 10
- Build mode: Release with `-O3 -march=native -mtune=native -ffast-math -funroll-loops`
- Threading: OpenMP disabled at configure time (single-thread)
- Sweep points: 14 reserve pairs
- Raw data: `/home/frankp/Codes/voronoi_dynamics/benchmarks/results/cell_arena_block_study.csv`

## Recommendation

Use **vertex=28, facet=20** as the smallest near-optimal setting under a 5.0% slowdown tolerance.

- Best runtime config:
  vertex=48, facet=32, time=152.765 ms
- Recommended config vs best:
  time=153.083 ms (+0.21%)
- Recommended config vs baseline (32/24):
  time=153.083 ms vs 153.330 ms (-0.16%)
  arena capacity=9570.3 KiB vs 10976.6 KiB (-12.81%)
  peak RSS=140532 kB vs 140480 kB (+0.04%)

## Results

| Reserve (v/f) | Time mean (ms) | Std (ms) | Slowdown vs best | Arena cap (KiB) | Arena slack (%) | Max RSS (kB) |
|---:|---:|---:|---:|---:|---:|---:|
| 28/20 | 153.083 | 0.176 | +0.21% | 9570.3 | 5.63 | 140532 |
| 28/22 | 153.208 | 1.005 | +0.29% | 9687.5 | 6.77 | 139660 |
| 32/20 | 154.181 | 1.294 | +0.93% | 10742.2 | 15.93 | 139352 |
| 16/12 | 153.035 | 0.435 | +0.18% | 10976.4 | 17.72 | 140348 |
| 32/24 | 153.330 | 1.252 | +0.37% | 10976.6 | 17.72 | 140480 |
| 36/24 | 153.188 | 0.402 | +0.28% | 12148.4 | 25.66 | 141068 |
| 18/14 | 153.256 | 0.285 | +0.32% | 12382.6 | 27.06 | 141944 |
| 20/16 | 153.103 | 0.081 | +0.22% | 12850.7 | 29.72 | 140784 |
| 40/28 | 154.859 | 0.573 | +1.37% | 13554.7 | 33.37 | 141504 |
| 22/16 | 153.298 | 1.444 | +0.35% | 14023.3 | 35.60 | 140440 |
| 24/16 | 153.092 | 0.483 | +0.21% | 15195.1 | 40.56 | 140280 |
| 24/18 | 153.663 | 1.115 | +0.59% | 15312.3 | 41.02 | 141192 |
| 24/20 | 153.986 | 1.064 | +0.80% | 15429.5 | 41.47 | 140952 |
| 48/32 | 152.765 | 0.669 | +0.00% | 16132.8 | 44.02 | 139604 |

## Insights

- Runtime is relatively flat across a broad reserve range; very small reserves tend to add modest overhead from reallocation/copy growth.
- Arena capacity scales nearly linearly with reserve settings; this is where memory savings are most directly visible.
- Peak process RSS is less sensitive than arena capacity alone because total process memory includes neighbor lists and other temporary structures.
- Practical tuning rule: choose the smallest pair inside the near-optimal runtime band instead of the absolute fastest pair.

## Robustness Notes

- Median runtime across sweep points: 153.232 ms
- If you want tighter confidence, rerun with higher reps (e.g., 30) and pin CPU frequency/governor.
