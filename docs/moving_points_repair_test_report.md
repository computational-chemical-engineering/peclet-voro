# Moving-Points Repair Test Report (N=1000)

## Scenario

- Particles: 1000
- Domain: periodic unit cube
- Initial positions: uniform random
- Velocities: normal distribution per component, mean 0, sigma 0.25
- Motion model: linear advection with fixed velocity
- Timesteps: 8
- Time step size: 0.01
- Seeds: position 20260327, velocity 20260328

## Method

1. Build initial tessellation.
2. For each timestep:
   - Advect positions with fixed velocity.
   - Wrap positions periodically.
   - Count non-convex cells before update.
   - Run `CellComplex::update`.
   - Count non-convex cells after update.
   - Count cells whose local topology signature changed (used as a repair-work proxy).
3. Build a clean tessellation from final positions.
4. Compare incremental final tessellation vs clean final tessellation by cell id.

## Per-step Statistics

| step | non_convex_before | non_convex_after | topology_changed_cells | no_nbr_cells_after |
|---:|---:|---:|---:|---:|
| 1 | 868 | 1 | 465 | 0 |
| 2 | 849 | 0 | 451 | 0 |
| 3 | 804 | 0 | 435 | 0 |
| 4 | 776 | 1 | 427 | 0 |
| 5 | 760 | 1 | 435 | 0 |
| 6 | 715 | 0 | 380 | 0 |
| 7 | 713 | 0 | 386 | 0 |
| 8 | 695 | 0 | 380 | 0 |

## Aggregate Statistics

- Cumulative non-convex-before count: 6180
- Cumulative topology-changed count (repair proxy): 3359

## Final Incremental vs Clean Comparison

- Signature mismatch cells: 952
- Max absolute volume difference: 1.2562929830660038e-01
- Max relative volume difference: 8.4807563724399387e+01
- Final correctness verdict: **no**

## Interpretation

For this displacement regime, the incrementally updated tessellation is not equivalent to a clean rebuild from the same final positions.

Note on repair metric: the code does not currently expose the exact internal repair-iteration count from `CellComplex::repair`, so `topology_changed_cells` is reported as a practical proxy for repair activity.

## One-Step Timestep Sweep (Boundary Search)

To identify a debugging-friendly case, a one-step sweep was run with the same seeds and velocity distribution.

### Coarse sweep (starting at 0.001)

All tested values were incorrect in one step:

- `dt = 0.001, 0.002, ..., 0.010, 0.012, 0.015, 0.020` -> final correctness `no`

So `dt=0.001` is already in the failing regime.

### Fine sweep below 0.001

| dt | final_correct | non_convex_before | non_convex_after | topology_changed_cells | mismatch_cells | max_rel_vol_diff |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00001 | yes | 0 | 0 | 0 | 0 | 2.0089e-15 |
| 0.00002 | yes | 5 | 0 | 2 | 0 | 1.4756e-15 |
| 0.00005 | no | 11 | 0 | 2 | 4 | 2.0323e-15 |
| 0.00010 | no | 33 | 0 | 10 | 6 | 3.1672e-11 |
| 0.00020 | no | 62 | 0 | 16 | 16 | 8.6595e-10 |
| 0.00050 | no | 122 | 0 | 32 | 32 | 4.3708e-07 |
| 0.00080 | no | 199 | 0 | 48 | 62 | 3.3811e-06 |
| 0.00100 | no | 222 | 0 | 55 | 68 | 7.9421e-06 |

### Boundary found

- Last known correct: `dt = 0.00002`
- First known incorrect: `dt = 0.00005`

### Recommended debugging case for next step

Use **one timestep with `dt = 0.00005`**:

- It is incorrect (so the bug is active).
- Only a small number of cells differ (`mismatch_cells = 4`), which should keep debugging tractable.
