#!/usr/bin/env python3
"""
Step 1 of the packed-bed interstitial-Voronoi example.

Uses peclet.dem to grow a random sphere packing in a cubic box, then scatters random points and
keeps only those in the INTERSTITIAL space (outside every sphere, i.e. union-of-balls SDF > margin).
Writes `packing.txt` for the C++ tessellator and `packing.npz` for the plotter.

Run (dem module built at ../../../dem/build):
    PYTHONPATH=../../../dem/build python3 pack_and_seed.py
"""
import os
import sys
import numpy as np

try:
    import dem  # peclet.dem, built as `dem` in dem/build
except ImportError:
    from peclet import dem

HERE = os.path.dirname(os.path.abspath(__file__))


def make_packing(n=250, r=0.44, phi=0.35, iters=150, dt=0.01, steps=1000, seed=1):
    """Settle `n` equal spheres of radius `r` into a non-overlapping PERIODIC packing at solid
    fraction `phi`. The domain side is sized from (n, r, phi); dem's XPBD position solver relaxes the
    random (mildly overlapping) initial state. Periodicity confines the particles to the box (a plain
    domain does not). Returns centres, radii, half."""
    half = 0.5 * (n * (4.0 / 3.0 * np.pi * r**3) / phi) ** (1.0 / 3.0)

    sim = dem.Simulation(n)
    sim.initialize(shape_type=1, radius=r)           # 1 = sphere
    sim.set_domain((-half, -half, -half), (half, half, half))
    sim.enable_periodicity(True, True, True)         # confine the packing to the box
    sim.set_gravity(0.0, 0.0, 0.0)
    sim.set_material_params(0.0, 0.0, 0.0)           # inelastic — settle
    sim.set_solver_iterations(iters, iters)

    rng = np.random.default_rng(seed)
    pos = rng.uniform(-half, half, (n, 4)).astype(np.float32)
    pos[:, 3] = 1.0
    sim.set_positions(pos)
    sim.set_velocities(np.zeros((n, 4), np.float32))
    sim.set_scales_uniform(1.0)                       # fixed radius (no growth)

    for _ in range(steps):
        sim.step(dt)

    centres = np.asarray(sim.get_positions())[:, :3].astype(np.float64)
    radii = np.full(n, r, np.float64)
    solid = (radii**3).sum() * 4.0 / 3.0 * np.pi / (2 * half) ** 3
    print(f"packing: n={n} phi={solid:.3f} r={r} half={half:.3f} "
          f"maxOverlap={sim.get_max_overlap():.2e} out={(np.abs(centres) > half).any(1).sum()}")
    return centres, radii, half


def union_sdf(pts, centres, radii):
    """min_i(|x-c_i| - r_i): <0 inside any ball, >0 in the fluid."""
    d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2) - radii[None, :]
    return d.min(axis=1)


def scatter_interstitial(centres, radii, half, n_try=6000, margin=0.008, seed=7):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-half, half, (n_try, 3))
    keep = union_sdf(pts, centres, radii) > margin       # strictly in the fluid
    seeds = pts[keep]
    print(f"seeds: kept {len(seeds)}/{n_try} interstitial points (margin={margin})")
    return seeds


def main():
    centres, radii, half = make_packing()
    seeds = scatter_interstitial(centres, radii, half)

    # packing.txt for the C++ tessellator: "Nsph Nseed lo hi", then spheres, then seeds.
    with open(os.path.join(HERE, "packing.txt"), "w") as f:
        f.write(f"{len(centres)} {len(seeds)} {-half:.10g} {half:.10g}\n")
        for c, r in zip(centres, radii):
            f.write(f"{c[0]:.10g} {c[1]:.10g} {c[2]:.10g} {r:.10g}\n")
        for s in seeds:
            f.write(f"{s[0]:.10g} {s[1]:.10g} {s[2]:.10g}\n")
    np.savez(os.path.join(HERE, "packing.npz"), centres=centres, radii=radii,
             seeds=seeds, half=half)
    print(f"wrote packing.txt ({len(centres)} spheres, {len(seeds)} seeds) and packing.npz")


if __name__ == "__main__":
    main()
