#!/usr/bin/env python3
"""Build a periodic random close packing of spheres with peclet.dem (Lubachevsky-Stillinger growth),
then write it as `packing.txt` for the voro volume-optimiser benchmark. Adapted from
peclet-examples/examples/random-packed-bed. The packing is shifted into [0, side)^3 (the voro
tessellator's periodic box convention).

Usage: python pack_bed.py [N] [out.txt]
Output file format:
    <Nsph> <side>
    x y z r         (one line per sphere, Nsph lines; coordinates in [0, side))
"""
import sys, os, math, time
import numpy as np
import dem


def pack_bed(N=180, phi_ref=0.63, radius=0.5, seed=3, dt=0.002, limit_time=7.0, iters=60,
             temperature=1.0, scale_init=0.05, criterion=5e-3, cooling_time=5.0, quench=1200,
             growth_accel=1.02, growth_decay=0.85, growth_rate_init=0.5):
    volp = (4 / 3) * math.pi * radius ** 3
    side = (N * volp / phi_ref) ** (1 / 3)
    half = side / 2.0
    gr = growth_rate_init
    cool = min(int(cooling_time / dt), int(limit_time / dt))
    rng = np.random.default_rng(seed)
    s = dem.Simulation(N)
    s.initialize(shape_type=1, radius=radius)
    s.set_domain((-half, -half, -half), (half, half, half))
    s.enable_periodicity(True, True, True); s.set_gravity(0, 0, 0)
    s.set_material_params(1.0, 1.0, 0.0); s.set_solver_iterations(iters, iters)
    pos = rng.uniform(-half, half, (N, 4)).astype(np.float32); pos[:, 3] = 1.0
    s.set_positions(pos)
    s.set_velocities(rng.normal(0.0, math.sqrt(temperature), (N, 3)).astype(np.float32))
    s.set_scales(np.full(N, 1.0, np.float32))
    s.set_growth_params(gr, scale_init); s.set_thermostat(temperature, dt)

    def overlap_frac():
        c = 2 * radius * float(s.get_scales().ravel().mean())
        return float(s.compute_overlaps()) / max(c, 1e-9)

    for step in range(int(limit_time / dt)):
        if step == cool:
            s.set_material_params(0.5, 1.0, 0.0); s.set_thermostat(0.0, 1e4 * dt)
        s.step(dt); mo = overlap_frac()
        if mo > criterion:
            it = 0
            while True:
                s.step(0.0); it += 1; mn = overlap_frac()
                if mn >= 0.95 * mo and it > 6:
                    break
                mo = mn
            if mo > criterion:
                gf = float(s.get_growth_factor()) * math.exp(-gr * dt)
                gr *= growth_decay; s.set_growth_params(gr, gf)
        else:
            gr = min(gr * growth_accel, growth_rate_init)
            s.set_growth_params(gr, float(s.get_growth_factor()))
    s.set_material_params(0.0, 0.0, 0.0); s.set_thermostat(0.0, 10 * dt)
    for _ in range(quench):
        s.step(dt)
    r_eff = radius * s.get_scales().ravel() * float(s.get_growth_factor())
    pos = s.get_positions()[:, :3].astype(float)
    phi = float(np.sum(4 / 3 * np.pi * r_eff ** 3) / side ** 3)
    return pos, r_eff.astype(float), float(side), phi


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    out = sys.argv[2] if len(sys.argv) > 2 else "packing.txt"
    t0 = time.time()
    pos, r, side, phi = pack_bed(N=N)
    # shift centres from [-half, half) into [0, side) and wrap (periodic)
    half = side / 2.0
    c = (pos + half) % side
    print(f"packing: N={len(pos)}  phi={phi:.3f}  porosity eps={1 - phi:.3f}  "
          f"mean r={r.mean():.4f}  side={side:.3f}  ({time.time() - t0:.0f}s)")
    with open(out, "w") as f:
        f.write(f"{len(pos)} {side:.10g}\n")
        for i in range(len(pos)):
            f.write(f"{c[i, 0]:.10g} {c[i, 1]:.10g} {c[i, 2]:.10g} {r[i]:.10g}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
