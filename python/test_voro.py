#!/usr/bin/env python3
"""Smoke test for the device-native `peclet.voro` nanobind module.

Exercises the two surfaces — the bare Tessellation (cold build + incremental repair) and the
compressible-Euler Simulation — on a small uniform point set and checks the basic invariants
(space-filling volume, plausible neighbour counts, finite energies). Run with the built module
on PYTHONPATH, e.g.:

    PYTHONPATH=build_nb python python/test_vorflow.py
"""
import numpy as np
from peclet import voro


def test_tessellation():
    rng = np.random.default_rng(0)
    N, L = 20_000, 1.0
    pos = rng.random((N, 3)) * L

    t = voro.Tessellation()
    t.set_box((L, L, L))
    t.build(pos)
    assert t.num_particles == N

    vol = t.volumes()
    assert vol.shape == (N,) and vol.dtype == np.float64
    # space-filling: cell volumes sum to the box volume
    assert abs(vol.sum() / L**3 - 1.0) < 1e-9, vol.sum()
    assert (vol > 0).all()

    nbr = t.neighbor_counts()
    assert nbr.shape == (N,) and nbr.dtype == np.int32
    # a 3D Voronoi cell has at least 4 faces; the Poisson mean is ~15.5
    assert nbr.min() >= 4 and 13 < nbr.mean() < 18, (nbr.min(), nbr.mean())

    # move + repair: volumes must stay space-filling, and tiny moves should flag few cells
    last = None
    for _ in range(20):
        pos = (pos + 2e-5 * rng.standard_normal((N, 3))) % L  # ~5e-4 of the spacing per step
        last = t.step(pos)
    assert set(last) == {"flagged", "pass1", "pass2", "rebuilt", "fell_back"}
    assert abs(t.volumes().sum() / L**3 - 1.0) < 1e-9
    assert last["flagged"] < N // 2  # small per-step displacement -> not a full rebuild
    print(f"  Tessellation: N={N}  vol_err={abs(t.volumes().sum()/L**3-1):.1e}  "
          f"mean_nbr={nbr.mean():.2f}  last_step_flagged={last['flagged']}")


def test_simulation():
    rng = np.random.default_rng(1)
    N, L = 4_000, 1.0
    pos = rng.random((N, 3)) * L
    vel = np.zeros((N, 3))
    mass = np.ones(N)

    s = voro.Simulation()
    s.set_l((L, L, L))
    s.set_positions(pos)
    s.set_velocities(vel)
    s.set_masses(mass)
    s.set_pressure(1.0)
    s.init()
    e0 = s.get_kinetic_energy() + s.get_internal_energy()
    s.step(5, 1e-4)
    e1 = s.get_kinetic_energy() + s.get_internal_energy()
    assert np.isfinite(e0) and np.isfinite(e1)
    assert s.get_positions().shape == (N, 3)
    assert abs(s.get_volumes().sum() / L**3 - 1.0) < 1e-9
    print(f"  Simulation:   N={N}  t={s.get_time():.2e}  KE={s.get_kinetic_energy():.3e}  "
          f"IE={s.get_internal_energy():.3e}")


if __name__ == "__main__":
    print(f"peclet.voro execution_space = {voro.execution_space}")
    test_tessellation()
    test_simulation()
    print("peclet.voro python smoke test: PASS")
