"""Smoke test / demo for the vorflow Python module (the first Python surface for vorflow).

Exercises the three solver classes on a jittered periodic grid: build the tessellation (init),
advance the dynamics (step), and round-trip particle state through numpy. Run:

    cmake -B build_suite -DVORONOI_BUILD_PYTHON=ON && cmake --build build_suite --target vorflow -j
    PYTHONPATH=build_suite/python python3 python/test_vorflow.py
"""
import sys
import numpy as np
import vorflow

L = 6.0
n = 6  # 6^3 = 216 particles


def jittered_grid(seed=0, jitter=0.15):
    rng = np.random.RandomState(seed)
    xs = (np.arange(n) + 0.5) * (L / n)
    P = np.array([[x, y, z] for x in xs for y in xs for z in xs], dtype=np.float64)
    P = (P + rng.uniform(-jitter, jitter, P.shape)) % L
    return P


pos = jittered_grid()
N = pos.shape[0]
vel = np.random.RandomState(1).normal(0.0, 0.1, (N, 3))
masses = np.full(N, (L ** 3) / N, dtype=np.float64)  # density 1 x mean cell volume
fails = 0


def check(name, cond):
    global fails
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    fails += 0 if cond else 1


# ---- ExplicitEuler ----
print("ExplicitEuler:")
s = vorflow.ExplicitEuler()
s.set_l([L, L, L])
s.set_mass_density(1.0)
s.set_positions(pos)
s.set_velocities(vel)
s.set_masses(masses)
s.set_pressure(1.0)
ok = s.init()
check("init() succeeds", ok)
ke0 = s.get_kinetic_energy()
s.step(10, 1e-3)
p1 = np.array(s.get_positions())
check("positions finite after step", np.isfinite(p1).all())
check("particle count preserved", p1.shape == (N, 3))
check("particles moved", float(np.abs(p1 - pos).max()) > 0)
vol = np.array(s.get_volumes())
nbrs = np.array(s.get_num_neighbors())
check("cell volumes tile the box (sum == L^3)", abs(vol.sum() - L ** 3) < 1e-6 * L ** 3)
check("every particle has a cell", (vol > 0).all())
check("neighbour counts are sane (>=4)", (nbrs >= 4).all())
edges = np.array(s.get_neighbor_pairs())
eset = set(map(tuple, edges))
check("connectivity matches neighbour counts", len(edges) == int(nbrs.sum()))
check("Voronoi adjacency is symmetric", all((j, i) in eset for i, j in edges))
print(f"    KE {ke0:.4g} -> {s.get_kinetic_energy():.4g}, U={s.get_internal_energy():.4g}, "
      f"sum(vol)={vol.sum():.4f}(box {L ** 3:.0f}), mean nbrs={nbrs.mean():.2f}")

# ---- NavierStokes (adds viscosity) ----
print("NavierStokes:")
ns = vorflow.NavierStokes()
ns.set_l([L, L, L])
ns.set_mass_density(1.0)
ns.set_positions(pos)
ns.set_velocities(vel)
ns.set_masses(masses)
ns.set_pressure(1.0)
ns.set_viscosity(0.05)
ns.set_bulk_viscosity(0.05)
check("init() succeeds", ns.init())
ns.step(10, 1e-3)
check("KE finite after viscous step", np.isfinite(ns.get_kinetic_energy()))
print(f"    KE -> {ns.get_kinetic_energy():.4g}")

# ---- IntfDyn (two-phase interface tension) ----
print("IntfDyn:")
ifd = vorflow.IntfDyn()
ifd.set_l([L, L, L])
ifd.set_mass_density(1.0)
types = (pos[:, 0] > L / 2).astype(np.uint8)  # split the box into two phases
ifd.set_positions(pos)
ifd.set_velocities(np.zeros((N, 3)))
ifd.set_masses(masses)
ifd.set_pressure(1.0)
ifd.set_viscosity(0.05)
ifd.set_types(types)
ifd.set_intf_tension(0.1, 0, 1)
check("init() succeeds", ifd.init())
e_intf = ifd.get_intf_energy()
ifd.step(10, 1e-3)
check("interface energy finite", np.isfinite(e_intf))
check("KE finite after step", np.isfinite(ifd.get_kinetic_energy()))
print(f"    interface energy={e_intf:.4g}, KE -> {ifd.get_kinetic_energy():.4g}")

print(f"\n{'OK' if fails == 0 else 'FAIL'}: vorflow smoke test ({fails} failures)")
sys.exit(0 if fails == 0 else 1)
