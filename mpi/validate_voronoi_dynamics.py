"""Distributed Voronoi DYNAMICS validated against serial.

Beyond the static tessellation (validate_voronoi.py), this runs the actual compressible-Euler
dynamics distributed: each step migrate owned particles, gather velocity-carrying ghosts one
interaction radius deep, tessellate owned+ghost and advance one step, keep the owned (pos, vel).
The force on an owned cell uses its full neighbourhood (owned or ghost), so the owned trajectory
matches the serial run. Voronoi analogue of packing-gpu/mpi/validate_exact.py.

Run: PYTHONPATH=<voro>/python:<core>/python/build mpirun -np 4 python3 mpi/validate_voronoi_dynamics.py
"""
import os
import sys
import numpy as np
from mpi4py import MPI
from peclet import voro
import peclet.core.mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

L = 6.0
n = 8
# 2-RING halo: the DYNAMICS needs a deeper halo than the tessellation. An owned cell's force uses its
# neighbours' cell *pressures* (= EOS of their volumes), so those neighbour cells must themselves be
# correct -> their neighbours (the 2nd ring) must be present. rcut ~= 2x the 1-ring tessellation depth
# makes the owned dynamics exact (machine precision); rcut=1.2 (1.5 rings) leaves a ~5e-6 force error.
rcut = float(os.environ.get("RCUT", "2.0"))
gs = [16, 16, 16]
nsteps = 6
dt = 2e-3
dens = 1.0
press = 1.0
mass = dens * L ** 3 / (n ** 3)


SOLVER = os.environ.get("SOLVER", "euler")  # "euler" or "ns" (NavierStokes, viscous)


def make_sim(pos, vel):
    s = voro.NavierStokes() if SOLVER == "ns" else voro.ExplicitEuler()
    s.set_l([L, L, L])
    s.set_mass_density(dens)
    s.set_positions(np.ascontiguousarray(pos % L))
    s.set_velocities(np.ascontiguousarray(vel))
    s.set_masses(np.full(pos.shape[0], mass))
    s.set_pressure(press)
    if SOLVER == "ns":
        s.set_viscosity(0.05)
        s.set_bulk_viscosity(0.05)
    return s


xs = (np.arange(n) + 0.5) * (L / n)
g_pos = np.array([[x, y, z] for x in xs for y in xs for z in xs], dtype=np.float64)
g_pos = (g_pos + np.random.RandomState(0).uniform(-0.08, 0.08, g_pos.shape)) % L
g_vel = np.random.RandomState(1).normal(0.0, 0.3, g_pos.shape)
N = g_pos.shape[0]
ids = np.arange(N)

# serial reference (rank 0)
ref = None
if rank == 0:
    s = make_sim(g_pos, g_vel)
    s.init()
    s.step(nsteps, dt)
    ref = np.array(s.get_positions()) % L

# distributed
mig = peclet.core.mpi.Migrator(origin=[0, 0, 0], size=[L, L, L], gsize=gs, periodic=[True, True, True])
own = np.array([mig.owner_of(tuple(p)) for p in g_pos])
mine = np.where(own == rank)[0]
pos, vel, idd = g_pos[mine].copy(), g_vel[mine].copy(), ids[mine].astype(np.float64)
for _ in range(nsteps):
    pay = np.column_stack([vel, idd]) if pos.shape[0] else np.zeros((0, 4))
    pos, pay = mig.migrate(pos, pay)
    vel, idd = pay[:, 0:3].copy(), pay[:, 3].copy()
    n_owned = pos.shape[0]
    if n_owned == 0:
        raise RuntimeError(f"rank {rank} owns 0 particles -- adjust N/np")
    gpos, gpay = mig.gather_ghosts(pos, pay, rcut)
    cpos = np.vstack([pos, gpos]) if gpos.shape[0] else pos
    cvel = np.vstack([vel, gpay[:, 0:3]]) if gpos.shape[0] else vel
    s = make_sim(cpos, cvel)
    s.init()
    s.step(1, dt)
    pos = (np.array(s.get_positions())[:n_owned]) % L
    vel = np.array(s.get_velocities())[:n_owned]

allp = comm.gather(pos, 0)
alli = comm.gather(idd.astype(np.int64), 0)
if rank == 0:
    D = np.full((N, 3), np.nan)
    for p, i in zip(allp, alli):
        D[i] = p
    assert np.isfinite(D).all(), "missing ids"
    d = D - ref
    d = (d + L / 2) % L - L / 2          # periodic-aware difference
    err = np.abs(d)
    maxe, meane = float(err.max()), float(err.mean())
    ok = maxe < 1e-6
    print(f"np={size}: distributed Euler dynamics vs serial, {nsteps} steps, N={N}  "
          f"max|d|={maxe:.3e} mean={meane:.3e}  ({'OK' if ok else 'FAIL'})")
    sys.exit(0 if ok else 1)
