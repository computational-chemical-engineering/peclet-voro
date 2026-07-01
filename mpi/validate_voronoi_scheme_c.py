"""Scheme C (force-communication) vs re-gather, for distributed Voronoi dynamics + MPI profiling.

Two distributed schemes for the same compressible-Euler dynamics, both compared to the serial run:

  RE-GATHER (replicate): every step migrate + gather_ghosts (full ghost STATE: pos+vel) and rebuild
      the local tessellation from scratch, integrate owned, discard ghosts.

  SCHEME C (force communication): build a PERSISTENT owner<->ghost halo once per `rebuild_every` steps;
      hold one local Simulation over owned+ghost; each step do velocity-Verlet on owned+ghost, but
      FORWARD the owner forces onto their ghost copies and integrate the ghosts locally with them, so
      ghosts track their owners with no per-step state re-gather. (For Voronoi the owned-cell force is
      already complete from the local closure, so the reverse contributes zero; only the forward is
      needed -- one force vector/step over a fixed topology, plus an incremental tessellation update
      instead of a full rebuild.)

Reports, for each scheme: max owned-position error vs serial, and the MPI/communication time and total
wall time per step. Run:
  PYTHONPATH=<vorflow>/python:<transport-core>/python/build mpirun -np 4 python3 mpi/validate_voronoi_scheme_c.py
"""
import os
import sys
import numpy as np
from mpi4py import MPI
from peclet import voro as vorflow
import tpx_mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
Wt = MPI.Wtime

L = 6.0
n = 8
dt = 2e-3
nsteps = 40
rcut = 2.0           # 2-ring halo (exact dynamics)
skin = 0.5           # extra so the ghost set stays valid between rebuilds
rebuild_every = int(os.environ.get("REBUILD", "10"))
gs = [16, 16, 16]
dens, press = 1.0, 1.0
mass = dens * L ** 3 / (n ** 3)

xs = (np.arange(n) + 0.5) * (L / n)
g_pos = np.array([[x, y, z] for x in xs for y in xs for z in xs], dtype=np.float64)
g_pos = (g_pos + np.random.RandomState(0).uniform(-0.08, 0.08, g_pos.shape)) % L
g_vel = np.random.RandomState(1).normal(0.0, 0.3, g_pos.shape)
N = g_pos.shape[0]
ids = np.arange(N)


def make_sim(pos, vel):
    s = vorflow.ExplicitEuler()
    s.set_l([L, L, L])
    s.set_mass_density(dens)
    s.set_positions(np.ascontiguousarray(pos % L))
    s.set_velocities(np.ascontiguousarray(vel))
    s.set_masses(np.full(pos.shape[0], mass))
    s.set_pressure(press)
    s.init()
    return s


# serial reference
ref = None
if rank == 0:
    s = make_sim(g_pos, g_vel)
    s.step(nsteps, dt)
    ref = np.array(s.get_positions()) % L


def own_initial():
    mig = tpx_mpi.Migrator(origin=[0, 0, 0], size=[L, L, L], gsize=gs, periodic=[True, True, True])
    o = np.array([mig.owner_of(tuple(p)) for p in g_pos])
    m = np.where(o == rank)[0]
    return mig, g_pos[m].copy(), g_vel[m].copy(), ids[m].astype(np.float64)


def run_regather():
    mig, pos, vel, idd = own_initial()
    t_mpi = 0.0
    for _ in range(nsteps):
        t0 = Wt()
        pos, pay = mig.migrate(pos, np.column_stack([vel, idd]))
        vel, idd = pay[:, 0:3].copy(), pay[:, 3].copy()
        no = pos.shape[0]
        gpos, gpay = mig.gather_ghosts(pos, np.column_stack([vel, idd]), rcut)
        t_mpi += Wt() - t0
        cpos = np.vstack([pos, gpos]) if gpos.shape[0] else pos
        cvel = np.vstack([vel, gpay[:, 0:3]]) if gpos.shape[0] else vel
        s = make_sim(cpos, cvel)
        s.step(1, dt)
        pos = np.array(s.get_positions())[:no] % L
        vel = np.array(s.get_velocities())[:no]
    return pos, idd, t_mpi


def run_scheme_c():
    mig, pos, vel, idd = own_initial()
    halo = tpx_mpi.Halo(origin=[0, 0, 0], size=[L, L, L], gsize=gs, periodic=[True, True, True])
    t_mpi = 0.0
    s = None
    x = v = F = None
    no = 0
    for step in range(nsteps):
        if step % rebuild_every == 0:
            # rebuild: canonical ownership + persistent halo + gather ghost state once
            t0 = Wt()
            pos, pay = mig.migrate(pos, np.column_stack([vel, idd]))
            vel, idd = pay[:, 0:3].copy(), pay[:, 3].copy()
            no = pos.shape[0]
            halo.build(pos, rcut + skin)
            gpos = halo.forward_positions(pos)
            gvel = halo.forward(vel)
            t_mpi += Wt() - t0
            s = make_sim(np.vstack([pos, gpos]) if gpos.shape[0] else pos,
                         np.vstack([vel, gvel]) if gpos.shape[0] else vel)
            x = (np.vstack([pos, gpos]) if gpos.shape[0] else pos) % L
            v = np.vstack([vel, gvel]) if gpos.shape[0] else vel
            F = np.array(s.get_forces())
        # velocity-Verlet on owned+ghost, with the owner-force forward as the only exchange
        v = v + F / mass * (dt / 2)
        x = (x + v * dt) % L
        s.set_positions(x)
        s.recompute_forces()
        F = np.array(s.get_forces())
        t0 = Wt()
        F[no:] = halo.forward(F[:no])          # forward owner forces -> ghosts (scheme C exchange)
        t_mpi += Wt() - t0
        v = v + F / mass * (dt / 2)
        pos, vel = x[:no] % L, v[:no]
    return pos, idd, t_mpi


def gather_err(pos, idd, label, t_mpi, t_tot):
    allp = comm.gather(pos, 0)
    alli = comm.gather(idd.astype(np.int64), 0)
    tmpi_max = comm.reduce(t_mpi, op=MPI.MAX, root=0)
    if rank == 0:
        D = np.full((N, 3), np.nan)
        for p, i in zip(allp, alli):
            D[i] = p
        d = (D - ref + L / 2) % L - L / 2
        print(f"  {label:9s}: max|d vs serial|={np.abs(d).max():.3e}  "
              f"MPI/step={1e3 * tmpi_max / nsteps:.3f} ms  total/step={1e3 * t_tot / nsteps:.3f} ms")


comm.Barrier(); t0 = Wt(); pr, ir, tm = run_regather(); comm.Barrier(); treg = Wt() - t0
comm.Barrier(); t0 = Wt(); pc, ic, tc = run_scheme_c(); comm.Barrier(); tscm = Wt() - t0
if rank == 0:
    print(f"np={size}: N={N}, {nsteps} steps, scheme-C rebuild_every={rebuild_every}")
gather_err(pr, ir, "re-gather", tm, treg)
gather_err(pc, ic, "scheme-C", tc, tscm)
