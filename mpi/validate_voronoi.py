"""Distributed Voronoi tessellation validated against the serial tessellation.

Block-decompose a periodic particle set across ranks (transport-core `tpx_mpi`), gather ghost
particles one interaction radius deep, and have each rank tessellate its owned+ghost set with vordyn,
keeping the OWNED cells. Because the domain is periodic, the ghosts need no special imaging: each rank
tessellates in the full periodic [0,L] box (`put_in_box` wraps the gathered images back to canonical
positions), and the far particles it omits are never neighbours of its owned cells -- so the owned
cells are identical to the serial full-box tessellation (down to machine precision).

This is the Voronoi analogue of packing-gpu/mpi/validate_exact.py. Run:
    PYTHONPATH=<vordyn build>/python:<transport-core>/python/build mpirun -np 4 python3 mpi/validate_voronoi.py
"""
import sys
import numpy as np
from mpi4py import MPI
import vordyn
import tpx_mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

L = 6.0
n = 8                      # 8^3 = 512 particles
rcut = 1.0                 # ghost depth; must exceed the largest owned-cell circumradius
gs = [16, 16, 16]


def tessellate(pos, box=L):
    s = vordyn.ExplicitEuler()
    s.set_l([box, box, box])
    s.set_mass_density(1.0)
    s.set_positions(np.ascontiguousarray(pos))
    s.set_velocities(np.zeros_like(pos))
    s.set_masses(np.full(pos.shape[0], box ** 3 / pos.shape[0]))
    s.set_pressure(1.0)
    if not s.init():
        raise RuntimeError("tessellation init failed")
    return np.array(s.get_volumes()), np.array(s.get_num_neighbors())


# global particle set (perturbed lattice -> a robust, non-degenerate tessellation)
xs = (np.arange(n) + 0.5) * (L / n)
g_pos = np.array([[x, y, z] for x in xs for y in xs for z in xs], dtype=np.float64)
g_pos = (g_pos + np.random.RandomState(0).uniform(-0.1, 0.1, g_pos.shape)) % L
N = g_pos.shape[0]
ids = np.arange(N)

# serial reference (rank 0): full periodic tessellation
vfull = None
if rank == 0:
    vfull, _ = tessellate(g_pos)

# distributed: own a block, gather ghosts, tessellate owned+ghost, keep owned cells
mig = tpx_mpi.Migrator(origin=[0, 0, 0], size=[L, L, L], gsize=gs, periodic=[True, True, True])
own = np.array([mig.owner_of(tuple(p)) for p in g_pos])
mine = np.where(own == rank)[0]
pos = g_pos[mine].copy()
pay = ids[mine].astype(np.float64).reshape(-1, 1)
pos, pay = mig.migrate(pos, pay)                 # canonical ownership (+ periodic wrap)
oid = pay[:, 0].astype(np.int64)
n_owned = pos.shape[0]

gpos, gpay = mig.gather_ghosts(pos, pay, rcut)    # ghost copies (periodic images near the block)
# combine owned + ghosts; the periodic box wraps the ghost images back to canonical positions.
combined = np.vstack([pos, gpos]) if gpos.shape[0] else pos
combined = combined % L                            # put_in_box equivalent
vol, _ = tessellate(combined)
vol_owned = vol[:n_owned]                           # owned cells are the first n_owned

# gather owned (global id, volume) to rank 0 and compare to the serial tessellation
allv = comm.gather(vol_owned, 0)
alli = comm.gather(oid, 0)
n_ghost_tot = comm.reduce(gpos.shape[0], op=MPI.SUM, root=0)
if rank == 0:
    D = np.full(N, np.nan)
    for v, i in zip(allv, alli):
        D[i] = v
    assert np.isfinite(D).all(), "some owned ids missing"
    err = np.abs(D - vfull)
    maxe, meane = float(err.max()), float(err.mean())
    ok = maxe < 1e-9 and abs(D.sum() - L ** 3) < 1e-6 * L ** 3
    print(f"np={size}: N={N}, ghosts(total)={n_ghost_tot}, sum(owned vol)={D.sum():.5f} (box {L**3:.0f})")
    print(f"np={size}: owned-cell volume vs serial  max|d|={maxe:.3e} mean={meane:.3e}  "
          f"({'OK' if ok else 'FAIL'})")
    sys.exit(0 if ok else 1)
