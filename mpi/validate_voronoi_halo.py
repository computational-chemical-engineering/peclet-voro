"""Distributed (MPI) Voronoi tessellation validated against the single-rank tessellation.

Uses the C++ ``peclet.voro.VoronoiHalo`` binding (ORB block decomposition + core particle halo):
each rank selects its owned seeds (``owned_mask``), gathers ghost seeds one interaction radius deep
(``gather``), and tessellates its owned+ghost subset with the single-rank ``peclet.voro.Tessellation``
-- keeping the first ``n_owned`` cells (they are bit-identical to the serial full-box cells, since a
Voronoi cell is fully determined by its neighbourhood and every neighbour within ``rcut`` is present).

The per-seed owned-cell volumes are gathered to rank 0 by global id and compared to a single-rank
tessellation of ALL seeds. Run:

    PYTHONPATH=<voro build_mpi> mpirun -np {1,2,4} python3 mpi/validate_voronoi_halo.py
"""
import sys
import numpy as np
from mpi4py import MPI
import peclet.voro as voro

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

L = 1.0
N = 4096                                   # random seeds in the periodic unit box
spacing = (L ** 3 / N) ** (1.0 / 3.0)      # mean inter-seed spacing
rcut = 3.0 * spacing                       # ghost depth; > largest owned-cell interaction distance
gs = [16, 16, 16]                          # ORB decomposition granularity


def tessellate(pos):
    """Single-rank tessellation of `pos` (n,3) in the periodic box -> (n,) volumes."""
    t = voro.Tessellation()
    t.set_box((L, L, L))
    t.build(np.ascontiguousarray(pos % L, dtype=np.float64))
    return np.asarray(t.volumes())


# Identical global seed set on every rank (deterministic).
g_pos = np.random.RandomState(12345).uniform(0.0, L, size=(N, 3)).astype(np.float64)
g_gid = np.arange(N, dtype=np.int64)

# Single-rank reference (rank 0): tessellate the full periodic box.
vref = tessellate(g_pos) if rank == 0 else None

# Distributed: own a block, gather ghosts within rcut, tessellate owned+ghost, keep owned cells.
halo = voro.VoronoiHalo(origin=(0.0, 0.0, 0.0), size=(L, L, L), gsize=gs,
                        periodic=(True, True, True))
mask = np.asarray(halo.owned_mask(g_pos))
mine = np.where(mask == 1)[0]
owned_pos = np.ascontiguousarray(g_pos[mine])
owned_gid = np.ascontiguousarray(g_gid[mine])
owned_w = np.zeros(mine.size, dtype=np.float64)

pos, gid, weight, n_owned = halo.gather(owned_pos, owned_gid, owned_w, rcut)
pos = np.asarray(pos)
gid = np.asarray(gid)
assert n_owned == mine.size, f"n_owned {n_owned} != owned count {mine.size}"

# Tessellate the combined owned+ghost set; the first n_owned cells are this rank's owned cells.
vol = tessellate(pos)
vol_owned = vol[:n_owned]
gid_owned = gid[:n_owned]

# Gather (global id, owned volume) to rank 0 and compare to the single-rank reference.
n_ghost_tot = comm.reduce(pos.shape[0] - n_owned, op=MPI.SUM, root=0)
all_gid = comm.gather(gid_owned, root=0)
all_vol = comm.gather(vol_owned, root=0)

if rank == 0:
    D = np.full(N, np.nan)
    for gg, vv in zip(all_gid, all_vol):
        D[gg] = vv
    assert np.isfinite(D).all(), "some owned global ids missing from the gather"
    err = np.abs(D - vref)
    maxe, meane = float(err.max()), float(err.mean())
    sum_ok = abs(D.sum() - L ** 3) < 1e-9 * L ** 3
    ok = maxe < 1e-7 and sum_ok
    print(f"np={size}: N={N} rcut={rcut:.4f} ({rcut / spacing:.1f} sp) "
          f"ghosts(total)={n_ghost_tot}")
    print(f"np={size}: sum(owned vol)={D.sum():.9f} (box {L ** 3:.1f})  "
          f"max|dvol|={maxe:.3e} mean|dvol|={meane:.3e}  ({'OK' if ok else 'FAIL'})")
    sys.exit(0 if ok else 1)
