# peclet.voro — distributed tessellation check (core halo)

Block-decomposed Voronoi tessellation across MPI ranks through the C++ `peclet.voro.VoronoiHalo`
binding (core ORB decomposition + particle halo), in an MPI build of `peclet.voro`
(`-DPECLET_VORO_MPI=ON`):

```bash
PYTHONPATH=<voro build_mpi> mpirun -np 4 python3 mpi/validate_voronoi_halo.py
```

- `validate_voronoi_halo.py` — each rank's **owned** Voronoi cells (tessellated from owned + gathered
  ghost seeds) match the **single-rank** tessellation of all seeds, gathered to rank 0 by global id, at
  np = 1, 2, 4.

The distributed engine itself is gated by the C++ ctests in `tests/kokkos_mpi` (np = 1, 2, 4).

The three pre-Kokkos Python drivers that first validated the scheme (static tessellation, dynamics,
and the scheme-C force-forward vs re-gather comparison) were written against the retired
`ExplicitEuler` API and were removed; their measured results are recorded in
[../docs/distributed_voronoi.md](../docs/distributed_voronoi.md), and the code is in git history at
`59d7e21` (`git show 59d7e21:mpi/validate_voronoi_scheme_c.py`).
