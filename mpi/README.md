# vorflow — distributed tessellation (core halo)

Block-decomposed Voronoi tessellation across MPI ranks, built on the shared `core` halo
(migration + ghost particles) via its `tpx_mpi` Python shim, and the `vorflow` Python module.

```bash
# build vorflow (this repo) and tpx_mpi (core), then:
PYTHONPATH=../build_suite/python:../../core/python/build \
    mpirun -np 4 python3 mpi/validate_voronoi.py
```

- `validate_voronoi.py` — each rank's **owned** Voronoi cells (tessellated from owned + gathered ghosts
  in the full periodic box) match the **serial** tessellation in volume *and* neighbour count to machine
  precision (~1e-15) at np=1/2/4.
- `validate_voronoi_dynamics.py` — 6 steps of distributed **dynamics** vs serial: ExplicitEuler matches
  to machine precision (~4e-15); NavierStokes (`SOLVER=ns`, viscous) to ~3e-9 (OpenMP reduction-order
  noise). NB: the dynamics needs a **2-ring** halo (`rcut≈2×` the tessellation depth) because the force
  uses neighbours' cell pressures; see the doc.

See [../docs/distributed_voronoi.md](../docs/distributed_voronoi.md) for the algorithm and why the
periodic case needs no special imaging or non-periodic tessellation mode.
