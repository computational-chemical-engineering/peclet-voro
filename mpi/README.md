# voronoi_dynamics — distributed tessellation (transport-core halo)

Block-decomposed Voronoi tessellation across MPI ranks, built on the shared `transport-core` halo
(migration + ghost particles) via its `tpx_mpi` Python shim, and the `vordyn` Python module.

```bash
# build vordyn (this repo) and tpx_mpi (transport-core), then:
PYTHONPATH=../build_suite/python:../../transport-core/python/build \
    mpirun -np 4 python3 mpi/validate_voronoi.py
```

`validate_voronoi.py` checks that each rank's **owned** Voronoi cells (tessellated from its owned +
gathered ghost particles, in the full periodic box) match the **serial** full-box tessellation. They
agree to machine precision (max ~1e-15) at np=1/2/4.

See [../docs/distributed_voronoi.md](../docs/distributed_voronoi.md) for the algorithm and why the
periodic case needs no special imaging or non-periodic tessellation mode.
