#!/usr/bin/env python3
"""Fixed-seed reference runs of every public entry path of `peclet.voro`, each printing the
SHA-256 of its final float64 state — the numerics gate of an API change (suite/docs/QUALITY_PLAN.md
§3.F: a tiering or renaming pass must reproduce every hash of the run before it).

Run at one thread (the tessellator's cold build is thread-order-independent, but keep the gate
strict) with the built module on PYTHONPATH:

    OMP_NUM_THREADS=1 OMP_PROC_BIND=false PYTHONPATH=<build> python python/state_hash.py

The single-process paths are hashed in this process; the MPI paths (`VoronoiHalo`,
`DistributedTessellation`) are run at np=2 by re-launching this script under mpirun with `--mpi`
(skipped, with a note, when the module was built without PECLET_VORO_MPI or mpirun is missing).
"""
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
from peclet import voro


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def report(name, *arrays):
    print(f"{name:34s} {sha(*arrays)}", flush=True)


def sphere_scene(center, radius):
    """One solid sphere in the flat node encoding of Tessellation.set_geometry."""
    ni = np.array([1, -1, -1], dtype=np.int32)
    nr = np.zeros(16)
    nr[0] = radius
    nr[8:11] = center
    nr[11:15] = (0.0, 0.0, 0.0, 1.0)
    nr[15] = 1.0
    return ni, nr


def run_tessellation():
    rng = np.random.default_rng(0)
    N, L = 4000, 1.0
    pos = rng.random((N, 3)) * L
    t = voro.Tessellation()
    t.set_domain(extent=(L, L, L))
    t.build(pos)
    for _ in range(5):
        pos = (pos + 2e-5 * rng.standard_normal((N, 3))) % L
        t.step(pos)
    report("Tessellation.build+step", t.get_volumes(), t.get_neighbor_counts())
    # SDF walls + power weights on the same path
    tw = voro.Tessellation()
    tw.set_domain(extent=(L, L, L))
    ni, nr = sphere_scene((0.5, 0.5, 0.5), 0.25)
    tw.set_geometry(ni, nr, root=0)
    spacing = (L**3 / N) ** (1.0 / 3.0)
    tw.set_weights(rng.random(N) * (0.05 * spacing) ** 2)
    tw.build(pos)
    for _ in range(3):
        pos = (pos + 2e-5 * rng.standard_normal((N, 3))) % L
        tw.step(pos)
    report("Tessellation.geometry+weights", tw.get_volumes(), tw.get_wall_counts())
    types = (np.linalg.norm(pos - 0.5, axis=1) < 0.4).astype(np.int32)
    r = tw.energy_forces(types, np.array([[0.0, 1.0], [1.0, 0.0]]), lloyd=1.0)
    report("Tessellation.energy_forces", r["force"], r["interface_energy"], r["lloyd_energy"])


def run_simulation():
    rng = np.random.default_rng(1)
    N, L = 2000, 1.0
    s = voro.Simulation()
    s.set_domain(extent=(L, L, L))
    s.set_positions(rng.random((N, 3)) * L)
    s.set_velocities(0.01 * rng.standard_normal((N, 3)))
    s.set_masses(np.ones(N))
    s.set_pressure(1.0)
    s.set_viscosities(np.full(N, 0.01))
    s.init()
    s.set_dt(1e-4)
    s.step(10)
    report("Simulation.step", s.get_positions(), s.get_velocities(), s.get_forces(),
           s.kinetic_energy(), s.internal_energy())


def run_flow_solver():
    rng = np.random.default_rng(5)
    n, L, nu = 8, 1.0, 0.01
    h = L / n
    g = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pos = (pos + rng.uniform(-0.2 * h, 0.2 * h, pos.shape)) % L
    k = 2 * np.pi
    U0 = np.zeros_like(pos)
    U0[:, 0] = np.sin(k * pos[:, 0]) * np.cos(k * pos[:, 1])
    U0[:, 1] = -np.cos(k * pos[:, 0]) * np.sin(k * pos[:, 1])
    t = voro.Tessellation()
    t.set_domain(extent=(L, L, L))
    t.build(pos)
    for layout in ("collocated", "covolume"):
        f = voro.FlowSolver(t, nu, layout=layout)
        f.set_body_force((1e-3, 0.0, 0.0))
        f.set_velocity(U0)
        f.set_dt(0.2 * h)
        f.step(10)
        report(f"FlowSolver[{layout}].step", f.get_velocities(), f.get_pressure(),
               f.kinetic_energy())
    f = voro.FlowSolver(t, nu, layout="collocated")
    f.set_implicit_diffusion(True)
    f.set_stokes(True)
    f.set_velocity(U0)
    f.set_dt(2.0 * h * h / nu)
    f.step(5)
    report("FlowSolver[implicit].step", f.get_velocities(), f.get_pressure())


def run_optimizers():
    rng = np.random.default_rng(7)
    N, L = 500, 1.0
    pos = rng.random((N, 3)) * L
    vset = 1.0 + 0.5 * np.sin(2 * np.pi * pos[:, 0])
    r = voro.optimize_volume_mesh(pos, vset, (L, L, L), max_iter=8)
    report("optimize_volume_mesh", r.positions, r.max_vol_err, r.mean_vol_err)
    rw = voro.optimize_volume_mesh(pos, vset, (L, L, L), max_iter=5, use_weights=True,
                                   method="colored_gs")
    report("optimize_volume_mesh[weights]", rw.positions, rw.weights)
    types = (np.linalg.norm(pos - 0.5, axis=1) < 0.3).astype(np.int32)
    ri = voro.minimize_interface(pos, types, (L, L, L), max_iter=8)
    report("minimize_interface", ri.positions, ri.energy, ri.energy_ratio)


def run_pore_mesh():
    from peclet.voro import pore_mesh, scenes
    rng = np.random.default_rng(11)
    L = 1.0
    centers = np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.6]])
    radii = np.array([0.18, 0.2])
    ext = (L, L, L)
    pos = rng.uniform(0, L, (900, 3))
    pos = np.ascontiguousarray(pos[scenes.sphere_union_sdf(pos, centers, radii, ext) > 0.03])
    report("scenes.sphere_union_sdf", scenes.sphere_union_sdf(pos, centers, radii, ext))
    ni, nr, root = scenes.sphere_union_scene(centers, radii)
    report("scenes.sphere_union_scene", ni, nr, root)
    cells = pore_mesh.sdf_voronoi_cells(pos, centers, radii, ext)
    report("pore_mesh.sdf_voronoi_cells", cells["points"], cells["volume"], cells["boundary"])
    sec = pore_mesh.sdf_voronoi_section(pos, centers, radii, ext, (0, 0, 0.5), (0, 0, 1))
    report("pore_mesh.sdf_voronoi_section", sec["verts"], sec["volume"])
    vref = np.full(len(pos), 1.0)
    ro = pore_mesh.optimize_pore_mesh(pos, vref, centers, radii, ext, max_iter=6)
    report("pore_mesh.optimize_pore_mesh", ro.positions, ro.max_vol_err)
    rr = pore_mesh.redistribute_pore_mesh(pos, centers, radii, ext, s_lo=0.12, s_hi=0.12,
                                          max_rounds=3, lloyd_steps=2)
    report("pore_mesh.redistribute_pore_mesh", rr.positions, rr.volumes, rr.max_rel, rr.rms_rel)


def run_mpi():
    """np=2: VoronoiHalo's gather + single-rank tessellation of the owned+ghost set, and
    DistributedTessellation's establish + step; owned volumes gathered to rank 0 by gid."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    L, N = 1.0, 3000
    g_pos = np.random.RandomState(12345).uniform(0.0, L, size=(N, 3))
    g_gid = np.arange(N, dtype=np.int64)
    spacing = (L**3 / N) ** (1.0 / 3.0)

    def collect(name, vol, gid, *extra):
        vols = comm.gather(vol, root=0)
        gids = comm.gather(gid, root=0)
        if comm.rank == 0:
            D = np.full(N, np.nan)
            for gg, vv in zip(gids, vols):
                D[gg] = vv
            assert np.isfinite(D).all(), name
            report(name, D, *extra)

    halo = voro.VoronoiHalo((8, 8, 8), extent=(L, L, L))
    mask = np.asarray(halo.owned_mask(g_pos)) == 1
    pos, gid, _w, n_owned = halo.gather(np.ascontiguousarray(g_pos[mask]),
                                        np.ascontiguousarray(g_gid[mask]), 3.0 * spacing)
    t = voro.Tessellation()
    t.set_domain(extent=(L, L, L))
    t.build(np.ascontiguousarray(np.asarray(pos) % L))
    collect("VoronoiHalo.gather+build", np.asarray(t.get_volumes())[:n_owned],
            np.asarray(gid)[:n_owned], halo.num_ranks)

    d = voro.DistributedTessellation((8, 8, 8), extent=(L, L, L))
    mask = np.asarray(d.owned_mask(g_pos)) == 1
    own = np.ascontiguousarray(g_pos[mask])
    d.establish(own, np.ascontiguousarray(g_gid[mask]))
    rng = np.random.default_rng(100 + comm.rank)
    for _ in range(5):
        own = (own + 2e-5 * rng.standard_normal(own.shape)) % L
        d.step(own)
    collect("DistributedTessellation.step", np.asarray(d.get_volumes()),
            np.asarray(d.get_combined_gids())[:d.num_owned], d.diagnostics.num_regathers)
    # self-check (no single-rank twin to hash against): the owned cells of the repaired distributed
    # tessellation equal a cold single-rank rebuild of the same global positions to the certificate
    # tolerance (the bench_repair_mpi gate)
    allpos = comm.gather(own, root=0)
    allgid = comm.gather(np.asarray(d.get_combined_gids())[:d.num_owned], root=0)
    allvol = comm.gather(np.asarray(d.get_volumes()), root=0)
    if comm.rank == 0:
        P = np.zeros((N, 3))
        V = np.zeros(N)
        for pp, gg, vv in zip(allpos, allgid, allvol):
            P[gg] = pp
            V[gg] = vv
        t = voro.Tessellation()
        t.set_domain(extent=(L, L, L))
        t.build(np.ascontiguousarray(P))
        err = np.abs(V - t.get_volumes()).max() / (L**3 / N)
        print(f"DistributedTessellation vs single-rank rebuild: max |dV|/V_mean = {err:.2e} "
              f"({'OK' if err < 1e-6 else 'FAIL'}, regathers={d.diagnostics.num_regathers})")
        assert err < 1e-6


def main():
    if "--mpi" in sys.argv:
        run_mpi()
        return
    print(f"peclet.voro execution_space = {voro.execution_space}")
    run_tessellation()
    run_simulation()
    run_flow_solver()
    run_optimizers()
    run_pore_mesh()
    mpirun = shutil.which("mpirun")
    if not hasattr(voro, "VoronoiHalo"):
        print("MPI paths: skipped (built without PECLET_VORO_MPI)")
    elif mpirun is None:
        print("MPI paths: skipped (no mpirun on PATH)")
    else:
        env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
        subprocess.run([mpirun, "-np", "2", sys.executable, os.path.abspath(__file__), "--mpi"],
                       check=True, env=env)


if __name__ == "__main__":
    main()
