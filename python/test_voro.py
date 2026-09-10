#!/usr/bin/env python3
"""Smoke test for the device-native `peclet.voro` nanobind module.

Exercises the three surfaces — the bare Tessellation (cold build + incremental repair, SDF
geometry, power weights, energy forces), the compressible-Euler Simulation, and the static
FlowSolver on the face mesh — plus the pore_mesh / scenes submodules and the API contract
(diagnostics tier, typed optimiser results, validated string modes), on small point sets, and
checks the basic invariants (space-filling volume, plausible neighbour counts, finite energies,
face divergence). Run with the built module on PYTHONPATH, e.g.:

    PYTHONPATH=<build> python python/test_voro.py
"""
import numpy as np
from peclet import voro


def test_tessellation():
    rng = np.random.default_rng(0)
    N, L = 20_000, 1.0
    pos = rng.random((N, 3)) * L

    t = voro.Tessellation()
    t.set_domain(extent=(L, L, L))
    t.build(pos)
    assert t.num_particles == N

    vol = t.get_volumes()
    assert vol.shape == (N,) and vol.dtype == np.float64
    # space-filling: cell volumes sum to the box volume
    assert abs(vol.sum() / L**3 - 1.0) < 1e-9, vol.sum()
    assert (vol > 0).all()

    nbr = t.get_neighbor_counts()
    assert nbr.shape == (N,) and nbr.dtype == np.int32
    # a 3D Voronoi cell has at least 4 faces; the Poisson mean is ~15.5
    assert nbr.min() >= 4 and 13 < nbr.mean() < 18, (nbr.min(), nbr.mean())

    # move + repair: volumes must stay space-filling, and tiny moves should flag few cells
    last = None
    for _ in range(20):
        pos = (pos + 2e-5 * rng.standard_normal((N, 3))) % L  # ~5e-4 of the spacing per step
        last = t.step(pos)
    # step() must keep reporting the repair-stats fields this test (and callers) rely on. A
    # SUPERSET check, not equality: the dict legitimately grows as the repair path gains
    # instrumentation -- it picked up extra/surgical/verify_passes, which broke a strict
    # `set(last) == {...}` here even though nothing had regressed. A removed or renamed key is
    # the real regression, and this still catches that.
    assert set(last) >= {"flagged", "pass1", "pass2", "rebuilt", "fell_back"}, sorted(last)
    assert abs(t.get_volumes().sum() / L**3 - 1.0) < 1e-9
    assert last["flagged"] < N // 2  # small per-step displacement -> not a full rebuild
    print(f"  Tessellation: N={N}  vol_err={abs(t.get_volumes().sum()/L**3-1):.1e}  "
          f"mean_nbr={nbr.mean():.2f}  last_step_flagged={last['flagged']}")


def test_api_contract():
    """The suite-wide names (suite/docs/NAMING.md): set_domain CHECKS origin / periodic, the time
    step is set_dt + dt, and step() without one raises instead of guessing."""
    t = voro.Tessellation()
    t.set_domain(extent=(2.0, 1.0, 1.0))
    assert tuple(t.extent) == (2.0, 1.0, 1.0)
    for bad in (dict(origin=(0.1, 0.0, 0.0)), dict(periodic=(True, True, False))):
        try:
            t.set_domain(extent=(1.0, 1.0, 1.0), **bad)
            raise AssertionError(f"set_domain must reject {bad}")
        except ValueError:
            pass
    s = voro.Simulation()
    s.set_domain(extent=(1.0, 1.0, 1.0))
    s.set_positions(np.random.default_rng(9).random((500, 3)))
    s.set_velocities(np.zeros((500, 3)))
    s.set_masses(np.ones(500))
    s.set_pressure(1.0)
    s.init()
    assert s.dt == 0.0
    try:
        s.step(1)
        raise AssertionError("step() without set_dt must raise")
    except ValueError:
        pass
    s.set_dt(1e-4)
    s.step(1)
    assert s.dt == 1e-4 and abs(s.time - 1e-4) < 1e-18
    # the state setters raise after init() with the order in the message
    try:
        s.set_pressure(2.0)
        raise AssertionError("set_pressure after init must raise")
    except RuntimeError as e:
        assert "after init()" in str(e)
    # the diagnostics tier: one nested object per class, a view onto its owner
    t.set_tolerance()                                   # the numeric default stays
    t.diagnostics.set_gate(True)
    t.build(np.random.default_rng(1).random((3000, 3)))
    rep = t.diagnostics.build_report()
    assert set(rep) == {"buried", "reach_exceeded", "empty", "overflow", "incomplete",
                        "over_buffer_rebuilds"}
    assert rep["over_buffer_rebuilds"] == 0
    t.diagnostics.set_profile(True)   # stderr only; the numerics do not move
    t.build(np.random.default_rng(1).random((3000, 3)))
    assert t.diagnostics.build_report() == rep
    t.diagnostics.set_profile(False)
    assert hasattr(s.diagnostics, "set_repair")
    # string modes: a bad value raises and the message lists the accepted set
    for call, expect in ((lambda: t.set_wall_mode("bogus"), "'exact', 'skin'"),
                         (lambda: voro.FlowSolver(t, 0.01, layout="staggered"), "'collocated', 'covolume'"),
                         (lambda: voro.optimize_volume_mesh(np.random.default_rng(2).random((50, 3)),
                                                            np.ones(50), (1.0, 1.0, 1.0), method="cg"),
                          "'jacobi', 'colored_gs', 'graphamg', 'steepest'")):
        try:
            call()
            raise AssertionError(f"expected a ValueError listing {expect}")
        except ValueError as e:
            assert expect in str(e), str(e)
    # typed results; triples as 3-sequences; the submodules
    r = voro.optimize_volume_mesh(np.random.default_rng(3).random((200, 3)), np.ones(200),
                                  (1.0, 1.0, 1.0), max_iter=2)
    assert isinstance(r, voro.OptimizeResult) and r.positions.shape == (200, 3) and r.weights is None
    f = voro.FlowSolver(t, 0.01)
    f.set_body_force([0.0, 0.0, 1e-3])
    assert f.diagnostics is not None and f.dt == 0.0
    assert "certificate_tolerance" in voro.defaults
    assert callable(voro.pore_mesh.redistribute_pore_mesh) and callable(voro.scenes.sphere_union_sdf)
    for gone in ("build_report", "set_gate", "set_local_certificate"):
        assert not hasattr(t, gone), gone
    for gone in ("_union_sdf", "redistribute_pore_mesh", "sphere_union_scene", "optimize_pore_mesh"):
        assert not hasattr(voro, gone), gone
    print("  API contract: set_domain checks, set_dt/dt/step, diagnostics tier, string modes OK")


def test_simulation():
    rng = np.random.default_rng(1)
    N, L = 4_000, 1.0
    pos = rng.random((N, 3)) * L
    vel = np.zeros((N, 3))
    mass = np.ones(N)

    s = voro.Simulation()
    s.set_domain(extent=(L, L, L))
    s.set_positions(pos)
    s.set_velocities(vel)
    s.set_masses(mass)
    s.set_pressure(1.0)
    s.init()
    e0 = s.kinetic_energy() + s.internal_energy()
    s.set_dt(1e-4)
    s.step(5)
    e1 = s.kinetic_energy() + s.internal_energy()
    assert np.isfinite(e0) and np.isfinite(e1)
    assert s.get_positions().shape == (N, 3)
    assert abs(s.get_volumes().sum() / L**3 - 1.0) < 1e-9
    print(f"  Simulation:   N={N}  t={s.time:.2e}  KE={s.kinetic_energy():.3e}  "
          f"IE={s.internal_energy():.3e}")


def sphere_scene(centre, radius):
    """Flat node encoding (3 int32 + 16 float64 per node) of one solid sphere — what
    peclet.core.geom.Scene.encode() would return for scene.add_sphere(radius, translation=centre)."""
    kSphere = 1
    node_ints = np.array([kSphere, -1, -1], dtype=np.int32)
    node_reals = np.zeros(16, dtype=np.float64)
    node_reals[0] = radius                       # params[0]
    node_reals[8:11] = centre                    # translation
    node_reals[11:15] = (0.0, 0.0, 0.0, 1.0)     # rotation quaternion (identity)
    node_reals[15] = 1.0                         # scale
    return node_ints, node_reals


def test_geometry():
    """Rung A0: an SDF solid on the Tessellation, carried through cold build + incremental steps."""
    rng = np.random.default_rng(2)
    N, L, R = 12_000, 1.0, 0.25
    pos = rng.random((N, 3)) * L
    ni, nr = sphere_scene((0.5, 0.5, 0.5), R)

    t = voro.Tessellation()
    t.set_domain(extent=(L, L, L))
    t.set_geometry(ni, nr, root=0)
    t.build(pos)
    vol = t.get_volumes()
    fluid = L**3 - 4.0 / 3.0 * np.pi * R**3
    inside = np.linalg.norm(pos - 0.5, axis=1) < R
    assert (vol[inside] == 0).all(), "seeds inside the solid must have no cell"
    # the fluid volume is tiled up to the tangent-plane clip's recession from the curved wall
    # (measured 0.65% here; rung A1 of the Voronoi methods plan tightens this to second order)
    err0 = abs(vol.sum() / fluid - 1.0)
    assert err0 < 2e-2, err0
    wc = t.get_wall_counts()
    assert wc.shape == (N,) and wc.dtype == np.int32 and (wc > 0).sum() > 0
    assert (wc[inside] == 0).all()
    # move + repair: the boundary watch must fire and the fluid volume stay tiled
    flagged = 0
    for _ in range(20):
        pos = (pos + 2e-5 * rng.standard_normal((N, 3))) % L
        st = t.step(pos)
        flagged += st["wall_flagged"]
        assert not st["fell_back"]
    assert flagged > 0
    vol = t.get_volumes()
    inside = np.linalg.norm(pos - 0.5, axis=1) < R
    assert (vol[inside] == 0).all()
    err1 = abs(vol.sum() / fluid - 1.0)
    assert err1 < 2e-2, err1
    # the same geometry on the Simulation (walls push back through the EOS pressure)
    s = voro.Simulation()
    s.set_domain(extent=(L, L, L))
    keep = ~inside
    s.set_positions(np.ascontiguousarray(pos[keep]))
    s.set_velocities(np.zeros((keep.sum(), 3)))
    s.set_masses(np.ones(keep.sum()))
    s.set_pressure(1.0)
    s.set_geometry(ni, nr)
    s.init()
    s.set_dt(1e-4)
    s.step(3)
    assert np.isfinite(s.kinetic_energy())
    p1 = s.get_positions()
    assert (np.linalg.norm(p1 - 0.5, axis=1) > R * 0.9).all(), "fluid seeds pushed into the solid"
    print(f"  Geometry:     N={N}  fluid_vol_err build={err0:.1e} after steps={err1:.1e}  "
          f"wall_cells={(wc > 0).sum()}  wall_flagged/step={flagged / 20:.0f}")


def test_weights():
    """Rung A0: power weights on the Tessellation — equal weights reproduce Voronoi exactly."""
    rng = np.random.default_rng(3)
    N, L = 8_000, 1.0
    pos = rng.random((N, 3)) * L
    t0 = voro.Tessellation()
    t0.set_domain(extent=(L, L, L))
    t0.build(pos)
    v0 = t0.get_volumes()
    t1 = voro.Tessellation()
    t1.set_domain(extent=(L, L, L))
    t1.set_weights(np.zeros(N))          # w == 0: the radical planes ARE the bisectors; the
    t1.build(pos)                        # weight-aware gather visits candidates in another
    v1 = t1.get_volumes()                    # order, so equality is to round-off, not bit-for-bit
    assert np.allclose(v1, v0, rtol=1e-12, atol=0), np.abs(v1 / v0 - 1).max()
    t1.set_weights(np.full(N, 1e-3))     # equal nonzero weights: the same cells
    t1.build(pos)
    assert np.allclose(t1.get_volumes(), v0, rtol=1e-10, atol=0)
    spacing = (L**3 / N) ** (1.0 / 3.0)
    w = rng.random(N) * (0.05 * spacing) ** 2   # small-weight regime
    t1.set_weights(w)
    t1.build(pos)
    v2 = t1.get_volumes()
    # the periodic min-image power diagram is not an exact partition at nonzero weight spread
    # (documented ~1e-2 floor; rung A2 of the Voronoi methods plan makes it exact)
    err = abs(v2.sum() / L**3 - 1.0)
    assert err < 1e-2 and (v2 >= 0).all(), err
    assert not np.array_equal(v0, v2)
    for _ in range(5):
        pos = (pos + 2e-5 * rng.standard_normal((N, 3))) % L
        st = t1.step(pos)
        assert not st["fell_back"]
    assert abs(t1.get_volumes().sum() / L**3 - 1.0) < 1e-2
    # A2a diagnostics: large-spread weights on overlapping (random) balls bury cells — reported,
    # warned, and raised under strict=True; the small weights above bury none.
    rep = t1.diagnostics.build_report()
    import warnings
    wbig = (rng.random(N) * 2.0 * spacing) ** 2
    t1.set_weights(wbig)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t1.build(pos)
        assert any("buried" in str(x.message) for x in w)
    big = t1.diagnostics.build_report()
    assert big["buried"] > 0
    try:
        t1.build(pos, strict=True)
        raise AssertionError("strict build must raise on buried cells")
    except RuntimeError:
        pass
    print(f"  Weights:      N={N}  power volumes sum err={err:.1e} (periodic min-image floor); "
          f"small-w report {rep}; large-w buried={big['buried']}")


def test_energy_forces():
    """Rung A3: interfacial / wetting / volume energies + gradients on the resident cells."""
    rng = np.random.default_rng(4)
    N, L, R = 6_000, 1.0, 0.25
    pos = rng.random((N, 3)) * L
    types = (np.linalg.norm(pos - 0.5, axis=1) < 0.2).astype(np.int32)   # a blob of species 1
    tension = np.array([[0.0, 1.0], [1.0, 0.0]])
    t = voro.Tessellation()
    t.set_domain(extent=(L, L, L))
    t.build(pos)
    r = t.energy_forces(types, tension)
    assert r["force"].shape == (N, 3) and np.isfinite(r["force"]).all()
    assert r["interface_energy"] > 0 and r["wall_energy"] == 0.0
    # descend along -force: the interfacial area must drop (surface tension rounds the blob)
    e0 = r["interface_energy"]
    spacing = (L**3 / N) ** (1.0 / 3.0)
    for _ in range(5):
        f = r["force"]
        step = 0.02 * spacing / max(np.abs(f).max(), 1e-30)
        pos = (pos - step * f) % L
        t.step(pos)
        r = t.energy_forces(types, tension)
    assert r["interface_energy"] < e0, (e0, r["interface_energy"])
    # with a wall: wetting energy of species 1 on a sphere, plus a volume-target term
    ni, nr = sphere_scene((0.5, 0.5, 0.8), 0.2)
    tw = voro.Tessellation()
    tw.set_domain(extent=(L, L, L))
    tw.set_geometry(ni, nr)
    tw.build(pos)
    vol = tw.get_volumes()
    vref = vol[vol > 0].mean()
    dEdV = np.where(vol > 0, 2.0 * (vol / vref - 1.0) / vref, 0.0)
    rw = tw.energy_forces(types, tension, sigma_wall=np.array([0.0, 0.5]), dEdV=dEdV)
    assert rw["wall_energy"] > 0 and np.isfinite(rw["force"]).all()
    # centroidal (Lloyd) relaxation of a random grid through the resident cells: E and the mean
    # seed-to-centroid distance both fall (the B1 gate in miniature)
    pos = rng.random((N, 3)) * L
    tg = voro.Tessellation()
    tg.set_domain(extent=(L, L, L))
    tg.build(pos)
    rl = tg.energy_forces(np.zeros(N, dtype=np.int32), np.zeros((1, 1)), lloyd=1.0)
    e_l0 = rl["lloyd_energy"]
    vol = tg.get_volumes()
    for _ in range(10):
        pos = (pos - rl["force"] / (2.0 * vol[:, None])) % L   # x <- centroid (Lloyd's step)
        tg.step(pos)
        vol = tg.get_volumes()
        rl = tg.energy_forces(np.zeros(N, dtype=np.int32), np.zeros((1, 1)), lloyd=1.0)
    assert rl["lloyd_energy"] < 0.8 * e_l0, (e_l0, rl["lloyd_energy"])
    print(f"  Energies:     N={N}  E_if {e0:.4f} -> {r['interface_energy']:.4f} after 5 descent steps;"
          f"  wetting E={rw['wall_energy']:.4f};  Lloyd E {e_l0:.3e} -> {rl['lloyd_energy']:.3e} in 10 steps")


def test_flow_solver():
    """Track C (C5): FlowSolver on a resident tessellation — a decaying Taylor–Green vortex on a
    jittered 16^3 lattice for both layouts; the energy must track exp(-4 nu k^2 t) (collocated
    within 3 %, covolume within 5 % at 0.2h jitter — the measured accuracies of the C2 tests at
    this resolution) and the transporting face flux must be divergence-free to round-off."""
    rng = np.random.default_rng(5)
    n, L, nu = 16, 1.0, 0.01
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
    T, dt = 0.25, 0.2 * h
    steps = int(np.ceil(T / dt))
    exact = np.exp(-4 * nu * k * k * T)
    for layout, tol in (("collocated", 0.03), ("covolume", 0.05)):
        f = voro.FlowSolver(t, nu, layout=layout)
        assert f.num_cells == n**3 and f.num_wall_faces == 0 and f.layout == layout
        f.set_velocity(U0)
        E0 = f.kinetic_energy()
        f.set_dt(T / steps)
        f.step(steps)
        ratio = f.kinetic_energy() / E0
        div = f.max_divergence()
        assert abs(ratio / exact - 1) < tol, (layout, ratio, exact)
        assert div < 1e-9, (layout, div)
        vel = f.get_velocities()
        assert vel.shape == (n**3, 3) and np.isfinite(vel).all()
        assert f.get_pressure().shape == (n**3,)
        assert abs(f.get_volumes().sum() - L**3) < 1e-9
        print(f"  FlowSolver[{layout}]: E/E0 {ratio:.5f} (exact {exact:.5f}), face div {div:.1e}, "
              f"PCG iters {f.pressure_iterations}")


def test_redistribute():
    """Rung B2: global redistribution of pore-space seeds. Six random non-overlapping spheres in the
    periodic unit box; a MISMATCHED start (uniform random seeds, twice too many) must reach the
    target by split / merge / relax with the graded wall shells, zero dead cells.
    MEASURED (2026-09-03): uniform s = 0.10: max |V/V_ref - 1| 0.10-0.14, rms 0.035-0.05 (the plan's
    < 0.1 gate met at rms level, at max level within 1.4x); graded s(phi) = clip(0.08 + 0.3 (phi -
    0.08), 0.08, 0.25): rms 0.07-0.08, max ~0.5 (the wall-layer cells stay ~1.5x too big — the
    first shell's radial extent); slope 1 (the example's clip(phi)) is unresolvable by any mesh
    (neighbouring targets differ 8x): rms ~0.3-0.5. Gates: uniform max < 0.2, rms < 0.07;
    graded-0.3 rms < 0.16; no dead cells."""
    rng = np.random.default_rng(11)
    L = 1.0
    centers, radii = [], []
    while len(centers) < 6:
        c, r = rng.uniform(0, L, 3), rng.uniform(0.14, 0.2)
        ok = all(np.linalg.norm((c - cc) - L * np.round((c - cc) / L)) > r + rr + 0.06
                 for cc, rr in zip(centers, radii))
        if ok:
            centers.append(c)
            radii.append(r)
    centers, radii = np.array(centers), np.array(radii)
    phi_solid = (4 / 3 * np.pi * radii**3).sum()
    pos = rng.uniform(0, L, (2500, 3))
    pos = pos[voro.scenes.sphere_union_sdf(pos, centers, radii, (L, L, L)) > 0.03]  # a mismatched start
    out = {}
    for name, kw in (("uniform", dict(s_lo=0.10, s_hi=0.10)),
                     ("graded", dict(s_lo=0.08, s_hi=0.25, slope=0.3))):
        res = voro.pore_mesh.redistribute_pore_mesh(pos, centers, radii, (L, L, L), **kw)
        n0, mx0 = res.history[0][0], res.history[0][1]
        print(f"  Redistribute[{name}]: solid fraction {phi_solid:.3f}; start N={n0} max|r|={mx0:.2f} "
              f"-> N={len(res.positions)} max|r|={res.max_rel:.3f} rms|r|={res.rms_rel:.3f} "
              f"dead={res.num_dead} in {res.rounds} rounds (+{res.num_added} -{res.num_removed})")
        assert res.num_dead == 0, name
        out[name] = res
    assert out["uniform"].max_rel < 0.2 and out["uniform"].rms_rel < 0.07
    assert out["graded"].rms_rel < 0.16  # 0.07-0.13 measured (thread-order variation)
    return out


def _pore_scene():
    """Two non-overlapping spheres in the periodic unit box and a fluid seeding with a clear
    wall gap (a seed closer than 0.03 to a wall would carry a wall-hugging sliver cell)."""
    rng = np.random.default_rng(11)
    L = 1.0
    centers = np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.6]])
    radii = np.array([0.18, 0.2])
    pos = rng.uniform(0, L, (900, 3))
    pos = np.ascontiguousarray(pos[voro.scenes.sphere_union_sdf(pos, centers, radii, (L, L, L)) > 0.03])
    return pos, centers, radii, L


def _check_pore_cells(cells, pos, L, pore_volume, tag):
    """The invariants of a VTK_POLYHEDRON cell list: every face a closed CCW polygon, the faces
    a closed manifold (each edge used exactly twice, once per direction), the seed strictly
    inside, the 'volume' equal to the divergence-theorem volume of the faces, and the cells
    tiling the pore space (to the tangent-plane wall clip's recession, < 2 % as in
    test_geometry)."""
    pts, faces, off = cells["points"], cells["faces"], cells["face_offsets"]
    vol, seed, boundary = cells["volume"], cells["seed"], cells["boundary"]
    nc = len(vol)
    assert nc > 0 and len(off) == nc + 1 and len(seed) == nc and len(boundary) == nc
    assert pts.shape[1] == 3 and (vol > 0).all() and len(set(seed)) == nc
    assert abs(vol.sum() / pore_volume - 1.0) < 2e-2, vol.sum() / pore_volume
    max_vol_err = 0.0
    for c in range(nc):
        blk = faces[off[c]:off[c + 1]]
        nf, k = int(blk[0]), 1
        assert nf >= 4
        edges, vsum, s = {}, 0.0, pos[seed[c]]
        for _ in range(nf):
            m = int(blk[k])
            ids = blk[k + 1:k + 1 + m]
            k += 1 + m
            assert m >= 3 and len(set(ids)) == m
            v = pts[ids]
            area = 0.5 * np.cross(v, np.roll(v, -1, axis=0)).sum(axis=0)   # outward for CCW faces
            assert np.linalg.norm(area) > 0
            n = area / np.linalg.norm(area)
            # planar, convex and CCW about the outward normal; the seed strictly inside. Sliver
            # faces (three vertices ~1e-6 apart) have an ill-conditioned normal, hence 1e-8 L.
            assert np.abs((v - v[0]) @ n).max() < 1e-8 * L
            for i in range(m):
                e = np.cross(v[(i + 1) % m] - v[i], v[(i + 2) % m] - v[(i + 1) % m]) @ n
                assert e > -1e-9 * L * L
                edges[(int(ids[i]), int(ids[(i + 1) % m]))] = edges.get((int(ids[i]), int(ids[(i + 1) % m])), 0) + 1
            assert (s - v[0]) @ n < -1e-9 * L, "seed outside its cell"
            vsum += (v[0] @ area) / 3.0
        assert k == len(blk)
        assert all(cnt == 1 and (b, a) in edges for (a, b), cnt in edges.items()), "faces not closed"
        max_vol_err = max(max_vol_err, abs(vsum / vol[c] - 1.0))
    assert max_vol_err < 1e-10, max_vol_err
    assert boundary.sum() > 0 and (boundary[np.isin(seed, seed)] <= 1).all()
    return max_vol_err


def _check_pore_section(sec, cells, point, normal, L, area_ref, tag):
    """Section polygons lie in the plane, are convex and CCW, carry their 3-D cell's volume, and
    tile the plane's pore cross-section (same recession tolerance as the volumes)."""
    verts, off, vol, seed = sec["verts"], sec["offsets"], sec["volume"], sec["seed"]
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    npoly = len(vol)
    assert npoly > 0 and len(off) == npoly + 1 and len(seed) == npoly
    assert np.abs((verts - np.asarray(point, float)) @ n).max() < 1e-12 * L
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    area = 0.0
    cvol = dict(zip(cells["seed"].tolist(), cells["volume"].tolist()))
    for p in range(npoly):
        v = verts[off[p]:off[p + 1]]
        m = len(v)
        assert m >= 3
        q = np.stack([v @ e1, v @ e2], axis=1)
        a = 0.5 * (q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]).sum()
        assert a > 0, "section polygon not CCW"
        for i in range(m):
            d = q[(i + 1) % m] - q[i]
            d2 = q[(i + 2) % m] - q[(i + 1) % m]
            assert d[0] * d2[1] - d[1] * d2[0] > -1e-9 * L * L, "section polygon not convex"
        area += a
        assert seed[p] in cvol and abs(vol[p] / cvol[seed[p]] - 1.0) < 1e-12  # same cell, same code
    assert abs(area / area_ref - 1.0) < 2e-2, (tag, area, area_ref)
    return area


def _same_point_sets(a, b, tol):
    """Two (n,3) point sets are equal as sets: every point of `a` has a match in `b` within tol
    and the counts agree (the reconstruction orders vertices by its own triangle list)."""
    if len(a) != len(b):
        return False
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return bool((d.min(axis=1) < tol).all() and (d.min(axis=0) < tol).all())


def test_pore_cells():
    """peclet.voro.pore_mesh.sdf_voronoi_cells / sdf_voronoi_section (the SDF-walled interstitial
    cells and a plane section of them) on a two-sphere scene: the geometric invariants of both
    outputs, and — the device port's gate (QUALITY_PLAN G.7) — the device result against the
    host-serial reconstruction kept as the test oracle (`_voro._sdf_voronoi_cells_host`,
    `_sdf_voronoi_section_host`): the same cells, volumes to 1e-12 relative, vertices equal as
    sets to 1e-9 L (the two gather the same planes in a different order, so the vertex ORDER and
    the last bits differ). Suite policy for a DEVICE backend (CUDA/HIP: FMA contraction, no
    bit-exactness): 1e-9 relative / 1e-7 L (measured 6e-11 on an RTX 5080)."""
    host_backend = voro.execution_space in ("OpenMP", "Serial")
    vtol, ptol = (1e-12, 1e-9) if host_backend else (1e-9, 1e-7)
    pos, centers, radii, L = _pore_scene()
    ext = (L, L, L)
    pore_volume = L**3 - (4.0 / 3.0 * np.pi * radii**3).sum()
    cells = voro.pore_mesh.sdf_voronoi_cells(pos, centers, radii, ext)
    # 'num_incomplete' is the conservative inscribed-sphere coverage criterion of the gather
    # window; at this small N the grid clamps the window to 3 blocks and flags a few tens of
    # cells whose volumes are nonetheless exact (the oracle comparison below). At 5k+ seeds the
    # window is not clamped and nothing is flagged.
    assert cells["num_overflow"] == 0 and cells["num_incomplete"] < 0.1 * len(pos), cells["num_incomplete"]
    big = np.random.default_rng(3).uniform(0, L, (6000, 3))
    big = np.ascontiguousarray(big[voro.scenes.sphere_union_sdf(big, centers, radii, ext) > 0.03])
    cb = voro.pore_mesh.sdf_voronoi_cells(big, centers, radii, ext)
    assert cb["num_overflow"] == 0 and cb["num_incomplete"] == 0, cb["num_incomplete"]
    assert len(cb["volume"]) == len(big) and abs(cb["volume"].sum() / pore_volume - 1.0) < 2e-2
    verr = _check_pore_cells(cells, pos, L, pore_volume, "device")
    point, normal = (0.0, 0.0, 0.3), (0.0, 0.0, 1.0)
    dz = np.abs(((centers[:, 2] - point[2]) + L / 2) % L - L / 2)
    area_ref = L * L - np.pi * np.clip(radii**2 - dz**2, 0.0, None).sum()
    sec = voro.pore_mesh.sdf_voronoi_section(pos, centers, radii, ext, point, normal)
    assert sec["num_overflow"] == 0 and sec["num_incomplete"] == cells["num_incomplete"]
    _check_pore_section(sec, cells, point, normal, L, area_ref, "device")
    # the host oracle (the pre-G.7 serial reconstruction, retained as `_voro._*_host`)
    from peclet.voro import _voro
    hc = _voro._sdf_voronoi_cells_host(pos, centers, radii, ext)
    _check_pore_cells(hc, pos, L, pore_volume, "host")
    assert set(cells["seed"].tolist()) == set(hc["seed"].tolist())
    order_d, order_h = np.argsort(cells["seed"]), np.argsort(hc["seed"])
    assert np.allclose(cells["volume"][order_d], hc["volume"][order_h], rtol=vtol, atol=0)
    assert np.array_equal(cells["boundary"][order_d], hc["boundary"][order_h])

    def cell_points(c, i):
        blk = c["faces"][c["face_offsets"][i]:c["face_offsets"][i + 1]]
        ids, k = set(), 1
        for _ in range(int(blk[0])):
            m = int(blk[k])
            ids.update(blk[k + 1:k + 1 + m].tolist())
            k += 1 + m
        return c["points"][sorted(ids)]

    for i, j in zip(order_d, order_h):
        assert _same_point_sets(cell_points(cells, i), cell_points(hc, j), ptol * L), cells["seed"][i]
    hs = _voro._sdf_voronoi_section_host(pos, centers, radii, ext, point, normal)
    _check_pore_section(hs, hc, point, normal, L, area_ref, "host")
    assert set(sec["seed"].tolist()) == set(hs["seed"].tolist())
    hoff = {int(s): (hs["offsets"][p], hs["offsets"][p + 1]) for p, s in enumerate(hs["seed"])}
    for p, s in enumerate(sec["seed"]):
        a, b = hoff[int(s)]
        assert _same_point_sets(sec["verts"][sec["offsets"][p]:sec["offsets"][p + 1]],
                                hs["verts"][a:b], ptol * L), s
    print(f"  Pore cells:   N={len(pos)} cells={len(cells['volume'])} (wall {int(cells['boundary'].sum())}, "
          f"{cells['num_incomplete']} flagged by the clamped window; N={len(big)}: 0) "
          f"vol err={abs(cells['volume'].sum() / pore_volume - 1):.1e} face-vol err={verr:.1e}; "
          f"section polys={len(sec['volume'])}; host oracle: volumes {vtol:g}, vertex sets {ptol:g} L OK")


if __name__ == "__main__":
    print(f"peclet.voro execution_space = {voro.execution_space}")
    test_tessellation()
    test_api_contract()
    test_simulation()
    test_geometry()
    test_weights()
    test_energy_forces()
    test_flow_solver()
    test_redistribute()
    test_pore_cells()
    print("peclet.voro python smoke test: PASS")
