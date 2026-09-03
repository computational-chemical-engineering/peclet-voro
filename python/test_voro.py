#!/usr/bin/env python3
"""Smoke test for the device-native `peclet.voro` nanobind module.

Exercises the two surfaces — the bare Tessellation (cold build + incremental repair) and the
compressible-Euler Simulation — on a small uniform point set and checks the basic invariants
(space-filling volume, plausible neighbour counts, finite energies). Run with the built module
on PYTHONPATH, e.g.:

    PYTHONPATH=build_nb python python/test_vorflow.py
"""
import numpy as np
from peclet import voro


def test_tessellation():
    rng = np.random.default_rng(0)
    N, L = 20_000, 1.0
    pos = rng.random((N, 3)) * L

    t = voro.Tessellation()
    t.set_box((L, L, L))
    t.build(pos)
    assert t.num_particles == N

    vol = t.volumes()
    assert vol.shape == (N,) and vol.dtype == np.float64
    # space-filling: cell volumes sum to the box volume
    assert abs(vol.sum() / L**3 - 1.0) < 1e-9, vol.sum()
    assert (vol > 0).all()

    nbr = t.neighbor_counts()
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
    assert abs(t.volumes().sum() / L**3 - 1.0) < 1e-9
    assert last["flagged"] < N // 2  # small per-step displacement -> not a full rebuild
    print(f"  Tessellation: N={N}  vol_err={abs(t.volumes().sum()/L**3-1):.1e}  "
          f"mean_nbr={nbr.mean():.2f}  last_step_flagged={last['flagged']}")


def test_simulation():
    rng = np.random.default_rng(1)
    N, L = 4_000, 1.0
    pos = rng.random((N, 3)) * L
    vel = np.zeros((N, 3))
    mass = np.ones(N)

    s = voro.Simulation()
    s.set_box((L, L, L))
    s.set_positions(pos)
    s.set_velocities(vel)
    s.set_masses(mass)
    s.set_pressure(1.0)
    s.init()
    e0 = s.get_kinetic_energy() + s.get_internal_energy()
    s.step(5, 1e-4)
    e1 = s.get_kinetic_energy() + s.get_internal_energy()
    assert np.isfinite(e0) and np.isfinite(e1)
    assert s.get_positions().shape == (N, 3)
    assert abs(s.get_volumes().sum() / L**3 - 1.0) < 1e-9
    print(f"  Simulation:   N={N}  t={s.get_time():.2e}  KE={s.get_kinetic_energy():.3e}  "
          f"IE={s.get_internal_energy():.3e}")


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
    t.set_box((L, L, L))
    t.set_geometry(ni, nr, root=0)
    t.build(pos)
    vol = t.volumes()
    fluid = L**3 - 4.0 / 3.0 * np.pi * R**3
    inside = np.linalg.norm(pos - 0.5, axis=1) < R
    assert (vol[inside] == 0).all(), "seeds inside the solid must have no cell"
    # the fluid volume is tiled up to the tangent-plane clip's recession from the curved wall
    # (measured 0.65% here; rung A1 of the Voronoi methods plan tightens this to second order)
    err0 = abs(vol.sum() / fluid - 1.0)
    assert err0 < 2e-2, err0
    wc = t.wall_counts()
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
    vol = t.volumes()
    inside = np.linalg.norm(pos - 0.5, axis=1) < R
    assert (vol[inside] == 0).all()
    err1 = abs(vol.sum() / fluid - 1.0)
    assert err1 < 2e-2, err1
    # the same geometry on the Simulation (walls push back through the EOS pressure)
    s = voro.Simulation()
    s.set_box((L, L, L))
    keep = ~inside
    s.set_positions(np.ascontiguousarray(pos[keep]))
    s.set_velocities(np.zeros((keep.sum(), 3)))
    s.set_masses(np.ones(keep.sum()))
    s.set_pressure(1.0)
    s.set_geometry(ni, nr)
    s.init()
    s.step(3, 1e-4)
    assert np.isfinite(s.get_kinetic_energy())
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
    t0.set_box((L, L, L))
    t0.build(pos)
    v0 = t0.volumes()
    t1 = voro.Tessellation()
    t1.set_box((L, L, L))
    t1.set_weights(np.zeros(N))          # w == 0: the radical planes ARE the bisectors; the
    t1.build(pos)                        # weight-aware gather visits candidates in another
    v1 = t1.volumes()                    # order, so equality is to round-off, not bit-for-bit
    assert np.allclose(v1, v0, rtol=1e-12, atol=0), np.abs(v1 / v0 - 1).max()
    t1.set_weights(np.full(N, 1e-3))     # equal nonzero weights: the same cells
    t1.build(pos)
    assert np.allclose(t1.volumes(), v0, rtol=1e-10, atol=0)
    spacing = (L**3 / N) ** (1.0 / 3.0)
    w = rng.random(N) * (0.05 * spacing) ** 2   # small-weight regime
    t1.set_weights(w)
    t1.build(pos)
    v2 = t1.volumes()
    # the periodic min-image power diagram is not an exact partition at nonzero weight spread
    # (documented ~1e-2 floor; rung A2 of the Voronoi methods plan makes it exact)
    err = abs(v2.sum() / L**3 - 1.0)
    assert err < 1e-2 and (v2 >= 0).all(), err
    assert not np.array_equal(v0, v2)
    for _ in range(5):
        pos = (pos + 2e-5 * rng.standard_normal((N, 3))) % L
        st = t1.step(pos)
        assert not st["fell_back"]
    assert abs(t1.volumes().sum() / L**3 - 1.0) < 1e-2
    # A2a diagnostics: large-spread weights on overlapping (random) balls bury cells — reported,
    # warned, and raised under strict=True; the small weights above bury none.
    rep = t1.build_report()
    import warnings
    wbig = (rng.random(N) * 2.0 * spacing) ** 2
    t1.set_weights(wbig)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t1.build(pos)
        assert any("buried" in str(x.message) for x in w)
    big = t1.build_report()
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
    t.set_box((L, L, L))
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
    tw.set_box((L, L, L))
    tw.set_geometry(ni, nr)
    tw.build(pos)
    vol = tw.volumes()
    vref = vol[vol > 0].mean()
    dEdV = np.where(vol > 0, 2.0 * (vol / vref - 1.0) / vref, 0.0)
    rw = tw.energy_forces(types, tension, sigma_wall=np.array([0.0, 0.5]), dEdV=dEdV)
    assert rw["wall_energy"] > 0 and np.isfinite(rw["force"]).all()
    # centroidal (Lloyd) relaxation of a random grid through the resident cells: E and the mean
    # seed-to-centroid distance both fall (the B1 gate in miniature)
    pos = rng.random((N, 3)) * L
    tg = voro.Tessellation()
    tg.set_box((L, L, L))
    tg.build(pos)
    rl = tg.energy_forces(np.zeros(N, dtype=np.int32), np.zeros((1, 1)), lloyd=1.0)
    e_l0 = rl["lloyd_energy"]
    vol = tg.volumes()
    for _ in range(10):
        pos = (pos - rl["force"] / (2.0 * vol[:, None])) % L   # x <- centroid (Lloyd's step)
        tg.step(pos)
        vol = tg.volumes()
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
    t.set_box((L, L, L))
    t.build(pos)
    T, dt = 0.25, 0.2 * h
    steps = int(np.ceil(T / dt))
    exact = np.exp(-4 * nu * k * k * T)
    for layout, tol in (("collocated", 0.03), ("covolume", 0.05)):
        f = voro.FlowSolver(t, nu, layout=layout)
        assert f.num_cells() == n**3 and f.num_wall_faces() == 0 and f.layout() == layout
        f.set_velocity(U0)
        E0 = f.kinetic_energy()
        f.step(steps, T / steps)
        ratio = f.kinetic_energy() / E0
        div = f.max_divergence()
        assert abs(ratio / exact - 1) < tol, (layout, ratio, exact)
        assert div < 1e-9, (layout, div)
        vel = f.get_velocity()
        assert vel.shape == (n**3, 3) and np.isfinite(vel).all()
        assert f.get_pressure().shape == (n**3,)
        assert abs(f.get_cell_volume().sum() - L**3) < 1e-9
        print(f"  FlowSolver[{layout}]: E/E0 {ratio:.5f} (exact {exact:.5f}), face div {div:.1e}, "
              f"PCG iters {f.pressure_iterations()}")


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
    centres, radii = [], []
    while len(centres) < 6:
        c, r = rng.uniform(0, L, 3), rng.uniform(0.14, 0.2)
        ok = all(np.linalg.norm((c - cc) - L * np.round((c - cc) / L)) > r + rr + 0.06
                 for cc, rr in zip(centres, radii))
        if ok:
            centres.append(c)
            radii.append(r)
    centres, radii = np.array(centres), np.array(radii)
    phi_solid = (4 / 3 * np.pi * radii**3).sum()
    pos = rng.uniform(0, L, (2500, 3))
    pos = pos[voro._union_sdf(pos, centres, radii, L) > 0.03]  # a uniform (mismatched) start
    out = {}
    for name, kw in (("uniform", dict(s_lo=0.10, s_hi=0.10)),
                     ("graded", dict(s_lo=0.08, s_hi=0.25, slope=0.3))):
        res = voro.redistribute_pore_mesh(pos, centres, radii, L, **kw)
        n0, mx0 = res["history"][0][0], res["history"][0][1]
        print(f"  Redistribute[{name}]: solid fraction {phi_solid:.3f}; start N={n0} max|r|={mx0:.2f} "
              f"-> N={len(res['positions'])} max|r|={res['max_rel']:.3f} rms|r|={res['rms_rel']:.3f} "
              f"dead={res['n_dead']} in {res['rounds']} rounds (+{res['n_added']} -{res['n_removed']})")
        assert res["n_dead"] == 0, name
        out[name] = res
    assert out["uniform"]["max_rel"] < 0.2 and out["uniform"]["rms_rel"] < 0.07
    assert out["graded"]["rms_rel"] < 0.16  # 0.07-0.13 measured (thread-order variation)
    return out


if __name__ == "__main__":
    print(f"peclet.voro execution_space = {voro.execution_space}")
    test_tessellation()
    test_simulation()
    test_geometry()
    test_weights()
    test_energy_forces()
    test_flow_solver()
    test_redistribute()
    print("peclet.voro python smoke test: PASS")
