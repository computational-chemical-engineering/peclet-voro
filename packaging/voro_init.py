"""peclet.voro — dynamic 3D Voronoi tessellation of moving particles.

A device-native (Kokkos) moving-cell Voronoi engine: periodic & Lees–Edwards boxes, incremental cell
repair, and compressible Euler / Navier–Stokes / multiphase dynamics on the moving cells. Also serves as
an unstructured-mesh generator that can feed an Eulerian solve in :mod:`peclet.flow`. The compiled
backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.voro.execution_space`` reports
which one this build has.

* :class:`peclet.voro.Tessellation`, :class:`peclet.voro.Simulation`.

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from ._voro import *  # noqa: F401,F403

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown". This replaces a
# hand-maintained literal that had drifted behind pyproject.toml in every package at 0.6.0.
try:
    from importlib.metadata import PackageNotFoundError as _PNF, version as _dist_version
    try:
        __version__ = _dist_version("peclet-voro")
    except _PNF:  # the CUDA wheel installs the same module under the -cu13 distribution name
        __version__ = _dist_version("peclet-voro-cu13")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"


# --------------------------------------------------------------------------------------------------
# Track B, rung B2 (Voronoi methods plan): global redistribution of interstitial seeds.
# --------------------------------------------------------------------------------------------------
def _union_sdf(pts, centres, radii, L):
    """min_i(|x − c_i|_minimage − r_i): < 0 inside a sphere, > 0 in the fluid (periodic box L)."""
    import numpy as np

    d = pts[:, None, :] - centres[None, :, :]
    d -= L * np.round(d / L)
    return (np.linalg.norm(d, axis=2) - radii[None, :]).min(axis=1)


def sphere_union_scene(centres, radii):
    """Flat scene encoding (node_ints (n,3) int32, node_reals (n,16) float64, root) of the CSG union
    of solid spheres — the geometry for :meth:`Tessellation.set_geometry`."""
    import numpy as np

    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    m = len(radii)
    ints, reals = [], []
    for c, r in zip(centres, radii):  # leaves
        ints.append([1, -1, -1])  # kSphere
        row = np.zeros(16)
        row[0] = r
        row[8:11] = c
        row[11:15] = (0.0, 0.0, 0.0, 1.0)
        row[15] = 1.0
        reals.append(row)
    root = 0
    for k in range(1, m):  # union chain: node m+k-1 = union(previous root, leaf k)
        ints.append([32, root, k])  # kUnion
        row = np.zeros(16)
        row[11:15] = (0.0, 0.0, 0.0, 1.0)
        row[15] = 1.0
        reals.append(row)
        root = len(ints) - 1
    return (np.ascontiguousarray(np.array(ints, dtype=np.int32)),
            np.ascontiguousarray(np.array(reals, dtype=np.float64)), root)


def redistribute_pore_mesh(positions, centres, radii, L, s_lo, s_hi, *, slope=1.0, beta=2.0,
                           beta_decay=0.9, max_rounds=40, lloyd_steps=8, lloyd_weight=0.3,
                           margin=None, max_change=0.1, tol=0.1, polish=False, wall_shell=True,
                           seed=0, verbose=False):
    """Global redistribution of pore-space seeds toward the graded target volume
    V_ref = s(φ)³, s(φ) = clip(φ, s_lo, s_hi) (φ = distance to the nearest sphere wall), by
    TOPOLOGICAL moves the position-only optimiser cannot make (it cannot move seeds between
    pores — rung B2 of the Voronoi methods plan):

      * split: a cell with V > β V_ref gets a second seed (offset from its centroid by ~s(φ)),
      * merge: a seed whose cell has V < V_ref/β, or no cell at all (dead / empty / buried), is
        removed — its neighbours absorb the volume,
      * relax: `lloyd_steps` Lloyd sweeps (seed → clipped-cell centroid, kept off the wall),

    repeated until max |V/V_ref − 1| < tol (V_ref renormalised so Σ V_ref = the fluid volume) or
    `max_rounds`; `max_change` caps the fraction of seeds changed per round. With `polish` the
    position-only optimiser (:func:`optimize_pore_mesh`, GraphAMG Gauss–Newton) finishes from the
    now-feasible start. Returns a dict: positions, volumes, vref, max_rel, rms_rel, rounds,
    n_added, n_removed, n_dead, history (per-round (N, max_rel, rms_rel, n_dead)).

    The sphere packing is the periodic wall geometry (centres (M,3), radii (M,), box L)."""
    import numpy as np

    from ._voro import Tessellation, optimize_pore_mesh

    rng = np.random.default_rng(seed)
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    pos = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(-1, 3) % L)
    if margin is None:
        margin = 0.4 * s_lo
    ni, nr, root = sphere_union_scene(centres, radii)
    def size_of(phi):
        # s(φ) = clip(s_lo + slope (φ − s_lo), s_lo, s_hi): slope 1 is the example's s = clip(φ);
        # a Voronoi cell's size cannot change faster than its neighbours' — with slope 1 the
        # targets of adjacent cells differ by up to 8x in volume, so the per-cell error floor is
        # O(1); slope ≲ 0.3 makes the target field resolvable
        return np.clip(s_lo + slope * (phi - s_lo), s_lo, s_hi)

    if wall_shell:
        # the wall layers by construction (the example's graded-shell heuristic): drop the seeds
        # closer than 3 s_hi... no — closer than the outermost shell — and lay concentric shells
        # around every sphere at distances d_k (radial step = in-surface spacing = the local size
        # s(d)), keeping each point near its own layer; the loop then handles the bulk
        dists, d = [], 0.6 * s_lo
        while d < 1.5 * s_hi:
            dists.append(d)
            d += float(size_of(d))
        phi0 = _union_sdf(pos, centres, radii, L)
        keep = phi0 >= dists[-1] + 0.5 * size_of(dists[-1])
        shells = []
        for d in dists:
            h = float(size_of(d))
            for c, r in zip(centres, radii):
                R = r + d
                n = max(6, int(4 * np.pi * R * R / (h * h)))
                i = np.arange(n) + 0.5
                th = np.arccos(1 - 2 * i / n)
                ph = np.pi * (1 + 5 ** 0.5) * i
                p = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)] * R + c
                p += rng.normal(0, 0.15 * h, p.shape)
                p %= L
                pd = _union_sdf(p, centres, radii, L)
                shells.append(p[np.abs(pd - d) < 0.5 * h])
        pos = np.ascontiguousarray(np.vstack([pos[keep]] + shells) % L)


    def measure(p):
        t = Tessellation()
        t.set_box((L, L, L))
        t.set_geometry(ni, nr, root=root)
        if verbose:
            print(f"    measure: build N={len(p)}", flush=True)
        t.build(p, strict=False)
        vol = t.volumes()
        rep = t.build_report()
        if verbose:
            print(f"    measure: report {rep}", flush=True)
        phi = _union_sdf(p, centres, radii, L)
        dead = (vol <= 0) | (phi <= 0)
        vref = size_of(phi) ** 3  # ABSOLUTE target: the seed count adjusts until Σ V_ref = fluid
        live = ~dead
        rel = np.where(live, vol / vref - 1.0, -1.0)
        # centroids from the Lloyd gradient dE/dx = 2 V (x − c)
        types, ten = np.zeros(len(p), dtype=np.int32), np.zeros((1, 1))
        f = t.energy_forces(types, ten, lloyd=1.0)["force"]
        cen = p.copy()
        cen[live] = p[live] - f[live] / (2.0 * vol[live])[:, None]
        # graded volume descent: E = Σ (V/V_ref − 1)², dE/dV = 2 r / V_ref → the gradient wrt the
        # seeds through the published facet areas; Newton-like step −r·s along −grad
        dEdV = np.where(live, 2.0 * rel / vref, 0.0)
        g = t.energy_forces(types, ten, dEdV=np.ascontiguousarray(dEdV))["force"]
        gn = np.linalg.norm(g, axis=1)
        step = np.zeros_like(p)
        ok = live & (gn > 0)
        step[ok] = -(g[ok] / gn[ok][:, None]) * (np.minimum(np.abs(rel[ok]), 0.5) * size_of(phi[ok]))[:, None]
        return vol, vref, rel, dead, cen, phi, rep, step

    history, n_added, n_removed = [], 0, 0
    rounds = 0
    best = None  # (max|r|, positions): the loop is a heuristic — return its best state
    for rounds in range(1, max_rounds + 1):
        vol, vref, rel, dead, cen, phi, rep, step = measure(pos)
        n = len(pos)
        mx, rms = float(np.abs(rel[~dead]).max()) if (~dead).any() else np.inf, \
            float(np.sqrt(np.mean(rel[~dead] ** 2))) if (~dead).any() else np.inf
        history.append((n, mx, rms, int(dead.sum())))
        if not dead.any() and (best is None or mx < best[0]):
            best = (mx, pos.copy())
        if verbose:
            print(f"  round {rounds:2d}: N={n} max|r|={mx:.3f} rms|r|={rms:.3f} dead={dead.sum()} "
                  f"added={n_added} removed={n_removed}")
        if mx < tol and not dead.any():
            break
        cap = max(1, int(max_change * n))
        # the split/merge thresholds tighten toward the tolerance as the rounds proceed
        bk = max(1.0 + 2.0 * tol, beta * beta_decay ** (rounds - 1))
        # merge: dead cells first, then the smallest cells beyond the threshold
        small = np.where(~dead & (rel < 1.0 / bk - 1.0))[0]
        small = small[np.argsort(rel[small])][:cap]
        remove = np.union1d(np.where(dead)[0], small)
        # split: the largest cells beyond the threshold
        big = np.where(~dead & (rel > bk - 1.0))[0]
        big = big[np.argsort(-rel[big])][:cap]
        new = []
        for i in big:
            s = size_of(phi[i])
            off = cen[i] - pos[i]
            if np.linalg.norm(off) < 0.2 * s:  # symmetric cell: a random direction
                off = rng.standard_normal(3)
            if phi[i] < s:  # wall cell: split ALONG the wall (outward seeds get merged away)
                d = pos[i] - centres
                d -= L * np.round(d / L)
                nrm = d[np.argmin(np.linalg.norm(d, axis=1) - radii)]
                nrm /= max(np.linalg.norm(nrm), 1e-300)
                off -= np.dot(off, nrm) * nrm
                if np.linalg.norm(off) < 1e-12:
                    off = rng.standard_normal(3)
                    off -= np.dot(off, nrm) * nrm
            off *= 0.6 * s / max(np.linalg.norm(off), 1e-300)
            q = (pos[i] + off) % L
            if _union_sdf(q[None, :], centres, radii, L)[0] > margin:
                new.append(q)
        keep = np.ones(n, dtype=bool)
        keep[remove] = False
        n_removed += int((~keep).sum())
        n_added += len(new)
        pos = np.vstack([pos[keep]] + ([np.array(new)] if new else []))
        pos = np.ascontiguousarray(pos % L)
        # relax: a Lloyd blend (cell shape) + the graded volume descent (cell size), seeds kept
        # off the wall
        for _ in range(lloyd_steps):
            vol, vref, rel, dead, cen, phi, rep, step = measure(pos)
            live = ~dead
            q = (pos + lloyd_weight * (cen - pos) + step) % L
            ok = live & (_union_sdf(q, centres, radii, L) > margin)
            pos[ok] = q[ok]
            pos = np.ascontiguousarray(pos % L)
    vol, vref, rel, dead, cen, phi, rep, step = measure(pos)
    if best is not None and (dead.any() or best[0] < float(np.abs(rel[~dead]).max())):
        pos = best[1]
        vol, vref, rel, dead, cen, phi, rep, step = measure(pos)
    if polish and not dead.any():
        r = optimize_pore_mesh(pos, np.ascontiguousarray(vref), centres, radii, L, sw=4,
                               max_iter=60, tol=1e-8, method="graphamg")
        p2 = np.ascontiguousarray(np.asarray(r["positions"]) % L)
        vol2, vref2, rel2, dead2, cen2, phi2, rep2, step2 = measure(p2)
        if not dead2.any() and np.abs(rel2).max() < np.abs(rel).max():
            pos, vol, vref, rel, dead = p2, vol2, vref2, rel2, dead2
    live = ~dead
    return dict(positions=pos, volumes=vol, vref=vref, rel=rel,
                max_rel=float(np.abs(rel[live]).max()) if live.any() else float("inf"),
                rms_rel=float(np.sqrt(np.mean(rel[live] ** 2))) if live.any() else float("inf"),
                rounds=rounds, n_added=n_added, n_removed=n_removed, n_dead=int(dead.sum()),
                history=history)
