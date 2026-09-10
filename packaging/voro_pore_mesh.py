"""peclet.voro.pore_mesh — the SDF-walled pore-space (interstitial Voronoi) family.

The wall geometry is a periodic packing of spheres (``sphere_centers`` (M,3), ``sphere_radii`` (M,))
in a CUBIC box ``extent`` = (L, L, L); every function takes the suite-wide ``extent`` triple and
checks it is cubic. Bound in ``_voro`` and re-exported here: :func:`optimize_pore_mesh` (the
position-only Gauss–Newton relaxation toward per-cell target volumes), :func:`sdf_voronoi_cells`
(the clipped polyhedra) and :func:`sdf_voronoi_section` (a plane cross-section). Implemented here:
:func:`redistribute_pore_mesh`, the topological split / merge / relax loop that drives a seeding to
a graded target (rung B2 of the Voronoi methods plan).
"""

from dataclasses import dataclass, field

import numpy as np

from ._voro import (  # noqa: F401
    OptimizeResult,
    Tessellation,
    optimize_pore_mesh,
    sdf_voronoi_cells,
    sdf_voronoi_section,
)
from .scenes import sphere_union_scene, sphere_union_sdf

__all__ = [
    "optimize_pore_mesh",
    "redistribute_pore_mesh",
    "sdf_voronoi_cells",
    "sdf_voronoi_section",
    "OptimizeResult",
    "RedistributeResult",
]


def _cubic_extent(extent, fn):
    e = np.asarray(extent, dtype=np.float64).reshape(-1)
    if e.shape != (3,) or not (e[0] > 0) or e[0] != e[1] or e[1] != e[2]:
        raise ValueError(f"voro: {fn}(extent=...) must be a cubic box (Lx == Ly == Lz > 0) — the "
                         "union-of-spheres wall SDF is cubic-periodic.")
    return float(e[0])


@dataclass
class RedistributeResult:
    """Result of :func:`redistribute_pore_mesh`."""
    positions: np.ndarray   #: the redistributed seeds (N,3)
    volumes: np.ndarray     #: their clipped cell volumes (N,)
    vref: np.ndarray        #: the graded target volume per seed (N,)
    rel: np.ndarray         #: V / V_ref - 1 per seed (-1 for a dead cell)
    max_rel: float          #: max |rel| over the live cells
    rms_rel: float          #: rms of rel over the live cells
    rounds: int             #: split/merge rounds run
    num_added: int          #: seeds added by splits
    num_removed: int        #: seeds removed by merges
    num_dead: int           #: seeds without a cell at the end (0 for a valid mesh)
    history: list = field(default_factory=list)  #: per round (N, max_rel, rms_rel, num_dead)


def redistribute_pore_mesh(positions, sphere_centers, sphere_radii, extent, s_lo, s_hi, *,
                           slope=1.0, beta=2.0, beta_decay=0.9, max_rounds=40, lloyd_steps=8,
                           lloyd_weight=0.3, margin=None, max_change=0.1, tol=0.1, polish=False,
                           wall_shell=True, seed=0, verbose=False):
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
    now-feasible start. Returns a :class:`RedistributeResult`.

    The sphere packing is the periodic wall geometry (sphere_centers (M,3), sphere_radii (M,)) in
    the cubic box `extent` = (L, L, L)."""
    L = _cubic_extent(extent, "redistribute_pore_mesh")
    ext = (L, L, L)
    rng = np.random.default_rng(seed)
    centers = np.asarray(sphere_centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(sphere_radii, dtype=np.float64).ravel()
    pos = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(-1, 3) % L)
    if margin is None:
        margin = 0.4 * s_lo
    ni, nr, root = sphere_union_scene(centers, radii)

    def size_of(phi):
        # s(φ) = clip(s_lo + slope (φ − s_lo), s_lo, s_hi): slope 1 is the example's s = clip(φ);
        # a Voronoi cell's size cannot change faster than its neighbours' — with slope 1 the
        # targets of adjacent cells differ by up to 8x in volume, so the per-cell error floor is
        # O(1); slope ≲ 0.3 makes the target field resolvable
        return np.clip(s_lo + slope * (phi - s_lo), s_lo, s_hi)

    if wall_shell:
        # the wall layers by construction (the example's graded-shell heuristic): drop the seeds
        # closer than the outermost shell and lay concentric shells around every sphere at
        # distances d_k (radial step = in-surface spacing = the local size s(d)), keeping each
        # point near its own layer; the loop then handles the bulk
        dists, d = [], 0.6 * s_lo
        while d < 1.5 * s_hi:
            dists.append(d)
            d += float(size_of(d))
        phi0 = sphere_union_sdf(pos, centers, radii, ext)
        keep = phi0 >= dists[-1] + 0.5 * size_of(dists[-1])
        shells = []
        for d in dists:
            h = float(size_of(d))
            for c, r in zip(centers, radii):
                R = r + d
                n = max(6, int(4 * np.pi * R * R / (h * h)))
                i = np.arange(n) + 0.5
                th = np.arccos(1 - 2 * i / n)
                ph = np.pi * (1 + 5 ** 0.5) * i
                p = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)] * R + c
                p += rng.normal(0, 0.15 * h, p.shape)
                p %= L
                pd = sphere_union_sdf(p, centers, radii, ext)
                shells.append(p[np.abs(pd - d) < 0.5 * h])
        pos = np.ascontiguousarray(np.vstack([pos[keep]] + shells) % L)

    def measure(p):
        t = Tessellation()
        t.set_domain(extent=ext)
        t.set_geometry(ni, nr, root=root)
        if verbose:
            print(f"    measure: build N={len(p)}", flush=True)
        t.build(p, strict=False)
        vol = t.get_volumes()
        rep = t.diagnostics.build_report()
        if verbose:
            print(f"    measure: report {rep}", flush=True)
        phi = sphere_union_sdf(p, centers, radii, ext)
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
                d = pos[i] - centers
                d -= L * np.round(d / L)
                nrm = d[np.argmin(np.linalg.norm(d, axis=1) - radii)]
                nrm /= max(np.linalg.norm(nrm), 1e-300)
                off -= np.dot(off, nrm) * nrm
                if np.linalg.norm(off) < 1e-12:
                    off = rng.standard_normal(3)
                    off -= np.dot(off, nrm) * nrm
            off *= 0.6 * s / max(np.linalg.norm(off), 1e-300)
            q = (pos[i] + off) % L
            if sphere_union_sdf(q[None, :], centers, radii, ext)[0] > margin:
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
            ok = live & (sphere_union_sdf(q, centers, radii, ext) > margin)
            pos[ok] = q[ok]
            pos = np.ascontiguousarray(pos % L)
    vol, vref, rel, dead, cen, phi, rep, step = measure(pos)
    if best is not None and (dead.any() or best[0] < float(np.abs(rel[~dead]).max())):
        pos = best[1]
        vol, vref, rel, dead, cen, phi, rep, step = measure(pos)
    if polish and not dead.any():
        r = optimize_pore_mesh(pos, np.ascontiguousarray(vref), centers, radii, ext,
                               search_window=4, max_iter=60, tol=1e-8, method="graphamg")
        p2 = np.ascontiguousarray(np.asarray(r.positions) % L)
        vol2, vref2, rel2, dead2, cen2, phi2, rep2, step2 = measure(p2)
        if not dead2.any() and np.abs(rel2).max() < np.abs(rel).max():
            pos, vol, vref, rel, dead = p2, vol2, vref2, rel2, dead2
    live = ~dead
    return RedistributeResult(
        positions=pos, volumes=vol, vref=vref, rel=rel,
        max_rel=float(np.abs(rel[live]).max()) if live.any() else float("inf"),
        rms_rel=float(np.sqrt(np.mean(rel[live] ** 2))) if live.any() else float("inf"),
        rounds=rounds, num_added=n_added, num_removed=n_removed, num_dead=int(dead.sum()),
        history=history)
