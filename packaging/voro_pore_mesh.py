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
    rng = np.random.default_rng(seed)
    centers = np.asarray(sphere_centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(sphere_radii, dtype=np.float64).ravel()
    pos = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(-1, 3) % L)
    if margin is None:
        margin = 0.4 * s_lo
    ni, nr, root = sphere_union_scene(centers, radii)
    scene = _PoreScene(centers, radii, L, ni, nr, root, s_lo, s_hi, slope)

    if wall_shell:
        pos = _wall_shell_seeds(scene, pos, rng)

    history, n_added, n_removed = [], 0, 0
    rounds = 0
    best = None  # (max|r|, positions): the loop is a heuristic — return its best state
    for rounds in range(1, max_rounds + 1):
        vol, vref, rel, dead, cen, phi, rep, step = _measure_pore_cells(scene, pos, verbose)
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
        new = _split_seeds(scene, pos, cen, phi, big, margin, rng)
        keep = np.ones(n, dtype=bool)
        keep[remove] = False
        n_removed += int((~keep).sum())
        n_added += len(new)
        pos = np.vstack([pos[keep]] + ([np.array(new)] if new else []))
        pos = np.ascontiguousarray(pos % L)
        pos = _relax_seeds(scene, pos, lloyd_steps, lloyd_weight, margin, verbose)
    vol, vref, rel, dead, cen, phi, rep, step = _measure_pore_cells(scene, pos, verbose)
    if best is not None and (dead.any() or best[0] < float(np.abs(rel[~dead]).max())):
        pos = best[1]
        vol, vref, rel, dead, cen, phi, rep, step = _measure_pore_cells(scene, pos, verbose)
    if polish and not dead.any():
        pos, vol, vref, rel, dead = _polish_seeds(scene, pos, vol, vref, rel, dead, verbose)
    live = ~dead
    return RedistributeResult(
        positions=pos, volumes=vol, vref=vref, rel=rel,
        max_rel=float(np.abs(rel[live]).max()) if live.any() else float("inf"),
        rms_rel=float(np.sqrt(np.mean(rel[live] ** 2))) if live.any() else float("inf"),
        rounds=rounds, num_added=n_added, num_removed=n_removed, num_dead=int(dead.sum()),
        history=history)


# ---- the phases of redistribute_pore_mesh ------------------------------------------------------

@dataclass(frozen=True)
class _PoreScene:
    """The periodic sphere packing and the graded size field s(φ) of redistribute_pore_mesh."""
    centers: np.ndarray
    radii: np.ndarray
    L: float
    ni: np.ndarray
    nr: np.ndarray
    root: int
    s_lo: float
    s_hi: float
    slope: float

    @property
    def ext(self):
        return (self.L, self.L, self.L)

    def size_of(self, phi):
        # s(φ) = clip(s_lo + slope (φ − s_lo), s_lo, s_hi): slope 1 is the example's s = clip(φ);
        # a Voronoi cell's size cannot change faster than its neighbours' — with slope 1 the
        # targets of adjacent cells differ by up to 8x in volume, so the per-cell error floor is
        # O(1); slope ≲ 0.3 makes the target field resolvable
        return np.clip(self.s_lo + self.slope * (phi - self.s_lo), self.s_lo, self.s_hi)

    def sdf(self, p):
        return sphere_union_sdf(p, self.centers, self.radii, self.ext)


def _wall_shell_seeds(scene, pos, rng):
    """The wall layers by construction (the example's graded-shell heuristic): drop the seeds
    closer than the outermost shell and lay concentric shells around every sphere at distances
    d_k (radial step = in-surface spacing = the local size s(d)), keeping each point near its own
    layer; the redistribution loop then handles the bulk."""
    s_lo, s_hi, L = scene.s_lo, scene.s_hi, scene.L
    dists, d = [], 0.6 * s_lo
    while d < 1.5 * s_hi:
        dists.append(d)
        d += float(scene.size_of(d))
    phi0 = scene.sdf(pos)
    keep = phi0 >= dists[-1] + 0.5 * scene.size_of(dists[-1])
    shells = []
    for d in dists:
        h = float(scene.size_of(d))
        for c, r in zip(scene.centers, scene.radii):
            R = r + d
            n = max(6, int(4 * np.pi * R * R / (h * h)))
            i = np.arange(n) + 0.5
            th = np.arccos(1 - 2 * i / n)
            ph = np.pi * (1 + 5 ** 0.5) * i
            p = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)] * R + c
            p += rng.normal(0, 0.15 * h, p.shape)
            p %= L
            pd = scene.sdf(p)
            shells.append(p[np.abs(pd - d) < 0.5 * h])
    return np.ascontiguousarray(np.vstack([pos[keep]] + shells) % L)


def _measure_pore_cells(scene, p, verbose):
    """Tessellate the seeds p against the walls and return, per seed, (volume, V_ref, rel, dead,
    Lloyd centroid, φ, the build report, the graded volume-descent step)."""
    t = Tessellation()
    t.set_domain(extent=scene.ext)
    t.set_geometry(scene.ni, scene.nr, root=scene.root)
    if verbose:
        print(f"    measure: build N={len(p)}", flush=True)
    t.build(p, strict=False)
    vol = t.get_volumes()
    rep = t.diagnostics.build_report()
    if verbose:
        print(f"    measure: report {rep}", flush=True)
    phi = scene.sdf(p)
    dead = (vol <= 0) | (phi <= 0)
    vref = scene.size_of(phi) ** 3  # ABSOLUTE target: the seed count adjusts until Σ V_ref = fluid
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
    step[ok] = -(g[ok] / gn[ok][:, None]) * (np.minimum(np.abs(rel[ok]), 0.5) * scene.size_of(phi[ok]))[:, None]
    return vol, vref, rel, dead, cen, phi, rep, step


def _split_seeds(scene, pos, cen, phi, big, margin, rng):
    """The second seed of every cell in `big`, offset from its centroid by ~s(φ) (along the wall
    for a wall cell); a candidate closer to the wall than `margin` is dropped."""
    centers, radii, L = scene.centers, scene.radii, scene.L
    new = []
    for i in big:
        s = scene.size_of(phi[i])
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
        if scene.sdf(q[None, :])[0] > margin:
            new.append(q)
    return new


def _relax_seeds(scene, pos, lloyd_steps, lloyd_weight, margin, verbose):
    """`lloyd_steps` sweeps of a Lloyd blend (cell shape) + the graded volume descent (cell
    size), seeds kept off the wall."""
    L = scene.L
    for _ in range(lloyd_steps):
        vol, vref, rel, dead, cen, phi, rep, step = _measure_pore_cells(scene, pos, verbose)
        live = ~dead
        q = (pos + lloyd_weight * (cen - pos) + step) % L
        ok = live & (scene.sdf(q) > margin)
        pos[ok] = q[ok]
        pos = np.ascontiguousarray(pos % L)
    return pos


def _polish_seeds(scene, pos, vol, vref, rel, dead, verbose):
    """The position-only optimiser from the feasible start; kept only if it lowers max |rel|
    without killing a cell."""
    L = scene.L
    r = optimize_pore_mesh(pos, np.ascontiguousarray(vref), scene.centers, scene.radii, scene.ext,
                           search_window=4, max_iter=60, tol=1e-8, method="graphamg")
    p2 = np.ascontiguousarray(np.asarray(r.positions) % L)
    vol2, vref2, rel2, dead2, cen2, phi2, rep2, step2 = _measure_pore_cells(scene, p2, verbose)
    if not dead2.any() and np.abs(rel2).max() < np.abs(rel).max():
        return p2, vol2, vref2, rel2, dead2
    return pos, vol, vref, rel, dead
