"""Distortion of a moving-grid frame (brief ALE_S1_3A_R3, item 4), from the generators alone.

Usage: python ale_mesh_distortion.py DUMP_DIR [--window SW]

DUMP_DIR: a dump of `test_ale_stepper --dump-on-error / --dump-final` (meta, weight, chain_<q>;
the last chain is the frame; zero weights, checked). The exact Voronoi neighbours are the edges
of the periodic Delaunay triangulation (scipy, the generators with their images within a margin;
the margin is certified against the longest edge found). Per cell: the largest neighbour distance
/ h, the facet count (= Delaunay degree), r_max / h (the farthest vertex = the largest circumradius
of the incident tetrahedra). Against the cold build's gather (SW blocks of the counting-sort grid,
block size csz as tess_grid.hpp computes it: one seed per block), the cells whose completeness
test fails (2 r_max >= SW csz: kIncomplete) and the cells with a neighbour beyond SW csz.
One line: DISTORTION ... (median / 99 % / max of each quantity, and the two counts).
"""

import sys

import numpy as np
from scipy.spatial import Delaunay


def main():
    d = sys.argv[1]
    sw = int(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 4
    meta = dict(kv.split('=', 1) for kv in open(d + '/meta').read().split())
    nch, h, L = int(meta['chain']), float(meta['h']), float(meta['L'])
    x = np.fromfile(f'{d}/chain_{nch - 1}').reshape(-1, 3) % L
    N = len(x)
    assert np.all(np.fromfile(f'{d}/weight') == 0), 'weighted dump: Voronoi truth only'
    m = min(0.25 * L, 10 * h)
    pts, ids = [x], [np.arange(N)]
    for s in np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).T.reshape(-1, 3):
        if not s.any():
            continue
        y = x + s * L
        k = np.all((y > -m) & (y < L + m), axis=1)
        pts.append(y[k])
        ids.append(np.where(k)[0])
    P, I = np.vstack(pts), np.concatenate(ids)
    tri = Delaunay(P)
    S = tri.simplices
    # circumradius of every tetrahedron
    A = P[S[:, 1:]] - P[S[:, :1]]
    b = 0.5 * np.einsum('tij,tij->ti', A, A)
    # flat tetrahedra (qhull's triangulation of a cospherical cluster, volume ~ 0) have no finite
    # circumsphere and are skipped; their count is reported
    det = np.linalg.det(A)
    flat = np.abs(det) <= 1e-12 * np.einsum('tij,tij->t', A, A) ** 1.5
    c = np.zeros((len(S), 3))
    c[~flat] = np.linalg.solve(A[~flat], b[~flat][..., None])[..., 0]
    rc = np.linalg.norm(c, axis=1)
    own = S < N
    rmax = np.zeros(N)
    for k in range(4):
        sel = own[:, k]
        np.maximum.at(rmax, S[sel, k], rc[sel])
    # neighbour pairs (edges with at least one owned end), distance and degree
    e = np.vstack([S[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
    e = e[(e[:, 0] < N) | (e[:, 1] < N)]
    e = np.unique(np.sort(e, axis=1), axis=0)
    dist = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)
    pair = np.unique(np.sort(np.stack([I[e[:, 0]], I[e[:, 1]]], 1), axis=1), axis=0, return_index=True)[1]
    e, dist = e[pair], dist[pair]
    assert dist.max() < m, f'margin {m / h:.1f} h not certified (longest edge {dist.max() / h:.2f} h)'
    gi, gj = I[e[:, 0]], I[e[:, 1]]
    deg = np.bincount(gi, minlength=N) + np.bincount(gj, minlength=N)
    dmax = np.zeros(N)
    np.maximum.at(dmax, gi, dist)
    np.maximum.at(dmax, gj, dist)
    spacing = np.cbrt(L**3 / N)  # one seed per block (Power and GPU; tess_grid.hpp)
    dim = max(1, int(np.floor(L / spacing)))
    cov = sw * L / dim
    q = lambda v: f'{np.median(v):.3f}/{np.quantile(v, 0.99):.3f}/{v.max():.3f}'
    print(f'DISTORTION dump={d.rstrip("/").split("/")[-1]} N={N} step={meta.get("step")} '
          f'max_nbr_dist/h(med/99%/max)={q(dmax / h)} facets(med/99%/max)={q(deg.astype(float))} '
          f'r_max/h(med/99%/max)={q(rmax / h)} window={sw} coverage/h={cov / h:.4f} '
          f'incomplete(2r_max>=cov)={int(np.sum(2 * rmax >= cov))} nbr_beyond_cov={int(np.sum(dmax > cov))} '
          f'need_window={int(np.ceil(dmax.max() / (L / dim)))} need_window_reach={int(np.ceil(2 * rmax.max() / (L / dim)))} flat_tets={int(flat.sum())}')


if __name__ == '__main__':
    main()
