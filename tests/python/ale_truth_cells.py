"""Independent truth for the cells of a frame dump (brief ALE_S1_3A_R3, item 2).

Usage: python ale_truth_cells.py DUMP_DIR FACETS_LOG [--radius R]

DUMP_DIR is a dump of `test_ale_stepper --dump-on-error` (meta, weight, chain_<q>; the last chain
is the frame); FACETS_LOG the output of `test_ale_frame --replay DUMP_DIR --facets` (REPLAYSTATUS,
REPLAYFACET lines: the engine's cold build). For every listed cell (listed=1: the open cells) and
every cell printed with its facets, the exact cell is computed with scipy (Voronoi of the
min-imaged generators within R h of the cell's generator; mesh Z has zero weights, checked) and
compared with the engine's: neighbours missing in the engine (truth only) and extra (engine only),
their distance / h and the truth facet area / the cell surface, and the engine's area error
sum_g |A_eng - A_true| / S. The truth is certified: every neighbour lies within 2 r_max of the
generator (r_max = the cell's farthest vertex), so R > 2 r_max / h + 1 is checked.
One line per listed cell (TRUTHCELL), one per missing/extra facet (TRUTHNBR), a summary (TRUTH).
"""

import sys

import numpy as np
from scipy.spatial import Voronoi


def polygon_area_vector(P, n):
    """Area vector of a planar convex polygon P (k x 3) with normal direction n."""
    c = P.mean(0)
    n = n / np.linalg.norm(n)
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    ang = np.arctan2((P - c) @ v, (P - c) @ u)
    Q = P[np.argsort(ang)]
    a = np.zeros(3)
    for k in range(len(Q)):
        a += np.cross(Q[k] - c, Q[(k + 1) % len(Q)] - c)
    a *= 0.5
    return a if a @ n >= 0 else -a


def truth_cell(x, L, h, i, R):
    """Exact Voronoi cell i: {neighbour id: (outward area vector, distance)}, r_max."""
    dx = (x - x[i] + L / 2) % L - L / 2
    sel = np.where(np.linalg.norm(dx, axis=1) < R * h)[0]
    p = dx[sel]
    vor = Voronoi(p)
    ic = int(np.where(sel == i)[0][0])
    reg = vor.regions[vor.point_region[ic]]
    assert -1 not in reg, "unbounded cell: radius too small"
    rmax = np.linalg.norm(vor.vertices[reg], axis=1).max()
    out = {}
    for (a, b), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if ic not in (a, b):
            continue
        o = b if a == ic else a
        P = vor.vertices[rv]
        A = polygon_area_vector(P, p[o])
        out[int(sel[o])] = (A, np.linalg.norm(p[o]))
    return out, rmax


def main():
    d, logf = sys.argv[1], sys.argv[2]
    R = float(sys.argv[sys.argv.index('--radius') + 1]) if '--radius' in sys.argv else 12.0
    meta = dict(kv.split('=', 1) for kv in open(d + '/meta').read().split())
    nch, h, L = int(meta['chain']), float(meta['h']), float(meta['L'])
    x = np.fromfile(f'{d}/chain_{nch - 1}').reshape(-1, 3)
    w = np.fromfile(f'{d}/weight')
    assert np.all(w == 0), 'weighted dump: this script computes the Voronoi (zero-weight) truth'
    eng, listed, status = {}, [], {}
    for line in open(logf):
        t = line.split()
        if t and t[0] == 'REPLAYFACET':
            eng.setdefault(int(t[1]), {})[int(t[2])] = np.array([float(s) for s in t[3:6]])
        elif t and t[0] == 'REPLAYSTATUS':
            kv = dict(s.split('=') for s in t[1:])
            status[int(kv['cell'])] = int(kv['status'])
            if kv['listed'] == '1':
                listed.append(int(kv['cell']))
    sign = None
    nmiss = nextra = 0
    for i in sorted(eng):
        tr, rmax = truth_cell(x, L, h, i, R)
        assert 2 * rmax / h + 1 < R, f'cell {i}: radius {R} h not certified (2 r_max = {2 * rmax / h:.2f} h)'
        e = eng[i]
        if sign is None:  # the engine's orientation: outward (+1) or inward (-1)
            j = next(iter(set(e) & set(tr)))
            sign = 1.0 if e[j] @ tr[j][0] > 0 else -1.0
        S = sum(np.linalg.norm(a) for a, _ in tr.values())
        miss = sorted(set(tr) - set(e))
        extra = sorted(set(e) - set(tr))
        common = set(tr) & set(e)
        err = sum(np.linalg.norm(sign * e[j] - tr[j][0]) for j in common)
        err += sum(np.linalg.norm(tr[j][0]) for j in miss) + sum(np.linalg.norm(e[j]) for j in extra)
        # the measured disagreement of the frame: sum_g |A_g + A_recip(g)| (both from the engine)
        dis = sum(np.linalg.norm(e[j] + eng[j][i]) for j in e if j in eng and i in eng[j])
        nmax = max(dd for _, dd in tr.values()) / h
        if i in listed:
            print(f'TRUTHCELL cell={i} status={status[i]} facets_true={len(tr)} facets_eng={len(e)} '
                  f'missing={len(miss)} extra={len(extra)} r_max/h={rmax / h:.3f} '
                  f'max_nbr_dist/h={nmax:.3f} S/h2={S / h**2:.4f} err_rel={err / S:.4e} '
                  f'missing_area_rel={sum(np.linalg.norm(tr[j][0]) for j in miss) / S:.4e}')
        for j in miss:
            nmiss += 1
            print(f'TRUTHNBR cell={i} missing j={j} dist/h={tr[j][1] / h:.3f} '
                  f'A/S={np.linalg.norm(tr[j][0]) / S:.4e} j_status={status.get(j, -1)} '
                  f'j_has_i={int(j in eng and i in eng[j])} listed={int(i in listed)}')
        for j in extra:
            nextra += 1
            print(f'TRUTHNBR cell={i} extra j={j} A_eng/S={np.linalg.norm(e[j]) / S:.4e} listed={int(i in listed)}')
        if i in listed:
            print(f'TRUTHDIS cell={i} engine_dis_rel(listed facets)={dis / S:.4e}')
    print(f'TRUTH cells_checked={len(eng)} listed={len(listed)} missing={nmiss} extra={nextra} '
          f'orientation={"outward" if sign > 0 else "inward"} radius={R}h')


if __name__ == '__main__':
    main()
