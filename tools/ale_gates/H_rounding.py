#!/usr/bin/env python3
"""Criterion H of the moving-grid design note (docs/ale_design.md §11 A21; brief ALE_S1_3A_R2): is what stays
open after the rounds rounding of the geometry at nearly degenerate vertices?

For each case: the generators of a dump (DIR/chain_<last>, written by `test_ale_stepper --dump-on-error` or
`--dump-final`) and the closure tail of the cold build of those generators, the REPLAYTAILCELL lines of
`test_ale_frame --replay DIR` (every cell whose closure measure, own closure + disagreement with the
neighbours' facets over the cell's surface, exceeds 2.5e-14). For every such cell, its worst facet (the one with
the largest disagreement, toward `worst_nbr`) is found in a scipy Voronoi diagram of the generators around the
cell (periodic box, zero weights only), and for each vertex of that facet the Delaunay tetrahedron of its four
nearest generators gets the shape quality q = 6√2·vol / l_max³ (1 regular, 0 flat; the definition of
build_s2/gate_logs/S1_3a_r.repro/geom.py). q_min is the smallest over the facet's vertices.

The reading holds for a cell if q_min < 1e-2 and defect · q_min / ε ≤ 1, with defect = the cell's closure
measure (closure_rel) and ε = 2.22e-16. A miss is a miss of the reading, not something to repair.

    python tools/ale_gates/H_rounding.py NAME:DUMPDIR:REPLAYLOG [NAME:DUMPDIR:REPLAYLOG ...]

Prints one HCELL line per cell, one HCASE line per case and ONE final line `H PASS|FAIL ...`; exit 1 on FAIL.
"""
import sys

import numpy as np
from scipy.spatial import Voronoi

EPS = 2.22e-16
Q_BAR = 1e-2


def meta_of(d):
    return dict(kv.split('=', 1) for kv in open(d + '/meta').read().split() if '=' in kv)


def tail_cells(log):
    out = []
    for ln in open(log):
        if ln.startswith('REPLAYTAILCELL'):
            kv = dict(t.split('=', 1) for t in ln.split()[2:] if '=' in t)
            out.append({'cell': int(kv['cell']), 'closure': float(kv['closure_rel']),
                        'worst_nbr': int(kv['worst_nbr']), 'worst_rel': float(kv['worst_rel'])})
    return out


def quality(T):
    vol = abs(np.linalg.det(T[1:] - T[0])) / 6
    lm = max(np.linalg.norm(T[i] - T[j]) for i in range(4) for j in range(i))
    return 6 * np.sqrt(2) * vol / lm ** 3


def worst_facet_q(x, L, h, cell, nbr):
    """q_min over the vertices of the facet cell -> nbr, and the number of its vertices (nan, 0: not found)."""
    dx = (x - x[cell] + L / 2) % L - L / 2
    sel = np.where(np.linalg.norm(dx, axis=1) < 3.5 * h)[0]
    p = dx[sel]
    v = Voronoi(p)
    ic = int(np.where(sel == cell)[0][0])
    for (a, b), rv in zip(v.ridge_points, v.ridge_vertices):
        if ic in (a, b) and sel[b if a == ic else a] == nbr:
            if -1 in rv:
                return float('nan'), 0
            qs = []
            for vi in rv:
                r = np.linalg.norm(p - v.vertices[vi], axis=1)
                qs.append(quality(p[np.argsort(r)[:4]]))
            return min(qs), len(rv)
    return float('nan'), 0


def main(args):
    ok_all, rows, ncells = True, [], 0
    for spec in args:
        name, d, log = spec.split(':', 2)
        meta = meta_of(d)
        nch, L, h = int(meta['chain']), float(meta['L']), float(meta['h'])
        x = np.fromfile(f'{d}/chain_{nch - 1}').reshape(-1, 3)
        w = np.fromfile(f'{d}/weight')
        cells = tail_cells(log)
        if w.size and np.any(w != 0):
            print(f'HCASE {name} FAIL nonzero weights: the Voronoi diagram is not the power diagram')
            ok_all = False
            continue
        if not any(ln.startswith('REPLAYTAIL ') for ln in open(log)):
            print(f'HCASE {name} FAIL no REPLAYTAIL line in {log}')
            ok_all = False
            continue
        ok_case = True
        for c in cells:
            q, nv = worst_facet_q(x, L, h, c['cell'], c['worst_nbr'])
            prod = c['closure'] * q / EPS
            good = q < Q_BAR and prod <= 1.0  # nan fails both
            ok_case = ok_case and good
            print(f"HCELL {name} cell={c['cell']} closure={c['closure']:.3e} worst_nbr={c['worst_nbr']} "
                  f"worst_rel={c['worst_rel']:.3e} facet_vertices={nv} q_min={q:.3e} defect_q_over_eps={prod:.3e} "
                  f"{'ok' if good else 'MISS'}")
            rows.append(good)
        ncells += len(cells)
        ok_all = ok_all and ok_case
        print(f"HCASE {name} {'PASS' if ok_case else 'FAIL'} cells_above_2.5e-14={len(cells)} N={x.shape[0]}")
    print(f"H {'PASS' if ok_all else 'FAIL'} cases={len(args)} cells={ncells} misses={rows.count(False)} "
          f"(q_min < {Q_BAR:g} on the worst facet and closure * q_min / eps <= 1, eps = {EPS:g})")
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
