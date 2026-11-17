#!/usr/bin/env python3
"""Independent scipy check of the cold power build's outside-seed and empty cells (gate G5 of
DESIGN_NOTE_OUTSIDE_SEED, M7 hub work/M7).

Input: the files `test_determinism --dump K1,K2,... FILE` writes (n^3 seeds in the unit periodic
box): FILE.K (positions then weights, raw float64) and FILE.K.cold (per cell
`i status volume nFacets sorted-neighbours...`). For every cell with status kNegOffset (8) or
kBuried (16), the power cell {y : r_j.y <= 1/2(|r_j|^2 + w_i - w_j)} over the min-image neighbours
within R is solved independently:
  * LP (scipy linprog): the Chebyshev radius t* of the cell, i.e. the largest normalised slack;
    empty <=> t* <= 0;
  * non-empty: HalfspaceIntersection + ConvexHull -> volume and the facet set (the half-spaces
    carrying hull area > 1e-14 h^2).
Pass: per kNegOffset cell, |V - V_dev| / V <= 1e-10 and facet sets equal; per kBuried cell, t* <= 0
(and conversely no cell flagged otherwise). Also reports the largest seed-to-cell distance over cell
size (|seed - centroid| / V^(1/3)) of the outside-seed cells, and an optional named cell.

Usage: oracle_power_cells.py N FILE K1,K2,... [--cell K:i]   -> one RESULT line, exit 1 on a miss.
"""
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

K_NEG, K_BURIED = 8, 16


def load(path, n):
    N = n**3
    a = np.fromfile(path)
    x, w = a[:3 * N].reshape(N, 3), a[3 * N:3 * N + N]
    cold = {}
    for line in open(path + ".cold"):
        f = line.split()
        cold[int(f[0])] = (int(f[1]), float(f[2]), sorted(int(v) for v in f[4:]))
    return x, w, cold


def power_cell(i, x, w, tree, h, R):
    idx = np.array([j for j in tree.query_ball_point(x[i], R) if j != i])
    r = x[idx] - x[i]
    r -= np.round(r)  # min image, seed frame
    off = 0.5 * ((r * r).sum(1) + w[i] - w[idx])
    nr = np.linalg.norm(r, axis=1)
    lp = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[r, nr], b_ub=off,
                 bounds=[(None, None)] * 3 + [(None, h)], method="highs")
    t, c = -lp.fun, lp.x[:3]
    if t <= 0:
        return t, 0.0, [], None
    hs = np.c_[r, -off]
    hi = HalfspaceIntersection(hs, c)
    pts = hi.intersections
    hull = ConvexHull(pts)
    area = np.zeros(len(idx))
    unit = r / nr[:, None]
    for s, eq in zip(hull.simplices, hull.equations):
        a = 0.5 * np.linalg.norm(np.cross(pts[s[1]] - pts[s[0]], pts[s[2]] - pts[s[0]]))
        k = np.argmax(unit @ eq[:3])  # the half-space whose outward normal this triangle carries
        area[k] += a
    facets = sorted(int(idx[k]) for k in np.nonzero(area > 1e-14 * h * h)[0])
    cen = pts.mean(0)
    return t, hull.volume, facets, cen


def main():
    n, base, frames = int(sys.argv[1]), sys.argv[2], [int(v) for v in sys.argv[3].split(",")]
    named = None
    if len(sys.argv) > 5 and sys.argv[4] == "--cell":
        k, i = sys.argv[5].split(":")
        named = (int(k), int(i))
    h = 1.0 / n
    R = 4 * h
    miss = 0
    nNeg = nBur = 0
    maxRel = 0.0
    worst = (0.0, 0.0)  # (dist/size, rel volume error)
    namedOut = ""
    for K in frames:
        path = base + "." + str(K) if len(frames) > 1 else base
        x, w, cold = load(path, n)
        tree = cKDTree(x, boxsize=1.0)
        for i, (st, vdev, fdev) in cold.items():
            if not (st & (K_NEG | K_BURIED)) and named != (K, i):
                continue
            t, V, facets, cen = power_cell(i, x, w, tree, h, R)
            if named == (K, i):
                namedOut = (f" named frame={K} cell={i}: status={st} V/h3={V / h**3:.4f} "
                            f"facets={len(facets)} (device V/h3={vdev / h**3:.4f} "
                            f"facets={len(fdev)})")
            if st & K_BURIED:
                nBur += 1
                if not (t <= 0 and vdev == 0 and not fdev):
                    miss += 1
                    print(f"MISS frame={K} cell={i}: kBuried but Chebyshev radius {t / h:.3e} h")
            elif st & K_NEG:
                nNeg += 1
                rel = abs(V - vdev) / V if V > 0 else float("inf")
                maxRel = max(maxRel, rel)
                if cen is not None:
                    ratio = np.linalg.norm(cen) / V ** (1 / 3)
                    if ratio > worst[0]:
                        worst = (ratio, rel)
                if not (t > 0 and rel <= 1e-10 and facets == fdev):
                    miss += 1
                    print(f"MISS frame={K} cell={i}: t={t / h:.3e}h V/h3={V / h**3:.6f} "
                          f"dev={vdev / h**3:.6f} facets oracle={facets} dev={fdev}")
    print(f"RESULT oracle_power_cells {'PASS' if miss == 0 else 'FAIL'} frames={len(frames)} "
          f"kNegOffset={nNeg} kBuried={nBur} miss={miss} maxRelVol={maxRel:.2e} "
          f"maxDist/size={worst[0]:.2f} (relVolErr there {worst[1]:.1e}){namedOut}")
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
