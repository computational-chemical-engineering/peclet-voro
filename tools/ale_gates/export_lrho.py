"""Export the two-phase pressure operator L_rho of the moving-grid solver for gate G0.4 (work order WO-S0.4).

    python tools/ale_gates/export_lrho.py <mesh dir> <out dir> NAME [NAME ...]

Reads each mesh `<mesh dir>/NAME.*` written by `test_ale_frame --export DIR --cells n` (mesh M1, jitter +-0.3h,
zero weights) with `ale_reference.Mesh`, builds L_rho = D diag(A_f/(rho_f d_f)) D^T with `inv_rho_face`
(1/rho_f = (h_B/rho_A + h_A/rho_B)/d_f) and writes, for three density fields,

    NAME.r1        rho = 1 everywhere
    NAME.r1000     rho = 1 within R = L/4 of the box center, 1000 outside (the two-phase case of test_ale_reference)
    NAME.drop1000  rho = 1000 within R, 1 outside (the heavy drop)

each as the pair PREFIX.csr / PREFIX.label read by core's `test_graph_amg_parallel --file PREFIX` (raw little-endian):

    PREFIX.csr    int64 n, int64 nnz, float64 diag[n], int64 start[n+1], int64 nbr[nnz], float64 coef[nnz]
                  (off-diagonal CSR, columns increasing within a row; duplicate faces between two cells summed)
    PREFIX.label  int64 label[n]: 1 within R, 0 outside (the same in all three files)

Prints one line per file: n, nnz, the symmetry defect and max |L 1| / max |L| (the constants are the null space).
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'python'))
import ale_reference as ar  # noqa: E402


def lrho_csr(m, rho):
    a = m.area * ar.inv_rho_face(m, rho) / m.d
    N = m.N
    rows = np.concatenate([m.A, m.B, m.A, m.B])
    cols = np.concatenate([m.A, m.B, m.B, m.A])
    vals = np.concatenate([a, a, -a, -a])
    L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    L.sum_duplicates()
    L.sort_indices()
    return L


def write(prefix, L, label):
    n = L.shape[0]
    diag = L.diagonal().copy()
    off = L.tolil()
    off.setdiag(0)
    off = off.tocsr()
    off.eliminate_zeros()
    off.sort_indices()
    with open(prefix + '.csr', 'wb') as f:
        np.array([n, off.nnz], dtype='<i8').tofile(f)
        diag.astype('<f8').tofile(f)
        off.indptr.astype('<i8').tofile(f)
        off.indices.astype('<i8').tofile(f)
        off.data.astype('<f8').tofile(f)
    label.astype('<i8').tofile(prefix + '.label')
    return off.nnz


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for name in sys.argv[3:]:
        m = ar.Mesh(src, name)
        inside = np.linalg.norm(m.pos - 0.5 * m.L, axis=1) < 0.25 * m.L[0]
        label = inside.astype(np.int64)
        for tag, rho in (('r1', np.ones(m.N)), ('r1000', np.where(inside, 1.0, 1000.0)),
                         ('drop1000', np.where(inside, 1000.0, 1.0))):
            L = lrho_csr(m, rho)
            nnz = write(os.path.join(out, f'{name}.{tag}'), L, label)
            sym = abs(L - L.T).max() / abs(L).max()
            const = np.abs(L @ np.ones(m.N)).max() / abs(L).max()
            print(f'LRHO {name}.{tag} n={m.N} nnz={nnz} inside={int(inside.sum())} sym={sym:.1e} '
                  f'|L1|/|L|={const:.1e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
