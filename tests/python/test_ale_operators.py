"""Gate G1.1, criteria K0 and K1 (work order WO-S1.1, docs/ale_design.md §7; criteria fixed 2026-11-15 in the brief
ALE_S1_1): the device kernels of include/peclet/voro/ale/operators.hpp against the host reference
tests/python/ale_reference.py on the meshes M1, M2, M3 exported by `test_ale_frame --export DIR`.

  python test_ale_operators.py inputs DIR [M1 M2 M3]
      K0, checks of the reference itself (a miss is a defect of the reference), then the random input fields
      DIR/<M>.in_<name> (fixed seed) for `test_ale_operators --apply DIR`:
        (i)   U = a + Bx through differences dU_f = B conn_f, mu = 1: 1/2 dU^T zeta_d dU = mu |sym B|^2 sum_i V_i
              (zeta_d the friction as a form on face differences, Matrices.zeta_d)                <= 1e-13 relative
        (ii)  face_gradient_mf: G_f n_f = dU_f/d_f, random field                                     <= 1e-14 relative
        (iii) the off-diagonal entries of the matrix L_rho equal -(A_f/d_f)(h_Bf/rho_A + h_Af/rho_B)/d_f, written out
              here, at density ratio 1000                                                            <= 1e-14 relative
        (iv)  C'U = tr(B) V_i for U = a + Bx with tr(B) != 0 (Cp_local_mf)                           <= 1e-13 relative
  python test_ale_operators.py compare DIR ODIR [M1 M2 M3]
      K1: every output ODIR/<M>.out_<name> of the --apply run against the reference on the same inputs
      (L_rho at rho = 1 and at ratio 1000 with the sphere R = L/4; zeta at mu = 1 and mu_f random in [1, 100];
      gamma at rho random in [1, 1000], psi = cos(2 pi y/L))                                       <= 1e-13 relative

Relative: max over entries of |device - reference| over max over entries of |reference|. One line per mesh and
check, then the summary line `K0 PASS|FAIL ...` or `K1 PASS|FAIL ...`; exit status 1 on FAIL.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ale_reference as ar  # noqa: E402

TOL_K0 = {'i': 1e-13, 'ii': 1e-14, 'iii': 1e-14, 'iv': 1e-13}
TOL_K1 = 1e-13
G_UNIFORM = np.array([0.3, -1.0, 0.2])
G_BAR = np.array([-0.2, 0.7, -1.3])


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(np.abs(a - b).max() / np.abs(b).max())


def sphere_rho(m):
    """rho = 1 within L/4 of the box center, 1000 elsewhere (as test_ale_reference.py R4)."""
    inside = np.linalg.norm(m.pos - 0.5 * m.L, axis=1) < 0.25 * m.L[0]
    return np.where(inside, 1.0, 1000.0)


def psi_fields(m):
    k = 2.0 * np.pi / m.L[1]
    return np.cos(k * m.pos[:, 1]), np.cos(k * (m.pos[m.A, 1] + m.cA[:, 1]))


def check_reference(m, rng):
    """K0 (i)-(iv) on one mesh: returns {criterion: relative error}."""
    out = {}
    mats = ar.Matrices(m)
    # (i) friction energy of a linear field through differences, uniform mu = 1
    B = rng.uniform(-1, 1, (3, 3))
    dU = m.conn @ B.T
    E = 0.5 * dU.ravel() @ (mats.zeta_d @ dU.ravel())
    S = 0.5 * (B + B.T)
    ref = float((S * S).sum() * m.V.sum())
    out['i'] = abs(E - ref) / abs(ref)
    # (ii) the normal part of the face gradient is the two-point difference
    U = rng.uniform(-1, 1, (m.N, 3))
    dU = ar.face_differences(m, U)
    Gf = ar.face_gradient_mf(m, dU=dU)
    out['ii'] = rel(np.einsum('fab,fb->fa', Gf, m.n), dU / m.d[:, None])
    # (iii) off-diagonal entries of L_rho at ratio 1000, the formula written out
    rho = sphere_rho(m)
    mats.set_rho(rho)
    expect = {}
    for f in range(m.F):
        a, b = int(m.A[f]), int(m.B[f])
        e = -(m.area[f] / m.d[f]) * (m.hB[f] / rho[a] + m.hA[f] / rho[b]) / m.d[f]
        expect[(a, b)] = expect.get((a, b), 0.0) + e
    L = mats.Lrho.tocsr()
    got = np.array([[L[a, b], L[b, a]] for (a, b) in expect])
    want = np.array([[e, e] for e in expect.values()])
    out['iii'] = rel(got, want)
    # (iv) C'U = tr(B) V_i for a linear field with nonzero trace
    B = rng.uniform(-1, 1, (3, 3)) + np.eye(3)
    a = rng.uniform(-1, 1, 3)
    U = a + m.pos @ B.T
    Cp = ar.Cp_local_mf(m, U, m.conn @ B.T)
    out['iv'] = rel(Cp, np.trace(B) * m.V)
    return out


def write_inputs(m, rng):
    N, F = m.N, m.F
    psi_c, psi_f = psi_fields(m)
    fields = {
        'U': rng.uniform(-1, 1, (N, 3)),
        'p': rng.uniform(-1, 1, N),
        'phi': rng.uniform(-1, 1, F),
        'xdot': rng.uniform(-1, 1, (N, 3)),
        'wdot': rng.uniform(-1, 1, N) * m.h,
        'rhoSphere': sphere_rho(m),
        'rhoRand': rng.uniform(1, 1000, N),
        'mu': rng.uniform(1, 100, F),
        'psiCell': psi_c,
        'psiFace': psi_f,
        'g': G_UNIFORM,
        'gbar': G_BAR,
    }
    for k, v in fields.items():
        np.ascontiguousarray(v, dtype='<f8').tofile(os.path.join(m.directory, f'{m.name}.in_{k}'))


def read_inputs(m):
    def rd(k, shape):
        return np.fromfile(os.path.join(m.directory, f'{m.name}.in_{k}'), dtype='<f8').reshape(shape)

    N, F = m.N, m.F
    return dict(U=rd('U', (N, 3)), p=rd('p', N), phi=rd('phi', F), xdot=rd('xdot', (N, 3)), wdot=rd('wdot', N),
                rhoSphere=rd('rhoSphere', N), rhoRand=rd('rhoRand', N), mu=rd('mu', F), psiCell=rd('psiCell', N),
                psiFace=rd('psiFace', F), g=rd('g', 3), gbar=rd('gbar', 3))


def laplacian_diag(m, rho):
    a = m.area * ar.inv_rho_face(m, rho) / m.d
    d = np.zeros(m.N)
    np.add.at(d, m.A, a)
    np.add.at(d, m.B, a)
    return d


def references(m, x):
    ones = np.ones(m.N)
    return {
        'div': ar.div_mf(m, x['phi']),
        'dn': ar.dn_mf(m, x['p']),
        'grad': ar.grad_mf(m, x['U']),
        'IF': ar.IF_mf(m, x['U']),
        'IFT': ar.IFT_twopass(m, x['phi']),
        'Cp': ar.Cp_mf(m, x['U']),
        'CpT': ar.CpT_mf(m, x['p'], twopass=True),
        'Ldiag_1': laplacian_diag(m, ones),
        'Lapply_1': ar.Lrho_mf(m, x['p'], ones),
        'Ldiag_1000': laplacian_diag(m, x['rhoSphere']),
        'Lapply_1000': ar.Lrho_mf(m, x['p'], x['rhoSphere']),
        'swept': ar.swept_mf(m, x['xdot'], x['wdot']),
        'zeta_1': ar.zeta_mf(m, x['U']),
        'zeta_mu': ar.zeta_mf(m, x['U'], mu=x['mu']),
        'gamma': ar.gamma_face(m, x['rhoRand'], x['psiCell'], x['psiFace']),
        'gammaU': ar.gamma_face_uniform(m, x['rhoRand'], x['g']),
        'bodyCell': x['rhoRand'][:, None] * m.V[:, None] * x['gbar'][None, :],
    }


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('inputs', 'compare'):
        print(__doc__)
        return 2
    mode, d = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    odir = None
    if mode == 'compare':
        odir, rest = rest[0], rest[1:]
    names = rest or ['M1', 'M2', 'M3']
    ok = True
    if mode == 'inputs':
        worst = {k: 0.0 for k in TOL_K0}
        for idx, name in enumerate(names):
            m = ar.Mesh(d, name)
            rng = np.random.default_rng(20261117 + idx)
            r = check_reference(m, rng)
            good = all(r[k] <= TOL_K0[k] for k in TOL_K0)
            ok &= good
            for k in worst:
                worst[k] = max(worst[k], r[k])
            print(f'K0 {name} {"PASS" if good else "FAIL"} ' + ' '.join(f'{k}={v:.2e}' for k, v in r.items()))
            write_inputs(m, rng)
        print(f'K0 {"PASS" if ok else "FAIL"} ' + ' '.join(f'{k}={v:.2e}' for k, v in worst.items()))
        return 0 if ok else 1
    worst, where = 0.0, ''
    for name in names:
        m = ar.Mesh(d, name)
        x = read_inputs(m)
        refs = references(m, x)
        for k, ref in refs.items():
            dev = np.fromfile(os.path.join(odir, f'{name}.out_{k}'), dtype='<f8')
            if dev.size != np.asarray(ref).size:
                print(f'K1 {name} {k} FAIL size {dev.size} != {np.asarray(ref).size}')
                ok = False
                continue
            e = rel(dev, ref)
            good = e <= TOL_K1
            ok &= good
            if e > worst:
                worst, where = e, f'{name}.{k}'
            print(f'K1 {name} {k} {"PASS" if good else "FAIL"} rel={e:.2e}')
        # localization aid: the scatter and two-pass forms of the reference agree on these inputs
        print(f'K1 {name} info IFT_scatter_vs_twopass={rel(ar.IFT_mf(m, x["phi"]), refs["IFT"]):.2e}')
    print(f'K1 {"PASS" if ok else "FAIL"} max={worst:.2e} ({where})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
