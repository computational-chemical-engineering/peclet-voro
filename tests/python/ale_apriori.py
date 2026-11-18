"""A-priori test of the bulk operators of the one-phase step on a static mesh (work order S1.2e).

The exact Taylor-Green or ABC field (rho = 1, nu = 0.01, t = 0, k = 2 pi, unit periodic box) is put into the
matrix-free operators of ale_reference.py; the discrete accelerations of transport (T), upwind/reconstruction jump
(Q), viscous friction (Z) and pressure (P) are compared with the continuous equation after the discrete projection
P_h a = a - M^-1 C'^T x, S x = C' a, S = C' M^-1 C'^T (the Stepper's exact_projection with a pyamg-preconditioned
solve of L). Five projected parts and their sum:
  Tp = P_h(aT - G)   Gp = P_h(G)   Qp = P_h(aQ)   Zp = P_h(aZ - nu Lap u)   Up = P_h(nu Lap u) - nu Lap u
with G = grad p / rho = -(u.grad)u (exact). Meshes: an exported FaceMesh (M1 = mesh Z, M2 = mesh W) or the cubic
lattice LAT. Matrices are never built (3N is too large above 16^3) except for the --check comparison at 16^3.

Usage: python ale_apriori.py EXPORT_DIR FIELD MESH N [--check]      FIELD tg|abc, MESH LAT|M1|M2 (EXPORT_DIR unused
for LAT). One RESULT line on stdout; --check prints CHECK id PASS|FAIL value lines and no RESULT.
"""
import argparse
import os
import sys
import time

import numpy as np
import pyamg
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ale_reference as ar  # noqa: E402
import ale_reference_stepper as ars  # noqa: E402

RHO, NU, K = 1.0, 0.01, 2.0 * np.pi
L_RTOL, P_RTOL, MAXIT = 1e-12, 1e-11, 500


# ------------------------------------------------------------------------------------------------- lattice mesh
class Lattice:
    """n^3 cubic cells, one face per cell and axis (A the cell, B its +axis neighbor, periodic), with every
    attribute that ar.Mesh sets."""

    def __init__(self, n):
        h = 1.0 / n
        N = n ** 3
        i = np.arange(N)
        ijk = np.stack([i % n, (i // n) % n, i // (n * n)], axis=1)
        self.N, self.F, self.numNegativeHeight = N, 3 * N, 0
        self.pos = (ijk + 0.5) * h
        axis = np.tile(np.arange(3), N)
        cell = np.repeat(i, 3)
        nb = ijk.copy()
        nb = np.repeat(nb, 3, axis=0)
        nb[np.arange(3 * N), axis] = (nb[np.arange(3 * N), axis] + 1) % n
        self.A = cell
        self.B = nb[:, 0] + n * (nb[:, 1] + n * nb[:, 2])
        self.n = np.eye(3)[axis]
        self.area = np.full(3 * N, h * h)
        self.d = np.full(3 * N, h)
        self.hA = np.full(3 * N, 0.5 * h)
        self.hB = np.full(3 * N, 0.5 * h)
        self.cA = 0.5 * h * self.n
        self.conn = h * self.n
        self.cB = self.cA - self.conn
        f = np.arange(3 * N)
        cells = np.concatenate([self.A, self.B])
        faces = np.concatenate([f, f])
        signs = np.concatenate([np.ones(3 * N), -np.ones(3 * N)])
        order = np.lexsort((faces, cells))
        self.cellFace = faces[order]
        self.cellFaceSign = signs[order]
        self.cellFaceOffset = np.concatenate([[0], np.cumsum(np.bincount(cells, minlength=N))])
        self.V = np.full(N, h ** 3)
        self.w = np.zeros(N)
        self.L = np.ones(3)
        self.directory, self.name = None, f'LAT_{n}'
        self.q = self.cA - self.hA[:, None] * self.n
        self.h = h


def load_mesh(export_dir, mesh, n):
    if mesh == 'LAT':
        return Lattice(n)
    return ar.Mesh(export_dir, f'{mesh}_{n}')


# --------------------------------------------------------------------------------------------------- the fields
def fields(field, x):
    """u, p, G = grad p / rho, nu Lap u at the points x (t = 0)."""
    u = ars.FIELDS[field](x, NU, 0.0)
    X, Y, Zc = x[:, 0], x[:, 1], x[:, 2]
    if field == 'tg':
        G = -0.5 * K * np.c_[np.sin(2 * K * X), np.sin(2 * K * Y), np.zeros(len(x))]
        lap = -2.0 * NU * K * K * u
    else:
        J = K * np.zeros((len(x), 3, 3))
        J[:, 0, 1], J[:, 0, 2] = -K * np.sin(K * Y), K * np.cos(K * Zc)
        J[:, 1, 0], J[:, 1, 2] = K * np.cos(K * X), -K * np.sin(K * Zc)
        J[:, 2, 0], J[:, 2, 1] = -K * np.sin(K * X), K * np.cos(K * Y)
        G = -np.einsum('iab,ia->ib', J, u)
        lap = -NU * K * K * u
    return u, pressure(field, x), G, lap


def pressure(field, x):
    if field == 'tg':
        return 0.25 * RHO * (np.cos(2 * K * x[:, 0]) + np.cos(2 * K * x[:, 1]))
    u = ars.FIELDS[field](x, NU, 0.0)
    return -0.5 * RHO * np.sum(u * u, axis=1)


# ------------------------------------------------------------------------------------------------------ solvers
class Solvers:
    """L (assembled from a, A, B), its V-cycle, and the discrete projection."""

    def __init__(self, m):
        self.m = m
        F = m.F
        f = np.arange(F)
        self.a = m.area * ar.inv_rho_face(m, np.full(m.N, RHO)) / m.d
        D = sp.coo_matrix((np.concatenate([np.ones(F), -np.ones(F)]),
                           (np.concatenate([m.A, m.B]), np.concatenate([f, f]))), shape=(m.N, F)).tocsr()
        self.L = (D @ sp.diags(self.a) @ D.T).tocsr()
        self.vcycle = pyamg.smoothed_aggregation_solver(self.L).aspreconditioner(cycle='V')
        self.rhoV = RHO * m.V
        self.its_max = 0

    def prec(self, r):
        z = self.vcycle.matvec(r - r.mean())
        return z - z.mean()

    def cg(self, op, b, rtol, atol=0.0):
        try:
            x, its = ars.pcg(op, b, self.prec, rtol, atol=atol, maxit=MAXIT)
        except RuntimeError:
            raise SystemExit(f'STOP: CG did not reach {rtol:g} in {MAXIT} iterations')
        self.its_max = max(self.its_max, its)
        return x - x.mean(), its

    def Lsolve(self, r):
        return self.cg(lambda v: self.L @ v, r - r.mean(), L_RTOL)[0]

    def flux(self, u):
        """Phi = I_F u and the compact projection F = Phi + a Delta_n phi."""
        m = self.m
        Phi = ar.IF_mf(m, u)
        phi = self.Lsolve(ar.div_mf(m, Phi))
        return Phi, Phi + self.a * ar.dn_mf(m, phi)

    def schur(self, q):
        m = self.m
        return ar.Cp_mf(m, ars.apply_inverse_mass(self.rhoV, ar.CpT_mf(m, q)))

    def project(self, a):
        """P_h a = a - M^-1 C'^T x, S x = C' a; returns P_h a and x."""
        m = self.m
        # absolute floor: C'a of a divergence-free field is rounding noise, a relative target is then unreachable
        atol = 100.0 * np.finfo(float).eps * np.sqrt(m.N) * 6.0 * m.h ** 2 * np.sqrt(np.sum(a * a, axis=1)).max()
        x, _ = self.cg(self.schur, ar.Cp_mf(m, a), P_RTOL, atol)
        return a - ars.apply_inverse_mass(self.rhoV, ar.CpT_mf(m, x)), x


def accelerations(m, u, p, Fm):
    """aT, aQ, aZ, aP (N, 3): force / (rho V)."""
    rV = (RHO * m.V)[:, None]
    mu = np.full(m.F, RHO * NU)
    aT = (-ar.transport_mf(m, u, Fm) - ar.face_scatter_mf(m, Fm[:, None] * ar.centroid_correction_mf(m, u))) / rV
    aQ = (-ar.upwind_mf(m, u, Fm)
          - ar.face_scatter_mf(m, (0.5 * np.abs(Fm))[:, None] * ar.reconstruction_jump_mf(m, u))) / rV
    aZ = -ar.zeta_mf(m, u, mu) / rV
    aP = ar.CpT_mf(m, p) / rV
    return aT, aQ, aZ, aP


def norm(m, a):
    return np.sqrt(np.sum(m.V * np.sum(a * a, axis=1)))


def dot(m, a, b):
    return np.sum(m.V * np.sum(a * b, axis=1))


def low_wavenumber(m, X, mmax):
    """sqrt(sum_{k in K_m} |Xhat_k|^2), K_m = {2 pi (a, b, c) integers, max |.| <= mmax},
    Xhat_k = sum_i V_i X_i exp(-i k.x_i) (complex 3-vector)."""
    tot = 0.0
    rng = range(-mmax, mmax + 1)
    for a in rng:
        for b in rng:
            for c in rng:
                ph = np.exp(-1j * K * (m.pos @ np.array([a, b, c], dtype=float)))
                xh = (m.V * ph) @ X
                tot += np.sum(np.abs(xh) ** 2)
    return np.sqrt(tot)


# ------------------------------------------------------------------------------------------------------- the case
def run_case(m, field):
    t0 = time.time()
    s = Solvers(m)
    u, p, G, lap = fields(field, m.pos)
    Phi, F = s.flux(u)
    Fm = RHO * F
    aT, aQ, aZ, aP = accelerations(m, u, p, Fm)
    raw = {'T': aT - G, 'Q': aQ, 'Z': aZ - lap, 'P': aP + G}
    Pl = s.project(lap)[0]
    parts = {'T': s.project(aT - G)[0], 'G': s.project(G)[0], 'Q': s.project(aQ)[0],
             'Z': s.project(aZ - lap)[0], 'U': Pl - lap}
    parts['TOT'] = sum(parts[k] for k in 'TGQZU')
    Su = norm(m, u)
    out = {'Sp': norm(m, G), 'Sz': norm(m, lap), 'Su': Su}
    out.update({f'raw_{k}': norm(m, v) for k, v in raw.items()})
    out.update({f'nrm_{k}': norm(m, v) for k, v in parts.items()})
    out.update({f'coh_{k}': dot(m, v, u) / Su ** 2 for k, v in parts.items()})
    for mm in (1, 2):
        for k, v in parts.items():
            out[f'lp{mm}_{k}'] = low_wavenumber(m, v - out[f'coh_{k}'] * u, mm)
    keys = ['Sp', 'Sz', 'Su'] + [f'raw_{k}' for k in 'TQZP'] + [f'nrm_{k}' for k in ('T', 'G', 'Q', 'Z', 'U', 'TOT')] \
        + [f'coh_{k}' for k in ('T', 'G', 'Q', 'Z', 'U', 'TOT')] \
        + [f'lp{mm}_{k}' for mm in (1, 2) for k in ('T', 'G', 'Q', 'Z', 'U', 'TOT')]
    return out, keys, s.its_max, time.time() - t0


# ---------------------------------------------------------------------------------------------------- the checks
def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def checks(m, field_names=('tg', 'abc')):
    ok_all = True

    def report(cid, ok, val):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f'CHECK {cid} {"PASS" if ok else "FAIL"} {val}')

    s = Solvers(m)
    st = ars.Stepper(m, rho=RHO, nu=NU)
    u, p, G, lap = fields('tg', m.pos)
    Phi, F = s.flux(u)
    Fm = RHO * F
    aT, aQ, aZ, aP = accelerations(m, u, p, Fm)
    N = m.N
    ue = u.ravel()
    rV = (RHO * m.V)[:, None]
    Wd = 0.5 * np.abs(Fm)
    Q = (st.tm.DeltaT @ sp.diags(np.repeat(Wd, 3)) @ st.tm.Delta).tocsr()
    e6 = -(st.tm.DeltaT @ (np.repeat(Fm, 3) * (st.tm.Delta_c @ ue)))
    e8 = -(st.tm.DeltaT @ (np.repeat(Wd, 3) * (st.tm.Jump @ ue)))
    refs = {'aT': ((-(st.tm.T(Fm) @ ue) + e6).reshape(N, 3) / rV, aT),
            'aQ': ((-(Q @ ue) + e8).reshape(N, 3) / rV, aQ),
            'aZ': (-(st.zeta @ ue).reshape(N, 3) / rV, aZ),
            'aP': ((st.CpT @ p).reshape(N, 3) / rV, aP)}
    e = {k: rel(v[1], v[0]) for k, v in refs.items() if v[0] is not None}
    rnd = np.random.default_rng(1).random(N)
    eL = rel(s.L @ rnd, ar.Lrho_mf(m, rnd, np.full(N, RHO)))
    eF = rel(F, st.project_flux(st.IF @ ue))
    report('C0', max(e.values()) <= 1e-12 and eL <= 1e-12 and eF <= 1e-9,
           'aT=%.2e aQ=%.2e aZ=%.2e aP=%.2e L=%.2e F=%.2e' % (e['aT'], e['aQ'], e['aZ'], e['aP'], eL, eF))
    Pa, _ = s.project(aT)
    r = np.linalg.norm(ar.Cp_mf(m, Pa)) / np.linalg.norm(ar.Cp_mf(m, aT))
    dp, _ = st.exact_projection(aT)
    ref = aT + ars.apply_inverse_mass(st.rhoV, (st.CpT @ dp).reshape(N, 3))
    eP = np.linalg.norm(Pa - ref) / np.linalg.norm(aT)
    report('C1', r <= 1e-9 and eP <= 1e-8, 'resid=%.2e vs_stepper=%.2e' % (r, eP))
    errs = []
    for fn in field_names:
        _, _, Gf, _ = fields(fn, m.pos)
        fd = np.zeros_like(Gf)
        for b in range(3):
            dx = np.zeros(3)
            dx[b] = 1e-5
            fd[:, b] = (pressure(fn, m.pos + dx) - pressure(fn, m.pos - dx)) / (2e-5 * RHO)
        errs.append(rel(Gf, fd))
    report('C2', max(errs) <= 1e-6, ' '.join('%s=%.2e' % (fn, er) for fn, er in zip(field_names, errs)))
    rP = norm(m, s.project(aP)[0]) / norm(m, aP)
    report('C3', rP <= 1e-8, '%.2e' % rP)
    su = norm(m, u)
    e5 = abs(low_wavenumber(m, u, 1) - su) / su
    report('C5', e5 <= (1e-12 if isinstance(m, Lattice) else 5e-2), '%.2e' % e5)
    if isinstance(m, Lattice):
        close = np.zeros((N, 3))
        cells = np.concatenate([m.A, m.B])
        faces = np.concatenate([np.arange(m.F)] * 2)
        sg = np.concatenate([np.ones(m.F), -np.ones(m.F)])
        np.add.at(close, cells, (sg * m.area[faces])[:, None] * m.n[faces])
        c = np.abs(close).max() / m.h ** 2
        dv = abs(m.V.sum() - 1.0)
        # the cell lists must carry the same faces and signs
        cl = np.zeros((N, 3))
        owner = np.repeat(np.arange(N), np.diff(m.cellFaceOffset))
        np.add.at(cl, owner, (m.cellFaceSign * m.area[m.cellFace])[:, None] * m.n[m.cellFace])
        c = max(c, np.abs(cl).max() / m.h ** 2)
        report('C4', c <= 1e-14 and dv <= 1e-14, 'closure=%.2e sumV-1=%.2e' % (c, dv))
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('export_dir')
    ap.add_argument('field', choices=['tg', 'abc'])
    ap.add_argument('mesh', choices=['LAT', 'M1', 'M2'])
    ap.add_argument('n', type=int)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    m = load_mesh(a.export_dir, a.mesh, a.n)
    if a.check:
        sys.exit(0 if checks(m) else 1)
    out, keys, its, wall = run_case(m, a.field)
    head = f'RESULT field={a.field} mesh={a.mesh} n={a.n} N={m.N} h={m.h:.6e} '
    print(head + ' '.join(f'{k}={out[k]:.6e}' for k in keys) + f' its_max={its} wall={wall:.1f}')


if __name__ == '__main__':
    main()
