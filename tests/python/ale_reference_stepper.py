"""Host-only reference of the time step of the moving-grid solver (docs/ale_design.md §4.1) for ONE phase on a
STATIC 3D mesh (work order S1.2a; gate G1.2r). The mesh is one exported FaceMesh (`test_ale_frame --export`), so
there is one frame, Phi^m = 0 and the volumes are constant. Operators come from ale_reference.py; this file adds
the step, its energy budget and the runs. It is meant to be plainly the formula, not fast.

The step (sign convention of §A.3: C' = D I_F, pressure force +C'^T p):
  S1  U_ext = 1.5 U^n - 0.5 U^{n-1} (U_ext = U^n at the first step).
  S3  Phi_raw = I_F U_ext; compact projection phi = L_rho^+ D Phi_raw, F = Phi_raw + a (Delta_n phi),
      a_f = A_f/(rho_f d_f), so D F = 0 to the solver tolerance; face mass flux Fm = rho F.
  S4  (M + tau/2 (T + Q) + tau/2 zeta) U* = (M - tau/2 (T + Q) - tau/2 zeta) U^n
          - tau Delta^T (Fm delta) - tau Delta^T W (R - Delta) U_ext + tau C'^T p^n + tau B
      (scheme v3f of the 2D oracle a4_cell_ale.py; T, Q, R - Delta, delta in ale_reference.py; B the body force of
      amendment A7, cell part rho V gbar plus face part I_F^T gamma).
  S5  mode F: delta p = -omega L_rho^+ C'U*/tau, omega = min(1, 1/mu_hat), mu_hat by power iteration on
      (C'M^-1C'^T, L_rho) (§4.3: 20 cold iterations at the first step, one per step after, mu_hat = max(lambda,
      0.9 mu_hat_prev)); mode R: delta p = -(C'M^-1C'^T)^+ C'U*/tau by CG preconditioned with L_rho^+.
      U^{n+1} = U* + tau M^-1 C'^T delta p, p^{n+1} = p^n + delta p.
  S6  Phi_1 = I_F U^{n+1} with its compact projection; max tau |D Phi_1| / V reported.

Energy budget, K = 1/2 U^T M U, Ubar = (U* + U^n)/2 (the oracle's B2-B8 in our sign convention, B1 = 0 on a
static mesh):
  B2 = -tau Ubar^T T Ubar  (T is skew: zero up to rounding; the oracle's B2 is the work of the diagonal part
       1/2 (D Fm) U, which the brief's T does not carry)
  B3 = +tau Ubar^T C'^T p^n            B4 = -1/2 (U^{n+1} - U*)^T M (U^{n+1} - U*)     B5 = +tau dp^T C' U^{n+1}
  B6 = -tau Ubar^T Delta^T (Fm delta)  B7 = -tau Ubar^T zeta Ubar                     B8a = -tau Ubar^T Q Ubar
  B8b = -tau Ubar^T Delta^T W (R - Delta) U_ext                                       B9 = +tau Ubar^T B
  Bres = -Ubar^T (rhs - A U*), the work of the residual of the predictor solve (§4.9).
closure = (K^{n+1} - K^n) - sum of all; B4, B7, B8a are <= 0, the others carry no sign.

Linear solves: L_rho by a grounded sparse LU (exact on the mean-zero subspace); predictor by a sparse LU when
3N <= PREDICTOR_DIRECT_MAX, else right-preconditioned GMRES (3x3 block Jacobi) to a relative 1e-13 with the
residual work booked; mode R by preconditioned CG to a relative 1e-13 (floor: the rounding level of C'U*).
The mu_hat power iteration uses the exact L_rho solve (the device uses a loose one, §4.3).

Usage: python ale_reference_stepper.py run EXPORT_DIR CASE [--log FILE]   one run, RESULT line at the end
       python ale_reference_stepper.py judge LOG_DIR                       gate G1.2r from the RESULT lines
       python ale_reference_stepper.py trace EXPORT_DIR CASE OUT_DIR [--steps K] [--pred-rtol X]
                                     start state and per-step U*, F, U, p for the device (gate G1.2 C1/C2)
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ale_reference as ar  # noqa: E402

TWO_PI = 2.0 * np.pi
PREDICTOR_DIRECT_MAX = 1536          # 3N: sparse LU at 8^3, GMRES above
PREDICTOR_RTOL = 1e-13
MODE_R_RTOL = 1e-13
MU_COLD_ITERATIONS = 20
MU_EXACT_ITERATIONS = 200
COURANT = 0.4
WALL_LIMIT = 7200.0                  # stop rule of the brief: a run projected beyond 2 h stops


# ------------------------------------------------------------------------------------------- mass (isolated)
# The ONE definition of the mass: lumped, M = diag(rho V_i) per component. Every mass in this file goes through
# these three functions (predictor, projection, Schur operator, energies); mass_matrix is the explicit form of
# apply_mass for the direct predictor solve and the block preconditioner. Another mass replaces this block only.
def apply_mass(rhoV, U):
    return rhoV[:, None] * U


def apply_inverse_mass(rhoV, U):
    return U / rhoV[:, None]


def mass_matrix(rhoV):
    return sp.diags(np.repeat(rhoV, 3))


def kinetic_energy(rhoV, U):
    return 0.5 * np.sum(U * apply_mass(rhoV, U))


# --------------------------------------------------------------------------------------------- solvers
def pcg(Sop, b, prec, rtol, atol=0.0, maxit=5000):
    """Preconditioned CG on S x = b; stops on the unpreconditioned residual |b - S x| <= max(rtol |b|, atol)."""
    x = np.zeros_like(b)
    r = b.copy()
    tol = max(rtol * np.linalg.norm(b), atol)
    if np.linalg.norm(r) <= tol:
        return x, 0
    z = prec(r)
    d = z.copy()
    rz = r @ z
    for k in range(1, maxit + 1):
        Sd = Sop(d)
        al = rz / (d @ Sd)
        x += al * d
        r -= al * Sd
        if np.linalg.norm(r) <= tol:
            return x, k
        z = prec(r)
        rzn = r @ z
        d = z + (rzn / rz) * d
        rz = rzn
    raise RuntimeError('PCG: no convergence')


def gmres(Aop, b, x0, prec, rtol, restart=60, maxit=3000):
    """Right-preconditioned restarted GMRES (modified Gram-Schmidt, Givens). Stops on the true residual
    |b - A x| <= rtol |b| (checked at every restart). Returns x and the number of inner iterations."""
    tol = rtol * np.linalg.norm(b)
    x = x0.copy()
    its = 0
    while True:
        r = b - Aop(x)
        beta = np.linalg.norm(r)
        if beta <= tol:
            return x, its
        if its >= maxit:
            raise RuntimeError(f'GMRES: no convergence ({beta / np.linalg.norm(b):.2e})')
        V = [r / beta]
        Z = []
        H = np.zeros((restart + 1, restart))
        cs, sn = np.zeros(restart), np.zeros(restart)
        g = np.zeros(restart + 1)
        g[0] = beta
        k = 0
        for j in range(restart):
            z = prec(V[j])
            Z.append(z)
            w = Aop(z)
            for i in range(j + 1):
                H[i, j] = w @ V[i]
                w -= H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            V.append(w / H[j + 1, j] if H[j + 1, j] > 0 else w)
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            den = np.hypot(H[j, j], H[j + 1, j])
            cs[j], sn[j] = H[j, j] / den, H[j + 1, j] / den
            H[j, j] = den
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            its += 1
            k = j + 1
            if abs(g[j + 1]) <= 0.5 * tol or H[j, j] == 0.0:
                break
        y = np.linalg.solve(np.triu(H[:k, :k]), g[:k])
        for i in range(k):
            x += y[i] * Z[i]


# ----------------------------------------------------------------------------------------------- fields
def taylor_green(x, nu, t):
    """2D Taylor-Green extruded along z, k = 2 pi: u = (sin kx cos ky, -cos kx sin ky, 0) e^{-2 nu k^2 t}."""
    k = TWO_PI
    u = np.c_[np.sin(k * x[:, 0]) * np.cos(k * x[:, 1]), -np.cos(k * x[:, 0]) * np.sin(k * x[:, 1]),
              np.zeros(len(x))]
    return u * np.exp(-2.0 * nu * k * k * t)


def abc_flow(x, nu, t):
    """ABC (Beltrami) flow A = B = C = 1, k = 2 pi: u = (sin kz + cos ky, sin kx + cos kz, sin ky + cos kx)
    e^{-nu k^2 t}; pressure -rho |u|^2/2."""
    k = TWO_PI
    X, Y, Zc = x[:, 0], x[:, 1], x[:, 2]
    u = np.c_[np.sin(k * Zc) + np.cos(k * Y), np.sin(k * X) + np.cos(k * Zc), np.sin(k * Y) + np.cos(k * X)]
    return u * np.exp(-nu * k * k * t)


FIELDS = {'tg': taylor_green, 'abc': abc_flow}


# ---------------------------------------------------------------------------------------------- stepper
class Stepper:
    """One phase (rho, nu uniform) on the static mesh m."""

    def __init__(self, m, rho=1.0, nu=0.01, upwind=1.0, pred_rtol=PREDICTOR_RTOL):
        self.m = m
        N, F = m.N, m.F
        self.rho, self.nu, self.upwind, self.pred_rtol = rho, nu, upwind, pred_rtol
        rho_c = np.full(N, rho)
        self.rhoV = rho * m.V
        self.mx = ar.Matrices(m, rho=rho_c, mu=np.full(F, rho * nu))
        self.tm = ar.TransportMatrices(self.mx)
        self.IF, self.D, self.Cp = self.mx.IF, self.mx.D, self.mx.Cp
        self.CpT = self.Cp.T.tocsr()
        self.absCp = abs(self.Cp).tocsr()
        self.zeta = self.mx.zeta
        self.a = m.area * ar.inv_rho_face(m, rho_c) / m.d
        e0 = sp.csc_matrix(([1.0], ([0], [0])), shape=(N, N))
        self.L = self.mx.Lrho
        self._lu = splu((self.L + e0).tocsc())
        self.Mmat = mass_matrix(self.rhoV)
        # diagonal 3x3 blocks of zeta (block-Jacobi preconditioner of the predictor)
        z = self.zeta.tocoo()
        keep = (z.row // 3) == (z.col // 3)
        self.zeta_blocks = np.zeros((N, 3, 3))
        np.add.at(self.zeta_blocks, (z.row[keep] // 3, z.row[keep] % 3, z.col[keep] % 3), z.data[keep])
        self.direct = 3 * N <= PREDICTOR_DIRECT_MAX
        self.mu_v = None
        self.mu_hat = None

    # ---- pressure-type solves
    def Lsolve(self, r):
        """L_rho^+ r on the mean-zero subspace (grounded LU, then the mean removed: exact)."""
        x = self._lu.solve(r - r.mean())
        return x - x.mean()

    def schur(self, q):
        """C' M^-1 C'^T q."""
        return self.Cp @ apply_inverse_mass(self.rhoV, (self.CpT @ q).reshape(-1, 3)).ravel()

    def project_flux(self, Phi):
        """Compact projection: phi = L_rho^+ D Phi, Phi + a (Delta_n phi) (D of the result = 0)."""
        phi = self.Lsolve(self.D @ Phi)
        return Phi + self.a * ar.dn_mf(self.m, phi)

    def courant(self, F, tau):
        out = np.zeros(self.m.N)
        np.add.at(out, self.m.A, np.maximum(F, 0.0))
        np.add.at(out, self.m.B, np.maximum(-F, 0.0))
        return tau * (out / self.m.V).max()

    def power_iteration(self, v, nit):
        """nit iterations v <- L^+ S v (L-normalized); returns v and the Rayleigh quotient v.Sv / v.Lv."""
        for _ in range(nit):
            v = self.Lsolve(self.schur(v))
            v /= np.sqrt(v @ (self.L @ v))
        return v, (v @ self.schur(v)) / (v @ (self.L @ v))

    def mu_seed(self):
        v = np.where(np.random.default_rng(20261117).random(self.m.N) < 0.5, -1.0, 1.0)
        return v - v.mean()

    def mu_exact(self):
        return self.power_iteration(self.mu_seed(), MU_EXACT_ITERATIONS)[1]

    def exact_projection(self, U, tau=1.0):
        """Mode R projection of U: dp = -(C'M^-1C'^T)^+ C'U/tau; returns dp and the CG iterations."""
        b = (self.Cp @ U.ravel()) / tau
        atol = 4.0 * np.finfo(float).eps * np.linalg.norm(self.absCp @ np.abs(U.ravel())) / tau
        x, its = pcg(self.schur, b, self.Lsolve, MODE_R_RTOL, atol=atol)
        return -x, its

    # ---- one step
    def step(self, U, Uext, p, tau, mode, body=None):
        m, rhoV = self.m, self.rhoV
        N = m.N
        r = {}
        # S3: transport flux from U_ext, compact projection
        Phi_raw = self.IF @ Uext.ravel()
        F = self.project_flux(Phi_raw)
        r['divF'] = tau * np.abs(self.D @ F).max() / m.V.min()
        r['cfl'] = self.courant(F, tau)
        Fm = self.rho * F
        # S4: predictor
        T = self.tm.T(Fm)
        Wd = self.upwind * 0.5 * np.abs(Fm)
        Q = (self.tm.DeltaT @ sp.diags(np.repeat(Wd, 3)) @ self.tm.Delta).tocsr()
        TQ = (T + Q).tocsr()
        ue = Uext.ravel()
        e6 = -(self.tm.DeltaT @ (np.repeat(Fm, 3) * (self.tm.Delta_c @ ue)))
        e8 = -(self.tm.DeltaT @ (np.repeat(Wd, 3) * (self.tm.Jump @ ue)))
        fp = self.CpT @ p
        bf = np.zeros(3 * N) if body is None else body.ravel()
        u0 = U.ravel()

        def Aop(x):
            return apply_mass(rhoV, x.reshape(N, 3)).ravel() + 0.5 * tau * (TQ @ x + self.zeta @ x)

        rhs = (apply_mass(rhoV, U).ravel() - 0.5 * tau * (TQ @ u0 + self.zeta @ u0)
               + tau * (e6 + e8 + fp + bf))
        if self.direct:
            A = (self.Mmat + 0.5 * tau * (TQ + self.zeta)).tocsc()
            us = splu(A).solve(rhs)
            r['pred_its'] = 0
        else:
            qd = np.zeros(N)
            np.add.at(qd, m.A, Wd)
            np.add.at(qd, m.B, Wd)
            blocks = (rhoV[:, None, None] * np.eye(3) + 0.5 * tau * (qd[:, None, None] * np.eye(3)
                                                                      + self.zeta_blocks))
            binv = np.linalg.inv(blocks)

            def prec(v):
                return np.einsum('iab,ib->ia', binv, v.reshape(N, 3)).ravel()

            us, r['pred_its'] = gmres(Aop, rhs, ue.copy(), prec, self.pred_rtol)
        res = rhs - Aop(us)
        r['Ustar'], r['F'] = us.reshape(N, 3), F        # for the trace subcommand (no other use)
        r['pred_res'] = np.linalg.norm(res) / np.linalg.norm(rhs) if np.linalg.norm(rhs) > 0 else 0.0
        ub = 0.5 * (us + u0)
        r['B2'] = -tau * ub @ (T @ ub)
        r['B3'] = tau * ub @ fp
        r['B6'] = tau * ub @ e6
        r['B7'] = -tau * ub @ (self.zeta @ ub)
        r['B8a'] = -tau * ub @ (Q @ ub)
        r['B8b'] = tau * ub @ e8
        r['B9'] = tau * ub @ bf
        r['Bres'] = -ub @ res
        r['res_mom'] = res.reshape(N, 3).sum(0)        # momentum of the residual: P* = P^n + tau sum B - this
        # S5: projection
        Us = us.reshape(N, 3)
        if mode == 'R':
            dp, r['projR_its'] = self.exact_projection(Us, tau)
            r['omega'] = 1.0
        else:
            if self.mu_v is None:
                self.mu_v, lam = self.power_iteration(self.mu_seed(), MU_COLD_ITERATIONS)
                self.mu_hat = lam
            else:
                self.mu_v, lam = self.power_iteration(self.mu_v, 1)
                self.mu_hat = max(lam, 0.9 * self.mu_hat)
            omega = min(1.0, 1.0 / self.mu_hat)
            dp = -omega * self.Lsolve((self.Cp @ us) / tau)
            r['omega'], r['mu_hat'], r['projR_its'] = omega, self.mu_hat, 0
        U1 = Us + tau * apply_inverse_mass(rhoV, (self.CpT @ dp).reshape(N, 3))
        p1 = p + dp
        dU = U1 - Us
        r['B4'] = -0.5 * np.sum(dU * apply_mass(rhoV, dU))
        r['B5'] = tau * dp @ (self.Cp @ U1.ravel())
        r['B1'] = 0.0
        r['dK'] = 0.5 * np.sum((U1 - U) * apply_mass(rhoV, U1 + U))
        r['K1'] = kinetic_energy(rhoV, U1)
        r['booked'] = sum(r[k] for k in BOOKED)
        r['closure'] = r['dK'] - r['booked']
        r['unsigned'] = sum(r[k] for k in UNSIGNED)
        # S6: compact flux of U^{n+1}
        Phi1 = self.project_flux(self.IF @ U1.ravel())
        r['divPhi1'] = tau * np.max(np.abs(self.D @ Phi1) / m.V)
        r['divU1'] = tau * np.max(np.abs(self.Cp @ U1.ravel()) / m.V)
        r['P1'] = apply_mass(rhoV, U1).sum(0)
        r['absP1'] = np.abs(apply_mass(rhoV, U1)).sum()
        return U1, p1, r


BOOKED = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8a', 'B8b', 'B9', 'Bres')
UNSIGNED = ('B2', 'B3', 'B5', 'B6', 'B8b', 'B9', 'Bres')     # B4, B7, B8a <= 0 by construction


# ----------------------------------------------------------------------------------------------- the cases
# name -> (mesh family, cells per side, field, mode, steps or None (to t = 1), options)
def case_table():
    c = {}
    c['r0_zero_F'] = ('W', 8, 'zero', 'F', 10, {})
    c['r0_zero_R'] = ('W', 8, 'zero', 'R', 10, {})
    c['r0_unif_F'] = ('W', 8, 'unif', 'F', 100, {})
    c['r0_unif_R'] = ('W', 8, 'unif', 'R', 100, {})
    for fam in 'ZW':
        for n in (8, 16, 32):
            for fld in ('tg', 'abc'):
                c[f'{fld}_{fam}{n}_F'] = (fam, n, fld, 'F', None, {})
            c[f'tg_{fam}{n}_R'] = (fam, n, 'tg', 'R', None, {})
    c['r3_tg_W16_F'] = ('W', 16, 'tg', 'F', 2000, {'sample': 100})
    # R2 control: the runs of R1/R2 with the predictor solved to a relative 1e-14 instead of 1e-13
    for fam in 'ZW':
        for mode in 'FR':
            c[f'tg_{fam}16_{mode}_rtol14'] = (fam, 16, 'tg', mode, None, {'pred_rtol': 1e-14})
    c['r6i_W16'] = ('W', 16, 'zero', 'F', 100, {'rho': 1000.0, 'force': 'psi_face'})
    c['r6iii_W16'] = ('W', 16, 'zero', 'F', 100, {'rho': 1000.0, 'force': 'psi_cell'})
    c['r6ii_rest_W16'] = ('W', 16, 'zero', 'F', 100, {'force': 'gbar'})
    c['r6ii_tg_W16'] = ('W', 16, 'tg', 'F', 100, {'force': 'gbar', 'twin': True})
    # R6 (ii') controls, reported: the run from rest with the predictor to a relative 1e-14, and at W 8^3 (direct)
    c['r6ii_rest_W16_rtol14'] = ('W', 16, 'zero', 'F', 100, {'force': 'gbar', 'pred_rtol': 1e-14})
    c['r6ii_rest_W8'] = ('W', 8, 'zero', 'F', 100, {'force': 'gbar'})
    # R5 control (stop rule): the same runs with the upwind part switched off
    for n in (8, 16, 32):
        c[f'tg_Z{n}_F_noupw'] = ('Z', n, 'tg', 'F', None, {'upwind': 0.0})
    return c


GBAR = np.array([0.0, 0.0, -1.0])


def mesh_name(fam, n):
    return {'Z': 'M1', 'W': 'M2'}[fam] + f'_{n}'


def body_force(st, kind):
    """Body force of amendment A7 for the run options: psi = cos(2 pi y / L) as face form (I_F^T gamma) or as the
    cell values rho V (-grad psi)(x_i) (control), or the uniform gbar in cell form rho V gbar."""
    m = st.m
    rho = np.full(m.N, st.rho)
    k = TWO_PI / m.L[1]
    if kind == 'psi_face':
        psi_c = np.cos(k * m.pos[:, 1])
        psi_f = np.cos(k * (m.pos[m.A, 1] + m.cA[:, 1]))
        return (st.IF.T @ ar.gamma_face(m, rho, psi_c, psi_f)).reshape(-1, 3), -st.rho * psi_c
    if kind == 'psi_cell':
        f = np.zeros((m.N, 3))
        f[:, 1] = st.rho * m.V * k * np.sin(k * m.pos[:, 1])
        return f, -st.rho * np.cos(k * m.pos[:, 1])
    if kind == 'gbar':
        return st.rho * m.V[:, None] * GBAR[None, :], None
    raise ValueError(kind)


def run_case(export_dir, name, log):
    fam, n, fld, mode, nsteps, opt = case_table()[name]
    t_start = time.time()
    m = ar.Mesh(export_dir, mesh_name(fam, n))
    rho = opt.get('rho', 1.0)
    nu = 0.01
    st = Stepper(m, rho=rho, nu=nu, upwind=opt.get('upwind', 1.0),
                 pred_rtol=opt.get('pred_rtol', PREDICTOR_RTOL))
    t_setup = time.time() - t_start
    # time step: flux Courant number 0.4 of the start field (Taylor-Green for the fields at rest), rounded so that
    # an integer number of steps reaches t = 1; the same rule at every resolution
    if fld == 'unif':
        U = np.tile([0.3, -0.7, 0.5], (m.N, 1))
    elif fld == 'zero':
        U = np.zeros((m.N, 3))
    else:
        U = FIELDS[fld](m.pos, nu, 0.0)
    Uc = U if fld != 'zero' else taylor_green(m.pos, nu, 0.0)
    dp0, its0 = st.exact_projection(Uc)
    Uc = Uc + apply_inverse_mass(st.rhoV, (st.CpT @ dp0).reshape(-1, 3))
    c1 = st.courant(st.project_flux(st.IF @ Uc.ravel()), 1.0)
    nst1 = int(np.ceil(c1 / COURANT))
    tau = 1.0 / nst1
    nst = nst1 if nsteps is None else nsteps
    body, p = None, np.zeros(m.N)
    if 'force' in opt:
        body, pstart = body_force(st, opt['force'])
        if pstart is not None:
            p = pstart
    mu_ex = st.mu_exact() if mode == 'F' else float('nan')
    log(f"## RUN {name} mesh={mesh_name(fam, n)} N={m.N} F={m.F} field={fld} mode={mode} rho={rho} nu={nu} "
        f"tau={tau:.6e} steps={nst} courant_start={c1 * tau:.4f} upwind={st.upwind} predictor="
        f"{'direct' if st.direct else 'gmres'} mu_exact={mu_ex:.6f} setup={t_setup:.1f}s")
    if fld in ('tg', 'abc'):
        U = Uc                                                 # start field projected exactly (oracle `run`)
        _, p, _ = st.step(U, U, np.zeros(m.N), tau, 'R', body)  # start pressure from one trial step, mode R
    U0 = U.copy()
    norm0 = np.sqrt(np.sum(m.V[:, None] * U0 ** 2))
    P0 = apply_mass(st.rhoV, U0).sum(0)
    runs = [(U, p, None)]
    if opt.get('twin'):
        st2 = Stepper.__new__(Stepper)
        st2.__dict__.update(st.__dict__)
        st2.mu_v, st2.mu_hat = None, None
        runs.append((U.copy(), p.copy(), st2))
    S = dict(closure=0.0, mom=0.0, divPhi1=0.0, divF=0.0, excess=-np.inf, pred_its=[], projR_its=[], cfl=0.0,
             omega_mu=0.0, mu_ratio=np.inf, maxU=0.0, pred_res=0.0, Bsum={k: 0.0 for k in BOOKED},
             unif_dev=0.0, gt_dev=0.0, mean_dev=0.0, twin_dev=0.0, mu_samples=0, mom_booked=0.0)
    res_mom = np.zeros(3)
    Uprev = None
    U, p = U0.copy(), p.copy()
    if opt.get('twin'):
        Ut, pt, Utprev = U0.copy(), p.copy(), None
    t_loop = time.time()
    last = None
    for k in range(nst):
        Uext = U if Uprev is None else 1.5 * U - 0.5 * Uprev
        U1, p1, r = st.step(U, Uext, p, tau, mode, body)
        if opt.get('twin'):
            Utext = Ut if Utprev is None else 1.5 * Ut - 0.5 * Utprev
            Ut1, pt1, _ = runs[1][2].step(Ut, Utext, pt, tau, mode, None)
            Utprev, Ut, pt = Ut, Ut1, pt1
        Uprev, U, p = U, U1, p1
        t = tau * (k + 1)
        K1 = r['K1']
        Kref = max(K1, kinetic_energy(st.rhoV, Uprev))
        if Kref > 0:
            S['closure'] = max(S['closure'], abs(r['closure']) / Kref)
            S['excess'] = max(S['excess'], (r['dK'] - r['unsigned']) / Kref)
        res_mom += r['res_mom']
        if r['absP1'] > 0 and body is None:
            S['mom'] = max(S['mom'], np.abs(r['P1'] - P0).max() / r['absP1'])
            S['mom_booked'] = max(S['mom_booked'], np.abs(r['P1'] - P0 + res_mom).max() / r['absP1'])
        for key in ('divPhi1', 'divF', 'cfl', 'pred_res'):
            S[key] = max(S[key], r[key])
        S['pred_its'].append(r['pred_its'])
        S['projR_its'].append(r['projR_its'])
        for key in BOOKED:
            S['Bsum'][key] += r[key]
        S['maxU'] = max(S['maxU'], np.abs(U).max())
        if mode == 'F':
            S['omega_mu'] = max(S['omega_mu'], r['omega'] * mu_ex)
            sample = opt.get('sample', 1)
            if (k + 1) % sample == 0 or k == 0:
                S['mu_ratio'] = min(S['mu_ratio'], r['mu_hat'] / mu_ex)
                S['mu_samples'] += 1
        if fld == 'unif':
            S['unif_dev'] = max(S['unif_dev'], np.abs(U - U0).max() / np.linalg.norm(U0[0]))
        if opt.get('force') == 'gbar':
            if fld == 'zero':
                S['gt_dev'] = max(S['gt_dev'], np.abs(U - GBAR * t).max() / (np.linalg.norm(GBAR) * t))
            mean = apply_mass(st.rhoV, U).sum(0) / st.rhoV.sum() - P0 / st.rhoV.sum()
            S['mean_dev'] = max(S['mean_dev'], np.abs(mean - GBAR * t).max() / (np.linalg.norm(GBAR) * t))
            if opt.get('twin'):
                S['twin_dev'] = max(S['twin_dev'], np.abs(U - Ut - GBAR * t).max() / (np.linalg.norm(GBAR) * t))
        if k == 4 and n >= 32:
            proj = (time.time() - t_loop) / 5 * nst + t_setup
            if proj > WALL_LIMIT:
                log(f"RESULT case={name} status=TIMEOUT wall_per_step={(time.time() - t_loop) / 5:.2f} "
                    f"projected={proj:.0f}")
                return dict(case=name, status='TIMEOUT')
        if opt.get('sample') and (k + 1) % opt['sample'] == 0:
            log(f"step n={k + 1} t={t:.4f} K={K1:.6e} closure/K={abs(r['closure']) / max(K1, 1e-300):.2e} "
                f"excess/K={(r['dK'] - r['unsigned']) / max(K1, 1e-300):+.2e} mu_hat={r.get('mu_hat', 0):.5f} "
                f"omega={r['omega']:.4f}")
        last = r
        if not np.isfinite(K1):
            log(f"RESULT case={name} status=BLOWUP step={k + 1}")
            return dict(case=name, status='BLOWUP')
    wall = (time.time() - t_loop) / nst
    out = dict(case=name, status='DONE', N=m.N, tau=tau, steps=nst, t=tau * nst, closure=S['closure'],
               mom=S['mom'], mom_booked=S['mom_booked'], divPhi1=S['divPhi1'], divF=S['divF'], cfl=S['cfl'],
               excess=S['excess'],
               pred_its_mean=np.mean(S['pred_its']), pred_its_max=np.max(S['pred_its']), pred_res=S['pred_res'],
               projR_its_mean=np.mean(S['projR_its']), projR_its_max=np.max(S['projR_its']),
               wall_per_step=wall, setup=t_setup, mu_exact=mu_ex, maxU=S['maxU'], P0=np.abs(P0).max(),
               K0=kinetic_energy(st.rhoV, U0), K1=kinetic_energy(st.rhoV, U))
    if mode == 'F':
        out.update(omega_mu=S['omega_mu'], mu_ratio=S['mu_ratio'], mu_samples=S['mu_samples'],
                   mu_hat=last['mu_hat'])
    if fld in ('tg', 'abc'):
        e = U - FIELDS[fld](m.pos, nu, tau * nst)
        dec = {'tg': np.exp(-2.0 * nu * TWO_PI ** 2 * tau * nst), 'abc': np.exp(-nu * TWO_PI ** 2 * tau * nst)}[fld]
        out['err'] = np.sqrt(np.sum(m.V[:, None] * e ** 2)) / norm0 / dec
    if fld == 'unif':
        out['unif_dev'] = S['unif_dev']
    if opt.get('force') in ('psi_face', 'psi_cell'):
        gmax = TWO_PI / m.L[1]
        out['maxU_rel_bound'] = S['maxU'] / (gmax * m.h) ** 0.5
        out['maxU_rel_drive'] = np.abs(U).max() / (tau * nst * gmax)
    if opt.get('force') == 'gbar':
        out.update(gt_dev=S['gt_dev'], mean_dev=S['mean_dev'])
        if opt.get('twin'):
            out['twin_dev'] = S['twin_dev']
    log('TERMS ' + name + ' last_step ' + ' '.join(f'{k}={last[k]:+.6e}' for k in BOOKED)
        + f" dK={last['dK']:+.6e} closure={last['closure']:+.3e} K={last['K1']:.6e}")
    log('TERMS ' + name + ' run_sum ' + ' '.join(f'{k}={v:+.6e}' for k, v in S['Bsum'].items()))
    log('RESULT ' + ' '.join(f'{k}={v:.6e}' if isinstance(v, float) else f'{k}={v}' for k, v in out.items()))
    return out


# ------------------------------------------------------------------------------- trace (work order S1.2b)
def trace_case(export_dir, name, out_dir, nsteps, pred_rtol):
    """Write the start state of case `name` and, per step k = 1..nsteps, U*, F, U, p (raw float64) and the
    scalars (k, omega, mu_hat) for the device comparison of gate G1.2 C1/C2. The start state is that of
    run_case (the same operations in the same order: the start field projected exactly, tau from the flux
    Courant number 0.4, the start pressure from one trial step in mode R); run_case itself is not changed."""
    fam, n, fld, mode, _, opt = case_table()[name]
    m = ar.Mesh(export_dir, mesh_name(fam, n))
    rho, nu = opt.get('rho', 1.0), 0.01
    st = Stepper(m, rho=rho, nu=nu, upwind=opt.get('upwind', 1.0), pred_rtol=pred_rtol)
    if fld not in ('tg', 'abc') or 'force' in opt:
        raise ValueError(f'trace: {name} is not a Taylor-Green or ABC case without a body force')
    U = FIELDS[fld](m.pos, nu, 0.0)
    dp0, _ = st.exact_projection(U)
    Uc = U + apply_inverse_mass(st.rhoV, (st.CpT @ dp0).reshape(-1, 3))
    c1 = st.courant(st.project_flux(st.IF @ Uc.ravel()), 1.0)
    tau = 1.0 / int(np.ceil(c1 / COURANT))
    U = Uc
    _, p, _ = st.step(U, U, np.zeros(m.N), tau, 'R', None)
    os.makedirs(out_dir, exist_ok=True)
    pre = os.path.join(out_dir, name + '.')
    with open(pre + 'meta', 'w') as f:
        f.write(f'mesh={mesh_name(fam, n)} mode={mode} field={fld} rho={rho!r} nu={nu!r} tau={tau!r} '
                f'steps={nsteps} pred_rtol={pred_rtol!r}\n')
    U.astype('<f8').tofile(pre + 'U0')
    p.astype('<f8').tofile(pre + 'p0')
    Uprev = None
    with open(pre + 'ref_scalars', 'w') as sc:
        for k in range(1, nsteps + 1):
            Uext = U if Uprev is None else 1.5 * U - 0.5 * Uprev
            U1, p1, r = st.step(U, Uext, p, tau, mode, None)
            Uprev, U, p = U, U1, p1
            for key, arr in (('Ustar', r['Ustar']), ('F', r['F']), ('U', U), ('p', p)):
                np.asarray(arr, dtype='<f8').tofile(pre + f'ref_{key}_{k}')
            sc.write(f"{k} {float(r['omega'])!r} {float(r.get('mu_hat', float('nan')))!r}\n")
    print(f'TRACE {name} mesh={mesh_name(fam, n)} mode={mode} tau={tau:.6e} steps={nsteps} written to {out_dir}')
    return 0


# ------------------------------------------------------------------------------------------------- the gate
def read_results(log_dir):
    res = {}
    for fn in sorted(os.listdir(log_dir)):
        if not (fn.startswith('S1_2a.') and fn.endswith('.log')):
            continue
        with open(os.path.join(log_dir, fn)) as f:
            for line in f:
                if line.startswith('RESULT '):
                    d = dict(kv.split('=', 1) for kv in line.split()[1:])
                    res[d['case']] = {k: (v if k in ('case', 'status') else float(v)) for k, v in d.items()}
    return res


def judge(log_dir):
    R = read_results(log_dir)
    lines, ok_all = [], True

    def get(case, key):
        d = R.get(case)
        if d is None or d.get('status') != 'DONE':
            return None
        return d.get(key)

    def verdict(cid, ok, text):
        nonlocal ok_all
        word = 'PASS' if ok is True else ('FAIL' if ok is False else 'REPORT')
        if ok is False:
            ok_all = False
        lines.append(f'{cid} {word} {text}')

    def miss(cases):
        return [c for c in cases if get(c, 'steps') is None]

    # R0
    cs = ['r0_zero_F', 'r0_zero_R', 'r0_unif_F', 'r0_unif_R']
    if miss(cs):
        verdict('R0', False, f'missing {miss(cs)}')
    else:
        z = max(get(c, 'maxU') for c in cs[:2])
        u = max(get(c, 'unif_dev') for c in cs[2:])
        verdict('R0', z == 0.0 and u <= 1e-13, f'zero: max|U|={z:.1e} (exactly 0) uniform: max dev/|U|={u:.2e} '
                f'(<=1e-13)')
    # R1, R2
    cs = ['tg_Z16_F', 'tg_Z16_R', 'tg_W16_F', 'tg_W16_R']
    if miss(cs):
        verdict('R1', False, f'missing {miss(cs)}')
        verdict("R2'", False, f'missing {miss(cs)}')
    else:
        c = max(get(x, 'closure') for x in cs)
        verdict('R1', c <= 1e-13, f'max|closure|/K={c:.2e} (<=1e-13) ' + ' '.join(
            f"{x}:{get(x, 'closure'):.1e}" for x in cs))
        mo = max(get(x, 'mom') for x in cs)
        mb = max(get(x, 'mom_booked') for x in cs)
        dv = max(get(x, 'divPhi1') for x in cs)
        ctl = [get(x + '_rtol14', 'mom') for x in cs]
        ctl = 'missing' if None in ctl else f'{max(ctl):.2e}'
        # R2' (A15, fixed 2026-11-17): the drift with the predictor residual's momentum booked
        verdict("R2'", mb <= 1e-13 and dv <= 1e-13, f'momentum drift with the residual booked '
                f'|P^n-P^0+sum res|/sum M|U|={mb:.2e} (<=1e-13) '
                + ' '.join(f"{x}:{get(x, 'mom_booked'):.1e}" for x in cs)
                + f' max tau|D Phi_1|/V={dv:.2e} (<=1e-13); reported: drift without booking {mo:.2e}, '
                f'without booking at predictor rtol 1e-14 {ctl}')
    # R3
    c = 'r3_tg_W16_F'
    if miss([c]):
        verdict('R3', False, f'missing {c}')
    else:
        ex, mr, om = get(c, 'excess'), get(c, 'mu_ratio'), get(c, 'omega_mu')
        mus = ' '.join(f"{f}{n}:{get(f'tg_{f}{n}_F', 'mu_exact') or float('nan'):.4f}" for f in 'ZW'
                       for n in (8, 16, 32))
        verdict('R3', ex <= 1e-13 and mr >= 0.9 and om <= 4.0 / 3.0,
                f'max(dK - unsigned)/K={ex:.2e} (<=1e-13; unsigned: {"+".join(UNSIGNED)}) '
                f'min mu_hat/mu_exact={mr:.4f} over {int(get(c, "mu_samples"))} samples (>=0.9) '
                f'omega*mu_exact max={om:.4f} (<=4/3) mu_exact {mus}')
    # R4
    cs = ['tg_Z16_F', 'tg_Z16_R', 'tg_Z32_F', 'tg_Z32_R']
    if miss(cs):
        verdict('R4', False, f'missing {miss(cs)}')
    else:
        d16 = abs(get('tg_Z16_F', 'err') / get('tg_Z16_R', 'err') - 1)
        d32 = abs(get('tg_Z32_F', 'err') / get('tg_Z32_R', 'err') - 1)
        # A15: the bound applies at 32^3 only; 16^3 is reported
        verdict('R4', d32 <= 0.02, f'|err_F/err_R - 1| 32^3={d32:.4f} (<=0.02); reported 16^3={d16:.4f}')
    # R5
    parts, ok5 = [], True
    for fam in 'ZW':
        for fld in ('tg', 'abc'):
            e = [get(f'{fld}_{fam}{n}_F', 'err') for n in (8, 16, 32)]
            if None in e:
                parts.append(f'{fld}{fam}:missing')
                ok5 = ok5 and fam != 'Z'
                continue
            o1, o2 = np.log2(e[0] / e[1]), np.log2(e[1] / e[2])
            parts.append(f'{fld}{fam}:{e[0]:.3e},{e[1]:.3e},{e[2]:.3e} order {o1:.2f},{o2:.2f}')
            if fam == 'Z':
                ok5 = ok5 and bool(o2 >= 1.5)
    verdict('R5', ok5, '(Z: order 16->32 >=1.5; W reported) ' + ' '.join(parts))
    # R6
    c1, c2, c3, c4 = 'r6i_W16', 'r6ii_rest_W16', 'r6ii_tg_W16', 'r6iii_W16'
    if miss([c1, c2, c3]):
        verdict('R6', False, f'missing {miss([c1, c2, c3])}')
    else:
        a = get(c1, 'maxU_rel_bound')
        b = get(c2, 'gt_dev')
        d = get(c3, 'mean_dev')
        tw = get(c3, 'twin_dev')
        ctl = get(c4, 'maxU_rel_drive')
        b14, b8 = get(c2 + '_rtol14', 'gt_dev'), get('r6ii_rest_W8', 'gt_dev')
        b14 = f'{b14:.2e}' if b14 is not None else 'missing'
        b8 = f'{b8:.2e}' if b8 is not None else 'missing'
        # (ii')' (A15): the fall from rest within 20 x the predictor's relative tolerance
        bb = 20.0 * PREDICTOR_RTOL
        verdict('R6', a <= 1e-12 and b <= bb and d <= 1e-12,
                f"(i) max|U|/(|grad psi|h)^0.5={a:.2e} (<=1e-12) (ii')' rest |U-gt|/|g|t={b:.2e} (<={bb:.0e}) "
                f'[controls: predictor rtol 1e-14 {b14}, W 8^3 direct {b8}] '
                f'TG mean |Ubar-gt|/|g|t={d:.2e} (<=1e-12) TG twin max|U-U0-gt|/|g|t={tw:.2e} (reported) '
                f'(iii) control |U|/(tau 100 |grad psi|)={ctl if ctl is not None else float("nan"):.3e} (reported)')
    # R7
    w, pi, ri = get('tg_Z32_F', 'wall_per_step'), get('tg_Z32_F', 'pred_its_mean'), get('tg_Z32_R', 'projR_its_mean')
    if None in (w, pi, ri):
        verdict('R7', None, 'missing 32^3 runs')
    else:
        verdict('R7', None, f'32^3 one thread: {w:.2f} s/step (mode F), predictor GMRES its mean={pi:.1f} '
                f"max={get('tg_Z32_F', 'pred_its_max'):.0f}, mode R CG its mean={ri:.1f} "
                f"max={get('tg_Z32_R', 'projR_its_max'):.0f}")
    for ln in lines:
        print(ln)
    print(f"G1.2r {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('run')
    a.add_argument('export_dir')
    a.add_argument('case')
    a.add_argument('--log')
    j = sub.add_parser('judge')
    j.add_argument('log_dir')
    sub.add_parser('cases')
    t = sub.add_parser('trace')
    t.add_argument('export_dir')
    t.add_argument('case')
    t.add_argument('out_dir')
    t.add_argument('--steps', type=int, default=10)
    t.add_argument('--pred-rtol', type=float, default=PREDICTOR_RTOL)
    args = ap.parse_args()
    if args.cmd == 'trace':
        return trace_case(args.export_dir, args.case, args.out_dir, args.steps, args.pred_rtol)
    if args.cmd == 'cases':
        print('\n'.join(case_table()))
        return 0
    if args.cmd == 'judge':
        return judge(args.log_dir)
    fh = open(args.log, 'w') if args.log else None

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + '\n')
            fh.flush()

    run_case(args.export_dir, args.case, log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
