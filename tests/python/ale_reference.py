"""Host-only reference of the bulk operators of the moving-grid solver (docs/ale_design.md §A.3, work order
WO-S0.3) on a 3D face mesh exported by `test_ale_frame --export DIR`.

Every operator is written twice: as an explicit scipy.sparse matrix assembled entry by entry, and as a matrix-free
face loop (np.add.at, as the 2D oracle peclet-papers/work/M7/pilots/fable/a4_cell_ale.py does). The adjoint of the
flux map is written a third time in the two-pass cell-gather form of §2.3 (moments T_i). The stage-1 device kernels
are checked against this file; it is meant to be plainly the formula, not fast.

Convention (§A.3, binding):
  - face f = (A, B), A the lower gid, n_f = conn_f/d_f the unit vector A -> B; conn_f = x_B - x_A (minimum image);
  - h_Af = (d_f^2 + w_A - w_B)/(2 d_f), h_Bf = d_f - h_Af (signed); c_f the face centroid, stored as c_f - x_A;
    seen from B, c_f - x_B = (c_f - x_A) - conn_f (positions are never differenced across the box);
  - D is OUTFLOW: (D phi)_i = sum_f s_if phi_f, s_if = +1 if i = A, -1 if i = B;  Delta_n = -D^T: p_B - p_A;
  - cell gradient G_i = (1/V_i) sum_{f in i} (A_f/d_f)(U_o - U_i) (x) (c_f - x_i), component (c,k) = dU_c/dx_k
    stored at 9i + 3c + k;
  - flux map (I_F U)_f = A_f [n_f.(h_Bf U_A + h_Af U_B)/d_f + n_f.Gbar_f q_f], Gbar_f = (G_A + G_B)/2,
    q_f = c_f - x_A - h_Af n_f;
  - C' = D I_F; the pressure force on cells is F_p = -I_F^T Delta_n p = +C'^T p;
  - L_rho = D diag(A_f/(rho_f d_f)) D^T, 1/rho_f = (h_Bf/rho_A + h_Af/rho_B)/d_f;
  - swept flux Phi^m_f = (A_f/d_f)[(c_f - x_A).xd_A - (c_f - x_B).xd_B + (wd_A - wd_B)/2];
  - friction: 1/2 U^T zeta U = sum_f mu_f V_f |sym G_f|^2, G_f = Gbar_f + [(U_B - U_A)/d_f - Gbar_f n_f] (x) n_f,
    V_f = A_f d_f/3.

Velocity fields are (N, 3) arrays (flattened 3i + c for the matrices). Every operator that differences cell values
across a face accepts the face differences dU_f = U_B - U_A explicitly (default: from the cell values), so that a
linear field a + B x is supplied as dU_f = B conn_f without wrapped positions; the gradient depends on dU only.
"""
import os

import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------------------------------------------------------ mesh
class Mesh:
    """The arrays of one exported FaceMesh (periodic box: every face is interior)."""

    def __init__(self, directory, name):
        def rd(arr, dtype=np.float64):
            return np.fromfile(os.path.join(directory, f'{name}.{arr}'), dtype=dtype)

        counts = np.fromfile(os.path.join(directory, f'{name}.counts'), dtype=np.int64, count=6)
        self.N, self.F, nInterior = int(counts[0]), int(counts[1]), int(counts[2])
        self.numNegativeHeight = int(counts[5])
        if nInterior != self.F:
            raise ValueError(f'{name}: {self.F - nInterior} boundary faces; the reference handles a periodic box')
        self.A = rd('faceCellA', np.int32).astype(np.int64)
        self.B = rd('faceCellB', np.int32).astype(np.int64)
        self.area = rd('faceArea')
        self.n = rd('faceNormal').reshape(-1, 3)
        self.d = rd('faceDist')
        self.hA = rd('faceHa')
        self.hB = rd('faceHb')
        self.cA = rd('faceCentroid').reshape(-1, 3)      # c_f - x_A
        self.conn = rd('faceConn').reshape(-1, 3)        # x_B - x_A (minimum image)
        self.cB = self.cA - self.conn                    # c_f - x_B
        self.cellFaceOffset = rd('cellFaceOffset', np.int32).astype(np.int64)
        self.cellFace = rd('cellFace', np.int32).astype(np.int64)
        self.cellFaceSign = rd('cellFaceSign')
        self.V = rd('cellVolume')
        self.pos = rd('pos').reshape(-1, 3)
        self.w = rd('weight')
        self.L = rd('L')
        self.directory, self.name = directory, name
        self.q = self.cA - self.hA[:, None] * self.n     # q_f = c_f - x_A - h_Af n_f
        self.h = (self.V.sum() / self.N) ** (1.0 / 3.0)

    def rates(self, variant):
        """Swept-flux data of a rate variant: xdot (N,3), wdot (N), tau (8), vol (8,N), changed (8,N)."""
        def rd(arr, dtype=np.float64):
            return np.fromfile(os.path.join(self.directory, f'{self.name}.{arr}_{variant}'), dtype=dtype)

        return (rd('xdot').reshape(-1, 3), rd('wdot'), rd('tau'), rd('vol').reshape(8, self.N),
                rd('changed', np.int32).reshape(8, self.N))


def face_differences(m, U):
    """dU_f = U_B - U_A (cell values; any trailing shape)."""
    return U[m.B] - U[m.A]


# ------------------------------------------------------------------------------------ matrix-free face loops
def div_mf(m, phi):
    """(D phi)_i = sum_f s_if phi_f (outflow)."""
    out = np.zeros(m.N)
    np.add.at(out, m.A, phi)
    np.add.at(out, m.B, -phi)
    return out


def dn_mf(m, p):
    """(Delta_n p)_f = p_B - p_A = -(D^T p)_f."""
    return p[m.B] - p[m.A]


def grad_mf(m, U=None, dU=None):
    """Cell gradient (N, 3, 3), [i, c, k] = dU_c/dx_k, from the face differences dU (default U_B - U_A)."""
    if dU is None:
        dU = face_differences(m, U)
    w = m.area / m.d
    G = np.zeros((m.N, 3, 3))
    np.add.at(G, m.A, w[:, None, None] * dU[:, :, None] * m.cA[:, None, :])
    np.add.at(G, m.B, w[:, None, None] * (-dU)[:, :, None] * m.cB[:, None, :])
    return G / m.V[:, None, None]


def IF_mf(m, U, dU=None):
    """Flux map (I_F U)_f. With dU given, U_B is taken as U_A + dU (U is read at A only)."""
    if dU is None:
        dU = face_differences(m, U)
    return IF_face_mf(m, U[m.A], dU)


def IF_face_mf(m, UA, dU):
    """Flux map from per-face values U_A (F, 3) and differences dU_f = U_B - U_A (F, 3). A linear field on the
    periodic box is supplied per cell frame: from the side of B, U_A = U_B - dU_f (see Cp_local_mf)."""
    G = grad_mf(m, dU=dU)
    Gb = 0.5 * (G[m.A] + G[m.B])
    UB = UA + dU
    n = m.n
    return m.area * ((m.hB * np.einsum('fa,fa->f', n, UA) + m.hA * np.einsum('fa,fa->f', n, UB)) / m.d
                     + np.einsum('fa,fab,fb->f', n, Gb, m.q))


def grad_adjoint_mf(m, T):
    """G^T applied to cell moments T (N, 3, 3): returns (N, 3). Face scatter of
    (A_f/d_f)[-(T_i/V_i)(c_f - x_i)] to i and +(T_i/V_i)(c_f - x_i) to the other cell o, for i = A and i = B."""
    w = m.area / m.d
    aA = (w / m.V[m.A])[:, None] * np.einsum('fab,fb->fa', T[m.A], m.cA)
    aB = (w / m.V[m.B])[:, None] * np.einsum('fab,fb->fa', T[m.B], m.cB)
    out = np.zeros((m.N, 3))
    np.add.at(out, m.B, aA)
    np.add.at(out, m.A, -aA)
    np.add.at(out, m.A, aB)
    np.add.at(out, m.B, -aB)
    return out


def IFT_mf(m, phi):
    """Adjoint of IF_mf, face-scatter form (as the 2D oracle IFT_mf): returns (N, 3)."""
    out = np.zeros((m.N, 3))
    np.add.at(out, m.A, (m.area * m.hB / m.d * phi)[:, None] * m.n)
    np.add.at(out, m.B, (m.area * m.hA / m.d * phi)[:, None] * m.n)
    H = (0.5 * m.area * phi)[:, None, None] * m.n[:, :, None] * m.q[:, None, :]
    T = np.zeros((m.N, 3, 3))
    np.add.at(T, m.A, H)
    np.add.at(T, m.B, H)
    return out + grad_adjoint_mf(m, T)


def IFT_twopass(m, phi):
    """Adjoint of I_F in the two-pass cell-gather form of §2.3 (one loop per cell over its face list, ascending
    face order). Pass 1: plain part and M_i = T_i/V_i, T_i = 1/2 sum_{f in i} phi_f A_f n_f (x) q_f.
    Pass 2: sum_{f in m} (A_f/d_f)[-M_m (c_f - x_m) + M_o (c_f - x_o)]."""
    N = m.N
    out = np.zeros((N, 3))
    M = np.zeros((N, 3, 3))
    for i in range(N):
        T = np.zeros((3, 3))
        for k in range(m.cellFaceOffset[i], m.cellFaceOffset[i + 1]):
            f = m.cellFace[k]
            h_other = m.hB[f] if m.cellFaceSign[k] > 0 else m.hA[f]
            out[i] += (m.area[f] * h_other / m.d[f] * phi[f]) * m.n[f]
            T += (0.5 * phi[f] * m.area[f]) * np.outer(m.n[f], m.q[f])
        M[i] = T / m.V[i]
    for i in range(N):
        acc = np.zeros(3)
        for k in range(m.cellFaceOffset[i], m.cellFaceOffset[i + 1]):
            f = m.cellFace[k]
            if m.cellFaceSign[k] > 0:
                o, ri, ro = m.B[f], m.cA[f], m.cB[f]
            else:
                o, ri, ro = m.A[f], m.cB[f], m.cA[f]
            acc += (m.area[f] / m.d[f]) * (-M[i] @ ri + M[o] @ ro)
        out[i] += acc
    return out


def Cp_mf(m, U, dU=None):
    """C' U = D I_F U."""
    return div_mf(m, IF_mf(m, U, dU))


def Cp_local_mf(m, U, dU):
    """C'U for a field given by cell values U and face differences dU that need not satisfy dU_f = U_B - U_A
    across the box (a linear field a + B x on the periodic box): every face flux is evaluated in the frame of the
    cell it is summed into, U_A = U_i on the A side and U_A = U_i - dU_f on the B side. Equal to Cp_mf when
    dU_f = U_B - U_A."""
    out = np.zeros(m.N)
    np.add.at(out, m.A, IF_face_mf(m, U[m.A], dU))
    np.add.at(out, m.B, -IF_face_mf(m, U[m.B] - dU, dU))
    return out


def CpT_mf(m, p, dp=None, twopass=False):
    """Pressure force F_p = +C'^T p = -I_F^T Delta_n p; dp the face differences p_B - p_A (default from p)."""
    if dp is None:
        dp = dn_mf(m, p)
    return -(IFT_twopass(m, dp) if twopass else IFT_mf(m, dp))


def inv_rho_face(m, rho):
    """1/rho_f = (h_Bf/rho_A + h_Af/rho_B)/d_f."""
    return (m.hB / rho[m.A] + m.hA / rho[m.B]) / m.d


def Lrho_mf(m, p, rho):
    """L_rho p = D diag(a_f) D^T p, a_f = A_f/(rho_f d_f)."""
    a = m.area * inv_rho_face(m, rho) / m.d
    return div_mf(m, -a * dn_mf(m, p))


def swept_mf(m, xd, wd):
    """Swept flux Phi^m_f (positive when the face moves along n_f)."""
    return m.area / m.d * (np.einsum('fa,fa->f', m.cA, xd[m.A]) - np.einsum('fa,fa->f', m.cB, xd[m.B])
                           + 0.5 * (wd[m.A] - wd[m.B]))


def face_gradient_mf(m, U=None, dU=None):
    """G_f = Gbar_f + [dU_f/d_f - Gbar_f n_f] (x) n_f, (F, 3, 3)."""
    if dU is None:
        dU = face_differences(m, U)
    G = grad_mf(m, dU=dU)
    Gb = 0.5 * (G[m.A] + G[m.B])
    corr = dU / m.d[:, None] - np.einsum('fab,fb->fa', Gb, m.n)
    return Gb + corr[:, :, None] * m.n[:, None, :]


def zeta_mf(m, U=None, mu=None, dU=None):
    """zeta U: face stress sigma_f = 2 mu_f V_f sym G_f, then the adjoint of U -> G_f:
    <sigma, dG_f> = <sigma (I - n n^T), dGbar_f> + (sigma n).dU_f/d_f."""
    if dU is None:
        dU = face_differences(m, U)
    if mu is None:
        mu = np.ones(m.F)
    Gf = face_gradient_mf(m, dU=dU)
    S = 0.5 * (Gf + np.transpose(Gf, (0, 2, 1)))
    Vf = m.area * m.d / 3.0
    sig = (2.0 * mu * Vf)[:, None, None] * S
    sn = np.einsum('fab,fb->fa', sig, m.n)
    tau = sig - sn[:, :, None] * m.n[:, None, :]
    T = np.zeros((m.N, 3, 3))
    np.add.at(T, m.A, 0.5 * tau)
    np.add.at(T, m.B, 0.5 * tau)
    out = grad_adjoint_mf(m, T)
    g = sn / m.d[:, None]
    np.add.at(out, m.B, g)
    np.add.at(out, m.A, -g)
    return out


# --------------------------------------------------------------------------------------- explicit matrices
def _coo(rows, cols, vals, shape):
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape).tocsr()


class Matrices:
    """The operators as sparse matrices, assembled entry by entry. Vector unknowns are flattened 3i + c, cell
    gradients 9i + 3c + k, face gradients 9f + 3c + k, face differences 3f + c."""

    def __init__(self, m, rho=None, mu=None):
        N, F = m.N, m.F
        f = np.arange(F)
        self.m = m
        # D (N x F), Delta_n = -D^T
        self.D = _coo([m.A, m.B], [f, f], [np.ones(F), -np.ones(F)], (N, F))
        self.Dn = -self.D.T.tocsr()
        # face differences and A-side selection, per component (3F x 3N)
        r, cA, cB, ones = [], [], [], []
        for c in range(3):
            r.append(3 * f + c); cA.append(3 * m.A + c); cB.append(3 * m.B + c)
        self.Delta3 = _coo(r + r, cB + cA, [np.ones(F)] * 3 + [-np.ones(F)] * 3, (3 * F, 3 * N))
        self.SelA = _coo(r, cA, [np.ones(F)] * 3, (3 * F, 3 * N))
        self.SelB = _coo(r, cB, [np.ones(F)] * 3, (3 * F, 3 * N))
        # gradient from face differences, Gd (9N x 3F): G_i[c,k] += (w/V_i) s (dU_f)_c r_ik
        w = m.area / m.d
        rows, cols, vals = [], [], []
        for c in range(3):
            for k in range(3):
                rows += [9 * m.A + 3 * c + k, 9 * m.B + 3 * c + k]
                cols += [3 * f + c, 3 * f + c]
                vals += [w / m.V[m.A] * m.cA[:, k], -w / m.V[m.B] * m.cB[:, k]]
        self.Gd = _coo(rows, cols, vals, (9 * N, 3 * F))
        self.G = (self.Gd @ self.Delta3).tocsr()
        # face average of cell gradients (9F x 9N)
        rows, cols, vals = [], [], []
        for j in range(9):
            rows += [9 * f + j, 9 * f + j]; cols += [9 * m.A + j, 9 * m.B + j]; vals += [np.full(F, 0.5)] * 2
        self.Avg = _coo(rows, cols, vals, (9 * F, 9 * N))
        # flux map pieces: Pa, Pb (F x 3F) on U_A, U_B per face; Qn (F x 9F): A n_c q_k
        rows, cols, va, vb = [], [], [], []
        for c in range(3):
            rows.append(f); cols.append(3 * f + c)
            va.append(m.area * m.hB / m.d * m.n[:, c]); vb.append(m.area * m.hA / m.d * m.n[:, c])
        self.Pa = _coo(rows, cols, va, (F, 3 * F))
        self.Pb = _coo(rows, cols, vb, (F, 3 * F))
        rows, cols, vals = [], [], []
        for c in range(3):
            for k in range(3):
                rows.append(f); cols.append(9 * f + 3 * c + k); vals.append(m.area * m.n[:, c] * m.q[:, k])
        self.Qn = _coo(rows, cols, vals, (F, 9 * F))
        self.QGd = (self.Qn @ self.Avg @ self.Gd).tocsr()          # F x 3F: gradient correction from dU
        self.IF = (self.Pa @ self.SelA + self.Pb @ self.SelB + self.QGd @ self.Delta3).tocsr()
        self.Cp = (self.D @ self.IF).tocsr()
        # swept flux S (F x 4N): columns xd (3N) then wd (N)
        rows, cols, vals = [], [], []
        for c in range(3):
            rows += [f, f]; cols += [3 * m.A + c, 3 * m.B + c]; vals += [w * m.cA[:, c], -w * m.cB[:, c]]
        rows += [f, f]; cols += [3 * N + m.A, 3 * N + m.B]; vals += [0.5 * w, -0.5 * w]
        self.S = _coo(rows, cols, vals, (F, 4 * N))
        # friction: face gradient from dU, Gf_d (9F x 3F) = (I - n n^T) on the k index of Gbar, plus dU/d (x) n
        rows, cols, vals = [], [], []
        for c in range(3):
            for k in range(3):
                for l in range(3):                     # Gbar_f[c,l] (delta_lk - n_l n_k)
                    proj = (1.0 if l == k else 0.0) - m.n[:, l] * m.n[:, k]
                    rows.append(9 * f + 3 * c + k); cols.append(9 * f + 3 * c + l); vals.append(proj)
        self.Proj = _coo(rows, cols, vals, (9 * F, 9 * F))
        rows, cols, vals = [], [], []
        for c in range(3):
            for k in range(3):
                rows.append(9 * f + 3 * c + k); cols.append(3 * f + c); vals.append(m.n[:, k] / m.d)
        self.Gf_d = (self.Proj @ self.Avg @ self.Gd + _coo(rows, cols, vals, (9 * F, 3 * F))).tocsr()
        rows, cols, vals = [], [], []
        for c in range(3):
            for k in range(3):
                rows += [9 * f + 3 * c + k, 9 * f + 3 * c + k]
                cols += [9 * f + 3 * c + k, 9 * f + 3 * k + c]
                vals += [np.full(F, 0.5), np.full(F, 0.5)]
        self.Sym = _coo(rows, cols, vals, (9 * F, 9 * F))
        self.SGf_d = (self.Sym @ self.Gf_d).tocsr()
        self.set_mu(mu)
        self.set_rho(rho)

    def set_mu(self, mu=None):
        m = self.m
        mu = np.ones(m.F) if mu is None else mu
        W = sp.diags(np.repeat(2.0 * mu * m.area * m.d / 3.0, 9))
        self.zeta_d = (self.SGf_d.T @ W @ self.SGf_d).tocsr()             # 3F x 3F, on face differences
        self.zeta = (self.Delta3.T @ self.zeta_d @ self.Delta3).tocsr()   # 3N x 3N

    def set_rho(self, rho=None):
        m = self.m
        rho = np.ones(m.N) if rho is None else rho
        a = m.area * inv_rho_face(m, rho) / m.d
        self.Lrho = (self.D @ sp.diags(a) @ self.D.T).tocsr()

    # ---- application with explicit face differences (linear fields without wrapped positions)
    def IF_apply(self, U, dU):
        return self.IF_face_apply(self.SelA @ U.ravel(), dU)

    def IF_face_apply(self, uA, dU):
        """Flux map from per-face U_A (3F) and face differences dU (3F)."""
        uA = np.asarray(uA).ravel()
        return self.Pa @ uA + self.Pb @ (uA + dU.ravel()) + self.QGd @ dU.ravel()

    def Cp_local_apply(self, U, dU):
        """As Cp_local_mf: each face flux in the frame of the cell it is summed into."""
        phiA = self.IF_face_apply(self.SelA @ U.ravel(), dU)
        phiB = self.IF_face_apply(self.SelB @ U.ravel() - dU.ravel(), dU)
        return (abs(self.D).multiply(self.D > 0)) @ phiA + (self.D.multiply(self.D < 0)) @ phiB

    def grad_apply(self, dU):
        return (self.Gd @ dU.ravel()).reshape(-1, 3, 3)

    def sym_face_gradient_apply(self, dU):
        return (self.SGf_d @ dU.ravel()).reshape(-1, 3, 3)

    def zeta_apply(self, dU):
        return (self.Delta3.T @ (self.zeta_d @ dU.ravel())).reshape(-1, 3)
