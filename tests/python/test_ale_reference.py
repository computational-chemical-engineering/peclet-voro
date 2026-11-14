"""Gate G0.3 (work order WO-S0.3, docs/ale_design.md §7; criteria fixed 2026-11-15 in the brief ALE_S0_3) on the
meshes M1, M2, M3 exported by `test_ale_frame --export DIR`:

  R0  matrix form against matrix-free form of every operator on random fields          <= 1e-14 relative
  R1  <I_F U, phi> = <U, I_F^T phi> (scatter and two-pass forms); two-pass I_F^T against the transposed matrix;
      <D Phi, p> = -<Phi, Delta_n p>                                                    <= 1e-14 relative
  R2  U = a + B x through differences B conn: G_i = B (relative to |B|), (I_F U)_f = A_f n_f.U(c_f) (relative
      to A_f |B| h); trace-free B: C'U = 0 (relative to |B| V_i)                         <= 1e-13
  R3  U = a + Omega x x: sym G_f = 0 (relative to |Omega|) and zeta U = 0 (relative to |zeta||dU|, the size of
      the terms that cancel) <= 1e-13; zeta symmetric <= 1e-14; lambda_min >= -1e-12 lambda_max
  R4  L_rho symmetric <= 1e-14, lambda_min >= -1e-12 lambda_max, second smallest eigenvalue > 0, at rho uniform
      and at density ratio 1000 (rho = 1 within L/4 of the box center, 1000 elsewhere)
  R5' p linear (differences b.conn): |sum_i (C'^T p)_i + (sum_i V_i) b| / ((sum_i V_i)|b|)   <= 1e-13
      (amendment A6 of the design note; the per-cell form of the former R5 is reported as INFO R5cell)
  R6' on cells whose neighbor set is unchanged at both +tau and -tau: max_i |tau (D Phi^m)_i - (V_i(tau) -
      V_i(-tau))/2|, the ratio per halving of tau in [6, 10] (three ratios), variant 'full', every mesh

Reported without threshold: conditioning of L_rho, sum_f A_f d_f/(6 V_i), cells used in R6', negative heights; the
per-cell R5 quantity (R5cell); and for R6' the forward-difference errors (V(tau) - V(0)) and the other rate
variants.

Usage: python test_ale_reference.py DIR [M1 M2 M3]. Prints one line per check and mesh, then the summary line
`G0.3' PASS|FAIL R0=... R6'=...`; exit status 1 on FAIL.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ale_reference as ar  # noqa: E402

TOL_R0 = 1e-14
TOL_R1 = 1e-14
TOL_R2 = 1e-13
TOL_R3 = 1e-13
TOL_SYM = 1e-14
TOL_PSD = 1e-12
TOL_R5 = 1e-13
R6_LO, R6_HI = 6.0, 10.0


def rel(a, b):
    """max |a - b| / max |b|."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def sparse_sym(M):
    return float(abs(M - M.T).max() / abs(M).max())


def check_mesh(directory, name, rng):
    m = ar.Mesh(directory, name)
    mats = ar.Matrices(m)
    N, F = m.N, m.F
    res = {}

    # ---------------------------------------------------------------- R0: matrix vs matrix-free
    U = rng.standard_normal((N, 3))
    p = rng.standard_normal(N)
    phi = rng.standard_normal(F)
    xd = rng.standard_normal((N, 3))
    wd = m.h * rng.standard_normal(N)
    rho = np.exp(rng.uniform(-3, 3, N))
    mu = rng.uniform(0.5, 2.0, F)
    mats.set_rho(rho)
    mats.set_mu(mu)
    r0 = {
        'D': rel(ar.div_mf(m, phi), mats.D @ phi),
        'Dn': rel(ar.dn_mf(m, p), mats.Dn @ p),
        'G': rel(ar.grad_mf(m, U).ravel(), mats.G @ U.ravel()),
        'IF': rel(ar.IF_mf(m, U), mats.IF @ U.ravel()),
        'IFT': rel(ar.IFT_mf(m, phi).ravel(), mats.IF.T @ phi),
        'Cp': rel(ar.Cp_mf(m, U), mats.Cp @ U.ravel()),
        'CpT': rel(ar.CpT_mf(m, p).ravel(), mats.Cp.T @ p),
        'Lrho': rel(ar.Lrho_mf(m, p, rho), mats.Lrho @ p),
        'swept': rel(ar.swept_mf(m, xd, wd), mats.S @ np.r_[xd.ravel(), wd]),
        'zeta': rel(ar.zeta_mf(m, U, mu).ravel(), mats.zeta @ U.ravel()),
        'Cp_local': rel(ar.Cp_local_mf(m, U, ar.face_differences(m, U)), mats.Cp @ U.ravel()),
    }
    res['R0'] = max(r0.values())
    print(f'R0 {name} {"PASS" if res["R0"] <= TOL_R0 else "FAIL"} max={res["R0"]:.2e} '
          + ' '.join(f'{k}={v:.1e}' for k, v in r0.items()))
    mats.set_rho(None)
    mats.set_mu(None)

    # ---------------------------------------------------------------- R1: adjointness
    IFU = ar.IF_mf(m, U)
    lhs = IFU @ phi
    sc = np.linalg.norm(IFU) * np.linalg.norm(phi)
    r1 = {
        'IF_scatter': abs(lhs - np.sum(U * ar.IFT_mf(m, phi))) / sc,
        'IF_twopass': abs(lhs - np.sum(U * ar.IFT_twopass(m, phi))) / sc,
        'twopass_vs_matT': rel(ar.IFT_twopass(m, phi).ravel(), mats.IF.T @ phi),
    }
    DPhi = ar.div_mf(m, phi)
    r1['D_Dn'] = abs(DPhi @ p + phi @ ar.dn_mf(m, p)) / (np.linalg.norm(DPhi) * np.linalg.norm(p))
    res['R1'] = max(r1.values())
    print(f'R1 {name} {"PASS" if res["R1"] <= TOL_R1 else "FAIL"} max={res["R1"]:.2e} '
          + ' '.join(f'{k}={v:.1e}' for k, v in r1.items()))

    # ---------------------------------------------------------------- R2: linear exactness
    a = rng.standard_normal(3)
    B = rng.standard_normal((3, 3))
    nB = np.linalg.norm(B)
    UL = a + m.pos @ B.T                       # cell values; only U_A is read with dU given
    dU = m.conn @ B.T                          # dU_f = B conn_f
    Uc = a + (m.pos[m.A] + m.cA) @ B.T         # U(c_f) in the frame of A
    exact = m.area * np.einsum('fa,fa->f', m.n, Uc)
    fscale = m.area * nB * m.h
    r2 = {
        'G_mf': float(np.abs(ar.grad_mf(m, dU=dU) - B).max() / nB),
        'G_mat': float(np.abs(mats.grad_apply(dU) - B).max() / nB),
        'IF_mf': float((np.abs(ar.IF_mf(m, UL, dU) - exact) / fscale).max()),
        'IF_mat': float((np.abs(mats.IF_apply(UL, dU) - exact) / fscale).max()),
    }
    B0 = B - np.trace(B) / 3.0 * np.eye(3)
    nB0 = np.linalg.norm(B0)
    U0 = a + m.pos @ B0.T
    dU0 = m.conn @ B0.T
    # every face flux in the frame of the cell it enters (a linear field is not periodic; no wrapped positions)
    r2['Cp_mf'] = float((np.abs(ar.Cp_local_mf(m, U0, dU0)) / (nB0 * m.V)).max())
    r2['Cp_mat'] = float((np.abs(mats.Cp_local_apply(U0, dU0)) / (nB0 * m.V)).max())
    res['R2'] = max(r2.values())
    # geometry identity behind R2: sum_f (A_f/d_f) conn (x) (c_f - x_i) = V_i I
    Gid = ar.grad_mf(m, dU=m.conn)
    geo = float(np.abs(Gid - np.eye(3)).max())
    print(f'R2 {name} {"PASS" if res["R2"] <= TOL_R2 else "FAIL"} max={res["R2"]:.2e} '
          + ' '.join(f'{k}={v:.1e}' for k, v in r2.items()) + f' geometry|sum(A/d)conn(c-x)/V-I|={geo:.1e}')
    if res['R2'] > TOL_R2:   # localize: plain part against the gradient correction
        plain = m.area * ((m.hB * np.einsum('fa,fa->f', m.n, UL[m.A])
                           + m.hA * np.einsum('fa,fa->f', m.n, UL[m.A] + dU)) / m.d)
        corr = m.area * np.einsum('fa,ab,fb->f', m.n, B, m.q)
        print(f'R2-LOCALIZE {name} plain+exact-correction err={float((np.abs(plain + corr - exact) / fscale).max()):.2e} '
              f'gradient-correction err={float((np.abs(ar.IF_mf(m, UL, dU) - plain - corr) / fscale).max()):.2e}')

    # ---------------------------------------------------------------- R3: rigid motion, zeta
    Om = rng.standard_normal(3)
    dR = np.cross(Om, m.conn)                  # dU_f = Omega x conn_f
    SGm = ar.face_gradient_mf(m, dU=dR)
    SGm = 0.5 * (SGm + np.transpose(SGm, (0, 2, 1)))
    zscale = float((abs(mats.Delta3).T @ (abs(mats.zeta_d) @ np.abs(dR).ravel())).max())
    r3 = {
        'symGf_mf': float(np.abs(SGm).max() / np.linalg.norm(Om)),
        'symGf_mat': float(np.abs(mats.sym_face_gradient_apply(dR)).max() / np.linalg.norm(Om)),
        'zetaU_mf': float(np.abs(ar.zeta_mf(m, dU=dR)).max() / zscale),
        'zetaU_mat': float(np.abs(mats.zeta_apply(dR)).max() / zscale),
    }
    zsym = sparse_sym(mats.zeta)
    ev = np.linalg.eigvalsh(mats.zeta.toarray())
    zpsd = ev[0] / ev[-1]
    ok3 = max(r3.values()) <= TOL_R3 and zsym <= TOL_SYM and zpsd >= -TOL_PSD
    res['R3'] = (max(r3.values()), zsym, zpsd)
    res['R3ok'] = ok3
    print(f'R3 {name} {"PASS" if ok3 else "FAIL"} rigid_max={max(r3.values()):.2e} '
          + ' '.join(f'{k}={v:.1e}' for k, v in r3.items())
          + f' zeta_sym={zsym:.1e} lambda_min/lambda_max={zpsd:.2e} lambda_max={ev[-1]:.3e}')

    # ---------------------------------------------------------------- R4: L_rho
    ctr = 0.5 * m.L
    inside = np.linalg.norm(m.pos - ctr, axis=1) < 0.25 * m.L[0]
    res['R4'] = {}
    ok4 = True
    for tag, rr in (('ratio1', np.ones(N)), ('ratio1000', np.where(inside, 1.0, 1000.0))):
        mats.set_rho(rr)
        Ls = sparse_sym(mats.Lrho)
        evL = np.linalg.eigvalsh(mats.Lrho.toarray())
        const = float(np.abs(mats.Lrho @ np.ones(N)).max() / abs(mats.Lrho).max())
        ok = Ls <= TOL_SYM and evL[0] >= -TOL_PSD * evL[-1] and evL[1] > 0
        ok4 &= ok
        res['R4'][tag] = (Ls, evL[0], evL[1], evL[-1])
        print(f'R4 {name} {tag} {"PASS" if ok else "FAIL"} sym={Ls:.1e} lambda1={evL[0]:.2e} lambda2={evL[1]:.6e} '
              f'lambda_max={evL[-1]:.6e} cond(lambda2/lambda_max)={evL[1] / evL[-1]:.3e} |L 1|/|L|={const:.1e} '
              f'inside={int(inside.sum())}')
    mats.set_rho(None)
    res['R4ok'] = ok4

    # ---------------------------------------------------------------- R5: hydrostatic identity
    b = rng.standard_normal(3)
    dp = m.conn @ b
    target = -m.V[:, None] * b[None, :]
    sc5 = (m.V * np.linalg.norm(b))[:, None]
    r5 = {
        'scatter': float((np.abs(ar.CpT_mf(m, None, dp) - target) / sc5).max()),
        'twopass': float((np.abs(ar.CpT_mf(m, None, dp, twopass=True) - target) / sc5).max()),
        'matrix': float((np.abs((-(mats.IF.T @ dp)).reshape(-1, 3) - target) / sc5).max()),
    }
    res['R5cell'] = max(r5.values())
    print(f'INFO R5cell {name} max={res["R5cell"]:.2e} ' + ' '.join(f'{k}={v:.1e}' for k, v in r5.items()))
    Vsum = float(m.V.sum())
    tot5 = {
        'scatter': ar.CpT_mf(m, None, dp).sum(0),
        'twopass': ar.CpT_mf(m, None, dp, twopass=True).sum(0),
        'matrix': (-(mats.IF.T @ dp)).reshape(-1, 3).sum(0),
    }
    r5s = {k: float(np.linalg.norm(v + Vsum * b) / (Vsum * np.linalg.norm(b))) for k, v in tot5.items()}
    res['R5'] = max(r5s.values())
    print(f"R5' {name} {'PASS' if res['R5'] <= TOL_R5 else 'FAIL'} max={res['R5']:.2e} "
          + ' '.join(f'{k}={v:.1e}' for k, v in r5s.items()))
    if res['R5'] > TOL_R5:   # localize (per cell): plain part of -I_F^T dp against the gradient correction
        plain = np.zeros((N, 3))
        np.add.at(plain, m.A, -(m.area * m.hB / m.d * dp)[:, None] * m.n)
        np.add.at(plain, m.B, -(m.area * m.hA / m.d * dp)[:, None] * m.n)
        total = ar.CpT_mf(m, None, dp)
        e_tot = np.linalg.norm(total - target, axis=1) / sc5[:, 0]
        e_pl = np.linalg.norm(plain - target, axis=1) / sc5[:, 0]
        g_sz = np.linalg.norm(total - plain, axis=1) / sc5[:, 0]
        mom = np.abs(total.sum(0) - target.sum(0)).max() / np.abs(target.sum(0)).max()
        print(f'R5-LOCALIZE {name} |F-target|/(V|b|) max={e_tot.max():.2e} median={np.median(e_tot):.2e} '
              f'cells<=1e-13: {int((e_tot <= TOL_R5).sum())}/{N}; plain part alone max={e_pl.max():.2e} '
              f'median={np.median(e_pl):.2e}; gradient correction size max={g_sz.max():.2e}; '
              f'sum over cells rel={mom:.1e}')

    # ---------------------------------------------------------------- R6: swept flux against Delta V
    variants = ['full', 'smooth'] + (['nowdot'] if os.path.exists(os.path.join(directory, f'{name}.vol_nowdot'))
                                     else [])
    for v in variants:
        xd6, wd6, tau, vol, changed = m.rates(v)
        DPhi6 = ar.div_mf(m, ar.swept_mf(m, xd6, wd6))
        keep = changed[:4].sum(0) == 0
        keepc = changed.sum(0) == 0
        ef = np.array([np.abs(tau[k] * DPhi6 - (vol[k] - m.V))[keep].max() for k in range(4)])
        ec = np.array([np.abs(tau[k] * DPhi6 - 0.5 * (vol[k] - vol[k + 4]))[keepc].max() for k in range(4)])
        rf = ef[:-1] / ef[1:]
        rc = ec[:-1] / ec[1:]
        if v == 'full':
            ok6 = bool(np.all((rc >= R6_LO) & (rc <= R6_HI)))
            res['R6'] = (rc, ok6, int(keepc.sum()))
            verdict = 'PASS' if ok6 else 'FAIL'
        else:
            verdict = 'info'
        print(f"R6' {name} {v} {verdict} cells={int(keepc.sum())} "
              f'err/h3={",".join(f"{e / m.h ** 3:.3e}" for e in ec)} ratios={",".join(f"{r:.2f}" for r in rc)}')
        print(f'INFO R6fwd {name} {v} cells={int(keep.sum())} err/h3={",".join(f"{e / m.h ** 3:.3e}" for e in ef)} '
              f'ratios={",".join(f"{r:.2f}" for r in rf)}')

    # ---------------------------------------------------------------- reported without threshold
    ratio = np.zeros(N)
    np.add.at(ratio, m.A, m.area * m.d)
    np.add.at(ratio, m.B, m.area * m.d)
    ratio /= 6.0 * m.V
    hneg = int(((m.hA < 0) | (m.hB < 0)).sum())
    print(f'INFO {name} N={N} F={F} sumAd/(6V) min={ratio.min():.6f} max={ratio.max():.6f} '
          f'negative_heights={hneg} (export counter {m.numNegativeHeight})')
    return res


def main():
    directory = sys.argv[1]
    names = sys.argv[2:] or ['M1', 'M2', 'M3']
    rng = np.random.default_rng(20261115)
    allres = {nm: check_mesh(directory, nm, rng) for nm in names}
    r0 = max(r['R0'] for r in allres.values())
    r1 = max(r['R1'] for r in allres.values())
    r2 = max(r['R2'] for r in allres.values())
    r3 = max(r['R3'][0] for r in allres.values())
    r5 = max(r['R5'] for r in allres.values())
    ok = {
        'R0': r0 <= TOL_R0, 'R1': r1 <= TOL_R1, 'R2': r2 <= TOL_R2, 'R3': all(r['R3ok'] for r in allres.values()),
        'R4': all(r['R4ok'] for r in allres.values()), 'R5': r5 <= TOL_R5,
        'R6': all(r['R6'][1] for r in allres.values()),
    }
    l2 = min(min(x[2] for x in r['R4'].values()) for r in allres.values())
    r6 = ';'.join(f'{nm}:' + ','.join(f'{x:.2f}' for x in r['R6'][0]) for nm, r in allres.items())
    word = lambda k: 'PASS' if ok[k] else 'FAIL'  # noqa: E731
    print(f"G0.3' {'PASS' if all(ok.values()) else 'FAIL'} R0={word('R0')}({r0:.1e}) R1={word('R1')}({r1:.1e}) "
          f"R2={word('R2')}({r2:.1e}) R3={word('R3')}({r3:.1e}) R4={word('R4')}(lambda2min={l2:.2e}) "
          f"R5'={word('R5')}({r5:.1e}) R6'={word('R6')}({r6})")
    return 0 if all(ok.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
