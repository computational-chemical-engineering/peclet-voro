"""Gate G1.2 of the moving-grid work order S1.2b (docs/ale_design.md §4.1-§4.5, §11 A15): the device time step of
include/peclet/voro/ale/stepper.hpp against the host reference (tests/python/ale_reference.py,
tests/python/ale_reference_stepper.py). Criteria fixed 2026-11-17 in the brief ALE_S1_2B; "relative" is the max over
entries of |a - b| over the max over entries of |b| (the reference).

  python test_ale_stepper.py inputs DIR [--cells 16]
      C0, host side: for each exported mesh <M>_<n> in DIR (test_ale_frame --export DIR --cells n --mesh all) writes
      random inputs in_U, in_Fm, in_mu, in_rho, in_tau and the reference outputs ref_<name>; checks the reference's
      face loops against its matrices first (line C0ref).
  python test_ale_stepper.py compare DIR ODIR [--cells 16]
      C0: the device outputs ODIR/<M>_<n>.out_<name> (test_ale_stepper --apply) against ref_<name>: <= 1e-13.
  python test_ale_stepper.py trace REFDIR DEVDIR
      C1/C2: the device trace (test_ale_stepper --trace) against the reference trace (ale_reference_stepper.py
      trace): tg_Z16_F step 1 F, U*, U <= 1e-10, p <= 1e-8, omega = 1 in both; 10 steps U <= 1e-9, p <= 1e-7;
      tg_W16_R 10 steps U <= 1e-8.
  python test_ale_stepper.py c3 DIR_A DIR_B [--tol 1e-9]
      the U of the C2 run (tg_Z16_F, 10 steps) of two device runs: max relative difference (and whether bitwise).
  python test_ale_stepper.py trace-moving DIR CASE [CASE ...] [--tol 1e-12]
      gate T of G1.3a (brief ALE_S1_3A, docs/ale_design.md §11 A18): each step of a moving-mesh trace
      (test_ale_stepper --trace-moving DIR CASE) recomputed by ale_reference_stepper.moving_step from the device's
      inputs (U^n, U_ext, p^n, V^n, xdot, the frames Fh and F1 as exported, tau, mode; omega of mode F is the
      device's) and compared per step: F, U*, U^1, p^1 relative <= tol; gcl and closure of both printed.
  python test_ale_stepper.py judge LOG_DIR REF_LOG_DIR
      R0-R6, L and T from the RESULT / LAP lines of LOG_DIR/S1_2b.run.<case>.<host|gpu>.log and
      LOG_DIR/S1_2b.lap.<n>.<host|gpu>.log; the reference's numbers from REF_LOG_DIR/S1_2a.<case>.log.
Each command prints one line per criterion and exits 1 on a FAIL.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ale_reference as ar  # noqa: E402
import ale_reference_stepper as ars  # noqa: E402

MESHES = ('M1', 'M2', 'M3')
C0_OUTPUTS = ('T', 'Q', 'TQ', 'RmD', 'delta', 'explicit', 'pred')


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    s = np.abs(b).max()
    d = np.abs(a - b).max()
    return d / s if s > 0 else d


# ------------------------------------------------------------------------------------------------- C0
def reference_outputs(m, U, Fm, mu, rho, tau):
    rhoV = rho * m.V
    out = {
        'T': ar.transport_mf(m, U, Fm),
        'Q': ar.upwind_mf(m, U, Fm),
        'RmD': ar.reconstruction_jump_mf(m, U),
        'delta': ar.centroid_correction_mf(m, U),
        'explicit': ar.transport_explicit_mf(m, U, Fm),
        'pred': ar.predictor_mf(m, U, Fm, tau, lambda x: rhoV[:, None] * x, mu),
    }
    out['TQ'] = out['T'] + out['Q']
    # the reference's own consistency: face loops against the assembled matrices
    mx = ar.Matrices(m, rho=rho, mu=mu)
    tm = ar.TransportMatrices(mx)
    u = U.ravel()
    chk = {
        'T': rel(out['T'].ravel(), tm.T(Fm) @ u),
        'Q': rel(out['Q'].ravel(), tm.Q(Fm) @ u),
        'RmD': rel(out['RmD'].ravel(), tm.Jump @ u),
        'delta': rel(out['delta'].ravel(), tm.Delta_c @ u),
        'explicit': rel(out['explicit'].ravel(), tm.explicit(Fm) @ u),
        'pred': rel(out['pred'].ravel(), tm.predictor(Fm, tau, ar.sp.diags(np.repeat(rhoV, 3))) @ u),
    }
    return out, chk


def cmd_inputs(d, cells):
    worst, ok = 0.0, True
    for name in MESHES:
        mn = f'{name}_{cells}'
        m = ar.Mesh(d, mn)
        rng = np.random.default_rng(20261118 + MESHES.index(name))
        U = rng.uniform(-1, 1, (m.N, 3))
        Fm = rng.uniform(-1, 1, m.F) * m.area
        mu = rng.uniform(0.005, 0.02, m.F)
        rho = rng.uniform(1.0, 1000.0, m.N)
        tau = np.array([0.013])
        pre = os.path.join(d, mn + '.')
        for k, v in (('U', U), ('Fm', Fm), ('mu', mu), ('rho', rho), ('tau', tau)):
            np.asarray(v, dtype='<f8').tofile(pre + 'in_' + k)
        out, chk = reference_outputs(m, U, Fm, mu, rho, float(tau[0]))
        for k, v in out.items():
            np.asarray(v, dtype='<f8').tofile(pre + 'ref_' + k)
        w = max(chk.values())
        worst = max(worst, w)
        ok = ok and w <= 1e-13
        print(f'C0ref {mn} ' + ' '.join(f'{k}={v:.1e}' for k, v in chk.items()))
    print(f"C0ref {'PASS' if ok else 'FAIL'} max={worst:.2e} (face loops against matrices, <=1e-13)")
    return 0 if ok else 1


def cmd_compare(d, od, cells):
    worst, ok, parts = 0.0, True, []
    for name in MESHES:
        mn = f'{name}_{cells}'
        for k in C0_OUTPUTS:
            ref = np.fromfile(os.path.join(d, f'{mn}.ref_{k}'))
            fn = os.path.join(od, f'{mn}.out_{k}')
            if not os.path.exists(fn):
                print(f'C0 FAIL missing {fn}')
                return 1
            dev = np.fromfile(fn)
            if dev.size != ref.size:
                print(f'C0 FAIL size {mn}.{k} {dev.size} != {ref.size}')
                return 1
            e = rel(dev, ref)
            worst = max(worst, e)
            ok = ok and e <= 1e-13
            parts.append(f'{name}.{k}={e:.1e}')
        print(f'C0 {mn} ' + ' '.join(p for p in parts if p.startswith(name)))
    print(f"C0 {'PASS' if ok else 'FAIL'} max={worst:.2e} (<=1e-13; {len(MESHES)} meshes at {cells}^3, "
          f"{','.join(C0_OUTPUTS)})")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------- C1, C2
def load_trace(dirname, case, who, k):
    pre = os.path.join(dirname, f'{case}.{who}_')
    return {q: np.fromfile(pre + f'{q}_{k}') for q in ('Ustar', 'F', 'U', 'p')}


def scalars(dirname, case, who):
    rows = {}
    with open(os.path.join(dirname, f'{case}.{who}_scalars')) as f:
        for line in f:
            t = line.split()
            rows[int(t[0])] = (float(t[1]), float(t[2]))
    return rows


def cmd_trace(refdir, devdir):
    ok_all = True
    # C1: tg_Z16_F, step 1
    c = 'tg_Z16_F'
    r, d = load_trace(refdir, c, 'ref', 1), load_trace(devdir, c, 'dev', 1)
    e = {q: rel(d[q], r[q]) for q in r}
    om_r, om_d = scalars(refdir, c, 'ref')[1][0], scalars(devdir, c, 'dev')[1][0]
    ok = e['F'] <= 1e-10 and e['Ustar'] <= 1e-10 and e['U'] <= 1e-10 and e['p'] <= 1e-8 and om_r == 1.0 \
        and om_d == 1.0
    ok_all = ok_all and ok
    print(f"C1 {'PASS' if ok else 'FAIL'} {c} step 1: F={e['F']:.2e} U*={e['Ustar']:.2e} U1={e['U']:.2e} "
          f"(<=1e-10) p={e['p']:.2e} (<=1e-8) omega ref={om_r:.6f} dev={om_d:.6f} (both 1)")
    # C2: 10 steps
    eu, ep, hist = 0.0, 0.0, []
    for k in range(1, 11):
        r, d = load_trace(refdir, c, 'ref', k), load_trace(devdir, c, 'dev', k)
        eu, ep = max(eu, rel(d['U'], r['U'])), max(ep, rel(d['p'], r['p']))
        hist.append(f"{rel(d['U'], r['U']):.1e}")
    c2 = 'tg_W16_R'
    ew = 0.0
    for k in range(1, 11):
        r, d = load_trace(refdir, c2, 'ref', k), load_trace(devdir, c2, 'dev', k)
        ew = max(ew, rel(d['U'], r['U']))
    ok = eu <= 1e-9 and ep <= 1e-7 and ew <= 1e-8
    ok_all = ok_all and ok
    print(f"C2 {'PASS' if ok else 'FAIL'} {c} 10 steps: U={eu:.2e} (<=1e-9) p={ep:.2e} (<=1e-7) "
          f"[U per step {','.join(hist)}]; {c2} 10 steps: U={ew:.2e} (<=1e-8)")
    return 0 if ok_all else 1


def cmd_c3(da, db, tol):
    a = np.fromfile(os.path.join(da, 'tg_Z16_F.dev_U_10'))
    b = np.fromfile(os.path.join(db, 'tg_Z16_F.dev_U_10'))
    e = rel(a, b)
    bit = bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))
    print(f"C3pair {'PASS' if e <= tol else 'FAIL'} U after 10 steps rel={e:.2e} (<={tol:.0e}) bitwise={int(bit)}")
    return 0 if e <= tol else 1


def cmd_trace_moving(d, cases, tol):
    ok_all = True
    for c in cases:
        meta = {}
        with open(os.path.join(d, f'{c}.meta')) as f:
            for tok in f.read().split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    meta[k] = v
        tau, rho, nu, mode = float(meta['tau']), float(meta['rho']), float(meta['nu']), meta['mode']
        rows = {}
        with open(os.path.join(d, f'{c}.dev_scalars')) as f:
            for line in f:
                t = line.split()
                rows[int(t[0])] = {'omega': float(t[1]), 'gcl': float(t[3]), 'closure': float(t[4]),
                                   **{kv.split('=')[0]: kv.split('=')[1] for kv in t[5:]}}
        worst = {q: 0.0 for q in ('F', 'Ustar', 'U', 'p')}
        for k in sorted(rows):
            def rd(q):
                return np.fromfile(os.path.join(d, f'{c}.{q}_{k}'))
            mh, m1 = ar.Mesh(d, f'{c}.Fh_{k}'), ar.Mesh(d, f'{c}.F1_{k}')
            N = mh.N
            U, Uext, xdot = (rd(q).reshape(N, 3) for q in ('in_U', 'in_Uext', 'in_xdot'))
            p, Vn = rd('in_p'), rd('in_Vn')
            U1, p1, r = ars.moving_step(mh, m1, U, Uext, p, Vn, xdot, tau, mode, rho=rho, nu=nu,
                                        omega=rows[k]['omega'])
            e = {'F': rel(rd('dev_F'), r['F']), 'Ustar': rel(rd('dev_Ustar'), r['Ustar']),
                 'U': rel(rd('dev_U'), U1), 'p': rel(rd('dev_p'), p1)}
            for q in worst:
                worst[q] = max(worst[q], e[q])
            dev = rows[k]
            K1 = max(r['K1'], 1e-300)
            print(f"Tstep {c} k={k} F={e['F']:.2e} U*={e['Ustar']:.2e} U1={e['U']:.2e} p={e['p']:.2e} "
                  f"omega={dev['omega']:.6f} gcl ref={r['gcl']:.2e} dev={dev['gcl']:.2e} "
                  f"closure/K ref={abs(r['closure']) / K1:.2e} dev={abs(dev['closure']) / K1:.2e} "
                  f"B1 ref={r['B1']:.6e} dev={float(dev['B1']):.6e} "
                  f"mom_booked/|P| ref={np.abs(r['mom_booked_step']).max() / r['absP1']:.2e} "
                  f"rebuilt={dev['rebuilt']} Fh={mh.F} F1={m1.F} faces")
        ok = all(v <= tol for v in worst.values())
        ok_all = ok_all and ok
        print(f"Tcase {'PASS' if ok else 'FAIL'} {c} {len(rows)} steps: F={worst['F']:.2e} U*={worst['Ustar']:.2e} "
              f"U1={worst['U']:.2e} p={worst['p']:.2e} (<={tol:.0e})")
    print(f"T {'PASS' if ok_all else 'FAIL'} {' '.join(cases)} (per step against moving_step, <={tol:.0e})")
    return 0 if ok_all else 1


# ------------------------------------------------------------------------------------------- the runs
def read_kv_lines(path, tag):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.startswith(tag + ' '):
                d = {}
                for kv in line.split()[1:]:
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        try:
                            d[k] = float(v)
                        except ValueError:
                            d[k] = v
                out.append(d)
    return out


def cmd_judge(log_dir, ref_dir):
    lines, ok_all = [], True

    def verdict(cid, ok, text):
        nonlocal ok_all
        word = 'PASS' if ok is True else ('FAIL' if ok is False else 'REPORT')
        if ok is False:
            ok_all = False
        lines.append(f'{cid} {word} {text}')

    def res(case, be):
        r = read_kv_lines(os.path.join(log_dir, f'S1_2b.run.{case}.{be}.log'), 'RESULT')
        return r[-1] if r and r[-1].get('status') == 'DONE' else None

    def ref(case):
        r = read_kv_lines(os.path.join(ref_dir, f'S1_2a.{case}.log'), 'RESULT')
        return r[-1] if r and r[-1].get('status') == 'DONE' else None

    BE = ('host', 'gpu')

    def gather(cases, key):
        """{(case, be): value}; missing runs give None."""
        return {(c, b): (res(c, b) or {}).get(key) for c in cases for b in BE}

    def missing(vals):
        return [f'{c}.{b}' for (c, b), v in vals.items() if v is None]

    def fmt(vals, f='{:.1e}'):
        return ' '.join(f'{c}.{b}:' + (f.format(v) if v is not None else 'missing') for (c, b), v in vals.items())

    # R0
    z = gather(['r0_zero_F', 'r0_zero_R'], 'maxU')
    u = gather(['r0_unif_F', 'r0_unif_R'], 'unif_dev')
    mz = missing(z) + missing(u)
    if mz:
        verdict('R0', False, f'missing {mz}')
    else:
        verdict('R0', all(v == 0.0 for v in z.values()) and max(u.values()) <= 1e-13,
                f'rest max|U| (exactly 0): {fmt(z)}; uniform max dev/|U| (<=1e-13): {fmt(u, "{:.2e}")}')
    # R1, R2'
    cs = ['tg_Z16_F', 'tg_Z16_R', 'tg_W16_F', 'tg_W16_R']
    cl, rd = gather(cs, 'closure'), gather(cs, 'res_over_diss')
    mb, dv, mo = gather(cs, 'mom_booked'), gather(cs, 'divPhi1'), gather(cs, 'mom')
    if missing(cl):
        verdict('R1', False, f'missing {missing(cl)}')
        verdict("R2'", False, f'missing {missing(cl)}')
    else:
        verdict('R1', max(cl.values()) <= 1e-13 and max(rd.values()) <= 1e-10,
                f'max|closure|/K={max(cl.values()):.2e} (<=1e-13) T1 max|Bres|/|B4+B7+B8a|={max(rd.values()):.2e} '
                f'(<=1e-10) [{fmt(cl)}]')
        verdict("R2'", max(mb.values()) <= 1e-13 and max(dv.values()) <= 1e-14,
                f'momentum drift with the residual booked={max(mb.values()):.2e} (<=1e-13) max tau|D Phi_1|/V='
                f'{max(dv.values()):.2e} (<=1e-14); reported: drift without booking {max(mo.values()):.2e}')
    # R3
    ref_mu = {f'{f}{n}': (ref(f'tg_{f}{n}_F') or {}).get('mu_exact') for f in 'ZW' for n in (8, 16, 32)}
    cs3 = ['r3_tg_W16_F', 'r3_tg_W32_F']
    ex, mr, om = gather(cs3, 'excess'), gather(cs3, 'mu_ratio'), gather(cs3, 'omega_mu')
    mus = {f'{f}{n}.{b}': (res(f'tg_{f}{n}_F', b) or {}).get('mu_exact') for f in 'ZW' for n in (8, 16, 32, 64)
           for b in BE}
    if missing(ex):
        verdict('R3', False, f'missing {missing(ex)}')
    else:
        dev1 = []
        for key, v in mus.items():
            mk = key.split('.')[0]
            if mk in ref_mu and ref_mu[mk] is not None:
                dev1.append(abs(v / ref_mu[mk] - 1) if v is not None else np.inf)
        md = max(dev1) if dev1 else np.inf
        verdict('R3', max(ex.values()) <= 1e-13 and min(mr.values()) >= 0.9 and max(om.values()) <= 4.0 / 3.0
                and md <= 0.01,
                f'max(dK-unsigned)/K={max(ex.values()):.2e} (<=1e-13) min mu_hat/mu_exact={min(mr.values()):.4f} '
                f'(>=0.9; samples {fmt(gather(cs3, "mu_samples"), "{:.0f}")}) max omega*mu_exact='
                f'{max(om.values()):.4f} (<=4/3) mu_exact vs reference 8-32^3 max rel dev={md:.2e} (<=0.01); '
                'mu_exact ' + ' '.join(f'{k}:{v:.4f}' if v is not None else f'{k}:missing' for k, v in mus.items()))
    # R4
    e = gather(['tg_Z16_F', 'tg_Z16_R', 'tg_Z32_F', 'tg_Z32_R'], 'err')
    if missing({k: v for k, v in e.items() if '32' in k[0]}):
        verdict('R4', False, f'missing {missing(e)}')
    else:
        d32 = {b: abs(e[('tg_Z32_F', b)] / e[('tg_Z32_R', b)] - 1) for b in BE}
        d16 = {b: (abs(e[('tg_Z16_F', b)] / e[('tg_Z16_R', b)] - 1) if None not in (e[('tg_Z16_F', b)],
               e[('tg_Z16_R', b)]) else float('nan')) for b in BE}
        verdict('R4', max(d32.values()) <= 0.02, '|err_F/err_R - 1| 32^3 ' + ' '.join(
            f'{b}={v:.4f}' for b, v in d32.items()) + ' (<=0.02); reported 16^3 ' + ' '.join(
            f'{b}={v:.4f}' for b, v in d16.items()))
    # R5
    ok5, parts = True, []
    for fam in 'ZW':
        for fld in ('tg', 'abc'):
            for b in BE:
                es = [(res(f'{fld}_{fam}{n}_F', b) or {}).get('err') for n in (16, 32, 64)]
                rs = [(ref(f'{fld}_{fam}{n}_F') or {}).get('err') for n in (16, 32)]
                if None in es or None in rs:
                    parts.append(f'{fld}{fam}.{b}:missing')
                    ok5 = False
                    continue
                a16, a32 = abs(es[0] / rs[0] - 1), abs(es[1] / rs[1] - 1)
                o1, o2 = np.log2(es[0] / es[1]), np.log2(es[1] / es[2])
                ok = (a16 <= 1e-4 if fam == 'Z' else True) and a32 <= 0.02 and o2 >= 1.8
                ok5 = ok5 and bool(ok)
                parts.append(f'{fld}{fam}.{b}:{es[0]:.4e},{es[1]:.4e},{es[2]:.4e} vs_ref16={a16:.1e} '
                             f'vs_ref32={a32:.1e} order16-32={o1:.2f} order32-64={o2:.2f}'
                             + ('' if ok else '[MISS]'))
    verdict('R5', ok5, '((a) Z 16^3 vs reference <=1e-4, 32^3 <=0.02; (b) order 32->64 >=1.8) ' + ' '.join(parts))
    # R6
    a = gather(['r6i_W16'], 'maxU_rel_bound')
    g = gather(['r6ii_rest_W16'], 'gt_dev')
    mdb = gather(['r6ii_tg_W16'], 'mean_dev_booked')
    md = gather(['r6ii_tg_W16'], 'mean_dev')
    if missing(a) + missing(g) + missing(mdb):
        verdict('R6', False, f'missing {missing(a) + missing(g) + missing(mdb)}')
    else:
        verdict('R6', max(a.values()) <= 1e-12 and max(g.values()) <= 2e-11 and max(mdb.values()) <= 1e-12,
                f"(i) max|U|/(|grad psi|h)^0.5 {fmt(a, '{:.2e}')} (<=1e-12) (ii')' rest |U-gt|/|g|t "
                f"{fmt(g, '{:.2e}')} (<=2e-11) TG mean with residual booked {fmt(mdb, '{:.2e}')} (<=1e-12); "
                f"reported without booking {fmt(md, '{:.2e}')}")
    # L
    lap = {}
    for b in BE:
        for n in (32, 64):
            for d in read_kv_lines(os.path.join(log_dir, f'S1_2b.lap.{n}.{b}.log'), 'LAP'):
                lap[(n, int(d['ratio']), b)] = d
    bound = {(32, 1): 14, (64, 1): 17, (32, 1000): 19, (64, 1000): 28}
    if len(lap) < 8:
        verdict('L', False, f'missing LAP lines ({len(lap)}/8)')
    else:
        okL, parts = True, []
        for (n, r), bd in bound.items():
            h, gp = lap[(n, r, 'host')], lap[(n, r, 'gpu')]
            ih, ig = int(h['iters']), int(gp['iters'])
            same = h['sizes'] == gp['sizes']
            ok = ih <= bd and abs(ig - ih) <= 1 and same and h['converged'] == 1 and gp['converged'] == 1
            okL = okL and ok
            parts.append(f'{n}^3/r{r}:host={ih},gpu={ig}(<={bd},+-1) levels_equal={int(same)}({h["sizes"]})')
        mem = lap[(64, 1000, 'gpu')]['setup_peak_MiB'], lap[(64, 1, 'gpu')]['setup_peak_MiB']
        okL = okL and max(mem) <= 1024.0
        verdict('L', okL, ' '.join(parts) + f' GPU setup memory at 64^3 r1/r1000={mem[1]:.1f}/{mem[0]:.1f} MiB '
                f'(<=1024; Kokkos Tools allocation high-water mark during setup) setup_ms 64^3 host/gpu='
                f"{lap[(64, 1, 'host')]['setup_ms']:.1f}/{lap[(64, 1, 'gpu')]['setup_ms']:.1f}")
    # T
    okT, parts = True, []
    for n in (32, 64):
        for b in BE:
            d = res(f'tg_Z{n}_F', b)
            if d is None:
                parts.append(f'{n}^3.{b}:missing')
                okT = False
                continue
            pa = d['pred_apps_mean']
            ok = pa <= 40
            if n == 64:
                ok = ok and max(d['s3_its_mean'], d['s5_its_mean'], d['s6_its_mean']) <= 20
            okT = okT and bool(ok)
            ms = {k: d[k] * 1e3 for k in ('wall_per_step', 't_pred', 't_s3', 't_s3_solve', 't_mu', 't_s5', 't_s6',
                                          't_s6_solve', 't_budget')}
            parts.append(f"{n}^3.{b}: {ms['wall_per_step']:.1f} ms/step [predictor {ms['t_pred']:.1f}, "
                         f"S3 flux map {ms['t_s3'] - ms['t_s3_solve']:.1f} + solve {ms['t_s3_solve']:.1f}, "
                         f"mu {ms['t_mu']:.1f}, S5 solve {ms['t_s5']:.1f}, S6 flux map "
                         f"{ms['t_s6'] - ms['t_s6_solve']:.1f} + solve {ms['t_s6_solve']:.1f}, budget "
                         f"{ms['t_budget']:.1f}] frame setup {d['frame_setup']:.3f}s "
                         f"pred_apps={pa:.1f}(<=40) its S3/S5/S6={d['s3_its_mean']:.1f}/{d['s5_its_mean']:.1f}/"
                         f"{d['s6_its_mean']:.1f}" + ('(<=20)' if n == 64 else '') + f" mu_its={d['mu_its_per_solve']:.1f}")
    verdict('T', okT, ' | '.join(parts))
    for ln in lines:
        print(ln)
    print(f"G1.2runs {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('inputs')
    a.add_argument('dir')
    a.add_argument('--cells', type=int, default=16)
    c = sub.add_parser('compare')
    c.add_argument('dir')
    c.add_argument('odir')
    c.add_argument('--cells', type=int, default=16)
    t = sub.add_parser('trace')
    t.add_argument('refdir')
    t.add_argument('devdir')
    c3 = sub.add_parser('c3')
    c3.add_argument('da')
    c3.add_argument('db')
    c3.add_argument('--tol', type=float, default=1e-9)
    tm = sub.add_parser('trace-moving')
    tm.add_argument('dir')
    tm.add_argument('cases', nargs='+')
    tm.add_argument('--tol', type=float, default=1e-12)
    j = sub.add_parser('judge')
    j.add_argument('log_dir')
    j.add_argument('ref_dir')
    args = ap.parse_args()
    if args.cmd == 'inputs':
        return cmd_inputs(args.dir, args.cells)
    if args.cmd == 'compare':
        return cmd_compare(args.dir, args.odir, args.cells)
    if args.cmd == 'trace':
        return cmd_trace(args.refdir, args.devdir)
    if args.cmd == 'trace-moving':
        return cmd_trace_moving(args.dir, args.cases, args.tol)
    if args.cmd == 'c3':
        return cmd_c3(args.da, args.db, args.tol)
    return cmd_judge(args.log_dir, args.ref_dir)


if __name__ == '__main__':
    sys.exit(main())
