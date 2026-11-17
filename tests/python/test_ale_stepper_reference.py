"""pytest: the host reference of the time step (ale_reference_stepper.py, work order S1.2a) at 8^3, the small
cases of gate G1.2r (criteria of the brief ALE_S1_2A, fixed 2026-11-17):

  operators  transport T, upwind Q, reconstruction jump R - Delta, centroid correction delta, the explicit part and
             the predictor operator: explicit matrix against face loop <= 1e-14 relative; T skew, Q symmetric;
             R U = 0 for a linear field (through differences, relative to |B| h)
  R0         W 8^3: U = 0 stays exactly 0 for 10 steps; uniform U = (0.3, -0.7, 0.5): max deviation <= 1e-13 |U|
             over 100 steps, modes F and R
  R1, R2     Taylor-Green, Z and W at 8^3, modes F and R, to t = 1: |closure| <= 1e-13 K at every step; momentum
             drift <= 1e-13 sum M|U|; max tau |D Phi_1| / V <= 1e-13

The meshes are those of `test_ale_frame --export DIR --cells 8 --mesh M1|M2` (M1 = Z, M2 = W); DIR is given by
ALE_EXPORT_DIR (input data only; the tests skip without it). tools/ale_gates/G1_2r.sh sets it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ale_reference as ar  # noqa: E402
import ale_reference_stepper as ars  # noqa: E402

EXPORT = os.environ.get('ALE_EXPORT_DIR', '')


def _have(name):
    return EXPORT and os.path.exists(os.path.join(EXPORT, f'{name}.counts'))


needs_meshes = pytest.mark.skipif(not (_have('M1_8') and _have('M2_8')),
                                  reason='ALE_EXPORT_DIR with M1_8, M2_8 not given')


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


@needs_meshes
@pytest.mark.parametrize('name', ['M1_8', 'M2_8'])
def test_transport_operators(name):
    m = ar.Mesh(EXPORT, name)
    mx = ar.Matrices(m, mu=np.full(m.F, 0.01))
    tm = ar.TransportMatrices(mx)
    rng = np.random.default_rng(20261117)
    U = rng.standard_normal((m.N, 3))
    Fm = rng.standard_normal(m.F)
    u = U.ravel()
    assert _rel(tm.T(Fm) @ u, ar.transport_mf(m, U, Fm).ravel()) <= 1e-14
    assert _rel(tm.Q(Fm) @ u, ar.upwind_mf(m, U, Fm).ravel()) <= 1e-14
    assert _rel(tm.Jump @ u, ar.reconstruction_jump_mf(m, U).ravel()) <= 1e-14
    assert _rel(tm.Delta_c @ u, ar.centroid_correction_mf(m, U).ravel()) <= 1e-14
    assert _rel(tm.explicit(Fm) @ u, ar.transport_explicit_mf(m, U, Fm).ravel()) <= 1e-14
    rhoV = 1.3 * m.V
    tau = 0.02
    A = tm.predictor(Fm, tau, ars.mass_matrix(rhoV))
    ref = ar.predictor_mf(m, U, Fm, tau, lambda X: ars.apply_mass(rhoV, X), np.full(m.F, 0.01))
    assert _rel(A @ u, ref.ravel()) <= 1e-14
    T = tm.T(Fm)
    assert abs(u @ (T @ u)) <= 1e-14 * np.abs(T).sum() * np.abs(u).max() ** 2
    Q = tm.Q(Fm)
    assert abs(Q - Q.T).max() <= 1e-14 * abs(Q).max()
    # R U = Delta U + (R - Delta) U = 0 for U = a + B x (through differences B conn)
    B = rng.standard_normal((3, 3))
    dU = m.conn @ B.T
    jump = -dU + ar.reconstruction_jump_mf(m, dU=dU)
    assert np.abs(jump).max() <= 1e-13 * np.abs(B).max() * m.h


def _run(case):
    out = ars.run_case(EXPORT, case, lambda s: None)
    assert out['status'] == 'DONE', out
    return out


@needs_meshes
@pytest.mark.parametrize('mode', ['F', 'R'])
def test_R0_rest_and_uniform(mode):
    z = _run(f'r0_zero_{mode}')
    assert z['maxU'] == 0.0
    u = _run(f'r0_unif_{mode}')
    assert u['unif_dev'] <= 1e-13


@needs_meshes
@pytest.mark.parametrize('fam', ['Z', 'W'])
@pytest.mark.parametrize('mode', ['F', 'R'])
def test_R1_R2_taylor_green(fam, mode):
    r = _run(f'tg_{fam}8_{mode}')
    assert r['closure'] <= 1e-13
    assert r['mom'] <= 1e-13
    assert r['divPhi1'] <= 1e-13
