/**
 * @file fv/collocated.hpp
 * \brief Track C, rung C2b (Voronoi methods plan, verdict V4): the COLLOCATED Navier–Stokes solver
 * on the Voronoi face mesh with the approximate projection of `peclet.flow`'s SolverColocated
 * (Almgren–Bell–Colella structure; user ruling 2026-09-03: as flow, NOT Rhie–Chow).
 *
 * flow's collocated design (flow/src/mac_approx_projection.hpp, gauge_exact_gradient.hpp,
 * doc/collocated_invisible_subspace.md) transferred to the unstructured mesh:
 *   * the velocity VECTOR lives at the seeds U_i, the constraint acts on the centre→face
 *     interpolation T (flow: centerToFace): D T U = 0 with D the face divergence;
 *   * the cell pressure gradient is the EXACT TRANSPOSE of the constraint, G = −(D T)ᵀ = Tᵀ∘grad_f
 *     (flow: the gauge-exact gradient = the transpose of centerToFace, "the pressure does no work
 *     on the constraint manifold"), and it is used BOTH in the incremental predictor (−G P^n) and
 *     in the cell correction (−Δt G φ) — momentum and constraint stay one operator family;
 *   * the FACE field is projected exactly (L φ = div T U* / Δt with the same two-point L as the
 *     covolume solver) and transports the momentum; the cell field is corrected approximately;
 *   * incremental pressure P += φ (flow: rotational Timmermans update P += (ρ/Δt)φ − μ div u*;
 *     the viscous part belongs to the semi-implicit viscous solve — here the diffusion is
 *     explicit, so the plain increment is the consistent form).
 * On a cubic lattice T is the ½/½ average and G the central difference — flow's kernels exactly.
 * The unstructured extension: the plain T interpolates at the CONNECTOR FOOT, which sits off the
 * face centroid by the skewness (B4: first-order flux consistency ∝ skewness); `skewCorrected`
 * extrapolates each cell's value to the centroid with its Green–Gauss gradient (T = T₀ + T₁∘GG)
 * and G is the exact transpose of THAT (faceInterpTranspose) — the same adjoint pairing, second
 * order on skewed faces. No Rhie–Chow term anywhere. Immersed solids do not exist on this mesh
 * (it is body-fitted), so flow's invisible-subspace / ghost machinery has no counterpart here.
 */
#ifndef PECLET_VORO_FV_COLLOCATED_HPP
#define PECLET_VORO_FV_COLLOCATED_HPP

#include <Kokkos_Core.hpp>

#include "peclet/voro/fv/covolume.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/fv/operators.hpp"

namespace peclet::voro::fv {

template <class Real>
struct CollocatedNS {
  using Exec = peclet::core::ExecSpace;
  FaceMesh<Real> m;
  Real nu = 0;
  Real convScale = 1;         // 0 = Stokes (diagnostic)
  bool incremental = true;    // flow's incremental_: false = classical Chorin (φ = the pressure)
  bool skewCorrected = true;  // centroid-consistent constraint pair (see the header); MEASURED
                              // TGV order 2.10 / 2.08 on jittered / CVT meshes vs 1.72 / 1.17 plain
  DV<Real> U, uf, p, force;   // 3N, nFaces, N, 3N (optional)
  DV<Real> Uwall, ub;         // 3 × nBoundary wall velocity, nBoundary wall flux (C3)
  bool wallQuadratic = true;  // second-order wall gradient (wallGradientLS) vs two-point
  DV<Real> wg;                // 3 × nBoundary wall gradient
  DV<Real> a, U1, U2, uf1, uf2, div, gphi, gp, phi, g9;  // workspace

  void setup(const FaceMesh<Real>& mesh, Real viscosity, bool amg = false) {
    m = mesh;
    nu = viscosity;
    const int N = m.nCells, F = m.nFaces;
    U = DV<Real>("co.U", 3 * N);
    uf = DV<Real>("co.uf", F);
    p = DV<Real>("co.p", N);
    a = DV<Real>("co.a", 3 * N);
    U1 = DV<Real>("co.U1", 3 * N);
    U2 = DV<Real>("co.U2", 3 * N);
    uf1 = DV<Real>("co.uf1", F);
    uf2 = DV<Real>("co.uf2", F);
    div = DV<Real>("co.div", N);
    gphi = DV<Real>("co.gphi", F);
    gp = DV<Real>("co.gp", 3 * N);
    phi = DV<Real>("co.phi", N);
    g9 = DV<Real>("co.g9", 9 * N);
    setWallVelocity(DV<Real>{});
    poisson.setup(m, amg);
  }
  /// Prescribe the boundary-face velocity (3 × nBoundary; empty = no-slip): boundary fluxes
  /// u_f = U_wall·n; the viscous wall flux and the convective wall value read U_wall.
  void setWallVelocity(const DV<Real>& Uw) {
    const int nB = m.nFaces - m.nInterior, nI = m.nInterior;
    Uwall = Uw;
    ub = DV<Real>("fv.ub", nB);
    if (Uw.extent(0) == 0)
      return;
    const FaceMesh<Real> mm = m;
    const DV<Real> bb = ub;
    Kokkos::parallel_for(
        "fv.ub", Kokkos::RangePolicy<Exec>(0, nB), KOKKOS_LAMBDA(const int b) {
          Real un = 0;
          for (int c = 0; c < 3; ++c)
            un += Uw(3 * b + c) * mm.normal(nI + b, c);
          bb(b) = un;
        });
  }
  /// Impose the boundary fluxes on a face field.
  void applyWallFlux(const DV<Real>& uf) const {
    const int nB = m.nFaces - m.nInterior, nI = m.nInterior;
    const DV<Real> bb = ub;
    Kokkos::parallel_for(
        "fv.wall", Kokkos::RangePolicy<Exec>(0, nB),
        KOKKOS_LAMBDA(const int b) { uf(nI + b) = bb(b); });
  }
  PressureSolver<Real> poisson;

  /// The constraint interpolation: face flux of the cell field.
  void toFaces(const DV<Real>& Uc, const DV<Real>& out) {
    if (skewCorrected) {
      vectorGreenGauss(m, Uc, g9, Uwall);
      projectToFaces(m, Uc, out, g9, ub);
    } else {
      projectToFaces(m, Uc, out, DV<Real>{}, ub);
    }
  }
  /// The cell gradient of a scalar q = the transpose of the constraint applied to grad_f q.
  void cellGrad(const DV<Real>& q, const DV<Real>& out) {
    faceGradient(m, q, gphi);
    faceInterpTranspose(m, gphi, out, skewCorrected, g9, Uwall.extent(0) > 0);
  }
  /// a = tendency of Uin transported by ufin, minus the current pressure gradient (incremental).
  void tendency(const DV<Real>& Uin, const DV<Real>& ufin, const DV<Real>& out) {
    if (wallQuadratic && m.nFaces > m.nInterior) {
      if (wg.extent(0) == 0)
        wg = DV<Real>("co.wg", 3 * (m.nFaces - m.nInterior));
      vectorGreenGauss(m, Uin, g9, Uwall);
      wallGradientLS(m, Uin, g9, Uwall, wg);
      cellTendency(m, ufin, Uin, nu, out, force, convScale, Uwall, wg);
    } else {
      cellTendency(m, ufin, Uin, nu, out, force, convScale, Uwall);
    }
    if (!incremental)
      return;
    const DV<Real> gc = gp;
    cellGrad(p, gc);
    Kokkos::parallel_for(
        "co.gradp", Kokkos::RangePolicy<Exec>(0, 3 * m.nCells),
        KOKKOS_LAMBDA(const int i) { out(i) -= gc(i); });
  }
  /// Approximate projection of the predictor Ustar (in place): T Ustar → exact face projection →
  /// cell correction with the transpose gradient; ufOut = the divergence-free face flux; P += φ.
  void project(const DV<Real>& Ustar, const DV<Real>& ufOut, Real dt) {
    const DV<Real> dv = div, g = gphi, gc = gp, ph = phi, pp = p;
    toFaces(Ustar, ufOut);
    divergence(m, ufOut, dv);
    const Real s = -Real(1) / dt;
    Kokkos::parallel_for(
        "co.rhsP", Kokkos::RangePolicy<Exec>(0, m.nCells),
        KOKKOS_LAMBDA(const int i) { dv(i) *= s; });
    poisson.solve(dv, ph);
    faceGradient(m, ph, g);
    Kokkos::parallel_for(
        "co.corrF", Kokkos::RangePolicy<Exec>(0, m.nFaces),
        KOKKOS_LAMBDA(const int f) { ufOut(f) -= dt * g(f); });
    cellGrad(ph, gc);
    Kokkos::parallel_for(
        "co.corrC", Kokkos::RangePolicy<Exec>(0, 3 * m.nCells),
        KOKKOS_LAMBDA(const int i) { Ustar(i) -= dt * gc(i); });
    if (incremental)
      Kokkos::parallel_for(
          "co.pinc", Kokkos::RangePolicy<Exec>(0, m.nCells),
          KOKKOS_LAMBDA(const int i) { pp(i) += ph(i); });
    else
      Kokkos::deep_copy(pp, ph);
  }
  /// Initialise from a cell field: U = Uin projected once (dt = 1, no pressure kept).
  void initialize(const DV<Real>& Uin) {
    Kokkos::deep_copy(U, Uin);
    Kokkos::deep_copy(p, Real(0));
    project(U, uf, Real(1));
    Kokkos::deep_copy(p, Real(0));
  }
  /// One SSP-RK3 step, flow's predictor/projection structure at every stage.
  void step(Real dt) {
    const int N3 = 3 * m.nCells;
    const DV<Real> UU = U, aa = a, V1 = U1, V2 = U2;
    tendency(UU, uf, aa);
    Kokkos::parallel_for(
        "co.s1", Kokkos::RangePolicy<Exec>(0, N3),
        KOKKOS_LAMBDA(const int i) { V1(i) = UU(i) + dt * aa(i); });
    project(V1, uf1, dt);
    tendency(V1, uf1, aa);
    Kokkos::parallel_for(
        "co.s2", Kokkos::RangePolicy<Exec>(0, N3), KOKKOS_LAMBDA(const int i) {
          V2(i) = Real(0.75) * UU(i) + Real(0.25) * (V1(i) + dt * aa(i));
        });
    project(V2, uf2, Real(0.25) * dt);
    tendency(V2, uf2, aa);
    Kokkos::parallel_for(
        "co.s3", Kokkos::RangePolicy<Exec>(0, N3),
        KOKKOS_LAMBDA(const int i) { UU(i) = (UU(i) + Real(2) * (V2(i) + dt * aa(i))) / Real(3); });
    project(UU, uf, Real(2) / Real(3) * dt);
  }
  /// ½ Σ V_i |U_i|²
  Real kineticEnergy() const {
    const FaceMesh<Real> mm = m;
    const DV<Real> UU = U;
    Real e = 0;
    Kokkos::parallel_reduce(
        "co.ke", Kokkos::RangePolicy<Exec>(0, m.nCells),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          acc += mm.cellVolume(i) * (UU(3 * i) * UU(3 * i) + UU(3 * i + 1) * UU(3 * i + 1) +
                                     UU(3 * i + 2) * UU(3 * i + 2));
        },
        e);
    return Real(0.5) * e;
  }
  /// max_i |div u_f|_i of the transporting face flux (exactly projected)
  Real maxFaceDivergence() { return maxDivOf(uf); }
  /// max_i |div (T U)|_i — the residual divergence of the CELL field (the "approximate" of ABC)
  Real maxCellDivergence() {
    toFaces(U, uf1);
    return maxDivOf(uf1);
  }
  Real maxDivOf(const DV<Real>& flux) {
    const DV<Real> dv = div;
    divergence(m, flux, dv);
    Real mx = 0;
    Kokkos::parallel_reduce(
        "co.maxdiv", Kokkos::RangePolicy<Exec>(0, m.nCells),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          const Real v = Kokkos::fabs(dv(i));
          if (v > acc)
            acc = v;
        },
        Kokkos::Max<Real>(mx));
    return mx;
  }
};

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_COLLOCATED_HPP
