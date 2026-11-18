/**
 * @file ale/moving_stepper.hpp
 * \brief Moving-grid stage 1, WO-S1.3a (docs/ale_design.md §4.1, §4.4, §11 A18): the time step for
 * ONE phase on a MOVING mesh, with two frames per step, as device code over the stages of
 * `ale/stepper.hpp`.
 *
 * One phase (uniform ρ and μ), periodic box, no body force. Masses are lumped: Mⁿ = ρVⁿ, M¹ = ρV¹,
 * ΔM = M¹ − Mⁿ. With Uⁿ, Uⁿ⁻¹, pⁿ, the generator positions xⁿ and the volumes Vⁿ of frame n:
 *  S1  U_ext = 1.5 Uⁿ − 0.5 Uⁿ⁻¹ (Uⁿ at the first step). Mesh rule (Q-MESH): Rest ẋ = 0, Follow
 *      ẋ = U_ext, Prescribed ẋ given per step by the caller; ẇ = 0 always (the weights never change
 *      in stage 1). x^h = xⁿ + ½τẋ, x¹ = xⁿ + τẋ, wrapped into the periodic box.
 *  S2  Fh = advanceFrame at x^h, then F1 = advanceFrame at x¹, each followed by `setupFrame`. A
 *      frame error (an empty cell, a one-sided facet, an open cell) is an error of the step: it
 *      throws; there is no retry and no smaller step.
 *  S3  on Fh: Φᵐ = sweptFlux(Fh, ẋ, 0); ψ from L_ρψ = DΦᵐ − (V¹ − Vⁿ)/τ (the operator pair of
 *      `compactFluxOf`, its stopping test max_i τ|r_i|/V_i ≤ compactTol, V of Fh), Φᵐ_c = Φᵐ +
 *      a ⊙ Δ_nψ, so that τDΦᵐ_c = V¹ − Vⁿ to compactTol V_i; a right-hand side that is exactly
 *      zero gives ψ = 0 without a solve. The solve stops on its recursive residual; while the
 *      recomputed measure max_i |τ(DΦᵐ_c)_i − (V¹_i − Vⁿ_i)|/V_i exceeds compactTol, the
 *      correction is solved again for that true residual and added (at most kMaxSweptRestarts).
 *      Φ = compactFluxOf(Fh, U_ext); F = Φ − Φᵐ_c; Fm = ρF.
 *  S4  on Fh: (M¹ − ¼ΔM + ½τ(S + Q + ζ)) U* = (Mⁿ + ¼ΔM − ½τ(S + Q + ζ)) Uⁿ + τ(e6 + e8 + C′ᵀpⁿ),
 *      S the skew transport, i.e. the flux form Σ_f s_if Fm_f ½(x_i + x_o) with (DFm)_i = −ΔM_i/τ.
 *      GMRES as on a static mesh; the preconditioner's mass is M¹ − ¼ΔM.
 *  S5, S6 on F1 with the mass M¹, exactly the stages of the static step.
 *  S8  Vⁿ ← V¹, xⁿ ← x¹, Uⁿ⁻¹ ← Uⁿ; the engine stays at x¹.
 * At rest (ẋ = 0) the frames equal frame n bitwise (§11 A11), ΔM = 0 exactly, Φᵐ = 0, ψ = 0, and
 * every operation of the step is that of `Stepper::step` on the static frame: U and p bit for bit
 * (gate V0.2).
 *
 * Energy budget, K = ½U·MU with the mass of its frame: the terms of `Stepper::step` with Fh's
 * operators in S4 and F1's in S5, plus B1 = +⅛ Σ_i ΔM_i |U*_i − Uⁿ_i|² (S is skew);
 * K¹ − Kⁿ = ½(U¹ − Uⁿ)·M¹(U¹ + Uⁿ) + ½ Σ_i ΔM_i |Uⁿ_i|². B1 is sign-indefinite (in unsignedSum).
 * Momentum: Σ M¹U¹ − Σ MⁿUⁿ = ΔP_def − Σ_i (rhs − AU*)_i, with the defect of S3 and of DΦ = 0
 * ΔP_def = ½ Σ_i [τ(DFm)_i + ΔM_i] Ū_i (Ū = ½(U* + Uⁿ); = ½ Σ ρ[τ(DF)_i + V¹_i − Vⁿ_i] Ū_i for one
 * phase), reported per step for the caller to book as the residual's momentum is booked.
 *
 * Single rank, periodic box, the power engine of `ale/frame.hpp` (equal weights on a moving mesh:
 * a power cell that empties is a frame error, §11 A5).
 */
#ifndef PECLET_VORO_ALE_MOVING_STEPPER_HPP
#define PECLET_VORO_ALE_MOVING_STEPPER_HPP

#include <algorithm>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

#include "peclet/voro/ale/frame.hpp"
#include "peclet/voro/ale/operators.hpp"
#include "peclet/voro/ale/stepper.hpp"

namespace peclet::voro::ale {

/// At most this many re-solves of the S3 correction for its true residual (brief ALE_S1_3A_R):
/// the solver stops on its recursive residual, which may sit just below compactTol while the
/// recomputed measure of A18 item 3 lands just above it.
inline constexpr int kMaxSweptRestarts = 3;

/// The mesh rule of S1 (Q-MESH, §11 A18 item 1).
enum class MeshMotion { Rest, Follow, Prescribed };

/// What one moving step did: the static report plus the terms of the moving mesh.
struct MovingStepReport : StepReport {
  double B1 = 0;   ///< ⅛ Σ ΔM |U* − Uⁿ|²
  double gcl = 0;  ///< max_i |τ(DΦᵐ_c)_i − (V¹_i − Vⁿ_i)|/V_i (V of Fh)
  double defectMomentum[3] = {0, 0, 0};  ///< ΔP_def per component
  krylov::SolveReport swept;  ///< the S3 correction solve (restarts' iterations included)
  int sweptRestarts = 0;      ///< re-solves of the S3 correction for its true residual
  bool sweptZero = false;     ///< the correction's right-hand side was exactly zero
  int rebuiltHalf = 0, rebuiltFull = 0;  ///< MeshFrame::numRebuilt of Fh, F1
  double timeFrames = 0, timeSetup = 0;  ///< the two advanceFrame, the two setupFrame (timed)
};

/// The mesh side of the state of a moving run: the positions and volumes of frame n, the warm start
/// of the S3 correction, and the two frames of the last step with their operators.
template <class Real>
struct MovingMesh {
  DV<Real> pos;     ///< 3·nCombined: xⁿ (wrapped into [0, L))
  DV<Real> volume;  ///< nCombined: Vⁿ
  DV<Real> psi;     ///< nCombined: warm start of the S3 correction (§4.4)
  Real L[3] = {1, 1, 1};
  FrameOptions<Real> frameOptions;
  MeshFrame<Real> half, full;  ///< Fh, F1 of the last step
  FrameData<Real> fh, f1;      ///< their operators
  DV<Real> muHalf, muFull;     ///< μ_f per face of Fh, F1

  /// Frame n = `f` (the initial frame): its positions and volumes.
  void start(const MeshFrame<Real>& f, const Real box[3]) {
    const int nc = f.mesh.nCombined;
    pos = DV<Real>("ale.mm.pos", 3 * nc);
    volume = DV<Real>("ale.mm.V", nc);
    psi = DV<Real>("ale.mm.psi", nc);
    Kokkos::deep_copy(pos, Kokkos::subview(f.pos, std::make_pair(0, 3 * nc)));
    Kokkos::deep_copy(volume, Kokkos::subview(f.mesh.cellVolume, std::make_pair(0, nc)));
    for (int k = 0; k < 3; ++k)
      L[k] = box[k];
  }
};

namespace detail {

/// out = x + s·ẋ per coordinate, wrapped into [0, L_d) as test_ale_frame's `advect`. K-cell.
template <class Real>
void movePositions(int n, const DV<Real>& x, const DV<Real>& xdot, Real s, const Real L[3],
                   const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  const Real L0 = L[0], L1 = L[1], L2 = L[2];
  Kokkos::parallel_for(
      "ale.movePos", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const int i) {
        const Real Ld[3] = {L0, L1, L2};
        for (int d = 0; d < 3; ++d) {
          Real p = x(3 * i + d) + s * xdot(3 * i + d);
          if (p >= Ld[d])
            p -= Ld[d];
          if (p < 0) {
            p += Ld[d];
            if (p >= Ld[d])
              p = 0;
          }
          out(3 * i + d) = p;
        }
      });
}

/// dV = V¹ − Vⁿ, ΔM = ρ dV, the masses of the predictor mL = ρV¹ − ¼ΔM and of its right-hand
/// side mR = ρVⁿ + ¼ΔM (at rest ΔM = 0 exactly and both are ρV bitwise). Per cell.
template <class Real>
void movingMasses(int n, const DV<Real>& rho, const DV<Real>& V1, const DV<Real>& Vn,
                  const DV<Real>& dV, const DV<Real>& dM, const DV<Real>& mL, const DV<Real>& mR) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.movingMasses", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const int i) {
        const Real d = V1(i) - Vn(i);
        const Real dm = rho(i) * d;
        dV(i) = d;
        dM(i) = dm;
        mL(i) = rho(i) * V1(i) - Real(0.25) * dm;
        mR(i) = rho(i) * Vn(i) + Real(0.25) * dm;
      });
}

/// out(3i + c) = mass_i x(3i + c): a lumped mass given per cell. Per cell.
template <class Real>
void applyCellMass(int n, const DV<Real>& mass, const DV<Real>& x, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.cellMass", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const int i) {
        const Real mi = mass(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = mi * x(3 * i + c);
      });
}

/// b_i ← b_i − dV_i/τ (b = DΦᵐ on entry). Per cell.
template <class Real>
void sweptRhs(int n, const DV<Real>& b, const DV<Real>& dV, Real tau) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.sweptRhs", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i) { b(i) = b(i) - dV(i) / tau; });
}

/// max_i |τ D_i − dV_i| / V_i (D = DΦᵐ_c). K5.
template <class Real>
double gclDefect(int n, const DV<Real>& D, const DV<Real>& dV, const DV<Real>& V, Real tau) {
  using Exec = peclet::core::ExecSpace;
  Real mx = 0;
  Kokkos::parallel_reduce(
      "ale.gcl", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        const Real e = Kokkos::fabs(tau * D(i) - dV(i)) / V(i);
        acc = e > acc ? e : acc;
      },
      Kokkos::Max<Real>(mx));
  return mx;
}

/// Σ_i w_i |a_i − b_i|² (3 components per cell). K5.
template <class Real>
double weightedSquareDifference(int n, const DV<Real>& w, const DV<Real>& a, const DV<Real>& b) {
  using Exec = peclet::core::ExecSpace;
  Real s = 0;
  Kokkos::parallel_reduce(
      "ale.wsq", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        Real q = 0;
        for (int c = 0; c < 3; ++c) {
          const Real d = a(3 * i + c) - b(3 * i + c);
          q += d * d;
        }
        acc += w(i) * q;
      },
      s);
  return s;
}

/// Σ_i w_i |a_i|². K5.
template <class Real>
double weightedSquare(int n, const DV<Real>& w, const DV<Real>& a) {
  using Exec = peclet::core::ExecSpace;
  Real s = 0;
  Kokkos::parallel_reduce(
      "ale.wsq0", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        Real q = 0;
        for (int c = 0; c < 3; ++c)
          q += a(3 * i + c) * a(3 * i + c);
        acc += w(i) * q;
      },
      s);
  return s;
}

/// ΔP_def = ½ Σ_i (τ DFm_i + ΔM_i) Ū_i per component. K5.
template <class Real>
void defectMomentum(int n, const DV<Real>& DFm, const DV<Real>& dM, const DV<Real>& ubar, Real tau,
                    double out[3]) {
  using Exec = peclet::core::ExecSpace;
  Real s0 = 0, s1 = 0, s2 = 0;
  Kokkos::parallel_reduce(
      "ale.defMom", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, Real& a0, Real& a1, Real& a2) {
        const Real w = Real(0.5) * (tau * DFm(i) + dM(i));
        a0 += w * ubar(3 * i);
        a1 += w * ubar(3 * i + 1);
        a2 += w * ubar(3 * i + 2);
      },
      s0, s1, s2);
  out[0] = s0;
  out[1] = s1;
  out[2] = s2;
}

/// A frame the step must not use (§11 A5, A12, A14): throws with what is wrong.
template <class Real>
void requireValidFrame(const MeshFrame<Real>& f, const char* which) {
  if (f.numEmptyCells > 0 || f.numOneSidedFacets > 0 || f.reciprocity.numOpenCells > 0)
    throw std::runtime_error(std::string("ale::MovingStepper: frame error in ") + which +
                             ": numEmptyCells=" + std::to_string(f.numEmptyCells) +
                             " numOneSidedFacets=" + std::to_string(f.numOneSidedFacets) +
                             " numOpenCells=" + std::to_string(f.reciprocity.numOpenCells));
}

}  // namespace detail

/// The time step of §11 A18 over the stages of `Stepper` (`base`).
template <class Real>
class MovingStepper {
 public:
  Stepper<Real> base;

  explicit MovingStepper(const StepOptions& o = {}) : base(o) {}

  /// One step. `rho` per cell (nCombined, uniform), `mu` the viscosity; `xdotGiven` (3·nCombined)
  /// the mesh velocity for MeshMotion::Prescribed (ignored otherwise); `body` must be empty (a
  /// body force on a moving mesh is work order S1.3c). On return s holds U^{n+1}, Uⁿ, p^{n+1}; mm
  /// holds x¹, V¹ and the two frames; the engine stands at x¹. Throws on a frame error.
  template <int MAXP, int MAXT, bool Weighted, class Sdf>
  MovingStepReport step(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                        MovingMesh<Real>& mm, FlowState<Real>& s, const DV<Real>& rho, Real mu,
                        Real tau, Projection mode, MeshMotion motion,
                        const DV<Real>& xdotGiven = DV<Real>(), const DV<Real>& body = DV<Real>()) {
    if (body.extent(0) > 0)
      throw std::invalid_argument("ale::MovingStepper: a body force on a moving mesh (S1.3c)");
    const StepOptions& opt = base.opt;
    auto& ws = base.ws;
    const int nc = (int)mm.volume.extent(0), n = 3 * nc;
    base.reserveAtLeast(nc, 0);
    allocateCells(nc);
    MovingStepReport rep;
    Kokkos::Timer timer;
    auto lap = [&](double& t) {
      if (opt.timed) {
        Kokkos::fence();
        t = timer.seconds();
        timer.reset();
      }
    };
    if (opt.timed)
      Kokkos::fence(), timer.reset();
    // S1: U_ext, mesh rates, positions
    if (s.hasPrevious)
      detail::combine(Real(1.5), s.U, Real(-0.5), s.Uprev, ws.Uext, n);
    else
      Kokkos::deep_copy(ws.Uext, s.U);
    if (motion == MeshMotion::Rest) {
      Kokkos::deep_copy(xdot_, Real(0));
    } else if (motion == MeshMotion::Follow) {
      Kokkos::deep_copy(xdot_, Kokkos::subview(ws.Uext, std::make_pair(0, n)));
    } else {
      if ((int)xdotGiven.extent(0) < n)
        throw std::invalid_argument(
            "ale::MovingStepper: Prescribed motion without a mesh velocity");
      Kokkos::deep_copy(xdot_, Kokkos::subview(xdotGiven, std::make_pair(0, n)));
    }
    detail::movePositions(nc, mm.pos, xdot_, Real(0.5) * tau, mm.L, xh_);
    detail::movePositions(nc, mm.pos, xdot_, tau, mm.L, x1_);
    // S2: the two frames and their operators
    double tPre = 0;
    lap(tPre);
    mm.half = advanceFrame(engine, xh_, mm.L, mm.frameOptions);
    detail::requireValidFrame(mm.half, "Fh");
    mm.full = advanceFrame(engine, x1_, mm.L, mm.frameOptions);
    detail::requireValidFrame(mm.full, "F1");
    rep.rebuiltHalf = mm.half.numRebuilt;
    rep.rebuiltFull = mm.full.numRebuilt;
    lap(rep.timeFrames);
    mm.muHalf = DV<Real>("ale.mm.muH", mm.half.mesh.nFaces);
    mm.muFull = DV<Real>("ale.mm.mu1", mm.full.mesh.nFaces);
    Kokkos::deep_copy(mm.muHalf, mu);
    Kokkos::deep_copy(mm.muFull, mu);
    setupFrame(mm.fh, mm.half.mesh, rho, mm.muHalf);
    setupFrame(mm.f1, mm.full.mesh, rho, mm.muFull);
    lap(rep.timeSetup);
    FrameData<Real>& fh = mm.fh;
    FrameData<Real>& f1 = mm.f1;
    const auto& mh = fh.mesh;
    const auto& m1 = f1.mesh;
    const int nfMax = std::max(mh.nFaces, m1.nFaces);
    base.reserveAtLeast(nc, nfMax);
    allocateFaces(nfMax);
    detail::movingMasses(nc, rho, m1.cellVolume, mm.volume, dV_, dM_, mL_, mR_);
    // S3: the swept flux corrected to the volume change, then F = Φ − Φᵐ_c
    ale::sweptFlux(mh, xdot_, wdot_, phiM_);
    ale::divergence(mh, phiM_, ws.b);
    detail::sweptRhs(nc, ws.b, dV_, tau);
    if (krylov::maxScaled(ws.b, DV<Real>(), nc) == Real(0)) {
      Kokkos::deep_copy(mm.psi, Real(0));
      rep.sweptZero = true;
    } else {
      krylov::PcgTest<Real> test;
      test.maxTol = Real(opt.compactTol) / tau;
      test.weight = fh.invVolume;
      fh.solver.maxIterations = opt.laplacianMaxIterations;
      rep.swept = fh.solver.solve(ws.b, mm.psi, test);
      detail::compactCorrect(mh, fh.aFace, mm.psi, phiM_);
      ale::divergence(mh, phiM_, ws.tN);
      rep.gcl = detail::gclDefect(nc, ws.tN, dV_, mh.cellVolume, tau);
      // the true measure above the bar: solve for the true residual from zero, add the correction.
      // The solver's own measure of that residual (τ|D_i − dV_i/τ|·(1/V_i), another rounding of
      // the same difference) may already pass compactTol, so a restart takes at least one step.
      krylov::PcgTest<Real> again = test;
      again.minIterations = 1;
      for (int r = 0; r < kMaxSweptRestarts && rep.gcl > opt.compactTol; ++r) {
        Kokkos::deep_copy(Kokkos::subview(ws.b, std::make_pair(0, nc)),
                          Kokkos::subview(ws.tN, std::make_pair(0, nc)));
        detail::sweptRhs(nc, ws.b, dV_, tau);
        Kokkos::deep_copy(dpsi_, Real(0));
        const krylov::SolveReport sr = fh.solver.solve(ws.b, dpsi_, again);
        rep.swept.iterations += sr.iterations;
        rep.swept.applications += sr.applications;
        rep.swept.residual = sr.residual;
        rep.swept.converged = sr.converged;
        detail::compactCorrect(mh, fh.aFace, dpsi_, phiM_);
        krylov::axpy(mm.psi, Real(1), dpsi_, nc);
        ale::divergence(mh, phiM_, ws.tN);
        rep.gcl = detail::gclDefect(nc, ws.tN, dV_, mh.cellVolume, tau);
        rep.sweptRestarts = r + 1;
      }
    }
    if (rep.sweptZero) {
      ale::divergence(mh, phiM_, ws.tN);
      rep.gcl = detail::gclDefect(nc, ws.tN, dV_, mh.cellVolume, tau);
    }
    rep.transportFlux = base.compactFluxOf(fh, ws.Uext, tau, s.phiTransport, ws.Phi);
    detail::combine(Real(1), ws.Phi, Real(-1), phiM_, ws.Phi, mh.nFaces);
    ale::divergence(mh, ws.Phi, ws.tN);
    detail::combine(Real(1), ws.tN, Real(1) / tau, dV_, ws.tN, nc);  // DF + dV/τ
    rep.divF = tau * krylov::maxScaled(ws.tN, DV<Real>(), nc) / detail::minVolume(mh);
    rep.cfl = detail::courant(mh, ws.Phi, tau);
    detail::massFlux(mh, rho, ws.Phi, ws.Fm);
    if (!opt.transport)
      Kokkos::deep_copy(ws.Fm, Real(0));
    lap(rep.timeTransportFlux);
    rep.timeSolveTransportFlux = base.lastSolveSeconds();
    // S4 on Fh
    predictor(fh, s, tau, rep);
    lap(rep.timePredictor);
    if (opt.budget) {
      base.predictorBudget(fh, s, tau, DV<Real>(), Ustar_, rep);
      rep.B1 = 0.125 * detail::weightedSquareDifference(nc, dM_, Ustar_, s.U);
      ale::divergence(mh, ws.Fm, ws.tN);
      detail::defectMomentum(nc, ws.tN, dM_, ws.ubar, tau, rep.defectMomentum);
    }
    lap(rep.timeBudget);
    // S5 on F1 (mass M¹ = ρV¹ of F1)
    base.projection(f1, s, tau, mode, Ustar_, U1_, dp_, rep);
    if (opt.timed)
      timer.reset();
    double tb = 0;
    if (opt.budget) {
      detail::combine(Real(1), U1_, Real(-1), Ustar_, ws.t3a, n);
      ale::applyMass(m1, rho, ws.t3a, ws.t3b);
      rep.B4 = -0.5 * krylov::dot(ws.t3a, ws.t3b, n);
      ale::cellGradient(m1, U1_, ws.G);
      ale::constraint(m1, U1_, ws.G, ws.sF, C1_);
      rep.B5 = tau * krylov::dot(dp_, C1_, nc);
      rep.divU1 = tau * krylov::maxScaled(C1_, f1.invVolume, nc);
      detail::combine(Real(1), U1_, Real(-1), s.U, ws.t3a, n);
      detail::combine(Real(1), U1_, Real(1), s.U, ws.t3c, n);
      ale::applyMass(m1, rho, ws.t3c, ws.t3b);
      rep.dK = 0.5 * krylov::dot(ws.t3a, ws.t3b, n) + 0.5 * detail::weightedSquare(nc, dM_, s.U);
      ale::applyMass(m1, rho, U1_, ws.t3b);
      rep.K1 = 0.5 * krylov::dot(U1_, ws.t3b, n);
      detail::componentSums(ws.t3b, nc, rep.momentum, &rep.absMomentum);
      rep.booked = rep.B1 + rep.B2 + rep.B3 + rep.B4 + rep.B5 + rep.B6 + rep.B7 + rep.B8a +
                   rep.B8b + rep.B9 + rep.Bres;
      rep.closure = rep.dK - rep.booked;
      rep.unsignedSum = rep.B1 + rep.B2 + rep.B3 + rep.B5 + rep.B6 + rep.B8b + rep.B9 + rep.Bres;
    }
    lap(tb);
    rep.timeBudget += tb;
    // S6 on F1
    rep.compactFlux = base.compactFluxOf(f1, U1_, tau, s.phiCompact, phi1_);
    rep.divPhi1 = base.maxDivergence(f1, phi1_, tau);
    lap(rep.timeCompactFlux);
    rep.timeSolveCompactFlux = base.lastSolveSeconds();
    // S8
    Kokkos::deep_copy(s.Uprev, s.U);
    Kokkos::deep_copy(s.U, U1_);
    s.hasPrevious = true;
    Kokkos::deep_copy(mm.volume, Kokkos::subview(m1.cellVolume, std::make_pair(0, nc)));
    Kokkos::deep_copy(mm.pos, x1_);
    return rep;
  }

  const DV<Real>& lastUstar() const { return Ustar_; }
  const DV<Real>& lastTransportFlux() const { return base.ws.Phi; }  ///< F on Fh (nFaces of Fh)
  const DV<Real>& lastSweptFlux() const { return phiM_; }            ///< Φᵐ_c on Fh
  const DV<Real>& lastPressureIncrement() const { return dp_; }
  const DV<Real>& lastMeshVelocity() const { return xdot_; }

 private:
  /// S4 of A18 on Fh into Ustar_ (explicit terms kept in base.ws for the budget).
  void predictor(FrameData<Real>& fr, FlowState<Real>& s, Real tau, StepReport& rep) {
    auto& ws = base.ws;
    const auto& m = fr.mesh;
    const int nc = m.nCells, n = 3 * nc;
    ale::cellGradient(m, ws.Uext, ws.G);
    ale::explicitTransport(m, ws.G, ws.Fm, ws.e6, ws.e8, ws.s6F);
    ale::pressureForce(m, s.p, ws.fp, ws.sF, ws.s9);
    // rhs = (Mⁿ + ¼ΔM − ½τ(S + Q + ζ))Uⁿ + τ(e6 + e8 + C′ᵀpⁿ)
    applyOperator(fr, ws.Fm, tau, s.U, ws.rhs, Real(-1), mR_);
    detail::addExplicit(ws.rhs, tau, ws.e6, ws.e8, ws.fp, DV<Real>(), n);
    // preconditioner: the diagonal of §4.2 with the mass M¹ − ¼ΔM
    detail::applyCellMass(nc, mL_, ws.ones, ws.massOnes);
    detail::predictorDiagonal(m, ws.massOnes, ws.Fm, fr.muFace, tau, ws.diag);
    Kokkos::deep_copy(Ustar_, ws.Uext);
    const DV<Real> Fm = ws.Fm, diag = ws.diag, mL = mL_;
    auto op = [this, &fr, Fm, tau, mL](const DV<Real>& in, const DV<Real>& out) {
      applyOperator(fr, Fm, tau, in, out, Real(1), mL);
    };
    auto prec = [diag, n](const DV<Real>& in, const DV<Real>& out) {
      detail::divideBy(in, diag, out, n);
    };
    rep.predictor = krylov::gmres<Real>(op, prec, ws.rhs, Ustar_, Real(base.opt.predictorRtol),
                                        base.opt.predictorRestart, base.opt.predictorMaxIterations,
                                        ws.gmres, n);
    rep.predictorResidual =
        rep.predictor.rhsNorm > 0 ? rep.predictor.residual / rep.predictor.rhsNorm : 0.0;
  }

  /// out = mass ⊙ x + sign · ½τ ((S + Q)x + ζx) on frame `fr` (the operator of
  /// Stepper::predictorOperator with the mass given per cell).
  void applyOperator(const FrameData<Real>& fr, const DV<Real>& Fm, Real tau, const DV<Real>& x,
                     const DV<Real>& out, Real sign, const DV<Real>& mass) {
    auto& ws = base.ws;
    const auto& m = fr.mesh;
    detail::applyCellMass(m.nCells, mass, x, out);
    ale::cellGradient(m, x, ws.G);
    ale::friction(m, x, ws.G, fr.muFace, ws.t3c, ws.sF12, ws.s9);
    detail::addTransportFriction(m, x, Fm, ws.t3c, sign * Real(0.5) * tau, out);
  }

  void allocateCells(int nc) {
    if ((int)dV_.extent(0) == nc)
      return;
    xdot_ = DV<Real>("ale.mv.xdot", 3 * nc);
    wdot_ = DV<Real>("ale.mv.wdot", nc);
    xh_ = DV<Real>("ale.mv.xh", 3 * nc);
    x1_ = DV<Real>("ale.mv.x1", 3 * nc);
    dV_ = DV<Real>("ale.mv.dV", nc);
    dM_ = DV<Real>("ale.mv.dM", nc);
    mL_ = DV<Real>("ale.mv.mL", nc);
    mR_ = DV<Real>("ale.mv.mR", nc);
    Ustar_ = DV<Real>("ale.mv.Ustar", 3 * nc);
    U1_ = DV<Real>("ale.mv.U1", 3 * nc);
    dp_ = DV<Real>("ale.mv.dp", nc);
    C1_ = DV<Real>("ale.mv.C1", nc);
    dpsi_ = DV<Real>("ale.mv.dpsi", nc);
  }
  void allocateFaces(int nf) {
    if ((int)phiM_.extent(0) >= nf)
      return;
    phiM_ = DV<Real>("ale.mv.phiM", nf + nf / 4);
    phi1_ = DV<Real>("ale.mv.phi1", nf + nf / 4);
  }

  DV<Real> xdot_, wdot_, xh_, x1_, dV_, dM_, mL_, mR_, Ustar_, U1_, dp_, C1_, dpsi_, phiM_, phi1_;
};

}  // namespace peclet::voro::ale

#endif  // PECLET_VORO_ALE_MOVING_STEPPER_HPP
