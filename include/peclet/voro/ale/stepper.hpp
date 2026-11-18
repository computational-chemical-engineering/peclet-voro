/**
 * @file ale/stepper.hpp
 * \brief Moving-grid stage 1, WO-S1.2b (docs/ale_design.md §4.1–§4.5, §4.9, §11 A7, A15): the time
 * step for ONE phase on a STATIC mesh as device code — stages, pressure solves, the energy budget
 * and the momentum with the predictor residual booked.
 *
 * The specification is the host reference `Stepper.step` of tests/python/ale_reference_stepper.py;
 * gate G1.2 (tools/ale_gates/G1_2.sh) checks this file against it operator by operator and step by
 * step. Sign convention of §A.3: C′ = D I_F, pressure force +C′ᵀp. On a static mesh there is one
 * frame, Φᵐ = 0 and V is constant. With Uⁿ, Uⁿ⁻¹, pⁿ:
 *  S1  U_ext = 1.5 Uⁿ − 0.5 Uⁿ⁻¹ (U_ext = Uⁿ at the first step).
 *  S3  Φ = I_F U_ext, compact projection φ = L_ρ⁺ DΦ, F = Φ + a ⊙ Δ_n φ, a_f = A_f/(ρ_f d_f), so
 *      D F = −(the solver's residual): max_i τ|(DF)_i|/V_i ≤ compactTol (§4.4). Fm = ρ F.
 *  S4  (M + ½τ(T + Q) + ½τζ) U* = (M − ½τ(T + Q) − ½τζ) Uⁿ + τ (e6 + e8 + C′ᵀpⁿ + B),
 *      e6, e8 the explicit transport from U_ext (ale/transport.hpp), B the body force (A7, given as
 *      a cell vector by the caller). GMRES, right-preconditioned by the diagonal of §4.2,
 *      M + ½τ Σ_f (½|Fm_f| + μ_f A_f/d_f), from U_ext.
 *  S5  mode F (Projection::Damped): μ̂ by power iteration on (C′M⁻¹C′ᵀ, L_ρ) (§4.3), ω = min(1,
 *      1/μ̂), δp = −ω L_ρ⁺ C′U* / τ; mode R (Projection::Exact, the oracle): δp = −(C′M⁻¹C′ᵀ)⁺ C′U*
 * / τ by PCG preconditioned with one V-cycle of L_ρ. U^{n+1} = U* + τ M⁻¹C′ᵀδp, p^{n+1} = pⁿ + δp.
 *  S6  Φ₁ = I_F U^{n+1} with its compact projection (the divergence contract of §4.4, reported).
 * Energy budget, K = ½UᵀMU, ū = ½(U* + Uⁿ) (B1 = 0 on a static mesh):
 *  B2 = −τ ūᵀTū (zero to rounding: T is skew), B3 = τ ūᵀC′ᵀpⁿ, B4 = −½ΔUᵀMΔU (ΔU = U^{n+1} − U*),
 *  B5 = τ δpᵀC′U^{n+1}, B6 = τ ūᵀe6, B7 = −τ ūᵀζū, B8a = −τ ūᵀQū, B8b = τ ūᵀe8, B9 = τ ūᵀB,
 *  Bres = −ūᵀ(rhs − A U*) (the work of the predictor residual, §4.9);
 *  closure = (K^{n+1} − Kⁿ) − Σ B. The predictor residual's momentum Σ_i (rhs − A U*)_i is
 *  reported per step (§11 A15: P* = Pⁿ + τ Σ forces − that), for the caller to book.
 *
 * The mass is ONE isolated pair, `applyMass` / `applyInverseMass` (lumped, M = diag(ρ_iV_i)); every
 * mass of this file goes through it (the predictor's diagonal preconditioner through applyMass of
 * the ones vector). Another mass replaces those two functions only (§4.3).
 *
 * Structure: the stages are member functions of (FrameData, FlowState) so that the moving-mesh
 * work order wraps them with two frames (§4.1, last paragraph). Per-step work runs on the device
 * (§2.1): kernels in the patterns of §2.2 (no atomics), reductions are `parallel_reduce` (K5),
 * the scalars of the Krylov recurrences and of the budget are host scalars. Scratch is owned by the
 * Stepper (`Workspace`), allocated on the first step of a frame size and reused.
 *
 * Single rank, periodic box (the exchange hooks of ale/operators.hpp stay unset; WO-S1.4).
 */
#ifndef PECLET_VORO_ALE_STEPPER_HPP
#define PECLET_VORO_ALE_STEPPER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <limits>

#include "peclet/core/solver/graph_amg_setup.hpp"
#include "peclet/voro/ale/krylov.hpp"
#include "peclet/voro/ale/laplacian_solver.hpp"
#include "peclet/voro/ale/operators.hpp"
#include "peclet/voro/ale/transport.hpp"

namespace peclet::voro::ale {

// ================================================================================ mass (isolated)
/// out = M x, M = diag(ρ_iV_i) per component (x, out 3·nCombined; owned cells written).
template <class Real>
void applyMass(const fv::FaceMesh<Real>& m, const DV<Real>& rho, const DV<Real>& x,
               const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.mass", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        const Real mi = rho(i) * m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = mi * x(3 * i + c);
      });
}

/// out = M⁻¹ x.
template <class Real>
void applyInverseMass(const fv::FaceMesh<Real>& m, const DV<Real>& rho, const DV<Real>& x,
                      const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.invMass", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        const Real mi = rho(i) * m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = x(3 * i + c) / mi;
      });
}

// ================================================================================ options, state
/// The projection of S5: mode F (damped, the default) or mode R (exact; the oracle, §4.3).
enum class Projection { Damped, Exact };

/// Tolerances and limits of the step (§4.9 defaults; gate G1.2 runs the comparison with the
/// reference at tighter values).
struct StepOptions {
  double predictorRtol = 1e-12;  ///< S4, relative to |rhs|₂ (true residual)
  int predictorRestart = 30;     ///< GMRES restart length
  int predictorMaxIterations = 200;
  double laplacianRtol = 1e-8;  ///< S5 mode F solve, relative
  double muRtol = 1e-2;         ///< the μ̂ solves (§4.3)
  int muColdIterations = 20;    ///< power iterations at the first step (§4.3)
  double compactTol = 1e-14;    ///< S3, S6: max_i τ|(DΦ)_i|/V_i (§4.4)
  double exactRtol = 1e-13;  ///< mode R, relative (floor: rounding of C′U*, see exactProjection)
  int exactMaxIterations = 5000;
  int laplacianMaxIterations = 1000;
  bool budget = true;     ///< evaluate the energy budget (B-terms)
  bool timed = false;     ///< fence and time the stages into StepReport::time*
  bool transport = true;  ///< false: the Stokes step (mass flux of S4 set to zero; the Courant
                          ///< number of S3 is still reported)
};

/// The operators of one frame: mesh, density per cell, viscosity per face, the compact-projection
/// coefficient a_f = A_f/(ρ_f d_f), 1/V, the matrix L_ρ and its solver (setup per frame).
template <class Real>
struct FrameData {
  fv::FaceMesh<Real> mesh;
  DV<Real> rho;        ///< nCombined
  DV<Real> muFace;     ///< nFaces: μ_f
  DV<Real> aFace;      ///< nFaces: A_f/(ρ_f d_f), the coefficient of L_ρ
  DV<Real> invVolume;  ///< nCombined: 1/V_i (weights of the cell-wise tests)
  LaplacianMatrix<Real> L;
  LaplacianSolver<Real> solver;
  double setupSeconds = 0;  ///< assembly + AMG setup (fenced)
};

/// The state carried from step to step: velocity, the previous velocity, pressure, and the solver
/// state that is not the solution (warm starts, the μ̂ vector).
template <class Real>
struct FlowState {
  DV<Real> U, Uprev, p;  ///< 3·nCombined, 3·nCombined, nCombined
  bool hasPrevious = false;
  DV<Real> muVector;  ///< the persistent power-iteration vector (§4.3)
  double muHat = 0;
  bool hasMu = false;
  DV<Real> phiTransport, phiCompact, projection;  ///< warm starts of S3, S6, S5 (nCombined)
  void allocate(int nCombined) {
    U = DV<Real>("ale.U", 3 * nCombined);
    Uprev = DV<Real>("ale.Uprev", 3 * nCombined);
    p = DV<Real>("ale.p", nCombined);
    muVector = DV<Real>("ale.muv", nCombined);
    phiTransport = DV<Real>("ale.phiT", nCombined);
    phiCompact = DV<Real>("ale.phiC", nCombined);
    projection = DV<Real>("ale.proj", nCombined);
  }
};

/// What one step did: solver reports, μ̂ and ω, the divergence contracts, the budget terms.
struct StepReport {
  krylov::SolveReport predictor, transportFlux, projection, compactFlux, mu;
  int muSolves = 0, muIterations = 0;
  double omega = 1, muHat = 0, lambda = 0;
  double divF = 0;               ///< τ max|DF|/min V (S3, as the reference)
  double cfl = 0;                ///< τ max_i Σ_f max(s_if F_f, 0)/V_i
  double divPhi1 = 0;            ///< τ max_i |(DΦ₁)_i|/V_i (S6)
  double divU1 = 0;              ///< τ max_i |(C′U^{n+1})_i|/V_i
  double predictorResidual = 0;  ///< |rhs − A U*|/|rhs|
  // budget (B1 = 0 on a static mesh)
  double B2 = 0, B3 = 0, B4 = 0, B5 = 0, B6 = 0, B7 = 0, B8a = 0, B8b = 0, B9 = 0, Bres = 0;
  double dK = 0, K1 = 0, booked = 0, closure = 0, unsignedSum = 0;
  double residualMomentum[3] = {0, 0, 0};  ///< Σ_i (rhs − A U*)_i per component
  double momentum[3] = {0, 0, 0};          ///< Σ_i (M U^{n+1})_i
  double absMomentum = 0;                  ///< Σ_i |M U^{n+1}|
  // stage times (StepOptions::timed): S3 (flux map + solve), S4 (explicit transport + GMRES), μ̂,
  // S5 (constraint + solve), S6 (flux map + solve), budget
  double timeTransportFlux = 0, timePredictor = 0, timeMu = 0, timeProjection = 0,
         timeCompactFlux = 0, timeBudget = 0;
  double timeSolveTransportFlux = 0, timeSolveCompactFlux = 0;  ///< the L_ρ solves within S3, S6
};

// ================================================================================ kernels
namespace detail {

/// a_f = A_f/(ρ_f d_f), 1/ρ_f = (h_Bf/ρ_A + h_Af/ρ_B)/d_f, in the operation order of
/// `assembleLaplacian` (bitwise its −coef). K1.
template <class Real>
void laplacianCoefficient(const fv::FaceMesh<Real>& m, const DV<Real>& rho, const DV<Real>& a) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.aFace", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int A = m.faceCellA(f), B = m.faceCellB(f);
        const Real d = m.faceDist(f);
        const Real invRho = (m.faceHb(f) / rho(A) + m.faceHa(f) / rho(B)) / d;
        a(f) = m.faceArea(f) * invRho / d;
      });
}

/// Φ_f ← Φ_f + a_f (φ_B − φ_A): the compact correction, D of the result = DΦ − L_ρφ. K1.
template <class Real>
void compactCorrect(const fv::FaceMesh<Real>& m, const DV<Real>& a, const DV<Real>& phi,
                    const DV<Real>& Phi) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.compact", Kokkos::RangePolicy<Exec>(0, m.nFaces),
      KOKKOS_LAMBDA(const int f) { Phi(f) += a(f) * (phi(m.faceCellB(f)) - phi(m.faceCellA(f))); });
}

/// Fm_f = ρ_A F_f (one phase, or a like face: ρ_A = ρ_B). K1.
template <class Real>
void massFlux(const fv::FaceMesh<Real>& m, const DV<Real>& rho, const DV<Real>& F,
              const DV<Real>& Fm) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.massFlux", Kokkos::RangePolicy<Exec>(0, m.nFaces),
      KOKKOS_LAMBDA(const int f) { Fm(f) = rho(m.faceCellA(f)) * F(f); });
}

/// 1/V_i. Per cell.
template <class Real>
void inverseVolume(const fv::FaceMesh<Real>& m, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.invV", Kokkos::RangePolicy<Exec>(0, m.nCells),
      KOKKOS_LAMBDA(const int i) { out(i) = Real(1) / m.cellVolume(i); });
}

/// out_i += scale · (fric_i + ((T + Q)x)_i): the transport and friction part of the predictor
/// operator added to its mass part. K2.
template <class Real>
void addTransportFriction(const fv::FaceMesh<Real>& m, const DV<Real>& x, const DV<Real>& Fm,
                          const DV<Real>& fric, Real scale, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.pred.add", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const bool isA = m.cellFaceSign(q) > 0;
          const int o = isA ? m.faceCellB(f) : m.faceCellA(f);
          const Real fm = Fm(f);
          const Real h = Real(0.5) * m.cellFaceSign(q) * fm;
          const Real w = Real(0.5) * Kokkos::fabs(fm);
          for (int c = 0; c < 3; ++c)
            acc[c] += h * x(3 * o + c) + w * (x(3 * i + c) - x(3 * o + c));
        }
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) += scale * (fric(3 * i + c) + acc[c]);
      });
}

/// The diagonal preconditioner of §4.2: out_i (3 components) = (M 1)_i + ½τ Σ_{f∋i} (½|Fm_f| +
/// μ_f A_f/d_f), with `massOnes` = applyMass of the ones vector. K2.
template <class Real>
void predictorDiagonal(const fv::FaceMesh<Real>& m, const DV<Real>& massOnes, const DV<Real>& Fm,
                       const DV<Real>& muFace, Real tau, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.pred.diag", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real s = 0;
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          s += Real(0.5) * Kokkos::fabs(Fm(f)) + muFace(f) * m.faceArea(f) / m.faceDist(f);
        }
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = massOnes(3 * i + c) + Real(0.5) * tau * s;
      });
}

/// out = x / diag (elementwise over [0, n)).
template <class Real>
void divideBy(const DV<Real>& x, const DV<Real>& diag, const DV<Real>& out, int n) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.divideBy", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i) { out(i) = x(i) / diag(i); });
}

/// out = a·x + b·y over [0, n).
template <class Real>
void combine(Real a, const DV<Real>& x, Real b, const DV<Real>& y, const DV<Real>& out, int n) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.combine", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i) { out(i) = a * x(i) + b * y(i); });
}

/// rhs += τ (e6 + e8 + fp + body) over [0, n) (body may be empty).
template <class Real>
void addExplicit(const DV<Real>& rhs, Real tau, const DV<Real>& e6, const DV<Real>& e8,
                 const DV<Real>& fp, const DV<Real>& body, int n) {
  using Exec = peclet::core::ExecSpace;
  const bool hasB = body.extent(0) > 0;
  Kokkos::parallel_for(
      "ale.rhs.explicit", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const int i) {
        rhs(i) += tau * (e6(i) + e8(i) + fp(i) + (hasB ? body(i) : Real(0)));
      });
}

/// τ max_i Σ_f max(s_if F_f, 0)/V_i. K2 + K5.
template <class Real>
double courant(const fv::FaceMesh<Real>& m, const DV<Real>& F, double tau) {
  using Exec = peclet::core::ExecSpace;
  Real mx = 0;
  Kokkos::parallel_reduce(
      "ale.courant", Kokkos::RangePolicy<Exec>(0, m.nCells),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        Real s = 0;
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const Real out = m.cellFaceSign(q) * F(m.cellFace(q));
          s += out > Real(0) ? out : Real(0);
        }
        s /= m.cellVolume(i);
        acc = s > acc ? s : acc;
      },
      Kokkos::Max<Real>(mx));
  return tau * mx;
}

/// min_i V_i.
template <class Real>
double minVolume(const fv::FaceMesh<Real>& m) {
  using Exec = peclet::core::ExecSpace;
  Real mn = 0;
  Kokkos::parallel_reduce(
      "ale.minV", Kokkos::RangePolicy<Exec>(0, m.nCells),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        const Real v = m.cellVolume(i);
        acc = v < acc ? v : acc;
      },
      Kokkos::Min<Real>(mn));
  return mn;
}

/// Σ_i x(3i + c) per component and Σ |x| over the owned cells. K5.
template <class Real>
void componentSums(const DV<Real>& x, int nCells, double out[3], double* absSum) {
  using Exec = peclet::core::ExecSpace;
  Real s0 = 0, s1 = 0, s2 = 0, sa = 0;
  Kokkos::parallel_reduce(
      "ale.compSums", Kokkos::RangePolicy<Exec>(0, nCells),
      KOKKOS_LAMBDA(const int i, Real& a0, Real& a1, Real& a2, Real& aa) {
        const Real x0 = x(3 * i), x1 = x(3 * i + 1), x2 = x(3 * i + 2);
        a0 += x0;
        a1 += x1;
        a2 += x2;
        aa += Kokkos::fabs(x0) + Kokkos::fabs(x1) + Kokkos::fabs(x2);
      },
      s0, s1, s2, sa);
  out[0] = s0;
  out[1] = s1;
  out[2] = s2;
  if (absSum)
    *absSum = sa;
}

/// The hash-seeded ±1 vector of §4.3 (splitmix64 of the gid; gid = slot on one rank), mean removed
/// by the caller.
template <class Real>
void hashSeed(int n, const DV<Real>& v) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.seed", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const int i) {
        v(i) = (peclet::core::solver::amg_setup::splitmix64((std::uint64_t)i) & 1u) ? Real(1)
                                                                                    : Real(-1);
      });
}

/// An upper bound of |C′| |U| per cell for the rounding floor of mode R: every term of D I_F U
/// taken in absolute value (|G| from |U_o| + |U_i|). Face bound into `faceScratch` (nFaces), the
/// absolute gradient into `G9` (9·nCombined). K2, K1, K2.
template <class Real>
void constraintBound(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& G9,
                     const DV<Real>& faceScratch, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.absGrad", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real g[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const bool isA = m.cellFaceSign(q) > 0;
          const int o = isA ? m.faceCellB(f) : m.faceCellA(f);
          const Real w = m.faceArea(f) / m.faceDist(f);
          for (int c = 0; c < 3; ++c) {
            const Real du = w * (Kokkos::fabs(U(3 * o + c)) + Kokkos::fabs(U(3 * i + c)));
            for (int k = 0; k < 3; ++k) {
              const Real r = isA ? m.centroid(f, k) : m.centroid(f, k) - m.conn(f, k);
              g[3 * c + k] += du * Kokkos::fabs(r);
            }
          }
        }
        const Real vol = m.cellVolume(i);
        for (int j = 0; j < 9; ++j)
          G9(9 * i + j) = g[j] / vol;
      });
  Kokkos::parallel_for(
      "ale.absIF", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        const Real ha = m.faceHa(f), hb = m.faceHb(f);
        Real s = 0, corr = 0;
        for (int c = 0; c < 3; ++c) {
          const Real n = Kokkos::fabs(m.normal(f, c));
          s += n * (Kokkos::fabs(hb) * Kokkos::fabs(U(3 * a + c)) +
                    Kokkos::fabs(ha) * Kokkos::fabs(U(3 * b + c)));
          for (int k = 0; k < 3; ++k) {
            const Real qk = Kokkos::fabs(m.centroid(f, k) - ha * m.normal(f, k));
            corr += n * (Real(0.5) * (G9(9 * a + 3 * c + k) + G9(9 * b + 3 * c + k))) * qk;
          }
        }
        faceScratch(f) = m.faceArea(f) * (s / m.faceDist(f) + corr);
      });
  Kokkos::parallel_for(
      "ale.absD", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc = 0;
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q)
          acc += faceScratch(m.cellFace(q));
        out(i) = acc;
      });
}

}  // namespace detail

/// Set up the operators of a frame: a_f, 1/V, L_ρ assembled and its multigrid built (on the
/// device). `rho` per combined cell, `muFace` per face; both are referenced.
template <class Real>
void setupFrame(FrameData<Real>& fr, const fv::FaceMesh<Real>& mesh, const DV<Real>& rho,
                const DV<Real>& muFace) {
  Kokkos::fence();
  Kokkos::Timer timer;
  fr.mesh = mesh;
  fr.rho = rho;
  fr.muFace = muFace;
  fr.aFace = DV<Real>("ale.aFace", mesh.nFaces);
  fr.invVolume = DV<Real>("ale.invV", mesh.nCombined);
  detail::laplacianCoefficient(mesh, rho, fr.aFace);
  detail::inverseVolume(mesh, fr.invVolume);
  ale::assembleLaplacian(mesh, rho, fr.L);
  fr.solver.setup(fr.L);
  Kokkos::fence();
  fr.setupSeconds = timer.seconds();
}

// ================================================================================ the stepper
template <class Real>
class Stepper {
 public:
  using Exec = peclet::core::ExecSpace;
  StepOptions opt;

  explicit Stepper(const StepOptions& o = {}) : opt(o) {}

  /// Scratch owned by the stepper, sized by the frame.
  struct Workspace {
    int nCombined = -1, nFaces = -1;
    DV<Real> Uext, rhs, e6, e8, fp, ubar, t3a, t3b, t3c, diag, ones, massOnes;  // 3·nCombined
    DV<Real> b, x, tN;                                                          // nCombined
    DV<Real> Phi, Fm, sF, s6F;                                                  // faces
    DV<Real> G, s9;                                                             // 9·nCombined
    DV<Real> sF12;                                                              // 12·nFaces
    krylov::GmresWork<Real> gmres;
    krylov::PcgWork<Real> pcg;
  };
  Workspace ws;

  void reserve(const fv::FaceMesh<Real>& m) {
    if (ws.nCombined == m.nCombined && ws.nFaces == m.nFaces)
      return;
    const int NC = m.nCombined, F = m.nFaces;
    ws.nCombined = NC;
    ws.nFaces = F;
    auto v3 = [NC](const char* s) { return DV<Real>(s, 3 * NC); };
    ws.Uext = v3("ale.ws.Uext");
    ws.rhs = v3("ale.ws.rhs");
    ws.e6 = v3("ale.ws.e6");
    ws.e8 = v3("ale.ws.e8");
    ws.fp = v3("ale.ws.fp");
    ws.ubar = v3("ale.ws.ubar");
    ws.t3a = v3("ale.ws.t3a");
    ws.t3b = v3("ale.ws.t3b");
    ws.t3c = v3("ale.ws.t3c");
    ws.diag = v3("ale.ws.diag");
    ws.ones = v3("ale.ws.ones");
    ws.massOnes = v3("ale.ws.massOnes");
    Kokkos::deep_copy(ws.ones, Real(1));
    ws.b = DV<Real>("ale.ws.b", NC);
    ws.x = DV<Real>("ale.ws.x", NC);
    ws.tN = DV<Real>("ale.ws.tN", NC);
    ws.Phi = DV<Real>("ale.ws.Phi", F);
    ws.Fm = DV<Real>("ale.ws.Fm", F);
    ws.sF = DV<Real>("ale.ws.sF", F);
    ws.s6F = DV<Real>("ale.ws.s6F", 6 * F);
    ws.G = DV<Real>("ale.ws.G", 9 * NC);
    ws.s9 = DV<Real>("ale.ws.s9", 9 * NC);
    ws.sF12 = DV<Real>("ale.ws.sF12", 12 * F);
  }

  // ---------------------------------------------------------------- operators (on a frame)
  /// out = M x + sign · ½τ ((T + Q)x + ζx): the predictor operator (sign = +1) or the explicit
  /// half of the Crank–Nicolson pair (sign = −1). Uses ws.G, ws.t3c, ws.sF12, ws.s9.
  void predictorOperator(const FrameData<Real>& fr, const DV<Real>& Fm, Real tau, const DV<Real>& x,
                         const DV<Real>& out, Real sign) {
    const auto& m = fr.mesh;
    ale::applyMass(m, fr.rho, x, out);
    ale::cellGradient(m, x, ws.G);
    ale::friction(m, x, ws.G, fr.muFace, ws.t3c, ws.sF12, ws.s9);
    detail::addTransportFriction(m, x, Fm, ws.t3c, sign * Real(0.5) * tau, out);
  }

  /// out = C′M⁻¹C′ᵀ q (the Schur operator of mode R and of μ̂). Uses ws.t3b, ws.t3c, ws.G, ws.sF,
  /// ws.s9.
  void schur(const FrameData<Real>& fr, const DV<Real>& q, const DV<Real>& out) {
    const auto& m = fr.mesh;
    ale::pressureForce(m, q, ws.t3b, ws.sF, ws.s9);
    ale::applyInverseMass(m, fr.rho, ws.t3b, ws.t3c);
    ale::cellGradient(m, ws.t3c, ws.G);
    ale::constraint(m, ws.t3c, ws.G, ws.sF, out);
  }

  /// Φ = compact projection of I_F U into `Phi` (nFaces), φ warm-started from `phiWarm`: the solve
  /// stops when max_i τ|(DΦ)_i|/V_i ≤ compactTol (§4.4). Uses ws.G, ws.b.
  krylov::SolveReport compactFluxOf(FrameData<Real>& fr, const DV<Real>& U, Real tau,
                                    const DV<Real>& phiWarm, const DV<Real>& Phi) {
    const auto& m = fr.mesh;
    ale::cellGradient(m, U, ws.G);
    ale::fluxMap(m, U, ws.G, Phi);
    ale::divergence(m, Phi, ws.b);
    krylov::PcgTest<Real> test;
    test.maxTol = Real(opt.compactTol) / tau;
    test.weight = fr.invVolume;
    fr.solver.maxIterations = opt.laplacianMaxIterations;
    Kokkos::Timer timer;
    if (opt.timed)
      Kokkos::fence(), timer.reset();
    const auto rep = fr.solver.solve(ws.b, phiWarm, test);
    if (opt.timed)
      Kokkos::fence(), lastSolveSeconds_ = timer.seconds();
    detail::compactCorrect(m, fr.aFace, phiWarm, Phi);
    return rep;
  }

  /// τ max_i |(DΦ)_i|/V_i. Uses ws.tN.
  double maxDivergence(const FrameData<Real>& fr, const DV<Real>& Phi, double tau) {
    ale::divergence(fr.mesh, Phi, ws.tN);
    return tau * krylov::maxScaled(ws.tN, fr.invVolume, fr.mesh.nCells);
  }

  // ---------------------------------------------------------------- μ̂ (§4.3)
  /// `nit` power iterations v ← L_ρ⁺(S v)/|·|_L (each solve from zero at relative `rtol`), then the
  /// Rayleigh quotient vᵀSv / vᵀL_ρv (a lower bound of λ_max(S, L_ρ) for any v). Uses ws.b, ws.tN.
  double powerIteration(FrameData<Real>& fr, const DV<Real>& v, int nit, double rtol,
                        StepReport* rep = nullptr) {
    const int n = fr.mesh.nCells;
    krylov::PcgTest<Real> test;
    test.rtol = Real(rtol);
    fr.solver.maxIterations = opt.laplacianMaxIterations;
    for (int k = 0; k < nit; ++k) {
      schur(fr, v, ws.b);
      Kokkos::deep_copy(v, Real(0));
      const auto r = fr.solver.solve(ws.b, v, test);
      if (rep) {
        rep->mu = r;
        ++rep->muSolves;
        rep->muIterations += r.iterations;
      }
      fr.solver.apply(v, ws.tN);
      const Real vl = krylov::dot(v, ws.tN, n);
      krylov::scale(v, Real(1) / std::sqrt(vl), n);
    }
    schur(fr, v, ws.b);
    const Real vs = krylov::dot(v, ws.b, n);
    fr.solver.apply(v, ws.tN);
    const Real vl = krylov::dot(v, ws.tN, n);
    return vs / vl;
  }

  /// The hash-seeded ±1 start vector of §4.3, mean removed.
  void muSeed(const FrameData<Real>& fr, const DV<Real>& v) {
    detail::hashSeed(fr.mesh.nCells, v);
    krylov::removeMean(v, fr.mesh.nCells);
  }

  /// λ_max(C′M⁻¹C′ᵀ, L_ρ) to the accuracy of `nit` iterations with solves at `rtol` from the seed
  /// (gate G1.2 R3: 200 iterations at 1e-12). `v` is scratch (nCombined).
  double muExact(FrameData<Real>& fr, const DV<Real>& v, int nit = 200, double rtol = 1e-12) {
    muSeed(fr, v);
    return powerIteration(fr, v, nit, rtol);
  }

  // ---------------------------------------------------------------- mode R (the oracle)
  /// δp = −(C′M⁻¹C′ᵀ)⁺ C′U/τ by PCG on the Schur operator, preconditioned by one V-cycle of L_ρ,
  /// from zero, to |r| ≤ max(exactRtol |b|, 4ε |C′||U|/τ) (the floor: the rounding level of C′U,
  /// bounded from above by `detail::constraintBound`). `dp` (nCombined) receives δp. Uses ws.b,
  /// ws.tN, ws.G, ws.sF, ws.s9, ws.t3b, ws.t3c.
  krylov::SolveReport exactProjection(FrameData<Real>& fr, const DV<Real>& U, Real tau,
                                      const DV<Real>& dp) {
    const auto& m = fr.mesh;
    const int n = m.nCells;
    ale::cellGradient(m, U, ws.G);
    ale::constraint(m, U, ws.G, ws.sF, ws.b);
    krylov::scale(ws.b, Real(1) / tau, n);
    detail::constraintBound(m, U, ws.s9, ws.sF, ws.tN);
    const Real bound = krylov::norm2(ws.tN, n);
    krylov::PcgTest<Real> test;
    test.rtol = Real(opt.exactRtol);
    test.atol = Real(4) * std::numeric_limits<Real>::epsilon() * bound / tau;
    // b stays in ws.b: the Schur apply writes only its `out` (the PCG's own work vector) and the
    // scratch ws.t3b, ws.t3c, ws.G, ws.sF, ws.s9
    Kokkos::deep_copy(dp, Real(0));
    auto op = [this, &fr](const DV<Real>& in, const DV<Real>& out) { schur(fr, in, out); };
    auto prec = [&fr](const DV<Real>& in, const DV<Real>& out) { fr.solver.precondition(in, out); };
    const auto rep =
        krylov::pcg<Real>(op, prec, ws.b, dp, test, true, opt.exactMaxIterations, ws.pcg, n);
    krylov::scale(dp, Real(-1), n);
    return rep;
  }

  // ---------------------------------------------------------------- the stages
  /// S1: U_ext = 1.5 Uⁿ − 0.5 Uⁿ⁻¹ (Uⁿ at the first step) into ws.Uext.
  void extrapolate(const FrameData<Real>& fr, const FlowState<Real>& s) {
    const int n = 3 * fr.mesh.nCells;
    if (s.hasPrevious)
      detail::combine(Real(1.5), s.U, Real(-0.5), s.Uprev, ws.Uext, n);
    else
      Kokkos::deep_copy(ws.Uext, s.U);
  }

  /// S3: the transport flux F = compact projection of I_F U_ext into ws.Phi and the mass flux
  /// Fm = ρF into ws.Fm.
  void transportFlux(FrameData<Real>& fr, FlowState<Real>& s, Real tau, StepReport& rep) {
    rep.transportFlux = compactFluxOf(fr, ws.Uext, tau, s.phiTransport, ws.Phi);
    ale::divergence(fr.mesh, ws.Phi, ws.tN);
    rep.divF =
        tau * krylov::maxScaled(ws.tN, DV<Real>(), fr.mesh.nCells) / detail::minVolume(fr.mesh);
    rep.cfl = detail::courant(fr.mesh, ws.Phi, tau);
    detail::massFlux(fr.mesh, fr.rho, ws.Phi, ws.Fm);
    if (!opt.transport)
      Kokkos::deep_copy(ws.Fm, Real(0));
  }

  /// S4: the predictor into `Ustar`, the explicit terms kept in ws (e6, e8, fp) for the budget.
  void predictor(FrameData<Real>& fr, FlowState<Real>& s, Real tau, const DV<Real>& body,
                 const DV<Real>& Ustar, StepReport& rep) {
    const auto& m = fr.mesh;
    const int n = 3 * m.nCells;
    // explicit terms from U_ext and pⁿ
    ale::cellGradient(m, ws.Uext, ws.G);
    ale::explicitTransport(m, ws.G, ws.Fm, ws.e6, ws.e8, ws.s6F);
    ale::pressureForce(m, s.p, ws.fp, ws.sF, ws.s9);
    // rhs = (M − ½τ(T + Q + ζ))Uⁿ + τ(e6 + e8 + C′ᵀpⁿ + B)
    predictorOperator(fr, ws.Fm, tau, s.U, ws.rhs, Real(-1));
    detail::addExplicit(ws.rhs, tau, ws.e6, ws.e8, ws.fp, body, n);
    // preconditioner: diagonal of §4.2
    ale::applyMass(m, fr.rho, ws.ones, ws.massOnes);
    detail::predictorDiagonal(m, ws.massOnes, ws.Fm, fr.muFace, tau, ws.diag);
    Kokkos::deep_copy(Ustar, ws.Uext);
    const DV<Real> Fm = ws.Fm, diag = ws.diag;
    auto op = [this, &fr, Fm, tau](const DV<Real>& in, const DV<Real>& out) {
      predictorOperator(fr, Fm, tau, in, out, Real(1));
    };
    auto prec = [diag, n](const DV<Real>& in, const DV<Real>& out) {
      detail::divideBy(in, diag, out, n);
    };
    rep.predictor =
        krylov::gmres<Real>(op, prec, ws.rhs, Ustar, Real(opt.predictorRtol), opt.predictorRestart,
                            opt.predictorMaxIterations, ws.gmres, n);
    rep.predictorResidual =
        rep.predictor.rhsNorm > 0 ? rep.predictor.residual / rep.predictor.rhsNorm : 0.0;
  }

  /// Budget terms of S4 (B2, B3, B6, B7, B8a, B8b, B9, Bres) and the residual's momentum; needs
  /// ws.e6, ws.e8, ws.fp, ws.Fm and the GMRES residual from `predictor`.
  void predictorBudget(FrameData<Real>& fr, FlowState<Real>& s, Real tau, const DV<Real>& body,
                       const DV<Real>& Ustar, StepReport& rep) {
    const auto& m = fr.mesh;
    const int n = 3 * m.nCells;
    const DV<Real> res = ws.gmres.r;
    detail::combine(Real(0.5), Ustar, Real(0.5), s.U, ws.ubar, n);
    ale::transport(m, ws.ubar, ws.Fm, ws.t3a);
    rep.B2 = -tau * krylov::dot(ws.ubar, ws.t3a, n);
    rep.B3 = tau * krylov::dot(ws.ubar, ws.fp, n);
    rep.B6 = tau * krylov::dot(ws.ubar, ws.e6, n);
    ale::cellGradient(m, ws.ubar, ws.G);
    ale::friction(m, ws.ubar, ws.G, fr.muFace, ws.t3a, ws.sF12, ws.s9);
    rep.B7 = -tau * krylov::dot(ws.ubar, ws.t3a, n);
    ale::upwind(m, ws.ubar, ws.Fm, ws.t3a);
    rep.B8a = -tau * krylov::dot(ws.ubar, ws.t3a, n);
    rep.B8b = tau * krylov::dot(ws.ubar, ws.e8, n);
    rep.B9 = body.extent(0) > 0 ? tau * krylov::dot(ws.ubar, body, n) : 0.0;
    rep.Bres = -krylov::dot(ws.ubar, res, n);
    detail::componentSums(res, m.nCells, rep.residualMomentum, nullptr);
  }

  /// S5: δp into `dp` (nCombined), U^{n+1} into `U1`, p^{n+1} into s.p (updated in place).
  void projection(FrameData<Real>& fr, FlowState<Real>& s, Real tau, Projection mode,
                  const DV<Real>& Ustar, const DV<Real>& U1, const DV<Real>& dp, StepReport& rep) {
    const auto& m = fr.mesh;
    const int nc = m.nCells, n = 3 * nc;
    Kokkos::Timer timer;
    if (mode == Projection::Exact) {
      rep.projection = exactProjection(fr, Ustar, tau, dp);
      rep.omega = 1;
    } else {
      // μ̂ (§4.3)
      if (opt.timed)
        Kokkos::fence(), timer.reset();
      if (!s.hasMu) {
        muSeed(fr, s.muVector);
        s.muHat = powerIteration(fr, s.muVector, opt.muColdIterations, opt.muRtol, &rep);
        rep.lambda = s.muHat;
        s.hasMu = true;
      } else {
        rep.lambda = powerIteration(fr, s.muVector, 1, opt.muRtol, &rep);
        s.muHat = std::max(rep.lambda, 0.9 * s.muHat);
      }
      rep.muHat = s.muHat;
      rep.omega = std::min(1.0, 1.0 / s.muHat);
      if (opt.timed)
        Kokkos::fence(), rep.timeMu = timer.seconds(), timer.reset();
      // δp = −ω L_ρ⁺ C′U* / τ, the solve warm-started from the previous step's
      ale::cellGradient(m, Ustar, ws.G);
      ale::constraint(m, Ustar, ws.G, ws.sF, ws.b);
      krylov::scale(ws.b, Real(1) / tau, nc);
      krylov::PcgTest<Real> test;
      test.rtol = Real(opt.laplacianRtol);
      fr.solver.maxIterations = opt.laplacianMaxIterations;
      rep.projection = fr.solver.solve(ws.b, s.projection, test);
      detail::combine(Real(-rep.omega), s.projection, Real(0), s.projection, dp, nc);
    }
    if (opt.timed)
      Kokkos::fence(), rep.timeProjection = timer.seconds();
    // U^{n+1} = U* + τ M⁻¹C′ᵀδp; p^{n+1} = pⁿ + δp
    ale::pressureForce(m, dp, ws.t3a, ws.sF, ws.s9);
    ale::applyInverseMass(m, fr.rho, ws.t3a, ws.t3b);
    detail::combine(Real(1), Ustar, tau, ws.t3b, U1, n);
    krylov::axpy(s.p, Real(1), dp, nc);
  }

  /// One step of §4.1 on a static frame: S1–S6 and the budget. `body` (3·nCombined, may be empty)
  /// is the body force B of amendment A7 as a cell vector. On return s.U = U^{n+1}, s.Uprev = Uⁿ,
  /// s.p = p^{n+1}; `Ustar`, `F` (= ws.Phi) and `dp` stay readable in the workspace until the next
  /// step (`lastUstar()`, `lastTransportFlux()`, `lastPressureIncrement()`).
  StepReport step(FrameData<Real>& fr, FlowState<Real>& s, Real tau, Projection mode,
                  const DV<Real>& body = DV<Real>()) {
    reserve(fr.mesh);
    if ((int)Ustar_.extent(0) != 3 * fr.mesh.nCombined) {
      Ustar_ = DV<Real>("ale.Ustar", 3 * fr.mesh.nCombined);
      U1_ = DV<Real>("ale.U1", 3 * fr.mesh.nCombined);
      dp_ = DV<Real>("ale.dp", fr.mesh.nCombined);
      Phi1_ = DV<Real>("ale.Phi1", fr.mesh.nFaces);
      C1_ = DV<Real>("ale.C1", fr.mesh.nCombined);
    }
    const auto& m = fr.mesh;
    const int nc = m.nCells, n = 3 * nc;
    StepReport rep;
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
    // S1, S3
    extrapolate(fr, s);
    transportFlux(fr, s, tau, rep);
    lap(rep.timeTransportFlux);
    rep.timeSolveTransportFlux = lastSolveSeconds_;
    // S4
    predictor(fr, s, tau, body, Ustar_, rep);
    lap(rep.timePredictor);
    if (opt.budget)
      predictorBudget(fr, s, tau, body, Ustar_, rep);
    lap(rep.timeBudget);
    // S5 (pⁿ is needed by nothing after S4's explicit force)
    projection(fr, s, tau, mode, Ustar_, U1_, dp_, rep);
    if (opt.timed)
      timer.reset();
    double tb = 0;
    if (opt.budget) {
      // B4, B5, ΔK, K^{n+1}
      detail::combine(Real(1), U1_, Real(-1), Ustar_, ws.t3a, n);
      ale::applyMass(m, fr.rho, ws.t3a, ws.t3b);
      rep.B4 = -0.5 * krylov::dot(ws.t3a, ws.t3b, n);
      ale::cellGradient(m, U1_, ws.G);
      ale::constraint(m, U1_, ws.G, ws.sF, C1_);
      rep.B5 = tau * krylov::dot(dp_, C1_, nc);
      rep.divU1 = tau * krylov::maxScaled(C1_, fr.invVolume, nc);
      detail::combine(Real(1), U1_, Real(-1), s.U, ws.t3a, n);
      detail::combine(Real(1), U1_, Real(1), s.U, ws.t3c, n);
      ale::applyMass(m, fr.rho, ws.t3c, ws.t3b);
      rep.dK = 0.5 * krylov::dot(ws.t3a, ws.t3b, n);
      ale::applyMass(m, fr.rho, U1_, ws.t3b);
      rep.K1 = 0.5 * krylov::dot(U1_, ws.t3b, n);
      detail::componentSums(ws.t3b, nc, rep.momentum, &rep.absMomentum);
      rep.booked = rep.B2 + rep.B3 + rep.B4 + rep.B5 + rep.B6 + rep.B7 + rep.B8a + rep.B8b +
                   rep.B9 + rep.Bres;
      rep.closure = rep.dK - rep.booked;
      rep.unsignedSum = rep.B2 + rep.B3 + rep.B5 + rep.B6 + rep.B8b + rep.B9 + rep.Bres;
    }
    lap(tb);
    rep.timeBudget += tb;
    // S6
    rep.compactFlux = compactFluxOf(fr, U1_, tau, s.phiCompact, Phi1_);
    rep.divPhi1 = maxDivergence(fr, Phi1_, tau);
    lap(rep.timeCompactFlux);
    rep.timeSolveCompactFlux = lastSolveSeconds_;
    // rotate (S8): Uⁿ⁻¹ ← Uⁿ, Uⁿ ← U^{n+1}
    Kokkos::deep_copy(s.Uprev, s.U);
    Kokkos::deep_copy(s.U, U1_);
    s.hasPrevious = true;
    return rep;
  }

  const DV<Real>& lastUstar() const { return Ustar_; }
  const DV<Real>& lastTransportFlux() const { return ws.Phi; }
  const DV<Real>& lastPressureIncrement() const { return dp_; }

 private:
  DV<Real> Ustar_, U1_, dp_, Phi1_, C1_;
  double lastSolveSeconds_ = 0;  ///< the L_ρ solve of the last compactFluxOf (timed)
};

}  // namespace peclet::voro::ale

#endif  // PECLET_VORO_ALE_STEPPER_HPP
