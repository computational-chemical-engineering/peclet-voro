/**
 * @file lagrangian/integrator.hpp
 * \brief One time step of the Lagrangian power-cell method NHV-W (M7 DESIGN_NOTE §10.3 item 6,
 * with §2.6 for the linear solves and §2.7 steps 3 and 5 for the heat and the step size):
 *
 *   1. kick 1 at x^n, friction fused (θ-midpoint), velocity constraint C^n v^{½} = 0 (by default
 *      C and ζ of kick 1 at x^n + (τ/2) v^n instead, option S,
 * IntegratorParams::midpointConstraint): (M + θ(τ/2)ζ^n) v^{½} + (τ/2) C^{nᵀ} λ₁ = (M −
 * (1−θ)(τ/2)ζ^n) v^n − (τ/2)(∇Ũ^n − M a(x^n));
 *   2. drift x^{n+1} = x^n + τ v^{½} (wrapped into the box); weights warm-started by
 *      H δw = −τ J_x v^{½} (at x^n) and solved by the damped Newton of weight_solve.hpp, every
 *      iterate an incremental REPAIR of the resident engine (no cold rebuild); then π^{n+1} and
 *      ∇Ũ^{n+1} = ∂_xU + J_xᵀπ (potential.hpp);
 *   3. kick 2: the same saddle at x^{n+1}, giving v^{n+1} and λ₂;
 *   4. bookkeeping: heat Q += (τ/2) v̄_θᵀ ζ (v_old + v_new)/2 per kick (§2.7 step 3), the
 *      constraint work Q_c += (τ/4) λ₂ᵀ C^{n+1} v^{½} (kick 2's; kick 1 does none), the body-force
 *      work W += (τ/2) (M a)ᵀ(v_old + v_new)/2 per kick, so that K + Ũ + Q + Q_c − W is conserved
 *      to the integrator's O(τ²);
 *   5. step size τ = min(C_cfl min_i h_i/|v_i|, C_σ √(ρ_σ h_min³/σ_max), τ_user) (stableStep);
 *   6. the gauge G″ (DESIGN_NOTE §12.3.2: strain-activated, Lloyd-only, ascent-free) after kick 2,
 *      every IntegratorParams::gaugeEvery steps (see gauge()); its ΔŨ and ΔK go to their own
 * counters, so K + Ũ + Q + Q_c − W − ΔŨ_G − ΔK_G is what the integrator conserves to O(τ²). Moving
 * no-slip walls: the wall term of R becomes ½ Σ_w μ_i A^w |P_w(v_i − u_w)|²/δ_w with u_w a linear
 * field (FlowParams::wallVelocity/wallGradient) at the wall-facet centroid; the affine part −f_w
 * enters the kick's right-hand side and its work f_wᵀv̄ goes to the body work W (C is unchanged: u_w
 * is taken tangential, P_w u_w). No topology bookkeeping beyond the repair's.
 *
 * The saddle [M + θ(τ/2)ζ, (τ/2)Cᵀ; C, 0] is solved by preconditioned CG on the Schur complement
 * S = C A⁻¹ Cᵀ (A = M + θ(τ/2)ζ) for μ = (τ/2)λ, preconditioned by AMG on the assembled
 * inertial Schur complement S₀ = C M⁻¹ Cᵀ (2-ring stencil, detail::schurHost; pinned at DOF 0,
 * diagonal kept; SaddleParams::schurAssembled, M7 WO4a) or on the two-point Laplacian L_ρ,
 * (L_ρ p)_i = Σ_f (A_f/d_f) ½(1/ρ_i + 1/ρ_j)(p_i − p_j) (§2.6, its compact approximation), plus
 * the viscous diagonal of Cahouet–Chabard: P⁻¹ = S₀⁻¹ (or L_ρ⁻¹) + θ(τ/2)·2μ_i/V_i.
 * S₀ (and its approximation L_ρ) is the Schur complement's inertial limit (A = M); when θ(τ/2)ζ
 * dominates M (ν τ/h² ≫ 1) S tends to C(θ(τ/2)ζ)⁻¹Cᵀ ≈ V/(2θ(τ/2)μ), a mass-matrix-like operator
 * S₀⁻¹ cannot precondition; the sum covers both limits (Cahouet & Chabard 1988). The velocity is
 * carried by the same recurrence as the residual, v ← v − α A⁻¹Cᵀp,   r ← r − α C (A⁻¹Cᵀp), with
 * the SAME vector A⁻¹Cᵀp in both, so r = C v holds to round-off whatever the inner accuracy: the
 * constraint is met to the outer tolerance ‖C v‖_∞ ≤ min(tol·max V/τ, tol·‖C‖·v_ref) (see solve),
 * and the inner tolerance (1e-10, §2.6) bounds only the momentum residual. The outer CG is the
 * flexible (Polak–Ribière) form, since inexact inner solves make S vary slightly between
 * iterations. Inner A-solves: PCG with Jacobi on the approximate diagonal m + θ(τ/2) Σ_f μ_f
 * A/(3d)(1 + n_a²) (+ the wall term), or, when max ν τ/h² exceeds 0.5 (§2.6), AMG on the scalar
 * two-point operator m + θ(τ/2)·(4/9) Σ_f μ_f A/d (n̂_a² averaged to 1/3) applied per component.
 * Inviscid (θτζ = 0): A = M, solved exactly. Any non-convergence throws.
 */
#ifndef PECLET_VORO_LAGRANGIAN_INTEGRATOR_HPP
#define PECLET_VORO_LAGRANGIAN_INTEGRATOR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/graph_amg_device.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/operators.hpp"
#include "peclet/voro/lagrangian/potential.hpp"
#include "peclet/voro/lagrangian/weight_solve.hpp"
#include "peclet/voro/plane_policy.hpp"

namespace peclet::voro::lagrangian {

namespace detail {
template <class Real>
Real maxAbsView(const LView<Real>& a) {
  Real m = 0;
  Kokkos::parallel_reduce(
      "lagrangian.maxabs", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, a.extent(0)),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        const Real v = Kokkos::fabs(a(i));
        acc = v > acc ? v : acc;
      },
      Kokkos::Max<Real>(m));
  return m;
}

/// Host CSR of a two-point operator over F's directed facets,
///   (K x)_i = extra(i) x_i + Σ_f coef(i, j, A, d) (x_i − x_j),
/// multi-image duplicates merged. `pin`: row and column 0 lose their off-diagonals (the diagonal
/// is kept) — the SPD preconditioner matrix of a singular (periodic / Neumann) Laplacian.
template <class Real, class Coef, class Extra>
peclet::core::solver::HostCsrOp twoPointHost(const CellFacets<Real>& F, Coef coef, Extra extra,
                                             bool pin) {
  const auto off = peclet::core::toVector(F.facetOffset);
  const auto nbr = peclet::core::toVector(F.nbr);
  const auto area = peclet::core::toVector(F.area);
  const auto conn = peclet::core::toVector(F.conn);
  const int N = F.nCells;
  peclet::core::solver::HostCsrOp K;
  K.n = N;
  K.diag.assign(N, 0.0);
  K.start.assign(N + 1, 0);
  std::vector<std::pair<int, double>> row;
  for (int i = 0; i < N; ++i) {
    row.clear();
    K.diag[i] = extra(i);
    for (int f = off[i]; f < off[i + 1]; ++f) {
      const double d = std::sqrt(conn[3 * f] * conn[3 * f] + conn[3 * f + 1] * conn[3 * f + 1] +
                                 conn[3 * f + 2] * conn[3 * f + 2]);
      const double w = coef(i, nbr[f], area[f], d);
      K.diag[i] += w;
      if (!pin || (i != 0 && nbr[f] != 0))
        row.emplace_back(nbr[f], -w);
    }
    std::sort(row.begin(), row.end());
    for (std::size_t q = 0; q < row.size(); ++q) {
      if (q > 0 && row[q].first == row[q - 1].first)
        K.coef.back() += row[q].second;
      else {
        K.nbr.push_back(row[q].first);
        K.coef.push_back(row[q].second);
      }
    }
    K.start[i + 1] = static_cast<peclet::core::Index>(K.nbr.size());
  }
  return K;
}

/// Host CSR of the assembled inertial Schur complement S₀ = C M⁻¹ Cᵀ (the 2-ring stencil), pinned
/// at DOF 0 like twoPointHost: S₀_ik = Σ_l (1/m_l) C_il·C_kl over the velocity cells l, with column
/// l of C holding −(Σ_f (A/d) c_f + Σ_w A_w n̂_w) at row l and (A/d)(c_f − d_f) at the row of each
/// neighbour across facet f of l (applyDivergence / applyDivergenceT).
template <class Real>
peclet::core::solver::HostCsrOp schurHost(const CellFacets<Real>& F, const LView<Real>& mass) {
  const auto off = peclet::core::toVector(F.facetOffset);
  const auto nbr = peclet::core::toVector(F.nbr);
  const auto area = peclet::core::toVector(F.area);
  const auto conn = peclet::core::toVector(F.conn);
  const auto cen = peclet::core::toVector(F.centroid);
  const auto woff = peclet::core::toVector(F.wallOffset);
  const auto warea = peclet::core::toVector(F.wallArea);
  const auto wn = peclet::core::toVector(F.wallNormal);
  const auto m = peclet::core::toVector(mass);
  const int N = F.nCells;
  // column l of C as (row, 3-vector) pairs
  std::vector<std::vector<std::pair<int, std::array<double, 3>>>> col(N);
  for (int l = 0; l < N; ++l) {
    std::array<double, 3> dl = {0, 0, 0};
    for (int f = off[l]; f < off[l + 1]; ++f) {
      const double d = std::sqrt(conn[3 * f] * conn[3 * f] + conn[3 * f + 1] * conn[3 * f + 1] +
                                 conn[3 * f + 2] * conn[3 * f + 2]);
      const double sc = area[f] / d;
      std::array<double, 3> cj;
      for (int a = 0; a < 3; ++a) {
        dl[a] -= sc * cen[3 * f + a];
        cj[a] = sc * (cen[3 * f + a] - conn[3 * f + a]);
      }
      col[l].emplace_back(nbr[f], cj);
    }
    for (int f = woff[l]; f < woff[l + 1]; ++f)
      for (int a = 0; a < 3; ++a)
        dl[a] -= warea[f] * wn[3 * f + a];
    col[l].emplace_back(l, dl);
  }
  std::vector<std::vector<std::pair<int, double>>> rows(N);
  for (int l = 0; l < N; ++l) {
    const double im = 1.0 / m[l];
    for (const auto& [i, ci] : col[l])
      for (const auto& [k, ck] : col[l])
        rows[i].emplace_back(k, im * (ci[0] * ck[0] + ci[1] * ck[1] + ci[2] * ck[2]));
  }
  peclet::core::solver::HostCsrOp K;
  K.n = N;
  K.diag.assign(N, 0.0);
  K.start.assign(N + 1, 0);
  for (int i = 0; i < N; ++i) {
    auto& row = rows[i];
    std::sort(row.begin(), row.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (std::size_t q = 0; q < row.size(); ++q) {
      const int k = row[q].first;
      if (k == i) {
        K.diag[i] += row[q].second;
        continue;
      }
      if (i == 0 || k == 0)
        continue;  // pinned
      if (q > 0 && row[q - 1].first == k)
        K.coef.back() += row[q].second;
      else {
        K.nbr.push_back(k);
        K.coef.push_back(row[q].second);
      }
    }
    std::vector<std::pair<int, double>>().swap(row);
    K.start[i + 1] = static_cast<peclet::core::Index>(K.nbr.size());
  }
  return K;
}
}  // namespace detail

/// The kick's saddle solver (M7 WO-PERF).
enum class SaddleMethod {
  kSchurCG,  ///< Schur-complement CG with an exact inner A-solve per iteration (WO4 default)
  kCoupled,  ///< A: one FGMRES on [A, Cᵀ; C, 0], block-triangular preconditioner (exact saddle)
  kProjection,  ///< B: rotational incremental pressure correction (one A-solve + one S₀ projection)
};

struct SaddleParams {
  double tol = 1e-12;       ///< outer: ‖C v‖_∞ ≤ tol · max_i V_i / τ (DESIGN_NOTE §2.6)
  double innerTol = 1e-10;  ///< relative residual of each inner A-solve
  int maxOuter = 2000, maxInner = 2000;
  double amgSwitch = 0.5;  ///< inner preconditioner: AMG when max ν τ/h² exceeds this, else Jacobi
  /// Schur preconditioner: true (default since M7 WO4a) = AMG on the assembled inertial Schur
  /// complement S₀ = C M⁻¹ Cᵀ (2-ring stencil); false = AMG on the two-point L_ρ (§2.6).
  /// Cahouet–Chabard's viscous term either way. Measured (Lloyd drop, ρ 1:10, σ 1, μ 0.1, gauge
  /// on): outer iterations per kick 96 → 25 and s/step 136 → 35 at N = 32768 (7.7 → 3.0 at
  /// 4096); inviscid 113 → 23 outer, but its 4× dearer setup makes the step ~7 % slower.
  bool schurAssembled = true;
  SaddleMethod method = SaddleMethod::kSchurCG;
  /// B: χ of the rotational multiplier update μ += φ + χ a (μ_min / V_i) (C v*)_i — flow's
  /// rotational term −χ μ_min ∇·u* (its default mode "min", χ = 1: the least cell viscosity, stable
  /// at any contrast); χ = 2 is the whole gradient part of the stress-form ζ (the Cahouet–Chabard
  /// 2μ/V). 0: plain incremental
  double rotational = 1.0;
  int restart = 40;      ///< A: FGMRES restart length
  int coupledInner = 2;  ///< A: Jacobi-PCG iterations of Â⁻¹ in the preconditioner (0: Jacobi)
};

struct SaddleStats {
  int outer = 0;          ///< Schur CG iterations
  int inner = 0;          ///< inner A-solve iterations, summed
  double constraint = 0;  ///< ‖C v‖_∞ τ / max V (recomputed from v)
  double viscNumber = 0;  ///< max ν τ / h²
  bool amgInner = false;  ///< inner preconditioner was AMG
  double tolUsed = 0;     ///< the ‖C v‖_∞ bound applied (after the physical scaling)
  // wall-clock split (diagnostics; seconds): AMG setups charged to this kick (the L_ρ hierarchy of
  // the last setup() and the viscous one), inner A-solves, Schur preconditioner applications
  double tSetup = 0, tInner = 0, tPrecond = 0, tTotal = 0;
};

/// The Stokes-type saddle of one NHV-W kick (see the file comment).
template <class Real>
class SaddleSolver {
 public:
  /// Geometry setup at the facets `F`: the L_ρ AMG and the approximate ζ diagonal. `rho`, `mu`,
  /// `mass`, `h` (N): cell density, viscosity, mass and length h_i = V⁰_i^{1/3}.
  /// `rebuildAmg` false keeps the L_ρ / S₀ hierarchy of an earlier setup as the preconditioner (it
  /// only preconditions: the Schur operator is applied matrix-free at F; a kind change forces it).
  void setup(const CellFacets<Real>& F, const LView<Real>& rho, const LView<Real>& mu,
             const LView<Real>& mass, const LView<Real>& h, bool rebuildAmg = true) {
    F_ = F;
    mu_ = mu;
    mass_ = mass;
    const int N = F.nCells;
    if (static_cast<int>(r_.extent(0)) != N) {
      const std::size_t n = static_cast<std::size_t>(N), n3 = 3 * n;
      r_ = LView<Real>("sd.r", n);
      z_ = LView<Real>("sd.z", n);
      zOld_ = LView<Real>("sd.zOld", n);
      p_ = LView<Real>("sd.p", n);
      s_ = LView<Real>("sd.s", n);
      w_ = LView<Real>("sd.w", n3);
      q_ = LView<Real>("sd.q", n3);
      ir_ = LView<Real>("sd.ir", n3);
      iz_ = LView<Real>("sd.iz", n3);
      ip_ = LView<Real>("sd.ip", n3);
      iq_ = LView<Real>("sd.iq", n3);
      zdiag_ = LView<Real>("sd.zdiag", n3);
      c0_ = LView<Real>("sd.c0", n);
      c1_ = LView<Real>("sd.c1", n);
      rStar_ = LView<Real>("sd.rStar", n);
      scr_ = DissipationScratch<Real>(N);
      kv_.clear();
    }
    const auto rh = peclet::core::toVector(rho);
    const auto mh = peclet::core::toVector(mu);
    const auto hh = peclet::core::toVector(h);
    viscous_ = false;
    nuH2_ = 0;
    for (int i = 0; i < N; ++i) {
      viscous_ = viscous_ || mh[i] > 0;
      nuH2_ = std::max(nuH2_, mh[i] / (rh[i] * hh[i] * hh[i]));
    }
    peclet::core::solver::AmgParams ap;
    ap.ndofPerNode = 1;
    {  // Cahouet–Chabard viscous diagonal 2μ_i/V_i (scaled by a = θτ/2 in solve)
      const auto vh = peclet::core::toVector(F.volume);
      std::vector<double> cc(N);
      for (int i = 0; i < N; ++i)
        cc[i] = vh[i] > 0 ? 2.0 * mh[i] / vh[i] : 0.0;
      cc_ = peclet::core::toDevice<Real>(cc, "sd.cc");
      const double muMin = N > 0 ? *std::min_element(mh.begin(), mh.end()) : 0.0;
      for (int i = 0; i < N; ++i)
        cc[i] = vh[i] > 0 ? muMin / vh[i] : 0.0;
      ccMin_ = peclet::core::toDevice<Real>(cc, "sd.ccMin");
    }
    Kokkos::Timer tb;
    if (!rebuildAmg && lrho_.numLevels() > 0 && builtAssembled_ == assembled_) {
      // the lagged hierarchy (no setup charged)
    } else if (assembled_)
      lrho_.build(detail::schurHost(F, mass), ap);
    else
      lrho_.build(detail::twoPointHost(
                      F,
                      [&](int i, int j, double A, double d) {
                        return A / d * 0.5 * (1.0 / rh[i] + 1.0 / rh[j]);
                      },
                      [](int) { return 0.0; }, /*pin=*/true),
                  ap);
    builtAssembled_ = assembled_;
    setupSeconds_ = tb.seconds();
    mh_ = mh;
    {  // ‖C‖ per unit velocity: max_i Σ_f (A/d)|c_f| + Σ_w A_w (the physically scaled stop in
       // solve)
      const CellFacets<Real> G = F;
      Real sc = 0;
      Kokkos::parallel_reduce(
          "sd.cscale", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
          KOKKOS_LAMBDA(const int i, Real& acc) {
            Real a = 0;
            for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
              Real d2 = 0, c2 = 0;
              for (int e = 0; e < 3; ++e) {
                d2 += G.conn(3 * f + e) * G.conn(3 * f + e);
                c2 += G.centroid(3 * f + e) * G.centroid(3 * f + e);
              }
              a += G.area(f) * Kokkos::sqrt(c2 / d2);
            }
            for (int f = G.wallOffset(i); f < G.wallOffset(i + 1); ++f)
              a += G.wallArea(f);
            acc = a > acc ? a : acc;
          },
          Kokkos::Max<Real>(sc));
      cScale_ = sc;
    }
    if (viscous_) {  // approximate diagonal of ζ: Σ_f μ_f A/(3d)(1 + n_a²) + wall μ A_w/δ_w (1 −
                     // n_wa²)
      const CellFacets<Real> G = F;
      const LView<Real> M = mu, zd = zdiag_;
      Kokkos::parallel_for(
          "sd.zdiag", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
          KOKKOS_LAMBDA(const int i) {
            Real o[3] = {0, 0, 0};
            for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
              const int j = G.nbr(f);
              Real d2 = 0;
              for (int a = 0; a < 3; ++a)
                d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
              const Real mi = M(i), mj = M(j);
              const Real muf = (mi + mj) > Real(0) ? Real(2) * mi * mj / (mi + mj) : Real(0);
              const Real c = muf * G.area(f) / (Real(3) * Kokkos::sqrt(d2));
              for (int a = 0; a < 3; ++a)
                o[a] += c * (Real(1) + G.conn(3 * f + a) * G.conn(3 * f + a) / d2);
            }
            for (int f = G.wallOffset(i); f < G.wallOffset(i + 1); ++f) {
              const Real nw[3] = {G.wallNormal(3 * f), G.wallNormal(3 * f + 1),
                                  G.wallNormal(3 * f + 2)};
              const Real dw = nw[0] * G.wallCentroid(3 * f) + nw[1] * G.wallCentroid(3 * f + 1) +
                              nw[2] * G.wallCentroid(3 * f + 2);
              if (!(dw > Real(0)))
                continue;
              for (int a = 0; a < 3; ++a)
                o[a] += M(i) * G.wallArea(f) / dw * (Real(1) - nw[a] * nw[a]);
            }
            for (int a = 0; a < 3; ++a)
              zd(3 * i + a) = o[a];
          });
    }
  }

  /// Solve (M + a ζ) v + Cᵀ μ = b, C v = 0, with μ = (τ/2) λ: `lam` holds the initial guess of
  /// λ (e.g. the previous kick's) and returns λ = μ / halfTau. `a` = θ τ/2, `tolAbs` the outer
  /// bound on ‖C v‖_∞.
  SaddleStats solve(Real a, Real halfTau, Real tolAbs, const LView<Real>& b, const LView<Real>& v,
                    const LView<Real>& lam, const SaddleParams& prm) {
    using Exec = peclet::core::ExecSpace;
    const int N = F_.nCells;
    SaddleStats st;
    Kokkos::Timer tAll, tPart;
    st.tSetup = setupSeconds_;  // charged once, to the first solve after setup()
    setupSeconds_ = 0;
    st.viscNumber = nuH2_ * 2 * halfTau;
    useAmg_ = viscous_ && a > Real(0) && st.viscNumber > prm.amgSwitch;
    st.amgInner = useAmg_;
    if (useAmg_) {
      tPart.reset();
      peclet::core::solver::AmgParams ap;
      ap.ndofPerNode = 1;
      const auto mh = peclet::core::toVector(mass_);
      const auto& muh = mh_;
      const double aa = a;
      visc_.build(detail::twoPointHost(
                      F_,
                      [&](int i, int j, double A, double d) {
                        const double muf =
                            muh[i] + muh[j] > 0 ? 2 * muh[i] * muh[j] / (muh[i] + muh[j]) : 0.0;
                        return aa * (4.0 / 9.0) * muf * A / d;
                      },
                      [&](int i) { return mh[i]; }, /*pin=*/false),
                  ap);
      st.tSetup += tPart.seconds();
    }
    const LView<Real> r = r_, z = z_, zo = zOld_, p = p_, s = s_, q = q_, vv = v, mu = lam;
    {  // the stop, physically scaled (M7 WO4a): ‖C v‖_∞ ≤ min(tolAbs, tol · ‖C‖ · v_ref), with
       // v_ref = max_i |b_i|/m_i the velocity the kick would produce unconstrained. tolAbs alone
       // (tol · max V/τ, §2.6) is relative to the grid velocity h/τ: at rest, where b is round-off,
       // it accepted the warm start with no iteration and its divergent part (measured 1.1e-13 v_g
       // on the walled 16³ layer, ratio 1000; 1.5e-15 v_g once projected)
      const LView<Real> B = b, m = mass_;
      Real vr = 0;
      Kokkos::parallel_reduce(
          "sd.vref", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i, Real& acc) {
            const Real q2 =
                B(3 * i) * B(3 * i) + B(3 * i + 1) * B(3 * i + 1) + B(3 * i + 2) * B(3 * i + 2);
            const Real u = Kokkos::sqrt(q2) / m(i);
            acc = u > acc ? u : acc;
          },
          Kokkos::Max<Real>(vr));
      const Real tp = Real(prm.tol) * cScale_ * vr;
      tolAbs = tp < tolAbs ? tp : tolAbs;
      st.tolUsed = tolAbs;
    }
    {  // warm start: μ₀ = (τ/2) λ_guess (mean removed), v₀ = A⁻¹(b − Cᵀμ₀)
      Kokkos::parallel_for(
          "sd.mu0", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i) { mu(i) *= halfTau; });
      removeMean(mu);
      applyDivergenceT(F_, mu, w_);
      const LView<Real> w = w_, B = b;
      Kokkos::parallel_for(
          "sd.b0", Kokkos::RangePolicy<Exec>(0, 3 * N),
          KOKKOS_LAMBDA(const int k) { w(k) = B(k) - w(k); });
    }
    tPart.reset();
    st.inner += solveA(a, w_, v, prm);
    Kokkos::fence();
    st.tInner += tPart.seconds();
    if (prm.method == SaddleMethod::kCoupled && viscous_ && a > Real(0))
      return solveCoupled(a, halfTau, tolAbs, b, v, lam, prm, st, tAll);
    const bool proj = prm.method == SaddleMethod::kProjection;
    applyDivergence(F_, v, r_);
    if (proj)
      Kokkos::deep_copy(rStar_, r_);  // C v* for the rotational update
    Real rinf = detail::maxAbsView(r);
    const Real rinit = rinf;
    if (rinf > tolAbs) {
      tPart.reset();
      schurPrecond(a, r, z);
      Kokkos::fence();
      st.tPrecond += tPart.seconds();
      Kokkos::deep_copy(p, z);
      Real rz = detail::dotView(r, z);
      bool done = false;
      for (int k = 0; k < prm.maxOuter; ++k) {
        applyDivergenceT(F_, p, w_);
        tPart.reset();
        // B projects with S₀ = C M⁻¹ Cᵀ: v = v* − M⁻¹Cᵀφ (the A-solve only in the predictor)
        st.inner += proj ? solveA(Real(0), w_, q, prm) : solveA(a, w_, q, prm);
        Kokkos::fence();
        st.tInner += tPart.seconds();
        applyDivergence(F_, q, s);
        const Real ps = detail::dotView(p, s);
        if (!(ps > Real(0)))
          throw std::runtime_error(
              "voro: NHV-W saddle: Schur CG broke down (p·Sp = " + std::to_string(ps) + ")");
        const Real al = rz / ps;
        Kokkos::parallel_for(
            "sd.upd", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
              mu(i) += al * p(i);
              r(i) -= al * s(i);
              for (int c = 0; c < 3; ++c)
                vv(3 * i + c) -= al * q(3 * i + c);
            });
        st.outer = k + 1;
        rinf = detail::maxAbsView(r);
        if (rinf <= tolAbs) {
          done = true;
          break;
        }
        Kokkos::deep_copy(zo, z);
        tPart.reset();
        schurPrecond(a, r, z);
        Kokkos::fence();
        st.tPrecond += tPart.seconds();
        const Real rzNew = detail::dotView(r, z);
        const Real be = (rzNew - detail::dotView(r, zo)) / rz;
        rz = rzNew;
        Kokkos::parallel_for(
            "sd.dir", Kokkos::RangePolicy<Exec>(0, N),
            KOKKOS_LAMBDA(const int i) { p(i) = z(i) + be * p(i); });
      }
      if (!done) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "voro: NHV-W saddle: Schur CG did not converge in %d iterations "
                      "(|Cv|_inf = %.3e, bound %.3e, initial %.3e)",
                      prm.maxOuter, double(rinf), double(tolAbs), double(rinit));
        throw std::runtime_error(msg);
      }
    }
    if (proj && viscous_ && a > Real(0) && prm.rotational > 0) {
      // rotational update: μ += χ a (μ_min / V) C v* — the viscous part of the multiplier
      // increment the S₀ projection leaves out (S⁻¹ ≈ S₀⁻¹ + a·2μ/V, Cahouet–Chabard; flow's −μ∇·u*)
      const LView<Real> rs = rStar_, cm = ccMin_;
      const Real ca = Real(prm.rotational) * a;
      Kokkos::parallel_for(
          "sd.rot", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i) { mu(i) += ca * cm(i) * rs(i); });
    }
    // The constant is the exact null space of Cᵀ (Cᵀ1 = 0, walls included); the CG picks up a
    // component along it from the preconditioner. It is harmless to v, but it enters the
    // constraint work λᵀC v^{½} through the round-off of 1ᵀC = 0, and with the warm start it
    // accumulated from kick to kick (measured: energy-identity defects up to 7e-4 in single
    // kicks). So μ is kept mean-free, at the warm start and here.
    removeMean(mu);
    const Real ih = Real(1) / halfTau;
    Kokkos::parallel_for(
        "sd.lam", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { mu(i) *= ih; });
    applyDivergence(F_, v, r_);
    st.constraint = detail::maxAbsView(r_);  // absolute; the caller scales by τ / max V
    Kokkos::fence();
    st.tTotal = tAll.seconds();
    return st;
  }

  bool viscous() const { return viscous_; }
  void setAssembled(bool on) { assembled_ = on; }  ///< SaddleParams::schurAssembled (next setup)
  const CellFacets<Real>& facets() const { return F_; }
  DissipationScratch<Real>& scratch() { return scr_; }

 private:
  /// z = (L_ρ⁻¹ + a·2μ/V) r (Cahouet–Chabard; the viscous term only with viscosity and a > 0).
  void schurPrecond(Real a, const LView<Real>& r, const LView<Real>& z) {
    lrho_.apply(r, z);
    if (viscous_ && a > Real(0)) {
      const LView<Real> R = r, Z = z, cc = cc_;
      Kokkos::parallel_for(
          "sd.cahouet", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, F_.nCells),
          KOKKOS_LAMBDA(const int i) { Z(i) += a * cc(i) * R(i); });
    }
  }

  static void removeMean(const LView<Real>& x) {
    const int n = static_cast<int>(x.extent(0));
    Real s = 0;
    Kokkos::parallel_reduce(
        "sd.mean", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc += x(i); }, s);
    const Real m = s / n;
    Kokkos::parallel_for(
        "sd.demean", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int i) { x(i) -= m; });
  }

  /// A (SaddleMethod::kCoupled): right-preconditioned restarted FGMRES on K = [A, Cᵀ; C, 0] from
  /// the predictor (v = A⁻¹(b − Cᵀμ₀), μ₀) the caller computed. Block upper-triangular
  /// preconditioner P = [Â, Cᵀ; 0, −Ŝ]: z_μ = −Ŝ⁻¹ r_μ, z_v = Â⁻¹(r_v − Cᵀ z_μ), Ŝ⁻¹ = the Schur
  /// preconditioner (AMG on S₀ + Cahouet–Chabard), Â⁻¹ = prm.coupledInner Jacobi-PCG iterations.
  /// The constraint rows are scaled by ς = ε_m / tolAbs (ε_m = innerTol ‖b‖₂) so that the 2-norm
  /// target ε_m bounds ‖C v‖_∞ ≤ tolAbs; converged on the true residuals ‖C v‖_∞ ≤ tolAbs and
  /// ‖b − A v − Cᵀμ‖₂ ≤ 10 ε_m. The exact saddle solution to that tolerance.
  SaddleStats solveCoupled(Real a, Real halfTau, Real tolAbs, const LView<Real>& b,
                           const LView<Real>& v, const LView<Real>& mu, const SaddleParams& prm,
                           SaddleStats st, Kokkos::Timer& tAll) {
    using Exec = peclet::core::ExecSpace;
    const int N = F_.nCells, n3 = 3 * N, m = prm.restart;
    if (static_cast<int>(kv_.size()) != m + 1) {
      kv_.clear();
      km_.clear();
      zv_.clear();
      zm_.clear();
      for (int j = 0; j <= m; ++j) {
        kv_.emplace_back("sd.kv", n3);
        km_.emplace_back("sd.km", N);
        if (j < m) {
          zv_.emplace_back("sd.zv", n3);
          zm_.emplace_back("sd.zm", N);
        }
      }
    }
    const LView<Real> M = mass_;
    const Real em = Real(prm.innerTol) * std::sqrt(detail::dotView(b, b));
    if (!(em > Real(0)) || !(tolAbs > Real(0)))
      return finishCoupled(halfTau, v, mu, st, tAll);
    const Real sc = em / tolAbs;
    Kokkos::Timer tPart;
    // out = K' (xv, xm): (M + aζ) xv + Cᵀxm, ς C xv
    auto applyK = [&](const LView<Real>& xv, const LView<Real>& xm, const LView<Real>& ov,
                      const LView<Real>& om) {
      applyDissipation(F_, mu_, xv, ov, scr_);
      applyDivergenceT(F_, xm, w_);
      const LView<Real> W = w_;
      Kokkos::parallel_for(
          "sd.Kv", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { ov(k) = M(k / 3) * xv(k) + a * ov(k) + W(k); });
      applyDivergence(F_, xv, om);
      Kokkos::parallel_for(
          "sd.Km", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { om(i) *= sc; });
    };
    // the true residual (b − A v − Cᵀμ, −ς C v) into (kv_[0], km_[0]); returns ‖C v‖_∞, ‖r_v‖₂
    auto residual = [&](Real& cinf, Real& rv2) {
      applyK(v, mu, kv_[0], km_[0]);
      const LView<Real> rv = kv_[0], rm = km_[0];
      Kokkos::parallel_for(
          "sd.Rv", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { rv(k) = b(k) - rv(k); });
      Kokkos::parallel_for(
          "sd.Rm", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { rm(i) = -rm(i); });
      cinf = detail::maxAbsView(rm) / sc;
      rv2 = std::sqrt(detail::dotView(rv, rv));
    };
    std::vector<double> H(static_cast<std::size_t>((m + 1) * m)), cs(m), sn(m), g(m + 1), y(m);
    auto Hij = [&](int i, int j) -> double& {
      return H[static_cast<std::size_t>(j * (m + 1) + i)];
    };
    Real cinf = 0, rv2 = 0;
    for (int cycle = 0;; ++cycle) {
      residual(cinf, rv2);
      if (cinf <= tolAbs && rv2 <= Real(10) * em)
        break;
      if (st.outer >= prm.maxOuter) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "voro: NHV-W saddle: coupled FGMRES did not converge in %d iterations "
                      "(|Cv|_inf = %.3e, bound %.3e; |r_v| = %.3e, bound %.3e)",
                      st.outer, double(cinf), double(tolAbs), double(rv2), double(10 * em));
        throw std::runtime_error(msg);
      }
      const double beta =
          std::sqrt(detail::dotView(kv_[0], kv_[0]) + detail::dotView(km_[0], km_[0]));
      scaleBlk(kv_[0], km_[0], Real(1.0 / beta));
      std::fill(g.begin(), g.end(), 0.0);
      g[0] = beta;
      int j = 0;
      for (; j < m; ++j) {
        // z_j = P⁻¹ v_j
        tPart.reset();
        {
          const LView<Real> vm = km_[j], zmj = zm_[j], z = z_, S = s_;
          Kokkos::parallel_for(
              "sd.Pm", Kokkos::RangePolicy<Exec>(0, N),
              KOKKOS_LAMBDA(const int i) { S(i) = vm(i) / sc; });
          schurPrecond(a, S, z);
          Kokkos::parallel_for(
              "sd.Pz", Kokkos::RangePolicy<Exec>(0, N),
              KOKKOS_LAMBDA(const int i) { zmj(i) = -z(i); });
          applyDivergenceT(F_, zmj, w_);
          const LView<Real> vv = kv_[j], W = w_, q = q_;
          Kokkos::parallel_for(
              "sd.Pv", Kokkos::RangePolicy<Exec>(0, n3),
              KOKKOS_LAMBDA(const int k) { q(k) = vv(k) - W(k); });
        }
        Kokkos::fence();
        st.tPrecond += tPart.seconds();
        tPart.reset();
        st.inner += solveAcap(a, q_, zv_[j], prm.coupledInner);
        Kokkos::fence();
        st.tInner += tPart.seconds();
        applyK(zv_[j], zm_[j], kv_[j + 1], km_[j + 1]);
        for (int i = 0; i <= j; ++i) {  // modified Gram–Schmidt
          const double hij =
              detail::dotView(kv_[j + 1], kv_[i]) + detail::dotView(km_[j + 1], km_[i]);
          Hij(i, j) = hij;
          axpyBlk(kv_[j + 1], km_[j + 1], kv_[i], km_[i], Real(-hij));
        }
        const double hn = std::sqrt(detail::dotView(kv_[j + 1], kv_[j + 1]) +
                                    detail::dotView(km_[j + 1], km_[j + 1]));
        Hij(j + 1, j) = hn;
        if (hn > 0)
          scaleBlk(kv_[j + 1], km_[j + 1], Real(1.0 / hn));
        for (int i = 0; i < j; ++i) {  // the earlier Givens rotations
          const double t = cs[i] * Hij(i, j) + sn[i] * Hij(i + 1, j);
          Hij(i + 1, j) = -sn[i] * Hij(i, j) + cs[i] * Hij(i + 1, j);
          Hij(i, j) = t;
        }
        const double den = std::hypot(Hij(j, j), Hij(j + 1, j));
        cs[j] = den > 0 ? Hij(j, j) / den : 1.0;
        sn[j] = den > 0 ? Hij(j + 1, j) / den : 0.0;
        Hij(j, j) = den;
        Hij(j + 1, j) = 0;
        g[j + 1] = -sn[j] * g[j];
        g[j] = cs[j] * g[j];
        ++st.outer;
        if (std::fabs(g[j + 1]) <= 0.5 * em || hn == 0 || st.outer >= prm.maxOuter) {
          ++j;
          break;
        }
      }
      for (int i = j - 1; i >= 0; --i) {  // y = H⁻¹ g, x += Z y
        double t = g[i];
        for (int k = i + 1; k < j; ++k)
          t -= Hij(i, k) * y[k];
        y[i] = t / Hij(i, i);
        axpyBlk(v, mu, zv_[i], zm_[i], Real(y[i]));
      }
    }
    return finishCoupled(halfTau, v, mu, st, tAll);
  }

  SaddleStats finishCoupled(Real halfTau, const LView<Real>& v, const LView<Real>& mu,
                            SaddleStats st, Kokkos::Timer& tAll) {
    removeMean(mu);
    const Real ih = Real(1) / halfTau;
    Kokkos::parallel_for(
        "sd.lamA", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, F_.nCells),
        KOKKOS_LAMBDA(const int i) { mu(i) *= ih; });
    applyDivergence(F_, v, r_);
    st.constraint = detail::maxAbsView(r_);
    Kokkos::fence();
    st.tTotal = tAll.seconds();
    return st;
  }

  static void scaleBlk(const LView<Real>& xv, const LView<Real>& xm, Real c) {
    Kokkos::parallel_for(
        "sd.scv", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, xv.extent(0)),
        KOKKOS_LAMBDA(const int k) { xv(k) *= c; });
    Kokkos::parallel_for(
        "sd.scm", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, xm.extent(0)),
        KOKKOS_LAMBDA(const int k) { xm(k) *= c; });
  }
  /// (xv, xm) += c (yv, ym)
  static void axpyBlk(const LView<Real>& xv, const LView<Real>& xm, const LView<Real>& yv,
                      const LView<Real>& ym, Real c) {
    Kokkos::parallel_for(
        "sd.axv", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, xv.extent(0)),
        KOKKOS_LAMBDA(const int k) { xv(k) += c * yv(k); });
    Kokkos::parallel_for(
        "sd.axm", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, xm.extent(0)),
        KOKKOS_LAMBDA(const int k) { xm(k) += c * ym(k); });
  }

  /// x ≈ A⁻¹ rhs by exactly `iters` Jacobi-PCG iterations from x = 0 (0: one Jacobi apply), the
  /// variable preconditioner block Â⁻¹ of A. Returns the iterations.
  int solveAcap(Real a, const LView<Real>& rhs, const LView<Real>& x, int iters) {
    using Exec = peclet::core::ExecSpace;
    const int n3 = 3 * F_.nCells;
    const LView<Real> m = mass_, X = x, B = rhs, r = ir_, z = iz_, p = ip_, q = iq_, zd = zdiag_;
    Kokkos::parallel_for(
        "sd.capJ", Kokkos::RangePolicy<Exec>(0, n3), KOKKOS_LAMBDA(const int k) {
          const Real dk = m(k / 3) + a * zd(k);
          X(k) = B(k) / dk;
          r(k) = B(k);
          z(k) = X(k);
          p(k) = X(k);
        });
    if (iters <= 0)
      return 0;
    Kokkos::deep_copy(X, Real(0));
    Real rz = detail::dotView(r, z);
    for (int it = 0; it < iters; ++it) {
      applyDissipation(F_, mu_, p, q, scr_);
      Kokkos::parallel_for(
          "sd.capAp", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { q(k) = m(k / 3) * p(k) + a * q(k); });
      const Real pq = detail::dotView(p, q);
      if (!(pq > Real(0)) || !(rz > Real(0)))
        return it;
      const Real al = rz / pq;
      Kokkos::parallel_for(
          "sd.capUpd", Kokkos::RangePolicy<Exec>(0, n3), KOKKOS_LAMBDA(const int k) {
            X(k) += al * p(k);
            r(k) -= al * q(k);
            z(k) = r(k) / (m(k / 3) + a * zd(k));
          });
      if (it + 1 == iters)
        return iters;
      const Real rzNew = detail::dotView(r, z);
      const Real be = rzNew / rz;
      rz = rzNew;
      Kokkos::parallel_for(
          "sd.capDir", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { p(k) = z(k) + be * p(k); });
    }
    return iters;
  }

  /// x = A⁻¹ rhs, A = M + a ζ, by PCG to the relative residual prm.innerTol. Returns iterations.
  int solveA(Real a, const LView<Real>& rhs, const LView<Real>& x, const SaddleParams& prm) {
    using Exec = peclet::core::ExecSpace;
    const int N = F_.nCells, n3 = 3 * N;
    const LView<Real> m = mass_, X = x, B = rhs;
    if (!(a > Real(0)) || !viscous_) {
      Kokkos::parallel_for(
          "sd.Minv", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { X(k) = B(k) / m(k / 3); });
      return 0;
    }
    const LView<Real> r = ir_, z = iz_, p = ip_, q = iq_, zd = zdiag_;
    Kokkos::deep_copy(X, Real(0));
    Kokkos::deep_copy(r, B);
    const Real r0 = std::sqrt(detail::dotView(r, r));
    if (!(r0 > Real(0)))
      return 0;
    auto precond = [&]() {
      if (!useAmg_) {
        Kokkos::parallel_for(
            "sd.jacobi", Kokkos::RangePolicy<Exec>(0, n3),
            KOKKOS_LAMBDA(const int k) { z(k) = r(k) / (m(k / 3) + a * zd(k)); });
        return;
      }
      const LView<Real> c0 = c0_, c1 = c1_;
      for (int c = 0; c < 3; ++c) {
        Kokkos::parallel_for(
            "sd.split", Kokkos::RangePolicy<Exec>(0, N),
            KOKKOS_LAMBDA(const int i) { c0(i) = r(3 * i + c); });
        visc_.apply(c0, c1);
        Kokkos::parallel_for(
            "sd.merge", Kokkos::RangePolicy<Exec>(0, N),
            KOKKOS_LAMBDA(const int i) { z(3 * i + c) = c1(i); });
      }
    };
    precond();
    Kokkos::deep_copy(p, z);
    Real rz = detail::dotView(r, z);
    for (int it = 0; it < prm.maxInner; ++it) {
      applyDissipation(F_, mu_, p, q, scr_);
      Kokkos::parallel_for(
          "sd.Ap", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { q(k) = m(k / 3) * p(k) + a * q(k); });
      const Real pq = detail::dotView(p, q);
      if (!(pq > Real(0)))
        throw std::runtime_error("voro: NHV-W saddle: inner A-solve broke down");
      const Real al = rz / pq;
      Kokkos::parallel_for(
          "sd.Aupd", Kokkos::RangePolicy<Exec>(0, n3), KOKKOS_LAMBDA(const int k) {
            X(k) += al * p(k);
            r(k) -= al * q(k);
          });
      if (std::sqrt(detail::dotView(r, r)) <= prm.innerTol * r0)
        return it + 1;
      precond();
      const Real rzNew = detail::dotView(r, z);
      const Real be = rzNew / rz;
      rz = rzNew;
      Kokkos::parallel_for(
          "sd.Adir", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { p(k) = z(k) + be * p(k); });
    }
    throw std::runtime_error("voro: NHV-W saddle: inner A-solve did not converge in " +
                             std::to_string(prm.maxInner) + " iterations");
  }

  CellFacets<Real> F_;
  LView<Real> mu_, mass_;
  std::vector<double> mh_;
  bool viscous_ = false, useAmg_ = false, assembled_ = false, builtAssembled_ = false;
  double nuH2_ = 0, setupSeconds_ = 0;
  Real cScale_ = 0;
  peclet::core::solver::GraphAMGDevice lrho_, visc_;
  LView<Real> cc_, ccMin_, r_, z_, zOld_, p_, s_, w_, q_, ir_, iz_, ip_, iq_, zdiag_, c0_, c1_,
      rStar_;
  /// A: FGMRES bases, Krylov (V) and preconditioned (Z), velocity (3N) and multiplier (N) blocks
  std::vector<LView<Real>> kv_, km_, zv_, zm_;
  DissipationScratch<Real> scr_;
};

/// The physical inputs of the integrator beyond the potential: per-type viscosity and density
/// (host tables, indexed by the cell type) and a body acceleration a(x) = amp·sin(k·x).
struct FlowParams {
  std::vector<double> viscosity;  ///< nTypes (μ ≥ 0)
  std::vector<double> density;    ///< nTypes (ρ > 0; the mass is ρ V⁰)
  double bodyAmp[3] = {0, 0, 0};
  double bodyWave[3] = {0, 0, 0};
  /// no-slip wall velocity u_w(y) = wallVelocity + wallGradient·y (row-major ∂u_a/∂y_b) at the
  /// wall-facet centroid y, tangential part only; zero: fixed walls
  double wallVelocity[3] = {0, 0, 0};
  double wallGradient[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct IntegratorParams {
  double theta = 0.5;
  SaddleParams saddle;
  double weightTol = 1e-12;  ///< max |V − V⁰|/V⁰ of the per-step weight solve
  int weightMaxIter = 50;
  double hTol = 1e-10;  ///< relative residual of the warm-start and Newton H solves
  /// relative residual of the π solve: ∇Ũ = ∂_xU + J_xᵀπ cancels O(ρ g z A) terms to its round-off,
  /// and at 1e-12 the π error was its floor (rest: 3e-13 vs 9e-16 at 1e-15, +3 PCG iterations;
  /// M7 WO4a)
  double adjointTol = 1e-14;
  int maxHalvings = 4;  ///< τ halvings when the weight solve fails (W-Q7)
  /// option S (the default since M7 WO4′, orchestrator decision 2026-11-08: second order in the
  /// energy, ×4.2 per τ halving vs ×2.4 at x^n): kick 1's C at x^n + (τ/2) v^n; false: at x^n
  bool midpointConstraint = true;
  double repairTol = 1e-10;  ///< repaired-topology volume check (cone fan vs engine), relative
  /// AMG reuse (M7 WO-PERF): the S₀ (L_ρ) and H hierarchies are kept as preconditioners for at most
  /// this many further setups, and rebuilt early when their solve needs more than 1.5× (+2) the
  /// iterations it took right after the build. 0: rebuilt at every setup (the WO4 behaviour).
  int amgReuse = 0;
  /// relative residual of the linear warm-start H solves (x^{n+1} and the option-S midpoint) and
  /// of the Newton H solves (the inexact-Newton forcing; the volumes still close to weightTol)
  double warmTol = 1e-10, newtonTol = 1e-10;
  // ---- the gauge G″ (DESIGN_NOTE §12.3.2) ----
  int gaugeEvery = 1;              ///< the gauge after kick 2 every this many steps (0: off)
  double gaugeStrain = 3;          ///< c_g: α_i = min(α_max, c_g τ ε̂_i)
  double gaugeAlphaMax = 0.2;      ///< α_max
  double gaugeSkip = 1e-6;         ///< skip the gauge (state untouched) when max_i α_i < this
  int gaugeRetries = 3;            ///< halvings of all α_i on a rejected trial before the skip
  double kernelLength = 2;         ///< ℓ/h_i of the interface-normal kernel (§2.2; support 3ℓ)
  double normalLloydShift = 0.25;  ///< W-Q2: a cell of I with an unlike |s|/d above this also
                                   ///< gets the normal Lloyd part m̂m̂ᵀ(b_i − x_i)
};

/// Per-step record (energies are the values AFTER the step).
struct StepReport {
  double dt = 0, time = 0;
  double kinetic = 0, potential = 0, heat = 0, constraintWork = 0, bodyWork = 0;
  double wrapOffset = 0;  ///< accumulated periodic-in-g energy jumps (Ũ + wrapOffset continuous)
  double maxSpeed = 0;
  double maxShift = 0, maxShiftUnlike = 0;  ///< max |s_f|/d_f, all facets / unlike facets
  SaddleStats kick1, kick2;
  int warmIters = 0, newtonIters = 0, evaluations = 0, coldRebuilds = 0, adjIters = 0;
  double weightErr = 0;
  int halvings = 0;
  int unmatched = 0;          ///< one-sided facets dropped at x^{n+1} (matchFacets)
  int repairFallbacks = 0;    ///< cold rebuilds after the repaired topology failed the check
  int coldStarts = 0;         ///< weight solves restarted from w^n after the warm start failed
  double repairMismatch = 0;  ///< that check's value (max relative volume difference)
  double unmatchedArea = 0;   ///< largest such area
  double tKick1 = 0, tKick2 = 0, tRepair = 0, tWeights = 0, tPotential = 0, tTotal = 0;
  /// tPotential's split: facet export + match, U and ∂U, the H hierarchy, the π solve (the rest is
  /// the L_ρ setup, charged to kick2.tSetup)
  double tFacets = 0, tEnergy = 0, tAmgH = 0, tAdjoint = 0;
  double minMargin =
      0;  ///< min_i m_i after the step (cellMargins; > 0: every seed inside its cell)
  // the gauge (zero when it did not run)
  int gaugeTries = 0;          ///< trials evaluated (0: skipped or off this step)
  bool gaugeAccepted = false;  ///< a trial was accepted
  int gaugeNormal = 0;         ///< cells of I with the W-Q2 normal Lloyd part
  int gaugeInterface = 0;      ///< |I|
  int gaugeProjected = 0;      ///< cells whose ascent component was removed (step 3)
  int gaugeCapped = 0;         ///< cells scaled down by the margin cap (step 4)
  double gaugeAlphaMax = 0;    ///< max_i α_i (computed also when the gauge is skipped)
  double gaugePotential = 0;   ///< ΔŨ of the accepted trial (≤ 0)
  double gaugeKinetic = 0;     ///< ΔK of the momentum projection (≤ 0)
  int gaugeSolveFails = 0;     ///< trials whose weight solve failed
  int gaugeMarginRejects = 0;  ///< trials with a seed on or behind its own facet
  int gaugeEnergyRejects = 0;  ///< trials with Ũ(x̃) > Ũ(x)
  double gaugeTrialDU = 0;     ///< ΔŨ of the last evaluated trial (diagnostics)
  double tGauge = 0;
};

/// Parameters of the t = 0 relaxation R0 (DESIGN_NOTE §12.3.1; the values of the P4 pilot).
struct RelaxParams {
  int maxIter = 5000;  ///< descent iterations at most
  int window = 250;    ///< the convergence test compares F over this many iterations
  double tol = 1e-6;   ///< stop when |ΔF|/F < tol over one window
  double kappa = 100;  ///< volume-penalty stiffness κ: F = E_σ + (κσh²/2) Σ ((V−V⁰)/V⁰)²
  double eta = 0.1;    ///< the descent step δ = −(η/κ) ∇F / σ
  double maxShift = 0.15;    ///< step-4 check: max |s_f|/d_f after the OT solve
  double volumeTol = 1e-12;  ///< step-4 check: max |V − V⁰|/V⁰ after the OT solve
};

/// Result of Integrator::relax (R0).
struct RelaxReport {
  int iterations = 0;          ///< descent iterations taken
  bool converged = false;      ///< the |ΔF|/F window test stopped the descent (else maxIter)
  std::vector<double> energy;  ///< F at iteration 0, window, 2·window, ... (and at the stop)
  std::vector<double> area;    ///< the unlike-facet area at the same iterations (Voronoi cells)
  int otIterations = 0;        ///< the closing OT Newton solve (from w = 0)
  bool otConverged = false;
  double maxVolumeError = 0;  ///< step-4 checks at the solved power cells
  double maxShift = 0;
  double minMargin = 0;
  double unlikeArea = 0;  ///< the unlike-facet area of the solved power cells
  bool passed = false;    ///< all step-4 checks hold
};

/// NHV-W on a resident POWER MovingTessellation `MT`. Holds the velocity, the reference volumes
/// and the state at x^n (facets, ∇Ũ, π, the L_ρ and H hierarchies) between steps.
template <class MT>
class Integrator {
 public:
  using Real = double;
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  static_assert(std::is_same_v<typename MT::PlanePolicy, Power>,
                "the NHV-W integrator needs a power tessellation");

  /// `pos`, `weight` are the engine's views (updated in place); `V0` the reference volumes (sum
  /// = the tessellated volume); `spec` the potential; `vel` (3N) the initial velocity.
  Integrator(MT& mt, LView<Real> pos, LView<Real> weight, const Real L[3], std::vector<double> V0,
             const PotentialSpec<Real>& spec, const FlowParams& fp, const std::vector<double>& vel,
             const IntegratorParams& prm = {})
      : mt_(&mt), pos_(pos), weight_(weight), V0h_(std::move(V0)), spec_(spec), fp_(fp), prm_(prm) {
    N_ = static_cast<int>(V0h_.size());
    for (int a = 0; a < 3; ++a)
      L_[a] = L[a];
    const auto th = peclet::core::toVector(spec.types);
    std::vector<double> rho(N_), mu(N_), mass(N_), h(N_);
    maxV0_ = 0;
    for (int i = 0; i < N_; ++i) {
      const int t = th[i];
      if (t < 0 || t >= static_cast<int>(fp.density.size()) ||
          t >= static_cast<int>(fp.viscosity.size()))
        throw std::invalid_argument("voro: NHV-W: a cell type has no density / viscosity");
      rho[i] = fp.density[t];
      mu[i] = fp.viscosity[t];
      mass[i] = rho[i] * V0h_[i];
      h[i] = std::cbrt(V0h_[i]);
      maxV0_ = std::max(maxV0_, V0h_[i]);
    }
    rho_ = peclet::core::toDevice<Real>(rho, "nhvw.rho");
    V0_ = peclet::core::toDevice<Real>(V0h_, "nhvw.V0");
    mu_ = peclet::core::toDevice<Real>(mu, "nhvw.mu");
    mass_ = peclet::core::toDevice<Real>(mass, "nhvw.mass");
    h_ = peclet::core::toDevice<Real>(h, "nhvw.h");
    vel_ = peclet::core::toDevice<Real>(vel, "nhvw.v");
    const std::size_t n = static_cast<std::size_t>(N_), n3 = 3 * n;
    vh_ = LView<Real>("nhvw.vhalf", n3);
    b_ = LView<Real>("nhvw.b", n3);
    zo_ = LView<Real>("nhvw.zo", n3);
    zn_ = LView<Real>("nhvw.zn", n3);
    grad_ = LView<Real>("nhvw.grad", n3);
    gx_ = LView<Real>("nhvw.gx", n3);
    gw_ = LView<Real>("nhvw.gw", n);
    pi_ = LView<Real>("nhvw.pi", n);
    lam1_ = LView<Real>("nhvw.lam1", n);
    lam2_ = LView<Real>("nhvw.lam2", n);
    tmpN_ = LView<Real>("nhvw.tmpN", n);
    dw_ = LView<Real>("nhvw.dw", n);
    pos0_ = LView<Real>("nhvw.pos0", n3);
    w0_ = LView<Real>("nhvw.w0", n);
    wrapS_ = LView<Real>("nhvw.wrap", n3);
    gd_ = LView<Real>("nhvw.gaugeDelta", n3);
    aI_ = LView<Real>("nhvw.areaVec", n3);
    lamG_ = LView<Real>("nhvw.lamGauge", n);
    gs_ = LView<Real>("nhvw.gradV", 9 * n);
    cls_ = Kokkos::View<int*, Mem>("nhvw.gaugeClass", n);
    aG_ = LView<Real>("nhvw.gaugeAlpha", n);
    mG_ = LView<Real>("nhvw.margin", n);
    iList_ = Kokkos::View<int*, Mem>("nhvw.iList", n);
    const double* ug = fp_.wallGradient;
    movingWall_ = fp_.wallVelocity[0] != 0 || fp_.wallVelocity[1] != 0 ||
                  fp_.wallVelocity[2] != 0 ||
                  std::any_of(ug, ug + 9, [](double x) { return x != 0; });
    if (movingWall_)
      fw_ = LView<Real>("nhvw.wallForce", n3);
    evalState();
    kinetic_ = kineticEnergy();
  }

  /// Advance one step of size `tau` (halved on a weight-solve failure, W-Q7).
  StepReport step(double tau) {
    StepReport rep;
    Kokkos::Timer total;
    Kokkos::deep_copy(pos0_, pos_);
    Kokkos::deep_copy(w0_, weight_);
    for (int att = 0;; ++att) {
      if (attempt(tau, rep))
        break;
      if (att >= prm_.maxHalvings)
        throw std::runtime_error(
            "voro: NHV-W: the weight solve failed after " + std::to_string(att) +
            " step halvings (max |V - V0|/V0 = " + std::to_string(rep.weightErr) + ")");
      // restore x^n, w^n and the engine; the state at x^n is untouched
      restoreEngine();
      tau *= 0.5;
      ++rep.halvings;
    }
    time_ += tau;
    if (prm_.gaugeEvery > 0 && ++stepCount_ % prm_.gaugeEvery == 0) {
      double dU = 0, dK = 0;
      gauge(tau, rep, dU, dK);
      gaugeU_ += dU;
      gaugeK_ += dK;
    }
    Kokkos::fence();
    rep.tTotal = total.seconds();
    rep.dt = tau;
    rep.time = time_;
    rep.kinetic = kinetic_;
    rep.potential = parts_.total();
    rep.heat = heat_;
    rep.constraintWork = qc_;
    rep.bodyWork = work_;
    rep.wrapOffset = wrapOffset_;
    rep.maxSpeed = maxSpeed();
    shifts(rep.maxShift, rep.maxShiftUnlike);
    rep.minMargin = minMargin();
    return rep;
  }

  /// τ = min(C_cfl min h_i/|v_i|, C_σ √(ρ_σ h_min³ / σ_max), τ_user) (DESIGN_NOTE §2.7 step 5);
  /// a limit whose constant is 0 is off. ρ_σ is the least density over the cells with a
  /// tensioned unlike facet. +inf when nothing limits.
  double stableStep(double cfl, double capillary, double tauUser) const {
    double tau = tauUser > 0 ? tauUser : std::numeric_limits<double>::infinity();
    const LView<Real> v = vel_, h = h_;
    if (cfl > 0) {
      Real m = 0;
      Kokkos::parallel_reduce(
          "nhvw.cfl", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& acc) {
            const Real s = Kokkos::sqrt(v(3 * i) * v(3 * i) + v(3 * i + 1) * v(3 * i + 1) +
                                        v(3 * i + 2) * v(3 * i + 2));
            const Real r = s / h(i);
            acc = r > acc ? r : acc;
          },
          Kokkos::Max<Real>(m));
      if (m > 0)
        tau = std::min(tau, cfl / m);
    }
    if (capillary > 0)
      tau = std::min(tau, capillaryStep(capillary));
    return tau;
  }

  /// C_σ √(ρ_σ h_min³ / σ_max) (+inf without tension).
  double capillaryStep(double capillary) const {
    if (spec_.tension.extent(0) == 0)
      return std::numeric_limits<double>::infinity();
    const auto ten = peclet::core::toVector(spec_.tension);
    double smax = 0;
    for (double s : ten)
      smax = std::max(smax, s);
    if (!(smax > 0))
      return std::numeric_limits<double>::infinity();
    const CellFacets<Real> G = F_;
    const auto T = spec_.types;
    const LView<Real> ten_ = spec_.tension, rho = rho_;
    const int nT = spec_.nTypes;
    Real rmin = std::numeric_limits<Real>::infinity();
    Kokkos::parallel_reduce(
        "nhvw.rhoSigma", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          bool iface = false;
          for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f)
            iface = iface || ten_(T(i) * nT + T(G.nbr(f))) > Real(0);
          if (iface && rho(i) < acc)
            acc = rho(i);
        },
        Kokkos::Min<Real>(rmin));
    if (!(rmin < std::numeric_limits<Real>::infinity()))
      return std::numeric_limits<double>::infinity();
    const auto hh = peclet::core::toVector(h_);
    const double hmin = *std::min_element(hh.begin(), hh.end());
    return capillary * std::sqrt(rmin * hmin * hmin * hmin / smax);
  }

  /// max_i ν_i / h_i² (for the θ rule of §2.7).
  double maxNuOverH2() const {
    const auto r = peclet::core::toVector(rho_), m = peclet::core::toVector(mu_),
               h = peclet::core::toVector(h_);
    double x = 0;
    for (int i = 0; i < N_; ++i)
      x = std::max(x, m[i] / (r[i] * h[i] * h[i]));
    return x;
  }

  /// The t = 0 relaxation R0 (DESIGN_NOTE §12.3.1; replaces the power-cell descent of §2.8
  /// "Initialisation"). The reference volumes V⁰ are the caller's (the Lloyd volumes).
  ///  1. On VORONOI cells (w ≡ 0, a cold rebuild), steepest descent on all generators of
  ///     F(x) = E_σ(x) + (κσh²/2) Σ_i ((V_i − V⁰_i)/V⁰_i)²,  δ = −(η/κ) ∇F / σ,
  ///     E_σ the interface energy alone (no wall, no gravity term), σ the largest tension, h the
  ///     mean spacing (|Ω|/N)^{1/3}; ∇F = ∂_xE_σ + J_xᵀ (∂F/∂V) at w = 0, each iterate an
  ///     incremental repair. No guard and no Lloyd. A Voronoi generator cannot leave its cell.
  ///     Stop after maxIter iterations, or when |ΔF|/F < tol over `window` iterations.
  ///  2. One OT Newton solve from w = 0 to V(x, w) = V⁰ (cold rebuilds, the §10.2 tolerance), then
  ///     the state (facets, ∇Ũ, π) at the solution. v is untouched.
  ///  3. The step-4 checks (RelaxReport::passed): max|V − V⁰|/V⁰ ≤ volumeTol, max |s_f|/d_f ≤
  ///     maxShift, min_i m_i > 0 (minMargin()). Call at t = 0 only; no energy counter is touched.
  RelaxReport relax(const RelaxParams& rp = {}) {
    RelaxReport r;
    fcValid_ = false;
    double sigma = 0;
    if (spec_.tension.extent(0) > 0)
      for (double s : peclet::core::toVector(spec_.tension))
        sigma = std::max(sigma, s);
    double vt = 0;
    for (double v : V0h_)
      vt += v;
    const double h = std::cbrt(vt / N_);
    PotentialSpec<Real> st = spec_;  // E_σ only
    st.sigmaWall = LView<Real>();
    st.density = LView<Real>();
    for (int a = 0; a < 3; ++a)
      st.gravity[a] = 0;
    Kokkos::deep_copy(weight_, Real(0));
    mt_->setWeights(weight_);
    mt_->rebuild(pos_);
    const Real c = rp.kappa * sigma * h * h;
    const Real step = sigma > 0 ? rp.eta / (rp.kappa * sigma) : 0;
    double Flast = 0;
    for (int it = 0;; ++it) {
      F_ = exportCellFacets<Real, Power>(mt_->store, pos_, weight_, N_, L_, mt_->wall, mt_->xRef,
                                         mt_->vol);
      matchFacets(F_);
      Kokkos::deep_copy(gx_, Real(0));
      Kokkos::deep_copy(gw_, Real(0));
      const PotentialParts pp = partitionPotential(*mt_, F_, pos_, weight_, L_, st, gx_, gw_);
      const LView<Real> vol = F_.volume, v0 = V0_, p = tmpN_;
      Real pen = 0;
      Kokkos::parallel_reduce(
          "nhvw.r0Penalty", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& acc) {
            const Real e = (vol(i) - v0(i)) / v0(i);
            p(i) = c * e / v0(i);  // ∂F/∂V_i
            acc += e * e;
          },
          pen);
      const double F = pp.interface + 0.5 * c * pen;
      r.iterations = it;
      const bool last = it >= rp.maxIter || step == 0;
      if (it % rp.window == 0 || last) {
        r.energy.push_back(F);
        r.area.push_back(unlikeArea());
      }
      if (it % rp.window == 0) {
        if (it > 0 && (!(F > 0) || std::fabs(F - Flast) < rp.tol * F)) {
          r.converged = true;
          break;
        }
        Flast = F;
      }
      if (last)
        break;
      applyVolumeJacobianT(F_, tmpN_, b_);
      const LView<Real> P = pos_, gx = gx_, jt = b_;
      const Kokkos::Array<Real, 3> L3 = {L_[0], L_[1], L_[2]};
      Kokkos::parallel_for(
          "nhvw.r0Step", Kokkos::RangePolicy<Exec>(0, 3 * N_), KOKKOS_LAMBDA(const int k) {
            const Real Lk = L3[k % 3];
            const Real x = P(k) - step * (gx(k) + jt(k));
            P(k) = x - Lk * Kokkos::floor(x / Lk);
          });
      mt_->step(pos_);
    }
    // ---- 2. OT from w = 0 ----
    Kokkos::deep_copy(weight_, Real(0));
    mt_->setWeights(weight_);
    mt_->rebuild(pos_);
    WeightSolveParams wp;
    wp.tol = prm_.weightTol;
    wp.maxIter = prm_.weightMaxIter;
    wp.pcgTol = prm_.hTol;
    const WeightSolveResult ws = solveWeights(*mt_, pos_, weight_, V0h_, L_, wp);
    r.otIterations = ws.iters;
    r.otConverged = ws.converged;
    // ---- 3. the state at the solution and the checks ----
    evalState();
    {
      const LView<Real> vol = F_.volume, v0 = V0_;
      Real e = 0;
      Kokkos::parallel_reduce(
          "nhvw.r0Vol", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& acc) {
            const Real q = Kokkos::fabs(vol(i) - v0(i)) / v0(i);
            acc = q > acc ? q : acc;
          },
          Kokkos::Max<Real>(e));
      r.maxVolumeError = e;
    }
    double sUnlike = 0;
    shifts(r.maxShift, sUnlike);
    r.minMargin = minMargin();
    r.unlikeArea = unlikeArea();
    r.passed = r.otConverged && r.maxVolumeError <= rp.volumeTol && r.maxShift <= rp.maxShift &&
               r.minMargin > 0;
    return r;
  }

  /// min_i m_i over the current facets (cellMargins).
  double minMargin() const {
    cellMargins(F_, tmpN_);
    const LView<Real> m = tmpN_;
    Real out = 0;
    Kokkos::parallel_reduce(
        "nhvw.minMargin", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc = m(i) < acc ? m(i) : acc; },
        Kokkos::Min<Real>(out));
    return out;
  }

  /// m (N) = the distance from x_i to its own facet planes, m_i = min_f (d_f/2 + s_{i,f}),
  /// s_{i,f} = (w_i − w_j)/(2d_f) (positive away from i), over i's interior facets (DESIGN_NOTE
  /// §12.3.2 step 4); m_i ≤ 0: the generator is on or behind a facet of its own cell.
  void cellMargins(const CellFacets<Real>& F, const LView<Real>& m) const {
    const CellFacets<Real> G = F;
    const LView<Real> w = weight_;
    Kokkos::parallel_for(
        "nhvw.margins", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
          Real mi = Kokkos::Experimental::finite_max_v<Real>;
          for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
            Real d2 = 0;
            for (int c = 0; c < 3; ++c)
              d2 += G.conn(3 * f + c) * G.conn(3 * f + c);
            const Real d = Kokkos::sqrt(d2);
            const Real q = Real(0.5) * d + (w(i) - w(G.nbr(f))) / (Real(2) * d);
            mi = q < mi ? q : mi;
          }
          m(i) = mi;
        });
  }

  /// Σ of the unlike-facet areas (each facet once).
  double unlikeArea() const {
    const CellFacets<Real> G = F_;
    const auto T = spec_.types;
    Real a = 0;
    Kokkos::parallel_reduce(
        "nhvw.unlikeArea", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f)
            if (T(G.nbr(f)) != T(i))
              acc += G.area(f);
        },
        a);
    return 0.5 * a;
  }

  void setTheta(double th) { prm_.theta = th; }
  void setMidpointConstraint(bool on) { prm_.midpointConstraint = on; }
  /// Solver knobs of the M7 WO4a trials (no numerics default changes): the assembled Schur
  /// preconditioner and the relative tolerance of the π (adjoint) solve.
  void setSolverOptions(bool schurAssembled, double adjointTol, double saddleTol) {
    prm_.saddle.schurAssembled = schurAssembled;
    prm_.adjointTol = adjointTol;
    prm_.saddle.tol = saddleTol;
  }
  /// M7 WO-PERF knobs: the saddle method, B's rotational χ, A-solve tolerances and AMG reuse.
  void setPerfOptions(SaddleMethod method, double rotational, int amgReuse, double warmTol,
                      double newtonTol) {
    prm_.saddle.method = method;
    prm_.saddle.rotational = rotational;
    prm_.amgReuse = amgReuse;
    prm_.warmTol = warmTol;
    prm_.newtonTol = newtonTol;
  }
  void setGauge(int every, double strain, double alphaMax) {
    prm_.gaugeEvery = every;
    prm_.gaugeStrain = strain;
    prm_.gaugeAlphaMax = alphaMax;
  }
  double theta() const { return prm_.theta; }
  double time() const { return time_; }
  void setTime(double t) { time_ = t; }
  /// Restore the energy accumulators (a re-prepared integrator continues the old run's totals).
  void setAccumulators(double heat, double qc, double work, double wrap, double gaugeU = 0,
                       double gaugeK = 0) {
    heat_ = heat;
    qc_ = qc;
    work_ = work;
    wrapOffset_ = wrap;
    gaugeU_ = gaugeU;
    gaugeK_ = gaugeK;
  }
  double gaugePotential() const { return gaugeU_; }  ///< Σ ΔŨ of the accepted gauge trials
  double gaugeKinetic() const { return gaugeK_; }    ///< Σ ΔK of the gauge's projections
  int restoreRebuilds() const { return restoreRebuilds_; }  ///< cold rebuilds by restoreEngine
  double kinetic() const { return kinetic_; }
  double heat() const { return heat_; }
  double constraintWork() const { return qc_; }
  double bodyWork() const { return work_; }
  double wrapOffset() const { return wrapOffset_; }
  const PotentialParts& potential() const { return parts_; }
  const LView<Real>& velocity() const { return vel_; }
  const LView<Real>& pi() const { return pi_; }
  const LView<Real>& lambda() const { return lam2_; }
  const LView<Real>& gradient() const { return grad_; }
  const CellFacets<Real>& facets() const { return F_; }
  int numCells() const { return N_; }

 private:
  /// From (pos0_, w0_) at the state x^n held in F_: x ← x^n + s·u (wrapped), weights warm-started
  /// by H δw = −s J_x u and solved by the repaired Newton. False if the weight solve failed.
  bool moveAndSolve(double s, const LView<Real>& u, StepReport& rep) {
    fcValid_ = false;
    applyVolumeJacobian(F_, u, tmpN_);
    {
      const LView<Real> r = tmpN_;
      Real sum = 0;
      Kokkos::parallel_reduce(
          "nhvw.msMean", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& acc) { acc += r(i); }, sum);
      const Real mean = sum / N_, ms = -s;
      Kokkos::parallel_for(
          "nhvw.msRhs", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i) { r(i) = i == 0 ? Real(0) : ms * (r(i) - mean); });
    }
    WeightSolveParams wp;
    wp.pcgTol = prm_.warmTol;
    wp.amg = &hAmg_;
    solvePinnedH(F_, tmpN_, dw_, wp);
    {
      const LView<Real> w = weight_, w0 = w0_, dw = dw_, P = pos_, P0 = pos0_, U = u, S = wrapS_;
      const Kokkos::Array<Real, 3> L3 = {L_[0], L_[1], L_[2]};
      const Real ss = s;
      Kokkos::parallel_for(
          "nhvw.msW", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i) { w(i) = w0(i) + dw(i); });
      Kokkos::parallel_for(
          "nhvw.msX", Kokkos::RangePolicy<Exec>(0, 3 * N_), KOKKOS_LAMBDA(const int k) {
            const Real Lk = L3[k % 3];
            const Real x = P0(k) + ss * U(k);
            const Real sh = Lk * Kokkos::floor(x / Lk);
            P(k) = x - sh;
            S(k) = sh;  // read only by the gauge's accumulateWrap (the drift rewrites it)
          });
    }
    wp.tol = prm_.weightTol;
    wp.maxIter = prm_.weightMaxIter;
    wp.repair = true;
    wp.pcgTol = prm_.newtonTol;
    const WeightSolveResult ws = solveWeights(*mt_, pos_, weight_, V0h_, L_, wp);
    rep.tRepair += ws.evalSeconds;
    rep.evaluations += ws.evaluations;
    if (ws.converged && repairMismatch() > prm_.repairTol) {
      ++rep.repairFallbacks;
      fcValid_ = false;
      mt_->rebuild(pos_);
      WeightSolveParams wc = wp;
      wc.repair = false;
      return solveWeights(*mt_, pos_, weight_, V0h_, L_, wc).converged;
    }
    return ws.converged;
  }

  /// max_i |V_i(export) − V_i(engine)| / V⁰_i of the resident topology at (pos, weight). The
  /// export is kept (fc_) for the next consumer at the same state (currentFacets()).
  double repairMismatch() {
    fc_ = exportCellFacets<Real, Power>(mt_->store, pos_, weight_, N_, L_, mt_->wall, mt_->xRef,
                                        mt_->vol);
    fcValid_ = true;
    const CellFacets<Real> F = fc_;
    const LView<Real> ve = mt_->vol, v0 = V0_, vf = F.volume;
    Real m = 0;
    Kokkos::parallel_reduce(
        "nhvw.repairCheck", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          const Real e = Kokkos::fabs(vf(i) - ve(i)) / v0(i);
          acc = e > acc ? e : acc;
        },
        Kokkos::Max<Real>(m));
    return m;
  }

  /// The facets at the current (pos, weight): the export repairMismatch() made at this state if it
  /// is still valid (every change of pos/weight/engine after it clears fcValid_), else a new one.
  CellFacets<Real> currentFacets() {
    if (fcValid_)
      return fc_;
    return exportCellFacets<Real, Power>(mt_->store, pos_, weight_, N_, L_, mt_->wall, mt_->xRef,
                                         mt_->vol);
  }

  /// saddle_.setup at F_, the S₀ hierarchy rebuilt or lagged per IntegratorParams::amgReuse.
  void setupSaddle() {
    const bool fresh = prm_.amgReuse <= 0 || sAge_ >= prm_.amgReuse || sStale_;
    saddle_.setAssembled(prm_.saddle.schurAssembled);
    saddle_.setup(F_, rho_, mu_, mass_, h_, fresh);
    if (fresh) {
      sAge_ = 0;
      sStale_ = false;
      sRef_ = -1;  // the next kick's outer count is the reference
    } else {
      ++sAge_;
    }
  }

  /// Facets, U and ∇Ũ, π, and the two hierarchies at the current (pos, weight).
  void evalState(double* tPot = nullptr) {
    Kokkos::Timer t, tp;
    F_ = currentFacets();
    match_ = matchFacets(F_);
    Kokkos::fence();
    evalSplit_[0] = tp.seconds();
    tp.reset();
    Kokkos::deep_copy(gx_, Real(0));
    Kokkos::deep_copy(gw_, Real(0));
    parts_ = partitionPotential(*mt_, F_, pos_, weight_, L_, spec_, gx_, gw_);
    Kokkos::fence();
    evalSplit_[1] = tp.seconds();
    tp.reset();
    peclet::core::solver::AmgParams ap;
    ap.ndofPerNode = 1;
    const bool hFresh =
        prm_.amgReuse <= 0 || hAge_ >= prm_.amgReuse || hStale_ || hAmg_.numLevels() == 0;
    if (hFresh) {
      hAmg_.build(detail::pinnedHostH(F_), ap);
      hAge_ = 0;
      hStale_ = false;
    } else {
      ++hAge_;
    }
    evalSplit_[2] = tp.seconds();
    tp.reset();
    WeightSolveParams wp;
    wp.pcgTol = prm_.adjointTol;
    wp.amg = &hAmg_;
    adj_ = reducedGradient(F_, gx_, gw_, pi_, grad_, wp);
    if (hFresh)
      hRef_ = adj_.pcgIters;
    else if (adj_.pcgIters > 1.5 * hRef_ + 2)
      hStale_ = true;
    Kokkos::fence();
    evalSplit_[3] = tp.seconds();
    setupSaddle();
    Kokkos::fence();
    if (tPot)
      *tPot = t.seconds();
  }

  /// One kick at the current state: vNew, lam from vOld; adds the heat and the body work.
  SaddleStats kick(double tau, const LView<Real>& vOld, const LView<Real>& vNew,
                   const LView<Real>& lam, const LView<Real>& posForce) {
    const Real th = prm_.theta, ht = Real(0.5) * tau;
    const bool visc = saddle_.viscous();
    const LView<Real> m = mass_, vo = vOld, b = b_, zo = zo_, g = grad_, P = posForce;
    if (visc)
      applyDissipation(F_, mu_, vOld, zo_, saddle_.scratch());
    else
      Kokkos::deep_copy(zo_, Real(0));
    const Kokkos::Array<Real, 3> amp = {fp_.bodyAmp[0], fp_.bodyAmp[1], fp_.bodyAmp[2]},
                                 kw = {fp_.bodyWave[0], fp_.bodyWave[1], fp_.bodyWave[2]};
    const bool body = amp[0] != 0 || amp[1] != 0 || amp[2] != 0;
    const bool mw = movingWall_ && visc;
    if (mw)
      wallForce();
    const LView<Real> fw = fw_;
    // b = M v − (1−θ)(τ/2) ζ v − (τ/2)(∇Ũ − M a(x) − f_w)
    Kokkos::parallel_for(
        "nhvw.rhs", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
          const Real s =
              body ? Kokkos::sin(kw[0] * P(3 * i) + kw[1] * P(3 * i + 1) + kw[2] * P(3 * i + 2))
                   : Real(0);
          for (int c = 0; c < 3; ++c) {
            const int k = 3 * i + c;
            b(k) = m(i) * vo(k) - (Real(1) - th) * ht * zo(k) - ht * (g(k) - m(i) * amp[c] * s);
            if (mw)
              b(k) += ht * fw(k);
          }
        });
    const Real tolAbs = prm_.saddle.tol * maxV0_ / tau;
    SaddleStats st = saddle_.solve(th * ht, ht, tolAbs, b_, vNew, lam, prm_.saddle);
    if (sRef_ < 0)
      sRef_ = st.outer;
    else if (st.outer > 1.5 * sRef_ + 2)
      sStale_ = true;
    st.constraint *= tau / maxV0_;
    // heat (τ/2) v̄_θᵀ ζ (v_old + v_new)/2 and body work (τ/2) (M a)ᵀ (v_old + v_new)/2
    const LView<Real> vn = vNew, zn = zn_;
    if (visc)
      applyDissipation(F_, mu_, vNew, zn_, saddle_.scratch());
    else
      Kokkos::deep_copy(zn_, Real(0));
    Real dq = 0, dw = 0;
    Kokkos::parallel_reduce(
        "nhvw.heat", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& aq, Real& aw) {
          const Real s =
              body ? Kokkos::sin(kw[0] * P(3 * i) + kw[1] * P(3 * i + 1) + kw[2] * P(3 * i + 2))
                   : Real(0);
          for (int c = 0; c < 3; ++c) {
            const int k = 3 * i + c;
            aq += (th * vn(k) + (Real(1) - th) * vo(k)) * Real(0.5) * (zo(k) + zn(k));
            aw += m(i) * amp[c] * s * Real(0.5) * (vo(k) + vn(k));
            if (mw)
              aw += fw(k) * Real(0.5) * (vo(k) + vn(k));
          }
        },
        dq, dw);
    heat_ += ht * dq;
    work_ += ht * dw;
    return st;
  }

  /// Steps 1–4 at τ; false if the weight solve failed (state at x^n untouched except pos/weight
  /// and the engine, which the caller restores).
  bool attempt(double tau, StepReport& rep) {
    Kokkos::Timer t;
    fcValid_ = false;
    const double heat0 = heat_, work0 = work_, qc0 = qc_;  // kick 1's increments undone on a retry
    // ---- kick 1 at x^n (λ₁ warm-started from the last λ₂, same geometry) ----
    Kokkos::deep_copy(lam1_, lam2_);
    if (prm_.midpointConstraint) {
      // option S (DESIGN_NOTE §10.3.6): C of kick 1 (and its ζ) at x^n + (τ/2)v^n, the weights
      // solved there (repair, warm start H δw = −(τ/2) J_x v^n); ∇Ũ and the body force stay at x^n
      const CellFacets<Real> Fn = F_;
      if (!moveAndSolve(0.5 * tau, vel_, rep)) {
        heat_ = heat0;
        work_ = work0;
        qc_ = qc0;
        return false;
      }
      F_ = currentFacets();
      fcValid_ = false;  // pos/weight return to x^n below
      matchFacets(F_);
      setupSaddle();
      rep.kick1 = kick(tau, vel_, vh_, lam1_, pos0_);
      // kick 1's constraint work (τ/4) λ₁ᵀ C^s v^n (C^s v^{½} = 0, C^s v^n ≠ 0)
      applyDivergence(F_, vel_, tmpN_);
      qc_ += 0.25 * tau * detail::dotView(lam1_, tmpN_);
      F_ = Fn;
      Kokkos::deep_copy(pos_, pos0_);
      Kokkos::deep_copy(weight_, w0_);
    } else {
      rep.kick1 = kick(tau, vel_, vh_, lam1_, pos_);
    }
    Kokkos::fence();
    rep.tKick1 = t.seconds();
    t.reset();
    // ---- warm start H δw = −τ J_x v^{½} (at x^n) ----
    {
      applyVolumeJacobian(F_, vh_, tmpN_);
      const LView<Real> r = tmpN_;
      Real s = 0;
      Kokkos::parallel_reduce(
          "nhvw.warmMean", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& acc) { acc += r(i); }, s);
      const Real mean = s / N_, mt = -tau;
      Kokkos::parallel_for(
          "nhvw.warmRhs", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i) { r(i) = i == 0 ? Real(0) : mt * (r(i) - mean); });
      WeightSolveParams wp;
      wp.pcgTol = prm_.warmTol;
      wp.amg = &hAmg_;
      rep.warmIters = solvePinnedH(F_, tmpN_, dw_, wp).first;
      const LView<Real> w = weight_, dw = dw_;
      Kokkos::parallel_for(
          "nhvw.warm", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i) { w(i) += dw(i); });
    }
    // ---- drift (wrapped into the box; the lattice shift is kept for the gravity jump) ----
    {
      const LView<Real> P = pos_, v = vh_, S = wrapS_;
      const Kokkos::Array<Real, 3> L3 = {L_[0], L_[1], L_[2]};
      Kokkos::parallel_for(
          "nhvw.drift", Kokkos::RangePolicy<Exec>(0, 3 * N_), KOKKOS_LAMBDA(const int k) {
            const Real Lk = L3[k % 3];
            const Real x = P(k) + tau * v(k);
            const Real sh = Lk * Kokkos::floor(x / Lk);
            P(k) = x - sh;
            S(k) = sh;
          });
    }
    // ---- weights by damped Newton, every iterate a repair of the resident engine ----
    WeightSolveParams wp;
    wp.tol = prm_.weightTol;
    wp.maxIter = prm_.weightMaxIter;
    wp.pcgTol = prm_.newtonTol;
    wp.repair = true;
    wp.amg = &hAmg_;
    WeightSolveResult ws = solveWeights(*mt_, pos_, weight_, V0h_, L_, wp);
    if (!ws.converged) {
      // the linear warm start can overshoot into a buried cell, from which the damped Newton
      // cannot recover (every trial keeps it empty): retry once from w^n at the same x^{n+1}
      ++rep.coldStarts;
      Kokkos::deep_copy(weight_, w0_);
      const WeightSolveResult w2 = solveWeights(*mt_, pos_, weight_, V0h_, L_, wp);
      ws.evaluations += w2.evaluations;
      ws.evalSeconds += w2.evalSeconds;
      ws.iters += w2.iters;
      ws.converged = w2.converged;
      ws.maxRelErr = w2.maxRelErr;
    }
    rep.newtonIters = ws.iters;
    rep.evaluations += ws.evaluations;
    rep.coldRebuilds = ws.coldRebuilds;
    rep.weightErr = ws.maxRelErr;
    rep.tRepair += ws.evalSeconds;
    Kokkos::fence();
    rep.tWeights = t.seconds() - ws.evalSeconds;
    if (!ws.converged) {
      heat_ = heat0;
      work_ = work0;
      qc_ = qc0;
      return false;
    }
    // ---- guard: the repaired topology must reproduce the engine's volumes (a stale face the
    // repair certificate missed shows as a cone-fan volume off the engine's); else cold-rebuild
    // and re-solve the weights from it ----
    rep.repairMismatch = repairMismatch();
    if (rep.repairMismatch > prm_.repairTol) {
      ++rep.repairFallbacks;
      fcValid_ = false;
      mt_->rebuild(pos_);
      WeightSolveParams wc = wp;
      wc.repair = false;
      const WeightSolveResult wr = solveWeights(*mt_, pos_, weight_, V0h_, L_, wc);
      rep.newtonIters += wr.iters;
      rep.weightErr = wr.maxRelErr;
      if (!wr.converged) {
        heat_ = heat0;
        work_ = work0;
        qc_ = qc0;
        return false;
      }
    }
    // ---- π, ∇Ũ and the hierarchies at x^{n+1} ----
    double tPot = 0;
    evalState(&tPot);
    rep.tPotential = tPot;
    rep.tFacets = evalSplit_[0];
    rep.tEnergy = evalSplit_[1];
    rep.tAmgH = evalSplit_[2];
    rep.tAdjoint = evalSplit_[3];
    rep.adjIters = adj_.pcgIters;
    rep.unmatched = match_.dropped;
    rep.unmatchedArea = match_.maxArea;
    accumulateWrap();
    // ---- kick 2 at x^{n+1}; constraint work (τ/4) λ₂ᵀ C^{n+1} v^{½} ----
    t.reset();
    Kokkos::deep_copy(lam2_, lam1_);
    rep.kick2 = kick(tau, vh_, vel_, lam2_, pos_);
    applyDivergence(F_, vh_, tmpN_);
    qc_ += 0.25 * tau * detail::dotView(lam2_, tmpN_);
    kinetic_ = kineticEnergy();
    Kokkos::fence();
    rep.tKick2 = t.seconds();
    return true;
  }

  /// The gauge G″ (DESIGN_NOTE §12.3.2) at x = x^{n+1}, v = v^{n+1}: strain-activated, Lloyd-only,
  /// ascent-free; it never descends Ũ (D12.1, D12.2).
  ///  1. activation  ε̇_i = ‖sym(G_i v)‖_F (the linear-exact cell gradient), ε̂_i = its max over i
  ///                 and i's face neighbours, α_i = min(α_max, c_g τ ε̂_i). max_i α_i < gaugeSkip:
  ///                 nothing else runs, the state is untouched bit for bit.
  ///  2. raw move    sets: CL = cells with both a wall facet and an unlike facet; I = the other
  ///                 cells with an unlike facet; the rest. CL: r_i = 0. I: r_i = P_i(b_i − x_i),
  ///                 plus m̂_im̂_iᵀ(b_i − x_i) when an unlike facet of i has |s|/d >
  ///                 normalLloydShift (W-Q2), P_i = I − m̂_im̂_iᵀ with m̂ the §2.8 step-1 kernel
  ///                 normal (m_i = Σ_{k∈I, t_k=t_i, r<3ℓ} w(r) a_k; â_i if |m_i| < 0.1|a_i|).
  ///                 The rest: r_i = b_i − x_i (b_i the centroid of the clipped power cell).
  ///                 δ_i = α_i r_i.
  ///  3. projection  g_i = ∇_iŨ: if g_i·δ_i > 0, δ_i −= (g_i·δ_i/|g_i|²) g_i (ΔŨ ≤ 0 to first
  ///                 order, cell by cell).
  ///  4. margin cap  |δ_i| ≤ ½ m_i (cellMargins; δ_i = 0 if m_i ≤ 0).
  ///  5. trial       x̃ = x + δ, OT warm-started from w^{n+1} (moveAndSolve); reject if OT fails,
  ///                 min_i m_i(x̃) ≤ 0 or Ũ(x̃) > Ũ(x) (with the periodic-in-g wrap jump); on a
  ///                 rejection halve all α_i (δ scales linearly), at most gaugeRetries times, then
  ///                 skip (x, w and the engine restored).
  ///  6. acceptance  v_i travels with its cell; the M-projection onto C(x̃)v = 0 (one saddle with
  ///                 A = M); ∇Ũ(x̃) recomputed (evalState).
  /// dU, dK: the accepted trial's ΔŨ ≤ 0 and ΔK ≤ 0 (0 when skipped).
  void gauge(double tau, StepReport& rep, double& dU, double& dK) {
    Kokkos::Timer t;
    dU = dK = 0;
    // ---- 1. activation ----
    cellGradient(F_, vel_, gs_);
    {
      const LView<Real> gs = gs_, e = tmpN_;
      Kokkos::parallel_for(
          "nhvw.gaugeStrain", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
            Real q = 0;
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b) {
                const Real sab = Real(0.5) * (gs(9 * i + 3 * a + b) + gs(9 * i + 3 * b + a));
                q += sab * sab;
              }
            e(i) = Kokkos::sqrt(q);
          });
    }
    {
      const CellFacets<Real> G = F_;
      const LView<Real> e = tmpN_, al = aG_;
      const Real cg = prm_.gaugeStrain * tau, amax = prm_.gaugeAlphaMax;
      Real am = 0;
      Kokkos::parallel_reduce(
          "nhvw.gaugeAlpha", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& acc) {
            Real m = e(i);
            for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
              const Real q = e(G.nbr(f));
              m = q > m ? q : m;
            }
            const Real a = cg * m < amax ? cg * m : amax;
            al(i) = a;
            acc = a > acc ? a : acc;
          },
          Kokkos::Max<Real>(am));
      rep.gaugeAlphaMax = am;
      if (!(am >= prm_.gaugeSkip)) {
        rep.tGauge = t.seconds();
        return;
      }
    }
    // ---- 2-4. the displacement ----
    rep.gaugeInterface = gaugeDisplacement(rep);
    Kokkos::deep_copy(pos0_, pos_);
    Kokkos::deep_copy(w0_, weight_);
    const double U0 = parts_.total() + wrapOffset_;
    double s = 1;
    for (int k = 0; k <= prm_.gaugeRetries; ++k, s *= 0.5) {
      ++rep.gaugeTries;
      if (moveAndSolve(s, gd_, rep)) {
        CellFacets<Real> Ft = currentFacets();
        matchFacets(Ft);
        cellMargins(Ft, mG_);
        const LView<Real> mg = mG_;
        Real mmin = 0;
        Kokkos::parallel_reduce(
            "nhvw.gaugeTrialMargin", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i, Real& acc) { acc = mg(i) < acc ? mg(i) : acc; },
            Kokkos::Min<Real>(mmin));
        if (mmin > Real(0)) {
          Kokkos::deep_copy(gx_, Real(0));
          Kokkos::deep_copy(gw_, Real(0));
          const PotentialParts pt =
              partitionPotential(*mt_, Ft, pos_, weight_, L_, spec_, gx_, gw_);
          rep.gaugeTrialDU = pt.total() + wrapOffset_ - wrapJump() - U0;
          if (pt.total() + wrapOffset_ - wrapJump() <= U0) {
            accumulateWrap();
            evalState();
            dU = parts_.total() + wrapOffset_ - U0;
            rep.gaugeAccepted = true;
            break;
          }
          ++rep.gaugeEnergyRejects;
        } else {
          ++rep.gaugeMarginRejects;
        }
      } else {
        ++rep.gaugeSolveFails;
      }
      restoreEngine();
    }
    if (rep.gaugeAccepted && maxSpeed() > 0) {
      const double K0 = kinetic_;
      const LView<Real> b = b_, v = vel_, m = mass_;
      Kokkos::parallel_for(
          "nhvw.gaugeMv", Kokkos::RangePolicy<Exec>(0, 3 * N_),
          KOKKOS_LAMBDA(const int k) { b(k) = m(k / 3) * v(k); });
      Kokkos::deep_copy(lamG_, Real(0));
      saddle_.solve(Real(0), Real(0.5 * tau), Real(prm_.saddle.tol * maxV0_ / tau), b_, vel_, lamG_,
                    prm_.saddle);
      kinetic_ = kineticEnergy();
      dK = kinetic_ - K0;
    }
    rep.gaugePotential = dU;
    rep.gaugeKinetic = dK;
    Kokkos::fence();
    rep.tGauge = t.seconds();
  }

  /// G″ steps 2-4 (see gauge()): δ (gd_) from α (aG_); returns |I| and counts the normal-part,
  /// projected and capped cells into `rep`.
  int gaugeDisplacement(StepReport& rep) {
    const CellFacets<Real> G = F_;
    const auto T = spec_.types;
    const LView<Real> a = aI_, g = grad_, d = gd_, h = h_, P = pos_, w = weight_, al = aG_,
                      mg = mG_;
    const Kokkos::View<int*, Mem> cls = cls_, list = iList_;
    // classes: 0 the rest, 1 I, 2 CL (an unlike and a wall facet)
    Kokkos::parallel_for(
        "nhvw.gaugeSets", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
          Real ai[3] = {0, 0, 0};
          bool iface = false;
          for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
            if (T(G.nbr(f)) == T(i))
              continue;
            iface = true;
            Real d2 = 0;
            for (int c = 0; c < 3; ++c)
              d2 += G.conn(3 * f + c) * G.conn(3 * f + c);
            const Real sc = G.area(f) / Kokkos::sqrt(d2);
            for (int c = 0; c < 3; ++c)
              ai[c] += sc * G.conn(3 * f + c);
          }
          cls(i) = !iface ? 0 : (G.wallOffset(i + 1) > G.wallOffset(i) ? 2 : 1);
          for (int e = 0; e < 3; ++e)
            a(3 * i + e) = ai[e];
        });
    int nI = 0;
    Kokkos::parallel_scan(
        "nhvw.gaugeList", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
          if (cls(i) == 1) {
            if (fin)
              list(upd) = i;
            ++upd;
          }
        },
        nI);
    cellMargins(F_, mG_);
    const int n = nI;
    const Kokkos::Array<Real, 3> L3 = {L_[0], L_[1], L_[2]};
    const Real kl = prm_.kernelLength, sThr = prm_.normalLloydShift;
    int nNormal = 0, nProj = 0, nCap = 0;
    // the kernel sum is a brute-force loop over I (|I| ~ N^{2/3}); a cell list is the upgrade
    Kokkos::parallel_reduce(
        "nhvw.gaugeDelta", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, int& cN, int& cP, int& cC) {
          Real di[3] = {0, 0, 0};
          if (cls(i) == 0) {
            for (int c = 0; c < 3; ++c)
              di[c] = G.cellCentroid(3 * i + c);
          } else if (cls(i) == 1) {
            const Real ell2 = kl * h(i) * kl * h(i), rc2 = Real(9) * ell2;
            Real m[3] = {0, 0, 0};
            for (int q = 0; q < n; ++q) {
              const int k = list(q);
              if (T(k) != T(i))
                continue;
              Real r2 = 0;
              for (int c = 0; c < 3; ++c) {
                Real dx = P(3 * k + c) - P(3 * i + c);
                dx -= L3[c] * Kokkos::floor(dx / L3[c] + Real(0.5));
                r2 += dx * dx;
              }
              if (r2 >= rc2)
                continue;
              const Real u = Real(1) - r2 / rc2;
              const Real wgt = Kokkos::exp(-r2 / ell2) * u * u * u;
              for (int c = 0; c < 3; ++c)
                m[c] += wgt * a(3 * k + c);
            }
            Real mn = Kokkos::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
            const Real an = Kokkos::sqrt(a(3 * i) * a(3 * i) + a(3 * i + 1) * a(3 * i + 1) +
                                         a(3 * i + 2) * a(3 * i + 2));
            if (mn < Real(0.1) * an) {  // thin film: the cell's own vector area
              for (int c = 0; c < 3; ++c)
                m[c] = a(3 * i + c);
              mn = an;
            }
            for (int c = 0; c < 3; ++c)
              di[c] = G.cellCentroid(3 * i + c);
            if (mn > Real(0)) {
              Real nh[3];
              for (int c = 0; c < 3; ++c)
                nh[c] = m[c] / mn;
              bool normal = false;
              for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
                const int j = G.nbr(f);
                if (T(j) == T(i))
                  continue;
                Real d2 = 0;
                for (int c = 0; c < 3; ++c)
                  d2 += G.conn(3 * f + c) * G.conn(3 * f + c);
                if (Kokkos::fabs(w(i) - w(j)) / (Real(2) * d2) > sThr)
                  normal = true;
              }
              if (!normal) {
                const Real bn = nh[0] * di[0] + nh[1] * di[1] + nh[2] * di[2];
                for (int c = 0; c < 3; ++c)
                  di[c] -= bn * nh[c];
              } else {
                ++cN;
              }
            }
          }
          // δ_i = α_i r_i
          for (int c = 0; c < 3; ++c)
            di[c] *= al(i);
          // 3. remove the ascent component
          const Real gi[3] = {g(3 * i), g(3 * i + 1), g(3 * i + 2)};
          const Real gd = gi[0] * di[0] + gi[1] * di[1] + gi[2] * di[2];
          const Real gg = gi[0] * gi[0] + gi[1] * gi[1] + gi[2] * gi[2];
          if (gd > Real(0) && gg > Real(0)) {
            for (int c = 0; c < 3; ++c)
              di[c] -= gd / gg * gi[c];
            ++cP;
          }
          // 4. the margin cap |δ_i| ≤ ½ m_i
          const Real dn = Kokkos::sqrt(di[0] * di[0] + di[1] * di[1] + di[2] * di[2]);
          const Real cap = Real(0.5) * mg(i);
          if (dn > cap) {
            const Real sc = cap > Real(0) ? cap / dn : Real(0);
            for (int c = 0; c < 3; ++c)
              di[c] *= sc;
            ++cC;
          }
          for (int c = 0; c < 3; ++c)
            d(3 * i + c) = di[c];
        },
        nNormal, nProj, nCap);
    rep.gaugeNormal = nNormal;
    rep.gaugeProjected = nProj;
    rep.gaugeCapped = nCap;
    return n;
  }

  /// pos ← pos0_, weight ← w0_ and the engine back to them: a repair, or a cold rebuild when the
  /// repaired volumes miss V⁰ by more than repairTol (a repair cannot revive a cell the failed
  /// trial buried; measured after the t = 0 relaxation, M7 WO4′).
  void restoreEngine() {
    fcValid_ = false;
    Kokkos::deep_copy(pos_, pos0_);
    Kokkos::deep_copy(weight_, w0_);
    mt_->setWeights(weight_);
    mt_->step(pos_);
    const LView<Real> ve = mt_->vol, v0 = V0_;
    Real m = 0;
    Kokkos::parallel_reduce(
        "nhvw.restoreCheck", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          const Real e = Kokkos::fabs(ve(i) - v0(i)) / v0(i);
          acc = e > acc ? e : acc;
        },
        Kokkos::Max<Real>(m));
    if (!(m <= prm_.repairTol)) {
      ++restoreRebuilds_;
      mt_->rebuild(pos_);
    }
  }

  /// f_w (3N) = Σ_w (μ_i A^w/δ_w) P_w u_w(y_w), y_w = x_i + c^w (moving no-slip walls).
  void wallForce() {
    const CellFacets<Real> G = F_;
    const LView<Real> f = fw_, mu = mu_, P = pos_;
    const Kokkos::Array<Real, 3> u0 = {fp_.wallVelocity[0], fp_.wallVelocity[1],
                                       fp_.wallVelocity[2]};
    Kokkos::Array<Real, 9> ug;
    for (int e = 0; e < 9; ++e)
      ug[e] = fp_.wallGradient[e];
    Kokkos::parallel_for(
        "nhvw.wallForce", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
          Real o[3] = {0, 0, 0};
          for (int w = G.wallOffset(i); w < G.wallOffset(i + 1); ++w) {
            const Real nw[3] = {G.wallNormal(3 * w), G.wallNormal(3 * w + 1),
                                G.wallNormal(3 * w + 2)};
            const Real dw = nw[0] * G.wallCentroid(3 * w) + nw[1] * G.wallCentroid(3 * w + 1) +
                            nw[2] * G.wallCentroid(3 * w + 2);
            if (!(dw > Real(0)))
              continue;
            const Real kw = mu(i) * G.wallArea(w) / dw;
            Real y[3], u[3];
            for (int c = 0; c < 3; ++c)
              y[c] = P(3 * i + c) + G.wallCentroid(3 * w + c);
            for (int c = 0; c < 3; ++c)
              u[c] = u0[c] + ug[3 * c] * y[0] + ug[3 * c + 1] * y[1] + ug[3 * c + 2] * y[2];
            const Real un = nw[0] * u[0] + nw[1] * u[1] + nw[2] * u[2];
            for (int c = 0; c < 3; ++c)
              o[c] += kw * (u[c] - un * nw[c]);
          }
          for (int c = 0; c < 3; ++c)
            f(3 * i + c) = o[c];
        });
  }

  /// Σ_i (ρ_i − ρ_ref) V⁰_i g·S_i of the last wrap shifts S (wrapS_); 0 without gravity.
  double wrapJump() const {
    if (spec_.density.extent(0) == 0)
      return 0;
    const LView<Real> S = wrapS_, rho = rho_;
    const Kokkos::Array<Real, 3> g = {spec_.gravity[0], spec_.gravity[1], spec_.gravity[2]};
    const Real rr = spec_.rhoRef;
    const LView<Real> v0 = V0_;
    Real j = 0;
    Kokkos::parallel_reduce(
        "nhvw.wrapJump", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          const Real gs = g[0] * S(3 * i) + g[1] * S(3 * i + 1) + g[2] * S(3 * i + 2);
          acc += (rho(i) - rr) * v0(i) * gs;
        },
        j);
    return j;
  }

  void accumulateWrap() { wrapOffset_ -= wrapJump(); }

  double kineticEnergy() const {
    const LView<Real> v = vel_, m = mass_;
    Real k = 0;
    Kokkos::parallel_reduce(
        "nhvw.K", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          acc += Real(0.5) * m(i) *
                 (v(3 * i) * v(3 * i) + v(3 * i + 1) * v(3 * i + 1) + v(3 * i + 2) * v(3 * i + 2));
        },
        k);
    return k;
  }

  double maxSpeed() const {
    const LView<Real> v = vel_;
    Real m = 0;
    Kokkos::parallel_reduce(
        "nhvw.vmax", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          const Real s = Kokkos::sqrt(v(3 * i) * v(3 * i) + v(3 * i + 1) * v(3 * i + 1) +
                                      v(3 * i + 2) * v(3 * i + 2));
          acc = s > acc ? s : acc;
        },
        Kokkos::Max<Real>(m));
    return m;
  }

  /// max |s_f|/d_f = |w_i − w_j| / (2 d²) over all facets and over unlike facets.
  void shifts(double& all, double& unlike) const {
    const CellFacets<Real> G = F_;
    const LView<Real> w = weight_;
    const auto T = spec_.types;
    Real a = 0, u = 0;
    Kokkos::parallel_reduce(
        "nhvw.shift", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& aa, Real& uu) {
          for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
            const int j = G.nbr(f);
            Real d2 = 0;
            for (int c = 0; c < 3; ++c)
              d2 += G.conn(3 * f + c) * G.conn(3 * f + c);
            const Real s = Kokkos::fabs(w(i) - w(j)) / (Real(2) * d2);
            aa = s > aa ? s : aa;
            if (T(i) != T(j))
              uu = s > uu ? s : uu;
          }
        },
        Kokkos::Max<Real>(a), Kokkos::Max<Real>(u));
    all = a;
    unlike = u;
  }

  MT* mt_;
  LView<Real> pos_, weight_;
  std::vector<double> V0h_;
  PotentialSpec<Real> spec_;
  FlowParams fp_;
  IntegratorParams prm_;
  int N_ = 0;
  Real L_[3] = {1, 1, 1};
  double maxV0_ = 0;
  LView<Real> V0_, rho_, mu_, mass_, h_, vel_, vh_, b_, zo_, zn_, grad_, gx_, gw_, pi_, lam1_,
      lam2_, tmpN_, dw_, pos0_, w0_, wrapS_, gd_, aI_, lamG_, gs_, fw_, aG_, mG_;
  Kokkos::View<int*, Mem> cls_, iList_;
  bool movingWall_ = false;
  long stepCount_ = 0;
  int restoreRebuilds_ = 0;
  // AMG reuse (IntegratorParams::amgReuse): setups since the build, the iteration reference of the
  // first solve after it, and the stale flag (iterations grew past 1.5× + 2)
  int sAge_ = 0, hAge_ = 0, sRef_ = -1, hRef_ = 0;
  bool sStale_ = false, hStale_ = false;
  double evalSplit_[4] = {0, 0, 0, 0};  ///< evalState: facets, energy, H AMG setup, adjoint (s)
  double gaugeU_ = 0, gaugeK_ = 0;
  CellFacets<Real> F_;
  CellFacets<Real> fc_;   ///< repairMismatch()'s export, reused at the same state (currentFacets)
  bool fcValid_ = false;  ///< fc_ describes the current (pos, weight, engine)
  PotentialParts parts_;
  FacetMatch match_;
  AdjointResult adj_;
  peclet::core::solver::GraphAMGDevice hAmg_;
  SaddleSolver<Real> saddle_;
  double time_ = 0, kinetic_ = 0, heat_ = 0, qc_ = 0, work_ = 0, wrapOffset_ = 0;
};

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_INTEGRATOR_HPP
