/**
 * @file lagrangian/integrator.hpp
 * \brief One time step of the Lagrangian power-cell method NHV-W (M7 DESIGN_NOTE §10.3 item 6,
 * with §2.6 for the linear solves and §2.7 steps 3 and 5 for the heat and the step size):
 *
 *   1. kick 1 at x^n, friction fused (θ-midpoint), velocity constraint C^n v^{½} = 0:
 *        (M + θ(τ/2)ζ^n) v^{½} + (τ/2) C^{nᵀ} λ₁ = (M − (1−θ)(τ/2)ζ^n) v^n − (τ/2)(∇Ũ^n − M
 * a(x^n));
 *   2. drift x^{n+1} = x^n + τ v^{½} (wrapped into the box); weights warm-started by
 *      H δw = −τ J_x v^{½} (at x^n) and solved by the damped Newton of weight_solve.hpp, every
 *      iterate an incremental REPAIR of the resident engine (no cold rebuild); then π^{n+1} and
 *      ∇Ũ^{n+1} = ∂_xU + J_xᵀπ (potential.hpp);
 *   3. kick 2: the same saddle at x^{n+1}, giving v^{n+1} and λ₂;
 *   4. bookkeeping: heat Q += (τ/2) v̄_θᵀ ζ (v_old + v_new)/2 per kick (§2.7 step 3), the
 *      constraint work Q_c += (τ/4) λ₂ᵀ C^{n+1} v^{½} (kick 2's; kick 1 does none), the body-force
 *      work W += (τ/2) (M a)ᵀ(v_old + v_new)/2 per kick, so that K + Ũ + Q + Q_c − W is conserved
 *      to the integrator's O(τ²);
 *   5. step size τ = min(C_cfl min_i h_i/|v_i|, C_σ √(ρ_σ h_min³/σ_max), τ_user) (stableStep).
 * No gauge (WO4′) and no topology bookkeeping beyond the repair's.
 *
 * The saddle [M + θ(τ/2)ζ, (τ/2)Cᵀ; C, 0] is solved by preconditioned CG on the Schur complement
 * S = C A⁻¹ Cᵀ (A = M + θ(τ/2)ζ) for μ = (τ/2)λ, preconditioned by AMG on the two-point
 * Laplacian L_ρ, (L_ρ p)_i = Σ_f (A_f/d_f) ½(1/ρ_i + 1/ρ_j)(p_i − p_j) (§2.6; pinned at DOF 0,
 * diagonal kept), plus the viscous diagonal of Cahouet–Chabard: P⁻¹ = L_ρ⁻¹ + θ(τ/2)·2μ_i/V_i.
 * L_ρ alone is the Schur complement's inertial limit (A ≈ M); when θ(τ/2)ζ dominates M (ν τ/h² ≫ 1)
 * S tends to C(θ(τ/2)ζ)⁻¹Cᵀ ≈ V/(2θ(τ/2)μ), a mass-matrix-like operator L_ρ⁻¹ cannot precondition;
 * the sum covers both limits (Cahouet & Chabard 1988). The velocity is carried by the same
 * recurrence as the residual, v ← v − α A⁻¹Cᵀp,   r ← r − α C (A⁻¹Cᵀp), with the SAME vector A⁻¹Cᵀp
 * in both, so r = C v holds to round-off whatever the inner accuracy: the constraint is met to the
 * outer tolerance ‖C v‖_∞ ≤ tol·max V/τ, and the inner tolerance (1e-10, §2.6) bounds only the
 * momentum residual. The outer CG is the flexible (Polak–Ribière) form, since inexact inner solves
 * make S vary slightly between iterations. Inner A-solves: PCG with Jacobi on the approximate
 * diagonal m + θ(τ/2) Σ_f μ_f A/(3d)(1 + n_a²) (+ the wall term), or, when max ν τ/h² exceeds 0.5
 * (§2.6), AMG on the scalar two-point operator m + θ(τ/2)·(4/9) Σ_f μ_f A/d (n̂_a² averaged to 1/3)
 * applied per component. Inviscid (θτζ = 0): A = M, solved exactly. Any non-convergence throws.
 */
#ifndef PECLET_VORO_LAGRANGIAN_INTEGRATOR_HPP
#define PECLET_VORO_LAGRANGIAN_INTEGRATOR_HPP

#include <algorithm>
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
}  // namespace detail

struct SaddleParams {
  double tol = 1e-12;       ///< outer: ‖C v‖_∞ ≤ tol · max_i V_i / τ (DESIGN_NOTE §2.6)
  double innerTol = 1e-10;  ///< relative residual of each inner A-solve
  int maxOuter = 2000, maxInner = 2000;
  double amgSwitch = 0.5;  ///< inner preconditioner: AMG when max ν τ/h² exceeds this, else Jacobi
};

struct SaddleStats {
  int outer = 0;          ///< Schur CG iterations
  int inner = 0;          ///< inner A-solve iterations, summed
  double constraint = 0;  ///< ‖C v‖_∞ τ / max V (recomputed from v)
  double viscNumber = 0;  ///< max ν τ / h²
  bool amgInner = false;  ///< inner preconditioner was AMG
};

/// The Stokes-type saddle of one NHV-W kick (see the file comment).
template <class Real>
class SaddleSolver {
 public:
  /// Geometry setup at the facets `F`: the L_ρ AMG and the approximate ζ diagonal. `rho`, `mu`,
  /// `mass`, `h` (N): cell density, viscosity, mass and length h_i = V⁰_i^{1/3}.
  void setup(const CellFacets<Real>& F, const LView<Real>& rho, const LView<Real>& mu,
             const LView<Real>& mass, const LView<Real>& h) {
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
      scr_ = DissipationScratch<Real>(N);
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
    }
    lrho_.build(detail::twoPointHost(
                    F,
                    [&](int i, int j, double A, double d) {
                      return A / d * 0.5 * (1.0 / rh[i] + 1.0 / rh[j]);
                    },
                    [](int) { return 0.0; }, /*pin=*/true),
                ap);
    mh_ = mh;
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
    st.viscNumber = nuH2_ * 2 * halfTau;
    useAmg_ = viscous_ && a > Real(0) && st.viscNumber > prm.amgSwitch;
    st.amgInner = useAmg_;
    if (useAmg_) {
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
    }
    const LView<Real> r = r_, z = z_, zo = zOld_, p = p_, s = s_, q = q_, vv = v, mu = lam;
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
    st.inner += solveA(a, w_, v, prm);
    applyDivergence(F_, v, r_);
    Real rinf = detail::maxAbsView(r);
    const Real rinit = rinf;
    if (rinf > tolAbs) {
      schurPrecond(a, r, z);
      Kokkos::deep_copy(p, z);
      Real rz = detail::dotView(r, z);
      bool done = false;
      for (int k = 0; k < prm.maxOuter; ++k) {
        applyDivergenceT(F_, p, w_);
        st.inner += solveA(a, w_, q, prm);
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
        schurPrecond(a, r, z);
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
    return st;
  }

  bool viscous() const { return viscous_; }
  const CellFacets<Real>& facets() const { return F_; }
  DissipationScratch<Real>& scratch() { return scr_; }

 private:
  /// z = (L_ρ⁻¹ + a·2μ/V) r (Cahouet–Chabard; the viscous term only with viscosity and a > 0).
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
  bool viscous_ = false, useAmg_ = false;
  double nuH2_ = 0;
  peclet::core::solver::GraphAMGDevice lrho_, visc_;
  LView<Real> cc_, r_, z_, zOld_, p_, s_, w_, q_, ir_, iz_, ip_, iq_, zdiag_, c0_, c1_;
  DissipationScratch<Real> scr_;
};

/// The physical inputs of the integrator beyond the potential: per-type viscosity and density
/// (host tables, indexed by the cell type) and a body acceleration a(x) = amp·sin(k·x).
struct FlowParams {
  std::vector<double> viscosity;  ///< nTypes (μ ≥ 0)
  std::vector<double> density;    ///< nTypes (ρ > 0; the mass is ρ V⁰)
  double bodyAmp[3] = {0, 0, 0};
  double bodyWave[3] = {0, 0, 0};
};

struct IntegratorParams {
  double theta = 0.5;
  SaddleParams saddle;
  double weightTol = 1e-12;  ///< max |V − V⁰|/V⁰ of the per-step weight solve
  int weightMaxIter = 50;
  double hTol = 1e-10;              ///< relative residual of the warm-start and Newton H solves
  double adjointTol = 1e-12;        ///< relative residual of the π solve
  int maxHalvings = 4;              ///< τ halvings when the weight solve fails (W-Q7)
  bool midpointConstraint = false;  ///< option S: kick 1's C at x^n + (τ/2) v^n
  double repairTol = 1e-10;  ///< repaired-topology volume check (cone fan vs engine), relative
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
      // restore x^n, w^n and repair the engine back to them; the state at x^n is untouched
      Kokkos::deep_copy(pos_, pos0_);
      Kokkos::deep_copy(weight_, w0_);
      mt_->setWeights(weight_);
      mt_->step(pos_);
      tau *= 0.5;
      ++rep.halvings;
    }
    time_ += tau;
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

  void setTheta(double th) { prm_.theta = th; }
  void setMidpointConstraint(bool on) { prm_.midpointConstraint = on; }
  double theta() const { return prm_.theta; }
  double time() const { return time_; }
  void setTime(double t) { time_ = t; }
  /// Restore the energy accumulators (a re-prepared integrator continues the old run's totals).
  void setAccumulators(double heat, double qc, double work, double wrap) {
    heat_ = heat;
    qc_ = qc;
    work_ = work;
    wrapOffset_ = wrap;
  }
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
    wp.pcgTol = prm_.hTol;
    wp.amg = &hAmg_;
    solvePinnedH(F_, tmpN_, dw_, wp);
    {
      const LView<Real> w = weight_, w0 = w0_, dw = dw_, P = pos_, P0 = pos0_, U = u;
      const Kokkos::Array<Real, 3> L3 = {L_[0], L_[1], L_[2]};
      const Real ss = s;
      Kokkos::parallel_for(
          "nhvw.msW", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i) { w(i) = w0(i) + dw(i); });
      Kokkos::parallel_for(
          "nhvw.msX", Kokkos::RangePolicy<Exec>(0, 3 * N_), KOKKOS_LAMBDA(const int k) {
            const Real Lk = L3[k % 3];
            const Real x = P0(k) + ss * U(k);
            P(k) = x - Lk * Kokkos::floor(x / Lk);
          });
    }
    wp.tol = prm_.weightTol;
    wp.maxIter = prm_.weightMaxIter;
    wp.repair = true;
    const WeightSolveResult ws = solveWeights(*mt_, pos_, weight_, V0h_, L_, wp);
    rep.tRepair += ws.evalSeconds;
    rep.evaluations += ws.evaluations;
    if (ws.converged && repairMismatch() > prm_.repairTol) {
      ++rep.repairFallbacks;
      mt_->rebuild(pos_);
      WeightSolveParams wc = wp;
      wc.repair = false;
      return solveWeights(*mt_, pos_, weight_, V0h_, L_, wc).converged;
    }
    return ws.converged;
  }

  /// max_i |V_i(export) − V_i(engine)| / V⁰_i of the resident topology at (pos, weight).
  double repairMismatch() const {
    const CellFacets<Real> F = exportCellFacets<Real, Power>(mt_->store, pos_, weight_, N_, L_,
                                                             mt_->wall, mt_->xRef, mt_->vol);
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

  /// Facets, U and ∇Ũ, π, and the two hierarchies at the current (pos, weight).
  void evalState(double* tPot = nullptr) {
    Kokkos::Timer t;
    F_ = exportCellFacets<Real, Power>(mt_->store, pos_, weight_, N_, L_, mt_->wall, mt_->xRef,
                                       mt_->vol);
    match_ = matchFacets(F_);
    Kokkos::deep_copy(gx_, Real(0));
    Kokkos::deep_copy(gw_, Real(0));
    parts_ = partitionPotential(*mt_, F_, pos_, weight_, L_, spec_, gx_, gw_);
    peclet::core::solver::AmgParams ap;
    ap.ndofPerNode = 1;
    hAmg_.build(detail::pinnedHostH(F_), ap);
    WeightSolveParams wp;
    wp.pcgTol = prm_.adjointTol;
    wp.amg = &hAmg_;
    adj_ = reducedGradient(F_, gx_, gw_, pi_, grad_, wp);
    saddle_.setup(F_, rho_, mu_, mass_, h_);
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
    // b = M v − (1−θ)(τ/2) ζ v − (τ/2)(∇Ũ − M a(x))
    Kokkos::parallel_for(
        "nhvw.rhs", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
          const Real s =
              body ? Kokkos::sin(kw[0] * P(3 * i) + kw[1] * P(3 * i + 1) + kw[2] * P(3 * i + 2))
                   : Real(0);
          for (int c = 0; c < 3; ++c) {
            const int k = 3 * i + c;
            b(k) = m(i) * vo(k) - (Real(1) - th) * ht * zo(k) - ht * (g(k) - m(i) * amp[c] * s);
          }
        });
    const Real tolAbs = prm_.saddle.tol * maxV0_ / tau;
    SaddleStats st = saddle_.solve(th * ht, ht, tolAbs, b_, vNew, lam, prm_.saddle);
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
      F_ = exportCellFacets<Real, Power>(mt_->store, pos_, weight_, N_, L_, mt_->wall, mt_->xRef,
                                         mt_->vol);
      matchFacets(F_);
      saddle_.setup(F_, rho_, mu_, mass_, h_);
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
      wp.pcgTol = prm_.hTol;
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
    wp.pcgTol = prm_.hTol;
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

  void accumulateWrap() {
    if (spec_.density.extent(0) == 0)
      return;
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
    wrapOffset_ -= j;
  }

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
      lam2_, tmpN_, dw_, pos0_, w0_, wrapS_;
  CellFacets<Real> F_;
  PotentialParts parts_;
  FacetMatch match_;
  AdjointResult adj_;
  peclet::core::solver::GraphAMGDevice hAmg_;
  SaddleSolver<Real> saddle_;
  double time_ = 0, kinetic_ = 0, heat_ = 0, qc_ = 0, work_ = 0, wrapOffset_ = 0;
};

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_INTEGRATOR_HPP
