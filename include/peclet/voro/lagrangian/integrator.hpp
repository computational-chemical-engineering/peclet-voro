/**
 * @file lagrangian/integrator.hpp
 * \brief One time step of the Lagrangian power-cell method NHV-W (M7 DESIGN_NOTE §10.3 item 6,
 * with §2.6 for the linear solves and §2.7 steps 3 and 5 for the heat and the step size):
 *
 *   1. kick 1 at x^n, friction fused (θ-midpoint), velocity constraint C^n v^{½} = 0 (with
 *      option S C and ζ of kick 1 at x^n + (τ/2) v^n instead,
 * IntegratorParams::kickConstraint): (M + θ(τ/2)ζ^n) v^{½} + (τ/2) C^{nᵀ} λ₁ = (M −
 * (1−θ)(τ/2)ζ^n) v^n − (τ/2)(∇Ũ^n − M a(x^n)); with KickConstraint::kPressure (D14.1, the
 * default since M7 WO-PERF3 P3.4) no constraint and no saddle: λ₁ is replaced by λ̄ = ¾λⁿ + ¼λⁿ⁻¹
 * (the last two kick-2 multipliers) on the right-hand side, and Q_c += (τ/4) λ̄ᵀCⁿ(vⁿ + v^{½}); a
 * restart step is option S;
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
 * counters, so K + Ũ + Q + Q_c − W − ΔŨ_G − ΔK_G is what the integrator conserves to O(τ²). With
 * IntegratorParams::gaugeMode = GaugeMode::kDrift (M7 WO-PERF3 P3.3a, §14.4.4) the gauge's
 * displacement δ is instead added in the next step's drift (driftGauge()); its ΔŨ_G is the
 * posterior trapezoidal estimate and ΔK_G = 0 (the projection is kick 2's constraint). Moving
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
/// The AMG configuration of the integrator's per-geometry hierarchies (S₀ or L_ρ, H, L_μ): core
/// C1's parallel setup, MIS-2 aggregation and a dense coarsest level (M7 WO-PERF3 P3.6, DESIGN_NOTE
/// §14.4.7, §14.5). The host oracle (Greedy, coarse CG) stays the default everywhere else.
inline peclet::core::solver::AmgParams m7AmgParams() {
  peclet::core::solver::AmgParams ap;
  ap.ndofPerNode = 1;
  ap.aggregation = peclet::core::solver::Aggregation::Mis2;
  ap.coarseSolve = peclet::core::solver::CoarseSolve::Dense;
  return ap;
}

/// Break-even AMG reuse (D14.6, DESIGN_NOTE §14.4.7; M7 WO-PERF3 P3.6), one per hierarchy. After a
/// setup the first recorded solve's iteration count is the reference; every later solve adds its
/// excess over it. The hierarchy is rebuilt before its next solve once the excess reaches R_setup
/// (its setup cost in iterations of that solve, a frozen constant), once it is maxAge steps old, or
/// after a failed step attempt. Counts only, so the iterates do not depend on wall time. maxAge ≤ 0:
/// rebuilt at every opportunity.
struct AmgReuse {
  int rSetup = 0;  ///< R_setup (iterations)
  int ref = -1, excess = 0;
  long builtStep = -1;
  bool failed = false;
  bool due(long step, int maxAge) const {
    return maxAge <= 0 || builtStep < 0 || failed || excess >= rSetup || step - builtStep >= maxAge;
  }
  void built(long step) {
    ref = -1;
    excess = 0;
    builtStep = step;
    failed = false;
  }
  void solved(int its) {
    if (ref < 0)
      ref = its;
    else
      excess += its > ref ? its - ref : 0;
  }
};
/// R_setup of §14.4.7, measured once on core C1 and frozen: S₀ in kick-saddle FGMRES iterations, H
/// in adjoint PCG iterations (V-cycles), L_μ in kick-saddle FGMRES iterations. N 32768 Q4 drop,
/// rebuilt at every opportunity, 1 / 4 threads: S₀ 12.6 / 12.3 (setup 756 / 207 ms), H 18.9 / 20.6,
/// L_μ 1.4 / 1.6 (μ 1, ν τ/h² 2.95) — the note's defaults 4 / 9 / 4 assumed a faster S₀ and H setup
/// (M7 WO-PERF3 P3.6; hub results/WO_PERF3_P36.md).
inline constexpr int kReuseSetupS0 = 12, kReuseSetupH = 20, kReuseSetupVisc = 2;

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
/// `hf` (optional): persistent host copies of the facets (M7 WO-PERF3 P3.1).
template <class Real, class Coef, class Extra>
peclet::core::solver::HostCsrOp twoPointHost(const CellFacets<Real>& F, Coef coef, Extra extra,
                                             bool pin, HostFacets* hf = nullptr) {
  HostFacets own;
  HostFacets& h = hf ? *hf : own;
  copyHostFacets(F, h);
  const std::vector<int>&off = h.off, &nbr = h.nbr;
  const std::vector<double>&area = h.area, &conn = h.conn;
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
    // stable: equal-k terms are summed in insertion order (l ascending, then the order of the
    // column entries), a defined order the device assembly reproduces bit for bit
    std::stable_sort(row.begin(), row.end(),
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

/// Heap sort of a[0, n) ascending, in place (device-callable; the S₀ sparsity pass).
KOKKOS_INLINE_FUNCTION void heapSortInts(int* a, int n) {
  auto sift = [a](int r, int e) {
    for (;;) {
      int c = 2 * r + 1;
      if (c >= e)
        return;
      if (c + 1 < e && a[c + 1] > a[c])
        ++c;
      if (a[r] >= a[c])
        return;
      const int t = a[r];
      a[r] = a[c];
      a[c] = t;
      r = c;
    }
  };
  for (int r = n / 2 - 1; r >= 0; --r)
    sift(r, n);
  for (int e = n - 1; e > 0; --e) {
    const int t = a[0];
    a[0] = a[e];
    a[e] = t;
    sift(0, e);
  }
}

/// S₀ = C M⁻¹ Cᵀ assembled on the device, bit-identical to schurHost (WO-PERF3 P3.1, §14.4.6
/// item 4). Columns of C are formed per velocity cell l in schurHost's order (facets of l, then the
/// self entry), with the same expressions; row i (one thread) visits its column cells l in
/// ascending order, the occurrences of i in column l in column order, and the entries k of column
/// l in column order — exactly the order in which schurHost's stable sort leaves the equal-k terms
/// of row i, so every coefficient is the same left-to-right sum. Off-diagonal slots start at −0
/// (−0 + v = v for every v, the host's "first term assigned"); the diagonal starts at +0 as on the
/// host. F must be reciprocal (matchFacets): the cells whose column holds row i are i and its face
/// neighbours. The off-diagonal sparsity (2-ring, pinned at DOF 0) is kept while F's facet
/// structure (facetOffset, nbr) is unchanged. All scratch is persistent.
template <class Real>
class SchurAssembly {
 public:
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using IV = Kokkos::View<int*, Mem>;

  /// S₀ as a host CSR (the test oracle's comparison; schurHost's layout).
  peclet::core::solver::HostCsrOp assemble(const CellFacets<Real>& F, const LView<Real>& mass) {
    compute(F, mass);
    const int N = F.nCells;
    peclet::core::solver::HostCsrOp K;
    K.n = N;
    K.diag = peclet::core::toVector(Kokkos::subview(diag_, std::make_pair(0, N)));
    K.start.assign(hStart_.begin(), hStart_.end());
    K.nbr.assign(hCols_.begin(), hCols_.end());
    K.coef = peclet::core::toVector(Kokkos::subview(coef_, std::make_pair(0, nnz_)));
    return K;
  }

  /// S₀ as a device CSR (the same arrays as assemble(), no host round trip): the input of core
  /// C1's parallel setup, GraphAMGDevice::build(DeviceCsr) (M7 WO-PERF3 P3.6, DESIGN_NOTE §14.5).
  /// The views alias this object's scratch; build() copies them.
  peclet::core::solver::DeviceCsr assembleDevice(const CellFacets<Real>& F,
                                                 const LView<Real>& mass) {
    compute(F, mass);
    const int N = F.nCells;
    peclet::core::solver::DeviceCsr K;
    K.n = N;
    K.diag = Kokkos::subview(diag_, std::make_pair(0, N));
    K.start = Kokkos::subview(iStart_, std::make_pair(0, N + 1));
    K.nbr = Kokkos::subview(iCols_, std::make_pair(0, nnz_));
    K.coef = Kokkos::subview(coef_, std::make_pair(0, nnz_));
    return K;
  }

 private:
  void compute(const CellFacets<Real>& F, const LView<Real>& mass) {
    const int N = F.nCells, nF = F.nFacets;
    const std::size_t nE = static_cast<std::size_t>(nF) + static_cast<std::size_t>(N);
    grow(colRow_, nE, "s0.colRow");
    grow(colVal_, 3 * nE, "s0.colVal");
    grow(ring_, nE, "s0.ring");
    grow(ringN_, static_cast<std::size_t>(N), "s0.ringN");
    grow(diag_, static_cast<std::size_t>(N), "s0.diag");
    const IV off = F.facetOffset, nbr = F.nbr, woff = F.wallOffset;
    const LView<Real> area = F.area, conn = F.conn, cen = F.centroid, warea = F.wallArea,
                      wn = F.wallNormal;
    const IV colRow = colRow_, ring = ring_, ringN = ringN_;
    const LView<Real> colVal = colVal_;
    // ---- columns of C: entries of column l at [off(l) + l, off(l + 1) + l + 1) ----
    Kokkos::parallel_for(
        "s0.columns", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int l) {
          double dl[3] = {0, 0, 0};
          int e = off(l) + l;
          for (int f = off(l); f < off(l + 1); ++f, ++e) {
            const double d =
                Kokkos::sqrt(conn(3 * f) * conn(3 * f) + conn(3 * f + 1) * conn(3 * f + 1) +
                             conn(3 * f + 2) * conn(3 * f + 2));
            const double sc = area(f) / d;
            for (int a = 0; a < 3; ++a) {
              dl[a] -= sc * cen(3 * f + a);
              colVal(3 * e + a) = sc * (cen(3 * f + a) - conn(3 * f + a));
            }
            colRow(e) = nbr(f);
          }
          for (int f = woff(l); f < woff(l + 1); ++f)
            for (int a = 0; a < 3; ++a)
              dl[a] -= warea(f) * wn(3 * f + a);
          for (int a = 0; a < 3; ++a)
            colVal(3 * e + a) = dl[a];
          colRow(e) = l;
        });
    // ---- the column cells of row i: {i} ∪ face neighbours, ascending, unique ----
    Kokkos::parallel_for(
        "s0.ring", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          const int b = off(i) + i;
          int n = 0;
          ring(b + n++) = i;
          for (int f = off(i); f < off(i + 1); ++f)
            ring(b + n++) = nbr(f);
          heapSortInts(&ring(b), n);
          int u = 0;
          for (int q = 0; q < n; ++q)
            if (u == 0 || ring(b + q) != ring(b + u - 1))
              ring(b + u++) = ring(b + q);
          ringN(i) = u;
        });
    if (!sameStructure(F))
      sparsity(F);
    // ---- values, row by row in the stable host order ----
    const IV start = start_, cols = cols_;
    const LView<Real> coef = coef_, dg = diag_, m = mass;
    Kokkos::parallel_for(
        "s0.values", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          const int s0 = start(i), s1 = start(i + 1);
          for (int q = s0; q < s1; ++q)
            coef(q) = -0.0;
          double di = 0;
          const int b = off(i) + i;
          for (int r = 0; r < ringN(i); ++r) {
            const int l = ring(b + r);
            const double im = 1.0 / m(l);
            const int e0 = off(l) + l, e1 = off(l + 1) + l + 1;
            for (int ei = e0; ei < e1; ++ei) {
              if (colRow(ei) != i)
                continue;
              const double c0 = colVal(3 * ei), c1 = colVal(3 * ei + 1), c2 = colVal(3 * ei + 2);
              for (int ek = e0; ek < e1; ++ek) {
                const int k = colRow(ek);
                const double v =
                    im * (c0 * colVal(3 * ek) + c1 * colVal(3 * ek + 1) + c2 * colVal(3 * ek + 2));
                if (k == i) {
                  di += v;
                  continue;
                }
                if (i == 0 || k == 0)
                  continue;  // pinned
                int lo = s0, hi = s1;
                while (hi - lo > 1) {
                  const int mid = (lo + hi) / 2;
                  if (cols(mid) <= k)
                    lo = mid;
                  else
                    hi = mid;
                }
                coef(lo) += v;
              }
            }
          }
          dg(i) = di;
        });
    Kokkos::fence();
  }

  template <class V>
  static void grow(V& v, std::size_t n, const char* label) {
    if (v.extent(0) < n)
      v = V(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), n + n / 4 + 64);
  }

  /// F's facet structure equals the one the cached sparsity was built for.
  bool sameStructure(const CellFacets<Real>& F) const {
    if (F.nCells != sN_ || F.nFacets != sF_ || sN_ < 0)
      return false;
    const IV off = F.facetOffset, nbr = F.nbr, so = sOff_, sn = sNbr_;
    const int nC = F.nCells;
    int bad = 0;
    Kokkos::parallel_reduce(
        "s0.same", Kokkos::RangePolicy<Exec>(0, F.nFacets + F.nCells + 1),
        KOKKOS_LAMBDA(const int q, int& acc) {
          if (q <= nC)
            acc += off(q) != so(q) ? 1 : 0;
          else
            acc += nbr(q - nC - 1) != sn(q - nC - 1) ? 1 : 0;
        },
        bad);
    return bad == 0;
  }

  /// The off-diagonal sparsity of S₀ (pinned: row 0 and column 0 empty), rows ascending.
  void sparsity(const CellFacets<Real>& F) {
    const int N = F.nCells, nF = F.nFacets;
    const IV off = F.facetOffset, ring = ring_, ringN = ringN_;
    grow(ub_, static_cast<std::size_t>(N) + 1, "s0.ub");
    grow(cnt_, static_cast<std::size_t>(N) + 1, "s0.cnt");
    const IV ub = ub_, cnt = cnt_;
    // upper bound per row: Σ over its column cells of the column length
    Kokkos::parallel_for(
        "s0.ub", Kokkos::RangePolicy<Exec>(0, N + 1), KOKKOS_LAMBDA(const int i) {
          int u = 0;
          if (i > 0) {
            const int r = i - 1, b = off(r) + r;
            for (int q = 0; q < ringN(r); ++q) {
              const int l = ring(b + q);
              u += off(l + 1) - off(l) + 1;
            }
          }
          ub(i) = u;
        });
    int tot = 0;
    Kokkos::parallel_scan(
        "s0.ubScan", N + 1,
        KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
          upd += ub(i);
          if (fin)
            ub(i) = upd;
        },
        tot);
    grow(cand_, static_cast<std::size_t>(tot), "s0.cand");
    const IV cand = cand_, colRow = colRow_;
    Kokkos::parallel_for(
        "s0.pattern", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          const int b = off(i) + i, c0 = ub(i);
          int n = 0;
          if (i != 0)
            for (int q = 0; q < ringN(i); ++q) {
              const int l = ring(b + q);
              for (int e = off(l) + l; e < off(l + 1) + l + 1; ++e) {
                const int k = colRow(e);
                if (k != i && k != 0)
                  cand(c0 + n++) = k;
              }
            }
          heapSortInts(&cand(c0), n);
          int u = 0;
          for (int q = 0; q < n; ++q)
            if (u == 0 || cand(c0 + q) != cand(c0 + u - 1))
              cand(c0 + u++) = cand(c0 + q);
          cnt(i + 1) = u;
        });
    Kokkos::deep_copy(Kokkos::subview(cnt, 0), 0);
    int nnz = 0;
    Kokkos::parallel_scan(
        "s0.cntScan", N + 1,
        KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
          upd += cnt(i);
          if (fin)
            cnt(i) = upd;
        },
        nnz);
    grow(cols_, static_cast<std::size_t>(nnz), "s0.cols");
    grow(coef_, static_cast<std::size_t>(nnz), "s0.coef");
    const IV cols = cols_;
    Kokkos::parallel_for(
        "s0.compact", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          const int c0 = ub(i);
          for (int q = cnt(i); q < cnt(i + 1); ++q)
            cols(q) = cand(c0 + q - cnt(i));
        });
    start_ = Kokkos::subview(cnt_, std::make_pair(0, N + 1));
    nnz_ = nnz;
    // host copies of the pattern (HostCsrOp) and the structure the pattern belongs to
    const auto hs = peclet::core::toVector(start_);
    const auto hc = peclet::core::toVector(Kokkos::subview(cols_, std::make_pair(0, nnz)));
    hStart_.assign(hs.begin(), hs.end());
    hCols_.assign(hc.begin(), hc.end());
    // the pattern in core's index type (DeviceCsr), for the device build
    grow(iStart_, static_cast<std::size_t>(N) + 1, "s0.iStart");
    grow(iCols_, static_cast<std::size_t>(nnz) + 1, "s0.iCols");
    {
      const IV st = start_;
      const XV is = iStart_, ic = iCols_;
      Kokkos::parallel_for(
          "s0.idxStart", Kokkos::RangePolicy<Exec>(0, N + 1),
          KOKKOS_LAMBDA(const int i) { is(i) = st(i); });
      Kokkos::parallel_for(
          "s0.idxCols", Kokkos::RangePolicy<Exec>(0, nnz),
          KOKKOS_LAMBDA(const int q) { ic(q) = cols(q); });
    }
    grow(sOff_, static_cast<std::size_t>(N) + 1, "s0.sOff");
    grow(sNbr_, static_cast<std::size_t>(nF), "s0.sNbr");
    Kokkos::deep_copy(Kokkos::subview(sOff_, std::make_pair(0, N + 1)), off);
    Kokkos::deep_copy(Kokkos::subview(sNbr_, std::make_pair(0, nF)), F.nbr);
    sN_ = N;
    sF_ = nF;
  }

  using XV = Kokkos::View<peclet::core::Index*, Mem>;
  IV colRow_, ring_, ringN_, ub_, cnt_, cand_, cols_, start_, sOff_, sNbr_;
  XV iStart_, iCols_;
  LView<Real> colVal_, coef_, diag_;
  std::vector<peclet::core::Index> hStart_, hCols_;
  int nnz_ = 0, sN_ = -1, sF_ = -1;
};
}  // namespace detail

/// Where the gauge's displacement goes (M7 WO-PERF3 P3.3a, DESIGN_NOTE §14.4.4).
enum class GaugeMode {
  kTrial,  ///< G″ (§12.3.2): a trial OT after kick 2, Ũ-guarded, then an M-projection (gauge())
  kDrift,  ///< δ added in the next step's drift, posterior-monitored (driftGauge(), §14.4.4)
};

/// Kick 1 of a step (IntegratorParams::kickConstraint; LagrangianFlow.set_kick_constraint).
enum class KickConstraint {
  kStart,     ///< a saddle with C (and ζ) at x^n: first order in the energy
  kMidpoint,  ///< option S (§10.3.6): the saddle with C and ζ at x^n + (τ/2)v^n, weights solved
              ///< there
  /// D14.1 (§14.4.1, M7 WO-PERF3 P3.4): no constraint and no saddle; the last pressures
  /// λ̄ = ¾λⁿ + ¼λⁿ⁻¹ act through Cⁿᵀ inside the CN A-solve at x^n. A restart step (the first after
  /// construction or setKickConstraint, and the retry after a τ-halving) is an option-S step, and
  /// sets λⁿ⁻¹ = λⁿ = its λ₂.
  kPressure,
};

/// The kick's saddle solver (M7 WO-PERF).
enum class SaddleMethod {
  kSchurCG,  ///< Schur-complement CG with an exact inner A-solve per iteration (WO4 default)
  kCoupled,  ///< A: one FGMRES on [A, Cᵀ; C, 0], block-triangular preconditioner (exact saddle)
  kProjection,  ///< B: rotational incremental pressure correction (one A-solve + one S₀ projection)
};

struct SaddleParams {
  /// η_K (DESIGN_NOTE D14.3, M7 WO-PERF3 P3.2): ‖C v‖_∞ ≤ tol · ‖C‖ · v_ref, relative alone (the
  /// former min with tol · max_i V_i / τ, §2.6, is dropped; 1e-12 before P3.2)
  double tol = 1e-9;
  /// relative residual of each inner A-solve, and the coupled stop's ‖r_v‖₂ target innerTol ‖b‖₂
  /// (D14.3: η_K; 1e-10 before P3.2)
  double innerTol = 1e-9;
  int maxOuter = 2000, maxInner = 2000;
  double amgSwitch = 0.5;  ///< inner preconditioner: AMG when max ν τ/h² exceeds this, else Jacobi
  /// Schur preconditioner: true (default since M7 WO4a) = AMG on the assembled inertial Schur
  /// complement S₀ = C M⁻¹ Cᵀ (2-ring stencil); false = AMG on the two-point L_ρ (§2.6).
  /// Cahouet–Chabard's viscous term either way. Measured (Lloyd drop, ρ 1:10, σ 1, μ 0.1, gauge
  /// on): outer iterations per kick 96 → 25 and s/step 136 → 35 at N = 32768 (7.7 → 3.0 at
  /// 4096); inviscid 113 → 23 outer, but its 4× dearer setup makes the step ~7 % slower.
  bool schurAssembled = true;
  /// default kCoupled since M7 WO-PERF (the fastest method that passes every gate; B fails rest)
  SaddleMethod method = SaddleMethod::kCoupled;
  /// B: χ of the rotational multiplier update μ += φ + χ a (μ_min / V_i) (C v*)_i — flow's
  /// rotational term −χ μ_min ∇·u* (its default mode "min", χ = 1: the least cell viscosity, stable
  /// at any contrast); χ = 2 is the whole gradient part of the stress-form ζ (the Cahouet–Chabard
  /// 2μ/V). 0: plain incremental
  double rotational = 1.0;
  int restart = 40;  ///< A: FGMRES restart length
  /// A: Â⁻¹ of the preconditioner and the guess. 0 (default since M7 WO-PERF3 P3.5, D14.7): Jacobi,
  /// or one viscous V-cycle per component above amgSwitch; k > 0: k Jacobi-PCG iterations (the
  /// pre-P3.5 Â, 2)
  int coupledInner = 0;
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
    }
    detail::hostCopy(rho, rhoH_);
    detail::hostCopy(mu, muH_);
    detail::hostCopy(h, hH_);
    const std::vector<double>&rh = rhoH_, &mh = muH_, &hh = hH_;
    viscous_ = false;
    nuH2_ = 0;
    for (int i = 0; i < N; ++i) {
      viscous_ = viscous_ || mh[i] > 0;
      nuH2_ = std::max(nuH2_, mh[i] / (rh[i] * hh[i] * hh[i]));
    }
    {  // Cahouet–Chabard viscous diagonal 2μ_i/V_i (scaled by a = θτ/2 in solve)
      detail::hostCopy(F.volume, volH_);
      const std::vector<double>& vh = volH_;
      std::vector<double>& cc = ccH_;
      cc.resize(N);
      for (int i = 0; i < N; ++i)
        cc[i] = vh[i] > 0 ? 2.0 * mh[i] / vh[i] : 0.0;
      toDeviceInto(cc, cc_, "sd.cc");
      const double muMin = N > 0 ? *std::min_element(mh.begin(), mh.end()) : 0.0;
      for (int i = 0; i < N; ++i)
        cc[i] = vh[i] > 0 ? muMin / vh[i] : 0.0;
      toDeviceInto(cc, ccMin_, "sd.ccMin");
    }
    setupSeconds_ = 0;
    if (rebuildAmg || lrho_.numLevels() == 0 || builtAssembled_ != assembled_)
      buildSchur();  // else the lagged hierarchy (no setup charged)
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
    if (viscous_)  // the ζ-plan of this geometry (D14.5): every ζ apply of the kicks uses it
      plan_.build(F, mu);
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
  /// λ (e.g. the previous kick's) and returns λ = μ / halfTau. `a` = θ τ/2. `tolAbs` is overwritten
  /// by the relative bound tol · ‖C‖ · v_ref (D14.3; it was a cap before M7 WO-PERF3 P3.2).
  SaddleStats solve(Real a, Real halfTau, Real tolAbs, const LView<Real>& b, const LView<Real>& v,
                    const LView<Real>& lam, const SaddleParams& prm) {
    using Exec = peclet::core::ExecSpace;
    const int N = F_.nCells;
    SaddleStats st;
    Kokkos::Timer tAll, tPart;
    prepareInner(a, halfTau, prm, st);
    const LView<Real> r = r_, z = z_, zo = zOld_, p = p_, s = s_, q = q_, vv = v, mu = lam;
    {  // the stop, physically scaled (M7 WO4a): ‖C v‖_∞ ≤ tol · ‖C‖ · v_ref, with v_ref =
       // max_i |b_i|/m_i the velocity the kick would produce unconstrained; relative alone since
       // M7 WO-PERF3 P3.2 (D14.3: the caller's tolAbs is no longer a cap). tolAbs alone
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
      tolAbs = Real(prm.tol) * cScale_ * vr;
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
    if (prm.method == SaddleMethod::kCoupled && viscous_ && a > Real(0)) {
      // A's guess (D14.7, §14.4.5): v₀ = Â⁻¹(b − Cᵀμ₀), not an A-solve (λ₀ = λⁿ by the caller)
      if (prm.coupledInner > 0)
        st.inner += solveAcap(a, w_, v, prm.coupledInner);
      else
        applyAhat(a, w_, v);
      Kokkos::fence();
      st.tInner += tPart.seconds();
      return solveCoupled(a, halfTau, tolAbs, b, v, lam, prm, st, tAll);
    }
    st.inner += solveA(a, w_, v, prm);
    Kokkos::fence();
    st.tInner += tPart.seconds();
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
      // increment the S₀ projection leaves out (S⁻¹ ≈ S₀⁻¹ + a·2μ/V, Cahouet–Chabard; flow's
      // −μ∇·u*)
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

  /// x = A⁻¹ b, A = M + a ζ, with no constraint: kick 1 of the 'pressure' step (DESIGN_NOTE §14.4.1
  /// step 1, D14.1), whose λ̄ the caller has put into b. PCG from the guess in `x` to
  /// ‖b − A x‖₂ ≤ prm.innerTol ‖b‖₂ (η_K), with solve()'s inner preconditioner (Jacobi, or the
  /// scalar viscous AMG above prm.amgSwitch); inviscid or a = 0: x = M⁻¹ b. `halfTau` = τ/2.
  SaddleStats solveMomentum(Real a, Real halfTau, const LView<Real>& b, const LView<Real>& x,
                            const SaddleParams& prm) {
    SaddleStats st;
    Kokkos::Timer tAll, tPart;
    prepareInner(a, halfTau, prm, st);
    tPart.reset();
    st.inner = solveA(a, b, x, prm, /*fromGuess=*/true);
    Kokkos::fence();
    st.tInner = tPart.seconds();
    st.tTotal = tAll.seconds();
    return st;
  }

  bool viscous() const { return viscous_; }
  /// z = one V-cycle of the last setup's S₀ (or L_ρ) hierarchy on r, without the Cahouet–Chabard
  /// term: the preconditioner of the drift gauge's weight relaxation W (§14.4.4).
  void applySchurHierarchy(const LView<Real>& r, const LView<Real>& z) const { lrho_.apply(r, z); }
  void setAssembled(bool on) { assembled_ = on; }  ///< SaddleParams::schurAssembled (next setup)
  const CellFacets<Real>& facets() const { return F_; }
  DissipationScratch<Real>& scratch() { return scr_; }
  /// The S₀ (assembled) or L_ρ hierarchy at the last setup's facets and masses; its time is charged
  /// to the next solve's st.tSetup. setup() calls it unless the hierarchy is lagged; the integrator
  /// calls it right before a kick saddle when the break-even rule says so (M7 WO-PERF3 P3.6).
  void buildSchur() {
    Kokkos::Timer tb;
    if (assembled_) {
      // the S₀ configuration of the P3.0-c study (§14.4.7, variant (e)): Chebyshev degree 3
      // (21 → 16 kick-2 iterations at N 4096 and 32768; M7 WO-PERF3 P3.6)
      peclet::core::solver::AmgParams s0 = detail::m7AmgParams();
      s0.chebDegree = 3;
      lrho_.build(schurAsm_.assembleDevice(F_, mass_), s0);
    } else {
      const std::vector<double>& rh = rhoH_;
      lrho_.build(detail::twoPointHost(
                      F_,
                      [&](int i, int j, double A, double d) {
                        return A / d * 0.5 * (1.0 / rh[i] + 1.0 / rh[j]);
                      },
                      [](int) { return 0.0; }, /*pin=*/true),
                  detail::m7AmgParams());
    }
    builtAssembled_ = assembled_;
    setupSeconds_ += tb.seconds();
  }
  bool hasSchur() const { return lrho_.numLevels() > 0 && builtAssembled_ == assembled_; }
  /// The step clock and age cap of the viscous hierarchy's break-even reuse (D14.6).
  void setAmgClock(long step, int maxAge) {
    amgClock_ = step;
    amgMaxAge_ = maxAge;
  }
  /// A kick saddle preconditioned by the viscous hierarchy took `its` outer iterations.
  void recordViscSolve(int its) { viscReuse_.solved(its); }
  /// A step attempt failed: the viscous hierarchy is rebuilt before its next solve.
  void markAmgFailed() { viscReuse_.failed = true; }
  /// out = ζ v at the last setup's facets and viscosity, by the ζ-plan (viscous only)
  void applyZeta(const LView<Real>& v, const LView<Real>& out) {
    applyDissipation(plan_, v, out, scr_);
  }

 private:
  /// The inner A-solve preconditioner of a solve at a = θτ/2: AMG on the scalar two-point viscous
  /// operator when max ν τ/h² exceeds prm.amgSwitch, else Jacobi; st.tSetup also takes the last
  /// setup()'s time (charged once). The viscous hierarchy is rebuilt (its time charged to
  /// st.tSetup) only when the break-even rule says so (viscReuse_; the solves recorded are the kick
  /// saddles', recordViscSolve) — kept across geometries and τ, like S₀ (M7 WO-PERF3 P3.6).
  void prepareInner(Real a, Real halfTau, const SaddleParams& prm, SaddleStats& st) {
    Kokkos::Timer tPart;
    st.tSetup = setupSeconds_;  // charged once, to the first solve after setup()
    setupSeconds_ = 0;
    st.viscNumber = nuH2_ * 2 * halfTau;
    useAmg_ = viscous_ && a > Real(0) && st.viscNumber > prm.amgSwitch;
    st.amgInner = useAmg_;
    if (useAmg_ && (visc_.numLevels() == 0 || viscReuse_.due(amgClock_, amgMaxAge_))) {
      tPart.reset();
      const peclet::core::solver::AmgParams ap = detail::m7AmgParams();
      detail::hostCopy(mass_, massH_);
      const std::vector<double>& mh = massH_;
      const auto& muh = mh_;
      const double aa = a;
      visc_.build(detail::twoPointHost(
                      F_,
                      [&](int i, int j, double A, double d) {
                        const double muf =
                            muh[i] + muh[j] > 0 ? 2 * muh[i] * muh[j] / (muh[i] + muh[j]) : 0.0;
                        // c = 4/9: the isotropic mean of ζ's compact two-point part
                        // μ_fA/(3d)(I + n̂n̂ᵀ) (zdiag), NOT the continuum L_μ = μ_fA/d of
                        // §13.4.4: on ζ itself (N 512, dense) κ((M + a c L)⁻¹(M + aζ)) is
                        // minimal at c ≈ 0.40–0.44 for every ν τ/h² in 0.6–100 (c = 1: +22–31 %
                        // at 0.6–10); spec(ζ, L_1⊗I) = [0.38, 1.81], median 0.49 (M7 WO-PERF3 P3.6)
                        return aa * (4.0 / 9.0) * muf * A / d;
                      },
                      [&](int i) { return mh[i]; }, /*pin=*/false, &hostF_),
                  ap);
      viscReuse_.built(amgClock_);
      st.tSetup += tPart.seconds();
    }
  }

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
  /// the guess (v₀ = Â⁻¹(b − Cᵀμ₀), μ₀) the caller computed (D14.7). Block upper-triangular
  /// preconditioner P = [Â, Cᵀ; 0, −Ŝ]: z_μ = −Ŝ⁻¹ r_μ, z_v = Â⁻¹(r_v − Cᵀ z_μ), Ŝ⁻¹ = the Schur
  /// preconditioner (AMG on S₀ + Cahouet–Chabard), Â⁻¹ = applyAhat (prm.coupledInner 0: Jacobi,
  /// or one viscous V-cycle per component above amgSwitch; k > 0: k Jacobi-PCG iterations, the
  /// pre-P3.5 Â). Orthogonalisation by CGS2 (§14.4.5): two fused multi-dot and two fused
  /// multi-axpy passes per iteration whatever the basis size, the second axpy also giving the norm.
  /// The constraint rows are scaled by ς = ε_m / tolAbs (ε_m = innerTol ‖b‖₂) so that the 2-norm
  /// target ε_m bounds ‖C v‖_∞ ≤ tolAbs; converged on the true residuals ‖C v‖_∞ ≤ tolAbs and
  /// ‖b − A v − Cᵀμ‖₂ ≤ 10 ε_m. The exact saddle solution to that tolerance.
  SaddleStats solveCoupled(Real a, Real halfTau, Real tolAbs, const LView<Real>& b,
                           const LView<Real>& v, const LView<Real>& mu, const SaddleParams& prm,
                           SaddleStats st, Kokkos::Timer& tAll) {
    using Exec = peclet::core::ExecSpace;
    const int N = F_.nCells, n3 = 3 * N, m = prm.restart;
    // the bases as columns of two flat Views: velocity (3N per column) and multiplier (N)
    const std::size_t nv = static_cast<std::size_t>(n3) * (m + 1),
                      nm = static_cast<std::size_t>(N) * (m + 1);
    if (KV_.extent(0) != nv || KM_.extent(0) != nm) {
      KV_ = LView<Real>("sd.KV", nv);
      KM_ = LView<Real>("sd.KM", nm);
      ZV_ = LView<Real>("sd.ZV", static_cast<std::size_t>(n3) * m);
      ZM_ = LView<Real>("sd.ZM", static_cast<std::size_t>(N) * m);
      hd_ = LView<Real>("sd.hd", static_cast<std::size_t>(m + 1));
    }
    auto kv = [&](int j) { return LView<Real>(KV_.data() + static_cast<std::size_t>(j) * n3, n3); };
    auto km = [&](int j) { return LView<Real>(KM_.data() + static_cast<std::size_t>(j) * N, N); };
    auto zv = [&](int j) { return LView<Real>(ZV_.data() + static_cast<std::size_t>(j) * n3, n3); };
    auto zm = [&](int j) { return LView<Real>(ZM_.data() + static_cast<std::size_t>(j) * N, N); };
    const LView<Real> M = mass_;
    const Real em = Real(prm.innerTol) * std::sqrt(detail::dotView(b, b));
    if (!(em > Real(0)) || !(tolAbs > Real(0)))
      return finishCoupled(halfTau, v, mu, st, tAll);
    const Real sc = em / tolAbs;
    Kokkos::Timer tPart;
    // out = K' (xv, xm): (M + aζ) xv + Cᵀxm (the ζ-plan's last gather adds M xv and Cᵀxm), ς C xv
    auto applyK = [&](const LView<Real>& xv, const LView<Real>& xm, const LView<Real>& ov,
                      const LView<Real>& om) {
      applyDivergenceT(F_, xm, w_);
      applyMassDissipation(plan_, M, a, xv, ov, scr_, w_);
      applyDivergence(F_, xv, om);
      Kokkos::parallel_for(
          "sd.Km", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { om(i) *= sc; });
    };
    // the true residual (b − A v − Cᵀμ, −ς C v) into column 0; returns ‖C v‖_∞, ‖r‖₂ of both
    // blocks, ‖r_v‖₂
    auto residual = [&](Real& cinf, Real& beta, Real& rv2) {
      const LView<Real> rv = kv(0), rm = km(0);
      applyK(v, mu, rv, rm);
      Real sv = 0, sm = 0, cm = 0;
      Kokkos::parallel_reduce(
          "sd.R", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i, Real& av, Real& am, Real& ac) {
            for (int c = 0; c < 3; ++c) {
              const Real r = b(3 * i + c) - rv(3 * i + c);
              rv(3 * i + c) = r;
              av += r * r;
            }
            const Real q = -rm(i);
            rm(i) = q;
            am += q * q;
            const Real aq = Kokkos::fabs(q);
            ac = aq > ac ? aq : ac;
          },
          sv, sm, Kokkos::Max<Real>(cm));
      cinf = cm / sc;
      rv2 = std::sqrt(sv);
      beta = std::sqrt(sv + sm);
    };
    std::vector<double> H(static_cast<std::size_t>((m + 1) * m)), cs(m), sn(m), g(m + 1), y(m);
    std::vector<Real> hv(m + 1), h2(m + 1);
    auto Hij = [&](int i, int j) -> double& {
      return H[static_cast<std::size_t>(j * (m + 1) + i)];
    };
    Real cinf = 0, rv2 = 0, beta = 0;
    for (int cycle = 0;; ++cycle) {
      residual(cinf, beta, rv2);
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
      scaleCol(kv(0), km(0), Real(1.0 / beta));
      std::fill(g.begin(), g.end(), 0.0);
      g[0] = beta;
      int j = 0;
      for (; j < m; ++j) {
        // z_j = P⁻¹ v_j
        tPart.reset();
        {
          const LView<Real> vm = km(j), zmj = zm(j), z = z_, S = s_;
          Kokkos::parallel_for(
              "sd.Pm", Kokkos::RangePolicy<Exec>(0, N),
              KOKKOS_LAMBDA(const int i) { S(i) = vm(i) / sc; });
          schurPrecond(a, S, z);
          Kokkos::parallel_for(
              "sd.Pz", Kokkos::RangePolicy<Exec>(0, N),
              KOKKOS_LAMBDA(const int i) { zmj(i) = -z(i); });
          applyDivergenceT(F_, zmj, w_);
          const LView<Real> vv = kv(j), W = w_, q = q_;
          Kokkos::parallel_for(
              "sd.Pv", Kokkos::RangePolicy<Exec>(0, n3),
              KOKKOS_LAMBDA(const int k) { q(k) = vv(k) - W(k); });
        }
        Kokkos::fence();
        st.tPrecond += tPart.seconds();
        tPart.reset();
        if (prm.coupledInner > 0)
          st.inner += solveAcap(a, q_, zv(j), prm.coupledInner);
        else
          applyAhat(a, q_, zv(j));
        Kokkos::fence();
        st.tInner += tPart.seconds();
        applyK(zv(j), zm(j), kv(j + 1), km(j + 1));
        // CGS2: h = Vᵀw, w −= V h, twice; the second axpy returns ‖w‖²
        multiDot(j + 1, kv(j + 1), km(j + 1), hv);
        multiAxpy(j + 1, KV_, KM_, hv, kv(j + 1), km(j + 1), false);
        multiDot(j + 1, kv(j + 1), km(j + 1), h2);
        const double hn =
            std::sqrt(double(multiAxpy(j + 1, KV_, KM_, h2, kv(j + 1), km(j + 1), true)));
        for (int i = 0; i <= j; ++i)
          Hij(i, j) = double(hv[i]) + double(h2[i]);
        Hij(j + 1, j) = hn;
        if (hn > 0)
          scaleCol(kv(j + 1), km(j + 1), Real(1.0 / hn));
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
      for (int i = j - 1; i >= 0; --i) {  // y = H⁻¹ g
        double t = g[i];
        for (int k = i + 1; k < j; ++k)
          t -= Hij(i, k) * y[k];
        y[i] = t / Hij(i, i);
      }
      for (int i = 0; i < j; ++i)  // x += Z y, one fused pass (multiAxpy subtracts)
        hv[i] = Real(-y[i]);
      multiAxpy(j, ZV_, ZM_, hv, v, mu, false);
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

  /// (xv, xm) *= c, one pass
  void scaleCol(const LView<Real>& xv, const LView<Real>& xm, Real c) const {
    Kokkos::parallel_for(
        "sd.scl", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, F_.nCells),
        KOKKOS_LAMBDA(const int i) {
          for (int e = 0; e < 3; ++e)
            xv(3 * i + e) *= c;
          xm(i) *= c;
        });
  }

  /// The fused multi-dot of CGS2: h_c = ⟨(KV_c, KM_c), (wv, wm)⟩ for the first `cnt` basis
  /// columns, in one pass over the cells (an array reduction).
  struct MultiDot {
    using value_type = Real[];
    using size_type = int;
    int value_count, n3, N;
    LView<Real> V, Mm, wv, wm;
    KOKKOS_INLINE_FUNCTION void operator()(const int i, value_type acc) const {
      const Real w0 = wv(3 * i), w1 = wv(3 * i + 1), w2 = wv(3 * i + 2), wq = wm(i);
      for (int c = 0; c < value_count; ++c) {
        const std::size_t o = static_cast<std::size_t>(c) * n3 + 3 * i;
        acc[c] += V(o) * w0 + V(o + 1) * w1 + V(o + 2) * w2 +
                  Mm(static_cast<std::size_t>(c) * N + i) * wq;
      }
    }
    KOKKOS_INLINE_FUNCTION void init(value_type acc) const {
      for (int c = 0; c < value_count; ++c)
        acc[c] = 0;
    }
    KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
      for (int c = 0; c < value_count; ++c)
        dst[c] += src[c];
    }
  };
  void multiDot(int cnt, const LView<Real>& wv, const LView<Real>& wm, std::vector<Real>& h) {
    const int N = F_.nCells;
    MultiDot f{cnt, 3 * N, N, KV_, KM_, wv, wm};
    Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> res(h.data(), cnt);
    Kokkos::parallel_reduce("sd.mdot", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N), f, res);
  }
  /// The fused multi-axpy: (wv, wm) −= Σ_{c < cnt} h_c (V_c, Mm_c) over the columns of the flat
  /// bases (V, Mm) (3N and N per column), one pass; `norm`: also returns ‖(wv, wm)‖² after it.
  Real multiAxpy(int cnt, const LView<Real>& V, const LView<Real>& Mm, const std::vector<Real>& h,
                 const LView<Real>& wv, const LView<Real>& wm, bool norm) {
    const int N = F_.nCells, n3 = 3 * N;
    Kokkos::deep_copy(
        Kokkos::subview(hd_, std::make_pair(0, cnt)),
        Kokkos::View<const Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h.data(), cnt));
    const LView<Real> hd = hd_;
    Real s = 0;
    Kokkos::parallel_reduce(
        "sd.maxpy", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          Real x[4] = {wv(3 * i), wv(3 * i + 1), wv(3 * i + 2), wm(i)};
          for (int c = 0; c < cnt; ++c) {
            const Real hc = hd(c);
            const std::size_t o = static_cast<std::size_t>(c) * n3 + 3 * i;
            x[0] -= hc * V(o);
            x[1] -= hc * V(o + 1);
            x[2] -= hc * V(o + 2);
            x[3] -= hc * Mm(static_cast<std::size_t>(c) * N + i);
          }
          for (int e = 0; e < 3; ++e)
            wv(3 * i + e) = x[e];
          wm(i) = x[3];
          if (norm)
            acc += x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        },
        s);
    return s;
  }

  /// z = Â⁻¹ r (§14.4.5): Jacobi D_A⁻¹ = (m + a diag ζ)⁻¹, or above prm.amgSwitch (useAmg_, set by
  /// prepareInner) one V-cycle of the viscous hierarchy per Cartesian component. The
  /// preconditioner of kick 1's PCG, of the coupled FGMRES and its guess.
  void applyAhat(Real a, const LView<Real>& r, const LView<Real>& z) {
    using Exec = peclet::core::ExecSpace;
    const int N = F_.nCells, n3 = 3 * N;
    const LView<Real> m = mass_, zd = zdiag_, R = r, Z = z;
    if (!useAmg_) {
      Kokkos::parallel_for(
          "sd.jacobi", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { Z(k) = R(k) / (m(k / 3) + a * zd(k)); });
      return;
    }
    const LView<Real> c0 = c0_, c1 = c1_;
    for (int c = 0; c < 3; ++c) {
      Kokkos::parallel_for(
          "sd.split", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i) { c0(i) = R(3 * i + c); });
      visc_.apply(c0, c1);
      Kokkos::parallel_for(
          "sd.merge", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i) { Z(3 * i + c) = c1(i); });
    }
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
      applyDissipation(plan_, p, q, scr_);
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

  /// x = A⁻¹ rhs, A = M + a ζ, by PCG to the relative residual ‖rhs − A x‖₂ ≤ prm.innerTol ‖rhs‖₂,
  /// from x = 0, or from the guess in x when `fromGuess`. Returns iterations.
  int solveA(Real a, const LView<Real>& rhs, const LView<Real>& x, const SaddleParams& prm,
             bool fromGuess = false) {
    using Exec = peclet::core::ExecSpace;
    const int N = F_.nCells, n3 = 3 * N;
    const LView<Real> m = mass_, X = x, B = rhs;
    if (!(a > Real(0)) || !viscous_) {
      Kokkos::parallel_for(
          "sd.Minv", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { X(k) = B(k) / m(k / 3); });
      return 0;
    }
    const LView<Real> r = ir_, z = iz_, p = ip_, q = iq_;
    Real r0 = 0;
    if (fromGuess) {  // r = rhs − (M + a ζ) x; the stop stays relative to ‖rhs‖₂
      r0 = std::sqrt(detail::dotView(B, B));
      if (!(r0 > Real(0))) {
        Kokkos::deep_copy(X, Real(0));
        return 0;
      }
      applyMassDissipation(plan_, m, a, X, q, scr_);
      Kokkos::parallel_for(
          "sd.r0", Kokkos::RangePolicy<Exec>(0, n3),
          KOKKOS_LAMBDA(const int k) { r(k) = B(k) - q(k); });
      if (std::sqrt(detail::dotView(r, r)) <= prm.innerTol * r0)
        return 0;
    } else {
      Kokkos::deep_copy(X, Real(0));
      Kokkos::deep_copy(r, B);
      r0 = std::sqrt(detail::dotView(r, r));
      if (!(r0 > Real(0)))
        return 0;
    }
    auto precond = [&]() { applyAhat(a, r, z); };
    precond();
    Kokkos::deep_copy(p, z);
    Real rz = detail::dotView(r, z);
    for (int it = 0; it < prm.maxInner; ++it) {
      applyMassDissipation(plan_, m, a, p, q, scr_);  // q = (M + aζ) p, one fused apply
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
  // persistent host copies (M7 WO-PERF3 P3.1): ρ, μ, h, V, the Cahouet–Chabard diagonals, the
  // masses and the facets of the viscous AMG matrix
  std::vector<double> rhoH_, muH_, hH_, volH_, ccH_, massH_;
  HostFacets hostF_;
  /// `dst` = `src` on the device: copied into dst's allocation when it has the length and nothing
  /// else references it, else a new View (`label`) as before.
  static void toDeviceInto(const std::vector<double>& src, LView<Real>& dst, const char* label) {
    if constexpr (std::is_same_v<Real, double>) {
      if (dst.extent(0) == src.size() && dst.use_count() == 1) {
        Kokkos::deep_copy(dst,
                          Kokkos::View<const double*, Kokkos::HostSpace>(src.data(), src.size()));
        return;
      }
    }
    dst = peclet::core::toDevice<Real>(src, label);
  }
  detail::SchurAssembly<Real> schurAsm_;  ///< S₀ on the device (bit-identical to schurHost)
  bool viscous_ = false, useAmg_ = false, assembled_ = false, builtAssembled_ = false;
  double nuH2_ = 0, setupSeconds_ = 0;
  Real cScale_ = 0;
  peclet::core::solver::GraphAMGDevice lrho_, visc_;
  detail::AmgReuse viscReuse_{detail::kReuseSetupVisc};
  long amgClock_ = 0;
  int amgMaxAge_ = 10;
  LView<Real> cc_, ccMin_, r_, z_, zOld_, p_, s_, w_, q_, ir_, iz_, ip_, iq_, zdiag_, c0_, c1_,
      rStar_;
  /// A: FGMRES bases, Krylov (V) and preconditioned (Z), velocity (3N) and multiplier (N) blocks
  LView<Real> KV_, KM_, ZV_, ZM_, hd_;  ///< flat column-major bases (m + 1 / m columns), H column
  DissipationScratch<Real> scr_;
  DissipationPlan<Real> plan_;  ///< the ζ-plan of the last setup (D14.5; viscous only)
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
  /// the warm-started π solve (DESIGN_NOTE §14.4.2, M7 WO-PERF3 P3.2): from the previous π to
  /// ‖Hπ + ∂_wU‖₂ ≤ max(adjointWarmTol ‖r₀‖₂, adjointTol ‖∂_wU‖₂), r₀ the warm-start residual (the
  /// first solve after construction has no previous π: from zero to adjointTol, as before P3.2)
  double adjointWarmTol = 1e-8;
  int maxHalvings = 4;  ///< τ halvings when the weight solve fails (W-Q7)
  /// kick 1: D14.1's pressure kick (kPressure, the default since M7 WO-PERF3 P3.4: one saddle
  /// per step, every P3.4 gate passed; energy ×4.1–4.8 per τ halving), option S (kMidpoint, the
  /// default from M7 WO4′ to P3.4 and the restart step: ×3.9; kStart: ×2.4)
  KickConstraint kickConstraint = KickConstraint::kPressure;
  double repairTol = 1e-10;  ///< repaired-topology volume check (cone fan vs engine), relative
  /// AMG reuse: the age cap, in accepted steps, of the S₀ (L_ρ), H and L_μ hierarchies under the
  /// break-even rule (D14.6, detail::AmgReuse; since M7 WO-PERF3 P3.6 — before: this many setups
  /// with a 1.5× (+2) trigger). 0: rebuilt at every setup / solve (the WO4 behaviour).
  int amgReuse = 10;
  /// relative residual of the linear warm-start (predictor) H solves (x^{n+1}, the option-S
  /// midpoint and the gauge trial) and of the Newton H solves when newtonForcing = 0 (the volumes
  /// still close to weightTol) (WO4: 1e-10 both; WO-PERF 1e-4 / 1e-3; the predictor 1e-3 since M7
  /// WO-PERF3 P3.2, DESIGN_NOTE §14.4.2)
  double warmTol = 1e-3, newtonTol = 1e-3;
  /// Eisenstat–Walker forcing of the per-step Newton H solves (§14.4.2, P3.2): Newton step k to
  /// the relative residual min(newtonForcing, r_k), r_k the max relative volume error; 0: newtonTol
  double newtonForcing = 0.1;
  /// the carry predictor of the x^{n+1} weights (D14.4, §14.4.3, P3.2): w_pred = w_lin + ρ_c cⁿ,
  /// cⁿ the previous step's Newton correction w_final − w_lin, ρ_c = min(carryCap,
  /// (‖dⁿ‖_∞/‖dⁿ⁻¹‖_∞)²), 0 after a restart (construction, a τ-halving) or ‖dⁿ⁻¹‖ = 0; the
  /// carry is dropped (w_lin evaluated) when w_pred's max relative volume error exceeds carryDrop.
  /// carryCap = 0: no carry (w_lin is evaluated first), the default since M7 WO-PERF3 P3.3a step 0:
  /// on the Q4 drop (N 4096) the carry raised the median drift evaluations 4 -> 5 at max|d|/h ~ 0.1
  /// and lowered it at none of the measured drift sizes (4 to enable it: §14.4.3 as designed)
  double carryCap = 0, carryDrop = 1e-2;
  // ---- the gauge G″ (DESIGN_NOTE §12.3.2) ----
  int gaugeEvery = 1;          ///< the gauge after kick 2 every this many steps (0: off)
  double gaugeStrain = 3;      ///< c_g: α_i = min(α_max, c_g τ ε̂_i)
  double gaugeAlphaMax = 0.2;  ///< α_max
  double gaugeSkip = 1e-6;     ///< skip the gauge (state untouched) when max_i α_i < this
  int gaugeRetries = 3;        ///< halvings of all α_i on a rejected trial before the skip
  /// G‴'s activation (§14.4.4; the drift gauge's since M7 WO-PERF3 P3.3b): α_max of G‴ (G″ keeps
  /// gaugeAlphaMax) and the shift scale s_* of the ŝ-feedback (1 + (ŝ_i/s_*)²)
  double gaugeAlphaMaxDrift = 0.5, gaugeShiftScale = 0.15;
  /// the fallback G‴+W (§14.4.4; M7 WO-PERF3 P3.3b): every gaugeWeightEvery-th step (0: off) the
  /// drift gauge adds δ^W = M⁻¹J_xᵀS_J⁻¹(β H w), S_J = J_xM⁻¹J_xᵀ, β = min(gaugeWeightBetaMax,
  /// 4 c_g τ ε̄), S_J⁻¹ by gaugeWeightIters PCG iterations preconditioned by the S₀ hierarchy
  int gaugeWeightEvery = 4, gaugeWeightIters = 5;
  double gaugeWeightBetaMax = 0.2;
  /// kTrial: G″'s trial after kick 2; kDrift: δ in the next drift with G‴'s activation (P3.3b:
  /// the ε̄ floor, the ŝ-feedback, α_max = gaugeAlphaMaxDrift, times the posterior-monitor factor
  /// f_G; no trial OT, no Ũ trial, no M-projection)
  GaugeMode gaugeMode = GaugeMode::kTrial;
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
  // the x^{n+1} weight solve (M7 WO-PERF3 P3.2): its evaluations, the max relative volume error
  // of its first evaluation (r₀ at w_pred), the carry factor ρ_c, and whether the carry was dropped
  int driftEvaluations = 0;
  double otR0 = 0, carryRho = 0, otR0Lin = 0;  ///< otR0Lin: r₀ at w_lin when the carry was dropped
  int carryDrops = 0;
  int halvings = 0;
  int kickRestart = 0;      ///< KickConstraint::kPressure: 1 if this step was a restart (option S)
  int unmatched = 0;        ///< one-sided facets dropped at x^{n+1} (matchFacets)
  int repairFallbacks = 0;  ///< cold rebuilds after the repaired topology failed the check
  int coldStarts = 0;       ///< weight solves restarted from w^n after the warm start failed
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
  // the drift gauge (GaugeMode::kDrift, §14.4.4): gaugePotential is then the posterior ΔŨ_G of
  // the δⁿ this step's drift added, gaugeKinetic 0; gaugeAlphaMax and gaugeCapped describe δⁿ⁺¹
  int gaugeApplied = 0;    ///< δⁿ was added in this step's drift (1) or not (0)
  int gaugeRedos = 0;      ///< drifts redone with δⁿ = 0 (OT failure or min_i m_i ≤ 0)
  int gaugeActive = 0;     ///< cells with δⁿ⁺¹_i ≠ 0
  double gaugeFactor = 0;  ///< f_G after this step's monitor update
  double gaugeBeta = 0;  ///< β of the weight relaxation W added to δⁿ⁺¹ (0: none this step)
  double gaugeWeightRes = 0;  ///< W's PCG residual ‖r₅‖₂/‖r₀‖₂ (0: none this step)
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
    lamPrev_ = LView<Real>("nhvw.lamPrev", n);
    tmpN_ = LView<Real>("nhvw.tmpN", n);
    dw_ = LView<Real>("nhvw.dw", n);
    wLin_ = LView<Real>("nhvw.wLin", n);
    carry_ = LView<Real>("nhvw.carry", n);
    pos0_ = LView<Real>("nhvw.pos0", n3);
    w0_ = LView<Real>("nhvw.w0", n);
    wrapS_ = LView<Real>("nhvw.wrap", n3);
    gd_ = LView<Real>("nhvw.gaugeDelta", n3);
    dW_ = LView<Real>("nhvw.gaugeWeightDisp", n3);
    wr_ = LView<Real>("nhvw.gaugeWeightR", n);
    wz_ = LView<Real>("nhvw.gaugeWeightZ", n);
    wp_ = LView<Real>("nhvw.gaugeWeightP", n);
    wq_ = LView<Real>("nhvw.gaugeWeightQ", n);
    wy_ = LView<Real>("nhvw.gaugeWeightY", n);
    dd_ = LView<Real>("nhvw.driftDisp", n3);
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
      s0Reuse_.failed = hReuse_.failed = true;  // D14.6: rebuild after a failed attempt
      saddle_.markAmgFailed();
      tau *= 0.5;
      ++rep.halvings;
      kickRestart_ = true;  // kPressure: the retry is an option-S step
    }
    time_ += tau;
    ++amgClock_;  // the hierarchies' age is counted in accepted steps (D14.6)
    if (prm_.kickConstraint == KickConstraint::kPressure) {
      if (kickRestart_)  // §14.4.1 step 0: λⁿ⁻¹ := λⁿ := the restart step's λ₂
        Kokkos::deep_copy(lamPrev_, lam2_);
      kickRestart_ = false;
      tauPrev_ = tau;
    }
    if (prm_.gaugeMode == GaugeMode::kDrift) {
      Kokkos::Timer tg;
      gaugeMonitor(rep);
      rep.tGauge = tg.seconds();
      if (prm_.gaugeEvery > 0 && ++stepCount_ % prm_.gaugeEvery == 0)
        driftGauge(tau, rep);
    } else if (prm_.gaugeEvery > 0 && ++stepCount_ % prm_.gaugeEvery == 0) {
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
    detail::hostCopy(spec_.tension, tenH_);
    const std::vector<double>& ten = tenH_;
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
    detail::hostCopy(h_, hH_);
    const std::vector<double>& hh = hH_;
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
  /// Kick 1's mode; the next step is a restart (kPressure: an option-S step).
  void setKickConstraint(KickConstraint k) {
    prm_.kickConstraint = k;
    kickRestart_ = true;
  }
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
  /// M7 WO-PERF3 accuracy knobs (provisional, opt-in; exact stays the default): η_K, the relative
  /// tolerance of the kick saddles (D14.3: ‖C v‖_∞ ≤ η_K ‖C‖ v_ref and ‖r_v‖₂ ≤ η_K ‖b‖₂; default
  /// 1e-9), and the OT volume tolerance max_i |V_i − V⁰_i| / V⁰_i (default 1e-12).
  void setKickTolerance(double eta) {
    prm_.saddle.tol = eta;
    prm_.saddle.innerTol = eta;
  }
  void setVolumeTolerance(double tol) { prm_.weightTol = tol; }
  void setGauge(int every, double strain, double alphaMax) {
    prm_.gaugeEvery = every;
    prm_.gaugeStrain = strain;
    prm_.gaugeAlphaMax = alphaMax;
    deltaPending_ = false;  // a pending drift δ belongs to the old parameters
  }
  /// M7 WO-PERF3 P3.3a (private perf knob): the gauge's displacement in a trial after kick 2
  /// (G″) or in the next drift (§14.4.4). A pending drift δ is dropped.
  void setGaugeMode(GaugeMode mode) {
    prm_.gaugeMode = mode;
    deltaPending_ = false;
  }
  /// f_G of the drift gauge's posterior monitor (§14.4.4; 1 at construction): a re-prepared
  /// integrator continues the old run's factor.
  double gaugeFactor() const { return gaugeFactor_; }
  void setGaugeFactor(double f) { gaugeFactor_ = f; }
  double theta() const { return prm_.theta; }
  double time() const { return time_; }
  void setTime(double t) { time_ = t; }
  /// The state a re-prepared integrator inherits beyond v, t and the accumulators (the relabelling
  /// of LagrangianFlow): the last λ₂ (kick 1's warm start, cell order of this integrator) and the
  /// step count (the gauge cadence).
  long stepCount() const { return stepCount_; }
  void setWarmStart(const std::vector<Real>& lambda, long stepCount) {
    if (static_cast<int>(lambda.size()) != N_)
      throw std::logic_error("voro: NHV-W setWarmStart: lambda is not (N,)");
    Kokkos::deep_copy(lam2_, Kokkos::View<const Real*, Kokkos::HostSpace>(lambda.data(), N_));
    stepCount_ = stepCount;
  }
  /// The 'pressure' kick's history (KickConstraint::kPressure, §14.4.1): λⁿ⁻¹ (N), v^{n−½} (3N,
  /// kick 1's guess) and τ_prev, in this integrator's cell order; empty when the next step is a
  /// restart anyway (another mode, or a restart pending). A re-prepared integrator (the relabelling
  /// of LagrangianFlow) continues it, as it continues λⁿ (setWarmStart); call setKickHistory after
  /// setKickConstraint, which makes the next step a restart.
  struct KickHistory {
    std::vector<Real> lamPrev, vHalf;
    double tauPrev = 0;
  };
  KickHistory kickHistory() const {
    KickHistory h;
    if (prm_.kickConstraint != KickConstraint::kPressure || kickRestart_)
      return h;
    h.lamPrev = peclet::core::toVector(lamPrev_);
    h.vHalf = peclet::core::toVector(vh_);
    h.tauPrev = tauPrev_;
    return h;
  }
  void setKickHistory(const KickHistory& h) {
    if (h.lamPrev.empty())
      return;
    if (static_cast<int>(h.lamPrev.size()) != N_ || static_cast<int>(h.vHalf.size()) != 3 * N_)
      throw std::logic_error("voro: NHV-W setKickHistory: lamPrev is not (N,) or vHalf not (N, 3)");
    Kokkos::deep_copy(lamPrev_, Kokkos::View<const Real*, Kokkos::HostSpace>(h.lamPrev.data(), N_));
    Kokkos::deep_copy(vh_,
                      Kokkos::View<const Real*, Kokkos::HostSpace>(h.vHalf.data(), h.vHalf.size()));
    tauPrev_ = h.tauPrev;
    kickRestart_ = false;
  }
  /// The drift gauge's pending δⁿ (GaugeMode::kDrift; (3N) in this integrator's cell order, empty
  /// when none is pending) and its restore: a re-prepared integrator (the relabelling of
  /// LagrangianFlow) continues the old run's δⁿ, as it continues f_G. Call setPendingDelta after
  /// setGauge/setGaugeMode, which drop a pending δ.
  std::vector<Real> pendingDelta() const {
    std::vector<Real> d;
    if (deltaPending_) {
      d.resize(3 * static_cast<std::size_t>(N_));
      Kokkos::deep_copy(Kokkos::View<Real*, Kokkos::HostSpace>(d.data(), d.size()), gd_);
    }
    return d;
  }
  void setPendingDelta(const std::vector<Real>& d) {
    if (d.empty()) {
      deltaPending_ = false;
      return;
    }
    if (static_cast<int>(d.size()) != 3 * N_)
      throw std::logic_error("voro: NHV-W setPendingDelta: delta is not (N, 3)");
    Kokkos::deep_copy(gd_, Kokkos::View<const Real*, Kokkos::HostSpace>(d.data(), d.size()));
    deltaPending_ = true;
  }
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
  /// The static pressure P = −(π − e) of the current state (M7 DESIGN_NOTE_VARIANTS D15.2), the
  /// total derivative dŨ/dV⁰_i: e_i = ∂U/∂V⁰_i at fixed (x, w), which only E_M has (facet energies:
  /// P = −π exactly). Output only: E_M is re-evaluated on a fresh export with the higher moments
  /// into scratch, and no state, counter or cache of the integrator is touched. The reported
  /// pressure is P − λ.
  LView<Real> staticPressure() {
    LView<Real> out("nhvw.staticP", N_);
    const LView<Real> p = pi_;
    const bool nonlocal =
        spec_.tension.extent(0) > 0 && spec_.interfaceEnergy == InterfaceEnergy::Nonlocal;
    if (!nonlocal) {
      Kokkos::parallel_for(
          "nhvw.staticP", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i) { out(i) = -p(i); });
      return out;
    }
    CellFacets<Real> Fm = exportCellFacets<Real, Power>(mt_->store, pos_, weight_, N_, L_,
                                                        mt_->wall, mt_->xRef, mt_->vol, true);
    matchFacets(Fm);
    LView<Real> gx("nhvw.staticP.gx", 3 * static_cast<std::size_t>(N_)), e("nhvw.staticP.e", N_);
    nonlocalInterfaceEnergy(Fm, pos_, spec_.types, spec_.tension, spec_.nTypes, spec_.refVolume,
                            spec_.nonlocal, L_, gx, LView<Real>(), nullptr, e);
    Kokkos::parallel_for(
        "nhvw.staticP", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i) { out(i) = -(p(i) - e(i)); });
    return out;
  }
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
    wp.scratch = &wss_;
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
    wp.forcingMax = prm_.newtonForcing;
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
                                        mt_->vol, false, &pool_);
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
                                         mt_->vol, false, &pool_);
  }

  /// saddle_.setup at F_; the S₀ hierarchy is built here only when there is none — otherwise
  /// lazily, right before a kick saddle, at that saddle's geometry, when the break-even rule says
  /// so (s0Reuse_, kick()).
  void setupSaddle() {
    saddle_.setAssembled(prm_.saddle.schurAssembled);
    saddle_.setAmgClock(amgClock_, prm_.amgReuse);
    const bool first = !saddle_.hasSchur();
    saddle_.setup(F_, rho_, mu_, mass_, h_, first);
    if (first)
      s0Reuse_.built(amgClock_);
  }

  /// Facets, U and ∇Ũ, π, and the two hierarchies at the current (pos, weight). `ready`: the
  /// facets at this state, already exported and matched (with their match statistics `rm`).
  void evalState(double* tPot = nullptr, const CellFacets<Real>* ready = nullptr,
                 const FacetMatch* rm = nullptr) {
    Kokkos::Timer t, tp;
    zetaValid_ = false;
    if (ready) {
      F_ = *ready;
      match_ = *rm;
    } else {
      F_ = currentFacets();
      match_ = matchFacets(F_, &pool_);
    }
    Kokkos::fence();
    evalSplit_[0] = tp.seconds();
    tp.reset();
    Kokkos::deep_copy(gx_, Real(0));
    Kokkos::deep_copy(gw_, Real(0));
    parts_ = partitionPotential(*mt_, F_, pos_, weight_, L_, spec_, gx_, gw_, &pots_);
    Kokkos::fence();
    evalSplit_[1] = tp.seconds();
    tp.reset();
    const peclet::core::solver::AmgParams ap = detail::m7AmgParams();
    if (hAmg_.numLevels() == 0 || hReuse_.due(amgClock_, prm_.amgReuse)) {
      hAmg_.build(detail::pinnedHostH(F_, &wss_), ap);
      hReuse_.built(amgClock_);
    }
    evalSplit_[2] = tp.seconds();
    tp.reset();
    WeightSolveParams wp;
    wp.pcgTol = prm_.adjointTol;
    wp.amg = &hAmg_;
    wp.scratch = &wss_;
    wp.warmStart = piValid_;
    wp.warmRelTol = prm_.adjointWarmTol;
    adj_ = reducedGradient(F_, gx_, gw_, pi_, grad_, wp);
    piValid_ = true;
    hReuse_.solved(adj_.pcgIters);  // H's break-even count: the adjoint solve of each evaluation
    Kokkos::fence();
    evalSplit_[3] = tp.seconds();
    setupSaddle();
    Kokkos::fence();
    if (tPot)
      *tPot = t.seconds();
  }

  /// One kick at the current state: vNew, lam from vOld; adds the heat and the body work.
  /// `pressure` (kick 1 of a KickConstraint::kPressure step, §14.4.1 step 1): no constraint;
  /// `lam` is λ̄ (input), (τ/2)Cᵀλ̄ joins the right-hand side and vNew holds the A-solve's guess.
  /// `zoReady`: zo_ already holds ζ vOld at this geometry (the last kick 2's, §14.4.1 step 1).
  SaddleStats kick(double tau, const LView<Real>& vOld, const LView<Real>& vNew,
                   const LView<Real>& lam, const LView<Real>& posForce, bool pressure = false,
                   bool zoReady = false) {
    const Real th = prm_.theta, ht = Real(0.5) * tau;
    const bool visc = saddle_.viscous();
    if (visc && !zoReady)
      saddle_.applyZeta(vOld, zo_);
    const LView<Real> m = mass_, vo = vOld, b = b_, zo = zo_, g = grad_, P = posForce;
    if (!visc)
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
    SaddleStats st;
    if (pressure) {
      // b −= (τ/2) Cᵀλ̄ (zn_ is free until the heat below); A v = b, no constraint
      applyDivergenceT(F_, lam, zn_);
      const LView<Real> ct = zn_;
      Kokkos::parallel_for(
          "nhvw.rhsLam", Kokkos::RangePolicy<Exec>(0, 3 * N_),
          KOKKOS_LAMBDA(const int k) { b(k) -= ht * ct(k); });
      st = saddle_.solveMomentum(th * ht, ht, b_, vNew, prm_.saddle);
    } else {
      const Real tolAbs = prm_.saddle.tol * maxV0_ / tau;
      if (s0Reuse_.due(amgClock_, prm_.amgReuse)) {  // D14.6: rebuilt before the next solve
        saddle_.buildSchur();
        s0Reuse_.built(amgClock_);
      }
      st = saddle_.solve(th * ht, ht, tolAbs, b_, vNew, lam, prm_.saddle);
      s0Reuse_.solved(st.outer);
      if (st.amgInner)
        saddle_.recordViscSolve(st.outer);
      st.constraint *= tau / maxV0_;
    }
    // heat (τ/2) v̄_θᵀ ζ (v_old + v_new)/2 and body work (τ/2) (M a)ᵀ (v_old + v_new)/2
    const LView<Real> vn = vNew, zn = zn_;
    if (visc)
      saddle_.applyZeta(vNew, zn_);
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
    // the carry (§14.4.3) holds only from a step completed at full τ: a failed attempt (the
    // τ-halving retry) restarts it
    const bool carryOk = carryOk_;
    carryOk_ = false;
    const double heat0 = heat_, work0 = work_, qc0 = qc_;  // kick 1's increments undone on a retry
    const bool pressure = prm_.kickConstraint == KickConstraint::kPressure && !kickRestart_;
    const bool zetaReady = zetaValid_;  // zn_ = ζ vel_ at F_ (the last kick 2), consumed here
    zetaValid_ = false;
    rep.kickRestart = prm_.kickConstraint == KickConstraint::kPressure && kickRestart_ ? 1 : 0;
    if (pressure) {
      // ---- kick 1, D14.1 (§14.4.1 step 1): λ̄ = ¾λⁿ + ¼λⁿ⁻¹ through Cⁿᵀ inside the CN A-solve at
      // x^n, no constraint; guess v^{n−½} + (1 + τ/τ_prev)(vⁿ − v^{n−½}); constraint work
      // Q_c += (τ/4) λ̄ᵀCⁿ(vⁿ + v^{½}) ----
      {
        const LView<Real> lb = lam1_, ln = lam2_, lp = lamPrev_, vh = vh_, v = vel_;
        Kokkos::parallel_for(
            "nhvw.lamBar", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i) { lb(i) = Real(0.75) * ln(i) + Real(0.25) * lp(i); });
        const Real c = Real(1) + Real(tau / tauPrev_);
        Kokkos::parallel_for(
            "nhvw.kickGuess", Kokkos::RangePolicy<Exec>(0, 3 * N_),
            KOKKOS_LAMBDA(const int k) { vh(k) += c * (v(k) - vh(k)); });
      }
      // Bⁿvⁿ uses the stored ζⁿvⁿ: kick 2's ζⁿ⁺¹vⁿ⁺¹ of the last step when neither the geometry
      // nor v moved since (D14.5; not after an accepted trial-gauge move) — the same numbers
      const bool zoReady = zetaReady && saddle_.viscous();
      if (zoReady)
        std::swap(zo_, zn_);
      rep.kick1 = kick(tau, vel_, vh_, lam1_, pos_, /*pressure=*/true, zoReady);
      {
        const LView<Real> s = b_, v = vel_, vh = vh_;
        Kokkos::parallel_for(
            "nhvw.vSum", Kokkos::RangePolicy<Exec>(0, 3 * N_),
            KOKKOS_LAMBDA(const int k) { s(k) = v(k) + vh(k); });
        applyDivergence(F_, b_, tmpN_);
        qc_ += 0.25 * tau * detail::dotView(lam1_, tmpN_);
      }
    } else {
      // ---- kick 1 at x^n (λ₁ warm-started from the last λ₂, same geometry) ----
      Kokkos::deep_copy(lam1_, lam2_);
      if (prm_.kickConstraint != KickConstraint::kStart) {  // option S (also a kPressure restart)
        // option S (DESIGN_NOTE §10.3.6): C of kick 1 (and its ζ) at x^n + (τ/2)v^n, the weights
        // solved there (repair, warm start H δw = −(τ/2) J_x v^n); ∇Ũ and the body force stay at
        // x^n
        const CellFacets<Real> Fn = F_;
        if (!moveAndSolve(0.5 * tau, vel_, rep)) {
          heat_ = heat0;
          work_ = work0;
          qc_ = qc0;
          return false;
        }
        F_ = currentFacets();
        fcValid_ = false;  // pos/weight return to x^n below
        matchFacets(F_, &pool_);
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
    }
    Kokkos::fence();
    rep.tKick1 = t.seconds();
    t.reset();
    // ---- the drift gauge's δⁿ (GaugeMode::kDrift, §14.4.1 step 2): dⁿ = τ v^{½} + δⁿ; on an OT
    // failure or a seed on/behind its own facet the drift is redone from here with δⁿ = 0 ----
    const bool withDelta = deltaPending_;
    deltaPending_ = false;
    deltaUsed_ = false;
    if (withDelta)  // the monitor's first half Σ_i ∇_iŨⁿ·δⁿ_i (∇Ũ is still at x^n)
      gaugeGradDelta0_ = double(detail::dotView(grad_, gd_));
    int driftEvals = 0;  // the evaluations of a drift redone with δⁿ = 0
    CellFacets<Real> Fnext;
    FacetMatch mNext;
    bool nextReady = false;
    for (bool useDelta = withDelta;; useDelta = false) {
      // ---- warm start H δw = −τ J_x v^{½} (at x^n); with δⁿ: H δw = −J_x dⁿ ----
      if (useDelta) {
        const LView<Real> dd = dd_, v = vh_, g = gd_;
        const Real tt = tau;
        Kokkos::parallel_for(
            "nhvw.driftDisp", Kokkos::RangePolicy<Exec>(0, 3 * N_),
            KOKKOS_LAMBDA(const int k) { dd(k) = tt * v(k) + g(k); });
      }
      {
        applyVolumeJacobian(F_, useDelta ? dd_ : vh_, tmpN_);
        const LView<Real> r = tmpN_;
        Real s = 0;
        Kokkos::parallel_reduce(
            "nhvw.warmMean", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i, Real& acc) { acc += r(i); }, s);
        const Real mean = s / N_, mt = useDelta ? Real(-1) : Real(-tau);
        Kokkos::parallel_for(
            "nhvw.warmRhs", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i) { r(i) = i == 0 ? Real(0) : mt * (r(i) - mean); });
        WeightSolveParams wp;
        wp.pcgTol = prm_.warmTol;
        wp.amg = &hAmg_;
        wp.scratch = &wss_;
        rep.warmIters = solvePinnedH(F_, tmpN_, dw_, wp).first;
        const LView<Real> w = weight_, dw = dw_;
        Kokkos::parallel_for(
            "nhvw.warm", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i) { w(i) += dw(i); });
      }
      // ---- the carry predictor w_pred = w_lin + ρ_c cⁿ (D14.4, §14.4.3 step 1) ----
      const double dn =
          useDelta ? double(detail::maxAbsView(dd_)) : tau * double(detail::maxAbsView(vh_));
      double rho = 0;
      if (carryOk && dPrev_ > 0) {
        const double q = dn / dPrev_;
        rho = std::min(prm_.carryCap, q * q);
      }
      rep.carryRho = rho;
      Kokkos::deep_copy(wLin_, weight_);
      if (rho > 0) {
        const LView<Real> w = weight_, c = carry_;
        const Real rc = rho;
        Kokkos::parallel_for(
            "nhvw.carry", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i) { w(i) += rc * c(i); });
      }
      // ---- drift (wrapped into the box; the lattice shift is kept for the gravity jump) ----
      if (useDelta) {
        const LView<Real> P = pos_, dd = dd_, S = wrapS_;
        const Kokkos::Array<Real, 3> L3 = {L_[0], L_[1], L_[2]};
        Kokkos::parallel_for(
            "nhvw.driftDelta", Kokkos::RangePolicy<Exec>(0, 3 * N_), KOKKOS_LAMBDA(const int k) {
              const Real Lk = L3[k % 3];
              const Real x = P(k) + dd(k);
              const Real sh = Lk * Kokkos::floor(x / Lk);
              P(k) = x - sh;
              S(k) = sh;
            });
      } else {
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
      wp.forcingMax = prm_.newtonForcing;
      wp.repair = true;
      wp.amg = &hAmg_;
      wp.scratch = &wss_;
      wp.fallback = rho > 0 ? &wLin_ : nullptr;  // r₀ > carryDrop: drop the carry, evaluate w_lin
      wp.fallbackAbove = prm_.carryDrop;
      WeightSolveResult ws = solveWeights(*mt_, pos_, weight_, V0h_, L_, wp);
      wp.fallback = nullptr;
      rep.otR0 = ws.r0;
      rep.carryDrops = ws.fallback ? 1 : 0;
      rep.otR0Lin = ws.rFallback;
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
      rep.driftEvaluations = driftEvals + ws.evaluations;
      rep.coldRebuilds = ws.coldRebuilds;
      rep.weightErr = ws.maxRelErr;
      rep.tRepair += ws.evalSeconds;
      Kokkos::fence();
      rep.tWeights = t.seconds() - ws.evalSeconds;
      if (!ws.converged) {
        if (useDelta) {  // §14.4.1 step 3: redo the drift with δⁿ = 0
          ++rep.gaugeRedos;
          driftEvals = rep.driftEvaluations;
          restoreEngine();
          continue;
        }
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
        rep.driftEvaluations += wr.evaluations;
        rep.weightErr = wr.maxRelErr;
        if (!wr.converged) {
          if (useDelta) {
            ++rep.gaugeRedos;
            driftEvals = rep.driftEvaluations;
            restoreEngine();
            continue;
          }
          heat_ = heat0;
          work_ = work0;
          qc_ = qc0;
          return false;
        }
      }
      // ---- with δⁿ: every seed strictly inside its own cell at x^{n+1} (min_i m_i > 0), read
      // from the export evalState then uses ----
      if (useDelta) {
        Fnext = currentFacets();
        mNext = matchFacets(Fnext, &pool_);
        cellMargins(Fnext, tmpN_);
        const LView<Real> m = tmpN_;
        Real mmin = 0;
        Kokkos::parallel_reduce(
            "nhvw.driftMargin", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i, Real& acc) { acc = m(i) < acc ? m(i) : acc; },
            Kokkos::Min<Real>(mmin));
        if (!(mmin > Real(0))) {
          ++rep.gaugeRedos;
          driftEvals = rep.driftEvaluations;
          Fnext = CellFacets<Real>();  // release the pool buffers
          restoreEngine();
          continue;
        }
        nextReady = true;
      }
      // ---- the next step's carry: cⁿ = w_final − w_lin, ‖dⁿ‖_∞ ----
      {
        const LView<Real> c = carry_, w = weight_, wl = wLin_;
        Kokkos::parallel_for(
            "nhvw.carryStore", Kokkos::RangePolicy<Exec>(0, N_),
            KOKKOS_LAMBDA(const int i) { c(i) = w(i) - wl(i); });
        dPrev_ = dn;
        carryOk_ = true;
      }
      deltaUsed_ = useDelta;
      rep.gaugeApplied = useDelta ? 1 : 0;
      break;
    }
    // ---- π, ∇Ũ and the hierarchies at x^{n+1} ----
    double tPot = 0;
    if (nextReady)
      evalState(&tPot, &Fnext, &mNext);
    else
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
    if (pressure)  // λⁿ⁻¹ := λⁿ; kick 2 warm-started from λⁿ (D14.7)
      Kokkos::deep_copy(lamPrev_, lam2_);
    else
      Kokkos::deep_copy(lam2_, lam1_);
    rep.kick2 = kick(tau, vh_, vel_, lam2_, pos_);
    zetaValid_ = true;  // zn_ = ζ vel_ at F_ (or 0, inviscid) until F_ or vel_ moves
    applyDivergence(F_, vh_, tmpN_);
    qc_ += 0.25 * tau * detail::dotView(lam2_, tmpN_);
    kinetic_ = kineticEnergy();
    Kokkos::fence();
    rep.tKick2 = t.seconds();
    return true;
  }

  /// The gauge's activation into aG_: ε̇_i = ‖sym(G_i v)‖_F, ε̂_i its max over i and i's face
  /// neighbours.
  ///  G″ (§12.3.2 step 1; g3 false): α_i = fG · min(α_max, c_g τ ε̂_i), fG = 1.
  ///  G‴ (§14.4.4; g3 true, the drift gauge): ε̄ = (Σ_i V_i ε̇_i² / Σ_i V_i)^{1/2} (V_i the held
  ///  volumes V⁰_i), ŝ_i = max_{f∋i} |s_f|/d_f (the facets to neighbour cells, as shifts()),
  ///  α_i = fG · min(α_max, c_g τ max(ε̂_i, ε̄)(1 + (ŝ_i/s_*)²)), α_max = gaugeAlphaMaxDrift,
  ///  s_* = gaugeShiftScale, fG the posterior monitor's f_G.
  /// Returns max_i α_i.
  double gaugeActivation(double tau, double fG, bool g3 = false) {
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
    if (g3)
      return gaugeActivationG3(tau, fG);
    {
      const CellFacets<Real> G = F_;
      const LView<Real> e = tmpN_, al = aG_;
      const Real cg = prm_.gaugeStrain * tau, amax = prm_.gaugeAlphaMax, fg = fG;
      Real am = 0;
      Kokkos::parallel_reduce(
          "nhvw.gaugeAlpha", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& acc) {
            Real m = e(i);
            for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
              const Real q = e(G.nbr(f));
              m = q > m ? q : m;
            }
            const Real a = fg * (cg * m < amax ? cg * m : amax);
            al(i) = a;
            acc = a > acc ? a : acc;
          },
          Kokkos::Max<Real>(am));
      return am;
    }
  }

  /// G‴'s α_i (see gaugeActivation) from ε̇_i in tmpN_.
  double gaugeActivationG3(double tau, double fG) {
    double ebar = 0;
    {
      const LView<Real> e = tmpN_, V = V0_;
      Real sve = 0, sv = 0;
      Kokkos::parallel_reduce(
          "nhvw.gaugeStrainMean", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, Real& a, Real& b) {
            a += V(i) * e(i) * e(i);
            b += V(i);
          },
          sve, sv);
      ebar = sv > 0 ? std::sqrt(double(sve) / double(sv)) : 0.0;
    }
    gaugeEbar_ = ebar;
    const CellFacets<Real> G = F_;
    const LView<Real> e = tmpN_, al = aG_, w = weight_;
    const Real cg = prm_.gaugeStrain * tau, amax = prm_.gaugeAlphaMaxDrift, fg = fG, eb = ebar,
               sinv = Real(1) / prm_.gaugeShiftScale;
    Real am = 0;
    Kokkos::parallel_reduce(
        "nhvw.gaugeAlpha3", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          Real m = e(i) > eb ? e(i) : eb, sh = 0;
          for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
            const int j = G.nbr(f);
            const Real q = e(j);
            m = q > m ? q : m;
            Real d2 = 0;
            for (int c = 0; c < 3; ++c)
              d2 += G.conn(3 * f + c) * G.conn(3 * f + c);
            const Real sf = Kokkos::fabs(w(i) - w(j)) / (Real(2) * d2);
            sh = sf > sh ? sf : sh;
          }
          const Real r = sh * sinv, c = cg * m * (Real(1) + r * r);
          const Real a = fg * (c < amax ? c : amax);
          al(i) = a;
          acc = a > acc ? a : acc;
        },
        Kokkos::Max<Real>(am));
    return am;
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
    {
      const double am = gaugeActivation(tau, 1);
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
        matchFacets(Ft, &pool_);
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
              partitionPotential(*mt_, Ft, pos_, weight_, L_, spec_, gx_, gw_, &pots_);
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
      zetaValid_ = false;
      kinetic_ = kineticEnergy();
      dK = kinetic_ - K0;
    }
    rep.gaugePotential = dU;
    rep.gaugeKinetic = dK;
    Kokkos::fence();
    rep.tGauge = t.seconds();
  }

  /// The drift gauge's posterior monitor (§14.4.4) for the δⁿ this step's drift added (nothing
  /// when it added none): ΔŨ_G = ½ Σ_i (∇_iŨⁿ + ∇_iŨⁿ⁺¹)·δⁿ_i, the δ-share of the trapezoidal
  /// path integral; Q_G −= ΔŨ_G (gaugeU_ += ΔŨ_G, the ledger's gauge counter); f_G ← max(f_G/2,
  /// ⅛) if ΔŨ_G > 0, else min(1, 2 f_G).
  void gaugeMonitor(StepReport& rep) {
    if (deltaUsed_) {
      const double dU = 0.5 * (gaugeGradDelta0_ + double(detail::dotView(grad_, gd_)));
      gaugeU_ += dU;
      rep.gaugePotential = dU;
      gaugeFactor_ =
          dU > 0 ? std::max(0.5 * gaugeFactor_, 0.125) : std::min(1.0, 2.0 * gaugeFactor_);
    }
    rep.gaugeFactor = gaugeFactor_;
  }

  /// The weight-relaxation displacement of G‴+W (§14.4.4) into dW_, at x^{n+1}, w^{n+1}:
  /// δ^W = M⁻¹J_xᵀ y, S_J y = β H w, S_J = J_xM⁻¹J_xᵀ (matrix-free), β = min(β_max, 4 c_g τ ε̄),
  /// y by gaugeWeightIters PCG iterations from 0, preconditioned by one V-cycle of kick 2's S₀
  /// hierarchy at x^{n+1}, no tolerance. Linearised, H δw = −J_xδ^W = −βHw: δw = −βw.
  /// (gaugeDisplacement then makes it tangential on I and 0 on CL, and applies the ascent
  /// projection and the margin cap to the sum δ = α r + δ^W.)
  void weightRelaxation(double tau, StepReport& rep) {
    const double beta =
        std::min(prm_.gaugeWeightBetaMax, 4.0 * prm_.gaugeStrain * tau * gaugeEbar_);
    rep.gaugeBeta = beta;
    const LView<Real> r = wr_, z = wz_, p = wp_, q = wq_, y = wy_, m = mass_, u = dW_;
    applyWeightHessian(F_, weight_, r);
    const Real b = beta;
    Kokkos::parallel_for(
        "nhvw.gaugeWRhs", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
          r(i) *= b;
          y(i) = 0;
        });
    const double r0 = std::sqrt(double(detail::dotView(r, r)));
    double rz = 0;
    for (int k = 0; k < prm_.gaugeWeightIters && r0 > 0; ++k) {
      saddle_.applySchurHierarchy(r, z);
      const double rzNew = double(detail::dotView(r, z));
      const Real bt = k == 0 ? Real(0) : Real(rzNew / rz);
      rz = rzNew;
      Kokkos::parallel_for(
          "nhvw.gaugeWDir", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i) { p(i) = z(i) + bt * p(i); });
      applyWeightSchur(p, q);
      const double pq = double(detail::dotView(p, q));
      if (!(pq > 0))
        break;
      const Real al = Real(rz / pq);
      Kokkos::parallel_for(
          "nhvw.gaugeWUpd", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
            y(i) += al * p(i);
            r(i) -= al * q(i);
          });
    }
    rep.gaugeWeightRes = r0 > 0 ? std::sqrt(double(detail::dotView(r, r))) / r0 : 0.0;
    applyVolumeJacobianT(F_, y, u);
    Kokkos::parallel_for(
        "nhvw.gaugeWDisp", Kokkos::RangePolicy<Exec>(0, 3 * N_),
        KOKKOS_LAMBDA(const int k) { u(k) /= m(k / 3); });
  }

  /// out = S_J x = J_x M⁻¹ J_xᵀ x (dW_ as the 3N scratch).
  void applyWeightSchur(const LView<Real>& x, const LView<Real>& out) {
    const LView<Real> u = dW_, m = mass_;
    applyVolumeJacobianT(F_, x, u);
    Kokkos::parallel_for(
        "nhvw.gaugeWMinv", Kokkos::RangePolicy<Exec>(0, 3 * N_),
        KOKKOS_LAMBDA(const int k) { u(k) /= m(k / 3); });
    applyVolumeJacobian(F_, u, out);
  }

  /// The drift gauge (GaugeMode::kDrift; DESIGN_NOTE §14.4.4, M7 WO-PERF3 P3.3a/b) at
  /// x = x^{n+1}, v = v^{n+1}: G‴'s α_i (gaugeActivation, g3); max_i α_i < gaugeSkip:
  /// δⁿ⁺¹ ≡ 0 and the next drift adds nothing (bit for bit). Otherwise δⁿ⁺¹ = G″ steps 2-4
  /// (gaugeDisplacement: the raw Lloyd move, the per-cell ascent projection with ∇Ũⁿ⁺¹, the
  /// ½-margin cap from this state's export) into gd_, pending for the next drift. No trial OT, no
  /// Ũ trial, no M-projection: the state is untouched here.
  void driftGauge(double tau, StepReport& rep) {
    Kokkos::Timer t;
    const double am = gaugeActivation(tau, gaugeFactor_, true);
    rep.gaugeAlphaMax = am;
    if (am >= prm_.gaugeSkip) {
      const bool withW = prm_.gaugeWeightEvery > 0 && stepCount_ % prm_.gaugeWeightEvery == 0;
      if (withW)
        weightRelaxation(tau, rep);
      rep.gaugeInterface = gaugeDisplacement(rep, withW);
      const LView<Real> d = gd_;
      int nA = 0;
      Kokkos::parallel_reduce(
          "nhvw.gaugeActive", Kokkos::RangePolicy<Exec>(0, N_),
          KOKKOS_LAMBDA(const int i, int& acc) {
            if (d(3 * i) != Real(0) || d(3 * i + 1) != Real(0) || d(3 * i + 2) != Real(0))
              ++acc;
          },
          nA);
      rep.gaugeActive = nA;
      deltaPending_ = nA > 0;
    }
    Kokkos::fence();
    rep.tGauge += t.seconds();
  }

  /// G″ steps 2-4 (see gauge()): δ (gd_) from α (aG_); returns |I| and counts the normal-part,
  /// projected and capped cells into `rep`.
  int gaugeDisplacement(StepReport& rep, bool withW = false) {
    const CellFacets<Real> G = F_;
    const auto T = spec_.types;
    const LView<Real> a = aI_, g = grad_, d = gd_, h = h_, P = pos_, w = weight_, al = aG_,
                      mg = mG_, dW = dW_;
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
          Real di[3] = {0, 0, 0}, tn[3] = {0, 0, 0};  // tn: I's unit normal (0: none)
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
              for (int c = 0; c < 3; ++c)
                tn[c] = nh[c];
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
          // G‴+W: + δ^W_i, tangential on I, 0 on CL
          if (withW && cls(i) != 2) {
            Real e[3] = {dW(3 * i), dW(3 * i + 1), dW(3 * i + 2)};
            const Real en = tn[0] * e[0] + tn[1] * e[1] + tn[2] * e[2];
            for (int c = 0; c < 3; ++c)
              di[c] += e[c] - en * tn[c];
          }
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
      lam2_, tmpN_, dw_, pos0_, w0_, wrapS_, gd_, aI_, lamG_, gs_, fw_, aG_, mG_, wLin_, carry_,
      dd_, dW_, wr_, wz_, wp_, wq_, wy_, lamPrev_;
  // the 'pressure' kick (KickConstraint::kPressure): the next step is a restart (option S), and the
  // previous step's τ (kick 1's guess v^{n−½} + (1 + τ/τ_prev)(vⁿ − v^{n−½})); λⁿ⁻¹ in lamPrev_
  bool kickRestart_ = true;
  double tauPrev_ = 0;
  // the carry predictor (IntegratorParams::carryCap): ‖dⁿ⁻¹‖_∞ and whether cⁿ is valid (false
  // after construction and after a failed attempt); π holds a previous solve (the adjoint warm
  // start)
  double dPrev_ = 0;
  bool carryOk_ = false, piValid_ = false;
  Kokkos::View<int*, Mem> cls_, iList_;
  bool movingWall_ = false;
  long stepCount_ = 0;
  int restoreRebuilds_ = 0;
  // break-even AMG reuse (D14.6): S₀ / L_ρ and H; the step clock their age is counted in (L_μ's
  // state lives in the SaddleSolver)
  detail::AmgReuse s0Reuse_{detail::kReuseSetupS0}, hReuse_{detail::kReuseSetupH};
  long amgClock_ = 0;
  double evalSplit_[4] = {0, 0, 0, 0};  ///< evalState: facets, energy, H AMG setup, adjoint (s)
  double gaugeU_ = 0, gaugeK_ = 0;
  // the drift gauge (GaugeMode::kDrift): δⁿ (gd_) is pending for the next drift; whether this
  // step's drift used it; Σ_i ∇_iŨⁿ·δⁿ_i (the monitor's first half); f_G
  bool deltaPending_ = false, deltaUsed_ = false;
  double gaugeGradDelta0_ = 0, gaugeFactor_ = 1;
  double gaugeEbar_ = 0;  ///< ε̄ of the last G‴ activation (W's β)
  // persistent scratch (M7 WO-PERF3 P3.1, DESIGN_NOTE §14.4.6 item 1): the facet exports (a pool:
  // an export takes a buffer no live CellFacets references — F_, fc_, the option-S copy of x^n, the
  // saddle's copy, a gauge trial), the weight solve / H solves / adjoint, the band publish
  FacetPool<Real> pool_;
  WeightSolveScratch wss_;
  PotentialScratch<Real> pots_;
  mutable std::vector<double> tenH_, hH_;  ///< capillaryStep's host copies
  CellFacets<Real> F_;
  CellFacets<Real> fc_;   ///< repairMismatch()'s export, reused at the same state (currentFacets)
  bool fcValid_ = false;  ///< fc_ describes the current (pos, weight, engine)
  /// zn_ holds ζ vel_ at F_ (kick 2's, for the heat): the next 'pressure' kick 1 reuses it as its
  /// ζⁿvⁿ (D14.5). Cleared by evalState (a new geometry), the gauge projection (a new v) and
  /// every attempt (which overwrites zn_).
  bool zetaValid_ = false;
  PotentialParts parts_;
  FacetMatch match_;
  AdjointResult adj_;
  peclet::core::solver::GraphAMGDevice hAmg_;
  SaddleSolver<Real> saddle_;
  double time_ = 0, kinetic_ = 0, heat_ = 0, qc_ = 0, work_ = 0, wrapOffset_ = 0;
};

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_INTEGRATOR_HPP
