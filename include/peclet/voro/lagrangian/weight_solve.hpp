/**
 * @file lagrangian/weight_solve.hpp
 * \brief The power weights for prescribed cell volumes, w*(x): V_i(x, w) = V⁰_i (semi-discrete
 * optimal transport), by the damped Newton method of Kitagawa–Mérigot–Thibert (2019) as M7 uses it
 * (DESIGN_NOTE §10.2):
 *
 *   H δw = V⁰ − V   (H = ∂V/∂w, the TPFA Laplacian A_ij/(2d_ij); one AMG-PCG solve),
 *   w ← w + α δw with α = 1, ½, ¼, … — the first α with min V > ½ min V⁰ and
 *   ‖V⁰ − V(w + αδw)‖ ≤ (1 − α/2) ‖V⁰ − V(w)‖,
 *   stop at max_i |V_i − V⁰_i| / V⁰_i ≤ tol.
 *
 * Each iterate is a cold rebuild of the resident power tessellation (MovingTessellation::rebuild
 * with the trial weights); the volumes are the engine's own, computed in each cell's generator
 * frame, so tol = 1e-12 is reachable (DESIGN_NOTE §10.2: in absolute coordinates it is not).
 *
 * H is singular (a constant weight shift changes nothing): δw_0 is pinned to zero (row and column
 * 0 replaced by the identity), which leaves the other equations exact because Σ_i (V⁰ − V)_i = 0
 * — the caller must give targets that sum to the tessellated volume; the round-off remainder of
 * that sum is removed from the right-hand side first (spread evenly instead of landing on cell 0).
 * The PCG runs on the device (matrix-free H, lagrangian/operators.hpp) preconditioned by core's
 * GraphAMGDevice, whose setup is host-side as everywhere in the suite (fv/covolume.hpp).
 *
 * This is NOT ot_optimizer.hpp's otVolumeControl: that routine takes the seed distance from the
 * published connector 2n (= 2αr, not r, on radical planes), regularises H, rescales the targets
 * and accepts by exact line minimisation; extending it would change its numerics. This routine is
 * the M7 specification.
 */
#ifndef PECLET_VORO_LAGRANGIAN_WEIGHT_SOLVE_HPP
#define PECLET_VORO_LAGRANGIAN_WEIGHT_SOLVE_HPP

#include <algorithm>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <string>
#include <utility>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/graph_amg_device.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/operators.hpp"
#include "peclet/voro/plane_policy.hpp"

namespace peclet::voro::lagrangian {

/// Host copies of the facet arrays a two-point operator reads (facetOffset, nbr, area, conn), kept
/// by a caller that assembles one every step (pinnedHostH, the integrator's twoPointHost).
struct HostFacets {
  std::vector<int> off, nbr;
  std::vector<double> area, conn;
};

/// Persistent scratch of the weight solve, the pinned H solves and the adjoint (M7 WO-PERF3 P3.1,
/// DESIGN_NOTE §14.4.6 item 1: no View allocation per steady step), held by a caller that solves
/// every step (the NHV-W integrator) and passed by WeightSolveParams::scratch. Device vectors are
/// exact-length leading subviews, zeroed on every use like the one-shot allocations they replace.
struct WeightSolveScratch {
  LView<double> r, z, p, q, tmp;   ///< solvePinnedH's PCG vectors
  LView<double> rhs, dw, wt;       ///< solveWeights' Newton vectors
  LView<double> adjRhs;            ///< reducedGradient's right-hand side
  std::vector<double> V, Vt, res;  ///< host volumes (current, trial) and the Newton rhs
  HostFacets host;                 ///< pinnedHostH's host copies of the facets
  FacetPool<double> facets;        ///< the Newton iterations' exports
};

struct WeightSolveParams {
  double tol = 1e-12;     ///< stop at max_i |V_i − V⁰_i| / V⁰_i ≤ tol
  int maxIter = 50;       ///< Newton iterations
  double pcgTol = 1e-10;  ///< relative residual of the inner H solve
  int pcgMaxIter = 1000;
  double minAlpha = 1e-6;  ///< give up halving below this step
  /// Evaluate each iterate by the incremental repair of the resident engine
  /// (MovingTessellation::step at the trial weights) instead of a cold rebuild — the per-step path
  /// of the integrator (lagrangian/integrator.hpp). A cell is then rejected as buried when its
  /// volume is not positive (the repair keeps no build report).
  bool repair = false;
  /// A prebuilt AMG hierarchy to precondition the H solves with (e.g. built on a nearby
  /// geometry); null: build one on the current H per solve.
  const peclet::core::solver::GraphAMGDevice* amg = nullptr;
  /// Persistent scratch (null: allocate per call, as before).
  WeightSolveScratch* scratch = nullptr;
};

struct WeightSolveStep {
  double maxRelErr = 0;  ///< at the START of the iteration
  double alpha = 0;      ///< accepted step
  int pcgIters = 0;
  double pcgRelRes = 0;
  int halvings = 0;
};

struct WeightSolveResult {
  int iters = 0;
  double maxRelErr = 0;
  bool converged = false;
  bool roundoffFloor = false;  ///< stalled with maxRelErr ≤ 1e3·tol (step at the round-off floor)
  int coldRebuilds = 0;        ///< repair mode: evaluations the repair routed to a cold rebuild
  int evaluations = 0;         ///< tessellation updates (repairs or rebuilds)
  double evalSeconds = 0;      ///< wall time of those updates
  std::vector<WeightSolveStep> history;
};

namespace detail {
/// Copy a contiguous rank-1 View into a host vector (resized to its extent) with no Kokkos
/// allocation: the values toVector returns, without its staging buffer.
template <class V, class T>
void hostCopy(const V& v, std::vector<T>& out) {
  out.resize(v.extent(0));
  if (out.empty())
    return;
  Kokkos::View<T*, typename V::array_layout, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h(out.data(), out.size());
  Kokkos::deep_copy(h, v);
}

/// `slot` of the scratch `s` (null: a one-shot zero-initialised View, as before) at length n.
inline LView<double> wsVector(WeightSolveScratch* s, LView<double> WeightSolveScratch::*slot, int n,
                              const char* label) {
  return scratchView(s ? &(s->*slot) : nullptr, static_cast<std::size_t>(n), label, true);
}

/// `F`'s facetOffset, nbr, area and conn into the host vectors of `h`.
template <class Real>
void copyHostFacets(const CellFacets<Real>& F, HostFacets& h) {
  hostCopy(F.facetOffset, h.off);
  hostCopy(F.nbr, h.nbr);
  hostCopy(F.area, h.area);
  hostCopy(F.conn, h.conn);
}

template <class Real>
Real dotView(const LView<Real>& a, const LView<Real>& b) {
  Real s = 0;
  Kokkos::parallel_reduce(
      "lagrangian.dot", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, a.extent(0)),
      KOKKOS_LAMBDA(const int i, Real& acc) { acc += a(i) * b(i); }, s);
  return s;
}

/// H with δw_0 pinned: y = P H P x + e_0 x_0.
template <class Real>
void applyPinnedH(const CellFacets<Real>& F, const LView<Real>& x, const LView<Real>& tmp,
                  const LView<Real>& y) {
  Kokkos::deep_copy(tmp, x);
  Kokkos::deep_copy(Kokkos::subview(tmp, 0), Real(0));
  applyWeightHessian(F, tmp, y);
  Kokkos::deep_copy(Kokkos::subview(y, 0), Kokkos::subview(x, 0));
}

/// The pinned H as a host CSR (off-diagonal duplicates of multi-image facets merged). `scr`
/// (optional): the host copies of the facets are its persistent vectors.
template <class Real>
peclet::core::solver::HostCsrOp pinnedHostH(const CellFacets<Real>& F,
                                            WeightSolveScratch* scr = nullptr) {
  static_assert(std::is_same_v<Real, double>, "the weight solve runs in double (GraphAMG)");
  HostFacets own;
  HostFacets& h = scr ? scr->host : own;
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
    for (int f = off[i]; f < off[i + 1]; ++f) {
      const double d = std::sqrt(conn[3 * f] * conn[3 * f] + conn[3 * f + 1] * conn[3 * f + 1] +
                                 conn[3 * f + 2] * conn[3 * f + 2]);
      const double w = area[f] / (2.0 * d);
      K.diag[i] += w;
      if (i != 0 && nbr[f] != 0)
        row.emplace_back(nbr[f], -w);
    }
    std::sort(row.begin(), row.end());
    for (std::size_t q = 0; q < row.size(); ++q) {
      if (!K.nbr.empty() && q > 0 && row[q].first == row[q - 1].first)
        K.coef.back() += row[q].second;
      else {
        K.nbr.push_back(row[q].first);
        K.coef.push_back(row[q].second);
      }
    }
    K.start[i + 1] = static_cast<peclet::core::Index>(K.nbr.size());
  }
  K.diag[0] = 1.0;
  return K;
}
}  // namespace detail

/// Solve the pinned H δw = r (r_0 must be 0) by AMG-PCG on the device. Returns {iters, rel res}.
template <class Real>
std::pair<int, double> solvePinnedH(const CellFacets<Real>& F, const LView<Real>& r,
                                    const LView<Real>& x, const WeightSolveParams& prm) {
  static_assert(std::is_same_v<Real, double>, "the weight solve runs in double (GraphAMG)");
  const int N = F.nCells;
  peclet::core::solver::GraphAMGDevice own;
  if (!prm.amg) {
    peclet::core::solver::AmgParams ap;
    ap.ndofPerNode = 1;
    own.build(detail::pinnedHostH(F, prm.scratch), ap);
  }
  const peclet::core::solver::GraphAMGDevice& amg = prm.amg ? *prm.amg : own;
  WeightSolveScratch* S = prm.scratch;
  const LView<Real> res = detail::wsVector(S, &WeightSolveScratch::r, N, "ws.r"),
                    z = detail::wsVector(S, &WeightSolveScratch::z, N, "ws.z"),
                    p = detail::wsVector(S, &WeightSolveScratch::p, N, "ws.p"),
                    q = detail::wsVector(S, &WeightSolveScratch::q, N, "ws.q"),
                    tmp = detail::wsVector(S, &WeightSolveScratch::tmp, N, "ws.tmp");
  Kokkos::deep_copy(x, Real(0));
  Kokkos::deep_copy(res, r);
  const Real r0 = std::sqrt(detail::dotView(res, res));
  if (!(r0 > Real(0)))
    return {0, 0.0};
  amg.apply(res, z);
  Kokkos::deep_copy(p, z);
  Real rz = detail::dotView(res, z), rn = r0;
  int it = 0;
  using Exec = peclet::core::ExecSpace;
  for (; it < prm.pcgMaxIter; ++it) {
    detail::applyPinnedH(F, p, tmp, q);
    const Real pq = detail::dotView(p, q);
    if (!(pq > Real(0)))
      break;
    const Real al = rz / pq;
    Kokkos::parallel_for(
        "ws.update", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          x(i) += al * p(i);
          res(i) -= al * q(i);
        });
    rn = std::sqrt(detail::dotView(res, res));
    if (rn <= prm.pcgTol * r0) {
      ++it;
      break;
    }
    amg.apply(res, z);
    const Real rzNew = detail::dotView(res, z);
    const Real be = rzNew / rz;
    rz = rzNew;
    Kokkos::parallel_for(
        "ws.dir", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i) { p(i) = z(i) + be * p(i); });
  }
  return {it, rn / r0};
}

/// Damped Newton for V(x, w) = V⁰ on the resident power tessellation `mt` (a MovingTessellation
/// with the Power policy). `weight` (N) is the initial guess and returns the solution (gauge:
/// mean zero); `mt` is left built at it. `V0` (N, host) must sum to the tessellated volume.
template <class MT, class Real>
WeightSolveResult solveWeights(MT& mt, const LView<Real>& pos, const LView<Real>& weight,
                               const std::vector<double>& V0, const Real L[3],
                               const WeightSolveParams& prm = {}) {
  static_assert(std::is_same_v<typename MT::PlanePolicy, Power>,
                "solveWeights needs a power tessellation");
  using Exec = peclet::core::ExecSpace;
  const int N = static_cast<int>(V0.size());
  double minV0 = V0[0];
  for (double v : V0)
    minV0 = std::min(minV0, v);
  // gauge: mean-zero weights
  {
    Real s = 0;
    Kokkos::parallel_reduce(
        "ws.wmean", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc += weight(i); }, s);
    const Real m = s / N;
    Kokkos::parallel_for(
        "ws.gauge", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i) { weight(i) -= m; });
  }
  WeightSolveResult R;
  // rebuild (or repair) at weights `w`; return the host volumes and whether a cell was buried
  auto evaluate = [&](const LView<Real>& w, std::vector<double>& V) {
    mt.setWeights(w);
    ++R.evaluations;
    Kokkos::Timer tm;
    if (prm.repair) {
      const auto st = mt.step(pos);
      Kokkos::fence();
      R.evalSeconds += tm.seconds();
      R.coldRebuilds += (st.fellBack || st.route == decltype(st)::kRebuildGate) ? 1 : 0;
      detail::hostCopy(Kokkos::subview(mt.vol, std::make_pair(0, N)), V);
      return *std::min_element(V.begin(), V.end()) > 0.0;
    }
    mt.rebuild(pos);
    Kokkos::fence();
    R.evalSeconds += tm.seconds();
    detail::hostCopy(Kokkos::subview(mt.vol, std::make_pair(0, N)), V);
    return mt.report().buried == 0;
  };
  auto stats = [&](const std::vector<double>& V, double& maxRel, double& l2, double& minV) {
    maxRel = 0;
    l2 = 0;
    minV = V[0];
    for (int i = 0; i < N; ++i) {
      const double r = V0[i] - V[i];
      maxRel = std::max(maxRel, std::fabs(r) / V0[i]);
      l2 += r * r;
      minV = std::min(minV, V[i]);
    }
    l2 = std::sqrt(l2);
  };

  WeightSolveScratch own;
  WeightSolveScratch& S = prm.scratch ? *prm.scratch : own;
  std::vector<double>&V = S.V, &Vt = S.Vt;
  evaluate(weight, V);
  double maxRel, nr, minV;
  stats(V, maxRel, nr, minV);
  const LView<Real> rhs = detail::wsVector(prm.scratch, &WeightSolveScratch::rhs, N, "ws.rhs"),
                    dw = detail::wsVector(prm.scratch, &WeightSolveScratch::dw, N, "ws.dw"),
                    wt = detail::wsVector(prm.scratch, &WeightSolveScratch::wt, N, "ws.wt");
  for (int it = 0; it < prm.maxIter; ++it) {
    R.iters = it;
    R.maxRelErr = maxRel;
    if (maxRel <= prm.tol) {
      R.converged = true;
      return R;
    }
    WeightSolveStep st;
    st.maxRelErr = maxRel;
    // H only (DESIGN_NOTE §14.4.6 item 2): the light export, bit-identical in what H reads
    const CellFacets<Real> F =
        exportFacetsLight<Real, Power>(mt.store, pos, weight, N, L, mt.wall, mt.xRef, mt.vol,
                                       prm.scratch ? &prm.scratch->facets : nullptr);
    {
      // the mean of V0 − V is round-off (Σ V0 = Σ V); remove it before pinning, or row 0 would
      // absorb the whole imbalance (a ~1e-12 relative floor on cell 0 at N = 4096)
      std::vector<double>& r = S.res;
      r.resize(N);
      double mean = 0;
      for (int i = 0; i < N; ++i) {
        r[i] = V0[i] - V[i];
        mean += r[i];
      }
      mean /= N;
      for (int i = 0; i < N; ++i)
        r[i] -= mean;
      r[0] = 0.0;  // pinned
      Kokkos::deep_copy(rhs, Kokkos::View<const double*, Kokkos::HostSpace>(r.data(), N));
    }
    const auto [pit, pres] = solvePinnedH(F, rhs, dw, prm);
    st.pcgIters = pit;
    st.pcgRelRes = pres;
    double al = 1.0, maxRelT = 0, nrT = 0, minVT = 0;
    for (;;) {
      const Real a = static_cast<Real>(al);
      Kokkos::parallel_for(
          "ws.trial", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i) { wt(i) = weight(i) + a * dw(i); });
      const bool ok = evaluate(wt, Vt);
      stats(Vt, maxRelT, nrT, minVT);
      if (ok && minVT > 0.5 * minV0 && nrT <= (1.0 - 0.5 * al) * nr)
        break;
      al *= 0.5;
      ++st.halvings;
      if (al < prm.minAlpha) {
        evaluate(weight, V);  // leave mt at the last accepted iterate
        R.history.push_back(st);
        R.roundoffFloor = maxRel <= 1e3 * prm.tol;
        return R;
      }
    }
    st.alpha = al;
    R.history.push_back(st);
    Kokkos::deep_copy(weight, wt);
    mt.setWeights(weight);  // wt is scratch: the resident engine must hold `weight`
    V.swap(Vt);
    maxRel = maxRelT;
    nr = nrT;
    minV = minVT;
  }
  R.iters = prm.maxIter;
  R.maxRelErr = maxRel;
  R.converged = maxRel <= prm.tol;
  return R;
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_WEIGHT_SOLVE_HPP
