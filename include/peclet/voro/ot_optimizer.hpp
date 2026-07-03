/**
 * @file ot_optimizer.hpp
 * \brief Semi-discrete optimal-transport VOLUME control on the power tessellation — the first
 * energy-minimisation optimiser for building unstructured grids by moving cells to target volumes.
 *
 * DOFs = the power weights w (seed positions fixed). For a target per-cell volume V_set_i we solve
 * for w such that V_i(w) = V_set_i. This is the minimiser of the convex semi-discrete OT energy
 * whose gradient is (V_i − V_set_i); its Newton system is
 *
 *     L(w) δw = (V_set − V(w)),      w ← w + α δw,
 *
 * where L is the graph Laplacian of the cell-adjacency graph with edge weights A_ij/(2 d_ij)
 * (A_ij = shared-face area, d_ij = seed distance) — exactly dV/dw, and exactly the pressure-Poisson
 * operator from docs/power_cell_solver_spec.md §4.2. L is assembled from the tessellation's facet
 * CSR (facetArea, facetConnVec) and solved with the suite's mesh-agnostic sparse-operator + Krylov
 * tooling (peclet::core::amr::MomentumOp / MomentumSolver, Jacobi-preconditioned BiCGStab).
 *
 * V_set may depend on the SDF (smaller targets near a solid ⇒ grid refinement at boundaries).
 *
 * The optimiser is host-orchestrated (rebuild the power tessellation each Newton step, assemble +
 * solve); the solve itself runs on the Kokkos device. Depends on Effort 1 (power cells).
 *
 * LIMITATION (deferred): the PERIODIC min-image power diagram is not an exact partition (Effort 1:
 * volumes off the box by ~1% at non-trivial weights), so the OT system V=V_set is inconsistent below
 * that floor and the Newton residual plateaus at ~1% of the box volume. The Hessian is FD-validated
 * correct and the solve/line-search are sound; a clean converging OT needs an exact-partition power
 * tessellation (a non-periodic / walled domain, or the multi-image gather). For pure-Voronoi
 * position-based volume control that has no such floor, use mesh_optimizer.hpp.
 */
#ifndef PECLET_VORO_OT_OPTIMIZER_HPP
#define PECLET_VORO_OT_OPTIMIZER_HPP

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/core/amr/momentum.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tessellator.hpp"

namespace peclet::voro {

struct OtResult {
  int iters = 0;
  double maxVolErr = 0.0;   ///< max_i |V_i − V_set_i|
  double meanVolErr = 0.0;  ///< mean |V_i − V_set_i|
  bool converged = false;
  long nEmpty = 0;  ///< buried/empty cells at the final state (should be 0 for a feasible target)
};

namespace detail {
template <class Real>
std::vector<Real> toHostVec(const Kokkos::View<Real*, peclet::core::MemSpace>& v) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<Real>(m.data(), m.data() + m.extent(0));
}
template <class T>
std::vector<T> toHostVecT(const Kokkos::View<T*, peclet::core::MemSpace>& v) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<T>(m.data(), m.data() + m.extent(0));
}
}  // namespace detail

/**
 * Drive the power weights `weight` (size N, updated in place) so the power-cell volumes match
 * `vsetIn` (size N). `pos` (size 3N) are the fixed seeds. Returns convergence diagnostics.
 *
 * @param sw          grid coverage half-width for the tessellation (default 4).
 * @param maxNewton   max Newton iterations.
 * @param tol         convergence on max_i |V_i − V_set_i| (absolute volume).
 * @param reg         tiny Laplacian regularisation ε (lifts the constant nullspace so L+εI is SPD).
 * @param damp        Newton step scale α (1 = full Newton; <1 damps).
 * @param verbose     print per-iteration residual.
 *
 * `vsetIn` is normalised internally so Σ V_set = Σ V (the total tessellated volume) — the constant
 * nullspace of L requires a compatible right-hand side (Σ residual = 0).
 */
template <class Real, class Sdf = NoSdf>
OtResult otVolumeControl(std::vector<Real>& pos, std::vector<Real>& weight,
                         const std::vector<Real>& vsetIn, const Real L[3], int N, int sw,
                         const Sdf& sdf, int maxNewton, Real tol, Real reg = Real(1e-6),
                         Real damp = Real(1), bool verbose = false) {
  using peclet::core::Index;
  using peclet::core::MemSpace;
  using MView = peclet::core::View<double>;
  const Real Larr[3] = {L[0], L[1], L[2]};

  Kokkos::View<Real*, MemSpace> dpos("ot.pos", 3 * N), dw("ot.w", N);
  Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(pos.data(), 3 * N));
  Kokkos::View<long*, MemSpace> gd;

  peclet::core::amr::MomentumSolver<21> solver;
  solver.setJacobi(2, 0.7);  // damped-Jacobi preconditioner (no AMG on the irregular graph yet)

  double sumVset = 0;
  for (int i = 0; i < N; ++i) sumVset += vsetIn[i];

  // Build the power tessellation at weights `wv`; return its result and the volume error statistics
  // (targets renormalised so Σ V_set = Σ V — L's constant nullspace requires Σ residual = 0).
  auto evaluate = [&](const std::vector<Real>& wv, std::vector<double>& vol, double& maxErr,
                      double& meanErr, double& resL2, long& nEmpty, double& scale) {
    Kokkos::deep_copy(dw, Kokkos::View<const Real*, Kokkos::HostSpace>(wv.data(), N));
    auto res = buildTessellation<Real, true, Sdf>(dpos, dw, N, Larr, sw, N, gd, sdf,
                                                  /*withForceGeom=*/true);
    vol = detail::toHostVec<Real>(res.view.cellVolume);
    double sumV = 0;
    nEmpty = 0;
    for (int i = 0; i < N; ++i) {
      sumV += vol[i];
      if (vol[i] <= 0.0) ++nEmpty;
    }
    scale = (sumVset > 0) ? sumV / sumVset : 1.0;
    maxErr = 0;
    meanErr = 0;
    double s2 = 0;
    for (int i = 0; i < N; ++i) {
      const double e = std::fabs(vol[i] - vsetIn[i] * scale);
      maxErr = std::max(maxErr, e);
      meanErr += e;
      s2 += e * e;
    }
    meanErr /= N;
    resL2 = std::sqrt(s2);  // OT gradient norm — the damped-Newton merit (KMT)
    return res;
  };

  OtResult R;
  std::vector<double> vol;
  double maxErr, meanErr, resL2, scale;
  long nEmpty;
  auto res = evaluate(weight, vol, maxErr, meanErr, resL2, nEmpty, scale);

  for (int it = 0; it < maxNewton; ++it) {
    R.iters = it;
    R.maxVolErr = maxErr;
    R.meanVolErr = meanErr;
    R.nEmpty = nEmpty;
    if (verbose)
      std::printf("  [ot] iter %2d  maxVolErr=%.3e meanVolErr=%.3e nEmpty=%ld\n", it, maxErr, meanErr,
                  nEmpty);
    if (maxErr < tol) {
      R.converged = true;
      break;
    }

    // assemble L (graph Laplacian, edge weight A_ij/(2 d_ij)) from the current facet CSR.
    auto off = detail::toHostVecT<int>(res.view.cellFacetOffset);
    auto cnt = detail::toHostVecT<int>(res.view.cellFacetCount);
    auto nbr = detail::toHostVecT<peclet::voro::gid_t>(res.view.facetNeighbor);
    auto area = detail::toHostVec<Real>(res.view.facetArea);
    auto conn = detail::toHostVec<Real>(res.view.facetConnVec);
    const int nFacets = (int)nbr.size();
    std::vector<double> diag(N, 0.0), coef, r(N);
    std::vector<Index> start(N + 1, 0), fnbr;
    coef.reserve(nFacets);
    fnbr.reserve(nFacets);
    double sumDiag = 0;
    for (int i = 0; i < N; ++i) {
      start[i] = (Index)fnbr.size();
      const int fend = off[i] + cnt[i];
      for (int f = off[i]; f < fend && f < nFacets; ++f) {
        const long j = (long)nbr[f];
        if (j < 0 || j >= N) continue;  // boundary / SDF wall facet: no DOF coupling
        const double Aij = std::sqrt(area[3 * f] * area[3 * f] + area[3 * f + 1] * area[3 * f + 1] +
                                     area[3 * f + 2] * area[3 * f + 2]);
        const double dij = std::sqrt(conn[3 * f] * conn[3 * f] + conn[3 * f + 1] * conn[3 * f + 1] +
                                     conn[3 * f + 2] * conn[3 * f + 2]);
        if (dij <= 0.0 || Aij <= 0.0) continue;
        const double w = Aij / (2.0 * dij);
        diag[i] += w;
        coef.push_back(-w);
        fnbr.push_back((Index)j);
      }
      sumDiag += diag[i];
    }
    start[N] = (Index)fnbr.size();

    if (verbose && it == 0) {  // FD-validate dV/dw against the assembled L (one cell/neighbour)
      const int c = N / 2;
      long jn = -1;
      double Acf = 0;
      for (int f = off[c]; f < off[c] + cnt[c] && f < nFacets; ++f) {
        const long j = (long)nbr[f];
        if (j < 0 || j >= N) continue;
        const double Aij = std::sqrt(area[3 * f] * area[3 * f] + area[3 * f + 1] * area[3 * f + 1] +
                                     area[3 * f + 2] * area[3 * f + 2]);
        const double dij = std::sqrt(conn[3 * f] * conn[3 * f] + conn[3 * f + 1] * conn[3 * f + 1] +
                                     conn[3 * f + 2] * conn[3 * f + 2]);
        jn = j;
        Acf = -Aij / (2 * dij);  // analytic dV_c/dw_j = L_cj
        break;
      }
      if (jn >= 0) {
        const double dd = 1e-6;
        std::vector<Real> wp = weight;
        wp[jn] += (Real)dd;
        std::vector<double> vp;
        double a, b, cq, sq;
        long ne;
        evaluate(wp, vp, a, b, cq, ne, sq);
        const double fd = (vp[c] - vol[c]) / dd;
        std::printf("      FD dV_%d/dw_%ld: analytic=%.4e fd=%.4e  (dV_c/dw_c analytic=%.4e)\n", c,
                    jn, Acf, fd, diag[c]);
      }
    }

    const double eps = reg * (sumDiag / N);  // tiny regularisation (numerical safety)
    for (int i = 0; i < N; ++i) {
      diag[i] += eps;
      r[i] = vsetIn[i] * scale - vol[i];  // Newton RHS: L δw = (V_set − V)
    }
    // Pin node 0 (Dirichlet gauge): the Laplacian is singular (constant-weight nullspace — a global
    // weight shift is a null move for the power diagram), so fix δw_0 = 0 by making row 0 the
    // identity. This makes L SPD/nonsingular for the Krylov solve, without changing the diagram.
    for (Index f = start[0]; f < start[1]; ++f) coef[f] = 0.0;
    diag[0] = 1.0;
    r[0] = 0.0;

    peclet::core::amr::MomentumOp op;
    op.n = (Index)N;
    op.diag = peclet::core::toDevice(diag, "ot.diag");
    op.faceStart = peclet::core::toDevice(start, "ot.start");
    op.faceNbr = peclet::core::toDevice(fnbr, "ot.nbr");
    op.faceCoef = peclet::core::toDevice(coef, "ot.coef");
    MView db("ot.rhs", N), ddw("ot.dw", N);
    Kokkos::deep_copy(db, Kokkos::View<const double*, Kokkos::HostSpace>(r.data(), N));
    Kokkos::deep_copy(ddw, 0.0);
    auto sr = solver.solveBiCGStab(op, ddw, MView(db), 2000, 1e-10);
    auto dwv = detail::toHostVec<double>(ddw);
    if (verbose) {
      double dwnorm = 0;
      for (int i = 0; i < N; ++i) dwnorm = std::max(dwnorm, std::fabs(dwv[i]));
      std::printf("      solve: %d iters res0=%.2e res=%.2e |dw|inf=%.3e\n", sr.iters, sr.res0,
                  sr.res, dwnorm);
    }

    // Exact line minimisation of the convex OT energy g(w) along the Newton direction: g is convex,
    // so g'(α) = (V(w+αδw) − V_set)·δw is increasing with g'(0) < 0. Bisect for the α where g' = 0
    // (the energy minimum along δw), capping α to keep every cell non-empty. Using g' (not ‖V−Vset‖)
    // is the fix for the stall — the gradient NORM need not decrease along a valid energy-descent
    // step, but the directional derivative crossing zero exactly locates the line minimum.
    std::vector<Real> wtry(N);
    std::vector<double> volT;
    double maxErrT, meanErrT, resL2T, scaleT;
    long nEmptyT;
    peclet::voro::TessellatorResult<Real> resT;
    auto probe = [&](double alpha, double& dd) -> bool {  // returns feasible; sets dd = g'(α)
      for (int i = 0; i < N; ++i) wtry[i] = weight[i] + (Real)(alpha * dwv[i]);
      resT = evaluate(wtry, volT, maxErrT, meanErrT, resL2T, nEmptyT, scaleT);
      dd = 0;
      for (int i = 0; i < N; ++i) dd += (volT[i] - vsetIn[i] * scaleT) * dwv[i];
      return nEmptyT == 0;
    };
    // cap α at the Newton point `damp` and shrink until feasible (no empty cells).
    double aHi = damp, ddHi = 0;
    bool feasHi = probe(aHi, ddHi);
    for (int g = 0; !feasHi && aHi > 1e-6 && g < 40; ++g) {
      aHi *= 0.5;
      feasHi = probe(aHi, ddHi);
    }
    if (!feasHi) break;  // cannot take any feasible step
    double aAcc = aHi;   // still descending at the cap ⇒ take the full feasible step
    if (ddHi >= 0.0) {   // g' changed sign inside [0, aHi] ⇒ bisect for the line minimum g' = 0
      double aLo = 0.0;
      for (int b = 0; b < 24; ++b) {
        const double aMid = 0.5 * (aLo + aHi);
        double ddMid;
        const bool feas = probe(aMid, ddMid);
        if (feas && ddMid < 0.0)
          aLo = aMid;
        else
          aHi = aMid;
      }
      aAcc = aLo;
    }
    if (aAcc <= 0.0) break;  // at the minimum (or stuck)
    double ddAcc;
    probe(aAcc, ddAcc);  // re-evaluate at the accepted α to capture the state
    weight = wtry;
    res = resT;
    vol = volT;
    maxErr = maxErrT;
    meanErr = meanErrT;
    resL2 = resL2T;
    nEmpty = nEmptyT;
    scale = scaleT;
    if (verbose) std::printf("      accepted alpha=%.4f resL2=%.4e\n", aAcc, resL2);
  }
  return R;
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_OT_OPTIMIZER_HPP
