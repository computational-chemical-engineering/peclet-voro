/**
 * @file mesh_optimizer.hpp
 * \brief Energy-minimisation mesh optimiser over seed positions (pure Voronoi) and, optionally, the
 * power WEIGHTS (Laguerre) — moves the DOFs so the cells minimise a geometry energy. A
 * Surface-Evolver-style tool on the differentiable (power-)Voronoi geometry.
 *
 * Energy (this file): E = Σ_i γ (V_i − V_set,i)², driving cells to target volumes (V_set from an SDF
 * ⇒ refinement near solids). DOFs = seed positions x (3N) and, when Weighted, the power weights w
 * (N more). Pure Voronoi (Weighted=false) partitions space exactly ⇒ Σ V = box, so the energy is
 * well-posed with no floor; adding weights gives FULLER volume control (positions alone can only
 * partially reach a target).
 *
 * Gradient (published facet CSR — no cell rebuilds):
 *   ∂V_c/∂x_j = facetConnect_k ,  ∂V_c/∂x_c = −Σ_k facetConnect_k   (r_k = x_j − x_i),
 *   ∂V_c/∂w_c = Σ_k A_k/(2 d_k) ,  ∂V_c/∂w_j = −A_k/(2 d_k).
 * Newton–Raphson with the Gauss-Newton Hessian H = 2γ Σ_c (∇V_c)(∇V_c)ᵀ, ASSEMBLED as a scalar CSR
 * over the flattened DOFs, solved by CG with a Jacobi OR multicolour Gauss–Seidel preconditioner
 * (peclet::core::amr::greedyColoring); an Armijo line search on E completes the step.
 */
#ifndef PECLET_VORO_MESH_OPTIMIZER_HPP
#define PECLET_VORO_MESH_OPTIMIZER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "peclet/core/amr/momentum.hpp"     // greedyColoring
#include "peclet/core/solver/graph_amg.hpp"  // smoothed-aggregation AMG (O(N) CG preconditioner)
#include "peclet/voro/ot_optimizer.hpp"      // OtResult + detail::toHostVec/toHostVecT
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tessellator.hpp"

namespace peclet::voro {

// CG preconditioner for the Gauss-Newton Hessian solve. Jacobi = O(N^4/3) (iterations grow with N);
// ColoredGS = a stronger single-level smoother; GraphAMG = smoothed-aggregation multigrid, the only
// one whose iteration count stays mesh-independent ⇒ an O(N) solve at large N (see
// peclet::core::solver::GraphAMG).
enum class Precond { Jacobi, ColoredGS, GraphAMG };

/**
 * Minimise E = Σ γ (V_i − V_set,i)² by damped Gauss-Newton over positions (and weights when
 * Weighted). `pos` (3N) and `weight` (N; used only when Weighted) are updated in place.
 *
 * @param prec     Jacobi or ColoredGS CG preconditioner.
 * @param cgIters  inner CG iterations per Newton step; @param tol on the gradient ∞-norm.
 */
template <class Real, bool Weighted = false, class Sdf = NoSdf>
OtResult meshVolumeOptimize(std::vector<Real>& pos, std::vector<Real>& weight,
                            const std::vector<Real>& vsetIn, const Real L[3], int N, int sw,
                            const Sdf& sdf, int maxNewton, Real tol, int cgIters = 300,
                            Precond prec = Precond::Jacobi, bool verbose = false) {
  using peclet::core::Index;
  using MemSpace = peclet::core::MemSpace;
  const Real Larr[3] = {L[0], L[1], L[2]};
  const double gamma = 1.0;
  const int nD = Weighted ? 4 * N : 3 * N;  // DOFs: [0,3N) positions 3i+d ; [3N,4N) weights 3N+i
  auto wdof = [N](int i) { return 3 * N + i; };

  Kokkos::View<long*, MemSpace> gd;
  const double boxVol = (double)L[0] * L[1] * L[2];
  double sumVset = 0;
  for (int i = 0; i < N; ++i) sumVset += vsetIn[i];
  std::vector<double> vset(N);
  for (int i = 0; i < N; ++i) vset[i] = vsetIn[i] * (boxVol / sumVset);

  struct Geo {
    std::vector<double> vol, dvr, area, conn;
    std::vector<int> off, cnt;
    std::vector<peclet::voro::gid_t> nbr;
    double E = 0, maxErr = 0, meanErr = 0;
    long nBad = 0;
  };
  auto build = [&](const std::vector<Real>& x, const std::vector<Real>& w, Geo& G) {
    Kokkos::View<Real*, MemSpace> dpos("mo.pos", 3 * N), dw;
    Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(x.data(), 3 * N));
    if constexpr (Weighted) {
      dw = Kokkos::View<Real*, MemSpace>("mo.w", N);
      Kokkos::deep_copy(dw, Kokkos::View<const Real*, Kokkos::HostSpace>(w.data(), N));
    }
    auto res = buildTessellation<Real, Weighted, Sdf>(dpos, dw, N, Larr, sw, N, gd, sdf, true);
    G.vol = detail::toHostVec<Real>(res.view.cellVolume);
    G.dvr = detail::toHostVec<Real>(res.view.facetConnect);
    G.area = detail::toHostVec<Real>(res.view.facetArea);
    G.conn = detail::toHostVec<Real>(res.view.facetConnVec);
    G.off = detail::toHostVecT<int>(res.view.cellFacetOffset);
    G.cnt = detail::toHostVecT<int>(res.view.cellFacetCount);
    G.nbr = detail::toHostVecT<peclet::voro::gid_t>(res.view.facetNeighbor);
    G.E = G.maxErr = G.meanErr = 0;
    G.nBad = 0;
    for (int i = 0; i < N; ++i) {
      const double e = G.vol[i] - vset[i];
      G.E += gamma * e * e;
      G.maxErr = std::max(G.maxErr, std::fabs(e));
      G.meanErr += std::fabs(e);
      if (G.vol[i] <= 0.0) ++G.nBad;
    }
    G.meanErr /= N;
  };

  // Per-cell sparse gradient ∇V_c over the DOFs (dof index, value). Positions from facetConnect,
  // weights from A/(2d). `Rc` scales it into the objective gradient / Hessian rank-1 update.
  auto stencil = [&](const Geo& G, int c, std::vector<std::pair<int, double>>& st) {
    st.clear();
    const int nF = (int)G.nbr.size();
    double sx = 0, sy = 0, sz = 0, sw2 = 0;
    for (int f = G.off[c]; f < G.off[c] + G.cnt[c] && f < nF; ++f) {
      const long j = (long)G.nbr[f];
      if (j < 0 || j >= N) continue;
      const double dx = G.dvr[3 * f], dy = G.dvr[3 * f + 1], dz = G.dvr[3 * f + 2];
      st.emplace_back(3 * (int)j, dx);
      st.emplace_back(3 * (int)j + 1, dy);
      st.emplace_back(3 * (int)j + 2, dz);
      sx -= dx;
      sy -= dy;
      sz -= dz;
      if constexpr (Weighted) {
        const double A = std::sqrt(G.area[3 * f] * G.area[3 * f] +
                                   G.area[3 * f + 1] * G.area[3 * f + 1] +
                                   G.area[3 * f + 2] * G.area[3 * f + 2]);
        const double d = std::sqrt(G.conn[3 * f] * G.conn[3 * f] +
                                   G.conn[3 * f + 1] * G.conn[3 * f + 1] +
                                   G.conn[3 * f + 2] * G.conn[3 * f + 2]);
        const double wgt = (d > 0) ? A / (2 * d) : 0;
        st.emplace_back(wdof((int)j), -wgt);
        sw2 += wgt;
      }
    }
    st.emplace_back(3 * c, sx);
    st.emplace_back(3 * c + 1, sy);
    st.emplace_back(3 * c + 2, sz);
    if constexpr (Weighted) st.emplace_back(wdof(c), sw2);
  };

  Geo G;
  build(pos, weight, G);
  auto pack = [&](const std::vector<Real>& x, const std::vector<Real>& w, std::vector<Real>& q) {
    q.assign(x.begin(), x.end());
    if constexpr (Weighted) q.insert(q.end(), w.begin(), w.end());
  };
  auto unpack = [&](const std::vector<Real>& q, std::vector<Real>& x, std::vector<Real>& w) {
    x.assign(q.begin(), q.begin() + 3 * N);
    if constexpr (Weighted) w.assign(q.begin() + 3 * N, q.end());
  };

  OtResult R;
  std::vector<double> g(nD), dq(nD);
  std::vector<std::pair<int, double>> st;
  for (int it = 0; it < maxNewton; ++it) {
    // gradient + assembled Gauss-Newton Hessian (rows as maps → CSR).
    std::fill(g.begin(), g.end(), 0.0);
    std::vector<std::unordered_map<int, double>> row(nD);
    for (int c = 0; c < N; ++c) {
      const double Rc = 2.0 * gamma * (G.vol[c] - vset[c]);
      stencil(G, c, st);
      for (auto& [a, va] : st) {
        g[a] += Rc * va;
        auto& ra = row[a];
        for (auto& [b, vb] : st) ra[b] += 2.0 * gamma * va * vb;
      }
    }
    double gnorm = 0;
    for (int i = 0; i < nD; ++i) gnorm = std::max(gnorm, std::fabs(g[i]));

    if (verbose && it == 0) {  // FD-validate the objective gradient (one DOF)
      const int d = 3 * (N / 2);
      const double h = 1e-7;
      std::vector<Real> xp = pos, wp = weight;
      Geo Gp, Gm;
      xp[d] += (Real)h;
      build(xp, wp, Gp);
      xp[d] -= (Real)(2 * h);
      build(xp, wp, Gm);
      std::printf("      FD dE/dx_%d: analytic=%.4e fd=%.4e\n", d, g[d], (Gp.E - Gm.E) / (2 * h));
      if constexpr (Weighted) {  // also validate a weight-DOF gradient
        const int dw = wdof(N / 3);
        std::vector<Real> wq = weight;
        wq[N / 3] += (Real)h;
        Geo Gwp, Gwm;
        build(pos, wq, Gwp);
        wq[N / 3] -= (Real)(2 * h);
        build(pos, wq, Gwm);
        std::printf("      FD dE/dw_%d: analytic=%.4e fd=%.4e\n", N / 3, g[dw],
                    (Gwp.E - Gwm.E) / (2 * h));
      }
    }

    R.iters = it;
    R.maxVolErr = G.maxErr;
    R.meanVolErr = G.meanErr;
    R.nEmpty = G.nBad;
    if (verbose)
      std::printf("  [vmesh] iter %2d  E=%.4e maxVolErr=%.3e gnorm=%.3e nBad=%ld\n", it, G.E,
                  G.maxErr, gnorm, G.nBad);
    if (gnorm < tol) {
      R.converged = true;
      break;
    }

    // flatten to CSR (Hcol/Hval incl. diagonal; Hdiag separate; off-diagonal CSR for colouring).
    std::vector<Index> Hstart(nD + 1, 0), Hcol, ostart(nD + 1, 0), onbr;
    std::vector<double> Hval, Hdiag(nD, 0.0);
    for (int i = 0; i < nD; ++i) {
      Hstart[i] = (Index)Hcol.size();
      ostart[i] = (Index)onbr.size();
      for (auto& [j, v] : row[i]) {
        Hcol.push_back(j);
        Hval.push_back(v);
        if (j == i)
          Hdiag[i] = v;
        else
          onbr.push_back(j);
      }
    }
    Hstart[nD] = (Index)Hcol.size();
    ostart[nD] = (Index)onbr.size();

    auto matvec = [&](const std::vector<double>& v, std::vector<double>& out) {
      for (int i = 0; i < nD; ++i) {
        double s = 0;
        for (Index k = Hstart[i]; k < Hstart[i + 1]; ++k) s += Hval[k] * v[Hcol[k]];
        out[i] = s;
      }
    };

    // preconditioner z ≈ H⁻¹ r.
    peclet::core::amr::Coloring col;
    std::vector<Index> colIdxHost;
    if (prec == Precond::ColoredGS) {
      col = peclet::core::amr::greedyColoring(ostart, onbr, (Index)nD);
      colIdxHost = detail::toHostVecT<Index>(col.idx);
    }
    // Smoothed-aggregation AMG: rebuilt each Newton step (H moves with the geometry) from the same
    // assembled Hessian CSR. Nodal aggregation over the 3-DOF-per-seed position blocks (the weighted
    // path segregates the N weight DOFs after the 3N positions, so it falls back to scalar
    // aggregation, s=1 — correct, just not block-aware; a block-aware weighted variant is a later
    // refinement). One V-cycle per CG iteration keeps the iteration count flat as N grows.
    peclet::core::solver::GraphAMG amg;
    if (prec == Precond::GraphAMG) {
      peclet::core::solver::HostCsrOp Aop;
      Aop.n = nD;
      Aop.diag = Hdiag;
      Aop.start.assign((std::size_t)nD + 1, 0);
      for (int i = 0; i < nD; ++i) {
        for (Index k = Hstart[i]; k < Hstart[i + 1]; ++k)
          if (Hcol[k] != i) {
            Aop.nbr.push_back(Hcol[k]);
            Aop.coef.push_back(Hval[k]);
          }
        Aop.start[(std::size_t)i + 1] = (Index)Aop.nbr.size();
      }
      peclet::core::solver::AmgParams ap;
      ap.ndofPerNode = Weighted ? 1 : 3;
      amg.build(Aop, ap);
    }
    auto precond = [&](const std::vector<double>& r, std::vector<double>& z) {
      if (prec == Precond::Jacobi) {
        for (int i = 0; i < nD; ++i) z[i] = std::fabs(Hdiag[i]) > 1e-30 ? r[i] / Hdiag[i] : r[i];
        return;
      }
      if (prec == Precond::GraphAMG) {
        amg.apply(r, z);
        return;
      }
      // symmetric multicolour Gauss–Seidel (forward + backward sweep), z = 0 start.
      std::fill(z.begin(), z.end(), 0.0);
      auto sweep = [&](bool forward) {
        for (int ci = 0; ci < col.nColors; ++ci) {
          const int cc = forward ? ci : col.nColors - 1 - ci;
          for (Index t = col.hStart[cc]; t < col.hStart[cc + 1]; ++t) {
            const Index i = colIdxHost[t];
            double s = r[i];
            for (Index k = Hstart[i]; k < Hstart[i + 1]; ++k)
              if (Hcol[k] != i) s -= Hval[k] * z[Hcol[k]];
            if (std::fabs(Hdiag[i]) > 1e-30) z[i] = s / Hdiag[i];
          }
        }
      };
      sweep(true);
      sweep(false);
    };

    // Jacobi-/GS-preconditioned CG for H dq = −g.
    std::fill(dq.begin(), dq.end(), 0.0);
    std::vector<double> rr(nD), z(nD), p(nD), Ap(nD);
    for (int i = 0; i < nD; ++i) rr[i] = -g[i];
    precond(rr, z);
    p = z;
    double rz = 0;
    for (int i = 0; i < nD; ++i) rz += rr[i] * z[i];
    const double rz0 = rz;
    for (int k = 0; k < cgIters && rz > 1e-18 * rz0; ++k) {
      matvec(p, Ap);
      double pAp = 0;
      for (int i = 0; i < nD; ++i) pAp += p[i] * Ap[i];
      if (pAp <= 0) break;
      const double a = rz / pAp;
      for (int i = 0; i < nD; ++i) {
        dq[i] += a * p[i];
        rr[i] -= a * Ap[i];
      }
      precond(rr, z);
      double rzn = 0;
      for (int i = 0; i < nD; ++i) rzn += rr[i] * z[i];
      const double beta = rzn / rz;
      for (int i = 0; i < nD; ++i) p[i] = z[i] + beta * p[i];
      rz = rzn;
    }

    // Armijo backtracking on E along dq.
    double gdq = 0;
    for (int i = 0; i < nD; ++i) gdq += g[i] * dq[i];
    std::vector<Real> q, qtry, xt, wt;
    pack(pos, weight, q);
    double alpha = 1.0;
    bool accepted = false;
    Geo Gt;
    for (int bt = 0; bt < 24; ++bt) {
      qtry.resize(nD);
      for (int i = 0; i < nD; ++i) qtry[i] = q[i] + (Real)(alpha * dq[i]);
      unpack(qtry, xt, wt);
      build(xt, wt, Gt);
      if (Gt.nBad == 0 && Gt.E <= G.E + 1e-4 * alpha * gdq) {
        pos = xt;
        if constexpr (Weighted) weight = wt;
        G = Gt;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (verbose)
      std::printf("      prec=%s cg rz %.2e->%.2e  alpha=%.4f\n",
                  prec == Precond::Jacobi ? "jac" : "cgs", rz0, rz, alpha);
    if (!accepted) break;
  }
  return R;
}

// ================================================================================================
// Interfacial-energy optimiser (Surface-Evolver-style) — move seed positions to minimise the total
// area of the faces between cells of DIFFERENT type: E = Σ_{f: type_i≠type_j} σ A_f. A two-phase
// interface-tension minimiser (foam/emulsion coarsening).
//
// The face-area gradient ∂A_f/∂x is not published, so each cell is RECONSTRUCTED as a ConvexCell
// from its facet neighbours and the validated per-triangle area-Jacobian geomVolumeAreaGrad is
// gathered into ∂A_k/∂n_l, then routed to positions by chainToDofs<Voronoi>. Steepest descent with
// an Armijo line search on E. Host (the oracle per docs/DEVICE_RESIDENCY_PLAN.md); the device port
// mirrors the volume path (assemble on device, reuse the same reconstruction kernel per cell).
// ================================================================================================
template <class Real, class Sdf = NoSdf>
OtResult interfaceMinimize(std::vector<Real>& pos, const std::vector<int>& type, double sigma,
                           const Real L[3], int N, int sw, const Sdf& sdf, int maxIter, Real tol,
                           bool verbose = false) {
  using MemSpace = peclet::core::MemSpace;
  using RCell = ConvexCell<Real, 128, 256>;
  const Real Larr[3] = {L[0], L[1], L[2]}, Lh[3] = {L[0] / 2, L[1] / 2, L[2] / 2};
  Kokkos::View<Real*, MemSpace> dw;
  Kokkos::View<long*, MemSpace> gd;

  // Build the tessellation at `x`, download the neighbour CSR, then per cell reconstruct the
  // ConvexCell and gather the interfacial-area energy E and its position gradient g (3N).
  auto energyGrad = [&](const std::vector<Real>& x, std::vector<double>& g, double& E) {
    Kokkos::View<Real*, MemSpace> dpos("if.pos", 3 * N);
    Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(x.data(), 3 * N));
    auto res = buildTessellation<Real, false, Sdf>(dpos, dw, N, Larr, sw, N, gd, sdf, true);
    auto off = detail::toHostVecT<int>(res.view.cellFacetOffset);
    auto cnt = detail::toHostVecT<int>(res.view.cellFacetCount);
    auto nbr = detail::toHostVecT<peclet::voro::gid_t>(res.view.facetNeighbor);
    const int nF = (int)nbr.size();
    g.assign(3 * N, 0.0);
    E = 0;
    for (int c = 0; c < N; ++c) {
      // gather + sort real neighbours by distance (min-image) → reconstruct the cell.
      std::vector<std::array<Real, 4>> nb;  // dist², rx, ry, rz
      std::vector<int> ids;
      for (int f = off[c]; f < off[c] + cnt[c] && f < nF; ++f) {
        const long j = (long)nbr[f];
        if (j < 0 || j >= N) continue;
        Real r[3];
        for (int d = 0; d < 3; ++d) {
          Real rr = x[3 * j + d] - x[3 * c + d];
          rr = rr > Lh[d] ? rr - L[d] : (rr < -Lh[d] ? rr + L[d] : rr);
          r[d] = rr;
        }
        nb.push_back({r[0] * r[0] + r[1] * r[1] + r[2] * r[2], r[0], r[1], r[2]});
        ids.push_back((int)j);
      }
      const int M = (int)ids.size();
      std::vector<int> ord(M);
      for (int i = 0; i < M; ++i) ord[i] = i;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) { return nb[a][0] < nb[b][0]; });
      std::vector<Real> rx(M), ry(M), rz(M);
      std::vector<int> id2(M);
      for (int i = 0; i < M; ++i) {
        rx[i] = nb[ord[i]][1];
        ry[i] = nb[ord[i]][2];
        rz[i] = nb[ord[i]][3];
        id2[i] = ids[ord[i]];
      }
      RCell cell;
      buildConvexCell(cell, Larr, rx.data(), ry.data(), rz.data(), id2.data(), M);
      if (cell.empty() || cell.overflow) continue;

      // gather per-face area magnitudes Ag[k] and the area Jacobian dA[k][l][cc] = ∂A_k/∂n_l.
      const int np = cell.np;
      std::vector<double> Ag(np, 0.0), dA((size_t)np * np * 3, 0.0);
      for (int t = 0; t < cell.nt; ++t) {
        if (!cell.alive[t]) continue;
        int pl[3];
        double cb[3], gr[3][3][3];
        cell.geomVolumeAreaGrad(t, pl, cb, gr);
        for (int ii = 0; ii < 3; ++ii) {
          Ag[pl[ii]] += cb[ii];
          for (int jj = 0; jj < 3; ++jj)
            for (int cc = 0; cc < 3; ++cc)
              dA[((size_t)pl[ii] * np + pl[jj]) * 3 + cc] += gr[ii][jj][cc];
        }
      }
      // interfacial energy of this cell (½ per face — each interface face is shared by two cells),
      // and dGeom/dn_l accumulated over the cell's interface faces.
      std::vector<double> gn(3 * np, 0.0);
      for (int k = 0; k < np; ++k) {
        const int j = cell.pnbr[k];
        if (j < 0 || j >= N || type[j] == type[c]) continue;  // only different-type faces
        E += 0.5 * sigma * Ag[k];
        for (int l = 0; l < np; ++l)
          for (int cc = 0; cc < 3; ++cc)
            gn[3 * l + cc] += 0.5 * sigma * dA[((size_t)k * np + l) * 3 + cc];
      }
      // route ∂E_c/∂n_l → positions (self + neighbours) and scatter into g.
      double gx[128], gy[128], gz[128];
      for (int l = 0; l < np; ++l) {
        gx[l] = gn[3 * l];
        gy[l] = gn[3 * l + 1];
        gz[l] = gn[3 * l + 2];
      }
      const double seed3[3] = {(double)x[3 * c], (double)x[3 * c + 1], (double)x[3 * c + 2]};
      double fSelf[3], fwSelf, fnx[128], fny[128], fnz[128], fwn[128];
      chainToDofs<Voronoi>(cell, seed3, (const double*)nullptr, 0.0, (const double*)nullptr,
                           (double)L[0], gx, gy, gz, fSelf, fwSelf, fnx, fny, fnz, fwn);
      g[3 * c] += fSelf[0];
      g[3 * c + 1] += fSelf[1];
      g[3 * c + 2] += fSelf[2];
      for (int l = 0; l < np; ++l) {
        const int j = cell.pnbr[l];
        if (j < 0 || j >= N) continue;
        g[3 * j] += fnx[l];
        g[3 * j + 1] += fny[l];
        g[3 * j + 2] += fnz[l];
      }
    }
  };

  OtResult R;
  std::vector<double> g;
  double E;
  energyGrad(pos, g, E);
  const double E0 = E;
  if (verbose) {  // FD-validate the interfacial-area gradient (one DOF)
    const int d = 3 * (N / 2);
    const double h = 1e-7;
    std::vector<Real> xp = pos;
    std::vector<double> gp, gm;
    double Ep, Em;
    xp[d] += (Real)h;
    energyGrad(xp, gp, Ep);
    xp[d] -= (Real)(2 * h);
    energyGrad(xp, gm, Em);
    std::printf("      FD dE/dx_%d: analytic=%.4e fd=%.4e\n", d, g[d], (Ep - Em) / (2 * h));
  }
  for (int it = 0; it < maxIter; ++it) {
    double gnorm = 0;
    for (double v : g) gnorm = std::max(gnorm, std::fabs(v));
    R.iters = it;
    R.maxVolErr = E;  // report energy in maxVolErr slot
    if (verbose) std::printf("  [iface] iter %2d  E=%.6e gnorm=%.3e\n", it, E, gnorm);
    if (gnorm < tol) {
      R.converged = true;
      break;
    }
    // steepest descent with Armijo line search on E (dq = −g, g·dq = −|g|² < 0). Bound the initial
    // step to a trust region (≈2% of the spacing) so it does not overshoot across the non-smooth
    // topology-change kinks of the interfacial energy.
    const double gdq = -std::inner_product(g.begin(), g.end(), g.begin(), 0.0);
    const double spacing = std::cbrt((double)L[0] * L[1] * L[2] / N);
    double alpha = std::min(1.0, 0.02 * spacing / std::max(gnorm, 1e-30));
    bool accepted = false;
    std::vector<Real> xtry(3 * N);
    std::vector<double> gt;
    double Et;
    for (int bt = 0; bt < 30; ++bt) {
      for (int i = 0; i < 3 * N; ++i) xtry[i] = pos[i] - (Real)(alpha * g[i]);
      energyGrad(xtry, gt, Et);
      if (Et <= E + 1e-4 * alpha * gdq) {
        pos = xtry;
        g = gt;
        E = Et;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }
  R.meanVolErr = E / std::max(E0, 1e-30);  // final/initial energy ratio
  return R;
}

// ================================================================================================
// Device-resident volume optimiser (pure Voronoi positions) — every per-Newton computation runs on
// the Kokkos device: gradient assembly, matrix-free Gauss-Newton matvec, Jacobi-preconditioned CG,
// line search, and the DOF update. Only convergence scalars (energy, dot products) cross to host.
// This is the device path per docs/DEVICE_RESIDENCY_PLAN.md; the host meshVolumeOptimize is its
// oracle. (Weight DOFs, colored-GS-on-device, and MPI halo exchange are the recorded next
// increments.)
// ================================================================================================
template <class Real, class Sdf = NoSdf>
OtResult meshVolumeOptimizeDevice(std::vector<Real>& posHost, const std::vector<Real>& vsetIn,
                                  const Real L[3], int N, int sw, const Sdf& sdf, int maxNewton,
                                  Real tol, int cgIters = 300, bool verbose = false) {
  using MemSpace = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using DV = Kokkos::View<double*, MemSpace>;
  const Real Larr[3] = {L[0], L[1], L[2]};
  const double gamma = 1.0, boxVol = (double)L[0] * L[1] * L[2];
  const int nD = 3 * N;

  double sumVset = 0;
  for (int i = 0; i < N; ++i) sumVset += vsetIn[i];
  DV vset("mo.vset", N);
  {
    auto h = Kokkos::create_mirror_view(vset);
    for (int i = 0; i < N; ++i) h(i) = vsetIn[i] * (boxVol / sumVset);
    Kokkos::deep_copy(vset, h);
  }
  Kokkos::View<Real*, MemSpace> dpos("mo.pos", 3 * N), dw;
  Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(posHost.data(), 3 * N));
  Kokkos::View<long*, MemSpace> gd;

  DV g("mo.g", nD), dq("mo.dq", nD), diag("mo.diag", nD);
  DV rr("mo.r", nD), z("mo.z", nD), pp("mo.p", nD), Ap("mo.Ap", nD), yy("mo.y", N);

  // Build the tessellation at `x` and return E + max/mean vol error + empty count (device reduce).
  auto evaluate = [&](const Kokkos::View<Real*, MemSpace>& x, TessellatorResult<Real>& res,
                      double& E, double& maxErr, double& meanErr, long& nBad) {
    res = buildTessellation<Real, false, Sdf>(x, dw, N, Larr, sw, N, gd, sdf, true);
    auto vol = res.view.cellVolume;
    auto vs = vset;
    double e = 0, mx = 0, mn = 0;
    long nb = 0;
    Kokkos::parallel_reduce(
        "mo.E", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(int i, double& le, double& lmx, double& lmn, long& lnb) {
          const double d = vol(i) - vs(i);
          le += gamma * d * d;
          lmx = Kokkos::max(lmx, Kokkos::fabs(d));
          lmn += Kokkos::fabs(d);
          if (vol(i) <= 0.0) ++lnb;
        },
        e, Kokkos::Max<double>(mx), mn, nb);
    E = e;
    maxErr = mx;
    meanErr = mn / N;
    nBad = nb;
  };

  TessellatorResult<Real> res;
  double E, maxErr, meanErr;
  long nBad;
  evaluate(dpos, res, E, maxErr, meanErr, nBad);

  OtResult R;
  for (int it = 0; it < maxNewton; ++it) {
    auto off = res.view.cellFacetOffset;
    auto cnt = res.view.cellFacetCount;
    auto nbr = res.view.facetNeighbor;
    auto dvr = res.view.facetConnect;  // ∂V_c/∂r per facet
    auto vol = res.view.cellVolume;
    auto vs = vset;

    // gradient g and Gauss-Newton diagonal (device, atomic scatter).
    Kokkos::deep_copy(g, 0.0);
    Kokkos::deep_copy(diag, 0.0);
    Kokkos::parallel_for(
        "mo.grad", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int c) {
          const double Rc = 2.0 * gamma * (vol(c) - vs(c));
          double sx = 0, sy = 0, sz = 0;
          for (int f = off(c); f < off(c) + cnt(c); ++f) {
            const int j = (int)nbr(f);
            if (j < 0 || j >= N) continue;
            const double dx = dvr(3 * f), dy = dvr(3 * f + 1), dz = dvr(3 * f + 2);
            Kokkos::atomic_add(&g(3 * j), Rc * dx);
            Kokkos::atomic_add(&g(3 * j + 1), Rc * dy);
            Kokkos::atomic_add(&g(3 * j + 2), Rc * dz);
            Kokkos::atomic_add(&diag(3 * j), 2.0 * gamma * dx * dx);
            Kokkos::atomic_add(&diag(3 * j + 1), 2.0 * gamma * dy * dy);
            Kokkos::atomic_add(&diag(3 * j + 2), 2.0 * gamma * dz * dz);
            sx -= dx;
            sy -= dy;
            sz -= dz;
          }
          Kokkos::atomic_add(&g(3 * c), Rc * sx);
          Kokkos::atomic_add(&g(3 * c + 1), Rc * sy);
          Kokkos::atomic_add(&g(3 * c + 2), Rc * sz);
          Kokkos::atomic_add(&diag(3 * c), 2.0 * gamma * sx * sx);
          Kokkos::atomic_add(&diag(3 * c + 1), 2.0 * gamma * sy * sy);
          Kokkos::atomic_add(&diag(3 * c + 2), 2.0 * gamma * sz * sz);
        });
    double gnorm = 0;
    Kokkos::parallel_reduce(
        "mo.gnorm", Kokkos::RangePolicy<Exec>(0, nD),
        KOKKOS_LAMBDA(int i, double& m) { m = Kokkos::max(m, Kokkos::fabs(g(i))); },
        Kokkos::Max<double>(gnorm));

    R.iters = it;
    R.maxVolErr = maxErr;
    R.meanVolErr = meanErr;
    R.nEmpty = nBad;
    if (verbose)
      std::printf("  [dvmesh] iter %2d  E=%.4e maxVolErr=%.3e gnorm=%.3e nBad=%ld\n", it, E, maxErr,
                  gnorm, nBad);
    if (gnorm < tol) {
      R.converged = true;
      break;
    }

    // matrix-free Gauss-Newton apply H v = 2γ Jᵀ(J v) (two device passes: J into yy, Jᵀ scatter).
    auto Hmul = [&](const DV& v, DV& out) {
      auto y = yy;
      Kokkos::parallel_for(
          "mo.Jv", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int c) {
            double yc = 0;
            for (int f = off(c); f < off(c) + cnt(c); ++f) {
              const int j = (int)nbr(f);
              if (j < 0 || j >= N) continue;
              for (int d = 0; d < 3; ++d) yc += dvr(3 * f + d) * (v(3 * j + d) - v(3 * c + d));
            }
            y(c) = yc;
          });
      Kokkos::deep_copy(out, 0.0);
      Kokkos::parallel_for(
          "mo.JTy", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int c) {
            const double s = 2.0 * gamma * y(c);
            for (int f = off(c); f < off(c) + cnt(c); ++f) {
              const int j = (int)nbr(f);
              if (j < 0 || j >= N) continue;
              for (int d = 0; d < 3; ++d) {
                Kokkos::atomic_add(&out(3 * j + d), s * dvr(3 * f + d));
                Kokkos::atomic_add(&out(3 * c + d), -s * dvr(3 * f + d));
              }
            }
          });
    };
    auto dot = [&](const DV& a, const DV& b) {
      double s = 0;
      Kokkos::parallel_reduce(
          "mo.dot", Kokkos::RangePolicy<Exec>(0, nD),
          KOKKOS_LAMBDA(int i, double& l) { l += a(i) * b(i); }, s);
      return s;
    };

    // Jacobi-preconditioned CG for H dq = −g (device).
    Kokkos::deep_copy(dq, 0.0);
    {
      auto G = g, D = diag;
      Kokkos::parallel_for(
          "mo.r0", Kokkos::RangePolicy<Exec>(0, nD), KOKKOS_LAMBDA(int i) {
            rr(i) = -G(i);
            z(i) = Kokkos::fabs(D(i)) > 1e-30 ? rr(i) / D(i) : rr(i);
            pp(i) = z(i);
          });
    }
    double rz = dot(rr, z), rz0 = rz;
    for (int k = 0; k < cgIters && rz > 1e-18 * rz0; ++k) {
      Hmul(pp, Ap);
      const double pAp = dot(pp, Ap);
      if (pAp <= 0) break;
      const double a = rz / pAp;
      {
        auto D = diag;
        Kokkos::parallel_for(
            "mo.cgupd", Kokkos::RangePolicy<Exec>(0, nD), KOKKOS_LAMBDA(int i) {
              dq(i) += a * pp(i);
              rr(i) -= a * Ap(i);
              z(i) = Kokkos::fabs(D(i)) > 1e-30 ? rr(i) / D(i) : rr(i);
            });
      }
      const double rzn = dot(rr, z);
      const double beta = rzn / rz;
      Kokkos::parallel_for(
          "mo.pupd", Kokkos::RangePolicy<Exec>(0, nD),
          KOKKOS_LAMBDA(int i) { pp(i) = z(i) + beta * pp(i); });
      rz = rzn;
    }
    const double gdq = dot(g, dq);

    // Armijo line search on E: x_try = x + α dq (device), rebuild + reduce E; backtrack.
    double alpha = 1.0;
    bool accepted = false;
    Kokkos::View<Real*, MemSpace> xtry("mo.xtry", 3 * N);
    for (int bt = 0; bt < 24; ++bt) {
      const double al = alpha;
      auto X = dpos, DQ = dq, XT = xtry;
      Kokkos::parallel_for(
          "mo.trial", Kokkos::RangePolicy<Exec>(0, nD),
          KOKKOS_LAMBDA(int i) { XT(i) = X(i) + (Real)(al * DQ(i)); });
      TessellatorResult<Real> resT;
      double Et, mxT, mnT;
      long nbT;
      evaluate(xtry, resT, Et, mxT, mnT, nbT);
      if (nbT == 0 && Et <= E + 1e-4 * alpha * gdq) {
        Kokkos::deep_copy(dpos, xtry);
        res = resT;
        E = Et;
        maxErr = mxT;
        meanErr = mnT;
        nBad = nbT;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }
  // download the optimised positions.
  auto h = Kokkos::create_mirror_view(dpos);
  Kokkos::deep_copy(h, dpos);
  for (int i = 0; i < 3 * N; ++i) posHost[i] = h(i);
  return R;
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_MESH_OPTIMIZER_HPP
