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

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <unordered_map>
#include <vector>

#include "peclet/core/amr/momentum.hpp"  // greedyColoring
#include "peclet/voro/ot_optimizer.hpp"  // OtResult + detail::toHostVec/toHostVecT
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tessellator.hpp"

namespace peclet::voro {

enum class Precond { Jacobi, ColoredGS };

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
    auto precond = [&](const std::vector<double>& r, std::vector<double>& z) {
      if (prec == Precond::Jacobi) {
        for (int i = 0; i < nD; ++i) z[i] = std::fabs(Hdiag[i]) > 1e-30 ? r[i] / Hdiag[i] : r[i];
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

}  // namespace peclet::voro

#endif  // PECLET_VORO_MESH_OPTIMIZER_HPP
