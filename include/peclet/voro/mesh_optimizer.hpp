/**
 * @file mesh_optimizer.hpp
 * \brief Energy-minimisation mesh optimiser over SEED POSITIONS (pure Voronoi) — moves the seeds so
 * the cells minimise an energy, e.g. Σ γ (V_i − V_set,i)² to drive cells to target volumes (V_set
 * from an SDF ⇒ refinement near solids). A Surface-Evolver-style tool built on the differentiable
 * Voronoi geometry.
 *
 * DOFs = the seed positions x (3N). Pure Voronoi cells PARTITION space exactly (Σ V = box volume,
 * constant), so the energy is directly evaluable and the optimiser is well-posed — unlike the
 * power-weight OT path, whose periodic min-image diagram is only approximately space-filling.
 *
 * Gradient: the tessellation publishes ∂V_c/∂r_k per facet (`facetConnect`, r_k = x_j − x_i), so
 *   ∂V_c/∂x_j = facetConnect_k ,  ∂V_c/∂x_c = −Σ_k facetConnect_k .
 * For E = Σ_c γ (V_c − V_set,c)²,  g_i = Σ_c 2γ (V_c − V_set,c) ∂V_c/∂x_i.
 *
 * Newton–Raphson with the Gauss-Newton Hessian H = 2γ JᵀJ (J_ci = ∂V_c/∂x_i), applied MATRIX-FREE
 * as two facet-CSR passes, solved by a Jacobi-preconditioned CG; a backtracking Armijo line search
 * on E completes the step. Host-orchestrated; the tessellation build runs on the Kokkos device.
 */
#ifndef PECLET_VORO_MESH_OPTIMIZER_HPP
#define PECLET_VORO_MESH_OPTIMIZER_HPP

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/voro/ot_optimizer.hpp"  // OtResult + detail::toHostVec
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tessellator.hpp"

namespace peclet::voro {

/**
 * Move the seed positions `pos` (size 3N, updated in place) so the pure-Voronoi cell volumes match
 * `vsetIn` (size N), minimising E = Σ γ (V_i − V_set,i)² by damped Gauss-Newton. Returns the volume
 * error diagnostics (OtResult).
 *
 * @param cgIters   inner CG iterations per Newton step.
 * @param tol       convergence on the gradient ∞-norm.
 */
template <class Real, class Sdf = NoSdf>
OtResult voronoiVolumeOptimize(std::vector<Real>& pos, const std::vector<Real>& vsetIn,
                               const Real L[3], int N, int sw, const Sdf& sdf, int maxNewton,
                               Real tol, int cgIters = 200, bool verbose = false) {
  using MemSpace = peclet::core::MemSpace;
  const Real Larr[3] = {L[0], L[1], L[2]};
  const double gamma = 1.0;

  Kokkos::View<Real*, MemSpace> dw;  // empty weights (Voronoi)
  Kokkos::View<long*, MemSpace> gd;

  // normalise the targets so Σ V_set = box volume (= Σ V for a space-filling Voronoi), once.
  double sumVset = 0, boxVol = (double)L[0] * L[1] * L[2];
  for (int i = 0; i < N; ++i) sumVset += vsetIn[i];
  std::vector<double> vset(N);
  for (int i = 0; i < N; ++i) vset[i] = vsetIn[i] * (boxVol / sumVset);

  // Build the Voronoi tessellation at positions `x`; download volumes + the per-facet ∂V/∂r
  // (facetConnect) and the cell→facet CSR. Returns energy E and the worst/mean volume error.
  struct Geo {
    std::vector<double> vol, dvr;                     // V (N), ∂V/∂r per facet (3*nFacets)
    std::vector<int> off, cnt;                        // cell→facet CSR
    std::vector<peclet::voro::gid_t> nbr;             // facet neighbour ids
    double E = 0, maxErr = 0, meanErr = 0;
    long nBad = 0;
  };
  auto build = [&](const std::vector<Real>& x, Geo& G) {
    Kokkos::View<Real*, MemSpace> dpos("mo.pos", 3 * N);
    Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(x.data(), 3 * N));
    auto res = buildTessellation<Real, false, Sdf>(dpos, dw, N, Larr, sw, N, gd, sdf,
                                                   /*withForceGeom=*/true);
    G.vol = detail::toHostVec<Real>(res.view.cellVolume);
    G.dvr = detail::toHostVec<Real>(res.view.facetConnect);
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

  const int nF0 = 0;
  (void)nF0;
  Geo G;
  build(pos, G);

  OtResult R;
  std::vector<double> g(3 * N), dx(3 * N), diag(3 * N);
  for (int it = 0; it < maxNewton; ++it) {
    const int nFacets = (int)G.nbr.size();
    // gradient g_i = Σ_c 2γ(V_c − V_set,c) ∂V_c/∂x_i, and the Gauss-Newton diagonal for Jacobi.
    std::fill(g.begin(), g.end(), 0.0);
    std::fill(diag.begin(), diag.end(), 0.0);
    for (int c = 0; c < N; ++c) {
      const double Rc = 2.0 * gamma * (G.vol[c] - vset[c]);
      double gcx = 0, gcy = 0, gcz = 0;  // ∂V_c/∂x_c = −Σ_k ∂V_c/∂r_k
      const int fe = G.off[c] + G.cnt[c];
      for (int f = G.off[c]; f < fe && f < nFacets; ++f) {
        const long j = (long)G.nbr[f];
        if (j < 0 || j >= N) continue;  // boundary/SDF-wall facet: fixed, no DOF
        const double dx0 = G.dvr[3 * f], dy0 = G.dvr[3 * f + 1], dz0 = G.dvr[3 * f + 2];
        g[3 * j] += Rc * dx0;
        g[3 * j + 1] += Rc * dy0;
        g[3 * j + 2] += Rc * dz0;
        gcx -= dx0;
        gcy -= dy0;
        gcz -= dz0;
        diag[3 * j] += 2.0 * gamma * dx0 * dx0;
        diag[3 * j + 1] += 2.0 * gamma * dy0 * dy0;
        diag[3 * j + 2] += 2.0 * gamma * dz0 * dz0;
      }
      g[3 * c] += Rc * gcx;
      g[3 * c + 1] += Rc * gcy;
      g[3 * c + 2] += Rc * gcz;
      diag[3 * c] += 2.0 * gamma * gcx * gcx;
      diag[3 * c + 1] += 2.0 * gamma * gcy * gcy;
      diag[3 * c + 2] += 2.0 * gamma * gcz * gcz;
    }
    double gnorm = 0;
    for (int i = 0; i < 3 * N; ++i) gnorm = std::max(gnorm, std::fabs(g[i]));

    if (verbose && it == 0) {  // FD-validate the gradient against E (one DOF)
      const int d = 3 * (N / 2);
      const double h = 1e-7;
      std::vector<Real> xp = pos;
      xp[d] += (Real)h;
      Geo Gp;
      build(xp, Gp);
      xp[d] -= (Real)(2 * h);
      Geo Gm;
      build(xp, Gm);
      std::printf("      FD dE/dx_%d: analytic=%.4e fd=%.4e\n", d, g[d], (Gp.E - Gm.E) / (2 * h));
    }

    R.iters = it;
    R.maxVolErr = G.maxErr;
    R.meanVolErr = G.meanErr;
    R.nEmpty = G.nBad;
    if (verbose)
      std::printf("  [vmesh] iter %2d  E=%.4e maxVolErr=%.3e meanVolErr=%.3e gnorm=%.3e nBad=%ld\n",
                  it, G.E, G.maxErr, G.meanErr, gnorm, G.nBad);
    if (gnorm < tol) {
      R.converged = true;
      break;
    }

    // matrix-free Gauss-Newton apply: H δx = 2γ Jᵀ(J δx), J_ci = ∂V_c/∂x_i.
    auto Hmul = [&](const std::vector<double>& v, std::vector<double>& out) {
      std::fill(out.begin(), out.end(), 0.0);
      for (int c = 0; c < N; ++c) {
        const int fe = G.off[c] + G.cnt[c];
        double yc = 0;  // (J v)_c = Σ_k ∂V_c/∂r_k · (v_nbr − v_c)
        for (int f = G.off[c]; f < fe && f < nFacets; ++f) {
          const long j = (long)G.nbr[f];
          if (j < 0 || j >= N) continue;
          for (int d = 0; d < 3; ++d) yc += G.dvr[3 * f + d] * (v[3 * j + d] - v[3 * c + d]);
        }
        const double s = 2.0 * gamma * yc;
        for (int f = G.off[c]; f < fe && f < nFacets; ++f) {
          const long j = (long)G.nbr[f];
          if (j < 0 || j >= N) continue;
          for (int d = 0; d < 3; ++d) {
            out[3 * j + d] += s * G.dvr[3 * f + d];
            out[3 * c + d] -= s * G.dvr[3 * f + d];
          }
        }
      }
    };

    // Jacobi-preconditioned CG for H δx = −g (δx = 0 start).
    std::fill(dx.begin(), dx.end(), 0.0);
    std::vector<double> rr(3 * N), z(3 * N), p(3 * N), Ap(3 * N);
    for (int i = 0; i < 3 * N; ++i) rr[i] = -g[i];
    auto prec = [&](const std::vector<double>& in, std::vector<double>& outv) {
      for (int i = 0; i < 3 * N; ++i) outv[i] = diag[i] > 1e-30 ? in[i] / diag[i] : in[i];
    };
    prec(rr, z);
    p = z;
    double rz = 0;
    for (int i = 0; i < 3 * N; ++i) rz += rr[i] * z[i];
    const double rz0 = rz;
    for (int k = 0; k < cgIters && rz > 1e-20 * rz0; ++k) {
      Hmul(p, Ap);
      double pAp = 0;
      for (int i = 0; i < 3 * N; ++i) pAp += p[i] * Ap[i];
      if (pAp <= 0) break;  // non-positive curvature guard
      const double a = rz / pAp;
      for (int i = 0; i < 3 * N; ++i) {
        dx[i] += a * p[i];
        rr[i] -= a * Ap[i];
      }
      prec(rr, z);
      double rzn = 0;
      for (int i = 0; i < 3 * N; ++i) rzn += rr[i] * z[i];
      const double beta = rzn / rz;
      for (int i = 0; i < 3 * N; ++i) p[i] = z[i] + beta * p[i];
      rz = rzn;
    }

    // Armijo backtracking on E: accept the largest α ∈ {1, ½, …} with E(x+αδx) ≤ E − c·α·gᵀδx and
    // no empty cell. gᵀδx < 0 (descent, since δx ≈ −H⁻¹g and H is SPD).
    double gdx = 0;
    for (int i = 0; i < 3 * N; ++i) gdx += g[i] * dx[i];
    double alpha = 1.0;
    bool accepted = false;
    std::vector<Real> xtry(3 * N);
    Geo Gt;
    for (int bt = 0; bt < 20; ++bt) {
      for (int i = 0; i < 3 * N; ++i) xtry[i] = pos[i] + (Real)(alpha * dx[i]);
      build(xtry, Gt);
      if (Gt.nBad == 0 && Gt.E <= G.E + 1e-4 * alpha * gdx) {
        pos = xtry;
        G = Gt;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (verbose) std::printf("      cg rz %.2e->%.2e  alpha=%.4f  gdx=%.3e\n", rz0, rz, alpha, gdx);
    if (!accepted) break;  // line search failed
  }
  return R;
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_MESH_OPTIMIZER_HPP
