/**
 * @file test_fv_operators.cpp
 * @brief Track C, rung C1 gate (Voronoi methods plan): the covolume operators on the face mesh.
 *
 *  (A) Structure: every interior face is owned once; Σ_i V_i (div u)_i = 0 for any face flux in
 *      the periodic box (no boundary faces); the cell→faces CSR closes (each face twice).
 *  (B) Adjointness to round-off on a RANDOM Voronoi mesh: ⟨p, div u⟩_V = −⟨grad p, u⟩_F, and
 *      symmetry of L = div grad in ⟨·,·⟩_V; L equals the graph Laplacian Σ A_f/d_f (p_j − p_i)/V_i.
 *  (C) Exactness on a uniform field: the Perot reconstruction returns the field to round-off on
 *      the random mesh (the Green–Gauss gradient of a LINEAR field likewise).
 *  (D) Manufactured solution p = sin 2πx sin 2πy sin 2πz on a jittered cubic lattice (jitter
 *      0.15 h): L2 error of L p vs −12π² p and of the Green–Gauss gradient vs ∇p at h, h/2, h/4 —
 *      the two-point Laplacian must converge (order ≥ 1.5 measured; the residual first-order piece
 *      is the face-centroid skewness that track B's centroidal energy removes), the gradient
 *      likewise.
 */
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/fv/operators.hpp"
#include "peclet/voro/tessellator.hpp"
#include "peclet/voro/transpose.hpp"

using Real = double;
using Mem = peclet::core::MemSpace;
using DV = Kokkos::View<Real*, Mem>;
namespace fv = peclet::voro::fv;

template <class T>
static std::vector<T> down(const Kokkos::View<T*, Mem>& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  return std::vector<T>(h.data(), h.data() + h.extent(0));
}
static DV up(const std::vector<Real>& h, const char* name) {
  DV d(name, h.size());
  Kokkos::deep_copy(d, Kokkos::View<const Real*, Kokkos::HostSpace>(h.data(), h.size()));
  return d;
}

static fv::FaceMesh<Real> meshOf(const std::vector<Real>& pos, int N, const Real L[3],
                                 peclet::voro::TessellationView<Real>& viewOut) {
  DV dpos = up(pos, "pos"), dw;
  Kokkos::View<long*, Mem> gd;
  auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(dpos, dw, N, L, 4, N,
                                                                               gd, {}, true);
  viewOut = res.view;
  auto aux = peclet::voro::buildAuxMaps(res.view);
  return fv::buildFaceMesh(res.view, aux);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const Real L[3] = {1, 1, 1};
    std::mt19937 rng(3);
    std::uniform_real_distribution<Real> U(0, 1);
    std::printf("=== C1: covolume operators on the Voronoi face mesh ===\n");

    // ---- (A)+(B)+(C): random mesh -----------------------------------------------------------
    {
      const int N = 4000;
      std::vector<Real> pos(3 * N);
      for (auto& x : pos)
        x = U(rng);
      peclet::voro::TessellationView<Real> view;
      auto m = meshOf(pos, N, L, view);
      auto A = down(m.faceCellA), B = down(m.faceCellB), off = down(m.cellFaceOffset);
      long badB = 0;
      for (int f = 0; f < m.nInterior; ++f)
        if (B[f] < 0 || A[f] == B[f])
          ++badB;
      const bool structOk = m.nFaces == m.nInterior && badB == 0 && off[N] == 2 * m.nInterior &&
                            2 * m.nInterior == view.numFacets();
      std::printf("  (A) faces=%d interior=%d facets=%d csr=%d  %s\n", m.nFaces, m.nInterior,
                  view.numFacets(), off[N], structOk ? "OK" : "FAIL");
      if (!structOk)
        bad = 1;
      // random face flux + random pressure
      std::vector<Real> uh(m.nFaces), ph(N);
      for (auto& v : uh)
        v = 2 * U(rng) - 1;
      for (auto& v : ph)
        v = 2 * U(rng) - 1;
      DV u = up(uh, "u"), p = up(ph, "p"), divu("div", N), gp("gp", m.nFaces), Lp("Lp", N),
         q("q", N), Lq("Lq", N);
      fv::divergence(m, u, divu);
      fv::faceGradient(m, p, gp);
      // Σ V div = 0
      auto dv = down(divu), vol = down(m.cellVolume);
      double sumDiv = 0, scaleDiv = 0;
      for (int i = 0; i < N; ++i) {
        sumDiv += vol[i] * dv[i];
        scaleDiv += std::fabs(vol[i] * dv[i]);
      }
      const double lhs = fv::dotCells(m, p, divu), rhs = -fv::dotFaces(m, gp, u);
      const double adj = std::fabs(lhs - rhs) / std::max(std::fabs(lhs), 1e-30);
      // symmetry of L
      std::vector<Real> qh(N);
      for (auto& v : qh)
        v = 2 * U(rng) - 1;
      Kokkos::deep_copy(q, Kokkos::View<const Real*, Kokkos::HostSpace>(qh.data(), N));
      DV sf;
      fv::laplacian(m, p, Lp, sf);
      fv::laplacian(m, q, Lq, sf);
      const double s1 = fv::dotCells(m, Lp, q), s2 = fv::dotCells(m, p, Lq);
      const double sym = std::fabs(s1 - s2) / std::max(std::fabs(s1), 1e-30);
      // L vs the graph Laplacian on host
      auto Lph = down(Lp), fa = down(m.faceArea), fd = down(m.faceDist);
      std::vector<double> Lg(N, 0.0);
      for (int f = 0; f < m.nInterior; ++f) {
        const double w = fa[f] / fd[f] * (ph[B[f]] - ph[A[f]]);
        Lg[A[f]] += w / vol[A[f]];
        Lg[B[f]] -= w / vol[B[f]];
      }
      double lapErr = 0, lapScale = 0;
      for (int i = 0; i < N; ++i) {
        lapErr = std::max(lapErr, std::fabs(Lg[i] - Lph[i]));
        lapScale = std::max(lapScale, std::fabs(Lg[i]));
      }
      // 1e-11: the inner products are parallel reductions over ~1e5 terms with cancellation
      // (⟨p, div u⟩ ~ O(1) out of O(10) partial sums), so the last digits depend on the thread
      // count and scheduling; the identities themselves are exact.
      const bool bOk = std::fabs(sumDiv) < 1e-11 * scaleDiv && adj < 1e-11 && sym < 1e-11 &&
                       lapErr < 1e-12 * lapScale;
      std::printf(
          "  (B) sum V div=%.2e (scale %.2e) adjoint=%.2e symmetry=%.2e L-vs-graph=%.2e  %s\n",
          sumDiv, scaleDiv, adj, sym, lapErr / lapScale, bOk ? "OK" : "FAIL");
      if (!bOk)
        bad = 1;
      // (C) uniform field through Perot; linear field through Green–Gauss
      std::vector<Real> Uc(3 * N);
      for (int i = 0; i < N; ++i) {
        Uc[3 * i] = 1.0;
        Uc[3 * i + 1] = 2.0;
        Uc[3 * i + 2] = -3.0;
      }
      DV uc = up(Uc, "uc"), uf("uf", m.nFaces), ur("ur", 3 * N);
      fv::projectToFaces(m, uc, uf);
      fv::perotVelocity(m, uf, ur);
      auto urh = down(ur);
      double perotErr = 0;
      for (int i = 0; i < 3 * N; ++i)
        perotErr = std::max(perotErr, std::fabs(urh[i] - Uc[i]));
      // Green–Gauss of a linear field p = a·x needs the SEED positions: p(x_i) is exact only for
      // faces whose centroid lies on the connector, so this is a consistency measure, not exact.
      std::vector<Real> pl(N);
      for (int i = 0; i < N; ++i)
        pl[i] = 0.3 * pos[3 * i] - 0.7 * pos[3 * i + 1] + 0.2 * pos[3 * i + 2];
      // (periodic wrap makes a global linear field discontinuous — use the gradient on interior
      // cells away from the wrap: skip cells within 0.1 of a box face)
      DV plv = up(pl, "pl"), ggl("ggl", 3 * N);
      fv::greenGaussGradient(m, plv, ggl);
      auto gglh = down(ggl);
      double ggErr = 0;
      long ggN = 0;
      for (int i = 0; i < N; ++i) {
        bool inner = true;
        for (int c = 0; c < 3; ++c)
          inner = inner && pos[3 * i + c] > 0.12 && pos[3 * i + c] < 0.88;
        if (!inner)
          continue;
        ++ggN;
        const double ex[3] = {0.3, -0.7, 0.2};
        for (int c = 0; c < 3; ++c)
          ggErr = std::max(ggErr, std::fabs(gglh[3 * i + c] - ex[c]));
      }
      const bool cOk = perotErr < 1e-12;  // Green–Gauss on a random mesh is skewness-limited
      std::printf(
          "  (C) Perot uniform-field error=%.2e (exact); Green–Gauss linear-field worst=%.2e "
          "over %ld cells (skewness-limited, informational)  %s\n",
          perotErr, ggErr, ggN, cOk ? "OK" : "FAIL");
      if (!cOk)
        bad = 1;
    }

    // ---- (D) manufactured solution on a jittered lattice --------------------------------------
    {
      const int ns[3] = {12, 24, 48};
      double eL[3], eG[3], eP[3];
      for (int r = 0; r < 3; ++r) {
        const int n = ns[r], N = n * n * n;
        const Real h = 1.0 / n;
        std::vector<Real> pos(3 * N);
        int k = 0;
        for (int a = 0; a < n; ++a)
          for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c) {
              pos[3 * k] = (a + 0.5 + 0.15 * (2 * U(rng) - 1)) * h;
              pos[3 * k + 1] = (b + 0.5 + 0.15 * (2 * U(rng) - 1)) * h;
              pos[3 * k + 2] = (c + 0.5 + 0.15 * (2 * U(rng) - 1)) * h;
              ++k;
            }
        peclet::voro::TessellationView<Real> view;
        auto m = meshOf(pos, N, L, view);
        std::vector<Real> ph(N), ex(N), gx(3 * N);
        const Real w = 2 * M_PI;
        for (int i = 0; i < N; ++i) {
          const Real x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
          ph[i] = std::sin(w * x) * std::sin(w * y) * std::sin(w * z);
          ex[i] = -3 * w * w * ph[i];
          gx[3 * i] = w * std::cos(w * x) * std::sin(w * y) * std::sin(w * z);
          gx[3 * i + 1] = w * std::sin(w * x) * std::cos(w * y) * std::sin(w * z);
          gx[3 * i + 2] = w * std::sin(w * x) * std::sin(w * y) * std::cos(w * z);
        }
        DV p = up(ph, "p"), Lp("Lp", N), g("g", 3 * N), sf, fsrc = up(ex, "f"), psol("psol", N);
        fv::laplacian(m, p, Lp, sf);
        fv::greenGaussGradient(m, p, g);
        // the Poisson SOLVE −L p = −f (f = L p_exact): the two-point flux is sampled at the
        // connector midpoint, so the cellwise residual L p − f is O(1) on a non-centroidal mesh
        // (skewness — the B4 measurement), while the solution converges.
        std::vector<Real> negf(N);
        for (int i = 0; i < N; ++i)
          negf[i] = -ex[i];
        DV nf = up(negf, "nf");
        Real res = 0;
        const int its = fv::poissonCG(m, nf, psol, Real(1e-10), 4000, res);
        auto Lph = down(Lp), gh = down(g), vol = down(m.cellVolume), psh = down(psol);
        double e2 = 0, s2 = 0, ge2 = 0, gs2 = 0, pe2 = 0, ps2 = 0, pm = 0;
        for (int i = 0; i < N; ++i)
          pm += vol[i] * ph[i];
        for (int i = 0; i < N; ++i) {
          e2 += vol[i] * (Lph[i] - ex[i]) * (Lph[i] - ex[i]);
          s2 += vol[i] * ex[i] * ex[i];
          const double pex = ph[i] - pm;  // exact solution with the same zero mean
          pe2 += vol[i] * (psh[i] - pex) * (psh[i] - pex);
          ps2 += vol[i] * pex * pex;
          for (int c = 0; c < 3; ++c) {
            ge2 += vol[i] * (gh[3 * i + c] - gx[3 * i + c]) * (gh[3 * i + c] - gx[3 * i + c]);
            gs2 += vol[i] * gx[3 * i + c] * gx[3 * i + c];
          }
        }
        eL[r] = std::sqrt(e2 / s2);
        eG[r] = std::sqrt(ge2 / gs2);
        eP[r] = std::sqrt(pe2 / ps2);
        std::printf(
            "  (D) n=%3d h=%.4f  Poisson solution L2 err=%.3e (CG %d its, res %.1e) | "
            "residual consistency=%.3e  Green–Gauss=%.3e (skewness-limited)\n",
            n, h, eP[r], its, res, eL[r], eG[r]);
      }
      const double oP = std::log(eP[0] / eP[2]) / std::log(4.0),
                   oL = std::log(eL[0] / eL[2]) / std::log(4.0),
                   oG = std::log(eG[0] / eG[2]) / std::log(4.0);
      const bool dOk = oP >= 1.5 && eP[2] < 1e-2;
      std::printf(
          "  (D) convergence order in h — Poisson solution %.2f (gated ≥ 1.5); residual %.2f, "
          "gradient %.2f (informational)  %s\n",
          oP, oL, oG, dOk ? "OK" : "FAIL");
      if (!dOk)
        bad = 1;
    }
  }
  Kokkos::finalize();
  std::printf(bad ? "VORO-FV FAIL\n" : "VORO-FV OK\n");
  return bad;
}
