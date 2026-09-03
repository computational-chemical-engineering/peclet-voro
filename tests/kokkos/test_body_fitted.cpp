/**
 * @file test_body_fitted.cpp
 * @brief Track C, rung C3 gate (Voronoi methods plan): body-fitted no-slip walls for both static
 * solvers — plane Poiseuille flow between two SDF slabs (periodic in x and z, body force f_x),
 * u(y) = (f/2ν)(y − y0)(y1 − y). The seeds are a cubic lattice in the gap with the walls halfway
 * between seed rows, so the interior two-point flux is EXACT for the parabola; the wall face
 * carries the two-point wall flux (U_i − U_wall)/h_A, which is the exact derivative at h_A/2, not
 * at the wall — the classic half-cell O(1) local residual on the wall row that converges at second
 * order globally (measured here, not assumed).
 *
 *  (A) A-priori residual of the exact parabola: the semi-discrete Stokes tendency ν Δ u + f
 *      vanishes to round-off on every cell not touching a wall (both solvers; the covolume face
 *      tendency Rᵀ a likewise on faces of interior cells); the wall-row residual is reported
 *      (expected f/4 for the two-point wall flux).
 *  (B) Time-march to the steady state (SSP-RK3, diffusion number 0.2) at n = 4, 8, 16 cells
 *      across the gap: velocity error vs the parabola for the collocated (default, skew-corrected)
 *      and the covolume solver — order ≥ 1.9 (gated); the flux is divergence-free to round-off and
 *      the wall faces carry zero flux.
 */
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/fv/collocated.hpp"
#include "peclet/voro/fv/covolume.hpp"
#include "peclet/voro/sdf.hpp"

// fluid for y0 < y < y1 (φ > 0 in the fluid)
struct Slab {
  Real y0, y1;
  KOKKOS_INLINE_FUNCTION Real eval(Real, Real y, Real) const {
    return Kokkos::fmin(y - y0, y1 - y);
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-5); }
};

static fv::FaceMesh<Real> slabMesh(const std::vector<Real>& pos, int N, const Real L[3],
                                   const Slab& sdf) {
  DV dpos = up(pos, "pos"), dw;
  Kokkos::View<long*, Mem> gd;
  auto res =
      peclet::voro::buildTessellation<Real, false, Slab>(dpos, dw, N, L, 4, N, gd, sdf, true);
  auto aux = peclet::voro::buildAuxMaps(res.view);
  return fv::buildFaceMesh(res.view, aux);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const Real y0 = 0.25, y1 = 0.75, H = y1 - y0, nu = 1.0, fx = 1.0;
    const Slab sdf{y0, y1};
    auto uex = [&](Real y) { return fx / (2 * nu) * (y - y0) * (y1 - y); };
    std::printf("=== C3: body-fitted Poiseuille between SDF slabs (both solvers) ===\n");
    auto seeds = [&](int n, std::vector<Real>& pos, Real L[3]) {
      const Real h = H / n;
      // a cubic periodic box: the tessellator's periodic gather needs every box extent to exceed
      // twice its coverage radius (a 0.25-thin z extent segfaults the cell builder — engine limit)
      const int nx = (int)std::lround(1.0 / h), nz = nx;
      L[0] = nx * h;
      L[1] = 1.0;
      L[2] = nz * h;
      pos.clear();
      for (int k = 0; k < nz; ++k)
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < nx; ++i) {
            pos.push_back((i + 0.5) * h);
            pos.push_back(y0 + (j + 0.5) * h);
            pos.push_back((k + 0.5) * h);
          }
      return (int)pos.size() / 3;
    };
    // wall-adjacent cells
    auto wallCells = [&](const fv::FaceMesh<Real>& m) {
      std::vector<char> w(m.nCells, 0);
      auto A = down(m.faceCellA);
      for (int f = m.nInterior; f < m.nFaces; ++f)
        w[A[f]] = 1;
      return w;
    };

    // ---- (A) a-priori residual ----------------------------------------------------------------
    {
      const int n = 16;
      std::vector<Real> pos;
      Real L[3];
      const int N = seeds(n, pos, L);
      auto m = slabMesh(pos, N, L, sdf);
      auto wc = wallCells(m);
      std::vector<Real> Uh(3 * N, 0.0), fh(3 * N, 0.0);
      for (int i = 0; i < N; ++i) {
        Uh[3 * i] = uex(pos[3 * i + 1]);
        fh[3 * i] = fx;
      }
      DV U = up(Uh, "U"), force = up(fh, "f"), a("a", 3 * N), zero("zero", m.nFaces);
      fv::cellTendency(m, zero, U, nu, a, force, Real(0));
      auto ah = down(a);
      double rin = 0, rwall = 0;
      for (int i = 0; i < N; ++i) {
        const double r = std::sqrt(ah[3 * i] * ah[3 * i] + ah[3 * i + 1] * ah[3 * i + 1] +
                                   ah[3 * i + 2] * ah[3 * i + 2]) /
                         fx;
        (wc[i] ? rwall : rin) = std::max(wc[i] ? rwall : rin, r);
      }
      // covolume: exact flux at the face centroids, the face tendency
      std::vector<Real> ufh(m.nFaces, 0.0);
      {
        auto A = down(m.faceCellA);
        auto C = down(m.faceCentroid), Nn = down(m.faceNormal);
        for (int f = 0; f < m.nFaces; ++f)
          ufh[f] = uex(pos[3 * A[f] + 1] + C[3 * f + 1]) * Nn[3 * f];
      }
      fv::CovolumeNS<Real> cv;
      cv.setup(m, nu);
      cv.force = force;
      cv.convScale = 0;
      Kokkos::deep_copy(cv.u, up(ufh, "uf"));
      DV k("k", m.nFaces);
      cv.rhs(cv.u, k);
      auto kh = down(k);
      auto Af = down(m.faceCellA), Bf = down(m.faceCellB);
      double fin = 0, fwall = 0;
      for (int f = 0; f < m.nInterior; ++f) {
        const bool w = wc[Af[f]] || wc[Bf[f]];
        (w ? fwall : fin) = std::max(w ? fwall : fin, std::fabs(kh[f]) / fx);
      }
      // adjointness of the wall-aware constraint pair (prescribed U_wall = 0): the linear part
      // of T (T U − T 0) must be the exact transpose of faceInterpTranspose(..., wallPrescribed)
      double adjW = 0;
      {
        std::mt19937 rng(3);
        std::uniform_real_distribution<Real> Ur(-1, 1);
        std::vector<Real> Uh2(3 * N), gh(m.nFaces);
        for (auto& v : Uh2)
          v = Ur(rng);
        for (auto& v : gh)
          v = Ur(rng);
        DV Uc = up(Uh2, "Uc"), g = up(gh, "g"), tu("tu", m.nFaces), t0("t0", m.nFaces),
           tt("tt", 3 * N), g9("g9", 9 * N), z3("z3", 3 * N), ub("ub", m.nFaces - m.nInterior);
        DV Uw("Uw", 3 * (m.nFaces - m.nInterior));
        fv::vectorGreenGauss(m, Uc, g9, Uw);
        fv::projectToFaces(m, Uc, tu, g9, ub);
        fv::vectorGreenGauss(m, z3, g9, Uw);
        fv::projectToFaces(m, z3, t0, g9, ub);
        fv::faceInterpTranspose(m, g, tt, true, g9, true);
        auto tuh = down(tu), t0h = down(t0), Af = down(m.faceArea), df = down(m.faceDist);
        double lhs = 0;
        for (int f = 0; f < m.nInterior; ++f)
          lhs += Af[f] * df[f] * (tuh[f] - t0h[f]) * gh[f];
        auto th = down(tt), vol = down(m.cellVolume);
        double rhs = 0;
        for (int i = 0; i < N; ++i)
          for (int c = 0; c < 3; ++c)
            rhs += vol[i] * Uh2[3 * i + c] * th[3 * i + c];
        adjW = std::fabs(lhs - rhs) / std::fabs(lhs);
      }
      std::printf(
          "      wall-aware constraint pair adjointness (skew-corrected, U_wall prescribed): "
          "%.1e\n",
          adjW);
      const bool aOk = rin < 1e-10 && fin < 1e-10 && adjW < 1e-12;  // round-off of O(1/h²) sums
      std::printf(
          "  (A) exact parabola, n=%d (%d cells, %d wall faces): Stokes residual/f — "
          "interior cells %.1e, wall-row cells %.3f (two-point wall flux: f/4 expected); "
          "covolume face tendency interior %.1e, wall-adjacent %.3f  %s\n",
          n, N, m.nFaces - m.nInterior, rin, rwall, fin, fwall, aOk ? "OK" : "FAIL");
      if (!aOk)
        bad = 1;
    }

    // ---- (B) steady state -----------------------------------------------------------------------
    {
      const int ns_[3] = {4, 8, 16};
      Real eCo[3], eCv[3], eCoF[3], eCvF[3];
      for (int r = 0; r < 3; ++r) {
        const int n = ns_[r];
        std::vector<Real> pos;
        Real L[3];
        const int N = seeds(n, pos, L);
        auto m = slabMesh(pos, N, L, sdf);
        const Real h = H / n, dt = 0.2 * h * h / nu, T = 0.3;
        const int steps = (int)std::ceil(T / dt);
        std::vector<Real> fh(3 * N, 0.0), exC(3 * N, 0.0), exF(m.nFaces, 0.0);
        for (int i = 0; i < N; ++i) {
          fh[3 * i] = fx;
          exC[3 * i] = uex(pos[3 * i + 1]);
        }
        {
          auto A = down(m.faceCellA);
          auto C = down(m.faceCentroid), Nn = down(m.faceNormal);
          for (int f = 0; f < m.nFaces; ++f)
            exF[f] = uex(pos[3 * A[f] + 1] + C[3 * f + 1]) * Nn[3 * f];
        }
        DV force = up(fh, "f");
        // collocated (default: skew-corrected adjoint pair)
        fv::CollocatedNS<Real> co;
        co.setup(m, nu, true);
        co.force = force;
        co.poisson.tol = 1e-11;
        DV U0("U0", 3 * N);
        co.initialize(U0);
        for (int s = 0; s < steps; ++s)
          co.step(T / steps);
        eCo[r] = relErrV(m, co.U, exC);
        eCoF[r] = relErrF(m, co.uf, exF);
        // covolume
        fv::CovolumeNS<Real> cv;
        cv.setup(m, nu, true);
        cv.force = force;
        cv.poisson.tol = 1e-11;
        Kokkos::deep_copy(cv.u, Real(0));
        for (int s = 0; s < steps; ++s)
          cv.step(T / steps);
        eCvF[r] = relErrF(m, cv.u, exF);
        DV Uc("Uc", 3 * N);
        fv::perotVelocity(m, cv.u, Uc);
        eCv[r] = relErrV(m, Uc, exC);
        // wall flux + divergence
        auto ufh = down(co.uf), cvh = down(cv.u);
        double wmax = 0;
        for (int f = m.nInterior; f < m.nFaces; ++f)
          wmax = std::max({wmax, std::fabs(ufh[f]), std::fabs(cvh[f])});
        std::printf(
            "      n=%2d (%5d cells, %4d steps): collocated cell err %.3e face %.3e | "
            "covolume face err %.3e Perot cell %.3e | max wall flux %.1e, face div %.1e / "
            "%.1e\n",
            n, N, steps, eCo[r], eCoF[r], eCvF[r], eCv[r], wmax, co.maxFaceDivergence(),
            cv.maxDivergence());
      }
      const Real oCo = std::log2(eCo[1] / eCo[2]), oCv = std::log2(eCvF[1] / eCvF[2]);
      const bool bOk = oCo >= 1.9 && oCv >= 1.9;
      std::printf(
          "  (B) Poiseuille order 8->16: collocated cell %.2f (face %.2f), covolume face "
          "%.2f (Perot cell %.2f)  (gated >= 1.9)  %s\n",
          oCo, std::log2(eCoF[1] / eCoF[2]), oCv, std::log2(eCv[1] / eCv[2]), bOk ? "OK" : "FAIL");
      if (!bOk)
        bad = 1;
    }
  }
  std::printf("VORO-BODYFITTED %s\n", bad ? "FAIL" : "OK");
  Kokkos::finalize();
  return bad;
}
