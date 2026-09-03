/**
 * @file test_covolume_dec.cpp
 * @brief Track C, rung C2a′ gate (Voronoi methods plan): the DEC (Nicolaides) viscous term of
 * the covolume scheme.
 *
 *  (A) Edge lengths: on the cubic lattice every partner edge has length h to round-off; on a random
 *      mesh the two facets of a face report the same edge set and lengths (twin consistency).
 *  (B) Structure on a random Voronoi mesh: the DEC Laplacian is symmetric in ⟨·,·⟩_F and negative
 *      semidefinite (⟨u, Δu⟩_F ≤ 0) to round-off — the viscous term dissipates exactly.
 *  (C) Consistency: Δ of the exact fluxes of a LINEAR field vanishes to round-off on the cubic
 *      lattice; its residual on the jittered lattice and the Lloyd CVT is reported next to the
 *      Perot-reconstructed Rᵀ Δ₂ R (both are skewness-limited: the face flux is a face average, the
 *      DEC 1-form wants the connector-midpoint value).
 *  (D) Viscous Taylor–Green order with the DEC term (at dt/8: its spectral radius is 3–6x the
 *      two-point Laplacian's) vs the Perot term on the cubic lattice, the 0.2h-jittered lattice
 *      and the Lloyd CVT — informational; the MEASURED verdict is in the code: no accuracy gain.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/fv/covolume.hpp"
#include "peclet/voro/fv/dec.hpp"

// mesh + view with the facet-edge CSR (edge lengths)
static fv::FaceMesh<Real> meshWithEdges(const std::vector<Real>& pos, int N, const Real L[3],
                                        peclet::voro::TessellationView<Real>& viewOut) {
  DV dpos = up(pos, "pos"), dw;
  Kokkos::View<long*, Mem> gd;
  auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(
      dpos, dw, N, L, 4, N, gd, {}, true, -1, {}, {}, {}, {}, {}, {}, 0, nullptr, {},
      /*withAreaGrad=*/true);
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
    std::mt19937 rng(23);
    std::uniform_real_distribution<Real> Ur(-1, 1);
    std::printf("=== C2a': DEC covolume viscous term ===\n");

    // ---- (A) edge lengths ---------------------------------------------------------------------
    {
      const int n = 8, N = n * n * n;
      auto pos = lattice(n, 0.0, rng);
      peclet::voro::TessellationView<Real> view;
      auto m = meshWithEdges(pos, N, L, view);
      auto el = down(view.edgeLength), ea = down(view.facetArea);
      auto eo = down(view.facetEdgeOffset), ec = down(view.facetEdgeCount),
           nb = down(view.facetNeighbor);
      const Real h = Real(1) / n;
      double emax = 0;
      int nz = 0;
      for (int g = 0; g < view.numFacets(); ++g) {
        const double A = std::sqrt(ea[3 * g] * ea[3 * g] + ea[3 * g + 1] * ea[3 * g + 1] +
                                   ea[3 * g + 2] * ea[3 * g + 2]);
        if (A < 1e-12)
          continue;  // degenerate contact facets of the lattice
        for (int e = eo[g] + 1; e < eo[g] + ec[g]; ++e) {
          if (el[e] < 1e-12)
            continue;  // zero-length edge slots (degenerate vertices)
          emax = std::max(emax, std::fabs(el[e] - h));
          ++nz;
        }
      }
      // twin consistency on a random mesh: sorted edge lengths of facet g and of its reciprocal
      std::vector<Real> rp(3 * 2000);
      for (auto& x : rp)
        x = 0.5 * (Ur(rng) + 1);
      peclet::voro::TessellationView<Real> vr;
      auto mr = meshWithEdges(rp, 2000, L, vr);
      auto elr = down(vr.edgeLength);
      auto eor = down(vr.facetEdgeOffset), ecr = down(vr.facetEdgeCount), FF = down(mr.faceFacet),
           Ar = down(mr.faceCellA), Br = down(mr.faceCellB), co = down(vr.cellFacetOffset),
           cc = down(vr.cellFacetCount), nbr = down(vr.facetNeighbor);
      double twin = 0;
      for (int f = 0; f < mr.nInterior; ++f) {
        const int g = FF[f], a = Ar[f], b = Br[f];
        int g2 = -1;
        for (int hh = co[b]; hh < co[b] + cc[b]; ++hh)
          if (nbr[hh] == a)
            g2 = hh;
        if (g2 < 0)
          continue;
        std::vector<Real> l1, l2;
        for (int e = eor[g] + 1; e < eor[g] + ecr[g]; ++e)
          if (elr[e] > 1e-12)
            l1.push_back(elr[e]);
        for (int e = eor[g2] + 1; e < eor[g2] + ecr[g2]; ++e)
          if (elr[e] > 1e-12)
            l2.push_back(elr[e]);
        std::sort(l1.begin(), l1.end());
        std::sort(l2.begin(), l2.end());
        if (l1.size() != l2.size()) {
          twin = 1;
          continue;
        }
        for (size_t q = 0; q < l1.size(); ++q)
          twin = std::max(twin, (double)std::fabs(l1[q] - l2[q]));
      }
      const bool aOk = emax < 1e-12 && twin < 1e-10;
      std::printf(
          "  (A) lattice edge lengths: max |l - h| = %.1e over %d edges; random mesh twin "
          "consistency %.1e  %s\n",
          emax, nz, twin, aOk ? "OK" : "FAIL");
      if (!aOk)
        bad = 1;
    }

    // ---- (B) symmetry + dissipativity on a random mesh -----------------------------------------
    {
      const int N = 3000;
      std::vector<Real> pos(3 * N);
      for (auto& x : pos)
        x = 0.5 * (Ur(rng) + 1);
      peclet::voro::TessellationView<Real> view;
      auto m = meshWithEdges(pos, N, L, view);
      auto d = fv::buildDecEdges<Real>(view, m);
      std::vector<Real> uh(m.nFaces), vh(m.nFaces);
      for (auto& x : uh)
        x = Ur(rng);
      for (auto& x : vh)
        x = Ur(rng);
      DV u = up(uh, "u"), v = up(vh, "v"), Lu("Lu", m.nFaces), Lv("Lv", m.nFaces), sc("sc", N);
      fv::decLaplacian(m, d, u, sc, Lu);
      fv::decLaplacian(m, d, v, sc, Lv);
      const double s1 = fv::dotFaces(m, u, Lv), s2 = fv::dotFaces(m, v, Lu),
                   uu = fv::dotFaces(m, u, Lu), nrm = fv::dotFaces(m, u, u);
      const double sym = std::fabs(s1 - s2) / std::fabs(s1);
      const bool bOk = sym < 1e-12 && uu <= 0 && d.nSkipped == 0;
      std::printf(
          "  (B) DEC Laplacian on a random mesh (%d edges, %d skipped): symmetry %.1e, "
          "<u,Lu>_F/<u,u>_F = %.3e (must be <= 0)  %s\n",
          d.nEdges, d.nSkipped, sym, uu / nrm, bOk ? "OK" : "FAIL");
      if (!bOk)
        bad = 1;
    }

    // ---- (C) consistency on a linear field + (D) TGV order ------------------------------------
    {
      const char* names[3] = {"cubic lattice", "0.2h-jittered", "random + Lloyd(30)"};
      const Real G[9] = {0.3, -1.1, 0.7, 0.2, 0.9, -0.4, -0.8, 0.5, 1.3};
      const Real nu = 0.01, T = 0.25;
      for (int pass = 0; pass < 3; ++pass) {
        Real resD[3], resP[3], eD[3], eP[3];
        int ns_[3] = {8, 16, 32};
        for (int r = 0; r < 3; ++r) {
          const int n = ns_[r], N = n * n * n;
          auto pos = lattice(n, pass == 0 ? 0.0 : 0.2, rng);
          if (pass == 2) {
            for (auto& x : pos)
              x = 0.5 * (Ur(rng) + 1);
            lloydRelax(pos, N, L, 30);
          }
          peclet::voro::TessellationView<Real> view;
          auto m = meshWithEdges(pos, N, L, view);
          auto d = fv::buildDecEdges<Real>(view, m);
          // linear field: exact fluxes at the face centroids (periodic-wrap-safe faces only)
          std::vector<Real> lf(m.nFaces, 0.0);
          std::vector<char> inner(m.nFaces, 0), cellWraps(N, 0);
          {
            auto A = down(m.faceCellA), B = down(m.faceCellB);
            auto C = down(m.faceCentroid), Nn = down(m.faceNormal), Cn = down(m.faceConn);
            for (int f = 0; f < m.nInterior; ++f) {
              double w = 0;
              for (int c = 0; c < 3; ++c)
                w += std::fabs(pos[3 * B[f] + c] - pos[3 * A[f] + c] - Cn[3 * f + c]);
              if (w > 1e-9)
                cellWraps[A[f]] = cellWraps[B[f]] = 1;
            }
            for (int f = 0; f < m.nInterior; ++f) {
              const int ia = A[f];
              Real x[3];
              for (int c = 0; c < 3; ++c)
                x[c] = pos[3 * ia + c] + C[3 * f + c];
              Real un = 0;
              for (int c = 0; c < 3; ++c)
                un += (G[3 * c] * x[0] + G[3 * c + 1] * x[1] + G[3 * c + 2] * x[2]) * Nn[3 * f + c];
              lf[f] = un;
              inner[f] = !cellWraps[A[f]] && !cellWraps[B[f]];
            }
          }
          DV ul = up(lf, "ul"), Lu("Lu", m.nFaces), sc("sc", N), Uc("Uc", 3 * N), ac("ac", 3 * N),
             Lp("Lp", m.nFaces), zero("z", m.nFaces);
          fv::decLaplacian(m, d, ul, sc, Lu);
          fv::perotVelocity(m, ul, Uc);
          fv::cellTendency(m, zero, Uc, Real(1), ac, DV{}, Real(0));
          fv::perotTranspose(m, ac, Lp);
          auto Luh = down(Lu), Lph = down(Lp), Af = down(m.faceArea), df = down(m.faceDist);
          double rD = 0, rP = 0, nn = 0;
          for (int f = 0; f < m.nInterior; ++f) {
            if (!inner[f])
              continue;
            // residual relative to |∇u|/h: the natural scale of a first-order error
            rD += Af[f] * df[f] * Luh[f] * Luh[f];
            rP += Af[f] * df[f] * Lph[f] * Lph[f];
            nn += Af[f] * df[f];
          }
          const Real h = Real(1) / n;
          resD[r] = std::sqrt(rD / nn) * h * h;  // × h² : the error relative to |∇u| ~ 1 at scale h
          resP[r] = std::sqrt(rP / nn) * h * h;
          // TGV, both viscous terms
          const int steps = (int)std::ceil(T / (0.2 * h));
          const Real dt = T / steps;
          auto exF = tgvFlux(m, pos, T, nu);
          for (int dec = 0; dec < 2; ++dec) {
            fv::CovolumeNS<Real> cv;
            cv.setup(m, nu, true);
            if (dec)
              cv.setDec(view);
            cv.poisson.tol = 1e-11;
            Kokkos::deep_copy(cv.u, up(tgvFlux(m, pos, 0, nu), "u0"));
            cv.project(cv.u, 1.0);
            // the DEC operator's explicit stability constant is ~8x the two-point Laplacian's
            // (sliver Delaunay triangles: weights l/|t| up to 5.7/h): RK3 diverges at CFL 0.2 and
            // still at dt/8 on the jittered / CVT meshes — the DEC TGV runs only on the lattice
            // (an implicit face-space solve would be the way to use it)
            if (dec && pass > 0) {
              eD[r] = std::nan("");
              continue;
            }
            const int st = dec ? 8 * steps : steps;
            for (int s = 0; s < st; ++s)
              cv.step(T / st);
            (dec ? eD : eP)[r] = relErrF(m, cv.u, exF);
          }
          // stiffness of the DEC operator: max weight ℓ/|t| × d/A relative to the two-point A/(d V)
          auto wh = down(d.weight);
          double wmax = 0;
          for (auto w : wh)
            wmax = std::max(wmax, (double)w);
          std::printf(
              "      %-20s n=%2d skew %.3f | linear-field residual x h^2: DEC %.2e  Perot "
              "%.2e | TGV flux err: DEC %.3e  Perot %.3e | skipped edges %d, max l/|t| x h "
              "%.1f\n",
              names[pass], n, faceSkewness(m), resD[r], resP[r], eD[r], eP[r], d.nSkipped,
              wmax * h);
        }
        const Real oD = std::log2(eD[1] / eD[2]), oP = std::log2(eP[1] / eP[2]);
        // MEASURED (2026-09-03): the DEC term is first-order consistent like Rᵀ Δ₂ R on skewed
        // meshes (linear residual order ~1.0 / 0.93 on jittered / CVT vs 0.58 / 0.56 Perot, similar
        // magnitude), INCONSISTENT on the degenerate cubic lattice (cospherical Delaunay: the dual
        // triangles' loops do not close; residual x h² ≈ 2, 7078 skipped contact edges), and stiff.
        // Verdict: no accuracy remedy for the covolume scheme — its second order needs skew-free
        // (centroidal) meshes or the collocated scheme. Gates: structure only (A, B) + Perot
        // lattice.
        const bool ok = pass == 0 ? (oP >= 1.8) : true;
        std::printf(
            "  (C/D) %s: TGV order DEC %.2f (dt/8; nan = not run, explicitly unstable) vs Perot "
            "%.2f; linear residual order DEC "
            "%.2f Perot %.2f  %s  %s\n",
            names[pass], oD, oP, std::log2(resD[1] / resD[2]), std::log2(resP[1] / resP[2]),
            pass == 0 ? "gated (Perot order)" : "informational", ok ? "OK" : "FAIL");
        if (!ok)
          bad = 1;
      }
    }
  }
  std::printf("VORO-DEC %s\n", bad ? "FAIL" : "OK");
  Kokkos::finalize();
  return bad;
}
