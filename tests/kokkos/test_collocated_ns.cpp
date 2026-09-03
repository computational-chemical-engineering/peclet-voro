/**
 * @file test_collocated_ns.cpp
 * @brief Track C, rung C2b gate (Voronoi methods plan): the collocated solver in peclet.flow's
 * SolverColocated structure (incremental predictor with the constraint-transpose pressure
 * gradient, centre→face constraint, exact face projection, transpose cell correction), its
 * unstructured extension (the skew-corrected constraint pair), and the covolume-vs-collocated
 * comparison the plan asks for.
 *
 *  (A) On a random Voronoi mesh: ⟨T U, g⟩_F = ⟨U, Tᵀ g⟩_V to round-off for the plain AND the
 *      skew-corrected constraint pair (the gauge-exact property); the skew-corrected T returns a
 *      linear field's face-centroid value to round-off while the plain T errs ∝ skewness;
 *      ⟨U, conv(U; u_f)⟩_V = 0 to round-off with the projected flux; face divergence ≤ 1e-10.
 *  (B) Inviscid Taylor–Green on a jittered lattice: energy drift of the plain and the
 *      skew-corrected pair at CFL 0.2 / 0.1 and an h-scan at CFL 0.2 (the approximate cell
 *      correction is not energy-neutral: the drift must vanish with dt AND h); face divergence
 *      ≤ 1e-11.
 *  (C) Viscous Taylor–Green order in h (cell velocity at the seeds, V-norm; face flux, F-norm) on
 *      the cubic lattice (gated ≥ 1.8), 0.2h-jittered lattices and Lloyd CVTs of random seeds,
 *      plain vs skew-corrected, Stokes-only column, and the covolume solver on the SAME meshes.
 */
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/fv/collocated.hpp"
#include "peclet/voro/fv/covolume.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const Real L[3] = {1, 1, 1};
    std::mt19937 rng(17);
    std::uniform_real_distribution<Real> Ur(-1, 1);
    std::printf("=== C2b: collocated Navier–Stokes (flow's approximate projection) ===\n");

    // ---- (A) ---------------------------------------------------------------------------------
    {
      const int N = 3000;
      std::vector<Real> pos(3 * N);
      for (auto& x : pos)
        x = 0.5 * (Ur(rng) + 1);
      auto m = meshOf(pos, N, L);
      auto vol = down(m.cellVolume);
      std::vector<Real> Uh(3 * N);
      for (auto& v : Uh)
        v = Ur(rng);
      // adjointness of (T, Tᵀ), plain and skew-corrected; linear-field exactness
      double adj[2], lin[2];
      {
        std::vector<Real> gh(m.nFaces);
        for (auto& v : gh)
          v = Ur(rng);
        DV g = up(gh, "g"), Uc = up(Uh, "Uc"), tu("tu", m.nFaces), tt("tt", 3 * N), g9("g9", 9 * N);
        std::vector<Real> lh(3 * N), exf(m.nFaces);
        const Real G[9] = {0.3, -1.1, 0.7, 0.2, 0.9, -0.4, -0.8, 0.5, 1.3},
                   b0[3] = {0.1, -0.2, 0.3};
        for (int i = 0; i < N; ++i)
          for (int c = 0; c < 3; ++c)
            lh[3 * i + c] = b0[c] + G[3 * c] * pos[3 * i] + G[3 * c + 1] * pos[3 * i + 1] +
                            G[3 * c + 2] * pos[3 * i + 2];
        // a linear field is not periodic: check only faces whose BOTH cells have no wrapping
        // face at all (their gradients read wrapped neighbours otherwise)
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
          for (int f = 0; f < m.nInterior; ++f)
            inner[f] = !cellWraps[A[f]] && !cellWraps[B[f]];
          for (int f = 0; f < m.nFaces; ++f) {
            const int ia = A[f];
            Real x[3];
            for (int c = 0; c < 3; ++c)
              x[c] = pos[3 * ia + c] + C[3 * f + c];
            Real un = 0;
            for (int c = 0; c < 3; ++c)
              un += (b0[c] + G[3 * c] * x[0] + G[3 * c + 1] * x[1] + G[3 * c + 2] * x[2]) *
                    Nn[3 * f + c];
            exf[f] = un;
          }
        }
        DV lin3 = up(lh, "lin");
        for (int sk2 = 0; sk2 < 2; ++sk2) {
          if (sk2) {
            fv::vectorGreenGauss(m, Uc, g9);
            fv::projectToFaces(m, Uc, tu, g9);
          } else {
            fv::projectToFaces(m, Uc, tu);
          }
          fv::faceInterpTranspose(m, g, tt, sk2 == 1, g9);
          const double lhs = fv::dotFaces(m, tu, g);
          auto th = down(tt);
          double rhs = 0;
          for (int i = 0; i < N; ++i)
            for (int c = 0; c < 3; ++c)
              rhs += vol[i] * Uh[3 * i + c] * th[3 * i + c];
          adj[sk2] = std::fabs(lhs - rhs) / std::fabs(lhs);
          if (sk2) {
            fv::vectorGreenGauss(m, lin3, g9);
            fv::projectToFaces(m, lin3, tu, g9);
          } else {
            fv::projectToFaces(m, lin3, tu);
          }
          auto tuh = down(tu), Af = down(m.faceArea), df = down(m.faceDist);
          double e = 0, nn = 0;
          for (int f = 0; f < m.nInterior; ++f) {
            if (!inner[f])
              continue;
            e += Af[f] * df[f] * (tuh[f] - exf[f]) * (tuh[f] - exf[f]);
            nn += Af[f] * df[f] * exf[f] * exf[f];
          }
          lin[sk2] = std::sqrt(e / nn);
        }
      }
      fv::CollocatedNS<Real> ns;
      ns.setup(m, 0.0);
      ns.poisson.tol = 1e-14;
      ns.initialize(up(Uh, "U0"));
      DV conv("conv", 3 * N);
      fv::cellTendency(m, ns.uf, ns.U, Real(0), conv);
      auto ch = down(conv), Uc2 = down(ns.U);
      double sk = 0, skAbs = 0;
      for (int i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c) {
          sk += vol[i] * Uc2[3 * i + c] * ch[3 * i + c];
          skAbs += std::fabs(vol[i] * Uc2[3 * i + c] * ch[3 * i + c]);
        }
      const double skew = std::fabs(sk) / skAbs, fd = ns.maxFaceDivergence(),
                   cd = ns.maxCellDivergence();
      const bool aOk =
          skew < 1e-11 && fd < 1e-10 && adj[0] < 1e-12 && adj[1] < 1e-12 && lin[1] < 1e-12;
      std::printf(
          "  (A) adjointness <TU,g>_F = <U,T^T g>_V: plain %.1e, skew-corrected %.1e; "
          "linear field at the face centroid: plain err %.2e (skewness %.3f), corrected "
          "%.1e; <U, conv U>_V / Σ|terms| = %.2e; face div %.1e, cell-field div %.2e  %s\n",
          adj[0], adj[1], lin[0], faceSkewness(m), lin[1], skew, fd, cd, aOk ? "OK" : "FAIL");
      if (!aOk)
        bad = 1;
    }

    // ---- (B) inviscid TGV ----------------------------------------------------------------------
    {
      const int n = 16;
      auto pos = lattice(n, 0.2, rng);
      const int N = n * n * n;
      auto m = meshOf(pos, N, L);
      const Real h = Real(1) / n, T = 100 * 0.2 * h;
      Real drift[2][2], maxDiv = 0;
      for (int sk2 = 0; sk2 < 2; ++sk2)
        for (int pass = 0; pass < 2; ++pass) {
          fv::CollocatedNS<Real> ns;
          ns.setup(m, 0.0, true);
          ns.skewCorrected = sk2 == 1;
          ns.poisson.tol = 1e-13;
          ns.initialize(up(tgvCell(pos, N, 0, 0), "U0"));
          const int steps = pass == 0 ? 100 : 200;
          const Real dt = T / steps;
          const Real E0 = ns.kineticEnergy();
          for (int s = 0; s < steps; ++s) {
            ns.step(dt);
            maxDiv = std::max(maxDiv, ns.maxFaceDivergence());
          }
          drift[sk2][pass] = ns.kineticEnergy() / E0 - 1;
        }
      Real driftH[2][3], cdivH[2][3];
      {
        int nn[3] = {8, 16, 32};
        for (int r = 0; r < 3; ++r) {
          auto ph = lattice(nn[r], 0.2, rng);
          const int NN = nn[r] * nn[r] * nn[r];
          auto mh = meshOf(ph, NN, L);
          for (int sk2 = 0; sk2 < 2; ++sk2) {
            fv::CollocatedNS<Real> ns;
            ns.setup(mh, 0.0, true);
            ns.skewCorrected = sk2 == 1;
            ns.poisson.tol = 1e-13;
            ns.initialize(up(tgvCell(ph, NN, 0, 0), "U0"));
            const int steps = (int)std::lround(T / (0.2 / nn[r]));
            const Real dt = T / steps;
            const Real E0 = ns.kineticEnergy();
            cdivH[sk2][r] = 0;
            for (int s = 0; s < steps; ++s) {
              ns.step(dt);
              cdivH[sk2][r] = std::max(cdivH[sk2][r], ns.maxCellDivergence());
            }
            driftH[sk2][r] = ns.kineticEnergy() / E0 - 1;
          }
        }
      }
      std::printf("  (B) inviscid TGV jittered lattices, T=%.3f, E/E0 - 1:\n", T);
      for (int sk2 = 0; sk2 < 2; ++sk2)
        std::printf(
            "      %-14s 16^3 CFL 0.2 %+.2e, CFL 0.1 %+.2e | h-scan CFL 0.2: %+.2e (8^3) "
            "%+.2e (16^3) %+.2e (32^3), h-order %.2f; max cell-field |div| %.2e %.2e %.2e\n",
            sk2 ? "skew-corrected" : "plain", drift[sk2][0], drift[sk2][1], driftH[sk2][0],
            driftH[sk2][1], driftH[sk2][2], std::log2(driftH[sk2][1] / driftH[sk2][2]),
            cdivH[sk2][0], cdivH[sk2][1], cdivH[sk2][2]);
      // MEASURED (2026-09-03): plain −9.7e-4 / −4.9e-4 (O(dt)), h-order 2.04 (O(dt·h²) — flow's
      // approximate-projection analysis); skew-corrected −7.1e-4 / −2.4e-4, h-order 1.69.
      const bool bOk = maxDiv < 1e-11 && std::fabs(drift[1][0]) < 2e-3 &&
                       std::fabs(drift[1][1]) < 0.7 * std::fabs(drift[1][0]) &&
                       std::fabs(driftH[1][2]) < 0.5 * std::fabs(driftH[1][1]);
      std::printf(
          "      max face |div| %.1e; default drift < 2e-3, O(dt) and vanishing with h  %s\n",
          maxDiv, bOk ? "OK" : "FAIL");
      if (!bOk)
        bad = 1;
    }

    // ---- (C) viscous TGV order + comparison --------------------------------------------------
    {
      const Real nu = 0.01, T = 0.25;
      const char* names[3] = {"cubic lattice", "0.2h-jittered", "random + Lloyd(30)"};
      // MEASURED (2026-09-03): see the plan's C2b row; gates = measured orders − margin.
      // MEASURED (2026-09-03): skew-corrected (the default) 1.97 / 2.10 / 2.08 (Stokes-only
      // 2.00 / 1.95 / 1.99); plain 1.97 / 1.72 / 1.17; covolume face 1.98 / 0.82 / 1.29.
      const Real gates[3] = {1.8, 1.8, 1.8};
      std::printf(
          "      %-20s %3s %6s | plain: cell err  face err | skew-corr: cell err  face "
          "err | covolume: face err  cell(Perot) | Stokes-only plain / skew\n",
          "mesh", "n", "skew");
      for (int pass = 0; pass < 3; ++pass) {
        Real eC[2][3], eCf[2][3], eS[2][3], eV[3], eVc[3];
        int ns_[3] = {8, 16, 32};
        for (int r = 0; r < 3; ++r) {
          const int n = ns_[r], N = n * n * n;
          auto pos = lattice(n, pass == 0 ? 0.0 : 0.2, rng);
          if (pass == 2) {
            for (auto& x : pos)
              x = 0.5 * (Ur(rng) + 1);
            lloydRelax(pos, N, L, 30);
          }
          auto m = meshOf(pos, N, L);
          const Real h = Real(1) / n;
          const int steps = (int)std::ceil(T / (0.2 * h));
          const Real dt = T / steps;
          auto exF = tgvFlux(m, pos, T, nu);
          auto exC = tgvCell(pos, N, T, nu);
          for (int sk2 = 0; sk2 < 2; ++sk2)
            for (int stokes = 0; stokes < 2; ++stokes) {
              fv::CollocatedNS<Real> ns;
              ns.setup(m, nu, true);
              ns.skewCorrected = sk2 == 1;
              ns.convScale = stokes ? 0 : 1;
              ns.poisson.tol = 1e-11;
              ns.initialize(up(tgvCell(pos, N, 0, nu), "U0"));
              for (int s = 0; s < steps; ++s)
                ns.step(dt);
              if (stokes) {
                eS[sk2][r] = relErrV(m, ns.U, exC);
              } else {
                eC[sk2][r] = relErrV(m, ns.U, exC);
                eCf[sk2][r] = relErrF(m, ns.uf, exF);
              }
            }
          {
            fv::CovolumeNS<Real> cv;
            cv.setup(m, nu, true);
            cv.poisson.tol = 1e-11;
            Kokkos::deep_copy(cv.u, up(tgvFlux(m, pos, 0, nu), "u0"));
            cv.project(cv.u, 1.0);
            for (int s = 0; s < steps; ++s)
              cv.step(dt);
            eV[r] = relErrF(m, cv.u, exF);
            DV Uc("Uc", 3 * N);
            fv::perotVelocity(m, cv.u, Uc);
            eVc[r] = relErrV(m, Uc, exC);
          }
          std::printf("      %-20s %3d %6.3f | %.3e  %.3e | %.3e  %.3e | %.3e  %.3e | %.3e  %.3e\n",
                      names[pass], n, faceSkewness(m), eC[0][r], eCf[0][r], eC[1][r], eCf[1][r],
                      eV[r], eVc[r], eS[0][r], eS[1][r]);
        }
        auto ord = [](const Real* e) { return std::log2(e[1] / e[2]); };
        const bool cOk = ord(eC[1]) >= gates[pass];  // gates the default (skew-corrected) pair
        std::printf(
            "  (C) %s: plain cell order %.2f (face %.2f, Stokes %.2f); skew-corrected cell "
            "%.2f (face %.2f, Stokes %.2f); covolume face %.2f (Perot cell %.2f)  %s  %s\n",
            names[pass], ord(eC[0]), ord(eCf[0]), ord(eS[0]), ord(eC[1]), ord(eCf[1]), ord(eS[1]),
            ord(eV), ord(eVc), "gated (skew-corrected)", cOk ? "OK" : "FAIL");
        if (!cOk)
          bad = 1;
      }
    }
    // ---- (D) implicit diffusion on the viscous TGV: first order in dt (informational) ----------
    {
      const int n = 16, N = n * n * n;
      auto pos = lattice(n, 0.2, rng);
      auto m = meshOf(pos, N, L);
      const Real nu = 0.01, T = 0.25, h = Real(1) / n;
      auto exC = tgvCell(pos, N, T, nu);
      Real e[3], eRK3;
      int st[3] = {(int)std::ceil(T / (0.2 * h)), (int)std::ceil(T / (0.1 * h)),
                   (int)std::ceil(T / (0.05 * h))};
      {
        fv::CollocatedNS<Real> ns;
        ns.setup(m, nu, true);
        ns.poisson.tol = 1e-11;
        ns.initialize(up(tgvCell(pos, N, 0, nu), "U0"));
        for (int s = 0; s < st[0]; ++s)
          ns.step(T / st[0]);
        eRK3 = relErrV(m, ns.U, exC);
      }
      for (int r = 0; r < 3; ++r) {
        fv::CollocatedNS<Real> ns;
        ns.setup(m, nu, true);
        ns.implicitDiffusion = true;
        ns.poisson.tol = 1e-11;
        ns.initialize(up(tgvCell(pos, N, 0, nu), "U0"));
        for (int s = 0; s < st[r]; ++s)
          ns.step(T / st[r]);
        e[r] = relErrV(m, ns.U, exC);
      }
      // MEASURED (2026-09-03): 2.97e-2 / 2.95e-2 / 2.94e-2 vs RK3 at CFL 0.2 — the spatial error
      // dominates at these steps, the first-order time error is below it (informational).
      std::printf(
          "  (D) implicit-diffusion TGV 16^3 jittered: error %.3e / %.3e / %.3e at CFL 0.2 / "
          "0.1 / 0.05 -> dt-order %.2f, %.2f (backward Euler + explicit convection; the "
          "RK3 scheme at CFL 0.2 gives %.3e — the spatial error dominates)  informational\n",
          e[0], e[1], e[2], std::log2(e[0] / e[1]), std::log2(e[1] / e[2]), eRK3);
    }
  }
  std::printf("VORO-COLLOCATED %s\n", bad ? "FAIL" : "OK");
  Kokkos::finalize();
  return bad;
}
