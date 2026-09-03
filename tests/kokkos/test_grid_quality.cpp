/**
 * @file test_grid_quality.cpp
 * @brief Track B, rung B4 gate (Voronoi methods plan): grid quality IS solver quality. The
 * Poisson problem of C1 (−L p = f, manufactured p = sin 2πx sin 2πy sin 2πz) on grids of
 * decreasing skewness at FIXED h, and its convergence in h on Lloyd-relaxed grids.
 *
 *  (A) fixed N = 16³: jittered lattice (0.3 h) relaxed by 0 / 3 / 10 / 30 Lloyd steps — the
 *      cellwise residual consistency and the solution error are reported against the mean
 *      skewness; gate: the residual error falls monotonically with skewness and ends >= 2x below
 *      the start (measured 2.5x). The solution error at fixed h is h-limited (3.0e-2 -> 2.9e-2,
 *      not monotone) — skewness limits flux consistency, not the pressure solve.
 *  (B) convergence on relaxed grids: 8³ / 16³ / 32³ jittered lattices, 10 Lloyd steps each —
 *      solution order ≥ 1.8 (gate).
 */
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/energy/lloyd.hpp"
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

struct Q {
  double skew = 0, resid = 0, sol = 0;
};

// Lloyd-relax `pos` in place for `steps` steps (x <- centroid), then measure.
static Q relaxAndMeasure(std::vector<Real>& pos, int N, const Real L[3], int steps) {
  DV force("f", 3 * N), dw;
  Kokkos::View<long*, Mem> gd;
  for (int it = 0; it <= steps; ++it) {
    DV dpos("pos", 3 * N);
    Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(pos.data(), 3 * N));
    auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(dpos, dw, N, L, 4,
                                                                                 N, gd, {}, true);
    if (it == steps) {
      Q q;
      auto aux = peclet::voro::buildAuxMaps(res.view);
      auto fm = fv::buildFaceMesh(res.view, aux);
      auto cen = down(fm.faceCentroid), conn = down(fm.faceConn), dist = down(fm.faceDist);
      double sk = 0;
      for (int f = 0; f < fm.nInterior; ++f) {
        double s2 = 0;
        for (int c = 0; c < 3; ++c) {
          const double d = cen[3 * f + c] - 0.5 * conn[3 * f + c];
          s2 += d * d;
        }
        sk += std::sqrt(s2) / dist[f];
      }
      q.skew = sk / fm.nInterior;
      std::vector<Real> ph(N), ex(N), nf(N);
      const Real w = 2 * M_PI;
      for (int i = 0; i < N; ++i) {
        ph[i] =
            std::sin(w * pos[3 * i]) * std::sin(w * pos[3 * i + 1]) * std::sin(w * pos[3 * i + 2]);
        ex[i] = -3 * w * w * ph[i];
        nf[i] = -ex[i];
      }
      DV p("p", N), Lp("Lp", N), sf, f("fsrc", N), ps("ps", N);
      Kokkos::deep_copy(p, Kokkos::View<const Real*, Kokkos::HostSpace>(ph.data(), N));
      Kokkos::deep_copy(f, Kokkos::View<const Real*, Kokkos::HostSpace>(nf.data(), N));
      fv::laplacian(fm, p, Lp, sf);
      Real rr = 0;
      fv::poissonCG(fm, f, ps, Real(1e-10), 5000, rr);
      auto Lph = down(Lp), psh = down(ps), vol = down(fm.cellVolume);
      double e2 = 0, s2 = 0, pe2 = 0, ps2 = 0, pm = 0;
      for (int i = 0; i < N; ++i)
        pm += vol[i] * ph[i];
      for (int i = 0; i < N; ++i) {
        e2 += vol[i] * (Lph[i] - ex[i]) * (Lph[i] - ex[i]);
        s2 += vol[i] * ex[i] * ex[i];
        const double pex = ph[i] - pm;
        pe2 += vol[i] * (psh[i] - pex) * (psh[i] - pex);
        ps2 += vol[i] * pex * pex;
      }
      q.resid = std::sqrt(e2 / s2);
      q.sol = std::sqrt(pe2 / ps2);
      return q;
    }
    // Lloyd step through the energy gradient: x <- x - g/(2V)
    Kokkos::deep_copy(force, Real(0));
    peclet::voro::energy::lloydEnergyForce<Real>(res.view, dpos, Real(1), force);
    auto g = down(force), vol = down(res.view.cellVolume);
    for (int i = 0; i < N; ++i)
      for (int c = 0; c < 3; ++c) {
        pos[3 * i + c] -= g[3 * i + c] / (2 * vol[i]);
        pos[3 * i + c] -= L[c] * std::floor(pos[3 * i + c] / L[c]);
      }
  }
  return Q{};
}

static std::vector<Real> jittered(int n, Real jit, std::mt19937& rng) {
  std::uniform_real_distribution<Real> U(-1, 1);
  std::vector<Real> pos(3 * n * n * n);
  int k = 0;
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      for (int c = 0; c < n; ++c) {
        pos[3 * k] = (a + 0.5 + jit * U(rng)) / n;
        pos[3 * k + 1] = (b + 0.5 + jit * U(rng)) / n;
        pos[3 * k + 2] = (c + 0.5 + jit * U(rng)) / n;
        ++k;
      }
  return pos;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const Real L[3] = {1, 1, 1};
    std::mt19937 rng(31);
    std::printf("=== B4: grid quality vs solver accuracy ===\n");
    // (A) fixed h, decreasing skewness
    const int steps[4] = {0, 3, 10, 30};
    Q qa[4];
    std::printf("  (A) n=16 jitter 0.3h: %6s %10s %12s %12s\n", "lloyd", "skewness", "residual",
                "solution");
    for (int r = 0; r < 4; ++r) {
      std::mt19937 rr(31);
      auto pos = jittered(16, 0.3, rr);
      qa[r] = relaxAndMeasure(pos, 16 * 16 * 16, L, steps[r]);
      std::printf("                          %6d %10.4f %12.4e %12.4e\n", steps[r], qa[r].skew,
                  qa[r].resid, qa[r].sol);
    }
    // MEASURED (2026-09-03): the residual consistency tracks the skewness (0.106 -> 0.043 as the
    // skewness falls 0.121 -> 0.041), but the Poisson SOLUTION error at fixed h barely moves
    // (3.0e-2 -> 2.9e-2, not monotone): the two-point scheme's solution error is h-limited
    // (second order by supra-convergence); skewness limits the FLUX / gradient consistency —
    // what transport and the collocated gradient feel, not the pressure solve. Gate: the
    // residual falls monotonically and by >= 2x; the solution error is reported.
    bool aOk = qa[3].resid * 2 <= qa[0].resid;
    for (int r = 1; r < 4; ++r)
      aOk = aOk && qa[r].skew <= qa[r - 1].skew && qa[r].resid <= qa[r - 1].resid * 1.05;
    std::printf(
        "  (A) residual follows skewness (%.4f -> %.4f as skewness %.3f -> %.3f); solution "
        "%.2e -> %.2e (h-limited, informational)  %s\n",
        qa[0].resid, qa[3].resid, qa[0].skew, qa[3].skew, qa[0].sol, qa[3].sol,
        aOk ? "OK" : "FAIL");
    if (!aOk)
      bad = 1;
    // (B) convergence on relaxed grids
    const int ns[3] = {8, 16, 32};
    double eS[3], eR[3];
    for (int r = 0; r < 3; ++r) {
      auto pos = jittered(ns[r], 0.3, rng);
      Q q = relaxAndMeasure(pos, ns[r] * ns[r] * ns[r], L, 10);
      eS[r] = q.sol;
      eR[r] = q.resid;
      std::printf("  (B) n=%2d (10 Lloyd steps): skewness %.4f residual %.3e solution %.3e\n",
                  ns[r], q.skew, q.resid, q.sol);
    }
    const double oS = std::log(eS[0] / eS[2]) / std::log(4.0),
                 oR = std::log(eR[0] / eR[2]) / std::log(4.0);
    const bool bOk = oS >= 1.8;
    std::printf(
        "  (B) order in h on relaxed grids — solution %.2f (gated ≥ 1.8), residual %.2f  %s\n", oS,
        oR, bOk ? "OK" : "FAIL");
    if (!bOk)
      bad = 1;
  }
  Kokkos::finalize();
  std::printf(bad ? "VORO-GRIDQ FAIL\n" : "VORO-GRIDQ OK\n");
  return bad;
}
