/**
 * @file test_grid_relax.cpp
 * @brief Track B, rung B1 gate (Voronoi methods plan): a random seed set relaxed under the
 * centroidal (Lloyd) energy of energy/lloyd.hpp reaches the low-skewness, near-equal-volume
 * state the two-point operators of track C need, and the Poisson residual consistency of
 * fv/operators.hpp improves with it (the B4 argument: grid quality IS solver quality).
 *
 * Descent: x ← x − ∂E/∂x / (2 V_i) is exactly Lloyd's step x ← c (the energy's Hessian diagonal
 * is 2V), so the driver is the plain gradient step with that scaling. Measured per iteration:
 * volume coefficient of variation, mean face skewness |c_f − n_f| / d_f (face centroid off the
 * connector midpoint), and — at the start and the end — the Poisson residual consistency of the
 * manufactured solution. Gates: skewness ↓ ≥ 4x, volume CV ↓ ≥ 2x, residual consistency ↓ ≥ 2x
 * over 40 iterations at N = 4000 (measured 2026-09-03 and recorded in the plan).
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

struct Metrics {
  double volCV = 0, skew = 0, resid = 0, E = 0;
};

static Metrics measure(const std::vector<Real>& pos, int N, const Real L[3], DV& force,
                       bool withRes) {
  DV dpos("pos", 3 * N), dw;
  Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(pos.data(), 3 * N));
  Kokkos::View<long*, Mem> gd;
  auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(
      dpos, dw, N, L, 4, N, gd, {}, true, -1, {}, {}, {}, {}, {}, {}, 0, nullptr, {}, false, {}, {},
      0, Real(0), false, /*withMoments=*/true);
  Metrics m;
  Kokkos::deep_copy(force, Real(0));
  m.E = peclet::voro::energy::lloydEnergyForce<Real>(res.view, dpos, Real(1), force);
  auto vol = down(res.view.cellVolume);
  double mean = 0, var = 0;
  for (Real v : vol)
    mean += v;
  mean /= N;
  for (Real v : vol)
    var += (v - mean) * (v - mean);
  m.volCV = std::sqrt(var / N) / mean;
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
  m.skew = sk / fm.nInterior;
  if (withRes) {
    std::vector<Real> ph(N), ex(N);
    const Real w = 2 * M_PI;
    for (int i = 0; i < N; ++i) {
      const Real x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
      ph[i] = std::sin(w * x) * std::sin(w * y) * std::sin(w * z);
      ex[i] = -3 * w * w * ph[i];
    }
    DV p("p", N), Lp("Lp", N), sf;
    Kokkos::deep_copy(p, Kokkos::View<const Real*, Kokkos::HostSpace>(ph.data(), N));
    fv::laplacian(fm, p, Lp, sf);
    auto Lph = down(Lp);
    double e2 = 0, s2 = 0;
    for (int i = 0; i < N; ++i) {
      e2 += vol[i] * (Lph[i] - ex[i]) * (Lph[i] - ex[i]);
      s2 += vol[i] * ex[i] * ex[i];
    }
    m.resid = std::sqrt(e2 / s2);
  }
  return m;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 4000, iters = 40;
    const Real L[3] = {1, 1, 1};
    std::mt19937 rng(17);
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> pos(3 * N);
    for (auto& x : pos)
      x = U(rng);
    DV force("force", 3 * N);
    std::printf("=== B1: centroidal relaxation of a random Voronoi grid (N=%d) ===\n", N);
    std::printf("%5s %12s %10s %10s %10s\n", "iter", "E_lloyd", "volCV", "skewness", "residual");
    Metrics m0 = measure(pos, N, L, force, true), m = m0;
    std::printf("%5d %12.5e %10.4f %10.4f %10.4f\n", 0, m.E, m.volCV, m.skew, m.resid);
    // per-cell volumes for the 1/(2V) scaling: recompute from the force's own build (cheap: the
    // measure() call already built; reuse by rebuilding once more here would double the cost —
    // instead scale by the mean volume, a slightly damped Lloyd step)
    for (int it = 1; it <= iters; ++it) {
      auto f = down(force);
      const double Vm = 1.0 / N;
      for (int i = 0; i < 3 * N; ++i) {
        pos[i] -= f[i] / (2.0 * Vm);
        pos[i] -= L[i % 3] * std::floor(pos[i] / L[i % 3]);
      }
      m = measure(pos, N, L, force, it == iters || it % 10 == 0);
      if (it % 10 == 0 || it == iters)
        std::printf("%5d %12.5e %10.4f %10.4f %10.4f\n", it, m.E, m.volCV, m.skew, m.resid);
    }
    const bool ok =
        m.skew * 4 <= m0.skew && m.volCV * 2 <= m0.volCV && m.resid * 2 <= m0.resid && m.E < m0.E;
    std::printf(
        "  skewness %.4f -> %.4f, volCV %.4f -> %.4f, residual consistency %.4f -> %.4f  %s\n",
        m0.skew, m.skew, m0.volCV, m.volCV, m0.resid, m.resid, ok ? "OK" : "FAIL");
    if (!ok)
      bad = 1;
  }
  Kokkos::finalize();
  std::printf(bad ? "VORO-GRID FAIL\n" : "VORO-GRID OK\n");
  return bad;
}
