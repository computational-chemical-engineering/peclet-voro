/**
 * @file bench_construct.cpp
 * \brief Isolate the "build a cell from a set of candidate planes" core (B + G), no gather.
 *
 * This is the *always-relevant* part of the engine (cold build, repair, re-clip all run it),
 * and the F2/F3 number that matters. We cache, per cell, its closest-K candidate planes once
 * (the neighbour query, amortised), then time the pure construction `initBox → clip* →
 * geometry(tier)` from the cached list at each geometry tier:
 *   G0 = topology only (vertices are a free byproduct)
 *   G1 = + cell volume
 *   G2 = + per-facet area vector, dV (volume gradient), connecting vector  [physics]
 * Reports kcells/s per tier (FP64 + FP32 via CC_FLOAT). Run: ./bench_construct [N]
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"

#ifdef CC_FLOAT
using real_t = float;
#else
using real_t = double;
#endif
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
static constexpr int KCAND = 64;  // cached closest candidates per cell
static constexpr int CC_MAXP = 64, CC_MAXT = 112;

int main(int argc, char** argv) {
  using MemSpace = tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  Kokkos::initialize(argc, argv);
  {
    std::printf("backend: %s  (%s)\n", Kokkos::DefaultExecutionSpace::name(),
                sizeof(real_t) == 4 ? "FP32" : "FP64");
    const int N = argc > 1 ? std::atoi(argv[1]) : 1000000;
    const real_t L = 1.0, spacing = std::cbrt(1.0 / N);

    // host: random points, build closest-K candidate planes per cell by brute force over a
    // grid block (one-time; this stands in for the neighbour query, which we are amortising).
    std::mt19937 rng(123 + N);
    std::uniform_real_distribution<real_t> U(0, 1);
    std::vector<real_t> pos(3 * N);
    for (auto& x : pos) x = U(rng);
    // simple grid
    const int dim = std::max(1, (int)std::floor(1.0 / spacing));
    const real_t csz = 1.0 / dim;
    std::vector<std::vector<int>> grid(dim * dim * dim);
    auto gid = [&](int x, int y, int z) { return (x + dim) % dim + dim * (((y + dim) % dim) + dim * ((z + dim) % dim)); };
    for (int i = 0; i < N; ++i) {
      int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz), gz = (int)(pos[3 * i + 2] / csz);
      grid[gid(gx, gy, gz)].push_back(i);
    }
    // candidate planes (rel x,y,z), KCAND per cell, padded with a far sentinel
    Kokkos::View<real_t*, MemSpace> cand(Kokkos::view_alloc("cand", Kokkos::WithoutInitializing),
                                         (size_t)N * KCAND * 3);
    Kokkos::View<int*, MemSpace> ncand("ncand", N);
    {
      auto hc = Kokkos::create_mirror_view(cand);
      auto hn = Kokkos::create_mirror_view(ncand);
      const int sw = 3;
      std::vector<std::array<real_t, 4>> tmp;
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz), gz = (int)(pos[3 * i + 2] / csz);
        tmp.clear();
        for (int dz = -sw; dz <= sw; ++dz)
          for (int dy = -sw; dy <= sw; ++dy)
            for (int dx = -sw; dx <= sw; ++dx)
              for (int j : grid[gid(gx + dx, gy + dy, gz + dz)]) {
                if (j == i) continue;
                real_t rx = pos[3 * j] - pos[3 * i], ry = pos[3 * j + 1] - pos[3 * i + 1],
                       rz = pos[3 * j + 2] - pos[3 * i + 2];
                rx -= std::round(rx);
                ry -= std::round(ry);
                rz -= std::round(rz);
                tmp.push_back({rx, ry, rz, real_t(0.5) * (rx * rx + ry * ry + rz * rz)});
              }
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a[3] < b[3]; });
        int k = std::min((int)tmp.size(), KCAND);
        hn(i) = k;
        for (int t = 0; t < k; ++t)
          for (int c = 0; c < 3; ++c) hc((size_t)(i * KCAND + t) * 3 + c) = tmp[t][c];
      }
      Kokkos::deep_copy(cand, hc);
      Kokkos::deep_copy(ncand, hn);
    }

    Kokkos::View<real_t*, MemSpace> outVol("vol", N);
    Kokkos::View<real_t*, MemSpace> outArea("area", N);  // accumulate to defeat dead-code elim

    auto run = [&](int tier) {
      auto kern = KOKKOS_LAMBDA(const int i) {
        vor::device::ConvexCell<real_t, CC_MAXP, CC_MAXT> c;
        c.initBox(L, L, L);
        const int k = ncand(i);
        for (int t = 0; t < k; ++t) {
          const real_t* r = &cand((size_t)(i * KCAND + t) * 3);
          const real_t off = real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
          if (off >= real_t(2) * c.maxVertexRsq()) continue;  // per-candidate cull
          const real_t n[3] = {r[0], r[1], r[2]};
          c.clip(n, off, t);
          if (c.overflow) break;
        }
        if (tier >= 1) outVol(i) = c.overflow ? real_t(0) : c.volume();
        if (tier >= 2) {
          real_t asum = 0;
          for (int kk = 0; kk < c.np; ++kk) {
            if (c.pnbr[kk] < 0) continue;
            real_t a[3], d[3], cv[3];
            if (c.facetGeometry(kk, a, d, cv))
              asum += a[0] + d[0] + cv[0];  // touch all outputs
          }
          outArea(i) = asum;
        }
      };
      Kokkos::parallel_for("warm", Kokkos::RangePolicy<Exec>(0, N), kern);
      Kokkos::fence();
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        auto t0 = clk::now();
        Kokkos::parallel_for("construct", Kokkos::RangePolicy<Exec>(0, N), kern);
        Kokkos::fence();
        auto t1 = clk::now();
        best = std::min(best, secs(t0, t1));
      }
      return N / best / 1e3;
    };
    const double g0 = run(0), g1 = run(1), g2 = run(2);
    std::printf("N=%d  construct-from-cache (no gather), kcells/s:\n", N);
    std::printf("  G0 topology only   : %8.1f\n", g0);
    std::printf("  G1 + volume        : %8.1f  (+volume %.0f%%)\n", g1, 100 * (1 / g1 - 1 / g0) / (1 / g1));
    std::printf("  G2 + derivatives   : %8.1f  (+deriv  %.0f%%)\n", g2, 100 * (1 / g2 - 1 / g1) / (1 / g2));
  }
  Kokkos::finalize();
  return 0;
}
