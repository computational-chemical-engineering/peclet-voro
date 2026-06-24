/**
 * Head-to-head: geogram VBW::ConvexCell (the SOTA cell, Ray et al. TOG 2018) vs our
 * vor::device::ConvexCell, SAME workload, SAME machine. Construct-from-cache, FP64, CPU
 * (single-thread + OpenMP). Mirrors our bench_construct: N random points in a unit periodic
 * box, K nearest neighbours per point (sorted, min-imaged), clip the cell by all K bisectors,
 * compute the cell volume. Validates Σvol == box volume for both. Reports Mcells/s.
 *
 * Build: see build.sh (g++ -O3 -fopenmp, geogram standalone + our header via a Kokkos shim).
 */
#include <geogram/voronoi/convex_cell.h>      // VBW::ConvexCell (STANDALONE_CONVEX_CELL)
#include "vorflow/device/convex_cell.hpp"      // vor::device::ConvexCell (via Kokkos shim)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using real_t = double;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
static constexpr int KCAND = 64;

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 1000000;
  const real_t spacing = std::cbrt(1.0 / N);
  int nthreads = 1;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif
  std::printf("N=%d  K=%d  threads=%d  (FP64, CPU)\n", N, KCAND, nthreads);

  // --- points + closest-K candidate planes per cell (grid brute force; one-time) ---
  std::mt19937 rng(123 + N);
  std::uniform_real_distribution<real_t> U(0, 1);
  std::vector<real_t> pos(3 * N);
  for (auto& x : pos) x = U(rng);
  const int dim = std::max(1, (int)std::floor(1.0 / spacing));
  const real_t csz = 1.0 / dim;
  std::vector<std::vector<int>> grid(dim * dim * dim);
  auto gid = [&](int x, int y, int z) {
    return (x + dim) % dim + dim * (((y + dim) % dim) + dim * ((z + dim) % dim));
  };
  for (int i = 0; i < N; ++i) {
    int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz), gz = (int)(pos[3 * i + 2] / csz);
    grid[gid(gx, gy, gz)].push_back(i);
  }
  std::vector<real_t> cand((size_t)N * KCAND * 3);
  std::vector<int> ncand(N);
  {
    const int sw = 3;
#pragma omp parallel
    {
      std::vector<std::array<real_t, 4>> tmp;
#pragma omp for schedule(dynamic, 256)
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
                rx -= std::round(rx); ry -= std::round(ry); rz -= std::round(rz);
                tmp.push_back({rx, ry, rz, real_t(0.5) * (rx * rx + ry * ry + rz * rz)});
              }
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a[3] < b[3]; });
        int k = std::min((int)tmp.size(), KCAND);
        ncand[i] = k;
        for (int t = 0; t < k; ++t)
          for (int c = 0; c < 3; ++c) cand[(size_t)(i * KCAND + t) * 3 + c] = tmp[t][c];
      }
    }
  }

  auto bench = [&](const char* tag, auto build_one) {
    std::vector<double> vols(N);
    // warm
#pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) vols[i] = build_one(i);
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      auto t0 = clk::now();
#pragma omp parallel for schedule(dynamic, 256)
      for (int i = 0; i < N; ++i) vols[i] = build_one(i);
      auto t1 = clk::now();
      best = std::min(best, secs(t0, t1));
    }
    double vsum = 0;
    for (int i = 0; i < N; ++i) vsum += vols[i];
    std::printf("  %-26s %8.1f kcells/s   Σvol/box err=%.2e\n", tag, N / best / 1e3,
                std::fabs(vsum - 1.0));
    return N / best / 1e3;
  };

  // --- geogram VBW::ConvexCell (one reused object per thread) ---
  std::vector<VBW::ConvexCell*> geocells(nthreads, nullptr);
  auto geo_one = [&](int i) -> double {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    if (!geocells[tid]) geocells[tid] = new VBW::ConvexCell();
    VBW::ConvexCell& C = *geocells[tid];
    C.init_with_box(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5);
    const int k = ncand[i];
    for (int t = 0; t < k; ++t) {
      const real_t* r = &cand[(size_t)(i * KCAND + t) * 3];
      // keep origin side: r·x <= 0.5|r|^2  <=>  geogram positive side: (-r)·x + 0.5|r|^2 >= 0
      VBW::vec4 P{-r[0], -r[1], -r[2], real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2])};
      C.clip_by_plane(P);
    }
    C.compute_geometry();
    return C.volume();
  };
  // geogram with the fast clip path (what their GPU Voronoi uses — skips robustness checks)
  auto geo_fast_one = [&](int i) -> double {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    if (!geocells[tid]) geocells[tid] = new VBW::ConvexCell();
    VBW::ConvexCell& C = *geocells[tid];
    C.init_with_box(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5);
    const int k = ncand[i];
    for (int t = 0; t < k; ++t) {
      const real_t* r = &cand[(size_t)(i * KCAND + t) * 3];
      VBW::vec4 P{-r[0], -r[1], -r[2], real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2])};
      C.clip_by_plane_fast(P);
    }
    C.compute_geometry();
    return C.volume();
  };

  // --- our ConvexCell (FP64) ---
  auto our_one = [&](int i) -> double {
    vor::device::ConvexCell<real_t, 64, 112> c;
    c.initBox(1.0, 1.0, 1.0);
    const int k = ncand[i];
    for (int t = 0; t < k; ++t) {
      const real_t* r = &cand[(size_t)(i * KCAND + t) * 3];
      const real_t n[3] = {r[0], r[1], r[2]};
      const real_t d = real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      c.clip(n, d, t);
      if (c.overflow) break;
    }
    return c.overflow ? 0.0 : c.volume();
  };

  std::printf("construct-from-cache (clip all K, + volume):\n");
  const double g = bench("geogram clip_by_plane", geo_one);
  const double gf = bench("geogram clip_by_plane_fast", geo_fast_one);
  const double o = bench("ours  vor::ConvexCell", our_one);
  std::printf("  ratio ours/geogram(fast) = %.2fx\n", o / gf);
  for (auto* p : geocells) delete p;
  return 0;
}
