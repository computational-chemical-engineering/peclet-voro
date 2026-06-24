/**
 * @file bench_construct_compact.cpp
 * \brief A/B the memory-lean ConvexCellCompact (packed tri + recomputed vertex) against the
 *        cached-vertex ConvexCell, construct-from-cache (no gather), to test whether the
 *        single-thread construct is local-memory/footprint-bound. Same candidates for both.
 *        Reports kcells/s (G0 topology, G1 + volume) and Σvol/box err. FP32 (CC_FLOAT) or FP64.
 *        Run: ./bench_construct_compact [N]
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"
#include "vorflow/device/convex_cell_compact.hpp"

#ifdef CC_FLOAT
using real_t = float;
#else
using real_t = double;
#endif
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
static constexpr int KCAND = 64;
static constexpr int CC_MAXP = 64, CC_MAXT = 112;

template <class Cell, int MaxT, int MinB, class CandV, class NcandV, class VolV>
static double run_tier_lb(int N, int tier, const CandV& cand, const NcandV& ncand, VolV outVol,
                          real_t L, double* volSumOut) {
  using Exec = tpx::ExecSpace;
  using Policy = Kokkos::RangePolicy<Exec, Kokkos::LaunchBounds<MaxT, MinB>>;
  auto kern = KOKKOS_LAMBDA(const int i) {
    Cell c;
    c.initBox(L, L, L);
    const int k = ncand(i);
    for (int t = 0; t < k; ++t) {
      const real_t* r = &cand((size_t)(i * KCAND + t) * 3);
      const real_t off = real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      if (off >= real_t(2) * c.maxVertexRsq()) continue;
      const real_t n[3] = {r[0], r[1], r[2]};
      c.clip(n, off, t);
      if (c.overflow) break;
    }
    outVol(i) = (tier >= 1 && !c.overflow) ? c.volume() : real_t(0);
  };
  Kokkos::parallel_for("warm", Policy(0, N), kern);
  Kokkos::fence();
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    auto t0 = clk::now();
    Kokkos::parallel_for("run", Policy(0, N), kern);
    Kokkos::fence();
    auto t1 = clk::now();
    best = std::min(best, secs(t0, t1));
  }
  if (volSumOut) {
    auto h = Kokkos::create_mirror_view(outVol);
    Kokkos::deep_copy(h, outVol);
    double s = 0;
    for (int i = 0; i < N; ++i) s += h(i);
    *volSumOut = s;
  }
  return N / best / 1e3;
}

template <class Cell, class CandV, class NcandV, class VolV>
static double run_tier(const char* tag, int N, int tier, const CandV& cand, const NcandV& ncand,
                       VolV outVol, real_t L, double* volSumOut) {
  using Exec = tpx::ExecSpace;
  auto kern = KOKKOS_LAMBDA(const int i) {
    Cell c;
    c.initBox(L, L, L);
    const int k = ncand(i);
    for (int t = 0; t < k; ++t) {
      const real_t* r = &cand((size_t)(i * KCAND + t) * 3);
      const real_t off = real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      if (off >= real_t(2) * c.maxVertexRsq()) continue;
      const real_t n[3] = {r[0], r[1], r[2]};
      c.clip(n, off, t);
      if (c.overflow) break;
    }
    outVol(i) = (tier >= 1 && !c.overflow) ? c.volume() : real_t(0);
  };
  Kokkos::parallel_for("warm", Kokkos::RangePolicy<Exec>(0, N), kern);
  Kokkos::fence();
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    auto t0 = clk::now();
    Kokkos::parallel_for("run", Kokkos::RangePolicy<Exec>(0, N), kern);
    Kokkos::fence();
    auto t1 = clk::now();
    best = std::min(best, secs(t0, t1));
  }
  if (volSumOut) {
    auto h = Kokkos::create_mirror_view(outVol);
    Kokkos::deep_copy(h, outVol);
    double s = 0;
    for (int i = 0; i < N; ++i) s += h(i);
    *volSumOut = s;
  }
  (void)tag;
  return N / best / 1e3;
}

int main(int argc, char** argv) {
  using MemSpace = tpx::MemSpace;
  Kokkos::initialize(argc, argv);
  {
    std::printf("backend: %s  (%s)\n", Kokkos::DefaultExecutionSpace::name(),
                sizeof(real_t) == 4 ? "FP32" : "FP64");
    const int N = argc > 1 ? std::atoi(argv[1]) : 1000000;
    const real_t L = 1.0, spacing = std::cbrt(1.0 / N);
    std::printf("cell sizes (bytes): ConvexCell=%zu  ConvexCellCompact=%zu\n",
                sizeof(vor::device::ConvexCell<real_t, CC_MAXP, CC_MAXT>),
                sizeof(vor::device::ConvexCellCompact<real_t, CC_MAXP, CC_MAXT>));

    // host: random points + closest-K candidate planes per cell (grid brute force; one-time).
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
                rx -= std::round(rx); ry -= std::round(ry); rz -= std::round(rz);
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
    using Orig = vor::device::ConvexCell<real_t, CC_MAXP, CC_MAXT>;
    using Comp = vor::device::ConvexCellCompact<real_t, CC_MAXP, CC_MAXT>;

    double vO = 0, vC = 0;
    const double oG0 = run_tier<Orig>("orig", N, 0, cand, ncand, outVol, L, nullptr);
    const double oG1 = run_tier<Orig>("orig", N, 1, cand, ncand, outVol, L, &vO);
    const double cG0 = run_tier<Comp>("comp", N, 0, cand, ncand, outVol, L, nullptr);
    const double cG1 = run_tier<Comp>("comp", N, 1, cand, ncand, outVol, L, &vC);

    std::printf("N=%d  construct-from-cache, kcells/s        G0(topo)     G1(+vol)   Σvol/box err\n", N);
    std::printf("  ConvexCell  (cached vertex, 16 B/tri) : %9.1f   %9.1f   %.2e\n", oG0, oG1,
                std::fabs(vO - 1.0));
    std::printf("  Compact     (packed tri, recompute)   : %9.1f   %9.1f   %.2e\n", cG0, cG1,
                std::fabs(vC - 1.0));
    std::printf("  speedup (compact / cached)            : %8.2fx   %8.2fx\n", cG0 / oG0, cG1 / oG1);

    // Occupancy test: force the compiler toward fewer registers / more blocks per SM via
    // LaunchBounds. If throughput rises, the construct is latency/occupancy-bound (=> more cells
    // in flight is the lever, and warp-cooperative which REDUCES occupancy is the wrong way).
    double l0 = 0, l1 = 0, l2 = 0, l3 = 0;
    const double b_def = run_tier_lb<Orig, 256, 0>(N, 1, cand, ncand, outVol, L, &l0);  // default
    const double b_4 = run_tier_lb<Orig, 256, 4>(N, 1, cand, ncand, outVol, L, &l1);    // >=4 blk/SM
    const double b_6 = run_tier_lb<Orig, 256, 6>(N, 1, cand, ncand, outVol, L, &l2);    // >=6 blk/SM
    const double b_8 = run_tier_lb<Orig, 128, 12>(N, 1, cand, ncand, outVol, L, &l3);   // push hard
    std::printf("\n  occupancy sweep (ConvexCell, G1), LaunchBounds<MaxThreads,MinBlocksPerSM>:\n");
    std::printf("    <256,0> default %.1f | <256,4> %.1f | <256,6> %.1f | <128,12> %.1f  k/s\n",
                b_def, b_4, b_6, b_8);
  }
  Kokkos::finalize();
  return 0;
}
