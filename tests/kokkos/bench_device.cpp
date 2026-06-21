/**
 * @file bench_device.cpp
 * \brief Throughput benchmark: device tessellation (GPU/CPU) vs legacy (CPU).
 *
 * Times a full tessellation build on the device path (whatever Kokkos backend the
 * binary was compiled for) against the legacy CellComplex (OpenMP CPU), over a
 * range of N, and reports cells/sec and the speedup. A volume-sum check confirms
 * the device build is correct on the backend. Not a ctest — run manually:
 *   ./bench_device [N1 N2 ...]
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/voronoi.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;
using clk = std::chrono::high_resolution_clock;

static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

static void run(int N) {
  const Vec3 L = {1.0, 1.0, 1.0};
  std::mt19937 rng(12345 + N);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d) pos[i][d] = L[d] * U(rng);

  // Legacy (CPU, OpenMP) — best of 2.
  vor::Box<real_t> box(L);
  double legacyBest = 1e30, legacyVol = 0;
  for (int rep = 0; rep < 2; ++rep) {
    vor::CellComplex<real_t> c(&box);
    auto t0 = clk::now();
    c.build(pos);
    auto t1 = clk::now();
    legacyBest = std::min(legacyBest, secs(t0, t1));
    if (rep == 0) {
      const auto& g = c.getGeoms();
      for (size_t i = 0; i < g.size(); ++i) legacyVol += g[i].getVolume();
    }
  }

  // Device (current Kokkos backend) — upload once, warm up, best of 3.
  Kokkos::View<real_t*, tpx::MemSpace> dPos(
      Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)N * 3);
  {
    auto h = Kokkos::create_mirror_view(dPos);
    for (int i = 0; i < N; ++i)
      for (int k = 0; k < 3; ++k) h(3 * i + k) = pos[i][k];
    Kokkos::deep_copy(dPos, h);
  }
  Kokkos::View<real_t*, tpx::MemSpace> dW("w", N);
  const real_t Larr[3] = {L[0], L[1], L[2]};
  auto warm = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr);
  Kokkos::fence();
  // device volume sum (correctness)
  double devVol = 0;
  {
    auto v = Kokkos::create_mirror_view(warm.view.cellVolume);
    Kokkos::deep_copy(v, warm.view.cellVolume);
    for (int i = 0; i < N; ++i) devVol += v(i);
  }
  double devBest = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    Kokkos::fence();
    auto t0 = clk::now();
    auto res = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr);
    Kokkos::fence();
    auto t1 = clk::now();
    devBest = std::min(devBest, secs(t0, t1));
    (void)res;
  }

  const double volErr = std::fabs(devVol - legacyVol) / legacyVol;
  std::printf(
      "N=%-8d  legacy %7.1f kcells/s (%.3fs)  device %7.1f kcells/s (%.3fs)  speedup %5.2fx  "
      "volErr=%.1e\n",
      N, N / legacyBest / 1e3, legacyBest, N / devBest / 1e3, devBest, legacyBest / devBest, volErr);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("backend: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::vector<int> Ns;
    for (int i = 1; i < argc; ++i) Ns.push_back(std::atoi(argv[i]));
    if (Ns.empty()) Ns = {10000, 30000, 100000};
    for (int N : Ns) run(N);
  }
  Kokkos::finalize();
  return 0;
}
