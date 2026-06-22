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

#ifdef VORFLOW_CUTTER_PROFILE
namespace vor {
namespace device {
CutterCounters g_cc{};  // count device cutCell2 calls inside the tessellator (serial)
}
}  // namespace vor
#endif

#ifdef VORFLOW_HAVE_VOROPP
#include "voro++.hh"
// Serial voro++ reference: build every cell once (compute_cell over all seeds).
static void voropp_build(const std::vector<std::array<double, 3>>& pos, double L) {
  const int n = static_cast<int>(pos.size());
  const int nbl = std::max(1, static_cast<int>(std::cbrt(n / 8.0)));
  voro::container_periodic con(L, 0, L, 0, 0, L, nbl, nbl, nbl, 8);
  for (int i = 0; i < n; ++i) con.put(i, pos[i][0], pos[i][1], pos[i][2]);
  voro::voronoicell_neighbor c(con);
  voro::c_loop_all_periodic vl(con);
  if (vl.start())
    do {
      con.compute_cell(c, vl);
    } while (vl.inc());
}
#endif

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

  // Legacy (CPU, OpenMP) — best of 2. VORFLOW_LEGACY_NOGEOM=1 builds topology only
  // (build(pos,false)) to compare against the historical build(false) scaling.
  // VORFLOW_DEVONLY=1 skips the (slow at large N) legacy + voro++ reference builds.
  const bool legacyGeom = std::getenv("VORFLOW_LEGACY_NOGEOM") == nullptr;
  const bool devOnly = std::getenv("VORFLOW_DEVONLY") != nullptr;
  vor::Box<real_t> box(L);
  double legacyBest = 1e30, legacyVol = 0;
  for (int rep = 0; rep < 2 && !devOnly; ++rep) {
    vor::CellComplex<real_t> c(&box);
    auto t0 = clk::now();
    c.build(pos, legacyGeom);
    auto t1 = clk::now();
    legacyBest = std::min(legacyBest, secs(t0, t1));
    if (rep == 0 && legacyGeom) {
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
  const int sw = std::getenv("VORFLOW_SW") ? std::atoi(std::getenv("VORFLOW_SW")) : 4;
  auto warm = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr, sw);
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
    auto res = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr, sw);
    Kokkos::fence();
    auto t1 = clk::now();
    devBest = std::min(devBest, secs(t0, t1));
    (void)res;
  }

  double voroBest = 0;
#ifdef VORFLOW_HAVE_VOROPP
  voroBest = 1e30;
  for (int rep = 0; rep < 2 && !devOnly; ++rep) {
    auto t0 = clk::now();
    voropp_build(pos, L[0]);
    auto t1 = clk::now();
    voroBest = std::min(voroBest, secs(t0, t1));
  }
#endif

#ifdef VORFLOW_CUTTER_PROFILE
  {
    vor::device::g_cc = vor::device::CutterCounters{};
    auto r = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr, sw);
    Kokkos::fence();
    (void)r;
    const auto& g = vor::device::g_cc;
    std::printf("  [cutprofile N=%d] cutCell2/cell=%.1f cdist/cell=%.1f trace/cell=%.1f "
                "findRsqMax/cell=%.1f(scan %.1f)\n",
                N, (double)g.cuts / N, (double)g.cdistCalls / N, (double)g.traceSteps / N,
                (double)g.findRsqMaxCalls / N, (double)g.findRsqMaxScans / N);
  }
#endif

  const double volErr = legacyVol > 0 ? std::fabs(devVol - legacyVol) / legacyVol : 0.0;
  const double legKs = devOnly ? 0.0 : N / legacyBest / 1e3;
  std::printf(
      "N=%-8d  voro++ %7.1f k/s  legacy %7.1f k/s  device %7.1f k/s  | dev/voro %4.2fx  "
      "dev/legacy %4.2fx  volErr=%.1e\n",
      N, voroBest > 0 ? N / voroBest / 1e3 : 0.0, legKs, N / devBest / 1e3,
      voroBest > 0 ? voroBest / devBest : 0.0, devOnly ? 0.0 : legacyBest / devBest, volErr);
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
