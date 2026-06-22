/**
 * @file bench_cutter.cpp
 * \brief Per-cell cutter profiler (Phase 0 of the cutter-rewrite plan).
 *
 * Attributes single-thread per-cell build time to its sub-phases so the rewrite
 * targets the real hot cost. Two layers:
 *
 *  1. Time split (no instrumentation needed): buildVoronoiCell does ONLY the plane
 *     cuts; volume() and computeGeometry() are separate. Timing cut-only vs +volume
 *     vs +geometry gives the first-order split (cut% / volume% / geometry%).
 *
 *  2. Cut-internal counters (compile with -DVORFLOW_CUTTER_PROFILE): cell_cutter.hpp
 *     bumps host-only counters (trace steps, cdist calls, findRsqMax scans, the
 *     exhaustive-search fallback, DFS steps) so we can see where inside cutCell2 the
 *     time goes.
 *
 * Single-thread by design (run with OMP_NUM_THREADS=1). Not a ctest.
 *   ./bench_cutter [N] [sampleCells] [repeats]
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "vorflow/device/cell_cutter.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;
using clk = std::chrono::high_resolution_clock;

#ifdef VORFLOW_CUTTER_PROFILE
namespace vor {
namespace device {
CutterCounters g_cc{};  // definition of the host-only profiling counters
}
}  // namespace vor
#endif

static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Build, for one seed, its neighbour list sorted by ascending rSqHalf (what the
// tessellator feeds buildVoronoiCell). Brute-force nearest-K with minimal image.
struct Nbrs {
  std::vector<real_t> rx, ry, rz;
  std::vector<int> id;
};

static Nbrs sortedNbrs(const std::vector<Vec3>& pos, int self, real_t L, int K) {
  const int n = (int)pos.size();
  std::vector<std::pair<real_t, int>> d;
  d.reserve(n);
  for (int j = 0; j < n; ++j) {
    if (j == self) continue;
    real_t dx = pos[j][0] - pos[self][0];
    real_t dy = pos[j][1] - pos[self][1];
    real_t dz = pos[j][2] - pos[self][2];
    dx -= L * std::round(dx / L);
    dy -= L * std::round(dy / L);
    dz -= L * std::round(dz / L);
    d.push_back({real_t(0.5) * (dx * dx + dy * dy + dz * dz), j});
  }
  int k = std::min(K, (int)d.size());
  std::partial_sort(d.begin(), d.begin() + k, d.end());
  Nbrs nb;
  for (int m = 0; m < k; ++m) {
    int j = d[m].second;
    real_t dx = pos[j][0] - pos[self][0];
    real_t dy = pos[j][1] - pos[self][1];
    real_t dz = pos[j][2] - pos[self][2];
    dx -= L * std::round(dx / L);
    dy -= L * std::round(dy / L);
    dz -= L * std::round(dz / L);
    nb.rx.push_back(dx);
    nb.ry.push_back(dy);
    nb.rz.push_back(dz);
    nb.id.push_back(j);
  }
  return nb;
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 20000;
  const int M = argc > 2 ? std::atoi(argv[2]) : 4000;   // sample cells per repeat
  const int R = argc > 3 ? std::atoi(argv[3]) : 20;     // repeats
  const real_t L = 1.0;
  const int K = 64;  // neighbours fed per cell (early-out stops well before this)

  std::mt19937 rng(12345);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d) pos[i][d] = L * U(rng);

  // Precompute sorted neighbour lists for M sample seeds (excluded from timing).
  std::vector<Nbrs> samples(M);
  for (int s = 0; s < M; ++s) samples[s] = sortedNbrs(pos, s, L, K);

  const real_t Larr[3] = {L, L, L};
  namespace vd = vor::device;

  auto timeMode = [&](int mode) -> double {  // 0=cut, 1=+volume, 2=+geometry
    double best = 1e30;
    volatile real_t sink = 0;
    for (int rep = 0; rep < R; ++rep) {
      auto t0 = clk::now();
      for (int s = 0; s < M; ++s) {
        const Nbrs& nb = samples[s];
        vd::ScratchCell<real_t> c;
        vd::CutStatus st = vd::buildVoronoiCell<real_t, false>(
            c, Larr, nb.rx.data(), nb.ry.data(), nb.rz.data(), nb.id.data(), nullptr,
            (int)nb.id.size(), real_t(0));
        if (st != vd::CutStatus::Ok) continue;
        if (mode >= 1) sink += c.volume();
        if (mode >= 2) {
          c.computeGeometry();
          sink += c.fArea[0][0];
        }
      }
      auto t1 = clk::now();
      best = std::min(best, secs(t0, t1));
    }
    (void)sink;
    return best;
  };

  double tCut = timeMode(0);
  double tVol = timeMode(1);
  double tGeo = timeMode(2);
  double cells = (double)M;
  std::printf("cutter profile  N=%d  sampleCells=%d  repeats=%d  (K=%d nbrs/cell)\n", N, M, R, K);
  std::printf("  cut only        : %8.1f kcells/s  (%.3f us/cell)\n", cells / tCut / 1e3,
              tCut / cells * 1e6);
  std::printf("  cut + volume    : %8.1f kcells/s  (volume = %.1f%%)\n", cells / tVol / 1e3,
              100.0 * (tVol - tCut) / tGeo);
  std::printf("  cut + vol + geom: %8.1f kcells/s  (geometry = %.1f%%, cut = %.1f%%)\n",
              cells / tGeo / 1e3, 100.0 * (tGeo - tVol) / tGeo, 100.0 * tCut / tGeo);

#ifdef VORFLOW_CUTTER_PROFILE
  // One untimed pass to populate the counters (cut only).
  vd::g_cc = vd::CutterCounters{};
  for (int s = 0; s < M; ++s) {
    const Nbrs& nb = samples[s];
    vd::ScratchCell<real_t> c;
    vd::buildVoronoiCell<real_t, false>(c, Larr, nb.rx.data(), nb.ry.data(), nb.rz.data(),
                                        nb.id.data(), nullptr, (int)nb.id.size(), real_t(0));
  }
  const auto& g = vd::g_cc;
  double perCell = 1.0 / cells;
  std::printf("  --- cut internals (per cell, %d cells) ---\n", M);
  std::printf("    cutCell2 calls       : %6.1f\n", g.cuts * perCell);
  std::printf("    cdist calls          : %6.1f\n", g.cdistCalls * perCell);
  std::printf("    trace-loop steps     : %6.1f\n", g.traceSteps * perCell);
  std::printf("    DFS-delete steps     : %6.1f\n", g.dfsSteps * perCell);
  std::printf("    findRsqMax calls     : %6.1f  (verts scanned: %.1f)\n", g.findRsqMaxCalls * perCell,
              g.findRsqMaxScans * perCell);
  std::printf("    exhaustive-search    : %6.1f  (verts scanned: %.1f)\n", g.exhaustiveCalls * perCell,
              g.exhaustiveScans * perCell);
#else
  std::printf("  (build with -DVORFLOW_CUTTER_PROFILE for the cut-internal counters)\n");
#endif
  return 0;
}
