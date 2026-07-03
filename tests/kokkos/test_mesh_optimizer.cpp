/**
 * @file test_mesh_optimizer.cpp
 * @brief Acceptance test for the pure-Voronoi POSITION optimiser (voronoiVolumeOptimize).
 *
 *   (A) uniform target: move the seeds to equalise cell volumes — the energy E and the volume
 *       spread must both fall substantially (a centroidal-style mesh).
 *   (B) graded target: small target volume in a slab near x=0.5 → the seeds crowd there and the
 *       slab cells become smaller than the bulk (grid refinement by moving positions).
 */
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/voro/mesh_optimizer.hpp"

using real_t = double;

namespace {

void volumes(const std::vector<real_t>& x, int N, real_t L, int sw, std::vector<real_t>& vol) {
  Kokkos::View<real_t*, peclet::core::MemSpace> dpos("p", 3 * N), dw;
  Kokkos::deep_copy(dpos, Kokkos::View<const real_t*, Kokkos::HostSpace>(x.data(), 3 * N));
  Kokkos::View<long*, peclet::core::MemSpace> gd;
  const real_t Larr[3] = {L, L, L};
  auto res = peclet::voro::buildTessellation<real_t, false>(dpos, dw, N, Larr, sw, N, gd,
                                                            peclet::voro::NoSdf{}, true);
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.view.cellVolume);
  vol.assign(m.data(), m.data() + N);
}

double spread(const std::vector<real_t>& v) {
  double mean = 0;
  for (double x : v) mean += x;
  mean /= v.size();
  double var = 0;
  for (double x : v) var += (x - mean) * (x - mean);
  return std::sqrt(var / v.size()) / mean;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("pure-Voronoi position mesh optimiser:\n");
    const int N = 1500, sw = 5;
    const real_t L = 1.0;
    const double boxVol = L * L * L;
    std::mt19937 rng(3);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<real_t> pos0(3 * N);
    for (auto& v : pos0) v = L * U(rng);

    // (A) uniform target: equalise volumes.
    {
      std::vector<real_t> pos = pos0, vol0, vol1;
      volumes(pos, N, L, sw, vol0);
      const double spread0 = spread(vol0);
      std::vector<real_t> vset(N, boxVol / N);
      auto R = peclet::voro::voronoiVolumeOptimize<real_t>(pos, vset, (real_t[3]){L, L, L}, N, sw,
                                                           peclet::voro::NoSdf{}, 60, 1e-10, 200,
                                                           true);
      volumes(pos, N, L, sw, vol1);
      const double spread1 = spread(vol1);
      const bool pass = R.nEmpty == 0 && spread1 < 0.6 * spread0;  // meaningfully more uniform
      std::printf("  (A) uniform  iters=%d  volSpread %.3f -> %.3f  nBad=%ld  %s\n", R.iters, spread0,
                  spread1, R.nEmpty, pass ? "OK" : "FAIL");
      rc |= pass ? 0 : 1;
    }

    // (B) graded: refine a slab |x-0.5|<0.15 (target = 1/3 the volume there).
    {
      std::vector<real_t> pos = pos0, vol1;
      std::vector<real_t> vset(N);
      for (int i = 0; i < N; ++i)
        vset[i] = std::fabs(pos[3 * i] - 0.5) < 0.15 ? (real_t)(boxVol / N / 3.0) : (real_t)(boxVol / N);
      auto R = peclet::voro::voronoiVolumeOptimize<real_t>(pos, vset, (real_t[3]){L, L, L}, N, sw,
                                                           peclet::voro::NoSdf{}, 80, 1e-10, 200,
                                                           false);
      volumes(pos, N, L, sw, vol1);
      double sfine = 0, sbulk = 0;
      long nfine = 0, nbulk = 0;
      for (int i = 0; i < N; ++i) {  // classify by FINAL position
        if (std::fabs(pos[3 * i] - 0.5) < 0.15) {
          sfine += vol1[i];
          ++nfine;
        } else {
          sbulk += vol1[i];
          ++nbulk;
        }
      }
      const double ratio = (sbulk / nbulk) / (sfine / nfine);
      // position-only Voronoi gives PARTIAL volume control (full target-volume control needs power
      // weights); require clear refinement (slab cells markedly smaller than the bulk).
      const bool pass = R.nEmpty == 0 && ratio > 1.6;
      std::printf("  (B) graded   iters=%d  meanVol bulk=%.2e slab=%.2e  ratio=%.2f  %s\n", R.iters,
                  sbulk / nbulk, sfine / nfine, ratio, pass ? "OK" : "FAIL");
      rc |= pass ? 0 : 1;
    }
    std::printf("%s\n", rc == 0 ? "MESH OPTIMIZER CHECKS PASS" : "MESH OPTIMIZER CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
