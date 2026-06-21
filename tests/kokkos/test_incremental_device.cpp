/**
 * @file test_incremental_device.cpp
 * \brief Phase-6 acceptance: CPU incremental update == full rebuild (new cutter).
 *
 * Moves a seed set along a trajectory; at each step the host Incremental driver
 * updates (reusing cached candidate lists while the Verlet skin holds) and we
 * compare its per-cell volumes against a fresh full rebuild built from scratch on
 * the same cutter. They must agree to machine precision, and most steps must reuse
 * the cache (only crossing the skin triggers a full re-gather).
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "vorflow/host/incremental.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const int N = 800, T = 20;
    const Vec3 L = {1.0, 1.0, 1.0};
    const real_t spacing = std::cbrt(1.0 / N);
    const real_t rcut = 4.0 * spacing, skin = 0.05, stepRms = 0.004;
    std::mt19937 rng(2024);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::normal_distribution<real_t> G(0.0, 1.0);
    std::vector<Vec3> pos(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);

    vor::host::Incremental<real_t> incr;
    incr.build(pos, L, rcut, skin);

    real_t worst = 0;
    int rebuilds = 0, reuses = 0;
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < N; ++i)
        for (int d = 0; d < 3; ++d) {
          pos[i][d] += stepRms * G(rng);
          pos[i][d] -= L[d] * std::floor(pos[i][d] / L[d]);
        }
      bool full = incr.update(pos);
      if (full)
        ++rebuilds;
      else
        ++reuses;
      const std::vector<real_t>& vol = incr.volumes();

      // Fresh full rebuild reference on the same cutter.
      vor::host::Incremental<real_t> ref;
      ref.build(pos, L, rcut, skin);
      const std::vector<real_t>& rvol = ref.volumes();
      for (int i = 0; i < N; ++i)
        worst = std::max(worst, std::fabs(vol[i] - rvol[i]) / rvol[i]);
    }

    if (worst > 1e-12) {
      std::fprintf(stderr, "FAIL incremental vs rebuild: worst rel vol %.3e\n", worst);
      ++failures;
    }
    if (reuses == 0) {
      std::fprintf(stderr, "FAIL: cache never reused (skin too small for motion)\n");
      ++failures;
    }
    std::printf("incremental_device: steps=%d reuses=%d rebuilds=%d worstVolErr=%.2e  %s\n", T,
                reuses, rebuilds, worst, failures == 0 ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "incremental_device PASS" : "incremental_device FAIL");
  return failures;
}
