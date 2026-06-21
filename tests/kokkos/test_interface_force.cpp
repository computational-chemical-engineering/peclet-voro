/**
 * @file test_interface_force.cpp
 * \brief Phase-3 acceptance: device/host interface force vs legacy IntfDyn.
 *
 * With pressEq=0 and zero viscosity the legacy IntfDyn force is exactly the
 * interface-tension force, so it isolates the gradFacetAreaSq port. The host
 * interfaceForce driver (new cutter) must reproduce it.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "vorflow/host/interface_force.hpp"
#include "vorflow/simulation.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const int N = 2000, nT = 3;
    const Vec3 L = {1.0, 1.0, 1.0};
    const real_t rcut = 4.0 * std::cbrt(1.0 / N);
    std::mt19937 rng(15);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<Vec3> pos(N), vel(N, Vec3{0, 0, 0});
    std::vector<std::uint8_t> types8(N);
    std::vector<int> types(N);
    for (int i = 0; i < N; ++i) {
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);
      int t = (int)(U(rng) * nT);
      types8[i] = (std::uint8_t)t;
      types[i] = t;
    }
    std::vector<real_t> tens(nT * nT, 0.0);
    tens[0 * nT + 1] = tens[1 * nT + 0] = 0.7;
    tens[0 * nT + 2] = tens[2 * nT + 0] = 1.1;
    tens[1 * nT + 2] = tens[2 * nT + 1] = 0.4;

    // Legacy IntfDyn with pressure + viscosity OFF -> force is interface-only.
    vor::IntfDyn<real_t> sim;
    sim.setL(L);
    sim.setMassDensity(1.0);
    sim.setPositions(pos);
    sim.setVelocities(vel);
    sim.setMasses(std::vector<real_t>(N, 1.0));
    sim.setViscosities(std::vector<real_t>(N, 0.0));
    sim.setBulkViscosities(std::vector<real_t>(N, 0.0));
    sim.setPressure(0.0);
    sim.setTypes(types8);
    for (int a = 0; a < nT; ++a)
      for (int b = 0; b < nT; ++b)
        if (tens[a * nT + b] > 0)
          sim.setIntfTension(tens[a * nT + b], a, b);
    if (!sim.init()) {
      std::fprintf(stderr, "FAIL init\n");
      Kokkos::finalize();
      return 1;
    }
    const std::vector<Vec3>& ref = sim.getForces();

    auto force = vor::host::interfaceForce<real_t>(pos, types, tens, nT, L, rcut);

    real_t maxErr = 0, maxRef = 0;
    for (int i = 0; i < N; ++i)
      for (int k = 0; k < 3; ++k) {
        maxErr = std::max(maxErr, std::fabs(force[i][k] - ref[i][k]));
        maxRef = std::max(maxRef, std::fabs(ref[i][k]));
      }
    real_t rel = maxErr / (maxRef + 1e-300);
    bool ok = rel < 1e-9;
    if (!ok)
      ++failures;
    std::printf("interface_force: N=%d maxForce=%.3e maxErr=%.3e rel=%.2e  %s\n", N, maxRef, maxErr,
                rel, ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "interface_force PASS" : "interface_force FAIL");
  return failures;
}
