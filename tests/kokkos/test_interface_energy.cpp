/**
 * @file test_interface_energy.cpp
 * \brief Phase-3 acceptance (partial): device interface energy vs legacy IntfDyn.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/physics/interface.hpp"
#include "vorflow/simulation.hpp"
#include "vorflow/tessellation_build.hpp"
#include "vorflow/tessellation_view.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const int N = 2000, nT = 3;
    const Vec3 L = {1.0, 1.0, 1.0};
    std::mt19937 rng(9);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<Vec3> pos(N), vel(N, Vec3{0, 0, 0});
    std::vector<std::uint8_t> types(N);
    for (int i = 0; i < N; ++i) {
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);
      types[i] = (std::uint8_t)(U(rng) * nT);  // phases 0,1,2
    }
    // Symmetric tension table.
    std::vector<real_t> tens(nT * nT, 0.0);
    tens[0 * nT + 1] = tens[1 * nT + 0] = 0.7;
    tens[0 * nT + 2] = tens[2 * nT + 0] = 1.1;
    tens[1 * nT + 2] = tens[2 * nT + 1] = 0.4;

    vor::IntfDyn<real_t> sim;
    sim.setL(L);
    sim.setMassDensity(1.0);
    sim.setPositions(pos);
    sim.setVelocities(vel);
    sim.setMasses(std::vector<real_t>(N, 1.0));
    sim.setViscosities(std::vector<real_t>(N, 0.1));
    sim.setBulkViscosities(std::vector<real_t>(N, 0.0));
    sim.setPressure(1.0);
    sim.setTypes(types);
    for (int a = 0; a < nT; ++a)
      for (int b = 0; b < nT; ++b)
        if (tens[a * nT + b] > 0)
          sim.setIntfTension(tens[a * nT + b], a, b);
    if (!sim.init()) {
      std::fprintf(stderr, "FAIL: init\n");
      Kokkos::finalize();
      return 1;
    }
    const real_t refE = sim.getIntfEnergy();

    // Device.
    auto host = vor::buildHostTessellation(sim.getCellComplex());
    vor::TessellationView<real_t> view = vor::upload(host);
    std::vector<int> typeCell(host.nCells);
    for (int i = 0; i < host.nCells; ++i)
      typeCell[i] = types[host.cellSeedId[i]];
    Kokkos::View<int*, tpx::MemSpace> dTypes("types", host.nCells);
    {
      auto h = Kokkos::create_mirror_view(dTypes);
      for (int i = 0; i < host.nCells; ++i)
        h(i) = typeCell[i];
      Kokkos::deep_copy(dTypes, h);
    }
    Kokkos::View<real_t*, tpx::MemSpace> dTens("tens", nT * nT);
    {
      auto h = Kokkos::create_mirror_view(dTens);
      for (int i = 0; i < nT * nT; ++i)
        h(i) = tens[i];
      Kokkos::deep_copy(dTens, h);
    }
    const real_t devE = vor::physics::interfaceEnergy(view, dTypes, dTens, nT);

    const real_t rel = std::fabs(devE - refE) / std::fabs(refE);
    bool ok = rel < 1e-12;
    if (!ok)
      ++failures;
    std::printf("interface_energy: refE=%.10f devE=%.10f rel=%.2e  %s\n", refE, devE, rel,
                ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "interface_energy PASS" : "interface_energy FAIL");
  return failures;
}
