/**
 * @file test_simulation_sparse_particles.cpp
 * @brief regression test for simulation stepping with inactive particles
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <vorflow/simulation.hpp>

using std::array;
using std::vector;
using vor::ExplicitEuler;
using vor::uint1;
using vor::uint2;

int main() {
  typedef double real_t;

  ExplicitEuler<real_t> sim;
  sim.setL(array<real_t, 3>{1.0, 1.0, 1.0});
  sim.setPressure(1.0);
  sim.setMassDensity(1.0);

  std::mt19937_64 rng(11);
  std::uniform_real_distribution<real_t> uni(real_t(0), real_t(1));
  std::normal_distribution<real_t> velDist(real_t(0), real_t(1.0e-3));

  const size_t numParticles = 192;
  vector<array<real_t, 3> > pos(numParticles);
  vector<array<real_t, 3> > vel(numParticles);
  for (size_t i = 0; i < numParticles; ++i) {
    for (uint1 k = 0; k < 3; ++k) {
      pos[i][k] = uni(rng);
      vel[i][k] = velDist(rng);
    }
  }

  sim.setPositions(pos);
  sim.setVelocities(vel);
  if (!sim.init()) {
    std::fprintf(stderr, "simulation init failed\n");
    return 1;
  }

  const vector<uint2> deletedIds = {2u, 9u, 17u, 51u, 88u, 121u};
  sim.getCellComplex().deactivateParticles(deletedIds);
  sim.step(1, 1.0e-3);

  const size_t expectedActive = numParticles - deletedIds.size();
  if (sim.getCellComplex().numCells() != expectedActive) {
    std::fprintf(stderr, "unexpected active cell count after sparse step: got %zu expected %zu\n",
                 sim.getCellComplex().numCells(), expectedActive);
    return 1;
  }
  for (size_t i = 0; i < deletedIds.size(); ++i) {
    if (sim.getCellComplex().getCellIndexForParticle(deletedIds[i]) != vor::noNbr) {
      std::fprintf(stderr, "deleted particle %u remained active after simulation step\n",
                   static_cast<unsigned>(deletedIds[i]));
      return 1;
    }
  }

  const real_t kinetic = sim.getKineticEnergy();
  const real_t internal = sim.getInternalEnergy();
  if (!std::isfinite(kinetic) || !std::isfinite(internal)) {
    std::fprintf(stderr, "non-finite simulation energies after sparse step\n");
    return 1;
  }

  std::printf("sparse simulation regression passed (%zu active cells)\n",
              sim.getCellComplex().numCells());
  return 0;
}
