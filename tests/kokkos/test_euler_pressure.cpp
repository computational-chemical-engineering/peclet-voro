/**
 * @file test_euler_pressure.cpp
 * \brief Phase-4 acceptance: atomic-free pressure force over TessellationView.
 *
 * Computes the compressible-Euler pressure force with the reference physics
 * module (which sees only the published view) and checks it equals the legacy
 * ExplicitEuler::computeForces (which scatters with atomics) to ~machine
 * precision. The gather form writes only F_i per work item — zero atomics.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/physics/euler_pressure.hpp"
#include "vorflow/simulation.hpp"
#include "vorflow/tessellation_build.hpp"
#include "vorflow/tessellation_view.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {

template <class T>
Kokkos::View<T*, tpx::MemSpace> upload(const std::vector<T>& h, const char* l) {
  Kokkos::View<T*, tpx::MemSpace> d(Kokkos::view_alloc(std::string(l), Kokkos::WithoutInitializing),
                                    h.size());
  auto hv = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < h.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d, hv);
  return d;
}

int runCase(int N, unsigned seed) {
  const Vec3 L = {1.0, 1.0, 1.0};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * U(rng);

  // Legacy reference forces.
  vor::ExplicitEuler<real_t> euler;
  euler.setL(L);
  euler.setMassDensity(1.0);
  euler.setPositions(pos);
  euler.setVelocities(std::vector<Vec3>(N, Vec3{0, 0, 0}));
  euler.setMasses(std::vector<real_t>(N, 1.0));
  euler.setPressure(1.0);
  if (!euler.init()) {
    std::fprintf(stderr, "FAIL: euler.init()\n");
    return 1;
  }
  const std::vector<Vec3>& ref = euler.getForces();

  // Published view from the same tessellation.
  const real_t pressEq = 1.0;
  auto host = vor::buildHostTessellation(euler.getCellComplex());
  const real_t volAvg = (L[0] * L[1] * L[2]) / std::max(1, host.nCells);
  std::vector<int> recip = vor::buildReciprocalMap(host);
  std::vector<int> cellOfFacet(host.nFacets, 0);
  for (int i = 0; i < host.nCells; ++i)
    for (int g = host.cellFacetOffset[i]; g < host.cellFacetOffset[i + 1]; ++g)
      cellOfFacet[g] = i;

  vor::TessellationView<real_t> view = vor::upload(host);
  auto dRecip = upload(recip, "recip");
  auto dCellOfFacet = upload(cellOfFacet, "cellOfFacet");
  Kokkos::View<real_t*, tpx::MemSpace> dForce("force", 3 * host.nCells);

  vor::physics::eulerPressureForce(view, dRecip, dCellOfFacet, pressEq, volAvg, dForce);
  Kokkos::fence();
  auto force = Kokkos::create_mirror_view(dForce);
  Kokkos::deep_copy(force, dForce);

  // Compare per cell (cell index == particle id for the dense build).
  real_t maxErr = 0, maxRef = 0;
  for (int i = 0; i < host.nCells; ++i) {
    int p = host.cellSeedId[i];
    for (int k = 0; k < 3; ++k) {
      real_t e = std::fabs(force(3 * i + k) - ref[p][k]);
      if (e > maxErr)
        maxErr = e;
      if (std::fabs(ref[p][k]) > maxRef)
        maxRef = std::fabs(ref[p][k]);
    }
  }
  real_t rel = maxErr / (maxRef + 1e-300);
  bool ok = rel < 1e-10;
  std::printf("  [N=%d] cells=%d maxForce=%.3e maxErr=%.3e rel=%.2e  %s\n", N, host.nCells, maxRef,
              maxErr, rel, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    failures += runCase(1000, 3);
    failures += runCase(3000, 8);
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "euler_pressure PASS" : "euler_pressure FAIL");
  return failures;
}
