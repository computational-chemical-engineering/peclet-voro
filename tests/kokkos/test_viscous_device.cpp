/**
 * @file test_viscous_device.cpp
 * \brief Phase-2 acceptance: device viscous force vs legacy NavierStokes.
 *
 * Device (Euler pressure + viscous, both atomic-free over TessellationView) must
 * equal the legacy NavierStokes::computeForces (pressure + viscous, atomic
 * scatter) to ~machine precision. External force off, so the legacy total is
 * exactly pressure + viscous.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/physics/euler_pressure.hpp"
#include "vorflow/physics/viscous.hpp"
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
  std::normal_distribution<real_t> G(0.0, 1.0);
  std::vector<Vec3> pos(N), vel(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d) {
      pos[i][d] = L[d] * U(rng);
      vel[i][d] = 0.1 * G(rng);  // non-trivial velocity field
    }
  const real_t mu = 0.7, mub = 0.3, pressEq = 1.0;

  // Legacy reference: NavierStokes total force (pressure + viscous, external 0).
  vor::NavierStokes<real_t> ns;
  ns.setL(L);
  ns.setMassDensity(1.0);
  ns.setPositions(pos);
  ns.setVelocities(vel);
  ns.setMasses(std::vector<real_t>(N, 1.0));
  ns.setViscosities(std::vector<real_t>(N, mu));
  ns.setBulkViscosities(std::vector<real_t>(N, mub));
  ns.setPressure(pressEq);
  if (!ns.init()) {
    std::fprintf(stderr, "FAIL: ns.init()\n");
    return 1;
  }
  const std::vector<Vec3>& ref = ns.getForces();

  // Published view + transpose from the same tessellation.
  auto host = vor::buildHostTessellation(ns.getCellComplex());
  const real_t volAvg = (L[0] * L[1] * L[2]) / std::max(1, host.nCells);
  std::vector<int> recip = vor::buildReciprocalMap(host);
  std::vector<int> cellOfFacet(host.nFacets, 0);
  for (int i = 0; i < host.nCells; ++i)
    for (int g = host.cellFacetOffset[i]; g < host.cellFacetOffset[i + 1]; ++g)
      cellOfFacet[g] = i;
  std::vector<real_t> velFlat(3 * host.nCells);
  for (int i = 0; i < host.nCells; ++i)
    for (int k = 0; k < 3; ++k)
      velFlat[3 * i + k] = vel[host.cellSeedId[i]][k];

  vor::TessellationView<real_t> view = vor::upload(host);
  auto dRecip = upload(recip, "recip");
  auto dCellOfFacet = upload(cellOfFacet, "cellOfFacet");
  auto dVel = upload(velFlat, "vel");
  auto dVisc = upload(std::vector<real_t>(host.nCells, mu), "visc");
  auto dBulk = upload(std::vector<real_t>(host.nCells, mub), "bulk");
  Kokkos::View<real_t*, tpx::MemSpace> dForce("force", 3 * host.nCells);
  Kokkos::deep_copy(dForce, 0.0);

  vor::physics::eulerPressureForce(view, dRecip, dCellOfFacet, pressEq, volAvg, dForce);
  vor::physics::viscousForce(view, dRecip, dCellOfFacet, dVel, dVisc, dBulk, dForce);
  Kokkos::fence();
  auto force = Kokkos::create_mirror_view(dForce);
  Kokkos::deep_copy(force, dForce);

  real_t maxErr = 0, maxRef = 0;
  for (int i = 0; i < host.nCells; ++i) {
    int p = host.cellSeedId[i];
    for (int k = 0; k < 3; ++k) {
      maxErr = std::max(maxErr, std::fabs(force(3 * i + k) - ref[p][k]));
      maxRef = std::max(maxRef, std::fabs(ref[p][k]));
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
  std::printf("%s\n", failures == 0 ? "viscous_device PASS" : "viscous_device FAIL");
  return failures;
}
