/**
 * @file test_device_step.cpp
 * \brief Phase-4 acceptance: device Euler trajectory vs legacy ExplicitEuler.
 *
 * Runs the device-native velocity-Verlet loop (tessellate -> force -> integrate)
 * and the legacy ExplicitEuler::step over the same steps, and checks the particle
 * state and energies agree. The legacy uses the incremental update and the device
 * a full rebuild each step; over a few steps these match to ~machine precision
 * (incremental == rebuild). Positions are compared modulo the box (the legacy
 * leaves them unwrapped, the device wraps).
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/physics/device_simulation.hpp"
#include "vorflow/simulation.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {
template <class T>
Kokkos::View<T*, tpx::MemSpace> up(const std::vector<T>& h, const char* l) {
  Kokkos::View<T*, tpx::MemSpace> d(Kokkos::view_alloc(std::string(l), Kokkos::WithoutInitializing),
                                    h.size());
  auto hv = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < h.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d, hv);
  return d;
}
real_t wrap1(real_t x, real_t L) {
  x -= L * std::floor(x / L);
  if (x >= L)
    x -= L;
  if (x < 0)
    x += L;
  return x;
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const int N = 1500, steps = 5;
    const real_t dt = 0.005, pressEq = 1.0;
    const Vec3 L = {1.0, 1.0, 1.0};
    std::mt19937 rng(4);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::normal_distribution<real_t> Gv(0.0, 0.05);
    std::vector<Vec3> pos(N), vel(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d) {
        pos[i][d] = L[d] * U(rng);
        vel[i][d] = Gv(rng);
      }

    // Legacy.
    vor::ExplicitEuler<real_t> ee;
    ee.setL(L);
    ee.setMassDensity(1.0);
    ee.setPositions(pos);
    ee.setVelocities(vel);
    ee.setMasses(std::vector<real_t>(N, 1.0));
    ee.setPressure(pressEq);
    ee.init();
    ee.step(steps, dt);
    const std::vector<Vec3>& lpos = ee.getPositions();
    const std::vector<Vec3>& lvel = ee.getVelocities();
    real_t lKE = ee.getKineticEnergy(), lIE = ee.getInternalEnergy();

    // Device.
    std::vector<real_t> posF(3 * N), velF(3 * N), invM(N, 1.0), mass(N, 1.0);
    for (int i = 0; i < N; ++i)
      for (int k = 0; k < 3; ++k) {
        posF[3 * i + k] = pos[i][k];
        velF[3 * i + k] = vel[i][k];
      }
    vor::physics::ExplicitEulerDevice<real_t> dev;
    dev.init(up(posF, "pos"), up(velF, "vel"), up(invM, "im"), L, pressEq);
    dev.step(steps, dt);
    auto dPos = Kokkos::create_mirror_view(dev.positions());
    auto dVel = Kokkos::create_mirror_view(dev.velocities());
    Kokkos::deep_copy(dPos, dev.positions());
    Kokkos::deep_copy(dVel, dev.velocities());
    auto dMass = up(mass, "m");
    real_t dKE = dev.kineticEnergy(dMass), dIE = dev.internalEnergy();

    real_t maxPos = 0, maxVel = 0;
    for (int i = 0; i < N; ++i)
      for (int k = 0; k < 3; ++k) {
        real_t dp = wrap1(dPos(3 * i + k), L[k]) - wrap1(lpos[i][k], L[k]);
        dp -= L[k] * std::round(dp / L[k]);  // minimal image
        maxPos = std::max(maxPos, std::fabs(dp));
        maxVel = std::max(maxVel, std::fabs(dVel(3 * i + k) - lvel[i][k]));
      }
    real_t keErr = std::fabs(dKE - lKE) / std::fabs(lKE);
    real_t ieErr = std::fabs(dIE - lIE) / std::fabs(lIE);
    bool ok = maxPos < 1e-9 && maxVel < 1e-9 && keErr < 1e-9 && ieErr < 1e-9;
    if (!ok)
      ++failures;
    std::printf("device_step: steps=%d maxPos=%.2e maxVel=%.2e keErr=%.2e ieErr=%.2e  %s\n", steps,
                maxPos, maxVel, keErr, ieErr, ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "device_step PASS" : "device_step FAIL");
  return failures;
}
