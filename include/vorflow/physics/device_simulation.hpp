/**
 * @file physics/device_simulation.hpp
 * \brief Device-native compressible-Euler simulation facade (de-legacy Phase 4).
 *
 * Replaces the legacy Simulation/ExplicitEuler driver: holds the particle state on
 * device and runs the velocity-Verlet loop entirely on the device path —
 * tessellate (full rebuild) -> publish view -> aux maps -> atomic-free pressure
 * force -> integrate. Faithful port of ExplicitEuler::step:
 *   v += (F/m)·dt/2 ;  x += v·dt (wrapped) ;  rebuild + force ;  v += (F/m)·dt/2.
 * The GPU does a full rebuild each step (the CPU-incremental path is separate).
 *
 * Cell index == seed index (dense periodic build), so the per-cell force maps
 * straight onto the particle. Energies are device reductions.
 */
#ifndef VORFLOW_PHYSICS_DEVICE_SIMULATION_HPP
#define VORFLOW_PHYSICS_DEVICE_SIMULATION_HPP

#include <array>
#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/device/transpose.hpp"
#include "vorflow/physics/euler_pressure.hpp"
#include "vorflow/physics/viscous.hpp"

namespace vor {
namespace physics {

template <class Real>
class ExplicitEulerDevice {
 public:
  using DView = Kokkos::View<Real*, tpx::MemSpace>;

  /// @param posFlat,vel  3*N device arrays (x-fastest per particle), pos in [0,L).
  /// @param invMass      1/m per particle (N). @param pressEq EOS constant.
  void init(const DView& posFlat, const DView& vel, const DView& invMass,
            const std::array<Real, 3>& L, Real pressEq) {
    N_ = static_cast<int>(invMass.extent(0));
    pos_ = posFlat;
    vel_ = vel;
    invMass_ = invMass;
    L_ = L;
    pressEq_ = pressEq;
    w_ = DView("w", N_);
    force_ = DView("force", 3 * N_);
    buildAndForce();
  }

  /// Enable the viscous term (NavierStokes); per-particle viscosities (size N).
  void setViscous(const DView& visc, const DView& bulkVisc) {
    visc_ = visc;
    bulkVisc_ = bulkVisc;
    viscous_ = true;
    buildAndForce();
  }

  void step(int nSteps, Real dt) {
    using Exec = tpx::ExecSpace;
    const Real halfDt = Real(0.5) * dt;
    const Real Lx = L_[0], Ly = L_[1], Lz = L_[2];
    auto pos = pos_;
    auto vel = vel_;
    auto im = invMass_;
    auto force = force_;
    for (int s = 0; s < nSteps; ++s) {
      Kokkos::parallel_for(
          "ee.kick1", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
            for (int k = 0; k < 3; ++k)
              vel(3 * i + k) += force(3 * i + k) * im(i) * halfDt;
          });
      Kokkos::parallel_for(
          "ee.drift", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
            Real x = pos(3 * i + 0) + vel(3 * i + 0) * dt;
            Real y = pos(3 * i + 1) + vel(3 * i + 1) * dt;
            Real z = pos(3 * i + 2) + vel(3 * i + 2) * dt;
            pos(3 * i + 0) = x - Lx * Kokkos::floor(x / Lx);
            pos(3 * i + 1) = y - Ly * Kokkos::floor(y / Ly);
            pos(3 * i + 2) = z - Lz * Kokkos::floor(z / Lz);
          });
      buildAndForce();
      force = force_;
      Kokkos::parallel_for(
          "ee.kick2", Kokkos::RangePolicy<Exec>(0, N_), KOKKOS_LAMBDA(const int i) {
            for (int k = 0; k < 3; ++k)
              vel(3 * i + k) += force(3 * i + k) * im(i) * halfDt;
          });
      time_ += dt;
    }
    Kokkos::fence();
  }

  Real kineticEnergy(const DView& mass) const {
    using Exec = tpx::ExecSpace;
    auto vel = vel_;
    Real e = 0;
    Kokkos::parallel_reduce(
        "ee.ke", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          acc += mass(i) * (vel(3 * i + 0) * vel(3 * i + 0) + vel(3 * i + 1) * vel(3 * i + 1) +
                            vel(3 * i + 2) * vel(3 * i + 2));
        },
        e);
    return Real(0.5) * e;
  }

  Real internalEnergy() const {
    using Exec = tpx::ExecSpace;
    auto vol = view_.cellVolume;
    const Real pe = pressEq_, va = volAvg_;
    Real e = 0;
    Kokkos::parallel_reduce(
        "ee.ie", Kokkos::RangePolicy<Exec>(0, N_),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc -= pe * va * Kokkos::log(vol(i) / va); }, e);
    return e;
  }

  const DView& positions() const { return pos_; }
  const DView& velocities() const { return vel_; }
  const DView& force() const { return force_; }
  const TessellationView<Real>& view() const { return view_; }
  int numParticles() const { return N_; }
  Real time() const { return time_; }

 private:
  void buildAndForce() {
    const Real Larr[3] = {L_[0], L_[1], L_[2]};
    auto res = device::buildTessellation<Real, false>(pos_, w_, N_, Larr);
    view_ = res.view;
    aux_ = device::buildAuxMaps(view_);
    volAvg_ = (L_[0] * L_[1] * L_[2]) / static_cast<Real>(N_);
    Kokkos::deep_copy(force_, Real(0));
    eulerPressureForce(view_, aux_.recip, aux_.cellOfFacet, pressEq_, volAvg_, force_);
    if (viscous_)
      viscousForce(view_, aux_.recip, aux_.cellOfFacet, vel_, visc_, bulkVisc_, force_);
  }

  int N_ = 0;
  std::array<Real, 3> L_{};
  Real pressEq_ = 0, volAvg_ = 0, time_ = 0;
  bool viscous_ = false;
  DView pos_, vel_, invMass_, w_, force_, visc_, bulkVisc_;
  TessellationView<Real> view_;
  device::AuxMaps<Real> aux_;
};

}  // namespace physics
}  // namespace vor

#endif  // VORFLOW_PHYSICS_DEVICE_SIMULATION_HPP
