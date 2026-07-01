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
#ifndef PECLET_VORO_PHYSICS_SIMULATION_HPP
#define PECLET_VORO_PHYSICS_SIMULATION_HPP

#include <array>
#include <Kokkos_Core.hpp>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/reeval_tessellation.hpp"  // reevalPublish (E1 repair -> force geometry)
#include "peclet/voro/repair.hpp"               // MovingTessellation (incremental update)
#include "peclet/voro/tessellator.hpp"
#include "peclet/voro/transpose.hpp"
#include "peclet/voro/physics/euler_pressure.hpp"
#include "peclet/voro/physics/viscous.hpp"

namespace peclet::voro {
namespace physics {

template <class Real>
class ExplicitEuler {
 public:
  using DView = Kokkos::View<Real*, peclet::core::MemSpace>;

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
    // Persistent viscous scratch (9*N each), allocated once here rather than every buildAndForce
    // (E4).
    viscGrad_ = DView(Kokkos::view_alloc("visc.grad", Kokkos::WithoutInitializing),
                      static_cast<std::size_t>(9) * N_);
    viscStress_ = DView(Kokkos::view_alloc("visc.stress", Kokkos::WithoutInitializing),
                        static_cast<std::size_t>(9) * N_);
    buildAndForce();
  }

  void step(int nSteps, Real dt) {
    using Exec = peclet::core::ExecSpace;
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
    using Exec = peclet::core::ExecSpace;
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
    using Exec = peclet::core::ExecSpace;
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

 public:
  /// Opt-in (E1 scaffolding, default off): use the incremental moving-point repair for the topology
  /// each step and reeval-publish the force geometry, instead of a full rebuild. Same physics
  /// contract; ~10× cheaper geometry per the repair benches. Set before init(). The physics
  /// wiring/tuning on top of this scaffolding is ongoing work.
  void setRepair(bool on) { useRepair_ = on; }
  bool repair() const { return useRepair_; }

 private:
  void buildAndForce() {
    const Real Larr[3] = {L_[0], L_[1], L_[2]};
    if (useRepair_) {
      // Incremental path: cold-build the resident tessellation once (establishes the topology store
      // + volumes), then repair it in place each subsequent step. Either way the force geometry is
      // re-evaluated from the store and published into the same facet-CSR view the full build emits.
      if (!mtInit_) {
        const double boxVol = static_cast<double>(L_[0]) * L_[1] * L_[2];
        const Real spacing = static_cast<Real>(std::cbrt(boxVol / (N_ > 0 ? N_ : 1)));
        mt_.alloc(N_, Larr, Real(1e-4) * spacing, Real(0.25) * spacing, /*sw=*/4, /*density=*/N_);
        mt_.rebuild(pos_);
        mtInit_ = true;
      } else {
        mt_.step(pos_);
      }
      view_ = peclet::voro::reevalPublish<Real, 64, 112>(mt_.store, pos_, mt_.vol, N_, Larr);
    } else {
      // Pass the persistent worklist cache (last arg) so the step-invariant worklist table is built
      // once and reused across steps (E3). All intermediate args are the buildTessellation defaults;
      // the Sdf template arg is named explicitly because a defaulted `{}` Sdf cannot be deduced.
      auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(
          pos_, w_, N_, Larr, /*sw=*/4, /*densityCount=*/-1, /*gid=*/{}, peclet::voro::NoSdf{},
          /*withForceGeom=*/true, /*nBuild=*/-1, /*outNp=*/{}, /*outNt=*/{}, /*outPnbr=*/{},
          /*outTri=*/{}, /*outCand=*/{}, /*outCandCnt=*/{}, /*candCap=*/0, &wlCache_);
      view_ = res.view;
    }
    aux_ = peclet::voro::buildAuxMaps(view_);
    volAvg_ = (L_[0] * L_[1] * L_[2]) / static_cast<Real>(N_);
    Kokkos::deep_copy(force_, Real(0));
    eulerPressureForce(view_, aux_.recip, aux_.cellOfFacet, pressEq_, volAvg_, force_);
    if (viscous_)
      viscousForce(view_, aux_.recip, aux_.cellOfFacet, vel_, visc_, bulkVisc_, force_, viscGrad_,
                   viscStress_);
  }

  int N_ = 0;
  std::array<Real, 3> L_{};
  Real pressEq_ = 0, volAvg_ = 0, time_ = 0;
  bool viscous_ = false;
  DView pos_, vel_, invMass_, w_, force_, visc_, bulkVisc_;
  DView viscGrad_, viscStress_;          // persistent 9*N viscous scratch (E4)
  peclet::voro::WorklistCache<Real> wlCache_;  // step-invariant worklist table, reused across steps (E3)
  TessellationView<Real> view_;
  peclet::voro::AuxMaps<Real> aux_;
  // E1 opt-in incremental path (default off): resident moving-point tessellation + its cold/repair
  // state. Kept as members (Views free before Kokkos::finalize).
  bool useRepair_ = false, mtInit_ = false;
  peclet::voro::MovingTessellation<Real, 64, 112> mt_;
};

}  // namespace physics
}  // namespace peclet::voro

#endif  // PECLET_VORO_PHYSICS_SIMULATION_HPP
