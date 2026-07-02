/**
 * @file device/verlet_skin.hpp
 * \brief Per-particle Verlet-skin tracker (Part II, Phase 1) — the *insertion* trigger.
 *
 * The convexity certificate (ConvexCell::isSelfConsistent) detects that a particle LEFT a region
 * (its old neighbours flag), but never that one ARRIVED: when a far-mover lands among cells that
 * did not move, those cells stay convex against their current planes and never flag, so the gained
 * adjacency is invisible to detection (Risk 1c in dynamic_update_decision_and_plan.md). The only
 * signal that reaches it is "this particle moved far" — the standard Verlet criterion: flag a
 * particle once it has moved more than skin/2 from its last-rebuild position (two seeds can each
 * move skin/2 and close the skin between them). Those flagged movers are rebuilt by gather in the
 * repair Pass 1 so their new neighbours are found.
 *
 * This is deliberately the *only* trigger built in Phase 1. The trigger value is an extensible bit
 * mask (`SkinTrigger`) so the deferred SDF boundary-contact trigger (Risk 1d: |SDF(p)| <
 * security-radius status change) can be OR'd in later without reworking the repair driver.
 *
 * Core header: Kokkos + core, no physics. One path for CUDA/HIP/OpenMP.
 */
#ifndef PECLET_VORO_VERLET_SKIN_HPP
#define PECLET_VORO_VERLET_SKIN_HPP

#include <Kokkos_Core.hpp>
#include <string>

#include "peclet/core/common/view.hpp"

namespace peclet::voro {

/// Per-particle rebuild-trigger bits. Extensible: Phase 1 only ever sets kSkinMover; the deferred
/// boundary trigger (Risk 1d) will add kSkinBoundary so a status-change near an SDF wall feeds the
/// same Pass-1 set. Stored as int so the flag array doubles as a stream-compaction mask.
enum SkinTrigger : int {
  kSkinNone = 0,
  kSkinMover = 1,
  // kSkinBoundary = 2,  // DEFERRED (Risk 1d): |SDF(p)| < security radius contact-status change.
};

/// Resident reference configuration for the Verlet skin. Holds the positions captured at the last
/// full (re)build and the skin width; `reset` is called right after a rebuild, `flag` each step.
template <class Real>
struct VerletSkin {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  Kokkos::View<Real*, Mem> xRef;  ///< 3N: seed positions at the last rebuild
  Real skin = 0;                  ///< skin width in absolute length units (caller picks vs spacing)
  int N = 0;

  void alloc(int n, Real skinWidth) {
    N = n;
    skin = skinWidth;
    xRef = Kokkos::View<Real*, Mem>(
        Kokkos::view_alloc(std::string("verlet.xRef"), Kokkos::WithoutInitializing), (size_t)n * 3);
  }

  /// Capture the reference configuration (call immediately after a full (re)build).
  void reset(const Kokkos::View<Real*, Mem>& pos) { Kokkos::deep_copy(xRef, pos); }
};

/// Flag every particle that has moved more than skin/2 from xRef (minimal image in box L), writing
/// `kSkinMover`/`kSkinNone` into outFlags, and return the number flagged. `outFlags` must be sized
/// N. The flag value is OR-combinable with future trigger bits; here it is overwritten each call
/// (the only Phase-1 trigger). One reduce + write pass; no host round-trip beyond the final count.
template <class Real>
int flagSkinMovers(const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                   const Kokkos::View<Real*, peclet::core::MemSpace>& xRef, Real skin,
                   const Real L[3], const Kokkos::View<int*, peclet::core::MemSpace>& outFlags) {
  using Exec = peclet::core::ExecSpace;
  const int N = (int)outFlags.extent(0);
  const Real half2 = Real(0.25) * skin * skin;  // (skin/2)^2
  const Real Lx = L[0], Ly = L[1], Lz = L[2];
  const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
  int count = 0;
  Kokkos::parallel_reduce(
      "verlet.flag", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, int& acc) {
        Real dx = pos(3 * i + 0) - xRef(3 * i + 0);
        Real dy = pos(3 * i + 1) - xRef(3 * i + 1);
        Real dz = pos(3 * i + 2) - xRef(3 * i + 2);
        dx = dx > Lxh ? dx - Lx : (dx < -Lxh ? dx + Lx : dx);
        dy = dy > Lyh ? dy - Ly : (dy < -Lyh ? dy + Ly : dy);
        dz = dz > Lzh ? dz - Lz : (dz < -Lzh ? dz + Lz : dz);
        const Real d2 = dx * dx + dy * dy + dz * dz;
        const bool mover = d2 > half2;
        outFlags(i) = mover ? kSkinMover : kSkinNone;
        if (mover)
          ++acc;
      },
      count);
  return count;
}

/// Largest per-seed displacement from xRef (minimal image), in absolute units. Diagnostics / the
/// Verlet rebuild decision (max-disp > skin/2 -> a global rebuild is the simplest safe response).
template <class Real>
Real maxDisplacement(const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                     const Kokkos::View<Real*, peclet::core::MemSpace>& xRef, const Real L[3]) {
  using Exec = peclet::core::ExecSpace;
  const int N = (int)(xRef.extent(0) / 3);
  const Real Lx = L[0], Ly = L[1], Lz = L[2];
  const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
  Real m2 = 0;
  Kokkos::parallel_reduce(
      "verlet.maxDisp", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& lm) {
        Real dx = pos(3 * i + 0) - xRef(3 * i + 0);
        Real dy = pos(3 * i + 1) - xRef(3 * i + 1);
        Real dz = pos(3 * i + 2) - xRef(3 * i + 2);
        dx = dx > Lxh ? dx - Lx : (dx < -Lxh ? dx + Lx : dx);
        dy = dy > Lyh ? dy - Ly : (dy < -Lyh ? dy + Ly : dy);
        dz = dz > Lzh ? dz - Lz : (dz < -Lzh ? dz + Lz : dz);
        const Real d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > lm)
          lm = d2;
      },
      Kokkos::Max<Real>(m2));
  return Kokkos::sqrt(m2);
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_VERLET_SKIN_HPP
