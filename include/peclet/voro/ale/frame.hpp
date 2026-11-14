/**
 * @file frame.hpp
 * \brief Moving-grid stage 0 frame pipeline (docs/ale_design.md §1.4, §1.5, §3.5, §5.1, WO-S0.2):
 * repair → publish → face mesh, for one `MovingTessellation`.
 *
 * A frame is the face mesh of the tessellation at one set of generator positions, with the
 * snapshot of the positions and weights it was built from and the telemetry of the repair that
 * produced it. `initialFrame` builds it after a cold `rebuild`, `advanceFrame` after a repair
 * `step`. The publish uses the engine's plane policy (Power for the weighted engine, the one the
 * moving-grid solver uses; Voronoi for the unweighted engine); the published view is transient,
 * a local of the function (the mesh keeps only the arrays it aliases).
 *
 * Changed set: the engine's `changedMask()` (1 for a cell the step rebuilt). A cold build (the
 * initial frame, a gate rebuild, the fallback) changes every cell; `numRebuilt` counts the cells
 * changed in the frame, over the repair step and the reciprocity rounds together.
 *
 * Reciprocity (§3.5): a facet i → j whose cell j has no facet back to i is non-reciprocal; the
 * face mesh drops it (`FaceMesh::nDropped`). With `enforceReciprocity` (default on), when such a
 * facet has area > kNonReciprocalAreaFrac · h² (h³ = the mean cell volume), both of its cells are
 * forced through `step(pos, forced, n)` (§5.1) and the frame is published again, at most
 * kMaxReciprocityRounds rounds; `reciprocity` records what remains and the rounds used.
 *
 * Single domain only (nProc == N); the distributed frame is a later work order.
 */
#ifndef PECLET_VORO_ALE_FRAME_HPP
#define PECLET_VORO_ALE_FRAME_HPP

#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/dynamic_validate.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/transpose.hpp"

namespace peclet::voro::ale {

/// Non-reciprocal facets with area above this fraction of h² force a rebuild of both cells (α).
inline constexpr double kNonReciprocalAreaFrac = 1e-12;
/// At most this many forced rebuild + publish rounds per frame.
inline constexpr int kMaxReciprocityRounds = 2;

/// The face-mesh options of a frame: deterministic order, gid orientation, generator geometry and
/// cell centroids (gid empty: the slot index). The frame fills `pos`, `weight` and `L`.
template <class Real>
fv::FaceMeshOptions<Real> frameMeshOptions() {
  fv::FaceMeshOptions<Real> o;
  o.deterministicOrder = o.gidOrientation = o.generatorGeometry = o.cellCentroids = true;
  return o;
}

template <class Real>
struct FrameOptions {
  /// Builder options; `pos`, `weight` and `L` are overwritten with the frame's snapshot. All
  /// flags off is the default builder.
  fv::FaceMeshOptions<Real> mesh = frameMeshOptions<Real>();
  bool enforceReciprocity = true;  ///< force the cells of large non-reciprocal facets (§3.5)
  bool timed = false;  ///< fence and time the phases into MeshFrame::timings (diagnostic)
};

/// Reciprocity of the published view of a frame (after the rounds, if any).
struct ReciprocityReport {
  double maxAreaAsym = 0;      ///< max |A(g) + A(recip g)| / |A(g)| over reciprocal pairs
  long nNonRecip = 0;          ///< facets toward a cell with no facet back
  int nDropped = 0;            ///< non-reciprocal facets the face mesh dropped
  double maxNonRecipArea = 0;  ///< largest area of a non-reciprocal facet
  double areaThreshold = 0;    ///< kNonReciprocalAreaFrac · h²
  int rounds = 0;              ///< forced rebuild + publish rounds used
};

/// Wall time of the phases of the last frame build, seconds (only with FrameOptions::timed).
struct FrameTimings {
  double repair = 0, publish = 0, auxMaps = 0, faceMesh = 0, reciprocity = 0;
};

template <class Real>
struct MeshFrame {
  using Mem = peclet::core::MemSpace;
  Kokkos::View<Real*, Mem> pos;     ///< 3N generator positions the frame was built from
  Kokkos::View<Real*, Mem> weight;  ///< N power weights (empty for the unweighted engine)
  fv::FaceMesh<Real> mesh;
  RepairStats stats;       ///< of the repair step (default for the initial frame)
  bool coldBuild = false;  ///< every cell changed: the initial frame, a gate rebuild, the fallback
  int numRebuilt = 0;      ///< cells changed in this frame (step and reciprocity rounds)
  ReciprocityReport reciprocity;
  FrameTimings timings;
};

namespace detail {

template <class Real>
Kokkos::View<Real*, peclet::core::MemSpace> snapshot(
    const Kokkos::View<Real*, peclet::core::MemSpace>& v, const char* name) {
  if (v.extent(0) == 0)
    return {};
  Kokkos::View<Real*, peclet::core::MemSpace> s(
      Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), v.extent(0));
  Kokkos::deep_copy(s, v);
  return s;
}

inline int countNonzero(const Kokkos::View<int*, peclet::core::MemSpace>& m, int n) {
  int c = 0;
  Kokkos::parallel_reduce(
      "ale.frame.count", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i, int& a) { a += m(i) ? 1 : 0; }, c);
  return c;
}

struct Clock {
  bool on;
  std::chrono::steady_clock::time_point t0;
  explicit Clock(bool on_) : on(on_) {
    if (on) {
      Kokkos::fence();
      t0 = std::chrono::steady_clock::now();
    }
  }
  double lap() {
    if (!on)
      return 0;
    Kokkos::fence();
    const auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    t0 = t1;
    return s;
  }
};

/// Publish the engine's tessellation at the frame's snapshot, build the face mesh, and measure the
/// reciprocity. With a non-empty `forcedMask` (N), marks in it both cells of every non-reciprocal
/// facet above the threshold (the mask is left untouched when there is none).
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
void publish(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine, MeshFrame<Real>& f,
             const Real L[3], const FrameOptions<Real>& opt,
             const Kokkos::View<int*, peclet::core::MemSpace>& forcedMask) {
  using Engine = MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>;
  using Policy = typename Engine::PlanePolicy;
  using Exec = peclet::core::ExecSpace;
  const int N = engine.N;
  Clock clk(opt.timed);
  TessellationView<Real> view = reevalPublish<Real, MAXP, MAXT, Policy>(
      engine.store, f.pos, engine.vol, N, L, engine.wall, engine.xRef, false, false, f.weight);
  f.timings.publish += clk.lap();
  AuxMaps<Real> aux = buildAuxMaps(view);
  f.timings.auxMaps += clk.lap();
  fv::FaceMeshOptions<Real> mo = opt.mesh;
  mo.pos = f.pos;
  mo.weight = f.weight;
  for (int d = 0; d < 3; ++d)
    mo.L[d] = L[d];
  f.mesh = fv::buildFaceMesh(view, aux, -1, mo);
  f.timings.faceMesh += clk.lap();

  // reciprocity: the invariants of the published view, the largest non-reciprocal facet area
  const double boxVol = (double)L[0] * (double)L[1] * (double)L[2];
  const Invariants inv = checkInvariants(view, aux, boxVol);
  double sumV = 0;
  Kokkos::parallel_reduce(
      "ale.frame.sumVol", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, double& a) { a += (double)view.volume(i); }, sumV);
  const double h2 = std::pow(sumV / (N > 0 ? N : 1), 2.0 / 3.0);
  const double thr = kNonReciprocalAreaFrac * h2;
  auto recip = aux.recip;
  double maxA = 0;
  Kokkos::parallel_reduce(
      "ale.frame.nonRecipArea", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, double& m) {
        for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
          const long j = (long)view.facetNbr(g);
          if (j < 0 || j >= N || recip(g) >= 0)
            continue;
          double a2 = 0;
          for (int c = 0; c < 3; ++c)
            a2 += (double)view.area(g, c) * (double)view.area(g, c);
          const double a = std::sqrt(a2);
          if (a > m)
            m = a;
        }
      },
      Kokkos::Max<double>(maxA));
  f.reciprocity.maxAreaAsym = inv.maxAreaAsym;
  f.reciprocity.nNonRecip = inv.nNonRecip;
  f.reciprocity.nDropped = f.mesh.nDropped;
  f.reciprocity.maxNonRecipArea = maxA;
  f.reciprocity.areaThreshold = thr;
  if (forcedMask.extent(0) > 0 && maxA > thr) {
    auto M = forcedMask;
    Kokkos::deep_copy(M, 0);
    Kokkos::parallel_for(
        "ale.frame.markNonRecip", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
            const long j = (long)view.facetNbr(g);
            if (j < 0 || j >= N || recip(g) >= 0)
              continue;
            double a2 = 0;
            for (int c = 0; c < 3; ++c)
              a2 += (double)view.area(g, c) * (double)view.area(g, c);
            if (std::sqrt(a2) > thr) {
              Kokkos::atomic_store(&M(i), 1);
              Kokkos::atomic_store(&M((int)j), 1);
            }
          }
        });
  }
  f.timings.reciprocity += clk.lap();
}

/// The cells marked in `mask[0..n)` as an ascending slot list (deterministic scan); returns the
/// count.
inline int maskToList(const Kokkos::View<int*, peclet::core::MemSpace>& mask, int n,
                      const Kokkos::View<int*, peclet::core::MemSpace>& list) {
  int total = 0;
  Kokkos::parallel_scan(
      "ale.frame.maskToList", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i, int& upd, const bool isFinal) {
        if (mask(i)) {
          if (isFinal)
            list(upd) = i;
          ++upd;
        }
      },
      total);
  return total;
}

/// Shared tail of initialFrame / advanceFrame: publish, then the reciprocity rounds. `changed`
/// enters as the frame's changed set when a round is needed (allocated here on demand).
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
void publishWithReciprocity(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                            const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                            MeshFrame<Real>& f, const Real L[3], const FrameOptions<Real>& opt) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  const int N = engine.N;
  Kokkos::View<int*, Mem> forcedMask;
  if (opt.enforceReciprocity)
    forcedMask = Kokkos::View<int*, Mem>(
        view_alloc(std::string("ale.frame.forcedMask"), WithoutInitializing), N);
  publish(engine, f, L, opt, forcedMask);
  if (!opt.enforceReciprocity)
    return;
  Kokkos::View<int*, Mem> changed, list;
  for (int round = 0; round < kMaxReciprocityRounds; ++round) {
    if (!(f.reciprocity.maxNonRecipArea > f.reciprocity.areaThreshold))
      break;
    if (changed.extent(0) == 0) {
      // the frame's changed set so far: all cells after a cold build, else the step's mask
      changed = Kokkos::View<int*, Mem>(
          view_alloc(std::string("ale.frame.changed"), WithoutInitializing), N);
      if (f.coldBuild)
        Kokkos::deep_copy(changed, 1);
      else
        Kokkos::deep_copy(changed, engine.changedMask());
      list = Kokkos::View<int*, Mem>(
          view_alloc(std::string("ale.frame.forced"), WithoutInitializing), N);
    }
    const int nF = maskToList(forcedMask, N, list);
    Clock clk(opt.timed);
    const RepairStats st = engine.step(pos, list, nF);
    f.timings.repair += clk.lap();
    if (st.route == RepairStats::kRebuildGate || st.fellBack) {
      f.coldBuild = true;
      Kokkos::deep_copy(changed, 1);
    } else {
      auto C = changed;
      auto R = engine.changedMask();
      Kokkos::parallel_for(
          "ale.frame.orChanged", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
            if (R(i))
              C(i) = 1;
          });
    }
    f.reciprocity.rounds = round + 1;
    publish(engine, f, L, opt, forcedMask);
  }
  if (changed.extent(0) > 0)
    f.numRebuilt = countNonzero(changed, N);
}

template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
void checkSingleDomain(const MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine) {
  if (engine.nProc != engine.N)
    throw std::invalid_argument("ale::MeshFrame: single domain only (nProc == N)");
}

}  // namespace detail

/// The frame after a cold `engine.rebuild(pos)`: every cell changed. `weight` is the engine's
/// power weights (ignored, and not kept, for the unweighted engine).
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
MeshFrame<Real> initialFrame(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& weight,
                             const Real L[3], const FrameOptions<Real>& opt = {}) {
  detail::checkSingleDomain(engine);
  MeshFrame<Real> f;
  detail::Clock clk(opt.timed);
  engine.rebuild(pos);
  f.timings.repair = clk.lap();
  f.pos = detail::snapshot(pos, "ale.frame.pos");
  if constexpr (Weighted)
    f.weight = detail::snapshot(weight, "ale.frame.weight");
  f.coldBuild = true;
  f.numRebuilt = engine.nProc;
  detail::publishWithReciprocity(engine, pos, f, L, opt);
  return f;
}

/// The frame after a repair `engine.step(pos)` from the engine's present state.
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
MeshFrame<Real> advanceFrame(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& weight,
                             const Real L[3], const FrameOptions<Real>& opt = {}) {
  detail::checkSingleDomain(engine);
  MeshFrame<Real> f;
  detail::Clock clk(opt.timed);
  f.stats = engine.step(pos);
  f.timings.repair = clk.lap();
  f.pos = detail::snapshot(pos, "ale.frame.pos");
  if constexpr (Weighted)
    f.weight = detail::snapshot(weight, "ale.frame.weight");
  f.coldBuild = f.stats.route == RepairStats::kRebuildGate || f.stats.fellBack;
  f.numRebuilt =
      f.coldBuild ? engine.nProc : detail::countNonzero(engine.changedMask(), engine.nProc);
  detail::publishWithReciprocity(engine, pos, f, L, opt);
  return f;
}

}  // namespace peclet::voro::ale

#endif  // PECLET_VORO_ALE_FRAME_HPP
