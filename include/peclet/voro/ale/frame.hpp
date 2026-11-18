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
 * Reciprocity (§3.5): a facet i → j whose cell j has no facet back to i is non-reciprocal
 * (one-sided); the face mesh drops it (`FaceMesh::nDropped`). With `enforceReciprocity` (default
 * on), both cells of every one-sided facet of nonzero area are forced through
 * `step(pos, forced, n)` (§5.1) and the frame is published again, at most kMaxReciprocityRounds
 * rounds; `reciprocity` records what remains and the rounds used.
 *
 * Closure (WO-S0.6): the repair keeps a topology within its certificate tolerance, so a kept cell
 * may carry an inverted corner (a short edge whose flip is not yet applied). Its facets are then
 * not convex, the angular vertex order of the publish does not follow the topology, and the cell
 * does not close (Σ A_f ≠ 0, up to 6e-5 of its surface at 1e-4 h); its neighbours' facets disagree
 * with it to second order. With `enforceClosure` (default on), every cell whose own closure plus
 * the disagreement of its reciprocal pairs exceeds kClosureFrac of its surface is forced in the
 * same rounds; a forced cell is rebuilt from its clip and closes like a cold one.
 *
 * Two counts make a frame an error the caller must act on; the frame is built regardless:
 * `numEmptyCells`, the cells of the published view with zero volume or no facet (an empty power
 * cell, §11 A5), and `numOneSidedFacets`, the one-sided facets of area > kOneSidedFacetAreaFrac ·
 * h² (h³ = the mean cell volume) left after the rounds.
 *
 * Weights: the frame snapshots the engine's own power weights (`engine.weight`), so the published
 * geometry and the repaired topology belong to one power diagram.
 *
 * Volumes: every frame publishes the cell volume of the repair step's certificate kernel. A cold
 * build (`rebuild`: the initial frame, a gate rebuild, the fallback, also inside a reciprocity
 * round) leaves the clipping kernel's volume, which a device compiler may contract differently (up
 * to 6 ulp under nvcc); it is followed by one `step(pos)` at the same positions, which rebuilds
 * nothing and re-evaluates every volume, so a frame after a cold build equals the frames at rest
 * that follow it bitwise (§11 A11). `stats`, `coldBuild` and `numRebuilt` describe the real step.
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
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/dynamic_validate.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/transpose.hpp"

namespace peclet::voro::ale {

/// A one-sided facet larger than this fraction of h² that survives the reciprocity rounds makes
/// the frame an error (MeshFrame::numOneSidedFacets). The face mesh drops a one-sided facet; a
/// dropped facet of area a leaves the cell a volume-rate defect of up to a · h · CFL per step, i.e.
/// a/h² of V_i at CFL 1, against the geometric conservation bar of 1e-14 V_i of stage 1. Smaller
/// facets are still forced (any area > 0) but tolerated if they remain.
inline constexpr double kOneSidedFacetAreaFrac = 1e-14;
/// At most this many forced rebuild + publish rounds per frame.
inline constexpr int kMaxReciprocityRounds = 2;
/// Closure bar of a published cell (WO-S0.6): |Σ_g A_g| + Σ_g |A_g + A_recip(g)| over the cell's
/// facets, against this fraction of Σ_g |A_g|. The first term is the cell's own closure, the
/// second the disagreement with its neighbours' facets, which the face mesh uses for the faces
/// they own; together they bound the closure of the cell in the face mesh. A cell above it is
/// forced (FrameOptions::enforceClosure). The bar of gate G0.5 H2 / G0.6 B1.
inline constexpr double kClosureFrac = 1e-13;

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
  bool enforceReciprocity = true;  ///< force the cells of every one-sided facet (§3.5)
  /// Force every cell above the closure bar kClosureFrac (WO-S0.6): a cell whose kept topology is
  /// stale within the repair's certificate tolerance publishes facets that do not close.
  bool enforceClosure = true;
  bool timed = false;  ///< fence and time the phases into MeshFrame::timings (diagnostic)
  /// Diagnostic (WO-S1.3a-r): record each round's forced cells and the open-cell count after each
  /// publish into MeshFrame::roundLog (host copies; the frame itself is unchanged bitwise).
  bool recordRounds = false;
};

/// Reciprocity of the published view of a frame (after the rounds, if any).
struct ReciprocityReport {
  double maxAreaAsym = 0;      ///< max |A(g) + A(recip g)| / |A(g)| over reciprocal pairs
  long nNonRecip = 0;          ///< facets toward a cell with no facet back
  int nDropped = 0;            ///< non-reciprocal facets the face mesh dropped
  double maxNonRecipArea = 0;  ///< largest area of a non-reciprocal facet
  double areaThreshold = 0;    ///< kOneSidedFacetAreaFrac · h² (the bar of numOneSidedFacets)
  int rounds = 0;              ///< forced rebuild + publish rounds used
  int numRoundRebuilt = 0;     ///< cells the rounds' steps rebuilt, summed over rounds (N if cold)
  int numOpenCells = 0;  ///< cells above the closure bar kClosureFrac (after the rounds, if any)
};

/// The rounds of a frame (only with FrameOptions::recordRounds): `open[k]`, `maxNonRecipArea[k]`
/// after publish k (k = 0 before any round), `forced[k]` the ascending slots forced in round k + 1.
struct RoundLog {
  std::vector<int> open;
  std::vector<double> maxNonRecipArea;
  std::vector<std::vector<int>> forced;
};

/// Wall time of the phases of the last frame build, seconds (only with FrameOptions::timed).
struct FrameTimings {
  double repair = 0, publish = 0, auxMaps = 0, faceMesh = 0, reciprocity = 0;
};

template <class Real>
struct MeshFrame {
  using Mem = peclet::core::MemSpace;
  Kokkos::View<Real*, Mem> pos;     ///< 3N generator positions the frame was built from
  Kokkos::View<Real*, Mem> weight;  ///< N engine power weights (empty for the unweighted engine)
  fv::FaceMesh<Real> mesh;
  RepairStats stats;  ///< of the repair step (default for the initial frame; not the step at rest)
  bool coldBuild = false;  ///< every cell changed: the initial frame, a gate rebuild, the fallback
  int numRebuilt = 0;      ///< cells changed in this frame (step and reciprocity rounds)
  /// Owned cells with zero volume or no facet in the published view (empty power cells): a frame
  /// with any is an error the caller must act on (docs/ale_design.md §11 A5).
  int numEmptyCells = 0;
  /// One-sided facets of area > kOneSidedFacetAreaFrac · h² left after the reciprocity rounds
  /// (with or without enforcement): a frame with any is an error the caller must act on, as for
  /// numEmptyCells.
  int numOneSidedFacets = 0;
  ReciprocityReport reciprocity;
  FrameTimings timings;
  RoundLog roundLog;  ///< only with FrameOptions::recordRounds
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

/// Closure test of one cell of a published view (WO-S0.6): its own closure |Σ_g A_g| plus the
/// disagreement Σ_g |A_g + A_recip(g)| with its neighbours' facets, above kClosureFrac · Σ_g |A_g|.
template <class Real>
struct OpenCell {
  TessellationView<Real> view;
  Kokkos::View<int*, peclet::core::MemSpace> recip;
  KOKKOS_INLINE_FUNCTION bool operator()(const int i) const {
    double v[3] = {0, 0, 0}, sum = 0, dis = 0;
    for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
      double a2 = 0, d2 = 0;
      const int r = recip(g);
      for (int d = 0; d < 3; ++d) {
        const double a = (double)view.area(g, d);
        v[d] += a;
        a2 += a * a;
        if (r >= 0) {
          const double e = a + (double)view.area(r, d);
          d2 += e * e;
        }
      }
      sum += Kokkos::sqrt(a2);
      dis += Kokkos::sqrt(d2);
    }
    return Kokkos::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + dis > kClosureFrac * sum;
  }
};

/// Publish the engine's tessellation at the frame's snapshot, build the face mesh, and measure the
/// reciprocity and the closure. With a non-empty `forcedMask` (N), marks in it both cells of every
/// non-reciprocal facet of nonzero area (enforceReciprocity) and every cell above the closure bar
/// (enforceClosure); the mask is left untouched when there is none.
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
  // The published cellVolume IS the engine's live `vol` (reevalPublish reuses it); the frame owns
  // a copy, so a later step of the engine does not rewrite this frame's volumes (§1.4).
  if (f.mesh.cellVolume.data() == engine.vol.data())
    f.mesh.cellVolume = snapshot(engine.vol, "ale.frame.cellVolume");
  f.timings.faceMesh += clk.lap();

  // reciprocity: the invariants of the published view, the largest non-reciprocal facet area
  const double boxVol = (double)L[0] * (double)L[1] * (double)L[2];
  const Invariants inv = checkInvariants(view, aux, boxVol);
  double sumV = 0;
  Kokkos::parallel_reduce(
      "ale.frame.sumVol", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, double& a) { a += (double)view.volume(i); }, sumV);
  const double h2 = std::pow(sumV / (N > 0 ? N : 1), 2.0 / 3.0);
  const double thr = kOneSidedFacetAreaFrac * h2;
  auto recip = aux.recip;
  double maxA = 0;
  int nOver = 0;
  Kokkos::parallel_reduce(
      "ale.frame.nonRecipArea", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, double& m, int& c) {
        for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
          const long j = (long)view.facetNbr(g);
          if (j < 0 || j >= N || recip(g) >= 0)
            continue;
          double a2 = 0;
          for (int d = 0; d < 3; ++d)
            a2 += (double)view.area(g, d) * (double)view.area(g, d);
          const double a = std::sqrt(a2);
          if (a > m)
            m = a;
          c += a > thr ? 1 : 0;
        }
      },
      Kokkos::Max<double>(maxA), nOver);
  int nEmpty = 0;
  Kokkos::parallel_reduce(
      "ale.frame.empty", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, int& a) {
        a += (!((double)view.volume(i) > 0) || view.facetEnd(i) == view.facetBegin(i)) ? 1 : 0;
      },
      nEmpty);
  f.numEmptyCells = nEmpty;
  f.numOneSidedFacets = nOver;
  f.reciprocity.maxAreaAsym = inv.maxAreaAsym;
  f.reciprocity.nNonRecip = inv.nNonRecip;
  f.reciprocity.nDropped = f.mesh.nDropped;
  f.reciprocity.maxNonRecipArea = maxA;
  f.reciprocity.areaThreshold = thr;
  // closure (WO-S0.6): the cell's own closure plus the disagreement of its reciprocal pairs
  const OpenCell<Real> isOpen{view, recip};
  int nOpen = 0;
  Kokkos::parallel_reduce(
      "ale.frame.open", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, int& a) { a += isOpen(i) ? 1 : 0; }, nOpen);
  f.reciprocity.numOpenCells = nOpen;
  const bool markRecip = opt.enforceReciprocity && maxA > 0;
  const bool markOpen = opt.enforceClosure && nOpen > 0;
  if (forcedMask.extent(0) > 0 && (markRecip || markOpen)) {
    auto M = forcedMask;
    Kokkos::deep_copy(M, 0);
    if (markOpen)
      Kokkos::parallel_for(
          "ale.frame.markOpen", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
            if (isOpen(i))
              M(i) = 1;
          });
    if (markRecip)
      Kokkos::parallel_for(
          "ale.frame.markNonRecip", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
            for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
              const long j = (long)view.facetNbr(g);
              if (j < 0 || j >= N || recip(g) >= 0)
                continue;
              double a2 = 0;
              for (int c = 0; c < 3; ++c)
                a2 += (double)view.area(g, c) * (double)view.area(g, c);
              if (a2 > 0) {
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

/// True if the step ended in a cold build (the rebuild gate or the fallback).
inline bool isCold(const RepairStats& st) {
  return st.route == RepairStats::kRebuildGate || st.fellBack;
}

/// After a cold build: one repair step at the same positions (§11 A11). It rebuilds nothing and
/// leaves in the engine the certificate kernel's cell volume; its statistics are not the frame's.
/// After a gate rebuild (which skips the adjacency) it also builds the adjacency, so the next step
/// certifies with the local certificate instead of the brute one (test_ale_frame part c, INFO).
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
void stepAtRest(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                const Kokkos::View<Real*, peclet::core::MemSpace>& pos) {
  engine.step(pos);
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
  if (opt.enforceReciprocity || opt.enforceClosure)
    forcedMask = Kokkos::View<int*, Mem>(
        view_alloc(std::string("ale.frame.forcedMask"), WithoutInitializing), N);
  publish(engine, f, L, opt, forcedMask);
  auto record = [&] {
    if (opt.recordRounds) {
      f.roundLog.open.push_back(f.reciprocity.numOpenCells);
      f.roundLog.maxNonRecipArea.push_back(f.reciprocity.maxNonRecipArea);
    }
  };
  record();
  if (!opt.enforceReciprocity && !opt.enforceClosure)
    return;
  Kokkos::View<int*, Mem> changed, list;
  for (int round = 0; round < kMaxReciprocityRounds; ++round) {
    const bool recipDue = opt.enforceReciprocity && f.reciprocity.maxNonRecipArea > 0;
    const bool closureDue = opt.enforceClosure && f.reciprocity.numOpenCells > 0;
    if (!recipDue && !closureDue)
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
    if (opt.recordRounds) {
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   Kokkos::subview(list, std::make_pair(0, nF)));
      f.roundLog.forced.emplace_back(h.data(), h.data() + nF);
    }
    Clock clk(opt.timed);
    const RepairStats st = engine.step(pos, list, nF);
    if (isCold(st))
      stepAtRest(engine, pos);
    f.timings.repair += clk.lap();
    if (isCold(st)) {
      f.coldBuild = true;
      Kokkos::deep_copy(changed, 1);
      f.reciprocity.numRoundRebuilt += N;
    } else {
      f.reciprocity.numRoundRebuilt += countNonzero(engine.changedMask(), N);
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
    record();
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

/// The closure terms of one cell of a published view (diagnostic, WO-S1.3a-r): its own closure
/// |Σ_g A_g|, the disagreement Σ_g |A_g + A_recip(g)|, the surface Σ_g |A_g|, the facet with the
/// largest disagreement (its neighbour and |A_g + A_recip(g)|), the facet count and the facets
/// without a reciprocal.
struct CellClosure {
  int cell = -1;
  double own = 0, dis = 0, sum = 0;
  long worstNbr = -1;
  double worstDis = 0;
  int facets = 0, nonRecip = 0;
  std::vector<long> nbrs;  ///< the neighbour of each facet, in facet order
};

/// Diagnostic (WO-S1.3a-r): publish the engine's present tessellation at `pos` (with `weight`,
/// the engine's power weights or empty), as `publish` of a frame does, and return the cells above
/// the closure bar kClosureFrac with their terms (ascending slots); with `frac` > 0, the cells
/// with own + dis > frac · sum instead (the tail of the closure distribution). Re-evaluates the
/// engine's cell volumes from its unchanged store (the values `publish` leaves).
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
std::vector<CellClosure> openCellsOf(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                                     const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                                     const Kokkos::View<Real*, peclet::core::MemSpace>& weight,
                                     const Real L[3], double frac = -1) {
  using Engine = MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>;
  using Policy = typename Engine::PlanePolicy;
  using Exec = peclet::core::ExecSpace;
  using Mem = peclet::core::MemSpace;
  const int N = engine.N;
  TessellationView<Real> view = reevalPublish<Real, MAXP, MAXT, Policy>(
      engine.store, pos, engine.vol, N, L, engine.wall, engine.xRef, false, false, weight);
  AuxMaps<Real> aux = buildAuxMaps(view);
  auto recip = aux.recip;
  const detail::OpenCell<Real> isOpen{view, recip};
  Kokkos::View<double* [5], Mem> t("ale.diag.closure", N);
  Kokkos::View<int* [4], Mem> k("ale.diag.closureIdx", N);
  Kokkos::parallel_for(
      "ale.diag.closure", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        double v[3] = {0, 0, 0}, sum = 0, dis = 0, worst = 0;
        long wn = -1;
        int nr = 0;
        for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
          double a2 = 0, d2 = 0;
          const int r = recip(g);
          nr += r >= 0 ? 0 : 1;
          for (int d = 0; d < 3; ++d) {
            const double a = (double)view.area(g, d);
            v[d] += a;
            a2 += a * a;
            if (r >= 0) {
              const double e = a + (double)view.area(r, d);
              d2 += e * e;
            }
          }
          sum += Kokkos::sqrt(a2);
          const double dg = Kokkos::sqrt(d2);
          dis += dg;
          if (dg > worst || wn < 0) {
            worst = dg;
            wn = (long)view.facetNbr(g);
          }
        }
        t(i, 0) = Kokkos::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        t(i, 1) = dis;
        t(i, 2) = sum;
        t(i, 3) = worst;
        t(i, 4) = (double)wn;
        k(i, 0) = (frac > 0 ? t(i, 0) + dis > frac * sum : isOpen(i)) ? 1 : 0;
        k(i, 1) = view.facetEnd(i) - view.facetBegin(i);
        k(i, 2) = nr;
      });
  auto th = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t);
  auto kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k);
  // the neighbours of every cell, MAXP per cell (a cell has at most MAXP planes, so facets)
  Kokkos::View<long*, Mem> nb("ale.diag.nbrs", (std::size_t)N * MAXP);
  Kokkos::parallel_for(
      "ale.diag.nbrs", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int q = 0;
        for (int g = view.facetBegin(i); g < view.facetEnd(i) && q < MAXP; ++g, ++q)
          nb((std::size_t)i * MAXP + q) = (long)view.facetNbr(g);
      });
  auto nbh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nb);
  std::vector<CellClosure> out;
  for (int i = 0; i < N; ++i)
    if (kh(i, 0)) {
      CellClosure c;
      c.cell = i;
      c.own = th(i, 0);
      c.dis = th(i, 1);
      c.sum = th(i, 2);
      c.worstDis = th(i, 3);
      c.worstNbr = (long)th(i, 4);
      c.facets = kh(i, 1);
      c.nonRecip = kh(i, 2);
      for (int q = 0; q < c.facets && q < MAXP; ++q)
        c.nbrs.push_back(nbh((std::size_t)i * MAXP + q));
      out.push_back(c);
    }
  return out;
}

/// Diagnostic (WO-S1.3a-r): continue the forced rounds of a finished frame `f` on the engine that
/// built it (standing at `f.pos`) until no round is due or `f.reciprocity.rounds` reaches
/// `maxRounds`; `f` is published again after each round and holds the last state. Returns the
/// log of the continued rounds (open[0] = the frame as it was). The rounds are those of
/// `publishWithReciprocity`; the frame's changed set is not tracked.
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
RoundLog continueRounds(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                        MeshFrame<Real>& f, const Real L[3], const FrameOptions<Real>& opt,
                        int maxRounds) {
  using Mem = peclet::core::MemSpace;
  const int N = engine.N;
  Kokkos::View<int*, Mem> mask("ale.diag.forcedMask", N), list("ale.diag.forced", N);
  detail::publish(engine, f, L, opt, mask);
  RoundLog log;
  log.open.push_back(f.reciprocity.numOpenCells);
  log.maxNonRecipArea.push_back(f.reciprocity.maxNonRecipArea);
  while (f.reciprocity.rounds < maxRounds) {
    const bool recipDue = opt.enforceReciprocity && f.reciprocity.maxNonRecipArea > 0;
    const bool closureDue = opt.enforceClosure && f.reciprocity.numOpenCells > 0;
    if (!recipDue && !closureDue)
      break;
    const int nF = detail::maskToList(mask, N, list);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                 Kokkos::subview(list, std::make_pair(0, nF)));
    log.forced.emplace_back(h.data(), h.data() + nF);
    const RepairStats st = engine.step(f.pos, list, nF);
    if (detail::isCold(st))
      detail::stepAtRest(engine, f.pos);
    f.reciprocity.rounds += 1;
    detail::publish(engine, f, L, opt, mask);
    log.open.push_back(f.reciprocity.numOpenCells);
    log.maxNonRecipArea.push_back(f.reciprocity.maxNonRecipArea);
  }
  return log;
}

/// The frame after a cold `engine.rebuild(pos)`: every cell changed. The power weights are the
/// engine's (`setWeights` before the call).
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
MeshFrame<Real> initialFrame(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                             const Real L[3], const FrameOptions<Real>& opt = {}) {
  detail::checkSingleDomain(engine);
  MeshFrame<Real> f;
  detail::Clock clk(opt.timed);
  engine.rebuild(pos);
  // One repair step at the same positions: it rebuilds nothing, but it leaves in the engine the
  // cell volume of the kernel every later step evaluates. The cold build's volume comes from the
  // clipping kernel; the two agree bitwise on the host, but a device compiler may contract
  // a·b + c differently in each (nvcc: up to 6 ulp), and the Eulerian limit (frames at rest equal
  // the initial frame bitwise, V0.1) needs one code path. Design note §11 A11.
  detail::stepAtRest(engine, pos);
  f.timings.repair = clk.lap();
  f.pos = detail::snapshot(pos, "ale.frame.pos");
  if constexpr (Weighted)
    f.weight = detail::snapshot(engine.weight, "ale.frame.weight");
  f.coldBuild = true;
  f.numRebuilt = engine.nProc;
  detail::publishWithReciprocity(engine, pos, f, L, opt);
  return f;
}

/// The frame after a repair `engine.step(pos)` from the engine's present state (and, if that step
/// was a cold build, the step at rest of §11 A11). The power weights are the engine's.
template <class Real, int MAXP, int MAXT, bool Weighted, class Sdf>
MeshFrame<Real> advanceFrame(MovingTessellation<Real, MAXP, MAXT, Weighted, Sdf>& engine,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                             const Real L[3], const FrameOptions<Real>& opt = {}) {
  detail::checkSingleDomain(engine);
  MeshFrame<Real> f;
  detail::Clock clk(opt.timed);
  f.stats = engine.step(pos);
  f.coldBuild = detail::isCold(f.stats);
  if (f.coldBuild)
    detail::stepAtRest(engine, pos);
  f.timings.repair = clk.lap();
  f.pos = detail::snapshot(pos, "ale.frame.pos");
  if constexpr (Weighted)
    f.weight = detail::snapshot(engine.weight, "ale.frame.weight");
  f.numRebuilt =
      f.coldBuild ? engine.nProc : detail::countNonzero(engine.changedMask(), engine.nProc);
  detail::publishWithReciprocity(engine, pos, f, L, opt);
  return f;
}

}  // namespace peclet::voro::ale

#endif  // PECLET_VORO_ALE_FRAME_HPP
