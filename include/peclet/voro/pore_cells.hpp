/**
 * @file pore_cells.hpp
 * @brief The SDF-walled interstitial (pore-space) Voronoi cells of a seeding as exported
 *        geometry, on the device: the clipped polyhedra of every seed (VTK_POLYHEDRON layout)
 *        and their cross-section by a plane — `peclet.voro.pore_mesh.sdf_voronoi_cells` /
 *        `sdf_voronoi_section` (QUALITY_PLAN G.7, the device-first directive).
 *
 * Both entry points run the tessellator's own gather (CellBuilder::gatherCell over the
 * counting-sort grid + presorted worklist of tess_grid.hpp — the exact worklist walk of the cold
 * build, at the resident capacity kMaxPlanes / kMaxTriangles) followed by the SDF clip
 * (clipCellAgainstSdf, as CellBuilder::finishCell applies it), one thread per cell, and export
 * the finished ConvexCell with the count → scan → fill pattern: a first pass over the cells
 * counts each cell's output (points and face-list entries, or section vertices), an exclusive
 * scan in SEED order fixes every cell's offset, and a second pass rebuilds the cell and writes
 * it in place. The output is therefore deterministic (seed order, independent of thread
 * scheduling) and no over-buffer is needed; the price is the second gather, which is the same
 * device-parallel work again.
 *
 * Capacity: the production 64/112 (kMaxPlanes / kMaxTriangles), the same the cold build clips
 * SDF-walled cells at; a cell that overflows it is skipped and counted (numOverflow). The larger
 * pore capacity (kPoreMaxPlanes / kPoreMaxTriangles = 128/256, the host oracle's) is NOT used
 * here on purpose: measured 2026-09-10 on an RTX 5080, the SDF clip of a 128/256 cell is WRONG
 * on the CUDA backend (every wall cell off by up to 22 % in volume, deterministic; the same
 * kernel at 64/112 matches the host oracle to 6e-11, and 128/256 is exact on OpenMP) — an open
 * CUDA-only defect of clipCellAgainstSdf at that capacity, recorded in the G.7 report.
 *
 * Layer 2 (tessellator + sdf + convex_cell).
 */
#ifndef PECLET_VORO_PORE_CELLS_HPP
#define PECLET_VORO_PORE_CELLS_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tess_grid.hpp"
#include "peclet/voro/tessellator.hpp"

namespace peclet::voro {

/// The clipped polyhedra, packed as flat arrays in seed order (VTK_POLYHEDRON layout).
template <class Real>
struct PoreCellsResult {
  using MemSpace = peclet::core::MemSpace;
  Kokkos::View<Real*, MemSpace> points;  ///< 3·numPoints, x-fastest, world frame
  /// Per exported cell: nf, then per face (m, m global point ids, CCW about the outward normal).
  Kokkos::View<int64_t*, MemSpace> faces;
  Kokkos::View<int64_t*, MemSpace> faceOffset;  ///< numCells+1: each cell's range in `faces`
  Kokkos::View<Real*, MemSpace> volume;         ///< numCells: the clipped cell volume
  Kokkos::View<int*, MemSpace> boundary;        ///< numCells: 1 where a face is a wall facet
  Kokkos::View<int*, MemSpace> seed;            ///< numCells: the seed index of each cell
  long numOverflow = 0;                         ///< seeds skipped because their cell hit a capacity
  long numIncomplete = 0;  ///< exported cells whose gather coverage did not close (raise sw)
};

/// The cross-section polygons, packed in seed order (vertices in the world frame, all on the
/// plane, CCW about the plane normal).
template <class Real>
struct PoreSectionResult {
  using MemSpace = peclet::core::MemSpace;
  Kokkos::View<Real*, MemSpace> verts;      ///< 3·numVerts
  Kokkos::View<int64_t*, MemSpace> offset;  ///< numPolygons+1: each polygon's vertex range
  Kokkos::View<Real*, MemSpace> volume;     ///< numPolygons: the 3-D volume of the cut cell
  Kokkos::View<int*, MemSpace> seed;        ///< numPolygons: the seed index of the cut cell
  long numOverflow = 0;                     ///< seeds skipped because their cell hit a capacity
  long numIncomplete = 0;                   ///< cut cells whose gather coverage did not close
};

namespace detail {

/// Export rule of one finished cell (shared with the host oracle in the bindings): a cell is
/// exported when it is neither empty nor overflowed and has at least four vertices and four
/// polygon faces. Counts its points (alive dual triangles) and face-list entries
/// (1 + Σ_faces (1 + m)), and whether one of its faces is a wall facet.
template <class Cell>
KOKKOS_INLINE_FUNCTION bool poreCellCount(const Cell& c, int& nPts, int& nEntries, bool& wall) {
  if (c.empty() || c.overflow)
    return false;
  nPts = 0;
  for (int t = 0; t < c.nt; ++t)
    if (c.alive[t])
      ++nPts;
  if (nPts < 4)
    return false;
  nEntries = 1;
  wall = false;
  int nf = 0;
  int fidx[Cell::MAXFV];
  for (int k = 0; k < c.np; ++k) {
    const int m = c.faceOrderedIdx(k, fidx);
    if (m < 3)
      continue;
    ++nf;
    nEntries += 1 + m;
    if (c.pnbr[k] == kBoundaryFacet)
      wall = true;
  }
  return nf >= 4;
}

/// Write one exported cell: its points (world frame, seed + vertex) at pts[3·(ptBase + q)] and
/// its face list at faces[fBase …) with global point ids ptBase + q.
template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION void poreCellFill(const Cell& c, const Real seed[3], int64_t ptBase,
                                         int64_t fBase, Real* pts, int64_t* faces) {
  int triToPt[Cell::kMaxT];
  int q = 0;
  for (int t = 0; t < c.nt; ++t) {
    triToPt[t] = -1;
    if (!c.alive[t])
      continue;
    triToPt[t] = q;
    const int64_t o = 3 * (ptBase + q);
    pts[o] = seed[0] + c.vx[t];
    pts[o + 1] = seed[1] + c.vy[t];
    pts[o + 2] = seed[2] + c.vz[t];
    ++q;
  }
  int fidx[Cell::MAXFV];
  int nf = 0;
  for (int k = 0; k < c.np; ++k)
    if (c.faceOrderedIdx(k, fidx) >= 3)
      ++nf;
  int64_t w = fBase;
  faces[w++] = nf;
  for (int k = 0; k < c.np; ++k) {
    const int m = c.faceOrderedIdx(k, fidx);
    if (m < 3)
      continue;
    faces[w++] = m;
    for (int i = 0; i < m; ++i)
      faces[w++] = ptBase + triToPt[fidx[i]];
  }
}

/// The tessellator's gather at the pore capacity, without its publish: the CellBuilder holds the
/// grid + worklist inputs only (every output view empty), and `clipped` hands back the finished
/// cell — the worklist walk of CellBuilder::gatherCell (early wall clip + neighbour planes), then
/// the SDF clip exactly as CellBuilder::finishCell applies it.
template <class Real, class Sdf, int MAXP, int MAXT>
struct PoreCellGather {
  using Builder = CellBuilder<Real, false, Sdf, false, MAXP, MAXT>;
  using Cell = typename Builder::Cell;
  Builder op;

  static PoreCellGather make(const TessGrid<Real>& grid, const Sdf& sdf) {
    PoreCellGather g{};
    Builder& op = g.op;
    op.binned = grid.binned;
    op.posSorted = grid.posSorted;
    op.wSorted = grid.wSorted;
    op.gidSorted = grid.gidSorted;
    op.cellStart = grid.cellStart;
    op.wlOff = grid.wlOff;
    op.wlRmin = grid.wlRmin;
    op.icx = grid.icx;
    op.icy = grid.icy;
    op.icz = grid.icz;
    op.Lx = grid.Lx;
    op.Ly = grid.Ly;
    op.Lz = grid.Lz;
    op.minCsz = grid.minCsz;
    op.wMaxAll = Real(0);
    op.dimx = grid.dimx;
    op.dimy = grid.dimy;
    op.dimz = grid.dimz;
    op.sw = grid.sw;
    op.nOff = grid.nOff;
    op.wlS = grid.wlS;
    op.useMorton = grid.useMorton;
    op.haveGid = grid.haveGid;
    op.sdf = sdf;
    return g;
  }

  /// The finished cell of grid slot `pi` in `c`, its seed (world) in `seedW`; returns the
  /// StatusBit mask (kEmpty / kOverflow mean "no cell", kIncomplete flags an unclosed gather).
  KOKKOS_INLINE_FUNCTION int clipped(int pi, Cell& c, Real seedW[3]) const {
    Real wSelf;
    int ncRec, nnRec;
    op.gatherCell(pi, c, seedW[0], seedW[1], seedW[2], wSelf, ncRec, nnRec);
    if (c.overflow)
      return kOverflow;
    clipCellAgainstSdf<Real, MAXP, MAXT>(c, seedW, op.sdf);
    if (c.overflow)
      return kOverflow;
    if (c.empty())
      return kEmpty;
    // Completeness of the EXPORTED (clipped) cell: every seed that could cut it lies within twice
    // its reach, and the worklist examined everything within the walked window (grid.sw blocks,
    // the clamped window the grid was actually built with). The cold build judges the un-clipped
    // cell, which near a solid extends across the hole the clip removes — a spurious flag here.
    const Real covSq = Real(op.sw) * op.minCsz * Real(op.sw) * op.minCsz;
    return (covSq > Voronoi::template blockReachSq<Real>(c.maxVertexRsq(), wSelf, op.wMaxAll))
               ? kOk
               : kIncomplete;
  }
};

/// Exclusive scan of a per-seed count into offsets (N+1 entries; the last is the total).
template <class Count, class Off>
inline void exclusiveScan(const Count& count, const Off& off, int N) {
  using Exec = peclet::core::ExecSpace;
  using T = typename Off::value_type;
  Kokkos::parallel_scan(
      "pore.scan", Kokkos::RangePolicy<Exec>(0, N + 1),
      KOKKOS_LAMBDA(const int i, T& acc, const bool fin) {
        const T v = (i < N) ? T(count(i)) : T(0);
        if (fin)
          off(i) = acc;
        acc += v;
      });
}

template <class Off>
inline typename Off::value_type lastOf(const Off& off, int N) {
  typename Off::value_type v;
  Kokkos::deep_copy(v, Kokkos::subview(off, N));
  return v;
}

template <class Status>
inline void statusCounts(const Status& status, int N, long& overflow, long& incomplete) {
  using Exec = peclet::core::ExecSpace;
  long ov = 0, inc = 0;
  Kokkos::parallel_reduce(
      "pore.overflow", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, long& a) { a += (status(i) & kOverflow) ? 1 : 0; }, ov);
  Kokkos::parallel_reduce(
      "pore.incomplete", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, long& a) {
        a += ((status(i) & kIncomplete) && !(status(i) & (kEmpty | kOverflow))) ? 1 : 0;
      },
      inc);
  overflow = ov;
  incomplete = inc;
}

}  // namespace detail

/**
 * The SDF-clipped interstitial polyhedra of every seed, on the device.
 *
 * @param posFlat  device seed positions, x-fastest (3·i + k), in [0, L).
 * @param N        seed count.
 * @param L        periodic box extent.
 * @param sdf      the wall solid (any provider with eval()/gradH(); NoSdf gives the bare cells).
 * @param sw       the gather window (grid blocks per axis; kPoreSearchWindow in the bindings).
 * Seeds inside the solid (empty cell) and cells that overflow the capacity are skipped; the
 * result counts the latter, and the cells whose gather coverage did not close (kIncomplete).
 */
template <class Real, class Sdf, int MAXP = kMaxPlanes, int MAXT = kMaxTriangles>
PoreCellsResult<Real> buildPoreCells(const Kokkos::View<Real*, peclet::core::MemSpace>& posFlat,
                                     int N, const Real L[3], const Sdf& sdf,
                                     int sw = kSearchWindow) {
  using MemSpace = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Gather = detail::PoreCellGather<Real, Sdf, MAXP, MAXT>;
  using Cell = typename Gather::Cell;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  Kokkos::View<Real*, MemSpace> noW;
  Kokkos::View<long*, MemSpace> noGid;
  auto grid = buildTessGrid<Real, false>(posFlat, noW, N, L, sw, N, noGid, nullptr);
  const Gather g = Gather::make(grid, sdf);
  auto binned = grid.binned;

  // Count pass (written at the ORIGINAL seed index, so the scan below is in seed order).
  Kokkos::View<int*, MemSpace> status("pore.status", N), emit("pore.emit", N), nPts("pore.nPts", N),
      nEnt("pore.nEnt", N), wall("pore.wall", N);
  Kokkos::parallel_for(
      "pore.count", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
        const int i = binned(pi);
        Cell c;
        Real s[3];
        const int st = g.clipped(pi, c, s);
        status(i) = st;
        if (st & (kEmpty | kOverflow))
          return;
        int np = 0, ne = 0;
        bool w = false;
        if (!detail::poreCellCount(c, np, ne, w))
          return;
        emit(i) = 1;
        nPts(i) = np;
        nEnt(i) = ne;
        wall(i) = w ? 1 : 0;
      });
  Kokkos::View<int*, MemSpace> cellOff("pore.cellOff", N + 1);
  Kokkos::View<int64_t*, MemSpace> ptOff("pore.ptOff", N + 1), entOff("pore.entOff", N + 1);
  detail::exclusiveScan(emit, cellOff, N);
  detail::exclusiveScan(nPts, ptOff, N);
  detail::exclusiveScan(nEnt, entOff, N);
  const int nCells = detail::lastOf(cellOff, N);
  const int64_t nPoints = detail::lastOf(ptOff, N), nEntries = detail::lastOf(entOff, N);

  PoreCellsResult<Real> res;
  res.points = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("pore.points"), WithoutInitializing), 3 * (size_t)nPoints);
  res.faces = Kokkos::View<int64_t*, MemSpace>(
      view_alloc(std::string("pore.faces"), WithoutInitializing), (size_t)nEntries);
  res.faceOffset = Kokkos::View<int64_t*, MemSpace>("pore.faceOffset", nCells + 1);
  res.volume = Kokkos::View<Real*, MemSpace>("pore.volume", nCells);
  res.boundary = Kokkos::View<int*, MemSpace>("pore.boundary", nCells);
  res.seed = Kokkos::View<int*, MemSpace>("pore.seed", nCells);
  auto pts = res.points;
  auto faces = res.faces;
  auto faceOffset = res.faceOffset;
  auto volume = res.volume;
  auto boundary = res.boundary;
  auto seed = res.seed;
  Kokkos::deep_copy(Kokkos::subview(faceOffset, nCells), nEntries);
  // Fill pass: the same gather again, written at the scanned offsets.
  Kokkos::parallel_for(
      "pore.fill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
        const int i = binned(pi);
        if (!emit(i))
          return;
        Cell c;
        Real s[3];
        g.clipped(pi, c, s);
        const int ci = cellOff(i);
        detail::poreCellFill(c, s, ptOff(i), entOff(i), pts.data(), faces.data());
        faceOffset(ci) = entOff(i);
        volume(ci) = c.volumePerVertex();
        boundary(ci) = wall(i);
        seed(ci) = i;
      });
  Kokkos::fence();
  detail::statusCounts(status, N, res.numOverflow, res.numIncomplete);
  return res;
}

/**
 * The cross-section of the SDF-clipped interstitial mesh by the plane through `point` with
 * `normal`: every cell cut directly (ConvexCell::sectionPolygon, from the dual edges, so the
 * polygons tile the section exactly). Same gather and skip rules as buildPoreCells; a cell the
 * plane misses (or grazes at fewer than three crossings) yields no polygon.
 */
template <class Real, class Sdf, int MAXP = kMaxPlanes, int MAXT = kMaxTriangles>
PoreSectionResult<Real> buildPoreSection(const Kokkos::View<Real*, peclet::core::MemSpace>& posFlat,
                                         int N, const Real L[3], const Sdf& sdf,
                                         const Real point[3], const Real normal[3],
                                         int sw = kSearchWindow) {
  using MemSpace = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Gather = detail::PoreCellGather<Real, Sdf, MAXP, MAXT>;
  using Cell = typename Gather::Cell;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  Kokkos::View<Real*, MemSpace> noW;
  Kokkos::View<long*, MemSpace> noGid;
  auto grid = buildTessGrid<Real, false>(posFlat, noW, N, L, sw, N, noGid, nullptr);
  const Gather g = Gather::make(grid, sdf);
  auto binned = grid.binned;
  const Real px = point[0], py = point[1], pz = point[2];
  const Real ux = normal[0], uy = normal[1], uz = normal[2];

  Kokkos::View<int*, MemSpace> status("pore.status", N), nV("pore.nV", N), emit("pore.emit", N);
  Kokkos::parallel_for(
      "pore.section.count", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
        const int i = binned(pi);
        Cell c;
        Real s[3];
        const int st = g.clipped(pi, c, s);
        status(i) = st;
        if (st & (kEmpty | kOverflow))
          return;
        const Real p0[3] = {px - s[0], py - s[1], pz - s[2]};  // the plane in the cell frame
        const Real u[3] = {ux, uy, uz};
        Real spx[Cell::MAXSV], spy[Cell::MAXSV], spz[Cell::MAXSV];
        const int m = c.sectionPolygon(p0, u, spx, spy, spz);
        if (m < 3)
          return;
        emit(i) = 1;
        nV(i) = m;
      });
  Kokkos::View<int*, MemSpace> polyOff("pore.polyOff", N + 1);
  Kokkos::View<int64_t*, MemSpace> vOff("pore.vOff", N + 1);
  detail::exclusiveScan(emit, polyOff, N);
  detail::exclusiveScan(nV, vOff, N);
  const int nPoly = detail::lastOf(polyOff, N);
  const int64_t nVerts = detail::lastOf(vOff, N);

  PoreSectionResult<Real> res;
  res.verts = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("pore.verts"), WithoutInitializing), 3 * (size_t)nVerts);
  res.offset = Kokkos::View<int64_t*, MemSpace>("pore.offset", nPoly + 1);
  res.volume = Kokkos::View<Real*, MemSpace>("pore.volume", nPoly);
  res.seed = Kokkos::View<int*, MemSpace>("pore.seed", nPoly);
  auto verts = res.verts;
  auto offset = res.offset;
  auto volume = res.volume;
  auto seed = res.seed;
  Kokkos::deep_copy(Kokkos::subview(offset, nPoly), nVerts);
  Kokkos::parallel_for(
      "pore.section.fill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
        const int i = binned(pi);
        if (!emit(i))
          return;
        Cell c;
        Real s[3];
        g.clipped(pi, c, s);
        const Real p0[3] = {px - s[0], py - s[1], pz - s[2]};
        const Real u[3] = {ux, uy, uz};
        Real spx[Cell::MAXSV], spy[Cell::MAXSV], spz[Cell::MAXSV];
        const int m = c.sectionPolygon(p0, u, spx, spy, spz);
        const int64_t base = vOff(i);
        for (int k = 0; k < m; ++k) {
          verts(3 * (base + k)) = s[0] + spx[k];
          verts(3 * (base + k) + 1) = s[1] + spy[k];
          verts(3 * (base + k) + 2) = s[2] + spz[k];
        }
        const int p = polyOff(i);
        offset(p) = base;
        volume(p) = c.volumePerVertex();
        seed(p) = i;
      });
  Kokkos::fence();
  detail::statusCounts(status, N, res.numOverflow, res.numIncomplete);
  return res;
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_PORE_CELLS_HPP
