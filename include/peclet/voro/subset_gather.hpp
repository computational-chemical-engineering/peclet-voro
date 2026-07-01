/**
 * @file device/subset_gather.hpp
 * \brief Subset gather (Part II, Phase 1) — the cold-build cell kernel restricted to an arbitrary
 *        index list, reusing the per-step grid. The device analogue of the legacy
 *        NbrList::setupSubset, and the single primitive both repair passes will call (Phase 2).
 *
 * Given a TessGrid built once for the current positions (peclet/voro/tess_grid.hpp),
 * `subsetGather` runs the EXACT same `CellBuilder::buildCell` the cold build runs — same worklist
 * gather, same ConvexCell clip — but only over the grid-sorted slots of the requested ORIGINAL seed
 * indices (mapped through grid.slotOf). It writes the resident topology store (np/nt/pnbr/tri) and
 * the cell volume at the original index, so a subset cell is byte-for-byte the cell
 * buildTessellation would produce for that index off the same grid (the gather is per-cell
 * independent; restricting the launch set changes nothing about any individual cell's computation).
 *
 * No repair loop, no two-pass driver, no candidate hints — just the gather primitive. Voronoi only
 * (the ConvexCell device path; Power is deferred with the rest of the weighted work).
 *
 * Core header: Kokkos + the tessellator (CellBuilder) + tess_grid. No physics.
 */
#ifndef PECLET_VORO_SUBSET_GATHER_HPP
#define PECLET_VORO_SUBSET_GATHER_HPP

#include <Kokkos_Core.hpp>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/tess_grid.hpp"
#include "peclet/voro/tessellator.hpp"  // CellBuilder, StatusBit, NoSdf

namespace peclet::voro {

/// Output bundle of a subset gather: the per-cell StatusBit mask, addressed by ORIGINAL seed index
/// (size N, only the requested subset is written; the topology store + the caller's cellVol view
/// are written in place). Flags the rare overflow/incomplete cell exactly as the cold build does.
template <class Real>
struct SubsetGatherResult {
  Kokkos::View<int*, peclet::core::MemSpace> status;  // N : per-cell StatusBit mask (subset entries written)
};

/**
 * Gather + clip the `nSubset` cells named by `indices` (original seed indices) off `grid`, writing
 * their topology into the caller's store views and their volume into the result.
 *
 * @param grid     a TessGrid built for the CURRENT positions (buildTessGrid). Reused, not rebuilt.
 * @param indices  device view of nSubset ORIGINAL seed indices to (re)build (values in [0,N)).
 * @param nSubset  number of indices to process.
 * @param outNp,outNt,outPnbr,outTri  topology store views, each sized as in TopologyStore::alloc(N)
 *                 (N, N, N*MAXP, N*MAXT) with MAXP=CellBuilder::kMaxP, MAXT=CellBuilder::kMaxT.
 * @param cellVol  N-sized view; the volume of each rebuilt cell is written at its original index.
 * @param withForceGeom  if true also compute the per-facet area/dV/connector (kept for parity with
 *                 the cold build); the facet CSR itself is not published here (the store is).
 * Default false: the store + volume are what Phase-1 needs (geometry is re-derived on reload).
 */
template <class Real, bool Weighted = false, bool TrackAdj = false, class Sdf = NoSdf>
SubsetGatherResult<Real> subsetGather(
    const TessGrid<Real>& grid, const Kokkos::View<int*, peclet::core::MemSpace>& indices, int nSubset,
    const Kokkos::View<int*, peclet::core::MemSpace>& outNp, const Kokkos::View<int*, peclet::core::MemSpace>& outNt,
    const Kokkos::View<int*, peclet::core::MemSpace>& outPnbr,
    const Kokkos::View<unsigned*, peclet::core::MemSpace>& outTri,
    const Kokkos::View<Real*, peclet::core::MemSpace>& cellVol,
    const Kokkos::View<unsigned char*, peclet::core::MemSpace>& outPoke4 = {}, Sdf sdf = {},
    bool withForceGeom = false) {
  using peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Builder = CellBuilder<Real, Weighted, Sdf, TrackAdj>;
  static_assert(!Weighted,
                "subsetGather: Power/Laguerre on the ConvexCell device path is deferred.");
  const int N = grid.N;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;

  SubsetGatherResult<Real> res;
  res.status = Kokkos::View<int*, MemSpace>("subset.status", N);  // zero-init (unwritten = kOk)

  // Throwaway facet over-buffer: buildCell reserves a contiguous CSR range per cell and writes the
  // neighbour id (+ optional geometry) there. Sized for the subset (mean ~15.5 faces/cell +
  // headroom); the topology store is the published Phase-1 output, this buffer is just buildCell's
  // scratch.
  Kokkos::View<int*, MemSpace> facetCount("subset.facetCount", N);
  Kokkos::View<int*, MemSpace> cellFacetBase("subset.cellFacetBase", N);
  constexpr size_t kMeanFacets = 18;
  const size_t facetCap = (size_t)(nSubset > 0 ? nSubset : 1) * kMeanFacets;
  Kokkos::View<int*, MemSpace> oNbr(view_alloc(std::string("subset.oNbr"), WithoutInitializing),
                                    facetCap);
  Kokkos::View<Real*, MemSpace> oArea(view_alloc(std::string("subset.oArea"), WithoutInitializing),
                                      facetCap * 3);
  Kokkos::View<Real*, MemSpace> oDV(view_alloc(std::string("subset.oDV"), WithoutInitializing),
                                    facetCap * 3);
  Kokkos::View<Real*, MemSpace> oConn(view_alloc(std::string("subset.oConn"), WithoutInitializing),
                                      facetCap * 3);
  Kokkos::View<int*, MemSpace> facetCursor("subset.facetCursor", 1);

  // Empty candidate-list outputs (skin emission is not wanted for the subset gather).
  Kokkos::View<int*, MemSpace> noCand, noCandCnt;
  Kokkos::View<Real*, MemSpace> noW;  // unused weights (Weighted==false)

  Builder op{grid.binned,
             grid.posSorted,
             noW,
             grid.gidSorted,
             grid.cellStart,
             grid.wlOff,
             grid.wlRmin,
             res.status,
             cellVol,
             facetCount,
             cellFacetBase,
             oNbr,
             oArea,
             oDV,
             oConn,
             facetCursor,
             grid.icx,
             grid.icy,
             grid.icz,
             grid.Lx,
             grid.Ly,
             grid.Lz,
             grid.minCsz,
             grid.dimx,
             grid.dimy,
             grid.dimz,
             grid.sw,
             grid.nOff,
             grid.wlS,
             grid.useMorton,
             grid.haveGid,
             withForceGeom,
             facetCap,
             sdf,
             outNp,
             outNt,
             outPnbr,
             outTri,
             outPoke4,
             noCand,
             noCandCnt,
             /*emitTopo=*/true,
             /*emitCand=*/false,
             /*candCap=*/0};

  auto slotOf = grid.slotOf;
  auto idx = indices;
  Kokkos::parallel_for(
      "subset.build", Kokkos::RangePolicy<Exec>(0, nSubset), KOKKOS_LAMBDA(const int s) {
        const int i = idx(s);  // original seed index to rebuild
        op.buildCell(
            slotOf(i));  // grid-sorted slot of that seed; writes outputs at binned(slot)==i
      });
  Kokkos::fence();
  return res;
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_SUBSET_GATHER_HPP
