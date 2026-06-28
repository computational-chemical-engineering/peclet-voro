/**
 * @file device/tessellator.hpp
 * \brief Full-system Voronoi/Power tessellation on device (migration Phase 3 + 7).
 *
 * Drives the Phase-2 cutter over every seed in a single device pass and writes
 * the result straight into the published CSR (TessellationView):
 *   1. cell-linked grid built by counting sort (parallel_scan offsets);
 *   2. per-seed build — gather the seeds in the surrounding (2·SW+1)³ grid block
 *      (periodic, minimal-image), then clip the cuboid closest-first. For the
 *      unweighted case a per-cell selection of the nearest unprocessed candidate
 *      stops as soon as it is beyond the security radius (2·rSqMax), so only the
 *      handful that can cut are touched (Phase-7 speedup, ~3× over cutting the
 *      whole block); Power applies every candidate (no security early-out). The
 *      cut is idempotent, so duplicate/periodic images are harmless;
 *   3. exclusive scan of per-cell facet counts -> CSR offsets;
 *   4. compaction of the per-cell facet records into the packed CSR.
 *
 * Each cell carries a security check: with grid cell size h and block half-width
 * SW the gathered region inscribes a sphere of radius SW·h, so the cell is
 * provably complete iff (SW·h)² > 4·rSqMax (the legacy rSqMin>4·rSq criterion).
 * Cells that fail it (or overflow the 128 cap) are flagged in the status array
 * for a host/large-scratch fallback — the plan's overflow path.
 *
 * Core header: Kokkos + transport-core + the cutter, no physics.
 */
#ifndef VORFLOW_DEVICE_TESSELLATOR_HPP
#define VORFLOW_DEVICE_TESSELLATOR_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "morton/morton.hpp"  // suite spatial-index primitive (after Kokkos_Core: MORTON_HD->KOKKOS_FUNCTION)
#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"
#include "vorflow/device/sdf.hpp"
#include "vorflow/device/tess_grid.hpp"  // counting-sort grid + worklist (kWlOffBias, morton3, TessGrid)
#include "vorflow/tessellation_view.hpp"

namespace vor {
namespace device {

/// Per-cell status bits written by the build pass.
enum StatusBit { kOk = 0, kOverflow = 1, kEmpty = 2, kIncomplete = 4 };

template <class Real>
struct TessellatorResult {
  TessellationView<Real> view;
  Kokkos::View<int*, tpx::MemSpace> status;  // per-cell StatusBit mask
};

/**
 * Per-cell build on the compact ConvexCell (dual-triangle) representation — one thread per
 * cell (RangePolicy; the cell lives in the per-thread register/local frame). Holds the
 * grid-sorted inputs, the published outputs, and the grid scalars; buildCell clips the
 * worklist neighbours on the fly and emits the per-facet CSR (neighbour id + area vector +
 * dV + connector) via ConvexCell::facetGeometry — the same quantities the retired half-edge
 * ScratchCell published, now from the leaner cell whose geometry is a cheap separate pass
 * (so re-eval over a fixed topology is possible: ConvexCell::reevalGeometry).
 */
template <class Real, bool Weighted, class Sdf>
struct CellBuilder {
  using MemSpace = tpx::MemSpace;
  static constexpr int kMaxP = 64;     // plane cap (overflow -> kOverflow)
  static constexpr int kMaxT = 112;    // dual-triangle (vertex) cap
  static constexpr int MAXF_TMP = 50;  // max facets one cell may publish
  using Cell = ConvexCell<Real, kMaxP, kMaxT>;

  // Grid-sorted inputs (read).
  Kokkos::View<int*, MemSpace> binned;
  Kokkos::View<Real*, MemSpace> posSorted;
  Kokkos::View<Real*, MemSpace> wSorted;
  Kokkos::View<gid_t*, MemSpace> gidSorted;
  Kokkos::View<int*, MemSpace> cellStart;
  // Per-sub-position worklist (both backends): for the wlS^3 sub-region a seed lands in, the
  // block offsets (packed (dx,dy,dz), kWlOffBias) sorted by nearest-corner dist^2 (wlRmin,
  // absolute). The Voronoi gather walks the presorted list and breaks once wlRmin exceeds the
  // security radius — a table lookup, no per-block geometry. ConvexCell is lean enough that
  // the worklist runs on the GPU too (its geometry pass is cheap, so no occupancy penalty).
  Kokkos::View<int*, MemSpace> wlOff;
  Kokkos::View<Real*, MemSpace> wlRmin;
  // Published outputs (write).
  Kokkos::View<int*, MemSpace> status;
  Kokkos::View<Real*, MemSpace> cellVol;
  Kokkos::View<int*, MemSpace> facetCount;
  Kokkos::View<int*, MemSpace> cellFacetBase;
  Kokkos::View<int*, MemSpace> oNbr;
  Kokkos::View<Real*, MemSpace> oArea, oDV, oConn;
  Kokkos::View<int*, MemSpace> facetCursor;
  // Grid scalars.
  Real icx, icy, icz, Lx, Ly, Lz, minCsz;
  int dimx, dimy, dimz, sw, nOff, wlS;
  bool useMorton, haveGid, withForceGeom;
  size_t facetCap;
  Sdf sdf;
  // Optional Part-II (moving-points) outputs — emitted only when the views are sized (else left empty so
  // existing callers are unaffected): the compact resident TOPOLOGY store (np/nt + pnbr/packed-triangles at
  // kMaxP/kMaxT strides, written from the final clipped cell) and the per-cell CANDIDATE (skin) list = the
  // neighbour ids this cell examined (within the worklist's security reach), capped at candCap. Together they
  // let a later step re-evaluate geometry (ConvexCell::reevalGeometry) and locally repair off the skin list
  // without re-gathering. See vorflow/docs/voronoi_dynamic_update_study.md.
  Kokkos::View<int*, MemSpace> oNp, oNt, oTopoPnbr;
  Kokkos::View<unsigned*, MemSpace> oTri;
  Kokkos::View<int*, MemSpace> oCand, oCandCnt;
  bool emitTopo, emitCand;
  int candCap;

  /// Minimal-image relative vector from the seed at (pix,piy,piz) to sorted seed q.
  KOKKOS_INLINE_FUNCTION void relVec(int q, Real pix, Real piy, Real piz, Real pv[3]) const {
    Real rx = posSorted(3 * q + 0) - pix, ry = posSorted(3 * q + 1) - piy,
         rz = posSorted(3 * q + 2) - piz;
    const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
    pv[0] = rx > Lxh ? rx - Lx : (rx < -Lxh ? rx + Lx : rx);
    pv[1] = ry > Lyh ? ry - Ly : (ry < -Lyh ? ry + Ly : ry);
    pv[2] = rz > Lzh ? rz - Lz : (rz < -Lzh ? rz + Lz : rz);
  }

  /// Linear/Morton index of the grid cell at raw (periodic-unwrapped) offset.
  KOKKOS_INLINE_FUNCTION int gridCell(int rgx, int rgy, int rgz) const {
    int gx = rgx < 0 ? rgx + dimx : (rgx >= dimx ? rgx - dimx : rgx);
    int gy = rgy < 0 ? rgy + dimy : (rgy >= dimy ? rgy - dimy : rgy);
    int gz = rgz < 0 ? rgz + dimz : (rgz >= dimz ? rgz - dimz : rgz);
    return useMorton ? morton3(gx, gy, gz) : (gx + gy * dimx + gz * dimx * dimy);
  }

  /// Home grid cell index (cx,cy,cz) of seed (pix,piy,piz).
  KOKKOS_INLINE_FUNCTION void homeCell(Real pix, Real piy, Real piz, int& cx, int& cy,
                                       int& cz) const {
    cx = ((((int)Kokkos::floor(pix * icx)) % dimx) + dimx) % dimx;
    cy = ((((int)Kokkos::floor(piy * icy)) % dimy) + dimy) % dimy;
    cz = ((((int)Kokkos::floor(piz * icz)) % dimz) + dimz) % dimz;
  }

  /// Flat base offset into the worklist tables for the wlS^3 sub-region the seed lands in.
  KOKKOS_INLINE_FUNCTION int subBase(Real pix, Real piy, Real piz) const {
    const Real fxi = pix * icx, fyi = piy * icy, fzi = piz * icz;
    int sx = (int)((fxi - Kokkos::floor(fxi)) * Real(wlS));
    int sy = (int)((fyi - Kokkos::floor(fyi)) * Real(wlS));
    int sz = (int)((fzi - Kokkos::floor(fzi)) * Real(wlS));
    sx = sx < 0 ? 0 : (sx >= wlS ? wlS - 1 : sx);
    sy = sy < 0 ? 0 : (sy >= wlS ? wlS - 1 : sy);
    sz = sz < 0 ? 0 : (sz >= wlS ? wlS - 1 : sz);
    return (sx + wlS * (sy + wlS * sz)) * nOff;
  }

  /// Decode worklist entry g (relative to base) into a grid-cell index.
  KOKKOS_INLINE_FUNCTION int worklistCell(int base, int g, int cx, int cy, int cz) const {
    const int packed = wlOff(base + g);
    const int rgx = cx + ((packed & 0xFF) - kWlOffBias);
    const int rgy = cy + (((packed >> 8) & 0xFF) - kWlOffBias);
    const int rgz = cz + (((packed >> 16) & 0xFF) - kWlOffBias);
    return gridCell(rgx, rgy, rgz);
  }

  /// Finish a built cell: completeness flag (judged on the un-clipped cell), optional SDF
  /// boundary clip, status/volume, and the per-facet CSR write (one atomic reservation into
  /// the over-buffer). covSq is the attained coverage^2; covSq > 4*rSqMax => complete.
  KOKKOS_INLINE_FUNCTION void finishCell(Cell& c, int i, Real pix, Real piy, Real piz,
                                         Real covSq) const {
    if (c.overflow) {
      status(i) = kOverflow;
      facetCount(i) = 0;
      cellFacetBase(i) = 0;
      cellVol(i) = Real(0);
      return;
    }
    const bool incomplete = !(covSq > Real(4) * c.maxVertexRsq());
    (void)sdf;
    if constexpr (!std::is_same_v<Sdf, NoSdf>) {
      const Real seedW[3] = {pix, piy, piz};
      clipCellAgainstSdf<Real, kMaxP, kMaxT>(c, seedW, sdf);
    }
    int st = kOk;
    if (c.overflow) st |= kOverflow;
    const bool empty = c.empty();
    if (empty) st |= kEmpty;
    if (incomplete) st |= kIncomplete;
    cellVol(i) = (empty || c.overflow) ? Real(0) : c.volumePerVertex();

    // Part-II: persist the final (post-SDF-clip) topology so a later step can re-eval/repair without a rebuild.
    if (emitTopo && !empty && !c.overflow) {
      oNp(i) = c.np;
      oNt(i) = c.nt;
      for (int k = 0; k < c.np; ++k) oTopoPnbr[(size_t)i * kMaxP + k] = c.pnbr[k];
      for (int t = 0; t < c.nt; ++t)
        oTri[(size_t)i * kMaxT + t] = (unsigned)c.t0[t] | ((unsigned)c.t1[t] << 8) |
                                      ((unsigned)c.t2[t] << 16) | ((c.alive[t] ? 1u : 0u) << 24);
    }

    // Collect this cell's live faces (a plane with >=3 incident live triangles), then reserve
    // a contiguous CSR range with one atomic and write the per-facet geometry straight in.
    int faces[MAXF_TMP];
    int nf = 0;
    if (!empty && !c.overflow) {
      for (int k = 0; k < c.np; ++k) {
        int cnt = 0;
        for (int t = 0; t < c.nt; ++t)
          if (c.alive[t] && (c.t0[t] == k || c.t1[t] == k || c.t2[t] == k)) ++cnt;
        if (cnt < 3) continue;  // not a polygon face
        if (nf >= MAXF_TMP) {
          st |= kOverflow;
          break;
        }
        faces[nf++] = k;
      }
    }
    status(i) = st;
    const int base = Kokkos::atomic_fetch_add(&facetCursor(0), nf);
    if ((size_t)base + (size_t)nf > facetCap) {
      status(i) |= kOverflow;
      facetCount(i) = 0;
      cellFacetBase(i) = 0;
      return;
    }
    facetCount(i) = nf;
    cellFacetBase(i) = base;
    for (int idx = 0; idx < nf; ++idx) {
      const int k = faces[idx];
      Real area[3] = {0, 0, 0}, dv[3] = {0, 0, 0}, conn[3];
      conn[0] = Real(2) * c.n[k][0];
      conn[1] = Real(2) * c.n[k][1];
      conn[2] = Real(2) * c.n[k][2];
      if (withForceGeom) c.facetGeometry(k, area, dv, conn);  // area vector + dV + connector
      oNbr((size_t)base + idx) = c.pnbr[k];
      const size_t o = ((size_t)base + idx) * 3;
      for (int cc = 0; cc < 3; ++cc) {
        oArea(o + cc) = area[cc];
        oDV(o + cc) = dv[cc];
        oConn(o + cc) = conn[cc];
      }
    }
  }

  /// Build the cell owning grid-sorted slot pi, writing the published outputs at the original
  /// seed index binned(pi). Worklist gather, ConvexCell clip on the fly (no candidate buffer).
  KOKKOS_INLINE_FUNCTION void buildCell(int pi) const {
    // Voronoi only. Power/Laguerre is NOT supported on the ConvexCell device path: its
    // foot-point half-space ({x : nf·x ≤ |nf|²}) always contains the seed, but a radical
    // plane can put the seed OUTSIDE its own cell (negative offset), which this
    // representation cannot express. Full Laguerre needs ConvexCell radical-plane geometry
    // (the planned-but-unbuilt Power policy in convex_cell.hpp); no production path uses it.
    static_assert(!Weighted,
                  "Power/Laguerre on the device is pending ConvexCell radical-plane geometry; "
                  "buildTessellation currently supports Voronoi (Weighted=false) only.");
    const int i = binned(pi);
    const Real pix = posSorted(3 * pi + 0), piy = posSorted(3 * pi + 1), piz = posSorted(3 * pi + 2);
    int cx, cy, cz;
    homeCell(pix, piy, piz, cx, cy, cz);
    const int base = subBase(pix, piy, piz);
    Cell c;
    c.initBox(Lx, Ly, Lz);

    // The worklist is sorted by nearest-corner dist², so once wlRmin exceeds the security
    // radius (4·rSqMax) every remaining block is too far to cut — break. The per-candidate
    // cull skips seeds past the radius; secR2 is recomputed only on a real cut.
    Real secR2 = Real(2) * c.maxVertexRsq();
    int ncRec = 0;  // Part-II: count of recorded candidate (skin) ids for this cell
    for (int g = 0; g < nOff && !c.overflow; ++g) {
      if (wlRmin(base + g) > Real(2) * secR2) break;  // sorted ⇒ rest are farther; cell closed
      const int gc = worklistCell(base, g, cx, cy, cz);
      for (int q = cellStart(gc); q < cellStart(gc + 1) && !c.overflow; ++q) {
        if (q == pi) continue;
        if (haveGid && gidSorted(q) == gidSorted(pi)) continue;
        // record the examined neighbour as a skin candidate (within the worklist's security reach), capped.
        if (emitCand && ncRec < candCap) oCand[(size_t)i * candCap + ncRec++] = binned(q);
        Real pv[3];
        relVec(q, pix, piy, piz, pv);
        const Real off = Real(0.5) * (pv[0] * pv[0] + pv[1] * pv[1] + pv[2] * pv[2]);
        if (off >= secR2) continue;  // beyond the radius: cannot cut
        if (c.clip(pv, off, binned(q))) secR2 = Real(2) * c.maxVertexRsq();
      }
    }
    if (emitCand) oCandCnt(i) = ncRec;
    // Completeness uses the conservative inscribed-sphere coverage (sw·minCsz)², the same
    // criterion the legacy gather used: complete iff (sw·minCsz)² > 4·rSqMax.
    const Real covSq = Real(sw) * minCsz * Real(sw) * minCsz;
    finishCell(c, i, pix, piy, piz, covSq);
  }
};

/**
 * Build the full tessellation on device.
 *
 * @param posFlat  device positions, x-fastest per seed (3*i + k), in [0,L).
 * @param weight   device per-seed weights (size N) for the Power policy; ignored
 *                 (may be empty) when Weighted == false.
 * @param N        number of seeds.
 * @param L        periodic box extent.
 * @param sw       grid-block half-width (default 4; coverage = sw·cellSize).
 * @param densityCount  seed count to derive the grid spacing from (cellSize ~
 *                 mean spacing). Defaults to N. In the distributed case the seeds
 *                 are a clustered owned+ghost subset, so pass the GLOBAL count to
 *                 keep the grid at the true local density.
 * @param nBuild   build a cell only for seeds whose ORIGINAL index is < nBuild;
 *                 the rest are still used as cut candidates (neighbours) but their own
 *                 cell is skipped. Defaults to N (build all). In the distributed case
 *                 the ghosts are appended after the nOwned owned seeds, so passing
 *                 nOwned tessellates only the kept (owned) cells — the ghost cells are
 *                 needed only as cutting seeds, so building them is wasted work (~the
 *                 ghost fraction of the cold build).
 */
template <class Real, bool Weighted, class Sdf = NoSdf>
TessellatorResult<Real> buildTessellation(const Kokkos::View<Real*, tpx::MemSpace>& posFlat,
                                          const Kokkos::View<Real*, tpx::MemSpace>& weight, int N,
                                          const Real L[3], int sw = 4, int densityCount = -1,
                                          Kokkos::View<long*, tpx::MemSpace> gid = {}, Sdf sdf = {},
                                          bool withForceGeom = true, int nBuild = -1,
                                          Kokkos::View<int*, tpx::MemSpace> outNp = {},
                                          Kokkos::View<int*, tpx::MemSpace> outNt = {},
                                          Kokkos::View<int*, tpx::MemSpace> outPnbr = {},
                                          Kokkos::View<unsigned*, tpx::MemSpace> outTri = {},
                                          Kokkos::View<int*, tpx::MemSpace> outCand = {},
                                          Kokkos::View<int*, tpx::MemSpace> outCandCnt = {},
                                          int candCap = 0) {
  using tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  // Part-II optional outputs (see CellBuilder): emit the resident topology store / candidate skin list only
  // when the caller supplies sized views (so existing callers, passing none, are byte-for-byte unaffected).
  const bool emitTopo = outNp.extent(0) == static_cast<size_t>(N);
  const bool emitCand = outCand.extent(0) > 0 && candCap > 0;
  // Optional global ids: skip a candidate sharing the cell's own id (its periodic
  // self-image, which can wrap exactly onto the seed -> a degenerate zero-distance
  // cut). Mirrors the legacy processNbrs `itr->id == m_id` guard. In the
  // single-domain case (gid empty) only the same local index is skipped.
  const bool haveGid = gid.extent(0) == static_cast<size_t>(N);
  // Seeds with original index >= nBuildEff are candidate-only (their cell is skipped).
  const int nBuildEff = (nBuild >= 0 && nBuild < N) ? nBuild : N;

  const bool prof = std::getenv("VORFLOW_PROFILE") != nullptr;
  Kokkos::Timer ptimer;
  double tGrid = 0, tBuild = 0, tCsr = 0;

  // Counting-sort grid + presorted worklist. Factored into buildTessGrid so the SAME grid backs
  // both this cold build and the moving-point subset gather (device/subset_gather.hpp); pure code
  // motion, so the cold-build output is byte-for-byte unchanged.
  auto grid = buildTessGrid<Real, Weighted>(posFlat, weight, N, L, sw, densityCount, gid);
  const Real Lx = grid.Lx, Ly = grid.Ly, Lz = grid.Lz;
  const Real icx = grid.icx, icy = grid.icy, icz = grid.icz, minCsz = grid.minCsz;
  const int dimx = grid.dimx, dimy = grid.dimy, dimz = grid.dimz;
  const int nOff = grid.nOff, wlS = grid.wlS;
  const bool useMorton = grid.useMorton;

  if (prof) {
    Kokkos::fence();
    tGrid = ptimer.seconds();
    ptimer.reset();
  }

  // --- pass 1: build each cell, record volume / facet count / status and the
  //     per-cell facet records (neighbour id + area vector) into a temp slab ---
  Kokkos::View<Real*, MemSpace> cellVol("cellVol", N);
  Kokkos::View<int*, MemSpace> facetCount("facetCount", N);
  Kokkos::View<int*, MemSpace> status("status", N);
  Kokkos::View<int*, MemSpace> cellFacetBase("cellFacetBase", N);  // per-cell facet base
  // Facet over-buffer + atomic cursor: the build packs each cell's facets directly into
  // a single contiguous CSR-layout buffer via atomic_fetch_add (one-pass "fusion") — no
  // fixed-stride temp slab, no exclusive scan, and the connecting vector is the cut's own
  // plane vector (no minimal-image recompute). The buffer is over-allocated to a tight
  // facet cap; the used prefix [0,nFacets) is copied to a compact view at the end.
  // Per-cell facet cap: the most facets one Voronoi/Power cell may publish (a single
  // cell rarely exceeds ~40 faces); bounds the per-cell `faces[]` stack array.
  constexpr int MAXF_TMP = 50;
  // Global over-buffer capacity: the published CSR holds the *sum* of all cells'
  // facets ≈ N × mean-faces-per-cell (~15.5 for random Poisson–Voronoi). Sizing it at
  // N × MAXF_TMP over-allocated ~3× and OOM'd the GPU at large N (≈15 GB at N=4M). A
  // mean-facet estimate with headroom (N×18, ~16% over the aggregate mean) is ample —
  // the *sum* has negligible relative variance — and the atomic overflow guard below
  // flags the rare cell whose reservation would exceed it instead of writing past the
  // end. (The interleaved copy-and-free in the pack keeps peak memory near this size.)
  constexpr size_t kMeanFacets = 18;
  const size_t facetCap = (size_t)N * kMeanFacets;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  Kokkos::View<int*, MemSpace> oNbr(view_alloc(std::string("oNbr"), WithoutInitializing), facetCap);
  Kokkos::View<Real*, MemSpace> oArea(view_alloc(std::string("oArea"), WithoutInitializing),
                                      facetCap * 3);
  Kokkos::View<Real*, MemSpace> oDV(view_alloc(std::string("oDV"), WithoutInitializing),
                                    facetCap * 3);
  Kokkos::View<Real*, MemSpace> oConn(view_alloc(std::string("oConn"), WithoutInitializing),
                                      facetCap * 3);
  Kokkos::View<int*, MemSpace> facetCursor("facetCursor", 1);  // zero-initialised
  if (prof) std::fprintf(stderr, "[worklist] sw=%d nOff=%d\n", sw, nOff);

  // Build each cell: one thread per cell (RangePolicy), the compact ConvexCell in the
  // per-thread frame, the cut applied on the fly. Same path on every backend (ConvexCell is
  // lean enough that the worklist + geometry run well on the GPU without a team variant).
  CellBuilder<Real, Weighted, Sdf> op{
      grid.binned, grid.posSorted, grid.wSorted, grid.gidSorted, grid.cellStart, grid.wlOff, grid.wlRmin,
      status, cellVol, facetCount, cellFacetBase, oNbr, oArea, oDV, oConn, facetCursor,
      icx, icy, icz, Lx, Ly, Lz, minCsz, dimx, dimy, dimz, sw, nOff, wlS,
      useMorton, haveGid, withForceGeom, facetCap, sdf,
      outNp, outNt, outPnbr, outTri, outCand, outCandCnt, emitTopo, emitCand, candCap};
  const int nBuildL = nBuildEff;
  auto binnedV0 = grid.binned;
  Kokkos::parallel_for(
      "tess.build", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
        if (binnedV0(pi) >= nBuildL) return;  // candidate-only seed: skip its cell
        op.buildCell(pi);
      });

  if (prof) {
    Kokkos::fence();
    tBuild = ptimer.seconds();
    ptimer.reset();
  }

  // --- pack the over-buffer prefix [0,nFacets) into a compact CSR ---
  // The atomic packing already produced a gapless CSR (in cell-finish order, with each
  // cell's facets contiguous at cellFacetBase(i)); we only copy its used prefix into a
  // right-sized view (a contiguous read+write — no exclusive scan, no strided gather,
  // no minimal-image recompute that the old temp->CSR compaction paid).
  int nFacetsRaw = 0;
  Kokkos::deep_copy(nFacetsRaw, Kokkos::subview(facetCursor, 0));
  // Clamp to capacity: if the over-buffer overflowed (rare; flagged per-cell above),
  // the cursor ran past the end and only [0,facetCap) holds valid, indexable facets.
  const int nFacets = (size_t)nFacetsRaw > facetCap ? (int)facetCap : nFacetsRaw;

  TessellationView<Real> view;
  view.cellFacetOffset = cellFacetBase;
  view.cellFacetCount = facetCount;
  view.cellVolume = cellVol;
  view.cellSeedId = Kokkos::View<gid_t*, MemSpace>(view_alloc(std::string("cellSeedId"),
                                                              WithoutInitializing), N);
  {
    auto vSeed = view.cellSeedId;
    Kokkos::parallel_for(
        "tess.seedId", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i) { vSeed(i) = (gid_t)i; });
  }

  // Pack each over-buffer component into a right-sized published view, then free the
  // source immediately. Copying-and-freeing one array at a time holds peak memory at
  // ~(over-buffer + one compact array) instead of (over-buffer + full compact CSR) —
  // the latter OOM'd the GPU at large N. numFacets() == nFacets, so the CSR contract is
  // intact (consumers see correctly sized views). The fence before each free ensures the
  // copy completed before the source is released.
  view.facetNeighbor = Kokkos::View<gid_t*, MemSpace>(
      view_alloc(std::string("facetNeighbor"), WithoutInitializing), nFacets);
  {
    auto v = view.facetNeighbor;
    auto src = oNbr;
    Kokkos::parallel_for("tess.packNbr", Kokkos::RangePolicy<Exec>(0, nFacets),
                         KOKKOS_LAMBDA(const int g) { v(g) = (gid_t)src(g); });
  }
  Kokkos::fence();
  oNbr = Kokkos::View<int*, MemSpace>();

  view.facetArea = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("facetArea"), WithoutInitializing), (size_t)nFacets * 3);
  {
    auto v = view.facetArea;
    auto src = oArea;
    Kokkos::parallel_for("tess.packArea", Kokkos::RangePolicy<Exec>(0, nFacets),
                         KOKKOS_LAMBDA(const int g) {
                           for (int cc = 0; cc < 3; ++cc) v(3 * g + cc) = src(3 * (size_t)g + cc);
                         });
  }
  Kokkos::fence();
  oArea = Kokkos::View<Real*, MemSpace>();

  view.facetConnect = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("facetConnect"), WithoutInitializing), (size_t)nFacets * 3);
  {
    auto v = view.facetConnect;
    auto src = oDV;
    Kokkos::parallel_for("tess.packDV", Kokkos::RangePolicy<Exec>(0, nFacets),
                         KOKKOS_LAMBDA(const int g) {
                           for (int cc = 0; cc < 3; ++cc) v(3 * g + cc) = src(3 * (size_t)g + cc);
                         });
  }
  Kokkos::fence();
  oDV = Kokkos::View<Real*, MemSpace>();

  view.facetConnVec = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("facetConnVec"), WithoutInitializing), (size_t)nFacets * 3);
  {
    auto v = view.facetConnVec;
    auto src = oConn;
    Kokkos::parallel_for("tess.packConn", Kokkos::RangePolicy<Exec>(0, nFacets),
                         KOKKOS_LAMBDA(const int g) {
                           for (int cc = 0; cc < 3; ++cc) v(3 * g + cc) = src(3 * (size_t)g + cc);
                         });
  }
  Kokkos::fence();
  oConn = Kokkos::View<Real*, MemSpace>();

  if (prof) {
    tCsr = ptimer.seconds();
    std::fprintf(stderr, "[tess N=%d] grid=%.4fs build=%.4fs csr=%.4fs (build %.0f%%)\n", N, tGrid,
                 tBuild, tCsr, 100.0 * tBuild / (tGrid + tBuild + tCsr));
    // Max published facets/cell — the empirical bound for sizing a shrunk shared cell
    // (vertices ~ 2·facets - 4). Reduce over the per-cell facet count.
    int maxF = 0;
    auto fc = facetCount;
    Kokkos::parallel_reduce(
        "tess.maxFacets", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int c, int& m) { m = fc(c) > m ? fc(c) : m; }, Kokkos::Max<int>(maxF));
    std::fprintf(stderr, "[tess N=%d] maxFacets/cell=%d (=> maxVerts ~ %d; CAP=%d)\n", N, maxF,
                 2 * maxF - 4, CellBuilder<Real, Weighted, Sdf>::kMaxT);
  }

  TessellatorResult<Real> res;
  res.view = view;
  res.status = status;
  return res;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TESSELLATOR_HPP
