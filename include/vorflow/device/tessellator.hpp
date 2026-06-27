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
#include "vorflow/tessellation_view.hpp"

namespace vor {
namespace device {

/// Per-cell status bits written by the build pass.
enum StatusBit { kOk = 0, kOverflow = 1, kEmpty = 2, kIncomplete = 4 };

/// Bias added to each signed block offset (dx,dy,dz) so it packs into one int (8 bits
/// per axis) in the worklist table — a single load + bit-unpack in the hot gather loop
/// instead of three scattered loads. Block offsets stay within [-sw,sw], sw << 128.
constexpr int kWlOffBias = 128;

/// 3D Morton (Z-order) code of a grid cell, via the suite's morton library
/// (`Morton<3,21>`, software bit path on device — no BMI2; bit-identical to the
/// former hand-rolled magic-bits spread). Used to order the cell grid so a cell's
/// spatial neighbourhood is near it in memory — the gather then reads
/// near-contiguous cellStart/posSorted instead of chasing z-neighbours dimx*dimy
/// entries apart. Good to 21 bits/axis (grid indices are far below that).
KOKKOS_INLINE_FUNCTION int morton3(int x, int y, int z) {
  return static_cast<int>(morton::Morton<3, 21>::encode(static_cast<std::uint32_t>(x),
                                                        static_cast<std::uint32_t>(y),
                                                        static_cast<std::uint32_t>(z))
                              .code());
}

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

  // --- grid dimensions: ~kSeedsPerCell seeds per cell ---
  // A coarser grid than 1 seed/cell makes the neighbour gather touch fewer, denser
  // grid cells whose seeds are contiguous in the grid-sorted arrays — far fewer
  // scattered cellStart/posSorted lookups (the gather is memory-latency bound, the
  // dominant serial cost; voro++ packs ~8 particles/block for the same reason). ~2/cell
  // measured the best correct trade-off (coarser over-gathers per cell). The expanding
  // search + the `sw` cap keep cells complete regardless. This is a HOST (CPU) win
  // only: on a GPU the gather is bandwidth/latency-hidden, not cache-bound, so a coarser
  // grid just adds per-thread candidate work — GPUs keep 1/cell. Power also keeps 1/cell
  // (its no-early-out full-sphere gather would blow past MAXCAND at 2/cell).
  constexpr bool kHostBackend =
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible;
  constexpr Real kSeedsPerCell = (!Weighted && kHostBackend) ? Real(2) : Real(1);
  const int dens = densityCount > 0 ? densityCount : N;
  const Real vol = L[0] * L[1] * L[2];
  const Real spacing = std::cbrt(kSeedsPerCell * vol / Real(dens > 0 ? dens : 1));
  int dim[3];
  Real csz[3], invcsz[3];
  for (int k = 0; k < 3; ++k) {
    dim[k] = (int)Kokkos::floor(L[k] / spacing);
    if (dim[k] < 1)
      dim[k] = 1;
    csz[k] = L[k] / dim[k];
    invcsz[k] = Real(1) / csz[k];
  }
  const int dimx = dim[0], dimy = dim[1], dimz = dim[2];
  const int ncell = dimx * dimy * dimz;

  // Morton (Z-order) cell indexing clusters each cell's spatial neighbourhood in memory.
  // GPU ONLY: the spatial order improves the gather's memory coalescing (measured +2.5%
  // pure / +18% with force geometry on an RTX 5080), and the per-cell encode is hidden by
  // spare ALU. On the CPU it is a net loss — the encode in the hot gather loop is not
  // hidden and the ~2-seeds/cell density already supplies the cache locality — so the CPU
  // keeps the compact linear index. Morton needs a power-of-two-padded range (2^mbits per
  // axis); also fall back to linear if that padding would inflate the cell array too much
  // (very rectangular boxes / huge grids).
  int mbits = 1;
  {
    int md = dimx > dimy ? dimx : dimy;
    md = md > dimz ? md : dimz;
    while ((1 << mbits) < md) ++mbits;
  }
  const size_t mortonNcell = (size_t)1 << (3 * mbits);
  const bool useMorton = !kHostBackend && mortonNcell <= (size_t)8 * (size_t)ncell;
  const int ncellEff = useMorton ? (int)mortonNcell : ncell;

  const Real Lx = L[0], Ly = L[1], Lz = L[2];
  const Real icx = invcsz[0], icy = invcsz[1], icz = invcsz[2];
  Real minCsz = csz[0] < csz[1] ? csz[0] : csz[1];
  minCsz = minCsz < csz[2] ? minCsz : csz[2];

  const bool prof = std::getenv("VORFLOW_PROFILE") != nullptr;
  Kokkos::Timer ptimer;
  double tGrid = 0, tBuild = 0, tCsr = 0;

  // --- counting sort: bin seeds into grid cells (Morton- or linear-indexed) ---
  Kokkos::View<int*, MemSpace> cellOf("cellOf", N);
  Kokkos::View<int*, MemSpace> counts("counts", ncellEff + 1);
  Kokkos::deep_copy(counts, 0);
  Kokkos::parallel_for(
      "tess.bin", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int gx = (int)Kokkos::floor(posFlat(3 * i + 0) * icx);
        int gy = (int)Kokkos::floor(posFlat(3 * i + 1) * icy);
        int gz = (int)Kokkos::floor(posFlat(3 * i + 2) * icz);
        gx = ((gx % dimx) + dimx) % dimx;
        gy = ((gy % dimy) + dimy) % dimy;
        gz = ((gz % dimz) + dimz) % dimz;
        int c = useMorton ? morton3(gx, gy, gz) : (gx + gy * dimx + gz * dimx * dimy);
        cellOf(i) = c;
        Kokkos::atomic_inc(&counts(c));
      });

  Kokkos::View<int*, MemSpace> cellStart("cellStart", ncellEff + 1);
  Kokkos::parallel_scan(
      "tess.scan", Kokkos::RangePolicy<Exec>(0, ncellEff + 1),
      KOKKOS_LAMBDA(const int c, int& acc, const bool fin) {
        int v = (c < ncellEff) ? counts(c) : 0;
        if (fin)
          cellStart(c) = acc;
        acc += v;
      });

  Kokkos::View<int*, MemSpace> cursor("cursor", ncellEff);
  Kokkos::parallel_for(
      "tess.cursor", Kokkos::RangePolicy<Exec>(0, ncellEff),
      KOKKOS_LAMBDA(const int c) { cursor(c) = cellStart(c); });
  Kokkos::View<int*, MemSpace> binned("binned", N);
  Kokkos::parallel_for(
      "tess.scatter", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int slot = Kokkos::atomic_fetch_add(&cursor(cellOf(i)), 1);
        binned(slot) = i;
      });

  // Reorder seed data into grid (binned) order. The build then gathers neighbours
  // from contiguous memory instead of chasing scattered original indices through
  // posFlat — the large-N cache-locality win (voro++ processes points block-by-block
  // for the same reason). Outputs are still written back to the original cell index
  // (= binned(p)), so the cell i == particle i contract is unchanged.
  Kokkos::View<Real*, MemSpace> posSorted(
      Kokkos::view_alloc(std::string("posSorted"), Kokkos::WithoutInitializing), (size_t)N * 3);
  Kokkos::View<gid_t*, MemSpace> gidSorted(
      Kokkos::view_alloc(std::string("gidSorted"), Kokkos::WithoutInitializing), haveGid ? N : 0);
  Kokkos::View<Real*, MemSpace> wSorted(
      Kokkos::view_alloc(std::string("wSorted"), Kokkos::WithoutInitializing), Weighted ? N : 0);
  {
    const bool haveGidL = haveGid;
    Kokkos::parallel_for(
        "tess.reorder", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int p) {
          int o = binned(p);
          posSorted(3 * p + 0) = posFlat(3 * o + 0);
          posSorted(3 * p + 1) = posFlat(3 * o + 1);
          posSorted(3 * p + 2) = posFlat(3 * o + 2);
          if (haveGidL) gidSorted(p) = gid(o);
          if (Weighted) wSorted(p) = weight(o);
        });
  }

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

  // --- per-sub-position worklist (computed once) ---
  // The (dx,dy,dz) grid offsets inside the coverage sphere (nearest-corner dist² ≤ sw²). For
  // the wlS³ sub-region a seed lands in, they are sorted by nearest-corner dist² (wlRmin,
  // absolute) and packed (kWlOffBias). The Voronoi build walks this presorted list and breaks
  // once wlRmin exceeds the security radius (4·rSqMax) — a table lookup, no per-block geometry.
  // The cut runs on the compact ConvexCell, whose geometry pass is cheap, so the worklist is
  // used on both backends (no GPU occupancy penalty).
  std::vector<int> offHx, offHy, offHz;
  {
    const int swInit = sw < 2 ? sw : 2;
    const int sw2 = sw * sw;
    for (int swl = swInit; swl <= sw; ++swl) {
      const int lo2 = (swl == swInit) ? -1 : (swl - 1) * (swl - 1);
      const int hi2 = swl * swl;
      for (int dz = -(swl + 1); dz <= swl + 1; ++dz) {
        const int ez = dz < -1 ? -dz - 1 : (dz > 1 ? dz - 1 : 0);
        for (int dy = -(swl + 1); dy <= swl + 1; ++dy) {
          const int ey = dy < -1 ? -dy - 1 : (dy > 1 ? dy - 1 : 0);
          for (int dx = -(swl + 1); dx <= swl + 1; ++dx) {
            const int ex = dx < -1 ? -dx - 1 : (dx > 1 ? dx - 1 : 0);
            const int d2 = ex * ex + ey * ey + ez * ez;
            if (d2 <= lo2 || d2 > hi2 || d2 > sw2) continue;
            offHx.push_back(dx);
            offHy.push_back(dy);
            offHz.push_back(dz);
          }
        }
      }
    }
  }
  const int nOff = (int)offHx.size();
  const int wlS = 3;  // worklist sub-grid per axis
  const int nSub = wlS * wlS * wlS;
  Kokkos::View<int*, MemSpace> wlOff(view_alloc(std::string("wlOff"), WithoutInitializing),
                                     (size_t)nSub * nOff);
  Kokkos::View<Real*, MemSpace> wlRmin(view_alloc(std::string("wlRmin"), WithoutInitializing),
                                       (size_t)nSub * nOff);
  {
    auto hOff = Kokkos::create_mirror_view(wlOff);
    auto hRmin = Kokkos::create_mirror_view(wlRmin);
    const Real cszx = csz[0], cszy = csz[1], cszz = csz[2];
    // axis-aligned gap² between query sub-interval [a0,a1] and target block [b0,b1].
    auto axisGap = [](Real a0, Real a1, Real b0, Real b1) {
      Real g = std::max(Real(0), std::max(b0 - a1, a0 - b1));
      return g * g;
    };
    std::vector<int> idx(nOff);
    std::vector<Real> rmn(nOff);
    for (int sz = 0; sz < wlS; ++sz)
      for (int sy = 0; sy < wlS; ++sy)
        for (int sx = 0; sx < wlS; ++sx) {
          const int p = sx + wlS * (sy + wlS * sz);
          const Real ax0 = (Real)sx / wlS * cszx, ax1 = (Real)(sx + 1) / wlS * cszx;
          const Real ay0 = (Real)sy / wlS * cszy, ay1 = (Real)(sy + 1) / wlS * cszy;
          const Real az0 = (Real)sz / wlS * cszz, az1 = (Real)(sz + 1) / wlS * cszz;
          for (int k = 0; k < nOff; ++k) {
            const Real bx0 = offHx[k] * cszx, bx1 = (offHx[k] + 1) * cszx;
            const Real by0 = offHy[k] * cszy, by1 = (offHy[k] + 1) * cszy;
            const Real bz0 = offHz[k] * cszz, bz1 = (offHz[k] + 1) * cszz;
            rmn[k] = axisGap(ax0, ax1, bx0, bx1) + axisGap(ay0, ay1, by0, by1) +
                     axisGap(az0, az1, bz0, bz1);
            idx[k] = k;
          }
          std::sort(idx.begin(), idx.end(), [&](int a, int b) { return rmn[a] < rmn[b]; });
          for (int g = 0; g < nOff; ++g) {
            const int k = idx[g];
            const size_t slot = (size_t)p * nOff + g;
            hOff(slot) = (offHx[k] + kWlOffBias) | ((offHy[k] + kWlOffBias) << 8) |
                         ((offHz[k] + kWlOffBias) << 16);
            hRmin(slot) = rmn[k];
          }
        }
    Kokkos::deep_copy(wlOff, hOff);
    Kokkos::deep_copy(wlRmin, hRmin);
  }
  if (prof) std::fprintf(stderr, "[worklist] sw=%d nOff=%d nSub=%d\n", sw, nOff, nSub);

  // Build each cell: one thread per cell (RangePolicy), the compact ConvexCell in the
  // per-thread frame, the cut applied on the fly. Same path on every backend (ConvexCell is
  // lean enough that the worklist + geometry run well on the GPU without a team variant).
  CellBuilder<Real, Weighted, Sdf> op{
      binned, posSorted, wSorted, gidSorted, cellStart, wlOff, wlRmin,
      status, cellVol, facetCount, cellFacetBase, oNbr, oArea, oDV, oConn, facetCursor,
      icx, icy, icz, Lx, Ly, Lz, minCsz, dimx, dimy, dimz, sw, nOff, wlS,
      useMorton, haveGid, withForceGeom, facetCap, sdf,
      outNp, outNt, outPnbr, outTri, outCand, outCandCnt, emitTopo, emitCand, candCap};
  const int nBuildL = nBuildEff;
  auto binnedV0 = binned;
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
