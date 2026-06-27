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
#include "vorflow/device/cell_cutter.hpp"
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

/// Sift element i down a binary min-heap of size n keyed on key[], moving id[] in
/// lockstep. Used to process candidate neighbours closest-first in O(applied·log n)
/// instead of an O(applied·n) selection scan.
template <class Real>
KOKKOS_INLINE_FUNCTION void heapSiftDown(Real* key, int* id, int i, int n) {
  for (;;) {
    int l = 2 * i + 1, r = l + 1, m = i;
    if (l < n && key[l] < key[m]) m = l;
    if (r < n && key[r] < key[m]) m = r;
    if (m == i) break;
    Real tk = key[i];
    key[i] = key[m];
    key[m] = tk;
    int ti = id[i];
    id[i] = id[m];
    id[m] = ti;
    i = m;
  }
}

template <class Real>
struct TessellatorResult {
  TessellationView<Real> view;
  Kokkos::View<int*, tpx::MemSpace> status;  // per-cell StatusBit mask
};

/**
 * Per-cell build, factored out of the build pass so the host (one thread per cell,
 * RangePolicy) and device (one team per cell, TeamPolicy, cell in shared scratch)
 * paths run the *same* numerics — bit-exact, no second copy to drift. Holds the
 * grid-sorted inputs, the published outputs, and the grid scalars as members; the
 * scratch cell + candidate arrays (their storage differs per backend: per-thread
 * stack on host, team shared memory on device) are passed in to `buildCell`.
 */
template <class Real, bool Weighted, class Sdf>
struct CellBuilder {
  using MemSpace = tpx::MemSpace;
  static constexpr int MAXCAND = 1024;  // full-sphere candidate cap (see docs/performance.md)
  static constexpr int MAXF_TMP = 50;   // max facets one cell may publish
  static constexpr int kTeamChunk = 64;  // worklist offsets gathered per team barrier round

  // Grid-sorted inputs (read).
  Kokkos::View<int*, MemSpace> binned;
  Kokkos::View<Real*, MemSpace> posSorted;
  Kokkos::View<Real*, MemSpace> wSorted;
  Kokkos::View<gid_t*, MemSpace> gidSorted;
  Kokkos::View<int*, MemSpace> cellStart;
  // Per-sub-position worklist: for each of the wlS³ sub-regions of a home cell, the
  // (2sw+1)³ block offsets (packed (dx,dy,dz), `kWlOffBias`) sorted by nearest-corner
  // dist² (`wlRmin`, absolute). The build walks this presorted list and breaks once
  // wlRmin exceeds the security radius — a table lookup, no per-block geometry.
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

  /// Minimal-image relative vector from the seed at (pix,piy,piz) to sorted seed q
  /// (single-image wrap; bit-identical to round(r/L) in range). Shared by the serial
  /// and team paths so both compute candidates identically.
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

  /// Home grid cell (cx,cy,cz) of seed (pix,piy,piz) and the flat base offset into the
  /// worklist tables for the wlS³ sub-region the seed lands in. The worklist for a
  /// sub-region p occupies [base, base+nOff); base = p*nOff with p = sx + wlS·(sy + wlS·sz).
  KOKKOS_INLINE_FUNCTION int homeCellBase(Real pix, Real piy, Real piz, int& cx, int& cy,
                                          int& cz) const {
    const Real fxi = pix * icx, fyi = piy * icy, fzi = piz * icz;
    const Real flx = Kokkos::floor(fxi), fly = Kokkos::floor(fyi), flz = Kokkos::floor(fzi);
    cx = (((int)flx % dimx) + dimx) % dimx;
    cy = (((int)fly % dimy) + dimy) % dimy;
    cz = (((int)flz % dimz) + dimz) % dimz;
    int sx = (int)((fxi - flx) * Real(wlS));
    int sy = (int)((fyi - fly) * Real(wlS));
    int sz = (int)((fzi - flz) * Real(wlS));
    sx = sx < 0 ? 0 : (sx >= wlS ? wlS - 1 : sx);
    sy = sy < 0 ? 0 : (sy >= wlS ? wlS - 1 : sy);
    sz = sz < 0 ? 0 : (sz >= wlS ? wlS - 1 : sz);
    return (sx + wlS * (sy + wlS * sz)) * nOff;
  }

  /// Decode worklist entry g (relative to `base`) into a raw (periodic-unwrapped) grid
  /// offset added to the home cell. Returns the linear/Morton grid-cell index.
  KOKKOS_INLINE_FUNCTION int worklistCell(int base, int g, int cx, int cy, int cz) const {
    const int packed = wlOff(base + g);
    const int rgx = cx + ((packed & 0xFF) - kWlOffBias);
    const int rgy = cy + (((packed >> 8) & 0xFF) - kWlOffBias);
    const int rgz = cz + (((packed >> 16) & 0xFF) - kWlOffBias);
    return gridCell(rgx, rgy, rgz);
  }

  /// Finish a built cell: completeness flag, optional SDF clip, status/volume,
  /// per-facet geometry, and the atomic-packed facet write into the over-buffer.
  /// `covSq` is the attained coverage² and `ovf` the running overflow flag. Shared by
  /// the serial buildCell and the team-per-cell leader so the published outputs match.
  template <int CAP>
  KOKKOS_INLINE_FUNCTION void finishCell(ScratchCell<Real, CAP>& c, int i, Real pix, Real piy,
                                         Real piz, Real covSq, bool ovf,
                                         bool geomDone = false) const {
    // Overflow short-circuit: a cell that overran its slot/candidate capacity may be a
    // corrupt half-edge mesh, so volume()/computeGeometry()/the facet walk could loop
    // forever or read garbage. Publish only the kOverflow flag (no facets) and leave it
    // for the full-capacity fallback re-run; never walk the broken cell. (For random data
    // on the full-CAP path this never triggers, so the default path is unaffected.)
    if (ovf) {
      status(i) = kOverflow;
      facetCount(i) = 0;
      cellFacetBase(i) = 0;
      cellVol(i) = Real(0);
      return;
    }
    // Voronoi completeness is judged on the un-clipped cell.
    const bool incomplete = !(covSq > Real(4) * c.rsq[c.vRsqMax]);
    // Optional SDF boundary clip. (CUDA extended lambdas cannot first-capture a
    // variable inside an if-constexpr, so reference sdf once in normal context first.)
    (void)sdf;
    if constexpr (!std::is_same_v<Sdf, NoSdf>) {
      const Real seedW[3] = {pix, piy, piz};
      clipCellAgainstSdf<Real>(c, seedW, sdf, &ovf);
    }
    int st = kOk;
    if (ovf) st |= kOverflow;
    if (c.emptyV()) st |= kEmpty;
    if (incomplete) st |= kIncomplete;
    status(i) = st;
    cellVol(i) = c.volume();
    // Per-facet geometry (area vector + dV volume gradient); skipped for pure tess and
    // when the team path already computed it in parallel (geomDone).
    if (!c.emptyV() && withForceGeom && !geomDone) c.computeGeometry();
    // Collect this cell's live facets, then reserve a contiguous CSR range with one
    // atomic and write straight into the over-buffer.
    int faces[MAXF_TMP];
    int nf = 0;
    for (int f = 0; f < c.numAllocF; ++f) {
      if (!c.aliveF[f]) continue;
      if (nf >= MAXF_TMP) {
        status(i) |= kOverflow;
        break;
      }
      faces[nf++] = f;
    }
    const int base = Kokkos::atomic_fetch_add(&facetCursor(0), nf);
    // Overflow guard: skip the write rather than clobbering past the over-buffer end.
    if ((size_t)base + (size_t)nf > facetCap) {
      status(i) |= kOverflow;
      facetCount(i) = 0;
      cellFacetBase(i) = 0;
    } else {
      facetCount(i) = nf;
      cellFacetBase(i) = base;
      for (int k = 0; k < nf; ++k) {
        const int f = faces[k];
        oNbr((size_t)base + k) = c.fnbr[f];
        const size_t o = ((size_t)base + k) * 3;
        for (int cc = 0; cc < 3; ++cc) {
          oArea(o + cc) = withForceGeom ? c.fArea[f][cc] : Real(0);
          oDV(o + cc) = withForceGeom ? c.fdV[f][cc] : Real(0);
          oConn(o + cc) = c.pvec[f][cc];  // plane vector == the connecting vector (exact)
        }
      }
    }
  }

  /// Build the cell owning grid-sorted slot `pi` into scratch `c`, writing the published
  /// outputs at the original seed index binned(pi). Worklist gather: the cut clips the
  /// neighbour seeds on the fly (no candidate buffer) in nearest-block-first order, which
  /// for a convex clip yields the same cell as closest-first; the per-block `wlRmin` break
  /// and the per-candidate security cull touch only the blocks/seeds that can actually cut.
  template <int CAP>
  KOKKOS_INLINE_FUNCTION void buildCell(ScratchCell<Real, CAP>& c, int pi) const {
    // pi is the grid-sorted slot; i = binned(pi) the original seed index that owns this
    // cell. Positions/weights/ids are read from the reordered (cache-local) arrays;
    // outputs are written back at the original index i.
    const int i = binned(pi);
    const Real pix = posSorted(3 * pi + 0), piy = posSorted(3 * pi + 1), piz = posSorted(3 * pi + 2);
    int cx, cy, cz;
    const int base = homeCellBase(pix, piy, piz, cx, cy, cz);
    const Real wi = Weighted ? wSorted(pi) : Real(0);
    bool ovf = false;
    const Real Larr[3] = {Lx, Ly, Lz};
    c.initCuboid(Larr);

    // covSq encodes the completeness verdict for finishCell: a value > 4·rSqMax means the
    // cell provably closed (no unexamined seed can reach it), 0 means it did not.
    Real covSq;
    if constexpr (Weighted) {
      // Power has no security early-out (a distant heavy seed can still cut), so walk the
      // whole worklist and apply every candidate; the cut is order-independent.
      for (int g = 0; g < nOff && !ovf; ++g) {
        const int gc = worklistCell(base, g, cx, cy, cz);
        for (int q = cellStart(gc); q < cellStart(gc + 1) && !ovf; ++q) {
          if (q == pi) continue;
          if (haveGid && gidSorted(q) == gidSorted(pi)) continue;  // periodic self-image
          Real pv[3];
          relVec(q, pix, piy, piz, pv);
          const Real rSqHalf = Real(0.5) * (pv[0] * pv[0] + pv[1] * pv[1] + pv[2] * pv[2]);
          const Real off = rSqHalf + Real(0.5) * (wi - wSorted(q));
          c.cutCell2(pv, off, binned(q), &ovf);  // store the ORIGINAL neighbour id
        }
      }
      covSq = Real(sw) * minCsz * Real(sw) * minCsz;
    } else {
      // Voronoi: the worklist is sorted by nearest-corner dist², so once wlRmin exceeds
      // the security radius (4·rSqMax) every remaining block is too far to cut — break.
      // radiusClosed ⇒ provably complete (the exhausted=K guard: a drained worklist that
      // never hit the break may have a neighbour outside the window, so flag it).
      bool radiusClosed = false;
      for (int g = 0; g < nOff && !ovf; ++g) {
        if (wlRmin(base + g) > Real(4) * c.rsq[c.vRsqMax]) {
          radiusClosed = true;
          break;
        }
        const int gc = worklistCell(base, g, cx, cy, cz);
        for (int q = cellStart(gc); q < cellStart(gc + 1) && !ovf; ++q) {
          if (q == pi) continue;
          if (haveGid && gidSorted(q) == gidSorted(pi)) continue;
          Real pv[3];
          relVec(q, pix, piy, piz, pv);
          const Real off = Real(0.5) * (pv[0] * pv[0] + pv[1] * pv[1] + pv[2] * pv[2]);
          if (off >= Real(2) * c.rsq[c.vRsqMax]) continue;  // beyond the radius: cannot cut
          c.cutCell2(pv, off, binned(q), &ovf);
        }
      }
      covSq = radiusClosed ? Real(1e30) : Real(0);
    }

    finishCell(c, i, pix, piy, piz, covSq, ovf);
  }

  /// Team-per-cell build: one team cooperates on the cell owning grid-sorted slot
  /// `pi`. The scratch cell + candidate arrays live in shared memory (passed in);
  /// `sh` is a small shared int scratch (>=4 ints) for the team's counters.
  ///
  /// The neighbour gather is parallelised across the team (Voronoi only); the cut and
  /// geometry stay on the leader for now. Bit-exactness is preserved because the
  /// Voronoi per-shell min-heap re-sorts each shell's candidates by key, so the cut
  /// order is independent of the (now parallel, racy) gather order. Power keeps the
  /// serial leader gather — it applies candidates in array order with no re-sort, so
  /// parallelising its gather would reorder the cuts; left for a later deterministic
  /// sort. Either way the cut is closest-first / radical-plane identical to buildCell.
  template <class Member, int CAP>
  KOKKOS_INLINE_FUNCTION void buildCellTeam(const Member& team, ScratchCell<Real, CAP>& c,
                                            Real* ckey, int* cjid, int maxCand, int* sh,
                                            int pi) const {
    const int i = binned(pi);
    const Real pix = posSorted(3 * pi + 0), piy = posSorted(3 * pi + 1), piz = posSorted(3 * pi + 2);
    int cx, cy, cz;
    const int base = homeCellBase(pix, piy, piz, cx, cy, cz);
    const Real wi = Weighted ? wSorted(pi) : Real(0);

    // sh[0]=nc, sh[1]=ovf, sh[2]=stop, sh[3]=radiusClosed. Leader seeds the cuboid + counters.
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      const Real Larr[3] = {Lx, Ly, Lz};
      c.initCuboid(Larr);
      sh[0] = 0;
      sh[1] = 0;
      sh[2] = 0;
      sh[3] = 0;
    });
    team.team_barrier();

    // Parallel gather of worklist offsets [a,b): each lane scans its grid cells and
    // atomically appends candidates to the shared ckey/cjid (slot from sh[0]).
    auto gatherRange = [&](int a, int b) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, a, b), [&](const int g) {
        const int gc = worklistCell(base, g, cx, cy, cz);
        for (int q = cellStart(gc); q < cellStart(gc + 1); ++q) {
          if (q == pi) continue;
          if (haveGid && gidSorted(q) == gidSorted(pi)) continue;
          Real pv[3];
          relVec(q, pix, piy, piz, pv);
          const Real rSqHalf = Real(0.5) * (pv[0] * pv[0] + pv[1] * pv[1] + pv[2] * pv[2]);
          const Real off = Weighted ? rSqHalf + Real(0.5) * (wi - wSorted(q)) : rSqHalf;
          const int slot = Kokkos::atomic_fetch_add(&sh[0], 1);
          if (slot < maxCand) {
            ckey[slot] = off;
            cjid[slot] = q;
          } else {
            Kokkos::atomic_fetch_add(&sh[1], 1);  // candidate overflow -> ovf
          }
        }
      });
    };

    Real covSq;
    if constexpr (Weighted) {
      // Power: gather + apply on the leader, in worklist order (no security early-out).
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        bool ovf = false;
        for (int g = 0; g < nOff && !ovf; ++g) {
          const int gc = worklistCell(base, g, cx, cy, cz);
          for (int q = cellStart(gc); q < cellStart(gc + 1) && !ovf; ++q) {
            if (q == pi) continue;
            if (haveGid && gidSorted(q) == gidSorted(pi)) continue;
            Real pv[3];
            relVec(q, pix, piy, piz, pv);
            const Real rSqHalf = Real(0.5) * (pv[0] * pv[0] + pv[1] * pv[1] + pv[2] * pv[2]);
            const Real off = rSqHalf + Real(0.5) * (wi - wSorted(q));
            c.cutCell2(pv, off, binned(q), &ovf);
          }
        }
        sh[1] = ovf ? 1 : 0;
      });
      team.team_barrier();
      covSq = Real(sw) * minCsz * Real(sw) * minCsz;
    } else {
      // Voronoi: process the worklist in chunks. The list is rmin-sorted, so if a chunk's
      // first (nearest) offset is already past the security radius, every remaining block
      // is too — stop (radiusClosed). Otherwise gather the chunk in parallel, then the
      // leader heap-sorts it and cuts closest-first with the security early-out. The
      // candidate buffer is reset per chunk (sh[0] back to 0), so it only holds one chunk.
      for (int a = 0; a < nOff; a += kTeamChunk) {
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          if (wlRmin(base + a) > Real(4) * c.rsq[c.vRsqMax]) {
            sh[2] = 1;
            sh[3] = 1;
          }
        });
        team.team_barrier();
        if (sh[2]) break;
        const int b = (a + kTeamChunk < nOff) ? a + kTeamChunk : nOff;
        gatherRange(a, b);  // appends from sh[0] == 0
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          int nc = sh[0];
          if (nc > maxCand) nc = maxCand;
          bool ovf = sh[1] != 0;
          Real* k = ckey;
          int* idp = cjid;
          int hn = nc;
          for (int s = hn / 2 - 1; s >= 0; --s) heapSiftDown(k, idp, s, hn);
          while (hn > 0 && !ovf) {
            if (k[0] >= Real(2) * c.rsq[c.vRsqMax]) break;
            const Real topKey = k[0];
            const int topId = idp[0];
            --hn;
            k[0] = k[hn];
            idp[0] = idp[hn];
            heapSiftDown(k, idp, 0, hn);
            Real pv[3];
            relVec(topId, pix, piy, piz, pv);
            c.cutCell2(pv, topKey, binned(topId), &ovf);
          }
          sh[0] = 0;  // reset candidate buffer for the next chunk
          sh[1] = ovf ? 1 : 0;
        });
        team.team_barrier();
        if (sh[1]) break;  // overflow -> all lanes stop
      }
      covSq = sh[3] ? Real(1e30) : Real(0);  // radiusClosed ⇒ provably complete
    }

    // Force-geometry, parallelised across the team: zero the per-facet accumulators
    // (leader; facets are few) then scatter each vertex's contribution in parallel with
    // atomic adds. Only for the no-SDF path (the geometry must run on the post-clip cell,
    // and the clip lives in the leader's finishCell); SDF cells keep the leader geometry.
    // The atomic scatter reorders the per-facet sum vs the serial order (~1e-15, well under
    // the 1e-9 oracle tolerance); the cut and topology stay bit-identical.
    const bool ovf = sh[1] != 0;
    bool geomDone = false;
    if constexpr (std::is_same_v<Sdf, NoSdf>) {
      if (withForceGeom && !ovf && !c.emptyV()) {
        Kokkos::single(Kokkos::PerTeam(team), [&]() { c.zeroGeometry(0, c.numAllocF); });
        team.team_barrier();
        const int nv = c.numAllocV;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, nv),
                             [&](const int vc) { c.template accumGeometryVertex<true>(vc); });
        team.team_barrier();
        geomDone = true;
      }
    }
    // Leader finishes the cell (status/volume/SDF clip/facet write; geometry already done
    // above when geomDone).
    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { finishCell(c, i, pix, piy, piz, covSq, ovf, geomDone); });
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
 */
template <class Real, bool Weighted, class Sdf = NoSdf>
TessellatorResult<Real> buildTessellation(const Kokkos::View<Real*, tpx::MemSpace>& posFlat,
                                          const Kokkos::View<Real*, tpx::MemSpace>& weight, int N,
                                          const Real L[3], int sw = 4, int densityCount = -1,
                                          Kokkos::View<long*, tpx::MemSpace> gid = {}, Sdf sdf = {},
                                          bool withForceGeom = true) {
  using tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  constexpr int MAXF = ScratchCell<Real>::CAP;
  // Optional global ids: skip a candidate sharing the cell's own id (its periodic
  // self-image, which can wrap exactly onto the seed -> a degenerate zero-distance
  // cut). Mirrors the legacy processNbrs `itr->id == m_id` guard. In the
  // single-domain case (gid empty) only the same local index is skipped.
  const bool haveGid = gid.extent(0) == static_cast<size_t>(N);

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
  // voro++-style gather: split each home grid cell into wlS³ sub-regions; for the
  // sub-region a seed lands in, store the (2sw+1)³ block offsets sorted by nearest-corner
  // distance² (`wlRmin`, absolute, from that sub-region to the block). The per-cell build
  // walks this presorted list and BREAKS once wlRmin exceeds the security radius (4·rSqMax)
  // — a table lookup, no runtime per-block geometry — so it touches only the blocks that
  // can cut, a tighter bound than a whole-cell shell sphere. wlS = host 3 / device 4
  // (the device tolerates the tighter bound / bigger table better); ported from
  // bench_convexcell's build_worklist. The full cube of offsets is kept (the rmin break
  // makes the far tail free); the build's radiusClosed flag is the exact completeness guard.
  const int wlS = kHostBackend ? 3 : 4;
  const int side = 2 * sw + 1;
  const int nOff = side * side * side;
  const int nSub = wlS * wlS * wlS;
  Kokkos::View<int*, MemSpace> wlOff(view_alloc(std::string("wlOff"), WithoutInitializing),
                                     (size_t)nSub * nOff);
  Kokkos::View<Real*, MemSpace> wlRmin(view_alloc(std::string("wlRmin"), WithoutInitializing),
                                       (size_t)nSub * nOff);
  {
    std::vector<int> oX(nOff), oY(nOff), oZ(nOff);
    {
      int t = 0;
      for (int dz = -sw; dz <= sw; ++dz)
        for (int dy = -sw; dy <= sw; ++dy)
          for (int dx = -sw; dx <= sw; ++dx) {
            oX[t] = dx;
            oY[t] = dy;
            oZ[t] = dz;
            ++t;
          }
    }
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
            const Real bx0 = oX[k] * cszx, bx1 = (oX[k] + 1) * cszx;
            const Real by0 = oY[k] * cszy, by1 = (oY[k] + 1) * cszy;
            const Real bz0 = oZ[k] * cszz, bz1 = (oZ[k] + 1) * cszz;
            rmn[k] = axisGap(ax0, ax1, bx0, bx1) + axisGap(ay0, ay1, by0, by1) +
                     axisGap(az0, az1, bz0, bz1);
            idx[k] = k;
          }
          std::sort(idx.begin(), idx.end(), [&](int a, int b) { return rmn[a] < rmn[b]; });
          for (int g = 0; g < nOff; ++g) {
            const int k = idx[g];
            const size_t slot = (size_t)p * nOff + g;
            hOff(slot) = (oX[k] + kWlOffBias) | ((oY[k] + kWlOffBias) << 8) |
                         ((oZ[k] + kWlOffBias) << 16);
            hRmin(slot) = rmn[k];
          }
        }
    Kokkos::deep_copy(wlOff, hOff);
    Kokkos::deep_copy(wlRmin, hRmin);
  }
  if (prof) std::fprintf(stderr, "[worklist] sw=%d nOff=%d nSub=%d\n", sw, nOff, nSub);

  // Build each cell with the CellBuilder functor. Two device parallelisations exist;
  // the per-cell numerics are identical (same functor) so they are bit-exact:
  //
  //   * default (host always; device default): one thread per cell, RangePolicy, the
  //     scratch cell on the thread stack / local frame and the cut applied on the fly
  //     (no candidate buffer). This is the production GPU path.
  //   * team-per-cell (device only, opt-in via VORFLOW_TEAM=<teamSize>): one *team* per
  //     cell with the scratch cell + candidate arrays in team shared memory, so the
  //     per-cell state lives on-chip rather than in a 32 KB/thread local frame. The
  //     occupancy is then shared-memory-limited (the cell is large), which only pays off
  //     once the gather and geometry are parallelised across the team; this is the
  //     in-progress redesign and stays behind the env flag until it beats the default.
  CellBuilder<Real, Weighted, Sdf> op{
      binned, posSorted, wSorted, gidSorted, cellStart, wlOff, wlRmin,
      status, cellVol, facetCount, cellFacetBase, oNbr, oArea, oDV, oConn, facetCursor,
      icx, icy, icz, Lx, Ly, Lz, minCsz, dimx, dimy, dimz, sw, nOff, wlS,
      useMorton, haveGid, withForceGeom, facetCap, sdf};
  using Builder = CellBuilder<Real, Weighted, Sdf>;
  // Team-per-cell shared footprint: a shrunk cell + candidate arrays so more teams fit
  // per SM (the team path's occupancy wall). A cell/gather that exceeds these is flagged
  // kOverflow and re-run at full capacity by the fallback pass below, so any value is
  // correct. Swept on an RTX 5080 at N=1M: throughput peaks at CAP~52 / maxCand~256
  // (≈1.55 Mcells/s pure-tess, 0.35% fallback) — a smaller CAP raises occupancy but the
  // fallback rate (cells past the shrunk vertex cap) climbs and turns it over; a smaller
  // maxCand starves the per-shell gather. VORFLOW_MAXCAND overrides the candidate buffer at
  // runtime for re-tuning on other GPUs (only the cell CAP is compile-time).
  constexpr int kCapShared = 52;
  constexpr int kMaxCandShared = 256;
  auto rangeBuild = [&]() {
    Kokkos::parallel_for(
        "tess.build", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
          ScratchCell<Real> c;
          op.buildCell(c, pi);
        });
  };
  if constexpr (kHostBackend) {
    rangeBuild();
  } else {
    // Power keeps the per-thread path: it has no security early-out, so it gathers the
    // whole coverage sphere (~nOff candidates) with no per-shell reuse — it would overflow
    // a shrunk candidate buffer on every cell — and its leader-serial gather/apply gains
    // nothing from a team. The shrunk team path is a Voronoi optimisation.
    const char* teamEnv = std::getenv("VORFLOW_TEAM");
    if (!teamEnv || Weighted) {
      rangeBuild();
    } else {
      // Team-per-cell: the shrunk cell + candidate arrays live in level-0 (shared) team
      // scratch. get_shmem_aligned carves the sub-buffers out (the cell's doubles need
      // 8-byte alignment). The build runs on the team leader except the parallel gather;
      // overflow cells (cut or candidate cap) are caught by the fallback pass below.
      using TeamPol = Kokkos::TeamPolicy<Exec>;
      using Member = typename TeamPol::member_type;
      using SharedCell = ScratchCell<Real, kCapShared>;
      const int teamSize = std::atoi(teamEnv) > 0 ? std::atoi(teamEnv) : 32;
      // Candidate buffer size is a runtime get_shmem allocation (only the cell CAP is a
      // compile-time template), so it can be tuned via VORFLOW_MAXCAND for the per-shell
      // gather; a shell that exceeds it overflows -> fallback. Default kMaxCandShared.
      int maxCand = kMaxCandShared;
      if (const char* e = std::getenv("VORFLOW_MAXCAND"))
        if (std::atoi(e) > 0) maxCand = std::atoi(e);
      const size_t shBytes = sizeof(SharedCell) + (size_t)maxCand * sizeof(Real) +
                             (size_t)maxCand * sizeof(int) + 8 * sizeof(int) +
                             128;  // + team counters + alignment pad
      TeamPol policy(N, teamSize);
      policy.set_scratch_size(0, Kokkos::PerTeam((int)shBytes));
      Kokkos::parallel_for(
          "tess.build", policy, KOKKOS_LAMBDA(const Member& team) {
            auto sc = team.team_scratch(0);
            auto* cp = (SharedCell*)sc.get_shmem_aligned(sizeof(SharedCell), alignof(SharedCell));
            Real* ckey =
                (Real*)sc.get_shmem_aligned((size_t)maxCand * sizeof(Real), alignof(Real));
            int* cjid = (int*)sc.get_shmem_aligned((size_t)maxCand * sizeof(int), alignof(int));
            int* sh = (int*)sc.get_shmem_aligned(8 * sizeof(int), alignof(int));
            op.buildCellTeam(team, *cp, ckey, cjid, maxCand, sh, team.league_rank());
          });
      // Fallback: re-run any cell the shrunk team path flagged kOverflow at full capacity
      // (CAP=128, MAXCAND=1024) on the per-thread path. Overflow is rare (random cells fit
      // the shrunk caps), so most threads early-out; the few re-runs append to the same
      // over-buffer cursor, keeping the CSR in finish-order.
      auto statusV = status;
      auto binnedV = binned;
      if (prof) {
        Kokkos::fence();
        long nOvf = 0;
        Kokkos::parallel_reduce(
            "tess.ovfCount", Kokkos::RangePolicy<Exec>(0, N),
            KOKKOS_LAMBDA(const int c, long& a) { a += (statusV(c) & kOverflow) ? 1 : 0; }, nOvf);
        std::fprintf(stderr, "[team] fallback cells=%ld/%d (%.2f%%) maxCand=%d CAP=%d\n", nOvf, N,
                     100.0 * (double)nOvf / N, maxCand, kCapShared);
      }
      Kokkos::parallel_for(
          "tess.fallback", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
            if (!(statusV(binnedV(pi)) & kOverflow)) return;
            ScratchCell<Real> c;
            op.buildCell(c, pi);
          });
    }
  }

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
                 2 * maxF - 4, ScratchCell<Real>::CAP);
  }

  TessellatorResult<Real> res;
  res.view = view;
  res.status = status;
  return res;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TESSELLATOR_HPP
