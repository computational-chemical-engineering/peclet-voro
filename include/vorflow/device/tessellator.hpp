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

#include <cmath>
#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>

#include "tpx/common/view.hpp"
#include "vorflow/device/cell_cutter.hpp"
#include "vorflow/device/sdf.hpp"
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
                                          Kokkos::View<long*, tpx::MemSpace> gid = {},
                                          Sdf sdf = {}) {
  using tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  constexpr int MAXF = ScratchCell<Real>::CAP;
  // Optional global ids: skip a candidate sharing the cell's own id (its periodic
  // self-image, which can wrap exactly onto the seed -> a degenerate zero-distance
  // cut). Mirrors the legacy processNbrs `itr->id == m_id` guard. In the
  // single-domain case (gid empty) only the same local index is skipped.
  const bool haveGid = gid.extent(0) == static_cast<size_t>(N);

  // --- grid dimensions: ~1 seed per cell (cellSize ~ mean spacing) ---
  const int dens = densityCount > 0 ? densityCount : N;
  const Real vol = L[0] * L[1] * L[2];
  const Real spacing = std::cbrt(vol / Real(dens > 0 ? dens : 1));
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
  const Real Lx = L[0], Ly = L[1], Lz = L[2];
  const Real icx = invcsz[0], icy = invcsz[1], icz = invcsz[2];
  Real minCsz = csz[0] < csz[1] ? csz[0] : csz[1];
  minCsz = minCsz < csz[2] ? minCsz : csz[2];
  const Real coverageSq = Real(sw) * minCsz * Real(sw) * minCsz;

  // --- counting sort: bin seeds into grid cells ---
  Kokkos::View<int*, MemSpace> cellOf("cellOf", N);
  Kokkos::View<int*, MemSpace> counts("counts", ncell + 1);
  Kokkos::deep_copy(counts, 0);
  Kokkos::parallel_for(
      "tess.bin", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int gx = (int)Kokkos::floor(posFlat(3 * i + 0) * icx);
        int gy = (int)Kokkos::floor(posFlat(3 * i + 1) * icy);
        int gz = (int)Kokkos::floor(posFlat(3 * i + 2) * icz);
        gx = ((gx % dimx) + dimx) % dimx;
        gy = ((gy % dimy) + dimy) % dimy;
        gz = ((gz % dimz) + dimz) % dimz;
        int c = gx + gy * dimx + gz * dimx * dimy;
        cellOf(i) = c;
        Kokkos::atomic_inc(&counts(c));
      });

  Kokkos::View<int*, MemSpace> cellStart("cellStart", ncell + 1);
  Kokkos::parallel_scan(
      "tess.scan", Kokkos::RangePolicy<Exec>(0, ncell + 1),
      KOKKOS_LAMBDA(const int c, int& acc, const bool fin) {
        int v = (c < ncell) ? counts(c) : 0;
        if (fin)
          cellStart(c) = acc;
        acc += v;
      });

  Kokkos::View<int*, MemSpace> cursor("cursor", ncell);
  Kokkos::parallel_for(
      "tess.cursor", Kokkos::RangePolicy<Exec>(0, ncell),
      KOKKOS_LAMBDA(const int c) { cursor(c) = cellStart(c); });
  Kokkos::View<int*, MemSpace> binned("binned", N);
  Kokkos::parallel_for(
      "tess.scatter", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int slot = Kokkos::atomic_fetch_add(&cursor(cellOf(i)), 1);
        binned(slot) = i;
      });

  // --- pass 1: build each cell, record volume / facet count / status and the
  //     per-cell facet records (neighbour id + area vector) into a temp slab ---
  Kokkos::View<Real*, MemSpace> cellVol("cellVol", N);
  Kokkos::View<int*, MemSpace> facetCount("facetCount", N);
  Kokkos::View<int*, MemSpace> status("status", N);
  Kokkos::View<int*, MemSpace> tmpNbr("tmpNbr", (size_t)N * MAXF);
  Kokkos::View<Real*, MemSpace> tmpArea("tmpArea", (size_t)N * MAXF * 3);
  Kokkos::View<Real*, MemSpace> tmpDV("tmpDV", (size_t)N * MAXF * 3);
  Kokkos::View<Real*, MemSpace> tmpConn("tmpConn", (size_t)N * MAXF * 3);

  const bool weighted = Weighted;
  Kokkos::parallel_for(
      "tess.build", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real pix = posFlat(3 * i + 0), piy = posFlat(3 * i + 1), piz = posFlat(3 * i + 2);
        int cx = ((((int)Kokkos::floor(pix * icx)) % dimx) + dimx) % dimx;
        int cy = ((((int)Kokkos::floor(piy * icy)) % dimy) + dimy) % dimy;
        int cz = ((((int)Kokkos::floor(piz * icz)) % dimz) + dimz) % dimz;

        // Gather the surrounding grid block's seeds as candidates (offset + id),
        // then process them closest-first so the cell shrinks fast and the
        // security early-out prunes the long tail (legacy processNbrs order).
        constexpr int MAXCAND = 1024;
        Real ckey[MAXCAND];  // plane offset (sort key)
        int cjid[MAXCAND];   // candidate local index
        int nc = 0;
        const Real wi = weighted ? weight(i) : Real(0);
        bool ovf = false;
        for (int dz = -sw; dz <= sw; ++dz)
          for (int dy = -sw; dy <= sw; ++dy)
            for (int dx = -sw; dx <= sw; ++dx) {
              int gx = (((cx + dx) % dimx) + dimx) % dimx;
              int gy = (((cy + dy) % dimy) + dimy) % dimy;
              int gz = (((cz + dz) % dimz) + dimz) % dimz;
              int gc = gx + gy * dimx + gz * dimx * dimy;
              for (int p = cellStart(gc); p < cellStart(gc + 1); ++p) {
                int j = binned(p);
                if (j == i)
                  continue;
                if (haveGid && gid(j) == gid(i))
                  continue;  // periodic self-image
                Real rx = posFlat(3 * j + 0) - pix;
                Real ry = posFlat(3 * j + 1) - piy;
                Real rz = posFlat(3 * j + 2) - piz;
                rx -= Lx * Kokkos::round(rx / Lx);
                ry -= Ly * Kokkos::round(ry / Ly);
                rz -= Lz * Kokkos::round(rz / Lz);
                Real rSqHalf = Real(0.5) * (rx * rx + ry * ry + rz * rz);
                Real off = weighted ? rSqHalf + Real(0.5) * (wi - weight(j)) : rSqHalf;
                if (nc < MAXCAND) {
                  ckey[nc] = off;
                  cjid[nc] = j;
                  ++nc;
                } else {
                  ovf = true;  // candidate overflow -> flag for fallback
                }
              }
            }

        ScratchCell<Real> c;
        const Real Larr[3] = {Lx, Ly, Lz};
        c.initCuboid(Larr);
        auto applyCand = [&](int j, Real off) {
          Real rx = posFlat(3 * j + 0) - pix;
          Real ry = posFlat(3 * j + 1) - piy;
          Real rz = posFlat(3 * j + 2) - piz;
          rx -= Lx * Kokkos::round(rx / Lx);
          ry -= Ly * Kokkos::round(ry / Ly);
          rz -= Lz * Kokkos::round(rz / Lz);
          const Real pv[3] = {rx, ry, rz};
          c.cutCell2(pv, off, j, &ovf);
        };
        if (weighted) {
          // Power has no security early-out (a distant heavy seed can still cut)
          // and the cut is order-independent, so apply every candidate unsorted.
          for (int s = 0; s < nc && !ovf; ++s)
            applyCand(cjid[s], ckey[s]);
        } else {
          // Repeatedly take the closest unprocessed candidate, stopping as soon as
          // it is beyond the security radius (2·rSqMax) — only the handful that can
          // actually cut are touched, so no full O(n²) sort of the block.
          for (int done = 0; done < nc && !ovf; ++done) {
            int m = done;
            for (int s = done + 1; s < nc; ++s)
              if (ckey[s] < ckey[m])
                m = s;
            if (ckey[m] >= Real(2) * c.rsq[c.vRsqMax])
              break;
            Real tk = ckey[done];
            int tj = cjid[done];
            ckey[done] = ckey[m];
            cjid[done] = cjid[m];
            ckey[m] = tk;
            cjid[m] = tj;
            applyCand(cjid[done], ckey[done]);
          }
        }

        // Voronoi completeness is judged on the un-clipped cell (the SDF clip only
        // shrinks it, which would mask an incomplete neighbour search).
        const bool incomplete = !(coverageSq > Real(4) * c.rsq[c.vRsqMax]);

        // Optional SDF boundary clip: clip the cell to the fluid region (sdf > 0),
        // emptying it if the seed is in the solid.
        // (CUDA extended lambdas cannot first-capture a variable inside an
        // if-constexpr, so reference sdf once in normal context first.)
        (void)sdf;
        if constexpr (!std::is_same_v<Sdf, NoSdf>) {
          const Real seedW[3] = {pix, piy, piz};
          clipCellAgainstSdf<Real>(c, seedW, sdf, &ovf);
        }

        int st = kOk;
        if (ovf)
          st |= kOverflow;
        if (c.emptyV())
          st |= kEmpty;
        if (incomplete)
          st |= kIncomplete;
        status(i) = st;
        cellVol(i) = c.volume();
        // Per-facet geometry (area vector, dV volume gradient, connecting vector)
        // from the half-edge mesh — what the physics forces consume.
        if (!c.emptyV())
          c.computeGeometry();
        int nf = 0;
        for (int f = 0; f < c.numAllocF; ++f) {
          if (!c.aliveF[f] || nf >= MAXF)
            continue;
          const size_t o = ((size_t)i * MAXF + nf) * 3;
          tmpNbr((size_t)i * MAXF + nf) = c.fnbr[f];
          for (int cc = 0; cc < 3; ++cc) {
            tmpArea(o + cc) = c.fArea[f][cc];
            tmpDV(o + cc) = c.fdV[f][cc];
            tmpConn(o + cc) = c.pvec[f][cc];
          }
          ++nf;
        }
        facetCount(i) = nf;
      });

  // --- exclusive scan of facet counts -> CSR offsets ---
  Kokkos::View<int*, MemSpace> offset("cellFacetOffset", N + 1);
  Kokkos::parallel_scan(
      "tess.facetScan", Kokkos::RangePolicy<Exec>(0, N + 1),
      KOKKOS_LAMBDA(const int i, int& acc, const bool fin) {
        int v = (i < N) ? facetCount(i) : 0;
        if (fin)
          offset(i) = acc;
        acc += v;
      });
  int nFacets = 0;
  Kokkos::deep_copy(nFacets, Kokkos::subview(offset, N));

  // --- compaction: temp slab -> packed CSR ---
  TessellationView<Real> view;
  view.cellFacetOffset = offset;
  view.cellVolume = cellVol;
  view.cellSeedId = Kokkos::View<gid_t*, MemSpace>("cellSeedId", N);
  view.facetNeighbor = Kokkos::View<gid_t*, MemSpace>("facetNeighbor", nFacets);
  view.facetArea = Kokkos::View<Real*, MemSpace>("facetArea", (size_t)nFacets * 3);
  view.facetConnect = Kokkos::View<Real*, MemSpace>("facetConnect", (size_t)nFacets * 3);
  view.facetConnVec = Kokkos::View<Real*, MemSpace>("facetConnVec", (size_t)nFacets * 3);
  auto vSeed = view.cellSeedId;
  auto vNbr = view.facetNeighbor;
  auto vArea = view.facetArea;
  auto vDV = view.facetConnect;
  auto vConn = view.facetConnVec;
  Kokkos::parallel_for(
      "tess.compact", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        vSeed(i) = (gid_t)i;
        int base = offset(i);
        int nf = facetCount(i);
        for (int k = 0; k < nf; ++k) {
          vNbr(base + k) = (gid_t)tmpNbr((size_t)i * MAXF + k);
          for (int cc = 0; cc < 3; ++cc) {
            const size_t src = ((size_t)i * MAXF + k) * 3 + cc;
            const size_t dst = (size_t)(base + k) * 3 + cc;
            vArea(dst) = tmpArea(src);
            vDV(dst) = tmpDV(src);
            vConn(dst) = tmpConn(src);
          }
        }
      });
  Kokkos::fence();

  TessellatorResult<Real> res;
  res.view = view;
  res.status = status;
  return res;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TESSELLATOR_HPP
