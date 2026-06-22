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
#include <cstdio>
#include <cstdlib>
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

  const bool prof = std::getenv("VORFLOW_PROFILE") != nullptr;
  Kokkos::Timer ptimer;
  double tGrid = 0, tBuild = 0, tCsr = 0;

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
  // Per-cell facet scratch (compacted into the CSR below). Sized by a tight facet
  // cap (Voronoi mean ~15, this comfortably covers the tail); cells that exceed it
  // are flagged kOverflow. Allocated WithoutInitializing — every slot the compaction
  // reads is written first, so the default zero-fill (here ~hundreds of MB/build,
  // pure memory-bandwidth that also throttles thread scaling) is wasted work. The
  // connecting vector is NOT stored — it is recomputed from the neighbour position
  // in the compaction (cheap, and saves a third of this traffic).
  constexpr int MAXF_TMP = 50;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  Kokkos::View<int*, MemSpace> tmpNbr(view_alloc(std::string("tmpNbr"), WithoutInitializing),
                                      (size_t)N * MAXF_TMP);
  Kokkos::View<Real*, MemSpace> tmpArea(view_alloc(std::string("tmpArea"), WithoutInitializing),
                                        (size_t)N * MAXF_TMP * 3);
  Kokkos::View<Real*, MemSpace> tmpDV(view_alloc(std::string("tmpDV"), WithoutInitializing),
                                      (size_t)N * MAXF_TMP * 3);

  const bool weighted = Weighted;
  Kokkos::parallel_for(
      "tess.build", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int pi) {
        // pi is the grid-sorted slot; i = binned(pi) the original seed index that
        // owns this cell. Positions/weights/ids are read from the reordered arrays
        // (cache-local), outputs are written back at the original index i.
        const int i = binned(pi);
        const Real pix = posSorted(3 * pi + 0), piy = posSorted(3 * pi + 1),
                   piz = posSorted(3 * pi + 2);
        int cx = ((((int)Kokkos::floor(pix * icx)) % dimx) + dimx) % dimx;
        int cy = ((((int)Kokkos::floor(piy * icy)) % dimy) + dimy) % dimy;
        int cz = ((((int)Kokkos::floor(piz * icz)) % dimz) + dimz) % dimz;

        // Candidate neighbours are seeds of the surrounding grid cells, gathered
        // (minimal-imaged) into ckey/cjid (sorted indices) and the cell cut closest-first.
        constexpr int MAXCAND = 1024;
        Real ckey[MAXCAND];  // plane offset (sort key)
        int cjid[MAXCAND];   // candidate sorted index
        int nc = 0;
        const Real wi = weighted ? wSorted(pi) : Real(0);
        bool ovf = false;

        ScratchCell<Real> c;
        const Real Larr[3] = {Lx, Ly, Lz};
        c.initCuboid(Larr);

        // Append the seeds of grid cell (gx,gy,gz) as minimal-imaged candidates.
        auto gatherGrid = [&](int gx, int gy, int gz) {
          int gc = gx + gy * dimx + gz * dimx * dimy;
          for (int q = cellStart(gc); q < cellStart(gc + 1); ++q) {
            if (q == pi) continue;
            if (haveGid && gidSorted(q) == gidSorted(pi)) continue;  // periodic self-image
            Real rx = posSorted(3 * q + 0) - pix;
            Real ry = posSorted(3 * q + 1) - piy;
            Real rz = posSorted(3 * q + 2) - piz;
            rx -= Lx * Kokkos::round(rx / Lx);
            ry -= Ly * Kokkos::round(ry / Ly);
            rz -= Lz * Kokkos::round(rz / Lz);
            Real rSqHalf = Real(0.5) * (rx * rx + ry * ry + rz * rz);
            Real off = weighted ? rSqHalf + Real(0.5) * (wi - wSorted(q)) : rSqHalf;
            if (nc < MAXCAND) {
              ckey[nc] = off;
              cjid[nc] = q;
              ++nc;
            } else {
              ovf = true;  // candidate overflow -> flag for fallback
            }
          }
        };
        auto applyCand = [&](int q, Real off) {
          Real rx = posSorted(3 * q + 0) - pix;
          Real ry = posSorted(3 * q + 1) - piy;
          Real rz = posSorted(3 * q + 2) - piz;
          rx -= Lx * Kokkos::round(rx / Lx);
          ry -= Ly * Kokkos::round(ry / Ly);
          rz -= Lz * Kokkos::round(rz / Lz);
          const Real pv[3] = {rx, ry, rz};
          c.cutCell2(pv, off, binned(q), &ovf);  // store the ORIGINAL neighbour id
        };

        // Coverage² actually attained for this cell (drives the security check). A
        // grid cell at offset (dx,dy,dz) has nearest-corner distance max(0,|·|-1)
        // cells, so cells with e²>sw² lie outside the coverage sphere of radius
        // sw·h and never matter.
        Real covSq;
        if (weighted) {
          // Power has no security early-out (a distant heavy seed can still cut), so
          // gather the whole coverage sphere and apply every candidate unsorted.
          const int sw2 = sw * sw;
          for (int dz = -(sw + 1); dz <= sw + 1; ++dz) {
            const int ez = dz < -1 ? -dz - 1 : (dz > 1 ? dz - 1 : 0);
            if (ez * ez > sw2) continue;
            for (int dy = -(sw + 1); dy <= sw + 1; ++dy) {
              const int ey = dy < -1 ? -dy - 1 : (dy > 1 ? dy - 1 : 0);
              if (ez * ez + ey * ey > sw2) continue;
              for (int dx = -(sw + 1); dx <= sw + 1; ++dx) {
                const int ex = dx < -1 ? -dx - 1 : (dx > 1 ? dx - 1 : 0);
                if (ex * ex + ey * ey + ez * ez > sw2) continue;
                gatherGrid((((cx + dx) % dimx) + dimx) % dimx, (((cy + dy) % dimy) + dimy) % dimy,
                           (((cz + dz) % dimz) + dimz) % dimz);
              }
            }
          }
          for (int s = 0; s < nc && !ovf; ++s) applyCand(cjid[s], ckey[s]);
          covSq = Real(sw) * minCsz * Real(sw) * minCsz;
        } else {
          // Adaptive expanding search (legacy's tight per-cell coverage): grow the
          // gathered sphere one grid-shell at a time, process each new shell
          // closest-first via a min-heap, and stop the moment the security radius is
          // met (coverage² > 4·rSqMax). Most cells close at the innermost level, so
          // we never pay the full sw-block gather/cut a fixed radius would force.
          const int swInit = sw < 2 ? sw : 2;
          int swUsed = swInit;
          for (int swl = swInit; swl <= sw && !ovf; ++swl) {
            const int lo2 = (swl == swInit) ? -1 : (swl - 1) * (swl - 1);
            const int hi2 = swl * swl;
            const int shellStart = nc;
            for (int dz = -(swl + 1); dz <= swl + 1; ++dz) {
              const int ez = dz < -1 ? -dz - 1 : (dz > 1 ? dz - 1 : 0);
              if (ez * ez > hi2) continue;
              for (int dy = -(swl + 1); dy <= swl + 1; ++dy) {
                const int ey = dy < -1 ? -dy - 1 : (dy > 1 ? dy - 1 : 0);
                if (ez * ez + ey * ey > hi2) continue;
                for (int dx = -(swl + 1); dx <= swl + 1; ++dx) {
                  const int ex = dx < -1 ? -dx - 1 : (dx > 1 ? dx - 1 : 0);
                  const int d2 = ex * ex + ey * ey + ez * ez;
                  if (d2 <= lo2 || d2 > hi2) continue;  // not in this shell
                  gatherGrid((((cx + dx) % dimx) + dimx) % dimx, (((cy + dy) % dimy) + dimy) % dimy,
                             (((cz + dz) % dimz) + dimz) % dimz);
                }
              }
            }
            // Process this shell's new candidates closest-first, stopping once the
            // nearest unprocessed one is beyond the security radius (2·rSqMax).
            Real* k = ckey + shellStart;
            int* idp = cjid + shellStart;
            int hn = nc - shellStart;
            for (int s = hn / 2 - 1; s >= 0; --s) heapSiftDown(k, idp, s, hn);
            while (hn > 0 && !ovf) {
              if (k[0] >= Real(2) * c.rsq[c.vRsqMax]) break;
              const Real topKey = k[0];
              const int topId = idp[0];
              --hn;
              k[0] = k[hn];
              idp[0] = idp[hn];
              heapSiftDown(k, idp, 0, hn);
              applyCand(topId, topKey);
            }
            swUsed = swl;
            if (Real(swl) * minCsz * Real(swl) * minCsz > Real(4) * c.rsq[c.vRsqMax]) break;
          }
          covSq = Real(swUsed) * minCsz * Real(swUsed) * minCsz;
        }

        // Voronoi completeness is judged on the un-clipped cell (the SDF clip only
        // shrinks it, which would mask an incomplete neighbour search).
        const bool incomplete = !(covSq > Real(4) * c.rsq[c.vRsqMax]);

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
          if (!c.aliveF[f])
            continue;
          if (nf >= MAXF_TMP) {  // more facets than the compaction scratch holds
            status(i) |= kOverflow;
            break;
          }
          const size_t o = ((size_t)i * MAXF_TMP + nf) * 3;
          tmpNbr((size_t)i * MAXF_TMP + nf) = c.fnbr[f];
          for (int cc = 0; cc < 3; ++cc) {
            tmpArea(o + cc) = c.fArea[f][cc];
            tmpDV(o + cc) = c.fdV[f][cc];
          }
          ++nf;
        }
        facetCount(i) = nf;
      });

  if (prof) {
    Kokkos::fence();
    tBuild = ptimer.seconds();
    ptimer.reset();
  }

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
  view.cellSeedId = Kokkos::View<gid_t*, MemSpace>(view_alloc(std::string("cellSeedId"),
                                                              WithoutInitializing), N);
  view.facetNeighbor = Kokkos::View<gid_t*, MemSpace>(
      view_alloc(std::string("facetNeighbor"), WithoutInitializing), nFacets);
  view.facetArea = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("facetArea"), WithoutInitializing), (size_t)nFacets * 3);
  view.facetConnect = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("facetConnect"), WithoutInitializing), (size_t)nFacets * 3);
  view.facetConnVec = Kokkos::View<Real*, MemSpace>(
      view_alloc(std::string("facetConnVec"), WithoutInitializing), (size_t)nFacets * 3);
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
        const Real pix = posFlat(3 * i + 0), piy = posFlat(3 * i + 1), piz = posFlat(3 * i + 2);
        for (int k = 0; k < nf; ++k) {
          int j = tmpNbr((size_t)i * MAXF_TMP + k);
          vNbr(base + k) = (gid_t)j;
          // Recompute the connecting vector (minimal image of pos[j]-pos[i]); bit-
          // identical to the pvec the cutter used, so it need not be stored.
          Real rx = posFlat(3 * j + 0) - pix;
          Real ry = posFlat(3 * j + 1) - piy;
          Real rz = posFlat(3 * j + 2) - piz;
          rx -= Lx * Kokkos::round(rx / Lx);
          ry -= Ly * Kokkos::round(ry / Ly);
          rz -= Lz * Kokkos::round(rz / Lz);
          const Real conn[3] = {rx, ry, rz};
          for (int cc = 0; cc < 3; ++cc) {
            const size_t src = ((size_t)i * MAXF_TMP + k) * 3 + cc;
            const size_t dst = (size_t)(base + k) * 3 + cc;
            vArea(dst) = tmpArea(src);
            vDV(dst) = tmpDV(src);
            vConn(dst) = conn[cc];
          }
        }
      });
  Kokkos::fence();

  if (prof) {
    tCsr = ptimer.seconds();
    std::fprintf(stderr, "[tess N=%d] grid=%.4fs build=%.4fs csr=%.4fs (build %.0f%%)\n", N, tGrid,
                 tBuild, tCsr, 100.0 * tBuild / (tGrid + tBuild + tCsr));
  }

  TessellatorResult<Real> res;
  res.view = view;
  res.status = status;
  return res;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TESSELLATOR_HPP
