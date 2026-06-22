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

  // --- gather offset worklist (computed once) ---
  // The (dx,dy,dz) grid offsets inside the coverage sphere (nearest-corner distance²
  // = max(0,|·|-1)² summed ≤ sw²), grouped by expanding shell. The per-cell build
  // iterates this flat list instead of re-running the dz/dy/dx cube triple-loop with
  // the d² filter every cell — that loop control (≈ hundreds–thousands of branchy
  // integer ops/cell) was the bulk of the gather cost (Phase 0).
  const int swInit = sw < 2 ? sw : 2;
  std::vector<int> offHx, offHy, offHz;
  std::vector<int> shellStartH(sw + 2, 0);
  {
    const int sw2 = sw * sw;
    for (int swl = swInit; swl <= sw; ++swl) {
      shellStartH[swl] = (int)offHx.size();
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
    shellStartH[sw + 1] = (int)offHx.size();
  }
  const int nOff = (int)offHx.size();
  if (prof) std::fprintf(stderr, "[worklist] sw=%d nOff=%d (full-sphere candidate cap)\n", sw, nOff);
  Kokkos::View<int*, MemSpace> offX(view_alloc(std::string("offX"), WithoutInitializing), nOff);
  Kokkos::View<int*, MemSpace> offY(view_alloc(std::string("offY"), WithoutInitializing), nOff);
  Kokkos::View<int*, MemSpace> offZ(view_alloc(std::string("offZ"), WithoutInitializing), nOff);
  Kokkos::View<int*, MemSpace> shellStart(
      view_alloc(std::string("shellStart"), WithoutInitializing), sw + 2);
  {
    auto hX = Kokkos::create_mirror_view(offX);
    auto hY = Kokkos::create_mirror_view(offY);
    auto hZ = Kokkos::create_mirror_view(offZ);
    auto hS = Kokkos::create_mirror_view(shellStart);
    for (int i = 0; i < nOff; ++i) {
      hX(i) = offHx[i];
      hY(i) = offHy[i];
      hZ(i) = offHz[i];
    }
    for (int s = 0; s < sw + 2; ++s) hS(s) = shellStartH[s];
    Kokkos::deep_copy(offX, hX);
    Kokkos::deep_copy(offY, hY);
    Kokkos::deep_copy(offZ, hZ);
    Kokkos::deep_copy(shellStart, hS);
  }

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

        // Half-box extents for the single-image minimal-image wrap below.
        const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;

        // Candidate neighbours are seeds of the surrounding grid cells, gathered
        // (minimal-imaged) into ckey/cjid (sorted indices) and the cell cut closest-first.
        // Sized for the full-sphere gather: any cell that expands to sw, and every Power
        // cell (no early-out), gathers ~nOff candidates (667 at sw=4, plus Poisson
        // fluctuation), so the cap cannot be meaningfully reduced — see docs/performance.md.
        constexpr int MAXCAND = 1024;
        Real ckey[MAXCAND];  // plane offset (sort key)
        int cjid[MAXCAND];   // candidate sorted index
        int nc = 0;
        const Real wi = weighted ? wSorted(pi) : Real(0);
        bool ovf = false;

        ScratchCell<Real> c;
        const Real Larr[3] = {Lx, Ly, Lz};
        c.initCuboid(Larr);

        // Append the seeds of grid cell at raw offset (rgx,rgy,rgz) as candidates.
        // The grid index and the minimal image both wrap with a single add/sub (offsets
        // are << dim and candidates are within one box image) — branchy compares instead
        // of the %dim modulo and round(r/L), bit-identical but far cheaper, and (unlike a
        // per-cell "interior" flag) independent of sw, so a large safety sw stays fast.
        auto gatherGrid = [&](int rgx, int rgy, int rgz) {
          int gx = rgx < 0 ? rgx + dimx : (rgx >= dimx ? rgx - dimx : rgx);
          int gy = rgy < 0 ? rgy + dimy : (rgy >= dimy ? rgy - dimy : rgy);
          int gz = rgz < 0 ? rgz + dimz : (rgz >= dimz ? rgz - dimz : rgz);
          int gc = useMorton ? morton3(gx, gy, gz) : (gx + gy * dimx + gz * dimx * dimy);
          for (int q = cellStart(gc); q < cellStart(gc + 1); ++q) {
            if (q == pi) continue;
            if (haveGid && gidSorted(q) == gidSorted(pi)) continue;  // periodic self-image
            Real rx = posSorted(3 * q + 0) - pix;
            Real ry = posSorted(3 * q + 1) - piy;
            Real rz = posSorted(3 * q + 2) - piz;
            rx = rx > Lxh ? rx - Lx : (rx < -Lxh ? rx + Lx : rx);
            ry = ry > Lyh ? ry - Ly : (ry < -Lyh ? ry + Ly : ry);
            rz = rz > Lzh ? rz - Lz : (rz < -Lzh ? rz + Lz : rz);
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
          rx = rx > Lxh ? rx - Lx : (rx < -Lxh ? rx + Lx : rx);
          ry = ry > Lyh ? ry - Ly : (ry < -Lyh ? ry + Ly : ry);
          rz = rz > Lzh ? rz - Lz : (rz < -Lzh ? rz + Lz : rz);
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
          // gather the whole coverage sphere (every worklist offset) and apply each
          // candidate unsorted.
          for (int o = 0; o < nOff; ++o) gatherGrid(cx + offX(o), cy + offY(o), cz + offZ(o));
          for (int s = 0; s < nc && !ovf; ++s) applyCand(cjid[s], ckey[s]);
          covSq = Real(sw) * minCsz * Real(sw) * minCsz;
        } else {
          // Adaptive expanding search (legacy's tight per-cell coverage): grow the
          // gathered sphere one shell of the worklist at a time, process each new
          // shell closest-first via a min-heap, and stop the moment the security
          // radius is met (coverage² > 4·rSqMax). Most cells close at the innermost
          // shell, so the full sw-block is never gathered.
          int swUsed = swInit;
          for (int swl = swInit; swl <= sw && !ovf; ++swl) {
            const int shellBase = nc;
            const int o0 = shellStart(swl), o1 = shellStart(swl + 1);
            for (int o = o0; o < o1; ++o) gatherGrid(cx + offX(o), cy + offY(o), cz + offZ(o));
            // Process this shell's new candidates closest-first, stopping once the
            // nearest unprocessed one is beyond the security radius (2·rSqMax).
            Real* k = ckey + shellBase;
            int* idp = cjid + shellBase;
            int hn = nc - shellBase;
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
        // Per-facet geometry (area vector + dV volume gradient) from the half-edge
        // mesh — what the physics forces consume. The O(27·V) gradient is ~37% of the
        // per-cell build; skip it (and publish zero area/dV) when forces aren't needed
        // (pure tessellation), the apples-to-apples comparison with voro++.
        if (!c.emptyV() && withForceGeom)
          c.computeGeometry();
        // Collect this cell's live facets, then reserve a contiguous CSR range with one
        // atomic and write straight into the over-buffer (no temp slab / scan / compaction).
        int faces[MAXF_TMP];
        int nf = 0;
        for (int f = 0; f < c.numAllocF; ++f) {
          if (!c.aliveF[f])
            continue;
          if (nf >= MAXF_TMP) {
            status(i) |= kOverflow;
            break;
          }
          faces[nf++] = f;
        }
        const int base = Kokkos::atomic_fetch_add(&facetCursor(0), nf);
        // Overflow guard: if this cell's reserved range would run past the over-buffer
        // (sized from the mean-facet estimate, not the worst case), flag it and skip
        // the write rather than clobbering memory past the end. The atomic still
        // advanced, so the final cursor may read beyond capacity; the pack clamps to it.
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
  }

  TessellatorResult<Real> res;
  res.view = view;
  res.status = status;
  return res;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TESSELLATOR_HPP
