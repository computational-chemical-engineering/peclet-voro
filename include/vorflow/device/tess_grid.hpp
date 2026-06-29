/**
 * @file device/tess_grid.hpp
 * \brief The counting-sort grid + presorted worklist that drives the Voronoi gather — factored
 *        out of buildTessellation so the SAME grid backs both the cold build and the moving-point
 *        subset gather (Part II). There is exactly ONE gather path; this header builds the inputs
 *        it reads, `buildTessellation` (and `subsetGather`) consume them.
 *
 * `buildTessGrid` reproduces buildTessellation's first half verbatim (counting-sort bin -> scan ->
 * scatter -> grid-order reorder, then the per-sub-position worklist table) and returns a `TessGrid`
 * holding the grid-sorted inputs + the worklist + the grid scalars. It additionally fills `slotOf`
 * (the inverse of `binned`: original seed index -> grid-sorted slot), which the subset gather needs
 * to launch the cold-build kernel over an arbitrary list of ORIGINAL indices (the device analogue of
 * the legacy NbrList::setupSubset). The cold-build output stays byte-for-byte identical (pure code
 * motion — the kernels and their launch order are unchanged).
 *
 * Core header: Kokkos + transport-core + morton, no physics.
 */
#ifndef VORFLOW_DEVICE_TESS_GRID_HPP
#define VORFLOW_DEVICE_TESS_GRID_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "morton/morton.hpp"  // suite spatial-index primitive (after Kokkos_Core: MORTON_HD->KOKKOS_FUNCTION)
#include "tpx/common/view.hpp"
#include "vorflow/tessellation_view.hpp"  // gid_t

namespace vor {
namespace device {

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

/// Grid-sorted inputs + presorted worklist + grid scalars. Trivially copyable (Views are
/// reference-counted handles), so it captures by value into device lambdas.
template <class Real>
struct TessGrid {
  using MemSpace = tpx::MemSpace;
  // Grid-sorted inputs (read by the gather).
  Kokkos::View<int*, MemSpace> binned;     // N: grid-sorted slot -> original seed index
  Kokkos::View<int*, MemSpace> slotOf;     // N: original seed index -> grid-sorted slot (inverse of binned)
  Kokkos::View<Real*, MemSpace> posSorted; // 3N (grid order)
  Kokkos::View<Real*, MemSpace> wSorted;   // N (Power) or 0
  Kokkos::View<gid_t*, MemSpace> gidSorted;// N (multi-domain) or 0
  Kokkos::View<int*, MemSpace> cellStart;  // ncellEff+1
  // Per-sub-position worklist (both backends).
  Kokkos::View<int*, MemSpace> wlOff;      // nSub*nOff : packed (dx,dy,dz)+kWlOffBias
  Kokkos::View<Real*, MemSpace> wlRmin;    // nSub*nOff : nearest-corner dist^2 (sorted)
  // Grid scalars.
  Real icx = 0, icy = 0, icz = 0, Lx = 0, Ly = 0, Lz = 0, minCsz = 0;
  int dimx = 0, dimy = 0, dimz = 0, sw = 0, nOff = 0, wlS = 0, N = 0;
  bool useMorton = false, haveGid = false;
};

/// Cache for the step-invariant presorted worklist table (E3). The table depends only on the grid
/// dimensions, the search width `sw`, and the cell sizes `csz` — none of which change as the seeds
/// move — so a stepper that rebuilds the tessellation every step can reuse it instead of redoing the
/// host std::sort + two H2D copies each time. Held by value in the simulation (Views are
/// reference-counted handles); passed by pointer into buildTessGrid/buildTessellation. Holding the
/// Views in the owning object (not a function-local static) keeps them freed before Kokkos::finalize.
template <class Real>
struct WorklistCache {
  Kokkos::View<int*, tpx::MemSpace> wlOff;
  Kokkos::View<Real*, tpx::MemSpace> wlRmin;
  int nOff = 0, wlS = 0, dimx = -1, dimy = -1, dimz = -1, sw = -1;
  Real cszx = 0, cszy = 0, cszz = 0;
  bool valid = false;
  bool matches(int dx, int dy, int dz, int s, Real cx, Real cy, Real cz) const {
    return valid && dimx == dx && dimy == dy && dimz == dz && sw == s && cszx == cx &&
           cszy == cy && cszz == cz;
  }
};

/**
 * Build the counting-sort grid + worklist for the Voronoi gather (the first half of the old
 * buildTessellation, unchanged). `Weighted` only affects the grid density (Power keeps 1 seed/cell;
 * unweighted host uses 2/cell for cache locality). See buildTessellation for the argument semantics.
 * `wlc` (optional) reuses/fills the step-invariant worklist table across calls (E3); nullptr rebuilds
 * it every call (the original behaviour, byte-for-byte).
 */
template <class Real, bool Weighted>
TessGrid<Real> buildTessGrid(const Kokkos::View<Real*, tpx::MemSpace>& posFlat,
                             const Kokkos::View<Real*, tpx::MemSpace>& weight, int N,
                             const Real L[3], int sw = 4, int densityCount = -1,
                             Kokkos::View<long*, tpx::MemSpace> gid = {},
                             WorklistCache<Real>* wlc = nullptr) {
  using tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  TessGrid<Real> grid;
  grid.N = N;
  grid.sw = sw;
  // Optional global ids: skip a candidate sharing the cell's own id (periodic self-image).
  const bool haveGid = gid.extent(0) == static_cast<size_t>(N);
  grid.haveGid = haveGid;

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
  grid.icx = icx; grid.icy = icy; grid.icz = icz;
  grid.Lx = Lx; grid.Ly = Ly; grid.Lz = Lz; grid.minCsz = minCsz;
  grid.dimx = dimx; grid.dimy = dimy; grid.dimz = dimz;
  grid.useMorton = useMorton;

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
  // (= binned(p)), so the cell i == particle i contract is unchanged. `slotOf` records
  // the inverse map so the subset gather can address cells by their ORIGINAL index.
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  Kokkos::View<Real*, MemSpace> posSorted(view_alloc(std::string("posSorted"), WithoutInitializing),
                                          (size_t)N * 3);
  Kokkos::View<int*, MemSpace> slotOf(view_alloc(std::string("slotOf"), WithoutInitializing), N);
  Kokkos::View<gid_t*, MemSpace> gidSorted(view_alloc(std::string("gidSorted"), WithoutInitializing),
                                           haveGid ? N : 0);
  Kokkos::View<Real*, MemSpace> wSorted(view_alloc(std::string("wSorted"), WithoutInitializing),
                                        Weighted ? N : 0);
  {
    const bool haveGidL = haveGid;
    Kokkos::parallel_for(
        "tess.reorder", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int p) {
          int o = binned(p);
          slotOf(o) = p;
          posSorted(3 * p + 0) = posFlat(3 * o + 0);
          posSorted(3 * p + 1) = posFlat(3 * o + 1);
          posSorted(3 * p + 2) = posFlat(3 * o + 2);
          if (haveGidL) gidSorted(p) = gid(o);
          if (Weighted) wSorted(p) = weight(o);
        });
  }
  grid.binned = binned;
  grid.slotOf = slotOf;
  grid.posSorted = posSorted;
  grid.gidSorted = gidSorted;
  grid.wSorted = wSorted;
  grid.cellStart = cellStart;

  // --- per-sub-position worklist (computed once) ---
  // The (dx,dy,dz) grid offsets inside the coverage sphere (nearest-corner dist^2 <= sw^2). For
  // the wlS^3 sub-region a seed lands in, they are sorted by nearest-corner dist^2 (wlRmin,
  // absolute) and packed (kWlOffBias). The Voronoi build walks this presorted list and breaks
  // once wlRmin exceeds the security radius (4*rSqMax) — a table lookup, no per-block geometry.
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
  grid.nOff = nOff;
  grid.wlS = wlS;
  // E3: the worklist table is a pure function of (grid dims, sw, cell sizes). If the cache already
  // holds it for this key, reuse the device Views and skip the host sort + two H2D copies entirely.
  if (wlc && wlc->matches(grid.dimx, grid.dimy, grid.dimz, sw, csz[0], csz[1], csz[2])) {
    grid.wlOff = wlc->wlOff;
    grid.wlRmin = wlc->wlRmin;
    return grid;
  }
  Kokkos::View<int*, MemSpace> wlOff(view_alloc(std::string("wlOff"), WithoutInitializing),
                                     (size_t)nSub * nOff);
  Kokkos::View<Real*, MemSpace> wlRmin(view_alloc(std::string("wlRmin"), WithoutInitializing),
                                       (size_t)nSub * nOff);
  {
    auto hOff = Kokkos::create_mirror_view(wlOff);
    auto hRmin = Kokkos::create_mirror_view(wlRmin);
    const Real cszx = csz[0], cszy = csz[1], cszz = csz[2];
    // axis-aligned gap^2 between query sub-interval [a0,a1] and target block [b0,b1].
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
  grid.wlOff = wlOff;
  grid.wlRmin = wlRmin;
  if (wlc) {  // populate the cache for subsequent same-key calls (E3)
    wlc->wlOff = wlOff;
    wlc->wlRmin = wlRmin;
    wlc->nOff = nOff;
    wlc->wlS = wlS;
    wlc->dimx = grid.dimx;
    wlc->dimy = grid.dimy;
    wlc->dimz = grid.dimz;
    wlc->sw = sw;
    wlc->cszx = csz[0];
    wlc->cszy = csz[1];
    wlc->cszz = csz[2];
    wlc->valid = true;
  }
  return grid;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TESS_GRID_HPP
