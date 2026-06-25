/**
 * @file bench_convexcell.cpp
 * \brief Prototype benchmark for the compact ConvexCell (option-A) path.
 *
 * Builds the full tessellation one cell per thread on the ConvexCell representation
 * (compact dual-triangle cell, register-resident), over whatever Kokkos backend the
 * binary was compiled for. Validates against the space-filling identity (Σ cell volume ==
 * box volume) and, if voro++ is available, against per-cell voro++ volumes. Reports
 * throughput across N. Run manually:  ./bench_convexcell [N1 N2 ...]
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"

#ifdef VORFLOW_HAVE_VOROPP
#include "voro++.hh"
#endif

#ifdef CC_FLOAT
using real_t = float;
#else
using real_t = double;
#endif
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// 21-bit Morton (Z-order) encode of a grid cell (gx,gy,gz). Used as the cell id so that the
// counting-sort orders points along a space-filling curve: a warp of consecutive seeds is a
// COMPACT 3D blob, so their 3x3x3 neighbour blocks overlap and the gather reuses L1/L2 — unlike
// a row-major id (gx + gy*dimx + gz*dimx*dimy), where y/z neighbours are dimx/dimx*dimy apart.
KOKKOS_INLINE_FUNCTION unsigned long mortonPart1By2(unsigned long x) {
  x &= 0x1ffffful;
  x = (x | (x << 32)) & 0x1f00000000fffful;
  x = (x | (x << 16)) & 0x1f0000ff0000fful;
  x = (x | (x << 8)) & 0x100f00f00f00f00ful;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ul;
  x = (x | (x << 2)) & 0x1249249249249249ul;
  return x;
}
KOKKOS_INLINE_FUNCTION unsigned long mortonEncode(int x, int y, int z) {
  return mortonPart1By2((unsigned long)x) | (mortonPart1By2((unsigned long)y) << 1) |
         (mortonPart1By2((unsigned long)z) << 2);
}

// Per-thread candidate cap and cell sizing (compact). Smaller caps shrink the per-thread cell
// state (pn/pd/pnbr[MAXP] + t*/v*/alive[MAXT]); below ~MAXT=48 it stops spilling to local memory
// and throughput jumps. Override at compile time to sweep.
#ifndef CC_MAXP
#define CC_MAXP 64
#endif
#ifndef CC_MAXT
#define CC_MAXT 112
#endif

struct Result {
  double volSum = 0;
  long faceSum = 0;
  long overflow = 0;
  long clipSum = 0;
  long clipMax = 0;
  double secsBest = 0;
  // spatially-sorted variant (points reordered into grid order, threads = sorted slots)
  double volSumS = 0;
  long faceSumS = 0;
  long overflowS = 0;
  double secsBestS = 0;
};

static Result run_once(const Kokkos::View<real_t*, tpx::MemSpace>& pos, int N, const real_t L[3],
                       int sw, bool timeOnly) {
  using MemSpace = tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  // grid ~CC_DENS seeds/cell (default 1). Coarser cells = fewer offset iterations to walk
  // (less per-offset morton/modulo/cellStart overhead), at the cost of examining more candidates
  // per cell. Sweep to find the throughput optimum for the fused gather.
  const real_t vol = L[0] * L[1] * L[2];
  const real_t dens = std::getenv("CC_DENS") ? std::atof(std::getenv("CC_DENS")) : 1.0;
  const real_t spacing = std::cbrt(dens * vol / std::max(1, N));
  int dim[3];
  real_t icsz[3], csz[3];
  for (int k = 0; k < 3; ++k) {
    dim[k] = std::max(1, (int)std::floor(L[k] / spacing));
    csz[k] = L[k] / dim[k];
    icsz[k] = 1.0 / csz[k];
  }
  const int dimx = dim[0], dimy = dim[1], dimz = dim[2];
  // Morton cell id => cellStart sized to the (power-of-two)^3 Z-order range, not dimx*dimy*dimz.
  int nbits = 1;
  while ((1 << nbits) < std::max({dimx, dimy, dimz})) ++nbits;
  const long ncell = 1L << (3 * nbits);
  real_t minCsz = std::min({csz[0], csz[1], csz[2]});

  // counting-sort grid (cell id = Morton(gx,gy,gz))
  Kokkos::View<int*, MemSpace> cellOf("cellOf", N), counts("counts", ncell + 1);
  Kokkos::deep_copy(counts, 0);
  const real_t icx = icsz[0], icy = icsz[1], icz = icsz[2];
  Kokkos::parallel_for(
      "bin", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int gx = ((((int)Kokkos::floor(pos(3 * i) * icx)) % dimx) + dimx) % dimx;
        int gy = ((((int)Kokkos::floor(pos(3 * i + 1) * icy)) % dimy) + dimy) % dimy;
        int gz = ((((int)Kokkos::floor(pos(3 * i + 2) * icz)) % dimz) + dimz) % dimz;
        int c = (int)mortonEncode(gx, gy, gz);
        cellOf(i) = c;
        Kokkos::atomic_inc(&counts(c));
      });
  Kokkos::View<int*, MemSpace> cellStart("cellStart", ncell + 1);
  Kokkos::parallel_scan(
      "scan", Kokkos::RangePolicy<Exec>(0, ncell + 1), KOKKOS_LAMBDA(const int c, int& a, bool f) {
        int v = (c < ncell) ? counts(c) : 0;
        if (f) cellStart(c) = a;
        a += v;
      });
  Kokkos::View<int*, MemSpace> cursor("cursor", ncell), binned("binned", N);
  Kokkos::parallel_for(
      "cur", Kokkos::RangePolicy<Exec>(0, ncell), KOKKOS_LAMBDA(const int c) { cursor(c) = cellStart(c); });
  Kokkos::parallel_for(
      "scat", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int s = Kokkos::atomic_fetch_add(&cursor(cellOf(i)), 1);
        binned(s) = i;
      });

  const real_t Lx = L[0], Ly = L[1], Lz = L[2];
  Kokkos::View<real_t*, MemSpace> cellVol("cellVol", N);
  Kokkos::View<int*, MemSpace> cellFaces("cellFaces", N), cellOvf("cellOvf", N);
  Kokkos::View<long*, MemSpace> cellClips("cellClips", N);  // profiling: clip() calls per cell
  const bool doVol = std::getenv("CC_NOVOL") == nullptr;    // CC_NOVOL=1 -> skip volume() (timing)

  // Grid offsets in the (2sw+1)^3 block, sorted by nearest-corner distance² (×minCsz²).
  // Half-space intersection is order-independent, so we clip on the fly with NO candidate
  // buffer — the cell is the only per-thread state. The offsets are visited closest-first
  // and we stop once the nearest possible seed is beyond the security radius (2·maxVrsq).
  std::vector<int> oX, oY, oZ;
  std::vector<real_t> oD;
  for (int dz = -sw; dz <= sw; ++dz)
    for (int dy = -sw; dy <= sw; ++dy)
      for (int dx = -sw; dx <= sw; ++dx) {
        int ex = dx < 0 ? -dx - 1 : (dx > 0 ? dx - 1 : 0);
        int ey = dy < 0 ? -dy - 1 : (dy > 0 ? dy - 1 : 0);
        int ez = dz < 0 ? -dz - 1 : (dz > 0 ? dz - 1 : 0);
        oX.push_back(dx);
        oY.push_back(dy);
        oZ.push_back(dz);
        oD.push_back((real_t)(ex * ex + ey * ey + ez * ez) * minCsz * minCsz);
      }
  const int nOff = (int)oX.size();
  // sort offsets by ascending nearest-corner distance²
  std::vector<int> ord(nOff);
  for (int i = 0; i < nOff; ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return oD[a] < oD[b]; });
  Kokkos::View<int*, MemSpace> offX("offX", nOff), offY("offY", nOff), offZ("offZ", nOff);
  Kokkos::View<real_t*, MemSpace> offD("offD", nOff);
  {
    auto hx = Kokkos::create_mirror_view(offX), hy = Kokkos::create_mirror_view(offY),
         hz = Kokkos::create_mirror_view(offZ);
    auto hd = Kokkos::create_mirror_view(offD);
    for (int i = 0; i < nOff; ++i) {
      hx(i) = oX[ord[i]]; hy(i) = oY[ord[i]]; hz(i) = oZ[ord[i]]; hd(i) = oD[ord[i]];
    }
    Kokkos::deep_copy(offX, hx); Kokkos::deep_copy(offY, hy); Kokkos::deep_copy(offZ, hz);
    Kokkos::deep_copy(offD, hd);
  }

  auto build = [&]() {
    Kokkos::parallel_for(
        "cc.build", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          const real_t pix = pos(3 * i), piy = pos(3 * i + 1), piz = pos(3 * i + 2);
          int cx = ((((int)Kokkos::floor(pix * icx)) % dimx) + dimx) % dimx;
          int cy = ((((int)Kokkos::floor(piy * icy)) % dimy) + dimy) % dimy;
          int cz = ((((int)Kokkos::floor(piz * icz)) % dimz) + dimz) % dimz;
          const real_t Lxh = 0.5 * Lx, Lyh = 0.5 * Ly, Lzh = 0.5 * Lz;
          vor::device::ConvexCell<real_t, CC_MAXP, CC_MAXT> c;
          c.initBox(Lx, Ly, Lz);
          long nclip = 0;
          for (int o = 0; o < nOff; ++o) {
            const real_t secR2 = 2.0 * c.maxVertexRsq();  // security radius² (cell shrinks)
            if (offD(o) > secR2) break;                   // no closer grid cell can cut (sorted)
            int gx = ((cx + offX(o)) % dimx + dimx) % dimx;
            int gy = ((cy + offY(o)) % dimy + dimy) % dimy;
            int gz = ((cz + offZ(o)) % dimz + dimz) % dimz;
            int gc = (int)mortonEncode(gx, gy, gz);
            for (int q = cellStart(gc); q < cellStart(gc + 1); ++q) {
              int j = binned(q);
              if (j == i) continue;
              ++nclip;  // EXAMINED: distance computed for this candidate (the gather's real work)
              real_t ax = pos(3 * j) - pix, ay = pos(3 * j + 1) - piy, az = pos(3 * j + 2) - piz;
              ax = ax > Lxh ? ax - Lx : (ax < -Lxh ? ax + Lx : ax);
              ay = ay > Lyh ? ay - Ly : (ay < -Lyh ? ay + Ly : ay);
              az = az > Lzh ? az - Lz : (az < -Lzh ? az + Lz : az);
              real_t off = 0.5 * (ax * ax + ay * ay + az * az);
              // Per-candidate security cull: a grid cell within the corner bound can still
              // hold seeds beyond the radius (the bound is the nearest corner, not the seed).
              // off >= 2·maxVrsq => the bisector is past every vertex => cannot cut. Skipping
              // avoids the O(#tri) clip scan for the ~85% of gathered candidates that are no-ops.
              if (off >= secR2) continue;
              real_t n[3] = {ax, ay, az};
              c.clip(n, off, j);
              if (c.overflow) break;
            }
            if (c.overflow) break;
          }
          (void)minCsz;
          cellVol(i) = (c.overflow || !doVol) ? 0.0 : c.volumePerVertex();
          cellFaces(i) = c.overflow ? 0 : c.countFaces();
          cellOvf(i) = c.overflow ? 1 : 0;
          cellClips(i) = nclip;  // candidates EXAMINED per cell (gather volume)
        });
    Kokkos::fence();
  };

  // --- spatially-sorted variant: reorder points into grid (binned) order ONCE, then assign
  // thread q -> sorted slot q. Now a warp of 32 threads = 32 spatially-adjacent seeds sharing
  // the same neighbour cells (L1/L2 reuse) and the neighbour position reads are contiguous —
  // no binned() indirection into an unsorted array. This is the SOTA gather pattern.
  Kokkos::View<real_t*, MemSpace> posS(
      Kokkos::view_alloc(std::string("posS"), Kokkos::WithoutInitializing), (size_t)N * 3);
  Kokkos::parallel_for(
      "reorder", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int q) {
        const int i = binned(q);
        posS(3 * q) = pos(3 * i);
        posS(3 * q + 1) = pos(3 * i + 1);
        posS(3 * q + 2) = pos(3 * i + 2);
      });
  Kokkos::View<real_t*, MemSpace> cellVolS("cellVolS", N);
  Kokkos::View<int*, MemSpace> cellFacesS("cellFacesS", N), cellOvfS("cellOvfS", N);
  auto build_sorted = [&]() {
    Kokkos::parallel_for(
        "cc.build_sorted", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int q) {
          const real_t pix = posS(3 * q), piy = posS(3 * q + 1), piz = posS(3 * q + 2);
          int cx = ((((int)Kokkos::floor(pix * icx)) % dimx) + dimx) % dimx;
          int cy = ((((int)Kokkos::floor(piy * icy)) % dimy) + dimy) % dimy;
          int cz = ((((int)Kokkos::floor(piz * icz)) % dimz) + dimz) % dimz;
          const real_t Lxh = 0.5 * Lx, Lyh = 0.5 * Ly, Lzh = 0.5 * Lz;
          vor::device::ConvexCell<real_t, CC_MAXP, CC_MAXT> c;
          c.initBox(Lx, Ly, Lz);
          real_t secR2 = 2.0 * c.maxVertexRsq();  // cached; only shrinks when a clip cuts
          for (int o = 0; o < nOff; ++o) {
            if (offD(o) > secR2) break;
            // branchless periodic wrap (offX in [-sw,sw], cx in [0,dim) => one add/sub suffices),
            // replacing 2 integer-modulo per axis — GPU idiv is ~20+ cycles and this runs ~100x/cell
            int gx = cx + offX(o); gx = gx < 0 ? gx + dimx : (gx >= dimx ? gx - dimx : gx);
            int gy = cy + offY(o); gy = gy < 0 ? gy + dimy : (gy >= dimy ? gy - dimy : gy);
            int gz = cz + offZ(o); gz = gz < 0 ? gz + dimz : (gz >= dimz ? gz - dimz : gz);
            int gc = (int)mortonEncode(gx, gy, gz);
            for (int q2 = cellStart(gc); q2 < cellStart(gc + 1); ++q2) {
              if (q2 == q) continue;
              real_t ax = posS(3 * q2) - pix, ay = posS(3 * q2 + 1) - piy, az = posS(3 * q2 + 2) - piz;
              ax = ax > Lxh ? ax - Lx : (ax < -Lxh ? ax + Lx : ax);
              ay = ay > Lyh ? ay - Ly : (ay < -Lyh ? ay + Ly : ay);
              az = az > Lzh ? az - Lz : (az < -Lzh ? az + Lz : az);
              real_t off = 0.5 * (ax * ax + ay * ay + az * az);
              if (off >= secR2) continue;
              real_t n[3] = {ax, ay, az};
              if (c.clip(n, off, q2)) secR2 = 2.0 * c.maxVertexRsq();  // recompute only on a cut
              if (c.overflow) break;
            }
            if (c.overflow) break;
          }
          cellVolS(q) = (c.overflow || !doVol) ? 0.0 : c.volumePerVertex();
          cellFacesS(q) = c.overflow ? 0 : c.countFaces();
          cellOvfS(q) = c.overflow ? 1 : 0;
        });
    Kokkos::fence();
  };

  build();         // warm
  build_sorted();  // warm
  Result r;
  if (!timeOnly) {
    auto hv = Kokkos::create_mirror_view(cellVol);
    auto hf = Kokkos::create_mirror_view(cellFaces);
    auto ho = Kokkos::create_mirror_view(cellOvf);
    auto hc = Kokkos::create_mirror_view(cellClips);
    Kokkos::deep_copy(hv, cellVol);
    Kokkos::deep_copy(hf, cellFaces);
    Kokkos::deep_copy(ho, cellOvf);
    Kokkos::deep_copy(hc, cellClips);
    for (int i = 0; i < N; ++i) {
      r.volSum += hv(i);
      r.faceSum += hf(i);
      r.overflow += ho(i);
      r.clipSum += hc(i);
      if (hc(i) > r.clipMax) r.clipMax = hc(i);
    }
    auto hvs = Kokkos::create_mirror_view(cellVolS);
    auto hfs = Kokkos::create_mirror_view(cellFacesS);
    auto hos = Kokkos::create_mirror_view(cellOvfS);
    Kokkos::deep_copy(hvs, cellVolS);
    Kokkos::deep_copy(hfs, cellFacesS);
    Kokkos::deep_copy(hos, cellOvfS);
    for (int i = 0; i < N; ++i) {
      r.volSumS += hvs(i);
      r.faceSumS += hfs(i);
      r.overflowS += hos(i);
    }
  }
  double best = 1e30, bestS = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    auto t0 = clk::now();
    build();
    auto t1 = clk::now();
    best = std::min(best, secs(t0, t1));
    auto t2 = clk::now();
    build_sorted();
    auto t3 = clk::now();
    bestS = std::min(bestS, secs(t2, t3));
  }
  r.secsBest = best;
  r.secsBestS = bestS;
  return r;
}

static void run(int N) {
  const real_t L[3] = {1.0, 1.0, 1.0};
  std::mt19937 rng(12345 + N);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> h(3 * N);
  for (int i = 0; i < 3 * N; ++i) h[i] = U(rng);
  Kokkos::View<real_t*, tpx::MemSpace> pos(
      Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)N * 3);
  {
    auto m = Kokkos::create_mirror_view(pos);
    for (int i = 0; i < 3 * N; ++i) m(i) = h[i];
    Kokkos::deep_copy(pos, m);
  }
  const int sw = std::getenv("CC_SW") ? std::atoi(std::getenv("CC_SW")) : 3;
  Result r = run_once(pos, N, L, sw, false);
  const double boxVol = L[0] * L[1] * L[2];
  const double volErr = std::fabs(r.volSum - boxVol) / boxVol;

#ifdef VORFLOW_HAVE_VOROPP
  // per-cell voro++ comparison on a small subsample (build all, compare a few)
  double voroVol = 0;
  {
    const int nbl = std::max(1, (int)std::cbrt(N / 8.0));
    voro::container_periodic con(L[0], 0, L[1], 0, 0, L[2], nbl, nbl, nbl, 8);
    for (int i = 0; i < N; ++i) con.put(i, h[3 * i], h[3 * i + 1], h[3 * i + 2]);
    voro::voronoicell c(con);
    voro::c_loop_all_periodic vl(con);
    if (vl.start()) do {
        if (con.compute_cell(c, vl)) voroVol += c.volume();
      } while (vl.inc());
  }
  const double voroErr = std::fabs(r.volSum - voroVol) / voroVol;
  std::printf(
      "N=%-8d  convexcell %7.1f k/s  | Σvol/box err=%.2e  Σvol/voro err=%.2e  faces/cell=%.2f  "
      "overflow=%ld examined/cell mean=%.1f max=%ld\n",
      N, N / r.secsBest / 1e3, volErr, voroErr, (double)r.faceSum / N, r.overflow, (double)r.clipSum / N, r.clipMax);
#else
  std::printf("N=%-8d  convexcell %7.1f k/s  | Σvol/box err=%.2e  faces/cell=%.2f  overflow=%ld examined/cell mean=%.1f max=%ld\n",
              N, N / r.secsBest / 1e3, volErr, (double)r.faceSum / N, r.overflow, (double)r.clipSum / N, r.clipMax);
#endif
  const double volErrS = std::fabs(r.volSumS - boxVol) / boxVol;
  std::printf("          spatial-sort %7.1f k/s (%.2fx) | Σvol/box err=%.2e  faces/cell=%.2f  overflow=%ld\n",
              N / r.secsBestS / 1e3, r.secsBest / r.secsBestS, volErrS, (double)r.faceSumS / N, r.overflowS);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("backend: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::vector<int> Ns;
    for (int i = 1; i < argc; ++i) Ns.push_back(std::atoi(argv[i]));
    if (Ns.empty()) Ns = {10000, 100000, 1000000};
    for (int N : Ns) run(N);
  }
  Kokkos::finalize();
  return 0;
}
