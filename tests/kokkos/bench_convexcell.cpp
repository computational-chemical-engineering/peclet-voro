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

// Per-thread candidate cap and cell sizing (compact).
static constexpr int CC_MAXP = 64;
static constexpr int CC_MAXT = 112;

struct Result {
  double volSum = 0;
  long faceSum = 0;
  long overflow = 0;
  double secsBest = 0;
};

static Result run_once(const Kokkos::View<real_t*, tpx::MemSpace>& pos, int N, const real_t L[3],
                       int sw, bool timeOnly) {
  using MemSpace = tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  // grid ~1 seed/cell
  const real_t vol = L[0] * L[1] * L[2];
  const real_t spacing = std::cbrt(vol / std::max(1, N));
  int dim[3];
  real_t icsz[3], csz[3];
  for (int k = 0; k < 3; ++k) {
    dim[k] = std::max(1, (int)std::floor(L[k] / spacing));
    csz[k] = L[k] / dim[k];
    icsz[k] = 1.0 / csz[k];
  }
  const int dimx = dim[0], dimy = dim[1], dimz = dim[2], ncell = dimx * dimy * dimz;
  real_t minCsz = std::min({csz[0], csz[1], csz[2]});

  // counting-sort grid
  Kokkos::View<int*, MemSpace> cellOf("cellOf", N), counts("counts", ncell + 1);
  Kokkos::deep_copy(counts, 0);
  const real_t icx = icsz[0], icy = icsz[1], icz = icsz[2];
  Kokkos::parallel_for(
      "bin", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int gx = ((((int)Kokkos::floor(pos(3 * i) * icx)) % dimx) + dimx) % dimx;
        int gy = ((((int)Kokkos::floor(pos(3 * i + 1) * icy)) % dimy) + dimy) % dimy;
        int gz = ((((int)Kokkos::floor(pos(3 * i + 2) * icz)) % dimz) + dimz) % dimz;
        int c = gx + gy * dimx + gz * dimx * dimy;
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
          for (int o = 0; o < nOff; ++o) {
            if (offD(o) > 2.0 * c.maxVertexRsq()) break;  // security radius: stop (sorted)
            int gx = ((cx + offX(o)) % dimx + dimx) % dimx;
            int gy = ((cy + offY(o)) % dimy + dimy) % dimy;
            int gz = ((cz + offZ(o)) % dimz + dimz) % dimz;
            int gc = gx + gy * dimx + gz * dimx * dimy;
            for (int q = cellStart(gc); q < cellStart(gc + 1); ++q) {
              int j = binned(q);
              if (j == i) continue;
              real_t ax = pos(3 * j) - pix, ay = pos(3 * j + 1) - piy, az = pos(3 * j + 2) - piz;
              ax = ax > Lxh ? ax - Lx : (ax < -Lxh ? ax + Lx : ax);
              ay = ay > Lyh ? ay - Ly : (ay < -Lyh ? ay + Ly : ay);
              az = az > Lzh ? az - Lz : (az < -Lzh ? az + Lz : az);
              real_t n[3] = {ax, ay, az};
              real_t off = 0.5 * (ax * ax + ay * ay + az * az);
              c.clip(n, off, j);
              if (c.overflow) break;
            }
            if (c.overflow) break;
          }
          (void)minCsz;
          cellVol(i) = c.overflow ? 0.0 : c.volume();
          cellFaces(i) = c.overflow ? 0 : c.countFaces();
          cellOvf(i) = c.overflow ? 1 : 0;
        });
    Kokkos::fence();
  };

  build();  // warm
  Result r;
  if (!timeOnly) {
    auto hv = Kokkos::create_mirror_view(cellVol);
    auto hf = Kokkos::create_mirror_view(cellFaces);
    auto ho = Kokkos::create_mirror_view(cellOvf);
    Kokkos::deep_copy(hv, cellVol);
    Kokkos::deep_copy(hf, cellFaces);
    Kokkos::deep_copy(ho, cellOvf);
    for (int i = 0; i < N; ++i) {
      r.volSum += hv(i);
      r.faceSum += hf(i);
      r.overflow += ho(i);
    }
  }
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    auto t0 = clk::now();
    build();
    auto t1 = clk::now();
    best = std::min(best, secs(t0, t1));
  }
  r.secsBest = best;
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
      "overflow=%ld\n",
      N, N / r.secsBest / 1e3, volErr, voroErr, (double)r.faceSum / N, r.overflow);
#else
  std::printf("N=%-8d  convexcell %7.1f k/s  | Σvol/box err=%.2e  faces/cell=%.2f  overflow=%ld\n",
              N, N / r.secsBest / 1e3, volErr, (double)r.faceSum / N, r.overflow);
#endif
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
