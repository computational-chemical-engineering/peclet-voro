/**
 * @file bench_bvh_gather.cpp
 * \brief F4 — ArborX BVH (kNN) neighbour query feeding the ConvexCell construction.
 *
 * The gather is the cold-build bottleneck (F1: ~half the time; construction alone is 12-17 M/s).
 * This replaces the uniform-grid gather with an ArborX BVH k-nearest-neighbour query (the SOTA
 * family's neighbour structure; robust to non-uniform/clustered point sets where a grid degrades),
 * then builds each cell from the k neighbours (closest-first + per-candidate cull + G2 geometry).
 * Periodicity via boundary ghosts. Reports BVH-build / kNN-query / construct times and total
 * throughput; validates Σvol == box volume + faces/cell. FP32. Run: ./bench_bvh_gather [N] [K]
 */
#include <ArborX.hpp>
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

using real_t = float;
using Exec = Kokkos::DefaultExecutionSpace;
using Mem = Exec::memory_space;
using Point = ArborX::Point<3, float>;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
static constexpr int CC_MAXP = 64, CC_MAXT = 112, KMAX = 96;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = argc > 1 ? std::atoi(argv[1]) : 1000000;
    const int K = argc > 2 ? std::atoi(argv[2]) : 48;  // neighbours per kNN query
    const float L = 1.0f, spacing = std::cbrt(1.0f / N);
    const float margin = 5.0f * spacing;  // boundary-ghost band (covers the kNN radius)
    std::printf("backend: %s  N=%d K=%d\n", Exec::name(), N, K);

    // host: random points + periodic boundary ghosts (ghost carries its real seed id)
    std::mt19937 rng(99 + N);
    std::uniform_real_distribution<float> U(0, 1);
    std::vector<float> px(N), py(N), pz(N);
    for (int i = 0; i < N; ++i) { px[i] = U(rng); py[i] = U(rng); pz[i] = U(rng); }
    std::vector<Point> hpts;
    std::vector<int> horig;
    hpts.reserve((size_t)N * 5 / 4);
    horig.reserve((size_t)N * 5 / 4);
    for (int i = 0; i < N; ++i) { hpts.push_back({px[i], py[i], pz[i]}); horig.push_back(i); }
    for (int i = 0; i < N; ++i)
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            if (!dx && !dy && !dz) continue;
            const float x = px[i] + dx * L, y = py[i] + dy * L, z = pz[i] + dz * L;
            if (x < -margin || x > L + margin || y < -margin || y > L + margin || z < -margin ||
                z > L + margin)
              continue;
            hpts.push_back({x, y, z});
            horig.push_back(i);
          }
    const int M = (int)hpts.size();
    std::printf("ghosts: %d (%.1f%% of N)\n", M - N, 100.0 * (M - N) / N);

    Kokkos::View<Point*, Mem> pts(Kokkos::view_alloc("pts", Kokkos::WithoutInitializing), M);
    Kokkos::View<int*, Mem> orig(Kokkos::view_alloc("orig", Kokkos::WithoutInitializing), M);
    {
      auto hp = Kokkos::create_mirror_view(pts);
      auto ho = Kokkos::create_mirror_view(orig);
      for (int i = 0; i < M; ++i) { hp(i) = hpts[i]; ho(i) = horig[i]; }
      Kokkos::deep_copy(pts, hp);
      Kokkos::deep_copy(orig, ho);
    }

    Exec space;
    Kokkos::View<real_t*, Mem> cellVol("vol", N);
    Kokkos::View<int*, Mem> cellFaces("faces", N), cellOvf("ovf", N);

    double tB = 0, tQ = 0, tC = 0;
    Kokkos::View<ArborX::PairValueIndex<Point, unsigned>*, Mem> values("values", 0);
    Kokkos::View<int*, Mem> offsets("offsets", 0);

    auto run = [&](bool timing) {
      // --- build BVH over all (real + ghost) points ---
      auto t0 = clk::now();
      ArborX::BoundingVolumeHierarchy bvh{space, ArborX::Experimental::attach_indices(pts)};
      space.fence();
      auto t1 = clk::now();
      // --- kNN query: K+1 nearest of each real seed (self included, skipped below) ---
      Kokkos::View<ArborX::Nearest<Point>*, Mem> preds(
          Kokkos::view_alloc(space, "preds", Kokkos::WithoutInitializing), N);
      Kokkos::parallel_for(
          "preds", Kokkos::RangePolicy<Exec>(space, 0, N),
          KOKKOS_LAMBDA(int i) { preds(i) = ArborX::nearest(pts(i), K + 1); });
      bvh.query(space, preds, values, offsets);
      space.fence();
      auto t2 = clk::now();
      // --- construct each cell from its k neighbours ---
      auto valuesL = values;
      auto offsetsL = offsets;
      auto ptsL = pts;
      auto origL = orig;
      Kokkos::parallel_for(
          "construct", Kokkos::RangePolicy<Exec>(space, 0, N), KOKKOS_LAMBDA(int i) {
            const Point pc = ptsL(i);
            real_t rx[KMAX], ry[KMAX], rz[KMAX], key[KMAX];
            int rid[KMAX], nc = 0;
            for (int g = offsetsL(i); g < offsetsL(i + 1); ++g) {
              const int m = (int)valuesL(g).index;
              if (m == i) continue;  // self
              const Point q = ptsL(m);
              const real_t ax = q[0] - pc[0], ay = q[1] - pc[1], az = q[2] - pc[2];
              if (nc < KMAX) {
                rx[nc] = ax; ry[nc] = ay; rz[nc] = az;
                key[nc] = real_t(0.5) * (ax * ax + ay * ay + az * az);
                rid[nc] = origL(m);
                ++nc;
              }
            }
            // insertion sort by ascending key (closest first)
            for (int a = 1; a < nc; ++a) {
              real_t kk = key[a], xx = rx[a], yy = ry[a], zz = rz[a];
              int ii = rid[a], b = a - 1;
              while (b >= 0 && key[b] > kk) {
                key[b + 1] = key[b]; rx[b + 1] = rx[b]; ry[b + 1] = ry[b]; rz[b + 1] = rz[b];
                rid[b + 1] = rid[b]; --b;
              }
              key[b + 1] = kk; rx[b + 1] = xx; ry[b + 1] = yy; rz[b + 1] = zz; rid[b + 1] = ii;
            }
            vor::device::ConvexCell<real_t, CC_MAXP, CC_MAXT> c;
            c.initBox(L, L, L);
            for (int s = 0; s < nc; ++s) {
              if (key[s] >= real_t(2) * c.maxVertexRsq()) break;  // sorted: no closer left
              const real_t n[3] = {rx[s], ry[s], rz[s]};
              c.clip(n, key[s], rid[s]);
              if (c.overflow) break;
            }
            cellVol(i) = c.overflow ? 0 : c.volume();  // G1, timed (matches the grid bench)
            if (!timing) {
              cellFaces(i) = c.overflow ? 0 : c.countFaces();
              cellOvf(i) = c.overflow ? 1 : 0;
              real_t s = 0;  // touch G2 so it isn't elided (validation only)
              for (int k = 0; k < c.np; ++k) {
                if (c.pnbr[k] < 0) continue;
                real_t a[3], d[3], cv[3];
                if (c.facetGeometry(k, a, d, cv)) s += d[0];
              }
              cellVol(i) += real_t(0) * s;
            }
          });
      space.fence();
      auto t3 = clk::now();
      tB = secs(t0, t1); tQ = secs(t1, t2); tC = secs(t2, t3);
    };

    run(false);  // warm + validate
    double volSum = 0; long faceSum = 0, ovf = 0;
    {
      auto hv = Kokkos::create_mirror_view(cellVol);
      auto hf = Kokkos::create_mirror_view(cellFaces);
      auto ho = Kokkos::create_mirror_view(cellOvf);
      Kokkos::deep_copy(hv, cellVol); Kokkos::deep_copy(hf, cellFaces); Kokkos::deep_copy(ho, cellOvf);
      for (int i = 0; i < N; ++i) { volSum += hv(i); faceSum += hf(i); ovf += ho(i); }
    }
    double bestTot = 1e30, bb = 0, bq = 0, bc = 0;
    for (int rep = 0; rep < 3; ++rep) {
      run(true);
      if (tB + tQ + tC < bestTot) { bestTot = tB + tQ + tC; bb = tB; bq = tQ; bc = tC; }
    }
    std::printf("Σvol/box err=%.2e  faces/cell=%.2f  overflow=%ld\n",
                std::fabs(volSum - 1.0), (double)faceSum / N, ovf);
    std::printf("BVH-build %.1f k/s   kNN-query %.1f k/s   construct %.1f k/s\n", N / bb / 1e3,
                N / bq / 1e3, N / bc / 1e3);
    std::printf("TOTAL (build+query+construct) %.1f kcells/s   [grid full build ~5400]\n",
                N / bestTot / 1e3);
  }
  Kokkos::finalize();
  return 0;
}
