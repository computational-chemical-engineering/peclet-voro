/**
 * @file bench_incremental.cpp
 * \brief Part II / Phase 1 — incremental geometry re-eval over resident topology vs full rebuild.
 *
 * The moving-point workload (DEM/fluid): each step the seeds move a little. Most cells keep their
 * topology (Phase 0: 73-94% stable at realistic displacement), so instead of re-gathering neighbours
 * and re-clipping, we REUSE each cell's resident topology and only recompute plane equations +
 * vertices + geometry (`ConvexCell::reevalGeometry`). This benchmark:
 *   - builds N cells once (storing each cell + its global-id topology),
 *   - displaces all seeds by `delta * spacing` (Gaussian),
 *   - times re-eval (no gather/clip) vs full rebuild (clip the K updated neighbours),
 *   - reports the speedup and the fraction of cells whose re-eval volume matches the rebuild
 *     (= topology-stable; the rest need Phase-1.5 repair),
 *   - sweeps `delta`.
 * FP32. Run: ./bench_incremental [N] [delta1 delta2 ...]
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

using real_t = float;
using Exec = tpx::ExecSpace;
using Mem = tpx::MemSpace;
using Cell = vor::device::ConvexCell<real_t, 64, 112>;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
static constexpr int KCAND = 64;

/// Build cell i by clipping its K stored neighbours at the CURRENT positions (security break).
/// Stores the GLOBAL neighbour id as the plane's nbr, so the topology can be re-evaluated later.
KOKKOS_INLINE_FUNCTION void buildCell(int i, Cell& c, const real_t* pos, const int* candId,
                                      const int* ncand, real_t L) {
  const real_t sx = pos[3 * i], sy = pos[3 * i + 1], sz = pos[3 * i + 2];
  const real_t Lh = 0.5f * L;
  c.initBox(L, L, L);
  real_t secR2 = 2.0f * c.maxVertexRsq();
  const int k = ncand[i];
  for (int t = 0; t < k; ++t) {
    const int g = candId[(size_t)i * KCAND + t];
    real_t rx = pos[3 * g] - sx, ry = pos[3 * g + 1] - sy, rz = pos[3 * g + 2] - sz;
    rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
    ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
    rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
    const real_t off = 0.5f * (rx * rx + ry * ry + rz * rz);
    if (off >= secR2) break;
    const real_t n[3] = {rx, ry, rz};
    if (c.clip(n, off, g)) secR2 = 2.0f * c.maxVertexRsq();
    if (c.overflow) break;
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = argc > 1 ? std::atoi(argv[1]) : 1000000;
    std::vector<real_t> deltas;
    for (int i = 2; i < argc; ++i) deltas.push_back(std::atof(argv[i]));
    if (deltas.empty()) deltas = {0.02f, 0.05f, 0.1f, 0.2f, 0.4f};
    const real_t L = 1.0f, spacing = std::cbrt(1.0f / N);
    std::printf("Part II Phase 1: re-eval vs rebuild.  N=%d  cell=%zu B  (FP32)\n", N, sizeof(Cell));

    // --- seeds + closest-K neighbour ids per cell (grid brute force; one-time) ---
    std::mt19937 rng(123 + N);
    std::uniform_real_distribution<real_t> U(0, 1);
    std::vector<real_t> pos0(3 * N);
    for (auto& x : pos0) x = U(rng);
    const int dim = std::max(1, (int)std::floor(1.0 / spacing));
    const real_t csz = 1.0 / dim;
    std::vector<std::vector<int>> grid(dim * dim * dim);
    auto gid = [&](int x, int y, int z) {
      return (x + dim) % dim + dim * (((y + dim) % dim) + dim * ((z + dim) % dim));
    };
    for (int i = 0; i < N; ++i) {
      int gx = (int)(pos0[3 * i] / csz), gy = (int)(pos0[3 * i + 1] / csz), gz = (int)(pos0[3 * i + 2] / csz);
      grid[gid(gx, gy, gz)].push_back(i);
    }
    Kokkos::View<int*, Mem> candId("candId", (size_t)N * KCAND);  // GLOBAL neighbour id per slot
    Kokkos::View<int*, Mem> ncand("ncand", N);
    {
      auto hId = Kokkos::create_mirror_view(candId);
      auto hn = Kokkos::create_mirror_view(ncand);
      const int sw = 3;
      std::vector<std::array<real_t, 2>> tmp;  // {dist2, (float)j}
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos0[3 * i] / csz), gy = (int)(pos0[3 * i + 1] / csz), gz = (int)(pos0[3 * i + 2] / csz);
        tmp.clear();
        for (int dz = -sw; dz <= sw; ++dz)
          for (int dy = -sw; dy <= sw; ++dy)
            for (int dx = -sw; dx <= sw; ++dx)
              for (int j : grid[gid(gx + dx, gy + dy, gz + dz)]) {
                if (j == i) continue;
                real_t rx = pos0[3 * j] - pos0[3 * i], ry = pos0[3 * j + 1] - pos0[3 * i + 1],
                       rz = pos0[3 * j + 2] - pos0[3 * i + 2];
                rx -= std::round(rx); ry -= std::round(ry); rz -= std::round(rz);
                tmp.push_back({rx * rx + ry * ry + rz * rz, (real_t)j});
              }
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        int k = std::min((int)tmp.size(), KCAND);
        hn(i) = k;
        for (int t = 0; t < k; ++t) hId((size_t)i * KCAND + t) = (int)tmp[t][1];
      }
      Kokkos::deep_copy(candId, hId);
      Kokkos::deep_copy(ncand, hn);
    }

    Kokkos::View<real_t*, Mem> pos("pos", 3 * N);  // current global positions
    Kokkos::View<Cell*, Mem> cells0("cells0", N);  // resident topology (built once)
    Kokkos::View<real_t*, Mem> volRe("volRe", N), volRb("volRb", N);
    const real_t* posRaw = pos.data();
    const int* candRaw = candId.data();
    const int* ncRaw = ncand.data();

    // step 0: build at the initial positions and store the resident topology
    Kokkos::deep_copy(pos, Kokkos::View<real_t*, Kokkos::HostSpace>(pos0.data(), 3 * N));
    auto cells0L = cells0;
    Kokkos::parallel_for("build0", Kokkos::RangePolicy<Exec>(0, N),
                         KOKKOS_LAMBDA(int i) { buildCell(i, cells0L(i), posRaw, candRaw, ncRaw, L); });
    Kokkos::fence();

    std::printf("  delta   re-eval     rebuild   speedup   topo-stable   Σvol(re) err\n");
    auto volReL = volRe;
    auto volRbL = volRb;
    for (real_t d : deltas) {
      // displace all seeds by d*spacing (Gaussian), wrap into [0,1)
      {
        auto h = Kokkos::create_mirror_view(pos);
        std::mt19937 r2(999);
        std::normal_distribution<real_t> Ng(0, d * spacing);
        for (int i = 0; i < 3 * N; ++i) {
          real_t v = pos0[i] + Ng(r2);
          v -= std::floor(v);
          h(i) = v;
        }
        Kokkos::deep_copy(pos, h);
      }

      auto reeval = [&] {
        Kokkos::parallel_for("reeval", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
          Cell c = cells0L(i);  // copy resident topology
          c.reevalGeometry(posRaw[3 * i], posRaw[3 * i + 1], posRaw[3 * i + 2], posRaw, L);
          volReL(i) = c.volume();
        });
        Kokkos::fence();
      };
      auto rebuild = [&] {
        Kokkos::parallel_for("rebuild", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
          Cell c;
          buildCell(i, c, posRaw, candRaw, ncRaw, L);
          volRbL(i) = c.overflow ? 0.0f : c.volume();
        });
        Kokkos::fence();
      };
      reeval(); rebuild();  // warm
      double tRe = 1e30, tRb = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        auto t0 = clk::now(); reeval(); auto t1 = clk::now(); tRe = std::min(tRe, secs(t0, t1));
        auto t2 = clk::now(); rebuild(); auto t3 = clk::now(); tRb = std::min(tRb, secs(t2, t3));
      }
      long stable = 0; double sre = 0;
      {
        auto hRe = Kokkos::create_mirror_view(volRe);
        auto hRb = Kokkos::create_mirror_view(volRb);
        Kokkos::deep_copy(hRe, volRe); Kokkos::deep_copy(hRb, volRb);
        for (int i = 0; i < N; ++i) {
          sre += hRe(i);
          const double rel = hRb(i) > 0 ? std::fabs(hRe(i) - hRb(i)) / hRb(i) : 1.0;
          if (rel < 1e-3) ++stable;
        }
      }
      std::printf("  %4.2f  %8.1f   %8.1f   %6.2fx   %8.1f%%    %.2e\n", d, N / tRe / 1e3,
                  N / tRb / 1e3, tRb / tRe, 100.0 * stable / N, std::fabs(sre - 1.0));
    }
  }
  Kokkos::finalize();
  return 0;
}
