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
static constexpr int CMAXP = 64, CMAXT = 112;
using Cell = vor::device::ConvexCell<real_t, CMAXP, CMAXT>;
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
    Kokkos::View<real_t*, Mem> volRe("volRe", N), volRb("volRb", N), volRc("volRc", N);
    Kokkos::View<int*, Mem> needfix("needfix", N);          // Phase 1.5 reclip flag
    Kokkos::View<int*, Mem> fixList("fixList", N);          // compacted flagged indices
    Kokkos::View<int, Mem> fixCount("fixCount");            // # flagged
    const real_t* posRaw = pos.data();
    const int* candRaw = candId.data();
    const int* ncRaw = ncand.data();

    // step 0: build at the initial positions and store the resident topology
    Kokkos::deep_copy(pos, Kokkos::View<real_t*, Kokkos::HostSpace>(pos0.data(), 3 * N));
    auto cells0L = cells0;
    Kokkos::parallel_for("build0", Kokkos::RangePolicy<Exec>(0, N),
                         KOKKOS_LAMBDA(int i) { buildCell(i, cells0L(i), posRaw, candRaw, ncRaw, L); });
    Kokkos::fence();

    // --- compact resident topology: pnbr + packed triangles only (no planes/vertices/box), ~200 B/cell
    Kokkos::View<int*, Mem> topoNp("topoNp", N), topoNt("topoNt", N);
    Kokkos::View<int*, Mem> topoPnbr("topoPnbr", (size_t)N * CMAXP);
    Kokkos::View<unsigned*, Mem> topoTri("topoTri", (size_t)N * CMAXT);
    auto topoNpL = topoNp; auto topoNtL = topoNt; auto topoPnbrL = topoPnbr; auto topoTriL = topoTri;
    Kokkos::parallel_for("extract", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
      const Cell& c = cells0L(i);
      topoNpL(i) = c.np; topoNtL(i) = c.nt;
      for (int k = 0; k < c.np; ++k) topoPnbrL((size_t)i * CMAXP + k) = c.pnbr[k];
      for (int t = 0; t < c.nt; ++t)
        topoTriL((size_t)i * CMAXT + t) = (unsigned)c.t0[t] | ((unsigned)c.t1[t] << 8) |
                                          ((unsigned)c.t2[t] << 16) | ((c.alive[t] ? 1u : 0u) << 24);
    });
    Kokkos::fence();
    std::printf("  storage/cell: full=%zu B   compact=%d B\n", sizeof(Cell),
                (int)(8 + CMAXP * 4 + CMAXT * 4));

    std::printf("  delta  reeval-full reeval-cmp  rebuild  cmp/rebuild  topo-stable  Σvol err\n");
    auto volReL = volRe;
    auto volRbL = volRb;
    auto volRcL = volRc;
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
      // re-eval from the COMPACT topology: rebuild a local cell from pnbr + packed triangles, reeval
      auto reevalCompact = [&] {
        Kokkos::parallel_for("reevalCmp", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
          Cell c;
          c.initBoxPlanes(L, L, L);
          const int np = topoNpL(i), nt = topoNtL(i);
          c.np = np; c.nt = nt; c.overflow = false;
          for (int k = 6; k < np; ++k) c.pnbr[k] = topoPnbrL((size_t)i * CMAXP + k);
          for (int t = 0; t < nt; ++t) {
            const unsigned w = topoTriL((size_t)i * CMAXT + t);
            c.t0[t] = w & 0xff; c.t1[t] = (w >> 8) & 0xff; c.t2[t] = (w >> 16) & 0xff;
            c.alive[t] = (w >> 24) & 1u;
          }
          c.reevalGeometry(posRaw[3 * i], posRaw[3 * i + 1], posRaw[3 * i + 2], posRaw, L);
          volRcL(i) = c.volume();
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
      reeval(); reevalCompact(); rebuild();  // warm
      double tRe = 1e30, tRc = 1e30, tRb = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        auto t0 = clk::now(); reeval(); auto t1 = clk::now(); tRe = std::min(tRe, secs(t0, t1));
        auto ta = clk::now(); reevalCompact(); auto tb = clk::now(); tRc = std::min(tRc, secs(ta, tb));
        auto t2 = clk::now(); rebuild(); auto t3 = clk::now(); tRb = std::min(tRb, secs(t2, t3));
      }
      long stable = 0; double sre = 0, scmp_match = 0;
      {
        auto hRe = Kokkos::create_mirror_view(volRe);
        auto hRb = Kokkos::create_mirror_view(volRb);
        auto hRc = Kokkos::create_mirror_view(volRc);
        Kokkos::deep_copy(hRe, volRe); Kokkos::deep_copy(hRb, volRb); Kokkos::deep_copy(hRc, volRc);
        for (int i = 0; i < N; ++i) {
          sre += hRc(i);
          const double rel = hRb(i) > 0 ? std::fabs(hRc(i) - hRb(i)) / hRb(i) : 1.0;
          if (rel < 1e-3) ++stable;
          scmp_match += std::fabs((double)hRc(i) - (double)hRe(i));  // compact vs full re-eval agree?
        }
      }
      std::printf("  %4.2f  %9.1f  %9.1f  %8.1f  %7.2fx   %8.1f%%   %.2e\n", d, N / tRe / 1e3,
                  N / tRc / 1e3, N / tRb / 1e3, tRb / tRc, 100.0 * stable / N, std::fabs(sre - 1.0));
      if (scmp_match > 1e-3) std::printf("    [warn] compact vs full re-eval differ: %.2e\n", scmp_match);

      // ---- Phase 1.5: re-eval + flag, stream-compact, re-clip only flagged ----
      auto needfixL = needfix; auto fixListL = fixList; auto fixCountL = fixCount;
      auto reevalFlag = [&] {
        Kokkos::deep_copy(fixCount, 0);
        Kokkos::parallel_for("reevalFlag", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
          Cell c;
          c.initBoxPlanes(L, L, L);
          const int np = topoNpL(i), nt = topoNtL(i);
          c.np = np; c.nt = nt; c.overflow = false;
          for (int k = 6; k < np; ++k) c.pnbr[k] = topoPnbrL((size_t)i * CMAXP + k);
          for (int t = 0; t < nt; ++t) {
            const unsigned w = topoTriL((size_t)i * CMAXT + t);
            c.t0[t] = w & 0xff; c.t1[t] = (w >> 8) & 0xff; c.t2[t] = (w >> 16) & 0xff;
            c.alive[t] = (w >> 24) & 1u;
          }
          const real_t sx = posRaw[3 * i], sy = posRaw[3 * i + 1], sz = posRaw[3 * i + 2];
          c.reevalGeometry(sx, sy, sz, posRaw, L);
          volRcL(i) = c.volume();
          // needs-reclip test: face beyond 2Rmax (lost) OR non-face kNN that actually cuts (gained)
          const real_t Rm2 = c.maxVertexRsq();  // cache: one O(nt) scan, not one per candidate
          const real_t Lh = 0.5f * L, rad2 = 4.0f * 1.1f * Rm2;  // (2Rmax)^2 +10% margin
          auto dist2 = [&](int g) {
            real_t rx = posRaw[3 * g] - sx, ry = posRaw[3 * g + 1] - sy, rz = posRaw[3 * g + 2] - sz;
            rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
            ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
            rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
            return rx * rx + ry * ry + rz * rz;
          };
          bool flag = false;
          // lost face: a current face-neighbour whose bisector is now beyond every vertex
          for (int k = 6; k < np && !flag; ++k)
            if (c.pnbr[k] >= 0 && dist2(c.pnbr[k]) > rad2) flag = true;
          // gained face: a non-face kNN whose bisector ACTUALLY cuts a vertex (n·v > d) — the real
          // test, not just "inside 2Rmax" (most candidates in that ball are no-ops, not new faces)
          const int kk = ncRaw[i];
          for (int t = 0; t < kk && !flag; ++t) {
            const int g = candRaw[(size_t)i * KCAND + t];
            real_t rx = posRaw[3 * g] - sx, ry = posRaw[3 * g + 1] - sy, rz = posRaw[3 * g + 2] - sz;
            rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
            ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
            rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
            const real_t dd = 0.5f * (rx * rx + ry * ry + rz * rz);
            if (dd >= 2.0f * Rm2) continue;  // bisector beyond the cell (cached Rmax) -> cannot cut
            bool isFace = false;
            for (int m = 6; m < np; ++m)
              if (c.pnbr[m] == g) { isFace = true; break; }
            if (isFace) continue;
            for (int t2 = 0; t2 < c.nt; ++t2)  // does this non-face bisector cut any vertex?
              if (c.alive[t2] && rx * c.vx[t2] + ry * c.vy[t2] + rz * c.vz[t2] > dd) { flag = true; break; }
          }
          needfixL(i) = flag;
          if (flag) { int s = Kokkos::atomic_fetch_add(&fixCountL(), 1); fixListL(s) = i; }
        });
        Kokkos::fence();
      };
      auto repair = [&](int nfix) {
        Kokkos::parallel_for("repair", Kokkos::RangePolicy<Exec>(0, nfix), KOKKOS_LAMBDA(int s) {
          const int i = fixListL(s);
          Cell c;
          buildCell(i, c, posRaw, candRaw, ncRaw, L);
          volRcL(i) = c.overflow ? 0.0f : c.volume();
        });
        Kokkos::fence();
      };
      reevalFlag();
      int nfix = 0;
      Kokkos::deep_copy(Kokkos::View<int, Kokkos::HostSpace>(&nfix), fixCount);
      repair(nfix);  // warm
      double tFlag = 1e30, tFix = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        auto t0 = clk::now(); reevalFlag(); auto t1 = clk::now(); tFlag = std::min(tFlag, secs(t0, t1));
        Kokkos::deep_copy(Kokkos::View<int, Kokkos::HostSpace>(&nfix), fixCount);
        auto t2 = clk::now(); repair(nfix); auto t3 = clk::now(); tFix = std::min(tFix, secs(t2, t3));
      }
      // residual error after repair (vs full rebuild) — validates no false negatives
      double resid = 0;
      {
        auto hRc = Kokkos::create_mirror_view(volRc);
        auto hRb = Kokkos::create_mirror_view(volRb);
        Kokkos::deep_copy(hRc, volRc); Kokkos::deep_copy(hRb, volRb);
        for (int i = 0; i < N; ++i) resid += std::fabs((double)hRc(i) - (double)hRb(i));
      }
      const double effMcs = N / (tFlag + tFix) / 1e3;
      std::printf("    Phase1.5: flagged %.1f%%  reeval+flag %.1f + repair %.1f ms  -> %.1f Mc/s "
                  "(%.2fx rebuild)  resid=%.2e\n",
                  100.0 * nfix / N, tFlag * 1e3, tFix * 1e3, effMcs, effMcs / (N / tRb / 1e3), resid);
    }
  }
  Kokkos::finalize();
  return 0;
}
