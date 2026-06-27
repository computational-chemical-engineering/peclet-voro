/**
 * @file bench_update_strategies.cpp
 * \brief Part II — comparison of moving-cell UPDATE STRATEGIES vs displacement, on the PRODUCTION path.
 *
 * The moving-particle workload: a periodic random point set, each seed given a fixed Gaussian velocity
 * (ballistic motion — this study is about UPDATING THE CELLS, not the underlying dynamics, so there are no
 * forces). Each step the seeds drift by `disp` cell-sizes (cellSize = cbrt(V/N)); we update the Voronoi
 * tessellation and compare strategies.
 *
 * ALL full rebuilds go through the production worklist tessellator `vor::device::buildTessellation`, which now
 * emits (opt-in) the resident TopologyStore (np/nt/pnbr/tri) AND a per-cell candidate (skin) list. Re-eval
 * runs `ConvexCell::reevalGeometry` over the store; local repair re-clips the stored skin list (no re-gather).
 * The oracle is also `buildTessellation` (ground truth), so every number is validated against true Voronoi.
 *
 *   S0  pure re-eval     — never rebuild topology after the seed (speed ceiling + error growth).
 *   S1  rebuild each step — buildTessellation every step (correctness/cost baseline == oracle).
 *   S2  displacement Verlet — rebuild when max displacement since the last rebuild exceeds skin/2; re-eval between.
 *   S3  Verlet + convexity + INDEPENDENT local repair (off the stored skin list, no propagation).
 *   S4  Verlet + convexity + PROPAGATING repair (star-splaying: re-flag asymmetric neighbours, iterate).
 *   S5  = S4 swept over skin (the optimal-skin study).
 *
 * Diagnostics vs the production oracle: cell-volume rel error, topoMism (topology differs from a fresh
 * rebuild — mostly volume-irrelevant marginal faces), listMiss (a TRUE neighbour absent from the cell's
 * candidate list — i.e. genuine list incompleteness; should be ~0).
 *
 * Precision: FP64 default, FP32 with -DCC_FLOAT. Run: ./bench_update_strategies [N] [nSteps] [disp1 ...]
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"
#include "vorflow/device/tessellator.hpp"  // production worklist tessellation: rebuild + ground-truth oracle
#include "vorflow/device/topology_store.hpp"

#ifdef CC_FLOAT
using real_t = float;
static constexpr const char* kPrec = "FP32";
#else
using real_t = double;
static constexpr const char* kPrec = "FP64";
#endif

using Exec = tpx::ExecSpace;
using Mem = tpx::MemSpace;
static constexpr int CMAXP = 64, CMAXT = 112;
static constexpr int KCAND = 256;  // candidate (skin) list size emitted by buildTessellation. Recording is
                                   // in worklist (block) order; the cap must exceed the worst-cell examined
                                   // count (~security ball, tail ~150-200) or a far face of a large cell is
                                   // truncated out of the list (shows up as listMiss). 256 covers the tail.
using Cell = vor::device::ConvexCell<real_t, CMAXP, CMAXT>;
using Store = vor::device::TopologyStore<CMAXP, CMAXT>;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

/// Local repair: rebuild cell i by clipping its stored candidate (skin) list. The list (from the last full
/// rebuild) is in worklist-examined order, NOT strictly distance-sorted, so we scan all of it (`continue`,
/// not `break`) and rely on the per-candidate security cull — correct regardless of order, and cheap (~K).
KOKKOS_INLINE_FUNCTION Cell repairCell(int i, const real_t* P, const int* candId, int nc, real_t L) {
  Cell c;
  const real_t Lh = real_t(0.5) * L;
  c.initBox(L, L, L);
  const real_t sx = P[3 * i], sy = P[3 * i + 1], sz = P[3 * i + 2];
  real_t secR2 = real_t(2) * c.maxVertexRsq();
  for (int t = 0; t < nc && !c.overflow; ++t) {
    const int j = candId[(size_t)i * KCAND + t];
    real_t rx = P[3 * j] - sx, ry = P[3 * j + 1] - sy, rz = P[3 * j + 2] - sz;
    rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
    ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
    rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
    const real_t off = real_t(0.5) * (rx * rx + ry * ry + rz * rz);
    if (off >= secR2) continue;  // beyond the security radius: cannot cut (list not sorted -> continue)
    const real_t nrm[3] = {rx, ry, rz};
    if (c.clip(nrm, off, j)) secR2 = real_t(2) * c.maxVertexRsq();
  }
  return c;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = argc > 1 ? std::atoi(argv[1]) : 300000;
    const int nSteps = argc > 2 ? std::atoi(argv[2]) : 16;
    std::vector<real_t> disps;
    for (int i = 3; i < argc; ++i) disps.push_back((real_t)std::atof(argv[i]));
    if (disps.empty()) disps = {(real_t)0.0005, (real_t)0.001, (real_t)0.002, (real_t)0.005};

    const real_t L = 1;
    const real_t spacing = std::cbrt((double)L * L * L / N);  // == cellSize
    const real_t dt = 1;
    std::printf("=== update-strategy study on the PRODUCTION builder (%s) N=%d nSteps=%d cellSize=%.4g ===\n",
                kPrec, N, nSteps, (double)spacing);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0, 1);
    std::normal_distribution<real_t> Ng(0, 1);
    std::vector<real_t> x0(3 * N), vel(3 * N);
    for (int i = 0; i < 3 * N; ++i) x0[i] = U(rng);
    for (int i = 0; i < 3 * N; ++i) vel[i] = Ng(rng);
    Kokkos::View<real_t*, Mem> x0d("x0", 3 * N), veld("vel", 3 * N), pos("pos", 3 * N);
    Kokkos::deep_copy(x0d, Kokkos::View<real_t*, Kokkos::HostSpace>(x0.data(), 3 * N));
    Kokkos::deep_copy(veld, Kokkos::View<real_t*, Kokkos::HostSpace>(vel.data(), 3 * N));

    Store store;
    store.alloc(N);  // strategy's resident topology (emitted by buildTessellation)
    Kokkos::View<real_t*, Mem> vol("vol", N), volOra("volOra", N);
    Kokkos::View<int*, Mem> flag("flag", N), active("active", N), nextA("nextA", N);
    Kokkos::View<int*, Mem> changed("changed", N), workList("workList", N);
    Kokkos::View<int, Mem> wcount("wcount");
    Kokkos::View<int*, Mem> candId(Kokkos::view_alloc(std::string("candId"), Kokkos::WithoutInitializing),
                                   (size_t)N * KCAND);
    Kokkos::View<int*, Mem> ncand("ncand", N);
    Kokkos::View<int*, Mem> candIdSnap(
        Kokkos::view_alloc(std::string("candIdSnap"), Kokkos::WithoutInitializing), (size_t)N * KCAND);
    Kokkos::View<int*, Mem> ncandSnap("ncandSnap", N);
    Kokkos::View<real_t*, Mem> xRef("xRef", 3 * N);
    vor::TessellationView<real_t> prodView;  // ground-truth oracle CSR

    auto P_at = [&](real_t scale, int step) {
      auto Pd = pos;
      auto X0 = x0d;
      auto V = veld;
      const real_t s = scale * step * dt, Ll = L;
      Kokkos::parallel_for(
          "advance", Kokkos::RangePolicy<Exec>(0, 3 * N), KOKKOS_LAMBDA(int i) {
            real_t v = X0(i) + V(i) * s;
            v -= Ll * Kokkos::floor(v / Ll);
            Pd(i) = v;
          });
    };
    auto scaleFor = [&](real_t disp) { return disp * spacing; };

    // STRATEGY full rebuild: production worklist tessellator, emitting the topology store + candidate list +
    // volume; reset the Verlet reference. This is the same fast path used for S1/S2/S3/S4/S5 rebuilds.
    auto fullRebuild = [&] {
      const real_t Larr[3] = {L, L, L};
      Kokkos::View<real_t*, Mem> wdummy;
      Kokkos::View<long*, Mem> gdummy;
      auto res = vor::device::buildTessellation<real_t, false>(
          pos, wdummy, N, Larr, 4, N, gdummy, vor::device::NoSdf{}, /*withForceGeom=*/false, -1, store.np,
          store.nt, store.pnbr, store.tri, candId, ncand, KCAND);
      Kokkos::deep_copy(vol, res.view.cellVolume);
      Kokkos::deep_copy(xRef, pos);
    };
    // GROUND-TRUTH oracle: an independent buildTessellation (volume + CSR for neighbour sets).
    auto buildOracle = [&] {
      const real_t Larr[3] = {L, L, L};
      Kokkos::View<real_t*, Mem> wdummy;
      auto res = vor::device::buildTessellation<real_t, false>(pos, wdummy, N, Larr);
      prodView = res.view;
      Kokkos::deep_copy(volOra, prodView.cellVolume);
    };

    auto reevalAll = [&] {
      auto P = pos;
      auto Vv = vol;
      Store st = store;
      Kokkos::parallel_for(
          "reeval", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
            Cell c;
            st.load(i, c, L, L, L);
            c.reevalGeometry(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), L);
            Vv(i) = c.volumePerVertex();
          });
      Kokkos::fence();
    };

    const real_t d2tol = (sizeof(real_t) == 4 ? real_t(2e-3) : real_t(1e-4)) * spacing;
    auto reevalFlagD2 = [&] {
      auto P = pos;
      auto Vv = vol;
      auto Fl = flag;
      Store st = store;
      const real_t tol = d2tol;
      Kokkos::parallel_for(
          "reevalFlagD2", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
            Cell c;
            st.load(i, c, L, L, L);
            c.reevalGeometry(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), L);
            Vv(i) = c.volumePerVertex();
            Fl(i) = c.isSelfConsistent(tol) ? 0 : 1;
          });
      Kokkos::fence();
    };

    auto compact = [&](Kokkos::View<int*, Mem> mask) -> int {
      Kokkos::deep_copy(wcount, 0);
      auto M = mask;
      auto WL = workList;
      auto wc = wcount;
      Kokkos::parallel_for(
          "compact", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(int i) { if (M(i)) WL(Kokkos::atomic_fetch_add(&wc(), 1)) = i; });
      int h = 0;
      Kokkos::deep_copy(h, wcount);
      return h;
    };

    // Local repair of the `cnt` work-list cells off the stored skin list (no re-gather). markChanged sets
    // changed(i) iff the rebuilt neighbour set differs from the stored one (np or Σpnbr) — propagation trigger.
    auto repairWork = [&](int cnt, bool markChanged) {
      auto P = pos;
      auto cId = candId;
      auto nc = ncand;
      auto Vv = vol;
      auto WL = workList;
      auto Ch = changed;
      Store st = store;
      Kokkos::parallel_for(
          "repairWork", Kokkos::RangePolicy<Exec>(0, cnt), KOKKOS_LAMBDA(int s) {
            const int i = WL(s);
            int onp = st.np(i);
            long osum = 0;
            for (int k = 6; k < onp; ++k) osum += st.pnbr((size_t)i * CMAXP + k);
            Cell c = repairCell(i, P.data(), cId.data(), nc(i), L);
            if (c.overflow) return;
            Vv(i) = c.volumePerVertex();
            if (markChanged) {
              long nsum = 0;
              for (int k = 6; k < c.np; ++k) nsum += c.pnbr[k];
              Ch(i) = (c.np != onp || nsum != osum) ? 1 : 0;
            }
            st.save(i, c);
          });
      Kokkos::fence();
    };

    auto propagateWork = [&](int cnt, Kokkos::View<int*, Mem> outNext) {
      Store st = store;
      auto WL = workList;
      auto Ch = changed;
      auto NX = outNext;
      Kokkos::parallel_for(
          "propagateWork", Kokkos::RangePolicy<Exec>(0, cnt), KOKKOS_LAMBDA(int s) {
            const int i = WL(s);
            if (!Ch(i)) return;
            const int npi = st.np(i);
            for (int k = 6; k < npi; ++k) {
              const int j = st.pnbr((size_t)i * CMAXP + k);
              if (j < 0) continue;
              const int npj = st.np(j);
              bool back = false;
              for (int m = 6; m < npj; ++m)
                if (st.pnbr((size_t)j * CMAXP + m) == i) { back = true; break; }
              if (!back) NX(j) = 1;
            }
          });
      Kokkos::fence();
    };

    auto maxDispCells = [&]() -> real_t {
      auto P = pos;
      auto XR = xRef;
      const real_t Ll = L, Lh = real_t(0.5) * L;
      real_t m2 = 0;
      Kokkos::parallel_reduce(
          "maxDisp", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(int i, real_t& lm) {
            real_t dx = P(3 * i) - XR(3 * i), dy = P(3 * i + 1) - XR(3 * i + 1),
                   dz = P(3 * i + 2) - XR(3 * i + 2);
            dx = dx > Lh ? dx - Ll : (dx < -Lh ? dx + Ll : dx);
            dy = dy > Lh ? dy - Ll : (dy < -Lh ? dy + Ll : dy);
            dz = dz > Lh ? dz - Ll : (dz < -Lh ? dz + Ll : dz);
            const real_t d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > lm) lm = d2;
          },
          Kokkos::Max<real_t>(m2));
      return Kokkos::sqrt(m2) / spacing;
    };

    auto accuracy = [&](double& meanRel, double& maxRel, long& mism) {
      auto hv = Kokkos::create_mirror_view(vol);
      auto ho = Kokkos::create_mirror_view(volOra);
      Kokkos::deep_copy(hv, vol);
      Kokkos::deep_copy(ho, volOra);
      double s = 0, mx = 0;
      long m = 0;
      for (int i = 0; i < N; ++i) {
        double r = ho(i) > 0 ? std::fabs((double)hv(i) - ho(i)) / ho(i) : 0;
        s += r;
        mx = std::max(mx, r);
        if (r > 1e-3) ++m;
      }
      meanRel = s / N;
      maxRel = mx;
      mism = m;
    };

    // vs the production oracle: topoMism = stored topology differs from truth; listMiss = a TRUE neighbour is
    // absent from the strategy's candidate-list snapshot (genuine list incompleteness).
    auto topoDiag = [&](long& topoMism, long& listMiss) {
      Store st = store;
      auto pv = prodView;
      auto cs = candIdSnap;
      auto ncs = ncandSnap;
      Kokkos::View<long, Mem> tm("tm"), lm("lm");
      Kokkos::deep_copy(tm, 0);
      Kokkos::deep_copy(lm, 0);
      Kokkos::parallel_for(
          "topoDiag", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
            const int snp = st.np(i);
            bool diff = false, miss = false;
            for (int g = pv.facetBegin(i); g < pv.facetEnd(i); ++g) {
              const int j = (int)pv.facetNbr(g);
              if (j < 0) continue;
              bool inStore = false;
              for (int m = 6; m < snp; ++m)
                if (st.pnbr((size_t)i * CMAXP + m) == j) { inStore = true; break; }
              if (!inStore) diff = true;
              bool inList = false;
              const int nc = ncs(i);
              for (int t = 0; t < nc; ++t)
                if (cs((size_t)i * KCAND + t) == j) { inList = true; break; }
              if (!inList) miss = true;
            }
            if (diff) Kokkos::atomic_inc(&tm());
            if (miss) Kokkos::atomic_inc(&lm());
          });
      Kokkos::deep_copy(topoMism, tm);
      Kokkos::deep_copy(listMiss, lm);
    };

    // close a run: snapshot the candidate list the strategy used, build the oracle, measure accuracy + diag.
    auto closeStep = [&](double& meanRel, double& maxRel, long& mism, long& topoM, long& listM) {
      Kokkos::deep_copy(candIdSnap, candId);
      Kokkos::deep_copy(ncandSnap, ncand);
      buildOracle();
      accuracy(meanRel, maxRel, mism);
      topoDiag(topoM, listM);
    };

    auto row = [&](const char* name, double disp, double skin, double ms, double rebPct, double touchPct,
                   double meanRel, double maxRel, long mism, long topoM, long listM) {
      std::printf("%-16s %6.4f %5.2f %8.2f %6.1f %6.1f %10.2e %10.2e %8ld %8ld %8ld\n", name, disp, skin,
                  ms, rebPct, touchPct, meanRel, maxRel, mism, topoM, listM);
    };

    std::printf("\n[parity] static harness vs oracle is now identical (both buildTessellation).\n");
    std::printf("%-16s %6s %5s %8s %6s %6s %10s %10s %8s %8s %8s\n", "strategy", "disp", "skin", "ms/step",
                "rebld%", "touch%", "meanRelV", "maxRelV", "mism", "topoMism", "listMiss");

    const int fallbackN = (int)(0.30 * N);
    // shared Verlet + repair driver (mode: 0 reeval-only, 1 independent repair, 2 propagating repair).
    auto runVerlet = [&](const char* name, real_t disp, real_t skin, int mode) {
      const real_t scale = scaleFor(disp);
      P_at(scale, 0);
      fullRebuild();
      double t = 0, meanRel = 0, maxRel = 0;
      long mism = 0, topoM = 0, listM = 0, touch = 0, rebuilds = 0;
      for (int s = 1; s <= nSteps; ++s) {
        P_at(scale, s);
        auto a = clk::now();
        if (maxDispCells() > real_t(0.5) * skin) {
          fullRebuild();
          ++rebuilds;
          touch += N;
        } else if (mode == 0) {
          reevalAll();
        } else {
          reevalFlagD2();
          Kokkos::deep_copy(active, flag);
          for (int sweep = 0; sweep < 12; ++sweep) {
            const int cnt = compact(active);
            if (cnt == 0) break;
            if (cnt > fallbackN) {
              fullRebuild();
              ++rebuilds;
              touch += N;
              break;
            }
            touch += cnt;
            Kokkos::deep_copy(changed, 0);
            repairWork(cnt, mode == 2);
            if (mode != 2) break;  // independent repair: no propagation
            Kokkos::deep_copy(nextA, 0);
            propagateWork(cnt, nextA);
            Kokkos::deep_copy(active, nextA);
          }
        }
        t += secs(a, clk::now());
        if (s == nSteps) closeStep(meanRel, maxRel, mism, topoM, listM);
      }
      row(name, disp, skin, 1e3 * t / nSteps, 100.0 * rebuilds / nSteps, 100.0 * touch / ((double)nSteps * N),
          meanRel, maxRel, mism, topoM, listM);
    };

    const real_t mainSkin = 0.1;
    for (real_t disp : disps) {
      const real_t scale = scaleFor(disp);
      // S0 pure re-eval
      {
        P_at(scale, 0);
        fullRebuild();
        double t = 0, meanRel = 0, maxRel = 0;
        long mism = 0, topoM = 0, listM = 0;
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          auto a = clk::now();
          reevalAll();
          t += secs(a, clk::now());
          if (s == nSteps) closeStep(meanRel, maxRel, mism, topoM, listM);
        }
        row("S0 pure-reeval", disp, 0, 1e3 * t / nSteps, 0, 0, meanRel, maxRel, mism, topoM, listM);
      }
      // S1 rebuild each step (baseline == oracle path)
      {
        P_at(scale, 0);
        fullRebuild();
        double t = 0, meanRel = 0, maxRel = 0;
        long mism = 0, topoM = 0, listM = 0;
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          auto a = clk::now();
          fullRebuild();
          t += secs(a, clk::now());
          if (s == nSteps) closeStep(meanRel, maxRel, mism, topoM, listM);
        }
        row("S1 rebuild-each", disp, 0, 1e3 * t / nSteps, 100, 100, meanRel, maxRel, mism, topoM, listM);
      }
      runVerlet("S2 disp-verlet", disp, mainSkin, 0);
      runVerlet("S3 convex-local", disp, mainSkin, 1);
      runVerlet("S4 convex-prop", disp, mainSkin, 2);
    }

    std::printf("\n--- S5 = S4 (propagating) swept over skin (cell-sizes) ---\n");
    for (real_t disp : disps)
      for (real_t skin : {(real_t)0.05, (real_t)0.1, (real_t)0.2, (real_t)0.4})
        runVerlet("S5 prop", disp, skin, 2);

    // space-filling sanity (oracle Σvol ≈ L^3)
    P_at(scaleFor(disps[0]), 0);
    buildOracle();
    {
      double sv = 0;
      auto ho = Kokkos::create_mirror_view(volOra);
      Kokkos::deep_copy(ho, volOra);
      for (int i = 0; i < N; ++i) sv += ho(i);
      std::printf("\n[check] oracle Σvol=%.6f (~%.6f)\n", sv, (double)L * L * L);
    }
  }
  Kokkos::finalize();
  return 0;
}
