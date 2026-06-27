/**
 * @file bench_update_strategies.cpp
 * \brief Part II — exhaustive comparison of moving-cell UPDATE STRATEGIES vs displacement.
 *
 * The moving-particle workload: a periodic random point set, each seed given a fixed Gaussian velocity
 * (ballistic motion — this study is about UPDATING THE CELLS, not the underlying dynamics, so there are no
 * forces). Each step the seeds drift by `disp` cell-sizes (cellSize = cbrt(V/N)); we then update the Voronoi
 * tessellation and compare strategies for the topology/geometry update:
 *
 *   S0  pure re-eval            — never rebuild topology; reevalGeometry over the step-0 store (speed ceiling
 *                                 + the error-growth curve).
 *   S1  full rebuild每step      — rebuild every cell from the grid each step (correctness/cost baseline; this
 *                                 is also the oracle).
 *   S2  displacement Verlet     — global rebuild when max accumulated displacement since the last rebuild
 *                                 exceeds skin/2; pure re-eval between.
 *   S3  convexity + local repair (independent) — reeval + D2 self-consistency flag; rebuild ONLY flagged
 *                                 cells, no propagation.
 *   S4  convexity + propagating sweep — reeval + D2 flag; rebuild flagged, then propagate to asymmetric
 *                                 neighbours (star-splaying-style) and iterate to a fixpoint (or fall back to
 *                                 a full rebuild if too many cells are touched).
 *
 * Per strategy/displacement we report: steady-state update time/step, the per-step rebuild/repair work
 * (cells touched), and accuracy vs the per-step full-rebuild ORACLE: mean & max cell-volume relative error,
 * and (the key scientific question) for the convexity strategies the post-update residual + the count of
 * cells that differ from the oracle yet were NOT flagged (D2 false negatives ⇒ does the convexity certificate
 * + propagation catch every topology change?).
 *
 * Precision: FP64 default, FP32 with -DCC_FLOAT. Run: ./bench_update_strategies [N] [nSteps] [disp1 disp2 ...]
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"
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
using Cell = vor::device::ConvexCell<real_t, CMAXP, CMAXT>;
using Store = vor::device::TopologyStore<CMAXP, CMAXT>;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// ----------------------------------------------------------------------------------------------------
// Uniform periodic grid (counting sort). Self-contained so the harness can rebuild arbitrary cell subsets
// (needed for local / propagating repair) with the same builder used for the full rebuild — a fair cost cmp.
// ----------------------------------------------------------------------------------------------------
struct GridView {
  Kokkos::View<int*, Mem> cellStart;  // ncell+1
  Kokkos::View<int*, Mem> binned;     // N (seed ids in grid order)
  int dim;
  real_t csz, L;
  KOKKOS_INLINE_FUNCTION int wrap(int g) const { return (g % dim + dim) % dim; }
  KOKKOS_INLINE_FUNCTION int cellOf(int gx, int gy, int gz) const {
    return wrap(gx) + dim * (wrap(gy) + dim * wrap(gz));
  }
};

struct Grid {
  GridView v;
  Kokkos::View<int*, Mem> counts, cursor;
  void alloc(int N, int dim, real_t csz, real_t L) {
    v.dim = dim;
    v.csz = csz;
    v.L = L;
    const int ncell = dim * dim * dim;
    v.cellStart = Kokkos::View<int*, Mem>("cellStart", ncell + 1);
    v.binned = Kokkos::View<int*, Mem>("binned", N);
    counts = Kokkos::View<int*, Mem>("counts", ncell + 1);
    cursor = Kokkos::View<int*, Mem>("cursor", ncell);
  }
  void build(const Kokkos::View<real_t*, Mem>& pos, int N) {
    const int ncell = v.dim * v.dim * v.dim;
    Kokkos::deep_copy(counts, 0);
    auto cnts = counts;
    GridView g = v;
    auto P = pos;
    Kokkos::parallel_for(
        "grid.bin", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
          int gx = (int)Kokkos::floor(P(3 * i) / g.csz);
          int gy = (int)Kokkos::floor(P(3 * i + 1) / g.csz);
          int gz = (int)Kokkos::floor(P(3 * i + 2) / g.csz);
          Kokkos::atomic_inc(&cnts(g.cellOf(gx, gy, gz)));
        });
    auto cs = v.cellStart;
    Kokkos::parallel_scan(
        "grid.scan", Kokkos::RangePolicy<Exec>(0, ncell + 1),
        KOKKOS_LAMBDA(int c, int& acc, bool fin) {
          int val = (c < ncell) ? cnts(c) : 0;
          if (fin) cs(c) = acc;
          acc += val;
        });
    auto cur = cursor;
    Kokkos::parallel_for(
        "grid.cur", Kokkos::RangePolicy<Exec>(0, ncell),
        KOKKOS_LAMBDA(int c) { cur(c) = cs(c); });
    auto bn = v.binned;
    Kokkos::parallel_for(
        "grid.scatter", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
          int gx = (int)Kokkos::floor(P(3 * i) / g.csz);
          int gy = (int)Kokkos::floor(P(3 * i + 1) / g.csz);
          int gz = (int)Kokkos::floor(P(3 * i + 2) / g.csz);
          int slot = Kokkos::atomic_fetch_add(&cur(g.cellOf(gx, gy, gz)), 1);
          bn(slot) = i;
        });
    Kokkos::fence();
  }
};

static constexpr int KCAND = 128;  // per-cell k-NN candidate list; sized for skin headroom (S5 Verlet
                                   // list): the security ball holds ~70, the extra entries are the skin.

/// Gather seed i's KCAND nearest neighbours from the grid window, SORTED by ascending distance, into
/// candId[i*KCAND..]. Closest-first order is what keeps the clip's committed-plane count bounded (a far
/// plane that would be redundant is never reached once the security radius closes the cell).
KOKKOS_INLINE_FUNCTION void gatherKNNOne(int i, const real_t* P, const GridView& g, int sw, int* candId,
                                         int* ncand) {
  const real_t L = g.L, Lh = real_t(0.5) * L;
  const real_t sx = P[3 * i], sy = P[3 * i + 1], sz = P[3 * i + 2];
  const int hx = (int)Kokkos::floor(sx / g.csz), hy = (int)Kokkos::floor(sy / g.csz),
            hz = (int)Kokkos::floor(sz / g.csz);
  real_t cd[KCAND];
  int ci[KCAND], m = 0;
  for (int dz = -sw; dz <= sw; ++dz)
    for (int dy = -sw; dy <= sw; ++dy)
      for (int dx = -sw; dx <= sw; ++dx) {
        const int gc = g.cellOf(hx + dx, hy + dy, hz + dz);
        for (int q = g.cellStart(gc); q < g.cellStart(gc + 1); ++q) {
          const int j = g.binned(q);
          if (j == i) continue;
          real_t rx = P[3 * j] - sx, ry = P[3 * j + 1] - sy, rz = P[3 * j + 2] - sz;
          rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
          ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
          rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
          const real_t d2 = rx * rx + ry * ry + rz * rz;
          int p;
          if (m < KCAND) {
            cd[m] = d2;
            ci[m] = j;
            p = m++;
          } else if (d2 < cd[KCAND - 1]) {
            cd[KCAND - 1] = d2;
            ci[KCAND - 1] = j;
            p = KCAND - 1;
          } else {
            continue;
          }
          for (; p > 0 && cd[p] < cd[p - 1]; --p) {  // bubble into sorted position
            real_t td = cd[p]; cd[p] = cd[p - 1]; cd[p - 1] = td;
            int tj = ci[p]; ci[p] = ci[p - 1]; ci[p - 1] = tj;
          }
        }
      }
  for (int t = 0; t < m; ++t) candId[(size_t)i * KCAND + t] = ci[t];
  ncand[i] = m;
}

/// Build cell i by clipping its sorted candidate list closest-first with the security early-out.
KOKKOS_INLINE_FUNCTION Cell buildFromCand(int i, const real_t* P, const int* candId, int nc, real_t L) {
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
    if (off >= secR2) break;  // sorted ⇒ all farther; cell closed
    const real_t nrm[3] = {rx, ry, rz};
    if (c.clip(nrm, off, j)) secR2 = real_t(2) * c.maxVertexRsq();
  }
  return c;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = argc > 1 ? std::atoi(argv[1]) : 300000;
    const int nSteps = argc > 2 ? std::atoi(argv[2]) : 24;
    std::vector<real_t> disps;
    for (int i = 3; i < argc; ++i) disps.push_back((real_t)std::atof(argv[i]));
    if (disps.empty()) disps = {(real_t)0.001, (real_t)0.005, (real_t)0.02, (real_t)0.05, (real_t)0.1};

    const real_t L = 1;
    const real_t spacing = std::cbrt((double)L * L * L / N);  // == cellSize
    const int dim = std::max(1, (int)std::floor(L / spacing));
    const real_t csz = L / dim;
    const int sw = 3;  // grid window half-width (verify space-filling identity below)
    const real_t dt = 1;
    std::printf("=== update-strategy study (%s) N=%d  nSteps=%d  cellSize=%.4g  grid=%d^3 sw=%d ===\n",
                kPrec, N, nSteps, (double)spacing, dim, sw);

    // initial random seeds + fixed unit-variance Gaussian velocity directions (scaled per-disp)
    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0, 1);
    std::normal_distribution<real_t> Ng(0, 1);
    std::vector<real_t> x0(3 * N), vel(3 * N);
    for (int i = 0; i < 3 * N; ++i) x0[i] = U(rng);
    for (int i = 0; i < 3 * N; ++i) vel[i] = Ng(rng);  // per-step disp = |vel|*dt*scale (scale set per-disp)

    Kokkos::View<real_t*, Mem> x0d("x0", 3 * N), veld("vel", 3 * N), pos("pos", 3 * N);
    Kokkos::deep_copy(x0d, Kokkos::View<real_t*, Kokkos::HostSpace>(x0.data(), 3 * N));
    Kokkos::deep_copy(veld, Kokkos::View<real_t*, Kokkos::HostSpace>(vel.data(), 3 * N));

    Grid grid;
    grid.alloc(N, dim, csz, L);
    Store store;
    store.alloc(N);            // strategy's resident topology
    Store oracleStore;
    oracleStore.alloc(N);      // oracle topology (for false-negative accounting)
    Kokkos::View<real_t*, Mem> vol("vol", N), volOra("volOra", N);
    Kokkos::View<int*, Mem> flag("flag", N), active("active", N), nextA("nextA", N);
    Kokkos::View<int*, Mem> changed("changed", N), workList("workList", N);
    Kokkos::View<int, Mem> wcount("wcount");  // compaction counter
    Kokkos::View<int*, Mem> candId(Kokkos::view_alloc(std::string("candId"), Kokkos::WithoutInitializing),
                                   (size_t)N * KCAND);
    Kokkos::View<int*, Mem> ncand("ncand", N);
    Kokkos::View<real_t*, Mem> xRef("xRef", 3 * N);  // positions at the last full rebuild (S5 Verlet ref)

    // k-NN gather (sorted) from the current grid, for all cells or a masked subset.
    auto gatherAll = [&] {
      GridView g = grid.v;
      auto P = pos;
      auto cId = candId;
      auto nc = ncand;
      Kokkos::parallel_for(
          "gatherAll", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(int i) { gatherKNNOne(i, P.data(), g, sw, cId.data(), nc.data()); });
      Kokkos::fence();
    };

    auto P_at = [&](real_t scale, int step) {  // pos = wrap(x0 + vel*scale*step*dt)
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

    // scale s.t. RMS per-step displacement (over 3N comps, var=1) == disp*spacing  => scale = disp*spacing
    auto scaleFor = [&](real_t disp) { return disp * spacing; };

    // Build the oracle (full rebuild from current grid) into volOra + oracleStore.
    auto buildOracle = [&] {
      grid.build(pos, N);
      gatherAll();
      auto P = pos;
      auto cId = candId;
      auto nc = ncand;
      auto Vo = volOra;
      Store st = oracleStore;
      Kokkos::parallel_for(
          "oracle", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
            Cell c = buildFromCand(i, P.data(), cId.data(), nc(i), L);
            Vo(i) = c.overflow ? real_t(0) : c.volumePerVertex();
            if (!c.overflow) st.save(i, c);
          });
      Kokkos::fence();
    };

    // reeval one cell from `store` -> vol(i)
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

    // full rebuild from grid into store + vol
    auto rebuildAll = [&] {
      grid.build(pos, N);
      gatherAll();
      auto P = pos;
      auto cId = candId;
      auto nc = ncand;
      auto Vv = vol;
      Store st = store;
      Kokkos::parallel_for(
          "rebuildAll", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
            Cell c = buildFromCand(i, P.data(), cId.data(), nc(i), L);
            Vv(i) = c.overflow ? real_t(0) : c.volumePerVertex();
            if (!c.overflow) st.save(i, c);
          });
      Kokkos::fence();
    };

    // accuracy of `vol` vs `volOra`
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

    // D2 convexity flag (reeval + self-consistency) -> flag(i), vol(i). tol scaled to cell size and to the
    // working precision: a self-consistent cell's vertices sit ON their 3 planes and inside the others, so
    // the "poke-out" of a genuine flip is O(displacement); the tol just rejects round-off. FP32 vertex
    // recompute is far noisier than FP64, so it needs a looser tol or marginal faces flicker (false flags
    // that inflate the repaired/propagated set — the Sugihara robustness regime).
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

    // Stream-compact a mask into workList; returns the count (so kernels iterate over WORK, not all N —
    // the difference between cost ∝ cells-touched and cost ∝ N for the local/propagating strategies).
    auto compact = [&](Kokkos::View<int*, Mem> mask) -> int {
      Kokkos::deep_copy(wcount, 0);
      auto M = mask;
      auto WL = workList;
      auto wc = wcount;
      Kokkos::parallel_for(
          "compact", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
            if (M(i)) WL(Kokkos::atomic_fetch_add(&wc(), 1)) = i;
          });
      int h = 0;
      Kokkos::deep_copy(h, wcount);
      return h;
    };

    // Gather + rebuild the `cnt` cells in workList; refresh store + vol. markChanged: also set changed(i)
    // iff the rebuilt neighbour set differs from the stored one (np or Σpnbr) — the propagation trigger.
    auto rebuildWork = [&](int cnt, bool markChanged) {
      GridView g = grid.v;
      auto P = pos;
      auto cId = candId;
      auto nc = ncand;
      auto Vv = vol;
      auto WL = workList;
      auto Ch = changed;
      Store st = store;
      Kokkos::parallel_for(
          "gatherWork", Kokkos::RangePolicy<Exec>(0, cnt),
          KOKKOS_LAMBDA(int s) { gatherKNNOne(WL(s), P.data(), g, sw, cId.data(), nc.data()); });
      Kokkos::parallel_for(
          "rebuildWork", Kokkos::RangePolicy<Exec>(0, cnt), KOKKOS_LAMBDA(int s) {
            const int i = WL(s);
            int onp = st.np(i);
            long osum = 0;
            for (int k = 6; k < onp; ++k) osum += st.pnbr((size_t)i * CMAXP + k);
            Cell c = buildFromCand(i, P.data(), cId.data(), nc(i), L);
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

    // Symmetry propagation: for each just-rebuilt cell i WHOSE TOPOLOGY CHANGED, any neighbour j it now
    // lists whose store does not list i back is asymmetric -> flag j into outNext (caller zeroes it).
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

    // S5 local repair: rebuild the `cnt` work-list cells from their EXISTING (skin) candidate list — NO grid,
    // NO re-gather. Valid while the Verlet criterion holds (every true neighbour is already in the stored
    // list). The sorted list + security early-out means extra skin candidates are never examined.
    auto repairFromList = [&](int cnt, bool markChanged) {
      auto P = pos;
      auto cId = candId;
      auto nc = ncand;
      auto Vv = vol;
      auto WL = workList;
      auto Ch = changed;
      Store st = store;
      Kokkos::parallel_for(
          "repairFromList", Kokkos::RangePolicy<Exec>(0, cnt), KOKKOS_LAMBDA(int s) {
            const int i = WL(s);
            int onp = st.np(i);
            long osum = 0;
            for (int k = 6; k < onp; ++k) osum += st.pnbr((size_t)i * CMAXP + k);
            Cell c = buildFromCand(i, P.data(), cId.data(), nc(i), L);
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

    // max per-seed displacement since xRef (min-image), in cell-sizes — the Verlet criterion.
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

    std::printf("\n%-26s %8s %8s %10s %10s %9s\n", "strategy / disp", "ms/step", "touch%",
                "meanRelV", "maxRelV", "mism>1e-3");

    for (real_t disp : disps) {
      const real_t scale = scaleFor(disp);

      // ---- S0: pure re-eval (store built once at step 0) ----
      {
        P_at(scale, 0);
        rebuildAll();  // seed store at step 0
        double t = 0;
        double meanRel = 0, maxRel = 0;
        long mism = 0;
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          auto a = clk::now();
          reevalAll();
          t += secs(a, clk::now());
          if (s == nSteps) { buildOracle(); accuracy(meanRel, maxRel, mism); }
        }
        std::printf("%-18s %5.3f %8.2f %8.1f %10.2e %10.2e %9ld\n", "S0 pure-reeval", (double)disp,
                    1e3 * t / nSteps, 0.0, meanRel, maxRel, mism);
      }

      // ---- S1: full rebuild every step (== oracle) ----
      {
        P_at(scale, 0);
        rebuildAll();
        double t = 0, meanRel = 0, maxRel = 0;
        long mism = 0;
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          auto a = clk::now();
          rebuildAll();
          t += secs(a, clk::now());
          if (s == nSteps) { buildOracle(); accuracy(meanRel, maxRel, mism); }
        }
        std::printf("%-18s %5.3f %8.2f %8.1f %10.2e %10.2e %9ld\n", "S1 rebuild-each", (double)disp,
                    1e3 * t / nSteps, 100.0, meanRel, maxRel, mism);
      }

      // ---- S2: displacement Verlet (rebuild when max disp since rebuild > skin/2) ----
      {
        const real_t skin = real_t(0.15) * spacing;  // budget in cell-sizes
        P_at(scale, 0);
        rebuildAll();
        int lastRebuild = 0;
        double t = 0, meanRel = 0, maxRel = 0;
        long mism = 0, rebuilds = 0;
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          // max accumulated displacement since lastRebuild = scale*(s-lastRebuild)*max|vel comp-vector|.
          // Bound conservatively by RMS: disp*(s-lastRebuild) cell-sizes (per-axis). Use vector RMS.
          const real_t accum = disp * std::sqrt((real_t)3) * (s - lastRebuild) * spacing;
          auto a = clk::now();
          if (accum > real_t(0.5) * skin) {
            rebuildAll();
            lastRebuild = s;
            ++rebuilds;
          } else {
            reevalAll();
          }
          t += secs(a, clk::now());
          if (s == nSteps) { buildOracle(); accuracy(meanRel, maxRel, mism); }
        }
        std::printf("%-18s %5.3f %8.2f %8.1f %10.2e %10.2e %9ld\n", "S2 disp-verlet", (double)disp,
                    1e3 * t / nSteps, 100.0 * rebuilds / nSteps, meanRel, maxRel, mism);
      }

      // ---- S3: convexity flag + independent local repair (no propagation) ----
      {
        P_at(scale, 0);
        rebuildAll();
        double t = 0, meanRel = 0, maxRel = 0;
        long mism = 0, touch = 0;
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          auto a = clk::now();
          reevalFlagD2();
          grid.build(pos, N);  // repair needs a current grid
          const int cnt = compact(flag);
          rebuildWork(cnt, false);
          t += secs(a, clk::now());
          touch += cnt;
          if (s == nSteps) { buildOracle(); accuracy(meanRel, maxRel, mism); }
        }
        std::printf("%-18s %5.3f %8.2f %8.1f %10.2e %10.2e %9ld\n", "S3 convex-local", (double)disp,
                    1e3 * t / nSteps, 100.0 * touch / ((double)nSteps * N), meanRel, maxRel, mism);
      }

      // ---- S4: convexity flag + propagating sweep (star-splaying-style) ----
      {
        P_at(scale, 0);
        rebuildAll();
        double t = 0, meanRel = 0, maxRel = 0;
        long mism = 0, touch = 0, fellBack = 0;
        const int fallbackN = (int)(0.30 * N);
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          auto a = clk::now();
          reevalFlagD2();
          grid.build(pos, N);
          Kokkos::deep_copy(active, flag);  // sweep-active mask
          long stepTouch = 0;
          bool fb = false;
          for (int sweep = 0; sweep < 12; ++sweep) {
            const int cnt = compact(active);
            if (cnt == 0) break;
            if (cnt > fallbackN) {
              rebuildAll();
              fb = true;
              stepTouch += N;
              break;
            }
            stepTouch += cnt;
            Kokkos::deep_copy(changed, 0);
            rebuildWork(cnt, true);            // rebuild flagged + mark which changed topology
            Kokkos::deep_copy(nextA, 0);
            propagateWork(cnt, nextA);         // spread only from changed cells to asymmetric neighbours
            Kokkos::deep_copy(active, nextA);
          }
          t += secs(a, clk::now());
          touch += stepTouch;
          if (fb) ++fellBack;
          if (s == nSteps) { buildOracle(); accuracy(meanRel, maxRel, mism); }
        }
        std::printf("%-18s %5.3f %8.2f %8.1f %10.2e %10.2e %9ld  (fb=%ld)\n", "S4 convex-prop",
                    (double)disp, 1e3 * t / nSteps, 100.0 * touch / ((double)nSteps * N), meanRel,
                    maxRel, mism, fellBack);
      }
    }

    // ================= S5: persistent skin-list + Verlet trigger + S4 inner loop =================
    // The combined strategy. Each cell keeps the candidate (skin) list gathered at the last FULL rebuild;
    // while max-displacement < skin/2 (Verlet) the list is guaranteed to contain every true neighbour, so
    // local repair clips the STORED list (no grid, no re-gather). When the criterion trips: rebuild the grid,
    // re-gather the lists (with skin), full rebuild, reset the reference. Sweep skin to find the optimum.
    std::printf("\n--- S5 combined (persistent skin-list + Verlet + propagating repair); skin in cell-sizes ---\n");
    std::printf("%-14s %6s %8s %8s %8s %10s %10s %9s\n", "disp", "skin", "ms/step", "rebld%", "touch%",
                "meanRelV", "maxRelV", "mism>1e-3");
    const int fallbackN5 = (int)(0.30 * N);
    for (real_t disp : disps) {
      const real_t scale = scaleFor(disp);
      for (real_t skin : {(real_t)0.05, (real_t)0.1, (real_t)0.2, (real_t)0.4}) {
        P_at(scale, 0);
        rebuildAll();
        Kokkos::deep_copy(xRef, pos);
        double t = 0, meanRel = 0, maxRel = 0;
        long mism = 0, touch = 0, rebuilds = 0;
        for (int s = 1; s <= nSteps; ++s) {
          P_at(scale, s);
          auto a = clk::now();
          if (maxDispCells() > real_t(0.5) * skin) {
            rebuildAll();  // grid + re-gather skin lists + full rebuild (worklist recreation)
            Kokkos::deep_copy(xRef, pos);
            ++rebuilds;
            touch += N;
          } else {
            reevalFlagD2();  // reeval from store + D2 flag; NO grid
            Kokkos::deep_copy(active, flag);
            for (int sweep = 0; sweep < 12; ++sweep) {
              const int cnt = compact(active);
              if (cnt == 0) break;
              if (cnt > fallbackN5) {
                rebuildAll();
                Kokkos::deep_copy(xRef, pos);
                ++rebuilds;
                touch += N;
                break;
              }
              touch += cnt;
              Kokkos::deep_copy(changed, 0);
              repairFromList(cnt, true);   // clip the STORED skin list — no grid, no gather
              Kokkos::deep_copy(nextA, 0);
              propagateWork(cnt, nextA);
              Kokkos::deep_copy(active, nextA);
            }
          }
          t += secs(a, clk::now());
          if (s == nSteps) { buildOracle(); accuracy(meanRel, maxRel, mism); }
        }
        std::printf("%-10.3f %6.2f %8.2f %8.1f %8.1f %10.2e %10.2e %9ld\n", (double)disp, (double)skin,
                    1e3 * t / nSteps, 100.0 * rebuilds / nSteps, 100.0 * touch / ((double)nSteps * N),
                    meanRel, maxRel, mism);
      }
    }

    // space-filling sanity for the chosen sw (oracle Σvol should ≈ L^3)
    P_at(scaleFor(disps[0]), 0);
    grid.build(pos, N);
    gatherAll();
    {
      auto P = pos;
      auto cId = candId;
      auto nc = ncand;
      Kokkos::View<int*, Mem> diag("diag", 4);  // [overflow, empty, npSum, ntSum]
      Kokkos::deep_copy(diag, 0);
      auto Vo = volOra;
      Kokkos::parallel_for(
          "diag", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
            Cell c = buildFromCand(i, P.data(), cId.data(), nc(i), L);
            Vo(i) = c.overflow ? real_t(0) : c.volumePerVertex();
            if (c.overflow) Kokkos::atomic_inc(&diag(0));
            if (c.empty()) Kokkos::atomic_inc(&diag(1));
            Kokkos::atomic_add(&diag(2), c.np);
            Kokkos::atomic_add(&diag(3), c.nt);
          });
      Kokkos::fence();
      auto hd = Kokkos::create_mirror_view(diag);
      Kokkos::deep_copy(hd, diag);
      double sv = 0;
      auto ho = Kokkos::create_mirror_view(volOra);
      Kokkos::deep_copy(ho, volOra);
      for (int i = 0; i < N; ++i) sv += ho(i);
      std::printf("\n[check] oracle Σvol=%.6f (~%.6f)  overflow=%d empty=%d  meanNp=%.1f meanNt=%.1f"
                  "  vol0=%.3e\n", sv, (double)L * L * L, hd(0), hd(1), (double)hd(2) / N,
                  (double)hd(3) / N, (double)ho(0));
    }
  }
  Kokkos::finalize();
  return 0;
}
