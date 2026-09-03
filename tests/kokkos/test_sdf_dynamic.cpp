/**
 * @file test_sdf_dynamic.cpp
 * @brief Rung A0 gate (Voronoi methods plan §3): SDF solids on the MOVING-POINT path.
 *
 * A periodic point set advanced ballistically through a CSG scene (a sphere ∪ a torus, straight
 * from the shared core vocabulary via SdfScene) is maintained by MovingTessellation<…, SdfScene>
 * — cold SDF build once, then the two-pass repair + the SDF boundary watch each step — and
 * compared, every few steps, against a fresh cold buildTessellation with the SAME scene at the SAME
 * positions (the oracle). Cells slide along the curved walls, enter the solid (become empty) and
 * re-emerge; a wall-clipped cell's planes are restored from the WallStore between re-gathers.
 *
 * Gates (exact wall mode, the default):
 *   - per-cell volume: max |V_repair − V_cold| / V_cold ≤ 10x the WALL-FREE repair's own figure
 *     on the same seeds + motion (the repair is exact only to its certificate tolerance);
 *   - emptiness agrees cell-for-cell (seed in the solid ⇔ volume 0 on both paths);
 *   - the boundary watch fires (wallFlagged > 0) and the repair never falls back to a rebuild;
 *   - the fluid volume Σ V agrees with the cold build to 10x the wall-free figure.
 * The skin wall mode (wallExact=false) is reported, only bounded: it trades exactness for fewer
 * wall re-gathers (stale tangent planes until the seed moved > wallSkin).
 *
 * MEASURED (host-openmp, 2026-09-03): with the seed-local certificate alone the wall-free repair
 * was exact only to ~1.6e-3 per cell over 400 steps (a face-gain blind spot, ~250 missed
 * neighbour relations per step); with the near-miss certificate (MovingTessellation::useNearMiss)
 * it is 5.5e-11 at tol=1e-4·spacing and 1.2e-15 at tol=1e-7, and the SDF path 5.9e-8. Before the
 * chord-plane fix in clipCellAgainstSdf the cold build produced silent zero-volume overflow cells
 * at curved walls and the two paths disagreed by O(1).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <random>
#include <type_traits>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tessellator.hpp"
#include "peclet/voro/topology_store.hpp"

using Real = double;
using Mem = peclet::core::MemSpace;
using Exec = peclet::core::ExecSpace;
using namespace peclet::core::geom;
using Scene = peclet::voro::SdfScene<Real>;

struct SceneHold {
  Kokkos::View<ShapeNode<Real>*, Mem> nodes;
  Kokkos::View<GridDesc<Real>*, Mem> grids;
  Kokkos::View<float*, Mem> pool;
};

// union( sphere r=0.2 @ (0.5,0.5,0.5) , torus R=0.22 r=0.07 @ (0.2,0.25,0.75) )
static Scene makeScene(SceneHold& h) {
  h.nodes = Kokkos::View<ShapeNode<Real>*, Mem>("nodes", 3);
  auto hn = Kokkos::create_mirror_view(h.nodes);
  hn(0).kind = kUnion;
  hn(0).aux0 = 1;
  hn(0).aux1 = 2;
  hn(1).kind = kSphere;
  hn(1).params[0] = 0.2;
  hn(1).transform.translation = peclet::core::Vec3<Real>{0.5, 0.5, 0.5};
  hn(2).kind = kTorus;
  hn(2).params[0] = 0.22;
  hn(2).params[1] = 0.07;
  hn(2).transform.translation = peclet::core::Vec3<Real>{0.2, 0.25, 0.75};
  Kokkos::deep_copy(h.nodes, hn);
  h.grids = Kokkos::View<GridDesc<Real>*, Mem>("grids", 1);
  h.pool = Kokkos::View<float*, Mem>("pool", 1);
  return Scene{h.nodes, h.grids, h.pool, 3, 0, Real(1e-5)};
}

static Real wrap1(Real x, Real L) {
  x -= L * std::floor(x / L);
  if (x >= L)
    x -= L;
  if (x < 0)
    x += L;
  return x;
}

struct CaseResult {
  double maxRelV = 0, sumRel = 0;
  long emptyMismatch = 0, faceMismatch = 0, wallFlag = 0, fellBack = 0, nWallCells = 0, nEmpty = 0,
       checks = 0;
};

template <class MT>
static int faceCountFromStore(MT& mt, Kokkos::View<int*, Mem>& out) {
  using Cell = peclet::voro::ConvexCell<Real, 64, 112, false>;
  auto st = mt.store;
  auto O = out;
  const Real Lx = mt.L[0], Ly = mt.L[1], Lz = mt.L[2];
  Kokkos::parallel_for(
      "faces", Kokkos::RangePolicy<Exec>(0, mt.N), KOKKOS_LAMBDA(int i) {
        Cell c;
        st.load(i, c, Lx, Ly, Lz);
        int nf = 0;
        for (int k = 0; k < c.np; ++k) {
          int cnt = 0;
          for (int t = 0; t < c.nt; ++t)
            if (c.alive[t] && (c.t0[t] == k || c.t1[t] == k || c.t2[t] == k))
              ++cnt;
          if (cnt >= 3)
            ++nf;
        }
        O(i) = nf;
      });
  Kokkos::fence();
  return 0;
}

template <class Sdf>
static CaseResult runCase(const Sdf& sdf, const std::vector<Real>& p0, const std::vector<Real>& vel,
                          int N, const Real L[3], Real spacing, bool exact, Real wallSkinFrac,
                          Real dispFrac, int nSteps, int checkEvery, Real tolFrac = Real(1e-4)) {
  using MT = peclet::voro::MovingTessellation<Real, 64, 112, false, Sdf>;
  constexpr bool hasSdf = !std::is_same_v<Sdf, peclet::voro::NoSdf>;
  constexpr bool hostAcc = Kokkos::SpaceAccessibility<Kokkos::HostSpace, Mem>::accessible;
  CaseResult r;
  MT mt;
  mt.sdf = sdf;
  mt.wallExact = exact;
  mt.wallSkin = wallSkinFrac * spacing;
  mt.alloc(N, L, tolFrac * spacing, Real(0.25) * spacing, 4, N);
  Kokkos::View<Real*, Mem> dPos(Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing),
                                (size_t)N * 3);
  auto hPos = Kokkos::create_mirror_view(dPos);
  auto place = [&](int s) {
    const Real sc = dispFrac * spacing * s;
    for (int i = 0; i < 3 * N; ++i)
      hPos(i) = wrap1(p0[i] + vel[i] * sc, L[i % 3]);
    Kokkos::deep_copy(dPos, hPos);
  };
  place(0);
  mt.rebuild(dPos);
  Kokkos::View<int*, Mem> faces("faces", N);
  int dbgShown = 0;
  Kokkos::View<Real*, Mem> wd;
  Kokkos::View<long*, Mem> gd;
  for (int s = 1; s <= nSteps; ++s) {
    place(s);
    auto st = mt.step(dPos);
    r.wallFlag += st.wallFlagged;
    r.fellBack += st.fellBack ? 1 : 0;
    if (s % checkEvery != 0 && s != nSteps)
      continue;
    auto cold = peclet::voro::buildTessellation<Real, false, Sdf>(dPos, wd, N, L, 4, N, gd, sdf,
                                                                  /*withForceGeom=*/false);
    faceCountFromStore(mt, faces);
    auto ov = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold.view.cellVolume);
    auto ofc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold.view.cellFacetCount);
    auto rv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.vol);
    auto rf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, faces);
    std::vector<int> wcv(N, 0);
    if constexpr (hasSdf) {
      auto wc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.wall.cnt);
      for (int i = 0; i < N; ++i)
        wcv[i] = wc(i);
    }
    if (std::getenv("VORO_SDF_DEBUG") && dbgShown < 24) {
      auto ost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold.status);
      auto snp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.store.np);
      auto snt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.store.nt);
      auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dPos);
      auto off =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold.view.cellFacetOffset);
      auto fnb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold.view.facetNeighbor);
      // isolation: a fresh subset gather of the cell off a grid at these positions (no repair)
      auto grid = peclet::voro::buildTessGrid<Real, false>(dPos, wd, N, L, 4, N, gd);
      Kokkos::View<int*, Mem> gNp("gNp", N), gNt("gNt", N), gPnbr("gPnbr", (size_t)N * 64);
      Kokkos::View<unsigned*, Mem> gTri("gTri", (size_t)N * 112);
      Kokkos::View<Real*, Mem> gVol("gVol", N);
      Kokkos::View<int*, Mem> one("one", 1);
      for (int i = 0; i < N && dbgShown < 24; ++i) {
        const double o = ov(i), v = rv(i);
        const bool bad = (o > 0 && std::fabs(v - o) / o > 1e-3) || (o == 0 && v != 0);
        if (!bad)
          continue;
        Kokkos::deep_copy(one, i);
        peclet::voro::subsetGather<Real, false, false, Sdf>(grid, one, 1, gNp, gNt, gPnbr, gTri,
                                                            gVol, {}, sdf, false);
        double gv = 0;
        Kokkos::deep_copy(gv, Kokkos::subview(gVol, i));
        // round trip: gather with a wall store, reload + reeval, volume again
        double rt = -1;
        int rtNp = -1, rtWall = -1;
        if constexpr (hasSdf) {
          peclet::voro::WallStore<Real> ws;
          ws.alloc(N);
          peclet::voro::subsetGather<Real, false, false, Sdf>(grid, one, 1, gNp, gNt, gPnbr, gTri,
                                                              gVol, {}, sdf, false, ws);
          peclet::voro::TopologyStore<64, 112> ts;
          ts.np = gNp;
          ts.nt = gNt;
          ts.pnbr = gPnbr;
          ts.tri = gTri;
          Kokkos::View<Real*, Mem> out("rt", 3);
          const Real Lx = L[0], Ly = L[1], Lz = L[2];
          Kokkos::parallel_for(
              "rt", 1, KOKKOS_LAMBDA(int) {
                peclet::voro::ConvexCell<Real, 64, 112, false> c;
                ts.load(i, c, Lx, Ly, Lz);
                const bool ok = ws.load(i, c, Real(0), Real(0), Real(0));
                c.reevalGeometry(dPos(3 * i), dPos(3 * i + 1), dPos(3 * i + 2), dPos.data(), Lx);
                out(0) = c.volumePerVertex();
                out(1) = c.np;
                out(2) = ok ? ws.cnt(i) : -ws.cnt(i);
              });
          Kokkos::fence();
          auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
          rt = ho(0);
          rtNp = (int)ho(1);
          rtWall = (int)ho(2);
        }
        int coldWall = 0;
        for (int f = off(i); f < off(i) + ofc(i); ++f)
          if (fnb(f) == peclet::voro::kBoundaryFacet)
            ++coldWall;
        double phi = 0;
        if constexpr (hasSdf && hostAcc)
          phi = sdf.eval(hp(3 * i), hp(3 * i + 1), hp(3 * i + 2));
        // host replay of the SDF clip on the wall-free cell (overflow cells only): where does it
        // overflow?
        if constexpr (hasSdf && hostAcc) {
          if (ost(i) & 1) {
            peclet::voro::subsetGather<Real, false, false, peclet::voro::NoSdf>(
                grid, one, 1, gNp, gNt, gPnbr, gTri, gVol, {}, peclet::voro::NoSdf{}, false);
            Kokkos::fence();
            peclet::voro::TopologyStore<64, 112> ts;
            ts.np = gNp;
            ts.nt = gNt;
            ts.pnbr = gPnbr;
            ts.tri = gTri;
            peclet::voro::ConvexCell<Real, 64, 112, false> c;
            ts.load(i, c, L[0], L[1], L[2]);
            c.reevalGeometry(hp(3 * i), hp(3 * i + 1), hp(3 * i + 2), hp.data(), L[0]);
            std::printf("      replay: voronoi cell np=%d nt=%d faces=%d vol=%.3e\n", c.np, c.nt,
                        c.countFaces(), c.volumePerVertex());
            const int removed = c.compactPlanes();
            std::printf("      replay: after compactPlanes np=%d (removed %d) vol=%.3e\n", c.np,
                        removed, c.volumePerVertex());
            const Real seed[3] = {hp(3 * i), hp(3 * i + 1), hp(3 * i + 2)};
            for (int it = 0; it < 24 && !c.overflow; ++it) {
              // most violating vertex
              Real probe[3] = {seed[0], seed[1], seed[2]};
              Real probePhi = sdf.eval(seed[0], seed[1], seed[2]);
              if (it > 0) {
                bool found = false;
                for (int t = 0; t < c.nt; ++t) {
                  if (!c.alive[t])
                    continue;
                  Real x = seed[0] + c.vx[t], y = seed[1] + c.vy[t], z = seed[2] + c.vz[t];
                  Real ph = sdf.eval(x, y, z);
                  if (!found || ph < probePhi) {
                    probe[0] = x;
                    probe[1] = y;
                    probe[2] = z;
                    probePhi = ph;
                    found = true;
                  }
                }
                if (!found || probePhi >= -1e-8) {
                  std::printf("      replay: done at cut %d\n", it);
                  break;
                }
              }
              Real g[3];
              peclet::voro::sdfGradient<Real>(sdf, probe[0], probe[1], probe[2], g);
              const Real gsq = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
              const Real ph = sdf.eval(probe[0], probe[1], probe[2]);
              Real surf[3], normal[3];
              for (int k = 0; k < 3; ++k) {
                surf[k] = probe[k] - ph * g[k] / gsq;
                normal[k] = g[k] / std::sqrt(gsq);
              }
              Real pv[3] = {-normal[0], -normal[1], -normal[2]};
              Real off = pv[0] * (surf[0] - seed[0]) + pv[1] * (surf[1] - seed[1]) +
                         pv[2] * (surf[2] - seed[2]);
              const bool cut = c.clip(pv, off, peclet::voro::kBoundaryFacet);
              std::printf(
                  "      replay: cut %2d probePhi=%.2e off=%.3e committed=%d np=%d nt=%d alive=%d "
                  "overflow=%d\n",
                  it, probePhi, off, cut ? 1 : 0, c.np, c.nt,
                  c.nt - (int)std::count(c.alive, c.alive + c.nt, false), c.overflow ? 1 : 0);
            }
          }
        }
        std::printf(
            "    step %4d cell %6d: cold=%.6e rep=%.6e gather=%.6e roundtrip=%.6e (np %d, "
            "wall %d) coldFaces=%d(wall %d) repFaces=%d status=%d np=%d nt=%d wallRec=%d "
            "phi=%.3e\n",
            s, i, o, v, gv, rt, rtNp, rtWall, ofc(i), coldWall, rf(i), ost(i), snp(i), snt(i),
            wcv[i], phi);
        ++dbgShown;
      }
    }
    double so = 0, sr = 0;
    long nEmpty = 0, nWall = 0;
    for (int i = 0; i < N; ++i) {
      const double o = ov(i), v = rv(i);
      so += o;
      sr += v;
      if (o > 0) {
        r.maxRelV = std::max(r.maxRelV, std::fabs(v - o) / o);
      } else {
        ++nEmpty;
        if (v != 0)
          ++r.emptyMismatch;
      }
      if (ofc(i) != rf(i))
        ++r.faceMismatch;
      if (wcv[i] > 0)
        ++nWall;
    }
    r.sumRel = std::max(r.sumRel, std::fabs(sr - so) / so);
    r.nEmpty = nEmpty;
    r.nWallCells = nWall;
    ++r.checks;
  }
  return r;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 12000;
    const Real L[3] = {1, 1, 1};
    const Real spacing = std::cbrt((L[0] * L[1] * L[2]) / N);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<Real> U(0.0, 1.0);
    std::normal_distribution<Real> G(0.0, 1.0);
    std::vector<Real> p0(3 * N), vel(3 * N);
    for (auto& x : p0)
      x = U(rng);
    for (auto& v : vel)
      v = G(rng);
    SceneHold hold;
    Scene sdf = makeScene(hold);

    // The repair is exact only up to its certificate tolerance (tol = tolFrac·spacing): a stale
    // face may survive while every vertex pokes past it by < tol. So the SDF path is gated
    // AGAINST THE WALL-FREE BASELINE at the same tolerance (same seeds, same motion), and both are
    // shown to tighten together as tol shrinks.
    struct Spec {
      const char* name;
      bool sdf, exact;
      Real skinFrac, disp, tolFrac;
      int steps, every;
    } specs[] = {
        {"voronoi 2e-3 x400 tol1e-4", false, true, 0.0, 2e-3, 1e-4, 400, 4},
        {"sdf     2e-3 x400 tol1e-4", true, true, 0.0, 2e-3, 1e-4, 400, 4},
        {"voronoi 2e-3 x400 tol1e-7", false, true, 0.0, 2e-3, 1e-7, 400, 4},
        {"sdf     2e-3 x400 tol1e-7", true, true, 0.0, 2e-3, 1e-7, 400, 4},
        {"sdf     1e-2 x100 tol1e-7", true, true, 0.0, 1e-2, 1e-7, 100, 2},
        {"sdf     5e-2 x40  tol1e-7", true, true, 0.0, 5e-2, 1e-7, 40, 2},
        {"sdf-skin 2e-3 x200 tol1e-7", true, false, 0.05, 2e-3, 1e-7, 200, 4},
    };
    std::printf("=== A0: SDF solids on the moving-point path (N=%d, sphere ∪ torus scene) ===\n",
                N);
    std::printf("%-27s %10s %10s %8s %8s %9s %8s %8s %6s\n", "case", "maxRelV", "sumRel", "emptyMM",
                "faceMM", "wallFlag", "wallCel", "empty", "fellbk");
    double baseline[2] = {0, 0}, baseSum[2] = {0, 0};  // wall-free maxRelV / sumRel per tol
    for (const auto& sp : specs) {
      CaseResult r = sp.sdf ? runCase(sdf, p0, vel, N, L, spacing, sp.exact, sp.skinFrac, sp.disp,
                                      sp.steps, sp.every, sp.tolFrac)
                            : runCase(peclet::voro::NoSdf{}, p0, vel, N, L, spacing, sp.exact,
                                      sp.skinFrac, sp.disp, sp.steps, sp.every, sp.tolFrac);
      std::printf("%-27s %10.2e %10.2e %8ld %8ld %9ld %8ld %8ld %6ld\n", sp.name, r.maxRelV,
                  r.sumRel, r.emptyMismatch, r.faceMismatch, r.wallFlag, r.nWallCells, r.nEmpty,
                  r.fellBack);
      const int tb = sp.tolFrac < 1e-5 ? 1 : 0;
      if (!sp.sdf) {
        baseline[tb] = r.maxRelV;
        baseSum[tb] = r.sumRel;
        continue;
      }
      if (sp.exact) {
        // 10x the wall-free figure, floored at 1e-6: with the near-miss certificate the wall-free
        // repair is exact to ~1e-11..1e-15 and the SDF path to ~6e-8 (the wall re-clip's own
        // round-off chain), so a purely relative gate would compare two round-off floors.
        const double allow = std::max(10.0 * baseline[tb], 1e-6);
        if (!(r.maxRelV <= allow)) {
          std::printf("  FAIL: maxRelV %.2e > 10x wall-free baseline %.2e\n", r.maxRelV, allow);
          bad = 1;
        }
        if (r.emptyMismatch != 0) {
          std::printf("  FAIL: emptiness mismatch\n");
          bad = 1;
        }
        const double allowSum = std::max(10.0 * baseSum[tb], 1e-10);
        if (!(r.sumRel <= allowSum)) {
          std::printf("  FAIL: fluid volume %.2e > 10x wall-free baseline %.2e\n", r.sumRel,
                      allowSum);
          bad = 1;
        }
        if (r.wallFlag <= 0) {
          std::printf("  FAIL: boundary watch never fired\n");
          bad = 1;
        }
        if (r.fellBack != 0) {
          std::printf("  FAIL: repair fell back to rebuild\n");
          bad = 1;
        }
        if (r.nWallCells <= 0) {
          std::printf("  FAIL: no wall-clipped cells\n");
          bad = 1;
        }
      } else {
        // skin mode keeps a moved wall cell's stale tangent planes until it moved > wallSkin: not
        // exact by construction (a plane anchored to a vertex that moved), only bounded.
        if (!(r.maxRelV < 1.0)) {
          std::printf("  FAIL: skin mode error unbounded\n");
          bad = 1;
        }
        if (r.emptyMismatch != 0) {
          std::printf("  FAIL: skin mode emptiness mismatch\n");
          bad = 1;
        }
      }
    }
  }
  Kokkos::finalize();
  std::printf(bad ? "VORO-SDF-DYNAMIC FAIL\n" : "VORO-SDF-DYNAMIC OK\n");
  return bad;
}
