/**
 * @file bench_report.cpp
 * \brief Machine-readable (CSV) performance + memory + accuracy benchmark for the report.
 *
 * Two modes (uniform Poisson point set in a periodic unit box):
 *
 *  --cold   : sweep system size N. For each N time (a) the bare production cold build
 *             (buildTessellation, topology+volume, NO resident store) and (b) the
 *             repair-ready cold build (MovingTessellation::rebuild — emits the resident
 *             topology store + the poke4 cert planes). Reports throughput (Mcells/s) for
 *             both, the per-cell resident-store memory, and the space-filling volume error
 *             (|Σvol/boxVol − 1|). CSV: N,mcells_plain,mcells_repair,store_bytes_per_cell,vol_err
 *
 *  --repair : fixed N, sweep the per-step displacement δ/h from tiny to large (→ full
 *             rebuild). Drives MovingTessellation over a ballistic trajectory; reports the
 *             repair throughput (Mcells/s), the cold-build throughput for reference, the
 *             rebuilt fractions (Pass-1/Pass-2/gate-rebuild), the cert-flagged count, the
 *             space-filling volume error, and the worst-cell volume error vs a fresh cold
 *             oracle. CSV: disp,cold_mcells,repair_mcells,speedup,p1pct,p2pct,gatepct,
 *                         flagpct,vol_err,max_rel_v,missed
 *
 * Env: VORF_NS="n1 n2 ..." (cold N list), VORF_DISPS, VORF_TOL, VORF_BRUTECERT, VORF_NOGATE.
 * Build FP32 with -DCC_FLOAT. Run: ./bench_report --cold | ./bench_report --repair [N] [nSteps]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/dynamic_validate.hpp"  // OracleDiff, compareVolumes, compareNeighbours
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"
#include "peclet/voro/topology_store.hpp"

#ifdef CC_FLOAT
using real_t = float;
#else
using real_t = double;
#endif
using Exec = peclet::core::ExecSpace;
using Mem = peclet::core::MemSpace;
static constexpr int CMAXP = 64, CMAXT = 112;
using Store = peclet::voro::TopologyStore<CMAXP, CMAXT>;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

static void uniform(int N, real_t L, std::mt19937& rng, std::vector<real_t>& x) {
  std::uniform_real_distribution<real_t> U(0, 1);
  x.assign(3 * N, 0);
  for (auto& v : x)
    v = L * U(rng);
}

// Sum of cell volumes (space-filling: should equal boxVol exactly).
static double sumVol(const Kokkos::View<real_t*, Mem>& vol, int N) {
  double s = 0;
  Kokkos::parallel_reduce(
      "sumvol", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(int i, double& a) { a += (double)vol(i); }, s);
  return s;
}

// Resident "repair infrastructure" bytes per cell (the topology store + poke4 + per-cell scratch
// the MovingTessellation keeps; excludes the transient grid/facet over-buffers and the position
// arrays).
static double storeBytesPerCell() {
  const double pnbr = CMAXP * sizeof(int);                 // neighbour id / plane
  const double tri = CMAXT * sizeof(unsigned);             // packed dual triangles
  const double poke4 = CMAXT * 4 * sizeof(unsigned char);  // cert plane set
  const double npnt = 2 * sizeof(int);                     // np, nt
  const double volxref = (1 + 3) * sizeof(real_t);         // vol + Verlet reference
  const double scratch =
      (4 + 3 + 1) * sizeof(int) + 8 * sizeof(int) + sizeof(int);  // masks/wl/extra
  return pnbr + tri + poke4 + npnt + volxref + scratch;
}

static int coldSweep(real_t L) {
  std::vector<int> Ns = {20000, 50000, 100000, 200000, 500000, 1000000};
  if (const char* e = std::getenv("VORF_NS")) {
    Ns.clear();
    std::stringstream ss(e);
    int n;
    while (ss >> n)
      Ns.push_back(n);
  }
  const real_t Larr[3] = {L, L, L};
  std::printf("# cold-build sweep  backend=%s prec=%s MAXP=%d MAXT=%d\n",
              Kokkos::DefaultExecutionSpace::name(), sizeof(real_t) == 4 ? "fp32" : "fp64", CMAXP,
              CMAXT);
  std::printf("N,mcells_plain,mcells_repair,store_bytes_per_cell,vol_err\n");
  Kokkos::View<real_t*, Mem> wd;
  Kokkos::View<long*, Mem> gd;
  for (int N : Ns) {
    std::mt19937 rng(1234);
    std::vector<real_t> xh;
    uniform(N, L, rng, xh);
    Kokkos::View<real_t*, Mem> pos("pos", 3 * N);
    Kokkos::deep_copy(pos, Kokkos::View<const real_t*, Kokkos::HostSpace>(xh.data(), 3 * N));
    const double boxVol = (double)L * L * L;
    const int reps = N <= 200000 ? 5 : 3;

    // (a) bare production cold build (topology + volume; no resident store).
    double tPlain = 1e30;
    double volErr = 0;
    for (int r = 0; r < reps; ++r) {
      Kokkos::fence();
      auto a = clk::now();
      auto res = peclet::voro::buildTessellation<real_t, false>(pos, wd, N, Larr, 4, N, gd,
                                                                peclet::voro::NoSdf{}, false);
      Kokkos::fence();
      tPlain = std::min(tPlain, secs(a, clk::now()));
      if (r == 0)
        volErr = std::abs(sumVol(res.view.cellVolume, N) / boxVol - 1.0);
    }
    // (b) repair-ready cold build: MovingTessellation::rebuild (emits store + poke4).
    peclet::voro::MovingTessellation<real_t, CMAXP, CMAXT> mt;
    const real_t spacing = (real_t)std::cbrt(boxVol / N);
    mt.alloc(N, Larr, real_t(1e-4) * spacing, real_t(0.25) * spacing, 4, N);
    double tRepair = 1e30;
    for (int r = 0; r < reps; ++r) {
      Kokkos::fence();
      auto a = clk::now();
      mt.rebuild(pos);
      Kokkos::fence();
      tRepair = std::min(tRepair, secs(a, clk::now()));
    }
    std::printf("%d,%.4f,%.4f,%.1f,%.3e\n", N, 1e-6 * N / tPlain, 1e-6 * N / tRepair,
                storeBytesPerCell(), volErr);
    std::fflush(stdout);
  }
  return 0;
}

static int repairSweep(int N, int nSteps, real_t L) {
  const real_t Larr[3] = {L, L, L};
  const double boxVol = (double)L * L * L;
  const real_t spacing = (real_t)std::cbrt(boxVol / N);
  const double tolMult = std::getenv("VORF_TOL") ? std::atof(std::getenv("VORF_TOL"))
                                                 : (sizeof(real_t) == 4 ? 2e-3 : 1e-4);
  const real_t tol = (real_t)tolMult * spacing, skin = real_t(0.25) * spacing;
  const bool bruteCert = std::getenv("VORF_BRUTECERT") != nullptr;
  const bool noGate = std::getenv("VORF_NOGATE") != nullptr;
  std::vector<real_t> disps = {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1};
  if (const char* e = std::getenv("VORF_DISPS")) {
    disps.clear();
    std::stringstream ss(e);
    double d;
    while (ss >> d)
      disps.push_back((real_t)d);
  }
  std::printf("# repair sweep  backend=%s prec=%s N=%d nSteps=%d cert=%s gate=%s\n",
              Kokkos::DefaultExecutionSpace::name(), sizeof(real_t) == 4 ? "fp32" : "fp64", N,
              nSteps, bruteCert ? "brute" : "local", noGate ? "off" : "on");
  std::printf(
      "disp,cold_mcells,repair_mcells,speedup,p1pct,p2pct,gatepct,flagpct,vol_err,max_rel_v,"
      "missed\n");

  Kokkos::View<real_t*, Mem> x0d("x0", 3 * N), veld("vel", 3 * N), pos("pos", 3 * N);
  Kokkos::View<real_t*, Mem> wd;
  Kokkos::View<long*, Mem> gd;
  Store oraStore;
  oraStore.alloc(N);
  std::mt19937 rng(7000);
  std::vector<real_t> x0h;
  uniform(N, L, rng, x0h);
  std::vector<real_t> velh(3 * N);
  std::normal_distribution<real_t> Ng(0, 1);
  for (auto& v : velh)
    v = Ng(rng);
  Kokkos::deep_copy(x0d, Kokkos::View<const real_t*, Kokkos::HostSpace>(x0h.data(), 3 * N));
  Kokkos::deep_copy(veld, Kokkos::View<const real_t*, Kokkos::HostSpace>(velh.data(), 3 * N));
  auto advance = [&](real_t scale, int step) {
    auto Pd = pos;
    auto X0 = x0d;
    auto V = veld;
    const real_t s = scale * step, Ll = L;
    Kokkos::parallel_for(
        "adv", Kokkos::RangePolicy<Exec>(0, 3 * N), KOKKOS_LAMBDA(int i) {
          real_t v = X0(i) + V(i) * s;
          v -= Ll * Kokkos::floor(v / Ll);
          Pd(i) = v;
        });
  };

  for (real_t disp : disps) {
    const real_t scale = disp * spacing;
    // cold-build reference throughput (a few steps suffice — it barely depends on disp)
    const int coldSteps = std::min(nSteps, 4);
    advance(scale, 0);
    double tCold = 0;
    for (int s = 1; s <= coldSteps; ++s) {
      advance(scale, s);
      Kokkos::fence();
      auto a = clk::now();
      auto r = peclet::voro::buildTessellation<real_t, false>(pos, wd, N, Larr, 4, N, gd,
                                                              peclet::voro::NoSdf{}, false);
      Kokkos::fence();
      tCold += secs(a, clk::now());
      (void)r;
    }
    tCold *= (double)nSteps / coldSteps;  // scale to the per-step×nSteps basis used below
    // repair over the same trajectory
    peclet::voro::MovingTessellation<real_t, CMAXP, CMAXT> mt;
    mt.localCert = !bruteCert;
    mt.useGate = !noGate;
    mt.alloc(N, Larr, tol, skin, 4, N);
    advance(scale, 0);
    mt.rebuild(pos);
    double tRep = 0;
    long p1 = 0, p2 = 0, gate = 0, flag = 0;
    for (int s = 1; s <= nSteps; ++s) {
      advance(scale, s);
      Kokkos::fence();
      auto a = clk::now();
      auto st = mt.step(pos);
      Kokkos::fence();
      tRep += secs(a, clk::now());
      p1 += st.pass1;
      p2 += st.pass2;
      flag += st.pass1Raw;
      gate += (st.route == peclet::voro::RepairStats::kRebuildGate) ? 1 : 0;
    }
    // accuracy vs a fresh cold oracle at the final positions
    auto ora = peclet::voro::buildTessellation<real_t, false>(
        pos, wd, N, Larr, 4, N, gd, peclet::voro::NoSdf{}, false, -1, oraStore.np, oraStore.nt,
        oraStore.pnbr, oraStore.tri);
    peclet::voro::OracleDiff d;
    peclet::voro::compareVolumes(mt.vol, ora.view, d);
    peclet::voro::compareNeighbours(mt.store.np, mt.store.nt, mt.store.pnbr, mt.store.tri, CMAXP,
                                    CMAXT, ora.view, d);
    const double volErr = std::abs(sumVol(mt.vol, N) / boxVol - 1.0);
    const double coldMc = 1e-6 * N * nSteps / tCold, repMc = 1e-6 * N * nSteps / tRep;
    std::printf("%.4f,%.3f,%.3f,%.2f,%.3f,%.3f,%.1f,%.3f,%.3e,%.3e,%ld\n", (double)disp, coldMc,
                repMc, repMc / coldMc, 100.0 * p1 / ((double)nSteps * N),
                100.0 * p2 / ((double)nSteps * N), 100.0 * gate / nSteps,
                100.0 * flag / ((double)nSteps * N), volErr, d.maxVolRelErr, d.missedNbr);
    std::fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const real_t L = 1.0;
    const char* mode = argc > 1 ? argv[1] : "--cold";
    if (std::strcmp(mode, "--cold") == 0) {
      rc = coldSweep(L);
    } else {
      const int N = argc > 2 ? std::atoi(argv[2]) : 100000;
      const int nSteps = argc > 3 ? std::atoi(argv[3]) : 20;
      rc = repairSweep(N, nSteps, L);
    }
  }
  Kokkos::finalize();
  return rc;
}
