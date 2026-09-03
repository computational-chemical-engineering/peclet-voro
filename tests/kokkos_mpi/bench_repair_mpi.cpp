/**
 * @file bench_repair_mpi.cpp
 * \brief Distributed (MPI) two-pass repair vs distributed cold build, as a function of
 * dimensionless displacement — the "serial + MPI, one process per core" configuration (run
 * OMP_NUM_THREADS=1).
 *
 * Each rank owns an ORB block (core, via VoronoiHalo). The distributed COLD build, per
 * step: gather every seed within rcut of the block (the MPI halo exchange), then tessellate the
 * owned cells over owned+ghost. The distributed REPAIR keeps a resident topology of the owned cells
 * and, while no owned seed has moved beyond the Verlet skin, only *refreshes* the ghost positions
 * on the established halo topology (VoronoiHalo::refreshPositions — same comm pattern, no
 * re-decomposition, combined order stable so the resident neighbour indices stay valid) and runs
 * the local two-pass gather repair (MovingTessellation::step over the owned cells). On a skin trip
 * it re-gathers + cold-rebuilds (the distributed fallback). The MPI halo exchange is common to both
 * paths; the difference is build vs reeval+repair compute per rank.
 *
 * Exactness: at the final step the owned-cell volumes from the repair are compared to a fresh cold
 * build over the SAME combined positions (so same ordering) — the distributed analogue of the
 * single-domain oracle. Space-filling (Σ owned volume == box) is also checked.
 *
 * Run:  OMP_NUM_THREADS=1 mpirun -np <R> --bind-to core ./bench_repair_mpi [N_global] [nSteps]
 * [--sdf]
 *
 * --sdf (rung A0): the same run through an SDF solid (a sphere ∪ torus core scene, replicated on
 * every rank as an SdfScene) — the distributed driver's tessellation carries the wall planes and
 * the boundary watch. Two extra gates: the distributed repair matches the distributed cold SDF
 * build (as above), AND the owned cells match a SINGLE-RANK cold SDF build of the whole point set
 * (every rank holds the global seeds), keyed by global id — the "np = 1, 2, 4 identical to single
 * rank" gate.
 */
#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/geom/scene.hpp"
#include "peclet/voro/mpi/distributed_moving.hpp"
#include "peclet/voro/mpi/voronoi_halo.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;
using Mem = peclet::core::MemSpace;
static constexpr int CMAXP = 64, CMAXT = 112;

static real_t wrap1(real_t x, real_t L) {
  x -= L * std::floor(x / L);
  if (x >= L)
    x -= L;
  if (x < 0)
    x += L;
  return x;
}

// Upload a combined (owned+ghost) Vec3 list, wrapped to [0,L), into a device 3*n view.
static Kokkos::View<real_t*, Mem> uploadCombined(const std::vector<Vec3>& p, const Vec3& L) {
  const int n = (int)p.size();
  Kokkos::View<real_t*, Mem> d(Kokkos::view_alloc(std::string("dPos"), Kokkos::WithoutInitializing),
                               (size_t)n * 3);
  auto h = Kokkos::create_mirror_view(d);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k)
      h(3 * i + k) = wrap1(p[i][k], L[k]);
  Kokkos::deep_copy(d, h);
  return d;
}

// Replicated SDF scene for --sdf: union( sphere r=0.2 @ centre , torus R=0.22 r=0.07 @
// (0.2,0.25,0.75) ).
struct SceneHold {
  Kokkos::View<peclet::core::geom::ShapeNode<real_t>*, Mem> nodes;
  Kokkos::View<peclet::core::geom::GridDesc<real_t>*, Mem> grids;
  Kokkos::View<float*, Mem> pool;
};
static peclet::voro::SdfScene<real_t> makeScene(SceneHold& h) {
  using namespace peclet::core::geom;
  h.nodes = Kokkos::View<ShapeNode<real_t>*, Mem>("nodes", 3);
  auto hn = Kokkos::create_mirror_view(h.nodes);
  hn(0).kind = kUnion;
  hn(0).aux0 = 1;
  hn(0).aux1 = 2;
  hn(1).kind = kSphere;
  hn(1).params[0] = 0.2;
  hn(1).transform.translation = peclet::core::Vec3<real_t>{0.5, 0.5, 0.5};
  hn(2).kind = kTorus;
  hn(2).params[0] = 0.22;
  hn(2).params[1] = 0.07;
  hn(2).transform.translation = peclet::core::Vec3<real_t>{0.2, 0.25, 0.75};
  Kokkos::deep_copy(h.nodes, hn);
  h.grids = Kokkos::View<GridDesc<real_t>*, Mem>("grids", 1);
  h.pool = Kokkos::View<float*, Mem>("pool", 1);
  return peclet::voro::SdfScene<real_t>{h.nodes, h.grids, h.pool, 3, 0, real_t(1e-5)};
}

template <class Sdf>
static int runBench(int rank, int nproc, int N, int nSteps, const Sdf& sdf, bool withSdf) {
  int rcG = 0;
  {
    const Vec3 L = {1.0, 1.0, 1.0};
    const real_t spacing = std::cbrt((L[0] * L[1] * L[2]) / N);
    const double rcut =
        (std::getenv("VORF_RCUT") ? std::atof(std::getenv("VORF_RCUT")) : 3.5) * spacing;
    const real_t tol = real_t(1e-4) * spacing;
    const real_t skin = real_t(0.25) * spacing;

    if (rank == 0)
      std::printf(
          "=== distributed repair vs cold build (np=%d, 1 proc/core) N=%d nSteps=%d sp=%.4g "
          "rcut=%.2f·sp%s ===\n",
          nproc, N, nSteps, (double)spacing, rcut / spacing, withSdf ? " [SDF scene]" : "");

    // identical global seed set + velocities on every rank (deterministic ballistic motion).
    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::normal_distribution<real_t> Ng(0, 1);
    std::vector<Vec3> p0(N), vel(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        p0[i][d] = L[d] * U(rng);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        vel[i][d] = Ng(rng);

    peclet::voro::mpi::VoronoiHalo<real_t> halo;
    halo.init({0, 0, 0}, {L[0], L[1], L[2]}, {16, 16, 16}, {true, true, true}, MPI_COMM_WORLD);

    // this rank's owned seeds (by ownership at t0) + their global ids + velocities.
    std::vector<long> ownedGid;
    std::vector<Vec3> ownedP0, ownedVel;
    std::vector<real_t> ownedW;
    for (int i = 0; i < N; ++i)
      if (halo.ownerOf(p0[i]) == rank) {
        ownedP0.push_back(p0[i]);
        ownedVel.push_back(vel[i]);
        ownedGid.push_back(i);
        ownedW.push_back(0.0);
      }
    const int nOwned = (int)ownedP0.size();

    auto advanceOwned = [&](real_t scale, int step, std::vector<Vec3>& out) {
      out.resize(nOwned);
      const real_t s = scale * step;
      for (int i = 0; i < nOwned; ++i)
        for (int d = 0; d < 3; ++d)
          out[i][d] = wrap1(ownedP0[i][d] + ownedVel[i][d] * s, L[d]);
    };

    std::vector<real_t> disps = {real_t(1e-4), real_t(2e-4), real_t(5e-4), real_t(1e-3),
                                 real_t(2e-3), real_t(5e-3), real_t(1e-2)};
    if (const char* e = std::getenv("VORF_DISPS")) {
      disps.clear();
      std::stringstream ss(e);
      double d;
      while (ss >> d)
        disps.push_back((real_t)d);
    }
    if (rank == 0)
      std::printf("%6s %10s %10s %8s %8s %9s %9s %10s\n", "disp", "cold_ms", "repair_ms", "speedup",
                  "regath%", "p1%", "p2%", "maxRelV");

    for (real_t disp : disps) {
      const real_t scale = disp * spacing;
      std::vector<Vec3> owned;

      // ---------- distributed COLD build timing (gather + buildTessellation(owned) every step)
      // ----------
      double tCold = 0;
      for (int s = 1; s <= nSteps; ++s) {
        advanceOwned(scale, s, owned);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        auto g = halo.gather(owned, ownedGid, ownedW, rcut);
        auto dPos = uploadCombined(g.pos, L);
        Kokkos::View<real_t*, Mem> wd;
        Kokkos::View<long*, Mem> gd;
        auto r = peclet::voro::buildTessellation<real_t, false, Sdf>(
            dPos, wd, (int)g.pos.size(), L.data(), 4, N, gd, sdf, false, g.nOwned);
        Kokkos::fence();
        tCold += MPI_Wtime() - t0;
        (void)r;
      }

      // ---------- distributed REPAIR timing (refresh + local two-pass repair; skin-trip re-gather)
      // ----------
      double tRep = 0;
      long regath = 0, p1 = 0, p2 = 0;
      // establish at t0 — the library-level distributed driver (VoronoiHalo + MovingTessellation
      // + the distributed Verlet-skin invariant now live in DistributedMovingTessellation; this
      // bench validates it via the exactness gate below).
      advanceOwned(scale, 0, owned);
      peclet::voro::mpi::DistributedMovingTessellation<real_t, CMAXP, CMAXT, Sdf> dmt;
      dmt.init({0, 0, 0}, {L[0], L[1], L[2]}, {16, 16, 16}, {true, true, true}, (real_t)rcut, skin,
               tol, MPI_COMM_WORLD, 4, N);
      dmt.setSdf(sdf);
      dmt.establish(owned, ownedGid, ownedW);

      for (int s = 1; s <= nSteps; ++s) {
        advanceOwned(scale, s, owned);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        auto st = dmt.step(owned);
        Kokkos::fence();
        tRep += MPI_Wtime() - t0;
        if (st.regathered) {
          ++regath;
        } else {
          p1 += st.repair.pass1;
          p2 += st.repair.pass2;
        }
      }

      // ---------- exactness: cold-build the SAME final combined positions, compare owned volumes
      // ----------
      double maxRelV = 0, maxRelSingle = 0;
      long emptyMM = 0;
      {
        Kokkos::View<real_t*, Mem> wd;
        Kokkos::View<long*, Mem> gd;
        auto& mt = dmt.tess();
        auto rr = peclet::voro::buildTessellation<real_t, false, Sdf>(
            dmt.positions(), wd, mt.N, L.data(), 4, N, gd, sdf, false, mt.nProc);
        auto ov = Kokkos::create_mirror_view(rr.view.cellVolume);
        auto rv = Kokkos::create_mirror_view(mt.vol);
        Kokkos::deep_copy(ov, rr.view.cellVolume);
        Kokkos::deep_copy(rv, mt.vol);
        for (int i = 0; i < mt.nProc; ++i) {
          const double o = ov(i);
          if (o > 0)
            maxRelV = std::max(maxRelV, std::fabs((double)rv(i) - o) / o);
          else if (rv(i) != 0)
            ++emptyMM;
        }
        // --sdf: the DISTRIBUTED cold SDF build of the owned cells vs a SINGLE-RANK cold SDF build
        // of the whole point set at the same (final) positions, keyed by global id — np=1,2,4
        // identical to single rank (cold vs cold: round-off). The repair's own deviation from the
        // cold build is the wall-free gate above (the certificate tolerance, not the SDF).
        if (withSdf) {
          std::vector<Vec3> all(N);
          const int sLast = nSteps;
          const real_t sc = scale * sLast;
          for (int i = 0; i < N; ++i)
            for (int d = 0; d < 3; ++d)
              all[i][d] = wrap1(p0[i][d] + vel[i][d] * sc, L[d]);
          auto dAll = uploadCombined(all, L);
          auto single = peclet::voro::buildTessellation<real_t, false, Sdf>(dAll, wd, N, L.data(),
                                                                            4, N, gd, sdf, false);
          auto sv = Kokkos::create_mirror_view(single.view.cellVolume);
          Kokkos::deep_copy(sv, single.view.cellVolume);
          const auto& cg = dmt.combinedGid();
          for (int i = 0; i < mt.nProc; ++i) {
            const double o = sv(cg[i]), d = ov(i);
            if (o > 0)
              maxRelSingle = std::max(maxRelSingle, std::fabs(d - o) / o);
            else if (d != 0)
              ++emptyMM;
          }
        }
      }

      // aggregate across ranks (worst build/repair time = the distributed step cost; max error).
      double maxCold = 0, maxRep = 0, gMaxRelV = 0, gMaxRelSingle = 0;
      long sumP1 = 0, sumP2 = 0, sumReg = 0, gEmptyMM = 0;
      const double coldMs = 1e3 * tCold / nSteps, repMs = 1e3 * tRep / nSteps;
      MPI_Allreduce(&coldMs, &maxCold, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&repMs, &maxRep, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&maxRelV, &gMaxRelV, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&maxRelSingle, &gMaxRelSingle, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&emptyMM, &gEmptyMM, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&p1, &sumP1, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&p2, &sumP2, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&regath, &sumReg, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (gMaxRelV > 1e-2)
        rcG = 1;
      if (withSdf && (gMaxRelSingle > 1e-9 || gEmptyMM != 0))
        rcG = 1;  // A0 gate: distributed cold SDF == single-rank cold SDF, empties agree
      if (rank == 0) {
        const double nFast = (double)nSteps * nproc - sumReg;  // step-rank fast-path count
        std::printf("%6.3f %10.2f %10.2f %8.2f %8.1f %9.1f %9.1f %10.2e\n", (double)disp, maxCold,
                    maxRep, maxCold / maxRep, 100.0 * sumReg / ((double)nSteps * nproc),
                    nFast > 0 ? 100.0 * sumP1 / (nFast * (N / nproc)) : 0.0,
                    nFast > 0 ? 100.0 * sumP2 / (nFast * (N / nproc)) : 0.0, gMaxRelV);
        if (withSdf)
          std::printf("        vs single-rank cold SDF build: maxRelV=%.2e  emptyMismatch=%ld\n",
                      gMaxRelSingle, gEmptyMM);
      }
    }
    if (rank == 0)
      std::printf("REPAIR(MPI%s) exactness: %s\n", withSdf ? ",SDF" : "",
                  rcG == 0 ? "PASS" : "FAIL");
  }
  return rcG;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int rcG = 0;
  {
    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const int N = (argc > 1) ? std::atoi(argv[1]) : 400000;
    const int nSteps = (argc > 2) ? std::atoi(argv[2]) : 8;
    bool withSdf = false;
    for (int a = 3; a < argc; ++a)
      if (std::string(argv[a]) == "--sdf")
        withSdf = true;
    if (withSdf) {
      SceneHold hold;
      auto sdf = makeScene(hold);
      rcG = runBench(rank, nproc, N, nSteps, sdf, true);
    } else {
      rcG = runBench(rank, nproc, N, nSteps, peclet::voro::NoSdf{}, false);
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return rcG;
}
