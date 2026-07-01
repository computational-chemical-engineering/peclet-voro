/**
 * @file bench_voronoi_mpi.cpp
 * \brief Distributed cold-build throughput benchmark for the Voronoi tessellator.
 *
 * The MPI sibling of bench_convexcell: it times the DISTRIBUTED cold build. Each rank owns a
 * block of the domain (core ORB via VoronoiHalo), gathers ghost seeds within rcut
 * (the MPI communication), and tessellates its owned+ghost subset with the device tessellator
 * (`peclet::voro::buildTessellation`) — the same path validated bit-exact vs single-rank by
 * test_voronoi_mpi. Reports, per rank and aggregated:
 *   - owned / ghost cell counts (the ghost fraction = the distributed overhead),
 *   - gather time (MPI halo exchange) vs build time (the cold-build compute),
 *   - per-rank throughput (owned cells / build time) = the "MPI process per core" number,
 *   - aggregate distributed throughput (Σ owned / max build time).
 *
 * Run:  mpirun -np <R> --bind-to core ./bench_voronoi_mpi [N_global]   (set OMP_NUM_THREADS=1
 * for one MPI rank per core). N_global default 1e6.
 *
 * NOTE: this exercises the device tessellator's own grid gather (tessellator.hpp), which is a
 * DIFFERENT (older) neighbour search than bench_convexcell's worklist gather — so per-core
 * numbers are not directly comparable to the single-process worklist bench; this measures the
 * distributed path as it stands.
 */
#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/tessellator.hpp"
#include "peclet/voro/mpi/voronoi_halo.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

static real_t wrap1(real_t x, real_t L) {
  x -= L * std::floor(x / L);
  if (x >= L)
    x -= L;
  if (x < 0)
    x += L;
  return x;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    const int N = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    const Vec3 L = {1.0, 1.0, 1.0};
    const real_t spacing = std::cbrt((L[0] * L[1] * L[2]) / N);
    // Ghost gather radius in units of mean spacing. The tessellator's adaptive worklist
    // closes most Poisson cells by ~2.6·spacing, so the conservative 5·spacing window (=
    // sw·csz, the search MAX) gathers ~2× the ghosts that are actually consumed. 3.5·spacing
    // keeps every owned cell complete (Σvol == box) while cutting the ghost shell. Override
    // with VORF_RCUT to sweep.
    const double rcutMult = std::getenv("VORF_RCUT") ? std::atof(std::getenv("VORF_RCUT")) : 3.0;
    const double rcut = rcutMult * spacing;

    // Identical global seed set on every rank (deterministic).
    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<Vec3> pos(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);

    // Block decomposition + ghost gather (core).
    peclet::voro::mpi::VoronoiHalo<real_t> halo;
    halo.init({0, 0, 0}, {L[0], L[1], L[2]}, {16, 16, 16}, {true, true, true}, MPI_COMM_WORLD);
    std::vector<Vec3> ownedPos;
    std::vector<long> ownedGid;
    std::vector<real_t> ownedW;
    for (int i = 0; i < N; ++i)
      if (halo.ownerOf(pos[i]) == rank) {
        ownedPos.push_back(pos[i]);
        ownedGid.push_back(i);
        ownedW.push_back(0.0);
      }

    // --- gather (MPI halo exchange) timing ---
    double gatherBest = 1e30;
    typename peclet::voro::mpi::VoronoiHalo<real_t>::Gathered g;
    for (int r = 0; r < reps; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      double t0 = MPI_Wtime();
      g = halo.gather(ownedPos, ownedGid, ownedW, rcut);
      double t1 = MPI_Wtime();
      gatherBest = std::min(gatherBest, t1 - t0);
    }
    const int nComb = (int)g.pos.size();

    // device inputs (owned+ghost subset)
    Kokkos::View<real_t*, peclet::core::MemSpace> dPos(
        Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)nComb * 3);
    {
      auto h = Kokkos::create_mirror_view(dPos);
      for (int i = 0; i < nComb; ++i)
        for (int k = 0; k < 3; ++k)
          h(3 * i + k) = wrap1(g.pos[i][k], L[k]);
      Kokkos::deep_copy(dPos, h);
    }
    Kokkos::View<real_t*, peclet::core::MemSpace> dW("w", nComb);
    Kokkos::View<long*, peclet::core::MemSpace> dGid(
        Kokkos::view_alloc(std::string("gid"), Kokkos::WithoutInitializing), nComb);
    {
      auto hg = Kokkos::create_mirror_view(dGid);
      for (int i = 0; i < nComb; ++i)
        hg(i) = g.gid[i];
      Kokkos::deep_copy(dGid, hg);
    }
    const real_t Larr[3] = {L[0], L[1], L[2]};

    // --- build (cold tessellation) timing ---
    double buildBest = 1e30;
    double ownedVol = 0;
    long badOwned =
        0;  // owned cells flagged incomplete/overflow/empty (must be 0 for completeness)
    for (int r = 0; r < reps + 1; ++r) {  // first = warm
      MPI_Barrier(MPI_COMM_WORLD);
      double t0 = MPI_Wtime();
      auto res = peclet::voro::buildTessellation<real_t, false>(
          dPos, dW, nComb, Larr, /*sw=*/4, /*density=*/N, dGid, {}, /*withForceGeom=*/true,
          /*nBuild=*/g.nOwned);  // build only owned cells; ghosts are candidate-only
      Kokkos::fence();
      double t1 = MPI_Wtime();
      if (r > 0)
        buildBest = std::min(buildBest, t1 - t0);
      if (r == reps) {  // last rep: sum owned volumes + count incomplete owned cells
        auto vol = Kokkos::create_mirror_view(res.view.cellVolume);
        auto st = Kokkos::create_mirror_view(res.status);
        Kokkos::deep_copy(vol, res.view.cellVolume);
        Kokkos::deep_copy(st, res.status);
        for (int i = 0; i < g.nOwned; ++i) {
          ownedVol += vol(i);
          if (st(i) & (peclet::voro::kOverflow | peclet::voro::kIncomplete | peclet::voro::kEmpty))
            ++badOwned;
        }
      }
    }

    // --- aggregate ---
    const double perRankKps = g.nOwned / buildBest / 1e3;  // owned cells/s per rank
    int totOwned = 0;
    long totBad = 0;
    double maxBuild = 0, sumGather = 0, totVol = 0;
    MPI_Allreduce(&g.nOwned, &totOwned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&buildBest, &maxBuild, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&gatherBest, &sumGather, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&ownedVol, &totVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&badOwned, &totBad, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    // per-rank lines (ordered)
    for (int r = 0; r < nproc; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (r == rank)
        std::printf(
            "  [rank %2d/%d] owned=%7d ghost=%7d (%.0f%%)  gather=%.2f ms  build=%.2f ms  "
            "perRank=%.1f kcell/s\n",
            rank, nproc, g.nOwned, nComb - g.nOwned,
            100.0 * (nComb - g.nOwned) / std::max(1, nComb), gatherBest * 1e3, buildBest * 1e3,
            perRankKps);
      fflush(stdout);
    }
    if (rank == 0) {
      const double aggMps = totOwned / maxBuild / 1e6;  // aggregate (Σ owned / max build)
      std::printf(
          "MPI np=%2d  N=%d  rcut=%.2f·sp  | aggregate build = %.3f Mcell/s  | per-core = %.1f "
          "kcell/s  | "
          "max gather = %.2f ms  build = %.2f ms  | totOwned=%d badOwned=%ld Σvol=%.6f\n",
          nproc, N, rcutMult, aggMps, aggMps * 1e3 / nproc, sumGather * 1e3, maxBuild * 1e3,
          totOwned, totBad, totVol);
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
