/**
 * @file bench_repair_mpi.cpp
 * \brief Distributed (MPI) two-pass repair vs distributed cold build, as a function of dimensionless
 *        displacement — the "serial + MPI, one process per core" configuration (run OMP_NUM_THREADS=1).
 *
 * Each rank owns an ORB block (transport-core, via VoronoiHalo). The distributed COLD build, per step:
 * gather every seed within rcut of the block (the MPI halo exchange), then tessellate the owned cells
 * over owned+ghost. The distributed REPAIR keeps a resident topology of the owned cells and, while no
 * owned seed has moved beyond the Verlet skin, only *refreshes* the ghost positions on the established
 * halo topology (VoronoiHalo::refreshPositions — same comm pattern, no re-decomposition, combined order
 * stable so the resident neighbour indices stay valid) and runs the local two-pass gather repair
 * (MovingTessellation::step over the owned cells). On a skin trip it re-gathers + cold-rebuilds (the
 * distributed fallback). The MPI halo exchange is common to both paths; the difference is build vs
 * reeval+repair compute per rank.
 *
 * Exactness: at the final step the owned-cell volumes from the repair are compared to a fresh cold
 * build over the SAME combined positions (so same ordering) — the distributed analogue of the
 * single-domain oracle. Space-filling (Σ owned volume == box) is also checked.
 *
 * Run:  OMP_NUM_THREADS=1 mpirun -np <R> --bind-to core ./bench_repair_mpi [N_global] [nSteps]
 */
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "tpx/common/view.hpp"
#include "vorflow/device/repair.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/mpi/voronoi_halo.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;
using Mem = tpx::MemSpace;
static constexpr int CMAXP = 64, CMAXT = 112;

static real_t wrap1(real_t x, real_t L) {
  x -= L * std::floor(x / L);
  if (x >= L) x -= L;
  if (x < 0) x += L;
  return x;
}

// Upload a combined (owned+ghost) Vec3 list, wrapped to [0,L), into a device 3*n view.
static Kokkos::View<real_t*, Mem> uploadCombined(const std::vector<Vec3>& p, const Vec3& L) {
  const int n = (int)p.size();
  Kokkos::View<real_t*, Mem> d(Kokkos::view_alloc(std::string("dPos"), Kokkos::WithoutInitializing),
                               (size_t)n * 3);
  auto h = Kokkos::create_mirror_view(d);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) h(3 * i + k) = wrap1(p[i][k], L[k]);
  Kokkos::deep_copy(d, h);
  return d;
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
    const Vec3 L = {1.0, 1.0, 1.0};
    const real_t spacing = std::cbrt((L[0] * L[1] * L[2]) / N);
    const double rcut = (std::getenv("VORF_RCUT") ? std::atof(std::getenv("VORF_RCUT")) : 3.5) * spacing;
    const real_t tol = real_t(1e-4) * spacing;
    const real_t skin = real_t(0.25) * spacing;

    if (rank == 0)
      std::printf("=== distributed repair vs cold build (np=%d, 1 proc/core) N=%d nSteps=%d sp=%.4g rcut=%.2f·sp ===\n",
                  nproc, N, nSteps, (double)spacing, rcut / spacing);

    // identical global seed set + velocities on every rank (deterministic ballistic motion).
    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::normal_distribution<real_t> Ng(0, 1);
    std::vector<Vec3> p0(N), vel(N);
    for (int i = 0; i < N; ++i) for (int d = 0; d < 3; ++d) p0[i][d] = L[d] * U(rng);
    for (int i = 0; i < N; ++i) for (int d = 0; d < 3; ++d) vel[i][d] = Ng(rng);

    vor::mpi::VoronoiHalo<real_t> halo;
    halo.init({0, 0, 0}, {L[0], L[1], L[2]}, {16, 16, 16}, {true, true, true}, MPI_COMM_WORLD);

    // this rank's owned seeds (by ownership at t0) + their global ids + velocities.
    std::vector<long> ownedGid;
    std::vector<Vec3> ownedP0, ownedVel;
    std::vector<real_t> ownedW;
    for (int i = 0; i < N; ++i)
      if (halo.ownerOf(p0[i]) == rank) {
        ownedP0.push_back(p0[i]); ownedVel.push_back(vel[i]); ownedGid.push_back(i); ownedW.push_back(0.0);
      }
    const int nOwned = (int)ownedP0.size();

    auto advanceOwned = [&](real_t scale, int step, std::vector<Vec3>& out) {
      out.resize(nOwned);
      const real_t s = scale * step;
      for (int i = 0; i < nOwned; ++i)
        for (int d = 0; d < 3; ++d) out[i][d] = wrap1(ownedP0[i][d] + ownedVel[i][d] * s, L[d]);
    };
    auto maxOwnedDispFrom = [&](const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
      real_t m2 = 0;
      for (int i = 0; i < nOwned; ++i) {
        real_t d2 = 0;
        for (int d = 0; d < 3; ++d) { real_t q = a[i][d] - b[i][d]; q -= std::round(q / L[d]) * L[d]; d2 += q * q; }
        m2 = std::max(m2, d2);
      }
      return std::sqrt(m2);
    };

    const std::vector<real_t> disps = {real_t(0.001), real_t(0.002), real_t(0.005), real_t(0.01)};
    if (rank == 0)
      std::printf("%6s %10s %10s %8s %8s %9s %9s %10s\n", "disp", "cold_ms", "repair_ms", "speedup",
                  "regath%", "p1%", "p2%", "maxRelV");

    for (real_t disp : disps) {
      const real_t scale = disp * spacing;
      std::vector<Vec3> owned, refPos;

      // ---------- distributed COLD build timing (gather + buildTessellation(owned) every step) ----------
      double tCold = 0;
      for (int s = 1; s <= nSteps; ++s) {
        advanceOwned(scale, s, owned);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        auto g = halo.gather(owned, ownedGid, ownedW, rcut);
        auto dPos = uploadCombined(g.pos, L);
        Kokkos::View<real_t*, Mem> wd; Kokkos::View<long*, Mem> gd;
        auto r = vor::device::buildTessellation<real_t, false>(dPos, wd, (int)g.pos.size(), L.data(),
            4, N, gd, vor::device::NoSdf{}, false, g.nOwned);
        Kokkos::fence();
        tCold += MPI_Wtime() - t0;
        (void)r;
      }

      // ---------- distributed REPAIR timing (refresh + local two-pass repair; skin-trip re-gather) ----------
      double tRep = 0;
      long regath = 0, p1 = 0, p2 = 0;
      // establish at t0
      advanceOwned(scale, 0, owned);
      auto g = halo.gather(owned, ownedGid, ownedW, rcut);
      int nComb = (int)g.pos.size();
      auto dPos = uploadCombined(g.pos, L);
      vor::device::MovingTessellation<real_t, CMAXP, CMAXT> mt;
      mt.alloc(nComb, L.data(), tol, skin, 4, N, g.nOwned);
      mt.rebuild(dPos);
      refPos = owned;  // Verlet reference (owned positions at last (re)gather)
      Kokkos::View<real_t*, Mem> lastDPos = dPos;

      for (int s = 1; s <= nSteps; ++s) {
        advanceOwned(scale, s, owned);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        if (maxOwnedDispFrom(owned, refPos) > real_t(0.5) * skin) {
          // skin trip: re-gather (re-establish topology) + cold rebuild — the distributed fallback.
          g = halo.gather(owned, ownedGid, ownedW, rcut);
          nComb = (int)g.pos.size();
          dPos = uploadCombined(g.pos, L);
          mt.alloc(nComb, L.data(), tol, skin, 4, N, g.nOwned);
          mt.rebuild(dPos);
          refPos = owned;
          ++regath;
        } else {
          // fast path: refresh ghost positions on the established topology, repair locally.
          std::vector<Vec3> comb;
          halo.refreshPositions(owned, comb);
          dPos = uploadCombined(comb, L);
          auto st = mt.step(dPos);
          p1 += st.pass1; p2 += st.pass2;
        }
        Kokkos::fence();
        tRep += MPI_Wtime() - t0;
        lastDPos = dPos;
      }

      // ---------- exactness: cold-build the SAME final combined positions, compare owned volumes ----------
      double maxRelV = 0;
      {
        Kokkos::View<real_t*, Mem> wd; Kokkos::View<long*, Mem> gd;
        auto rr = vor::device::buildTessellation<real_t, false>(lastDPos, wd, mt.N, L.data(), 4, N, gd,
            vor::device::NoSdf{}, false, mt.nProc);
        auto ov = Kokkos::create_mirror_view(rr.view.cellVolume);
        auto rv = Kokkos::create_mirror_view(mt.vol);
        Kokkos::deep_copy(ov, rr.view.cellVolume);
        Kokkos::deep_copy(rv, mt.vol);
        for (int i = 0; i < mt.nProc; ++i) {
          const double o = ov(i);
          if (o > 0) maxRelV = std::max(maxRelV, std::fabs((double)rv(i) - o) / o);
        }
      }

      // aggregate across ranks (worst build/repair time = the distributed step cost; max error).
      double maxCold = 0, maxRep = 0, gMaxRelV = 0;
      long sumP1 = 0, sumP2 = 0, sumReg = 0;
      const double coldMs = 1e3 * tCold / nSteps, repMs = 1e3 * tRep / nSteps;
      MPI_Allreduce(&coldMs, &maxCold, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&repMs, &maxRep, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&maxRelV, &gMaxRelV, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&p1, &sumP1, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&p2, &sumP2, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&regath, &sumReg, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (gMaxRelV > 1e-2) rcG = 1;
      if (rank == 0) {
        const double nFast = (double)nSteps * nproc - sumReg;  // step-rank fast-path count
        std::printf("%6.3f %10.2f %10.2f %8.2f %8.1f %9.1f %9.1f %10.2e\n", (double)disp, maxCold, maxRep,
                    maxCold / maxRep, 100.0 * sumReg / ((double)nSteps * nproc),
                    nFast > 0 ? 100.0 * sumP1 / (nFast * (N / nproc)) : 0.0,
                    nFast > 0 ? 100.0 * sumP2 / (nFast * (N / nproc)) : 0.0, gMaxRelV);
      }
    }
    if (rank == 0) std::printf("REPAIR(MPI) exactness: %s\n", rcG == 0 ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return rcG;
}
