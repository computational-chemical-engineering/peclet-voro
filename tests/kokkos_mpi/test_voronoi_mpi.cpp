/**
 * @file test_voronoi_mpi.cpp
 * \brief Phase-6 acceptance: distributed tessellation == single-rank.
 *
 * Every rank holds the same global seed set (for the oracle). Each rank:
 *   1. selects the seeds it owns (BlockDecomposer / ParticleMigrator.ownerOf);
 *   2. gathers ghost seeds within rcut via transport-core's ParticleHalo
 *      (VoronoiHalo), forwarding their global ids;
 *   3. tessellates the owned+ghost subset with the device tessellator in the
 *      full periodic box and keeps its owned cells.
 * It then checks every owned cell's volume and neighbour set (mapped back to
 * global ids) against the serial full-box tessellation (built locally from the
 * global set). Across ranks the owned cells partition the global set exactly.
 *
 * Run under mpirun at np = 1, 2, 4. Returns non-zero on any rank that mismatches.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/mpi/voronoi_halo.hpp"
#include "vorflow/voronoi.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {

real_t wrap1(real_t x, real_t L) {
  x -= L * std::floor(x / L);
  if (x >= L)
    x -= L;
  if (x < 0)
    x += L;
  return x;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    const int N = 1500;
    const Vec3 L = {1.0, 1.0, 1.0};
    const real_t boxVol = L[0] * L[1] * L[2];
    const real_t spacing = std::cbrt(boxVol / N);
    const double rcut = 5.0 * spacing;  // >= sw(4)*spacing, and < block extent

    // Identical global seed set on every rank.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<Vec3> pos(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);

    // Serial reference (full-box) tessellation: per-gid volume + neighbour set.
    vor::Box<real_t> box(L);
    vor::CellComplex<real_t> serial(&box);
    serial.build(pos);
    std::vector<vor::Cell<real_t> > scells;
    serial.materializeCells(scells);
    const std::vector<vor::CellGeometry<real_t> >& sgeom = serial.getGeoms();
    std::map<int, real_t> refVol;
    std::map<int, std::set<int> > refNbr;
    for (size_t i = 0; i < scells.size(); ++i) {
      int gid = (int)scells[i].getID();
      refVol[gid] = sgeom[i].getVolume();
      std::set<int>& s = refNbr[gid];
      for (vor::uint1 f = 0; f < scells[i].numFacets(); ++f) {
        vor::uint2 g = scells[i].getNbr(f);
        if (g != vor::noNbr && g != vor::boundaryNbr)
          s.insert((int)g);
      }
    }

    // Distributed: own + gather ghosts via transport-core.
    vor::mpi::VoronoiHalo<real_t> halo;
    halo.init({0, 0, 0}, {L[0], L[1], L[2]}, {16, 16, 16}, {true, true, true}, MPI_COMM_WORLD);
    std::vector<Vec3> ownedPos;
    std::vector<long> ownedGid;
    std::vector<real_t> ownedW;
    for (int i = 0; i < N; ++i) {
      if (halo.ownerOf(pos[i]) == rank) {
        ownedPos.push_back(pos[i]);
        ownedGid.push_back(i);
        ownedW.push_back(0.0);
      }
    }
    auto g = halo.gather(ownedPos, ownedGid, ownedW, rcut);
    const int nComb = (int)g.pos.size();

    // Device tessellation of the owned+ghost subset (grid at the GLOBAL density).
    Kokkos::View<real_t*, tpx::MemSpace> dPos(
        Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)nComb * 3);
    {
      auto h = Kokkos::create_mirror_view(dPos);
      for (int i = 0; i < nComb; ++i)
        for (int k = 0; k < 3; ++k)
          h(3 * i + k) = wrap1(g.pos[i][k], L[k]);
      Kokkos::deep_copy(dPos, h);
    }
    Kokkos::View<real_t*, tpx::MemSpace> dW("w", nComb);
    Kokkos::View<long*, tpx::MemSpace> dGid(
        Kokkos::view_alloc(std::string("gid"), Kokkos::WithoutInitializing), nComb);
    {
      auto hg = Kokkos::create_mirror_view(dGid);
      for (int i = 0; i < nComb; ++i)
        hg(i) = g.gid[i];
      Kokkos::deep_copy(dGid, hg);
    }
    const real_t Larr[3] = {L[0], L[1], L[2]};
    auto res = vor::device::buildTessellation<real_t, false>(
        dPos, dW, nComb, Larr, /*sw=*/4, /*densityCount=*/N, dGid, {}, /*withForceGeom=*/true,
        /*nBuild=*/g.nOwned);  // build only owned cells (ghosts are candidate-only)
    auto vol = Kokkos::create_mirror_view(res.view.cellVolume);
    auto off = Kokkos::create_mirror_view(res.view.cellFacetOffset);
    auto cnt = Kokkos::create_mirror_view(res.view.cellFacetCount);
    auto nbr = Kokkos::create_mirror_view(res.view.facetNeighbor);
    auto st = Kokkos::create_mirror_view(res.status);
    Kokkos::deep_copy(vol, res.view.cellVolume);
    Kokkos::deep_copy(off, res.view.cellFacetOffset);
    Kokkos::deep_copy(cnt, res.view.cellFacetCount);
    Kokkos::deep_copy(nbr, res.view.facetNeighbor);
    Kokkos::deep_copy(st, res.status);

    int volMis = 0, setMis = 0, badStatus = 0;
    for (int i = 0; i < g.nOwned; ++i) {  // owned cells are the first nOwned
      int gid = (int)g.gid[i];
      if (st(i) & (vor::device::kOverflow | vor::device::kIncomplete | vor::device::kEmpty))
        ++badStatus;
      if (std::fabs(vol(i) - refVol[gid]) / refVol[gid] > 1e-9)
        ++volMis;
      std::set<int> dn;
      for (int k = off(i); k < off(i) + cnt(i); ++k) {
        int lj = nbr(k);
        if (lj >= 0)
          dn.insert((int)g.gid[lj]);  // local index -> global id
      }
      if (dn != refNbr[gid])
        ++setMis;
    }

    // Partition check: owned cells across ranks cover the global set exactly once.
    int totOwned = 0;
    MPI_Allreduce(&g.nOwned, &totOwned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int localFail = (volMis || setMis || badStatus || totOwned != N) ? 1 : 0;
    std::printf(
        "  [rank %d/%d] owned=%d ghost=%d volMis=%d setMis=%d badStatus=%d totOwned=%d %s\n", rank,
        nproc, g.nOwned, nComb - g.nOwned, volMis, setMis, badStatus, totOwned,
        localFail ? "FAIL" : "PASS");
    MPI_Allreduce(MPI_IN_PLACE, &localFail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    failures = localFail;
    if (rank == 0)
      std::printf("voronoi_mpi (np=%d): %s\n", nproc, failures == 0 ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return failures;
}
