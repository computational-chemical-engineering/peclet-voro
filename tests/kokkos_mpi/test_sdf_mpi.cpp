/**
 * @file test_sdf_mpi.cpp
 * \brief Distributed SDF-bounded tessellation == single-rank.
 *
 * Like test_voronoi_mpi, but each rank clips its owned+ghost cells against a
 * replicated solid (an analytic ball). The SDF is read-only geometry replicated
 * on every rank, so no extra exchange is needed: each rank's owned clipped cells
 * must match the serial SDF-bounded build (a legacy SignedDistanceBoundary over
 * the same ball), and seeds inside the solid have no owned cell anywhere.
 *
 * Run under mpirun at np = 1, 2, 4.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/sdf.hpp"
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

// Legacy boundary adapter mirroring the device SdfSphere (central-diff gradient).
struct SphereBnd : vor::SignedDistanceBoundary<real_t> {
  vor::device::SdfSphere<real_t> s;
  real_t value(const Vec3& x) const override { return s.eval(x[0], x[1], x[2]); }
  Vec3 gradient(const Vec3& x) const override {
    const real_t h = s.gradH();
    return {(s.eval(x[0] + h, x[1], x[2]) - s.eval(x[0] - h, x[1], x[2])) / (2 * h),
            (s.eval(x[0], x[1] + h, x[2]) - s.eval(x[0], x[1] - h, x[2])) / (2 * h),
            (s.eval(x[0], x[1], x[2] + h) - s.eval(x[0], x[1], x[2] - h)) / (2 * h)};
  }
};

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
    const real_t spacing = std::cbrt(1.0 / N);
    const double rcut = 5.0 * spacing;
    vor::device::SdfSphere<real_t> sphere{0.5, 0.5, 0.5, 0.25};

    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<Vec3> pos(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);

    // Serial SDF-bounded reference (cell index == seed; empty => volume 0).
    vor::Box<real_t> box(L);
    vor::CellComplex<real_t> serial(&box);
    SphereBnd bnd;
    bnd.s = sphere;
    serial.setBoundary(&bnd);
    serial.build(pos);
    const std::vector<vor::CellGeometry<real_t> >& sg = serial.getGeoms();
    std::map<int, real_t> refVol;
    for (int i = 0; i < N && i < (int)sg.size(); ++i)
      if (sg[i].getVolume() > 0)
        refVol[i] = sg[i].getVolume();

    // Distributed: own + gather ghosts, clip owned+ghost against the replicated SDF.
    vor::mpi::VoronoiHalo<real_t> halo;
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
    auto g = halo.gather(ownedPos, ownedGid, ownedW, rcut);
    const int nComb = (int)g.pos.size();

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
    auto res = vor::device::buildTessellation<real_t, false, vor::device::SdfSphere<real_t> >(
        dPos, dW, nComb, Larr, /*sw=*/4, /*densityCount=*/N, dGid, sphere);
    auto vol = Kokkos::create_mirror_view(res.view.cellVolume);
    auto st = Kokkos::create_mirror_view(res.status);
    Kokkos::deep_copy(vol, res.view.cellVolume);
    Kokkos::deep_copy(st, res.status);

    const double meanVol = (1.0) / std::max<size_t>(1, refVol.size());
    int volMis = 0, emptyMis = 0, ownedFluid = 0;
    for (int i = 0; i < g.nOwned; ++i) {  // owned cells first
      int gid = (int)g.gid[i];
      const bool devEmpty = (st(i) & vor::device::kEmpty) != 0;
      auto it = refVol.find(gid);
      if (it == refVol.end()) {  // serial: seed in solid
        if (!devEmpty)
          ++emptyMis;
        continue;
      }
      if (devEmpty) {
        ++emptyMis;
        continue;
      }
      ++ownedFluid;
      if (std::fabs(vol(i) - it->second) / meanVol > 1e-6)
        ++volMis;
    }
    int totOwnedFluid = 0;
    MPI_Allreduce(&ownedFluid, &totOwnedFluid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int localFail = (volMis || emptyMis || totOwnedFluid != (int)refVol.size()) ? 1 : 0;
    std::printf("  [rank %d/%d] ownedFluid=%d volMis=%d emptyMis=%d totFluid=%d (serial %zu) %s\n",
                rank, nproc, ownedFluid, volMis, emptyMis, totOwnedFluid, refVol.size(),
                localFail ? "FAIL" : "PASS");
    MPI_Allreduce(MPI_IN_PLACE, &localFail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    failures = localFail;
    if (rank == 0)
      std::printf("sdf_mpi (np=%d): %s\n", nproc, failures == 0 ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return failures;
}
