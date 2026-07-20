/// @file
/// @brief DistributedMovingTessellation — the library-level distributed dynamic-Voronoi driver.
///
/// Promotes the validated orchestration that previously lived only in
/// tests/kokkos_mpi/bench_repair_mpi.cpp into a reusable header: VoronoiHalo (host ghost
/// gather/refresh over core's decomposition + NBX halo) composed with the device
/// MovingTessellation (two-pass repair) under the distributed Verlet-skin invariant.
///
/// Per step, with this rank's owned positions:
///   - trip test: if ANY rank's max owned displacement since the last (re)gather exceeds
///     skin/2, ALL ranks re-gather (halo.gather rebuilds the ghost topology via NBX) and cold-
///     rebuild the tessellation. The trip is a LOCAL test but the gather and refresh paths are
///     both collective, so it is reduced to a GLOBAL decision (MPI_Allreduce MAX) — without this
///     the run deadlocks once ranks diverge on the trip (observed at np>=4).
///   - fast path: refresh ghost positions on the established topology (halo.refreshPositions)
///     and run the local device two-pass repair (MovingTessellation::step).
///
/// Owned cells are [0, nOwned) of the combined view; their geometry equals a cold rebuild of the
/// same combined positions to the repair tolerance (gated by the repair_mpi ctests at np=1,2,4).
#ifndef PECLET_VORO_MPI_DISTRIBUTED_MOVING_HPP
#define PECLET_VORO_MPI_DISTRIBUTED_MOVING_HPP

#include <mpi.h>

#include <array>
#include <cmath>
#include <vector>

#include "../repair.hpp"
#include "voronoi_halo.hpp"

namespace peclet::voro::mpi {

template <class Real, int MAXP = 64, int MAXT = 112>
struct DistributedMovingTessellation {
  using Vec3 = std::array<Real, 3>;
  using Mem = typename Kokkos::DefaultExecutionSpace::memory_space;

  struct StepStats {
    bool regathered = false;  ///< this step took the re-gather + cold-rebuild path (collective)
    RepairStats repair;       ///< fast-path repair stats (valid when !regathered)
  };

  /// One-time setup. gsize is the ORB decomposition granularity (cells per axis).
  void init(std::array<Real, 3> origin, std::array<Real, 3> size, std::array<long, 3> gsize,
            std::array<bool, 3> periodic, Real rcut, Real skin, Real tol, MPI_Comm comm,
            int searchWindow = 4, int densityCount = -1) {
    halo_.init(origin, size, gsize, periodic, comm);
    comm_ = comm;
    for (int d = 0; d < 3; ++d)
      L_[d] = size[d];
    rcut_ = rcut;
    skin_ = skin;
    tol_ = tol;
    sw_ = searchWindow;
    density_ = densityCount;
  }

  int ownerOf(const Vec3& x) const { return halo_.ownerOf(x); }

  /// Establish the tessellation from this rank's owned seeds (collective). Call once after init
  /// and whenever ownership changes externally (e.g. after a rebalance/migration).
  void establish(const std::vector<Vec3>& ownedPos, const std::vector<long>& ownedGid,
                 const std::vector<Real>& ownedW) {
    gid_ = ownedGid;
    w_ = ownedW;
    regather(ownedPos);
  }

  /// Advance to new owned positions (same ownership as establish; collective every step).
  StepStats step(const std::vector<Vec3>& ownedPos) {
    StepStats st;
    int localTrip = (maxDispFrom(ownedPos, refPos_) > Real(0.5) * skin_) ? 1 : 0;
    int anyTrip = 0;
    MPI_Allreduce(&localTrip, &anyTrip, 1, MPI_INT, MPI_MAX, comm_);
    if (anyTrip) {
      regather(ownedPos);
      st.regathered = true;
    } else {
      std::vector<Vec3> comb;
      halo_.refreshPositions(ownedPos, comb);
      upload(comb);
      st.repair = mt_.step(dPos_);
    }
    return st;
  }

  /// Combined owned+ghost count and owned count of the current tessellation.
  int nCombined() const { return nComb_; }
  int nOwned() const { return nOwned_; }
  /// The device tessellation (cells [0, nOwned) are this rank's owned cells).
  MovingTessellation<Real, MAXP, MAXT>& tess() { return mt_; }
  const Kokkos::View<Real*, Mem>& positions() const { return dPos_; }
  /// Global ids of the combined (owned+ghost) seeds after the last (re)gather.
  const std::vector<long>& combinedGid() const { return combGid_; }
  long numRegathers() const { return nRegather_; }

 private:
  void regather(const std::vector<Vec3>& ownedPos) {
    auto g = halo_.gather(ownedPos, gid_, w_, rcut_);
    nComb_ = (int)g.pos.size();
    nOwned_ = g.nOwned;
    combGid_ = g.gid;
    upload(g.pos);
    mt_.alloc(nComb_, L_, tol_, skin_, sw_, density_, nOwned_);
    mt_.rebuild(dPos_);
    refPos_ = ownedPos;
    ++nRegather_;
  }

  void upload(const std::vector<Vec3>& p) {
    const int n = (int)p.size();
    // exact size: MovingTessellation deep_copies the whole extent (xRef), so a slack buffer trips
    // the extent check
    if ((int)dPos_.extent(0) != 3 * n)
      dPos_ = Kokkos::View<Real*, Mem>(
          Kokkos::view_alloc(std::string("dmt_pos"), Kokkos::WithoutInitializing), (size_t)n * 3);
    auto h = Kokkos::create_mirror_view(dPos_);
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < 3; ++k)
        h(3 * i + k) = wrap1(p[i][k], L_[k]);
    Kokkos::deep_copy(dPos_, h);
  }

  Real maxDispFrom(const std::vector<Vec3>& a, const std::vector<Vec3>& b) const {
    if (a.size() != b.size())
      return Real(1e30);  // ownership changed under us: force a re-gather
    Real m2 = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      Real d2 = 0;
      for (int d = 0; d < 3; ++d) {
        Real q = a[i][d] - b[i][d];
        q -= std::round(q / L_[d]) * L_[d];
        d2 += q * q;
      }
      m2 = std::max(m2, d2);
    }
    return std::sqrt(m2);
  }

  static Real wrap1(Real x, Real L) {
    Real y = std::fmod(x, L);
    return y < 0 ? y + L : y;
  }

  VoronoiHalo<Real> halo_;
  MovingTessellation<Real, MAXP, MAXT> mt_;
  Kokkos::View<Real*, Mem> dPos_;
  std::vector<Vec3> refPos_;   // owned positions at the last (re)gather (Verlet reference)
  std::vector<long> gid_, combGid_;
  std::vector<Real> w_;
  Real L_[3] = {1, 1, 1};
  Real rcut_ = 0, skin_ = 0, tol_ = 0;
  int sw_ = 4, density_ = -1, nComb_ = 0, nOwned_ = 0;
  long nRegather_ = 0;
  MPI_Comm comm_ = MPI_COMM_WORLD;
};

}  // namespace peclet::voro::mpi

#endif  // PECLET_VORO_MPI_DISTRIBUTED_MOVING_HPP
