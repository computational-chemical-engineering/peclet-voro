/**
 * @file mpi/voronoi_halo.hpp
 * \brief Distributed Voronoi halo over transport-core (migration Phase 6).
 *
 * A Voronoi/power cell is fully determined by its local neighbourhood, so the
 * distributed tessellation needs ONE ghost exchange and no iteration: decompose
 * the periodic domain into blocks, gather every seed within the security radius
 * of the block, tessellate owned+ghost locally, and keep the owned cells (they
 * are bit-identical to the serial cells, since all their neighbours are present).
 *
 * This wrapper reuses transport-core for all of that — exactly the infrastructure
 * the dem distributed step uses, and the C++ counterpart of the validated
 * mpi/validate_voronoi.py recipe:
 *   - tpx::decomp::BlockDecomposer<3> : ORB block ownership;
 *   - tpx::halo::ParticleMigrator<3>  : ownerOf() + periodic wrap;
 *   - tpx::halo::ParticleHaloTopology<3>      : gather ghost seeds within rcut and
 *                                       forward owner fields (global id, weight)
 *                                       onto the ghost copies.
 *
 * Header-only host driver; requires MPI + transport-core (VORFLOW_MPI build).
 */
#ifndef VORFLOW_MPI_VORONOI_HALO_HPP
#define VORFLOW_MPI_VORONOI_HALO_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "tpx/common/mpi.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_halo_topology.hpp"
#include "tpx/halo/particle_migrator.hpp"

namespace vor {
namespace mpi {

template <class Real>
class VoronoiHalo {
 public:
  using Vec3 = std::array<Real, 3>;

  /// Combined owned+ghost seeds (owned first), with global ids and weights.
  struct Gathered {
    std::vector<Vec3> pos;     ///< [0,nOwned) owned, [nOwned,size) ghost
    std::vector<long> gid;     ///< global seed id per entry
    std::vector<Real> weight;  ///< per entry (0 if unweighted)
    int nOwned = 0;
  };

  /// @param origin,size  physical box (origin + extent L); seeds live in [origin, origin+size).
  /// @param gsize        ORB cell grid (decomposition granularity).
  /// @param periodic     per-axis periodicity.
  void init(std::array<Real, 3> origin, std::array<Real, 3> size, std::array<long, 3> gsize,
            std::array<bool, 3> periodic, MPI_Comm comm = MPI_COMM_WORLD) {
    comm_ = comm;
    int sz = 1;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &sz);
    dec_.init(static_cast<std::size_t>(sz), tpx::IVec<3>{gsize[0], gsize[1], gsize[2]});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = origin[i];
      map.cellSize[i] = size[i] / static_cast<double>(gsize[i]);
      map.periodic[i] = periodic[i];
    }
    mig_.init(dec_, rank_, map, comm_);
    halo_.init(mig_);
  }

  int rank() const { return rank_; }
  int size() const {
    int s = 1;
    MPI_Comm_size(comm_, &s);
    return s;
  }

  /// Rank that owns position x.
  int ownerOf(const Vec3& x) const {
    tpx::Vec<3> v{x[0], x[1], x[2]};
    return mig_.ownerOf(v);
  }

  /// Build the owner<->ghost topology over this rank's owned seeds and gather the
  /// ghost seeds within rcut, forwarding their global ids and weights.
  Gathered gather(const std::vector<Vec3>& ownedPos, const std::vector<long>& ownedGid,
                  const std::vector<Real>& ownedWeight, double rcut) {
    const int nOwned = static_cast<int>(ownedPos.size());
    std::vector<tpx::Vec<3>> pv(nOwned);
    for (int i = 0; i < nOwned; ++i)
      pv[i] = tpx::Vec<3>{ownedPos[i][0], ownedPos[i][1], ownedPos[i][2]};
    // includePeriodicSelf=false: the tessellator is periodic-native (minimal image),
    // so two same-rank seeds interact across a periodic face directly — only
    // cross-rank neighbours (which the build still gathers, periodic images
    // included) need to travel as ghosts. Self-ghosts would just duplicate owned
    // seeds (wrapping back onto them) and inflate the set.
    halo_.build(pv, rcut, /*includePeriodicSelf=*/false);
    const int ng = static_cast<int>(halo_.numGhost());

    // Forward owner global-id and weight onto the ghost copies.
    std::vector<long> ghostGid(ng);
    std::vector<Real> ghostW(ng);
    halo_.forward(ownedGid.data(), ghostGid.data());
    halo_.forward(ownedWeight.data(), ghostW.data());
    const std::vector<tpx::Vec<3>>& gpos = halo_.ghostPositions();

    Gathered g;
    g.nOwned = nOwned;
    g.pos.resize(nOwned + ng);
    g.gid.resize(nOwned + ng);
    g.weight.resize(nOwned + ng);
    for (int i = 0; i < nOwned; ++i) {
      g.pos[i] = ownedPos[i];
      g.gid[i] = ownedGid[i];
      g.weight[i] = ownedWeight[i];
    }
    for (int j = 0; j < ng; ++j) {
      g.pos[nOwned + j] = Vec3{gpos[j][0], gpos[j][1], gpos[j][2]};
      g.gid[nOwned + j] = ghostGid[j];
      g.weight[nOwned + j] = ghostW[j];
    }
    return g;
  }

 private:
  MPI_Comm comm_ = MPI_COMM_WORLD;
  int rank_ = 0;
  tpx::decomp::BlockDecomposer<3> dec_;
  tpx::halo::ParticleMigrator<3> mig_;
  tpx::halo::ParticleHaloTopology<3> halo_;
};

}  // namespace mpi
}  // namespace vor

#endif  // VORFLOW_MPI_VORONOI_HALO_HPP
