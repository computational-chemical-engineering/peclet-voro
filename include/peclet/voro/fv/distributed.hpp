/**
 * @file fv/distributed.hpp
 * \brief Track C, rung C5 (Voronoi methods plan): the distributed static solvers — the ghost
 * field exchange over VoronoiHalo (core's ORB decomposition + particle halo) and the global
 * reductions, installed on CollocatedNS / PressureSolver through their hooks.
 *
 * Layout: this rank's tessellation is built for its owned seeds [0, nOwned) with the gathered
 * ghost seeds [nOwned, nCombined) as candidates (DistributedMovingTessellation's layout); the face
 * mesh (`buildFaceMesh(view, aux, nOwned)`) keeps the facets toward ghosts as INTERFACE faces
 * owned here. Cell fields are sized nCombined; the owned part is stepped, the ghost part is
 * refreshed from the owning ranks by `exchange` (VoronoiHalo::forward over the established
 * topology) — velocity, its gradient, the pressure, the transpose moments — so an interface face
 * computes the same flux on both ranks from identical inputs. Pressure PCG: the matvec exchanges
 * the search direction, the dot products are all-reduced, the AMG preconditioner is the per-rank
 * owned block (block Jacobi). np = 1 is the single-rank code path (no ghosts, identity hooks).
 *
 * The exchange round-trips through the host (VoronoiHalo is a host halo) — adequate for the
 * per-stage exchanges of these solvers at the test sizes; a device-resident halo is the
 * device-first follow-up.
 */
#ifndef PECLET_VORO_FV_DISTRIBUTED_HPP
#define PECLET_VORO_FV_DISTRIBUTED_HPP

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <vector>

#include "peclet/voro/fv/operators.hpp"
#include "peclet/voro/mpi/voronoi_halo.hpp"

namespace peclet::voro::fv {

template <class Real>
struct GhostExchange {
  peclet::voro::mpi::VoronoiHalo<Real>* halo = nullptr;
  int nOwned = 0, nGhost = 0;

  void init(peclet::voro::mpi::VoronoiHalo<Real>& h, int owned) {
    halo = &h;
    nOwned = owned;
    nGhost = h.numGhost();
  }
  /// Refresh the ghost entries [nOwned, nOwned + nGhost) of an `nc`-component cell field.
  void exchange(const DV<Real>& f, int nc) const {
    if (!halo || nGhost == 0)
      return;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);
    std::vector<Real> own(nOwned), gh(nGhost);
    for (int c = 0; c < nc; ++c) {
      for (int i = 0; i < nOwned; ++i)
        own[i] = h(nc * i + c);
      halo->forward(own.data(), gh.data());
      for (int j = 0; j < nGhost; ++j)
        h(nc * (nOwned + j) + c) = gh[j];
    }
    Kokkos::deep_copy(f, h);
  }
  Real sum(Real v) const {
    if (!halo)
      return v;
    Real g = 0;
    MPI_Allreduce(&v, &g, 1, std::is_same_v<Real, double> ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM,
                  halo->comm());
    return g;
  }
  Real max(Real v) const {
    if (!halo)
      return v;
    Real g = 0;
    MPI_Allreduce(&v, &g, 1, std::is_same_v<Real, double> ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX,
                  halo->comm());
    return g;
  }
};

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_DISTRIBUTED_HPP
