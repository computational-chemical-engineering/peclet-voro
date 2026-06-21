/**
 * @file physics/interface.hpp
 * \brief Multiphase interface-tension energy over the published view (Phase 3).
 *
 * Faithful port of IntfDyn::getIntfEnergy, view-only: each cell of phase t_i sums,
 * over its facets to a LOWER-phase neighbour (so each interface is counted once),
 * the surface tension γ(t_i, t_j) times the facet area. types() is per cell; the
 * tension table is row-major nT×nT.
 *
 * The interface FORCE (gradFacetAreaSq) needs the per-cell half-edge mesh on the
 * device and is ported together with the planned methodological improvement.
 */
#ifndef VORFLOW_PHYSICS_INTERFACE_HPP
#define VORFLOW_PHYSICS_INTERFACE_HPP

#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"
#include "vorflow/tessellation_view.hpp"

namespace vor {
namespace physics {

/// Total interfacial energy Σ γ(t_i,t_j)·area over phase-boundary facets.
template <class Real>
Real interfaceEnergy(const TessellationView<Real>& view,
                     const Kokkos::View<int*, tpx::MemSpace>& types,
                     const Kokkos::View<Real*, tpx::MemSpace>& tension, int nTypes) {
  using Exec = tpx::ExecSpace;
  const int N = view.numCells();
  Real e = 0;
  Kokkos::parallel_reduce(
      "intf.energy", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        const int ti = types(i);
        if (ti == 0)
          return;
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
          const int j = view.facetNbr(f);
          if (j < 0 || j >= N)
            continue;
          const int tj = types(j);
          if (tj < ti) {
            Real ax = view.area(f, 0), ay = view.area(f, 1), az = view.area(f, 2);
            acc += tension(ti * nTypes + tj) * Kokkos::sqrt(ax * ax + ay * ay + az * az);
          }
        }
      },
      e);
  return e;
}

}  // namespace physics
}  // namespace vor

#endif  // VORFLOW_PHYSICS_INTERFACE_HPP
