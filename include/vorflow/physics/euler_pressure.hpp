/**
 * @file physics/euler_pressure.hpp
 * \brief Reference physics consumer: compressible-Euler pressure force (Phase 4).
 *
 * The first physics module written against the published TessellationView ONLY —
 * it never touches the half-edge internals (it includes tessellation_view.hpp,
 * not voronoi.hpp / simulation.hpp), which is what proves the engine/physics
 * decoupling (plan §1, §4).
 *
 * Isothermal EOS: press_i = pressEq · volAvg / vol_i. The legacy force scatters
 *   F[nbr] += press_i · dV_i[f];  F[self] -= press_i · dV_i[f]
 * with atomics. The atomic-free GATHER form computed here (plan §2.5) makes each
 * cell assemble its OWN force from its facets and the reciprocal-facet transpose,
 * so every work item writes only F_i — zero atomics:
 *   F_i = Σ_{f in i, f->j} [ press_j · dV[recip(f)] − press_i · dV[f] ].
 * This is algebraically identical to the legacy scatter (each shared facet's two
 * half-contributions are gathered by the two cells independently).
 */
#ifndef VORFLOW_PHYSICS_EULER_PRESSURE_HPP
#define VORFLOW_PHYSICS_EULER_PRESSURE_HPP

#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"
#include "vorflow/tessellation_view.hpp"

namespace vor {
namespace physics {

/**
 * Atomic-free compressible-Euler pressure force over the published view.
 *
 * @param view        published tessellation (volumes, facet neighbours, dV in facetConnect).
 * @param recip       reciprocal-facet transpose (buildReciprocalMap), size numFacets.
 * @param cellOfFacet owning cell index per packed facet, size numFacets.
 * @param pressEq,volAvg  EOS constants (press_i = pressEq·volAvg/vol_i).
 * @param force       output, size 3·numCells (x-fastest 3*i+k). Overwritten.
 */
template <class Real>
void eulerPressureForce(const TessellationView<Real>& view,
                        const Kokkos::View<int*, tpx::MemSpace>& recip,
                        const Kokkos::View<int*, tpx::MemSpace>& cellOfFacet, Real pressEq,
                        Real volAvg, const Kokkos::View<Real*, tpx::MemSpace>& force) {
  using Exec = tpx::ExecSpace;
  const int N = view.numCells();
  Kokkos::parallel_for(
      "euler.pressure", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real pressI = pressEq * volAvg / view.volume(i);
        Real fx = 0, fy = 0, fz = 0;
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
          // self half-contribution (always present, including boundary facets)
          fx -= pressI * view.connect(f, 0);
          fy -= pressI * view.connect(f, 1);
          fz -= pressI * view.connect(f, 2);
          const int g = recip(f);
          if (g < 0)
            continue;  // boundary / no reciprocal
          const Real pressJ = pressEq * volAvg / view.volume(cellOfFacet(g));
          fx += pressJ * view.connect(g, 0);
          fy += pressJ * view.connect(g, 1);
          fz += pressJ * view.connect(g, 2);
        }
        force(3 * i + 0) = fx;
        force(3 * i + 1) = fy;
        force(3 * i + 2) = fz;
      });
}

}  // namespace physics
}  // namespace vor

#endif  // VORFLOW_PHYSICS_EULER_PRESSURE_HPP
