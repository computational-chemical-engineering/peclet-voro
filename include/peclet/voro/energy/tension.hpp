/**
 * @file energy/tension.hpp
 * \brief Track B, rung B1: uniform facet tension E = σ Σ_faces A_f (every interior face once) —
 * the "roundness" term of grid generation (surface tension on all faces shrinks total face area,
 * which rounds the cells) — on the published view through the facet-edge area Jacobians, routed
 * by energy/route.hpp. Wall facets are not included (their area is the wetting term's).
 */
#ifndef PECLET_VORO_ENERGY_TENSION_HPP
#define PECLET_VORO_ENERGY_TENSION_HPP

#include <Kokkos_Core.hpp>

#include "peclet/voro/energy/route.hpp"
#include "peclet/voro/tessellation_view.hpp"

namespace peclet::voro::energy {

template <class Real, class Policy = Voronoi, class Sdf = NoSdf>
Real facetTensionEnergyForce(const TessellationView<Real>& view, Real sigma,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& weight, Real L,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& force,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& forceW = {},
                             const Sdf& sdf = Sdf{}) {
  using Exec = peclet::core::ExecSpace;
  const int N = view.numCells();
  Real E = 0;
  Kokkos::parallel_reduce(
      "energy.tension", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        Real gl[detail::kMaxLocalFacets * 3];
        const int b = view.facetBegin(i), nf = view.facetEnd(i) - b;
        if (nf > detail::kMaxLocalFacets)
          return;
        for (int k = 0; k < 3 * nf; ++k)
          gl[k] = Real(0);
        const Real half = Real(0.5) * sigma;  // each interior face is seen from both cells
        bool any = false;
        for (int f = b; f < b + nf; ++f) {
          const int j = view.facetNbr(f);
          if (j < 0 || j >= N)
            continue;
          any = true;
          const Real ax = view.area(f, 0), ay = view.area(f, 1), az = view.area(f, 2);
          acc += half * Kokkos::sqrt(ax * ax + ay * ay + az * az);
          for (int e = view.edgeBegin(f); e < view.edgeEnd(f); ++e) {
            const int p = view.edgePartner(e) - b;
            for (int c = 0; c < 3; ++c)
              gl[3 * p + c] += half * view.areaGrad(e, c);
          }
        }
        if (any)
          detail::routeCellGradient<Real, Policy>(view, i, gl, nf, pos, weight, L, sdf, force,
                                                  forceW);
      },
      E);
  return E;
}

}  // namespace peclet::voro::energy

#endif  // PECLET_VORO_ENERGY_TENSION_HPP
