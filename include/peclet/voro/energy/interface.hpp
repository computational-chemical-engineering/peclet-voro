/**
 * @file energy/interface.hpp
 * \brief Rung A3: interfacial (surface-tension) energy between cells of different species on the
 * published view, with its exact gradient — no cell reconstruction.
 *
 *     E = Σ_{facets (i,j), t_i != t_j} σ(t_i, t_j) A_ij           (each interface counted once)
 *
 * Per cell the energy is taken as ½ σ A per interface facet (both cells see it), and the
 * gradient uses the facet-edge area Jacobians the build publishes (`withAreaGrad`): a facet's
 * area depends on its own plane and on the planes of the facets it shares an edge with, so
 * g_l = Σ_{interface facets f} ½ σ_f ∂A_f/∂n_l over the cell's facets l, then routed to the DOFs by
 * energy/route.hpp (Voronoi or Power chain; wall planes through the seed-foot chain).
 * The tension table is row-major nTypes × nTypes and must be symmetric.
 */
#ifndef PECLET_VORO_ENERGY_INTERFACE_HPP
#define PECLET_VORO_ENERGY_INTERFACE_HPP

#include <Kokkos_Core.hpp>

#include "peclet/voro/energy/route.hpp"
#include "peclet/voro/tessellation_view.hpp"

namespace peclet::voro::energy {

/// Interfacial energy + gradient. `types` per cell (N), `tension` nTypes×nTypes. Adds dE/dx into
/// `force` (3N) and dE/dw into `forceW` (N; only under a weighted policy). Returns E. The view
/// must carry the area-Jacobian CSR (build with withAreaGrad = true).
template <class Real, class Policy = Voronoi, class Sdf = NoSdf>
Real interfaceEnergyForce(const TessellationView<Real>& view,
                          const Kokkos::View<int*, peclet::core::MemSpace>& types,
                          const Kokkos::View<Real*, peclet::core::MemSpace>& tension, int nTypes,
                          const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                          const Kokkos::View<Real*, peclet::core::MemSpace>& weight, Real L,
                          const Kokkos::View<Real*, peclet::core::MemSpace>& force,
                          const Kokkos::View<Real*, peclet::core::MemSpace>& forceW = {},
                          const Sdf& sdf = Sdf{}) {
  using Exec = peclet::core::ExecSpace;
  const int N = view.numCells();
  Real E = 0;
  Kokkos::parallel_reduce(
      "energy.interface", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        Real gl[detail::kMaxLocalFacets * 3];
        const int b = view.facetBegin(i), nf = view.facetEnd(i) - b;
        if (nf > detail::kMaxLocalFacets)
          return;
        for (int k = 0; k < 3 * nf; ++k)
          gl[k] = Real(0);
        const int ti = types(i);
        bool any = false;
        for (int f = b; f < b + nf; ++f) {
          const int j = view.facetNbr(f);
          if (j < 0 || j >= N)
            continue;
          const int tj = types(j);
          if (tj == ti)
            continue;
          const Real sig = Real(0.5) * tension(ti * nTypes + tj);
          if (sig == Real(0))
            continue;
          any = true;
          const Real ax = view.area(f, 0), ay = view.area(f, 1), az = view.area(f, 2);
          acc += sig * Kokkos::sqrt(ax * ax + ay * ay + az * az);
          for (int e = view.edgeBegin(f); e < view.edgeEnd(f); ++e) {
            const int p = view.edgePartner(e) - b;
            for (int c = 0; c < 3; ++c)
              gl[3 * p + c] += sig * view.areaGrad(e, c);
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

#endif  // PECLET_VORO_ENERGY_INTERFACE_HPP
