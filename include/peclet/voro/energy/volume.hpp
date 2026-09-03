/**
 * @file energy/volume.hpp
 * \brief Rung A3: per-cell volume energies e_i(V_i) on the published view — target volume,
 * log-barrier and the ideal-gas free energy the mesh optimiser validated — and their gradient
 * through dV/dn.
 *
 * Any e(V) routes the same way: g_k = e'(V_i) ∂V_i/∂n_k. The build publishes ∂V_i/∂r_k
 * (facetConnect, r_k = 2 n_k the connector), so ∂V/∂n_k = 2·facetConnect for a neighbour facet;
 * a wall facet moves only by translation under the seed-foot model, ∂V/∂n_wall = its outward
 * area vector (facetArea) — exactly the optimiser's wall block.
 */
#ifndef PECLET_VORO_ENERGY_VOLUME_HPP
#define PECLET_VORO_ENERGY_VOLUME_HPP

#include <Kokkos_Core.hpp>

#include "peclet/voro/energy/route.hpp"
#include "peclet/voro/tessellation_view.hpp"

namespace peclet::voro::energy {

/// e(V) = (V/Vref − 1)²: e' = 2 (V/Vref − 1)/Vref.
template <class Real>
KOKKOS_INLINE_FUNCTION Real dTargetVolume(Real V, Real Vref) {
  return Real(2) * (V / Vref - Real(1)) / Vref;
}
/// e(V) = −μ log(V/Vref) (log-barrier against collapse): e' = −μ/V.
template <class Real>
KOKKOS_INLINE_FUNCTION Real dLogBarrier(Real V, Real mu) {
  return V > Real(0) ? -mu / V : Real(0);
}
/// e(V) = −Vref log V (ideal-gas free energy; equilibrium at equal pressure Vref/V): e' = −Vref/V.
template <class Real>
KOKKOS_INLINE_FUNCTION Real dFreeEnergy(Real V, Real Vref) {
  return V > Real(0) ? -Vref / V : Real(0);
}

/// Gradient of Σ_i e_i(V_i) for a caller-supplied e'(V_i) per cell (`dEdV`, N). Adds into
/// `force` / `forceW`. (The energy value itself is the caller's Σ e_i — it needs no geometry.)
template <class Real, class Policy = Voronoi, class Sdf = NoSdf>
void volumeGradientForce(const TessellationView<Real>& view,
                         const Kokkos::View<Real*, peclet::core::MemSpace>& dEdV,
                         const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                         const Kokkos::View<Real*, peclet::core::MemSpace>& weight, Real L,
                         const Kokkos::View<Real*, peclet::core::MemSpace>& force,
                         const Kokkos::View<Real*, peclet::core::MemSpace>& forceW = {},
                         const Sdf& sdf = Sdf{}) {
  using Exec = peclet::core::ExecSpace;
  const int N = view.numCells();
  Kokkos::parallel_for(
      "energy.volume", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real de = dEdV(i);
        if (de == Real(0))
          return;
        Real gl[detail::kMaxLocalFacets * 3];
        const int b = view.facetBegin(i), nf = view.facetEnd(i) - b;
        if (nf > detail::kMaxLocalFacets)
          return;
        for (int k = 0; k < nf; ++k) {
          const int f = b + k;
          if (view.facetNbr(f) == kBoundaryFacet) {
            for (int c = 0; c < 3; ++c)
              gl[3 * k + c] = de * view.area(f, c);
          } else {
            for (int c = 0; c < 3; ++c)
              gl[3 * k + c] = Real(2) * de * view.connect(f, c);
          }
        }
        detail::routeCellGradient<Real, Policy>(view, i, gl, nf, pos, weight, L, sdf, force,
                                                forceW);
      });
}

}  // namespace peclet::voro::energy

#endif  // PECLET_VORO_ENERGY_VOLUME_HPP
