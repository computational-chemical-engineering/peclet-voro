/**
 * @file energy/wall.hpp
 * \brief The ONE seed-foot SDF wall chain (Voronoi methods plan, rung A3) + the wall interfacial
 * energy on the published view.
 *
 * A wall facet (pnbr == kBoundaryFacet) has no partner seed: its plane moves only through the
 * cell's own seed. The seed-foot model takes the wall as the tangent plane at the seed's foot
 * point on φ = 0, foot vector n(x) = −φ ∇φ/|∇φ|² (the Newton step to φ(x+n) = 0, valid for a
 * GENERAL level-set function, not only |∇φ| = 1). Its Jacobian is
 *     dn/dx = −û ûᵀ − (φ/|∇φ|) (I − û ûᵀ) H,   û = ∇φ/|∇φ|,  H = ∇²φ,
 * whose leading term is |∇φ|-INDEPENDENT (the |∇φ|² cancels), so a geometry gradient g_w
 * aggregated over the cell's wall planes routes to the seed as
 *     f_self += −(û·g_w) û − (φ/|∇φ|) H (g_w − (û·g_w) û).
 * Exact for a flat wall (H = 0, one plane); first-order for a curved wall (the clip approximates
 * the curve by several vertex-anchored planes, modelled as one effective seed-foot plane).
 *
 * The chain itself is sdf.hpp::sdfWallChain (the single source): sdf.hpp::addSdfWallForce, the
 * host and device mesh optimisers and the energy terms here all call it. (Before A3 the chain
 * lived in three copies; addSdfWallForce's carried an extra |∇φ| factor from the |∇φ| = 1
 * shortcut.)
 */
#ifndef PECLET_VORO_ENERGY_WALL_HPP
#define PECLET_VORO_ENERGY_WALL_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "peclet/voro/energy/route.hpp"
#include "peclet/voro/sdf.hpp"

namespace peclet::voro::energy {

/// Solid–fluid (wetting) energy E = Σ_c σ_s(t_c) Σ_{wall facets f of c} |A_f| over the published
/// view, with a PER-SPECIES wall tension σ_s(t) (`sigmaS`, nTypes; `types` per cell). A uniform
/// σ_s on a closed wall is a constant (the wall is tiled by the wall facets), so only the
/// difference between species does work — Young's angle emerges from σ_sg − σ_sl (plan D2).
/// Needs the facet-edge area-Jacobian CSR (build with withAreaGrad). Adds dE/dx (and dE/dw
/// under a weighted policy) into `force` / `forceW`. Returns E.
template <class Real, class Policy = Voronoi, class Sdf = NoSdf>
Real wallEnergyForce(const TessellationView<Real>& view,
                     const Kokkos::View<int*, peclet::core::MemSpace>& types,
                     const Kokkos::View<Real*, peclet::core::MemSpace>& sigmaS, const Sdf& sdf,
                     const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                     const Kokkos::View<Real*, peclet::core::MemSpace>& weight, Real L,
                     const Kokkos::View<Real*, peclet::core::MemSpace>& force,
                     const Kokkos::View<Real*, peclet::core::MemSpace>& forceW = {}) {
  using Exec = peclet::core::ExecSpace;
  const int N = view.numCells();
  Real E = 0;
  Kokkos::parallel_reduce(
      "energy.wall", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        const Real sig = sigmaS(types(i));
        if (sig == Real(0))
          return;
        Real gl[detail::kMaxLocalFacets * 3];
        const int b = view.facetBegin(i), nf = view.facetEnd(i) - b;
        if (nf > detail::kMaxLocalFacets)
          return;
        for (int k = 0; k < 3 * nf; ++k)
          gl[k] = Real(0);
        bool any = false;
        const bool fd = view.hasWallFD();  // exact wall-plane part (sdfWallFD) instead of the chain
        for (int f = b; f < b + nf; ++f) {
          if (view.facetNbr(f) != kBoundaryFacet)
            continue;
          any = true;
          const Real ax = view.area(f, 0), ay = view.area(f, 1), az = view.area(f, 2);
          acc += sig * Kokkos::sqrt(ax * ax + ay * ay + az * az);
          for (int e = view.edgeBegin(f); e < view.edgeEnd(f); ++e) {
            const int p = view.edgePartner(e) - b;
            if (fd && view.facetNbr(b + p) == kBoundaryFacet)
              continue;  // wall-plane dependence comes from the FD part below
            for (int c = 0; c < 3; ++c)
              gl[3 * p + c] += sig * view.areaGrad(e, c);
          }
        }
        if (any) {
          detail::routeCellGradient<Real, Policy>(view, i, gl, nf, pos, weight, L, sdf, force,
                                                  forceW);
          if (fd)
            for (int c = 0; c < 3; ++c)
              Kokkos::atomic_add(&force(3 * i + c), sig * view.wallDA(i, c));
        }
      },
      E);
  return E;
}

}  // namespace peclet::voro::energy

#endif  // PECLET_VORO_ENERGY_WALL_HPP
