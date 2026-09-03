/**
 * @file energy/route.hpp
 * \brief Rung A3: route a cell's per-facet geometry gradient to the seed DOFs on the PUBLISHED
 * view — the one combiner every energy term uses (no cell reconstruction).
 *
 * Given, for cell i, g_k = ∂E/∂n_k for each published facet k (SoA, local facet index), the
 * neighbour facets go through the plane policy's chain (Voronoi: ∓½ g; Power: the radical-plane
 * Jacobians from the current positions + weights) onto the cell's own seed and the neighbour
 * seed (atomic scatter), the SDF wall facets are aggregated and routed through the seed-foot wall
 * chain (energy/wall.hpp) onto the own seed, and the box facets carry no DOF dependence. This is
 * chainToDofs + addSdfWallForce expressed on TessellationView data.
 */
#ifndef PECLET_VORO_ENERGY_ROUTE_HPP
#define PECLET_VORO_ENERGY_ROUTE_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tessellation_view.hpp"

namespace peclet::voro::energy {

namespace detail {

/// Facets one cell may publish (CellBuilder::MAXF_TMP); bounds the per-thread gradient array.
constexpr int kMaxLocalFacets = 50;

/// Route cell i's per-facet gradient `gl` (3 per local facet, nf facets) to the DOFs. `pos` /
/// `weight` / `L` feed the weighted policy's chain (unused by Voronoi; pass empty views).
template <class Real, class Policy, class Sdf>
KOKKOS_INLINE_FUNCTION void routeCellGradient(
    const TessellationView<Real>& view, int i, const Real* gl, int nf,
    const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
    const Kokkos::View<Real*, peclet::core::MemSpace>& weight, Real L, const Sdf& sdf,
    const Kokkos::View<Real*, peclet::core::MemSpace>& force,
    const Kokkos::View<Real*, peclet::core::MemSpace>& forceW) {
  const int N = view.numCells();
  const int b = view.facetBegin(i);
  const Real Lh = Real(0.5) * L;
  Real fSelf[3] = {Real(0), Real(0), Real(0)};
  Real fwSelf = Real(0);
  double gw[3] = {0.0, 0.0, 0.0};
  Real sx = Real(0), sy = Real(0), sz = Real(0);
  if constexpr (Policy::kHasWeightDof) {
    sx = pos(3 * i);
    sy = pos(3 * i + 1);
    sz = pos(3 * i + 2);
  }
  for (int k = 0; k < nf; ++k) {
    const Real g[3] = {gl[3 * k], gl[3 * k + 1], gl[3 * k + 2]};
    if (g[0] == Real(0) && g[1] == Real(0) && g[2] == Real(0))
      continue;
    const int j = view.facetNbr(b + k);
    if (j == kBoundaryFacet) {
      gw[0] += g[0];
      gw[1] += g[1];
      gw[2] += g[2];
      continue;
    }
    if (j < 0 || j >= N)
      continue;  // box facet: no DOF dependence
    Real r[3] = {Real(0), Real(0), Real(0)}, rho = Real(0), cc = Real(0);
    if constexpr (Policy::kHasWeightDof) {
      const Real s[3] = {sx, sy, sz};
      for (int d = 0; d < 3; ++d) {
        Real rr = pos(3 * j + d) - s[d];
        rr = rr > Lh ? rr - L : (rr < -Lh ? rr + L : rr);
        r[d] = rr;
      }
      rho = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      cc = weight(i) - weight(j);
    }
    Real fs[3], fn[3], fws = Real(0), fwn = Real(0);
    Policy::template chain<Real>(g, r, rho, cc, fs, fn, fws, fwn);
    fSelf[0] += fs[0];
    fSelf[1] += fs[1];
    fSelf[2] += fs[2];
    fwSelf += fws;
    Kokkos::atomic_add(&force(3 * j), fn[0]);
    Kokkos::atomic_add(&force(3 * j + 1), fn[1]);
    Kokkos::atomic_add(&force(3 * j + 2), fn[2]);
    if constexpr (Policy::kHasWeightDof)
      if (forceW.extent(0) > 0)
        Kokkos::atomic_add(&forceW(j), fwn);
  }
  if constexpr (!std::is_same_v<Sdf, NoSdf>) {
    double fw[3] = {0.0, 0.0, 0.0};
    Real px = pos.extent(0) > 0 ? pos(3 * i) : Real(0);
    Real py = pos.extent(0) > 0 ? pos(3 * i + 1) : Real(0);
    Real pz = pos.extent(0) > 0 ? pos(3 * i + 2) : Real(0);
    sdfWallChain<Real>(sdf, px, py, pz, gw, fw);
    fSelf[0] += (Real)fw[0];
    fSelf[1] += (Real)fw[1];
    fSelf[2] += (Real)fw[2];
  } else {
    (void)sdf;
  }
  Kokkos::atomic_add(&force(3 * i), fSelf[0]);
  Kokkos::atomic_add(&force(3 * i + 1), fSelf[1]);
  Kokkos::atomic_add(&force(3 * i + 2), fSelf[2]);
  if constexpr (Policy::kHasWeightDof)
    if (forceW.extent(0) > 0)
      Kokkos::atomic_add(&forceW(i), fwSelf);
}

}  // namespace detail
}  // namespace peclet::voro::energy

#endif  // PECLET_VORO_ENERGY_ROUTE_HPP
