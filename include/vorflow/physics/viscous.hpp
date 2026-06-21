/**
 * @file physics/viscous.hpp
 * \brief Viscous Navier-Stokes force over the published view (de-legacy Phase 2).
 *
 * Faithful port of NavierStokes::computeViscousForces, view-only (no engine). The
 * per-cell velocity gradient is the same Green-Gauss reconstruction as the legacy
 * computeGradients, but in the atomic-free GATHER form (each cell assembles its
 * own gradient from its facets and the reciprocal-facet transpose, plan §2.5):
 *   grad_i = (1/V_i) Σ_{f in i} [ v_i ⊗ dV_i[f] − v_{nbr} ⊗ dV[recip(f)] ].
 * The Newtonian stress τ_i = μ_i (∇v + ∇vᵀ) + (μ_b − 2/3 μ) (∇·v) I is then
 * gathered onto the force atomic-free, each cell writing only F_i:
 *   F_i += Σ_{f in i} (τ_{nbr} − τ_i) · dV_i[f].
 *
 * Stress/gradient are stored 9-per-cell (9*i + 3*k + m). dV is facetConnect; the
 * neighbour cell index is facetNeighbor (dense single-domain build). Boundary
 * facets (recip < 0) are skipped — the periodic dynamics has none.
 */
#ifndef VORFLOW_PHYSICS_VISCOUS_HPP
#define VORFLOW_PHYSICS_VISCOUS_HPP

#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"
#include "vorflow/tessellation_view.hpp"

namespace vor {
namespace physics {

/// Per-cell velocity gradient (Green-Gauss, atomic-free). vel/grad are device
/// Views sized 3*N and 9*N (row-major 3x3: 9*i + 3*k + m = d v_k / d x_m).
template <class Real>
void velocityGradient(const TessellationView<Real>& view,
                      const Kokkos::View<int*, tpx::MemSpace>& recip,
                      const Kokkos::View<int*, tpx::MemSpace>& cellOfFacet,
                      const Kokkos::View<Real*, tpx::MemSpace>& vel,
                      const Kokkos::View<Real*, tpx::MemSpace>& grad) {
  using Exec = tpx::ExecSpace;
  const int N = view.numCells();
  Kokkos::parallel_for(
      "visc.grad", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Real g[9];
        for (int t = 0; t < 9; ++t)
          g[t] = 0;
        const Real vi[3] = {vel(3 * i + 0), vel(3 * i + 1), vel(3 * i + 2)};
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
          const int r = recip(f);
          if (r < 0)
            continue;  // boundary facet
          const int nb = cellOfFacet(r);
          for (int p = 0; p < 3; ++p) {
            const Real vn = vel(3 * nb + p);
            for (int k = 0; k < 3; ++k)
              g[3 * p + k] += vi[p] * view.connect(f, k) - vn * view.connect(r, k);
          }
        }
        const Real invV = Real(1) / view.volume(i);
        for (int t = 0; t < 9; ++t)
          grad(9 * i + t) = g[t] * invV;
      });
}

/// Newtonian stress per cell from the gradient (τ stored 9-per-cell, symmetric).
template <class Real>
void viscousStress(const Kokkos::View<Real*, tpx::MemSpace>& grad,
                   const Kokkos::View<Real*, tpx::MemSpace>& visc,
                   const Kokkos::View<Real*, tpx::MemSpace>& bulkVisc,
                   const Kokkos::View<Real*, tpx::MemSpace>& stress) {
  using Exec = tpx::ExecSpace;
  const int N = static_cast<int>(visc.extent(0));
  Kokkos::parallel_for(
      "visc.stress", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real mu = visc(i), mub = bulkVisc(i);
        Real gk[9];
        for (int t = 0; t < 9; ++t)
          gk[t] = grad(9 * i + t);
        Real divv = gk[0] + gk[4] + gk[8];
        Real s[9];
        for (int k = 0; k < 3; ++k)
          for (int m = 0; m < 3; ++m)
            s[3 * k + m] = mu * (gk[3 * k + m] + gk[3 * m + k]);
        const Real lam = (mub - Real(2) / Real(3) * mu) * divv;
        s[0] += lam;
        s[4] += lam;
        s[8] += lam;
        for (int t = 0; t < 9; ++t)
          stress(9 * i + t) = s[t];
      });
}

/// Atomic-free gather of the viscous force, ADDED to `force` (size 3*N):
///   F_i += Σ_f (τ_nbr − τ_i) · dV_i[f].
template <class Real>
void addViscousForce(const TessellationView<Real>& view,
                     const Kokkos::View<Real*, tpx::MemSpace>& stress,
                     const Kokkos::View<Real*, tpx::MemSpace>& force) {
  using Exec = tpx::ExecSpace;
  const int N = view.numCells();
  Kokkos::parallel_for(
      "visc.force", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Real f[3] = {0, 0, 0};
        for (int ff = view.facetBegin(i); ff < view.facetEnd(i); ++ff) {
          const int nb = view.facetNbr(ff);
          if (nb < 0)
            continue;  // boundary facet
          for (int k = 0; k < 3; ++k)
            for (int m = 0; m < 3; ++m)
              f[k] +=
                  (stress(9 * nb + 3 * k + m) - stress(9 * i + 3 * k + m)) * view.connect(ff, m);
        }
        for (int k = 0; k < 3; ++k)
          force(3 * i + k) += f[k];
      });
}

/// Full viscous force: gradient -> stress -> gather (ADDED to force). Allocates
/// two 9*N scratch Views.
template <class Real>
void viscousForce(const TessellationView<Real>& view,
                  const Kokkos::View<int*, tpx::MemSpace>& recip,
                  const Kokkos::View<int*, tpx::MemSpace>& cellOfFacet,
                  const Kokkos::View<Real*, tpx::MemSpace>& vel,
                  const Kokkos::View<Real*, tpx::MemSpace>& visc,
                  const Kokkos::View<Real*, tpx::MemSpace>& bulkVisc,
                  const Kokkos::View<Real*, tpx::MemSpace>& force) {
  const int N = view.numCells();
  Kokkos::View<Real*, tpx::MemSpace> grad("visc.grad", (size_t)9 * N);
  Kokkos::View<Real*, tpx::MemSpace> stress("visc.stress", (size_t)9 * N);
  velocityGradient(view, recip, cellOfFacet, vel, grad);
  viscousStress(grad, visc, bulkVisc, stress);
  addViscousForce(view, stress, force);
}

}  // namespace physics
}  // namespace vor

#endif  // VORFLOW_PHYSICS_VISCOUS_HPP
