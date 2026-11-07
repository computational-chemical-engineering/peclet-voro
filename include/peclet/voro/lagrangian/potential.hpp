/**
 * @file lagrangian/potential.hpp
 * \brief The potential of the Lagrangian power-cell method at prescribed volumes (M7 NHV-W,
 * DESIGN_NOTE §2.2, §4.2, §10.2): the facet (partition) gravity with its derivatives in positions
 * AND weights, and the adjoint that reduces any U(x, w) to Ũ(x) = U(x, w*(x)), w* the weights
 * that hold the cell volumes (lagrangian/weight_solve.hpp):
 *
 *   H π = −∂_w U   (H = ∂V/∂w, the TPFA Laplacian; Σ_i ∂_{w_i}U = 0),   ∇Ũ = ∂_x U + J_xᵀ π.
 *
 * Facet gravity (cells integrated in coordinates unwrapped at their generator, ρ′ = ρ − ρ_ref):
 *
 *   U_g = −Σ_i ρ′_i g·∫_{C_i} y dy.
 *
 * A facet f = (i, j) moves with the normal velocity (toward j)
 *   u_n(y) = [ (y − x_i)·δx_i + (x_j′ − y)·δx_j + (δw_i − δw_j)/2 ] / d,
 * the same on Voronoi and power facets (the radical plane is the zero set of
 * |y − x_i|² − w_i − |y − x_j|² + w_j, whose gradient is 2(x_j′ − x_i)). Moving f by u_n moves the
 * volume u_n dA from C_j to C_i, and a point of f has the coordinate y in i's frame and y + s in
 * j's, s = x_j − x_j′ the lattice shift of the min image. Hence, per facet in cell i's frame,
 *
 *   ∂U_g/∂x_i = Σ_f [ −(ρ′_i − ρ′_j) ∫_f (g·y)(y − x_i) dA + ρ′_j (g·s) A (c − x_i) ] / d,
 *   ∂U_g/∂w_i = Σ_f [ −(ρ′_i − ρ′_j) ∫_f (g·y) dA       + ρ′_j (g·s) A           ] / (2d),
 *
 * a GATHER over each cell's own facets (the j-side of the same facet gives the identical
 * expression), with ∫_f (g·y)(y − x_i) dA = (g·x_i) A c̃ + (S + A c̃c̃ᵀ) g from the facet's area,
 * generator-relative centroid c̃ and centroidal second moment S (lagrangian/facets.hpp).
 *
 * The two forms of DESIGN_NOTE §10.2 are the two values of ρ_ref:
 *  - walls ⟂ g (no facet crosses the seam along g, so g·s = 0): ρ_ref = 0 is the partition
 *    potential −Σ_a ρ_a g·∫_{Ω_a} y dy, §2.2 item 2 / §4.2, only unlike facets contribute;
 *  - periodic along g: ρ_ref = ρ̄ = Σ m / |Ω|; the force is periodic, the energy jumps when a
 *    generator wraps (the caller logs that), and the like facets that cross the seam carry the
 *    ρ′_j (g·s) term.
 * Walls are fixed: wall facets carry no derivative.
 */
#ifndef PECLET_VORO_LAGRANGIAN_POTENTIAL_HPP
#define PECLET_VORO_LAGRANGIAN_POTENTIAL_HPP

#include <Kokkos_Core.hpp>
#include <utility>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/operators.hpp"
#include "peclet/voro/lagrangian/weight_solve.hpp"

namespace peclet::voro::lagrangian {

/// Facet gravity U_g = −Σ_i (ρ_{t_i} − ρ_ref) g·∫_{C_i} y dy (see the file comment). `types` (N)
/// index `density` (nTypes); `g` the gravity vector; `L` the periodic box. ADDS ∂U_g/∂x into
/// `gradX` (3N) and ∂U_g/∂w into `gradW` (N; may be empty for a Voronoi tessellation) and returns
/// U_g.
template <class Real>
Real facetGravity(const CellFacets<Real>& F, const LView<Real>& pos,
                  const Kokkos::View<int*, peclet::core::MemSpace>& types,
                  const LView<Real>& density, const Real g[3], Real rhoRef, const Real L[3],
                  const LView<Real>& gradX, const LView<Real>& gradW) {
  const CellFacets<Real> G = F;
  const LView<Real> P = pos, rho = density, gx = gradX, gw = gradW;
  const Kokkos::View<int*, peclet::core::MemSpace> T = types;
  const bool hasW = gradW.extent(0) > 0;
  const Kokkos::Array<Real, 3> gv = {g[0], g[1], g[2]}, L3 = {L[0], L[1], L[2]};
  Real U = 0;
  Kokkos::parallel_reduce(
      "lagrangian.facetGravity", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, G.nCells),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        const Real ri = rho(T(i)) - rhoRef;
        const Real xi[3] = {P(3 * i), P(3 * i + 1), P(3 * i + 2)};
        const Real gxi = gv[0] * xi[0] + gv[1] * xi[1] + gv[2] * xi[2];
        // value: −ρ′_i g·(V_i x_i + ∫_{C_i} (y − x_i) dy)
        const Real V = G.volume(i);
        acc -= ri * V *
               (gxi + gv[0] * G.cellCentroid(3 * i) + gv[1] * G.cellCentroid(3 * i + 1) +
                gv[2] * G.cellCentroid(3 * i + 2));
        Real fx[3] = {0, 0, 0}, fw = 0;
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          const Real rj = rho(T(j)) - rhoRef;
          const Real k = ri - rj;
          Real d2 = 0, gs = 0, gc = 0;
          Real c[3], Sg[3];
          for (int a = 0; a < 3; ++a) {
            const Real dd = G.conn(3 * f + a);
            d2 += dd * dd;
            // lattice shift of the min image, s = x_j − (x_i + d): a multiple of the box,
            // rounded to it so that g·s carries no round-off (it is exactly 0 off the seam)
            gs += gv[a] * L3[a] * Kokkos::round((P(3 * j + a) - xi[a] - dd) / L3[a]);
            c[a] = G.centroid(3 * f + a);
            gc += gv[a] * c[a];
            Sg[a] = G.second(9 * f + 3 * a) * gv[0] + G.second(9 * f + 3 * a + 1) * gv[1] +
                    G.second(9 * f + 3 * a + 2) * gv[2];
          }
          const Real d = Kokkos::sqrt(d2);
          const Real A = G.area(f);
          if (k == Real(0) && gs == Real(0))
            continue;
          for (int a = 0; a < 3; ++a)
            fx[a] += (-k * ((gxi + gc) * A * c[a] + Sg[a]) + rj * gs * A * c[a]) / d;
          fw += (-k * A * (gxi + gc) + rj * gs * A) / (Real(2) * d);
        }
        for (int a = 0; a < 3; ++a)
          gx(3 * i + a) += fx[a];
        if (hasW)
          gw(i) += fw;
      },
      U);
  return U;
}

struct AdjointResult {
  int pcgIters = 0;
  double pcgRelRes = 0;
};

/// The adjoint reduction: π from H π = −∂_wU (H singular on constants: the round-off mean of the
/// right-hand side is removed and π_0 pinned as in the weight solve, then π is shifted to mean
/// zero), and `grad` = ∂_xU + J_xᵀπ = ∇Ũ. `dUdx` (3N), `dUdw` (N) in; `pi` (N), `grad` (3N) out.
/// `prm.pcgTol` is the relative residual of the H solve.
template <class Real>
AdjointResult reducedGradient(const CellFacets<Real>& F, const LView<Real>& dUdx,
                              const LView<Real>& dUdw, const LView<Real>& pi,
                              const LView<Real>& grad, const WeightSolveParams& prm = {}) {
  using Exec = peclet::core::ExecSpace;
  const int N = F.nCells;
  LView<Real> rhs("adj.rhs", N);
  {
    const LView<Real> b = dUdw, r = rhs;
    Real s = 0;
    Kokkos::parallel_reduce(
        "adj.mean", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc += b(i); }, s);
    const Real m = s / N;
    Kokkos::parallel_for(
        "adj.rhs", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i) { r(i) = i == 0 ? Real(0) : -(b(i) - m); });
  }
  const auto [it, res] = solvePinnedH(F, rhs, pi, prm);
  {
    const LView<Real> p = pi;
    Real s = 0;
    Kokkos::parallel_reduce(
        "adj.pimean", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc += p(i); }, s);
    const Real m = s / N;
    Kokkos::parallel_for(
        "adj.gauge", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { p(i) -= m; });
  }
  applyVolumeJacobianT(F, pi, grad);
  {
    const LView<Real> gr = grad, gx = dUdx;
    Kokkos::parallel_for(
        "adj.grad", Kokkos::RangePolicy<Exec>(0, 3 * N),
        KOKKOS_LAMBDA(const int q) { gr(q) += gx(q); });
  }
  return {it, res};
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_POTENTIAL_HPP
