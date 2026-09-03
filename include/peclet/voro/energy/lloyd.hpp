/**
 * @file energy/lloyd.hpp
 * \brief Track B, rung B1: the CENTROIDAL (Lloyd / CVT) energy on the published view.
 *
 *     E = Σ_i ∫_{V_i} |y − x_i|² dy
 *
 * Its gradient is exact and needs no shape derivative: because |y − x_i|² = |y − x_j|² on the
 * bisector between i and j, the boundary terms of the shape variation cancel pairwise
 * (Du–Faber–Gunzburger), leaving
 *     ∂E/∂x_i = 2 V_i (x_i − c_i),   c_i the cell centroid.
 * Minimising it moves every seed to its cell's centroid — the Lloyd iteration — which is exactly
 * what removes the SKEWNESS (face centroid off the seed connector) that limits the two-point
 * operators of track C (fv/operators.hpp): grid generation and solver accuracy are one problem.
 *
 * The centroid comes from the facet CSR alone: the cell is the union of pyramids from the seed
 * over its faces, pyramid f has volume h_f A_f / 3 and centroid ¾ c_f (c_f the face centroid,
 * seed-relative, c_f = r_f − dV_f |r_f| / |A_f| from the published dV/dr). The ENERGY value needs
 * the faces' second moments ∫_face |s|² dA about the foot point (a pyramid's ∫|y|² = h (h² A +
 * M2)/5), published as `facetMoment2` when the build is asked for it (`withMoments`); without it
 * only the gradient is available (enough for descent, not for a line search on E).
 */
#ifndef PECLET_VORO_ENERGY_LLOYD_HPP
#define PECLET_VORO_ENERGY_LLOYD_HPP

#include <Kokkos_Core.hpp>

#include "peclet/voro/energy/route.hpp"
#include "peclet/voro/tessellation_view.hpp"

namespace peclet::voro::energy {

/// Cell centroids (3N, world = seed + relative) from the facet CSR. `pos` supplies the seeds.
template <class Real>
void cellCentroids(const TessellationView<Real>& view,
                   const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                   const Kokkos::View<Real*, peclet::core::MemSpace>& out) {
  using Exec = peclet::core::ExecSpace;
  const int N = view.numCells();
  Kokkos::parallel_for(
      "energy.centroid", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Real m[3] = {0, 0, 0}, V = 0;
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
          const Real ax = view.area(f, 0), ay = view.area(f, 1), az = view.area(f, 2);
          const Real A = Kokkos::sqrt(ax * ax + ay * ay + az * az);
          if (!(A > Real(0)))
            continue;
          const Real rx = view.connVec(f, 0), ry = view.connVec(f, 1), rz = view.connVec(f, 2);
          const Real rl = Kokkos::sqrt(rx * rx + ry * ry + rz * rz);
          const Real h = Real(0.5) * rl;  // foot distance (conn = 2 n)
          const Real s = rl / A;
          const Real cx = rx - view.connect(f, 0) * s, cy = ry - view.connect(f, 1) * s,
                     cz = rz - view.connect(f, 2) * s;  // face centroid, seed-relative
          const Real Vf = h * A / Real(3);
          V += Vf;
          m[0] += Vf * Real(0.75) * cx;
          m[1] += Vf * Real(0.75) * cy;
          m[2] += Vf * Real(0.75) * cz;
        }
        const Real iv = V > Real(0) ? Real(1) / V : Real(0);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = pos(3 * i + c) + m[c] * iv;
      });
}

/// Lloyd energy gradient dE/dx = 2 V_i (x_i − c_i) added into `force` (3N); returns E when the
/// view carries facet second moments (`withMoments`), else −1 (gradient only).
template <class Real>
Real lloydEnergyForce(const TessellationView<Real>& view,
                      const Kokkos::View<Real*, peclet::core::MemSpace>& pos, Real gamma,
                      const Kokkos::View<Real*, peclet::core::MemSpace>& force) {
  using Exec = peclet::core::ExecSpace;
  const int N = view.numCells();
  const bool haveM2 = view.hasMoments();
  Real E = 0;
  Kokkos::parallel_reduce(
      "energy.lloyd", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        Real m[3] = {0, 0, 0}, V = 0, e = 0;
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
          const Real ax = view.area(f, 0), ay = view.area(f, 1), az = view.area(f, 2);
          const Real A = Kokkos::sqrt(ax * ax + ay * ay + az * az);
          if (!(A > Real(0)))
            continue;
          const Real rx = view.connVec(f, 0), ry = view.connVec(f, 1), rz = view.connVec(f, 2);
          const Real rl = Kokkos::sqrt(rx * rx + ry * ry + rz * rz);
          const Real h = Real(0.5) * rl;
          const Real s = rl / A;
          const Real cx = rx - view.connect(f, 0) * s, cy = ry - view.connect(f, 1) * s,
                     cz = rz - view.connect(f, 2) * s;
          const Real Vf = h * A / Real(3);
          V += Vf;
          m[0] += Vf * Real(0.75) * cx;
          m[1] += Vf * Real(0.75) * cy;
          m[2] += Vf * Real(0.75) * cz;
          if (haveM2)
            e += h * (h * h * A + view.moment2(f)) / Real(5);
        }
        // gradient: 2 V (x − c) = −2 m  (m = V (c − x))
        for (int c = 0; c < 3; ++c)
          Kokkos::atomic_add(&force(3 * i + c), Real(-2) * gamma * m[c]);
        acc += gamma * e;
      },
      E);
  return haveM2 ? E : Real(-1);
}

}  // namespace peclet::voro::energy

#endif  // PECLET_VORO_ENERGY_LLOYD_HPP
