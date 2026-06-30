/**
 * @file device/plane_policy.hpp
 * \brief Phase 3: plane-from-DOF policies + the chain-rule combiner.
 *
 * The ConvexCell geometry kernels are differentiated toward the foot-point normals n_k only
 * (dV/dn, area-vectors, dA/dn). A *policy* says how each plane's normal is built from the physics
 * degrees of freedom and supplies the chain — for a Voronoi cell the plane between seeds i and j is
 *   n_ij = ½ (p_j − p_i),   so   dn_ij/dp_i = −½ I,   dn_ij/dp_j = +½ I.
 * `chainToDofs<Policy>` then routes a per-plane geometry gradient g_k = dGeom/dn_k to the seed
 * DOFs: it sums the self-contributions onto the cell's own seed and emits, per plane, the
 * contribution onto the neighbour seed `pnbr[k]` (which the caller scatters into a global per-seed
 * array). Box planes (pnbr < 0) move with the seed and carry no DOF dependence, so they contribute
 * nothing.
 *
 * Adding a cell type (power/radical, additively-weighted spheres, SDF boundary) = a new policy with
 * its own buildNormal + chain; the geometry kernels and the combiner are unchanged.
 */
#ifndef VORFLOW_DEVICE_PLANE_POLICY_HPP
#define VORFLOW_DEVICE_PLANE_POLICY_HPP

#include <Kokkos_Core.hpp>

namespace vor {
namespace device {

/// Voronoi plane policy: n_ij = ½(p_j − p_i). The Jacobians are constant (∓½ I), so the chain is a
/// pure sign/scale split of the geometry gradient between the cell's own seed and the neighbour
/// seed.
struct Voronoi {
  /// Foot-point normal of the bisector to a neighbour at pNbr (periodic min-image, box length L).
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void buildNormal(const Real pSelf[3], const Real pNbr[3], Real L,
                                                 Real nOut[3]) {
    const Real Lh = Real(0.5) * L;
    for (int d = 0; d < 3; ++d) {
      Real r = pNbr[d] - pSelf[d];
      r = r > Lh ? r - L : (r < -Lh ? r + L : r);
      nOut[d] = Real(0.5) * r;
    }
  }

  /// Chain one plane's geometry gradient g = dGeom/dn_k to the seed DOFs:
  ///   dGeom/dp_self += (dn_k/dp_self)^T g = −½ g ,   dGeom/dp_nbr += (dn_k/dp_nbr)^T g = +½ g.
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void chain(const Real g[3], Real fSelf[3], Real fNbr[3]) {
    for (int d = 0; d < 3; ++d) {
      fSelf[d] = Real(-0.5) * g[d];
      fNbr[d] = Real(0.5) * g[d];
    }
  }
};

/// Chain a cell's per-plane geometry gradient (SoA gx/gy/gz, e.g. from geomVolumeGrad) to the
/// physics DOFs under `Policy`. Accumulates the derivative w.r.t. the cell's OWN seed into fSelf,
/// and writes the derivative w.r.t. each neighbour seed into (fnx,fny,fnz)[k] — the caller scatters
/// that into the global per-seed array at index pnbr[k]. Box planes (pnbr<0) contribute nothing
/// (zeroed).
template <class Policy, class Cell, class Real>
KOKKOS_INLINE_FUNCTION void chainToDofs(const Cell& c, const Real* gx, const Real* gy,
                                        const Real* gz, Real fSelf[3], Real* fnx, Real* fny,
                                        Real* fnz) {
  fSelf[0] = fSelf[1] = fSelf[2] = Real(0);
  for (int k = 0; k < c.np; ++k) {
    if (c.pnbr[k] < 0) {
      fnx[k] = fny[k] = fnz[k] = Real(0);
      continue;
    }  // box plane: no DOF dependence
    const Real g[3] = {gx[k], gy[k], gz[k]};
    Real fs[3], fn[3];
    Policy::template chain<Real>(g, fs, fn);
    fSelf[0] += fs[0];
    fSelf[1] += fs[1];
    fSelf[2] += fs[2];
    fnx[k] = fn[0];
    fny[k] = fn[1];
    fnz[k] = fn[2];
  }
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_PLANE_POLICY_HPP
