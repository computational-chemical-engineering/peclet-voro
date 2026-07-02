/**
 * @file device/plane_policy.hpp
 * \brief Phase 3: plane-from-DOF policies + the chain-rule combiner.
 *
 * The ConvexCell geometry kernels are differentiated toward the foot-point normals n_k only
 * (dV/dn, area-vectors, dA/dn). A *policy* says how each plane's normal is built from the physics
 * degrees of freedom and supplies the chain:
 *   - Voronoi cell: the plane between seeds i and j is the perpendicular bisector,
 *       n_ij = ½ (p_j − p_i),   so   dn_ij/dp_i = −½ I,   dn_ij/dp_j = +½ I,   no weight DOF.
 *   - Power/Laguerre cell: the plane is the *radical* plane. With r = p_j − p_i, ρ = |r|²,
 *     c = w_i − w_j, α = (ρ+c)/(2ρ), the foot-point normal is n_ij = α r (offset d = ½(ρ+c)).
 *     Both positions AND weights are DOFs: dn/dr = α I − (c/ρ²) r rᵀ and dn/dw_{i,j} = ±r/(2ρ).
 *
 * `chainToDofs<Policy>` routes a per-plane geometry gradient g_k = dGeom/dn_k to the seed DOFs: it
 * sums the self-contributions onto the cell's own seed (position fSelf, weight fwSelf) and emits,
 * per plane, the contribution onto the neighbour seed `pnbr[k]` (position fnx/fny/fnz, weight fwn),
 * which the caller scatters into the global per-seed arrays. Box planes (pnbr == -1) move with the
 * seed and carry no DOF dependence; SDF walls (pnbr == -2) carry a *self-only* DOF dependence
 * handled by a separate SDF chain (see sdf.hpp) — both are skipped here.
 *
 * Each policy also supplies `planeFromNeighbour` (the FORWARD clip: neighbour → half-space
 * {x : pdir·x ≤ off}) and `buildNormal` (the stored foot-point normal), so the clip/re-eval sites
 * are policy-driven rather than hard-wiring the Voronoi bisector.
 *
 * Adding a cell type (power/radical, additively-weighted spheres, SDF boundary) = a new policy with
 * its own planeFromNeighbour/buildNormal + chain; the geometry kernels and the combiner are
 * unchanged.
 */
#ifndef PECLET_VORO_PLANE_POLICY_HPP
#define PECLET_VORO_PLANE_POLICY_HPP

#include <Kokkos_Core.hpp>

namespace peclet::voro {

/// Voronoi plane policy: n_ij = ½(p_j − p_i). The Jacobians are constant (∓½ I), so the chain is a
/// pure sign/scale split of the geometry gradient between the cell's own seed and the neighbour
/// seed. No weight degree of freedom.
struct Voronoi {
  /// Whether cells carry a per-seed weight DOF (Laguerre). Voronoi cells do not.
  static constexpr bool kHasWeightDof = false;

  /// Half-space offset for the forward clip {x : r·x ≤ off} given the min-image relative vector
  /// r = p_j − p_i (pdir == r for both Voronoi and Power). Voronoi: off = ½|r|² (weights ignored).
  template <class Real>
  KOKKOS_INLINE_FUNCTION static Real offsetFromRel(const Real r[3], Real wSelf, Real wNbr) {
    (void)wSelf;
    (void)wNbr;
    return Real(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  }

  /// FORWARD clip: half-space {x : pdir·x ≤ off} cutting off the neighbour at pNbr. For a Voronoi
  /// bisector pdir = r (min-image p_j − p_i), off = ½|r|². Weights are ignored.
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void planeFromNeighbour(const Real pSelf[3], const Real pNbr[3],
                                                        Real wSelf, Real wNbr, Real L, Real pdir[3],
                                                        Real& off) {
    const Real Lh = Real(0.5) * L;
    for (int d = 0; d < 3; ++d) {
      Real r = pNbr[d] - pSelf[d];
      r = r > Lh ? r - L : (r < -Lh ? r + L : r);
      pdir[d] = r;
    }
    off = offsetFromRel(pdir, wSelf, wNbr);
  }

  /// Foot-point normal of the bisector to a neighbour at pNbr (periodic min-image, box length L).
  /// n = (off/|pdir|²) pdir = ½ r. Weights are ignored.
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void buildNormal(const Real pSelf[3], const Real pNbr[3], Real wSelf,
                                                 Real wNbr, Real L, Real nOut[3]) {
    Real pdir[3], off;
    planeFromNeighbour(pSelf, pNbr, wSelf, wNbr, L, pdir, off);
    const Real l2 = pdir[0] * pdir[0] + pdir[1] * pdir[1] + pdir[2] * pdir[2];
    const Real a = l2 > Real(0) ? off / l2 : Real(0);
    for (int d = 0; d < 3; ++d) nOut[d] = a * pdir[d];
  }

  /// Chain one plane's geometry gradient g = dGeom/dn_k to the seed DOFs. The Voronoi Jacobians are
  /// constant (∓½ I) so r/rho/c are unused; the weight outputs are zero (no weight DOF):
  ///   dGeom/dp_self += −½ g ,   dGeom/dp_nbr += +½ g.
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void chain(const Real g[3], const Real r[3], Real rho, Real c,
                                           Real fSelf[3], Real fNbr[3], Real& fwSelf, Real& fwNbr) {
    (void)r;
    (void)rho;
    (void)c;
    for (int d = 0; d < 3; ++d) {
      fSelf[d] = Real(-0.5) * g[d];
      fNbr[d] = Real(0.5) * g[d];
    }
    fwSelf = Real(0);
    fwNbr = Real(0);
  }
};

/// Power / Laguerre plane policy: the radical plane between weighted seeds. With r = p_j − p_i,
/// ρ = |r|², c = w_i − w_j, α = (ρ+c)/(2ρ), the foot-point normal is n = α r and the half-space
/// offset is d = ½(ρ+c). Unlike the bisector, d can be negative (the seed can lie OUTSIDE its own
/// cell) — the generic ConvexCell::clip already tolerates that. Both positions and weights are DOFs.
struct Power {
  static constexpr bool kHasWeightDof = true;

  /// Half-space offset for the forward clip {x : r·x ≤ off} given the min-image relative vector
  /// r = p_j − p_i (pdir == r). Power: off = d = ½(|r|² + w_self − w_nbr), which can be negative.
  template <class Real>
  KOKKOS_INLINE_FUNCTION static Real offsetFromRel(const Real r[3], Real wSelf, Real wNbr) {
    return Real(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + wSelf - wNbr);
  }

  /// FORWARD clip: pdir = r (min-image), off = d = ½(|r|² + w_self − w_nbr). ConvexCell::clip then
  /// stores n = (off/|pdir|²) pdir = α r.
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void planeFromNeighbour(const Real pSelf[3], const Real pNbr[3],
                                                        Real wSelf, Real wNbr, Real L, Real pdir[3],
                                                        Real& off) {
    const Real Lh = Real(0.5) * L;
    for (int d = 0; d < 3; ++d) {
      Real r = pNbr[d] - pSelf[d];
      r = r > Lh ? r - L : (r < -Lh ? r + L : r);
      pdir[d] = r;
    }
    off = offsetFromRel(pdir, wSelf, wNbr);
  }

  /// Foot-point normal n = (off/|pdir|²) pdir = α r.
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void buildNormal(const Real pSelf[3], const Real pNbr[3], Real wSelf,
                                                 Real wNbr, Real L, Real nOut[3]) {
    Real pdir[3], off;
    planeFromNeighbour(pSelf, pNbr, wSelf, wNbr, L, pdir, off);
    const Real l2 = pdir[0] * pdir[0] + pdir[1] * pdir[1] + pdir[2] * pdir[2];
    const Real a = l2 > Real(0) ? off / l2 : Real(0);
    for (int d = 0; d < 3; ++d) nOut[d] = a * pdir[d];
  }

  /// Chain g = dGeom/dn_k to the seed DOFs. n = α r, α = ½ + c/(2ρ), c = w_self − w_nbr.
  ///   Position: J = ∂n/∂r = α I − (c/ρ²) r rᵀ, ∂r/∂p_nbr = +I ⇒
  ///     dGeom/dp_nbr = α g − (c/ρ²)(r·g) r ,   dGeom/dp_self = −(that).
  ///   Weight:  ∂n/∂w_self = r/(2ρ), ∂n/∂w_nbr = −r/(2ρ) ⇒
  ///     dGeom/dw_self = (g·r)/(2ρ) ,   dGeom/dw_nbr = −(that).
  /// c = 0 recovers the Voronoi split (∓½ g, zero weight force).
  template <class Real>
  KOKKOS_INLINE_FUNCTION static void chain(const Real g[3], const Real r[3], Real rho, Real c,
                                           Real fSelf[3], Real fNbr[3], Real& fwSelf, Real& fwNbr) {
    const Real invRho = rho > Real(0) ? Real(1) / rho : Real(0);
    const Real alpha = Real(0.5) + Real(0.5) * c * invRho;
    const Real rg = r[0] * g[0] + r[1] * g[1] + r[2] * g[2];
    const Real kk = c * invRho * invRho * rg;  // (c/ρ²)(r·g)
    for (int d = 0; d < 3; ++d) {
      const Real jn = alpha * g[d] - kk * r[d];  // dGeom/dp_nbr component
      fNbr[d] = jn;
      fSelf[d] = -jn;
    }
    const Real w = Real(0.5) * invRho * rg;  // (g·r)/(2ρ)
    fwSelf = w;
    fwNbr = -w;
  }
};

/// Chain a cell's per-plane geometry gradient (SoA gx/gy/gz, e.g. from geomVolumeGrad) to the
/// physics DOFs under `Policy`. Accumulates the derivative w.r.t. the cell's OWN seed into fSelf
/// (position) and fwSelf (weight), and writes the derivative w.r.t. each neighbour seed into
/// (fnx,fny,fnz)[k] (position) and fwn[k] (weight) — the caller scatters those into the global
/// per-seed arrays at index pnbr[k]. Box/SDF planes (pnbr < 0) contribute nothing here (zeroed);
/// SDF-wall self-DOF forces are added by the SDF chain in sdf.hpp.
///
/// `seed`/`pos`/`wSelf`/`weight`/`L` recompute each plane's (r, ρ, c) for weighted policies; they
/// are untouched for Voronoi (kHasWeightDof == false), so a caller with no weights may pass nullptr
/// for `pos`/`weight` and 0 for `wSelf` on the Voronoi path.
template <class Policy, class Cell, class Real>
KOKKOS_INLINE_FUNCTION void chainToDofs(const Cell& c, const Real seed[3], const Real* pos,
                                        Real wSelf, const Real* weight, Real L, const Real* gx,
                                        const Real* gy, const Real* gz, Real fSelf[3], Real& fwSelf,
                                        Real* fnx, Real* fny, Real* fnz, Real* fwn) {
  fSelf[0] = fSelf[1] = fSelf[2] = Real(0);
  fwSelf = Real(0);
  const Real Lh = Real(0.5) * L;
  for (int k = 0; k < c.np; ++k) {
    const int j = c.pnbr[k];
    if (j < 0) {  // box (−1) / SDF wall (−2): no particle-DOF dependence here
      fnx[k] = fny[k] = fnz[k] = Real(0);
      fwn[k] = Real(0);
      continue;
    }
    const Real g[3] = {gx[k], gy[k], gz[k]};
    Real r[3] = {Real(0), Real(0), Real(0)}, rho = Real(0), cc = Real(0);
    if constexpr (Policy::kHasWeightDof) {
      for (int d = 0; d < 3; ++d) {
        Real rr = pos[3 * j + d] - seed[d];
        rr = rr > Lh ? rr - L : (rr < -Lh ? rr + L : rr);
        r[d] = rr;
      }
      rho = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      cc = wSelf - weight[j];
    }
    Real fs[3], fn[3], fws = Real(0), fwn1 = Real(0);
    Policy::template chain<Real>(g, r, rho, cc, fs, fn, fws, fwn1);
    fSelf[0] += fs[0];
    fSelf[1] += fs[1];
    fSelf[2] += fs[2];
    fwSelf += fws;
    fnx[k] = fn[0];
    fny[k] = fn[1];
    fnz[k] = fn[2];
    fwn[k] = fwn1;
  }
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_PLANE_POLICY_HPP
