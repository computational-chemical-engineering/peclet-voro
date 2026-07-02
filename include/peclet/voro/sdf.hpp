/**
 * @file device/sdf.hpp
 * \brief Device-callable signed-distance geometry + per-cell SDF boundary clip.
 *
 * Embeds a solid geometry into the tessellation: a cell that would reach into the
 * solid is clipped by a plane located at the sdf = 0 surface with normal ∇sdf
 * (suite convention: sdf < 0 inside solid, > 0 in fluid, ∇sdf outward). This is a
 * faithful KOKKOS_FUNCTION port of the legacy
 * CellComplex::clipCellAgainstBoundary + SignedDistanceBoundary::closestPoint, on
 * the suite's shared geometry (core peclet::core::geom):
 *   - analytic providers (SdfSphere / SdfBox / SdfHollowCylinder) port the
 *     peclet::core::geom::{Sphere,Box,HollowCylinder} eval formulas verbatim;
 *   - SdfGrid trilinearly evaluates a device-resident sampled field (any geometry,
 *     analytic baked via peclet::core::geom::sample or loaded from a VTI via
 * peclet::core::geom::readVti). A provider is any POD with `eval(x,y,z)` and `gradH()`; gradients
 * are central differences (so a host peclet::core::geom adapter and the device agree bit-for-bit).
 *
 * Core header: Kokkos + the cutter, no physics.
 */
#ifndef PECLET_VORO_SDF_HPP
#define PECLET_VORO_SDF_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"

namespace peclet::voro {

/// Neighbour id stamped on a facet produced by an SDF cut (a wall). Distinct from
/// the initial-cuboid boundary (-1); published as facetNbr < 0 either way.
constexpr int kBoundaryFacet = -2;

/// Sentinel "no geometry" provider — the default; the clip stage is skipped.
struct NoSdf {};

/// Solid ball (negative inside). eval ported from peclet::core::geom::Sphere.
template <class Real>
struct SdfSphere {
  Real cx = 0, cy = 0, cz = 0, radius = 1;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real dx = x - cx, dy = y - cy, dz = z - cz;
    return Kokkos::sqrt(dx * dx + dy * dy + dz * dz) - radius;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

/// Axis-aligned solid box of half-extents (hx,hy,hz). eval ported from peclet::core::geom::Box.
template <class Real>
struct SdfBox {
  Real cx = 0, cy = 0, cz = 0, hx = Real(0.5), hy = Real(0.5), hz = Real(0.5);
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real qx = Kokkos::fabs(x - cx) - hx, qy = Kokkos::fabs(y - cy) - hy,
         qz = Kokkos::fabs(z - cz) - hz;
    Real ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0, oz = qz > 0 ? qz : 0;
    Real outside = Kokkos::sqrt(ox * ox + oy * oy + oz * oz);
    Real mx = qx > qy ? qx : qy;
    mx = mx > qz ? mx : qz;
    Real inside = mx < 0 ? mx : 0;
    return outside + inside;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

/// Solid hollow cylinder (tube wall) about `axis`. eval ported from
/// peclet::core::geom::HollowCylinder.
template <class Real>
struct SdfHollowCylinder {
  Real cx = 0, cy = 0, cz = 0, rOuter = 1, rInner = Real(0.5), height = 1;
  int axis = 2;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real d[3] = {x - cx, y - cy, z - cz};
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
    Real r = Kokkos::sqrt(d[a0] * d[a0] + d[a1] * d[a1]);
    Real zz = d[axis];
    Real m = r - rOuter;
    Real b = rInner - r;
    if (b > m)
      m = b;
    Real c = Kokkos::fabs(zz) - height * Real(0.5);
    if (c > m)
      m = c;
    return m;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

/// Device-resident sampled SDF (trilinear; x-fastest), the universal provider for
/// analytic-baked or VTI geometry. eval ported from peclet::core::geom::GridSdf.
template <class Real>
struct SdfGrid {
  Kokkos::View<const float*, peclet::core::MemSpace> values;  // i + nx*(j + ny*k)
  int nx = 0, ny = 0, nz = 0;
  Real ox = 0, oy = 0, oz = 0, sx = 1, sy = 1, sz = 1;
  KOKKOS_INLINE_FUNCTION Real at(int i, int j, int k) const {
    return static_cast<Real>(values(i + nx * (j + ny * k)));
  }
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real p[3] = {x, y, z};
    Real o[3] = {ox, oy, oz}, s[3] = {sx, sy, sz};
    int dims[3] = {nx, ny, nz};
    int i0[3], i1[3];
    Real f[3];
    for (int d = 0; d < 3; ++d) {
      Real c = (p[d] - o[d]) / s[d];
      Real hi = static_cast<Real>(dims[d] - 1);
      c = c < 0 ? Real(0) : (c > hi ? hi : c);
      i0[d] = static_cast<int>(Kokkos::floor(c));
      i1[d] = i0[d] + 1 < dims[d] ? i0[d] + 1 : dims[d] - 1;
      f[d] = c - static_cast<Real>(i0[d]);
    }
    Real c000 = at(i0[0], i0[1], i0[2]), c100 = at(i1[0], i0[1], i0[2]);
    Real c010 = at(i0[0], i1[1], i0[2]), c110 = at(i1[0], i1[1], i0[2]);
    Real c001 = at(i0[0], i0[1], i1[2]), c101 = at(i1[0], i0[1], i1[2]);
    Real c011 = at(i0[0], i1[1], i1[2]), c111 = at(i1[0], i1[1], i1[2]);
    Real c00 = c000 * (1 - f[0]) + c100 * f[0];
    Real c10 = c010 * (1 - f[0]) + c110 * f[0];
    Real c01 = c001 * (1 - f[0]) + c101 * f[0];
    Real c11 = c011 * (1 - f[0]) + c111 * f[0];
    Real c0 = c00 * (1 - f[1]) + c10 * f[1];
    Real c1 = c01 * (1 - f[1]) + c11 * f[1];
    return c0 * (1 - f[2]) + c1 * f[2];
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const {
    Real m = sx < sy ? sx : sy;
    m = m < sz ? m : sz;
    return Real(0.25) * m;
  }
};

/// Central-difference gradient of any provider with eval() (matches peclet::core::geom::gradient).
template <class Real, class Sdf>
KOKKOS_INLINE_FUNCTION void sdfGradient(const Sdf& s, Real x, Real y, Real z, Real g[3]) {
  const Real h = s.gradH();
  g[0] = (s.eval(x + h, y, z) - s.eval(x - h, y, z)) / (2 * h);
  g[1] = (s.eval(x, y + h, z) - s.eval(x, y - h, z)) / (2 * h);
  g[2] = (s.eval(x, y, z + h) - s.eval(x, y, z - h)) / (2 * h);
}

/// Central-difference Hessian H_ab = ∂²φ/∂x_a∂x_b (symmetrised), via differences of the gradient
/// (same stencil as sdfGradient, so host/device agree). Used by the differentiable SDF wall force.
template <class Real, class Sdf>
KOKKOS_INLINE_FUNCTION void sdfHessian(const Sdf& s, Real x, Real y, Real z, Real H[3][3]) {
  const Real h = s.gradH();
  Real gp[3], gm[3];
  sdfGradient<Real>(s, x + h, y, z, gp);
  sdfGradient<Real>(s, x - h, y, z, gm);
  for (int r = 0; r < 3; ++r) H[r][0] = (gp[r] - gm[r]) / (2 * h);
  sdfGradient<Real>(s, x, y + h, z, gp);
  sdfGradient<Real>(s, x, y - h, z, gm);
  for (int r = 0; r < 3; ++r) H[r][1] = (gp[r] - gm[r]) / (2 * h);
  sdfGradient<Real>(s, x, y, z + h, gp);
  sdfGradient<Real>(s, x, y, z - h, gm);
  for (int r = 0; r < 3; ++r) H[r][2] = (gp[r] - gm[r]) / (2 * h);
  for (int a = 0; a < 3; ++a)
    for (int b = a + 1; b < 3; ++b) {
      const Real m = Real(0.5) * (H[a][b] + H[b][a]);
      H[a][b] = H[b][a] = m;
    }
}

/// Differentiable SDF wall force (Effort 2, Option A — the seed-foot model). A wall facet
/// (pnbr == kBoundaryFacet) is modelled as the tangent plane at the seed's foot point on sdf=0, so
/// its seed-relative foot-point normal is n_wall(s) = −φ(s) û(s), û = ∇φ/|∇φ|. Its Jacobian is
///   J_wall = ∂n_wall/∂s = −|∇φ| û ûᵀ − (φ/|∇φ|)(I − û ûᵀ) H,   H = ∇²φ,
/// so the wall's contribution to dGeom/dseed is J_wallᵀ g summed over the cell's wall facets
/// (g = dGeom/dn_k from geomVolumeGrad). EXACT for a flat wall (φ linear ⇒ H=0, one facet);
/// first-order for a curved wall (the clip approximates the curve by several vertex-anchored tangent
/// facets, modelled here as one effective seed-foot plane). Call AFTER chainToDofs<Policy> (which
/// zeroes pnbr<0 planes); this adds the wall self-force into fSelf. No-op for NoSdf.
template <class Real, int MAXP, int MAXT, bool TrackAdj, class Sdf>
KOKKOS_INLINE_FUNCTION void addSdfWallForce(const ConvexCell<Real, MAXP, MAXT, TrackAdj>& c,
                                            const Real seed[3], const Sdf& sdf, const Real* gx,
                                            const Real* gy, const Real* gz, Real fSelf[3]) {
  if constexpr (std::is_same_v<Sdf, NoSdf>) {
    (void)c;
    (void)seed;
    (void)sdf;
    (void)gx;
    (void)gy;
    (void)gz;
    (void)fSelf;
    return;
  } else {
    // Aggregate the wall facets' geometry gradients (all move together under the seed-foot model).
    Real gw[3] = {Real(0), Real(0), Real(0)};
    bool any = false;
    for (int k = 0; k < c.np; ++k)
      if (c.pnbr[k] == kBoundaryFacet) {
        gw[0] += gx[k];
        gw[1] += gy[k];
        gw[2] += gz[k];
        any = true;
      }
    if (!any) return;
    const Real phi = sdf.eval(seed[0], seed[1], seed[2]);
    Real g[3];
    sdfGradient<Real>(sdf, seed[0], seed[1], seed[2], g);
    const Real gn = Kokkos::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (gn <= Real(1e-12)) return;  // degenerate gradient (box crease/corner) — no wall force (guard)
    const Real u[3] = {g[0] / gn, g[1] / gn, g[2] / gn};
    Real H[3][3];
    sdfHessian<Real>(sdf, seed[0], seed[1], seed[2], H);
    const Real ug = u[0] * gw[0] + u[1] * gw[1] + u[2] * gw[2];
    const Real perp[3] = {gw[0] - ug * u[0], gw[1] - ug * u[1], gw[2] - ug * u[2]};
    const Real Hp[3] = {H[0][0] * perp[0] + H[0][1] * perp[1] + H[0][2] * perp[2],
                        H[1][0] * perp[0] + H[1][1] * perp[1] + H[1][2] * perp[2],
                        H[2][0] * perp[0] + H[2][1] * perp[1] + H[2][2] * perp[2]};
    const Real cH = phi / gn;
    for (int d = 0; d < 3; ++d) fSelf[d] += -gn * ug * u[d] - cH * Hp[d];
  }
}

/**
 * Clip a built scratch cell against the SDF solid. Faithful port of
 * clipCellAgainstBoundary (m_boundaryMaxCuts=24, m_boundaryTol=1e-8): empties the
 * cell if its seed is in the solid; otherwise iteratively projects the most
 * violating vertex onto sdf=0 and clips by the tangent plane there, so a curved
 * surface is approximated by a few planar wall facets.
 *
 * @param seed  seed world position (the cell's vpos are relative to it).
 * @return true if the cell was emptied (seed inside solid).
 */
template <class Real, int MAXP, int MAXT, bool TrackAdj, class Sdf>
KOKKOS_INLINE_FUNCTION bool clipCellAgainstSdf(ConvexCell<Real, MAXP, MAXT, TrackAdj>& c,
                                               const Real seed[3], const Sdf& sdf) {
  const Real tol = Real(1e-8);
  const int maxCuts = 24;
  const Real phiCenter = sdf.eval(seed[0], seed[1], seed[2]);
  if (phiCenter <= Real(0)) {  // seed inside solid -> no cell
    for (int t = 0; t < c.nt; ++t)
      c.alive[t] = false;
    return true;
  }
  // cell circumradius (dual vertices are seed-relative)
  const Real maxRsq = c.maxVertexRsq();
  const Real radius = Kokkos::sqrt(maxRsq > 0 ? maxRsq : Real(0));
  if (phiCenter > radius + tol)
    return false;  // cell fully in fluid

  bool seedPlaneApplied = false;
  for (int iter = 0; iter < maxCuts; ++iter) {
    Real probe[3] = {seed[0], seed[1], seed[2]};
    Real probePhi = phiCenter;
    if (seedPlaneApplied) {
      bool found = false;
      for (int t = 0; t < c.nt; ++t) {
        if (!c.alive[t])
          continue;
        Real x = seed[0] + c.vx[t], y = seed[1] + c.vy[t], z = seed[2] + c.vz[t];
        Real phi = sdf.eval(x, y, z);
        if (!found || phi < probePhi) {
          probe[0] = x;
          probe[1] = y;
          probe[2] = z;
          probePhi = phi;
          found = true;
        }
      }
      if (!found || probePhi >= -tol)
        break;
    }
    // closest point on sdf=0 + outward normal
    Real g[3];
    sdfGradient<Real>(sdf, probe[0], probe[1], probe[2], g);
    Real gsq = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    if (gsq <= Real(0))
      break;
    Real phi = sdf.eval(probe[0], probe[1], probe[2]);
    Real invGsq = Real(1) / gsq, invG = Real(1) / Kokkos::sqrt(gsq);
    Real surf[3], normal[3];
    for (int k = 0; k < 3; ++k) {
      surf[k] = probe[k] - phi * g[k] * invGsq;
      normal[k] = g[k] * invG;
    }
    // orient the normal into the fluid (positive side)
    Real eps = Real(1e-3) * (radius + Real(1));
    if (eps < Real(1e-6))
      eps = Real(1e-6);
    if (sdf.eval(surf[0] + eps * normal[0], surf[1] + eps * normal[1], surf[2] + eps * normal[2]) <=
        Real(0))
      for (int k = 0; k < 3; ++k)
        normal[k] = -normal[k];
    // plane in seed-relative coords: dist(v) = v·pv - off; vertices in the solid (off the fluid
    // side) get dist > 0 and are removed.
    Real pv[3] = {-normal[0], -normal[1], -normal[2]};
    Real off =
        pv[0] * (surf[0] - seed[0]) + pv[1] * (surf[1] - seed[1]) + pv[2] * (surf[2] - seed[2]);
    c.clip(pv, off, kBoundaryFacet);
    seedPlaneApplied = true;
    if (c.empty())
      break;
  }
  return false;
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_SDF_HPP
