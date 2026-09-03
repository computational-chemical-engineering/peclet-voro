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
#include <string>
#include <type_traits>

#include "peclet/core/common/view.hpp"
#include "peclet/core/geom/primitives.hpp"
#include "peclet/core/geom/scene.hpp"
#include "peclet/voro/convex_cell.hpp"

namespace peclet::voro {

/// Neighbour id stamped on a facet produced by an SDF cut (a wall). Distinct from
/// the initial-cuboid boundary (-1); published as facetNbr < 0 either way.
constexpr int kBoundaryFacet = -2;

/// Sentinel "no geometry" provider — the default; the clip stage is skipped.
struct NoSdf {};

/// Rung A1: wrap a provider to DISABLE the second-order (sagitta) wall placement and cut with the
/// bare tangent plane, as the legacy clip did — the first-order baseline the curved-wall gate
/// measures against. `TangentOnly<SdfSphere<Real>>{{...}}`.
template <class Sdf>
struct TangentOnly : Sdf {
  static constexpr bool kTangentOnly = true;
};
namespace detail {
template <class T, class = void>
struct tangentOnly : std::false_type {};
template <class T>
struct tangentOnly<T, std::void_t<decltype(T::kTangentOnly)>>
    : std::bool_constant<T::kTangentOnly> {};
template <class T>
inline constexpr bool tangentOnly_v = tangentOnly<T>::value;
}  // namespace detail

/// Solid ball (negative inside). DELEGATES to peclet::core::geom::prim::Sphere -- the shared leaf
/// is the single source of the formula (Layer 0 rung 4, suite/docs/ANALYTIC_SDF_GEOMETRY.md); this
/// struct keeps voro's provider shape (a centre + the (x,y,z) scalar signature the clipper calls).
template <class Real>
struct SdfSphere {
  Real cx = 0, cy = 0, cz = 0, radius = 1;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    return peclet::core::geom::prim::Sphere<Real>{radius}.eval(
        peclet::core::Vec3<Real>{x - cx, y - cy, z - cz});
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

/// Axis-aligned solid box of half-extents (hx,hy,hz). DELEGATES to
/// peclet::core::geom::prim::Box. The leaf uses fmax/fmin where this body used ternaries; those
/// agree on every input including NaN and -0.0 (fmax(-0,0) and (-0 > 0 ? -0 : 0) both give +0),
/// and the rung-4 bit-capture confirms the values are unchanged on host and CUDA.
template <class Real>
struct SdfBox {
  Real cx = 0, cy = 0, cz = 0, hx = Real(0.5), hy = Real(0.5), hz = Real(0.5);
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    return peclet::core::geom::prim::Box<Real>{hx, hy, hz}.eval(
        peclet::core::Vec3<Real>{x - cx, y - cy, z - cz});
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

/// Solid hollow cylinder (tube wall) about `axis`. DELEGATES to
/// peclet::core::geom::prim::HollowCylinderShell -- the MAX-OF-HALFSPACES form, which is what this
/// provider has always computed. It is deliberately NOT core's prim::HollowCylinder: that is the
/// distance-exact tube dem uses, a genuinely different function that would shift voro's
/// mesh-optimiser wall force (see the TRAP section of docs/ANALYTIC_SDF_GEOMETRY.md).
/// The axis permutation stays here; the leaf's evalRZ takes the cross-section (r, z) directly.
template <class Real>
struct SdfHollowCylinder {
  Real cx = 0, cy = 0, cz = 0, rOuter = 1, rInner = Real(0.5), height = 1;
  int axis = 2;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real d[3] = {x - cx, y - cy, z - cz};
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
    Real r = Kokkos::sqrt(d[a0] * d[a0] + d[a1] * d[a1]);
    return peclet::core::geom::prim::HollowCylinderShell<Real>{rOuter, rInner, height}.evalRZ(
        r, d[axis]);
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

/// Device-resident sampled SDF (trilinear; x-fastest), the universal provider for analytic-baked
/// or VTI geometry.
///
/// DELEGATES to peclet::core::geom::sampleGrid, the suite's single trilinear grid-SDF routine
/// (Layer 0). The descriptor is held as a MEMBER, built once by fromSpacing() -- building it
/// per call inside the sampler is what shifted nvcc's FMA contraction in dem's rung-3 port.
/// The extension policy is kClamp: the field flattens at the box face, which is what this
/// provider has always done (dem's kObject/kContainer residual policies are for bodies and
/// containers whose far field must keep growing).
///
/// NOTE the shared routine multiplies by an inverse spacing where this body used to divide by
/// spacing, so the last bit of an off-lattice sample can differ. Inert in practice: nothing in
/// voro constructed an SdfGrid -- it was a provider written ahead of its first consumer.
template <class Real>
struct SdfGrid {
  Kokkos::View<const float*, peclet::core::MemSpace> values;  // i + nx*(j + ny*k)
  peclet::core::geom::GridDesc<Real> desc{};

  /// Build from the natural (origin, spacing) description; inverts the spacing once, here.
  ///
  /// CALL THIS AT SETUP, on the host, and pass the resulting provider into the kernel by value --
  /// that is the whole point of caching the descriptor (see the rung-3 FMA note in
  /// docs/ANALYTIC_SDF_GEOMETRY.md). It is marked device-callable only so that calling it in the
  /// wrong place is a performance mistake rather than a silent one: without the annotation nvcc
  /// accepts a host function inside a device lambda WITHOUT error and quietly corrupts the kernel.
  KOKKOS_INLINE_FUNCTION static SdfGrid fromSpacing(
      Kokkos::View<const float*, peclet::core::MemSpace> v, int nx, int ny, int nz, Real ox,
      Real oy, Real oz, Real sx, Real sy, Real sz) {
    SdfGrid g;
    g.values = v;
    g.desc.nx = nx;
    g.desc.ny = ny;
    g.desc.nz = nz;
    g.desc.offset = 0;
    g.desc.origin = peclet::core::Vec3<Real>{ox, oy, oz};
    g.desc.invSpacing = peclet::core::Vec3<Real>{sx != Real(0) ? Real(1) / sx : Real(0),
                                                 sy != Real(0) ? Real(1) / sy : Real(0),
                                                 sz != Real(0) ? Real(1) / sz : Real(0)};
    g.desc.extension = peclet::core::geom::GridExtension::kClamp;
    return g;
  }

  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    return peclet::core::geom::sampleGrid(peclet::core::Vec3<Real>{x, y, z}, desc, values);
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const {
    // quarter of the smallest cell, as before -- recovered from the stored inverse spacing
    const Real ix = desc.invSpacing.x, iy = desc.invSpacing.y, iz = desc.invSpacing.z;
    Real mx = ix > iy ? ix : iy;
    mx = mx > iz ? mx : iz;  // largest inverse == smallest spacing
    return mx > Real(0) ? Real(0.25) / mx : Real(1e-4);
  }
};

/// A whole core shape SCENE as a voro provider: the full shared analytic vocabulary (sphere, box,
/// both hollow-cylinder forms, capsule, torus, cone, ellipsoid, superquadric), CSG union /
/// intersection / difference, per-node conformal transforms, and sampled grids -- all reachable
/// from voro's cell clipper and mesh optimiser through the one `eval(x,y,z)` + `gradH()` interface
/// they already expect.
///
/// This is what lets voro use analytic geometry beyond the three legacy providers above without
/// voro owning any geometry code: the nodes are core's POD ShapeNode records, the evaluation is
/// core's recursion-free evalTree.
template <class Real>
struct SdfScene {
  Kokkos::View<const peclet::core::geom::ShapeNode<Real>*, peclet::core::MemSpace> nodes;
  Kokkos::View<const peclet::core::geom::GridDesc<Real>*, peclet::core::MemSpace> grids;
  Kokkos::View<const float*, peclet::core::MemSpace> samples;
  int nodeCount = 0;
  int root = 0;
  Real h = Real(1e-4);

  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    return peclet::core::geom::evalTree<Real>(nodes, nodeCount, root,
                                              peclet::core::Vec3<Real>{x, y, z}, grids, samples);
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return h; }
};

/// Periodic UNION of solid balls: sdf(x) = min_i(|x−c_i|_minimage − r_i) (<0 inside a ball, >0 in the
/// fluid). Device-callable; holds Views of the centres (3M, x-fastest c_{ix},c_{iy},c_{iz}) and radii
/// (M). L > 0 ⇒ periodic min-image (L = box edge); L ≤ 0 ⇒ non-periodic. The packed-bed / pore-space
/// wall geometry for the volume mesh optimiser.
template <class Real>
struct SdfSpheres {
  Kokkos::View<const Real*, peclet::core::MemSpace> cen;  // 3*M
  Kokkos::View<const Real*, peclet::core::MemSpace> rad;  // M
  int n = 0;
  Real L = 0;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real m = Real(1e30);
    for (int i = 0; i < n; ++i) {
      Real dx = x - cen(3 * i), dy = y - cen(3 * i + 1), dz = z - cen(3 * i + 2);
      if (L > Real(0)) {
        dx -= L * Kokkos::round(dx / L);
        dy -= L * Kokkos::round(dy / L);
        dz -= L * Kokkos::round(dz / L);
      }
      const Real d = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) - rad(i);
      if (d < m) m = d;
    }
    return m;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
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

/// The seed-foot SDF wall chain (rung A3 — the single source; see energy/wall.hpp): f_self +=
/// J_wallᵀ g_w with J_wall = dn/dx of the foot vector n = −φ∇φ/|∇φ|². Coordinates in Real,
/// accumulation in double — the exact arithmetic of the optimiser's device kernel it replaces.
/// No-op for NoSdf.
template <class Real, class Sdf>
KOKKOS_INLINE_FUNCTION void sdfWallChain(const Sdf& sdf, Real x, Real y, Real z, const double gw[3],
                                         double fSelf[3]) {
  if constexpr (std::is_same_v<Sdf, NoSdf>) {
    (void)sdf;
    (void)x;
    (void)y;
    (void)z;
    (void)gw;
    (void)fSelf;
    return;
  } else {
    if (gw[0] == 0.0 && gw[1] == 0.0 && gw[2] == 0.0)
      return;
    Real gg[3];
    sdfGradient<Real>(sdf, x, y, z, gg);
    const double gn =
        Kokkos::sqrt((double)gg[0] * gg[0] + (double)gg[1] * gg[1] + (double)gg[2] * gg[2]);
    if (gn <= 1e-12)
      return;  // degenerate gradient (crease / corner): no wall force (guard)
    const double u0 = gg[0] / gn, u1 = gg[1] / gn, u2 = gg[2] / gn;
    const double ug = u0 * gw[0] + u1 * gw[1] + u2 * gw[2];
    const double p0 = gw[0] - ug * u0, p1 = gw[1] - ug * u1, p2 = gw[2] - ug * u2;
    Real Hm[3][3];
    sdfHessian<Real>(sdf, x, y, z, Hm);
    const double Hp0 = Hm[0][0] * p0 + Hm[0][1] * p1 + Hm[0][2] * p2;
    const double Hp1 = Hm[1][0] * p0 + Hm[1][1] * p1 + Hm[1][2] * p2;
    const double Hp2 = Hm[2][0] * p0 + Hm[2][1] * p1 + Hm[2][2] * p2;
    const double cH = (double)sdf.eval(x, y, z) / gn;
    fSelf[0] += -ug * u0 - cH * Hp0;
    fSelf[1] += -ug * u1 - cH * Hp1;
    fSelf[2] += -ug * u2 - cH * Hp2;
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
    // Rung A3: the shared |∇φ|-independent seed-foot chain (foot vector n = −φ∇φ/|∇φ|²). The
    // earlier body carried an extra |∇φ| factor from the n = −φû shortcut (identical for a true
    // SDF, |∇φ| = 1; wrong for a general level set).
    const double gwd[3] = {(double)gw[0], (double)gw[1], (double)gw[2]};
    double fw[3] = {0.0, 0.0, 0.0};
    sdfWallChain<Real>(sdf, seed[0], seed[1], seed[2], gwd, fw);
    for (int d = 0; d < 3; ++d)
      fSelf[d] += (Real)fw[d];
  }
}

/// Upper bound on the wall planes one SDF clip can commit: clipCellAgainstSdf runs at most
/// `maxCuts` (24) iterations and each commits at most one plane.
constexpr int kMaxWallPlanes = 24;

/**
 * Resident per-cell SDF wall planes for the moving-point path (rung A0 of the Voronoi methods
 * plan). The TopologyStore keeps only neighbour ids + triangles: a neighbour plane is rebuilt from
 * the neighbour's current position (reevalGeometry), but a wall plane has no partner seed, so its
 * equation must be PERSISTED. Each wall plane is stored in the form (û, h): û the unit foot-point
 * direction (from the seed into the solid) and h the seed→plane distance at the time of the clip.
 * When the seed has moved by d since then, the SAME world plane has foot distance h' = h − û·d, so
 * the cell-frame plane is n = h' û, nn = h'² — exact for the stored tangent plane. h' ≤ 0 means the
 * seed crossed its own wall plane: the cell is invalid and must be re-gathered.
 *
 * Records are listed in plane order among the cell's planes k ≥ 6 with pnbr[k] < 0 (the six box
 * planes k < 6 are re-seeded by initBoxPlanes and never stored). Allocated only when an SDF is in
 * use; a default-constructed (empty) store is a no-op in load/save.
 */
template <class Real>
struct WallStore {
  using MemSpace = peclet::core::MemSpace;
  static constexpr int kMax = kMaxWallPlanes;
  Kokkos::View<Real*, MemSpace> rec;  // N*kMax*4 : (ux, uy, uz, h) per wall plane
  Kokkos::View<int*, MemSpace> cnt;   // N : wall planes stored for the cell

  void alloc(int n) {
    rec = Kokkos::View<Real*, MemSpace>(
        Kokkos::view_alloc(std::string("wall.rec"), Kokkos::WithoutInitializing),
        (size_t)n * kMax * 4);
    cnt = Kokkos::View<int*, MemSpace>("wall.cnt", n);
  }
  KOKKOS_INLINE_FUNCTION bool active() const { return cnt.extent(0) > 0; }

  /// Persist the wall planes of finalised cell `c` at slot i (cell frame = the seed position at
  /// clip time). Returns the number stored.
  template <class Cell>
  KOKKOS_INLINE_FUNCTION int save(int i, const Cell& c) const {
    if (!active())
      return 0;
    int r = 0;
    for (int k = 6; k < c.np && r < kMax; ++k) {
      if (c.pnbr[k] >= 0)
        continue;
      const Real h = Kokkos::sqrt(c.nn[k]);
      const Real ih = h > Real(0) ? Real(1) / h : Real(0);
      const size_t o = ((size_t)i * kMax + r) * 4;
      rec(o + 0) = c.n[k][0] * ih;
      rec(o + 1) = c.n[k][1] * ih;
      rec(o + 2) = c.n[k][2] * ih;
      rec(o + 3) = h;
      ++r;
    }
    cnt(i) = r;
    return r;
  }

  /// Restore the wall-plane equations of slot i into a cell just reloaded by TopologyStore::load,
  /// for a seed displaced by (dx,dy,dz) (min-imaged) from where the planes were saved. Returns
  /// false if the seed crossed one of its wall planes (h' ≤ 0) — the cell must be re-gathered.
  template <class Cell>
  KOKKOS_INLINE_FUNCTION bool load(int i, Cell& c, Real dx, Real dy, Real dz) const {
    if (!active())
      return true;
    bool ok = true;
    int r = 0;
    const int nrec = cnt(i);
    for (int k = 6; k < c.np; ++k) {
      if (c.pnbr[k] >= 0)
        continue;
      if (r >= nrec) {  // store out of sync with the topology: treat as invalid
        ok = false;
        break;
      }
      const size_t o = ((size_t)i * kMax + r) * 4;
      const Real ux = rec(o + 0), uy = rec(o + 1), uz = rec(o + 2);
      const Real h = rec(o + 3) - (ux * dx + uy * dy + uz * dz);
      c.n[k][0] = h * ux;
      c.n[k][1] = h * uy;
      c.n[k][2] = h * uz;
      c.nn[k] = h * h;
      if (!(h > Real(0)))
        ok = false;
      ++r;
    }
    return ok;
  }
};

/**
 * Rung A1 — second-order wall placement. A tangent plane at a tangency point `surf` (unit normal
 * into the fluid) is exact for a flat wall but INSCRIBED for a curved one: over its face the true
 * surface deviates from the plane by h(p) = φ(surf + p)/|∇φ| ≈ ½ pᵀ H p / |∇φ| (p in-plane,
 * H = ∇²φ), so the tangent cut removes the fluid sliver between plane and wall on a convex solid
 * and leaves solid inside the cell on a concave one. Once the multi-plane tangent clip has
 * settled the cell, each committed wall plane k gets the mean sliver depth over its FINAL face
 * polygon,
 *     δ_k = V_sag / A_k,   V_sag = ½ Σ_ab H_ab M_ab / |∇φ|,   M = ∫_face p pᵀ dA about surf,
 * and clipCellAgainstSdf RE-CLIPS the un-cut cell with every plane translated INTO the solid by
 * its δ_k (a re-clip, not a plane move, so the result is a valid convex clip and the resident
 * certificate stays clean). The planar cell then has the curved cell's volume to second order.
 * Shifts below round-off of the Hessian stencil (|δ| < 1e-8·h_k, i.e. flat walls) are zeroed so
 * a flat wall is bit-identical to the tangent clip. Writes δ_k into `delta` (0 for planes with
 * no face); returns the number of nonzero shifts.
 */
template <class Real, int MAXP, int MAXT, bool TrackAdj, class Sdf>
KOKKOS_INLINE_FUNCTION int sdfSagittaShifts(const ConvexCell<Real, MAXP, MAXT, TrackAdj>& c,
                                            const Real seed[3], const Sdf& sdf, const int* planes,
                                            const Real* tangency, int nPlanes, Real* delta) {
  using Cell = ConvexCell<Real, MAXP, MAXT, TrackAdj>;
  int nz = 0;
  for (int q = 0; q < nPlanes; ++q) {
    delta[q] = Real(0);
    const int k = planes[q];
    if (k < 0)
      continue;  // chord plane: no tangency
    Real fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
    const int m = c.faceOrdered(k, fx, fy, fz);
    if (m < 3)
      continue;
    const Real hk = Kokkos::sqrt(c.nn[k]);
    if (!(hk > Real(0)))
      continue;
    const Real u[3] = {c.n[k][0] / hk, c.n[k][1] / hk, c.n[k][2] / hk};  // seed -> wall
    const Real* surf = tangency + 3 * q;                                 // world tangency point
    const Real p0[3] = {surf[0] - seed[0], surf[1] - seed[1], surf[2] - seed[2]};
    Real A = 0, M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < m; ++i) {
      const int j = (i + 1) % m;
      const Real a[3] = {fx[i] - p0[0], fy[i] - p0[1], fz[i] - p0[2]};
      const Real b[3] = {fx[j] - p0[0], fy[j] - p0[1], fz[j] - p0[2]};
      const Real cr[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0]};
      const Real At = Real(0.5) * (cr[0] * u[0] + cr[1] * u[1] + cr[2] * u[2]);
      A += At;
      const Real sm[3] = {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
      const Real w = At / Real(12);
      for (int r = 0; r < 3; ++r)
        for (int qq = 0; qq < 3; ++qq)
          M[r][qq] += w * (a[r] * a[qq] + b[r] * b[qq] + sm[r] * sm[qq]);
    }
    if (A < Real(0)) {
      A = -A;
      for (int r = 0; r < 3; ++r)
        for (int qq = 0; qq < 3; ++qq)
          M[r][qq] = -M[r][qq];
    }
    if (!(A > Real(0)))
      continue;
    Real g[3];
    sdfGradient<Real>(sdf, surf[0], surf[1], surf[2], g);
    const Real gn = Kokkos::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (!(gn > Real(0)))
      continue;
    Real H[3][3];
    sdfHessian<Real>(sdf, surf[0], surf[1], surf[2], H);
    Real tr = 0;
    for (int r = 0; r < 3; ++r)
      for (int qq = 0; qq < 3; ++qq)
        tr += H[r][qq] * M[r][qq];
    Real d = Real(0.5) * tr / gn / A;
    if (d < Real(-0.5) * hk)  // a concave shift keeps the plane on the far side of the seed
      d = Real(-0.5) * hk;
    if (Kokkos::fabs(d) < Real(1e-8) * hk)
      continue;  // Hessian-stencil round-off (flat wall): no shift
    delta[q] = d;
    ++nz;
  }
  return nz;
}

/**
 * The wall cut plane for a probe point (the seed for the first cut, a violating vertex after):
 * closest point on φ = 0 + outward normal (oriented into the fluid), the half-space
 * {x : pv·x ≤ off} in seed-relative coordinates with pv = −normal, plus the tangency point
 * `surf` (world) for the sagitta post-pass. Shared by clipCellAgainstSdf and the moving-point
 * boundary watch (sdfWouldClip), so both make the same decision. Returns false if the gradient
 * is degenerate (no plane).
 */
template <class Real, int MAXP, int MAXT, bool TrackAdj, class Sdf>
KOKKOS_INLINE_FUNCTION bool sdfCutPlane(const ConvexCell<Real, MAXP, MAXT, TrackAdj>& c,
                                        const Real seed[3], const Sdf& sdf, const Real probe[3],
                                        Real radius, Real pv[3], Real& off, Real surf[3]) {
  (void)c;
  Real g[3];
  sdfGradient<Real>(sdf, probe[0], probe[1], probe[2], g);
  const Real gsq = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
  if (gsq <= Real(0))
    return false;
  const Real phi = sdf.eval(probe[0], probe[1], probe[2]);
  const Real invGsq = Real(1) / gsq, gn = Kokkos::sqrt(gsq), invG = Real(1) / gn;
  Real normal[3];
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
  pv[0] = -normal[0];
  pv[1] = -normal[1];
  pv[2] = -normal[2];
  off = pv[0] * (surf[0] - seed[0]) + pv[1] * (surf[1] - seed[1]) + pv[2] * (surf[2] - seed[2]);
  return true;
}

/**
 * Would clipCellAgainstSdf commit at least one wall plane on this (un-clipped) cell? The exact
 * decision the cold build makes, so the moving-point boundary watch flags precisely the cells whose
 * cold rebuild would differ from a wall-free re-evaluation: the seed is in the solid (the cell
 * would be emptied), OR the cell is within its circumradius of the surface AND (some vertex lies
 * beyond the seed-foot tangent plane — the first cut — OR some vertex is inside the solid by more
 * than the clip tolerance — a later cut). Reads the same tolerance and probe rule as the clip.
 */
template <class Real, int MAXP, int MAXT, bool TrackAdj, class Sdf>
KOKKOS_INLINE_FUNCTION bool sdfWouldClip(const ConvexCell<Real, MAXP, MAXT, TrackAdj>& c,
                                         const Real seed[3], const Sdf& sdf) {
  const Real tol = Real(1e-8);
  const Real phiCenter = sdf.eval(seed[0], seed[1], seed[2]);
  if (phiCenter <= Real(0))
    return true;
  const Real maxRsq = c.maxVertexRsq();
  const Real radius = Kokkos::sqrt(maxRsq > 0 ? maxRsq : Real(0));
  if (phiCenter > radius + tol)
    return false;
  // first cut: the same plane clipCellAgainstSdf would commit for probe = seed (tangent at the
  // seed's foot point, sagitta-shifted unless TangentOnly). ConvexCell::clip commits iff some live
  // vertex satisfies nf·v > nf·nf with nf = (off/|pv|²) pv; |pv| = 1 here, so the test is
  // pv·v > off. The clip skips a first cut whose offset is not positive, so only a positive
  // offset can commit here.
  {
    Real pv[3], off, surf[3];
    if (sdfCutPlane(c, seed, sdf, seed, radius, pv, off, surf) && off > Real(0))
      for (int t = 0; t < c.nt; ++t) {
        if (!c.alive[t])
          continue;
        if (pv[0] * c.vx[t] + pv[1] * c.vy[t] + pv[2] * c.vz[t] > off)
          return true;
      }
  }
  // later cuts: the most violating vertex must be inside the solid by more than tol
  for (int t = 0; t < c.nt; ++t) {
    if (!c.alive[t])
      continue;
    if (sdf.eval(seed[0] + c.vx[t], seed[1] + c.vy[t], seed[2] + c.vz[t]) < -tol)
      return true;
  }
  return false;
}

/**
 * Clip a built scratch cell against the SDF solid. Port of clipCellAgainstBoundary
 * (m_boundaryMaxCuts=24, m_boundaryTol=1e-8): empties the cell if its seed is in the solid;
 * otherwise iteratively projects the most violating vertex onto sdf=0 and clips by the tangent
 * plane there, so a curved surface is approximated by a few planar wall facets. Rung A0 added the
 * chord-plane fallback for a tangent plane that would exclude the seed (see the body) — the
 * legacy port committed such planes and produced the dead cells seen in pore meshing.
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
  // Make room for the wall planes: the committed-plane list carries the clip's redundant
  // candidates, and maxCuts more would overflow MAXP on a wall-hugging cell (see compactPlanes).
  if (c.np + maxCuts > MAXP)
    c.compactPlanes();

  // Rung A1: the un-cut cell is kept for the second-order RE-CLIP (see sdfSagittaShifts), and
  // every committed plane is recorded with its half-space and tangency point (chord planes get
  // plane index -1: no tangency, no shift).
  constexpr bool kSagitta = !detail::tangentOnly_v<Sdf>;
  ConvexCell<Real, MAXP, MAXT, TrackAdj> saved;
  if constexpr (kSagitta)
    saved = c;
  int tpPlane[kMaxWallPlanes];
  Real tpSurf[kMaxWallPlanes * 3], tpPv[kMaxWallPlanes * 3], tpOff[kMaxWallPlanes];
  int nTp = 0;

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
    // the cut plane: tangent at the probe's foot point, sagitta-shifted (rung A1) unless
    // TangentOnly; in seed-relative coords dist(v) = v·pv − off, vertices in the solid (off the
    // fluid side) get dist > 0 and are removed.
    Real pv[3], off, surf[3];
    if (!sdfCutPlane(c, seed, sdf, probe, radius, pv, off, surf))
      break;
    if (off <= Real(0)) {
      // The tangent plane at the probe's foot point puts the SEED on its solid side: the tangent
      // approximation is invalid here (surface curvature strong on the scale of the cell — e.g. a
      // seed hugging a sphere while a far vertex sits inside it near the equator). The foot-point
      // half-space {n·x <= n·n} cannot even represent such a plane (n·n >= 0): committing it cut
      // the WRONG side, the violating vertex survived, the same cut was re-applied maxCuts times
      // and the cell overflowed into a silent zero-volume "dead cell" (the rim/collapse symptom
      // seen in pore meshing). Cut instead with the CHORD plane: through the surface crossing of
      // the seed->probe segment, normal along the segment — always a valid half-space that
      // contains the seed and excludes the probe, so the iteration makes progress.
      if (!seedPlaneApplied) {  // probe == seed: nothing to chord; leave the first cut out
        seedPlaneApplied = true;
        continue;
      }
      Real d[3] = {probe[0] - seed[0], probe[1] - seed[1], probe[2] - seed[2]};
      const Real len = Kokkos::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (len <= Real(0))
        break;
      for (int k = 0; k < 3; ++k)
        d[k] /= len;
      Real lo = Real(0), hi = len;  // phi(seed) > 0 >= phi(probe)
      for (int it2 = 0; it2 < 40; ++it2) {
        const Real mid = Real(0.5) * (lo + hi);
        const Real ph = sdf.eval(seed[0] + mid * d[0], seed[1] + mid * d[1], seed[2] + mid * d[2]);
        if (ph > Real(0))
          lo = mid;
        else
          hi = mid;
      }
      const Real t = Real(0.5) * (lo + hi);
      if (t <= Real(0))
        break;
      if (c.clip(d, t, kBoundaryFacet) && nTp < kMaxWallPlanes) {
        tpPlane[nTp] = -1;  // chord plane: re-clipped verbatim
        for (int k = 0; k < 3; ++k) {
          tpPv[3 * nTp + k] = d[k];
          tpSurf[3 * nTp + k] = Real(0);
        }
        tpOff[nTp] = t;
        ++nTp;
      }
      if (c.empty())
        break;
      continue;
    }
    const bool committed = c.clip(pv, off, kBoundaryFacet);
    seedPlaneApplied = true;
    if (c.empty())
      break;
    if (committed) {
      if (nTp < kMaxWallPlanes) {
        tpPlane[nTp] = c.np - 1;
        for (int k = 0; k < 3; ++k) {
          tpSurf[3 * nTp + k] = surf[k];
          tpPv[3 * nTp + k] = pv[k];
        }
        tpOff[nTp] = off;
        ++nTp;
      }
    } else if (iter > 0) {
      break;  // the violating vertex is not cut by its own plane: no progress possible
    }
  }
  if constexpr (kSagitta) {
    if (nTp > 0 && !c.empty()) {
      Real delta[kMaxWallPlanes];
      if (sdfSagittaShifts(c, seed, sdf, tpPlane, tpSurf, nTp, delta) > 0) {
        // second pass: the same planes, in the same order, translated into the solid by δ_k
        c = saved;
        for (int q = 0; q < nTp; ++q) {
          const Real pv2[3] = {tpPv[3 * q], tpPv[3 * q + 1], tpPv[3 * q + 2]};
          c.clip(pv2, tpOff[q] + delta[q], kBoundaryFacet);
          if (c.empty())
            break;
        }
      }
    }
  }
  return false;
}

/**
 * Rung A1 (force half) — the wall part of a cell's geometry gradient, exact for WHATEVER the clip
 * does (tangent planes, chord planes, the sagitta re-clip): central differences of the clipped
 * cell's volume and wall area with respect to the seed position, with the neighbour planes held
 * fixed in the seed frame (the chain-rule split the energy layer uses: neighbour planes go
 * through the plane policy, wall planes through this). `unclipped` is the cell BEFORE the SDF
 * clip; six SDF clips of a copy at seed ± h e_d. Writes dV/dx and dA_wall/dx (3 each).
 */
template <class Real, int MAXP, int MAXT, bool TrackAdj, class Sdf>
KOKKOS_INLINE_FUNCTION void sdfWallFD(const ConvexCell<Real, MAXP, MAXT, TrackAdj>& unclipped,
                                      const Real seed[3], const Sdf& sdf, Real h, Real dV[3],
                                      Real dA[3]) {
  for (int d = 0; d < 3; ++d) {
    Real vpm[2], apm[2];
    for (int s = 0; s < 2; ++s) {
      ConvexCell<Real, MAXP, MAXT, TrackAdj> c = unclipped;
      Real sd[3] = {seed[0], seed[1], seed[2]};
      sd[d] += (s == 0 ? h : -h);
      clipCellAgainstSdf<Real, MAXP, MAXT, TrackAdj>(c, sd, sdf);
      vpm[s] = (c.empty() || c.overflow) ? Real(0) : c.volumePerVertex();
      Real area[MAXP];
      for (int k = 0; k < c.np; ++k)
        area[k] = Real(0);
      if (!(c.empty() || c.overflow))
        c.facetAreasPerVertex(area);
      Real aw = Real(0);
      for (int k = 0; k < c.np; ++k)
        if (c.pnbr[k] == kBoundaryFacet)
          aw += area[k];
      apm[s] = aw;
    }
    dV[d] = (vpm[0] - vpm[1]) / (Real(2) * h);
    dA[d] = (apm[0] - apm[1]) / (Real(2) * h);
  }
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_SDF_HPP
