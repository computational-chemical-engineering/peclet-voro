/**
 * @file lagrangian/facets.hpp
 * \brief The facet geometry a Lagrangian moving-cell method needs (M7 NHV-W, DESIGN_NOTE §2.1,
 * §10.2), exported from the RESIDENT tessellation: per cell, a CSR of its directed interior facets
 * (area A, centroid c, the min-image connector d = x_j′ − x_i, the second-moment tensor
 * ∫_f (y−c)(y−c)ᵀ dA), its wall facets (area, centroid, outward unit normal), and the cell's volume
 * and centroid.
 *
 * Every quantity is computed in the cell's own frame — coordinates relative to the generator x_i,
 * i.e. "unwrapped at the generator" — so a cell that straddles the periodic seam needs no
 * unwrapping and the volume residuals of the OT solve are free of the absolute-coordinate
 * round-off floor (DESIGN_NOTE §10.2).
 *
 * Unlike `reevalPublish` (the force-geometry publish), the re-evaluation here honours the plane
 * policy: a power tessellation is re-evaluated on its radical planes (`reevalGeometry<Power>` with
 * the resident weights), so the facets are the power facets, and the connector is taken from the
 * positions (for a radical plane the stored foot point 2n = 2αr is not the seed connector r).
 *
 * Both sides of an interior facet are kept (cell i's facet toward j and cell j's toward i), each
 * computed from its own cell. The two agree to round-off; keeping both lets every operator in
 * lagrangian/operators.hpp be an atomic-free gather over a cell's own facets.
 *
 * Valid in voro's small-weight power regime (every seed inside its own cell, d_ij > 0): there the
 * outward normal of the facet toward j is d/|d| and every cone from the generator has a positive
 * height. A buried power cell is emptied by the engine (build_report()['buried']); it exports no
 * facets.
 */
#ifndef PECLET_VORO_LAGRANGIAN_FACETS_HPP
#define PECLET_VORO_LAGRANGIAN_FACETS_HPP

#include <Kokkos_Core.hpp>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/sdf.hpp"  // WallStore
#include "peclet/voro/topology_store.hpp"

namespace peclet::voro::lagrangian {

/// Facet geometry of a tessellation, device-resident. Interior facets of cell i are
/// [facetOffset(i), facetOffset(i+1)); wall facets [wallOffset(i), wallOffset(i+1)). 3-vectors are
/// flat (3*f + a), tensors row-major (9*f + 3*a + b). Positions are generator-relative (cell i's
/// frame): the absolute centroid is x_i + centroid.
template <class Real>
struct CellFacets {
  using Mem = peclet::core::MemSpace;
  int nCells = 0, nFacets = 0, nWall = 0;
  Kokkos::View<int*, Mem> facetOffset;    // nCells+1
  Kokkos::View<int*, Mem> nbr;            // nFacets: neighbour cell j
  Kokkos::View<Real*, Mem> area;          // nFacets: A
  Kokkos::View<Real*, Mem> centroid;      // 3*nFacets: c − x_i
  Kokkos::View<Real*, Mem> conn;          // 3*nFacets: x_j′ − x_i (min image)
  Kokkos::View<Real*, Mem> second;        // 9*nFacets: ∫_f (y−c)(y−c)ᵀ dA
  Kokkos::View<int*, Mem> wallOffset;     // nCells+1
  Kokkos::View<Real*, Mem> wallArea;      // nWall
  Kokkos::View<Real*, Mem> wallCentroid;  // 3*nWall: c − x_i
  Kokkos::View<Real*, Mem> wallNormal;    // 3*nWall: outward unit normal
  Kokkos::View<Real*, Mem> volume;        // nCells: V_i (cone fan from the generator)
  Kokkos::View<Real*, Mem> cellCentroid;  // 3*nCells: b_i − x_i
  // Higher moments, exported only on request (exportCellFacets(..., withHigherMoments = true);
  // empty otherwise): the facet's central third moment ∫_f η⊗η⊗η dA, η = y − c, as its ten
  // symmetric components in the sym3Index order (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz),
  // and the cell's second moment about the generator ∫_{C_i} (y−x_i)(y−x_i)ᵀ dV (cone fan, like
  // volume and cellCentroid). Used by the nonlocal capillary energy
  // (lagrangian/nonlocal_energy.hpp).
  Kokkos::View<Real*, Mem> third;       // 10*nFacets (optional)
  Kokkos::View<Real*, Mem> cellSecond;  // 9*nCells (optional), row-major
  bool hasHigherMoments() const { return cellSecond.extent(0) > 0; }
};

/// Index of the symmetric third-moment component (a ≤ b ≤ c after sorting) in the 10-vector.
KOKKOS_INLINE_FUNCTION int sym3Index(int a, int b, int c) {
  // sort a ≤ b ≤ c
  if (a > b) {
    const int t = a;
    a = b;
    b = t;
  }
  if (b > c) {
    const int t = b;
    b = c;
    c = t;
  }
  if (a > b) {
    const int t = a;
    a = b;
    b = t;
  }
  // lexicographic over a ≤ b ≤ c in {0,1,2}: offsets of a = 0 (6 entries), a = 1 (3), a = 2 (1)
  const int baseA = a == 0 ? 0 : (a == 1 ? 6 : 9);
  // within a: pairs (b, c) with a ≤ b ≤ c
  int k = 0;
  for (int bb = a; bb <= 2; ++bb)
    for (int cc = bb; cc <= 2; ++cc) {
      if (bb == b && cc == c)
        return baseA + k;
      ++k;
    }
  return -1;
}

/// Geometry of face k of cell `c` (vertices generator-relative): area, centroid, second-moment
/// tensor about the centroid, and the face's cone from the generator (signed volume and first
/// moment). The polygon is fanned from its first vertex with SIGNED triangle areas projected on
/// the outward unit normal, so a non-convex round-off sliver cannot flip a sign. Returns false for
/// a degenerate face (fewer than three vertices or a zero plane normal).
template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION bool faceMoments(const Cell& c, int k, Real& area, Real cen[3], Real sec[9],
                                        Real& coneVol, Real coneMom[3]) {
  Real fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
  const int m = c.faceOrdered(k, fx, fy, fz);
  if (m < 3)
    return false;
  const Real nl = Kokkos::sqrt(c.nn[k]);
  if (!(nl > Real(0)))
    return false;
  const Real u[3] = {c.n[k][0] / nl, c.n[k][1] / nl, c.n[k][2] / nl};  // outward unit normal
  Real A = 0, cx = 0, cy = 0, cz = 0;
  for (int q = 1; q + 1 < m; ++q) {
    const Real a[3] = {fx[q] - fx[0], fy[q] - fy[0], fz[q] - fz[0]};
    const Real b[3] = {fx[q + 1] - fx[0], fy[q + 1] - fy[0], fz[q + 1] - fz[0]};
    const Real at =
        Real(0.5) * ((a[1] * b[2] - a[2] * b[1]) * u[0] + (a[2] * b[0] - a[0] * b[2]) * u[1] +
                     (a[0] * b[1] - a[1] * b[0]) * u[2]);
    A += at;
    cx += at * (fx[0] + fx[q] + fx[q + 1]);
    cy += at * (fy[0] + fy[q] + fy[q + 1]);
    cz += at * (fz[0] + fz[q] + fz[q + 1]);
  }
  if (!(A > Real(0)))
    return false;
  cen[0] = cx / (Real(3) * A);
  cen[1] = cy / (Real(3) * A);
  cen[2] = cz / (Real(3) * A);
  {
    // A degenerate face (a lattice vertex or edge split by round-off: A ≈ 0 by cancellation of
    // signed fan triangles) has no meaningful centroid: Σ a_t y_t / Σ a_t is unbounded as
    // Σ a_t → 0, and the second moment about it with it (measured: |S| up to 1.7e11 at A = 8e-20
    // on the perturbed 16³ lattice, a 1.8e-5 jump in ∇Ũ when such faces appear). The centroid of a
    // valid (convex) face lies in its vertex hull, |c − m| ≤ max_v |v − m| with m the vertex
    // mean; outside it, take c = m, so that A c, A c cᵀ and S stay O(A · diam²) and vanish with A.
    // Never triggers on a convex face (bit-identical there).
    Real vm[3] = {0, 0, 0};
    for (int q = 0; q < m; ++q) {
      vm[0] += fx[q];
      vm[1] += fy[q];
      vm[2] += fz[q];
    }
    for (int a = 0; a < 3; ++a)
      vm[a] /= Real(m);
    Real r2 = 0;
    for (int q = 0; q < m; ++q) {
      const Real dx = fx[q] - vm[0], dy = fy[q] - vm[1], dz = fz[q] - vm[2];
      const Real d2 = dx * dx + dy * dy + dz * dz;
      r2 = d2 > r2 ? d2 : r2;
    }
    const Real ex = cen[0] - vm[0], ey = cen[1] - vm[1], ez = cen[2] - vm[2];
    if (!(ex * ex + ey * ey + ez * ez <= r2))
      for (int a = 0; a < 3; ++a)
        cen[a] = vm[a];
  }
  area = A;
  // second moment about the centroid: a triangle (p, q, r) has ∫ y yᵀ dA =
  // (A_t/12)(p pᵀ + q qᵀ + r rᵀ + s sᵀ), s = p + q + r (vertices relative to the centroid)
  for (int e = 0; e < 9; ++e)
    sec[e] = Real(0);
  coneVol = Real(0);
  coneMom[0] = coneMom[1] = coneMom[2] = Real(0);
  const Real p[3] = {fx[0] - cen[0], fy[0] - cen[1], fz[0] - cen[2]};
  for (int q = 1; q + 1 < m; ++q) {
    const Real a[3] = {fx[q] - cen[0], fy[q] - cen[1], fz[q] - cen[2]};
    const Real b[3] = {fx[q + 1] - cen[0], fy[q + 1] - cen[1], fz[q + 1] - cen[2]};
    const Real e1[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]};
    const Real e2[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
    const Real at = Real(0.5) * ((e1[1] * e2[2] - e1[2] * e2[1]) * u[0] +
                                 (e1[2] * e2[0] - e1[0] * e2[2]) * u[1] +
                                 (e1[0] * e2[1] - e1[1] * e2[0]) * u[2]);
    const Real s[3] = {p[0] + a[0] + b[0], p[1] + a[1] + b[1], p[2] + a[2] + b[2]};
    const Real w = at / Real(12);
    for (int r = 0; r < 3; ++r)
      for (int t = 0; t < 3; ++t)
        sec[3 * r + t] += w * (p[r] * p[t] + a[r] * a[t] + b[r] * b[t] + s[r] * s[t]);
    // cone (generator, v0, vq, vq+1): signed volume det/6, centroid (v0+vq+vq+1)/4
    const Real v0[3] = {fx[0], fy[0], fz[0]}, v1[3] = {fx[q], fy[q], fz[q]},
               v2[3] = {fx[q + 1], fy[q + 1], fz[q + 1]};
    const Real vt =
        (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0]) +
         v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])) /
        Real(6);
    coneVol += vt;
    for (int r = 0; r < 3; ++r)
      coneMom[r] += vt * Real(0.25) * (v0[r] + v1[r] + v2[r]);
  }
  return true;
}

/// Higher moments of face k of cell `c` about the point `cen` (vertices generator-relative), on
/// the same signed fan as faceMoments: the third moment ∫_f (y−cen)^{⊗3} dA (ten symmetric
/// components, sym3Index order) and the second moment of the face's cone from the generator,
/// ∫_cone y yᵀ dV. Per fan triangle (p, q, r) the third moment is the cubic-exact rule
/// (A_t/60)[3 Σ_v v^{⊗3} + 8 Σ_edges m^{⊗3} + 27 g^{⊗3}] (vertices, edge midpoints, centroid),
/// so the result does not depend on the fan; a cone tetrahedron (0, v0, v1, v2) of signed volume
/// V_t has ∫ y yᵀ = (V_t/20)(Σ v vᵀ + s sᵀ), s = v0 + v1 + v2.
template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION void faceHigherMoments(const Cell& c, int k, const Real cen[3],
                                              Real third[10], Real coneSec[9]) {
  for (int e = 0; e < 10; ++e)
    third[e] = Real(0);
  for (int e = 0; e < 9; ++e)
    coneSec[e] = Real(0);
  Real fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
  const int m = c.faceOrdered(k, fx, fy, fz);
  const Real nl = Kokkos::sqrt(c.nn[k]);
  if (m < 3 || !(nl > Real(0)))
    return;
  const Real u[3] = {c.n[k][0] / nl, c.n[k][1] / nl, c.n[k][2] / nl};
  const Real p[3] = {fx[0] - cen[0], fy[0] - cen[1], fz[0] - cen[2]};
  for (int q = 1; q + 1 < m; ++q) {
    const Real a[3] = {fx[q] - cen[0], fy[q] - cen[1], fz[q] - cen[2]};
    const Real b[3] = {fx[q + 1] - cen[0], fy[q + 1] - cen[1], fz[q + 1] - cen[2]};
    const Real e1[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]};
    const Real e2[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
    const Real at = Real(0.5) * ((e1[1] * e2[2] - e1[2] * e2[1]) * u[0] +
                                 (e1[2] * e2[0] - e1[0] * e2[2]) * u[1] +
                                 (e1[0] * e2[1] - e1[1] * e2[0]) * u[2]);
    // the seven nodes and their weights (×60/A_t)
    Real pts[7][3];
    const Real wts[7] = {3, 3, 3, 8, 8, 8, 27};
    for (int r = 0; r < 3; ++r) {
      pts[0][r] = p[r];
      pts[1][r] = a[r];
      pts[2][r] = b[r];
      pts[3][r] = Real(0.5) * (p[r] + a[r]);
      pts[4][r] = Real(0.5) * (a[r] + b[r]);
      pts[5][r] = Real(0.5) * (b[r] + p[r]);
      pts[6][r] = (p[r] + a[r] + b[r]) / Real(3);
    }
    const Real w = at / Real(60);
    for (int n = 0; n < 7; ++n) {
      const Real* y = pts[n];
      const Real ww = w * wts[n];
      int e = 0;
      for (int r = 0; r < 3; ++r)
        for (int s = r; s < 3; ++s)
          for (int t = s; t < 3; ++t)
            third[e++] += ww * y[r] * y[s] * y[t];
    }
    const Real v0[3] = {fx[0], fy[0], fz[0]}, v1[3] = {fx[q], fy[q], fz[q]},
               v2[3] = {fx[q + 1], fy[q + 1], fz[q + 1]};
    const Real vt =
        (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0]) +
         v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])) /
        Real(6);
    const Real sv[3] = {v0[0] + v1[0] + v2[0], v0[1] + v1[1] + v2[1], v0[2] + v1[2] + v2[2]};
    const Real wv = vt / Real(20);
    for (int r = 0; r < 3; ++r)
      for (int t = 0; t < 3; ++t)
        coneSec[3 * r + t] += wv * (v0[r] * v0[t] + v1[r] * v1[t] + v2[r] * v2[t] + sv[r] * sv[t]);
  }
}

namespace detail {
/// A plane is a face iff ≥3 live triangles use it (reevalPublish's criterion).
template <class Cell>
KOKKOS_INLINE_FUNCTION bool isFace(const Cell& c, int k) {
  int cnt = 0;
  for (int t = 0; t < c.nt; ++t)
    if (c.alive[t] && (c.t0[t] == k || c.t1[t] == k || c.t2[t] == k))
      ++cnt;
  return cnt >= 3;
}

/// Load slot i of the store and re-evaluate it under `Policy` on the current positions (and the
/// resident weights for Power), restoring the resident SDF wall planes first (as reevalPublish).
template <class Policy, class Store, class Real, class Cell>
KOKKOS_INLINE_FUNCTION void loadCell(const Store& st, const WallStore<Real>& Wl,
                                     const Kokkos::View<Real*, peclet::core::MemSpace>& P,
                                     const Kokkos::View<Real*, peclet::core::MemSpace>& W,
                                     const Kokkos::View<Real*, peclet::core::MemSpace>& XR,
                                     const Kokkos::Array<Real, 3>& L, int i, Cell& c) {
  st.load(i, c, L[0], L[1], L[2]);
  if (Wl.active()) {
    Real d[3] = {P(3 * i) - XR(3 * i), P(3 * i + 1) - XR(3 * i + 1), P(3 * i + 2) - XR(3 * i + 2)};
    for (int a = 0; a < 3; ++a) {
      const Real h = Real(0.5) * L[a];
      d[a] = d[a] > h ? d[a] - L[a] : (d[a] < -h ? d[a] + L[a] : d[a]);
    }
    Wl.load(i, c, d[0], d[1], d[2]);
  }
  if constexpr (Policy::kHasWeightDof)
    c.template reevalGeometry<Policy>(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), L[0], W(i),
                                      W.data());
  else
    c.template reevalGeometry<Policy>(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), L[0]);
}
}  // namespace detail

/// Re-evaluate every resident cell of `store` on the current positions (and, for the Power
/// policy, the resident weights) and export its facet geometry. `L` is the cubic periodic box
/// (the engine's minimal-image contract); `wall`/`xRef` restore the resident SDF wall planes as
/// in reevalPublish (leave empty without geometry). `weight` may be empty for Voronoi.
/// `engineVol` (optional, the engine's per-cell volumes): a face toward a neighbour whose cell is
/// EMPTY (its seed lies in the SDF solid, or a buried power cell) is not a DOF coupling — it is
/// exported as a wall facet of the live cell (outward normal toward the dead seed).
template <class Real, class Policy, int MAXP, int MAXT>
CellFacets<Real> exportCellFacets(const TopologyStore<MAXP, MAXT>& store,
                                  const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                                  const Kokkos::View<Real*, peclet::core::MemSpace>& weight, int N,
                                  const Real L[3], WallStore<Real> wall = {},
                                  Kokkos::View<Real*, peclet::core::MemSpace> xRef = {},
                                  Kokkos::View<Real*, peclet::core::MemSpace> engineVol = {},
                                  bool withHigherMoments = false) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Cell = ConvexCell<Real, MAXP, MAXT, false>;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  const Real Lx = L[0], Ly = L[1], Lz = L[2];
  const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
  const TopologyStore<MAXP, MAXT> st = store;
  const Kokkos::View<Real*, Mem> P = pos, W = weight, XR = xRef, EV = engineVol;
  const bool hasEV = engineVol.extent(0) > 0;
  const WallStore<Real> Wl = wall;
  const Kokkos::Array<Real, 3> L3 = {Lx, Ly, Lz};

  // ---- 1. count interior / wall faces per cell ----
  Kokkos::View<int*, Mem> fOff("lag.facetOffset", static_cast<std::size_t>(N) + 1);
  Kokkos::View<int*, Mem> wOff("lag.wallOffset", static_cast<std::size_t>(N) + 1);
  Kokkos::parallel_for(
      "lagrangian.facets.count", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Cell c;
        detail::loadCell<Policy>(st, Wl, P, W, XR, L3, i, c);
        int ni = 0, nw = 0;
        for (int k = 0; k < c.np; ++k) {
          if (!detail::isFace(c, k))
            continue;
          Real a, cen[3], sec[9], cv, cm[3];
          if (!faceMoments(c, k, a, cen, sec, cv, cm))
            continue;
          const int j = c.pnbr[k];
          if (j >= 0 && (!hasEV || EV(j) > Real(0)))
            ++ni;
          else
            ++nw;
        }
        fOff(i + 1) = ni;
        wOff(i + 1) = nw;
      });
  int nF = 0, nW = 0;
  Kokkos::parallel_scan(
      "lagrangian.facets.scan", N + 1,
      KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
        const int v = fOff(i);
        upd += v;
        if (fin)
          fOff(i) = upd;
      },
      nF);
  Kokkos::parallel_scan(
      "lagrangian.facets.scanWall", N + 1,
      KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
        const int v = wOff(i);
        upd += v;
        if (fin)
          wOff(i) = upd;
      },
      nW);

  // ---- 2. fill ----
  CellFacets<Real> F;
  F.nCells = N;
  F.nFacets = nF;
  F.nWall = nW;
  F.facetOffset = fOff;
  F.wallOffset = wOff;
  const std::size_t sF = static_cast<std::size_t>(nF), sW = static_cast<std::size_t>(nW);
  F.nbr = Kokkos::View<int*, Mem>(view_alloc(std::string("lag.nbr"), WithoutInitializing), sF);
  F.area = Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.area"), WithoutInitializing), sF);
  F.centroid =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.cen"), WithoutInitializing), 3 * sF);
  F.conn =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.conn"), WithoutInitializing), 3 * sF);
  F.second =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.sec"), WithoutInitializing), 9 * sF);
  F.wallArea =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.wArea"), WithoutInitializing), sW);
  F.wallCentroid =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.wCen"), WithoutInitializing), 3 * sW);
  F.wallNormal =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.wNormal"), WithoutInitializing), 3 * sW);
  F.volume = Kokkos::View<Real*, Mem>("lag.volume", static_cast<std::size_t>(N));
  F.cellCentroid = Kokkos::View<Real*, Mem>("lag.cellCen", 3 * static_cast<std::size_t>(N));
  if (withHigherMoments) {
    F.third = Kokkos::View<Real*, Mem>(view_alloc(std::string("lag.third"), WithoutInitializing),
                                       10 * sF);
    F.cellSecond = Kokkos::View<Real*, Mem>("lag.cellSec", 9 * static_cast<std::size_t>(N));
  }
  const bool hm = withHigherMoments;
  const CellFacets<Real> G = F;
  Kokkos::parallel_for(
      "lagrangian.facets.fill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Cell c;
        detail::loadCell<Policy>(st, Wl, P, W, XR, L3, i, c);
        int fi = G.facetOffset(i), wi = G.wallOffset(i);
        Real V = 0, M[3] = {0, 0, 0}, Q[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < c.np; ++k) {
          if (!detail::isFace(c, k))
            continue;
          Real a, cen[3], sec[9], cv, cm[3];
          if (!faceMoments(c, k, a, cen, sec, cv, cm))
            continue;
          V += cv;
          M[0] += cm[0];
          M[1] += cm[1];
          M[2] += cm[2];
          Real th[10], cq[9];
          if (hm) {
            faceHigherMoments(c, k, cen, th, cq);
            for (int e = 0; e < 9; ++e)
              Q[e] += cq[e];
          }
          const int j = c.pnbr[k];
          if (j >= 0 && (!hasEV || EV(j) > Real(0))) {
            if (hm)
              for (int e = 0; e < 10; ++e)
                G.third(10 * fi + e) = th[e];
            G.nbr(fi) = j;
            G.area(fi) = a;
            Real r[3] = {P(3 * j) - P(3 * i), P(3 * j + 1) - P(3 * i + 1),
                         P(3 * j + 2) - P(3 * i + 2)};
            r[0] = r[0] > Lxh ? r[0] - Lx : (r[0] < -Lxh ? r[0] + Lx : r[0]);
            r[1] = r[1] > Lyh ? r[1] - Ly : (r[1] < -Lyh ? r[1] + Ly : r[1]);
            r[2] = r[2] > Lzh ? r[2] - Lz : (r[2] < -Lzh ? r[2] + Lz : r[2]);
            for (int d = 0; d < 3; ++d) {
              G.centroid(3 * fi + d) = cen[d];
              G.conn(3 * fi + d) = r[d];
            }
            for (int e = 0; e < 9; ++e)
              G.second(9 * fi + e) = sec[e];
            ++fi;
          } else {
            const Real nl = Kokkos::sqrt(c.nn[k]);
            G.wallArea(wi) = a;
            for (int d = 0; d < 3; ++d) {
              G.wallCentroid(3 * wi + d) = cen[d];
              G.wallNormal(3 * wi + d) = c.n[k][d] / nl;
            }
            ++wi;
          }
        }
        G.volume(i) = V;
        const Real iv = V > Real(0) ? Real(1) / V : Real(0);
        for (int d = 0; d < 3; ++d)
          G.cellCentroid(3 * i + d) = M[d] * iv;
        if (hm)
          for (int e = 0; e < 9; ++e)
            G.cellSecond(9 * i + e) = Q[e];
      });
  return F;
}

/// Result of matchFacets: directed facets dropped for want of a reciprocal.
struct FacetMatch {
  int dropped = 0;
  double maxArea = 0;  ///< largest dropped area
};

/// Keep only RECIPROCATED interior facets: a facet of cell i toward j survives iff j holds a facet
/// toward i with the opposite connector (the same periodic image). The resident topology of a
/// repaired engine can hold a sliver facet on one side only (an incomplete neighbour set at a
/// near-degenerate event — measured areas ≤ 1e-12 h², volumes still exact to 1e-12); one-sided
/// facets break the pairing every gather in lagrangian/operators.hpp relies on (1ᵀC = 0, Cᵀ the
/// transpose of C, H symmetric). Dropping the sliver changes cell i's closure by its area only.
/// Returns the count and the largest area dropped; F is compacted in place when any is.
template <class Real>
FacetMatch matchFacets(CellFacets<Real>& F) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  const int N = F.nCells, nF = F.nFacets;
  const CellFacets<Real> G = F;
  Kokkos::View<int*, Mem> keep("lag.keep", static_cast<std::size_t>(nF));
  Kokkos::View<int*, Mem> off("lag.offKeep", static_cast<std::size_t>(N) + 1);
  FacetMatch res;
  int dropped = 0;
  Real amax = 0;
  Kokkos::parallel_reduce(
      "lagrangian.facets.match", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, int& nd, Real& am) {
        int nk = 0;
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          bool found = false;
          for (int g = G.facetOffset(j); g < G.facetOffset(j + 1) && !found; ++g) {
            if (G.nbr(g) != i)
              continue;
            Real e2 = 0;
            for (int a = 0; a < 3; ++a) {
              const Real e = G.conn(3 * g + a) + G.conn(3 * f + a);
              e2 += e * e;
            }
            found = e2 <= Real(1e-18) * d2;
          }
          keep(f) = found ? 1 : 0;
          nk += found ? 1 : 0;
          if (!found) {
            ++nd;
            am = G.area(f) > am ? G.area(f) : am;
          }
        }
        off(i + 1) = nk;
      },
      dropped, Kokkos::Max<Real>(amax));
  res.dropped = dropped;
  res.maxArea = dropped > 0 ? amax : 0.0;
  if (dropped == 0)
    return res;
  int nK = 0;
  Kokkos::parallel_scan(
      "lagrangian.facets.matchScan", N + 1,
      KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
        const int v = off(i);
        upd += v;
        if (fin)
          off(i) = upd;
      },
      nK);
  const std::size_t sK = static_cast<std::size_t>(nK);
  Kokkos::View<int*, Mem> nbr("lag.nbr", sK);
  Kokkos::View<Real*, Mem> area("lag.area", sK), cen("lag.cen", 3 * sK), conn("lag.conn", 3 * sK),
      sec("lag.sec", 9 * sK);
  const bool hm = F.third.extent(0) > 0;
  Kokkos::View<Real*, Mem> thd("lag.third", hm ? 10 * sK : 0);
  Kokkos::parallel_for(
      "lagrangian.facets.matchFill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int o = off(i);
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          if (!keep(f))
            continue;
          nbr(o) = G.nbr(f);
          area(o) = G.area(f);
          for (int a = 0; a < 3; ++a) {
            cen(3 * o + a) = G.centroid(3 * f + a);
            conn(3 * o + a) = G.conn(3 * f + a);
          }
          for (int e = 0; e < 9; ++e)
            sec(9 * o + e) = G.second(9 * f + e);
          if (hm)
            for (int e = 0; e < 10; ++e)
              thd(10 * o + e) = G.third(10 * f + e);
          ++o;
        }
      });
  F.nFacets = nK;
  F.facetOffset = off;
  F.nbr = nbr;
  F.area = area;
  F.centroid = cen;
  F.conn = conn;
  F.second = sec;
  if (hm)
    F.third = thd;
  return res;
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_FACETS_HPP
