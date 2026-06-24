/**
 * @file device/convex_cell.hpp
 * \brief Compact "ConvexCell" Voronoi cell — the option-A prototype.
 *
 * The Voronoi cell is the intersection of half-spaces {x : pn_k·x <= pd_k} (one per
 * neighbour, plus the six bounding-box planes). Instead of the half-edge ScratchCell's
 * ~21 KB of explicit topology, the cell is stored in the **dual**: each primal vertex
 * is the intersection of three planes, kept as a *triple of plane indices* (one byte
 * each). The whole cell is then a small triangle list (~a few hundred bytes) that lives
 * in registers / a tiny local frame — the point of the prototype is that this lifts GPU
 * occupancy far above the 32 KB-frame half-edge path (which is occupancy-bound at ~29%).
 *
 * Clipping by a new plane p (GEOGRAM / Ray-et-al. convex-cell clip): mark the triangles
 * whose dual vertex falls outside p (cut away), find the horizon (edges shared between a
 * cut and a kept triangle), and add a new triangle (x,y,p) per horizon edge — p becomes a
 * new face. No stored adjacency: the triangle sharing an edge {x,y} is found by a small
 * scan (the cell is tiny). Cuts are applied closest-first with the same security radius
 * early-out as the half-edge path.
 *
 * This is a DIFFERENT algorithm from the legacy half-edge cutter, so it is NOT bit-exact
 * with that oracle; it is validated to ~1e-9 against voro++ and the space-filling identity
 * (Σ cell volumes == box volume). Core header: Kokkos only.
 */
#ifndef VORFLOW_DEVICE_CONVEX_CELL_HPP
#define VORFLOW_DEVICE_CONVEX_CELL_HPP

#include <Kokkos_Core.hpp>

namespace vor {
namespace device {

/// Compact convex Voronoi cell. MAXP planes (<=255 so a plane index fits in a byte),
/// MAXT dual triangles (= primal vertices). Trivially default-constructible (POD) so it
/// lives in registers / a per-thread stack.
template <class Real, int MAXP = 64, int MAXT = 96>
struct ConvexCell {
  static_assert(MAXP <= 255, "plane index must fit in unsigned char");
  Real pn[MAXP][3];  ///< plane normals (seed-relative; = neighbour rel-pos for Voronoi)
  Real pd[MAXP];     ///< plane offsets: half-space {x : pn·x <= pd}
  int pnbr[MAXP];    ///< neighbour seed id per plane (<0 => bounding box)
  int np;            ///< number of planes
  unsigned char t0[MAXT], t1[MAXT], t2[MAXT];  ///< triangle = triple of plane indices
  Real vx[MAXT], vy[MAXT], vz[MAXT];           ///< cached dual-vertex position per triangle
  bool alive[MAXT];  ///< triangle live flag
  int nt;            ///< triangle high-water mark
  bool overflow;     ///< set if MAXP/MAXT exceeded -> cell invalid, caller falls back

  /// Seed the cell with the big cuboid of extent (L0,L1,L2) centred on the seed.
  /// Planes 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z; the 8 corners are the dual triangles.
  KOKKOS_INLINE_FUNCTION void initBox(Real L0, Real L1, Real L2) {
    const Real h[3] = {Real(0.5) * L0, Real(0.5) * L1, Real(0.5) * L2};
    for (int ax = 0; ax < 3; ++ax) {
      for (int s = 0; s < 2; ++s) {
        const int k = 2 * ax + s;
        pn[k][0] = pn[k][1] = pn[k][2] = 0;
        pn[k][ax] = (s == 0) ? Real(1) : Real(-1);
        pd[k] = h[ax];
        pnbr[k] = -1;
      }
    }
    np = 6;
    nt = 0;
    overflow = false;
    // 8 corners: x-plane (0|1), y-plane (2|3), z-plane (4|5)
    for (int sx = 0; sx < 2; ++sx)
      for (int sy = 0; sy < 2; ++sy)
        for (int sz = 0; sz < 2; ++sz) {
          t0[nt] = (unsigned char)(0 + sx);
          t1[nt] = (unsigned char)(2 + sy);
          t2[nt] = (unsigned char)(4 + sz);
          alive[nt] = true;
          computeVertex(nt);
          ++nt;
        }
  }

  /// Compute and cache the dual vertex of triangle t (where its three planes meet,
  /// Cramer's rule). Called once when a triangle is created; all later reads use the
  /// cache (vx/vy/vz), so the per-cell vertex solves are O(#triangles), not O(#clips ·
  /// #triangles) — the difference between compute-bound and not.
  KOKKOS_INLINE_FUNCTION void computeVertex(int t) {
    const Real* a = pn[t0[t]];
    const Real* b = pn[t1[t]];
    const Real* c = pn[t2[t]];
    const Real da = pd[t0[t]], db = pd[t1[t]], dc = pd[t2[t]];
    const Real bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                        b[0] * c[1] - b[1] * c[0]};
    const Real ca[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2],
                        c[0] * a[1] - c[1] * a[0]};
    const Real ab[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
    const Real det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
    const Real inv = det != Real(0) ? Real(1) / det : Real(0);
    vx[t] = (da * bc[0] + db * ca[0] + dc * ab[0]) * inv;
    vy[t] = (da * bc[1] + db * ca[1] + dc * ab[1]) * inv;
    vz[t] = (da * bc[2] + db * ca[2] + dc * ab[2]) * inv;
  }

  /// Set ONLY the six bounding-box planes (0:+x 1:-x 2:+y 3:-y 4:+z 5:-z), np=6, no triangles.
  /// Used to rebuild a cell from a compact stored topology: the caller then overwrites np/nt and the
  /// triangle structure and calls reevalGeometry() (which fills the neighbour planes + all vertices).
  KOKKOS_INLINE_FUNCTION void initBoxPlanes(Real L0, Real L1, Real L2) {
    const Real h[3] = {Real(0.5) * L0, Real(0.5) * L1, Real(0.5) * L2};
    for (int ax = 0; ax < 3; ++ax)
      for (int s = 0; s < 2; ++s) {
        const int k = 2 * ax + s;
        pn[k][0] = pn[k][1] = pn[k][2] = 0;
        pn[k][ax] = (s == 0) ? Real(1) : Real(-1);
        pd[k] = h[ax];
        pnbr[k] = -1;
      }
    np = 6;
    overflow = false;
  }

  /// Part II (moving points): re-evaluate this cell's geometry IN PLACE after the seeds moved,
  /// REUSING the resident topology (the surviving plane set `pnbr` + the dual-triangle structure
  /// t0/t1/t2/alive). No gather, no clip: each neighbour plane is rebuilt from the neighbour's new
  /// position (box planes pnbr<0 are fixed), then every live vertex is recomputed. Correct exactly
  /// when the topology is unchanged (the common case for small per-step displacement); a topology
  /// flip must be detected + repaired separately (Phase 1.5). `pos` is the global seed array (x-
  /// fastest, 3*N); `L` the periodic box length for min-imaging.
  KOKKOS_INLINE_FUNCTION void reevalGeometry(Real sx, Real sy, Real sz, const Real* pos, Real L) {
    const Real Lh = Real(0.5) * L;
    for (int k = 0; k < np; ++k) {
      if (pnbr[k] < 0) continue;  // bounding-box plane — does not move
      const int g = pnbr[k];
      Real rx = pos[3 * g] - sx, ry = pos[3 * g + 1] - sy, rz = pos[3 * g + 2] - sz;
      rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
      ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
      rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
      pn[k][0] = rx; pn[k][1] = ry; pn[k][2] = rz;
      pd[k] = Real(0.5) * (rx * rx + ry * ry + rz * rz);
    }
    for (int t = 0; t < nt; ++t)
      if (alive[t]) computeVertex(t);
  }

  /// Largest squared dual-vertex radius over live triangles (drives the security radius).
  KOKKOS_INLINE_FUNCTION Real maxVertexRsq() const {
    Real m = 0;
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      const Real r = vx[t] * vx[t] + vy[t] * vy[t] + vz[t] * vz[t];
      if (r > m) m = r;
    }
    return m;
  }

  /// Find a *live* triangle != self that contains both plane indices x and y. In a valid
  /// triangulation an interior edge is shared by exactly two triangles. Dead triangles
  /// (alive==false) keep stale plane indices, so they MUST be skipped or they create
  /// phantom horizon edges (the cascade bug). During a clip the about-to-be-removed
  /// triangles are still alive==true (removed only after horizon collection), so they are
  /// correctly considered here and rejected by the caller's kill[] test.
  KOKKOS_INLINE_FUNCTION int findSharing(int self, int x, int y) const {
    for (int s = 0; s < nt; ++s) {
      if (s == self || !alive[s]) continue;
      const int a = t0[s], b = t1[s], c = t2[s];
      const bool hasX = (a == x || b == x || c == x);
      const bool hasY = (a == y || b == y || c == y);
      if (hasX && hasY) return s;
    }
    return -1;
  }

  KOKKOS_INLINE_FUNCTION int allocTri() {
    for (int s = 0; s < nt; ++s)
      if (!alive[s]) return s;
    if (nt < MAXT) return nt++;
    overflow = true;
    return -1;
  }

  /// Add plane (n, d) for neighbour `nbr` and clip the cell by it. Returns true if the
  /// cell was modified. The plane is only stored if it actually cuts (so redundant
  /// candidates don't grow `np`).
  KOKKOS_INLINE_FUNCTION bool clip(const Real n[3], Real d, int nbr) {
    if (np >= MAXP) {
      overflow = true;
      return false;
    }
    const int pi = np;  // tentative index
    pn[pi][0] = n[0];
    pn[pi][1] = n[1];
    pn[pi][2] = n[2];
    pd[pi] = d;
    pnbr[pi] = nbr;

    // Mark triangles whose dual vertex is outside the new half-space (n·v > d).
    bool kill[MAXT];
    bool any = false;
    for (int t = 0; t < nt; ++t) {
      kill[t] = false;
      if (!alive[t]) continue;
      const Real s = n[0] * vx[t] + n[1] * vy[t] + n[2] * vz[t] - d;
      if (s > 0) {
        kill[t] = true;
        any = true;
      }
    }
    if (!any) return false;  // candidate does not cut -> no-op, plane not committed

    np = pi + 1;  // commit the plane

    // Horizon: for each killed triangle edge whose sharing triangle is alive, the new
    // plane gets a triangle (x, y, pi). Collect first, then mutate.
    unsigned char nA[MAXT], nB[MAXT];
    int nnew = 0;
    for (int t = 0; t < nt; ++t) {
      if (!kill[t]) continue;
      const int vtx[3] = {t0[t], t1[t], t2[t]};
      for (int e = 0; e < 3; ++e) {
        const int x = vtx[e], y = vtx[(e + 1) % 3];
        const int other = findSharing(t, x, y);
        if (other >= 0 && !kill[other]) {
          if (nnew < MAXT) {
            nA[nnew] = (unsigned char)x;
            nB[nnew] = (unsigned char)y;
            ++nnew;
          } else {
            overflow = true;
          }
        }
      }
    }
    // Remove killed triangles.
    for (int t = 0; t < nt; ++t)
      if (kill[t]) alive[t] = false;
    // Add the new fan around plane pi.
    for (int i = 0; i < nnew; ++i) {
      const int slot = allocTri();
      if (slot < 0) break;
      t0[slot] = nA[i];
      t1[slot] = nB[i];
      t2[slot] = (unsigned char)pi;
      alive[slot] = true;
      computeVertex(slot);
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION bool empty() const {
    for (int t = 0; t < nt; ++t)
      if (alive[t]) return false;
    return true;
  }

  KOKKOS_INLINE_FUNCTION int countFaces() const {
    int nf = 0;
    for (int k = 0; k < np; ++k) {
      bool used = false;
      for (int t = 0; t < nt && !used; ++t)
        if (alive[t] && (t0[t] == k || t1[t] == k || t2[t] == k)) used = true;
      if (used) ++nf;
    }
    return nf;
  }

  static constexpr int MAXFV = 28;  // max vertices on one face polygon

  /// Gather face k's vertices (the alive dual triangles containing plane k) and order them
  /// CCW around the face normal (insertion sort by in-plane angle). Returns the count, or
  /// <3 if k is not a (≥triangle) face. Shared by volume() and facetGeometry() — the G0/G1/G2
  /// geometry tiers all start here.
  KOKKOS_INLINE_FUNCTION int faceOrdered(int k, Real fx[MAXFV], Real fy[MAXFV],
                                         Real fz[MAXFV]) const {
    int m = 0;
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      if (t0[t] != k && t1[t] != k && t2[t] != k) continue;
      if (m < MAXFV) {
        fx[m] = vx[t];
        fy[m] = vy[t];
        fz[m] = vz[t];
        ++m;
      }
    }
    if (m < 3) return m;
    const Real nx = pn[k][0], ny = pn[k][1], nz = pn[k][2];
    const Real nlen = Kokkos::sqrt(nx * nx + ny * ny + nz * nz);
    if (nlen == Real(0)) return 0;
    const Real un[3] = {nx / nlen, ny / nlen, nz / nlen};
    Real e1[3];
    if (Kokkos::fabs(un[0]) <= Kokkos::fabs(un[1]) && Kokkos::fabs(un[0]) <= Kokkos::fabs(un[2])) {
      e1[0] = 0;
      e1[1] = -un[2];
      e1[2] = un[1];
    } else if (Kokkos::fabs(un[1]) <= Kokkos::fabs(un[2])) {
      e1[0] = -un[2];
      e1[1] = 0;
      e1[2] = un[0];
    } else {
      e1[0] = -un[1];
      e1[1] = un[0];
      e1[2] = 0;
    }
    const Real e1l = Kokkos::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    e1[0] /= e1l;
    e1[1] /= e1l;
    e1[2] /= e1l;
    const Real e2[3] = {un[1] * e1[2] - un[2] * e1[1], un[2] * e1[0] - un[0] * e1[2],
                        un[0] * e1[1] - un[1] * e1[0]};
    Real cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < m; ++i) {
      cx += fx[i];
      cy += fy[i];
      cz += fz[i];
    }
    cx /= m;
    cy /= m;
    cz /= m;
    Real ang[MAXFV];
    for (int i = 0; i < m; ++i) {
      const Real dx = fx[i] - cx, dy = fy[i] - cy, dz = fz[i] - cz;
      ang[i] = Kokkos::atan2(dx * e2[0] + dy * e2[1] + dz * e2[2],
                             dx * e1[0] + dy * e1[1] + dz * e1[2]);
    }
    for (int i = 1; i < m; ++i) {
      Real ka = ang[i], kx = fx[i], ky = fy[i], kz = fz[i];
      int j = i - 1;
      while (j >= 0 && ang[j] > ka) {
        ang[j + 1] = ang[j];
        fx[j + 1] = fx[j];
        fy[j + 1] = fy[j];
        fz[j + 1] = fz[j];
        --j;
      }
      ang[j + 1] = ka;
      fx[j + 1] = kx;
      fy[j + 1] = ky;
      fz[j + 1] = kz;
    }
    return m;
  }

  /// Polygon area vector (0.5 Σ vi × v(i+1)) of an ordered face.
  KOKKOS_INLINE_FUNCTION static void polyAreaVec(const Real fx[], const Real fy[], const Real fz[],
                                                 int m, Real A[3]) {
    Real ax = 0, ay = 0, az = 0;
    for (int i = 0; i < m; ++i) {
      const int j = (i + 1 == m) ? 0 : i + 1;
      ax += fy[i] * fz[j] - fz[i] * fy[j];
      ay += fz[i] * fx[j] - fx[i] * fz[j];
      az += fx[i] * fy[j] - fy[i] * fx[j];
    }
    A[0] = Real(0.5) * ax;
    A[1] = Real(0.5) * ay;
    A[2] = Real(0.5) * az;
  }

  /// Cell volume (G1): V = (1/3) Σ_faces support_k · area_k. Seed (origin) is interior so
  /// every support distance is positive.
  KOKKOS_INLINE_FUNCTION Real volume() const {
    Real vol = 0;
    Real fx[MAXFV], fy[MAXFV], fz[MAXFV];
    for (int k = 0; k < np; ++k) {
      const int m = faceOrdered(k, fx, fy, fz);
      if (m < 3) continue;
      Real A[3];
      polyAreaVec(fx, fy, fz, m, A);
      const Real area = Kokkos::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
      const Real nlen = Kokkos::sqrt(pn[k][0] * pn[k][0] + pn[k][1] * pn[k][1] + pn[k][2] * pn[k][2]);
      vol += (pd[k] / nlen) * area;
    }
    return vol * (Real(1) / Real(3));
  }

  /// Per-facet physics geometry (G2 tier) for plane k: the outward face area VECTOR, the
  /// volume gradient dV = ∂V/∂r_k, and the connecting vector r_k (= the plane normal). Matches
  /// the half-edge facetArea / facetConnect / facetConnVec. Returns false if k is not a face.
  ///
  /// Derivation of dV: the cell is { r_k·x ≤ ½|r_k|² }. Perturbing r_k → r_k+δ moves a point x
  /// on face k off the plane by δ·(x−r_k); the outward normal displacement is −δ·(x−r_k)/|r_k|,
  /// so δV = ∫_face (−δ·(x−r_k)/|r_k|) dA = (area/|r_k|) δ·(r_k − c_k), with c_k the face
  /// centroid. Holding all other planes fixed, this is first-order exact ⇒
  ///   dV = ∂V/∂r_k = (area/|r_k|)·(r_k − c_k).
  KOKKOS_INLINE_FUNCTION bool facetGeometry(int k, Real areaVec[3], Real dv[3], Real conn[3]) const {
    Real fx[MAXFV], fy[MAXFV], fz[MAXFV];
    const int m = faceOrdered(k, fx, fy, fz);
    if (m < 3) return false;
    Real A[3];
    polyAreaVec(fx, fy, fz, m, A);
    const Real* r = pn[k];
    if (A[0] * r[0] + A[1] * r[1] + A[2] * r[2] < Real(0)) {  // orient outward (toward neighbour)
      A[0] = -A[0];
      A[1] = -A[1];
      A[2] = -A[2];
    }
    const Real area = Kokkos::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
    // area-weighted centroid (fan-triangulate from v0)
    Real cx = 0, cy = 0, cz = 0, asum = 0;
    for (int i = 1; i + 1 < m; ++i) {
      const Real ux = fx[i] - fx[0], uy = fy[i] - fy[0], uz = fz[i] - fz[0];
      const Real wx = fx[i + 1] - fx[0], wy = fy[i + 1] - fy[0], wz = fz[i + 1] - fz[0];
      const Real tx = uy * wz - uz * wy, ty = uz * wx - ux * wz, tz = ux * wy - uy * wx;
      const Real at = Real(0.5) * Kokkos::sqrt(tx * tx + ty * ty + tz * tz);
      cx += at * (fx[0] + fx[i] + fx[i + 1]);
      cy += at * (fy[0] + fy[i] + fy[i + 1]);
      cz += at * (fz[0] + fz[i] + fz[i + 1]);
      asum += at;
    }
    const Real inv = asum > Real(0) ? Real(1) / (Real(3) * asum) : Real(0);
    cx *= inv;
    cy *= inv;
    cz *= inv;
    const Real s = area / Kokkos::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    areaVec[0] = A[0];
    areaVec[1] = A[1];
    areaVec[2] = A[2];
    dv[0] = s * (r[0] - cx);
    dv[1] = s * (r[1] - cy);
    dv[2] = s * (r[2] - cz);
    conn[0] = r[0];
    conn[1] = r[1];
    conn[2] = r[2];
    return true;
  }
};

/// Build a Voronoi cell from neighbour relative positions sorted by ascending distance
/// (rSqHalf). Clips closest-first with the security-radius early-out: once the next
/// candidate's plane offset exceeds 2·max-vertex-rsq, no farther seed can cut. Mirrors
/// the half-edge buildVoronoiCell, but on the compact ConvexCell.
template <class Real, int MAXP, int MAXT>
KOKKOS_INLINE_FUNCTION void buildConvexCell(ConvexCell<Real, MAXP, MAXT>& c, const Real L[3],
                                            const Real* relx, const Real* rely, const Real* relz,
                                            const int* ids, int nNbr) {
  c.initBox(L[0], L[1], L[2]);
  for (int i = 0; i < nNbr; ++i) {
    const Real rx = relx[i], ry = rely[i], rz = relz[i];
    const Real off = Real(0.5) * (rx * rx + ry * ry + rz * rz);  // plane: r·x <= 0.5|r|²
    if (!(off < Real(2) * c.maxVertexRsq())) break;              // security radius
    const Real n[3] = {rx, ry, rz};
    c.clip(n, off, ids[i]);
    if (c.overflow) return;
  }
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_CONVEX_CELL_HPP
