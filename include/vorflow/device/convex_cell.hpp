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
      const Real px = dx * e1[0] + dy * e1[1] + dz * e1[2];  // in-plane coords
      const Real py = dx * e2[0] + dy * e2[1] + dz * e2[2];
      // diamond pseudo-angle: monotonic in true angle, in [0,4), NO atan2 (transcendental)
      const Real s = Kokkos::fabs(px) + Kokkos::fabs(py);
      const Real t = (s > Real(0)) ? py / s : Real(0);            // [-1,1]
      ang[i] = (px < Real(0)) ? (Real(2) - t) : (py < Real(0) ? Real(4) + t : t);
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

  unsigned short adjT[MAXT][3];  ///< (optional) neighbour triangle across the edge OPPOSITE local
                                 ///< vertex 0/1/2 — built once for the O(1)-hop order-free volume.

  /// Fill adjT: for each triangle, the neighbour across each of its 3 edges. O(nt²) — call ONCE
  /// (e.g. after the cold build); re-eval then reuses it. Edge opposite vertex 0 is (t1,t2), etc.
  KOKKOS_INLINE_FUNCTION void buildAdjacency() {
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      adjT[t][0] = (unsigned short)findSharing(t, t1[t], t2[t]);
      adjT[t][1] = (unsigned short)findSharing(t, t0[t], t2[t]);
      adjT[t][2] = (unsigned short)findSharing(t, t0[t], t1[t]);
    }
  }

  /// Cell volume (G1), order-free AND search-free: walk each face's boundary via the stored
  /// adjacency (O(1) hops, no findSharing, no atan2, no sort). One O(nt) pass picks a start
  /// triangle per face; the per-face walk then accumulates A_k = ½ Σ v×v' via the local winding.
  KOKKOS_INLINE_FUNCTION Real volumeAdj() const {
    short faceTri[MAXP];
    for (int k = 0; k < np; ++k) faceTri[k] = -1;
    for (int t = 0; t < nt; ++t) {  // one start triangle per face, single pass
      if (!alive[t]) continue;
      if (faceTri[t0[t]] < 0) faceTri[t0[t]] = (short)t;
      if (faceTri[t1[t]] < 0) faceTri[t1[t]] = (short)t;
      if (faceTri[t2[t]] < 0) faceTri[t2[t]] = (short)t;
    }
    Real vol = 0;
    for (int k = 0; k < np; ++k) {
      int tstart = faceTri[k];
      if (tstart < 0) continue;
      int inplane = (t0[tstart] != k) ? t0[tstart] : t1[tstart];  // arrival edge (k, inplane)
      Real Ax = 0, Ay = 0, Az = 0;
      int tcur = tstart;
      for (int g = 0; g <= nt; ++g) {
        const int q0 = t0[tcur], q1 = t1[tcur], q2 = t2[tcur];
        // leave via edge (k, outplane); neighbour = edge OPPOSITE the arrival plane `inplane`
        const int outplane = (q0 != k && q0 != inplane) ? q0 : ((q1 != k && q1 != inplane) ? q1 : q2);
        const int li = (q0 == inplane) ? 0 : ((q1 == inplane) ? 1 : 2);
        const int tnext = adjT[tcur][li];
        if (tnext < 0 || tnext >= nt) break;
        Ax += vy[tcur] * vz[tnext] - vz[tcur] * vy[tnext];
        Ay += vz[tcur] * vx[tnext] - vx[tcur] * vz[tnext];
        Az += vx[tcur] * vy[tnext] - vy[tcur] * vx[tnext];
        inplane = outplane;
        tcur = tnext;
        if (tcur == tstart) break;
      }
      const Real area = Real(0.5) * Kokkos::sqrt(Ax * Ax + Ay * Ay + Az * Az);
      const Real nlen = Kokkos::sqrt(pn[k][0] * pn[k][0] + pn[k][1] * pn[k][1] + pn[k][2] * pn[k][2]);
      vol += (pd[k] / nlen) * area;
    }
    return vol * (Real(1) / Real(3));
  }

  /// Cell volume (G1), order-free / atan2-free. Same V = (1/3) Σ_k h_k A_k, but each face's area
  /// vector A_k = ½ Σ v_i × v_{i+1} is summed by WALKING the face boundary in topological order
  /// instead of sorting its vertices by angle: two face-k vertices are adjacent iff their dual
  /// triangles share a second plane, so we hop along that shared plane around the face. No gather,
  /// no atan2, no sort — just cross products in connectivity order.
  KOKKOS_INLINE_FUNCTION Real volumeWalk() const {
    Real vol = 0;
    for (int k = 0; k < np; ++k) {
      int tstart = -1;  // any live triangle on face k
      for (int t = 0; t < nt; ++t)
        if (alive[t] && (t0[t] == k || t1[t] == k || t2[t] == k)) { tstart = t; break; }
      if (tstart < 0) continue;  // plane k carries no face
      Real Ax = 0, Ay = 0, Az = 0;
      int tcur = tstart, inplane;
      {  // arrive at tstart via one of its two non-k planes; leave via the other
        const int p[3] = {t0[tcur], t1[tcur], t2[tcur]};
        inplane = (p[0] != k) ? p[0] : p[1];
      }
      for (int guard = 0; guard <= nt + 1; ++guard) {
        const int p[3] = {t0[tcur], t1[tcur], t2[tcur]};
        int outplane = -1;  // the plane of tcur that is neither k nor the arrival edge
        for (int e = 0; e < 3; ++e)
          if (p[e] != k && p[e] != inplane) outplane = p[e];
        const int tnext = findSharing(tcur, k, outplane);  // other triangle on edge (k, outplane)
        if (tnext < 0) break;
        Ax += vy[tcur] * vz[tnext] - vz[tcur] * vy[tnext];
        Ay += vz[tcur] * vx[tnext] - vx[tcur] * vz[tnext];
        Az += vx[tcur] * vy[tnext] - vy[tcur] * vx[tnext];
        inplane = outplane;
        tcur = tnext;
        if (tcur == tstart) break;  // face boundary closed
      }
      const Real area = Real(0.5) * Kokkos::sqrt(Ax * Ax + Ay * Ay + Az * Az);
      const Real nlen = Kokkos::sqrt(pn[k][0] * pn[k][0] + pn[k][1] * pn[k][1] + pn[k][2] * pn[k][2]);
      vol += (pd[k] / nlen) * area;
    }
    return vol * (Real(1) / Real(3));
  }

  // ---- Vertex-local, SORT-FREE geometry (Ray/Sokolov/Lefebvre/Lévy TOG 2018; geogram ConvexCell) ----
  // Plane stored as a (non-unit) normal n with interior {x : n·x ≤ n·n}; then x=n is the foot of the
  // perpendicular from the origin onto the plane (the facet point), and |n| is the origin→plane distance.
  // Our cell stores (pn,pd) as {pn·x ≤ pd}, so n = (pd/|pn|²) pn. Volume by the divergence theorem,
  // coning every boundary flag (facet n_i, edge foot f, vertex v) to the origin: each tetra (0,n_i,f,v)
  // is LOCAL to one vertex + its 3 planes, and the signed sum is exact even when feet fall outside their
  // faces. No vertex ordering, no adjacency — a pure per-vertex scatter.

  /// n-representation of plane k (foot of perpendicular from origin): n = (pd/|pn|²) pn.
  KOKKOS_INLINE_FUNCTION void planeN(int k, Real n[3]) const {
    const Real l2 = pn[k][0] * pn[k][0] + pn[k][1] * pn[k][1] + pn[k][2] * pn[k][2];
    const Real a = (l2 > Real(0)) ? pd[k] / l2 : Real(0);
    n[0] = a * pn[k][0]; n[1] = a * pn[k][1]; n[2] = a * pn[k][2];
  }
  KOKKOS_INLINE_FUNCTION static void xprod(const Real a[3], const Real b[3], Real o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
  }
  KOKKOS_INLINE_FUNCTION static Real dot3(const Real a[3], const Real b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  KOKKOS_INLINE_FUNCTION static Real det3(const Real a[3], const Real b[3], const Real c[3]) {
    Real bc[3]; xprod(b, c, bc); return dot3(a, bc);
  }
  /// foot of the perpendicular from v onto the edge line of direction ck: v − (v·ck/ck·ck) ck.
  KOKKOS_INLINE_FUNCTION static void edgeFoot(const Real v[3], const Real ck[3], Real f[3]) {
    const Real c2 = dot3(ck, ck), s = (c2 > Real(0)) ? dot3(v, ck) / c2 : Real(0);
    f[0] = v[0] - s * ck[0]; f[1] = v[1] - s * ck[1]; f[2] = v[2] - s * ck[2];
  }

  /// Cell volume (G1) by the vertex-local flag/divergence sum — NO ordering, NO adjacency, NO atan2,
  /// NO findSharing. One pass over vertices; each contributes 3 signed determinants.
  KOKKOS_INLINE_FUNCTION Real volumePerVertex() const {
    Real vol = 0;
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      Real n1[3], n2[3], n3[3];
      planeN(t0[t], n1); planeN(t1[t], n2); planeN(t2[t], n3);
      Real c1[3], c2[3], c3[3];
      xprod(n2, n3, c1); xprod(n3, n1, c2); xprod(n1, n2, c3);
      Real D = dot3(n1, c1);
      if (D < Real(0))  // canonical order D>0: swap (n2,n3) and (c2,c3); v is unchanged
        for (int a = 0; a < 3; ++a) {
          Real tmp = n2[a]; n2[a] = n3[a]; n3[a] = tmp;
          tmp = c2[a]; c2[a] = c3[a]; c3[a] = tmp;
        }
      const Real v[3] = {vx[t], vy[t], vz[t]};
      Real f12[3], f23[3], f31[3];
      edgeFoot(v, c3, f12); edgeFoot(v, c1, f23); edgeFoot(v, c2, f31);
      Real e[3], d = 0;
      e[0] = n1[0] - n2[0]; e[1] = n1[1] - n2[1]; e[2] = n1[2] - n2[2]; d += det3(e, f12, v);
      e[0] = n2[0] - n3[0]; e[1] = n2[1] - n3[1]; e[2] = n2[2] - n3[2]; d += det3(e, f23, v);
      e[0] = n3[0] - n1[0]; e[1] = n3[1] - n1[1]; e[2] = n3[2] - n1[2]; d += det3(e, f31, v);
      vol += d;
    }
    return vol * (Real(1) / Real(6));
  }

  /// Per-facet areas by the same vertex-local scatter (area[k] zeroed by caller, size np). Each vertex
  /// scatters into its 3 incident facets; |n_i| is the only sqrt (per facet). Order/adjacency-free.
  KOKKOS_INLINE_FUNCTION void facetAreasPerVertex(Real* area) const {
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      int k1 = t0[t], k2 = t1[t], k3 = t2[t];
      Real n1[3], n2[3], n3[3];
      planeN(k1, n1); planeN(k2, n2); planeN(k3, n3);
      Real c1[3], c2[3], c3[3];
      xprod(n2, n3, c1); xprod(n3, n1, c2); xprod(n1, n2, c3);
      Real D = dot3(n1, c1);
      if (D < Real(0)) {  // canonical: swap planes 2,3 (normals, cross, AND indices)
        for (int a = 0; a < 3; ++a) { Real tm = n2[a]; n2[a] = n3[a]; n3[a] = tm; tm = c2[a]; c2[a] = c3[a]; c3[a] = tm; }
        int tk = k2; k2 = k3; k3 = tk;
      }
      const Real v[3] = {vx[t], vy[t], vz[t]};
      Real f12[3], f23[3], f31[3];
      edgeFoot(v, c3, f12); edgeFoot(v, c1, f23); edgeFoot(v, c2, f31);
      Real g[3];
      g[0] = f12[0] - f31[0]; g[1] = f12[1] - f31[1]; g[2] = f12[2] - f31[2];
      area[k1] += det3(n1, g, v) / (Real(2) * Kokkos::sqrt(dot3(n1, n1)));
      g[0] = f23[0] - f12[0]; g[1] = f23[1] - f12[1]; g[2] = f23[2] - f12[2];
      area[k2] += det3(n2, g, v) / (Real(2) * Kokkos::sqrt(dot3(n2, n2)));
      g[0] = f31[0] - f23[0]; g[1] = f31[1] - f23[1]; g[2] = f31[2] - f23[2];
      area[k3] += det3(n3, g, v) / (Real(2) * Kokkos::sqrt(dot3(n3, n3)));
    }
  }

  /// Scatter one facet's area + first moment (∫x dA) at a vertex into the per-facet accumulators.
  /// Facet i's local boundary at v is the path f_first → v → f_last with apex n_i: two signed
  /// triangles (n_i,f_first,v) and (n_i,v,f_last). |n_i| is the one sqrt (per facet).
  KOKKOS_INLINE_FUNCTION static void scatterFacetMoment(int ki, const Real ni[3], const Real ff[3],
                                                        const Real fl[3], const Real v[3], Real* area,
                                                        Real* mx, Real* my, Real* mz) {
    const Real inv2n = Real(0.5) / Kokkos::sqrt(dot3(ni, ni));
    const Real sa = det3(ni, ff, v) * inv2n;   // triangle (ni, ff, v)
    const Real sb = -det3(ni, fl, v) * inv2n;  // triangle (ni, v, fl)
    area[ki] += sa + sb;
    mx[ki] += sa * (ni[0] + ff[0] + v[0]) / Real(3) + sb * (ni[0] + v[0] + fl[0]) / Real(3);
    my[ki] += sa * (ni[1] + ff[1] + v[1]) / Real(3) + sb * (ni[1] + v[1] + fl[1]) / Real(3);
    mz[ki] += sa * (ni[2] + ff[2] + v[2]) / Real(3) + sb * (ni[2] + v[2] + fl[2]) / Real(3);
  }

  /// Per-facet area + first moment ∫x dA by the same vertex-local scatter (caller zeroes the np-sized
  /// arrays). The area-weighted facet centroid is c_k = (mx,my,mz)[k]/area[k]; the volume gradient
  /// (force) is then dV_k = ∂V/∂r_k = (area_k/|r_k|)(r_k − c_k) with r_k = pn[k] — sort/adjacency-free.
  KOKKOS_INLINE_FUNCTION void facetMomentsPerVertex(Real* area, Real* mx, Real* my, Real* mz) const {
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      int k1 = t0[t], k2 = t1[t], k3 = t2[t];
      Real n1[3], n2[3], n3[3];
      planeN(k1, n1); planeN(k2, n2); planeN(k3, n3);
      Real c1[3], c2[3], c3[3];
      xprod(n2, n3, c1); xprod(n3, n1, c2); xprod(n1, n2, c3);
      if (dot3(n1, c1) < Real(0)) {  // canonical: swap planes 2,3 (normals, cross, indices)
        for (int a = 0; a < 3; ++a) { Real tm = n2[a]; n2[a] = n3[a]; n3[a] = tm; tm = c2[a]; c2[a] = c3[a]; c3[a] = tm; }
        int tk = k2; k2 = k3; k3 = tk;
      }
      const Real v[3] = {vx[t], vy[t], vz[t]};
      Real f12[3], f23[3], f31[3];
      edgeFoot(v, c3, f12); edgeFoot(v, c1, f23); edgeFoot(v, c2, f31);
      scatterFacetMoment(k1, n1, f12, f31, v, area, mx, my, mz);
      scatterFacetMoment(k2, n2, f23, f12, v, area, mx, my, mz);
      scatterFacetMoment(k3, n3, f31, f23, v, area, mx, my, mz);
    }
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
