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

// Loop unroll hint, applied ONLY in the CUDA/HIP device passes (where it lets the compiler scalarize
// small dynamically-indexed local arrays into registers); a no-op on the host so gcc sees no unknown pragma.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define VOR_UNROLL _Pragma("unroll")
#else
#define VOR_UNROLL
#endif

namespace vor {
namespace device {

namespace detail {
// Minimal forward-mode dual number (value + K partials) used by ConvexCell::geomVolumeAreaGrad (the geomFull
// area-Jacobian). Templated on Real so it routes through nvcc/hipcc like the rest of the header.
template <class Real, int K>
struct Dual {
  Real v;
  Real d[K];
};
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> dnum(Real c) {
  Dual<Real, K> r; r.v = c;
  for (int i = 0; i < K; ++i) r.d[i] = Real(0);
  return r;
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> dseed(Real c, int s) {
  Dual<Real, K> r = dnum<Real, K>(c); r.d[s] = Real(1);
  return r;
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> operator+(const Dual<Real, K>& a, const Dual<Real, K>& b) {
  Dual<Real, K> r; r.v = a.v + b.v;
  for (int i = 0; i < K; ++i) r.d[i] = a.d[i] + b.d[i];
  return r;
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> operator-(const Dual<Real, K>& a, const Dual<Real, K>& b) {
  Dual<Real, K> r; r.v = a.v - b.v;
  for (int i = 0; i < K; ++i) r.d[i] = a.d[i] - b.d[i];
  return r;
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> operator*(const Dual<Real, K>& a, const Dual<Real, K>& b) {
  Dual<Real, K> r; r.v = a.v * b.v;
  for (int i = 0; i < K; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
  return r;
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> operator/(const Dual<Real, K>& a, const Dual<Real, K>& b) {
  Dual<Real, K> r; const Real inv = Real(1) / b.v; r.v = a.v * inv;
  for (int i = 0; i < K; ++i) r.d[i] = (a.d[i] - r.v * b.d[i]) * inv;
  return r;
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> dsqrt(const Dual<Real, K>& a) {
  Dual<Real, K> r; r.v = Kokkos::sqrt(a.v);
  const Real inv = (r.v > Real(0)) ? Real(1) / (Real(2) * r.v) : Real(0);
  for (int i = 0; i < K; ++i) r.d[i] = a.d[i] * inv;
  return r;
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> ddot(const Dual<Real, K> a[3], const Dual<Real, K> b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION void dcross(const Dual<Real, K> a[3], const Dual<Real, K> b[3], Dual<Real, K> o[3]) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION Dual<Real, K> ddet3(const Dual<Real, K> a[3], const Dual<Real, K> b[3], const Dual<Real, K> c[3]) {
  Dual<Real, K> bc[3]; dcross(b, c, bc); return ddot(a, bc);
}
template <class Real, int K> KOKKOS_INLINE_FUNCTION void dedgeFoot(const Dual<Real, K> v[3], const Dual<Real, K> c[3], Dual<Real, K> f[3]) {
  const Dual<Real, K> s = ddot(v, c) / ddot(c, c);
  f[0] = v[0] - s * c[0]; f[1] = v[1] - s * c[1]; f[2] = v[2] - s * c[2];
}
}  // namespace detail

/// Compact convex Voronoi cell. MAXP planes (<=255 so a plane index fits in a byte),
/// MAXT dual triangles (= primal vertices). Trivially default-constructible (POD) so it
/// lives in registers / a per-thread stack.
template <class Real, int MAXP = 64, int MAXT = 96>
struct ConvexCell {
  static_assert(MAXP <= 255, "plane index must fit in unsigned char");
  Real n[MAXP][3];   ///< foot-point normal of each plane: half-space {x : n·x <= n·n}, so x=n is the
                     ///< foot of the perpendicular from the seed (origin) and |n| the seed->plane dist
                     ///< (= ½·(neighbour rel-pos) for Voronoi; the connector r = 2n).
  Real nn[MAXP];     ///< cached n·n, i.e. the half-space offset (so the cull test is n·x <= nn)
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
        n[k][0] = n[k][1] = n[k][2] = 0;
        n[k][ax] = (s == 0) ? h[ax] : -h[ax];  // foot point = (±h) e_ax
        nn[k] = h[ax] * h[ax];
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
    const Real* a = n[t0[t]];
    const Real* b = n[t1[t]];
    const Real* c = n[t2[t]];
    const Real da = nn[t0[t]], db = nn[t1[t]], dc = nn[t2[t]];
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
        n[k][0] = n[k][1] = n[k][2] = 0;
        n[k][ax] = (s == 0) ? h[ax] : -h[ax];  // foot point = (±h) e_ax
        nn[k] = h[ax] * h[ax];
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
      // foot point of the Voronoi bisector: n = ½r, offset nn = |n|² = ¼|r|²
      const Real hx = Real(0.5) * rx, hy = Real(0.5) * ry, hz = Real(0.5) * rz;
      n[k][0] = hx; n[k][1] = hy; n[k][2] = hz;
      nn[k] = hx * hx + hy * hy + hz * hz;
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

  /// Add a plane for neighbour `nbr` and clip the cell by it. The plane is passed in
  /// (direction, offset) form `{x : pdir·x <= d}`; it is stored internally as its
  /// foot-point normal `nf = (d/|pdir|²)·pdir` (so `{x : nf·x <= nf·nf}` is the same
  /// half-space). Returns true if the cell was modified. The plane is only committed if
  /// it actually cuts (so redundant candidates don't grow `np`).
  KOKKOS_INLINE_FUNCTION bool clip(const Real pdir[3], Real d, int nbr) {
    if (np >= MAXP) {
      overflow = true;
      return false;
    }
    const int pi = np;  // tentative index
    const Real l2 = pdir[0] * pdir[0] + pdir[1] * pdir[1] + pdir[2] * pdir[2];
    const Real a = (l2 > Real(0)) ? d / l2 : Real(0);  // foot scale: nf = a·pdir
    n[pi][0] = a * pdir[0];
    n[pi][1] = a * pdir[1];
    n[pi][2] = a * pdir[2];
    nn[pi] = n[pi][0] * n[pi][0] + n[pi][1] * n[pi][1] + n[pi][2] * n[pi][2];
    pnbr[pi] = nbr;

    // Mark triangles whose dual vertex is outside the new half-space (nf·v > nf·nf).
    bool kill[MAXT];
    bool any = false;
    for (int t = 0; t < nt; ++t) {
      kill[t] = false;
      if (!alive[t]) continue;
      const Real s = n[pi][0] * vx[t] + n[pi][1] * vy[t] + n[pi][2] * vz[t] - nn[pi];
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
    const Real nx = n[k][0], ny = n[k][1], nz = n[k][2];
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
      const Real support = Kokkos::sqrt(nn[k]);  // |n_k| = seed->plane distance
      vol += support * area;
    }
    return vol * (Real(1) / Real(3));
  }

  // ---- Vertex-local, SORT-FREE geometry (Ray/Sokolov/Lefebvre/Lévy TOG 2018; geogram ConvexCell) ----
  // Plane stored as the (non-unit) foot-point normal n with interior {x : n·x ≤ n·n}; then x=n is the
  // foot of the perpendicular from the origin onto the plane (the facet point), and |n| is the origin→
  // plane distance. The cell now stores n directly (member `n`, with `nn = n·n`), so this is just a read.
  // Volume by the divergence theorem, coning every boundary flag (facet n_i, edge foot f, vertex v) to
  // the origin: each tetra (0,n_i,f,v) is LOCAL to one vertex + its 3 planes, and the signed sum is exact
  // even when feet fall outside their faces. No vertex ordering, no adjacency — a pure per-vertex scatter.

  /// Foot-point normal of plane k — now stored directly, so just a copy (the reconstruction divide is gone).
  KOKKOS_INLINE_FUNCTION void planeN(int k, Real out[3]) const {
    out[0] = n[k][0]; out[1] = n[k][1]; out[2] = n[k][2];
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
  /// The three edge feet edgeFoot(v,c1/c2/c3) with ONE divide instead of three: each s_k = (v·ck)/|ck|²
  /// needs 1/|ck|², and the three reciprocals come from a single 1/(g1·g2·g3) (g_k=|c_k|²) via the
  /// product-of-the-other-two trick — exact for a non-degenerate triangle (c1,c2,c3 nonzero). Falls back
  /// to the per-edge edgeFoot (with its cc>0 guard) if degenerate. Used by the per-vertex area/moment/
  /// geometry kernels, which were divide-bound (see docs/voronoi_simd_cells_prototype.md).
  KOKKOS_INLINE_FUNCTION static void edgeFeet3(const Real v[3], const Real c1[3], const Real c2[3],
                                               const Real c3[3], Real f1[3], Real f2[3], Real f3[3]) {
    const Real g1 = dot3(c1, c1), g2 = dot3(c2, c2), g3 = dot3(c3, c3);
    const Real D = g1 * g2 * g3;
    if (!(D > Real(0))) { edgeFoot(v, c1, f1); edgeFoot(v, c2, f2); edgeFoot(v, c3, f3); return; }
    const Real inv = Real(1) / D;
    const Real s1 = dot3(v, c1) * (g2 * g3 * inv);
    const Real s2 = dot3(v, c2) * (g1 * g3 * inv);
    const Real s3 = dot3(v, c3) * (g1 * g2 * inv);
    for (int a = 0; a < 3; ++a) {
      f1[a] = v[a] - s1 * c1[a];
      f2[a] = v[a] - s2 * c2[a];
      f3[a] = v[a] - s3 * c3[a];
    }
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
      // det3(e, edgeFoot(v,ck), v) = (v·ck)(e·(v×ck))/(ck·ck): edgeFoot(v,ck)×v = (v·ck/|ck|²)(v×ck), so
      // the foot drops out and each term carries one 1/|ck|². Fold the three reciprocals into ONE divide
      // over the common denominator g1·g2·g3 (g_k = |c_k|²) — exact for a non-degenerate triangle (c1,c2,c3
      // all nonzero), turning 3 divides/triangle into 1 (the per-vertex geometry kernel was divide-bound;
      // see docs/voronoi_simd_cells_prototype.md). Pairing: edge (n1,n2)↔c3, (n2,n3)↔c1, (n3,n1)↔c2.
      Real vc1[3], vc2[3], vc3[3];
      xprod(v, c1, vc1); xprod(v, c2, vc2); xprod(v, c3, vc3);
      Real e[3];
      e[0] = n1[0] - n2[0]; e[1] = n1[1] - n2[1]; e[2] = n1[2] - n2[2];
      const Real numc3 = dot3(v, c3) * dot3(e, vc3);
      e[0] = n2[0] - n3[0]; e[1] = n2[1] - n3[1]; e[2] = n2[2] - n3[2];
      const Real numc1 = dot3(v, c1) * dot3(e, vc1);
      e[0] = n3[0] - n1[0]; e[1] = n3[1] - n1[1]; e[2] = n3[2] - n1[2];
      const Real numc2 = dot3(v, c2) * dot3(e, vc2);
      const Real g1 = dot3(c1, c1), g2 = dot3(c2, c2), g3 = dot3(c3, c3);
      const Real den = g1 * g2 * g3;
      vol += (den > Real(0)) ? (numc3 * g1 * g2 + numc1 * g2 * g3 + numc2 * g1 * g3) / den : Real(0);
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
      edgeFeet3(v, c1, c2, c3, f23, f31, f12);  // 3 divides → 1 (f23=foot c1, f31=foot c2, f12=foot c3)
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
  /// (force) is then dV_k = ∂V/∂r_k = (area_k/|r_k|)(r_k − c_k) with connector r_k = 2·n[k] — sort/adjacency-free.
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
      edgeFeet3(v, c1, c2, c3, f23, f31, f12);  // 3 divides → 1
      scatterFacetMoment(k1, n1, f12, f31, v, area, mx, my, mz);
      scatterFacetMoment(k2, n2, f23, f12, v, area, mx, my, mz);
      scatterFacetMoment(k3, n3, f31, f23, v, area, mx, my, mz);
    }
  }

  /// Scatter one facet's contribution to the volume-gradient NUMERATOR 2·S_a·n_k − S_m AT a vertex, with
  /// NO sqrt and into just the 3 output arrays. The trick: n_k is the SAME normal (= ni) at every vertex
  /// of facet k, so the per-vertex term 2·da·ni − mom already carries the facet normal; summed over the
  /// facet's vertices it is exactly 2·(Σda)·n_k − Σmom = 2·S_a·n_k − S_m. `da`, `mom` are RAW (no 1/(2|n|)
  /// factor); geomVolumeGrad's fold supplies the shared 1/(2·nn[k]) once per facet (also sqrt-free).
  KOKKOS_INLINE_FUNCTION static void scatterGradRaw(int ki, const Real ni[3], const Real ff[3],
                                                    const Real fl[3], const Real v[3], Real* dgx,
                                                    Real* dgy, Real* dgz) {
    const Real sa = det3(ni, ff, v);   // raw signed area of triangle (ni, ff, v)
    const Real sb = -det3(ni, fl, v);  // raw signed area of triangle (ni, v, fl)
    const Real da = sa + sb;
    const Real momx = sa * (ni[0] + ff[0] + v[0]) / Real(3) + sb * (ni[0] + v[0] + fl[0]) / Real(3);
    const Real momy = sa * (ni[1] + ff[1] + v[1]) / Real(3) + sb * (ni[1] + v[1] + fl[1]) / Real(3);
    const Real momz = sa * (ni[2] + ff[2] + v[2]) / Real(3) + sb * (ni[2] + v[2] + fl[2]) / Real(3);
    dgx[ki] += Real(2) * da * ni[0] - momx;  // 2·da·n_k − mom  (numerator piece)
    dgy[ki] += Real(2) * da * ni[1] - momy;
    dgz[ki] += Real(2) * da * ni[2] - momz;
  }

  /// geomVolumeGrad tier (Phase 2): cell volume + the volume gradient dV/dn_k per plane, with NO area/
  /// centroid arrays exposed, NO sqrt anywhere, and only the 3 output arrays touched (so it keeps the
  /// compact cell's GPU occupancy). dV/dn_k = (2·A_k·n_k − m_k)/|n_k|; scattering the raw numerator
  /// 2·S_a·n_k − S_m per vertex (S_a = 2|n_k|·A_k, S_m = 2|n_k|·m_k), the per-facet fold is
  /// dV/dn_k = (numerator)/(2·nn[k]) — the |n_k| cancels into the stored nn=|n|². Caller zeroes the
  /// np-sized dgx/dgy/dgz (accumulate the numerator, then overwritten in place by the fold). Matches
  /// geometryPerVertex's closed form to round-off. Voronoi position force dV/dr_k = ½·(dgx,dgy,dgz)[k];
  /// the policy layer (Phase 3) owns that chain.
  KOKKOS_INLINE_FUNCTION void geomVolumeGrad(Real& vol, Real* dgx, Real* dgy, Real* dgz) const {
    Real V = 0;
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      int k1 = t0[t], k2 = t1[t], k3 = t2[t];
      Real n1[3], n2[3], n3[3];
      planeN(k1, n1); planeN(k2, n2); planeN(k3, n3);
      Real c1[3], c2[3], c3[3];
      xprod(n2, n3, c1); xprod(n3, n1, c2); xprod(n1, n2, c3);
      if (dot3(n1, c1) < Real(0)) {  // canonical D>0: swap planes 2,3 (normals, cross, indices)
        for (int a = 0; a < 3; ++a) { Real tm = n2[a]; n2[a] = n3[a]; n3[a] = tm; tm = c2[a]; c2[a] = c3[a]; c3[a] = tm; }
        int tk = k2; k2 = k3; k3 = tk;
      }
      const Real v[3] = {vx[t], vy[t], vz[t]};
      Real f12[3], f23[3], f31[3];
      edgeFoot(v, c3, f12); edgeFoot(v, c1, f23); edgeFoot(v, c2, f31);
      Real e[3];
      e[0] = n1[0] - n2[0]; e[1] = n1[1] - n2[1]; e[2] = n1[2] - n2[2]; V += det3(e, f12, v);
      e[0] = n2[0] - n3[0]; e[1] = n2[1] - n3[1]; e[2] = n2[2] - n3[2]; V += det3(e, f23, v);
      e[0] = n3[0] - n1[0]; e[1] = n3[1] - n1[1]; e[2] = n3[2] - n1[2]; V += det3(e, f31, v);
      scatterGradRaw(k1, n1, f12, f31, v, dgx, dgy, dgz);
      scatterGradRaw(k2, n2, f23, f12, v, dgx, dgy, dgz);
      scatterGradRaw(k3, n3, f31, f23, v, dgx, dgy, dgz);
    }
    for (int k = 0; k < np; ++k) {  // per-facet fold, sqrt-free: numerator/(2·nn)
      const Real inv = (nn[k] > Real(0)) ? Real(1) / (Real(2) * nn[k]) : Real(0);
      dgx[k] *= inv; dgy[k] *= inv; dgz[k] *= inv;
    }
    vol = V * (Real(1) / Real(6));
  }

  /// Scatter one facet's RAW area S_a (=Σdet3) and RAW first moment S_m (=Σdet3·centroid_tri) AT a vertex
  /// — scatterFacetMoment without the 1/(2|n|) factor (so NO sqrt). geomVolumeArea defers the shared
  /// 1/(2|n|) to its per-facet fold, where it cancels against |n| for the area-vector and the gradient.
  KOKKOS_INLINE_FUNCTION static void scatterAreaMomentRaw(int ki, const Real ni[3], const Real ff[3],
                                                          const Real fl[3], const Real v[3], Real* sa_acc,
                                                          Real* smx, Real* smy, Real* smz) {
    const Real sa = det3(ni, ff, v);   // raw signed area of triangle (ni, ff, v)
    const Real sb = -det3(ni, fl, v);  // raw signed area of triangle (ni, v, fl)
    sa_acc[ki] += sa + sb;
    smx[ki] += sa * (ni[0] + ff[0] + v[0]) / Real(3) + sb * (ni[0] + v[0] + fl[0]) / Real(3);
    smy[ki] += sa * (ni[1] + ff[1] + v[1]) / Real(3) + sb * (ni[1] + v[1] + fl[1]) / Real(3);
    smz[ki] += sa * (ni[2] + ff[2] + v[2]) / Real(3) + sb * (ni[2] + v[2] + fl[2]) / Real(3);
  }

  /// geomVolumeArea tier (Phase 2): cell volume + the outward area-VECTOR A_k·n_k/|n_k| AND the volume
  /// gradient dV/dn_k per plane — the full G2 physics set (areas for fluxes/momentum, gradient for force).
  /// Like geomVolumeGrad it is fully SQRT-FREE: both outputs are |n|²-normalized — areaVec_k =
  /// S_a·n_k/(2·nn[k]) and dV/dn_k = (2·S_a·n_k − S_m)/(2·nn[k]) — with the raw S_a (=2|n_k|·A_k) and
  /// S_m (=2|n_k|·m_k) scattered per vertex. No extra scratch: during the scatter `avx` holds S_a and
  /// (dgx,dgy,dgz) hold S_m; the per-facet fold overwrites them in place (so the compact-cell stack stays
  /// at the 6 caller arrays). Caller zeroes all six. The facet centroid, if wanted, is c_k = S_m/S_a
  /// (also sqrt-free) — derivable but not exposed here. Matches facetGeometry's areaVec/dv to round-off.
  KOKKOS_INLINE_FUNCTION void geomVolumeArea(Real& vol, Real* avx, Real* avy, Real* avz, Real* dgx,
                                             Real* dgy, Real* dgz) const {
    Real V = 0;
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      int k1 = t0[t], k2 = t1[t], k3 = t2[t];
      Real n1[3], n2[3], n3[3];
      planeN(k1, n1); planeN(k2, n2); planeN(k3, n3);
      Real c1[3], c2[3], c3[3];
      xprod(n2, n3, c1); xprod(n3, n1, c2); xprod(n1, n2, c3);
      if (dot3(n1, c1) < Real(0)) {  // canonical D>0: swap planes 2,3 (normals, cross, indices)
        for (int a = 0; a < 3; ++a) { Real tm = n2[a]; n2[a] = n3[a]; n3[a] = tm; tm = c2[a]; c2[a] = c3[a]; c3[a] = tm; }
        int tk = k2; k2 = k3; k3 = tk;
      }
      const Real v[3] = {vx[t], vy[t], vz[t]};
      Real f12[3], f23[3], f31[3];
      edgeFoot(v, c3, f12); edgeFoot(v, c1, f23); edgeFoot(v, c2, f31);
      Real e[3];
      e[0] = n1[0] - n2[0]; e[1] = n1[1] - n2[1]; e[2] = n1[2] - n2[2]; V += det3(e, f12, v);
      e[0] = n2[0] - n3[0]; e[1] = n2[1] - n3[1]; e[2] = n2[2] - n3[2]; V += det3(e, f23, v);
      e[0] = n3[0] - n1[0]; e[1] = n3[1] - n1[1]; e[2] = n3[2] - n1[2]; V += det3(e, f31, v);
      scatterAreaMomentRaw(k1, n1, f12, f31, v, avx, dgx, dgy, dgz);  // avx <- S_a, (dgx,dgy,dgz) <- S_m
      scatterAreaMomentRaw(k2, n2, f23, f12, v, avx, dgx, dgy, dgz);
      scatterAreaMomentRaw(k3, n3, f31, f23, v, avx, dgx, dgy, dgz);
    }
    for (int k = 0; k < np; ++k) {  // per-facet fold, sqrt-free
      const Real inv = (nn[k] > Real(0)) ? Real(1) / (Real(2) * nn[k]) : Real(0);
      const Real sa = avx[k];  // raw area S_a (accumulated here during the scatter)
      const Real smx = dgx[k], smy = dgy[k], smz = dgz[k];
      dgx[k] = (Real(2) * sa * n[k][0] - smx) * inv;  // dV/dn_k
      dgy[k] = (Real(2) * sa * n[k][1] - smy) * inv;
      dgz[k] = (Real(2) * sa * n[k][2] - smz) * inv;
      avx[k] = sa * n[k][0] * inv;  // outward area-vector A_k·n_k/|n_k|
      avy[k] = sa * n[k][1] * inv;
      avz[k] = sa * n[k][2] * inv;
    }
    vol = V * (Real(1) / Real(6));
  }

  /// Adjoint of a cross-product edge direction: (∂(N[p]×N[q])/∂n_j)^T w. Used by the analytic geomVolumeAreaGrad.
  KOKKOS_INLINE_FUNCTION static void edgeCrossAdj(int j, int p, int q, const Real Nn[3][3],
                                                  const Real w[3], Real out[3]) {
    if (j == p) {
      xprod(Nn[q], w, out);
    } else if (j == q) {
      Real tm[3]; xprod(Nn[p], w, tm);
      out[0] = -tm[0]; out[1] = -tm[1]; out[2] = -tm[2];
    } else {
      out[0] = out[1] = out[2] = Real(0);
    }
  }

  /// geomFull primitive: the per-dual-triangle ("subtetrahedron") area-Jacobian block — the building
  /// block of the FULL coupled dA_k/dn_l (the OT/power-diagram Hessian), with NO ordering and NO
  /// adjacency. ANALYTIC, no-recompute version: it reuses the CACHED vertex v and computes the per-triangle
  /// intermediates (edge directions, feet) once in Real, then forms the derivatives in closed form. The
  /// vertex derivative is the rank-1 ∂v/∂n_j = (c_j/D)(2n_j − v)^T (c_j the cofactor, D the determinant);
  /// the area numerator N_i=det3(n_i,g_i,v) is differentiated trilinearly (∂/∂n_i = g_i×v, ∂/∂g via the
  /// foot adjoints, ∂/∂v via the rank-1 above), and the 1/(2|n_i|) factor adds the −δ_ij A_i n_i/nn_i term.
  /// Outputs: `pl[3]` = the (canonically ordered) plane indices; `contrib[3]` = the area contributions
  /// (Σ over triangles reproduces facetAreasPerVertex's area[k]); `grad[i][j][c]` = ∂contrib_i/∂n_{pl[j]}[c].
  /// Consumer gathers A_{pl[i]} += contrib[i], dA_{pl[i]}/dn_{pl[j]} += grad[i][j], in any order.
  KOKKOS_INLINE_FUNCTION void geomVolumeAreaGrad(int t, int pl[3], Real contrib[3], Real grad[3][3][3]) const {
    // planes + canonical swap (matches facetAreasPerVertex), reusing the cached vertex — nothing recomputed.
    // Every loop is VOR_UNROLL'd so the small local arrays scalarize into registers on GPU (no local-memory
    // spill from runtime indices); on CPU the macro is empty.
    int kk[3] = {t0[t], t1[t], t2[t]};
    Real Nn[3][3];
    VOR_UNROLL for (int a = 0; a < 3; ++a) { Nn[a][0] = n[kk[a]][0]; Nn[a][1] = n[kk[a]][1]; Nn[a][2] = n[kk[a]][2]; }
    { Real c01[3]; xprod(Nn[1], Nn[2], c01);
      if (dot3(Nn[0], c01) < Real(0)) {
        VOR_UNROLL for (int d = 0; d < 3; ++d) { Real tm = Nn[1][d]; Nn[1][d] = Nn[2][d]; Nn[2][d] = tm; }
        int tk = kk[1]; kk[1] = kk[2]; kk[2] = tk;
      }
    }
    pl[0] = kk[0]; pl[1] = kk[1]; pl[2] = kk[2];
    const Real v[3] = {vx[t], vy[t], vz[t]};
    Real nn3[3], u[3][3];  // u[a] = 2 n_a − v
    VOR_UNROLL for (int a = 0; a < 3; ++a) { nn3[a] = dot3(Nn[a], Nn[a]); VOR_UNROLL for (int d = 0; d < 3; ++d) u[a][d] = Real(2) * Nn[a][d] - v[d]; }
    // edge directions e[m] = N[eP]×N[eQ] for edges (0,1),(1,2),(2,0); cofactor of plane j is c_j = e[(j+1)%3]
    const int eP[3] = {0, 1, 2}, eQ[3] = {1, 2, 0};
    Real e[3][3], ecc[3], es[3], F[3][3];
    VOR_UNROLL for (int m = 0; m < 3; ++m) {
      xprod(Nn[eP[m]], Nn[eQ[m]], e[m]);
      ecc[m] = dot3(e[m], e[m]);
      const Real evc = dot3(v, e[m]);
      es[m] = (ecc[m] > Real(0)) ? evc / ecc[m] : Real(0);
      VOR_UNROLL for (int d = 0; d < 3; ++d) F[m][d] = v[d] - es[m] * e[m][d];  // foot of v on edge line m
    }
    const Real D = dot3(Nn[0], e[1]);  // e[1] = N1×N2 = cofactor c_0;  D = det(n0,n1,n2) > 0
    Real g[3][3], den[3];
    VOR_UNROLL for (int i = 0; i < 3; ++i) {  // facet i: g_i = F[i] − F[(i+2)%3]
      const int mb = (i + 2) % 3;
      VOR_UNROLL for (int d = 0; d < 3; ++d) g[i][d] = F[i][d] - F[mb][d];
      const Real Ni = det3(Nn[i], g[i], v);
      den[i] = Real(2) * Kokkos::sqrt(nn3[i]);
      contrib[i] = (den[i] > Real(0)) ? Ni / den[i] : Real(0);
    }
    VOR_UNROLL for (int i = 0; i < 3; ++i) {
      Real wi[3]; xprod(v, Nn[i], wi);         // w_i = v×n_i  (∂det3(n_i,·,v)/∂g)
      Real nixg[3]; xprod(Nn[i], g[i], nixg);  // n_i×g_i      (∂det3(n_i,g_i,·)/∂v)
      const int mb = (i + 2) % 3;
      VOR_UNROLL for (int j = 0; j < 3; ++j) {
        const Real* cj = e[(j + 1) % 3];  // cofactor of plane j
        Real gN[3] = {0, 0, 0};
        if (i == j) { Real gxv[3]; xprod(g[i], v, gxv); VOR_UNROLL for (int d = 0; d < 3; ++d) gN[d] += gxv[d]; }  // ∂/∂n_i
        const Real alpha = dot3(nixg, cj) / D;  // ∂/∂v (rank-1)
        VOR_UNROLL for (int d = 0; d < 3; ++d) gN[d] += alpha * u[j][d];
        // ∂/∂g = (∂F[i]/∂n_j − ∂F[(i+2)%3]/∂n_j)^T w_i  — the two foot adjoints
        VOR_UNROLL for (int s = 0; s < 2; ++s) {
          const int m = (s == 0) ? i : mb;
          const Real sgn = (s == 0) ? Real(1) : Real(-1);
          const int p = eP[m], q = eQ[m];
          const Real cjw = dot3(cj, wi);     // for vAdj
          const Real cje = dot3(cj, e[m]);   // for dS
          const Real emw = dot3(e[m], wi);
          Real eaW[3], eaV[3], eaE[3];
          edgeCrossAdj(j, p, q, Nn, wi, eaW);
          edgeCrossAdj(j, p, q, Nn, v, eaV);
          edgeCrossAdj(j, p, q, Nn, e[m], eaE);
          const Real inv_ecc = (ecc[m] > Real(0)) ? Real(1) / ecc[m] : Real(0);
          Real dS[3];  // ∂s_m/∂n_j
          VOR_UNROLL for (int d = 0; d < 3; ++d)
            dS[d] = ((cje / D) * u[j][d] + eaV[d]) * inv_ecc - (es[m] * inv_ecc) * Real(2) * eaE[d];
          // (∂F[m]/∂n_j)^T w_i = (c_j·w_i/D) u_j − (e_m·w_i) dS − s_m (∂e_m/∂n_j)^T w_i
          VOR_UNROLL for (int d = 0; d < 3; ++d)
            gN[d] += sgn * ((cjw / D) * u[j][d] - emw * dS[d] - es[m] * eaW[d]);
        }
        const Real invden = (den[i] > Real(0)) ? Real(1) / den[i] : Real(0);
        VOR_UNROLL for (int d = 0; d < 3; ++d) grad[i][j][d] = gN[d] * invden;
        if (i == j) {  // 1/(2|n_i|) factor: −A_i n_i/nn_i
          const Real f = (nn3[i] > Real(0)) ? contrib[i] / nn3[i] : Real(0);
          VOR_UNROLL for (int d = 0; d < 3; ++d) grad[i][j][d] -= f * Nn[i][d];
        }
      }
    }
  }

  /// Reference (forward-AD) implementation of the per-triangle area-Jacobian, kept only as the
  /// machine-precision ORACLE for the analytic geomVolumeAreaGrad (it recomputes everything in dual arithmetic,
  /// including the vertex — the redundancy the analytic version removes). Same outputs as geomVolumeAreaGrad.
  KOKKOS_INLINE_FUNCTION void geomVolumeAreaGradAD(int t, int pl[3], Real contrib[3], Real grad[3][3][3]) const {
    using D = detail::Dual<Real, 9>;
    int k1 = t0[t], k2 = t1[t], k3 = t2[t];
    Real rn1[3] = {n[k1][0], n[k1][1], n[k1][2]};
    Real rn2[3] = {n[k2][0], n[k2][1], n[k2][2]};
    Real rn3[3] = {n[k3][0], n[k3][1], n[k3][2]};
    {  // canonical D>0 swap on the real values (same as facetAreasPerVertex) so contributions land right
      Real cc[3]; xprod(rn2, rn3, cc);
      if (dot3(rn1, cc) < Real(0)) {
        for (int a = 0; a < 3; ++a) { Real tm = rn2[a]; rn2[a] = rn3[a]; rn3[a] = tm; }
        int tk = k2; k2 = k3; k3 = tk;
      }
    }
    pl[0] = k1; pl[1] = k2; pl[2] = k3;
    D n1[3], n2[3], n3[3];
    for (int a = 0; a < 3; ++a) {  // seed: n1 <- slots 0..2, n2 <- 3..5, n3 <- 6..8
      n1[a] = detail::dseed<Real, 9>(rn1[a], a);
      n2[a] = detail::dseed<Real, 9>(rn2[a], 3 + a);
      n3[a] = detail::dseed<Real, 9>(rn3[a], 6 + a);
    }
    const D nn1 = detail::ddot<Real, 9>(n1, n1), nn2 = detail::ddot<Real, 9>(n2, n2), nn3 = detail::ddot<Real, 9>(n3, n3);
    D c1[3], c2[3], c3[3];
    detail::dcross<Real, 9>(n2, n3, c1); detail::dcross<Real, 9>(n3, n1, c2); detail::dcross<Real, 9>(n1, n2, c3);
    const D Dd = detail::ddot<Real, 9>(n1, c1);  // det
    D v[3];
    for (int a = 0; a < 3; ++a) v[a] = (nn1 * c1[a] + nn2 * c2[a] + nn3 * c3[a]) / Dd;  // Cramer vertex
    D f12[3], f23[3], f31[3];
    detail::dedgeFoot<Real, 9>(v, c3, f12); detail::dedgeFoot<Real, 9>(v, c1, f23); detail::dedgeFoot<Real, 9>(v, c2, f31);
    const D two = detail::dnum<Real, 9>(Real(2));
    D g[3], aa[3];
    for (int a = 0; a < 3; ++a) g[a] = f12[a] - f31[a];
    aa[0] = detail::ddet3<Real, 9>(n1, g, v) / (two * detail::dsqrt<Real, 9>(nn1));
    for (int a = 0; a < 3; ++a) g[a] = f23[a] - f12[a];
    aa[1] = detail::ddet3<Real, 9>(n2, g, v) / (two * detail::dsqrt<Real, 9>(nn2));
    for (int a = 0; a < 3; ++a) g[a] = f31[a] - f23[a];
    aa[2] = detail::ddet3<Real, 9>(n3, g, v) / (two * detail::dsqrt<Real, 9>(nn3));
    for (int i = 0; i < 3; ++i) {
      contrib[i] = aa[i].v;
      for (int j = 0; j < 3; ++j)
        for (int c = 0; c < 3; ++c) grad[i][j][c] = aa[i].d[3 * j + c];
    }
  }

  /// MERGED vertex-local geometry: cell volume + per-facet area + per-facet first moment ∫x dA, all in
  /// ONE pass (shares n,c,D,v,feet per vertex). The production G1+G2 kernel — sort-free, adjacency-free.
  /// Caller zeroes the np-sized arrays. With connector r_k = 2·n[k] (so |r_k| = 2·sqrt(nn[k])), derive
  /// per facet k:
  ///   areaVec_k = area[k] · r_k/|r_k| = area[k] · n[k]/|n[k]|   (outward, toward the neighbour)
  ///   centroid_k = (mx,my,mz)[k]/area[k]
  ///   force dV_k = (area[k]/|r_k|)·(r_k − centroid_k)
  KOKKOS_INLINE_FUNCTION void geometryPerVertex(Real& vol, Real* area, Real* mx, Real* my, Real* mz) const {
    Real V = 0;
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      int k1 = t0[t], k2 = t1[t], k3 = t2[t];
      Real n1[3], n2[3], n3[3];
      planeN(k1, n1); planeN(k2, n2); planeN(k3, n3);
      Real c1[3], c2[3], c3[3];
      xprod(n2, n3, c1); xprod(n3, n1, c2); xprod(n1, n2, c3);
      if (dot3(n1, c1) < Real(0)) {  // canonical D>0: swap planes 2,3 (normals, cross, indices)
        for (int a = 0; a < 3; ++a) { Real tm = n2[a]; n2[a] = n3[a]; n3[a] = tm; tm = c2[a]; c2[a] = c3[a]; c3[a] = tm; }
        int tk = k2; k2 = k3; k3 = tk;
      }
      const Real v[3] = {vx[t], vy[t], vz[t]};
      Real f12[3], f23[3], f31[3];
      edgeFeet3(v, c1, c2, c3, f23, f31, f12);  // 3 divides → 1; feet kept for the moment scatter
      Real e[3];
      e[0] = n1[0] - n2[0]; e[1] = n1[1] - n2[1]; e[2] = n1[2] - n2[2]; V += det3(e, f12, v);
      e[0] = n2[0] - n3[0]; e[1] = n2[1] - n3[1]; e[2] = n2[2] - n3[2]; V += det3(e, f23, v);
      e[0] = n3[0] - n1[0]; e[1] = n3[1] - n1[1]; e[2] = n3[2] - n1[2]; V += det3(e, f31, v);
      scatterFacetMoment(k1, n1, f12, f31, v, area, mx, my, mz);
      scatterFacetMoment(k2, n2, f23, f12, v, area, mx, my, mz);
      scatterFacetMoment(k3, n3, f31, f23, v, area, mx, my, mz);
    }
    vol = V * (Real(1) / Real(6));
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
    const Real r[3] = {Real(2) * n[k][0], Real(2) * n[k][1], Real(2) * n[k][2]};  // connector r = 2·foot
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
