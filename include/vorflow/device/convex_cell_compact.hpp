/**
 * @file convex_cell_compact.hpp
 * \brief Memory-lean ConvexCell variant — packed triangles, NO cached vertex.
 *
 * The single-thread ConvexCell is local-memory-resident (its arrays are dynamically indexed, so
 * CUDA places them on the per-thread stack — 3536 B for MAXP=64/MAXT=112, confirmed by ptxas).
 * The construct is therefore bound on local-memory traffic, not FP (RTX 5080 has ~4× idle FP32).
 * This variant attacks the footprint to test whether the kernel is L1/L2-capacity-bound:
 *
 *   - triangle = ONE uint32: bits 0-9 / 10-19 / 20-29 = the three plane indices (MAXP<=1024),
 *     bit 31 = alive.  (16 B  ->  4 B  per triangle; the cached vx/vy/vz[MAXT] = 38% of the cell
 *     is gone entirely.)
 *   - the dual vertex is RECOMPUTED from its three planes (Cramer) wherever it is needed, trading
 *     ~17 KB/cell of local vertex traffic for Cramer FP32 on the idle units.
 *   - pnbr is dropped from the hot struct (neighbour ids are recovered from the plane index at the
 *     end if needed), so a plane is just pn[3]+pd = 16 B.
 *
 * Algorithm is a faithful port of convex_cell.hpp (same clip / horizon retriangulation / security
 * radius); only the storage and the cache-vs-recompute choice differ. A/B'd in bench_construct_compact.
 */
#pragma once
#include <Kokkos_Core.hpp>

namespace vor {
namespace device {

template <class Real, int MAXP = 64, int MAXT = 112>
struct ConvexCellCompact {
  static_assert(MAXP <= 1024, "plane index must fit in 10 bits");
  Real pn[MAXP][3];        ///< plane normals (seed-relative; = neighbour rel-pos for Voronoi)
  Real pd[MAXP];           ///< plane offsets: half-space {x : pn·x <= pd}
  unsigned int tri[MAXT];  ///< packed triangle: [p0:10][p1:10][p2:10][_:1][alive:1]
  int np;                  ///< number of planes
  int nt;                  ///< triangle high-water mark
  bool overflow;           ///< set if MAXP/MAXT exceeded -> cell invalid, caller falls back

  static constexpr unsigned int ALIVE = 0x80000000u;
  KOKKOS_INLINE_FUNCTION static unsigned int pack(int a, int b, int c, bool alive) {
    return (unsigned)a | ((unsigned)b << 10) | ((unsigned)c << 20) | (alive ? ALIVE : 0u);
  }
  KOKKOS_INLINE_FUNCTION static int p0(unsigned int t) { return t & 0x3ff; }
  KOKKOS_INLINE_FUNCTION static int p1(unsigned int t) { return (t >> 10) & 0x3ff; }
  KOKKOS_INLINE_FUNCTION static int p2(unsigned int t) { return (t >> 20) & 0x3ff; }
  KOKKOS_INLINE_FUNCTION static bool isAlive(unsigned int t) { return t & ALIVE; }

  /// Dual vertex of triangle word `t` (where its three planes meet, Cramer's rule). Recomputed
  /// on demand — there is no cache. The three planes are a tiny, hot, L1-resident array.
  KOKKOS_INLINE_FUNCTION void vertexOf(unsigned int t, Real v[3]) const {
    const Real* a = pn[p0(t)];
    const Real* b = pn[p1(t)];
    const Real* c = pn[p2(t)];
    const Real da = pd[p0(t)], db = pd[p1(t)], dc = pd[p2(t)];
    const Real bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]};
    const Real ca[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]};
    const Real ab[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    const Real det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
    const Real inv = det != Real(0) ? Real(1) / det : Real(0);
    v[0] = (da * bc[0] + db * ca[0] + dc * ab[0]) * inv;
    v[1] = (da * bc[1] + db * ca[1] + dc * ab[1]) * inv;
    v[2] = (da * bc[2] + db * ca[2] + dc * ab[2]) * inv;
  }

  /// Seed the cell with the big cuboid of extent (L0,L1,L2) centred on the seed.
  KOKKOS_INLINE_FUNCTION void initBox(Real L0, Real L1, Real L2) {
    const Real h[3] = {Real(0.5) * L0, Real(0.5) * L1, Real(0.5) * L2};
    for (int ax = 0; ax < 3; ++ax)
      for (int s = 0; s < 2; ++s) {
        const int k = 2 * ax + s;
        pn[k][0] = pn[k][1] = pn[k][2] = 0;
        pn[k][ax] = (s == 0) ? Real(1) : Real(-1);
        pd[k] = h[ax];
      }
    np = 6;
    nt = 0;
    overflow = false;
    for (int sx = 0; sx < 2; ++sx)
      for (int sy = 0; sy < 2; ++sy)
        for (int sz = 0; sz < 2; ++sz) tri[nt++] = pack(0 + sx, 2 + sy, 4 + sz, true);
  }

  KOKKOS_INLINE_FUNCTION Real maxVertexRsq() const {
    Real m = 0, v[3];
    for (int t = 0; t < nt; ++t) {
      if (!isAlive(tri[t])) continue;
      vertexOf(tri[t], v);
      const Real r = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      if (r > m) m = r;
    }
    return m;
  }

  /// Find a *live* triangle != self that contains both plane indices x and y.
  KOKKOS_INLINE_FUNCTION int findSharing(int self, int x, int y) const {
    for (int s = 0; s < nt; ++s) {
      if (s == self || !isAlive(tri[s])) continue;
      const int a = p0(tri[s]), b = p1(tri[s]), c = p2(tri[s]);
      const bool hasX = (a == x || b == x || c == x);
      const bool hasY = (a == y || b == y || c == y);
      if (hasX && hasY) return s;
    }
    return -1;
  }

  KOKKOS_INLINE_FUNCTION int allocTri() {
    for (int s = 0; s < nt; ++s)
      if (!isAlive(tri[s])) return s;
    if (nt < MAXT) return nt++;
    overflow = true;
    return -1;
  }

  /// Add plane (n, d) and clip the cell by it. Returns true if the cell was modified.
  KOKKOS_INLINE_FUNCTION bool clip(const Real n[3], Real d, int /*nbr*/) {
    if (np >= MAXP) {
      overflow = true;
      return false;
    }
    const int pi = np;  // tentative index
    pn[pi][0] = n[0];
    pn[pi][1] = n[1];
    pn[pi][2] = n[2];
    pd[pi] = d;

    // Mark triangles whose dual vertex is outside the new half-space (n·v > d).
    bool kill[MAXT];
    bool any = false;
    Real v[3];
    for (int t = 0; t < nt; ++t) {
      kill[t] = false;
      if (!isAlive(tri[t])) continue;
      vertexOf(tri[t], v);
      if (n[0] * v[0] + n[1] * v[1] + n[2] * v[2] - d > 0) {
        kill[t] = true;
        any = true;
      }
    }
    if (!any) return false;  // candidate does not cut -> no-op, plane not committed

    np = pi + 1;  // commit the plane

    // Horizon: collect (x,y) edges between a killed triangle and a live sharing triangle.
    unsigned char nA[MAXT], nB[MAXT];
    int nnew = 0;
    for (int t = 0; t < nt; ++t) {
      if (!kill[t]) continue;
      const int vtx[3] = {p0(tri[t]), p1(tri[t]), p2(tri[t])};
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
      if (kill[t]) tri[t] &= ~ALIVE;
    // Add the new fan around plane pi.
    for (int i = 0; i < nnew; ++i) {
      const int slot = allocTri();
      if (slot < 0) break;
      tri[slot] = pack(nA[i], nB[i], pi, true);
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION int countFaces() const {
    int nf = 0;
    for (int k = 0; k < np; ++k) {
      bool used = false;
      for (int t = 0; t < nt && !used; ++t)
        if (isAlive(tri[t]) && (p0(tri[t]) == k || p1(tri[t]) == k || p2(tri[t]) == k)) used = true;
      if (used) ++nf;
    }
    return nf;
  }

  static constexpr int MAXFV = 28;

  /// Gather + CCW-order face k's vertices (recomputed). Returns the count, or <3 if not a face.
  KOKKOS_INLINE_FUNCTION int faceOrdered(int k, Real fx[MAXFV], Real fy[MAXFV], Real fz[MAXFV]) const {
    int m = 0;
    Real v[3];
    for (int t = 0; t < nt; ++t) {
      if (!isAlive(tri[t])) continue;
      if (p0(tri[t]) != k && p1(tri[t]) != k && p2(tri[t]) != k) continue;
      if (m < MAXFV) {
        vertexOf(tri[t], v);
        fx[m] = v[0];
        fy[m] = v[1];
        fz[m] = v[2];
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
      e1[0] = 0; e1[1] = -un[2]; e1[2] = un[1];
    } else if (Kokkos::fabs(un[1]) <= Kokkos::fabs(un[2])) {
      e1[0] = -un[2]; e1[1] = 0; e1[2] = un[0];
    } else {
      e1[0] = -un[1]; e1[1] = un[0]; e1[2] = 0;
    }
    const Real e1l = Kokkos::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    e1[0] /= e1l; e1[1] /= e1l; e1[2] /= e1l;
    const Real e2[3] = {un[1] * e1[2] - un[2] * e1[1], un[2] * e1[0] - un[0] * e1[2],
                        un[0] * e1[1] - un[1] * e1[0]};
    Real cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < m; ++i) { cx += fx[i]; cy += fy[i]; cz += fz[i]; }
    cx /= m; cy /= m; cz /= m;
    Real ang[MAXFV];
    for (int i = 0; i < m; ++i) {
      const Real dx = fx[i] - cx, dy = fy[i] - cy, dz = fz[i] - cz;
      ang[i] = Kokkos::atan2(dx * e2[0] + dy * e2[1] + dz * e2[2], dx * e1[0] + dy * e1[1] + dz * e1[2]);
    }
    for (int i = 1; i < m; ++i) {
      Real ka = ang[i], kx = fx[i], ky = fy[i], kz = fz[i];
      int j = i - 1;
      while (j >= 0 && ang[j] > ka) {
        ang[j + 1] = ang[j]; fx[j + 1] = fx[j]; fy[j + 1] = fy[j]; fz[j + 1] = fz[j]; --j;
      }
      ang[j + 1] = ka; fx[j + 1] = kx; fy[j + 1] = ky; fz[j + 1] = kz;
    }
    return m;
  }

  KOKKOS_INLINE_FUNCTION static void polyAreaVec(const Real fx[], const Real fy[], const Real fz[],
                                                 int m, Real A[3]) {
    Real ax = 0, ay = 0, az = 0;
    for (int i = 0; i < m; ++i) {
      const int j = (i + 1 == m) ? 0 : i + 1;
      ax += fy[i] * fz[j] - fz[i] * fy[j];
      ay += fz[i] * fx[j] - fx[i] * fz[j];
      az += fx[i] * fy[j] - fy[i] * fx[j];
    }
    A[0] = Real(0.5) * ax; A[1] = Real(0.5) * ay; A[2] = Real(0.5) * az;
  }

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
};

}  // namespace device
}  // namespace vor
