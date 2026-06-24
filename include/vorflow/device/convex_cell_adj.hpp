/**
 * @file convex_cell_adj.hpp
 * \brief SOTA-style ConvexCell — triangle adjacency + free list, adjacency-walked horizon.
 *
 * Port of the geogram VBW::ConvexCell representation (Ray/Sokolov/Lefebvre/Lévy, "Meshless Voronoi
 * on the GPU", TOG 2018). The point of the redesign: our original clip does the horizon
 * retriangulation with `findSharing` (an O(n) scan per horizon edge) and allocates with an O(n)
 * scan — and those O(n) searches are the *measured* construct bottleneck. SOTA does both in O(1):
 *
 *   - each triangle stores its 3 vertex (plane) indices AND its 3 adjacent triangles (one per edge);
 *   - the conflict zone's border is found by walking adjacency (O(border)), one new triangle/edge;
 *   - a free list gives O(1) allocation/deletion (no scan for a dead slot);
 *   - dual vertices are recomputed on demand (no per-triangle cache — footprint is not the bound).
 *
 * Edge convention: edge e of triangle (v0,v1,v2) is OPPOSITE vertex e, i.e. edge e connects
 * v_{(e+1)%3} and v_{(e+2)%3}; tadj[t][e] is the triangle across that edge.
 */
#pragma once
#include <Kokkos_Core.hpp>

namespace vor {
namespace device {

template <class Real, int MAXP = 64, int MAXT = 112, bool CACHE = false>
struct ConvexCellAdj {
  static_assert(MAXP <= 65535, "plane index must fit in unsigned short");
  Real pn[MAXP][3];               ///< plane normals (seed-relative)
  Real pd[MAXP];                  ///< plane offsets: half-space {x : pn·x <= pd}
  int np;                         ///< number of planes
  unsigned short tv[MAXT][3];     ///< triangle = three plane (vertex) indices
  unsigned short tadj[MAXT][3];   ///< adjacent triangle across edge e (opposite vertex e)
  Real vcx[CACHE ? MAXT : 1];     ///< cached dual vertex (only when CACHE) — avoids the per-test division
  Real vcy[CACHE ? MAXT : 1];
  Real vcz[CACHE ? MAXT : 1];
  bool alive[MAXT];               ///< triangle live flag
  int nt;                         ///< triangle high-water mark
  int freeTop;                    ///< free-slot stack top
  unsigned short freeSlot[MAXT];  ///< stack of reusable (deleted) triangle slots
  bool overflow;

  static constexpr int MAXBORDER = 32;  // max conflict-zone border length (= new triangles per clip)

  KOKKOS_INLINE_FUNCTION int allocTri() {
    if (freeTop > 0) return freeSlot[--freeTop];
    if (nt < MAXT) return nt++;
    overflow = true;
    return -1;
  }
  KOKKOS_INLINE_FUNCTION void freeTri(int t) {
    alive[t] = false;
    if (freeTop < MAXT) freeSlot[freeTop++] = (unsigned short)t;
  }

  /// Dual vertex of triangle t (Cramer over its three planes).
  KOKKOS_INLINE_FUNCTION void computeVertex(int t, Real v[3]) const {
    const Real* a = pn[tv[t][0]];
    const Real* b = pn[tv[t][1]];
    const Real* c = pn[tv[t][2]];
    const Real da = pd[tv[t][0]], db = pd[tv[t][1]], dc = pd[tv[t][2]];
    const Real bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]};
    const Real ca[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]};
    const Real ab[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    const Real det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
    const Real inv = det != Real(0) ? Real(1) / det : Real(0);
    v[0] = (da * bc[0] + db * ca[0] + dc * ab[0]) * inv;
    v[1] = (da * bc[1] + db * ca[1] + dc * ab[1]) * inv;
    v[2] = (da * bc[2] + db * ca[2] + dc * ab[2]) * inv;
  }
  /// Cache triangle t's vertex (no-op unless CACHE). Call on every triangle creation.
  KOKKOS_INLINE_FUNCTION void touchVertex(int t) {
    if constexpr (CACHE) {
      Real v[3];
      computeVertex(t, v);
      vcx[t] = v[0]; vcy[t] = v[1]; vcz[t] = v[2];
    }
  }
  /// Read triangle t's vertex (cached read, or recompute).
  KOKKOS_INLINE_FUNCTION void vertexOf(int t, Real v[3]) const {
    if constexpr (CACHE) {
      v[0] = vcx[t]; v[1] = vcy[t]; v[2] = vcz[t];
    } else {
      computeVertex(t, v);
    }
  }

  /// Seed the cell with the big cuboid of extent (L0,L1,L2). Planes 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z.
  /// The 8 corner triangles carry full adjacency: corner (sx,sy,sz) holds planes (x,y,z); its edge
  /// e (opposite the axis-e plane) is shared with the corner differing in axis e.
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
    nt = 8;
    freeTop = 0;
    overflow = false;
    auto cid = [](int sx, int sy, int sz) { return sx * 4 + sy * 2 + sz; };
    for (int sx = 0; sx < 2; ++sx)
      for (int sy = 0; sy < 2; ++sy)
        for (int sz = 0; sz < 2; ++sz) {
          const int t = cid(sx, sy, sz);
          tv[t][0] = (unsigned short)(0 + sx);  // x-plane (vertex 0)
          tv[t][1] = (unsigned short)(2 + sy);  // y-plane (vertex 1)
          tv[t][2] = (unsigned short)(4 + sz);  // z-plane (vertex 2)
          tadj[t][0] = (unsigned short)cid(1 - sx, sy, sz);  // edge 0 opposite x -> flip x
          tadj[t][1] = (unsigned short)cid(sx, 1 - sy, sz);  // edge 1 opposite y -> flip y
          tadj[t][2] = (unsigned short)cid(sx, sy, 1 - sz);  // edge 2 opposite z -> flip z
          alive[t] = true;
          touchVertex(t);
        }
  }

  KOKKOS_INLINE_FUNCTION Real maxVertexRsq() const {
    Real m = 0, v[3];
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      vertexOf(t, v);
      const Real r = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      if (r > m) m = r;
    }
    return m;
  }

  /// Clip by the half-space {x : n·x <= d}. Returns true if the cell was modified.
  KOKKOS_INLINE_FUNCTION bool clip(const Real n[3], Real d, int /*nbr*/) {
    if (np >= MAXP) {
      overflow = true;
      return false;
    }
    const int pi = np;  // tentative plane index
    pn[pi][0] = n[0];
    pn[pi][1] = n[1];
    pn[pi][2] = n[2];
    pd[pi] = d;

    // 1. mark conflict triangles (dual vertex outside the new half-space). Linear scan (robust;
    //    SOTA does the same — "no more than a few tens of vertices").
    const int nt0 = nt;
    bool conflict[MAXT];
    bool any = false;
    Real v[3];
    for (int t = 0; t < nt0; ++t) {
      conflict[t] = false;
      if (!alive[t]) continue;
      vertexOf(t, v);
      if (n[0] * v[0] + n[1] * v[1] + n[2] * v[2] - d > 0) {
        conflict[t] = true;
        any = true;
      }
    }
    if (!any) return false;  // candidate does not cut -> plane not committed
    np = pi + 1;             // commit

    // 2. border edges: edge e of a conflict triangle whose neighbour is NOT in conflict. Make one
    //    new triangle (va, vb, pi) per border edge; wire it to the surviving neighbour across the
    //    (va,vb) edge (= new triangle's edge 2, opposite the new plane pi).
    int newT[MAXBORDER];
    int nnew = 0;
    for (int t = 0; t < nt0; ++t) {
      if (!conflict[t]) continue;
      for (int e = 0; e < 3; ++e) {
        const int tp = tadj[t][e];
        if (tp < nt0 && conflict[tp]) continue;  // interior to the conflict zone
        const int va = tv[t][(e + 1) % 3], vb = tv[t][(e + 2) % 3];
        const int N = allocTri();
        if (N < 0) return true;  // overflow flagged
        if (nnew >= MAXBORDER) {
          overflow = true;
          return true;
        }
        tv[N][0] = (unsigned short)va;
        tv[N][1] = (unsigned short)vb;
        tv[N][2] = (unsigned short)pi;
        alive[N] = true;
        touchVertex(N);
        tadj[N][2] = (unsigned short)tp;  // edge 2 (opposite pi) faces the surviving neighbour
        for (int e2 = 0; e2 < 3; ++e2)
          if (tadj[tp][e2] == t) {
            tadj[tp][e2] = (unsigned short)N;
            break;
          }
        newT[nnew++] = N;
      }
    }

    // 3. ring-link the new triangles. N=(va,vb,pi): edge 1 (opposite vb) = (va,pi) joins the other
    //    new triangle sharing va; edge 0 (opposite va) = (vb,pi) joins the one sharing vb.
    for (int a = 0; a < nnew; ++a) {
      const int Ni = newT[a];
      const int va = tv[Ni][0], vb = tv[Ni][1];
      for (int b = 0; b < nnew; ++b) {
        if (b == a) continue;
        const int Nj = newT[b];
        if (tv[Nj][0] == va || tv[Nj][1] == va) tadj[Ni][1] = (unsigned short)Nj;
        if (tv[Nj][0] == vb || tv[Nj][1] == vb) tadj[Ni][0] = (unsigned short)Nj;
      }
    }

    // 4. retire conflict triangles (free list).
    for (int t = 0; t < nt0; ++t)
      if (conflict[t]) freeTri(t);
    return true;
  }

  KOKKOS_INLINE_FUNCTION int countFaces() const {
    int nf = 0;
    for (int k = 0; k < np; ++k) {
      bool used = false;
      for (int t = 0; t < nt && !used; ++t)
        if (alive[t] && (tv[t][0] == k || tv[t][1] == k || tv[t][2] == k)) used = true;
      if (used) ++nf;
    }
    return nf;
  }

  static constexpr int MAXFV = 28;

  KOKKOS_INLINE_FUNCTION int faceOrdered(int k, Real fx[MAXFV], Real fy[MAXFV], Real fz[MAXFV]) const {
    int m = 0;
    Real v[3];
    for (int t = 0; t < nt; ++t) {
      if (!alive[t]) continue;
      if (tv[t][0] != k && tv[t][1] != k && tv[t][2] != k) continue;
      if (m < MAXFV) {
        vertexOf(t, v);
        fx[m] = v[0]; fy[m] = v[1]; fz[m] = v[2];
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
