/**
 * @file fv/polymesh.hpp
 * \brief Track B, rung B3: the internal polyhedral mesh (`PolyMesh`) of a resident tessellation —
 * shared vertices, ordered face polygons, owner/neighbour, wall patches — assembled on the host
 * from the topology store, plus a VTU (VTK_POLYHEDRON) writer for viewing. This is what track C
 * consumes beyond the face mesh (vertices are needed for output and for the cell-vertex
 * quantities of later rungs); the solver itself runs on fv::FaceMesh.
 *
 * Assembly: every cell is reloaded from the store and re-evaluated at the current positions
 * (ConvexCell::reevalGeometry — the same geometry the solver saw), each face polygon is walked
 * CCW (faceOrdered), and vertices are shared by quantised world position (the four cells around a
 * Voronoi vertex compute it from the same three planes to round-off, so a 1e-9·L bucket merges
 * them; a genuinely degenerate configuration with two vertices closer than that would be merged
 * too — recorded, not handled). Interior faces are emitted once (from the lower cell) with the
 * outward orientation of the owner; wall faces (kBoundaryFacet) go to patch 1, box faces to
 * patch 2 (a periodic box has none).
 */
#ifndef PECLET_VORO_FV_POLYMESH_HPP
#define PECLET_VORO_FV_POLYMESH_HPP

#include <cmath>
#include <cstdio>
#include <functional>
#include <Kokkos_Core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/topology_store.hpp"

namespace peclet::voro::fv {

template <class Real>
struct PolyMesh {
  int nCells = 0, nFaces = 0, nInterior = 0;
  std::vector<Real> points;         // 3 * nPoints (world)
  std::vector<int> faceOffset;      // nFaces + 1
  std::vector<int> faceVerts;       // point ids, CCW about the owner's outward normal
  std::vector<int> owner;           // nFaces
  std::vector<int> neighbour;       // nFaces, -1 on boundary
  std::vector<int> patch;           // nFaces: 0 interior, 1 wall, 2 box
  std::vector<int> cellFaceOffset;  // nCells + 1
  std::vector<int> cellFace;        // faces of each cell (interior faces in both)
  std::vector<Real> cellVolume;     // nCells, from the assembled polygons
  int nPoints() const { return (int)points.size() / 3; }
};

namespace detail {
struct VertexKey {
  long long a, b, c;
  bool operator==(const VertexKey& o) const { return a == o.a && b == o.b && c == o.c; }
};
struct VertexKeyHash {
  std::size_t operator()(const VertexKey& k) const {
    std::size_t h = (std::size_t)k.a * 73856093u;
    h ^= (std::size_t)k.b * 19349663u;
    h ^= (std::size_t)k.c * 83492791u;
    return h;
  }
};
}  // namespace detail

/// Conforming wall faces (rung B3 follow-up): every wall cell clips its wall-adjacent faces with
/// ITS OWN tangent plane of a curved wall, so the two copies of a shared edge's wall endpoint (one
/// per cell) differ by the tangent-plane mismatch (~the sagitta, O(h²/R)). This post-pass merges
/// wall vertices closer than `tol` (union–find on a hash grid, periodic min-image distances) into
/// their mean, rewrites the face polygons, drops repeated vertices and degenerate (< 3-vertex)
/// faces. MEASURED (2026-09-04, test_polymesh): it does NOT make the wall layer conforming — the
/// mismatch is topological (the two cells' copies of a shared interface face are clipped by
/// different tangent planes into polygons of different shape), 106 of 108 non-conforming wall
/// cells keep failing Euler. Conforming walls need one wall plane per wall EDGE inside the
/// clipper. Kept as a coincident-vertex cleanup. Returns the number of vertices merged away.
template <class Real>
int mergeWallVertices(PolyMesh<Real>& m, const Real L[3], Real tol) {
  const int nP = m.nPoints();
  std::vector<char> onWall(nP, 0);
  for (int f = 0; f < m.nFaces; ++f)
    if (m.patch[f] == 1)
      for (int q = m.faceOffset[f]; q < m.faceOffset[f + 1]; ++q)
        onWall[m.faceVerts[q]] = 1;
  std::vector<int> ids;
  for (int p = 0; p < nP; ++p)
    if (onWall[p])
      ids.push_back(p);
  // hash grid of cell size tol over the periodic box
  std::unordered_map<detail::VertexKey, std::vector<int>, detail::VertexKeyHash> grid;
  long long Mq[3];
  for (int k = 0; k < 3; ++k)
    Mq[k] = std::max<long long>(1, (long long)std::floor(L[k] / tol));
  auto cellOf = [&](Real x, int k) {
    long long c = (long long)std::floor(x / L[k] * (Real)Mq[k]);
    c %= Mq[k];
    return c < 0 ? c + Mq[k] : c;
  };
  for (int p : ids)
    grid[detail::VertexKey{cellOf(m.points[3 * p], 0), cellOf(m.points[3 * p + 1], 1),
                           cellOf(m.points[3 * p + 2], 2)}]
        .push_back(p);
  std::vector<int> parent(nP);
  for (int p = 0; p < nP; ++p)
    parent[p] = p;
  std::function<int(int)> find = [&](int a) {
    while (parent[a] != a)
      a = parent[a] = parent[parent[a]];
    return a;
  };
  const Real tol2 = tol * tol;
  for (int p : ids) {
    const long long c0[3] = {cellOf(m.points[3 * p], 0), cellOf(m.points[3 * p + 1], 1),
                             cellOf(m.points[3 * p + 2], 2)};
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          auto w = [&](long long c, long long M) { return ((c % M) + M) % M; };
          auto it = grid.find(
              detail::VertexKey{w(c0[0] + dx, Mq[0]), w(c0[1] + dy, Mq[1]), w(c0[2] + dz, Mq[2])});
          if (it == grid.end())
            continue;
          for (int q : it->second) {
            if (q <= p)
              continue;
            Real d2 = 0;
            for (int k = 0; k < 3; ++k) {
              Real d = m.points[3 * q + k] - m.points[3 * p + k];
              d -= L[k] * std::round(d / L[k]);
              d2 += d * d;
            }
            if (d2 <= tol2) {
              const int ra = find(p), rb = find(q);
              if (ra != rb)
                parent[rb] = ra;
            }
          }
        }
  }
  // merged position = the mean of the group (min-image about the root)
  std::vector<Real> acc(3 * (size_t)nP, Real(0));
  std::vector<int> cnt(nP, 0);
  int merged = 0;
  for (int p : ids) {
    const int r = find(p);
    for (int k = 0; k < 3; ++k) {
      Real d = m.points[3 * p + k] - m.points[3 * r + k];
      d -= L[k] * std::round(d / L[k]);
      acc[3 * r + k] += d;
    }
    ++cnt[r];
    if (r != p)
      ++merged;
  }
  for (int p : ids)
    if (find(p) == p && cnt[p] > 1)
      for (int k = 0; k < 3; ++k)
        m.points[3 * p + k] += acc[3 * p + k] / (Real)cnt[p];
  // rewrite the faces
  std::vector<int> newVerts, newOff(1, 0), newOwner, newNbr, newPatch;
  std::vector<int> faceMap(m.nFaces, -1);
  for (int f = 0; f < m.nFaces; ++f) {
    std::vector<int> poly;
    for (int q = m.faceOffset[f]; q < m.faceOffset[f + 1]; ++q) {
      const int v = find(m.faceVerts[q]);
      if (poly.empty() || poly.back() != v)
        poly.push_back(v);
    }
    while (poly.size() > 1 && poly.front() == poly.back())
      poly.pop_back();
    if (poly.size() < 3)
      continue;  // a degenerate face (collapsed by the merge)
    faceMap[f] = (int)newOff.size() - 1;
    newVerts.insert(newVerts.end(), poly.begin(), poly.end());
    newOff.push_back((int)newVerts.size());
    newOwner.push_back(m.owner[f]);
    newNbr.push_back(m.neighbour[f]);
    newPatch.push_back(m.patch[f]);
  }
  std::vector<int> cf, cfo(1, 0);
  for (int i = 0; i < m.nCells; ++i) {
    for (int q = m.cellFaceOffset[i]; q < m.cellFaceOffset[i + 1]; ++q)
      if (faceMap[m.cellFace[q]] >= 0)
        cf.push_back(faceMap[m.cellFace[q]]);
    cfo.push_back((int)cf.size());
  }
  int nInt = 0;
  for (int f = 0; f < (int)newPatch.size(); ++f)
    nInt += newPatch[f] == 0;
  m.faceVerts = newVerts;
  m.faceOffset = newOff;
  m.owner = newOwner;
  m.neighbour = newNbr;
  m.patch = newPatch;
  m.cellFace = cf;
  m.cellFaceOffset = cfo;
  m.nFaces = (int)newPatch.size();
  m.nInterior = nInt;
  return merged;
}

/// Assemble the polyhedral mesh of a store at positions `posHost` (3N, world, in [0,L)).
template <class Real, int MAXP, int MAXT>
PolyMesh<Real> buildPolyMesh(const TopologyStore<MAXP, MAXT>& store,
                             const Kokkos::View<Real*, peclet::core::MemSpace>& pos, int N,
                             const Real L[3], WallStore<Real> wall = {},
                             Kokkos::View<Real*, peclet::core::MemSpace> xRef = {},
                             Real quantum = Real(1e-9)) {
  using Cell = ConvexCell<Real, MAXP, MAXT, false>;
  auto np = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, store.np);
  auto nt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, store.nt);
  auto pnbr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, store.pnbr);
  auto tri = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, store.tri);
  auto ph = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pos);
  // host decode of the store (the same steps as TopologyStore::load, on the mirrors)
  auto loadHost = [&](int i, Cell& c) {
    c.initBoxPlanes(L[0], L[1], L[2]);
    const int npi = np(i), nti = nt(i);
    c.np = npi;
    c.nt = nti;
    c.overflow = false;
    for (int k = 6; k < npi; ++k)
      c.pnbr[k] = pnbr((size_t)i * MAXP + k);
    for (int t = 0; t < nti; ++t) {
      const unsigned w = tri((size_t)i * MAXT + t);
      c.t0[t] = (unsigned char)(w & 0xffu);
      c.t1[t] = (unsigned char)((w >> 8) & 0xffu);
      c.t2[t] = (unsigned char)((w >> 16) & 0xffu);
      c.alive[t] = ((w >> 24) & 1u) != 0u;
    }
  };
  const bool hasWall = wall.cnt.extent(0) > 0;
  typename Kokkos::View<Real*, peclet::core::MemSpace>::host_mirror_type wrecM, xrM;
  typename Kokkos::View<int*, peclet::core::MemSpace>::host_mirror_type wcntM;
  if (hasWall) {
    wrecM = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, wall.rec);
    wcntM = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, wall.cnt);
    xrM = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, xRef);
  }
  PolyMesh<Real> m;
  m.nCells = N;
  m.cellVolume.assign(N, Real(0));
  std::unordered_map<detail::VertexKey, int, detail::VertexKeyHash> vmap;
  vmap.reserve((size_t)N * 8);
  // keys are PERIODIC: the same vertex seen from a cell across the box boundary is offset by L
  const long long M[3] = {(long long)std::llround(Real(1) / quantum),
                          (long long)std::llround(Real(1) / quantum),
                          (long long)std::llround(Real(1) / quantum)};
  auto wrapq = [](long long v, long long m) {
    v %= m;
    return v < 0 ? v + m : v;
  };
  auto vertexId = [&](Real x, Real y, Real z) {
    detail::VertexKey k{wrapq((long long)std::llround(x / (quantum * L[0])), M[0]),
                        wrapq((long long)std::llround(y / (quantum * L[1])), M[1]),
                        wrapq((long long)std::llround(z / (quantum * L[2])), M[2])};
    auto it = vmap.find(k);
    if (it != vmap.end())
      return it->second;
    const int id = m.nPoints();
    m.points.push_back(x);
    m.points.push_back(y);
    m.points.push_back(z);
    vmap.emplace(k, id);
    return id;
  };
  // interior faces are keyed by (min cell, max cell) to be emitted once
  std::vector<std::vector<int>> cellFaces(N);
  std::vector<int> faceOwner, faceNbr, facePatch, faceOff{0}, faceV;
  const Real Lh[3] = {L[0] / 2, L[1] / 2, L[2] / 2};
  for (int i = 0; i < N; ++i) {
    Cell c;
    loadHost(i, c);
    const Real sx = ph(3 * i), sy = ph(3 * i + 1), sz = ph(3 * i + 2);
    if (hasWall) {
      Real dx = sx - xrM(3 * i), dy = sy - xrM(3 * i + 1), dz = sz - xrM(3 * i + 2);
      dx = dx > Lh[0] ? dx - L[0] : (dx < -Lh[0] ? dx + L[0] : dx);
      dy = dy > Lh[1] ? dy - L[1] : (dy < -Lh[1] ? dy + L[1] : dy);
      dz = dz > Lh[2] ? dz - L[2] : (dz < -Lh[2] ? dz + L[2] : dz);
      // restore the wall planes from the host mirrors (same arithmetic as WallStore::load)
      int r = 0;
      const int nrec = wcntM(i);
      for (int k = 6; k < c.np; ++k) {
        if (c.pnbr[k] >= 0 || r >= nrec)
          continue;
        const size_t o = ((size_t)i * WallStore<Real>::kMax + r) * 4;
        const Real ux = wrecM(o), uy = wrecM(o + 1), uz = wrecM(o + 2);
        const Real h = wrecM(o + 3) - (ux * dx + uy * dy + uz * dz);
        c.n[k][0] = h * ux;
        c.n[k][1] = h * uy;
        c.n[k][2] = h * uz;
        c.nn[k] = h * h;
        ++r;
      }
    }
    c.reevalGeometry(sx, sy, sz, ph.data(), L[0]);
    Real V = 0;
    for (int k = 0; k < c.np; ++k) {
      Real fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
      const int nv = c.faceOrdered(k, fx, fy, fz);
      if (nv < 3)
        continue;
      const int j = c.pnbr[k];
      // pyramid volume from the seed (exact for the convex cell)
      for (int q = 1; q + 1 < nv; ++q) {
        const Real ax = fx[0], ay = fy[0], az = fz[0], bx = fx[q], by = fy[q], bz = fz[q],
                   cx = fx[q + 1], cy = fy[q + 1], cz = fz[q + 1];
        V += std::fabs(ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) +
                       az * (bx * cy - by * cx)) /
             Real(6);
      }
      if (j >= 0 && j < i)
        continue;  // emitted from the lower cell
      const int fid = (int)faceOwner.size();
      faceOwner.push_back(i);
      faceNbr.push_back(j >= 0 ? j : -1);
      facePatch.push_back(j >= 0 ? 0 : (j == kBoundaryFacet ? 1 : 2));
      for (int q = 0; q < nv; ++q)
        faceV.push_back(vertexId(sx + fx[q], sy + fy[q], sz + fz[q]));
      faceOff.push_back((int)faceV.size());
      cellFaces[i].push_back(fid);
      if (j >= 0)
        cellFaces[j].push_back(fid);
    }
    m.cellVolume[i] = V;
  }
  // order: interior faces first
  const int nF = (int)faceOwner.size();
  std::vector<int> perm(nF), inv(nF);
  int q = 0;
  for (int f = 0; f < nF; ++f)
    if (facePatch[f] == 0)
      perm[q++] = f;
  m.nInterior = q;
  for (int f = 0; f < nF; ++f)
    if (facePatch[f] != 0)
      perm[q++] = f;
  for (int f = 0; f < nF; ++f)
    inv[perm[f]] = f;
  m.nFaces = nF;
  m.faceOffset.assign(1, 0);
  for (int f = 0; f < nF; ++f) {
    const int o = perm[f];
    m.owner.push_back(faceOwner[o]);
    m.neighbour.push_back(faceNbr[o]);
    m.patch.push_back(facePatch[o]);
    for (int v = faceOff[o]; v < faceOff[o + 1]; ++v)
      m.faceVerts.push_back(faceV[v]);
    m.faceOffset.push_back((int)m.faceVerts.size());
  }
  m.cellFaceOffset.assign(1, 0);
  for (int i = 0; i < N; ++i) {
    for (int f : cellFaces[i])
      m.cellFace.push_back(inv[f]);
    m.cellFaceOffset.push_back((int)m.cellFace.size());
  }
  return m;
}

/// Write the mesh as an ASCII VTU of VTK_POLYHEDRON cells (with per-cell volume and, if given, a
/// per-cell scalar), readable by ParaView / pyvista.
template <class Real>
bool writeVtu(const std::string& path, const PolyMesh<Real>& m,
              const std::vector<Real>* cellScalar = nullptr, const char* scalarName = "scalar") {
  std::FILE* fp = std::fopen(path.c_str(), "w");
  if (!fp)
    return false;
  const int nP = m.nPoints(), nC = m.nCells;
  std::fprintf(fp,
               "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
               "byte_order=\"LittleEndian\">\n<UnstructuredGrid>\n<Piece NumberOfPoints=\"%d\" "
               "NumberOfCells=\"%d\">\n",
               nP, nC);
  std::fprintf(
      fp, "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int p = 0; p < nP; ++p)
    std::fprintf(fp, "%.17g %.17g %.17g\n", (double)m.points[3 * p], (double)m.points[3 * p + 1],
                 (double)m.points[3 * p + 2]);
  std::fprintf(fp, "</DataArray>\n</Points>\n<Cells>\n");
  // connectivity: the union of a cell's face vertices (unique), offsets, types = 42
  std::vector<int> conn, connOff, faces, faceOff;
  std::vector<char> seen(nP, 0);
  for (int i = 0; i < nC; ++i) {
    std::vector<int> touched;
    for (int q = m.cellFaceOffset[i]; q < m.cellFaceOffset[i + 1]; ++q) {
      const int f = m.cellFace[q];
      for (int v = m.faceOffset[f]; v < m.faceOffset[f + 1]; ++v) {
        const int p = m.faceVerts[v];
        if (!seen[p]) {
          seen[p] = 1;
          touched.push_back(p);
          conn.push_back(p);
        }
      }
    }
    for (int p : touched)
      seen[p] = 0;
    connOff.push_back((int)conn.size());
    const int nf = m.cellFaceOffset[i + 1] - m.cellFaceOffset[i];
    faces.push_back(nf);
    for (int q = m.cellFaceOffset[i]; q < m.cellFaceOffset[i + 1]; ++q) {
      const int f = m.cellFace[q];
      const int nv = m.faceOffset[f + 1] - m.faceOffset[f];
      faces.push_back(nv);
      // orientation: outward for the owner; reversed for the neighbour
      if (m.owner[f] == i)
        for (int v = m.faceOffset[f]; v < m.faceOffset[f + 1]; ++v)
          faces.push_back(m.faceVerts[v]);
      else
        for (int v = m.faceOffset[f + 1] - 1; v >= m.faceOffset[f]; --v)
          faces.push_back(m.faceVerts[v]);
    }
    faceOff.push_back((int)faces.size());
  }
  std::fprintf(fp, "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
  for (int v : conn)
    std::fprintf(fp, "%d\n", v);
  std::fprintf(fp, "</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
  for (int o : connOff)
    std::fprintf(fp, "%d\n", o);
  std::fprintf(fp, "</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < nC; ++i)
    std::fprintf(fp, "42\n");
  std::fprintf(fp, "</DataArray>\n<DataArray type=\"Int64\" Name=\"faces\" format=\"ascii\">\n");
  for (int v : faces)
    std::fprintf(fp, "%d\n", v);
  std::fprintf(fp,
               "</DataArray>\n<DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"ascii\">\n");
  for (int o : faceOff)
    std::fprintf(fp, "%d\n", o);
  std::fprintf(fp, "</DataArray>\n</Cells>\n<CellData>\n");
  std::fprintf(fp, "<DataArray type=\"Float64\" Name=\"volume\" format=\"ascii\">\n");
  for (int i = 0; i < nC; ++i)
    std::fprintf(fp, "%.17g\n", (double)m.cellVolume[i]);
  std::fprintf(fp, "</DataArray>\n");
  if (cellScalar) {
    std::fprintf(fp, "<DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", scalarName);
    for (int i = 0; i < nC; ++i)
      std::fprintf(fp, "%.17g\n", (double)(*cellScalar)[i]);
    std::fprintf(fp, "</DataArray>\n");
  }
  std::fprintf(fp, "</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n");
  std::fclose(fp);
  return true;
}

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_POLYMESH_HPP
