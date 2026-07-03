/**
 * @file sdf_voronoi_vtu.cpp
 * @brief Interstitial Voronoi tessellation of a sphere packing, with the packing's SDF as a wall,
 *        written as a ParaView unstructured grid (VTU, one VTK_POLYHEDRON per cell).
 *
 * Reads `packing.txt` (produced by pack_and_seed.py): a sphere packing + a set of seed points that
 * live in the interstitial space (outside every sphere). For each seed it builds the Voronoi cell
 * (bisectors against the other seeds, closest-first with the security-radius early-out), clips it to
 * the domain box, and clips it against the union-of-balls SDF so the sphere surfaces act as curved
 * walls (a few tangent-plane facets per sphere it touches). The clipped cells are emitted as solid
 * polyhedra to `cells.vtu`, coloured by cell volume + a boundary flag.
 *
 * Usage:  sdf_voronoi_vtu <packing.txt> <cells.vtu>
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/sdf.hpp"

using Real = double;
using Cell = peclet::voro::ConvexCell<Real, 128, 256>;

namespace {

// Union-of-balls SDF: sdf(x) = min_i(|x - c_i| - r_i)  (< 0 inside any ball, > 0 in the fluid,
// gradient points outward into the fluid — the suite convention). Host provider (eval + gradH).
struct SdfSpheres {
  const Real* c;  // 3*n centres
  const Real* r;  // n radii
  int n;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real m = Real(1e30);
    for (int i = 0; i < n; ++i) {
      const Real dx = x - c[3 * i], dy = y - c[3 * i + 1], dz = z - c[3 * i + 2];
      const Real d = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) - r[i];
      if (d < m) m = d;
    }
    return m;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

// Ordered alive-triangle INDICES forming face k (a replica of ConvexCell::faceOrdered that returns
// triangle indices, not positions). Each dual triangle is a unique cell vertex and each triangle
// carries exactly its 3 defining planes, so referencing faces by triangle index makes the emitted
// polyhedron watertight (every edge is shared by exactly two faces) — position matching is not.
int faceOrderedIdx(const Cell& c, int k, int out[Cell::MAXFV]) {
  int m = 0;
  Real fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
  for (int t = 0; t < c.nt; ++t) {
    if (!c.alive[t]) continue;
    if (c.t0[t] != k && c.t1[t] != k && c.t2[t] != k) continue;
    if (m < Cell::MAXFV) {
      out[m] = t;
      fx[m] = c.vx[t];
      fy[m] = c.vy[t];
      fz[m] = c.vz[t];
      ++m;
    }
  }
  if (m < 3) return m;
  const Real nx = c.n[k][0], ny = c.n[k][1], nz = c.n[k][2];
  const Real nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (nlen == Real(0)) return 0;
  const Real un[3] = {nx / nlen, ny / nlen, nz / nlen};
  Real e1[3];
  if (std::fabs(un[0]) <= std::fabs(un[1]) && std::fabs(un[0]) <= std::fabs(un[2])) {
    e1[0] = 0; e1[1] = -un[2]; e1[2] = un[1];
  } else if (std::fabs(un[1]) <= std::fabs(un[2])) {
    e1[0] = -un[2]; e1[1] = 0; e1[2] = un[0];
  } else {
    e1[0] = -un[1]; e1[1] = un[0]; e1[2] = 0;
  }
  const Real e1l = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  e1[0] /= e1l; e1[1] /= e1l; e1[2] /= e1l;
  const Real e2[3] = {un[1] * e1[2] - un[2] * e1[1], un[2] * e1[0] - un[0] * e1[2],
                      un[0] * e1[1] - un[1] * e1[0]};
  Real cx = 0, cy = 0, cz = 0;
  for (int i = 0; i < m; ++i) { cx += fx[i]; cy += fy[i]; cz += fz[i]; }
  cx /= m; cy /= m; cz /= m;
  Real ang[Cell::MAXFV];
  for (int i = 0; i < m; ++i) {
    const Real dx = fx[i] - cx, dy = fy[i] - cy, dz = fz[i] - cz;
    const Real px = dx * e1[0] + dy * e1[1] + dz * e1[2];
    const Real py = dx * e2[0] + dy * e2[1] + dz * e2[2];
    const Real s = std::fabs(px) + std::fabs(py);
    const Real tt = (s > Real(0)) ? py / s : Real(0);
    ang[i] = (px < Real(0)) ? (Real(2) - tt) : (py < Real(0) ? Real(4) + tt : tt);
  }
  for (int i = 1; i < m; ++i) {  // insertion sort by angle, carrying indices
    const Real ka = ang[i];
    const int ki = out[i];
    int j = i - 1;
    while (j >= 0 && ang[j] > ka) {
      ang[j + 1] = ang[j];
      out[j + 1] = out[j];
      --j;
    }
    ang[j + 1] = ka;
    out[j + 1] = ki;
  }
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <packing.txt> <cells.vtu>\n", argv[0]);
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  int nSph = 0, nSeed = 0;
  Real lo = 0, hi = 1;
  in >> nSph >> nSeed >> lo >> hi;
  std::vector<Real> sc(3 * nSph), sr(nSph), seed(3 * nSeed);
  for (int i = 0; i < nSph; ++i) in >> sc[3 * i] >> sc[3 * i + 1] >> sc[3 * i + 2] >> sr[i];
  for (int i = 0; i < nSeed; ++i) in >> seed[3 * i] >> seed[3 * i + 1] >> seed[3 * i + 2];

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    SdfSpheres sdf{sc.data(), sr.data(), nSph};

    // VTU accumulators (per-cell points appended; connectivity/faces use global point ids).
    std::vector<Real> px, py, pz;             // points
    std::vector<long> conn, offs;             // cell -> its point ids
    std::vector<long> faces, faceoffs;        // polyhedron faces (VTK format)
    std::vector<Real> cellVol;                // scalar for colouring
    std::vector<int> cellBoundary;            // 1 if the cell touches a sphere wall

    std::vector<std::pair<Real, int>> ord;    // neighbours by distance²
    ord.reserve(nSeed);
    long connCursor = 0, faceCursor = 0;
    int nBuilt = 0;

    for (int i = 0; i < nSeed; ++i) {
      const Real sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
      // gather + sort the other seeds by distance (closest-first for the security early-out)
      ord.clear();
      for (int j = 0; j < nSeed; ++j) {
        if (j == i) continue;
        const Real dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
        ord.emplace_back(dx * dx + dy * dy + dz * dz, j);
      }
      std::sort(ord.begin(), ord.end());
      std::vector<Real> rx(ord.size()), ry(ord.size()), rz(ord.size());
      std::vector<int> ids(ord.size());
      for (size_t k = 0; k < ord.size(); ++k) {
        const int j = ord[k].second;
        rx[k] = seed[3 * j] - sx;
        ry[k] = seed[3 * j + 1] - sy;
        rz[k] = seed[3 * j + 2] - sz;
        ids[k] = j;
      }
      Cell c;
      const Real Lbig[3] = {2 * (hi - lo), 2 * (hi - lo), 2 * (hi - lo)};  // box far away; walls added
      peclet::voro::buildConvexCell(c, Lbig, rx.data(), ry.data(), rz.data(), ids.data(),
                                    (int)ord.size());
      // clip to the domain AABB (seed-relative offsets), marked as box planes (-1)
      const Real axes[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      const Real off6[6] = {hi - sx, sx - lo, hi - sy, sy - lo, hi - sz, sz - lo};
      for (int a = 0; a < 6; ++a) c.clip(axes[a], off6[a], -1);
      const Real seedW[3] = {sx, sy, sz};
      peclet::voro::clipCellAgainstSdf<Real, 128, 256, false>(c, seedW, sdf);
      if (c.empty() || c.overflow) continue;

      // map alive dual triangles (= cell vertices) to local point ids; append world positions.
      const long base = (long)px.size();
      std::vector<int> triToPt(c.nt, -1);
      int np = 0;
      for (int t = 0; t < c.nt; ++t) {
        if (!c.alive[t]) continue;
        triToPt[t] = np++;
        px.push_back(sx + c.vx[t]);
        py.push_back(sy + c.vy[t]);
        pz.push_back(sz + c.vz[t]);
      }
      if (np < 4) continue;  // degenerate

      // faces: for each plane, the ordered vertex loop (by triangle index -> watertight).
      std::vector<std::vector<long>> cellFaces;
      bool touchesWall = false;
      for (int k = 0; k < c.np; ++k) {
        int fidx[Cell::MAXFV];
        const int m = faceOrderedIdx(c, k, fidx);
        if (m < 3) continue;
        std::vector<long> face;
        face.reserve(m);
        for (int q = 0; q < m; ++q)
          if (triToPt[fidx[q]] >= 0) face.push_back(base + triToPt[fidx[q]]);
        if ((int)face.size() >= 3) {
          cellFaces.push_back(std::move(face));
          if (c.pnbr[k] == peclet::voro::kBoundaryFacet) touchesWall = true;
        }
      }
      if (cellFaces.size() < 4) continue;

      // append this polyhedron to the VTU arrays
      for (int p = 0; p < np; ++p) conn.push_back(base + p);
      connCursor += np;
      offs.push_back(connCursor);
      faces.push_back((long)cellFaces.size());
      for (auto& f : cellFaces) {
        faces.push_back((long)f.size());
        for (long id : f) faces.push_back(id);
      }
      faceCursor = (long)faces.size();
      faceoffs.push_back(faceCursor);
      cellVol.push_back(c.volumePerVertex());
      cellBoundary.push_back(touchesWall ? 1 : 0);
      ++nBuilt;
    }

    // ---- write the VTU (XML UnstructuredGrid, VTK_POLYHEDRON = 42) ----
    std::ofstream out(argv[2]);
    const long nPts = (long)px.size(), nCells = (long)offs.size();
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
           "byte_order=\"LittleEndian\">\n  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << nPts << "\" NumberOfCells=\"" << nCells << "\">\n";
    out << "      <Points>\n        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
           "format=\"ascii\">\n";
    for (long p = 0; p < nPts; ++p) out << px[p] << ' ' << py[p] << ' ' << pz[p] << '\n';
    out << "        </DataArray>\n      </Points>\n      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (long v : conn) out << v << ' ';
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (long v : offs) out << v << ' ';
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (long ci = 0; ci < nCells; ++ci) out << "42 ";
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int64\" Name=\"faces\" format=\"ascii\">\n";
    for (long v : faces) out << v << ' ';
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"ascii\">\n";
    for (long v : faceoffs) out << v << ' ';
    out << "\n        </DataArray>\n      </Cells>\n";
    out << "      <CellData Scalars=\"volume\">\n";
    out << "        <DataArray type=\"Float64\" Name=\"volume\" format=\"ascii\">\n";
    for (Real v : cellVol) out << v << ' ';
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"boundary\" format=\"ascii\">\n";
    for (int v : cellBoundary) out << v << ' ';
    out << "\n        </DataArray>\n      </CellData>\n";
    out << "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
    out.close();
    std::printf("wrote %s: %ld cells, %ld points (%d/%d seeds tessellated)\n", argv[2], nCells, nPts,
                nBuilt, nSeed);
  }
  Kokkos::finalize();
  return rc;
}
