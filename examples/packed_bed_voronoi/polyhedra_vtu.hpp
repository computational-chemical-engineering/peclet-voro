/**
 * @file polyhedra_vtu.hpp
 * \brief The VTU writer the two packed-bed examples share: clipped ConvexCells appended as
 * watertight VTK_POLYHEDRON cells (faces referenced by triangle index, ConvexCell::faceOrderedIdx),
 * written as an ASCII XML UnstructuredGrid with per-cell scalars and a wall-touch flag.
 */
#pragma once

#include <cstddef>
#include <fstream>
#include <utility>
#include <vector>

#include "peclet/voro/convex_cell.hpp"

template <class Real>
struct PolyhedraVtu {
  std::vector<Real> px, py, pz;       // points (world positions of the cell vertices)
  std::vector<long> conn, offs;       // cell -> its point ids
  std::vector<long> faces, faceoffs;  // polyhedron faces (VTK format)
  std::vector<int> boundary;          // 1 if the cell touches a wall facet
  long connCursor = 0;

  long numCells() const { return (long)offs.size(); }
  long numPoints() const { return (long)px.size(); }

  /// Append cell c (seed-relative vertices, seed at s) as one polyhedron. Returns false, with no
  /// cell added, if it has fewer than 4 vertices or 4 faces. The vertices are appended to the
  /// point list BEFORE those checks, so a rejected cell leaves its points behind (unreferenced) —
  /// the examples' output has always carried them.
  template <class Cell>
  bool append(const Cell& c, Real sx, Real sy, Real sz) {
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
    if (np < 4) return false;  // degenerate

    // faces: for each plane, the ordered vertex loop (by triangle index -> watertight).
    std::vector<std::vector<long>> cellFaces;
    bool touchesWall = false;
    for (int k = 0; k < c.np; ++k) {
      int fidx[Cell::MAXFV];
      const int m = c.faceOrderedIdx(k, fidx);
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
    if (cellFaces.size() < 4) return false;

    for (int p = 0; p < np; ++p) conn.push_back(base + p);
    connCursor += np;
    offs.push_back(connCursor);
    faces.push_back((long)cellFaces.size());
    for (auto& f : cellFaces) {
      faces.push_back((long)f.size());
      for (long id : f) faces.push_back(id);
    }
    faceoffs.push_back((long)faces.size());
    boundary.push_back(touchesWall ? 1 : 0);
    return true;
  }

  /// Write the XML UnstructuredGrid (VTK_POLYHEDRON = 42): the Float64 cell scalars in the order
  /// given (the first one is the active scalar, named "volume" by both examples), then `boundary`.
  void write(const char* fname,
             const std::vector<std::pair<const char*, const std::vector<Real>*>>& scalars) const {
    std::ofstream out(fname);
    const long nPts = numPoints(), nCells = numCells();
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
    out << "      <CellData Scalars=\"" << scalars.front().first << "\">\n";
    for (std::size_t s = 0; s < scalars.size(); ++s) {
      out << (s == 0 ? "" : "\n        </DataArray>\n") << "        <DataArray type=\"Float64\" Name=\""
          << scalars[s].first << "\" format=\"ascii\">\n";
      for (Real v : *scalars[s].second) out << v << ' ';
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"boundary\" format=\"ascii\">\n";
    for (int v : boundary) out << v << ' ';
    out << "\n        </DataArray>\n      </CellData>\n";
    out << "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
  }
};
