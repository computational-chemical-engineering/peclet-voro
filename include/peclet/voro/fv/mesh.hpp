/**
 * @file fv/mesh.hpp
 * \brief Track C, rung C1 (Voronoi methods plan): the FACE mesh of a published tessellation — one
 * record per geometric face (an interior face appears in two cells' facet lists; the mesh keeps it
 * once, owned by the lower cell), with the geometry the finite-volume operators need, and a
 * cell → faces CSR with the orientation sign. Built on device from TessellationView + AuxMaps.
 *
 * Voronoi orthogonality is the whole point of this track: the connector between the two seeds is
 * perpendicular to the face, so the two-point flux (p_B − p_A)/d is consistent with no
 * non-orthogonal correction, and the pressure Laplacian is the graph Laplacian A_f/d_f that the
 * OT optimiser already assembles. The only mesh-quality error left is SKEWNESS — the face
 * centroid off the connector — which the centroidal energy term of track B minimises.
 *
 * Face geometry from the view (Voronoi planes; a power tessellation's connector is the same
 * vector but the seed distance is not |conn| — pass positions when that matters):
 *   area vector A (owner side, outward), connector r = 2n (owner seed → plane foot, doubled = to
 *   the neighbour seed), dV/dr = (|A|/|r|)(r − c) ⇒ face centroid c = r − dV·|r|/|A|
 * (owner-relative).
 */
#ifndef PECLET_VORO_FV_MESH_HPP
#define PECLET_VORO_FV_MESH_HPP

#include <Kokkos_Core.hpp>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/tessellation_view.hpp"
#include "peclet/voro/transpose.hpp"

namespace peclet::voro::fv {

template <class Real>
struct FaceMesh {
  using Mem = peclet::core::MemSpace;
  int nCells = 0, nFaces = 0, nInterior = 0;
  // faces [0, nInterior) are interior (two cells), [nInterior, nFaces) boundary (wall / box).
  Kokkos::View<int*, Mem> faceCellA, faceCellB;  // nFaces; B = -1 on a boundary face
  Kokkos::View<int*, Mem> faceFacet;             // nFaces: owner (A-side) facet index in the view
  Kokkos::View<Real*, Mem> faceArea;             // nFaces: |A|
  Kokkos::View<Real*, Mem> faceNormal;           // 3*nFaces: unit, A → B (outward from A)
  Kokkos::View<Real*, Mem> faceDist;        // nFaces: seed distance d (interior); h_A (boundary)
  Kokkos::View<Real*, Mem> faceHa, faceHb;  // nFaces: seed-to-plane distances (h_B = 0 boundary)
  Kokkos::View<Real*, Mem> faceCentroid;    // 3*nFaces: centroid, A-seed-relative
  Kokkos::View<Real*, Mem> faceConn;        // 3*nFaces: connector A → B (min-image), 0 boundary
  Kokkos::View<int*, Mem> cellFaceOffset;   // nCells+1
  Kokkos::View<int*, Mem> cellFace;         // 2*nInterior + nBoundary
  Kokkos::View<Real*, Mem> cellFaceSign;    // same: +1 if the cell is A, -1 if B
  Kokkos::View<Real*, Mem> cellVolume;      // nCells (alias of the view's)

  KOKKOS_INLINE_FUNCTION int cellFacesBegin(int i) const { return cellFaceOffset(i); }
  KOKKOS_INLINE_FUNCTION int cellFacesEnd(int i) const { return cellFaceOffset(i + 1); }
  KOKKOS_INLINE_FUNCTION Real normal(int f, int c) const { return faceNormal(3 * f + c); }
  KOKKOS_INLINE_FUNCTION Real centroid(int f, int c) const { return faceCentroid(3 * f + c); }
  KOKKOS_INLINE_FUNCTION Real conn(int f, int c) const { return faceConn(3 * f + c); }
};

/// Build the face mesh. `aux` = buildAuxMaps(view). Interior faces are owned by the facet with the
/// lower index of the reciprocal pair; wall facets (facetNbr == kBoundaryFacet) and box facets
/// (−1) become boundary faces.
template <class Real>
FaceMesh<Real> buildFaceMesh(const TessellationView<Real>& view, const AuxMaps<Real>& aux) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  FaceMesh<Real> m;
  const int N = view.numCells(), nF = view.numFacets();
  m.nCells = N;
  auto recip = aux.recip;
  auto cellOf = aux.cellOfFacet;
  // classify facets: 1 = interior owner, 2 = boundary, 0 = twin (not owned)
  Kokkos::View<int*, Mem> kind("fv.kind", nF);
  Kokkos::parallel_for(
      "fv.kind", Kokkos::RangePolicy<Exec>(0, nF), KOKKOS_LAMBDA(const int f) {
        const int j = view.facetNbr(f);
        if (j < 0)
          kind(f) = 2;
        else if (recip(f) >= 0)
          kind(f) = (f < recip(f)) ? 1 : 0;
        else
          kind(f) = 2;  // no reciprocal (should not happen in a dense periodic build)
      });
  // enumerate: interior first, then boundary (two scans)
  Kokkos::View<int*, Mem> idxI(view_alloc(std::string("fv.idxI"), WithoutInitializing), nF);
  Kokkos::View<int*, Mem> idxB(view_alloc(std::string("fv.idxB"), WithoutInitializing), nF);
  int nI = 0, nB = 0;
  Kokkos::parallel_scan(
      "fv.scanI", nF,
      KOKKOS_LAMBDA(const int f, int& upd, const bool fin) {
        if (fin)
          idxI(f) = upd;
        upd += (kind(f) == 1) ? 1 : 0;
      },
      nI);
  Kokkos::parallel_scan(
      "fv.scanB", nF,
      KOKKOS_LAMBDA(const int f, int& upd, const bool fin) {
        if (fin)
          idxB(f) = upd;
        upd += (kind(f) == 2) ? 1 : 0;
      },
      nB);
  m.nInterior = nI;
  m.nFaces = nI + nB;
  const int nFaces = m.nFaces;
  m.faceCellA =
      Kokkos::View<int*, Mem>(view_alloc(std::string("fv.A"), WithoutInitializing), nFaces);
  m.faceCellB =
      Kokkos::View<int*, Mem>(view_alloc(std::string("fv.B"), WithoutInitializing), nFaces);
  m.faceFacet =
      Kokkos::View<int*, Mem>(view_alloc(std::string("fv.facet"), WithoutInitializing), nFaces);
  m.faceArea =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("fv.area"), WithoutInitializing), nFaces);
  m.faceNormal = Kokkos::View<Real*, Mem>(view_alloc(std::string("fv.normal"), WithoutInitializing),
                                          3 * (size_t)nFaces);
  m.faceDist =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("fv.dist"), WithoutInitializing), nFaces);
  m.faceHa =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("fv.ha"), WithoutInitializing), nFaces);
  m.faceHb =
      Kokkos::View<Real*, Mem>(view_alloc(std::string("fv.hb"), WithoutInitializing), nFaces);
  m.faceCentroid = Kokkos::View<Real*, Mem>(view_alloc(std::string("fv.cen"), WithoutInitializing),
                                            3 * (size_t)nFaces);
  m.faceConn = Kokkos::View<Real*, Mem>(view_alloc(std::string("fv.conn"), WithoutInitializing),
                                        3 * (size_t)nFaces);
  {
    auto A = m.faceCellA;
    auto B = m.faceCellB;
    auto FF = m.faceFacet;
    auto AR = m.faceArea;
    auto NM = m.faceNormal;
    auto DS = m.faceDist;
    auto HA = m.faceHa;
    auto HB = m.faceHb;
    auto CE = m.faceCentroid;
    auto CN = m.faceConn;
    Kokkos::parallel_for(
        "fv.fill", Kokkos::RangePolicy<Exec>(0, nF), KOKKOS_LAMBDA(const int f) {
          const int k = kind(f);
          if (k == 0)
            return;
          const int g = (k == 1) ? idxI(f) : nI + idxB(f);
          const int i = cellOf(f);
          A(g) = i;
          B(g) = (k == 1) ? cellOf(recip(f)) : -1;
          FF(g) = f;
          const Real ax = view.area(f, 0), ay = view.area(f, 1), az = view.area(f, 2);
          const Real ar = Kokkos::sqrt(ax * ax + ay * ay + az * az);
          AR(g) = ar;
          const Real iar = ar > Real(0) ? Real(1) / ar : Real(0);
          NM(3 * g) = ax * iar;
          NM(3 * g + 1) = ay * iar;
          NM(3 * g + 2) = az * iar;
          const Real rx = view.connVec(f, 0), ry = view.connVec(f, 1), rz = view.connVec(f, 2);
          const Real rl = Kokkos::sqrt(rx * rx + ry * ry + rz * rz);
          // foot distance of the owner seed to the plane: |n| = |conn|/2 (conn = 2n)
          const Real ha = Real(0.5) * rl;
          // centroid: dV/dr = (|A|/|r|)(r − c)  ⇒  c = r − dV·|r|/|A|
          const Real dvx = view.connect(f, 0), dvy = view.connect(f, 1), dvz = view.connect(f, 2);
          const Real s = (ar > Real(0)) ? rl / ar : Real(0);
          CE(3 * g) = rx - dvx * s;
          CE(3 * g + 1) = ry - dvy * s;
          CE(3 * g + 2) = rz - dvz * s;
          if (k == 1) {
            CN(3 * g) = rx;
            CN(3 * g + 1) = ry;
            CN(3 * g + 2) = rz;
            DS(g) = rl;  // Voronoi: seed distance == |conn|
            HA(g) = ha;
            HB(g) = rl - ha;
          } else {
            CN(3 * g) = CN(3 * g + 1) = CN(3 * g + 2) = Real(0);
            DS(g) = ha;
            HA(g) = ha;
            HB(g) = Real(0);
          }
        });
  }
  // cell -> faces CSR (each interior face in two cells, boundary in one)
  Kokkos::View<int*, Mem> cnt("fv.cnt", N + 1);
  {
    auto A = m.faceCellA;
    auto B = m.faceCellB;
    Kokkos::parallel_for(
        "fv.count", Kokkos::RangePolicy<Exec>(0, nFaces), KOKKOS_LAMBDA(const int g) {
          Kokkos::atomic_inc(&cnt(A(g)));
          if (B(g) >= 0)
            Kokkos::atomic_inc(&cnt(B(g)));
        });
  }
  m.cellFaceOffset = Kokkos::View<int*, Mem>("fv.cellFaceOffset", N + 1);
  {
    auto off = m.cellFaceOffset;
    Kokkos::parallel_scan(
        "fv.scanCells", N + 1, KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
          if (fin)
            off(i) = upd;
          upd += (i < N) ? cnt(i) : 0;
        });
  }
  const int nEntries = 2 * nI + nB;
  m.cellFace = Kokkos::View<int*, Mem>(view_alloc(std::string("fv.cellFace"), WithoutInitializing),
                                       nEntries);
  m.cellFaceSign = Kokkos::View<Real*, Mem>(
      view_alloc(std::string("fv.cellFaceSign"), WithoutInitializing), nEntries);
  {
    Kokkos::View<int*, Mem> cursor("fv.cursor", N);
    auto off = m.cellFaceOffset;
    auto CF = m.cellFace;
    auto CS = m.cellFaceSign;
    auto A = m.faceCellA;
    auto B = m.faceCellB;
    Kokkos::parallel_for(
        "fv.fillCells", Kokkos::RangePolicy<Exec>(0, nFaces), KOKKOS_LAMBDA(const int g) {
          const int a = A(g);
          const int pa = off(a) + Kokkos::atomic_fetch_add(&cursor(a), 1);
          CF(pa) = g;
          CS(pa) = Real(1);
          const int b = B(g);
          if (b >= 0) {
            const int pb = off(b) + Kokkos::atomic_fetch_add(&cursor(b), 1);
            CF(pb) = g;
            CS(pb) = Real(-1);
          }
        });
  }
  m.cellVolume = view.cellVolume;
  Kokkos::fence();
  return m;
}

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_MESH_HPP
