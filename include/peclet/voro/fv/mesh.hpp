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
 * (owner-relative). The centroid formula holds for every plane policy (the published connector is
 * 2·foot and dV = (|A|/|r|)(r − c) for Voronoi and power cells alike); the seed distance d = |r|
 * and h_B = d − h_A hold for Voronoi cells only.
 *
 * Opt-in extensions (`FaceMeshOptions`, the moving-grid design note docs/ale_design.md §1.5 and
 * §11 A1/A2; all off reproduces the arrays of the plain builder bitwise):
 *  - `deterministicOrder`: no atomics. An owned cell emits a face to an owned neighbour only if it
 *    has the lower gid (interface and boundary faces always); faces are numbered interior and
 *    interface first, then boundary, each by (emitting cell slot, facet position in that cell);
 *    each cell's face list is sorted by ascending face index. On a cell-major view with gid = slot
 *    this is the plain builder's single-thread result.
 *  - `gidOrientation`: A = the lower gid, B = the higher (an interface face may have a ghost as A).
 *    Requires `generatorGeometry`, which supplies the re-referenced geometry.
 *  - `generatorGeometry`: conn = minimum image of x_B − x_A, d = |conn|,
 *    h_A = (d² + w_A − w_B)/(2d), h_B = d − h_A (signed); the normal points A → B and the
 *    centroid is relative to the seed of A whichever cell emitted the face. Diagnostics in
 *    `FaceMesh::generatorDiag`.
 *  - `cellCentroids`: `FaceMesh::cellCentroid` (seed-relative), from the pyramids with apex at the
 *    seed: V_i (c_i − x_i) = ¼ Σ_f h_f A_f (c_f − x_i), h_f = n_f·(c_f − x_i), n_f outward of i.
 */
#ifndef PECLET_VORO_FV_MESH_HPP
#define PECLET_VORO_FV_MESH_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/tessellation_view.hpp"
#include "peclet/voro/transpose.hpp"

namespace peclet::voro::fv {

template <class Real>
struct FaceMesh {
  using Mem = peclet::core::MemSpace;
  int nCells = 0, nFaces = 0, nInterior = 0;
  int nCombined = 0;  // owned + ghost cells (distributed; == nCells on one rank): cell fields are
                      // sized nCombined, cell loops run over the nCells owned cells
  int nDropped = 0;   // non-reciprocal fluid facets dropped (degenerate edge/vertex contacts)
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
  Kokkos::View<Real*, Mem> cellCentroid;    // 3*nCells, seed-relative (FaceMeshOptions only)
  /// Of the `generatorGeometry` option (zero otherwise), over interior and interface faces.
  struct GeneratorDiagnostics {
    Real maxNormalConnSine = 0;  ///< max |n_f × conn|/d over faces with A_f > 1e-10 h²,
                                 ///< h³ = the mean volume of the owned cells
    int numNegativeHeight = 0;   ///< faces with h_A < 0 or h_B < 0
  } generatorDiag;

  KOKKOS_INLINE_FUNCTION int cellFacesBegin(int i) const { return cellFaceOffset(i); }
  KOKKOS_INLINE_FUNCTION int cellFacesEnd(int i) const { return cellFaceOffset(i + 1); }
  KOKKOS_INLINE_FUNCTION Real normal(int f, int c) const { return faceNormal(3 * f + c); }
  KOKKOS_INLINE_FUNCTION Real centroid(int f, int c) const { return faceCentroid(3 * f + c); }
  KOKKOS_INLINE_FUNCTION Real conn(int f, int c) const { return faceConn(3 * f + c); }
};

/// Opt-in extensions of `buildFaceMesh` (see the file comment). Per-cell inputs are indexed by the
/// view's combined cell slot (owned + ghost).
template <class Real>
struct FaceMeshOptions {
  using Mem = peclet::core::MemSpace;
  bool deterministicOrder = false;  ///< no atomics; reproducible face and cell→face order
  bool gidOrientation = false;      ///< A = lower gid (requires generatorGeometry)
  bool generatorGeometry = false;   ///< conn, d, h_A, h_B from positions and weights
  bool cellCentroids = false;       ///< fill FaceMesh::cellCentroid
  Kokkos::View<const std::int64_t*, Mem> gid;  ///< N global ids; empty: the slot index
  Kokkos::View<const Real*, Mem> pos;          ///< 3N seed positions (generatorGeometry)
  Kokkos::View<const Real*, Mem> weight;       ///< N power weights; empty: zero
  Real L[3] = {0, 0, 0};  ///< periodic box for the minimum image (0 on an axis: no wrap)

  bool any() const {
    return deterministicOrder || gidOrientation || generatorGeometry || cellCentroids;
  }
};

/// Build the face mesh. `aux` = buildAuxMaps(view). Interior faces are owned by the facet with the
/// lower index of the reciprocal pair; wall facets (facetNbr == kBoundaryFacet) and box facets
/// (−1) become boundary faces. Distributed (rung C5): `nOwned` ≥ 0 marks cells [0, nOwned) as this
/// rank's owned cells (the tessellation was built for them only, ghosts [nOwned, N) are
/// candidates without facets); a facet toward a ghost is an INTERFACE face owned here (B = the
/// ghost's combined index; the other rank owns its own copy of the same face and computes the
/// same flux from the exchanged fields).
template <class Real>
FaceMesh<Real> buildFaceMesh(const TessellationView<Real>& view, const AuxMaps<Real>& aux,
                             int nOwned = -1) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  FaceMesh<Real> m;
  const int N = view.numCells(), nF = view.numFacets();
  const int nOwn = (nOwned >= 0 && nOwned < N) ? nOwned : N;
  m.nCells = nOwn;
  m.nCombined = N;
  auto recip = aux.recip;
  auto cellOf = aux.cellOfFacet;
  // classify facets: 1 = interior owner, 2 = boundary, 0 = twin (not owned)
  Kokkos::View<int*, Mem> kind("fv.kind", nF);
  Kokkos::parallel_for(
      "fv.kind", Kokkos::RangePolicy<Exec>(0, nF), KOKKOS_LAMBDA(const int f) {
        const int j = view.facetNbr(f);
        if (j < 0)
          kind(f) = 2;
        else if (j >= nOwn)
          kind(f) = 1;  // interface face toward a ghost cell: owned here
        else if (recip(f) >= 0)
          kind(f) = (f < recip(f)) ? 1 : 0;
        else
          kind(f) = 3;  // no reciprocal: a degenerate (zero-area) edge/vertex contact of a
                        // symmetric lattice, seen from one side only — not a face, dropped
      });
  {
    int nd = 0;
    Kokkos::parallel_reduce(
        "fv.dropped", Kokkos::RangePolicy<Exec>(0, nF),
        KOKKOS_LAMBDA(const int f, int& acc) { acc += kind(f) == 3 ? 1 : 0; }, nd);
    m.nDropped = nd;
  }
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
          if (k == 0 || k == 3)
            return;
          const int g = (k == 1) ? idxI(f) : nI + idxB(f);
          const int i = cellOf(f);
          A(g) = i;
          // interior: the reciprocal facet's cell; interface (no reciprocal): the ghost index
          B(g) = (k == 1) ? (recip(f) >= 0 ? cellOf(recip(f)) : view.facetNbr(f)) : -1;
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
          if (B(g) >= 0 && B(g) < nOwn)
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
          if (b >= 0 && b < nOwn) {
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

namespace detail {
/// The cell → face entries of owned cell i in the deterministic order: fn(g, sign) for every face
/// cell i emits (facet position order) and, after each facet, for every face its owned neighbour
/// emits toward that facet; sign = +1 if i is A. The caller sorts the entries by face index.
template <class Real>
struct CellFaceEntries {
  TessellationView<Real> view;
  Kokkos::View<int*, peclet::core::MemSpace> kind, faceOf, recip, faceCellA;
  int nOwn;
  template <class Fn>
  KOKKOS_INLINE_FUNCTION void operator()(const int i, Fn&& fn) const {
    for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
      const int k = kind(f);
      if (k == 1 || k == 2) {
        const int g = faceOf(f);
        fn(g, faceCellA(g) == i ? Real(1) : Real(-1));
      }
      const int j = view.facetNbr(f);
      if (j < 0 || j >= nOwn)
        continue;
      for (int h = view.facetBegin(j); h < view.facetEnd(j); ++h)
        if (kind(h) == 1 && recip(h) == f) {
          const int g = faceOf(h);
          // a self-pair (j == i, A == B == i) already has its A entry from the emitting facet
          fn(g, (faceCellA(g) == i && j != i) ? Real(1) : Real(-1));
        }
    }
  }
};
}  // namespace detail

/// Build the face mesh with the opt-in extensions of `opt` (file comment; design note
/// docs/ale_design.md §1.5 items 1–3, §11 A1/A2). All options off is the plain builder above.
template <class Real>
FaceMesh<Real> buildFaceMesh(const TessellationView<Real>& view, const AuxMaps<Real>& aux,
                             int nOwned, const FaceMeshOptions<Real>& opt) {
  if (!opt.any())
    return buildFaceMesh(view, aux, nOwned);
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  const int N = view.numCells(), nF = view.numFacets();
  const bool det = opt.deterministicOrder, gidOri = opt.gidOrientation, gen = opt.generatorGeometry;
  if (gidOri && !gen)
    throw std::invalid_argument(
        "buildFaceMesh: gidOrientation needs generatorGeometry (the view's geometry is relative "
        "to the emitting cell and cannot be re-referenced to a lower-gid neighbour without "
        "positions)");
  const bool hasGid = opt.gid.extent(0) > 0;
  if (hasGid && opt.gid.extent(0) < static_cast<std::size_t>(N))
    throw std::invalid_argument("buildFaceMesh: FaceMeshOptions::gid needs one id per cell");
  if (gen && opt.pos.extent(0) < 3 * static_cast<std::size_t>(N))
    throw std::invalid_argument("buildFaceMesh: generatorGeometry needs 3 positions per cell");
  const bool hasW = opt.weight.extent(0) > 0;
  if (hasW && opt.weight.extent(0) < static_cast<std::size_t>(N))
    throw std::invalid_argument("buildFaceMesh: FaceMeshOptions::weight needs one per cell");
  FaceMesh<Real> m;
  const int nOwn = (nOwned >= 0 && nOwned < N) ? nOwned : N;
  m.nCells = nOwn;
  m.nCombined = N;
  auto recip = aux.recip;
  auto cellOf = aux.cellOfFacet;
  auto gidV = opt.gid;
  // classify facets: 1 = emitted interior/interface, 2 = boundary, 0 = twin, 3 = dropped
  Kokkos::View<int*, Mem> kind("fv.kind", nF);
  Kokkos::parallel_for(
      "fv.kindOpt", Kokkos::RangePolicy<Exec>(0, nF), KOKKOS_LAMBDA(const int f) {
        const int j = view.facetNbr(f);
        if (j < 0) {
          kind(f) = 2;
        } else if (j >= nOwn) {
          kind(f) = 1;
        } else if (recip(f) >= 0) {
          bool emits = f < recip(f);  // owner facet: the lower facet index of the pair
          if (det) {                  // owner cell: the lower gid (a self-pair: owner facet)
            const int i = cellOf(f);
            const std::int64_t gi = hasGid ? gidV(i) : std::int64_t(i);
            const std::int64_t gj = hasGid ? gidV(j) : std::int64_t(j);
            emits = gi < gj || (gi == gj && emits);
          }
          kind(f) = emits ? 1 : 0;
        } else {
          kind(f) = 3;
        }
      });
  {
    int nd = 0;
    Kokkos::parallel_reduce(
        "fv.droppedOpt", Kokkos::RangePolicy<Exec>(0, nF),
        KOKKOS_LAMBDA(const int f, int& acc) { acc += kind(f) == 3 ? 1 : 0; }, nd);
    m.nDropped = nd;
  }
  // face index of every emitted facet (-1 otherwise)
  Kokkos::View<int*, Mem> faceOf(view_alloc(std::string("fv.faceOf"), WithoutInitializing), nF);
  int nI = 0, nB = 0;
  if (!det) {  // the plain builder's enumeration: two scans over the facet index
    Kokkos::View<int*, Mem> idxI(view_alloc(std::string("fv.idxI"), WithoutInitializing), nF);
    Kokkos::View<int*, Mem> idxB(view_alloc(std::string("fv.idxB"), WithoutInitializing), nF);
    Kokkos::parallel_scan(
        "fv.scanIOpt", nF,
        KOKKOS_LAMBDA(const int f, int& upd, const bool fin) {
          if (fin)
            idxI(f) = upd;
          upd += (kind(f) == 1) ? 1 : 0;
        },
        nI);
    Kokkos::parallel_scan(
        "fv.scanBOpt", nF,
        KOKKOS_LAMBDA(const int f, int& upd, const bool fin) {
          if (fin)
            idxB(f) = upd;
          upd += (kind(f) == 2) ? 1 : 0;
        },
        nB);
    const int nIc = nI;
    Kokkos::parallel_for(
        "fv.faceOf", Kokkos::RangePolicy<Exec>(0, nF), KOKKOS_LAMBDA(const int f) {
          const int k = kind(f);
          faceOf(f) = k == 1 ? idxI(f) : (k == 2 ? nIc + idxB(f) : -1);
        });
  } else {  // (emitting cell slot, facet position): per-cell counts, scans over cells
    Kokkos::View<int*, Mem> baseI(view_alloc(std::string("fv.baseI"), WithoutInitializing), N);
    Kokkos::View<int*, Mem> baseB(view_alloc(std::string("fv.baseB"), WithoutInitializing), N);
    Kokkos::parallel_scan(
        "fv.scanCellsI", N,
        KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
          int c = 0;
          for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f)
            c += kind(f) == 1 ? 1 : 0;
          if (fin)
            baseI(i) = upd;
          upd += c;
        },
        nI);
    Kokkos::parallel_scan(
        "fv.scanCellsB", N,
        KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
          int c = 0;
          for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f)
            c += kind(f) == 2 ? 1 : 0;
          if (fin)
            baseB(i) = upd;
          upd += c;
        },
        nB);
    const int nIc = nI;
    Kokkos::parallel_for(
        "fv.faceOfCells", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          int gi = baseI(i), gb = nIc + baseB(i);
          for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
            const int k = kind(f);
            faceOf(f) = k == 1 ? gi++ : (k == 2 ? gb++ : -1);
          }
        });
  }
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
    auto P = opt.pos;
    auto W = opt.weight;
    const Real Lx = opt.L[0], Ly = opt.L[1], Lz = opt.L[2];
    const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
    Kokkos::parallel_for(
        "fv.fillOpt", Kokkos::RangePolicy<Exec>(0, nF), KOKKOS_LAMBDA(const int f) {
          const int k = kind(f);
          if (k == 0 || k == 3)
            return;
          const int g = faceOf(f);
          const int e = cellOf(f);  // emitting cell
          // interior: the reciprocal facet's cell; interface (no reciprocal): the ghost index
          const int o = (k == 1) ? (recip(f) >= 0 ? cellOf(recip(f)) : view.facetNbr(f)) : -1;
          bool flip = false;  // A is the other cell (gid orientation)
          if (k == 1 && gidOri) {
            const std::int64_t ge = hasGid ? gidV(e) : std::int64_t(e);
            const std::int64_t go = hasGid ? gidV(o) : std::int64_t(o);
            flip = go < ge;
          }
          const int a = flip ? o : e;
          const int b = flip ? e : o;
          A(g) = a;
          B(g) = b;
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
          // centroid: dV/dr = (|A|/|r|)(r − c)  ⇒  c = r − dV·|r|/|A|  (emitter-relative)
          const Real dvx = view.connect(f, 0), dvy = view.connect(f, 1), dvz = view.connect(f, 2);
          const Real s = (ar > Real(0)) ? rl / ar : Real(0);
          CE(3 * g) = rx - dvx * s;
          CE(3 * g + 1) = ry - dvy * s;
          CE(3 * g + 2) = rz - dvz * s;
          if (k == 1 && gen) {
            Real cx = P(3 * b) - P(3 * a), cy = P(3 * b + 1) - P(3 * a + 1),
                 cz = P(3 * b + 2) - P(3 * a + 2);
            cx = cx > Lxh ? cx - Lx : (cx < -Lxh ? cx + Lx : cx);
            cy = cy > Lyh ? cy - Ly : (cy < -Lyh ? cy + Ly : cy);
            cz = cz > Lzh ? cz - Lz : (cz < -Lzh ? cz + Lz : cz);
            const Real d2 = cx * cx + cy * cy + cz * cz;
            const Real d = Kokkos::sqrt(d2);
            const Real wa = hasW ? W(a) : Real(0), wb = hasW ? W(b) : Real(0);
            const Real hA = (d2 + wa - wb) / (Real(2) * d);
            CN(3 * g) = cx;
            CN(3 * g + 1) = cy;
            CN(3 * g + 2) = cz;
            DS(g) = d;
            HA(g) = hA;
            HB(g) = d - hA;  // signed: a power seed may lie outside its cell
            // the normal points A → B: flip the area vector if needed (emitter B)
            if (NM(3 * g) * cx + NM(3 * g + 1) * cy + NM(3 * g + 2) * cz < Real(0)) {
              NM(3 * g) = -NM(3 * g);
              NM(3 * g + 1) = -NM(3 * g + 1);
              NM(3 * g + 2) = -NM(3 * g + 2);
            }
            if (flip) {  // c − x_A = (c − x_B) + conn
              CE(3 * g) += cx;
              CE(3 * g + 1) += cy;
              CE(3 * g + 2) += cz;
            }
          } else if (k == 1) {
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
  if (gen && nI > 0) {  // diagnostics of the generator geometry
    Real vsum = 0;
    auto vol = view.cellVolume;
    Kokkos::parallel_reduce(
        "fv.meanVolume", Kokkos::RangePolicy<Exec>(0, nOwn),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc += vol(i); }, vsum);
    const Real h = Kokkos::cbrt(vsum / Real(nOwn > 0 ? nOwn : 1));
    const Real aMin = Real(1e-10) * h * h;
    auto AR = m.faceArea;
    auto NM = m.faceNormal;
    auto CN = m.faceConn;
    auto DS = m.faceDist;
    auto HA = m.faceHa;
    auto HB = m.faceHb;
    Real sMax = 0;
    Kokkos::parallel_reduce(
        "fv.normalConnSine", Kokkos::RangePolicy<Exec>(0, nI),
        KOKKOS_LAMBDA(const int g, Real& acc) {
          if (!(AR(g) > aMin))
            return;
          const Real nx = NM(3 * g), ny = NM(3 * g + 1), nz = NM(3 * g + 2);
          const Real cx = CN(3 * g), cy = CN(3 * g + 1), cz = CN(3 * g + 2);
          const Real tx = ny * cz - nz * cy, ty = nz * cx - nx * cz, tz = nx * cy - ny * cx;
          const Real sn = Kokkos::sqrt(tx * tx + ty * ty + tz * tz) / DS(g);
          acc = sn > acc ? sn : acc;
        },
        Kokkos::Max<Real>(sMax));
    int nNeg = 0;
    Kokkos::parallel_reduce(
        "fv.negativeHeight", Kokkos::RangePolicy<Exec>(0, nI),
        KOKKOS_LAMBDA(const int g, int& acc) {
          acc += (HA(g) < Real(0) || HB(g) < Real(0)) ? 1 : 0;
        },
        nNeg);
    m.generatorDiag.maxNormalConnSine = sMax;
    m.generatorDiag.numNegativeHeight = nNeg;
  }
  // cell -> faces CSR (each interior face in two cells, boundary in one); sign +1 if the cell is A
  const int nEntries = 2 * nI + nB;
  m.cellFaceOffset = Kokkos::View<int*, Mem>("fv.cellFaceOffset", N + 1);
  m.cellFace = Kokkos::View<int*, Mem>(view_alloc(std::string("fv.cellFace"), WithoutInitializing),
                                       nEntries);
  m.cellFaceSign = Kokkos::View<Real*, Mem>(
      view_alloc(std::string("fv.cellFaceSign"), WithoutInitializing), nEntries);
  {
    auto off = m.cellFaceOffset;
    auto CF = m.cellFace;
    auto CS = m.cellFaceSign;
    auto A = m.faceCellA;
    auto B = m.faceCellB;
    if (!det) {  // the plain builder's atomic fill (order depends on the thread schedule)
      Kokkos::View<int*, Mem> cnt("fv.cnt", N + 1);
      Kokkos::parallel_for(
          "fv.countOpt", Kokkos::RangePolicy<Exec>(0, nFaces), KOKKOS_LAMBDA(const int g) {
            if (A(g) < nOwn)
              Kokkos::atomic_inc(&cnt(A(g)));
            if (B(g) >= 0 && B(g) < nOwn)
              Kokkos::atomic_inc(&cnt(B(g)));
          });
      Kokkos::parallel_scan(
          "fv.scanCellsOpt", N + 1, KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
            if (fin)
              off(i) = upd;
            upd += (i < N) ? cnt(i) : 0;
          });
      Kokkos::View<int*, Mem> cursor("fv.cursor", N);
      Kokkos::parallel_for(
          "fv.fillCellsOpt", Kokkos::RangePolicy<Exec>(0, nFaces), KOKKOS_LAMBDA(const int g) {
            const int a = A(g);
            if (a < nOwn) {
              const int pa = off(a) + Kokkos::atomic_fetch_add(&cursor(a), 1);
              CF(pa) = g;
              CS(pa) = Real(1);
            }
            const int b = B(g);
            if (b >= 0 && b < nOwn) {
              const int pb = off(b) + Kokkos::atomic_fetch_add(&cursor(b), 1);
              CF(pb) = g;
              CS(pb) = Real(-1);
            }
          });
    } else {  // per-cell: the cell's emitted facets, and the faces its neighbours emit toward it
      const detail::CellFaceEntries<Real> entries{view, kind, faceOf, recip, A, nOwn};
      Kokkos::parallel_scan(
          "fv.scanCellFaces", N + 1, KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
            int c = 0;
            if (i < nOwn)
              entries(i, [&](int, Real) { ++c; });
            if (fin)
              off(i) = upd;
            upd += c;
          });
      Kokkos::parallel_for(
          "fv.fillCellFaces", Kokkos::RangePolicy<Exec>(0, nOwn), KOKKOS_LAMBDA(const int i) {
            const int p0 = off(i);
            int p = p0;
            entries(i, [&](int g, Real sg) {
              CF(p) = g;
              CS(p) = sg;
              ++p;
            });
            // insertion sort by (face index ascending, +1 before -1)
            for (int q = p0 + 1; q < p; ++q) {
              const int g = CF(q);
              const Real sg = CS(q);
              int r = q - 1;
              while (r >= p0 && (CF(r) > g || (CF(r) == g && CS(r) < sg))) {
                CF(r + 1) = CF(r);
                CS(r + 1) = CS(r);
                --r;
              }
              CF(r + 1) = g;
              CS(r + 1) = sg;
            }
          });
    }
  }
  if (opt.cellCentroids) {  // V_i (c_i − x_i) = ¼ Σ_f h_f A_f (c_f − x_i), n_f outward of i
    m.cellCentroid = Kokkos::View<Real*, Mem>("fv.cellCentroid", 3 * (size_t)nOwn);
    auto CC = m.cellCentroid;
    auto off = m.cellFaceOffset;
    auto CF = m.cellFace;
    auto CS = m.cellFaceSign;
    auto AR = m.faceArea;
    auto NM = m.faceNormal;
    auto CE = m.faceCentroid;
    auto CN = m.faceConn;
    auto vol = view.cellVolume;
    Kokkos::parallel_for(
        "fv.cellCentroid", Kokkos::RangePolicy<Exec>(0, nOwn), KOKKOS_LAMBDA(const int i) {
          Real mx = 0, my = 0, mz = 0;
          for (int q = off(i); q < off(i + 1); ++q) {
            const int g = CF(q);
            const Real sg = CS(q);
            // c_f − x_i: A-relative as stored; for i = B subtract the connector A → B
            Real rx = CE(3 * g), ry = CE(3 * g + 1), rz = CE(3 * g + 2);
            if (sg < Real(0)) {
              rx -= CN(3 * g);
              ry -= CN(3 * g + 1);
              rz -= CN(3 * g + 2);
            }
            const Real hf = sg * (NM(3 * g) * rx + NM(3 * g + 1) * ry + NM(3 * g + 2) * rz);
            const Real w = hf * AR(g);
            mx += w * rx;
            my += w * ry;
            mz += w * rz;
          }
          const Real v = vol(i);
          const Real s = v != Real(0) ? Real(0.25) / v : Real(0);
          CC(3 * i) = mx * s;
          CC(3 * i + 1) = my * s;
          CC(3 * i + 2) = mz * s;
        });
  }
  m.cellVolume = view.cellVolume;
  Kokkos::fence();
  return m;
}

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_MESH_HPP
