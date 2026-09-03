/**
 * @file fv/dec.hpp
 * \brief Track C, rung C2a′ (Voronoi methods plan): the discrete-exterior-calculus (Nicolaides
 * covolume) vector Laplacian of the face-normal flux field, Δu = grad div u − curl curl u, on the
 * Voronoi–Delaunay pair.
 *
 * The face flux u_f is the primal 1-form on the Delaunay edge e = (A, B) (value u_f d_f). Its
 * exterior derivative lives on the Delaunay triangles t = (A, B, k), each dual to a Voronoi EDGE
 * ε of the face f (the edge where f's plane meets the plane toward k): the circulation
 *   Γ_t = u_{AB} d_{AB} + u_{Bk} d_{Bk} + u_{kA} d_{kA}   (loop A → B → k → A),
 * so that ω·t̂ = Γ_t / |t| is the vorticity along ε, and Stokes on the face polygon gives
 *   (curl ω)·n_f = (1/A_f) Σ_{ε∈∂f} ℓ_ε Γ_t / |t|
 * (the loop orientation A→B→k makes t̂ = (r_AB × r_Ak)/|…| the counterclockwise traversal
 * direction of ε about n_f, so every sign is +). With the Hodge stars ⋆₁ = A_f/d_f and
 * ⋆₂ = ℓ_ε/|t| this is exactly −⋆₁⁻¹ dᵀ ⋆₂ d: symmetric negative semidefinite in ⟨·,·⟩_F —
 * the viscous term is dissipative to round-off, unlike the Perot-reconstructed Rᵀ Δ₂ R only
 * approximately (C2a). grad div: the two-point face gradient of the cell divergence (C1).
 *
 * Inputs: the face mesh plus the view's facet-edge CSR with edge lengths (`withAreaGrad`,
 * `edgeLength`). Single rank (the third leg B–k needs cell B's facets; ghosts have none).
 * Faces adjacent to walls are skipped (their triangles are undefined): the caller keeps the
 * two-point wall treatment there.
 */
#ifndef PECLET_VORO_FV_DEC_HPP
#define PECLET_VORO_FV_DEC_HPP

#include <Kokkos_Core.hpp>

#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/fv/operators.hpp"
#include "peclet/voro/tessellation_view.hpp"

namespace peclet::voro::fv {

/// Per interior face: its Voronoi edges with the DEC weight ℓ_ε/|t| and the two other legs of the
/// Delaunay triangle as (face, orientation sign) pairs; sign s: +1 if the leg's direction agrees
/// with the face's stored A→B orientation.
template <class Real>
struct DecEdges {
  using Mem = peclet::core::MemSpace;
  Kokkos::View<int*, Mem> offset;         // nInterior + 1
  Kokkos::View<Real*, Mem> weight;        // nEdges: ℓ_ε / |t|
  Kokkos::View<int*, Mem> face2, face3;   // nEdges: faces (B,k) and (k,A)
  Kokkos::View<Real*, Mem> sign2, sign3;  // nEdges: orientation of the legs B→k and k→A
  int nEdges = 0, nSkipped = 0;           // skipped: legs not found (wall / ghost / degenerate)
};

/// Build the DEC edge structure. `facetOfFace`-style inverse maps are built here from the mesh.
template <class Real>
DecEdges<Real> buildDecEdges(const TessellationView<Real>& view, const FaceMesh<Real>& m) {
  using Exec = peclet::core::ExecSpace;
  using Mem = peclet::core::MemSpace;
  DecEdges<Real> d;
  const int nI = m.nInterior, nF = view.numFacets();
  // facet -> face (+ sign): every interior face is the owner facet (+1) and its reciprocal (−1)
  Kokkos::View<int*, Mem> facetFace("dec.facetFace", nF);
  Kokkos::View<Real*, Mem> facetSign("dec.facetSign", nF);
  Kokkos::deep_copy(facetFace, -1);
  {
    auto FF = m.faceFacet;
    auto A = m.faceCellA;
    auto B = m.faceCellB;
    Kokkos::parallel_for(
        "dec.inv", Kokkos::RangePolicy<Exec>(0, nI), KOKKOS_LAMBDA(const int f) {
          const int g = FF(f);
          facetFace(g) = f;
          facetSign(g) = Real(1);
          // the reciprocal facet: cell B's facet whose neighbour is A
          const int b = B(f), a = A(f);
          if (b < 0 || b >= view.numCells())
            return;
          const int beg = view.cellFacetOffset(b), end = beg + view.cellFacetCount(b);
          for (int h = beg; h < end; ++h)
            if (view.facetNbr(h) == a) {
              facetFace(h) = f;
              facetSign(h) = Real(-1);
              break;
            }
        });
  }
  // count edges per interior face (partner slots of the owner facet)
  Kokkos::View<int*, Mem> cnt("dec.cnt", nI + 1);
  {
    auto FF = m.faceFacet;
    Kokkos::parallel_for(
        "dec.cnt", Kokkos::RangePolicy<Exec>(0, nI), KOKKOS_LAMBDA(const int f) {
          const int g = FF(f);
          cnt(f) = view.facetEdgeCount(g) > 0 ? view.facetEdgeCount(g) - 1 : 0;
        });
  }
  d.offset = Kokkos::View<int*, Mem>("dec.offset", nI + 1);
  {
    auto off = d.offset;
    Kokkos::parallel_scan(
        "dec.scan", Kokkos::RangePolicy<Exec>(0, nI + 1),
        KOKKOS_LAMBDA(const int f, int& upd, const bool fin) {
          if (fin)
            off(f) = upd;
          upd += f < nI ? cnt(f) : 0;
        });
  }
  int nE = 0;
  Kokkos::deep_copy(nE, Kokkos::subview(d.offset, nI));
  d.nEdges = nE;
  d.weight = Kokkos::View<Real*, Mem>("dec.weight", nE);
  d.face2 = Kokkos::View<int*, Mem>("dec.face2", nE);
  d.face3 = Kokkos::View<int*, Mem>("dec.face3", nE);
  d.sign2 = Kokkos::View<Real*, Mem>("dec.sign2", nE);
  d.sign3 = Kokkos::View<Real*, Mem>("dec.sign3", nE);
  int skipped = 0;
  {
    auto FF = m.faceFacet;
    auto A = m.faceCellA;
    auto B = m.faceCellB;
    auto off = d.offset;
    auto W = d.weight;
    auto F2 = d.face2;
    auto F3 = d.face3;
    auto S2 = d.sign2;
    auto S3 = d.sign3;
    const int nCells = m.nCells;
    Kokkos::parallel_reduce(
        "dec.fill", Kokkos::RangePolicy<Exec>(0, nI),
        KOKKOS_LAMBDA(const int f, int& sk) {
          const int g = FF(f), a = A(f), b = B(f);
          const int eb = view.facetEdgeOffset(g), ec = view.facetEdgeCount(g);
          int out = off(f);
          // r_AB = connector of facet g
          const Real rAB[3] = {view.connVec(g, 0), view.connVec(g, 1), view.connVec(g, 2)};
          for (int e = eb + 1; e < eb + ec; ++e, ++out) {
            const int g3 = view.edgePartner(e);  // facet of cell A toward k
            const int k = view.facetNbr(g3);
            W(out) = Real(0);
            F2(out) = F3(out) = -1;
            S2(out) = S3(out) = Real(0);
            if (k < 0 || k >= nCells || b < 0 || b >= nCells) {
              ++sk;
              continue;
            }
            const Real rAk[3] = {view.connVec(g3, 0), view.connVec(g3, 1), view.connVec(g3, 2)};
            // |t| = ½ |r_AB × r_Ak|
            const Real cx = rAB[1] * rAk[2] - rAB[2] * rAk[1],
                       cy = rAB[2] * rAk[0] - rAB[0] * rAk[2],
                       cz = rAB[0] * rAk[1] - rAB[1] * rAk[0];
            const Real tArea = Real(0.5) * Kokkos::sqrt(cx * cx + cy * cy + cz * cz);
            const Real len = view.edgeLen(e);
            // leg k→A: facet g3 of A toward k; the face's stored orientation is A→k if A is its
            // cell A, so k→A = −(A→k)
            const int f3 = facetFace(g3);
            if (f3 < 0 || tArea <= Real(0)) {
              ++sk;
              continue;
            }
            const Real s3 = -facetSign(g3);
            // leg B→k: cell B's facet toward k
            int f2 = -1;
            Real s2 = 0;
            const int bb = view.cellFacetOffset(b), be = bb + view.cellFacetCount(b);
            for (int h = bb; h < be; ++h)
              if (view.facetNbr(h) == k) {
                f2 = facetFace(h);
                s2 = facetSign(h);  // +1 if that face is stored B→k
                break;
              }
            if (f2 < 0) {
              ++sk;
              continue;
            }
            W(out) = len / tArea;
            F2(out) = f2;
            S2(out) = s2;
            F3(out) = f3;
            S3(out) = s3;
          }
        },
        skipped);
  }
  d.nSkipped = skipped;
  return d;
}

/// The DEC Laplacian of the face flux u (interior faces): out_f = grad_f(div u) − (curl curl u)_f,
/// with grad div through the C1 operators; `divScratch` (nCells). Faces with a skipped edge get
/// only the grad-div part (flagged by weight 0 — the caller decides how to treat them).
template <class Real>
void decLaplacian(const FaceMesh<Real>& m, const DecEdges<Real>& d, const DV<Real>& u,
                  const DV<Real>& divScratch, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  divergence(m, u, divScratch);
  faceGradient(m, divScratch, out);  // grad div (boundary faces: 0)
  const int nI = m.nInterior;
  Kokkos::parallel_for(
      "dec.curlcurl", Kokkos::RangePolicy<Exec>(0, nI), KOKKOS_LAMBDA(const int f) {
        Real cc = 0;
        const Real g1 = u(f) * m.faceDist(f);  // leg A→B
        for (int e = d.offset(f); e < d.offset(f + 1); ++e) {
          if (d.weight(e) <= Real(0))
            continue;
          const int f2 = d.face2(e), f3 = d.face3(e);
          const Real gamma =
              g1 + d.sign2(e) * u(f2) * m.faceDist(f2) + d.sign3(e) * u(f3) * m.faceDist(f3);
          cc += d.weight(e) * gamma;
        }
        // degenerate (zero-area) faces carry no flux: no curl-curl contribution
        if (m.faceArea(f) > Real(1e-14))
          out(f) -= cc / m.faceArea(f);
        else
          out(f) = Real(0);
      });
}

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_DEC_HPP
