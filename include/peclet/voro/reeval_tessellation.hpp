/**
 * @file device/reeval_tessellation.hpp
 * \brief Publish a force-geometry TessellationView from a resident topology store (E1 scaffolding).
 *
 * The moving-point repair (MovingTessellation::step) maintains the per-cell topology + volumes in a
 * TopologyStore but, for speed, does NOT emit the per-facet force geometry (area vector, dV,
 * connector) that the physics modules (euler_pressure / viscous) read through a TessellationView.
 * `reevalPublish` fills that gap: it re-evaluates each cell's geometry over the stored topology on
 * the current positions (ConvexCell::reevalGeometry) and packs the same facet-CSR a full
 * buildTessellation would — so the fluid step can use the cheap incremental repair for the topology
 * and this reeval for the forces, instead of a full rebuild every step.
 *
 * It reproduces buildCell's publish half exactly (the >=3-incident-live-triangle face criterion,
 * the facetGeometry(k) call with conn = 2·n[k]), differing only in that the per-cell facet base is
 * a deterministic exclusive prefix sum (contiguous by cell index) rather than the build's atomic
 * cell-finish order — both are valid facetBegin/facetEnd bases, and the physics + buildAuxMaps are
 * agnostic to which. The repair's already-computed volumes are reused for cellVolume.
 *
 * NOTE (physics still WIP): this is the scaffolding to let the incremental repair feed the physics;
 * the physics wiring/tuning on top is the user's ongoing work. Cubic-box minimal image (L = Lx),
 * matching the repair's reevalGeometry.
 */
#ifndef PECLET_VORO_REEVAL_TESSELLATION_HPP
#define PECLET_VORO_REEVAL_TESSELLATION_HPP

#include <Kokkos_Core.hpp>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/sdf.hpp"        // WallStore
#include "peclet/voro/tess_grid.hpp"  // scratchView
#include "peclet/voro/tessellation_view.hpp"
#include "peclet/voro/tessellator.hpp"  // areaGradEnumerate / areaGradFill (rung A3)
#include "peclet/voro/topology_store.hpp"

namespace peclet::voro {

/// Re-evaluate the geometry of every cell in `store` on positions `pos` and publish a
/// force-geometry TessellationView (facet neighbour / area / dV / connector CSR), reusing `vol` for
/// the per-cell volume. `MAXP`/`MAXT` are the store's ConvexCell capacities; `N` cells; `L` the
/// (cubic) box.
/// `wall` + `xRef` (optional, rung A0): the resident SDF wall planes and the per-cell positions at
/// which they were saved, so a wall-clipped cell's planes are restored for the seed's current
/// position before the re-eval (see WallStore). Leave both empty for a wall-free tessellation.
/// `withAreaGrad` (rung A3): also publish the facet-edge area-Jacobian CSR (the same layout the
/// cold build emits, with a deterministic prefix-sum reservation), so the energy layer runs on
/// the incremental path too.
/// `Policy` + `weight` (rung M7): the plane policy of the resident engine. A Power tessellation is
/// re-evaluated on its radical planes with the resident weights (`weight`, N), so the published
/// geometry is the power geometry of the power topology; Voronoi (the default) ignores `weight`
/// and is byte-identical to the historical path.
/// Persistent arrays of reevalPublish (M7 WO-PERF3 P3.1), held by a caller that publishes every
/// step (the M7 band publish of lagrangian/potential.hpp). The view a call returns aliases them:
/// it is valid until the next call with the same scratch.
template <class Real>
struct ReevalScratch {
  using Mem = peclet::core::MemSpace;
  Kokkos::View<int*, Mem> facetCount, edgeCount, base, ebase, eOff, eCnt, eFacet;
  Kokkos::View<gid_t*, Mem> fNbr, seedId;
  Kokkos::View<Real*, Mem> fArea, fDV, fConn, eGrad, eLen, fM2, edgeLength;
};

template <class Real, int MAXP, int MAXT, class Policy = Voronoi>
TessellationView<Real> reevalPublish(const TopologyStore<MAXP, MAXT>& store,
                                     const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                                     const Kokkos::View<Real*, peclet::core::MemSpace>& vol, int N,
                                     const Real L[3], WallStore<Real> wall = {},
                                     Kokkos::View<Real*, peclet::core::MemSpace> xRef = {},
                                     bool withAreaGrad = false, bool withMoments = false,
                                     Kokkos::View<Real*, peclet::core::MemSpace> weight = {},
                                     Kokkos::View<int*, peclet::core::MemSpace> mask = {},
                                     ReevalScratch<Real>* scr = nullptr) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Cell = ConvexCell<Real, MAXP, MAXT, false>;
  const Real Lx = L[0], Ly = L[1], Lz = L[2];
  TopologyStore<MAXP, MAXT> st = store;
  Kokkos::View<Real*, Mem> P = pos;

  // ---- 1. per-cell live-face count (topology only: the >=3-incident-live-triangle criterion) ----
  auto facetCount = scratchView(scr ? &scr->facetCount : nullptr, static_cast<std::size_t>(N),
                                "rp.facetCount", true);
  auto edgeCount =
      scratchView(scr ? &scr->edgeCount : nullptr, withAreaGrad ? static_cast<std::size_t>(N) : 0,
                  "rp.edgeCount", true);
  constexpr int kMaxF = MAXP;  // == CellBuilder::MAXF_TMP: what the cold build can publish
  const bool wag = withAreaGrad;
  // `mask` (N, optional): cells with mask(i) == 0 are published with no facets (and no area
  // gradients) — the band publish of M7 (DESIGN_NOTE §14.4.6 item 3); empty: every cell
  const Kokkos::View<int*, Mem> msk = mask;
  const bool masked = mask.extent(0) > 0;
  Kokkos::parallel_for(
      "reevalPublish.count", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        if (masked && msk(i) == 0) {
          facetCount(i) = 0;
          if (wag)
            edgeCount(i) = 0;
          return;
        }
        Cell c;
        st.load(i, c, Lx, Ly, Lz);
        int nf = 0;
        int faces[kMaxF];
        for (int k = 0; k < c.np; ++k) {
          int cnt = 0;
          for (int t = 0; t < c.nt; ++t)
            if (c.alive[t] && (c.t0[t] == k || c.t1[t] == k || c.t2[t] == k))
              ++cnt;
          if (cnt >= 3) {
            if (nf < kMaxF)
              faces[nf] = k;
            ++nf;
          }
        }
        facetCount(i) = nf;
        if (wag) {
          unsigned char part[kMaxF * Cell::MAXFV], pcnt[kMaxF];
          bool full = false;
          edgeCount(i) = (nf > 0 && nf <= kMaxF)
                             ? areaGradEnumerate<Cell, kMaxF>(c, faces, nf, part, pcnt, full)
                             : 0;
        }
      });

  // ---- 2. exclusive prefix sum -> per-cell facet base + total facet count ----
  auto base =
      scratchView(scr ? &scr->base : nullptr, static_cast<std::size_t>(N), "rp.base", false);
  int nFacets = 0;
  {
    Kokkos::View<int*, Mem> fc = facetCount, bs = base;
    Kokkos::parallel_scan(
        "reevalPublish.scan", N,
        KOKKOS_LAMBDA(const int i, int& upd, const bool final_pass) {
          if (final_pass)
            bs(i) = upd;
          upd += fc(i);
        },
        nFacets);
  }
  auto ebase = scratchView(scr ? &scr->ebase : nullptr,
                           withAreaGrad ? static_cast<std::size_t>(N) : 0, "rp.ebase", false);
  int nEdges = 0;
  if (withAreaGrad) {
    Kokkos::View<int*, Mem> ec = edgeCount, eb = ebase;
    Kokkos::parallel_scan(
        "reevalPublish.scanEdges", N,
        KOKKOS_LAMBDA(const int i, int& upd, const bool final_pass) {
          if (final_pass)
            eb(i) = upd;
          upd += ec(i);
        },
        nEdges);
  }

  // ---- 3. fill the facet CSR: reeval geometry, then facetGeometry per live face ----
  const std::size_t nF = static_cast<std::size_t>(nFacets);
  auto fNbr = scratchView(scr ? &scr->fNbr : nullptr, nF, "rp.fNbr", false);
  auto fArea = scratchView(scr ? &scr->fArea : nullptr, nF * 3, "rp.fArea", false);
  auto fDV = scratchView(scr ? &scr->fDV : nullptr, nF * 3, "rp.fDV", false);
  auto fConn = scratchView(scr ? &scr->fConn : nullptr, nF * 3, "rp.fConn", false);
  const std::size_t nE = static_cast<std::size_t>(nEdges);
  auto eOff = scratchView(scr ? &scr->eOff : nullptr, withAreaGrad ? nF : 0, "rp.eOff", false);
  auto eCnt = scratchView(scr ? &scr->eCnt : nullptr, withAreaGrad ? nF : 0, "rp.eCnt", false);
  auto eFacet =
      scratchView(scr ? &scr->eFacet : nullptr, withAreaGrad ? nE : 0, "rp.eFacet", false);
  auto eGrad =
      scratchView(scr ? &scr->eGrad : nullptr, withAreaGrad ? nE * 3 : 0, "rp.eGrad", false);
  auto eLen = scratchView(scr ? &scr->eLen : nullptr, withAreaGrad ? nE * 4 : 0, "rp.eLen", false);
  auto fM2 = scratchView(scr ? &scr->fM2 : nullptr, withMoments ? nF : 0, "rp.fM2", false);
  const bool wm = withMoments;
  {
    Kokkos::View<int*, Mem> bs = base, eb = ebase;
    WallStore<Real> Wl = wall;
    Kokkos::View<Real*, Mem> XR = xRef;
    Kokkos::View<Real*, Mem> W = weight;
    const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
    Kokkos::parallel_for(
        "reevalPublish.fill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          if (masked && msk(i) == 0)
            return;
          Cell c;
          st.load(i, c, Lx, Ly, Lz);
          if (Wl.active()) {  // restore the wall planes for the seed's displacement since save
            Real dx = P(3 * i) - XR(3 * i), dy = P(3 * i + 1) - XR(3 * i + 1),
                 dz = P(3 * i + 2) - XR(3 * i + 2);
            dx = dx > Lxh ? dx - Lx : (dx < -Lxh ? dx + Lx : dx);
            dy = dy > Lyh ? dy - Ly : (dy < -Lyh ? dy + Ly : dy);
            dz = dz > Lzh ? dz - Lz : (dz < -Lzh ? dz + Lz : dz);
            Wl.load(i, c, dx, dy, dz);
          }
          const Real* wPtr = W.data();  // force-capture W OUTSIDE the constexpr-if (nvcc rule)
          (void)wPtr;
          if constexpr (Policy::kHasWeightDof)
            c.template reevalGeometry<Policy>(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), Lx,
                                              wPtr[i], wPtr);
          else
            c.reevalGeometry(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), Lx);
          int idx = 0;
          const int b = bs(i);
          int faces[kMaxF];
          for (int k = 0; k < c.np; ++k) {
            int cnt = 0;
            for (int t = 0; t < c.nt; ++t)
              if (c.alive[t] && (c.t0[t] == k || c.t1[t] == k || c.t2[t] == k))
                ++cnt;
            if (cnt < 3)
              continue;  // not a polygon face — same criterion as the count pass
            if (idx < kMaxF)
              faces[idx] = k;
            Real area[3] = {0, 0, 0}, dv[3] = {0, 0, 0}, conn[3];
            conn[0] = Real(2) * c.n[k][0];
            conn[1] = Real(2) * c.n[k][1];
            conn[2] = Real(2) * c.n[k][2];
            c.facetGeometry(k, area, dv, conn);  // zeros for a degenerate face (matches buildCell)
            const std::size_t g = static_cast<std::size_t>(b + idx);
            fNbr(g) = static_cast<gid_t>(c.pnbr[k]);
            for (int cc = 0; cc < 3; ++cc) {
              fArea(3 * g + cc) = area[cc];
              fDV(3 * g + cc) = dv[cc];
              fConn(3 * g + cc) = conn[cc];
            }
            if (wm)
              fM2(g) = c.faceMoment2(k);
            ++idx;
          }
          if (wag && idx > 0 && idx <= kMaxF) {
            unsigned char part[kMaxF * Cell::MAXFV], pcnt[kMaxF];
            bool full = false;
            areaGradEnumerate<Cell, kMaxF>(c, faces, idx, part, pcnt, full);
            areaGradFill<Cell, kMaxF>(c, faces, idx, part, pcnt, b, eb(i), eOff, eCnt, eFacet,
                                      eGrad, eLen);
          }
        });
  }

  // ---- 4. assemble the published view (cellVolume reuses the repair's volumes) ----
  TessellationView<Real> view;
  view.cellFacetOffset = base;
  view.cellFacetCount = facetCount;
  view.cellVolume = vol;
  view.cellSeedId =
      scratchView(scr ? &scr->seedId : nullptr, static_cast<std::size_t>(N), "rp.seedId", false);
  {
    Kokkos::View<gid_t*, Mem> vSeed = view.cellSeedId;
    Kokkos::parallel_for(
        "reevalPublish.seedId", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i) { vSeed(i) = static_cast<gid_t>(i); });
  }
  view.facetNeighbor = fNbr;
  view.facetArea = fArea;
  view.facetConnect = fDV;
  view.facetConnVec = fConn;
  if (withMoments)
    view.facetMoment2 = fM2;
  if (withAreaGrad) {
    view.facetEdgeOffset = eOff;
    view.facetEdgeCount = eCnt;
    view.edgeFacet = eFacet;
    view.edgeAreaGrad = eGrad;
    view.edgeLength = scratchView(scr ? &scr->edgeLength : nullptr, nE, "rp.edgeLength", false);
    {
      auto el = view.edgeLength;
      Kokkos::parallel_for(
          "reevalPublish.edgeLength", Kokkos::RangePolicy<Exec>(0, nE),
          KOKKOS_LAMBDA(const int e) { el(e) = eLen((size_t)e * 4); });
    }
  }
  return view;
}

}  // namespace peclet::voro

#endif  // PECLET_VORO_REEVAL_TESSELLATION_HPP
