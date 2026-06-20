/**
 * @file tessellation_view.hpp
 * \brief Published, read-only device data layer for the tessellation (migration §2.1, §3).
 *
 * This is the stable boundary between the tessellation engine and its consumers
 * (physics, microstructure analysis, ...). Consumers `parallel_for` over a
 * data-oriented CSR view of per-cell / per-facet quantities and never touch the
 * half-edge internals — which is what keeps the engine reusable (plan §1).
 *
 * Pieces:
 *   - HostTessellation : plain std::vector CSR buffers (the legacy<->View seam).
 *   - buildReciprocalMap() : the NbrsToFacets transpose for atomic-free gather.
 *   - TessellationView : device-resident Kokkos Views + KOKKOS_INLINE_FUNCTION
 *                        accessors; built by upload().
 * The legacy CellComplex -> HostTessellation converter (buildHostTessellation)
 * lives in tessellation_build.hpp so view consumers need not include the engine.
 *
 * Storage matches docs/update_to_kokkos_plan.md §3: an exclusive-scan facet
 * offset (mean cell ~15 facets, so CSR, not the 128 cap) plus packed facet
 * arrays. Per-facet 3-vectors are stored flat (3*f + c) so the layout is
 * unambiguous across backends; coalescing layout is retuned in Phase 7.
 *
 * Requires Kokkos; only included by the device build (-DVORFLOW_KOKKOS=ON). This
 * is a *core* header: it must never include a physics header (see §1).
 */
#ifndef VORFLOW_TESSELLATION_VIEW_HPP
#define VORFLOW_TESSELLATION_VIEW_HPP

#include <cmath>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <map>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/vor_types.hpp"  // uint1/uint2 only; the legacy engine is NOT pulled in here

namespace vor {

/// Seed/neighbour identifier in the published view. Signed so the legacy
/// sentinels map cleanly to negatives: noNbr (0xFFFFFFFF) -> -1, boundaryNbr
/// (0xFFFFFFFE) -> -2. Consumers treat facetNbr(f) < 0 as a boundary facet.
using gid_t = std::int32_t;

inline gid_t toGid(uint2 id) {
  return static_cast<gid_t>(static_cast<std::int32_t>(id));
}

/// Host-side CSR buffers — the conversion target and the round-trip oracle.
template <class Real>
struct HostTessellation {
  int nCells = 0;
  int nFacets = 0;
  std::vector<int> cellFacetOffset;  ///< size nCells+1, exclusive scan of facet counts
  std::vector<gid_t> cellSeedId;     ///< size nCells, stable seed id per cell
  std::vector<Real> cellVolume;      ///< size nCells
  std::vector<gid_t> facetNeighbor;  ///< size nFacets (negative => boundary/none)
  std::vector<Real> facetArea;       ///< size 3*nFacets, area vectors (3*f + c)
  std::vector<Real> facetConnect;    ///< size 3*nFacets, dV (volume gradient)
  std::vector<Real> facetConnVec;    ///< size 3*nFacets, connecting vector to neighbour seed
};

/**
 * Reciprocal-facet transpose (NbrsToFacets, plan §2.5). For each packed facet g
 * (cell i -> seed j) returns the packed index of the facet in cell j that points
 * back to seed i, or -1 for a boundary facet or if no reciprocal exists. This is
 * the map the atomic-free gather force reads through: a cell fetches its
 * neighbour's reciprocal-facet quantities without scattering (no atomics).
 *
 * Periodic multi-image facets (several g in i -> same j) are disambiguated by
 * area-vector negation (the reciprocal has the most opposite area vector).
 */
template <class Real>
std::vector<int> buildReciprocalMap(const HostTessellation<Real>& t) {
  std::map<gid_t, int> seedToCell;
  for (int i = 0; i < t.nCells; ++i)
    seedToCell[t.cellSeedId[i]] = i;
  std::vector<int> recip(t.nFacets, -1);
  for (int i = 0; i < t.nCells; ++i) {
    const gid_t idi = t.cellSeedId[i];
    for (int g = t.cellFacetOffset[i]; g < t.cellFacetOffset[i + 1]; ++g) {
      const gid_t j = t.facetNeighbor[g];
      if (j < 0)
        continue;  // boundary
      auto it = seedToCell.find(j);
      if (it == seedToCell.end())
        continue;
      const int cj = it->second;
      int best = -1;
      Real bestErr = Real(1e300);
      for (int h = t.cellFacetOffset[cj]; h < t.cellFacetOffset[cj + 1]; ++h) {
        if (t.facetNeighbor[h] != idi)
          continue;
        Real e = 0;  // prefer the facet whose area best negates g's (same interface)
        for (int c = 0; c < 3; ++c) {
          Real s = t.facetArea[3 * g + c] + t.facetArea[3 * h + c];
          e += s * s;
        }
        if (e < bestErr) {
          bestErr = e;
          best = h;
        }
      }
      recip[g] = best;
    }
  }
  return recip;
}

/// Device-resident published view. Trivially copyable into kernels (Views are
/// reference-counted handles); accessors are KOKKOS_INLINE_FUNCTION.
template <class Real>
struct TessellationView {
  Kokkos::View<int*, tpx::MemSpace> cellFacetOffset;  // nCells+1
  Kokkos::View<gid_t*, tpx::MemSpace> cellSeedId;     // nCells
  Kokkos::View<Real*, tpx::MemSpace> cellVolume;      // nCells
  Kokkos::View<gid_t*, tpx::MemSpace> facetNeighbor;  // nFacets
  Kokkos::View<Real*, tpx::MemSpace> facetArea;       // 3*nFacets
  Kokkos::View<Real*, tpx::MemSpace> facetConnect;    // 3*nFacets (dV)
  Kokkos::View<Real*, tpx::MemSpace> facetConnVec;    // 3*nFacets

  KOKKOS_INLINE_FUNCTION int numCells() const { return static_cast<int>(cellSeedId.extent(0)); }
  KOKKOS_INLINE_FUNCTION int numFacets() const { return static_cast<int>(facetNeighbor.extent(0)); }
  KOKKOS_INLINE_FUNCTION gid_t cellSeed(int i) const { return cellSeedId(i); }
  KOKKOS_INLINE_FUNCTION Real volume(int i) const { return cellVolume(i); }
  KOKKOS_INLINE_FUNCTION int facetBegin(int i) const { return cellFacetOffset(i); }
  KOKKOS_INLINE_FUNCTION int facetEnd(int i) const { return cellFacetOffset(i + 1); }
  KOKKOS_INLINE_FUNCTION gid_t facetNbr(int f) const { return facetNeighbor(f); }
  KOKKOS_INLINE_FUNCTION Real area(int f, int c) const { return facetArea(3 * f + c); }
  KOKKOS_INLINE_FUNCTION Real connect(int f, int c) const { return facetConnect(3 * f + c); }
  KOKKOS_INLINE_FUNCTION Real connVec(int f, int c) const { return facetConnVec(3 * f + c); }
};

namespace detail {
template <class T>
Kokkos::View<T*, tpx::MemSpace> deviceFrom(const std::vector<T>& h, const std::string& label) {
  // NOTE: pass a std::string label — a bare const char* is interpreted by
  // Kokkos::view_alloc as a user pointer-to-memory, not an allocation label.
  Kokkos::View<T*, tpx::MemSpace> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing),
                                    h.size());
  if (!h.empty()) {
    using Host =
        Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    Host hv(h.data(), h.size());
    Kokkos::deep_copy(d, hv);
  }
  return d;
}
}  // namespace detail

/// Upload a HostTessellation to device Views.
template <class Real>
TessellationView<Real> upload(const HostTessellation<Real>& h) {
  TessellationView<Real> v;
  v.cellFacetOffset = detail::deviceFrom(h.cellFacetOffset, "tess.cellFacetOffset");
  v.cellSeedId = detail::deviceFrom(h.cellSeedId, "tess.cellSeedId");
  v.cellVolume = detail::deviceFrom(h.cellVolume, "tess.cellVolume");
  v.facetNeighbor = detail::deviceFrom(h.facetNeighbor, "tess.facetNeighbor");
  v.facetArea = detail::deviceFrom(h.facetArea, "tess.facetArea");
  v.facetConnect = detail::deviceFrom(h.facetConnect, "tess.facetConnect");
  v.facetConnVec = detail::deviceFrom(h.facetConnVec, "tess.facetConnVec");
  return v;
}

}  // namespace vor

#endif  // VORFLOW_TESSELLATION_VIEW_HPP
