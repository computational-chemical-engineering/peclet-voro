/**
 * @file device/transpose.hpp
 * \brief Device-native reciprocal-facet transpose + facet->cell map.
 *
 * The atomic-free physics gathers read through the reciprocal-facet map (a facet
 * i->j paired with the facet j->i, plan §2.5). buildReciprocalMap() in
 * tessellation_view.hpp builds it on the host; this builds it directly on device
 * from a TessellationView, so the whole pipeline (tessellate -> publish -> force
 * -> integrate) stays device-resident with no host round-trip.
 *
 * Dense single-domain build: facetNeighbor stores the neighbour's CELL INDEX and
 * cellSeedId(i) == i, which is what the device search uses. Periodic multi-image
 * facets are disambiguated by area-vector negation.
 */
#ifndef VORFLOW_DEVICE_TRANSPOSE_HPP
#define VORFLOW_DEVICE_TRANSPOSE_HPP

#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"
#include "vorflow/tessellation_view.hpp"

namespace vor {
namespace device {

template <class Real>
struct AuxMaps {
  Kokkos::View<int*, tpx::MemSpace> recip;        // nFacets: reciprocal facet, or -1
  Kokkos::View<int*, tpx::MemSpace> cellOfFacet;  // nFacets: owning cell index
};

/// Build {recip, cellOfFacet} on device from a published view (dense single-domain).
template <class Real>
AuxMaps<Real> buildAuxMaps(const TessellationView<Real>& view) {
  using Exec = tpx::ExecSpace;
  const int N = view.numCells();
  const int nF = view.numFacets();
  AuxMaps<Real> aux;
  aux.recip = Kokkos::View<int*, tpx::MemSpace>("recip", nF);
  aux.cellOfFacet = Kokkos::View<int*, tpx::MemSpace>("cellOfFacet", nF);
  auto recip = aux.recip;
  auto cellOfFacet = aux.cellOfFacet;

  Kokkos::parallel_for(
      "aux.cellOfFacet", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f)
          cellOfFacet(f) = i;
      });

  Kokkos::parallel_for(
      "aux.recip", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const gid_t idi = view.cellSeed(i);
        for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
          const gid_t j = view.facetNbr(g);
          recip(g) = -1;
          if (j < 0 || j >= N)
            continue;
          int best = -1;
          Real bestErr = Real(1e300);
          for (int h = view.facetBegin((int)j); h < view.facetEnd((int)j); ++h) {
            if (view.facetNbr(h) != idi)
              continue;
            Real e = 0;  // area-vector negation picks the matching interface
            for (int c = 0; c < 3; ++c) {
              Real s = view.area(g, c) + view.area(h, c);
              e += s * s;
            }
            if (e < bestErr) {
              bestErr = e;
              best = h;
            }
          }
          recip(g) = best;
        }
      });
  Kokkos::fence();
  return aux;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TRANSPOSE_HPP
