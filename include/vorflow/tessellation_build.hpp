/**
 * @file tessellation_build.hpp
 * \brief Legacy CellComplex -> HostTessellation CSR converter (migration Phase 1).
 *
 * Separated from tessellation_view.hpp so the published view (and its physics
 * consumers) need not pull in the legacy engine: this is the one place that
 * includes voronoi.hpp to read the legacy arenas. Engine-side; not a physics
 * header.
 */
#ifndef VORFLOW_TESSELLATION_BUILD_HPP
#define VORFLOW_TESSELLATION_BUILD_HPP

#include <array>
#include <vector>

#include "vorflow/tessellation_view.hpp"
#include "vorflow/voronoi.hpp"

namespace vor {

/// Extract the CSR from a built legacy CellComplex (computeGeometry must have run).
template <class Real, bool Weighted>
HostTessellation<Real> buildHostTessellation(const CellComplex<Real, Weighted>& complex) {
  HostTessellation<Real> t;
  std::vector<Cell<Real> > cells;
  complex.materializeCells(cells);
  const std::vector<CellGeometry<Real> >& geom = complex.getGeoms();

  t.nCells = static_cast<int>(cells.size());
  t.cellFacetOffset.resize(t.nCells + 1, 0);
  t.cellSeedId.resize(t.nCells);
  t.cellVolume.resize(t.nCells);

  // Pass 1: facet counts -> exclusive scan.
  for (int i = 0; i < t.nCells; ++i)
    t.cellFacetOffset[i + 1] = t.cellFacetOffset[i] + cells[i].numFacets();
  t.nFacets = t.cellFacetOffset[t.nCells];

  t.facetNeighbor.resize(t.nFacets);
  t.facetArea.assign(3 * t.nFacets, Real(0));
  t.facetConnect.assign(3 * t.nFacets, Real(0));
  t.facetConnVec.assign(3 * t.nFacets, Real(0));

  // Pass 2: pack per-cell scalars and per-facet vectors.
  for (int i = 0; i < t.nCells; ++i) {
    t.cellSeedId[i] = toGid(cells[i].getID());
    t.cellVolume[i] = geom[i].getVolume();
    const std::vector<std::array<Real, 3> >& areas = geom[i].getAreas();
    const std::vector<std::array<Real, 3> >& dV = geom[i].getdV();
    const std::vector<std::array<Real, 3> >& cv = geom[i].getConnVect();
    const int base = t.cellFacetOffset[i];
    const int nf = cells[i].numFacets();
    for (int f = 0; f < nf; ++f) {
      const int g = base + f;
      t.facetNeighbor[g] = toGid(cells[i].getNbr(static_cast<uint1>(f)));
      for (int c = 0; c < 3; ++c) {
        if (f < static_cast<int>(areas.size()))
          t.facetArea[3 * g + c] = areas[f][c];
        if (f < static_cast<int>(dV.size()))
          t.facetConnect[3 * g + c] = dV[f][c];
        if (f < static_cast<int>(cv.size()))
          t.facetConnVec[3 * g + c] = cv[f][c];
      }
    }
  }
  return t;
}

}  // namespace vor

#endif  // VORFLOW_TESSELLATION_BUILD_HPP
