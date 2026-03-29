/**
 * @file test_build_rebuild.cpp
 * @brief regression test: periodic random displacements followed by incremental update
 *
 * The incremental update path is compared against a clean rebuild on the same
 * final positions. This guards against missed topology changes under normal
 * periodic boundary conditions.
 */

#include <algorithm>
#include <boost/random.hpp>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>
#include <voronoi_dynamics/voronoi.hpp>

using std::clock;
using std::clock_t;
using std::sort;
using std::vector;
using vor::Box;
using vor::CellComplex;
using vor::CellGeometry;
using vor::CellView;
using vor::uint1;
using vor::uint2;

namespace {

template <typename real_t>
real_t sumCellVolumes(std::vector<CellGeometry<real_t> > &geoms) {
  long double vol = 0.0L;
  for (size_t i = 0; i < geoms.size(); ++i) {
    geoms[i].computeVolume();
    vol += static_cast<long double>(geoms[i].getVolume());
  }
  return static_cast<real_t>(vol);
}

template <typename real_t>
void collectSortedNbrs(const CellView<real_t> &cell, std::vector<uint2> &nbrs) {
  nbrs.clear();
  nbrs.reserve(cell.numFacets());
  for (uint1 facet = 0; facet < cell.numFacets(); ++facet) {
    const uint2 nbrId = cell.getNbr(facet);
    if (nbrId != vor::noNbr)
      nbrs.push_back(nbrId);
  }
  sort(nbrs.begin(), nbrs.end());
}

}  // namespace

int main() {
  typedef double real_t;
  typedef boost::mt11213b base_generator_type;
  typedef boost::uniform_01<real_t> distribution_type;
  typedef boost::variate_generator<base_generator_type &, distribution_type> gen_type;

  base_generator_type rng(1);
  gen_type pointGen(rng, distribution_type());

  std::array<real_t, 3> L;
  L[0] = 1;
  L[1] = 1;
  L[2] = 1;
  Box<real_t> box(L);
  CellComplex<real_t> complex(&box);
  CellComplex<real_t> reference(&box);
  vector<CellGeometry<real_t> > &geoms(complex.getGeoms());
  vector<CellGeometry<real_t> > &refGeoms(reference.getGeoms());

  vector<std::array<real_t, 3> > p(10000);
  for (size_t i = 0; i < p.size(); ++i)
    for (uint1 k = 0; k < 3; ++k)
      p[i][k] = L[k] * pointGen();

  clock_t start = clock();
  complex.build(p);
  real_t duration = real_t(clock() - start) / real_t(CLOCKS_PER_SEC);
  printf("%lu Voronoi cells created in %f seconds\n", static_cast<unsigned long>(complex.numCells()),
         duration);

  reference.build(p);

  const real_t fraction = 0.01;
  const real_t density = real_t(p.size()) / (L[0] * L[1] * L[2]);
  const real_t dx = fraction * std::pow(density, real_t(-1.0 / 3.0));
  const int nRepeat = 10;
  duration = 0;

  for (int step = 0; step < nRepeat; ++step) {
    for (size_t i = 0; i < p.size(); ++i) {
      p[i][0] += dx * (pointGen() - real_t(0.5));
      p[i][1] += dx * (pointGen() - real_t(0.5));
      p[i][2] += dx * (pointGen() - real_t(0.5));
    }
    box.putInBox(p);

    start = clock();
    complex.update(p);
    duration += real_t(clock() - start);

    reference.build(p);
  }
  printf("updating took %f s on average\n", duration / (real_t(CLOCKS_PER_SEC) * real_t(nRepeat)));

  const real_t volUpdate = sumCellVolumes(geoms);
  const real_t volReference = sumCellVolumes(refGeoms);
  printf("summed volume update:    %.12f\n", volUpdate);
  printf("summed volume reference: %.12f\n", volReference);

  std::vector<uint2> nbrsUpdate;
  std::vector<uint2> nbrsReference;
  size_t mismatchCell = p.size();
  for (size_t i = 0; i < p.size(); ++i) {
    collectSortedNbrs(complex.getCellView(i), nbrsUpdate);
    collectSortedNbrs(reference.getCellView(i), nbrsReference);
    if (nbrsUpdate != nbrsReference) {
      mismatchCell = i;
      break;
    }
  }

  const real_t tol = 1.0e-10;
  if (std::abs(volUpdate - real_t(1)) > tol || std::abs(volUpdate - volReference) > tol ||
      mismatchCell != p.size()) {
    if (mismatchCell != p.size()) {
      collectSortedNbrs(complex.getCellView(mismatchCell), nbrsUpdate);
      collectSortedNbrs(reference.getCellView(mismatchCell), nbrsReference);
      printf("topology mismatch in cell %lu: update facets=%lu reference facets=%lu\n",
             static_cast<unsigned long>(mismatchCell), static_cast<unsigned long>(nbrsUpdate.size()),
             static_cast<unsigned long>(nbrsReference.size()));
    }
    std::fprintf(stderr,
                 "Periodic rebuild regression failed: update volume %.12f, reference volume %.12f, "
                 "topologyMismatch=%s\n",
                 volUpdate, volReference, (mismatchCell == p.size() ? "false" : "true"));
    return 1;
  }

  return 0;
}
