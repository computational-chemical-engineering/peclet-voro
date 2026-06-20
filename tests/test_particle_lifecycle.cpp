/**
 * @file test_particle_lifecycle.cpp
 * @brief regression test for particle insertion, deletion, and renumbering
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <vorflow/voronoi.hpp>

using std::array;
using std::sort;
using std::vector;
using vor::Box;
using vor::CellComplex;
using vor::GeometryView;
using vor::ParticleRenumberResult;
using vor::uint1;
using vor::uint2;

namespace {

template <typename real_t>
void collectSortedNbrsByParticle(const CellComplex<real_t>& complex, uint2 particleId,
                                 vector<uint2>& nbrs) {
  nbrs.clear();
  const uint2 cellIndex = complex.getCellIndexForParticle(particleId);
  if (cellIndex == vor::noNbr)
    return;
  const vor::CellView<real_t> cell = complex.getCellView(cellIndex);
  nbrs.reserve(cell.numFacets());
  for (uint1 facet = 0; facet < cell.numFacets(); ++facet) {
    const uint2 nbrId = cell.getNbr(facet);
    if (nbrId != vor::noNbr)
      nbrs.push_back(nbrId);
  }
  sort(nbrs.begin(), nbrs.end());
}

template <typename real_t>
bool compareByParticleId(const CellComplex<real_t>& a, const CellComplex<real_t>& b,
                         const vector<uint2>& particleIds, real_t tol, const char* label) {
  vector<uint2> nbrsA;
  vector<uint2> nbrsB;
  for (size_t i = 0; i < particleIds.size(); ++i) {
    const uint2 particleId = particleIds[i];
    const uint2 cellA = a.getCellIndexForParticle(particleId);
    const uint2 cellB = b.getCellIndexForParticle(particleId);
    if (cellA == vor::noNbr || cellB == vor::noNbr) {
      std::fprintf(stderr, "%s: missing cell for particle %u\n", label,
                   static_cast<unsigned>(particleId));
      return false;
    }

    collectSortedNbrsByParticle(a, particleId, nbrsA);
    collectSortedNbrsByParticle(b, particleId, nbrsB);
    if (nbrsA != nbrsB) {
      std::fprintf(stderr, "%s: neighbour mismatch for particle %u\n", label,
                   static_cast<unsigned>(particleId));
      return false;
    }

    const GeometryView<real_t> geomA = a.getGeometryView(cellA);
    const GeometryView<real_t> geomB = b.getGeometryView(cellB);
    if (std::abs(geomA.getVolume() - geomB.getVolume()) > tol) {
      std::fprintf(stderr, "%s: volume mismatch for particle %u (%.16e vs %.16e)\n", label,
                   static_cast<unsigned>(particleId), static_cast<double>(geomA.getVolume()),
                   static_cast<double>(geomB.getVolume()));
      return false;
    }
  }
  return true;
}

template <typename real_t>
real_t sumVolumes(const CellComplex<real_t>& complex) {
  real_t sum = real_t(0);
  for (size_t i = 0; i < complex.numCells(); ++i)
    sum += complex.getGeometryView(i).getVolume();
  return sum;
}

}  // namespace

int main() {
  typedef double real_t;

  std::mt19937_64 rng(7);
  std::uniform_real_distribution<real_t> uni(real_t(0), real_t(1));

  Box<real_t> box(array<real_t, 3>{1.0, 1.0, 1.0});
  CellComplex<real_t> complex(&box);
  CellComplex<real_t> sparseReference(&box);
  CellComplex<real_t> denseReference(&box);

  vector<array<real_t, 3> > pos(256);
  for (size_t i = 0; i < pos.size(); ++i)
    for (uint1 k = 0; k < 3; ++k)
      pos[i][k] = uni(rng);

  complex.build(pos);

  const size_t oldSize = pos.size();
  vector<array<real_t, 3> > inserted(12);
  for (size_t i = 0; i < inserted.size(); ++i)
    for (uint1 k = 0; k < 3; ++k)
      inserted[i][k] = uni(rng);

  complex.insertParticles(pos, inserted);
  if (pos.size() != oldSize + inserted.size()) {
    std::fprintf(stderr, "insertParticles did not append new particles\n");
    return 1;
  }
  for (size_t i = 0; i < inserted.size(); ++i) {
    if (pos[oldSize + i] != inserted[i]) {
      std::fprintf(stderr, "insertParticles did not preserve appended particle order\n");
      return 1;
    }
  }

  vector<uint8_t> active(pos.size(), 1u);
  const vector<uint2> deletedIds = {3u, 17u, 42u, static_cast<uint2>(oldSize + 1u),
                                    static_cast<uint2>(oldSize + 5u)};
  complex.deactivateParticles(deletedIds);
  for (size_t i = 0; i < deletedIds.size(); ++i)
    active[deletedIds[i]] = 0u;

  const real_t displacement = 0.015;
  for (size_t i = 0; i < pos.size(); ++i) {
    if (!active[i])
      continue;
    for (uint1 k = 0; k < 3; ++k)
      pos[i][k] += displacement * (uni(rng) - real_t(0.5));
  }
  box.putInBox(pos);

  complex.update(pos);
  sparseReference.build(pos, active);

  vector<uint2> activeIds;
  for (size_t i = 0; i < active.size(); ++i)
    if (active[i] != 0u)
      activeIds.push_back(static_cast<uint2>(i));

  if (complex.numCells() != activeIds.size()) {
    std::fprintf(stderr, "active cell count mismatch: got %zu expected %zu\n", complex.numCells(),
                 activeIds.size());
    return 1;
  }
  for (size_t i = 0; i < deletedIds.size(); ++i) {
    if (complex.getCellIndexForParticle(deletedIds[i]) != vor::noNbr) {
      std::fprintf(stderr, "deleted particle %u is still present in the tessellation\n",
                   static_cast<unsigned>(deletedIds[i]));
      return 1;
    }
  }

  const real_t tol = 1.0e-10;
  if (!compareByParticleId(complex, sparseReference, activeIds, tol, "sparse compare"))
    return 1;
  if (std::abs(sumVolumes(complex) - real_t(1)) > tol) {
    std::fprintf(stderr, "sparse update volume conservation failed\n");
    return 1;
  }

  const ParticleRenumberResult map = complex.renumberParticles(pos);
  if (pos.size() != activeIds.size()) {
    std::fprintf(stderr, "renumberParticles produced %zu particles, expected %zu\n", pos.size(),
                 activeIds.size());
    return 1;
  }
  for (size_t i = 0; i < deletedIds.size(); ++i) {
    if (map.old_to_new[deletedIds[i]] != vor::noNbr) {
      std::fprintf(stderr, "deleted particle %u was not dropped during renumbering\n",
                   static_cast<unsigned>(deletedIds[i]));
      return 1;
    }
  }
  for (size_t newId = 0; newId < map.new_to_old.size(); ++newId) {
    if (map.old_to_new[map.new_to_old[newId]] != newId) {
      std::fprintf(stderr, "renumber mapping is inconsistent at new id %zu\n", newId);
      return 1;
    }
  }
  if (complex.getParticleActivity().size() != pos.size()) {
    std::fprintf(stderr, "particle activity size mismatch after renumbering\n");
    return 1;
  }
  for (size_t i = 0; i < complex.getParticleActivity().size(); ++i) {
    if (!complex.getParticleActivity()[i]) {
      std::fprintf(stderr, "inactive particle remained after renumbering\n");
      return 1;
    }
  }

  denseReference.build(pos);
  vector<uint2> denseIds(pos.size());
  for (size_t i = 0; i < denseIds.size(); ++i)
    denseIds[i] = static_cast<uint2>(i);

  if (!compareByParticleId(complex, denseReference, denseIds, tol, "dense compare"))
    return 1;
  if (std::abs(sumVolumes(complex) - real_t(1)) > tol) {
    std::fprintf(stderr, "dense rebuild volume conservation failed after renumbering\n");
    return 1;
  }

  std::printf("particle lifecycle regression passed (%zu active particles after renumbering)\n",
              pos.size());
  return 0;
}
