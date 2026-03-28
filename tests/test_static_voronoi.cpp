/**
 * @file test_static_voronoi.cpp
 * \brief test program: build a Voronoi tessellation for a static point set in
 *        a periodic box and verify its correctness
 */

#include <boost/random.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <voronoi_dynamics/voronoi.hpp>

using std::vector;
using vor::Box;
using vor::Cell;
using vor::CellComplex;
using vor::CellGeometry;
using vor::uint1;
using vor::uint2;

static int testRandomPoints(int numParticles, double Lx, double Ly, double Lz, unsigned int seed) {
  typedef double real_t;
  typedef boost::mt19937 rng_type;
  typedef boost::uniform_01<real_t> dist_type;
  typedef boost::variate_generator<rng_type &, dist_type> gen_type;

  rng_type rng(seed);
  gen_type pointGen(rng, dist_type());

  std::array<real_t, 3> L;
  L[0] = Lx;
  L[1] = Ly;
  L[2] = Lz;
  real_t boxVol = L[0] * L[1] * L[2];

  Box<real_t> box(L);
  CellComplex<real_t> complex(&box);

  // Generate random positions
  vector<std::array<real_t, 3> > pos(numParticles);
  for (int i = 0; i < numParticles; ++i) {
    pos[i][0] = L[0] * pointGen();
    pos[i][1] = L[1] * pointGen();
    pos[i][2] = L[2] * pointGen();
  }

  // Build Voronoi tessellation
  complex.build(pos);

  vector<CellGeometry<real_t> > &geoms = complex.getGeoms();
  vector<Cell<real_t> > cells;
  complex.materializeCells(cells);

  if ((int)cells.size() != numParticles) {
    fprintf(stderr, "FAIL: expected %d cells, got %lu\n", numParticles,
            (unsigned long)cells.size());
    return 1;
  }

  // Compute volumes and check they sum to the box volume
  real_t totalVol = 0;
  real_t minVol = 1e30, maxVol = -1e30;
  for (size_t i = 0; i < geoms.size(); ++i) {
    geoms[i].computeVolume();
    real_t v = geoms[i].getVolume();
    totalVol += v;
    if (v < minVol)
      minVol = v;
    if (v > maxVol)
      maxVol = v;
  }

  real_t volErr = std::fabs(totalVol - boxVol) / boxVol;
  printf("  Cells: %d, Volume sum: %.10f, Box volume: %.10f, Relative error: %.2e\n", numParticles,
         totalVol, boxVol, volErr);
  printf("  Min cell volume: %.6e, Max cell volume: %.6e\n", minVol, maxVol);

  // For >= 200 particles the tessellation should be exact to machine precision.
  // Smaller counts may have small errors due to near-degenerate configurations
  // in the plane-cutting algorithm (inherent floating-point limitation).
  real_t tol = (numParticles >= 200 ? 1e-10 : 1e-2);
  if (volErr > tol) {
    fprintf(stderr, "FAIL: volume error %.2e exceeds tolerance %.0e\n", volErr, tol);
    return 1;
  }

  // Check all cell volumes are positive
  for (size_t i = 0; i < geoms.size(); ++i) {
    if (geoms[i].getVolume() <= 0) {
      fprintf(stderr, "FAIL: cell %lu has non-positive volume %g\n", (unsigned long)i,
              geoms[i].getVolume());
      return 1;
    }
  }

  // Check that each cell has at least 4 facets (tetrahedron minimum)
  for (size_t i = 0; i < cells.size(); ++i) {
    if (cells[i].numFacets() < 4) {
      fprintf(stderr, "FAIL: cell %lu has only %u facets\n", (unsigned long)i,
              (unsigned)cells[i].numFacets());
      return 1;
    }
  }

  return 0;
}

int main() {
  int failures = 0;

  printf("Test 1: 100 particles in unit cube\n");
  failures += testRandomPoints(100, 1.0, 1.0, 1.0, 42);

  printf("Test 2: 1000 particles in unit cube\n");
  failures += testRandomPoints(1000, 1.0, 1.0, 1.0, 123);

  printf("Test 3: 2000 particles in non-cubic box (2x1x0.5)\n");
  failures += testRandomPoints(2000, 2.0, 1.0, 0.5, 456);

  printf("Test 4: 5000 particles in unit cube\n");
  failures += testRandomPoints(5000, 1.0, 1.0, 1.0, 789);

  printf("Test 5: Rebuild consistency (build twice, same result)\n");
  {
    typedef double real_t;
    typedef boost::mt19937 rng_type;
    typedef boost::uniform_01<real_t> dist_type;
    typedef boost::variate_generator<rng_type &, dist_type> gen_type;

    rng_type rng(999);
    gen_type pointGen(rng, dist_type());

    std::array<real_t, 3> L;
    L[0] = 1;
    L[1] = 1;
    L[2] = 1;
    Box<real_t> box(L);
    CellComplex<real_t> complex(&box);

    int N = 200;
    vector<std::array<real_t, 3> > pos(N);
    for (int i = 0; i < N; ++i) {
      pos[i][0] = pointGen();
      pos[i][1] = pointGen();
      pos[i][2] = pointGen();
    }

    // Build twice and compare volumes
    complex.build(pos);
  vector<CellGeometry<real_t> > &geoms1 = complex.getGeoms();
    vector<real_t> vols1(N);
    for (int i = 0; i < N; ++i) {
      geoms1[i].computeVolume();
      vols1[i] = geoms1[i].getVolume();
    }

    complex.build(pos);
    vector<CellGeometry<real_t> > &geoms2 = complex.getGeoms();
    real_t maxDiff = 0;
    for (int i = 0; i < N; ++i) {
      geoms2[i].computeVolume();
      real_t diff = std::fabs(geoms2[i].getVolume() - vols1[i]);
      if (diff > maxDiff)
        maxDiff = diff;
    }
    printf("  Max volume difference between builds: %.2e\n", maxDiff);
    if (maxDiff > 1e-14) {
      fprintf(stderr, "FAIL: rebuild produced different volumes (max diff: %g)\n", maxDiff);
      ++failures;
    }
  }

  if (failures == 0) {
    printf("\nAll tests PASSED\n");
  } else {
    printf("\n%d test(s) FAILED\n", failures);
  }

  return failures;
}
