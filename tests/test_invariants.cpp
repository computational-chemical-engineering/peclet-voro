/**
 * @file test_invariants.cpp
 * \brief Geometric-invariant oracle for the tessellation (suite migration §5).
 *
 * These invariants are the cross-platform / cross-rank truth used throughout the
 * Kokkos+MPI migration: exact half-edge topology may legitimately differ between
 * host/device or between ranks (near-degenerate sign flips yield different-but-valid
 * cells), but every valid tessellation must satisfy them. The harness runs against
 * the legacy CPU build here (Phase 0), establishing the oracle that the ported
 * device kernels are later validated against.
 *
 * Checks (cf. docs/update_to_kokkos_plan.md §5):
 *   1. Space-filling     : Sum of cell volumes == box volume (periodic).
 *   2. Positive volumes  : every cell volume > 0.
 *   3. Min facets        : every cell has >= 4 facets (tetrahedron minimum).
 *   4. Facet reciprocity : facet i->j exists  =>  facet j->i exists.
 *   5. Area negation     : area_vec(i->j) == -area_vec(j->i).
 *   6. Area closure      : Sum of facet area vectors over a cell == 0.
 *   7. Euler (3-valence) : general-position Voronoi vertices are 3-valent, so
 *                          V - E + F = 2 reduces to 2F - V = 4.
 */

#include <array>
#include <boost/random.hpp>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>
#include <vorflow/voronoi.hpp>

using std::vector;
using vor::boundaryNbr;
using vor::Box;
using vor::Cell;
using vor::CellComplex;
using vor::CellGeometry;
using vor::noNbr;
using vor::uint1;
using vor::uint2;

namespace {

typedef double real_t;
typedef std::array<real_t, 3> Vec3;

real_t norm(const Vec3& a) {
  return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

// Locate the unique facet of cell `c` whose neighbour id equals `target`.
// Returns the facet index, or -1 if there is no facet or more than one (periodic
// self/multi-image facets are ambiguous and are skipped by the caller).
int uniqueFacetToward(const Cell<real_t>& c, uint2 target) {
  int found = -1;
  for (uint1 f = 0; f < c.numFacets(); ++f) {
    if (c.getNbr(f) == target) {
      if (found >= 0)
        return -1;  // ambiguous (multiple images)
      found = f;
    }
  }
  return found;
}

struct Tol {
  real_t volSum;     // relative, summed-volume vs box
  real_t areaNeg;    // relative, |A_ij + A_ji| / |A_ij|
  real_t areaClose;  // relative, |sum A_f| / mean|A_f|
};

// Run all invariants on a freshly built complex. Returns number of failures.
int checkInvariants(const char* label, int numParticles, Vec3 L, unsigned seed, bool checkEuler,
                    const Tol& tol) {
  typedef boost::mt19937 rng_t;
  typedef boost::uniform_01<real_t> dist_t;
  typedef boost::variate_generator<rng_t&, dist_t> gen_t;
  rng_t rng(seed);
  gen_t gen(rng, dist_t());

  const real_t boxVol = L[0] * L[1] * L[2];
  Box<real_t> box(L);
  CellComplex<real_t> complex(&box);

  vector<Vec3> pos(numParticles);
  for (int i = 0; i < numParticles; ++i)
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * gen();

  complex.build(pos);  // computeGeometry=true -> volumes/areas populated

  vector<Cell<real_t> > cells;
  complex.materializeCells(cells);
  const vector<CellGeometry<real_t> >& geom = complex.getGeoms();

  const size_t n = cells.size();
  int failures = 0;
  if (n != (size_t)numParticles || geom.size() != n) {
    fprintf(stderr, "  [%s] FAIL: cell count %zu / geom %zu != %d\n", label, n, geom.size(),
            numParticles);
    return 1;
  }

  // id -> cell-index map (dense build: id == index, but do not assume it).
  std::map<uint2, size_t> idToCell;
  for (size_t i = 0; i < n; ++i)
    idToCell[cells[i].getID()] = i;

  // (1) space-filling + (2) positive volumes.
  real_t totalVol = 0;
  int nonPositive = 0;
  for (size_t i = 0; i < n; ++i) {
    real_t v = geom[i].getVolume();
    totalVol += v;
    if (v <= 0)
      ++nonPositive;
  }
  real_t volErr = std::fabs(totalVol - boxVol) / boxVol;
  if (volErr > tol.volSum) {
    fprintf(stderr, "  [%s] FAIL space-filling: rel err %.2e > %.0e\n", label, volErr, tol.volSum);
    ++failures;
  }
  if (nonPositive) {
    fprintf(stderr, "  [%s] FAIL positive-volume: %d non-positive cells\n", label, nonPositive);
    ++failures;
  }

  // (3) min facets, (6) area closure, (7) Euler.
  int minFacetFail = 0, closureFail = 0, eulerFail = 0;
  for (size_t i = 0; i < n; ++i) {
    const Cell<real_t>& c = cells[i];
    if (c.numFacets() < 4)
      ++minFacetFail;

    const vector<Vec3>& areas = geom[i].getAreas();
    Vec3 sum = {0, 0, 0};
    real_t meanMag = 0;
    for (size_t f = 0; f < areas.size(); ++f) {
      for (int d = 0; d < 3; ++d)
        sum[d] += areas[f][d];
      meanMag += norm(areas[f]);
    }
    if (!areas.empty())
      meanMag /= areas.size();
    if (meanMag > 0 && norm(sum) / meanMag > tol.areaClose)
      ++closureFail;

    if (checkEuler) {
      const int V = c.numVertices();
      const int F = c.numFacets();
      if (2 * F - V != 4)
        ++eulerFail;
    }
  }
  if (minFacetFail) {
    fprintf(stderr, "  [%s] FAIL min-facets: %d cells with <4 facets\n", label, minFacetFail);
    ++failures;
  }
  if (closureFail) {
    fprintf(stderr, "  [%s] FAIL area-closure: %d cells exceed %.0e\n", label, closureFail,
            tol.areaClose);
    ++failures;
  }
  if (eulerFail) {
    fprintf(stderr, "  [%s] FAIL euler(2F-V=4): %d cells violate\n", label, eulerFail);
    ++failures;
  }

  // (4) reciprocity + (5) area negation.
  int recipFail = 0, negFail = 0;
  for (size_t i = 0; i < n; ++i) {
    const Cell<real_t>& ci = cells[i];
    const uint2 idi = ci.getID();
    const vector<Vec3>& areasI = geom[i].getAreas();
    for (uint1 f = 0; f < ci.numFacets(); ++f) {
      const uint2 nbr = ci.getNbr(f);
      if (nbr == noNbr || nbr == boundaryNbr)
        continue;  // boundary facet
      if (nbr == idi)
        continue;  // periodic self-image: skip
      std::map<uint2, size_t>::iterator it = idToCell.find(nbr);
      if (it == idToCell.end()) {
        ++recipFail;  // neighbour id has no cell
        continue;
      }
      const size_t j = it->second;
      const Cell<real_t>& cj = cells[j];
      const int gf = uniqueFacetToward(cj, idi);
      if (gf < 0) {
        // No reciprocal facet at all is a hard failure; ambiguous (multi-image)
        // is skipped for the *negation* test but reciprocity still holds if any
        // facet toward idi exists.
        bool any = false;
        for (uint1 g = 0; g < cj.numFacets(); ++g)
          if (cj.getNbr(g) == idi) {
            any = true;
            break;
          }
        if (!any)
          ++recipFail;
        continue;
      }
      // (5) area negation, only when both directions are unambiguous.
      if (uniqueFacetToward(ci, nbr) != (int)f)
        continue;  // i->j ambiguous, skip
      const Vec3& aij = areasI[f];
      const Vec3& aji = geom[j].getAreas()[gf];
      Vec3 s = {aij[0] + aji[0], aij[1] + aji[1], aij[2] + aji[2]};
      real_t mag = norm(aij);
      if (mag > 0 && norm(s) / mag > tol.areaNeg)
        ++negFail;
    }
  }
  if (recipFail) {
    fprintf(stderr, "  [%s] FAIL reciprocity: %d unmatched facets\n", label, recipFail);
    ++failures;
  }
  if (negFail) {
    fprintf(stderr, "  [%s] FAIL area-negation: %d facets exceed %.0e\n", label, negFail,
            tol.areaNeg);
    ++failures;
  }

  printf("  [%s] N=%d  volErr=%.2e  facets/cell-ok  %s\n", label, numParticles, volErr,
         failures == 0 ? "PASS" : "FAIL");
  return failures;
}

}  // namespace

int main() {
  int failures = 0;
  // Tolerances: volume sum is exact to ~1e-15 at N>=200; area identities are
  // computed from the same vertex set so they close tightly.
  const Tol tol = {1e-10, 1e-9, 1e-9};

  printf("Invariant oracle (legacy CPU build):\n");
  failures += checkInvariants("uniform-1000", 1000, {1.0, 1.0, 1.0}, 123, true, tol);
  failures += checkInvariants("uniform-2000-noncubic", 2000, {2.0, 1.0, 0.5}, 456, true, tol);
  failures += checkInvariants("uniform-5000", 5000, {1.0, 1.0, 1.0}, 789, true, tol);
  failures += checkInvariants("uniform-500", 500, {1.0, 1.0, 1.0}, 2024, true, tol);

  if (failures == 0)
    printf("\nAll invariant checks PASSED\n");
  else
    printf("\n%d invariant check(s) FAILED\n", failures);
  return failures;
}
