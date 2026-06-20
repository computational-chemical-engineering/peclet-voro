/**
 * @file test_golden.cpp
 * \brief Golden-master capture for the migration (suite §5.6 / Phase 0).
 *
 * Builds the legacy tessellation on a frozen seed set (data/pos_eq_10000.dat) and
 * compares order-independent geometric/topological summaries against a committed
 * reference (tests/golden/pos_eq_10000.golden). Because exact half-edge topology
 * may legitimately differ between host/device or between ranks, the reference is
 * a set of order-invariant aggregates (total/spread of volumes, facet-count
 * histogram), not a facet-for-facet dump. The ported device kernels are diffed
 * against this same reference in later phases.
 *
 * If the golden file is absent it is written and the test passes with a notice
 * (first-run capture); commit the file to freeze the reference.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <vorflow/voronoi.hpp>

#ifndef VORONOI_GOLDEN_DIR
#define VORONOI_GOLDEN_DIR "."
#endif

using std::vector;
using vor::Box;
using vor::Cell;
using vor::CellComplex;
using vor::CellGeometry;

namespace {

typedef double real_t;
typedef std::array<real_t, 3> Vec3;

const int kMinF = 4;
const int kMaxF = 40;  // histogram buckets [kMinF, kMaxF]; overflow into last bucket

struct Summary {
  long nCells = 0;
  double totalVol = 0;
  double sumVolSq = 0;
  double minVol = 0;
  double maxVol = 0;
  long totalFacets = 0;
  std::vector<long> facetHist;  // size kMaxF-kMinF+1
};

bool loadPositions(const std::string& path, vector<Vec3>& pos) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f)
    return false;
  Vec3 p;
  int id;
  pos.clear();
  while (fscanf(f, "%d %lf %lf %lf", &id, &p[0], &p[1], &p[2]) == 4)
    pos.push_back(p);
  fclose(f);
  return !pos.empty();
}

Summary compute(const vector<Vec3>& pos) {
  Vec3 L = {1.0, 1.0, 1.0};
  Box<real_t> box(L);
  CellComplex<real_t> complex(&box);
  vector<Vec3> p = pos;
  box.putInBox(p);
  complex.build(p);

  vector<Cell<real_t> > cells;
  complex.materializeCells(cells);
  const vector<CellGeometry<real_t> >& geom = complex.getGeoms();

  Summary s;
  s.nCells = (long)cells.size();
  s.facetHist.assign(kMaxF - kMinF + 1, 0);
  s.minVol = 1e300;
  s.maxVol = -1e300;
  for (size_t i = 0; i < cells.size(); ++i) {
    double v = geom[i].getVolume();
    s.totalVol += v;
    s.sumVolSq += v * v;
    if (v < s.minVol)
      s.minVol = v;
    if (v > s.maxVol)
      s.maxVol = v;
    int nf = cells[i].numFacets();
    s.totalFacets += nf;
    int b = nf < kMinF ? 0 : (nf > kMaxF ? kMaxF - kMinF : nf - kMinF);
    s.facetHist[b]++;
  }
  return s;
}

bool readGolden(const std::string& path, Summary& g) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f)
    return false;
  g.facetHist.assign(kMaxF - kMinF + 1, 0);
  int ok = fscanf(f, "%ld %lf %lf %lf %lf %ld", &g.nCells, &g.totalVol, &g.sumVolSq, &g.minVol,
                  &g.maxVol, &g.totalFacets);
  for (size_t i = 0; i < g.facetHist.size(); ++i)
    ok += fscanf(f, "%ld", &g.facetHist[i]);
  fclose(f);
  return ok == 6 + (int)g.facetHist.size();
}

void writeGolden(const std::string& path, const Summary& s) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) {
    fprintf(stderr, "  cannot write golden %s\n", path.c_str());
    return;
  }
  fprintf(f, "%ld %.17g %.17g %.17g %.17g %ld\n", s.nCells, s.totalVol, s.sumVolSq, s.minVol,
          s.maxVol, s.totalFacets);
  for (size_t i = 0; i < s.facetHist.size(); ++i)
    fprintf(f, "%ld ", s.facetHist[i]);
  fprintf(f, "\n");
  fclose(f);
}

int compare(const Summary& a, const Summary& g) {
  int fail = 0;
  if (a.nCells != g.nCells) {
    fprintf(stderr, "  FAIL nCells %ld != golden %ld\n", a.nCells, g.nCells);
    ++fail;
  }
  // Volumes are exact to ~1e-12 relative on the same build; allow slack so a
  // valid device topology with different near-degenerate decisions still passes.
  double volTol = 1e-9;
  auto rel = [](double x, double y) { return std::fabs(x - y) / (std::fabs(y) + 1e-300); };
  if (rel(a.totalVol, g.totalVol) > volTol) {
    fprintf(stderr, "  FAIL totalVol %.12g vs %.12g\n", a.totalVol, g.totalVol);
    ++fail;
  }
  if (rel(a.sumVolSq, g.sumVolSq) > 1e-6) {
    fprintf(stderr, "  FAIL sumVolSq %.12g vs %.12g\n", a.sumVolSq, g.sumVolSq);
    ++fail;
  }
  // Facet histogram: total facet count must match closely; per-bucket allowed a
  // small drift (a handful of cells may pick a different valid topology).
  if (std::labs(a.totalFacets - g.totalFacets) > g.totalFacets / 1000 + 4) {
    fprintf(stderr, "  FAIL totalFacets %ld vs %ld\n", a.totalFacets, g.totalFacets);
    ++fail;
  }
  long histDrift = 0;
  for (size_t i = 0; i < g.facetHist.size() && i < a.facetHist.size(); ++i)
    histDrift += std::labs(a.facetHist[i] - g.facetHist[i]);
  if (histDrift > a.nCells / 200 + 8) {
    fprintf(stderr, "  FAIL facet histogram drift %ld (cells %ld)\n", histDrift, a.nCells);
    ++fail;
  }
  return fail;
}

}  // namespace

int main() {
  const std::string data = "pos_eq_10000.dat";  // cwd is data/ under ctest
  const std::string golden = std::string(VORONOI_GOLDEN_DIR) + "/pos_eq_10000.golden";

  vector<Vec3> pos;
  if (!loadPositions(data, pos)) {
    fprintf(stderr, "FAIL: cannot read %s\n", data.c_str());
    return 1;
  }
  Summary s = compute(pos);
  printf("Golden: nCells=%ld totalVol=%.12g meanFacets=%.3f\n", s.nCells, s.totalVol,
         (double)s.totalFacets / s.nCells);

  Summary g;
  if (!readGolden(golden, g)) {
    printf("  golden absent -> writing %s (first-run capture)\n", golden.c_str());
    writeGolden(golden, s);
    return 0;
  }
  int fail = compare(s, g);
  printf("%s\n", fail == 0 ? "Golden master MATCH" : "Golden master MISMATCH");
  return fail;
}
