/**
 * @file test_tessellation_view.cpp
 * \brief Phase-1 acceptance: legacy <-> HostTessellation <-> device round-trip.
 *
 * Validates the published data layer (tessellation_view.hpp):
 *   A. HostTessellation reproduces the legacy per-cell / per-facet values
 *      (volume, neighbour id, area vector) exactly.
 *   B. upload()/mirror round-trip is bit-exact device<->host.
 *   C. Invariants hold *through the view*: space-filling and per-cell area
 *      closure are evaluated on the device via TessellationView accessors;
 *      facet reciprocity is checked on the host CSR.
 *
 * No Boost (uses std::mt19937); no physics. Exit non-zero on any failure.
 */

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <vector>

#include "vorflow/tessellation_view.hpp"
#include "vorflow/voronoi.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {

int checkHostReproducesLegacy(const vor::HostTessellation<real_t>& t,
                              const std::vector<vor::Cell<real_t> >& cells,
                              const std::vector<vor::CellGeometry<real_t> >& geom) {
  int fail = 0;
  if (t.nCells != (int)cells.size()) {
    std::fprintf(stderr, "FAIL nCells %d != %zu\n", t.nCells, cells.size());
    return 1;
  }
  for (int i = 0; i < t.nCells && fail < 10; ++i) {
    if (std::fabs(t.cellVolume[i] - geom[i].getVolume()) > 1e-15) {
      std::fprintf(stderr, "FAIL volume cell %d: %.17g vs %.17g\n", i, t.cellVolume[i],
                   geom[i].getVolume());
      ++fail;
    }
    const int nf = cells[i].numFacets();
    if (t.cellFacetOffset[i + 1] - t.cellFacetOffset[i] != nf) {
      std::fprintf(stderr, "FAIL facet count cell %d\n", i);
      ++fail;
      continue;
    }
    const int base = t.cellFacetOffset[i];
    const std::vector<Vec3>& areas = geom[i].getAreas();
    for (int f = 0; f < nf; ++f) {
      if (t.facetNeighbor[base + f] != vor::toGid(cells[i].getNbr((vor::uint1)f))) {
        std::fprintf(stderr, "FAIL neighbour cell %d facet %d\n", i, f);
        ++fail;
      }
      for (int c = 0; c < 3; ++c)
        if (std::fabs(t.facetArea[3 * (base + f) + c] - areas[f][c]) > 1e-15) {
          std::fprintf(stderr, "FAIL area cell %d facet %d comp %d\n", i, f, c);
          ++fail;
        }
    }
  }
  return fail;
}

// Facet reciprocity on the host CSR: every non-boundary facet i->seed has a
// reciprocal facet seed->id_i (skipping periodic self/multi-image ambiguities).
int checkReciprocity(const vor::HostTessellation<real_t>& t) {
  std::map<vor::gid_t, int> seedToCell;
  for (int i = 0; i < t.nCells; ++i)
    seedToCell[t.cellSeedId[i]] = i;
  int fail = 0;
  for (int i = 0; i < t.nCells && fail < 10; ++i) {
    const vor::gid_t idi = t.cellSeedId[i];
    for (int g = t.cellFacetOffset[i]; g < t.cellFacetOffset[i + 1]; ++g) {
      const vor::gid_t nbr = t.facetNeighbor[g];
      if (nbr < 0 || nbr == idi)
        continue;  // boundary or self-image
      auto it = seedToCell.find(nbr);
      if (it == seedToCell.end()) {
        std::fprintf(stderr, "FAIL reciprocity: seed %d has no cell\n", (int)nbr);
        ++fail;
        continue;
      }
      const int j = it->second;
      bool back = false;
      for (int h = t.cellFacetOffset[j]; h < t.cellFacetOffset[j + 1]; ++h)
        if (t.facetNeighbor[h] == idi) {
          back = true;
          break;
        }
      if (!back) {
        std::fprintf(stderr, "FAIL reciprocity: %d->%d has no reverse\n", (int)idi, (int)nbr);
        ++fail;
      }
    }
  }
  return fail;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const int N = 1500;
    const Vec3 L = {1.0, 1.0, 1.0};
    const real_t boxVol = L[0] * L[1] * L[2];

    std::mt19937 rng(2024);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<Vec3> pos(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);

    vor::Box<real_t> box(L);
    vor::CellComplex<real_t> complex(&box);
    complex.build(pos);

    std::vector<vor::Cell<real_t> > cells;
    complex.materializeCells(cells);
    const std::vector<vor::CellGeometry<real_t> >& geom = complex.getGeoms();

    // (A) host CSR reproduces legacy.
    vor::HostTessellation<real_t> host = vor::buildHostTessellation(complex);
    int a = checkHostReproducesLegacy(host, cells, geom);
    failures += a;
    failures += checkReciprocity(host);

    // (B) upload + mirror back == host (bit-exact).
    vor::TessellationView<real_t> view = vor::upload(host);
    auto volH = Kokkos::create_mirror_view(view.cellVolume);
    Kokkos::deep_copy(volH, view.cellVolume);
    auto areaH = Kokkos::create_mirror_view(view.facetArea);
    Kokkos::deep_copy(areaH, view.facetArea);
    int rt = 0;
    for (int i = 0; i < host.nCells; ++i)
      if (volH(i) != host.cellVolume[i])
        ++rt;
    for (int g = 0; g < 3 * host.nFacets; ++g)
      if (areaH(g) != host.facetArea[g])
        ++rt;
    if (rt) {
      std::fprintf(stderr, "FAIL round-trip: %d device/host mismatches\n", rt);
      ++failures;
    }

    // (C) invariants through the device view.
    // Space-filling: sum of volumes over the published view.
    double totalVol = 0;
    Kokkos::parallel_reduce(
        "tess.spacefill", Kokkos::RangePolicy<tpx::ExecSpace>(0, view.numCells()),
        KOKKOS_LAMBDA(const int i, double& acc) { acc += view.volume(i); }, totalVol);
    if (std::fabs(totalVol - boxVol) / boxVol > 1e-10) {
      std::fprintf(stderr, "FAIL device space-filling: %.12g vs %.12g\n", totalVol, boxVol);
      ++failures;
    }
    // Area closure: max over cells of |sum_f area(f)| / mean|area(f)|.
    double maxClosure = 0;
    Kokkos::parallel_reduce(
        "tess.closure", Kokkos::RangePolicy<tpx::ExecSpace>(0, view.numCells()),
        KOKKOS_LAMBDA(const int i, double& worst) {
          double sx = 0, sy = 0, sz = 0, mag = 0;
          int nf = 0;
          for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f) {
            sx += view.area(f, 0);
            sy += view.area(f, 1);
            sz += view.area(f, 2);
            mag +=
                Kokkos::sqrt(view.area(f, 0) * view.area(f, 0) + view.area(f, 1) * view.area(f, 1) +
                             view.area(f, 2) * view.area(f, 2));
            ++nf;
          }
          if (mag > 0) {
            double rel = Kokkos::sqrt(sx * sx + sy * sy + sz * sz) / (mag / nf);
            if (rel > worst)
              worst = rel;
          }
        },
        Kokkos::Max<double>(maxClosure));
    if (maxClosure > 1e-9) {
      std::fprintf(stderr, "FAIL device area-closure: max rel %.3e\n", maxClosure);
      ++failures;
    }

    std::printf("tessellation_view: N=%d cells=%d facets=%d totalVol=%.12g closure=%.2e\n", N,
                view.numCells(), view.numFacets(), totalVol, maxClosure);
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "tessellation_view PASS" : "tessellation_view FAIL");
  return failures;
}
