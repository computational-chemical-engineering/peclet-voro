/**
 * @file test_device_geometry.cpp
 * \brief The device tessellator publishes the full per-facet physics geometry.
 *
 * Validates that buildTessellation now fills facetArea, facetConnect (dV) and
 * facetConnVec (the connecting vector) — ported from CellGeometry::computeEdgeInv
 * + diffVolume into the device cutter — against the legacy CellGeometry, matched
 * per (cell, neighbour). This is what makes the physics (pressure, viscous, ...)
 * runnable on the DEVICE-built tessellation, with no legacy/host geometry.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/voronoi.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const int N = 1500;
    const Vec3 L = {1.0, 1.0, 1.0};
    std::mt19937 rng(7);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<Vec3> pos(N);
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        pos[i][d] = L[d] * U(rng);

    // Legacy geometry, keyed by (cell id, neighbour id).
    vor::Box<real_t> box(L);
    vor::CellComplex<real_t> legacy(&box);
    legacy.build(pos);
    std::vector<vor::Cell<real_t> > lc;
    legacy.materializeCells(lc);
    const std::vector<vor::CellGeometry<real_t> >& lg = legacy.getGeoms();
    std::map<std::pair<int, int>, std::array<Vec3, 3> > ref;  // (i,nbr) -> {dV, area, conn}
    for (size_t i = 0; i < lc.size(); ++i) {
      int id = (int)lc[i].getID();
      for (vor::uint1 j = 0; j < lc[i].numFacets(); ++j)
        ref[{id, (int)lc[i].getNbr(j)}] = {lg[i].getdV()[j], lg[i].getAreas()[j],
                                           lg[i].getConnVect()[j]};
    }

    // Device tessellation + its published geometry.
    Kokkos::View<real_t*, tpx::MemSpace> dPos(
        Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)N * 3);
    {
      auto h = Kokkos::create_mirror_view(dPos);
      for (int i = 0; i < N; ++i)
        for (int k = 0; k < 3; ++k)
          h(3 * i + k) = pos[i][k];
      Kokkos::deep_copy(dPos, h);
    }
    Kokkos::View<real_t*, tpx::MemSpace> dW("w", N);
    const real_t Larr[3] = {L[0], L[1], L[2]};
    auto res = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr);
    auto off = Kokkos::create_mirror_view(res.view.cellFacetOffset);
    auto nbr = Kokkos::create_mirror_view(res.view.facetNeighbor);
    auto area = Kokkos::create_mirror_view(res.view.facetArea);
    auto dv = Kokkos::create_mirror_view(res.view.facetConnect);
    auto conn = Kokkos::create_mirror_view(res.view.facetConnVec);
    Kokkos::deep_copy(off, res.view.cellFacetOffset);
    Kokkos::deep_copy(nbr, res.view.facetNeighbor);
    Kokkos::deep_copy(area, res.view.facetArea);
    Kokkos::deep_copy(dv, res.view.facetConnect);
    Kokkos::deep_copy(conn, res.view.facetConnVec);

    int dvMis = 0, areaMis = 0, connMis = 0, unmatched = 0, checked = 0;
    real_t maxDV = 0, maxArea = 0, maxConn = 0;
    for (int i = 0; i < N; ++i)
      for (int f = off(i); f < off(i + 1); ++f) {
        const int nb = nbr(f);
        if (nb < 0)
          continue;
        auto it = ref.find({i, nb});
        if (it == ref.end()) {
          ++unmatched;
          continue;
        }
        ++checked;
        for (int c = 0; c < 3; ++c) {
          maxDV = std::max(maxDV, std::fabs(dv(3 * f + c) - it->second[0][c]));
          maxArea = std::max(maxArea, std::fabs(area(3 * f + c) - it->second[1][c]));
          maxConn = std::max(maxConn, std::fabs(conn(3 * f + c) - it->second[2][c]));
        }
        for (int c = 0; c < 3; ++c) {
          if (std::fabs(dv(3 * f + c) - it->second[0][c]) > 1e-9)
            ++dvMis;
          if (std::fabs(area(3 * f + c) - it->second[1][c]) > 1e-9)
            ++areaMis;
          if (std::fabs(conn(3 * f + c) - it->second[2][c]) > 1e-9)
            ++connMis;
        }
      }
    if (dvMis || areaMis || connMis || unmatched > checked / 1000 + 2) {
      std::fprintf(stderr, "  dvMis=%d areaMis=%d connMis=%d unmatched=%d/%d\n", dvMis, areaMis,
                   connMis, unmatched, checked);
      ++failures;
    }
    std::printf("device_geometry: checked=%d maxDV=%.2e maxArea=%.2e maxConn=%.2e %s\n", checked,
                maxDV, maxArea, maxConn, failures == 0 ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "device_geometry PASS" : "device_geometry FAIL");
  return failures;
}
