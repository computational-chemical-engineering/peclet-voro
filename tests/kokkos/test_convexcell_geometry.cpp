/**
 * @file test_convexcell_geometry.cpp
 * \brief G2 validation: ConvexCell per-facet geometry vs the half-edge oracle.
 *
 * The Voronoi cell is unique, so its per-facet area vector, volume gradient dV, and
 * connecting vector are well-defined geometric quantities independent of representation.
 * This builds the same random seeds with the validated half-edge tessellator (whose
 * facetArea/facetConnect/facetConnVec are validated against legacy CellGeometry) and with
 * the ConvexCell + `facetGeometry`, and compares per (cell, neighbour). Confirms the
 * derivative tier (G2) is computable on the compact dual representation. Serial host.
 */
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"
#include "vorflow/device/tessellator.hpp"

using real_t = double;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const int N = 1500;
    const real_t L = 1.0;
    std::mt19937 rng(7);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<real_t> pos(3 * N);
    for (int i = 0; i < 3 * N; ++i) pos[i] = L * U(rng);

    // --- reference: half-edge tessellator, published per-(cell,nbr) geometry ---
    Kokkos::View<real_t*, tpx::MemSpace> dPos(
        Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)N * 3);
    {
      auto h = Kokkos::create_mirror_view(dPos);
      for (int i = 0; i < 3 * N; ++i) h(i) = pos[i];
      Kokkos::deep_copy(dPos, h);
    }
    Kokkos::View<real_t*, tpx::MemSpace> dW("w", N);
    const real_t Larr[3] = {L, L, L};
    auto res = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr);
    auto off = Kokkos::create_mirror_view(res.view.cellFacetOffset);
    auto cnt = Kokkos::create_mirror_view(res.view.cellFacetCount);
    auto nbr = Kokkos::create_mirror_view(res.view.facetNeighbor);
    auto area = Kokkos::create_mirror_view(res.view.facetArea);
    auto dv = Kokkos::create_mirror_view(res.view.facetConnect);
    auto conn = Kokkos::create_mirror_view(res.view.facetConnVec);
    Kokkos::deep_copy(off, res.view.cellFacetOffset);
    Kokkos::deep_copy(cnt, res.view.cellFacetCount);
    Kokkos::deep_copy(nbr, res.view.facetNeighbor);
    Kokkos::deep_copy(area, res.view.facetArea);
    Kokkos::deep_copy(dv, res.view.facetConnect);
    Kokkos::deep_copy(conn, res.view.facetConnVec);
    std::map<std::pair<int, int>, std::array<std::array<real_t, 3>, 3>> ref;  // (i,nbr)->{area,dv,conn}
    for (int i = 0; i < N; ++i)
      for (int f = off(i); f < off(i) + cnt(i); ++f) {
        const int nb = nbr(f);
        if (nb < 0) continue;
        ref[{i, nb}] = {{{area(3 * f), area(3 * f + 1), area(3 * f + 2)},
                         {dv(3 * f), dv(3 * f + 1), dv(3 * f + 2)},
                         {conn(3 * f), conn(3 * f + 1), conn(3 * f + 2)}}};
      }

    // --- ConvexCell: brute-force gather per cell, build, facetGeometry per face ---
    real_t maxArea = 0, maxDV = 0, maxConn = 0;
    int checked = 0, missing = 0;
    std::vector<std::array<real_t, 4>> cand;  // (relx,rely,relz,key)
    std::vector<int> cid;
    for (int i = 0; i < N; ++i) {
      cand.clear();
      cid.clear();
      for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        real_t rx = pos[3 * j] - pos[3 * i], ry = pos[3 * j + 1] - pos[3 * i + 1],
               rz = pos[3 * j + 2] - pos[3 * i + 2];
        rx -= std::round(rx);  // minimal image, L=1
        ry -= std::round(ry);
        rz -= std::round(rz);
        const real_t k2 = 0.5 * (rx * rx + ry * ry + rz * rz);
        if (k2 > 0.5) continue;  // far beyond any cell
        cand.push_back({rx, ry, rz, k2});
        cid.push_back(j);
      }
      std::vector<int> ord(cand.size());
      for (size_t k = 0; k < ord.size(); ++k) ord[k] = (int)k;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cand[a][3] < cand[b][3]; });

      vor::device::ConvexCell<real_t, 128, 256> c;
      c.initBox(L, L, L);
      for (int o : ord) {
        const real_t key = cand[o][3];
        if (!(key < 2.0 * c.maxVertexRsq())) break;
        const real_t n[3] = {cand[o][0], cand[o][1], cand[o][2]};
        c.clip(n, key, cid[o]);
        if (c.overflow) break;
      }
      if (c.overflow) continue;
      for (int k = 0; k < c.np; ++k) {
        if (c.pnbr[k] < 0) continue;
        real_t a[3], d[3], cv[3];
        if (!c.facetGeometry(k, a, d, cv)) continue;
        auto it = ref.find({i, c.pnbr[k]});
        if (it == ref.end()) {
          ++missing;
          continue;
        }
        ++checked;
        for (int cc = 0; cc < 3; ++cc) {
          maxArea = std::max(maxArea, std::fabs(a[cc] - it->second[0][cc]));
          maxDV = std::max(maxDV, std::fabs(d[cc] - it->second[1][cc]));
          maxConn = std::max(maxConn, std::fabs(cv[cc] - it->second[2][cc]));
        }
      }
    }
    const real_t tol = 1e-9;
    const bool ok = maxArea < tol && maxDV < tol && maxConn < tol && missing < checked / 100 + 5;
    if (!ok) ++failures;
    std::printf("convexcell_geometry: checked=%d missing=%d  maxArea=%.2e maxDV=%.2e maxConn=%.2e  %s\n",
                checked, missing, maxArea, maxDV, maxConn, ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "convexcell_geometry PASS" : "convexcell_geometry FAIL");
  return failures;
}
