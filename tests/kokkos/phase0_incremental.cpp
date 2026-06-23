/**
 * @file phase0_incremental.cpp
 * \brief Phase 0 of the GPU-Voronoi-for-physics program (docs/voronoi_gpu_research_program.md):
 * characterise the moving-point workload.
 *
 * The real cost in a simulation is the per-step UPDATE, not the cold rebuild. Topology
 * (which seeds are neighbours) changes rarely; geometry (coords/areas/volumes/derivatives)
 * changes every step but is a closed-form re-evaluation given fixed topology. This program
 * measures the **topology-stable fraction** as a function of the per-step displacement, and
 * how many faces flip in the cells that do change — the number that decides whether the
 * clip is a side-show (most cells geometry-only) or the main event.
 *
 * Method: build the tessellation at P0, displace every point by a random vector of scale
 * `frac · spacing` (periodic), rebuild, and compare each cell's sorted neighbour set
 * (cell i == seed i across builds). Pure tessellation (no force geometry) — only topology.
 * Run:  ./phase0_incremental [N]   (env: P0_FRACS="0.001 0.003 ..." overrides the sweep)
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/tessellator.hpp"

using real_t = double;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Build the tessellation and return, per cell, its sorted list of neighbour seed ids
// (boundary facets, neighbour < 0, dropped). cell i corresponds to seed i.
static double build_neighbour_sets(const std::vector<real_t>& h, int N, const real_t L[3], int sw,
                                   std::vector<std::vector<int>>& nbr) {
  Kokkos::View<real_t*, tpx::MemSpace> dPos(
      Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)N * 3);
  {
    auto m = Kokkos::create_mirror_view(dPos);
    for (int i = 0; i < 3 * N; ++i) m(i) = h[i];
    Kokkos::deep_copy(dPos, m);
  }
  Kokkos::View<real_t*, tpx::MemSpace> dW("w", 0);
  Kokkos::View<long*, tpx::MemSpace> noGid;
  auto t0 = clk::now();
  auto res = vor::device::buildTessellation<real_t, false>(dPos, dW, N, L, sw, -1, noGid,
                                                           vor::device::NoSdf{}, false);
  Kokkos::fence();
  auto t1 = clk::now();

  auto base = Kokkos::create_mirror_view(res.view.cellFacetOffset);
  auto cnt = Kokkos::create_mirror_view(res.view.cellFacetCount);
  auto fn = Kokkos::create_mirror_view(res.view.facetNeighbor);
  Kokkos::deep_copy(base, res.view.cellFacetOffset);
  Kokkos::deep_copy(cnt, res.view.cellFacetCount);
  Kokkos::deep_copy(fn, res.view.facetNeighbor);

  nbr.assign(N, {});
  for (int i = 0; i < N; ++i) {
    const int b = base(i), c = cnt(i);
    auto& v = nbr[i];
    v.reserve(c);
    for (int g = b; g < b + c; ++g) {
      const int j = (int)fn(g);
      if (j >= 0) v.push_back(j);
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());  // periodic multi-image -> one neighbour
  }
  return secs(t0, t1);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("backend: %s\n", Kokkos::DefaultExecutionSpace::name());
    const int N = argc > 1 ? std::atoi(argv[1]) : 200000;
    const real_t L[3] = {1.0, 1.0, 1.0};
    const int sw = std::getenv("VORFLOW_SW") ? std::atoi(std::getenv("VORFLOW_SW")) : 4;
    const real_t spacing = std::cbrt(L[0] * L[1] * L[2] / N);

    std::vector<real_t> fracs = {0.001, 0.003, 0.01, 0.03, 0.1, 0.3};
    if (const char* e = std::getenv("P0_FRACS")) {
      fracs.clear();
      std::stringstream ss(e);
      real_t f;
      while (ss >> f) fracs.push_back(f);
    }

    std::mt19937 rng(2024);
    std::uniform_real_distribution<real_t> U(0.0, 1.0);
    std::vector<real_t> p0(3 * N);
    for (int i = 0; i < 3 * N; ++i) p0[i] = U(rng);

    std::vector<std::vector<int>> nbr0, nbr1;
    double tBuild = build_neighbour_sets(p0, N, L, sw, nbr0);
    double meanFaces = 0;
    for (auto& v : nbr0) meanFaces += v.size();
    meanFaces /= N;
    std::printf("N=%d  spacing=%.4e  full-build=%.3fs (%.0f kcells/s)  mean faces/cell=%.2f\n", N,
                spacing, tBuild, N / tBuild / 1e3, meanFaces);
    std::printf("%-10s %-12s %-14s %-18s %-14s\n", "frac", "rms_disp/sp", "stable_cells",
                "mean_flipped_faces", "any-flip_faces");

    for (real_t frac : fracs) {
      const real_t a = frac * spacing;  // per-axis uniform[-a,a]
      std::uniform_real_distribution<real_t> D(-a, a);
      std::vector<real_t> p1(3 * N);
      double sumsq = 0;
      for (int i = 0; i < 3 * N; ++i) {
        real_t x = p0[i] + D(rng);
        x -= std::floor(x);  // periodic wrap into [0,1)
        p1[i] = x;
      }
      for (int i = 0; i < N; ++i) {
        real_t dx = p1[3 * i] - p0[3 * i], dy = p1[3 * i + 1] - p0[3 * i + 1],
               dz = p1[3 * i + 2] - p0[3 * i + 2];
        // minimal image for the displacement magnitude
        dx -= std::round(dx);
        dy -= std::round(dy);
        dz -= std::round(dz);
        sumsq += dx * dx + dy * dy + dz * dz;
      }
      const double rms = std::sqrt(sumsq / N) / spacing;

      build_neighbour_sets(p1, N, L, sw, nbr1);
      long stable = 0, flippedFacesSum = 0, cellsWithFlip = 0;
      for (int i = 0; i < N; ++i) {
        // symmetric difference size of the two sorted neighbour sets
        const auto& A = nbr0[i];
        const auto& B = nbr1[i];
        int ai = 0, bi = 0, diff = 0;
        while (ai < (int)A.size() && bi < (int)B.size()) {
          if (A[ai] == B[bi]) { ++ai; ++bi; }
          else if (A[ai] < B[bi]) { ++ai; ++diff; }
          else { ++bi; ++diff; }
        }
        diff += (int)A.size() - ai + (int)B.size() - bi;
        if (diff == 0) ++stable;
        else { ++cellsWithFlip; flippedFacesSum += diff; }
      }
      std::printf("%-10.4g %-12.4f %-14.4f %-18.3f %-14.2f\n", (double)frac, rms,
                  (double)stable / N, cellsWithFlip ? (double)flippedFacesSum / cellsWithFlip : 0.0,
                  (double)flippedFacesSum / N);
    }
    std::printf(
        "\n(stable_cells = fraction whose neighbour set is unchanged = needs GEOMETRY-ONLY update;\n"
        " 1-stable = needs re-clip. mean_flipped_faces = symmetric-difference size among changed "
        "cells.)\n");
  }
  Kokkos::finalize();
  return 0;
}
