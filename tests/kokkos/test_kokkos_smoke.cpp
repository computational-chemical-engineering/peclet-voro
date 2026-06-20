/**
 * @file test_kokkos_smoke.cpp
 * \brief Phase-0 smoke test for the Kokkos device build.
 *
 * Proves the suite toolchain end-to-end before any real device kernels exist:
 *   - Kokkos initializes and runs a parallel_reduce on the default backend;
 *   - transport-core's tpx::View / tpx::toDevice round-trips host<->device;
 *   - vorflow's C++17-clean core types (vor_types.hpp) compile under the C++20
 *     Kokkos build and interoperate with device Views.
 * Exit non-zero on any mismatch (ctest oracle, no framework).
 */

#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/vor_types.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    // (1) device parallel_reduce: sum 0..N-1 == N(N-1)/2.
    const int N = 100000;
    long sum = 0;
    Kokkos::parallel_reduce(
        "smoke_sum", Kokkos::RangePolicy<tpx::ExecSpace>(0, N),
        KOKKOS_LAMBDA(const int i, long& acc) { acc += i; }, sum);
    const long expect = (long)N * (N - 1) / 2;
    if (sum != expect) {
      std::fprintf(stderr, "FAIL reduce: %ld != %ld\n", sum, expect);
      ++failures;
    }

    // (2) tpx::View round-trip: upload, scale on device, copy back.
    std::vector<double> h(1024);
    for (size_t i = 0; i < h.size(); ++i)
      h[i] = (double)i;
    tpx::View<double> d = tpx::toDevice(h, "smoke_field");
    Kokkos::parallel_for(
        "smoke_scale", Kokkos::RangePolicy<tpx::ExecSpace>(0, d.extent(0)),
        KOKKOS_LAMBDA(const int i) { d(i) *= 2.0; });
    auto hm = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(hm, d);
    for (size_t i = 0; i < h.size(); ++i) {
      if (hm(i) != 2.0 * (double)i) {
        std::fprintf(stderr, "FAIL view round-trip at %zu: %g\n", i, hm(i));
        ++failures;
        break;
      }
    }

    // (3) vorflow core types/constants are usable in the Kokkos TU: pack a
    // (facet, vertex, edge) half-edge label and unpack via the mask constants.
    const vor::uint1 facet = 5, vert = 2, edge = 1;
    const vor::uint1 label = (vor::uint1)((facet << vor::shiftFacet) | (vert << 2) | edge);
    const vor::uint1 gotFacet = (vor::uint1)(label >> vor::shiftFacet);
    const vor::uint1 gotVert = (vor::uint1)((label & vor::maskVertex) >> 2);
    const vor::uint1 gotEdge = (vor::uint1)(label & vor::maskEdge);
    if (gotFacet != facet || gotVert != vert || gotEdge != edge) {
      std::fprintf(stderr, "FAIL vor_types label round-trip: f=%u v=%u e=%u\n", gotFacet, gotVert,
                   gotEdge);
      ++failures;
    }
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "kokkos smoke PASS" : "kokkos smoke FAIL");
  return failures;
}
