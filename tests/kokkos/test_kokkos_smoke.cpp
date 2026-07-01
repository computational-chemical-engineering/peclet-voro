/**
 * @file test_kokkos_smoke.cpp
 * \brief Phase-0 smoke test for the Kokkos device build.
 *
 * Proves the suite toolchain end-to-end before any real device kernels exist:
 *   - Kokkos initializes and runs a parallel_reduce on the default backend;
 *   - core's peclet::core::View / peclet::core::toDevice round-trips host<->device;
 * *   - core's peclet::core::View round-trips through a device kernel under the C++20
 *     Kokkos build and interoperate with device Views.
 * Exit non-zero on any mismatch (ctest oracle, no framework).
 */

#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/view.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    // (1) device parallel_reduce: sum 0..N-1 == N(N-1)/2.
    const int N = 100000;
    long sum = 0;
    Kokkos::parallel_reduce(
        "smoke_sum", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(const int i, long& acc) { acc += i; }, sum);
    const long expect = (long)N * (N - 1) / 2;
    if (sum != expect) {
      std::fprintf(stderr, "FAIL reduce: %ld != %ld\n", sum, expect);
      ++failures;
    }

    // (2) peclet::core::View round-trip: upload, scale on device, copy back.
    std::vector<double> h(1024);
    for (size_t i = 0; i < h.size(); ++i)
      h[i] = (double)i;
    peclet::core::View<double> d = peclet::core::toDevice(h, "smoke_field");
    Kokkos::parallel_for(
        "smoke_scale", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, d.extent(0)),
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
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "kokkos smoke PASS" : "kokkos smoke FAIL");
  return failures;
}
