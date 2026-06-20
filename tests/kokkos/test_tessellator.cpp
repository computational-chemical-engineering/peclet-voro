/**
 * @file test_tessellator.cpp
 * \brief Phase-3 acceptance: full-system device tessellation vs legacy.
 *
 * Runs the whole device pipeline (cell-linked grid -> per-cell cutter -> CSR
 * write-out) and checks the published TessellationView against the legacy
 * CellComplex: per-cell volume, neighbour set, space-filling, and that every
 * cell is complete (status == Ok, i.e. the grid search closed the cell and the
 * 128 cap held). Reports a cells/sec baseline. Voronoi and Power.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/voronoi.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {

Kokkos::View<real_t*, tpx::MemSpace> uploadPos(const std::vector<Vec3>& pos) {
  Kokkos::View<real_t*, tpx::MemSpace> d(
      Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), pos.size() * 3);
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < pos.size(); ++i)
    for (int k = 0; k < 3; ++k)
      h(3 * i + k) = pos[i][k];
  Kokkos::deep_copy(d, h);
  return d;
}

template <bool Weighted>
int runCase(const char* tag, int N, real_t Lc, unsigned seed) {
  const Vec3 L = {Lc, Lc, Lc};
  const real_t boxVol = L[0] * L[1] * L[2];
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  std::vector<real_t> w(N, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * U(rng);
    if (Weighted)
      w[i] = 0.01 * U(rng);
  }

  // Legacy reference.
  vor::Box<real_t> box(L);
  vor::CellComplex<real_t, Weighted> complex(&box);
  if constexpr (Weighted)
    complex.setWeights(w);
  complex.build(pos);
  std::vector<vor::Cell<real_t> > legacy;
  complex.materializeCells(legacy);
  const std::vector<vor::CellGeometry<real_t> >& geom = complex.getGeoms();
  std::vector<int> idToLegacy(N, -1);
  for (int li = 0; li < (int)legacy.size(); ++li)
    idToLegacy[(int)legacy[li].getID()] = li;

  // Device tessellation.
  auto dPos = uploadPos(pos);
  Kokkos::View<real_t*, tpx::MemSpace> dW("w", N);
  {
    auto hw = Kokkos::create_mirror_view(dW);
    for (int i = 0; i < N; ++i)
      hw(i) = w[i];
    Kokkos::deep_copy(dW, hw);
  }
  const real_t Larr[3] = {L[0], L[1], L[2]};
  Kokkos::fence();
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = vor::device::buildTessellation<real_t, Weighted>(dPos, dW, N, Larr);
  Kokkos::fence();
  auto t1 = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();

  // Mirror results.
  auto vol = Kokkos::create_mirror_view(res.view.cellVolume);
  auto off = Kokkos::create_mirror_view(res.view.cellFacetOffset);
  auto nbr = Kokkos::create_mirror_view(res.view.facetNeighbor);
  auto st = Kokkos::create_mirror_view(res.status);
  Kokkos::deep_copy(vol, res.view.cellVolume);
  Kokkos::deep_copy(off, res.view.cellFacetOffset);
  Kokkos::deep_copy(nbr, res.view.facetNeighbor);
  Kokkos::deep_copy(st, res.status);

  int fail = 0, volMis = 0, setMis = 0, badStatus = 0, emptyMis = 0;
  double totalVol = 0;
  for (int i = 0; i < N; ++i) {  // seed id == i
    const int li = idToLegacy[i];
    const bool devEmpty = (st(i) & vor::device::kEmpty) != 0;
    if (li < 0) {  // legacy buried this seed
      if (!devEmpty)
        ++emptyMis;
      continue;
    }
    if (st(i) & (vor::device::kOverflow | vor::device::kIncomplete))
      ++badStatus;
    if (devEmpty) {
      ++emptyMis;
      continue;
    }
    totalVol += vol(i);
    if (std::fabs(vol(i) - geom[li].getVolume()) / geom[li].getVolume() > 1e-9)
      ++volMis;
    std::vector<int> dn, ln;
    for (int k = off(i); k < off(i + 1); ++k)
      if (nbr(k) >= 0)
        dn.push_back(nbr(k));
    for (vor::uint1 f = 0; f < legacy[li].numFacets(); ++f) {
      vor::uint2 g = legacy[li].getNbr(f);
      if (g != vor::noNbr && g != vor::boundaryNbr)
        ln.push_back((int)g);
    }
    std::sort(dn.begin(), dn.end());
    std::sort(ln.begin(), ln.end());
    if (dn != ln)
      ++setMis;
  }
  double volErr = std::fabs(totalVol - boxVol) / boxVol;
  if (volMis || setMis || badStatus || emptyMis || volErr > 1e-9) {
    std::fprintf(stderr, "  [%s N=%d] volMis=%d setMis=%d badStatus=%d emptyMis=%d volErr=%.2e\n",
                 tag, N, volMis, setMis, badStatus, emptyMis, volErr);
    ++fail;
  }
  std::printf(
      "  [%s N=%d] cells=%zu volErr=%.2e volMis=%d setMis=%d status!=ok:%d  %.3f Mcells/s  %s\n",
      tag, N, legacy.size(), volErr, volMis, setMis, badStatus, N / secs / 1e6,
      fail == 0 ? "PASS" : "FAIL");
  return fail;
}

// Fallback detection: with an undersized grid block (sw=1) the gathered region
// cannot close every cell, and the security check must FLAG those cells
// (kIncomplete) rather than silently emit a wrong cell — the hook a host /
// larger-scratch fallback pass keys off (plan §3 overflow fallback).
int runFallbackStress() {
  const int N = 1000;
  const Vec3 L = {1.0, 1.0, 1.0};
  std::mt19937 rng(99);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * U(rng);
  auto dPos = uploadPos(pos);
  Kokkos::View<real_t*, tpx::MemSpace> dW("w", N);
  const real_t Larr[3] = {1.0, 1.0, 1.0};
  auto res = vor::device::buildTessellation<real_t, false>(dPos, dW, N, Larr, /*sw=*/1);
  auto st = Kokkos::create_mirror_view(res.status);
  Kokkos::deep_copy(st, res.status);
  int flagged = 0;
  for (int i = 0; i < N; ++i)
    if (st(i) & vor::device::kIncomplete)
      ++flagged;
  // sw=1 (coverage = 1 cell width ~ mean spacing) cannot satisfy
  // coverage^2 > 4 rSqMax for most cells, so the vast majority must be flagged.
  bool ok = flagged > N / 2;
  std::printf("  [fallback] sw=1 flagged %d/%d incomplete cells  %s\n", flagged, N,
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    failures += runCase<false>("voronoi", 1000, 1.0, 3);
    failures += runCase<false>("voronoi", 4000, 1.0, 5);
    failures += runCase<true>("power", 2000, 1.0, 11);
    failures += runFallbackStress();
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "tessellator PASS" : "tessellator FAIL");
  return failures;
}
