/**
 * @file test_cospherical.cpp
 * @brief Cold build and moving-point repair on exactly cospherical (mirrored) seed sets.
 *
 * A seed set mirrored exactly across a plane makes every Voronoi edge on that plane a 4-cell edge
 * and every vertex on it a 6-cell vertex: the clip sees groups of dual vertices that coincide in
 * exact arithmetic but differ by round-off, so the sign of a later cut can split such a group. The
 * construction must still yield a valid convex cell (Sugihara–Iri: topology first, floating point
 * only a guide) — and the partition must be exact.
 *
 * Cases (periodic unit box, so the seam z = 0 is a second mirror plane): random and lattice halves
 * mirrored about z = 1/2; the same quantised to dyadic coordinates; jittered by 1e-15; the power
 * policy with mirrored weights; and MovingTessellation::step under a mirrored displacement (the set
 * stays cospherical), at a tight and at the production certificate tolerance (see repairCase).
 * Every case gates the oracle-free invariants of test_tessellation_invariants (space-filling to
 * 1e-12, positive volumes, every cell Ok up to the informational kDegenerate bit, area closure) and
 * facet reciprocity in absolute form: |A_ij + A_ji| <= 1e-10 h^2 over every facet (a one-sided
 * facet counts with its own area, so only zero-area touching contacts may be one-sided; the exact
 * mirrors land at ~1e-14, the 1e-15-jittered set — ill-conditioned by construction, its
 * near-coincident corners amplify round-off by eps / jitter — near 1e-12).
 */
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/dynamic_validate.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"

using real_t = double;
using Mem = peclet::core::MemSpace;

namespace {

/// `half` holds seeds with z in (0, 1/2); returns them followed by their mirror images about
/// z = 1/2, optionally jittered (normal deviate) and/or quantised to multiples of 2^-dyadic.
std::vector<real_t> mirrored(const std::vector<real_t>& half, real_t jitter, int dyadic,
                             unsigned seed) {
  const int n = (int)half.size() / 3;
  std::vector<real_t> p(6 * n);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) {
      p[3 * i + d] = half[3 * i + d];
      p[3 * (n + i) + d] = d == 2 ? real_t(1) - half[3 * i + d] : half[3 * i + d];
    }
  std::mt19937 rng(seed);
  std::normal_distribution<real_t> G(0.0, 1.0);
  for (auto& v : p) {
    if (jitter > 0)
      v += jitter * G(rng);
    if (dyadic > 0)
      v = std::round(std::ldexp(v, dyadic)) / std::ldexp(real_t(1), dyadic);
    v -= std::floor(v);
    if (v >= real_t(1))
      v = 0;
  }
  return p;
}

/// A generic (non-degenerate) half: an m x m x m/2 lattice in z < 1/2, every point displaced
/// uniformly by up to 0.35 spacings per axis — random, but well separated (no near pairs, so the
/// small power weights below bury no cell), like the Lloyd meshes the bug was found on.
std::vector<real_t> randomHalf(int m, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(-0.35, 0.35);
  std::vector<real_t> h;
  const real_t a = real_t(1) / m;
  for (int k = 0; k < m / 2; ++k)
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i) {
        h.push_back((i + real_t(0.5) + U(rng)) * a);
        h.push_back((j + real_t(0.5) + U(rng)) * a);
        h.push_back((k + real_t(0.5) + U(rng)) * a);
      }
  return h;
}

/// A cubic lattice of m x m x m/2 points in the lower half, offset by a quarter spacing in z so
/// the mirror is a NEW plane of cospherical vertices, not a lattice plane.
std::vector<real_t> latticeHalf(int m) {
  std::vector<real_t> h;
  const real_t a = real_t(1) / m;
  for (int k = 0; k < m / 2; ++k)
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i) {
        h.push_back((i + real_t(0.5)) * a);
        h.push_back((j + real_t(0.5)) * a);
        h.push_back((k + real_t(0.25)) * a);
      }
  return h;
}

Kokkos::View<real_t*, Mem> toDevice(const std::vector<real_t>& h, const char* name) {
  Kokkos::View<real_t*, Mem> d(name, h.size());
  Kokkos::deep_copy(d, Kokkos::View<const real_t*, Kokkos::HostSpace>(h.data(), h.size()));
  return d;
}

/// Gate the published view: the invariants of test_tessellation_invariants + the status mask.
int gate(const std::string& name, const peclet::voro::TessellationView<real_t>& view,
         const Kokkos::View<int*, Mem>& status, int N) {
  auto aux = peclet::voro::buildAuxMaps(view);
  auto inv = peclet::voro::checkInvariants(view, aux, 1.0);
  double minVol = 1e300;
  long nOk = N;
  {
    auto V = view.cellVolume;
    Kokkos::parallel_reduce(
        "minvol", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i, double& mn) { mn = Kokkos::min(mn, (double)V(i)); },
        Kokkos::Min<double>(minVol));
    if (status.extent(0) > 0) {
      auto S = status;
      Kokkos::parallel_reduce(
          "nok", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
          KOKKOS_LAMBDA(int i, long& c) {
            c += ((S(i) & ~peclet::voro::kDegenerate) == peclet::voro::kOk) ? 1 : 0;
          },
          nOk);
    }
  }
  // Facet reciprocity, absolute: every facet must be seen from both sides with equal area. A
  // facet without a reciprocal counts with its own area, so a one-sided zero-area contact (a
  // plane that only touches the cell at a cospherical vertex) passes, a lost face does not.
  const double h2 = std::pow(double(N), -2.0 / 3.0);  // spacing^2 of the unit box
  double maxPairErr = 0;
  {
    auto recip = aux.recip;
    const int nF = view.numFacets();
    Kokkos::parallel_reduce(
        "pairerr", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, nF),
        KOKKOS_LAMBDA(int g, double& m) {
          const int r = recip(g);
          double e = 0;
          for (int c = 0; c < 3; ++c) {
            const double d = (double)view.area(g, c) + (r >= 0 ? (double)view.area(r, c) : 0.0);
            e += d * d;
          }
          m = Kokkos::max(m, Kokkos::sqrt(e));
        },
        Kokkos::Max<double>(maxPairErr));
    maxPairErr /= h2;
  }
  const bool pass = inv.volRelErr < 1e-12 && minVol > 0.0 && nOk == N && inv.sumAreaMag < 1e-9 &&
                    maxPairErr <= 1e-10;
  std::printf(
      "  %-34s N=%-6d volRelErr=%.2e minVol=%.2e nOk=%ld areaClosure=%.2e pairErr/h2=%.2e "
      "(one-sided %ld) %s\n",
      name.c_str(), N, inv.volRelErr, minVol, nOk, inv.sumAreaMag, maxPairErr, inv.nNonRecip,
      pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

int coldCase(const std::string& name, const std::vector<real_t>& xh, bool weighted) {
  const int N = (int)xh.size() / 3;
  const real_t Larr[3] = {1, 1, 1};
  auto pos = toDevice(xh, "pos");
  Kokkos::View<long*, Mem> gd;
  if (!weighted) {
    Kokkos::View<real_t*, Mem> wd;
    auto res = peclet::voro::buildTessellation<real_t, false>(pos, wd, N, Larr, 4, N, gd,
                                                              peclet::voro::NoSdf{}, true);
    return gate(name, res.view, res.status, N);
  }
  // mirrored weights (the image of seed i is seed i + N/2): small, so no cell is buried
  const real_t spacing = real_t(1) / std::cbrt(real_t(N));
  std::mt19937 rng(11);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> wh(N);
  for (int i = 0; i < N / 2; ++i)
    wh[i] = wh[N / 2 + i] = real_t(0.05) * spacing * spacing * U(rng);
  auto wd = toDevice(wh, "w");
  auto res = peclet::voro::buildTessellation<real_t, true>(pos, wd, N, Larr, 4, N, gd,
                                                           peclet::voro::NoSdf{}, true);
  return gate(name, res.view, res.status, N);
}

/// Cold build at the mirrored set, then MovingTessellation::step to a mirrored displacement of it
/// (the moved set is exactly mirrored too). At a tight certificate tolerance (1e-6 spacing) the
/// repaired tessellation must pass the full gate. At the production tolerance (1e-4 spacing) the
/// certificate by design accepts a stored topology whose violation stays below it — and a
/// cospherical tie flips under ANY motion, by a violation far below tol — so there the gate is the
/// repair's equivalence contract: per-cell volume within 1e-9 of a cold rebuild, space-filling.
int repairCase(const std::string& name, const std::vector<real_t>& half, real_t jitter, int dyadic,
               real_t tolFrac) {
  const int n = (int)half.size() / 3, N = 2 * n;
  const real_t spacing = real_t(1) / std::cbrt(real_t(N));
  std::mt19937 rng(5);
  std::uniform_real_distribution<real_t> U(-1.0, 1.0);
  std::vector<real_t> moved(half);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d)
      moved[3 * i + d] += real_t(0.02) * spacing * U(rng);
  auto pos0 = toDevice(mirrored(half, jitter, dyadic, 1), "p0");
  auto pos1 = toDevice(mirrored(moved, jitter, dyadic, 2), "p1");
  const real_t Larr[3] = {1, 1, 1};
  const real_t tol = tolFrac * spacing, skin = real_t(0.25) * spacing;
  peclet::voro::MovingTessellation<real_t, 64, 112, false> mt;
  mt.alloc(N, Larr, tol, skin, 4, N);
  mt.rebuild(pos0);
  int rc = gate(name + " (cold)", peclet::voro::reevalPublish(mt.store, pos0, mt.vol, N, Larr),
                mt.lastStatus, N);
  mt.step(pos1);
  // the equivalence contract: every cell's volume against a cold rebuild at the new positions
  Kokkos::View<long*, Mem> gd;
  Kokkos::View<real_t*, Mem> wd;
  auto cold = peclet::voro::buildTessellation<real_t, false>(pos1, wd, N, Larr, 4, N, gd,
                                                             peclet::voro::NoSdf{}, false);
  auto vc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold.view.cellVolume);
  auto vs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.vol);
  double maxRel = 0;
  for (int i = 0; i < N; ++i)
    maxRel = std::max(maxRel, std::abs((double)vs(i) - (double)vc(i)) / (double)vc(i));
  auto view = peclet::voro::reevalPublish(mt.store, pos1, mt.vol, N, Larr);
  if (tolFrac <= real_t(1e-6)) {
    rc |= gate(name + " (step)", view, Kokkos::View<int*, Mem>(), N);
    rc |= maxRel <= 1e-12 ? 0 : 1;
  } else {
    auto aux = peclet::voro::buildAuxMaps(view);
    auto inv = peclet::voro::checkInvariants(view, aux, 1.0);
    const bool pass = inv.volRelErr < 1e-12 && maxRel <= 1e-9;
    std::printf("  %-34s N=%-6d volRelErr=%.2e max|V-V_cold|/V=%.2e (contract 1e-9) %s\n",
                (name + " (step)").c_str(), N, inv.volRelErr, maxRel, pass ? "OK" : "FAIL");
    rc |= pass ? 0 : 1;
  }
  std::printf("  %-34s max|V-V_cold|/V=%.2e\n", "", maxRel);
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("cospherical (mirrored) seed sets (backend=%s):\n",
                Kokkos::DefaultExecutionSpace::name());
    for (int m : {12, 26}) {
      const auto h = randomHalf(m, 3u + m);
      const std::string tag = " m=" + std::to_string(m);
      rc |= coldCase("mirror" + tag, mirrored(h, 0, 0, 1), false);
      rc |= coldCase("mirror dyadic 2^-20" + tag, mirrored(h, 0, 20, 1), false);
      rc |= coldCase("mirror dyadic 2^-12" + tag, mirrored(h, 0, 12, 1), false);
      rc |= coldCase("mirror jitter 1e-15" + tag, mirrored(h, 1e-15, 0, 1), false);
      rc |= coldCase("mirror power" + tag, mirrored(h, 0, 0, 1), true);
      rc |= coldCase("mirror dyadic power" + tag, mirrored(h, 0, 20, 1), true);
    }
    rc |= coldCase("lattice+mirror", mirrored(latticeHalf(16), 0, 0, 1), false);
    rc |= coldCase("lattice+mirror power", mirrored(latticeHalf(16), 0, 0, 1), true);
    for (real_t tolFrac : {real_t(1e-6), real_t(1e-4)}) {
      const std::string tag = tolFrac < real_t(1e-5) ? " tol 1e-6" : " tol 1e-4";
      rc |= repairCase("repair mirror" + tag, randomHalf(20, 9u), 0, 0, tolFrac);
      rc |= repairCase("repair mirror dyadic" + tag, randomHalf(20, 9u), 0, 20, tolFrac);
      rc |= repairCase("repair mirror jitter 1e-15" + tag, randomHalf(20, 9u), 1e-15, 0, tolFrac);
    }
    std::printf("%s\n", rc == 0 ? "ALL COSPHERICAL CASES PASS" : "COSPHERICAL CASES FAILED");
  }
  Kokkos::finalize();
  return rc;
}
