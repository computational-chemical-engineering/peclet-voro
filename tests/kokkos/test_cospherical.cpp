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
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tess_grid.hpp"
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

// ---- near-lattice power cells: the cold build's clip vs the repair gather's (M7, 2026-11-08) ----
//
// The cold build clips with the lean ConvexCell, the repair gather (MovingTessellation::gatherSet)
// with the adjacency-tracking one (TrackAdj): the same cut decisions, a different horizon
// collection. On an (almost) exact cubic lattice with small power weights every cell has 8-cell
// vertices and 4-cell edges, and the gather order passes through further cospherical corners of
// the still-open cell, so the clip's growDisk fallback runs on many cuts. growDisk exempted the
// edges to candidates that might still join K; one that never joined then failed the horizon
// walk's inconsistentFan, which kept K's triangle and broke the disk — the gathered cell was
// published EMPTY (an "overflow") while the cold build had it whole: the M7 rest gate's weight
// solve then saw a buried cell. The cases below gate the two clips against each other.

/// The failing cell, recorded: seed 824 of the M7 walled 16^3 rest gate (ratio 1000, option S,
/// step 33), its weights ~1e-17 h^2 apart, in the gather's clip order — the early wall plane, then
/// the worklist. Row: neighbour id, plane direction, offset (exactly as passed to clip).
constexpr double kLatticeCell[][5] = {
    {peclet::voro::kBoundaryFacet, 0, 0, 1, 0.4375},
    {1027, 0.0625, 0.0625, -0.0625, 0.005859375},
    {644, -0.0625, 0.0625, 0, 0.00390625},
    {836, 0, 0.0625, 0, 0.001953125},
    {1028, 0.0625, 0.0625, 0, 0.00390625},
    {1004, 0.0625, -0.0625, 0, 0.00390625},
    {1016, 0.0625, 0, 0, 0.001953125},
    {620, -0.0625, -0.0625, 0, 0.00390625},
    {632, -0.0625, 0, 0, 0.001953125},
    {812, 0, -0.0625, 0, 0.001953125},
    {619, -0.0625, -0.0625, -0.0625, 0.005859375},
    {631, -0.0625, 0, -0.0625, 0.00390625},
    {811, 0, -0.0625, -0.0625, 0.00390625},
    {823, 0, 0, -0.0625, 0.001953125},
    {1003, 0.0625, -0.0625, -0.0625, 0.005859375},
    {1015, 0.0625, 0, -0.0625, 0.00390625},
    {643, -0.0625, 0.0625, -0.0625, 0.005859375},
    {835, 0, 0.0625, -0.0625, 0.00390625},
    {1005, 0.0625, -0.0625, 0.0625, 0.0058593750000000009},
    {1006, 0.0625, -0.0625, 0.125, 0.011718749999999998},
    {1017, 0.0625, 0, 0.0625, 0.0039062500000000009},
    {1018, 0.0625, 0, 0.125, 0.0097656249999999983},
    {452, -0.125, 0.0625, 0, 0.009765625},
    {645, -0.0625, 0.0625, 0.0625, 0.0058593750000000009},
    {646, -0.0625, 0.0625, 0.125, 0.011718749999999998},
    {837, 0, 0.0625, 0.0625, 0.0039062500000000009},
    {838, 0, 0.0625, 0.125, 0.0097656249999999983},
    {621, -0.0625, -0.0625, 0.0625, 0.0058593750000000009},
    {622, -0.0625, -0.0625, 0.125, 0.011718749999999998},
    {633, -0.0625, 0, 0.0625, 0.0039062500000000009},
    {634, -0.0625, 0, 0.125, 0.0097656249999999983},
    {813, 0, -0.0625, 0.0625, 0.0039062500000000009},
    {814, 0, -0.0625, 0.125, 0.0097656249999999983},
    {825, 0, 0, 0.0625, 0.0019531250000000004},
};

template <bool TrackAdj>
double clipRecorded(bool& overflow) {
  peclet::voro::ConvexCell<double, peclet::voro::kMaxPlanes, peclet::voro::kMaxTriangles, TrackAdj>
      c;
  c.initBox(1.0, 1.0, 1.0);
  for (const auto& r : kLatticeCell) {
    const double pdir[3] = {r[1], r[2], r[3]};
    c.clip(pdir, r[4], (int)r[0]);
  }
  overflow = c.overflow;
  return c.overflow ? 0.0 : c.volumePerVertex();
}

int latticeCellCase() {
  const double h3 = std::pow(1.0 / 16, 3);
  bool o0 = false, o1 = false;
  const double v0 = clipRecorded<false>(o0), v1 = clipRecorded<true>(o1);
  const double e0 = std::abs(v0 / h3 - 1), e1 = std::abs(v1 / h3 - 1);
  const bool pass = !o0 && !o1 && e0 <= 1e-12 && e1 <= 1e-12;
  std::printf("  %-34s |V/h^3-1| lean %.2e (ovf %d) adj %.2e (ovf %d) %s\n", "M7 lattice cell 824",
              e0, (int)o0, e1, (int)o1, pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

/// Uniform deviate in [-1, 1) from a counter (splitmix64): the same draws on every backend.
KOKKOS_INLINE_FUNCTION double hashU(unsigned long long k) {
  k += 0x9E3779B97F4A7C15ull;
  k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9ull;
  k = (k ^ (k >> 27)) * 0x94D049BB133111EBull;
  k ^= k >> 31;
  return double(k >> 11) * 0x1.0p-52 - 1.0;
}

/// `trials` interior cells of the 16^3 lattice (seed and its 124 neighbours in [-2,2]^3, each
/// displaced by ex h and weighted W0 + ew h^2 times a deviate; W0 the M7 state's common weight),
/// clipped in distance order with random ties (the worklist's shell structure; order 0) or in
/// random order (order 1, a harder stress), by the lean and by the TrackAdj ConvexCell. Gate:
/// neither overflows, |V/h^3 - 1| <= 1e-6 (the perturbation moves V by ~ex) and the two volumes
/// agree to 1e-12 h^3.
int latticeClipCase(double ex, double ew, int order, int trials) {
  constexpr int K = 124;
  const double h = 1.0 / 16, h3 = h * h * h;
  long bad = 0;
  double maxErr = 0, maxDiff = 0;
  Kokkos::parallel_reduce(
      "latticeClip", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, trials),
      KOKKOS_LAMBDA(int tr, long& nb, double& me, double& md) {
        unsigned long long key = 1000003ull * (unsigned long long)tr;
        double c[3], xi[3];
        for (int d = 0; d < 3; ++d) {
          c[d] = (2 + int(6 * (hashU(key++) + 1)) + 0.5) * h;  // lattice site in [2, 13]
          xi[d] = c[d] + ex * h * hashU(key++);
        }
        const double w0 = -7.1e-14 * h * h;
        const double wi = w0 + ew * h * h * hashU(key++);
        double r[K][3], off[K], kk[K];
        int ord[K], m = 0;
        for (int a = -2; a <= 2; ++a)
          for (int b = -2; b <= 2; ++b)
            for (int e = -2; e <= 2; ++e) {
              if (a == 0 && b == 0 && e == 0)
                continue;
              const int o[3] = {a, b, e};
              for (int d = 0; d < 3; ++d)
                r[m][d] = (c[d] + o[d] * h + ex * h * hashU(key++)) - xi[d];
              const double wj = w0 + ew * h * h * hashU(key++);
              off[m] = peclet::voro::Power::offsetFromRel<double>(r[m], wi, wj);
              kk[m] = (order == 0 ? double(a * a + b * b + e * e) : 0.0) + 0.5 * hashU(key++);
              ord[m] = m;
              ++m;
            }
        for (int i = 1; i < K; ++i)  // insertion sort by key
          for (int j = i; j > 0 && kk[ord[j]] < kk[ord[j - 1]]; --j) {
            const int t = ord[j];
            ord[j] = ord[j - 1];
            ord[j - 1] = t;
          }
        peclet::voro::ConvexCell<double, peclet::voro::kMaxPlanes, peclet::voro::kMaxTriangles,
                                 false>
            lean;
        peclet::voro::ConvexCell<double, peclet::voro::kMaxPlanes, peclet::voro::kMaxTriangles,
                                 true>
            adj;
        lean.initBox(1.0, 1.0, 1.0);
        adj.initBox(1.0, 1.0, 1.0);
        for (int q = 0; q < K; ++q) {
          lean.clip(r[ord[q]], off[ord[q]], ord[q]);
          adj.clip(r[ord[q]], off[ord[q]], ord[q]);
        }
        const double vl = lean.overflow ? 0.0 : lean.volumePerVertex();
        const double va = adj.overflow ? 0.0 : adj.volumePerVertex();
        const double err = Kokkos::max(Kokkos::fabs(vl / h3 - 1), Kokkos::fabs(va / h3 - 1));
        const double diff = Kokkos::fabs(vl - va) / h3;
        me = Kokkos::max(me, err);
        md = Kokkos::max(md, diff);
        nb += (lean.overflow || adj.overflow || !(err <= 1e-6) || !(diff <= 1e-12)) ? 1 : 0;
      },
      bad, Kokkos::Max<double>(maxErr), Kokkos::Max<double>(maxDiff));
  char name[64];
  std::snprintf(name, sizeof name, "lattice clip %s ex %.0e ew %.0e", order ? "rand" : "dist", ex,
                ew);
  std::printf("  %-34s %d cells: bad %ld, max|V/h^3-1| %.1e, max|V_lean-V_adj|/h^3 %.1e %s\n", name,
              trials, bad, maxErr, maxDiff, bad == 0 ? "OK" : "FAIL");
  return bad == 0 ? 0 : 1;
}

/// The tessellation-level contract on a weighted near-lattice (periodic 16^3, positions displaced
/// by ex h, weights W0 + ew h^2 times a deviate) at the M7 certificate tolerance 1e-10 h: a cold
/// build (gated), then the repair's gather of EVERY cell at the same input — its volumes must equal
/// the cold build's to 1e-12 — then MovingTessellation::step to an independently perturbed input
/// against a cold build there (gated; the 1e-9 equivalence contract).
int latticeRepairCase(double ex, double ew) {
  const int n = 16, N = n * n * n;
  const real_t h = real_t(1) / n, Larr[3] = {1, 1, 1};
  auto sample = [&](unsigned long long key, std::vector<real_t>& x, std::vector<real_t>& w) {
    x.resize(3 * N);
    w.resize(N);
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
          const int c = i + n * (j + n * k), g[3] = {i, j, k};
          for (int d = 0; d < 3; ++d)
            x[3 * c + d] = (g[d] + real_t(0.5)) * h + ex * h * hashU(key++);
          w[c] = (real_t(-7.1e-14) + ew * hashU(key++)) * h * h;
        }
  };
  std::vector<real_t> x0, w0, x1, w1;
  sample(17, x0, w0);
  sample(91000017, x1, w1);
  auto pos0 = toDevice(x0, "p0"), wt0 = toDevice(w0, "w0");
  auto pos1 = toDevice(x1, "p1"), wt1 = toDevice(w1, "w1");
  char name[64];
  std::snprintf(name, sizeof name, "lattice power ex %.0e ew %.0e", ex, ew);
  Kokkos::View<long*, Mem> gd;
  auto cold0 = peclet::voro::buildTessellation<real_t, true>(pos0, wt0, N, Larr, 4, N, gd,
                                                             peclet::voro::NoSdf{}, true);
  int rc = gate(std::string(name) + " (cold)", cold0.view, cold0.status, N);
  auto vCold = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold0.view.cellVolume);
  peclet::voro::MovingTessellation<real_t, 64, 112, true> mt;
  mt.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
  mt.setWeights(wt0);
  mt.rebuild(pos0);
  auto grid = peclet::voro::buildTessGrid<real_t, true>(pos0, wt0, N, Larr, 4, N, gd);
  Kokkos::View<int*, Mem> all("all", N);
  Kokkos::parallel_for(
      "iota", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
      KOKKOS_LAMBDA(int i) { all(i) = i; });
  mt.gatherSet(grid, all, N, pos0);
  auto vGath = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.vol);
  double dGath = 0;
  for (int i = 0; i < N; ++i)
    dGath = std::max(dGath, std::abs((double)vGath(i) - (double)vCold(i)) / (double)vCold(i));
  rc |= dGath <= 1e-12 ? 0 : 1;
  // step to the second sample (weights and positions both change) vs a cold build there
  mt.setWeights(wt1);
  mt.step(pos1);
  auto cold1 = peclet::voro::buildTessellation<real_t, true>(pos1, wt1, N, Larr, 4, N, gd,
                                                             peclet::voro::NoSdf{}, true);
  rc |= gate(std::string(name) + " (cold, moved)", cold1.view, cold1.status, N);
  auto vc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold1.view.cellVolume);
  auto vs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.vol);
  double dStep = 0;
  for (int i = 0; i < N; ++i)
    dStep = std::max(dStep, std::abs((double)vs(i) - (double)vc(i)) / (double)vc(i));
  rc |= dStep <= 1e-9 ? 0 : 1;
  std::printf("  %-34s gather: max|V-V_cold|/V %.2e (1e-12); step: %.2e (1e-9) %s\n", "", dGath,
              dStep, rc == 0 ? "OK" : "FAIL");
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
    rc |= latticeCellCase();
    for (double ex : {0.0, 1e-16, 1e-14, 1e-12, 1e-9})
      for (double ew : {0.0, 1e-15, 1e-9})
        rc |= latticeClipCase(ex, ew, 0, 4000);
    for (double ex : {0.0, 1e-16, 1e-14, 1e-12})
      rc |= latticeClipCase(ex, 1e-15, 1, 30000);
    for (double ex : {0.0, 1e-16, 1e-12, 1e-9})
      rc |= latticeRepairCase(ex, 1e-15);
    std::printf("%s\n", rc == 0 ? "ALL COSPHERICAL CASES PASS" : "COSPHERICAL CASES FAILED");
  }
  Kokkos::finalize();
  return rc;
}
