/**
 * @file test_repair_vanishing.cpp
 * @brief The repair publishes a vanishing / appearing power cell exactly as the cold build does
 *        (DESIGN_NOTE_OUTSIDE_SEED §10, WO-3c).
 *
 * An exact 1↔4 flip of size far below the certificate tolerance: a hidden point i enters (gain)
 * or leaves (loss) the regular tetrahedron j1…j4 around their common Voronoi vertex v.
 *
 * Construction (fully fixed by the note, §10.5 item 1):
 *   - L = 1, a 4³ jittered lattice (h = 0.25, offsets uniform in ±0.2 h, mt19937_64 seed
 *     20261117), equal weights.
 *   - Cell 0 by a host brute-force oracle (all N − 1 neighbours, min image). Each live triangle
 *     that touches no box plane gives a vertex v = x_0 + (vx, vy, vz), the tetrahedron
 *     (j1 = 0, j2…j4 = its three pnbr in stored order) and R² = |v − x_0|². Margin
 *     m = min over the other 60 seeds of |v − x_k|² − R²; take the triangle with the largest m
 *     (require m ≥ 1e-3 h²).
 *   - Seed i = 64 (N = 65): x_i = 0.4 x_j1 + 0.3 x_j2 + 0.2 x_j3 + 0.1 x_j4 (unwrapped about v),
 *     u = x_i − v (require |u| ≥ 0.01 h). Weights R² for every lattice seed, w_i = |u|² + δ_A, so
 *     i owns v iff δ = w_i − |x_i − v|² > 0.
 *   - Frame A: x_i as above. Frame B moves only i to v + (|u| + s) û, s = −|u| + √(|u|² + 2δ_A):
 *     δ_B = −δ_A. Gain δ_A = −2e-6 h², loss δ_A = +2e-6 h².
 *   - MovingTessellation (tol 1e-4 h, skin 0.25 h) rebuilt at A, one step to B, against a cold
 *     build at B; variants localCert × useNearMiss.
 * Assertions at B: (1) i empty in the repair ⟺ empty in cold; (2) i and j1…j4 carry cold's
 * facet-neighbour sets, |V_rep − V_cold| ≤ 1e-12·V_cold for a non-empty i and ≤ 1e-15 h³ for the
 * j's; (3) every published V ≥ −1e-24 h³.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"

namespace {

constexpr int kP = peclet::voro::kMaxPlanes;
constexpr int kT = peclet::voro::kMaxTriangles;
using Engine = peclet::voro::MovingTessellation<Real, kP, kT, true>;
using OracleCell = peclet::voro::ConvexCell<Real, 200, 400>;

constexpr int kN = 4;              // lattice per axis
constexpr int kNL = kN * kN * kN;  // lattice seeds
constexpr int kNTot = kNL + 1;     // + the hidden point i
constexpr Real kH = Real(1) / kN;

Real minImg(Real d) {
  return d > Real(0.5) ? d - 1 : (d < Real(-0.5) ? d + 1 : d);
}
Real wrap(Real p) {
  return p < 0 ? p + 1 : (p >= 1 ? p - 1 : p);
}

/// Host brute-force Voronoi cell of seed 0 (equal weights): every other lattice seed, closest
/// first, min image, in a seed-centred box of extent L.
void bruteCell0(const std::vector<Real>& x, OracleCell& c) {
  std::vector<std::pair<Real, int>> ord;
  for (int j = 1; j < kNL; ++j) {
    Real rho = 0;
    for (int d = 0; d < 3; ++d) {
      const Real r = minImg(x[3 * j + d] - x[d]);
      rho += r * r;
    }
    ord.emplace_back(rho, j);
  }
  std::sort(ord.begin(), ord.end());
  c.initBox(1, 1, 1);
  for (const auto& pr : ord) {
    Real r[3];
    for (int d = 0; d < 3; ++d)
      r[d] = minImg(x[3 * pr.second + d] - x[d]);
    c.clip(r, Real(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), pr.second);
  }
}

struct Setup {
  std::vector<Real> xA, xB, w;
  int j[4];
  Real v[3], margin, uNorm;
};

/// Builds frames A and B for one δ_A (in units of h²). Returns false (with a message) if a
/// construction requirement fails.
bool construct(Real deltaA, Setup& s) {
  std::mt19937_64 rng(20261117ULL);
  std::uniform_real_distribution<Real> U(-1, 1);
  std::vector<Real> x(3 * kNL);
  for (int k = 0; k < kN; ++k)
    for (int jj = 0; jj < kN; ++jj)
      for (int ii = 0; ii < kN; ++ii) {
        const int c = ii + kN * (jj + kN * k);
        const int ijk[3] = {ii, jj, k};
        for (int d = 0; d < 3; ++d)
          x[3 * c + d] = wrap((ijk[d] + Real(0.5)) * kH + Real(0.2) * kH * U(rng));
      }
  OracleCell c;
  bruteCell0(x, c);
  if (c.overflow || c.nt == 0) {
    std::printf("  construct: oracle cell 0 overflow/empty\n");
    return false;
  }
  int best = -1;
  Real bestM = -1e300;
  for (int t = 0; t < c.nt; ++t) {
    if (!c.alive[t])
      continue;
    const int q[3] = {c.pnbr[c.t0[t]], c.pnbr[c.t1[t]], c.pnbr[c.t2[t]]};
    if (q[0] < 0 || q[1] < 0 || q[2] < 0)
      continue;  // touches a box plane
    const Real vv[3] = {x[0] + c.vx[t], x[1] + c.vy[t], x[2] + c.vz[t]};
    const Real R2 = c.vx[t] * c.vx[t] + c.vy[t] * c.vy[t] + c.vz[t] * c.vz[t];
    Real m = 1e300;
    for (int k = 1; k < kNL; ++k) {
      if (k == q[0] || k == q[1] || k == q[2])
        continue;
      Real d2 = 0;
      for (int d = 0; d < 3; ++d) {
        const Real r = minImg(vv[d] - x[3 * k + d]);
        d2 += r * r;
      }
      m = std::min(m, d2 - R2);
    }
    if (m > bestM) {
      bestM = m;
      best = t;
    }
  }
  if (best < 0 || bestM < Real(1e-3) * kH * kH) {
    std::printf("  construct: largest margin %.3e h^2 < 1e-3 h^2\n", bestM / (kH * kH));
    return false;
  }
  s.j[0] = 0;
  s.j[1] = c.pnbr[c.t0[best]];
  s.j[2] = c.pnbr[c.t1[best]];
  s.j[3] = c.pnbr[c.t2[best]];
  s.v[0] = x[0] + c.vx[best];
  s.v[1] = x[1] + c.vy[best];
  s.v[2] = x[2] + c.vz[best];
  s.margin = bestM;
  const Real R2 = c.vx[best] * c.vx[best] + c.vy[best] * c.vy[best] + c.vz[best] * c.vz[best];
  const Real coef[4] = {Real(0.4), Real(0.3), Real(0.2), Real(0.1)};
  Real xi[3] = {0, 0, 0};
  for (int k = 0; k < 4; ++k)
    for (int d = 0; d < 3; ++d)
      xi[d] += coef[k] * (s.v[d] + minImg(x[3 * s.j[k] + d] - s.v[d]));  // unwrapped about v
  const Real u[3] = {xi[0] - s.v[0], xi[1] - s.v[1], xi[2] - s.v[2]};
  const Real un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  s.uNorm = un;
  if (un < Real(0.01) * kH) {
    std::printf("  construct: |u| = %.3e h < 0.01 h\n", un / kH);
    return false;
  }
  const Real dA = deltaA * kH * kH;
  s.w.assign(kNTot, R2);
  s.w[kNL] = un * un + dA;
  const Real sstep = -un + std::sqrt(un * un + 2 * dA);
  s.xA = x;
  s.xB = x;
  for (int d = 0; d < 3; ++d) {
    s.xA.push_back(wrap(xi[d]));
    s.xB.push_back(wrap(s.v[d] + (un + sstep) * (u[d] / un)));
  }
  return true;
}

struct CellPub {
  std::vector<Real> vol;
  std::vector<std::vector<long>> nbr;  // facet neighbours over facets with A ≠ 0, sorted
  bool empty(int i) const { return vol[i] == Real(0) && nbr[i].empty(); }
};

CellPub summarize(const peclet::voro::TessellationView<Real>& v) {
  CellPub p;
  p.vol = down(v.cellVolume);
  const auto off = down(v.cellFacetOffset), cnt = down(v.cellFacetCount);
  const auto nbr = down(v.facetNeighbor);
  const auto area = down(v.facetArea);
  p.nbr.resize(kNTot);
  for (int i = 0; i < kNTot; ++i) {
    for (int f = off[i]; f < off[i] + cnt[i]; ++f)
      if (area[3 * f] != 0 || area[3 * f + 1] != 0 || area[3 * f + 2] != 0)
        p.nbr[i].push_back(static_cast<long>(nbr[f]));
    std::sort(p.nbr[i].begin(), p.nbr[i].end());
  }
  return p;
}

CellPub cold(const std::vector<Real>& x, const DV& w) {
  const Real L[3] = {1, 1, 1};
  const auto res =
      peclet::voro::buildTessellation<Real, true, peclet::voro::NoSdf>(up(x, "pos"), w, kNTot, L);
  return summarize(res.view);
}

std::vector<long> sortedJ(const Setup& s) {
  std::vector<long> q(s.j, s.j + 4);
  std::sort(q.begin(), q.end());
  return q;
}

/// One case (gain or loss): preconditions, then the four engine variants. Returns failures.
int runCase(const char* name, Real deltaA) {
  Setup s;
  if (!construct(deltaA, s)) {
    std::printf("CASE %s: construction FAILED\n", name);
    return 1;
  }
  const int I = kNL;
  const DV w = up(s.w, "w");
  const CellPub cA = cold(s.xA, w), cB = cold(s.xB, w);
  const Real h3 = kH * kH * kH;
  std::printf(
      "CASE %s: j = {%d %d %d %d} margin %.3e h^2 |u| %.3e h | cold A: V_i %.3e h^3 "
      "nf %zu | cold B: V_i %.3e h^3 nf %zu\n",
      name, s.j[0], s.j[1], s.j[2], s.j[3], s.margin / (kH * kH), s.uNorm / kH, cA.vol[I] / h3,
      cA.nbr[I].size(), cB.vol[I] / h3, cB.nbr[I].size());
  // Preconditions: the δ > 0 frame has i non-empty with exactly {j1…j4}; the δ < 0 frame empty;
  // no skin mover (only i moves, by |s| ≪ skin/2).
  const CellPub& pos = deltaA > 0 ? cA : cB;
  const CellPub& neg = deltaA > 0 ? cB : cA;
  Real disp2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real r = minImg(s.xB[3 * I + d] - s.xA[3 * I + d]);
    disp2 += r * r;
  }
  const Real skin = Real(0.25) * kH;
  const bool pre = !pos.empty(I) && pos.vol[I] > 0 && pos.nbr[I] == sortedJ(s) && neg.empty(I) &&
                   disp2 <= Real(0.25) * skin * skin;
  if (!pre) {
    std::printf(
        "CASE %s: precondition FAILED (pos-frame i nonempty=%d nbrs-ok=%d, neg-frame i "
        "empty=%d, disp %.3e h)\n",
        name, !pos.empty(I), pos.nbr[I] == sortedJ(s), neg.empty(I), std::sqrt(disp2) / kH);
    return 1;
  }
  int fails = 0;
  const Real L[3] = {1, 1, 1};
  for (int lc = 1; lc >= 0; --lc)
    for (int nm = 1; nm >= 0; --nm) {
      Engine mt;
      mt.alloc(kNTot, L, Real(1e-4) * kH, skin, 4, kNTot);
      mt.setWeights(w);
      mt.localCert = lc != 0;
      mt.useNearMiss = nm != 0;
      DV p = up(s.xA, "pos");
      mt.rebuild(p);
      Kokkos::deep_copy(p, Kokkos::View<const Real*, Kokkos::HostSpace>(s.xB.data(), s.xB.size()));
      const auto st = mt.step(p);
      using Policy = typename Engine::PlanePolicy;
      const auto pub = peclet::voro::reevalPublish<Real, kP, kT, Policy>(
          mt.store, p, mt.vol, kNTot, L, mt.wall, mt.xRef, false, false, w);
      const CellPub r = summarize(pub);
      const bool a1 = r.empty(I) == cB.empty(I);
      bool a2 = r.nbr[I] == cB.nbr[I] &&
                (cB.empty(I) || std::fabs(r.vol[I] - cB.vol[I]) <= Real(1e-12) * cB.vol[I]);
      Real djMax = 0;
      for (int k = 0; k < 4; ++k) {
        const int j = s.j[k];
        a2 = a2 && r.nbr[j] == cB.nbr[j];
        djMax = std::max(djMax, std::fabs(r.vol[j] - cB.vol[j]));
      }
      a2 = a2 && djMax <= Real(1e-15) * h3;
      Real vMin = 1e300;
      for (int k = 0; k < kNTot; ++k)
        vMin = std::min(vMin, r.vol[k]);
      const bool a3 = vMin >= Real(-1e-24) * h3;
      const bool ok = a1 && a2 && a3;
      fails += ok ? 0 : 1;
      // reported: i's relative volume difference and its seed distance over its size (the
      // seed-frame conditioning, amendment 3: relative error ∝ ε·(dist/size)³)
      const Real relI = cB.vol[I] > 0 ? std::fabs(r.vol[I] - cB.vol[I]) / cB.vol[I] : Real(0);
      const Real dOverS = cB.vol[I] > 0 ? s.uNorm / std::cbrt(cB.vol[I]) : Real(0);
      std::printf(
          "  %s localCert=%d nearMiss=%d: route=%d pass1=%d pass2=%d near=%d | i: rep "
          "empty=%d V %.6e h^3 nf %zu, cold empty=%d V %.6e h^3 nf %zu, rel %.3e, "
          "dist/size %.3e | max|dV_j| %.3e h^3 | minV %.3e h^3 | (1)%s (2)%s (3)%s %s\n",
          name, lc, nm, static_cast<int>(st.route), st.pass1, st.pass2, st.nearFlagged, r.empty(I),
          r.vol[I] / h3, r.nbr[I].size(), cB.empty(I), cB.vol[I] / h3, cB.nbr[I].size(), relI,
          dOverS, djMax / h3, vMin / h3, a1 ? "ok" : "MISS", a2 ? "ok" : "MISS", a3 ? "ok" : "MISS",
          ok ? "PASS" : "FAIL");
    }
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    std::printf("test_repair_vanishing\n");
    fails += runCase("gain", Real(-2e-6));
    fails += runCase("loss", Real(2e-6));
  }
  Kokkos::finalize();
  std::printf("%s (%d failing variants)\n", fails == 0 ? "PASS" : "FAIL", fails);
  return fails == 0 ? 0 : 1;
}
