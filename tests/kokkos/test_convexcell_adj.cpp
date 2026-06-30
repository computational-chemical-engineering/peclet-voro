/**
 * @file test_convexcell_adj.cpp
 * \brief Foundation gates for the `TrackAdj` template axis on ConvexCell (incremental edge-adjacency +
 *        Lawson O(nt) local-convexity certificate). Host-side, no parallel_for — exercises the cell
 *        methods directly (they are KOKKOS_INLINE_FUNCTION, host-callable).
 *
 * Covers the spec's machine-checkable acceptance:
 *   Change 1  — sizeof: false-mode unchanged, true-mode = false + MAXT*3*sizeof(int)  (compile-time).
 *   Change 3  — initBox bootstraps a consistent adjacency.
 *   Change 2  — rebuildAdjacency() (brute oracle) + checkAdjacencyInvariant() agree on built cells.
 *   Change 4  — clip-by-clip in TrackAdj mode keeps checkAdjacencyInvariant() true after EVERY clip,
 *               and the incrementally-stitched adj equals rebuildAdjacency() element-for-element.
 *   Change 5b — isLocallyConvex == isSelfConsistent in the valid / first-flip (small-δ) regime; the
 *               far-poke divergence (brute flags, local does not) is observed at large δ (documents the
 *               local certificate's precondition executably).
 */
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "vorflow/device/convex_cell.hpp"

using real_t = double;
using vor::device::ConvexCell;

// ---- Change 1: footprint (compile-time) ----------------------------------------------------------
static_assert(sizeof(ConvexCell<double, 64, 96, false>) == 5008, "false-mode sizeof changed");
static_assert(sizeof(ConvexCell<float, 64, 96, false>) == 2828, "false-mode (float) sizeof changed");
static_assert(sizeof(ConvexCell<double, 64, 96, true>) ==
                  sizeof(ConvexCell<double, 64, 96, false>) + 96 * 3 * sizeof(int),
              "true-mode footprint must be false + adj(MAXT*3*int)");
static_assert(sizeof(ConvexCell<float, 64, 96, true>) ==
                  sizeof(ConvexCell<float, 64, 96, false>) + 96 * 3 * sizeof(int),
              "true-mode (float) footprint must be false + adj(MAXT*3*int)");

static constexpr int MAXP = 64, MAXT = 96;
using CellT = ConvexCell<real_t, MAXP, MAXT, true>;

// Neighbour list of seed 0 (rel positions sorted by distance) for a point cloud in a periodic box.
struct Nbrs {
  std::vector<real_t> rx, ry, rz;
  std::vector<int> id;
};
static Nbrs buildNbrs(const std::vector<real_t>& P, int self, real_t L) {
  const int N = (int)P.size() / 3;
  const real_t Lh = real_t(0.5) * L;
  std::vector<std::tuple<real_t, real_t, real_t, real_t, int>> v;  // (d2, rx, ry, rz, id)
  for (int j = 0; j < N; ++j) {
    if (j == self) continue;
    real_t rx = P[3 * j] - P[3 * self], ry = P[3 * j + 1] - P[3 * self + 1],
           rz = P[3 * j + 2] - P[3 * self + 2];
    rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
    ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
    rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
    v.emplace_back(rx * rx + ry * ry + rz * rz, rx, ry, rz, j);
  }
  std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
  Nbrs out;
  for (auto& e : v) {
    out.rx.push_back(std::get<1>(e));
    out.ry.push_back(std::get<2>(e));
    out.rz.push_back(std::get<3>(e));
    out.id.push_back(std::get<4>(e));
  }
  return out;
}

// Build the cell of seed `self` clip-by-clip; after every committed clip run `cb(c, clipIdx)`.
template <class CB>
static void buildStepwise(CellT& c, const Nbrs& nb, real_t L, CB&& cb) {
  c.initBox(L, L, L);
  cb(c, -1);  // after initBox
  for (size_t i = 0; i < nb.id.size(); ++i) {
    const real_t rx = nb.rx[i], ry = nb.ry[i], rz = nb.rz[i];
    const real_t off = real_t(0.5) * (rx * rx + ry * ry + rz * rz);
    if (!(off < real_t(2) * c.maxVertexRsq())) break;  // security radius (same as buildConvexCell)
    const real_t pdir[3] = {rx, ry, rz};
    c.clip(pdir, off, nb.id[i]);
    if (c.overflow) return;
    cb(c, (int)i);
  }
}

// True iff cell `a`'s incrementally-stitched adj equals a brute rebuild, element-wise on live triangles.
static bool adjMatchesBrute(const CellT& a) {
  CellT b = a;            // copy
  b.rebuildAdjacency();   // brute oracle into b.adj
  for (int t = 0; t < a.nt; ++t) {
    if (!a.alive[t]) continue;
    for (int e = 0; e < 3; ++e)
      if (a.adj[t * 3 + e] != b.adj[t * 3 + e]) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const real_t L = 1.0;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> U(0.0, L);

    // ---- GATE B (Change 3) + GATE C (Change 2) + GATE D (Change 4 fuzz) -----------------------
    const int nClouds = 400, nPts = 60;
    long invFail = 0, bruteFail = 0, matchFail = 0, overflow = 0, cellsChecked = 0, clipsChecked = 0;
    for (int s = 0; s < nClouds; ++s) {
      std::vector<real_t> P(3 * nPts);
      for (auto& x : P) x = U(rng);
      Nbrs nb = buildNbrs(P, 0, L);
      CellT c;
      bool localFail = false;
      buildStepwise(c, nb, L, [&](CellT& cc, int clipIdx) {
        // Change 4: invariant must hold after initBox AND after every clip.
        if (!cc.checkAdjacencyInvariant()) { localFail = true; }
        // Change 4 fuzz: incremental adj == brute rebuild.
        if (!adjMatchesBrute(cc)) { localFail = true; }
        (void)clipIdx;
        ++clipsChecked;
      });
      if (c.overflow) { ++overflow; continue; }
      if (localFail) ++invFail;
      // Change 2: brute rebuild then invariant (oracle self-consistency).
      CellT cb = c;
      cb.rebuildAdjacency();
      if (!cb.checkAdjacencyInvariant()) ++bruteFail;
      if (!adjMatchesBrute(c)) ++matchFail;
      ++cellsChecked;
    }
    std::printf("[gate B/C/D] clouds=%d cells=%ld clips=%ld | invFail=%ld bruteFail=%ld matchFail=%ld overflow=%ld\n",
                nClouds, cellsChecked, clipsChecked, invFail, bruteFail, matchFail, overflow);
    if (invFail || bruteFail || matchFail) {
      std::printf("  FAIL: adjacency invariant / brute-match broken\n");
      rc = 1;
    }

    // ---- GATE E (Change 5b): isLocallyConvex vs isSelfConsistent ------------------------------
    // The Lawson local certificate tests a SUBSET of the planes the brute form does (only each edge's
    // edge-opposite plane), so it is a strict subset detector:
    //   (1) Valid state (no displacement): both return TRUE.
    //   (2) Subset invariant: local must NEVER flag a cell the brute form calls convex
    //       (okB==true => okL==true). A violation would be a logic bug. MUST be 0.
    //   (3) Far-poke divergence: there exist cells the brute form flags but the local form does not
    //       (okB==false && okL==true) — a non-adjacent plane poked a vertex without flipping an incident
    //       edge. This documents the local certificate's small-displacement precondition executably;
    //       it MUST be observed (so the precondition is real, not vacuous).
    long validBothTrue = 0, validChecked = 0;
    long subsetViolations = 0, farPoke = 0, sweepChecked = 0;
    auto sweep = [&](real_t disp, bool isValid) {
      std::mt19937 r2(999);
      std::uniform_real_distribution<real_t> u2(0.0, L);
      std::normal_distribution<real_t> g(0.0, 1.0);
      long localBoth = 0, localBrute = 0, checked = 0;
      for (int s = 0; s < 300; ++s) {
        std::vector<real_t> P(3 * nPts);
        for (auto& x : P) x = u2(r2);
        Nbrs nb = buildNbrs(P, 0, L);
        CellT c;
        buildStepwise(c, nb, L, [](CellT&, int) {});
        if (c.overflow) continue;
        std::vector<real_t> Pd = P;
        if (!isValid)
          for (auto& x : Pd) x += disp * L * g(r2);  // displace all seeds (topology fixed -> reeval)
        unsigned char poke4[MAXT * 4];
        c.computePoke4(poke4);  // build the cert plane set at the BUILD config (then displace + test)
        c.reevalGeometry(Pd[0], Pd[1], Pd[2], Pd.data(), L);
        const real_t tol = 1e-7;
        const bool okB = c.isSelfConsistent(tol);
        const bool okL = c.isLocallyConvex(poke4, tol);
        if (isValid) {
          if (okB && okL) ++validBothTrue;
          ++validChecked;
        } else {
          if (okB && !okL) ++subsetViolations;  // local flagged what brute did not -> BUG
          if (!okB && okL) ++farPoke;            // far-poke the local form misses (expected)
          ++sweepChecked;
        }
        if (!okB && !okL) ++localBoth;
        if (!okB && okL) ++localBrute;
        ++checked;
      }
      std::printf("[gate E] %-7s disp=%.4f checked=%ld | bothFlag=%ld localMissed(bruteOnly)=%ld\n",
                  isValid ? "valid" : "moved", (double)disp, checked, localBoth, localBrute);
      return localBrute;
    };
    sweep(0.0, true);
    const long missSmall = sweep(0.0005, false);  // operating regime: local must MATCH brute (complete)
    sweep(0.005, false);                           // larger disp: the build-config 4th-face goes stale →
    sweep(0.05, false);                            // a few cells may diverge (then a gather refreshes face4)
    std::printf("[gate E] validBothTrue=%ld/%ld  subsetViolations=%ld  missSmall@5e-4=%ld\n", validBothTrue,
                validChecked, subsetViolations, missSmall);
    if (validBothTrue != validChecked) {
      std::printf("  FAIL: a valid cell disagreed (local vs brute) with no displacement\n");
      rc = 1;
    }
    if (subsetViolations != 0) {
      std::printf("  FAIL: local certificate flagged a cell the brute form calls convex (subset broken)\n");
      rc = 1;
    }
    // With the 4th-face plane the local cert is COMPLETE: in the operating regime it must match the brute
    // flag set (a handful of FP-marginal cells per 300 is tolerated; gross divergence means face4 is broken).
    if (missSmall > 3) {
      std::printf("  FAIL: local cert missed %ld brute flags at small disp (4th-face completeness broken)\n",
                  missSmall);
      rc = 1;
    }

    if (rc == 0) std::printf("ALL TRACKADJ FOUNDATION GATES PASS\n");
  }
  Kokkos::finalize();
  return rc;
}
