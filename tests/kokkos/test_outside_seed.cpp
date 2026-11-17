/**
 * @file test_outside_seed.cpp
 * \brief ConvexCell with the seed OUTSIDE its own cell (sign-safe planes, design note
 *        DESIGN_NOTE_OUTSIDE_SEED §5.1–§5.5, WO-1): exact expectations on known cells, run on the
 *        host and in a device kernel.
 *
 *  (a) initBox(2,2,2), clip p = e_x, d = −0.25 (cell [−1,−0.25]×[−1,1]², seed outside): nn < 0,
 *      volume() = volumePerVertex() = 3, facet areas 4 (cut), 4 (x = −1), 1.5 ×4, cut-facet
 * centroid (−0.25,0,0), faceMoment2 = 8/3, signedSupport = −0.25, self-consistent. (b)
 * initBox(2,2,2), clip p = e_x, d = 0 (evaluated as ε|p|²): V = 4 to 1e-14. (c) after (a), clip p =
 * −e_x, d = −0.25 (x ≥ 0.25): the cell is empty — clip returns true, nt = 0, no overflow, empty();
 * a further clip returns false. (d) two seeds in L = 1, x1 − x0 = 0.25 e_x, w0 = 0: clip with w1 =
 * 0.0225 (d = +0.02), then w1 = 0.1025 (d = −0.02) and reevalGeometry<Power> on the same topology:
 * V = 0.42, equal to a fresh clip. (e) after (a), the touching planes p = (1,1,0), d = 0.75 and p =
 * (5,1,0), d = −0.25 (both touch the edge x = −0.25, y = 1): no overflow, V = 3, self-consistent.
 */
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>

#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/plane_policy.hpp"

using R = double;
using Cell = peclet::voro::ConvexCell<R>;
constexpr int kOut = 64;

/// Area-weighted centroid of face k (convex polygon, fan from its first vertex).
KOKKOS_INLINE_FUNCTION void faceCentroid(const Cell& c, int k, R out[3]) {
  R fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
  const int m = c.faceOrdered(k, fx, fy, fz);
  R A = 0, cx = 0, cy = 0, cz = 0;
  for (int i = 1; i + 1 < m; ++i) {
    const R a[3] = {fx[i] - fx[0], fy[i] - fy[0], fz[i] - fz[0]};
    const R b[3] = {fx[i + 1] - fx[0], fy[i + 1] - fy[0], fz[i + 1] - fz[0]};
    const R cr[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]};
    const R t = R(0.5) * Kokkos::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
    A += t;
    cx += t * (fx[0] + fx[i] + fx[i + 1]) / R(3);
    cy += t * (fy[0] + fy[i] + fy[i + 1]) / R(3);
    cz += t * (fz[0] + fz[i] + fz[i + 1]) / R(3);
  }
  out[0] = cx / A;
  out[1] = cy / A;
  out[2] = cz / A;
}

KOKKOS_INLINE_FUNCTION void caseA(Cell& c) {
  c.initBox(2, 2, 2);
  const R p[3] = {1, 0, 0};
  c.clip(p, R(-0.25), 0);
}

/// Every case; results into out[0..kOut).
KOKKOS_FUNCTION void runCases(R* out) {
  for (int i = 0; i < kOut; ++i)
    out[i] = R(-999);
  {  // (a)
    Cell c;
    caseA(c);
    out[0] = c.nn[6];
    out[1] = c.volume();
    out[2] = c.volumePerVertex();
    R area[Cell::kMaxP];
    for (int k = 0; k < c.np; ++k)
      area[k] = 0;
    c.facetAreasPerVertex(area);
    out[3] = area[6];
    out[4] = area[1];
    for (int k = 2; k < 6; ++k)
      out[3 + k] = area[k];  // out[5..8]
    R cen[3];
    faceCentroid(c, 6, cen);
    out[9] = cen[0];
    out[10] = cen[1];
    out[11] = cen[2];
    out[12] = c.faceMoment2(6);
    out[13] = c.signedSupport(6);
    out[14] = c.isSelfConsistent(R(0)) ? 1 : 0;
    out[15] = c.overflow ? 1 : 0;
    out[16] = c.np;
  }
  {  // (b)
    Cell c;
    c.initBox(2, 2, 2);
    const R p[3] = {1, 0, 0};
    c.clip(p, R(0), 0);
    out[20] = c.volume();
    out[21] = c.volumePerVertex();
    out[22] = c.overflow ? 1 : 0;
    out[23] = c.nn[6];
  }
  {  // (c)
    Cell c;
    caseA(c);
    const R p[3] = {-1, 0, 0};
    out[30] = c.clip(p, R(-0.25), 1) ? 1 : 0;
    out[31] = c.nt;
    out[32] = c.overflow ? 1 : 0;
    out[33] = c.empty() ? 1 : 0;
    const R p2[3] = {0, 1, 0};
    out[34] = c.clip(p2, R(0.1), 2) ? 1 : 0;
    out[35] = c.volumePerVertex();
  }
  {  // (d)
    const R pos[6] = {0.5, 0.5, 0.5, 0.75, 0.5, 0.5};
    R w[2] = {0, 0.0225};
    const R r[3] = {0.25, 0, 0};
    Cell c;
    c.initBox(1, 1, 1);
    c.clip(r, peclet::voro::Power::offsetFromRel<R>(r, w[0], w[1]), 1);
    out[39] = c.volumePerVertex();  // 0.58
    w[1] = R(0.1025);
    c.reevalGeometry<peclet::voro::Power>(pos[0], pos[1], pos[2], pos, R(1), w[0], w);
    out[40] = c.volumePerVertex();
    out[41] = c.volume();
    out[43] = c.nn[6];
    out[44] = c.isSelfConsistent(R(0)) ? 1 : 0;
    Cell f;
    f.initBox(1, 1, 1);
    f.clip(r, peclet::voro::Power::offsetFromRel<R>(r, w[0], w[1]), 1);
    out[42] = f.volumePerVertex();
  }
  {  // (e)
    Cell c;
    caseA(c);
    const R p1[3] = {1, 1, 0}, p2[3] = {5, 1, 0};
    c.clip(p1, R(0.75), 1);
    c.clip(p2, R(-0.25), 2);
    out[50] = c.overflow ? 1 : 0;
    out[51] = c.volume();
    out[52] = c.volumePerVertex();
    out[53] = c.isSelfConsistent(R(0)) ? 1 : 0;
  }
}

static int fails = 0;
static void check(const char* where, const char* what, bool ok, double got, double exp) {
  if (!ok)
    ++fails;
  std::printf("  %-6s %-34s got %.17g exp %.17g %s\n", where, what, got, exp, ok ? "ok" : "FAIL");
}
static bool rel(double a, double b, double t) {
  return std::fabs(a - b) <= t * std::fabs(b);
}

static void verify(const char* w, const R* o) {
  check(w, "(a) nn of the cut plane < 0", o[0] < 0, o[0], -0.0625);
  check(w, "(a) volume()", rel(o[1], 3, 1e-15), o[1], 3);
  check(w, "(a) volumePerVertex()", rel(o[2], 3, 1e-15), o[2], 3);
  check(w, "(a) area cut facet", std::fabs(o[3] - 4) <= 1e-14, o[3], 4);
  check(w, "(a) area x=-1 facet", std::fabs(o[4] - 4) <= 1e-14, o[4], 4);
  for (int k = 5; k <= 8; ++k)
    check(w, "(a) area side facet", std::fabs(o[k] - 1.5) <= 1e-14, o[k], 1.5);
  check(w, "(a) cut centroid x", std::fabs(o[9] + 0.25) <= 1e-15, o[9], -0.25);
  check(w, "(a) cut centroid y", std::fabs(o[10]) <= 1e-15, o[10], 0);
  check(w, "(a) cut centroid z", std::fabs(o[11]) <= 1e-15, o[11], 0);
  check(w, "(a) faceMoment2 cut facet", rel(o[12], 8.0 / 3.0, 1e-15), o[12], 8.0 / 3.0);
  check(w, "(a) signedSupport", o[13] == -0.25, o[13], -0.25);
  check(w, "(a) isSelfConsistent(0)", o[14] == 1, o[14], 1);
  check(w, "(a) no overflow", o[15] == 0, o[15], 0);
  check(w, "(b) d=0 volume()", std::fabs(o[20] - 4) <= 1e-14, o[20], 4);
  check(w, "(b) d=0 volumePerVertex()", std::fabs(o[21] - 4) <= 1e-14, o[21], 4);
  check(w, "(b) d=0 nn > 0 (+eps)", o[23] > 0 && o[22] == 0, o[23], 0);
  check(w, "(c) emptying clip returns true", o[30] == 1, o[30], 1);
  check(w, "(c) nt == 0", o[31] == 0, o[31], 0);
  check(w, "(c) no overflow", o[32] == 0, o[32], 0);
  check(w, "(c) empty()", o[33] == 1, o[33], 1);
  check(w, "(c) further clip returns false", o[34] == 0, o[34], 0);
  check(w, "(c) volume 0", o[35] == 0, o[35], 0);
  check(w, "(d) d=+0.02 volume", rel(o[39], 0.58, 1e-14), o[39], 0.58);
  check(w, "(d) reeval d=-0.02 volumePerVertex", rel(o[40], 0.42, 1e-14), o[40], 0.42);
  check(w, "(d) reeval d=-0.02 volume()", rel(o[41], 0.42, 1e-14), o[41], 0.42);
  check(w, "(d) reeval == fresh clip", rel(o[40], o[42], 1e-14), o[40], o[42]);
  check(w, "(d) nn < 0 after reeval", o[43] < 0, o[43], -0.0064);
  check(w, "(d) isSelfConsistent(0)", o[44] == 1, o[44], 1);
  check(w, "(e) no overflow", o[50] == 0, o[50], 0);
  check(w, "(e) volume()", rel(o[51], 3, 1e-14), o[51], 3);
  check(w, "(e) volumePerVertex()", rel(o[52], 3, 1e-14), o[52], 3);
  check(w, "(e) isSelfConsistent(0)", o[53] == 1, o[53], 1);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    R host[kOut];
    runCases(host);
    Kokkos::View<R*> dev("out", kOut);
    Kokkos::parallel_for(
        "outside_seed.cases", Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) { runCases(dev.data()); });
    auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev);
    std::printf("test_outside_seed\n");
    verify("host", host);
    verify("device", hd.data());
    double dmax = 0;
    for (int i = 0; i < kOut; ++i)
      dmax = std::fmax(dmax, std::fabs(host[i] - hd(i)));
    std::printf("RESULT test_outside_seed %s fails=%d maxHostDeviceDiff=%.3e\n",
                fails == 0 ? "PASS" : "FAIL", fails, dmax);
  }
  Kokkos::finalize();
  return fails == 0 ? 0 : 1;
}
