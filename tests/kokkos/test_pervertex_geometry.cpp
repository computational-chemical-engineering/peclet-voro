/**
 * @file test_pervertex_geometry.cpp
 * \brief Acceptance criteria for the vertex-local, SORT-FREE geometry (design note §7).
 *
 * Validates ConvexCell::volumePerVertex / facetAreasPerVertex (flag/divergence construction, no
 * vertex ordering, no adjacency) against the ordered reference (volume()/faceOrdered area), the
 * internal divergence identity, label invariance, obtuse/skinny robustness, and a finite-difference
 * gradient check. FP64. Cells are built by clipping random nearest neighbours (incl. an anisotropic
 * batch that produces feet-outside-face cells), and (13) power cells whose seed lies outside the
 * cell (translation oracle, FD and AD of the foot-point gradients). Host-side, single thread.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/plane_policy.hpp"

using real_t = double;
using Cell = peclet::voro::ConvexCell<real_t, 64, 128>;

// build one cell at site `i` from its K nearest neighbours (periodic min-image, security break)
static bool buildCell(Cell& c, int i, const std::vector<real_t>& pos, real_t L,
                      const std::vector<int>& nbr) {
  const real_t sx = pos[3 * i], sy = pos[3 * i + 1], sz = pos[3 * i + 2], Lh = 0.5 * L;
  c.initBox(L, L, L);
  real_t secR2 = 2.0 * c.maxVertexRsq();
  for (int g : nbr) {
    real_t rx = pos[3 * g] - sx, ry = pos[3 * g + 1] - sy, rz = pos[3 * g + 2] - sz;
    rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
    ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
    rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
    const real_t off = 0.5 * (rx * rx + ry * ry + rz * rz);
    if (off >= secR2)
      break;
    const real_t n[3] = {rx, ry, rz};
    if (c.clip(n, off, g))
      secR2 = 2.0 * c.maxVertexRsq();
    if (c.overflow)
      return false;
  }
  return !c.overflow;
}

// ---------------------------------------------------------------------------------------------
// (13) Outside-seed power cells (DESIGN_NOTE_OUTSIDE_SEED §5.5, work order WO-4, gate G9): every
// per-vertex value kernel against the TRANSLATION ORACLE (the same cell rebuilt in the frame of an
// interior point y, where every plane has d > 0, so no σ = −1 code path runs), and the gradients —
// derivatives w.r.t. the foot point q_k = σ_k n_k — against central differences of the value
// kernels on the fixed topology (q_k → q_k ± δ e_a, plane rewritten as n = σq, nn = σ|q|², live
// vertices recomputed) and against the dual-number AD oracle. Plus a REPORT (not gated, amendment
// 4) of Power::chain's neighbour-position force against FD as d → 0.
namespace outside_seed {

struct Plane {
  real_t p[3];
  real_t d;
  int j;
};

// All N−1 radical planes {x : p·x ≤ d} of site i (p = min-image connector, d = ½(|p|² + w_i −
// w_j)), sorted by |p|² then id.
static std::vector<Plane> powerPlanes(int i, const std::vector<real_t>& x,
                                      const std::vector<real_t>& w, int N, real_t L) {
  std::vector<Plane> P;
  P.reserve(N - 1);
  for (int j = 0; j < N; ++j) {
    if (j == i)
      continue;
    Plane q;
    for (int a = 0; a < 3; ++a) {
      real_t r = x[3 * j + a] - x[3 * i + a];
      q.p[a] = r - L * std::round(r / L);
    }
    q.d = 0.5 * (q.p[0] * q.p[0] + q.p[1] * q.p[1] + q.p[2] * q.p[2] + w[i] - w[j]);
    q.j = j;
    P.push_back(q);
  }
  std::sort(P.begin(), P.end(), [](const Plane& a, const Plane& b) {
    const real_t ra = a.p[0] * a.p[0] + a.p[1] * a.p[1] + a.p[2] * a.p[2];
    const real_t rb = b.p[0] * b.p[0] + b.p[1] * b.p[1] + b.p[2] * b.p[2];
    return ra < rb || (ra == rb && a.j < b.j);
  });
  return P;
}

// Clip every plane, in order, in the frame of y (offsets d − p·y), into a box of edge B about y.
static bool buildIn(Cell& c, const std::vector<Plane>& P, const real_t y[3], real_t B) {
  c.initBox(B, B, B);
  for (const Plane& q : P) {
    c.clip(q.p, q.d - (q.p[0] * y[0] + q.p[1] * y[1] + q.p[2] * y[2]), q.j);
    if (c.overflow)
      return false;
  }
  return true;
}

static bool hasNegFacet(const Cell& c) {
  for (int t = 0; t < c.nt; ++t)
    if (c.alive[t] && (c.nn[c.t0[t]] < 0 || c.nn[c.t1[t]] < 0 || c.nn[c.t2[t]] < 0))
      return true;
  return false;
}

static std::vector<char> facetPlanes(const Cell& c) {
  std::vector<char> f(c.np, 0);
  for (int t = 0; t < c.nt; ++t)
    if (c.alive[t])
      f[c.t0[t]] = f[c.t1[t]] = f[c.t2[t]] = 1;
  return f;
}

// Move plane k's foot point by s along axis a (keeping σ_k), then recompute the live vertices.
static void moveFoot(Cell& c, int k, int a, real_t s) {
  real_t q[3];
  c.footPoint(k, q);
  const real_t sg = c.nn[k] < 0 ? -1.0 : 1.0;
  q[a] += s;
  for (int d = 0; d < 3; ++d)
    c.n[k][d] = sg * q[d];
  c.nn[k] = sg * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
  for (int t = 0; t < c.nt; ++t)
    if (c.alive[t])
      c.computeVertex(t);
}

struct Vals {   // per-cell values keyed by plane index
  double V[5];  // volumePerVertex, volume, geomVolumeGrad, geomVolumeArea, geometryPerVertex
  double A[64], Am[64], Ag[64], av[64][3], m[64][3], mg[64][3], M2c[64];
};

static void values(const Cell& c, Vals& o) {
  double a1[64] = {}, a2[64] = {}, mx[64] = {}, my[64] = {}, mz[64] = {};
  double gx[64] = {}, gy[64] = {}, gz[64] = {};
  double avx[64] = {}, avy[64] = {}, avz[64] = {}, dx[64] = {}, dy[64] = {}, dz[64] = {};
  double a4[64] = {}, nx[64] = {}, ny[64] = {}, nz[64] = {};
  o.V[0] = c.volumePerVertex();
  o.V[1] = c.volume();
  c.geomVolumeGrad(o.V[2], dx, dy, dz);
  c.geomVolumeArea(o.V[3], avx, avy, avz, gx, gy, gz);
  c.geometryPerVertex(o.V[4], a4, nx, ny, nz);
  c.facetAreasPerVertex(a1);
  c.facetMomentsPerVertex(a2, mx, my, mz);
  for (int k = 0; k < c.np; ++k) {
    o.A[k] = a1[k];
    o.Am[k] = a2[k];
    o.Ag[k] = a4[k];
    o.av[k][0] = avx[k];
    o.av[k][1] = avy[k];
    o.av[k][2] = avz[k];
    o.m[k][0] = mx[k];
    o.m[k][1] = my[k];
    o.m[k][2] = mz[k];
    o.mg[k][0] = nx[k];
    o.mg[k][1] = ny[k];
    o.mg[k][2] = nz[k];
    // central second moment ∫|x − x̄|² dA = faceMoment2 (about q) − A|x̄ − q|² (frame-invariant)
    o.M2c[k] = 0;
    if (a2[k] > 0) {
      real_t q[3];
      c.footPoint(k, q);
      double e2 = 0;
      for (int d = 0; d < 3; ++d) {
        const double e = o.m[k][d] / a2[k] - q[d];
        e2 += e * e;
      }
      o.M2c[k] = c.faceMoment2(k) - a2[k] * e2;
    }
  }
}

static int run() {
  const int n = 8, N = n * n * n;
  const real_t L = 1.0, h = L / n, W = 1.0, h2 = h * h, h3 = h2 * h, h4 = h3 * h;
  std::mt19937 rng(20261117);
  std::uniform_real_distribution<real_t> U(0, 1);
  std::vector<real_t> x(3 * N), w(N);
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const int s = i + n * (j + n * k);
        const int ijk[3] = {i, j, k};
        for (int a = 0; a < 3; ++a) {
          real_t v = (ijk[a] + 0.5 + 0.6 * (U(rng) - 0.5)) * h;  // jitter ±0.3h
          x[3 * s + a] = v - L * std::floor(v / L);
        }
      }
  for (auto& v : w)
    v = W * h2 * U(rng);

  const real_t zero[3] = {0, 0, 0};
  const real_t delta = 1e-6 * h;
  long nOS = 0, nEmpty = 0, nOverflow = 0, nNotInterior = 0, nSetDiff = 0, nBoxFace = 0;
  double eV = 0, eA = 0, eM = 0, eM2 = 0, eAv = 0;  // translation oracle (abs, in h units)
  double eGradV = 0, eGradVA = 0, eAD = 0, eFDan = 0, eFDad = 0, maxDistSize = 0;
  // REPORT: the dV/dq FD error at other steps (truncation ∝ δ², roundoff ∝ ε·dist³/δ), and the
  // cell where the gated step's error peaks.
  const real_t sweep[3] = {1e-4 * h, 1e-5 * h, 1e-7 * h};
  double eSweep[3] = {0, 0, 0}, worstDistSize = 0, worstV = 0;
  for (int i = 0; i < N; ++i) {
    const std::vector<Plane> P = powerPlanes(i, x, w, N, L);
    Cell c;
    if (!buildIn(c, P, zero, L)) {
      ++nOverflow;
      continue;
    }
    if (c.empty()) {
      ++nEmpty;
      continue;
    }
    if (!hasNegFacet(c))
      continue;
    const std::vector<char> fp = facetPlanes(c);
    Vals a;
    values(c, a);
    bool box = false;
    for (int k = 0; k < c.np; ++k)
      box |= fp[k] && c.pnbr[k] < 0 && a.A[k] > 0;
    if (box) {
      ++nBoxFace;
      continue;
    }
    ++nOS;
    // ---- translation oracle
    real_t y[3] = {0, 0, 0};
    long nv = 0;
    for (int t = 0; t < c.nt; ++t)
      if (c.alive[t]) {
        y[0] += c.vx[t];
        y[1] += c.vy[t];
        y[2] += c.vz[t];
        ++nv;
      }
    for (auto& v : y)
      v /= real_t(nv);
    const double dist = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
    maxDistSize = std::max(maxDistSize, dist / std::cbrt(std::max(a.V[0], 1e-300)));
    bool interior = true;
    for (const Plane& q : P)
      interior &= q.d - (q.p[0] * y[0] + q.p[1] * y[1] + q.p[2] * y[2]) > 0;
    Cell c2;
    if (!interior || !buildIn(c2, P, y, 2 * L)) {
      ++nNotInterior;
      continue;
    }
    Vals b;
    values(c2, b);
    for (int v = 0; v < 5; ++v)
      eV = std::max(eV, std::fabs(a.V[v] - b.V[v]) / h3);
    // match facets by neighbour id (planes with area > 0 in either frame)
    std::vector<int> map2(c.np, -1);
    for (int k = 0; k < c.np; ++k)
      for (int l = 0; l < c2.np; ++l)
        if (c2.pnbr[l] == c.pnbr[k])
          map2[k] = l;
    for (int l = 0; l < c2.np; ++l) {  // a facet of the y-frame cell missing from the seed frame
      bool found = false;
      for (int k = 0; k < c.np; ++k)
        found |= c.pnbr[k] == c2.pnbr[l] && fp[k] && a.A[k] > 0;
      if (b.A[l] > 0 && c2.pnbr[l] >= 0 && !found)
        ++nSetDiff;
    }
    for (int k = 0; k < c.np; ++k) {
      if (!fp[k] || c.pnbr[k] < 0 || a.A[k] <= 0)
        continue;
      const int l = map2[k];
      if (l < 0 || b.A[l] <= 0) {
        ++nSetDiff;
        continue;
      }
      eA = std::max({eA, std::fabs(a.A[k] - b.A[l]) / h2, std::fabs(a.Am[k] - b.Am[l]) / h2,
                     std::fabs(a.Ag[k] - b.Ag[l]) / h2});
      for (int d = 0; d < 3; ++d) {
        eAv = std::max(eAv, std::fabs(a.av[k][d] - b.av[l][d]) / h2);
        eM = std::max({eM, std::fabs(a.m[k][d] - (b.m[l][d] + b.Am[l] * y[d])) / h3,
                       std::fabs(a.mg[k][d] - (b.mg[l][d] + b.Ag[l] * y[d])) / h3});
      }
      eM2 = std::max(eM2, std::fabs(a.M2c[k] - b.M2c[l]) / h4);
    }
    // ---- dV/dq (geomVolumeGrad, geomVolumeArea) vs FD of volumePerVertex
    double vg, va, dgx[64] = {}, dgy[64] = {}, dgz[64] = {}, avx[64] = {}, avy[64] = {},
                   avz[64] = {}, ggx[64] = {}, ggy[64] = {}, ggz[64] = {};
    c.geomVolumeGrad(vg, dgx, dgy, dgz);
    c.geomVolumeArea(va, avx, avy, avz, ggx, ggy, ggz);
    double gmax = 0, gmaxA = 0;
    for (int k = 0; k < c.np; ++k)
      if (fp[k]) {
        gmax = std::max({gmax, std::fabs(dgx[k]), std::fabs(dgy[k]), std::fabs(dgz[k])});
        gmaxA = std::max({gmaxA, std::fabs(ggx[k]), std::fabs(ggy[k]), std::fabs(ggz[k])});
      }
    // ---- area Jacobian: analytic and AD per triangle, gathered to J[k][l][a]
    static double Jan[64][64][3], Jad[64][64][3];
    for (int k = 0; k < c.np; ++k)
      for (int l = 0; l < c.np; ++l)
        for (int d = 0; d < 3; ++d)
          Jan[k][l][d] = Jad[k][l][d] = 0;
    for (int t = 0; t < c.nt; ++t) {
      if (!c.alive[t])
        continue;
      int pa[3], pb[3];
      double ca[3], cb[3], ga[3][3][3], gb[3][3][3];
      c.geomVolumeAreaGrad(t, pa, ca, ga);
      c.geomVolumeAreaGradAD(t, pb, cb, gb);
      double num = 0, den = 0;
      for (int i2 = 0; i2 < 3; ++i2)
        for (int j2 = 0; j2 < 3; ++j2)
          for (int d = 0; d < 3; ++d) {
            Jan[pa[i2]][pa[j2]][d] += ga[i2][j2][d];
            Jad[pb[i2]][pb[j2]][d] += gb[i2][j2][d];
            den = std::max(den, std::fabs(gb[i2][j2][d]));
          }
      bool same = pa[0] == pb[0] && pa[1] == pb[1] && pa[2] == pb[2];
      for (int i2 = 0; i2 < 3 && same; ++i2)
        for (int j2 = 0; j2 < 3; ++j2)
          for (int d = 0; d < 3; ++d)
            num = std::max(num, std::fabs(ga[i2][j2][d] - gb[i2][j2][d]));
      eAD = std::max(eAD, same ? (den > 0 ? num / den : num) : 1.0);
    }
    double jmax = 0;
    for (int k = 0; k < c.np; ++k)
      for (int l = 0; l < c.np; ++l)
        for (int d = 0; d < 3; ++d)
          jmax = std::max(jmax, std::fabs(Jad[k][l][d]));
    for (int l = 0; l < c.np; ++l) {
      if (!fp[l])
        continue;
      for (int d = 0; d < 3; ++d) {
        Cell cp = c, cm = c;
        moveFoot(cp, l, d, delta);
        moveFoot(cm, l, d, -delta);
        const double fd = (cp.volumePerVertex() - cm.volumePerVertex()) / (2 * delta);
        const double g = d == 0 ? dgx[l] : (d == 1 ? dgy[l] : dgz[l]);
        const double gA = d == 0 ? ggx[l] : (d == 1 ? ggy[l] : ggz[l]);
        if (std::fabs(fd - g) / gmax > eGradV) {
          worstDistSize = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]) /
                          std::cbrt(std::max(a.V[0], 1e-300));
          worstV = a.V[0] / h3;
        }
        eGradV = std::max(eGradV, std::fabs(fd - g) / gmax);
        for (int sIdx = 0; sIdx < 3; ++sIdx) {
          Cell sp = c, sm = c;
          moveFoot(sp, l, d, sweep[sIdx]);
          moveFoot(sm, l, d, -sweep[sIdx]);
          const double fs = (sp.volumePerVertex() - sm.volumePerVertex()) / (2 * sweep[sIdx]);
          eSweep[sIdx] = std::max(eSweep[sIdx], std::fabs(fs - g) / gmax);
        }
        eGradVA = std::max(eGradVA, std::fabs(fd - gA) / gmaxA);
        double ap[64] = {}, am[64] = {};
        cp.facetAreasPerVertex(ap);
        cm.facetAreasPerVertex(am);
        for (int k = 0; k < c.np; ++k) {
          if (!fp[k])
            continue;
          const double fdA = (ap[k] - am[k]) / (2 * delta);
          eFDan = std::max(eFDan, std::fabs(fdA - Jan[k][l][d]) / jmax);
          eFDad = std::max(eFDad, std::fabs(fdA - Jad[k][l][d]) / jmax);
        }
      }
    }
  }
  std::printf(
      "outside-seed power cells (8^3 jittered ±0.3h, W = %.1f h^2): checked=%ld (>= 20) "
      "empty=%ld overflow=%ld boxFace=%ld notInterior=%ld facetSetDiff=%ld "
      "maxDist/size=%.2f\n",
      W, nOS, nEmpty, nOverflow, nBoxFace, nNotInterior, nSetDiff, maxDistSize);
  std::printf(
      "  (13a) translation oracle: |dV|/h3=%.2e (<=1e-12) |dA|/h2=%.2e |dAreaVec|/h2=%.2e "
      "(<=1e-12) |dm|/h3=%.2e (<=1e-12) |dM2c|/h4=%.2e (<=1e-12)\n",
      eV, eA, eAv, eM, eM2);
  std::printf(
      "  (13b) dV/dq vs FD (delta=1e-6h, rel to the cell's max): geomVolumeGrad %.2e "
      "geomVolumeArea %.2e (<=1e-6); dA/dq analytic-vs-AD %.2e (<=1e-12, per triangle); "
      "vs FD: analytic %.2e AD %.2e (<=1e-6)\n",
      eGradV, eGradVA, eAD, eFDan, eFDad);
  std::printf(
      "  (13b') REPORT dV/dq FD error vs step: delta/h=1e-4:%.2e 1e-5:%.2e 1e-6:%.2e "
      "1e-7:%.2e; worst cell at 1e-6h: dist/size=%.2f V/h3=%.2e\n",
      eSweep[0], eSweep[1], eGradV, eSweep[2], worstDistSize, worstV);
  // ---- REPORT (amendment 4): Power::chain dV_i/dx_j vs FD in x_j as d_ij → 0, both signs.
  {
    const int i = 0;
    std::vector<Plane> P = powerPlanes(i, x, w, N, L);
    Cell c;
    buildIn(c, P, zero, L);
    double A[64] = {};
    c.facetAreasPerVertex(A);
    int kBig = -1;
    for (int k = 0; k < c.np; ++k)
      if (c.pnbr[k] >= 0 && (kBig < 0 || A[k] > A[kBig]))
        kBig = k;
    const int j = c.pnbr[kBig];
    real_t r2 = 0;
    for (const Plane& q : P)
      if (q.j == j)
        r2 = q.p[0] * q.p[0] + q.p[1] * q.p[1] + q.p[2] * q.p[2];
    std::printf("  (13c) REPORT chain dV_i/dx_j vs FD (cell %d, neighbour %d, delta=1e-6h):", i, j);
    for (double sgn : {1.0, -1.0})
      for (double t : {1e-2, 1e-6, 1e-12}) {
        std::vector<real_t> w2 = w, x2 = x;
        w2[j] = r2 + w2[i] - 2 * sgn * t * r2;  // d_ij = sgn·t·|r|²
        Cell cc;
        buildIn(cc, powerPlanes(i, x2, w2, N, L), zero, L);
        int kj = -1;
        for (int k = 0; k < cc.np; ++k)
          if (cc.pnbr[k] == j)
            kj = k;
        if (kj < 0) {
          std::printf(" %+.0e:nofacet", sgn * t);
          continue;
        }
        double vg, gx[64] = {}, gy[64] = {}, gz[64] = {}, fS[3], fwS, fx[64], fy[64], fz[64],
                   fw[64];
        cc.geomVolumeGrad(vg, gx, gy, gz);
        const real_t seed[3] = {x2[3 * i], x2[3 * i + 1], x2[3 * i + 2]};
        peclet::voro::chainToDofs<peclet::voro::Power>(cc, seed, x2.data(), w2[i], w2.data(), L, gx,
                                                       gy, gz, fS, fwS, fx, fy, fz, fw);
        const double F[3] = {fx[kj], fy[kj], fz[kj]};
        double num = 0, den = 0;
        for (int d = 0; d < 3; ++d) {
          double vpm[2];
          for (int s = 0; s < 2; ++s) {
            std::vector<real_t> x3 = x2;
            x3[3 * j + d] += (s == 0 ? 1 : -1) * delta;
            Cell c3;
            buildIn(c3, powerPlanes(i, x3, w2, N, L), zero, L);
            vpm[s] = c3.volumePerVertex();
          }
          const double fd = (vpm[0] - vpm[1]) / (2 * delta);
          num = std::max(num, std::fabs(fd - F[d]));
          den = std::max(den, std::fabs(fd));
        }
        std::printf(" d/r2=%+.0e:%.1e", sgn * t, num / den);
      }
    std::printf("\n");
  }
  const bool pass = nOS >= 20 && nOverflow == 0 && nNotInterior == 0 && nSetDiff == 0 &&
                    eV <= 1e-12 && eA <= 1e-12 && eAv <= 1e-12 && eM <= 1e-12 && eM2 <= 1e-12 &&
                    eGradV <= 1e-6 && eGradVA <= 1e-6 && eAD <= 1e-12 && eFDan <= 1e-6 &&
                    eFDad <= 1e-6;
  std::printf("  (13) %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace outside_seed

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int N = 60000;
    const real_t L = 1.0, spacing = std::cbrt(1.0 / N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<real_t> U(0, 1);

    // two batches: isotropic, and anisotropic (squashed z -> skinny/obtuse cells, feet outside
    // faces)
    for (int batch = 0; batch < 2; ++batch) {
      const real_t sz_scale = (batch == 0) ? 1.0 : 0.35;  // squash z in batch 1
      std::vector<real_t> pos(3 * N);
      for (int i = 0; i < N; ++i) {
        pos[3 * i] = U(rng);
        pos[3 * i + 1] = U(rng);
        pos[3 * i + 2] = U(rng) * sz_scale + 0.5 * (1 - sz_scale);
      }
      // neighbour lists (brute-force grid)
      const int dim = std::max(1, (int)std::floor(1.0 / spacing));
      const real_t csz = 1.0 / dim;
      std::vector<std::vector<int>> grid(dim * dim * dim);
      auto gid = [&](int x, int y, int z) {
        return (x + dim) % dim + dim * (((y + dim) % dim) + dim * ((z + dim) % dim));
      };
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz),
            gz = (int)(pos[3 * i + 2] / csz);
        grid[gid(std::min(gx, dim - 1), std::min(gy, dim - 1), std::min(gz, dim - 1))].push_back(i);
      }

      double maxV = 0, maxDiv = 0, maxArea = 0, maxAreaAbs = 0, maxLabel = 0, maxDVforce = 0,
             maxMerged = 0;
      double maxGrad = 0, maxGradVol = 0;  // (9) geomVolumeGrad vs closed-form dV/dn
      double maxAreaVec = 0, maxGradA = 0,
             maxAreaVol = 0;  // (10) geomVolumeArea vs facetGeometry/closed form
      double maxAreaGather = 0, maxdAfd = 0, maxAreaJacAD = 0;  // (11) geomFull dA/dn
      long ndAtot = 0, ndApass = 0, nFull = 0;
      double maxChain = 0;
      long nChainTot = 0, nChainPass = 0, nChainCells = 0;  // (12) Voronoi chainToDofs
      long nchecked = 0, nfaceChecked = 0, nGradTot = 0, nGradPass = 0;
      double cellScale = spacing * spacing;  // typical face area ~ spacing²
      const int sw = 3;
      std::vector<std::array<real_t, 2>> tmp;
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz),
            gz = (int)(pos[3 * i + 2] / csz);
        gx = std::min(gx, dim - 1);
        gy = std::min(gy, dim - 1);
        gz = std::min(gz, dim - 1);
        tmp.clear();
        for (int dz = -sw; dz <= sw; ++dz)
          for (int dy = -sw; dy <= sw; ++dy)
            for (int dx = -sw; dx <= sw; ++dx)
              for (int j : grid[gid(gx + dx, gy + dy, gz + dz)]) {
                if (j == i)
                  continue;
                real_t rx = pos[3 * j] - pos[3 * i], ry = pos[3 * j + 1] - pos[3 * i + 1],
                       rz = pos[3 * j + 2] - pos[3 * i + 2];
                rx -= std::round(rx);
                ry -= std::round(ry);
                rz -= std::round(rz);
                tmp.push_back({rx * rx + ry * ry + rz * rz, (real_t)j});
              }
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        std::vector<int> nbr;
        for (int t = 0; t < (int)tmp.size() && t < 64; ++t)
          nbr.push_back((int)tmp[t][1]);

        Cell c;
        if (!buildCell(c, i, pos, L, nbr))
          continue;
        const double Vref = c.volume();
        if (Vref <= 0)
          continue;
        ++nchecked;

        // (1) volume: vertex-local vs ordered reference
        const double Vpv = c.volumePerVertex();
        maxV = std::max(maxV, std::fabs(Vpv - Vref) / Vref);

        // areas + first moments (vertex-local scatter)
        double area[64], mx[64], my[64], mz[64];
        for (int k = 0; k < c.np; ++k) {
          area[k] = mx[k] = my[k] = mz[k] = 0.0;
        }
        c.facetMomentsPerVertex(area, mx, my, mz);

        // merged G1+G2 kernel must match the separate calls
        {
          double vg = 0, ag[64], gmx[64], gmy[64], gmz[64];
          for (int k = 0; k < c.np; ++k) {
            ag[k] = gmx[k] = gmy[k] = gmz[k] = 0.0;
          }
          c.geometryPerVertex(vg, ag, gmx, gmy, gmz);
          maxMerged = std::max(maxMerged, std::fabs(vg - Vpv) / Vref);
          for (int k = 0; k < c.np; ++k)
            maxMerged = std::max(maxMerged, std::fabs(ag[k] - area[k]) / cellScale);
        }

        // (9) geomVolumeGrad: areas-free reverse-scatter dV/dn vs the closed form (2·area·n −
        // m)/|n|
        {
          double vg = 0, dgx[64], dgy[64], dgz[64];
          for (int k = 0; k < c.np; ++k) {
            dgx[k] = dgy[k] = dgz[k] = 0.0;
          }
          c.geomVolumeGrad(vg, dgx, dgy, dgz);
          maxGradVol = std::max(maxGradVol, std::fabs(vg - Vpv) / Vref);
          for (int k = 0; k < c.np; ++k) {
            if (area[k] < 0.05 * cellScale)
              continue;  // skip near-degenerate facets (mag → 0)
            const double invn = 1.0 / std::sqrt(c.nn[k]);
            const double cfx = (2.0 * area[k] * c.n[k][0] - mx[k]) * invn;
            const double cfy = (2.0 * area[k] * c.n[k][1] - my[k]) * invn;
            const double cfz = (2.0 * area[k] * c.n[k][2] - mz[k]) * invn;
            const double err =
                std::sqrt((dgx[k] - cfx) * (dgx[k] - cfx) + (dgy[k] - cfy) * (dgy[k] - cfy) +
                          (dgz[k] - cfz) * (dgz[k] - cfz));
            const double mag = std::sqrt(cfx * cfx + cfy * cfy + cfz * cfz);
            if (mag > 1e-9)
              maxGrad = std::max(maxGrad, err / mag);
          }
        }

        // (10) geomVolumeArea: areas-free sqrt-free area-vectors + dV/dn vs facetGeometry / closed
        // form
        {
          double varea = 0, avx[64], avy[64], avz[64], dgx[64], dgy[64], dgz[64];
          for (int k = 0; k < c.np; ++k) {
            avx[k] = avy[k] = avz[k] = dgx[k] = dgy[k] = dgz[k] = 0.0;
          }
          c.geomVolumeArea(varea, avx, avy, avz, dgx, dgy, dgz);
          maxAreaVol = std::max(maxAreaVol, std::fabs(varea - Vpv) / Vref);
          real_t aV[3], dvRef[3], cn[3];
          for (int k = 6; k < c.np; ++k) {
            if (area[k] < 0.05 * cellScale)
              continue;
            // area-vector vs facetGeometry's oriented polygon area vector
            if (c.facetGeometry(k, aV, dvRef, cn)) {
              const double aerr = std::sqrt((avx[k] - aV[0]) * (avx[k] - aV[0]) +
                                            (avy[k] - aV[1]) * (avy[k] - aV[1]) +
                                            (avz[k] - aV[2]) * (avz[k] - aV[2]));
              const double amag =
                  std::sqrt((double)aV[0] * aV[0] + (double)aV[1] * aV[1] + (double)aV[2] * aV[2]);
              if (amag > 1e-9)
                maxAreaVec = std::max(maxAreaVec, aerr / amag);
            }
            // dV/dn vs closed form (2·area·n − m)/|n|
            const double invn = 1.0 / std::sqrt(c.nn[k]);
            const double cfx = (2.0 * area[k] * c.n[k][0] - mx[k]) * invn;
            const double cfy = (2.0 * area[k] * c.n[k][1] - my[k]) * invn;
            const double cfz = (2.0 * area[k] * c.n[k][2] - mz[k]) * invn;
            const double err =
                std::sqrt((dgx[k] - cfx) * (dgx[k] - cfx) + (dgy[k] - cfy) * (dgy[k] - cfy) +
                          (dgz[k] - cfz) * (dgz[k] - cfz));
            const double mag = std::sqrt(cfx * cfx + cfy * cfy + cfz * cfz);
            if (mag > 1e-9)
              maxGradA = std::max(maxGradA, err / mag);
          }
        }

        // (11) geomFull: gather the per-triangle area-Jacobian geomVolumeAreaGrad into the full
        // dA_k/dn_l and (on a sample) finite-difference check it. Expensive, so cap the sample per
        // batch.
        if (nFull < 60) {
          ++nFull;
          const int np = c.np;
          std::vector<double> dA((size_t)np * np * 3, 0.0), Ag(np, 0.0);
          for (int t = 0; t < c.nt; ++t) {
            if (!c.alive[t])
              continue;
            int pl[3];
            double cb[3], gr[3][3][3];
            c.geomVolumeAreaGrad(t, pl, cb, gr);
            {  // analytic vs forward-AD oracle (must match to machine precision)
              int plA[3];
              double cbA[3], grA[3][3][3];
              c.geomVolumeAreaGradAD(t, plA, cbA, grA);
              double sc = 0;
              for (int ii = 0; ii < 3; ++ii)
                sc = std::max(sc, std::fabs(cb[ii]));
              sc = std::max(sc, 1e-30);
              for (int ii = 0; ii < 3; ++ii)
                for (int jj = 0; jj < 3; ++jj)
                  for (int dd = 0; dd < 3; ++dd)
                    maxAreaJacAD = std::max(
                        maxAreaJacAD, std::fabs(gr[ii][jj][dd] - grA[ii][jj][dd]) / (sc / spacing));
            }
            for (int ii = 0; ii < 3; ++ii) {
              Ag[pl[ii]] += cb[ii];
              for (int jj = 0; jj < 3; ++jj)
                for (int cc = 0; cc < 3; ++cc)
                  dA[((size_t)pl[ii] * np + pl[jj]) * 3 + cc] += gr[ii][jj][cc];
            }
          }
          for (int k = 0; k < np;
               ++k)  // gathered area must reproduce facetAreasPerVertex's area[k]
            maxAreaGather = std::max(maxAreaGather, std::fabs(Ag[k] - area[k]) / cellScale);
          // finite-difference dA_k/dn_l vs the gathered analytic, over real faces
          const double eps = 1e-7;
          for (int l = 6; l < np; ++l) {
            if (area[l] < 0.05 * cellScale)
              continue;
            for (int cc = 0; cc < 3; ++cc) {
              auto areasAt = [&](double dd) {
                Cell cp = c;
                cp.n[l][cc] += dd;
                cp.nn[l] =
                    cp.n[l][0] * cp.n[l][0] + cp.n[l][1] * cp.n[l][1] + cp.n[l][2] * cp.n[l][2];
                for (int t = 0; t < cp.nt; ++t)
                  if (cp.alive[t])
                    cp.computeVertex(t);
                std::vector<double> ar(cp.np, 0.0);
                cp.facetAreasPerVertex(ar.data());
                return ar;
              };
              auto ap = areasAt(eps), am = areasAt(-eps);
              for (int k = 6; k < np; ++k) {
                if (area[k] < 0.05 * cellScale)
                  continue;
                const double fd = (ap[k] - am[k]) / (2 * eps);
                const double an = dA[((size_t)k * np + l) * 3 + cc];
                if (std::fabs(fd) > 0.02 * spacing) {  // only significant couplings (FD-resolvable)
                  ++ndAtot;
                  if (std::fabs(an - fd) / std::fabs(fd) < 2e-3)
                    ++ndApass;
                  maxdAfd = std::max(maxdAfd, std::fabs(an - fd) / std::fabs(fd));
                }
              }
            }
          }
        }

        // (12) Phase 3: Voronoi policy + chainToDofs — dV/dp_self (geomVolumeGrad's dV/dn chained
        // to the own seed) vs a finite difference of the rebuilt cell's volume. End-to-end
        // geometry→DOF check.
        if (nChainCells < 40) {
          ++nChainCells;
          double vg = 0, dgx[64], dgy[64], dgz[64];
          for (int k = 0; k < c.np; ++k)
            dgx[k] = dgy[k] = dgz[k] = 0.0;
          c.geomVolumeGrad(vg, dgx, dgy, dgz);
          double fSelf[3], fwSelf, fnx[64], fny[64], fnz[64], fwn[64];
          const double seed3[3] = {(double)pos[3 * i], (double)pos[3 * i + 1],
                                   (double)pos[3 * i + 2]};
          peclet::voro::chainToDofs<peclet::voro::Voronoi>(
              c, seed3, (const double*)nullptr, 0.0, (const double*)nullptr, (double)L, dgx, dgy,
              dgz, fSelf, fwSelf, fnx, fny, fnz, fwn);
          const double fmag =
              std::sqrt(fSelf[0] * fSelf[0] + fSelf[1] * fSelf[1] + fSelf[2] * fSelf[2]);
          const double denom = std::max(fmag, cellScale);
          const double eps = 1e-7;
          for (int cc = 0; cc < 3;
               ++cc) {  // FD of V wrt the self seed, rebuilding from the same neighbours
            std::vector<real_t> pp = pos;
            pp[3 * i + cc] += eps;
            Cell cpp;
            const bool okp = buildCell(cpp, i, pp, L, nbr);
            pp[3 * i + cc] -= 2 * eps;
            Cell cpm;
            const bool okm = buildCell(cpm, i, pp, L, nbr);
            if (!okp || !okm)
              continue;
            const double fd = (cpp.volumePerVertex() - cpm.volumePerVertex()) / (2 * eps);
            ++nChainTot;
            const double rel = std::fabs(fd - fSelf[cc]) / denom;
            if (rel < 2e-3)
              ++nChainPass;  // FD-limited; misses are topology events
            maxChain = std::max(maxChain, rel);
          }
        }

        // (4) divergence identity  V == (1/3) Σ_f |n_f| A_f   (|n_f| = sqrt(nn) = seed->plane
        // distance)
        double Vdiv = 0;
        for (int k = 0; k < c.np; ++k)
          Vdiv += std::sqrt(c.nn[k]) * area[k];
        Vdiv /= 3.0;
        maxDiv = std::max(maxDiv, std::fabs(Vdiv - Vpv) / Vref);

        // (3) per-facet areas vs ordered polygon area
        real_t fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
        for (int k = 0; k < c.np; ++k) {
          const int m = c.faceOrdered(k, fx, fy, fz);
          if (m < 3)
            continue;
          real_t A[3];
          Cell::polyAreaVec(fx, fy, fz, m, A);
          const double aref = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
          maxAreaAbs =
              std::max(maxAreaAbs, std::fabs(area[k] - aref) / cellScale);  // abs, scale-normed
          if (aref >
              0.05 * cellScale) {  // skip near-degenerate faces (relative metric blows up there)
            maxArea = std::max(maxArea, std::fabs(area[k] - aref) / aref);
            ++nfaceChecked;
          }
        }

        // (7) volume gradient / FORCE: per-vertex dV vs the oracle-validated facetGeometry.dv
        real_t aV[3], dvRef[3], cn[3];
        for (int k = 6; k < c.np; ++k) {
          if (area[k] < 0.05 * cellScale)
            continue;
          if (!c.facetGeometry(k, aV, dvRef, cn))
            continue;
          // connector r = 2·n (foot point); force dV/dr = (area/|r|)(r − centroid)
          const double rx = 2.0 * c.n[k][0], ry = 2.0 * c.n[k][1], rz = 2.0 * c.n[k][2];
          const double rl = std::sqrt(rx * rx + ry * ry + rz * rz);
          const double s = area[k] / rl;
          const double cxk = mx[k] / area[k], cyk = my[k] / area[k], czk = mz[k] / area[k];
          const double dvx = s * (rx - cxk), dvy = s * (ry - cyk), dvz = s * (rz - czk);
          const double err =
              std::sqrt((dvx - dvRef[0]) * (dvx - dvRef[0]) + (dvy - dvRef[1]) * (dvy - dvRef[1]) +
                        (dvz - dvRef[2]) * (dvz - dvRef[2]));
          const double mag =
              std::sqrt(dvRef[0] * dvRef[0] + dvRef[1] * dvRef[1] + dvRef[2] * dvRef[2]);
          if (mag > 1e-9)
            maxDVforce = std::max(maxDVforce, err / mag);
        }

        // (5) label invariance: cyclically rotate every triangle's plane triple
        Cell c2 = c;
        for (int t = 0; t < c2.nt; ++t)
          if (c2.alive[t]) {
            unsigned char a = c2.t0[t];
            c2.t0[t] = c2.t1[t];
            c2.t1[t] = c2.t2[t];
            c2.t2[t] = a;
          }
        maxLabel = std::max(maxLabel, std::fabs(c2.volumePerVertex() - Vpv) / Vref);

        // (6) gradient: dV/dh_f finite-diff vs analytic A_f  (one real face per cell). Perturb
        // plane kf along its normal by dh: scale the foot point n[kf] by (h+dh)/h, set nn =
        // (h+dh)².
        int kf = -1;
        for (int k = 6; k < c.np; ++k)
          if (area[k] > 1e-6) {
            kf = k;
            break;
          }
        if (kf >= 0) {
          const double h = std::sqrt(c.nn[kf]);  // |n_kf| = seed->plane distance
          const double eps =
              1e-8;  // small step; topology stays fixed away from combinatorial events
          auto Vat = [&](double dh) {
            Cell cc = c;
            const double sc = (h + dh) / h;
            cc.n[kf][0] *= sc;
            cc.n[kf][1] *= sc;
            cc.n[kf][2] *= sc;
            cc.nn[kf] = (h + dh) * (h + dh);
            for (int t = 0; t < cc.nt; ++t)
              if (cc.alive[t])
                cc.computeVertex(t);
            return cc.volumePerVertex();
          };
          const double dVdh = (Vat(eps) - Vat(-eps)) / (2 * eps);
          const double analytic = area[kf];  // dV/dh = A_f
          ++nGradTot;
          if (std::fabs(dVdh - analytic) / std::fabs(analytic) < 1e-4)
            ++nGradPass;  // smooth cells
        }
      }
      std::printf("batch %d (%s): cells=%ld faces=%ld\n", batch,
                  batch ? "anisotropic/obtuse" : "isotropic", nchecked, nfaceChecked);
      const double gradFrac = nGradTot ? (double)nGradPass / nGradTot : 1.0;
      std::printf("  (1) Vpv vs ordered       max rel = %.3e\n", maxV);
      std::printf(
          "  (3) areas vs ordered     max rel(non-degenerate) = %.3e   max abs/scale = %.3e\n",
          maxArea, maxAreaAbs);
      std::printf("  (4) divergence identity  max rel = %.3e\n", maxDiv);
      std::printf("  (5) label invariance     max rel = %.3e\n", maxLabel);
      std::printf("  (7) dV(force) vs oracle  max rel = %.3e\n", maxDVforce);
      std::printf("  (8) merged kernel match  max rel = %.3e\n", maxMerged);
      std::printf("  (9) geomVolumeGrad dV/dn  max rel = %.3e   (vol match = %.3e)\n", maxGrad,
                  maxGradVol);
      std::printf(
          "  (10) geomVolumeArea       areaVec rel = %.3e   dV/dn rel = %.3e   (vol match = "
          "%.3e)\n",
          maxAreaVec, maxGradA, maxAreaVol);
      const double dAfrac = ndAtot ? (double)ndApass / ndAtot : 1.0;
      std::printf(
          "  (11) geomFull dA/dn vs FD  %.4f%% of %ld couplings (max rel %.2e)   gather %.3e   "
          "analytic-vs-AD %.3e\n",
          100.0 * dAfrac, ndAtot, maxdAfd, maxAreaGather, maxAreaJacAD);
      const double chainFrac = nChainTot ? (double)nChainPass / nChainTot : 1.0;
      std::printf("  (12) Voronoi chain dV/dp_self vs FD  %.4f%% of %ld comps (max rel %.2e)\n",
                  100.0 * chainFrac, nChainTot, maxChain);
      std::printf("  (6) gradient (FD<1e-4)   %.4f%% of cells (rest are at topology events)\n",
                  100.0 * gradFrac);
      const double tol = 1e-9;
      if (maxV > tol || maxArea > tol || maxDiv > tol || maxLabel > 1e-12 || maxAreaAbs > tol ||
          maxDVforce > 1e-9 || maxMerged > 1e-9 || maxGrad > 1e-9 || maxGradVol > 1e-9 ||
          maxAreaVec > tol || maxGradA > 1e-9 || maxAreaVol > 1e-9 || maxAreaGather > tol ||
          dAfrac < 0.95 || maxAreaJacAD > 1e-9 || chainFrac < 0.95 || gradFrac < 0.98) {
        std::printf("  FAIL\n");
        rc = 1;
      } else {
        std::printf("  PASS\n");
      }
    }
    rc |= outside_seed::run();
  }
  Kokkos::finalize();
  return rc;
}
