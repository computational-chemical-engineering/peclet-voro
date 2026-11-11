/**
 * @file test_polygon_transform.cpp
 * @brief The exact polygon Fourier transforms Φ_P(q) = ∫_P e^{iq·ξ} dA and Φ1_P(q) = ∫_P ξ e^{iq·ξ}
 *        dA (lagrangian/polygon_transform.hpp, M7 DESIGN_NOTE_STATICS §17.5), gate G-S.1(a):
 *
 *   (A) the reference: a product Gauss–Legendre cubature in long double on the fan triangles
 *       from ξ_0 (Duffy-collapsed, independent of every closed form in the header), against the
 *       analytic unit square, and its self-convergence (N 40 vs 32);
 *   (B) Φ, Φ1 (auto branch) against the reference over random convex polygons (3–9 vertices, on
 *       ellipses of aspect ≤ 3, generator frame: h ∈ [0.05, 0.8], size ≤ 0.7) × five q classes
 *       (|q| ≤ 12 ≥ k_max h = 6.79 for ε_G 10⁻¹⁰): general; |q_t|D ∈ [0, 0.3]; |q_t|D within 1 %
 *       of the 0.25 switch; one edge with |a| = |q·Δ_e| ≤ 0.6; q ∥ n̂ (q = 0 included). Error
 *       |Φ − Φ_ref|/A and |Φ1 − Φ1_ref|/(A max_v|ξ_v|) ≤ 10⁻¹³ (|Φ| ≤ A and |Φ1| ≤ A max|ξ|, so
 *       this is relative to the transform's scale; Φ has zeros, where a pointwise ratio means
 *       nothing — the pointwise ratio is reported where |Φ| ≥ 10⁻² A);
 *   (C) both branches forced, at |q_t|D = 0.25 exactly: they agree to ≤ 10⁻¹³; both forced, each
 *       against the reference over |q_t|D ∈ [0.2, 0.3] (each branch is exact on both sides);
 *       F₀, F₁ continuous across |a| = 0.5;
 *   (D) device: the same evaluations inside a parallel_for on the default execution space equal
 *       the host calls (≤ 10⁻¹⁴; bit-identical expected on a host backend);
 *   (E) report only: slivers (aspect 5–30, near-coincident vertices), where the edge branch's
 *       cancellation scales as D²/A.
 *
 * Usage: test_polygon_transform [num_polygons=10000] [--dump file count] (the dump writes cases
 * and the long-double reference for an offline mpmath cross-check).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/polygon_transform.hpp"

using real_t = double;
using Cx = Kokkos::complex<real_t>;
namespace lag = peclet::voro::lagrangian;
using LD = long double;

namespace {

constexpr int kMaxV = 9;
constexpr double kQmax = 12.0;

struct Poly {
  int nv = 0;
  double xi[3 * kMaxV];
  double n[3];
};

// ── reference: long-double Gauss–Legendre on the fan from ξ_0 ──────────────────────────────────
struct Gauss {
  std::vector<LD> x, w;  // on [0, 1]
  explicit Gauss(int N) : x(N), w(N) {
    const LD pi = 3.141592653589793238462643383279502884L;
    for (int i = 0; i < N; ++i) {
      LD z = std::cos(pi * (i + 0.75L) / (N + 0.5L)), dp = 0;
      for (int it = 0; it < 100; ++it) {
        LD p0 = 1, p1 = z;
        for (int k = 2; k <= N; ++k) {
          const LD p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
          p0 = p1;
          p1 = p2;
        }
        dp = N * (z * p1 - p0) / (z * z - 1);
        const LD dz = p1 / dp;
        z -= dz;
        if (std::fabs(dz) < 1e-21L)
          break;
      }
      x[i] = (1 - z) / 2;
      w[i] = 1 / ((1 - z * z) * dp * dp);  // 2/((1−z²)P′²) on [−1,1], halved
    }
  }
};

// Φ and Φ1 by cubature: x = p0 + u(p1 − p0) + u w (p2 − p1), dA = |(p1−p0)×(p2−p0)| u du dw.
void reference(const Poly& P, const double* q, const Gauss& G, LD phi[2], LD phi1[3][2]) {
  for (int k = 0; k < 2; ++k) {
    phi[k] = 0;
    for (int d = 0; d < 3; ++d)
      phi1[d][k] = 0;
  }
  const int N = int(G.x.size());
  for (int t = 1; t + 1 < P.nv; ++t) {
    LD p0[3], e1[3], e2[3];
    for (int d = 0; d < 3; ++d) {
      p0[d] = P.xi[d];
      e1[d] = LD(P.xi[3 * t + d]) - p0[d];
      e2[d] = LD(P.xi[3 * (t + 1) + d]) - LD(P.xi[3 * t + d]);
    }
    const LD f[3] = {e1[1] * (e1[2] + e2[2]) - e1[2] * (e1[1] + e2[1]),
                     e1[2] * (e1[0] + e2[0]) - e1[0] * (e1[2] + e2[2]),
                     e1[0] * (e1[1] + e2[1]) - e1[1] * (e1[0] + e2[0])};
    const LD J = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (int i = 0; i < N; ++i) {
      const LD u = G.x[i];
      for (int j = 0; j < N; ++j) {
        const LD v = G.x[j], wt = G.w[i] * G.w[j] * u * J;
        LD x[3];
        for (int d = 0; d < 3; ++d)
          x[d] = p0[d] + u * e1[d] + u * v * e2[d];
        const LD ph = q[0] * x[0] + q[1] * x[1] + q[2] * x[2];
        const LD c = wt * std::cos(ph), s = wt * std::sin(ph);
        phi[0] += c;
        phi[1] += s;
        for (int d = 0; d < 3; ++d) {
          phi1[d][0] += x[d] * c;
          phi1[d][1] += x[d] * s;
        }
      }
    }
  }
}

// ── polygons and wave vectors ─────────────────────────────────────────────────────────────────
struct Gen {
  std::mt19937_64 rng;
  std::uniform_real_distribution<double> U{0.0, 1.0};
  std::normal_distribution<double> Nrm{0.0, 1.0};
  explicit Gen(unsigned long long s) : rng(s) {}
  double u(double a, double b) { return a + (b - a) * U(rng); }
  void unit(double* v) {
    double r;
    do {
      for (int d = 0; d < 3; ++d)
        v[d] = Nrm(rng);
      r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    } while (r < 1e-3);
    for (int d = 0; d < 3; ++d)
      v[d] /= r;
  }
};

void basis(const double* n, double* e1, double* e2) {
  const double a[3] = {std::fabs(n[0]) < 0.9 ? 1.0 : 0.0, std::fabs(n[0]) < 0.9 ? 0.0 : 1.0, 0.0};
  e1[0] = a[1] * n[2] - a[2] * n[1];
  e1[1] = a[2] * n[0] - a[0] * n[2];
  e1[2] = a[0] * n[1] - a[1] * n[0];
  const double r = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  for (int d = 0; d < 3; ++d)
    e1[d] /= r;
  e2[0] = n[1] * e1[2] - n[2] * e1[1];
  e2[1] = n[2] * e1[0] - n[0] * e1[2];
  e2[2] = n[0] * e1[1] - n[1] * e1[0];
}

// A convex polygon: nv points at jittered increasing angles on an ellipse (CCW about n).
Poly makePoly(Gen& g, double aspMin, double aspMax, double jitter) {
  Poly P;
  P.nv = 3 + int(g.U(g.rng) * 7);
  if (P.nv > kMaxV)
    P.nv = kMaxV;
  g.unit(P.n);
  double e1[3], e2[3];
  basis(P.n, e1, e2);
  const double h = g.u(0.05, 0.8), rho = g.u(0.15, 0.7), asp = g.u(aspMin, aspMax);
  const double rot = g.u(0, 2 * M_PI), th0 = g.u(0, 2 * M_PI);
  const double orad = 0.4 * std::sqrt(g.U(g.rng)), oang = g.u(0, 2 * M_PI);
  const double o1 = orad * std::cos(oang), o2 = orad * std::sin(oang);
  for (int k = 0; k < P.nv; ++k) {
    const double th = th0 + 2 * M_PI * (k + jitter * g.u(-1, 1)) / P.nv;
    const double a = rho * std::cos(th), b = rho / asp * std::sin(th);
    const double s1 = o1 + a * std::cos(rot) - b * std::sin(rot);
    const double s2 = o2 + a * std::sin(rot) + b * std::cos(rot);
    for (int d = 0; d < 3; ++d)
      P.xi[3 * k + d] = h * P.n[d] + s1 * e1[d] + s2 * e2[d];
  }
  return P;
}

// q of class c (0 general, 1 |q_t|D ≤ 0.3, 2 at the switch ±1 %, 3 |a| ≤ 0.6 on one edge, 4 q ∥ n̂)
void makeQ(Gen& g, const Poly& P, const lag::PolygonFrame<double>& F, int c, int idx, double* q) {
  double e1[3], e2[3];
  basis(P.n, e1, e2);
  const double ang = g.u(0, 2 * M_PI), qn = g.u(-kQmax, kQmax);
  double mag = 0;
  switch (c) {
    case 0: {
      double v[3];
      g.unit(v);
      const double r = kQmax * std::cbrt(g.U(g.rng));
      for (int d = 0; d < 3; ++d)
        q[d] = r * v[d];
      return;
    }
    case 1:
      mag = g.u(0, 0.3) / F.diam;
      break;
    case 2:
      mag = 0.25 * (1 + g.u(-0.01, 0.01)) / F.diam;
      break;
    case 3: {
      const int e = int(g.U(g.rng) * P.nv) % P.nv;
      const double* a = P.xi + 3 * e;
      const double* b = P.xi + 3 * ((e + 1) % P.nv);
      double D[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      const double l = std::sqrt(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]);
      for (int d = 0; d < 3; ++d)
        D[d] /= l;
      const double p[3] = {P.n[1] * D[2] - P.n[2] * D[1], P.n[2] * D[0] - P.n[0] * D[2],
                           P.n[0] * D[1] - P.n[1] * D[0]};
      const double kap = g.u(0.25, 6.0) / F.diam, aa = g.u(-0.6, 0.6), qnn = 0.5 * qn;
      for (int d = 0; d < 3; ++d)
        q[d] = kap * p[d] + aa / l * D[d] + qnn * P.n[d];
      return;
    }
    case 4: {
      const double lam = (idx % 10 == 0) ? 0.0 : qn;
      for (int d = 0; d < 3; ++d)
        q[d] = lam * P.n[d];
      return;
    }
  }
  for (int d = 0; d < 3; ++d)
    q[d] = mag * (std::cos(ang) * e1[d] + std::sin(ang) * e2[d]) + qn * P.n[d];
}

double rmax(const Poly& P) {
  double r = 0;
  for (int v = 0; v < P.nv; ++v)
    r = std::max(r, std::sqrt(P.xi[3 * v] * P.xi[3 * v] + P.xi[3 * v + 1] * P.xi[3 * v + 1] +
                              P.xi[3 * v + 2] * P.xi[3 * v + 2]));
  return r;
}

// Errors of (phi, phi1) against a long-double reference, normalised by A and A·max|ξ|.
void errs(const Poly& P, const lag::PolygonFrame<double>& F, const Cx& phi, const Cx* phi1,
          const LD rp[2], const LD r1[3][2], double& ep, double& e1, double& epw) {
  const LD dr = LD(phi.real()) - rp[0], di = LD(phi.imag()) - rp[1];
  const LD ae = std::sqrt(dr * dr + di * di), am = std::sqrt(rp[0] * rp[0] + rp[1] * rp[1]);
  ep = double(ae / F.area);
  epw = (am >= 1e-2L * F.area) ? double(ae / am) : 0.0;
  LD s = 0;
  for (int d = 0; d < 3; ++d) {
    const LD a = LD(phi1[d].real()) - r1[d][0], b = LD(phi1[d].imag()) - r1[d][1];
    s += a * a + b * b;
  }
  e1 = double(std::sqrt(s) / (F.area * rmax(P)));
}

const char* kClass[5] = {"general |q|<=12", "|q_t|D<=0.3", "switch +-1%", "edge |a|<=0.6",
                         "q || n (incl. 0)"};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    int nPoly = 10000;
    const char* dumpFile = nullptr;
    int dumpCount = 0;
    for (int a = 1; a < argc; ++a) {
      if (!std::strcmp(argv[a], "--dump") && a + 2 < argc) {
        dumpFile = argv[++a];
        dumpCount = std::atoi(argv[++a]);
      } else if (argv[a][0] != '-') {
        nPoly = std::atoi(argv[a]);
      }
    }
    std::printf("Polygon transforms Φ, Φ1 (backend=%s, long double %d bits):\n",
                Kokkos::DefaultExecutionSpace::name(), std::numeric_limits<LD>::digits);
    if (std::numeric_limits<LD>::digits < 64) {
      std::printf("  long double carries < 64 bits: the cubature reference is too coarse; SKIP\n");
      Kokkos::finalize();
      return 0;
    }
    const Gauss G40(40), G32(32);
    const double tol = 1e-13;

    // (A) the reference vs the analytic unit square at z = h: Φ = e^{iq_z h} G0(q_x) G0(q_y),
    //     Φ1_x = e^{iq_z h} G1(q_x) G0(q_y), Φ1_z = h Φ; G0, G1 = ∫₀¹ (1, t) e^{iat} dt.
    {
      Poly S;
      S.nv = 4;
      const double h = 0.37, V[12] = {0, 0, h, 1, 0, h, 1, 1, h, 0, 1, h};
      std::copy(V, V + 12, S.xi);
      S.n[0] = S.n[1] = 0;
      S.n[2] = 1;
      const auto F = lag::polygonFrame(S.xi, S.nv, S.n);
      auto G01 = [](LD a, LD& g0r, LD& g0i, LD& g1r, LD& g1i) {
        if (std::fabs(a) < 1) {  // Σ (ia)ⁿ/n! · [1/(n+1), 1/(n+2)], to below 1e-30
          g0r = g0i = g1r = g1i = 0;
          LD p = 1;
          for (int n = 0; n < 40; ++n) {
            if (n > 0)
              p *= a / n;
            const LD sg = (n & 2) ? -p : p;
            (n & 1 ? g0i : g0r) += sg / (n + 1);
            (n & 1 ? g1i : g1r) += sg / (n + 2);
          }
          return;
        }
        const LD s = std::sin(a), c = std::cos(a);
        g0r = s / a;
        g0i = (1 - c) / a;
        g1r = s / a + (c - 1) / (a * a);
        g1i = -c / a + s / (a * a);
      };
      Gen g(7);
      double eRef = 0, eT = 0;
      for (int k = 0; k < 200; ++k) {
        double q[3] = {g.u(-kQmax, kQmax), g.u(-kQmax, kQmax), g.u(-kQmax, kQmax)};
        if (k % 20 == 0)
          q[0] = 0;
        if (k % 33 == 0)
          q[0] = q[1] = 0;
        LD x0r, x0i, x1r, x1i, y0r, y0i, y1r, y1i;
        G01(q[0], x0r, x0i, x1r, x1i);
        G01(q[1], y0r, y0i, y1r, y1i);
        const LD er = std::cos(LD(q[2]) * LD(h)), ei = std::sin(LD(q[2]) * LD(h));
        auto mul = [](LD ar, LD ai, LD br, LD bi, LD& r, LD& i) {
          r = ar * br - ai * bi;
          i = ar * bi + ai * br;
        };
        LD tr, ti, ex[2], ey[2], ez[2], ep[2];
        mul(x0r, x0i, y0r, y0i, tr, ti);
        mul(er, ei, tr, ti, ep[0], ep[1]);
        mul(x1r, x1i, y0r, y0i, tr, ti);
        mul(er, ei, tr, ti, ex[0], ex[1]);
        mul(x0r, x0i, y1r, y1i, tr, ti);
        mul(er, ei, tr, ti, ey[0], ey[1]);
        ez[0] = h * ep[0];
        ez[1] = h * ep[1];
        const LD ea1[3][2] = {{ex[0], ex[1]}, {ey[0], ey[1]}, {ez[0], ez[1]}};
        LD rp[2], r1[3][2];
        reference(S, q, G40, rp, r1);
        LD m = std::hypot(rp[0] - ep[0], rp[1] - ep[1]);
        for (int d = 0; d < 3; ++d)
          m = std::max(m, std::hypot(r1[d][0] - ea1[d][0], r1[d][1] - ea1[d][1]));
        eRef = std::max(eRef, double(m));
        Cx phi, phi1[3];
        lag::polygonPhiPhi1(S.xi, S.nv, F, q, phi, phi1);
        double a, b, c;
        errs(S, F, phi, phi1, ep, ea1, a, b, c);
        eT = std::max({eT, a, b});
      }
      const bool ok = eRef <= 1e-17 && eT <= tol;
      std::printf(
          "  (A) unit square: cubature vs analytic %.1e (<= 1e-17), Φ/Φ1 vs analytic %.1e\n", eRef,
          eT);
      rc |= ok ? 0 : 1;
    }

    // (B) auto branch vs the reference; (A') reference self-convergence on every 16th case
    {
      Gen g(20261111ULL);
      double eP[5] = {}, e1[5] = {}, ePw[5] = {}, eConv = 0;
      long nPw[5] = {}, nCase = 0, nSeries = 0;
      std::vector<Poly> polys(nPoly);
      std::vector<double> qs(15 * size_t(nPoly));
      for (int p = 0; p < nPoly; ++p) {
        polys[p] = makePoly(g, 1.0, 3.0, 0.45);
        const auto F = lag::polygonFrame(polys[p].xi, polys[p].nv, polys[p].n);
        for (int c = 0; c < 5; ++c)
          makeQ(g, polys[p], F, c, p, &qs[15 * size_t(p) + 3 * c]);
      }
      std::vector<double> res(5 * 4 * size_t(nPoly));  // ep, e1, epw, conv
      std::vector<int> ser(5 * size_t(nPoly));
      Kokkos::parallel_for(
          "ref", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nPoly), [&](int p) {
            const Poly& P = polys[p];
            const auto F = lag::polygonFrame(P.xi, P.nv, P.n);
            for (int c = 0; c < 5; ++c) {
              const double* q = &qs[15 * size_t(p) + 3 * c];
              LD rp[2], r1[3][2];
              reference(P, q, G40, rp, r1);
              Cx phi, phi1[3];
              lag::polygonPhiPhi1(P.xi, P.nv, F, q, phi, phi1);
              const Cx phi0 = lag::polygonPhi(P.xi, P.nv, F, q);
              double* r = &res[4 * (5 * size_t(p) + c)];
              errs(P, F, phi, phi1, rp, r1, r[0], r[1], r[2]);
              if (phi0 != phi)
                r[0] = 1.0;  // polygonPhi must equal polygonPhiPhi1's Φ
              ser[5 * p + c] = lag::detail::useSeries(F, q) ? 1 : 0;
              r[3] = 0;
              if (p % 16 == 0) {
                LD sp[2], s1[3][2];
                reference(P, q, G32, sp, s1);
                LD m = std::hypot(sp[0] - rp[0], sp[1] - rp[1]);
                for (int d = 0; d < 3; ++d)
                  m = std::max(m, std::hypot(s1[d][0] - r1[d][0], s1[d][1] - r1[d][1]));
                r[3] = double(m / LD(F.area));
              }
            }
          });
      for (int p = 0; p < nPoly; ++p)
        for (int c = 0; c < 5; ++c) {
          const double* r = &res[4 * (5 * size_t(p) + c)];
          eP[c] = std::max(eP[c], r[0]);
          e1[c] = std::max(e1[c], r[1]);
          if (r[2] > 0) {
            ePw[c] = std::max(ePw[c], r[2]);
            ++nPw[c];
          }
          eConv = std::max(eConv, r[3]);
          nSeries += ser[5 * p + c];
          ++nCase;
        }
      const bool okConv = eConv <= 1e-17;
      std::printf("  (A') cubature N=40 vs N=32 on every 16th case: %.1e (<= 1e-17)\n", eConv);
      rc |= okConv ? 0 : 1;
      std::printf("  (B) %d polygons x 5 q = %ld cases (%ld series branch):\n", nPoly, nCase,
                  nSeries);
      for (int c = 0; c < 5; ++c) {
        const bool ok = eP[c] <= tol && e1[c] <= tol;
        std::printf(
            "      %-18s |dΦ|/A %.1e  |dΦ1|/(A max|ξ|) %.1e  pointwise |dΦ|/|Φ| %.1e"
            " (%ld with |Φ|>=1e-2 A)  %s\n",
            kClass[c], eP[c], e1[c], ePw[c], nPw[c], ok ? "ok" : "FAIL");
        rc |= ok ? 0 : 1;
      }

      // (C) both branches forced
      double eSw = 0, eE = 0, eS = 0;
      Gen h(99);
      for (int p = 0; p < std::min(nPoly, 2000); ++p) {
        const Poly& P = polys[p];
        const auto F = lag::polygonFrame(P.xi, P.nv, P.n);
        double e1v[3], e2v[3];
        basis(P.n, e1v, e2v);
        for (int k = 0; k < 2; ++k) {
          const double ang = h.u(0, 2 * M_PI), qn = h.u(-kQmax, kQmax);
          const double mag = (k == 0 ? 0.25 : h.u(0.2, 0.3)) / F.diam;
          double q[3];
          for (int d = 0; d < 3; ++d)
            q[d] = mag * (std::cos(ang) * e1v[d] + std::sin(ang) * e2v[d]) + qn * P.n[d];
          Cx pE, pS, qE[3], qS[3];
          lag::detail::phiPhi1Edge(P.xi, P.nv, F, q, pE, qE);
          lag::detail::phiPhi1Series(P.xi, P.nv, F, q, pS, qS);
          if (k == 0) {
            double m = Kokkos::abs(pE - pS) / F.area;
            for (int d = 0; d < 3; ++d)
              m = std::max(m, Kokkos::abs(qE[d] - qS[d]) / (F.area * rmax(P)));
            eSw = std::max(eSw, m);
          } else {
            LD rp[2], r1[3][2];
            reference(P, q, G40, rp, r1);
            double a, b, c;
            errs(P, F, pE, qE, rp, r1, a, b, c);
            eE = std::max({eE, a, b});
            errs(P, F, pS, qS, rp, r1, a, b, c);
            eS = std::max({eS, a, b});
          }
        }
      }
      // F₀, F₁ across |a| = 0.5: one ulp either side, against the derivative bound |F′| ≤ 1/2
      double eF = 0;
      for (double s : {-1.0, 1.0}) {
        const double a1 = s * 0.5, a0 = std::nextafter(a1, 0.0);
        Cx F0a, F1a, F0b, F1b;
        lag::detail::edgeF01(a0, F0a, F1a);
        lag::detail::edgeF01(a1, F0b, F1b);
        eF = std::max({eF, Kokkos::abs(F0a - F0b), Kokkos::abs(F1a - F1b)});
      }
      const bool okC = eSw <= tol && eE <= tol && eS <= tol && eF <= 1e-15;
      std::printf(
          "  (C) at |q_t|D = 0.25 edge vs series %.1e; on [0.2, 0.3] edge %.1e, series %.1e"
          " vs ref; F01 jump at |a| = 0.5 %.1e\n",
          eSw, eE, eS, eF);
      rc |= okC ? 0 : 1;

      // (D) device = host
      {
        const int nd = std::min(nPoly, 1000);
        std::vector<double> hx(3 * kMaxV * size_t(nd)), hn(3 * size_t(nd)), hq(15 * size_t(nd));
        std::vector<int> hv(nd);
        for (int p = 0; p < nd; ++p) {
          std::copy(polys[p].xi, polys[p].xi + 3 * kMaxV, &hx[3 * kMaxV * size_t(p)]);
          std::copy(polys[p].n, polys[p].n + 3, &hn[3 * size_t(p)]);
          std::copy(&qs[15 * size_t(p)], &qs[15 * size_t(p)] + 15, &hq[15 * size_t(p)]);
          hv[p] = polys[p].nv;
        }
        const auto dx = peclet::core::toDevice<double>(hx, "xi");
        const auto dn = peclet::core::toDevice<double>(hn, "n");
        const auto dq = peclet::core::toDevice<double>(hq, "q");
        const auto dv = peclet::core::toDevice<int>(hv, "nv");
        Kokkos::View<double*, peclet::core::MemSpace> out("out", 40 * size_t(nd));
        Kokkos::parallel_for(
            "polygon_transform_device", nd, KOKKOS_LAMBDA(int p) {
              const double* xi = &dx(3 * kMaxV * p);
              const auto F = lag::polygonFrame(xi, dv(p), &dn(3 * p));
              for (int c = 0; c < 5; ++c) {
                Cx phi, phi1[3];
                lag::polygonPhiPhi1(xi, dv(p), F, &dq(15 * p + 3 * c), phi, phi1);
                double* o = &out(40 * p + 8 * c);
                o[0] = phi.real();
                o[1] = phi.imag();
                for (int d = 0; d < 3; ++d) {
                  o[2 + 2 * d] = phi1[d].real();
                  o[3 + 2 * d] = phi1[d].imag();
                }
              }
            });
        Kokkos::fence();
        const auto ho = peclet::core::toVector(out);
        double eDev = 0;
        long nDiff = 0;
        for (int p = 0; p < nd; ++p) {
          const auto F = lag::polygonFrame(polys[p].xi, polys[p].nv, polys[p].n);
          for (int c = 0; c < 5; ++c) {
            Cx phi, phi1[3];
            lag::polygonPhiPhi1(polys[p].xi, polys[p].nv, F, &qs[15 * size_t(p) + 3 * c], phi,
                                phi1);
            const double hv8[8] = {phi.real(),     phi.imag(),     phi1[0].real(), phi1[0].imag(),
                                   phi1[1].real(), phi1[1].imag(), phi1[2].real(), phi1[2].imag()};
            for (int k = 0; k < 8; ++k) {
              const double d = std::fabs(ho[40 * size_t(p) + 8 * c + k] - hv8[k]);
              nDiff += d != 0;
              eDev = std::max(eDev, d / (F.area * std::max(1.0, rmax(polys[p]))));
            }
          }
        }
        const bool ok = eDev <= 1e-14;
        std::printf("  (D) device vs host, %d polygons x 5 q: max %.1e, %ld of %d values differ\n",
                    nd, eDev, nDiff, 40 * nd);
        rc |= ok ? 0 : 1;
      }

      // dump for the offline mpmath cross-check of the reference
      if (dumpFile) {
        FILE* f = std::fopen(dumpFile, "w");
        for (int p = 0; p < std::min(dumpCount, nPoly); ++p)
          for (int c = 0; c < 5; ++c) {
            const Poly& P = polys[p];
            const double* q = &qs[15 * size_t(p) + 3 * c];
            LD rp[2], r1[3][2];
            reference(P, q, G40, rp, r1);
            std::fprintf(f, "%d", P.nv);
            for (int k = 0; k < 3 * P.nv; ++k)
              std::fprintf(f, " %.17g", P.xi[k]);
            for (int d = 0; d < 3; ++d)
              std::fprintf(f, " %.17g", q[d]);
            std::fprintf(f, " %.21Lg %.21Lg", rp[0], rp[1]);
            for (int d = 0; d < 3; ++d)
              std::fprintf(f, " %.21Lg %.21Lg", r1[d][0], r1[d][1]);
            std::fprintf(f, "\n");
          }
        std::fclose(f);
        std::printf("  dumped %d x 5 cases to %s\n", std::min(dumpCount, nPoly), dumpFile);
      }
    }

    // (E) report: slivers
    {
      Gen g(31337);
      const int ns = std::min(nPoly, 2000);
      double eP = 0, e1 = 0, eD2 = 0, minAD = 1;
      for (int p = 0; p < ns; ++p) {
        const Poly P = makePoly(g, 5.0, 30.0, 0.49);
        const auto F = lag::polygonFrame(P.xi, P.nv, P.n);
        minAD = std::min(minAD, F.area / (F.diam * F.diam));
        for (int c = 0; c < 4; ++c) {
          double q[3];
          makeQ(g, P, F, c, p, q);
          LD rp[2], r1[3][2];
          reference(P, q, G40, rp, r1);
          Cx phi, phi1[3];
          lag::polygonPhiPhi1(P.xi, P.nv, F, q, phi, phi1);
          double a, b, w;
          errs(P, F, phi, phi1, rp, r1, a, b, w);
          eP = std::max(eP, a);
          e1 = std::max(e1, b);
          eD2 = std::max(eD2, a * F.area / (F.diam * F.diam));
        }
      }
      std::printf(
          "  (E) report, %d slivers (min A/D^2 %.1e): |dΦ|/A %.1e, |dΦ1|/(A max|ξ|) %.1e,"
          " |dΦ|/D^2 %.1e\n",
          ns, minAD, eP, e1, eD2);
    }
    std::printf("%s\n",
                rc == 0 ? "ALL POLYGON TRANSFORM CHECKS PASS" : "POLYGON TRANSFORM CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
