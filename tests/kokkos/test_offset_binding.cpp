/**
 * @file test_offset_binding.cpp
 * @brief The offset binding U_b (lagrangian/offset_binding.hpp, M7 DESIGN_NOTE_BINDING §18.5 with
 *        the floor amendment; WO-BD4 unit gates) on a jittered 8³ lattice of power cells (weights
 *        solved for ±20 % targets), a drop of radius 0.3, σ = 1, at the defaults β 0.1, β₀ 1e-3, at
 *        the band alone (β₀ 0) and at the floor alone (β 0):
 *
 *   (A) device = host: U_b and its partials from the Kokkos passes against a serial host oracle
 *       (brute-force pairs in index order, the facet terms written from §18.5), max relative
 *       difference ≤ 1e-12 (the pair sums run in a different order);
 *   (B) ∂_xU_b and ∂_wU_b against central differences at fixed w / fixed x (≤ 1e-7 of the term
 *       scale Σ|∂U_b·d|), all-cell and band-cell directions;
 *   (C) translation of every generator (weights kept): |ΔU_b|/U_b ≤ 1e-12; Σ_i ∂_{x_i}U_b and
 *       Σ_i ∂_{w_i}U_b vanish to round-off (≤ 1e-12 of Σ|·|);
 *   (D) the skip rule: β = β₀ = 0, σ = 0, all labels equal → U_b = 0 and the gradient untouched
 *       bit for bit; partitionPotential with the binding off is bit-identical to the old spec.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/offset_binding.hpp"
#include "peclet/voro/lagrangian/potential.hpp"
#include "peclet/voro/lagrangian/weight_solve.hpp"
#include "peclet/voro/repair.hpp"

using real_t = double;
using Mem = peclet::core::MemSpace;
using DV = Kokkos::View<real_t*, Mem>;
namespace lag = peclet::voro::lagrangian;

namespace {

constexpr int kP = 64, kT = 112;
using MT = peclet::voro::MovingTessellation<real_t, kP, kT, true>;

DV toDev(const std::vector<real_t>& h, const char* label) {
  return peclet::core::toDevice<real_t>(h, label);
}
std::vector<real_t> toHost(const DV& v) {
  return peclet::core::toVector(v);
}
double sumAbs(const std::vector<real_t>& a) {
  double m = 0;
  for (real_t v : a)
    m += std::fabs(v);
  return m;
}
double maxAbs(const std::vector<real_t>& a) {
  double m = 0;
  for (real_t v : a)
    m = std::max(m, std::fabs(v));
  return m;
}
bool bitEqual(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

/// The serial host oracle of §18.5 (3D, h^{d−3} = 1): brute-force unlike pairs in index order.
struct Oracle {
  double U = 0;
  std::vector<double> gx, gw;
};
Oracle oracle(const lag::CellFacets<real_t>& Fd, const std::vector<real_t>& x,
              const std::vector<int>& ty, const std::vector<double>& V0, double L, double sigma,
              double beta, double beta0) {
  const int N = static_cast<int>(ty.size());
  const auto off = peclet::core::toVector(Fd.facetOffset);
  const auto nbr = peclet::core::toVector(Fd.nbr);
  const auto area = toHost(Fd.area), cen = toHost(Fd.centroid), conn = toHost(Fd.conn),
             sec = toHost(Fd.second), vol = toHost(Fd.volume), cc = toHost(Fd.cellCentroid);
  std::vector<double> rb(N), psi(N, 0.0), chi(N), a(3 * N), e(N);
  for (int i = 0; i < N; ++i)
    rb[i] = 2.5 * std::cbrt(V0[i]);
  auto rmin = [&](int i, int j, double r[3]) {
    double r2 = 0;
    for (int c = 0; c < 3; ++c) {
      double d = x[3 * i + c] - x[3 * j + c];
      d -= L * std::round(d / L);
      r[c] = d;
      r2 += d * d;
    }
    return std::sqrt(r2);
  };
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      if (ty[j] == ty[i])
        continue;
      double r[3];
      const double q = rmin(i, j, r) / rb[i];
      if (q < 1)
        psi[i] += std::pow(1 - q, 4) * (1 + 4 * q);
    }
  Oracle o;
  o.gx.assign(3 * N, 0.0);
  o.gw.assign(N, 0.0);
  for (int i = 0; i < N; ++i) {
    const double ex = std::exp(-psi[i] / 0.1);
    chi[i] = 1 - ex;
    double o2 = 0;
    for (int c = 0; c < 3; ++c)
      o2 += cc[3 * i + c] * cc[3 * i + c];
    const double coef = sigma * (beta0 + beta * chi[i]);
    o.U += coef * o2;
    for (int c = 0; c < 3; ++c)
      a[3 * i + c] = -2 * coef * cc[3 * i + c];
    e[i] = beta * sigma * o2 * ex / 0.1;
  }
  for (int k = 0; k < N; ++k) {
    for (int c = 0; c < 3; ++c)
      o.gx[3 * k + c] += a[3 * k + c];
    for (int j = 0; j < N; ++j) {  // ψ_k's and ψ_j's dependence on x_k
      if (ty[j] == ty[k])
        continue;
      double r[3];
      const double d = rmin(k, j, r);
      double f = 0;
      if (d / rb[k] < 1)
        f += -20 * e[k] * std::pow(1 - d / rb[k], 3) / (rb[k] * rb[k]);
      if (d / rb[j] < 1)
        f += -20 * e[j] * std::pow(1 - d / rb[j], 3) / (rb[j] * rb[j]);
      for (int c = 0; c < 3; ++c)
        o.gx[3 * k + c] += f * r[c];
    }
    for (int f = off[k]; f < off[k + 1]; ++f) {
      const int c2 = nbr[f];
      if (!(beta0 > 0 || chi[k] > 0 || chi[c2] > 0))
        continue;
      // Ψ(ξ) = a_k·(ξ + o_k)/V_k − a_c·(ξ − D + o_c)/V_c, integrated with the facet moments
      double D[3], d2 = 0;
      for (int c = 0; c < 3; ++c) {
        D[c] = conn[3 * f + c];
        d2 += D[c] * D[c];
      }
      const double d = std::sqrt(d2), A = area[f];
      auto Psi = [&](const double xi[3]) {
        double v = 0;
        for (int c = 0; c < 3; ++c)
          v += a[3 * k + c] * (xi[c] - cc[3 * k + c]) / vol[k] -
               a[3 * c2 + c] * (xi[c] - D[c] - cc[3 * c2 + c]) / vol[c2];
        return v;
      };
      const double xf[3] = {cen[3 * f], cen[3 * f + 1], cen[3 * f + 2]};
      double grad[3];  // ∇_ξΨ (constant)
      for (int c = 0; c < 3; ++c)
        grad[c] = a[3 * k + c] / vol[k] - a[3 * c2 + c] / vol[c2];
      const double P0 = Psi(xf);
      for (int c = 0; c < 3; ++c) {
        double I1 = A * P0 * xf[c];
        for (int b = 0; b < 3; ++b)
          I1 += sec[9 * f + 3 * c + b] * grad[b];
        o.gx[3 * k + c] -= I1 / d;
      }
      o.gw[k] -= A * P0 / (2 * d);
    }
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("Offset binding U_b (backend=%s):\n", Kokkos::DefaultExecutionSpace::name());
    const int n = 8, N = n * n * n;
    const real_t L = 1.0, h = L / n;
    const real_t Larr[3] = {L, L, L};
    std::mt19937 rng(11u);
    std::uniform_real_distribution<real_t> U(-1, 1);
    std::vector<real_t> x(3 * N);
    for (int k = 0, q = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i, ++q) {
          const int ijk[3] = {i, j, k};
          for (int a = 0; a < 3; ++a) {
            real_t p = (ijk[a] + real_t(0.5) + real_t(0.3) * U(rng)) * h;
            x[3 * q + a] = p < 0 ? p + L : (p >= L ? p - L : p);
          }
        }
    const DV pos = toDev(x, "pos");
    DV w("w", N);
    MT mp;
    mp.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
    mp.setWeights(w);
    mp.rebuild(pos);
    std::vector<double> V0(N);
    {
      const auto F0 = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, pos, w, N, Larr);
      const auto V = toHost(F0.volume);
      double s = 0, s0 = 0;
      for (int i = 0; i < N; ++i) {
        V0[i] = V[i] * (1 + 0.2 * U(rng));
        s += V[i];
        s0 += V0[i];
      }
      for (auto& v : V0)
        v *= s / s0;
    }
    const auto R = lag::solveWeights(mp, pos, w, V0, Larr);
    std::printf("  weight solve: converged=%d max|V-V0|/V0=%.1e\n", (int)R.converged, R.maxRelErr);
    rc |= R.converged ? 0 : 1;
    const std::vector<real_t> wh = toHost(w);

    std::vector<int> ty(N), same(N, 0);
    for (int i = 0; i < N; ++i) {
      const real_t dx = x[3 * i] - 0.5, dy = x[3 * i + 1] - 0.5, dz = x[3 * i + 2] - 0.5;
      ty[i] = dx * dx + dy * dy + dz * dz < 0.09 ? 1 : 0;
    }
    const auto dty = peclet::core::toDevice<int>(ty, "ty");
    const auto dsame = peclet::core::toDevice<int>(same, "same");
    const DV ten = toDev({0.0, 1.0, 1.0, 0.0}, "ten"), ten0 = toDev({0.0, 0.0, 0.0, 0.0}, "ten0");

    // U_b (and its partials into gx, gw when given) at (xx, wt) on a fresh matched export
    auto evalB = [&](const std::vector<real_t>& xx, const std::vector<real_t>& wt,
                     const lag::BindingParams<real_t>& bp, DV gx, DV gw,
                     lag::CellFacets<real_t>* Fout = nullptr) {
      const DV p = toDev(xx, "p"), wd = toDev(wt, "wt");
      mp.setWeights(wd);
      mp.rebuild(p);
      auto F = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, p, wd, N, Larr, {}, {},
                                                                  mp.vol);
      lag::matchFacets(F);
      if (gx.extent(0) == 0) {
        gx = DV("gx", 3 * N);
        gw = DV("gw", N);
      }
      if (Fout)
        *Fout = F;
      return lag::offsetBinding(F, p, dty, ten, 2, bp, Larr, gx, gw, false);
    };

    struct Case {
      const char* name;
      double beta, beta0;
    };
    const Case cases[3] = {{"defaults (0.1, 1e-3)", 0.1, 1e-3},
                           {"band alone (0.1, 0)", 0.1, 0.0},
                           {"floor alone (0, 1e-3)", 0.0, 1e-3}};
    for (const Case& cs : cases) {
      const auto bp = lag::bindingParams<real_t>(V0, cs.beta, cs.beta0);
      std::printf(" [%s]\n", cs.name);
      DV gx("gx", 3 * N), gw("gw", N);
      lag::CellFacets<real_t> F;
      lag::BindingScratch<real_t> scr;
      const DV p0 = toDev(x, "p0"), w0 = toDev(wh, "w0");
      mp.setWeights(w0);
      mp.rebuild(p0);
      F = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, p0, w0, N, Larr, {}, {},
                                                             mp.vol);
      lag::matchFacets(F);
      const real_t Ub = lag::offsetBinding(F, p0, dty, ten, 2, bp, Larr, gx, gw, false, &scr);
      const auto hx = toHost(gx), hw = toHost(gw);
      std::printf("  U_b = %.6e, band %d cells, |o|/h max %.3e rms %.3e\n", Ub, scr.nBand,
                  scr.offMax, scr.offRms);
      rc |= scr.active && Ub > 0 ? 0 : 1;

      // (A) device = host
      if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Mem>::accessible) {
        const Oracle o = oracle(F, x, ty, V0, L, 1.0, cs.beta, cs.beta0);
        double dx = 0, dw = 0;
        for (int q = 0; q < 3 * N; ++q)
          dx = std::max(dx, std::fabs(hx[q] - o.gx[q]));
        for (int q = 0; q < N; ++q)
          dw = std::max(dw, std::fabs(hw[q] - o.gw[q]));
        const double eU = std::fabs(Ub - o.U) / o.U, eX = dx / maxAbs(o.gx), eW = dw / maxAbs(o.gw);
        std::printf("  (A) device = host: U %.1e, dU/dx %.1e, dU/dw %.1e (<= 1e-12, rel. max)\n",
                    eU, eX, eW);
        rc |= (eU <= 1e-12 && eX <= 1e-12 && eW <= 1e-12) ? 0 : 1;
      } else {
        std::printf("  (A) device = host: skipped (device memory)\n");
      }

      // (B) FD at fixed w (x directions) and at fixed x (w directions)
      {
        const auto chi = toHost(scr.chi);
        std::vector<int> band;
        for (int i = 0; i < N; ++i)
          if (chi[i] > 0)
            band.push_back(i);
        if (band.empty())
          for (int i = 0; i < N; ++i)
            if (ty[i] == 1)
              band.push_back(i);
        double worstX = 0, worstW = 0;
        std::mt19937 r2(5u);
        for (int dir = 0; dir < 6; ++dir) {
          const bool all = dir < 3;
          std::vector<real_t> d(3 * N, 0.0), dwv(N, 0.0);
          if (all) {
            for (auto& v : d)
              v = U(r2);
            for (auto& v : dwv)
              v = U(r2);
          } else {
            for (int k = 0; k < 8 && k < (int)band.size(); ++k) {
              const int i = band[(k * 37 + dir * 11) % band.size()];
              for (int a = 0; a < 3; ++a)
                d[3 * i + a] = U(r2);
              dwv[i] = U(r2);
            }
          }
          double ga = 0, sc = 0, gwa = 0, scw = 0;
          for (int q = 0; q < 3 * N; ++q) {
            ga += hx[q] * d[q];
            sc += std::fabs(hx[q] * d[q]);
          }
          for (int q = 0; q < N; ++q) {
            gwa += hw[q] * dwv[q];
            scw += std::fabs(hw[q] * dwv[q]);
          }
          double bx = 1e300, bw = 1e300;
          for (double st : {1e-6, 1e-5, 1e-4}) {
            const double s = st * h;
            std::vector<real_t> xp(x), xm(x);
            for (int q = 0; q < 3 * N; ++q) {
              xp[q] += s * d[q];
              xm[q] -= s * d[q];
            }
            const double fd =
                (evalB(xp, wh, bp, DV{}, DV{}) - evalB(xm, wh, bp, DV{}, DV{})) / (2 * s);
            bx = std::min(bx, std::fabs(fd - ga) / sc);
            const double sw = st * h * h;
            std::vector<real_t> wp(wh), wm(wh);
            for (int q = 0; q < N; ++q) {
              wp[q] += sw * dwv[q];
              wm[q] -= sw * dwv[q];
            }
            const double fdw =
                (evalB(x, wp, bp, DV{}, DV{}) - evalB(x, wm, bp, DV{}, DV{})) / (2 * sw);
            bw = std::min(bw, std::fabs(fdw - gwa) / scw);
          }
          worstX = std::max(worstX, bx);
          worstW = std::max(worstW, bw);
        }
        std::printf(
            "  (B) FD (best step of 1e-6..1e-4 h), max over 3 all + 3 band directions: "
            "dU/dx %.1e, dU/dw %.1e (<= 1e-7 of sum|g.d|)\n",
            worstX, worstW);
        rc |= (worstX <= 1e-7 && worstW <= 1e-7) ? 0 : 1;
      }

      // (C) translation; Σ ∂x and Σ ∂w
      {
        std::vector<real_t> xs(x);
        const real_t t[3] = {0.0123 * h, -0.371 * h, 0.2549 * h};
        for (int i = 0; i < N; ++i)
          for (int a = 0; a < 3; ++a) {
            real_t p = xs[3 * i + a] + t[a];
            xs[3 * i + a] = p < 0 ? p + L : (p >= L ? p - L : p);
          }
        const double Us = evalB(xs, wh, bp, DV{}, DV{});
        double sx[3] = {0, 0, 0}, ax = sumAbs(hx), sw = 0;
        for (int i = 0; i < N; ++i) {
          for (int a = 0; a < 3; ++a)
            sx[a] += hx[3 * i + a];
          sw += hw[i];
        }
        const double eT = std::fabs(Us - Ub) / Ub,
                     eX = std::max({std::fabs(sx[0]), std::fabs(sx[1]), std::fabs(sx[2])}) / ax,
                     eW = std::fabs(sw) / sumAbs(hw);
        std::printf(
            "  (C) translation |dU|/U %.1e (<= 1e-12); |sum dU/dx| %.1e, |sum dU/dw| %.1e "
            "(<= 1e-12 of sum|.|)\n",
            eT, eX, eW);
        rc |= (eT <= 1e-12 && eX <= 1e-12 && eW <= 1e-12) ? 0 : 1;
      }
    }

    // (D) the skip rule and the off spec
    {
      const DV p0 = toDev(x, "p0"), w0 = toDev(wh, "w0");
      mp.setWeights(w0);
      mp.rebuild(p0);
      auto F = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, p0, w0, N, Larr, {}, {},
                                                                  mp.vol);
      lag::matchFacets(F);
      std::vector<real_t> seed(3 * N), seedw(N);
      for (auto& v : seed)
        v = U(rng);
      for (auto& v : seedw)
        v = U(rng);
      int bad = 0;
      auto check = [&](const lag::BindingParams<real_t>& bp, const Kokkos::View<int*, Mem>& tt,
                       const DV& tn) {
        DV gx = toDev(seed, "gx"), gw = toDev(seedw, "gw");
        lag::BindingScratch<real_t> scr;
        const real_t u = lag::offsetBinding(F, p0, tt, tn, 2, bp, Larr, gx, gw, false, &scr);
        const auto hx = toHost(gx), hw = toHost(gw);
        int b = (u == 0 && !scr.active) ? 0 : 1;
        for (int q = 0; q < 3 * N; ++q)
          b += bitEqual(hx[q], seed[q]) ? 0 : 1;
        for (int q = 0; q < N; ++q)
          b += bitEqual(hw[q], seedw[q]) ? 0 : 1;
        return b;
      };
      const auto bpd = lag::bindingParams<real_t>(V0, 0.1, 1e-3);
      bad += check(lag::bindingParams<real_t>(V0, 0.0, 0.0), dty, ten);
      bad += check(bpd, dty, ten0);
      bad += check(bpd, dsame, ten);
      // partitionPotential (facet energy): binding off vs a spec without the field
      lag::PotentialSpec<real_t> s0;
      s0.types = dty;
      s0.nTypes = 2;
      s0.tension = ten;
      lag::PotentialSpec<real_t> s1 = s0;
      s1.binding = lag::bindingParams<real_t>(V0, 0.0, 0.0);
      DV a1("a1", 3 * N), a2("a2", N), b1("b1", 3 * N), b2("b2", N);
      const auto e0 = lag::partitionPotential(mp, F, p0, w0, Larr, s0, a1, a2);
      const auto e1 = lag::partitionPotential(mp, F, p0, w0, Larr, s1, b1, b2);
      const auto ha1 = toHost(a1), ha2 = toHost(a2), hb1 = toHost(b1), hb2 = toHost(b2);
      bad += bitEqual(e0.total(), e1.total()) ? 0 : 1;
      for (int q = 0; q < 3 * N; ++q)
        bad += bitEqual(ha1[q], hb1[q]) ? 0 : 1;
      for (int q = 0; q < N; ++q)
        bad += bitEqual(ha2[q], hb2[q]) ? 0 : 1;
      std::printf(
          "  (D) skip rule (beta = beta0 = 0, sigma = 0, one label) and the off spec: "
          "%d values differ (== 0)\n",
          bad);
      rc |= bad == 0 ? 0 : 1;
    }
    std::printf(rc == 0 ? "PASS\n" : "FAIL\n");
  }
  Kokkos::finalize();
  return rc;
}
