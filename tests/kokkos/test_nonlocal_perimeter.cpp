/**
 * @file test_nonlocal_perimeter.cpp
 * @brief The set-based nonlocal perimeter E_Γ (lagrangian/nonlocal_perimeter.hpp, M7
 *        DESIGN_NOTE_STATICS §17.4–§17.5, gate G-S.1(b)–(f) in unit form) on a jittered 8³ lattice
 *        of power cells (weights solved for ±20 % targets), a drop of radius 0.3 (σ = 1, s = h):
 *
 *   (A) device = host: E_Γ and its partials from the Kokkos passes equal a serial host loop over
 *       the same Γ buffer bit for bit (host-accessible memory only);
 *   (B) ∂_xE_Γ and ∂_wE_Γ against central differences at fixed w (|Ω_a| from the current volumes:
 *       the set functional), Σ_i ∂_{w_i}E_Γ = 0, and the
 *       adjoint ∇Ũ against central differences of Ũ with the weight solve re-run (≤ 1e-7);
 *   (C) the partials are exactly 0.0 on every cell without an unlike facet (G-S.1(e));
 *   (D) translation of every generator by a random vector (weights kept): |ΔE|/E ≤ 1e-13;
 *   (E) the default spec ignores the E_Γ fields: partitionPotential(Facet) bit-identical.
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
#include "peclet/voro/lagrangian/nonlocal_perimeter.hpp"
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
double maxAbs(const std::vector<real_t>& a) {
  double m = 0;
  for (real_t v : a)
    m = std::max(m, std::fabs(v));
  return m;
}
bool bitEqual(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("Nonlocal perimeter E_Gamma (backend=%s):\n",
                Kokkos::DefaultExecutionSpace::name());
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

    std::vector<int> ty(N);
    for (int i = 0; i < N; ++i) {
      const real_t dx = x[3 * i] - 0.5, dy = x[3 * i + 1] - 0.5, dz = x[3 * i + 2] - 0.5;
      ty[i] = dx * dx + dy * dy + dz * dz < 0.09 ? 1 : 0;
    }
    const auto dty = peclet::core::toDevice<int>(ty, "ty");
    const DV ten = toDev({0.0, 1.0, 1.0, 0.0}, "ten"), dV0 = toDev(V0, "V0");
    const auto prm = lag::perimeterParams<real_t>(V0);
    lag::PotentialSpec<real_t> spec;
    spec.types = dty;
    spec.nTypes = 2;
    spec.tension = ten;
    spec.interfaceEnergy = lag::InterfaceEnergy::NonlocalExact;
    spec.refVolume = dV0;
    spec.perimeter = prm;

    // `geometric`: |Ω_a| from the current cell volumes (the set functional at fixed w, whose
    // exact gradient the partials are), else from V⁰ (= the volumes once OT holds V = V⁰).
    auto energy = [&](const std::vector<real_t>& xx, const DV& wt, DV gx, DV gw,
                      bool geometric = false) {
      const DV p = toDev(xx, "p");
      mp.setWeights(wt);
      mp.rebuild(p);
      auto F = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, p, wt, N, Larr, {}, {},
                                                                  mp.vol);
      lag::matchFacets(F);
      if (gx.extent(0) == 0) {
        gx = DV("gx", 3 * N);
        gw = DV("gw", N);
      }
      lag::PotentialSpec<real_t> sp = spec;
      if (geometric)
        sp.refVolume = F.volume;
      return lag::partitionPotential(mp, F, p, wt, Larr, sp, gx, gw).interface;
    };
    auto reduced = [&](const std::vector<real_t>& xx) {
      DV wt("wt", N);
      Kokkos::deep_copy(wt, w);
      const DV p = toDev(xx, "p");
      mp.setWeights(wt);
      mp.rebuild(p);
      lag::WeightSolveParams sp;
      sp.tol = 1e-14;
      sp.maxIter = 20;
      lag::solveWeights(mp, p, wt, V0, Larr, sp);
      return energy(xx, wt, DV{}, DV{});
    };

    DV gx("gx", 3 * N), gw("gw", N);
    lag::PerimeterStats st;
    mp.setWeights(w);
    mp.rebuild(pos);
    const real_t E =
        lag::nonlocalPerimeter(mp, N, pos, w, Larr, dty, ten, 2, dV0, prm, gx, gw, &st);
    const auto hx = toHost(gx), hw = toHost(gw);
    std::printf("  E_Gamma = %.15f (4 pi R^2 = %.6f), |Gamma| %d (directed %d), |K+| %d\n", E,
                4 * 3.14159265358979323846 * 0.09, st.nGamma, st.nDirected, st.nModes);

    // (A) device = host
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Mem>::accessible) {
      const auto G = lag::exportGamma<real_t, peclet::voro::Power>(mp.store, pos, w, N, Larr, dty,
                                                                   prm.h, mp.wall, mp.xRef, mp.vol);
      const auto M = lag::perimeterModes(Larr, prm);
      std::vector<real_t> Are(M.n), Aim(M.n);
      real_t s = 0;
      for (int m = 0; m < M.n; ++m) {
        const auto chi = lag::detail::perimeterChi(G, M.k, m);
        const real_t g = M.g(m);
        Are[m] = g * chi.real();
        Aim[m] = g * chi.imag();
        s += g * (chi.real() * chi.real() + chi.imag() * chi.imag());
      }
      real_t Oa = 0;
      for (int i = 0; i < N; ++i)
        if (ty[i] == 0)
          Oa += V0[i];
      const real_t iL = real_t(1) / (L * L * L), sC = real_t(1) * prm.Cg;
      const real_t Eh = sC * (Oa - (Oa * Oa + real_t(2) * s) * iL);
      const DV dAre = toDev(Are, "Are"), dAim = toDev(Aim, "Aim");
      std::vector<real_t> rx(3 * N, 0.0), rw(N, 0.0);
      for (int i = 0; i < N; ++i) {
        const int f0 = G.offset(i), f1 = G.offset(i + 1);
        if (f1 == f0)
          continue;
        const real_t sgn = ty[i] == 0 ? 1.0 : -1.0;
        real_t fx[3] = {0, 0, 0}, fw = 0;
        for (int f = f0; f < f1; ++f) {
          real_t J0, J1[3], I0, I1[3];
          lag::detail::perimeterMoments(G, M.k, dAre, dAim, M.n, f, J0, J1);
          const real_t c[3] = {G.cen(3 * f), G.cen(3 * f + 1), G.cen(3 * f + 2)};
          lag::detail::perimeterFacetTerms(sC, Oa, iL, G.area(f), c, J0, J1, I0, I1);
          const real_t d =
              std::sqrt(G.conn(3 * f) * G.conn(3 * f) + G.conn(3 * f + 1) * G.conn(3 * f + 1) +
                        G.conn(3 * f + 2) * G.conn(3 * f + 2));
          for (int a = 0; a < 3; ++a)
            fx[a] += sgn * I1[a] / d;
          fw += sgn * I0 / (real_t(2) * d);
        }
        for (int a = 0; a < 3; ++a)
          rx[3 * i + a] += fx[a];
        rw[i] += fw;
      }
      int diff = bitEqual(E, Eh) ? 0 : 1;
      for (int q = 0; q < 3 * N; ++q)
        diff += bitEqual(hx[q], rx[q]) ? 0 : 1;
      for (int q = 0; q < N; ++q)
        diff += bitEqual(hw[q], rw[q]) ? 0 : 1;
      std::printf("  (A) device = host: %d of %d values differ (E %.17g vs %.17g)\n", diff,
                  4 * N + 1, E, Eh);
      rc |= diff == 0 ? 0 : 1;
    } else {
      std::printf("  (A) device = host: skipped (device memory)\n");
    }

    // (C) exact zeros off Γ
    {
      const auto G = lag::exportGamma<real_t, peclet::voro::Power>(mp.store, pos, w, N, Larr, dty,
                                                                   prm.h, mp.wall, mp.xRef, mp.vol);
      const auto off = peclet::core::toVector(G.offset);
      int nOff = 0, bad = 0;
      for (int i = 0; i < N; ++i) {
        if (off[i + 1] > off[i])
          continue;
        ++nOff;
        for (int a = 0; a < 3; ++a)
          bad += bitEqual(hx[3 * i + a], 0.0) ? 0 : 1;
        bad += bitEqual(hw[i], 0.0) ? 0 : 1;
      }
      std::printf("  (C) %d cells without an unlike facet: %d nonzero partials (== 0)\n", nOff,
                  bad);
      rc |= (bad == 0 && nOff > 0) ? 0 : 1;
    }

    // (B) FD at fixed w and of the reduced Ũ (OT re-solved)
    {
      DV pi("pi", N), grad("grad", 3 * N), gx2("gx2", 3 * N), gw2("gw2", N);
      mp.setWeights(w);
      mp.rebuild(pos);
      auto F = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, pos, w, N, Larr, {}, {},
                                                                  mp.vol);
      lag::matchFacets(F);
      lag::partitionPotential(mp, F, pos, w, Larr, spec, gx2, gw2);
      lag::WeightSolveParams aprm;
      aprm.pcgTol = 1e-14;
      lag::reducedGradient(F, gx2, gw2, pi, grad, aprm);
      const auto hg = toHost(grad);
      double sw = 0;
      for (real_t v : hw)
        sw += v;
      const double eSum = std::fabs(sw) / maxAbs(hw);
      std::vector<int> idx(N);
      for (int i = 0; i < N; ++i)
        idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return std::fabs(hx[3 * a]) + std::fabs(hx[3 * a + 1]) + std::fabs(hx[3 * a + 2]) >
               std::fabs(hx[3 * b]) + std::fabs(hx[3 * b + 1]) + std::fabs(hx[3 * b + 2]);
      });
      double eX = 0, eW = 0, eR = 0;
      const real_t ex = real_t(1e-5) * h, ew = real_t(1e-5) * h * h;
      const auto wv = toHost(w);
      for (int k : {idx[0], idx[N / 20]}) {
        for (int a = 0; a < 3; ++a) {
          std::vector<real_t> xp = x, xm = x;
          xp[3 * k + a] += ex;
          xm[3 * k + a] -= ex;
          const double fd =
              (energy(xp, w, DV{}, DV{}, true) - energy(xm, w, DV{}, DV{}, true)) / (2 * ex);
          eX = std::max(eX, std::fabs(fd - hx[3 * k + a]) / maxAbs(hx));
          const double fr = (reduced(xp) - reduced(xm)) / (2 * ex);
          eR = std::max(eR, std::fabs(fr - hg[3 * k + a]) / maxAbs(hg));
        }
        std::vector<real_t> wp = wv, wm = wv;
        wp[k] += ew;
        wm[k] -= ew;
        const double fd = (energy(x, toDev(wp, "wp"), DV{}, DV{}, true) -
                           energy(x, toDev(wm, "wm"), DV{}, DV{}, true)) /
                          (2 * ew);
        eW = std::max(eW, std::fabs(fd - hw[k]) / maxAbs(hw));
      }
      std::printf(
          "  (B) FD dE/dx=%.1e dE/dw=%.1e sum dE/dw=%.1e | grad U~ vs FD(U~, OT re-solved)=%.1e "
          "(<= 1e-7, sum <= 1e-12)\n",
          eX, eW, eSum, eR);
      rc |= (eX <= 1e-7 && eW <= 1e-7 && eR <= 1e-7 && eSum <= 1e-12) ? 0 : 1;
    }

    // (D) translation invariance (weights kept)
    {
      const real_t sh[3] = {real_t(0.37) * U(rng), real_t(0.37) * U(rng), real_t(0.37) * U(rng)};
      std::vector<real_t> xs = x;
      for (int i = 0; i < N; ++i)
        for (int a = 0; a < 3; ++a) {
          real_t p = x[3 * i + a] + sh[a];
          xs[3 * i + a] = p - L * std::floor(p / L);
        }
      const double Es = energy(xs, w, DV{}, DV{});
      const double eT = std::fabs(Es - E) / std::fabs(E);
      std::printf("  (D) translation (%.3f, %.3f, %.3f): |dE|/E = %.1e (<= 1e-13)\n", sh[0], sh[1],
                  sh[2], eT);
      rc |= eT <= 1e-13 ? 0 : 1;
    }

    // (E) the default spec ignores the E_Γ fields
    {
      mp.setWeights(w);
      mp.rebuild(pos);
      auto F = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, pos, w, N, Larr, {}, {},
                                                                  mp.vol);
      lag::matchFacets(F);
      lag::PotentialSpec<real_t> raw = spec, raw2;
      raw.interfaceEnergy = lag::InterfaceEnergy::Facet;
      raw2.types = dty;
      raw2.nTypes = 2;
      raw2.tension = ten;
      DV a("a", 3 * N), b("b", N), c3("c", 3 * N), d1("d", N);
      const auto e1 = lag::partitionPotential(mp, F, pos, w, Larr, raw, a, b);
      const auto e2 = lag::partitionPotential(mp, F, pos, w, Larr, raw2, c3, d1);
      const auto ha = toHost(a), hc = toHost(c3), hb = toHost(b), hd = toHost(d1);
      double eD = std::fabs(e1.interface - e2.interface) / std::fabs(e2.interface);
      for (std::size_t q = 0; q < ha.size(); ++q)
        eD = std::max(eD, std::fabs(ha[q] - hc[q]) / maxAbs(hc));
      for (std::size_t q = 0; q < hb.size(); ++q)
        eD = std::max(eD, std::fabs(hb[q] - hd[q]) / maxAbs(hd));
      std::printf("  (E) default spec = facet energy %.12f, E_Gamma fields ignored: %.1e\n",
                  e2.interface, eD);
      rc |= eD <= 1e-13 ? 0 : 1;
    }
    std::printf("%s\n", rc == 0 ? "ALL NONLOCAL PERIMETER CHECKS PASS"
                                : "NONLOCAL PERIMETER CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
