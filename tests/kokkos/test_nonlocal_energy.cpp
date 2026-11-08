/**
 * @file test_nonlocal_energy.cpp
 * @brief The moment-blob nonlocal capillary energy E_M (lagrangian/nonlocal_energy.hpp, M7
 *        DESIGN_NOTE §12.3.3) on a jittered 16³ lattice of power cells (weights solved for ±20 %
 *        target volumes), a drop of radius 0.3 (two types, σ = 1):
 *
 *   (A) the facet third moments: the cell second moment Q_i from the divergence theorem on the
 *       facet third moments, Q_ab = Σ_f n_c ∫_f ξ_a ξ_b ξ_c dA / (1 + δ_ac + δ_bc) for each fixed
 *       c (no sum), against the tetrahedral (cone-fan) Q_i: ≤ 1e-13 relative;
 *   (B) the normalisation: I_s by composite Gauss–Legendre against twice the panels (≤ 1e-15) and,
 *       untapered limit, the closed form s/√(2π) to the taper's share;
 *   (C) ∂_xE_M and ∂_wE_M against central differences at fixed w, Σ_i ∂_{w_i}E_M = 0, and the
 *       adjoint ∇Ũ against central differences of Ũ(x) = U(x, w*(x)) with the weight solve re-run
 *       per evaluation (gate W0-M): all ≤ 1e-6 relative;
 *   (D) the default ('facet') path is untouched: partitionPotential with the default spec equals
 *       the raw interface energy.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/nonlocal_energy.hpp"
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

// (A) Q from the facet third moments (divergence theorem) vs the cone fan
double checkThird(const lag::CellFacets<real_t>& F) {
  const int N = F.nCells;
  const auto off = peclet::core::toVector(F.facetOffset);
  const auto A = toHost(F.area), c = toHost(F.centroid), d = toHost(F.conn), S = toHost(F.second),
             K = toHost(F.third), Q = toHost(F.cellSecond);
  if (F.nWall != 0)
    return 1.0;  // periodic only
  double err = 0, qmax = 0;
  for (real_t v : Q)
    qmax = std::max(qmax, std::fabs(v));
  for (int i = 0; i < N; ++i) {
    double Qd[3][3][3] = {};  // [c][a][b]
    for (int f = off[i]; f < off[i + 1]; ++f) {
      const double dn = std::sqrt(d[3 * f] * d[3 * f] + d[3 * f + 1] * d[3 * f + 1] +
                                  d[3 * f + 2] * d[3 * f + 2]);
      const double* cf = &c[3 * f];
      const double* Sf = &S[9 * f];
      for (int cc = 0; cc < 3; ++cc) {
        const double n = d[3 * f + cc] / dn;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            // ∫ ξ_a ξ_b ξ_c = A c_a c_b c_c + c_a S_bc + c_b S_ac + c_c S_ab + K_abc
            const double m3 = A[f] * cf[a] * cf[b] * cf[cc] + cf[a] * Sf[3 * b + cc] +
                              cf[b] * Sf[3 * a + cc] + cf[cc] * Sf[3 * a + b] +
                              K[10 * f + lag::sym3Index(a, b, cc)];
            Qd[cc][a][b] += n * m3 / (1.0 + (a == cc) + (b == cc));
          }
      }
    }
    for (int cc = 0; cc < 3; ++cc)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          err = std::max(err, std::fabs(Qd[cc][a][b] - Q[9 * i + 3 * a + b]) / qmax);
  }
  return err;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("Nonlocal capillary energy E_M (backend=%s):\n",
                Kokkos::DefaultExecutionSpace::name());
    const int n = 16, N = n * n * n;
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
    // power cells: weights solved for ±20 % targets
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

    // (A)
    const auto Fh = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, pos, w, N, Larr,
                                                                       {}, {}, mp.vol, true);
    const double eQ = checkThird(Fh);
    std::printf("  (A) Q_i from facet third moments vs cone fan: %.1e (<= 1e-13)\n", eQ);
    rc |= eQ <= 1e-13 ? 0 : 1;

    // (B)
    const double I1 = lag::nonlocalUnitIntegral(), I2 = lag::nonlocalUnitIntegral(32, 20);
    const double eI = std::fabs(I1 - I2) / I2;
    std::printf("  (B) I_s/s = %.16f (untapered 1/sqrt(2pi) = %.16f), panel doubling %.1e\n", I1,
                1.0 / std::sqrt(2 * 3.14159265358979323846), eI);
    rc |= (eI <= 1e-15 && I1 < 1.0 / std::sqrt(2 * 3.14159265358979323846)) ? 0 : 1;

    // (C) a drop of radius 0.3, σ = 1
    std::vector<int> ty(N);
    for (int i = 0; i < N; ++i) {
      const real_t dx = x[3 * i] - 0.5, dy = x[3 * i + 1] - 0.5, dz = x[3 * i + 2] - 0.5;
      ty[i] = dx * dx + dy * dy + dz * dz < 0.09 ? 1 : 0;
    }
    lag::PotentialSpec<real_t> spec;
    spec.types = peclet::core::toDevice<int>(ty, "ty");
    spec.nTypes = 2;
    spec.tension = toDev({0.0, 1.0, 1.0, 0.0}, "ten");
    spec.interfaceEnergy = lag::InterfaceEnergy::Nonlocal;
    spec.refVolume = toDev(V0, "V0");
    spec.nonlocal = lag::nonlocalParams<real_t>(V0);
    auto energy = [&](const std::vector<real_t>& xx, const DV& wt, DV gx, DV gw,
                      lag::CellFacets<real_t>* Fout = nullptr) {
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
      const auto e = lag::partitionPotential(mp, F, p, wt, Larr, spec, gx, gw);
      if (Fout)
        *Fout = F;
      return e.interface;
    };
    auto reduced = [&](const std::vector<real_t>& xx) {
      DV wt("wt", N);
      Kokkos::deep_copy(wt, w);
      const DV p = toDev(xx, "p");
      mp.setWeights(wt);
      mp.rebuild(p);
      lag::WeightSolveParams prm;
      prm.tol = 1e-14;
      prm.maxIter = 20;
      lag::solveWeights(mp, p, wt, V0, Larr, prm);
      return energy(xx, wt, DV{}, DV{});
    };
    DV gx("gx", 3 * N), gw("gw", N), pi("pi", N), grad("grad", 3 * N);
    lag::CellFacets<real_t> F;
    const double E = energy(x, w, gx, gw, &F);
    lag::WeightSolveParams aprm;
    aprm.pcgTol = 1e-14;
    lag::reducedGradient(F, gx, gw, pi, grad, aprm);
    const auto hx = toHost(gx), hw = toHost(gw), hg = toHost(grad);
    double sw = 0;
    for (real_t v : hw)
      sw += v;
    const double eSum = std::fabs(sw) / maxAbs(hw);
    // probe the interface cells with the largest |∂_xE|
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i)
      idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      return std::fabs(hx[3 * a]) + std::fabs(hx[3 * a + 1]) + std::fabs(hx[3 * a + 2]) >
             std::fabs(hx[3 * b]) + std::fabs(hx[3 * b + 1]) + std::fabs(hx[3 * b + 2]);
    });
    double eX = 0, eW = 0, eR = 0;
    const real_t ex = real_t(1e-4) * h, ew = real_t(1e-4) * h * h;
    const auto wv = toHost(w);
    for (int k : {idx[0], idx[N / 50]}) {
      for (int a = 0; a < 3; ++a) {
        std::vector<real_t> xp = x, xm = x;
        xp[3 * k + a] += ex;
        xm[3 * k + a] -= ex;
        const double fd = (energy(xp, w, DV{}, DV{}) - energy(xm, w, DV{}, DV{})) / (2 * ex);
        eX = std::max(eX, std::fabs(fd - hx[3 * k + a]) / maxAbs(hx));
        const double fr = (reduced(xp) - reduced(xm)) / (2 * ex);
        eR = std::max(eR, std::fabs(fr - hg[3 * k + a]) / maxAbs(hg));
      }
      std::vector<real_t> wp = wv, wm = wv;
      wp[k] += ew;
      wm[k] -= ew;
      const double fd =
          (energy(x, toDev(wp, "wp"), DV{}, DV{}) - energy(x, toDev(wm, "wm"), DV{}, DV{})) /
          (2 * ew);
      eW = std::max(eW, std::fabs(fd - hw[k]) / maxAbs(hw));
    }
    std::printf(
        "  (C) E_M = %.12f (4 pi R^2 = %.6f): FD dE/dx=%.1e dE/dw=%.1e sum dE/dw=%.1e | "
        "grad U~ vs FD(U~, OT re-solved)=%.1e  (<= 1e-6)\n",
        E, 4 * 3.14159265358979323846 * 0.09, eX, eW, eSum, eR);
    rc |= (eX <= 1e-6 && eW <= 1e-6 && eR <= 1e-6 && eSum <= 1e-12) ? 0 : 1;

    // (D) the default spec is the raw facet energy
    {
      mp.setWeights(w);
      mp.rebuild(pos);
      lag::PotentialSpec<real_t> raw = spec;
      raw.interfaceEnergy = lag::InterfaceEnergy::Facet;
      DV a("a", 3 * N), b("b", N), c3("c", 3 * N), d1("d", N);
      const auto e1 = lag::partitionPotential(mp, F, pos, w, Larr, raw, a, b);
      lag::PotentialSpec<real_t> raw2;
      raw2.types = spec.types;
      raw2.nTypes = 2;
      raw2.tension = spec.tension;
      const auto e2 = lag::partitionPotential(mp, F, pos, w, Larr, raw2, c3, d1);
      // the raw path sums with atomics: equal to round-off (bit-identity of the default
      // path is proved by python/state_hash.py at OMP_NUM_THREADS=1)
      const auto ha = toHost(a), hc = toHost(c3), hb = toHost(b), hd = toHost(d1);
      double eD = std::fabs(e1.interface - e2.interface) / std::fabs(e2.interface);
      for (std::size_t q = 0; q < ha.size(); ++q)
        eD = std::max(eD, std::fabs(ha[q] - hc[q]) / maxAbs(hc));
      for (std::size_t q = 0; q < hb.size(); ++q)
        eD = std::max(eD, std::fabs(hb[q] - hd[q]) / maxAbs(hd));
      const bool same = eD <= 1e-13;
      std::printf("  (D) default spec = facet energy %.12f, nonlocal fields ignored: %.1e\n",
                  e2.interface, eD);
      rc |= same ? 0 : 1;
    }
    std::printf("%s\n",
                rc == 0 ? "ALL NONLOCAL ENERGY CHECKS PASS" : "NONLOCAL ENERGY CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
