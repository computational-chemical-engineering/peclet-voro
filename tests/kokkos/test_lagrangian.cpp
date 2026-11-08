/**
 * @file test_lagrangian.cpp
 * @brief The Lagrangian power-cell layer (lagrangian/): facet export, cell operators, weight solve.
 *
 *   (A) Facet geometry, Voronoi and power cells (weights from the weight solve, ±20 % target
 *       volumes): per cell the closure Σ_f A n̂ = 0 and the moment identity Σ_f A (c−x_i) ⊗ n̂ = V I
 *       (walls absent: periodic), V from the cone fan vs the engine's volume, Σ V = box, and each
 *       facet's second-moment tensor vs the engine's independent scalar faceMoment2
 *       (tr Σ_f + A |c − foot|² = ∫_f |y − foot|² dA, in units of h⁴).
 *   (B) Operators: H 1 = 0, C 1 = 0 (constant velocity), Cᵀ1 = 0, J_xᵀ1 = 0 (Σ V conserved),
 *       ζ(translation) = 0; adjointness ⟨C v, l⟩ = ⟨v, Cᵀ l⟩, ⟨J_x v, p⟩ = ⟨v, J_xᵀ p⟩ and ζ
 *       symmetric, all to round-off.
 *   (C) Weight solve: damped Newton from w = 0 to max|V − V⁰|/V⁰ ≤ 1e-12.
 *   (D) Potential (lagrangian/potential.hpp), periodic facet gravity (ρ_ref = mean density) on
 *       power cells: ∂_xU and ∂_wU against central differences at fixed w, Σ ∂_wU = 0, and the
 *       adjoint ∇Ũ = ∂_xU + J_xᵀπ against central differences of Ũ(x) = U(x, w*(x)) with the
 *       weight solve re-run per evaluation (DESIGN_NOTE §10.2, gate W0), all ≤ 1e-6 relative.
 *   (E) NHV-W (lagrangian/integrator.hpp): the saddle solve meets ‖Cv‖_∞ ≤ 1e-12·max V/τ and its
 *       momentum equation to 1e-8; a viscous two-phase drop with tension, integrated at τ and τ/2
 *       with option S, holds every volume to 1e-12 (each Newton iterate a REPAIR, checked against
 *       a cold rebuild to 1e-9) and its K + Ũ + Q + Q_c error falls ≥ 3× at τ/2 (second order);
 *       a force-free state stays exactly at rest.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/integrator.hpp"
#include "peclet/voro/lagrangian/operators.hpp"
#include "peclet/voro/lagrangian/potential.hpp"
#include "peclet/voro/lagrangian/weight_solve.hpp"
#include "peclet/voro/repair.hpp"

using real_t = double;
using Mem = peclet::core::MemSpace;
using DV = Kokkos::View<real_t*, Mem>;
namespace lag = peclet::voro::lagrangian;

namespace {

constexpr int kP = 64, kT = 112;

DV toDev(const std::vector<real_t>& h, const char* label) {
  return peclet::core::toDevice<real_t>(h, label);
}
std::vector<real_t> toHost(const DV& v) {
  return peclet::core::toVector(v);
}

double dot(const std::vector<real_t>& a, const std::vector<real_t>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  return s;
}
double maxAbs(const std::vector<real_t>& a) {
  double m = 0;
  for (real_t v : a)
    m = std::max(m, std::fabs(v));
  return m;
}

// (A) geometry identities of an exported tessellation; returns the worst relative error
template <class MT>
double checkGeometry(const MT& mt, const lag::CellFacets<real_t>& F, const DV& pos, const DV& w,
                     real_t L, real_t h) {
  using Policy = typename MT::PlanePolicy;
  const int N = F.nCells;
  const auto off = peclet::core::toVector(F.facetOffset);
  const auto A = toHost(F.area), c = toHost(F.centroid), d = toHost(F.conn);
  const auto V = toHost(F.volume);
  const auto Ve = peclet::core::toVector(Kokkos::subview(mt.vol, std::make_pair(0, N)));
  double eClose = 0, eMom = 0, eVol = 0, sumV = 0;
  for (int i = 0; i < N; ++i) {
    double cl[3] = {0, 0, 0}, T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int f = off[i]; f < off[i + 1]; ++f) {
      const double dl = std::sqrt(d[3 * f] * d[3 * f] + d[3 * f + 1] * d[3 * f + 1] +
                                  d[3 * f + 2] * d[3 * f + 2]);
      for (int a = 0; a < 3; ++a) {
        cl[a] += A[f] * d[3 * f + a] / dl;
        for (int b = 0; b < 3; ++b)
          T[3 * a + b] += A[f] * c[3 * f + a] * d[3 * f + b] / dl;
      }
    }
    for (int a = 0; a < 3; ++a) {
      eClose = std::max(eClose, std::fabs(cl[a]) / (h * h));
      for (int b = 0; b < 3; ++b)
        eMom = std::max(eMom, std::fabs(T[3 * a + b] - (a == b ? V[i] : 0.0)) / V[i]);
    }
    eVol = std::max(eVol, std::fabs(V[i] - Ve[i]) / Ve[i]);
    sumV += V[i];
  }
  const double eSum = std::fabs(sumV - L * L * L) / (L * L * L);
  // second moment vs the engine's scalar ∫|y − foot|² dA, per face, on the device
  double eM2 = 0;
  {
    using Cell = peclet::voro::ConvexCell<real_t, kP, kT, false>;
    const auto st = mt.store;
    const peclet::voro::WallStore<real_t> Wl{};
    const DV P = pos, W = w, XR;
    const Kokkos::Array<real_t, 3> L3 = {L, L, L};
    const real_t hh = h * h;
    Kokkos::parallel_reduce(
        "test.m2", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(const int i, double& acc) {
          Cell cc;
          lag::detail::loadCell<Policy>(st, Wl, P, W, XR, L3, i, cc);
          for (int k = 0; k < cc.np; ++k) {
            if (!lag::detail::isFace(cc, k))
              continue;
            real_t a, cen[3], sec[9], cv, cm[3];
            if (!lag::faceMoments(cc, k, a, cen, sec, cv, cm))
              continue;
            const real_t df[3] = {cen[0] - cc.n[k][0], cen[1] - cc.n[k][1], cen[2] - cc.n[k][2]};
            const real_t m2 =
                sec[0] + sec[4] + sec[8] + a * (df[0] * df[0] + df[1] * df[1] + df[2] * df[2]);
            const real_t ref = cc.faceMoment2(k);
            // in units of h⁴: faceMoment2 fans from the foot point, which for a sliver face lies
            // far outside it, so its own round-off is ~ε h⁴, not ε A h²
            const real_t e = Kokkos::fabs(m2 - ref) / (hh * hh);
            acc = e > acc ? e : acc;
          }
        },
        Kokkos::Max<double>(eM2));
  }
  std::printf("      closure=%.1e moment=%.1e V-vs-engine=%.1e sumV=%.1e M2=%.1e\n", eClose, eMom,
              eVol, eSum, eM2);
  return std::max({eClose, eMom, eVol, eSum, eM2});
}

// (B) operator null spaces and adjointness; returns the worst relative error
double checkOperators(const lag::CellFacets<real_t>& F, std::mt19937& rng) {
  const int N = F.nCells;
  std::normal_distribution<real_t> G(0, 1);
  std::vector<real_t> v(3 * N), u(3 * N), l(N), p(N), mu(N), one(N, 1.0), tr(3 * N);
  for (auto& x : v)
    x = G(rng);
  for (auto& x : u)
    x = G(rng);
  for (auto& x : l)
    x = G(rng);
  for (auto& x : p)
    x = G(rng);
  for (auto& x : mu)
    x = std::exp(real_t(0.5) * G(rng));
  for (int i = 0; i < N; ++i) {
    tr[3 * i] = 1.0;
    tr[3 * i + 1] = -2.0;
    tr[3 * i + 2] = 0.5;
  }
  const DV dv = toDev(v, "v"), du = toDev(u, "u"), dl = toDev(l, "l"), dp = toDev(p, "p"),
           dmu = toDev(mu, "mu"), d1 = toDev(one, "1"), dtr = toDev(tr, "tr");
  DV sN("sN", N), s3("s3", 3 * N), t3("t3", 3 * N);
  lag::DissipationScratch<real_t> scr(N);
  double e = 0;
  // scales: |C v| ~ |J v| ~ A |v|; use the norm of the generic result
  lag::applyDivergence(F, dv, sN);
  const auto Cv = toHost(sN);
  lag::applyDivergenceT(F, dl, s3);
  const auto Ctl = toHost(s3);
  const double eCadj = std::fabs(dot(Cv, l) - dot(v, Ctl)) / std::sqrt(dot(Cv, Cv) * dot(l, l));
  lag::applyVolumeJacobian(F, dv, sN);
  const auto Jv = toHost(sN);
  lag::applyVolumeJacobianT(F, dp, s3);
  const auto Jtp = toHost(s3);
  const double eJadj = std::fabs(dot(Jv, p) - dot(v, Jtp)) / std::sqrt(dot(Jv, Jv) * dot(p, p));
  lag::applyDissipation(F, dmu, du, s3, scr);
  const auto Zu = toHost(s3);
  lag::applyDissipation(F, dmu, dv, t3, scr);
  const auto Zv = toHost(t3);
  const double eZsym = std::fabs(dot(Zu, v) - dot(u, Zv)) / std::sqrt(dot(Zu, Zu) * dot(v, v));
  // null spaces, relative to the generic result's size
  lag::applyWeightHessian(F, d1, sN);
  const double eH1 = maxAbs(toHost(sN)) / maxAbs(Cv);
  lag::applyDivergence(F, dtr, sN);
  const double eC1 = maxAbs(toHost(sN)) / maxAbs(Cv);
  lag::applyDivergenceT(F, d1, s3);
  const double eCt1 = maxAbs(toHost(s3)) / maxAbs(Ctl);
  lag::applyVolumeJacobianT(F, d1, s3);
  const double eJt1 = maxAbs(toHost(s3)) / maxAbs(Jtp);
  lag::applyDissipation(F, dmu, dtr, s3, scr);
  const double eZtr = maxAbs(toHost(s3)) / maxAbs(Zv);
  std::printf(
      "      <Cv,l>-<v,C'l>=%.1e <Jv,p>-<v,J'p>=%.1e zeta-sym=%.1e H1=%.1e C(const)=%.1e "
      "C'1=%.1e J'1=%.1e zeta(transl)=%.1e\n",
      eCadj, eJadj, eZsym, eH1, eC1, eCt1, eJt1, eZtr);
  e = std::max({eCadj, eJadj, eZsym, eH1, eC1, eCt1, eJt1, eZtr});
  return e;
}

// (E) the NHV-W saddle and integrator on a solved power tessellation (x, weights w0, volumes V0)
int checkIntegrator(const std::vector<real_t>& x, const std::vector<real_t>& w0,
                    const std::vector<double>& V0, real_t L, real_t h) {
  using MT = peclet::voro::MovingTessellation<real_t, kP, kT, true>;
  const int N = static_cast<int>(V0.size());
  const real_t Larr[3] = {L, L, L};
  int rc = 0;
  DV pos = toDev(x, "e.pos"), w = toDev(w0, "e.w");
  MT mt;
  mt.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
  mt.setWeights(w);
  mt.rebuild(pos);
  std::vector<int> ty(N);
  for (int i = 0; i < N; ++i) {
    const real_t dx = x[3 * i] - 0.5, dy = x[3 * i + 1] - 0.5, dz = x[3 * i + 2] - 0.5;
    ty[i] = dx * dx + dy * dy + dz * dz < 0.09 ? 1 : 0;
  }
  Kokkos::View<int*, Mem> dty("e.ty", N);
  Kokkos::deep_copy(dty, Kokkos::View<const int*, Kokkos::HostSpace>(ty.data(), N));
  std::mt19937 rng(5);
  std::normal_distribution<real_t> G(0, 1);
  std::vector<real_t> v(3 * N);
  for (auto& q : v)
    q = real_t(0.01) * G(rng);

  // --- saddle: constraint to the outer bound, momentum to the inner tolerance ---
  {
    const auto F = lag::exportCellFacets<real_t, peclet::voro::Power>(mt.store, pos, w, N, Larr);
    std::vector<real_t> rho(N), mu(N), mass(N), hh(N);
    for (int i = 0; i < N; ++i) {
      rho[i] = ty[i] ? 10.0 : 1.0;
      mu[i] = ty[i] ? 0.3 : 0.1;
      mass[i] = rho[i] * V0[i];
      hh[i] = std::cbrt(V0[i]);
    }
    const DV drho = toDev(rho, "e.rho"), dmu = toDev(mu, "e.mu"), dm = toDev(mass, "e.m"),
             dh = toDev(hh, "e.h");
    lag::SaddleSolver<real_t> S;
    S.setup(F, drho, dmu, dm, dh);
    const real_t tau = 0.2 * h, a = 0.5 * 0.5 * tau, ht = 0.5 * tau;
    std::vector<real_t> b(3 * N);
    for (auto& q : b)
      q = G(rng) * 1e-4;
    const DV db = toDev(b, "e.b"), dv("e.v", 3 * N), dl("e.l", N);
    double maxV = 0;
    for (double q : V0)
      maxV = std::max(maxV, q);
    const lag::SaddleParams prm;
    const auto st = S.solve(a, ht, real_t(prm.tol * maxV / tau), db, dv, dl, prm);
    // momentum residual (M + aζ) v + (τ/2) Cᵀλ − b
    DV zv("e.zv", 3 * N), cl("e.cl", 3 * N);
    lag::DissipationScratch<real_t> scr(N);
    lag::applyDissipation(F, dmu, dv, zv, scr);
    lag::applyDivergenceT(F, dl, cl);
    const auto vh = toHost(dv), zh = toHost(zv), ch = toHost(cl);
    double rn = 0, bn = 0;
    for (int k = 0; k < 3 * N; ++k) {
      const double r = mass[k / 3] * vh[k] + a * zh[k] + ht * ch[k] - b[k];
      rn += r * r;
      bn += b[k] * b[k];
    }
    const double cons = st.constraint * tau / maxV, mom = std::sqrt(rn / bn);
    std::printf("  (E) saddle: outer=%d inner=%d |Cv|_inf tau/maxV=%.1e momentum resid=%.1e\n",
                st.outer, st.inner, cons, mom);
    rc |= (cons <= 1e-12 && mom <= 1e-8) ? 0 : 1;
  }

  // --- a viscous drop with tension over T = 8e-3 at τ and τ/2 (option S: kick 1 constrained at
  // the midpoint): volumes held, the repair against a cold rebuild, K + Ũ + Q + Q_c second order
  // ---
  {
    lag::PotentialSpec<real_t> spec;
    spec.types = dty;
    spec.nTypes = 2;
    spec.tension = toDev({0.0, 1.0, 1.0, 0.0}, "e.ten");
    lag::FlowParams fp;
    fp.density = {1.0, 10.0};
    fp.viscosity = {0.1, 0.3};
    lag::IntegratorParams ip;
    ip.midpointConstraint = true;
    ip.gaugeEvery = 0;  // the integrator's own order (the gauge is checked in (F))
    double err[2] = {0, 0}, worstV = 0, worstCold = 0;
    int fallbacks = 0;
    for (int run = 0; run < 2; ++run) {
      DV p2 = toDev(x, "e.pos2"), w2 = toDev(w0, "e.w2");
      MT m2;
      m2.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);  // tight certificate (M7)
      m2.setWeights(w2);
      m2.rebuild(p2);
      lag::Integrator<MT> I(m2, p2, w2, Larr, V0, spec, fp, v, ip);
      const double E0 = I.kinetic() + I.potential().total();
      const int n = 4 << run;
      const double tau = 8e-3 / n;
      for (int s = 0; s < n; ++s) {
        const auto r = I.step(tau);
        fallbacks += r.repairFallbacks;
        const double E = r.kinetic + r.potential + r.heat + r.constraintWork;
        err[run] = std::max(err[run], std::fabs(E - E0) / E0);
        const auto Vh = toHost(Kokkos::subview(m2.vol, std::make_pair(0, N)));
        for (int i = 0; i < N; ++i)
          worstV = std::max(worstV, std::fabs(Vh[i] - V0[i]) / V0[i]);
        MT mc;
        mc.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
        mc.setWeights(w2);
        mc.rebuild(p2);
        const auto Vc = toHost(Kokkos::subview(mc.vol, std::make_pair(0, N)));
        for (int i = 0; i < N; ++i)
          worstCold = std::max(worstCold, std::fabs(Vc[i] - Vh[i]) / V0[i]);
      }
    }
    std::printf(
        "  (E) drop, option S: max|V-V0|/V0=%.1e repair vs cold=%.1e fallbacks=%d |dE|/E0 at tau, "
        "tau/2 = %.2e, %.2e (ratio %.2f)\n",
        worstV, worstCold, fallbacks, err[0], err[1], err[0] / err[1]);
    rc |= (worstV <= 1e-12 && worstCold <= 1e-9 && err[0] / err[1] >= 3.0) ? 0 : 1;
  }

  // --- (F) the gauge: relax() descends Ũ at held volumes; one gauged inviscid step keeps Σ m v
  // and C v = 0, with ΔŨ, ΔK ≤ 0 ---
  {
    lag::PotentialSpec<real_t> spec;
    spec.types = dty;
    spec.nTypes = 2;
    spec.tension = toDev({0.0, 1.0, 1.0, 0.0}, "f.ten");
    lag::FlowParams fp;
    fp.density = {1.0, 10.0};
    fp.viscosity = {0.0, 0.0};
    DV p2 = toDev(x, "f.pos"), w2 = toDev(w0, "f.w");
    MT m2;
    m2.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
    m2.setWeights(w2);
    m2.rebuild(p2);
    double A0 = 0, A1 = 0, dU = 0, worstV = 0;
    int iters = 0, acc = 0, rebuilds = 0;
    {
      lag::Integrator<MT> I(m2, p2, w2, Larr, V0, spec, fp, std::vector<real_t>(3 * N, 0.0));
      A0 = I.unlikeArea();
      const auto rr = I.relax(1e-4, 10, 100);
      A1 = I.unlikeArea();
      dU = rr.potentialChange;
      iters = rr.iterations;
      acc = rr.accepted;
      rebuilds = I.restoreRebuilds();
      const auto Vh = toHost(Kokkos::subview(m2.vol, std::make_pair(0, N)));
      for (int i = 0; i < N; ++i)
        worstV = std::max(worstV, std::fabs(Vh[i] - V0[i]) / V0[i]);
    }
    lag::Integrator<MT> I(m2, p2, w2, Larr, V0, spec, fp, v);
    double P0[3] = {0, 0, 0}, P1[3] = {0, 0, 0}, pn = 0, maxV = 0;
    for (int i = 0; i < N; ++i) {
      const double m = (ty[i] ? 10.0 : 1.0) * V0[i];
      maxV = std::max(maxV, V0[i]);
      for (int a = 0; a < 3; ++a) {
        P0[a] += m * v[3 * i + a];
        pn += m * std::fabs(v[3 * i + a]);
      }
    }
    const double tau = 2e-3;
    const auto r = I.step(tau);
    const auto vh = toHost(I.velocity());
    for (int i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a)
        P1[a] += (ty[i] ? 10.0 : 1.0) * V0[i] * vh[3 * i + a];
    double dP = 0;
    for (int a = 0; a < 3; ++a)
      dP = std::max(dP, std::fabs(P1[a] - P0[a]) / pn);
    DV cv("f.cv", N);
    lag::applyDivergence(I.facets(), I.velocity(), cv);
    double cmax = 0;
    for (double c : toHost(cv))
      cmax = std::max(cmax, std::fabs(c) * tau / maxV);
    std::printf(
        "  (F) gauge: relax %d it (%d accepted, %d restore rebuilds) area %.4f -> %.4f dU=%.2e "
        "max|V-V0|/V0=%.1e; step: tries=%d accepted=%d dU=%.2e dK=%.2e |dP|/sum m|v|=%.1e |Cv| "
        "tau/maxV=%.1e\n",
        iters, acc, rebuilds, A0, A1, dU, worstV, r.gaugeTries, int(r.gaugeAccepted),
        r.gaugePotential, r.gaugeKinetic, dP, cmax);
    rc |= (dU <= 0 && A1 < A0 && acc > 0 && worstV <= 1e-12 && r.gaugeTries > 0 &&
           r.gaugePotential <= 0 && r.gaugeKinetic <= 0 && dP <= 1e-12 && cmax <= 1e-11)
              ? 0
              : 1;
  }

  // --- rest: one phase, no forces, v = 0 stays exactly 0 (the gauge on: Ũ ≡ 0, v ≡ 0) ---
  {
    Kokkos::deep_copy(dty, 0);
    lag::PotentialSpec<real_t> spec;
    spec.types = dty;
    spec.nTypes = 1;
    lag::FlowParams fp;
    fp.density = {1.0};
    fp.viscosity = {0.1};
    lag::Integrator<MT> I(mt, pos, w, Larr, V0, spec, fp, std::vector<real_t>(3 * N, 0.0));
    double vmax = 0;
    for (int s = 0; s < 3; ++s)
      vmax = std::max(vmax, I.step(0.01).maxSpeed);
    std::printf("  (E) rest: max|v| after 3 steps = %.1e (exactly 0 expected)\n", vmax);
    rc |= vmax == 0.0 ? 0 : 1;
  }
  return rc;
}

// (D) facet gravity + adjoint by finite differences; returns the worst relative error
template <class MT>
double checkPotential(MT& mp, const std::vector<real_t>& x, const DV& w,
                      const std::vector<double>& V0, real_t L, real_t h) {
  const int N = static_cast<int>(V0.size());
  const real_t Larr[3] = {L, L, L};
  const real_t g[3] = {real_t(0.2), real_t(-0.1), real_t(-1)};
  std::vector<int> ty(N);
  std::vector<real_t> rho = {1.0, 3.0};
  double m = 0, vol = 0;
  for (int i = 0; i < N; ++i) {
    const real_t dx = x[3 * i] - 0.5 * L, dy = x[3 * i + 1] - 0.5 * L, dz = x[3 * i + 2] - 0.5 * L;
    ty[i] = dx * dx + dy * dy + dz * dz < 0.09 * L * L ? 1 : 0;
    m += rho[ty[i]] * V0[i];
    vol += V0[i];
  }
  const real_t rhoRef = m / vol;
  const Kokkos::View<int*, Mem> dty = peclet::core::toDevice<int>(ty, "ty");
  const DV drho = toDev(rho, "rho");
  // U(x, wt), and optionally ∂_xU, ∂_wU, at a rebuild of mp
  auto energy = [&](const std::vector<real_t>& xx, const DV& wt, DV gx, DV gw) {
    const DV p = toDev(xx, "p");
    mp.setWeights(wt);
    mp.rebuild(p);
    const auto F = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, p, wt, N, Larr);
    if (gx.extent(0) == 0) {
      gx = DV("gx", 3 * N);
      gw = DV("gw", N);
    }
    return std::make_pair(lag::facetGravity(F, p, dty, drho, g, rhoRef, Larr, gx, gw), F);
  };
  // Ũ(xx): the weights re-solved for V0 from a warm start
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
    return energy(xx, wt, DV{}, DV{}).first;
  };
  DV gx("gx", 3 * N), gw("gw", N), pi("pi", N), grad("grad", 3 * N);
  const auto F = energy(x, w, gx, gw).second;
  lag::WeightSolveParams aprm;
  aprm.pcgTol = 1e-14;
  const auto ar = lag::reducedGradient(F, gx, gw, pi, grad, aprm);
  const auto hx = toHost(gx), hw = toHost(gw), hg = toHost(grad);
  double sw = 0;
  for (real_t v : hw)
    sw += v;
  const double eSum = std::fabs(sw) / maxAbs(hw);
  double eX = 0, eW = 0, eR = 0, eXonly = 0;
  const real_t ex = real_t(1e-4) * h, ew = real_t(1e-4) * h * h;
  for (int k : {N / 3, (2 * N) / 3}) {  // interior generators: no wrap within ±ex
    for (int a = 0; a < 3; ++a) {
      std::vector<real_t> xp = x, xm = x;
      xp[3 * k + a] += ex;
      xm[3 * k + a] -= ex;
      const double fd =
          (energy(xp, w, DV{}, DV{}).first - energy(xm, w, DV{}, DV{}).first) / (2 * ex);
      eX = std::max(eX, std::fabs(fd - hx[3 * k + a]) / maxAbs(hx));
      const double fr = (reduced(xp) - reduced(xm)) / (2 * ex);
      eR = std::max(eR, std::fabs(fr - hg[3 * k + a]) / maxAbs(hg));
      eXonly = std::max(eXonly, std::fabs(fr - hx[3 * k + a]) / maxAbs(hg));
    }
    std::vector<real_t> wv = toHost(w);
    std::vector<real_t> wp = wv, wm = wv;
    wp[k] += ew;
    wm[k] -= ew;
    const double fd = (energy(x, toDev(wp, "wp"), DV{}, DV{}).first -
                       energy(x, toDev(wm, "wm"), DV{}, DV{}).first) /
                      (2 * ew);
    eW = std::max(eW, std::fabs(fd - hw[k]) / maxAbs(hw));
  }
  mp.setWeights(w);  // leave mp at the reference state
  mp.rebuild(toDev(x, "x"));
  std::printf(
      "  (D) facet gravity FD: dU/dx=%.1e dU/dw=%.1e sum dU/dw=%.1e | adjoint (pcg %d it, %.1e): "
      "grad U~ vs FD(U~)=%.1e (dU/dx alone %.1e)\n",
      eX, eW, eSum, ar.pcgIters, ar.pcgRelRes, eR, eXonly);
  return std::max({eX, eW, eR, eSum * 1e6});
}

int run(int n, unsigned seed) {
  const real_t L = 1.0, h = L / n;
  const int N = n * n * n;
  std::mt19937 rng(seed);
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
  const real_t Larr[3] = {L, L, L};
  int rc = 0;
  const double tolGeom = 1e-12, tolOp = 1e-12;

  // --- Voronoi ---
  peclet::voro::MovingTessellation<real_t, kP, kT, false> mv;
  mv.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
  mv.rebuild(pos);
  const auto Fv =
      lag::exportCellFacets<real_t, peclet::voro::Voronoi>(mv.store, pos, DV{}, N, Larr);
  std::printf("  (A) Voronoi  N=%d facets=%d\n", N, Fv.nFacets);
  double eg = checkGeometry(mv, Fv, pos, DV{}, L, h);
  double eo = checkOperators(Fv, rng);
  rc |= (eg <= tolGeom && eo <= tolOp) ? 0 : 1;

  // --- power: weights from the weight solve on ±20 % targets (C) ---
  const auto Vvor = toHost(Fv.volume);
  std::vector<double> V0(N);
  double s = 0, s0 = 0;
  for (int i = 0; i < N; ++i) {
    V0[i] = Vvor[i] * (1 + 0.2 * U(rng));
    s += Vvor[i];
    s0 += V0[i];
  }
  for (auto& v : V0)
    v *= s / s0;
  DV w("w", N);
  peclet::voro::MovingTessellation<real_t, kP, kT, true> mp;
  mp.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
  mp.setWeights(w);
  mp.rebuild(pos);
  const auto R = lag::solveWeights(mp, pos, w, V0, Larr);
  std::printf("  (C) weight solve: converged=%d iterations=%d max|V-V0|/V0=%.2e (",
              (int)R.converged, R.iters, R.maxRelErr);
  for (const auto& st : R.history)
    std::printf(" %.1e/a=%.2g/pcg=%d", st.maxRelErr, st.alpha, st.pcgIters);
  std::printf(" )\n");
  rc |= (R.converged && R.maxRelErr <= 1e-12) ? 0 : 1;
  const auto Fp = lag::exportCellFacets<real_t, peclet::voro::Power>(mp.store, pos, w, N, Larr);
  std::printf("  (A) power    N=%d facets=%d\n", N, Fp.nFacets);
  eg = checkGeometry(mp, Fp, pos, w, L, h);
  eo = checkOperators(Fp, rng);
  rc |= (eg <= tolGeom && eo <= tolOp) ? 0 : 1;
  // the exported volumes are the solved ones
  {
    const auto Vp = toHost(Fp.volume);
    double e = 0;
    for (int i = 0; i < N; ++i)
      e = std::max(e, std::fabs(Vp[i] - V0[i]) / V0[i]);
    std::printf("      exported V vs targets: %.1e\n", e);
    rc |= e <= 1e-11 ? 0 : 1;
  }
  const auto wSolved = toHost(w);
  rc |= checkPotential(mp, x, w, V0, L, h) <= 1e-6 ? 0 : 1;
  rc |= checkIntegrator(x, wSolved, V0, L, h);
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("Lagrangian power-cell layer (backend=%s):\n",
                Kokkos::DefaultExecutionSpace::name());
    rc |= run(12, 1u);
    rc |= run(16, 7u);
    std::printf("%s\n", rc == 0 ? "ALL LAGRANGIAN CHECKS PASS" : "LAGRANGIAN CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
