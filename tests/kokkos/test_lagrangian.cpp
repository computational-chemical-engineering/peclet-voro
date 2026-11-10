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
 *       a force-free state stays exactly at rest. The same drop with the 'pressure' kick 1
 *       (KickConstraint::kPressure, D14.1: one saddle per step, restart steps by option S) at τ,
 *       τ/2, τ/4: volumes to 1e-12 and the energy error falling ≥ 3.5× per halving.
 *   (B′) The ζ-plan (DissipationPlan, D14.5) against the direct ζ apply on random fields
 *       (‖Δ‖₂/‖ζv‖₂), its symmetry, and its rigid-mode null (translations; rotations on the rows
 *       free of periodic wraps), each ≤ 1e-14; on the SDF-walled tessellation of (G) the plan
 *       against the direct apply only (the per-side SDF clip makes ζ itself non-symmetric there).
 *   (G) The light export of an OT evaluation (exportFacetsLight, M7 WO-PERF3 P3.1): nFacets,
 *       facetOffset, nbr, area and conn bit-identical to the full export's, on a walled power
 *       tessellation (SDF sphere solid) with seeds inside the solid (empty cells, whose faces are
 *       exported as wall facets of their live neighbours); and a FacetPool export bit-identical to
 *       a one-shot one in every array.
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

// (B′) the ζ-plan (DissipationPlan, M7 DESIGN_NOTE D14.5) against the direct applyDissipation on
// random fields and a random positive viscosity: ‖ζv − ζv_direct‖₂/‖ζv_direct‖₂; the plan's
// symmetry |⟨ζu, v⟩ − ⟨u, ζv⟩|/(‖ζu‖‖v‖); its rigid modes, relative to a random field's
// max|ζu|/max|u|: translations (all rows, `walls` false) and the rotations v = Ω x on the rows
// whose 3-ring holds no periodic wrap (ζ v_i reads v up to three facet rings away), when `x` (the
// generators, 3N) is given. Returns the worst; each ≤ 1e-14 is the P3.5 gate.
double checkZetaPlan(const lag::CellFacets<real_t>& F, const std::vector<real_t>& x, bool walls,
                     const char* tag) {
  const int N = F.nCells;
  std::mt19937 rng(41);  // its own generator: the problems drawn after it stay as they were
  std::normal_distribution<real_t> G(0, 1);
  std::vector<real_t> v(3 * N), u(3 * N), mu(N);
  for (auto& q : v)
    q = G(rng);
  for (auto& q : u)
    q = G(rng);
  for (auto& q : mu)
    q = std::exp(real_t(0.5) * G(rng));
  const DV dv = toDev(v, "zp.v"), du = toDev(u, "zp.u"), dmu = toDev(mu, "zp.mu");
  DV o1("zp.o1", 3 * N), o2("zp.o2", 3 * N);
  lag::DissipationScratch<real_t> scr(N);
  lag::DissipationPlan<real_t> P;
  P.build(F, dmu);
  P.build(F, dmu);  // a rebuild reuses the buffers
  double eDir = 0;
  for (const DV& f : {dv, du}) {
    lag::applyDissipation(F, dmu, f, o1, scr);
    lag::applyDissipation(P, f, o2, scr);
    const auto a = toHost(o1), b = toHost(o2);
    double dd = 0, nn = 0;
    for (int k = 0; k < 3 * N; ++k) {
      dd += (a[k] - b[k]) * (a[k] - b[k]);
      nn += a[k] * a[k];
    }
    eDir = std::max(eDir, std::sqrt(dd / nn));
  }
  lag::applyDissipation(P, du, o1, scr);
  lag::applyDissipation(P, dv, o2, scr);
  const auto Zu = toHost(o1), Zv = toHost(o2);
  const double eSym = std::fabs(dot(Zu, v) - dot(u, Zv)) / std::sqrt(dot(Zu, Zu) * dot(v, v));
  const double scale = maxAbs(Zu) / maxAbs(u);
  double eTr = 0, eRot = 0;
  int nRows = 0;
  if (!walls) {
    std::vector<real_t> t(3 * N);
    for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
          t[3 * i + c] = c == a ? 1.0 : 0.0;
      lag::applyDissipation(P, toDev(t, "zp.t"), o1, scr);
      eTr = std::max(eTr, maxAbs(toHost(o1)) / scale);
    }
  }
  if (!x.empty()) {
    const auto off = peclet::core::toVector(F.facetOffset), nb = peclet::core::toVector(F.nbr);
    const auto cn = toHost(F.conn);
    std::vector<char> near(N, 0);
    for (int i = 0; i < N; ++i)
      for (int f = off[i]; f < off[i + 1]; ++f)
        for (int c = 0; c < 3; ++c)
          if (std::fabs(x[3 * nb[f] + c] - x[3 * i + c] - cn[3 * f + c]) > 1e-9)
            near[i] = 1;
    for (int ring = 0; ring < 3; ++ring) {
      std::vector<char> nx = near;
      for (int i = 0; i < N; ++i)
        for (int f = off[i]; f < off[i + 1]; ++f)
          if (near[nb[f]])
            nx[i] = 1;
      near.swap(nx);
    }
    const double Om[3][3] = {{0, 0.7, -1.3}, {-0.7, 0, 0.4}, {1.3, -0.4, 0}};
    std::vector<real_t> r(3 * N);
    for (int i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a)
        r[3 * i + a] = Om[a][0] * x[3 * i] + Om[a][1] * x[3 * i + 1] + Om[a][2] * x[3 * i + 2];
    lag::applyDissipation(P, toDev(r, "zp.r"), o1, scr);
    const auto Zr = toHost(o1);
    double m = 0;
    for (int i = 0; i < N; ++i)
      if (!near[i]) {
        ++nRows;
        for (int c = 0; c < 3; ++c)
          m = std::max(m, double(std::fabs(Zr[3 * i + c])));
      }
    eRot = m / (scale * maxAbs(r));
  }
  std::printf(
      "      zeta-plan %s: vs direct %.1e, symmetry %.1e, translation %.1e, rotation %.1e (%d "
      "rows)\n",
      tag, eDir, eSym, eTr, eRot, nRows);
  // walls: the SDF clip is per cell side, so the two sides of a cut facet differ and ζ itself
  // (direct or planned) is not symmetric there; the plan is gated against the direct apply only
  return walls ? eDir : std::max({eDir, eSym, eTr, eRot});
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
    const double mom = std::sqrt(rn / bn);
    // D14.3 (M7 WO-PERF3 P3.2): ‖C v‖_∞ ≤ η_K ‖C‖ v_ref, η_K = 1e-9. Checked here independently of
    // the solver (WO-PERF3-R1, review F2a: st.constraint ≤ st.tolUsed is the solver's own exit
    // test, so it cannot fail): ‖C v‖_∞ recomputed from the returned v by applyDivergence, ‖C‖ =
    // max_i (Σ_f A_f |c_f|/|d_f| + Σ_wall A_w) and v_ref = max_i |b_i|/m_i from the export and b
    // on the host, η_K the D14.3 literal (a changed SaddleParams default fails here)
    DV cvd("e.cv", N);
    lag::applyDivergence(F, dv, cvd);
    const auto cv = toHost(cvd), Ah = toHost(F.area), ch3 = toHost(F.centroid),
               dh3 = toHost(F.conn), Aw = toHost(F.wallArea);
    const auto fo = peclet::core::toVector(F.facetOffset),
               wo = peclet::core::toVector(F.wallOffset);
    double cvInf = 0, cNorm = 0, vRef = 0;
    for (int i = 0; i < N; ++i) {
      cvInf = std::max(cvInf, std::fabs(double(cv[i])));
      double ai = 0;
      for (int f = fo[i]; f < fo[i + 1]; ++f) {
        double c2 = 0, d2 = 0;
        for (int e = 0; e < 3; ++e) {
          c2 += double(ch3[3 * f + e]) * ch3[3 * f + e];
          d2 += double(dh3[3 * f + e]) * dh3[3 * f + e];
        }
        ai += Ah[f] * std::sqrt(c2 / d2);
      }
      for (int f = wo[i]; f < wo[i + 1]; ++f)
        ai += Aw[f];
      cNorm = std::max(cNorm, ai);
      vRef = std::max(vRef,
                      std::sqrt(double(b[3 * i]) * b[3 * i] + double(b[3 * i + 1]) * b[3 * i + 1] +
                                double(b[3 * i + 2]) * b[3 * i + 2]) /
                          mass[i]);
    }
    const double etaK = 1e-9, bound = etaK * cNorm * vRef;
    std::printf(
        "  (E) saddle: outer=%d inner=%d |Cv|_inf recomputed %.2e <= eta_K |C| v_ref = %.2e (%.3f "
        "of it; solver's own bound %.2e) momentum resid=%.1e\n",
        st.outer, st.inner, cvInf, bound, cvInf / bound, double(st.tolUsed), mom);
    rc |= (cvInf <= bound && mom <= 1e-8) ? 0 : 1;
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
    ip.kickConstraint = lag::KickConstraint::kMidpoint;
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

  // --- the same drop with the 'pressure' kick 1 (D14.1, M7 WO-PERF3 P3.4; restart = option S):
  // one saddle per step, volumes held, K + Ũ + Q + Q_c second order at three τ (≥ 3.5× per
  // halving, DESIGN_NOTE §14.7) ---
  {
    lag::PotentialSpec<real_t> spec;
    spec.types = dty;
    spec.nTypes = 2;
    spec.tension = toDev({0.0, 1.0, 1.0, 0.0}, "ep.ten");
    lag::FlowParams fp;
    fp.density = {1.0, 10.0};
    fp.viscosity = {0.1, 0.3};
    lag::IntegratorParams ip;
    ip.kickConstraint = lag::KickConstraint::kPressure;
    ip.gaugeEvery = 0;
    double err[3] = {0, 0, 0}, worstV = 0;
    int restarts = 0, saddles = 0, steps = 0;
    for (int run = 0; run < 3; ++run) {
      DV p2 = toDev(x, "ep.pos2"), w2 = toDev(w0, "ep.w2");
      MT m2;
      m2.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
      m2.setWeights(w2);
      m2.rebuild(p2);
      lag::Integrator<MT> I(m2, p2, w2, Larr, V0, spec, fp, v, ip);
      const double E0 = I.kinetic() + I.potential().total();
      const int n = 4 << run;
      const double tau = 8e-3 / n;
      for (int s = 0; s < n; ++s) {
        const auto r = I.step(tau);
        restarts += r.kickRestart;
        saddles += 1 + (r.kick1.outer > 0 ? 1 : 0);
        ++steps;
        const double E = r.kinetic + r.potential + r.heat + r.constraintWork;
        err[run] = std::max(err[run], std::fabs(E - E0) / E0);
        const auto Vh = toHost(Kokkos::subview(m2.vol, std::make_pair(0, N)));
        for (int i = 0; i < N; ++i)
          worstV = std::max(worstV, std::fabs(Vh[i] - V0[i]) / V0[i]);
      }
    }
    std::printf(
        "  (E) drop, pressure kick: max|V-V0|/V0=%.1e restarts=%d saddles/step=%.2f |dE|/E0 at "
        "tau, tau/2, tau/4 = %.2e, %.2e, %.2e (ratios %.2f, %.2f)\n",
        worstV, restarts, double(saddles) / steps, err[0], err[1], err[2], err[0] / err[1],
        err[1] / err[2]);
    rc |= (worstV <= 1e-12 && restarts == 3 && err[0] / err[1] >= 3.5 && err[1] / err[2] >= 3.5)
              ? 0
              : 1;
  }

  // --- trajectory self-convergence (M7 WO-PERF3-R1, review F2d): the ledger above closes to the
  // trapezoid error of Ũ whatever the pressures are, so its ratio is blind to an O(τ) pressure
  // error. Here the state (x, v) at T = 8e-3 for τ, τ/2, τ/4: e_k = max|q_τk − q_τk/2| (x by
  // minimum image), e_0/e_1 ≥ 3.5 for a second-order trajectory. The pressure field comes from
  // facet gravity on the 1:10 drop, not tension: raw-facet E_σ is not smooth in x (DESIGN_NOTE §12,
  // the reason for E_M) and with σ = 1 the velocity converges at first order in both kick modes
  // (measured ratio 2.0); E_M needs a larger box than 12³. The start is on the constraint (v after
  // one 1e-7 step). Gated: the default 'pressure' kick. Reported only: 'midpoint' (option S), whose
  // velocity ratio falls with τ (2.83, 2.21 at 12³; 4.10, 3.11 at 16³ for τ/2, τ/4 and τ/4, τ/8;
  // 'pressure' 3.89, 3.95 and 3.90, 3.96) — WO-PERF3-R1 reports it to the orchestrator ---
  {
    lag::PotentialSpec<real_t> spec;
    spec.types = dty;
    spec.nTypes = 2;
    spec.density = toDev({1.0, 10.0}, "et.rho");
    spec.gravity[2] = -1.0;
    lag::FlowParams fp;
    fp.density = {1.0, 10.0};
    fp.viscosity = {0.1, 0.3};
    lag::IntegratorParams ip;
    ip.gaugeEvery = 0;
    std::vector<real_t> xs, ws, vs;  // the projected start
    {
      DV p2 = toDev(x, "et.pos0"), w2 = toDev(w0, "et.w0");
      MT m2;
      m2.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
      m2.setWeights(w2);
      m2.rebuild(p2);
      ip.kickConstraint = lag::KickConstraint::kMidpoint;
      lag::Integrator<MT> I(m2, p2, w2, Larr, V0, spec, fp, v, ip);
      I.step(1e-7);
      xs = toHost(p2);
      ws = toHost(w2);
      vs = toHost(I.velocity());
      xs.resize(3 * N);
      ws.resize(N);
      vs.resize(3 * N);
    }
    const lag::KickConstraint modes[2] = {lag::KickConstraint::kPressure,
                                          lag::KickConstraint::kMidpoint};
    const char* names[2] = {"pressure", "midpoint"};
    for (int md = 0; md < 2; ++md) {
      ip.kickConstraint = modes[md];
      std::vector<real_t> xT[3], vT[3];
      for (int run = 0; run < 3; ++run) {
        DV p2 = toDev(xs, "et.pos"), w2 = toDev(ws, "et.w");
        MT m2;
        m2.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
        m2.setWeights(w2);
        m2.rebuild(p2);
        lag::Integrator<MT> I(m2, p2, w2, Larr, V0, spec, fp, vs, ip);
        const int n = 4 << run;
        for (int s = 0; s < n; ++s)
          I.step(8e-3 / n);
        xT[run] = toHost(p2);
        vT[run] = toHost(I.velocity());
      }
      double ex[2] = {0, 0}, ev[2] = {0, 0}, dxMax = 0, vMax = 0;
      const auto mi = [L](double d) { return d - L * std::nearbyint(d / L); };
      for (int k = 0; k < 3 * N; ++k) {
        dxMax = std::max(dxMax, std::fabs(mi(xT[2][k] - xs[k])));
        vMax = std::max(vMax, std::fabs(double(vT[2][k])));
        for (int e = 0; e < 2; ++e) {
          ex[e] = std::max(ex[e], std::fabs(mi(xT[e][k] - xT[e + 1][k])));
          ev[e] = std::max(ev[e], std::fabs(double(vT[e][k]) - vT[e + 1][k]));
        }
      }
      const double rx = ex[0] / ex[1], rv = ev[0] / ev[1];
      std::printf(
          "  (E) trajectory at T, %s kick: |dx|/max|x-x0| = %.2e, %.2e (ratio %.2f), |dv|/max|v| "
          "= %.2e, %.2e (ratio %.2f); ratios >= 3.5%s\n",
          names[md], ex[0] / dxMax, ex[1] / dxMax, rx, ev[0] / vMax, ev[1] / vMax, rv,
          modes[md] == lag::KickConstraint::kPressure ? "" : " (reported, not gated)");
      // the differences must also stand far above the η_K = 1e-9 solve noise
      if (modes[md] == lag::KickConstraint::kPressure)
        rc |= (rx >= 3.5 && rv >= 3.5 && ex[1] >= 1e-6 * dxMax && ev[1] >= 1e-6 * vMax) ? 0 : 1;
    }
  }

  // --- (F) R0 (DESIGN_NOTE §12.3.1): the Voronoi-cell descent of F = E_σ + volume penalty from
  // the Voronoi volumes, then OT; F and the area fall, the closing checks hold. Then the gauge:
  // one gauged inviscid step keeps Σ m v and C v = 0, with ΔŨ, ΔK ≤ 0 ---
  {
    lag::PotentialSpec<real_t> spec;
    spec.types = dty;
    spec.nTypes = 2;
    spec.tension = toDev({0.0, 1.0, 1.0, 0.0}, "f.ten");
    lag::FlowParams fp;
    fp.density = {1.0, 10.0};
    fp.viscosity = {0.0, 0.0};
    lag::RelaxReport rr;
    {
      DV p3 = toDev(x, "f.pos3"), w3("f.w3", N);
      MT m3;
      m3.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
      m3.setWeights(w3);
      m3.rebuild(p3);
      const auto Vh = toHost(Kokkos::subview(m3.vol, std::make_pair(0, N)));
      const std::vector<double> Vvor(Vh.begin(), Vh.end());
      lag::Integrator<MT> I(m3, p3, w3, Larr, Vvor, spec, fp, std::vector<real_t>(3 * N, 0.0));
      lag::RelaxParams rp;
      rp.maxIter = 200;
      rp.window = 50;
      rr = I.relax(rp);
    }
    std::printf(
        "  (F) R0: %d it (converged %d) F %.4f -> %.4f, Voronoi area %.4f -> %.4f; OT %d it "
        "max|V-V0|/V0=%.1e max s/d=%.3f min margin/h=%.3f power area %.4f passed=%d\n",
        rr.iterations, int(rr.converged), rr.energy.front(), rr.energy.back(), rr.area.front(),
        rr.area.back(), rr.otIterations, rr.maxVolumeError, rr.maxShift, rr.minMargin / h,
        rr.unlikeArea, int(rr.passed));
    rc |= (rr.passed && rr.energy.back() < rr.energy.front() && rr.area.back() < rr.area.front())
              ? 0
              : 1;
    DV p2 = toDev(x, "f.pos"), w2 = toDev(w0, "f.w");
    MT m2;
    m2.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
    m2.setWeights(w2);
    m2.rebuild(p2);
    lag::Integrator<MT> I(m2, p2, w2, Larr, V0, spec, fp, v);
    I.setGaugeMode(lag::GaugeMode::kTrial);  // G″'s trial (the default; set so the test says so)
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
        "  (F) gauge G'' step: max alpha=%.2e |I|=%d projected=%d capped=%d tries=%d accepted=%d "
        "dU=%.2e dK=%.2e min margin/h=%.3f |dP|/sum m|v|=%.1e |Cv| tau/maxV=%.1e\n",
        r.gaugeAlphaMax, r.gaugeInterface, r.gaugeProjected, r.gaugeCapped, r.gaugeTries,
        int(r.gaugeAccepted), r.gaugePotential, r.gaugeKinetic, r.minMargin / h, dP, cmax);
    rc |= (r.gaugeTries > 0 && r.gaugeAccepted && r.gaugePotential <= 0 && r.gaugeKinetic <= 0 &&
           r.minMargin > 0 && dP <= 1e-12 && cmax <= 10 * lag::SaddleParams{}.tol)
              ? 0
              : 1;
    // the drift gauge (GaugeMode::kDrift, §14.4.4) from the same state: step 1 computes δ¹ (no
    // trial), step 2's drift adds it; momentum and C v as for G″
    DV p4 = toDev(x, "f.pos4"), w4 = toDev(w0, "f.w4");
    MT m4;
    m4.alloc(N, Larr, real_t(1e-10) * h, real_t(0.25) * h, 4, N);
    m4.setWeights(w4);
    m4.rebuild(p4);
    lag::Integrator<MT> J(m4, p4, w4, Larr, V0, spec, fp, v);
    J.setGaugeMode(lag::GaugeMode::kDrift);
    const auto r1 = J.step(tau);
    const auto r2 = J.step(tau);
    const auto vj = toHost(J.velocity());
    double Pj[3] = {0, 0, 0};
    for (int i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a)
        Pj[a] += (ty[i] ? 10.0 : 1.0) * V0[i] * vj[3 * i + a];
    double dPj = 0;
    for (int a = 0; a < 3; ++a)
      dPj = std::max(dPj, std::fabs(Pj[a] - P0[a]) / pn);
    lag::applyDivergence(J.facets(), J.velocity(), cv);
    double cj = 0;
    for (double c : toHost(cv))
      cj = std::max(cj, std::fabs(c) * tau / maxV);
    std::printf(
        "  (F) drift gauge: step 1 max alpha=%.2e active=%d applied=%d; step 2 applied=%d redos=%d "
        "dU_G=%.2e f_G=%.3f tries=%d min margin/h=%.3f |dP|/sum m|v|=%.1e |Cv| tau/maxV=%.1e\n",
        r1.gaugeAlphaMax, r1.gaugeActive, r1.gaugeApplied, r2.gaugeApplied, r2.gaugeRedos,
        r2.gaugePotential, r2.gaugeFactor, r1.gaugeTries + r2.gaugeTries, r2.minMargin / h, dPj,
        cj);
    rc |= (r1.gaugeActive > 0 && r1.gaugeApplied == 0 && r2.gaugeApplied == 1 &&
           r2.gaugeRedos == 0 && r1.gaugeTries + r2.gaugeTries == 0 && r2.minMargin > 0 &&
           dPj <= 1e-12 && cj <= 10 * lag::SaddleParams{}.tol)
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
    int tries = 0;
    for (int s = 0; s < 3; ++s) {
      const auto r = I.step(0.01);
      vmax = std::max(vmax, r.maxSpeed);
      tries += r.gaugeTries;
    }
    std::printf(
        "  (E) rest: max|v| after 3 steps = %.1e (exactly 0 expected), gauge trials %d (0: "
        "skipped at rest)\n",
        vmax, tries);
    rc |= (vmax == 0.0 && tries == 0) ? 0 : 1;
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

template <class T>
bool sameBits(const Kokkos::View<T*, Mem>& a, const Kokkos::View<T*, Mem>& b) {
  if (a.extent(0) != b.extent(0))
    return false;
  const auto ha = peclet::core::toVector(a), hb = peclet::core::toVector(b);
  return ha.empty() || std::memcmp(ha.data(), hb.data(), ha.size() * sizeof(T)) == 0;
}

// (G) the light export vs the full one, and pooled vs one-shot exports
int checkLightExport() {
  using Sdf = peclet::voro::SdfSphere<real_t>;
  using peclet::voro::Power;
  const real_t L = 1.0, Larr[3] = {L, L, L};
  const Sdf ball{0.5, 0.5, 0.5, 0.25};
  std::mt19937 rng(29);
  std::uniform_real_distribution<real_t> U(0, 1);
  std::vector<real_t> x;
  int nSolid = 0;
  while (x.size() < 3 * 1500) {
    const real_t p[3] = {U(rng), U(rng), U(rng)};
    const bool solid = !(ball.eval(p[0], p[1], p[2]) > 0);
    if (solid && nSolid >= 4)
      continue;
    nSolid += solid ? 1 : 0;
    x.insert(x.end(), p, p + 3);
  }
  const int N = static_cast<int>(x.size() / 3);
  const real_t h = std::cbrt(L * L * L / N);
  std::vector<real_t> wh(N);
  for (auto& v : wh)
    v = real_t(0.05) * h * h * (2 * U(rng) - 1);
  const DV pos = toDev(x, "pos"), w = toDev(wh, "w");
  peclet::voro::MovingTessellation<real_t, kP, kT, true, Sdf> mt;
  mt.sdf = ball;
  mt.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
  mt.setWeights(w);
  mt.rebuild(pos);
  int nEmpty = 0;
  for (real_t v : toHost(mt.vol))
    nEmpty += v > 0 ? 0 : 1;
  const auto full =
      lag::exportCellFacets<real_t, Power>(mt.store, pos, w, N, Larr, mt.wall, mt.xRef, mt.vol);
  const auto light =
      lag::exportFacetsLight<real_t, Power>(mt.store, pos, w, N, Larr, mt.wall, mt.xRef, mt.vol);
  lag::FacetPool<real_t> pool;
  const auto lightP = lag::exportFacetsLight<real_t, Power>(mt.store, pos, w, N, Larr, mt.wall,
                                                            mt.xRef, mt.vol, &pool);
  const auto fullP = lag::exportCellFacets<real_t, Power>(mt.store, pos, w, N, Larr, mt.wall,
                                                          mt.xRef, mt.vol, false, &pool);
  bool okL = true;
  for (const auto* Fl : {&light, &lightP})
    okL = okL && Fl->nCells == full.nCells && Fl->nFacets == full.nFacets &&
          sameBits(Fl->facetOffset, full.facetOffset) && sameBits(Fl->nbr, full.nbr) &&
          sameBits(Fl->area, full.area) && sameBits(Fl->conn, full.conn);
  const bool okP =
      fullP.nFacets == full.nFacets && fullP.nWall == full.nWall &&
      sameBits(fullP.facetOffset, full.facetOffset) && sameBits(fullP.nbr, full.nbr) &&
      sameBits(fullP.area, full.area) && sameBits(fullP.centroid, full.centroid) &&
      sameBits(fullP.conn, full.conn) && sameBits(fullP.second, full.second) &&
      sameBits(fullP.wallOffset, full.wallOffset) && sameBits(fullP.wallArea, full.wallArea) &&
      sameBits(fullP.wallCentroid, full.wallCentroid) &&
      sameBits(fullP.wallNormal, full.wallNormal) && sameBits(fullP.volume, full.volume) &&
      sameBits(fullP.cellCentroid, full.cellCentroid);
  std::printf(
      "  (G) light export N=%d facets=%d wall facets=%d empty cells=%d: light == full %s, "
      "pooled == one-shot %s (pool %d buffers)\n",
      N, full.nFacets, full.nWall, nEmpty, okL ? "bit-identical" : "DIFFERS",
      okP ? "bit-identical" : "DIFFERS", pool.size());
  const double ez = checkZetaPlan(full, {}, true, "walled (SDF)");
  return (okL && okP && full.nWall > 0 && ez <= 1e-14) ? 0 : 1;
}

/// S₀ assembled on the device (detail::SchurAssembly) equals schurHost bit for bit — diagonal,
/// pattern and coefficients — on a fresh sparsity and on the cached one (same structure, new
/// masses).
int checkSchurDevice(const lag::CellFacets<real_t>& F0, std::mt19937& rng) {
  auto F = F0;
  const auto drop = lag::matchFacets(F);
  const int N = F.nCells;
  std::uniform_real_distribution<real_t> U(0.5, 2.0);
  const auto vol = toHost(F.volume);
  lag::detail::SchurAssembly<real_t> dev;
  auto same = [](const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
  };
  int bad = 0;
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<real_t> m(N);
    for (int i = 0; i < N; ++i)
      m[i] = vol[i] * U(rng);
    const DV mass = toDev(m, "mass");
    const auto H = lag::detail::schurHost(F, mass);
    const auto D = dev.assemble(F, mass);
    const bool ok = H.n == D.n && H.start == D.start && H.nbr == D.nbr && same(H.diag, D.diag) &&
                    same(H.coef, D.coef);
    std::printf("      S0 device vs host (pass %d, nnz %zu, dropped %d): %s\n", pass, H.coef.size(),
                drop.dropped, ok ? "bit-identical" : "DIFFERENT");
    bad |= ok ? 0 : 1;
  }
  return bad;
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
  rc |= checkZetaPlan(Fv, x, false, "Voronoi") <= 1e-14 ? 0 : 1;

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
  rc |= checkZetaPlan(Fp, x, false, "power") <= 1e-14 ? 0 : 1;
  // the exported volumes are the solved ones
  {
    const auto Vp = toHost(Fp.volume);
    double e = 0;
    for (int i = 0; i < N; ++i)
      e = std::max(e, std::fabs(Vp[i] - V0[i]) / V0[i]);
    std::printf("      exported V vs targets: %.1e\n", e);
    rc |= e <= 1e-11 ? 0 : 1;
  }
  rc |= checkSchurDevice(Fp, rng);
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
    rc |= checkLightExport();
    rc |= run(12, 1u);
    rc |= run(16, 7u);
    std::printf("%s\n", rc == 0 ? "ALL LAGRANGIAN CHECKS PASS" : "LAGRANGIAN CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
