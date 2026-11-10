/**
 * @file lagrangian/nonlocal_energy.hpp
 * \brief The moment-blob nonlocal capillary energy E_M of the Lagrangian power-cell method (M7
 * NHV-W, DESIGN_NOTE §12.3.3), opt-in (`interface_energy = 'nonlocal'`), with its derivatives in
 * positions AND weights:
 *
 *   E_M(x, w) = Σ_{a<b} σ_ab C_s Σ_{i∈a} Σ_{k∈b} V⁰_i V⁰_k N(z_ik; S_ik) T(|z_ik|/r_c),
 *
 * the second-order, moment-matched value of the Gaussian nonlocal perimeter C_s ∫∫_{Ω_a×Ω_b} G_s.
 * Per cell, in generator-local coordinates ξ = y − x_i: M_i = ∫_{C_i} ξ dV, Q_i = ∫_{C_i} ξξᵀ dV,
 * c_i = M_i/V⁰_i, Σ_i = Q_i/V⁰_i − c_i c_iᵀ; per pair
 *
 *   z_ik = (x_i − x_k)_min-image + c_i − c_k,   S_ik = s² I + Σ_i + Σ_k,
 *   N(z; S) = (2π)^{−3/2} (det S)^{−1/2} exp(−½ zᵀS⁻¹z),
 *
 * s = κ_s h (κ_s = 1, h = (Σ V⁰/N)^{1/3}), r_c = 5 s, T the C² taper (1 on q ≤ 0.75, the quintic
 * smoothstep down to 0 at q = 1), C_s = 1/I_s with I_s = π ∫_0^{r_c} ρ³ (2πs²)^{−3/2} e^{−ρ²/2s²}
 * T(ρ/r_c) dρ (composite Gauss–Legendre to round-off). V⁰ (not V) weights the pairs: the two agree
 * on the OT solution, so Ũ is unchanged and the partials need only ∂M and ∂Q.
 *
 * Derivatives. Per pair ∂e/∂z = e_0 [−T p + T′ z/(|z| r_c)] (p = S⁻¹z, e_0 = σC_sV⁰_iV⁰_kN) and
 * ∂e/∂S = ½ e_0 T (p pᵀ − S⁻¹), gathered per cell into g_i = ∂E/∂c_i (= the explicit ∂E/∂x_i of
 * the min-image term) and G_i = ∂E/∂Σ_i. With c = M/V⁰ and Σ = Q/V⁰ − c cᵀ,
 *
 *   δE = Σ_i a_i·δM_i + B_i:δQ_i + g_i·δx_i,   a_i = (g_i − 2 G_i c_i)/V⁰_i,  B_i = G_i/V⁰_i.
 *
 * A facet f of cell i toward j (connector r = x_j′ − x_i, d = |r|) moves outward from i by
 * u_f(ξ) = [ξ·δx_i − (ξ − r)·δx_j]/d + (δw_i − δw_j)/(2d), and δ∫_{C_i} g dV = Σ_f ∫_f g u_f dA
 * for an integrand fixed in space; the local integrand ξ = y − x_i adds −V_i δx_i to δM_i and
 * −(M_i δx_iᵀ + δx_i M_iᵀ) to δQ_i. With φ_i(ξ) = a_i·ξ + ξᵀB_iξ and, on the facet,
 * Φ_f(ξ) = φ_i(ξ) − φ_j(ξ − r) (the j-side's moments, read in i's frame), the facet terms gather
 * over each cell's own facets (the j-side of the same facet gives the identical expression):
 *
 *   ∂E/∂x_i = g_i − V_i a_i − 2 B_i M_i + Σ_f ∫_f Φ_f ξ dA / d,   ∂E/∂w_i = Σ_f ∫_f Φ_f dA /(2d).
 *
 * Φ_f is quadratic, so ∫_f Φ_f ξ dA needs the facet third moment (lagrangian/facets.hpp,
 * exportCellFacets(..., withHigherMoments = true)); everything is expanded about the facet
 * centroid. Wall facets are fixed and contribute nothing.
 *
 * The static pressure (M7 DESIGN_NOTE_VARIANTS D15.2) needs e_i = ∂E_M/∂V⁰_i at fixed x and w:
 * the pair prefactor and the moment normalisation (c = M/V⁰ and Σ = Q/V⁰ − c cᵀ, so that
 * ∂c_i/∂V⁰_i = −c_i/V⁰_i and ∂Σ_i/∂V⁰_i = (c_i c_iᵀ − Σ_i)/V⁰_i):
 *
 *   e_i = [ Σ_k e_ik − g_i·c_i + G_i:(c_i c_iᵀ − Σ_i) ] / V⁰_i.
 *
 * The widths' dependence on V⁰ (through h = (Σ V⁰/N)^{1/3}) adds the same term to every cell and
 * is left out: a uniform shift of the pressure. e_i = 0 on a cell without pairs.
 *
 * Pairs: unlike (i, k) (σ_{t_i t_k} ≠ 0) with |x_i − x_k|_min-image < r_c + 2h, kept if |z| < r_c,
 * found through a uniform bin grid (cell size ≥ r_c + 2h, contents sorted by index so that every
 * sum is in a fixed order: the result is deterministic). The min image is unique while
 * r_c + 2h ≤ L/2 on every axis (checked).
 */
#ifndef PECLET_VORO_LAGRANGIAN_NONLOCAL_ENERGY_HPP
#define PECLET_VORO_LAGRANGIAN_NONLOCAL_ENERGY_HPP

#include <cmath>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/operators.hpp"  // LView

namespace peclet::voro::lagrangian {

/// The widths and the normalisation of E_M (DESIGN_NOTE §12.3.3).
template <class Real>
struct NonlocalParams {
  Real h = 0;   ///< (Σ V⁰ / N)^{1/3}
  Real s = 0;   ///< κ_s h
  Real rc = 0;  ///< 5 s
  Real Cs = 0;  ///< 1 / I_s
  bool valid() const { return s > Real(0) && Cs > Real(0); }
};

namespace detail {

/// n-point Gauss–Legendre nodes and weights on [−1, 1] (Newton on P_n, host).
inline void gaussLegendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < (n + 1) / 2; ++i) {
    double z = std::cos(pi * (i + 0.75) / (n + 0.5)), dp = 0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1, p1 = z;
      for (int k = 2; k <= n; ++k) {
        const double p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = n * (z * p1 - p0) / (z * z - 1);
      const double dz = p1 / dp;
      z -= dz;
      if (std::fabs(dz) < 1e-16)
        break;
    }
    {  // final derivative at the converged node
      double p0 = 1, p1 = z;
      for (int k = 2; k <= n; ++k) {
        const double p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = n * (z * p1 - p0) / (z * z - 1);
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2.0 / ((1 - z * z) * dp * dp);
  }
}

/// The C² taper T(q) and dT/dq.
template <class Real>
KOKKOS_INLINE_FUNCTION void nonlocalTaper(Real q, Real& T, Real& dT) {
  if (q <= Real(0.75)) {
    T = Real(1);
    dT = Real(0);
  } else if (q >= Real(1)) {
    T = Real(0);
    dT = Real(0);
  } else {
    const Real u = (q - Real(0.75)) / Real(0.25);
    const Real u2 = u * u, v = Real(1) - u;
    T = Real(1) - u2 * u * (Real(10) - Real(15) * u + Real(6) * u2);
    dT = -Real(120) * u2 * v * v;  // −30u²(1−u)² / 0.25
  }
}

}  // namespace detail

/// I_s / s = π ∫_0^{5} ρ³ (2π)^{−3/2} e^{−ρ²/2} T(ρ/5) dρ (the integral in units of s, r_c = 5s),
/// by composite Gauss–Legendre on the two smooth pieces [0, 3.75] and [3.75, 5] (`panels` panels
/// of `order` points each). The taper is C² only at 3.75, so the split is what makes the rule
/// converge to round-off; the defaults agree with twice the panels to ≤ 1e-15 relative.
inline double nonlocalUnitIntegral(int panels = 16, int order = 20) {
  std::vector<double> gx, gw;
  detail::gaussLegendre(order, gx, gw);
  const double pi = 3.14159265358979323846, rcs = 5.0;
  const double pref = pi * std::pow(2 * pi, -1.5);
  double I = 0;
  const double br[3] = {0.0, 0.75 * rcs, rcs};
  for (int piece = 0; piece < 2; ++piece) {
    const double a0 = br[piece], a1 = br[piece + 1], hp = (a1 - a0) / panels;
    for (int p = 0; p < panels; ++p) {
      const double lo = a0 + p * hp, mid = lo + 0.5 * hp;
      double sp = 0;
      for (int q = 0; q < order; ++q) {
        const double r = mid + 0.5 * hp * gx[q];
        double T, dT;
        detail::nonlocalTaper(r / rcs, T, dT);
        sp += gw[q] * r * r * r * std::exp(-0.5 * r * r) * T;
      }
      I += 0.5 * hp * sp;
    }
  }
  return pref * I;
}

/// The E_M widths from the reference volumes: h = (Σ V⁰/N)^{1/3}, s = κ_s h, r_c = 5 s,
/// C_s = 1/I_s (I_s = s · nonlocalUnitIntegral()).
template <class Real>
NonlocalParams<Real> nonlocalParams(const std::vector<double>& V0, double kappaS = 1.0) {
  if (V0.empty())
    throw std::invalid_argument("voro: nonlocal interface energy needs the reference volumes");
  double sum = 0;
  for (double v : V0)
    sum += v;
  NonlocalParams<Real> p;
  const double h = std::cbrt(sum / static_cast<double>(V0.size()));
  const double s = kappaS * h;
  p.h = static_cast<Real>(h);
  p.s = static_cast<Real>(s);
  p.rc = static_cast<Real>(5.0 * s);
  p.Cs = static_cast<Real>(1.0 / (s * nonlocalUnitIntegral()));
  return p;
}

/// E_M on the facets `F` (exported WITH higher moments) at positions `pos` (3N): `types` (N) index
/// `tension` (nTypes², σ_ab), `V0` (N) the reference volumes, `prm` the widths, `L` the periodic
/// box. ADDS ∂E_M/∂x into `gradX` (3N) and ∂E_M/∂w into `gradW` (N; may be empty: Voronoi) and
/// returns E_M. `nPairs` (optional) receives the number of kept (ordered) pairs; `dEdV0` (optional,
/// N) receives e_i = ∂E_M/∂V⁰_i at fixed (x, w) (see the file comment; output only — every other
/// result is unchanged by it).
template <class Real>
Real nonlocalInterfaceEnergy(const CellFacets<Real>& F, const LView<Real>& pos,
                             const Kokkos::View<int*, peclet::core::MemSpace>& types,
                             const LView<Real>& tension, int nTypes, const LView<Real>& V0,
                             const NonlocalParams<Real>& prm, const Real L[3],
                             const LView<Real>& gradX, const LView<Real>& gradW,
                             long* nPairs = nullptr, const LView<Real>& dEdV0 = LView<Real>()) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  if (!F.hasHigherMoments())
    throw std::logic_error(
        "voro: nonlocalInterfaceEnergy needs facets exported with higher moments");
  if (!prm.valid())
    throw std::logic_error("voro: nonlocalInterfaceEnergy: widths not set (nonlocalParams)");
  const int N = F.nCells;
  const Real rcand = prm.rc + Real(2) * prm.h;
  for (int a = 0; a < 3; ++a)
    if (!(rcand <= Real(0.5) * L[a]))
      throw std::runtime_error(
          "voro: nonlocal interface energy: the pair radius r_c + 2h = " + std::to_string(rcand) +
          " exceeds half the box (" + std::to_string(0.5 * L[a]) +
          "); the minimum image would not be unique (more cells needed)");
  const CellFacets<Real> G = F;
  const LView<Real> P = pos, ten = tension, v0 = V0, gx = gradX, gw = gradW, ev = dEdV0;
  const Kokkos::View<int*, Mem> T = types;
  const bool hasW = gradW.extent(0) > 0, hasE = dEdV0.extent(0) > 0;
  const int nT = nTypes;
  const Kokkos::Array<Real, 3> L3 = {L[0], L[1], L[2]};
  const std::size_t sN = static_cast<std::size_t>(N);

  // ---- 1. per-cell moments c_i, Σ_i (sym: xx xy xz yy yz zz), M_i ----
  LView<Real> cm("nl.c", 3 * sN), sg("nl.sigma", 6 * sN), mm("nl.M", 3 * sN);
  Kokkos::parallel_for(
      "nl.moments", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real V = G.volume(i), iv0 = Real(1) / v0(i);
        Real M[3], c[3];
        for (int a = 0; a < 3; ++a) {
          M[a] = V * G.cellCentroid(3 * i + a);
          c[a] = M[a] * iv0;
          mm(3 * i + a) = M[a];
          cm(3 * i + a) = c[a];
        }
        int e = 0;
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b)
            sg(6 * i + e++) = G.cellSecond(9 * i + 3 * a + b) * iv0 - c[a] * c[b];
      });

  // ---- 2. bins (cell size ≥ r_c + 2h), contents sorted by index ----
  int nb[3];
  for (int a = 0; a < 3; ++a) {
    nb[a] = static_cast<int>(std::floor(L[a] / rcand));
    nb[a] = nb[a] < 1 ? 1 : nb[a];
  }
  const int nBins = nb[0] * nb[1] * nb[2];
  const Kokkos::Array<int, 3> NB = {nb[0], nb[1], nb[2]};
  Kokkos::View<int*, Mem> bin("nl.bin", sN), cnt("nl.cnt", static_cast<std::size_t>(nBins) + 1);
  Kokkos::parallel_for(
      "nl.binOf", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        int b3[3];
        for (int a = 0; a < 3; ++a) {
          Real y = P(3 * i + a) - L3[a] * Kokkos::floor(P(3 * i + a) / L3[a]);
          int q = static_cast<int>(y / L3[a] * NB[a]);
          b3[a] = q < 0 ? 0 : (q >= NB[a] ? NB[a] - 1 : q);
        }
        const int b = b3[0] + NB[0] * (b3[1] + NB[1] * b3[2]);
        bin(i) = b;
        Kokkos::atomic_add(&cnt(b + 1), 1);
      });
  Kokkos::parallel_scan(
      "nl.binScan", nBins + 1, KOKKOS_LAMBDA(const int b, int& upd, const bool fin) {
        const int v = cnt(b);
        upd += v;
        if (fin)
          cnt(b) = upd;
      });
  Kokkos::View<int*, Mem> fill("nl.fill", static_cast<std::size_t>(nBins)), item("nl.item", sN);
  Kokkos::parallel_for(
      "nl.binFill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const int b = bin(i);
        item(cnt(b) + Kokkos::atomic_fetch_add(&fill(b), 1)) = i;
      });
  Kokkos::parallel_for(
      "nl.binSort", Kokkos::RangePolicy<Exec>(0, nBins), KOKKOS_LAMBDA(const int b) {
        for (int p = cnt(b) + 1; p < cnt(b + 1); ++p) {  // insertion sort: fixed order
          const int v = item(p);
          int q = p - 1;
          while (q >= cnt(b) && item(q) > v) {
            item(q + 1) = item(q);
            --q;
          }
          item(q + 1) = v;
        }
      });

  // ---- 3. pair gather: E_i, g_i (into gradX), G_i; then a_i, B_i ----
  LView<Real> av("nl.a", 3 * sN), bv("nl.B", 6 * sN);
  const Real s2 = prm.s * prm.s, rc = prm.rc, rc2 = rcand * rcand, Cs = prm.Cs;
  const Real norm = static_cast<Real>(std::pow(2.0 * 3.14159265358979323846, -1.5));  // (2π)^{−3/2}
  Real E = 0;
  long np = 0;
  Kokkos::parallel_reduce(
      "nl.pairs", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc, long& npAcc) {
        const int ti = T(i);
        const Real xi[3] = {P(3 * i), P(3 * i + 1), P(3 * i + 2)};
        const Real ci[3] = {cm(3 * i), cm(3 * i + 1), cm(3 * i + 2)};
        Real g[3] = {0, 0, 0}, Gs[6] = {0, 0, 0, 0, 0, 0}, Ei = 0;
        const int bi = bin(i);
        const int b3[3] = {bi % NB[0], (bi / NB[0]) % NB[1], bi / (NB[0] * NB[1])};
        int lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
          if (NB[a] >= 3) {
            lo[a] = b3[a] - 1;
            hi[a] = b3[a] + 1;
          } else {
            lo[a] = 0;
            hi[a] = NB[a] - 1;
          }
        }
        for (int oz = lo[2]; oz <= hi[2]; ++oz)
          for (int oy = lo[1]; oy <= hi[1]; ++oy)
            for (int ox = lo[0]; ox <= hi[0]; ++ox) {
              const int bx = (ox + NB[0]) % NB[0], by = (oy + NB[1]) % NB[1],
                        bz = (oz + NB[2]) % NB[2];
              const int b = bx + NB[0] * (by + NB[1] * bz);
              for (int p = cnt(b); p < cnt(b + 1); ++p) {
                const int k = item(p);
                const int tk = T(k);
                if (tk == ti)
                  continue;
                const Real sig = ten(ti * nT + tk);
                if (sig == Real(0))
                  continue;
                Real r[3], r2 = 0;
                for (int a = 0; a < 3; ++a) {
                  Real dd = xi[a] - P(3 * k + a);
                  dd -= L3[a] * Kokkos::round(dd / L3[a]);
                  r[a] = dd;
                  r2 += dd * dd;
                }
                if (!(r2 < rc2))
                  continue;
                Real z[3], zz = 0;
                for (int a = 0; a < 3; ++a) {
                  z[a] = r[a] + ci[a] - cm(3 * k + a);
                  zz += z[a] * z[a];
                }
                const Real zn = Kokkos::sqrt(zz), q = zn / rc;
                if (!(q < Real(1)))
                  continue;
                // S = s²I + Σ_i + Σ_k (sym), its inverse (adjugate / det)
                Real S[6];
                for (int e = 0; e < 6; ++e)
                  S[e] = sg(6 * i + e) + sg(6 * k + e);
                S[0] += s2;
                S[3] += s2;
                S[5] += s2;
                const Real a00 = S[3] * S[5] - S[4] * S[4], a01 = S[2] * S[4] - S[1] * S[5],
                           a02 = S[1] * S[4] - S[2] * S[3], a11 = S[0] * S[5] - S[2] * S[2],
                           a12 = S[1] * S[2] - S[0] * S[4], a22 = S[0] * S[3] - S[1] * S[1];
                const Real det = S[0] * a00 + S[1] * a01 + S[2] * a02;
                const Real id = Real(1) / det;
                const Real Si[6] = {a00 * id, a01 * id, a02 * id, a11 * id, a12 * id, a22 * id};
                const Real pz[3] = {Si[0] * z[0] + Si[1] * z[1] + Si[2] * z[2],
                                    Si[1] * z[0] + Si[3] * z[1] + Si[4] * z[2],
                                    Si[2] * z[0] + Si[4] * z[1] + Si[5] * z[2]};
                const Real quad = z[0] * pz[0] + z[1] * pz[1] + z[2] * pz[2];
                Real Tq, dTq;
                detail::nonlocalTaper(q, Tq, dTq);
                const Real e0 = sig * Cs * v0(i) * v0(k) * norm / Kokkos::sqrt(det) *
                                Kokkos::exp(Real(-0.5) * quad);
                const Real e = e0 * Tq;
                Ei += e;
                ++npAcc;
                const Real tz = zn > Real(0) ? e0 * dTq / (zn * rc) : Real(0);
                for (int a = 0; a < 3; ++a)
                  g[a] += -e * pz[a] + tz * z[a];
                const Real hE = Real(0.5) * e;
                Gs[0] += hE * (pz[0] * pz[0] - Si[0]);
                Gs[1] += hE * (pz[0] * pz[1] - Si[1]);
                Gs[2] += hE * (pz[0] * pz[2] - Si[2]);
                Gs[3] += hE * (pz[1] * pz[1] - Si[3]);
                Gs[4] += hE * (pz[1] * pz[2] - Si[4]);
                Gs[5] += hE * (pz[2] * pz[2] - Si[5]);
              }
            }
        acc += Real(0.5) * Ei;
        const Real iv0 = Real(1) / v0(i);
        // a_i = (g − 2 G c)/V⁰, B_i = G/V⁰
        const Real Gc[3] = {Gs[0] * ci[0] + Gs[1] * ci[1] + Gs[2] * ci[2],
                            Gs[1] * ci[0] + Gs[3] * ci[1] + Gs[4] * ci[2],
                            Gs[2] * ci[0] + Gs[4] * ci[1] + Gs[5] * ci[2]};
        for (int a = 0; a < 3; ++a) {
          av(3 * i + a) = (g[a] - Real(2) * Gc[a]) * iv0;
          gx(3 * i + a) += g[a];  // the explicit min-image term
        }
        for (int e = 0; e < 6; ++e)
          bv(6 * i + e) = Gs[e] * iv0;
        if (hasE) {  // e_i = [E_i − g·c + G:(c cᵀ − Σ)] / V⁰ (Σ, G symmetric: off-diagonals twice)
          const Real D[6] = {ci[0] * ci[0] - sg(6 * i),     ci[0] * ci[1] - sg(6 * i + 1),
                             ci[0] * ci[2] - sg(6 * i + 2), ci[1] * ci[1] - sg(6 * i + 3),
                             ci[1] * ci[2] - sg(6 * i + 4), ci[2] * ci[2] - sg(6 * i + 5)};
          const Real GD = Gs[0] * D[0] + Gs[3] * D[3] + Gs[5] * D[5] +
                          Real(2) * (Gs[1] * D[1] + Gs[2] * D[2] + Gs[4] * D[4]);
          ev(i) = (Ei - (g[0] * ci[0] + g[1] * ci[1] + g[2] * ci[2]) + GD) * iv0;
        }
      },
      E, np);
  if (nPairs)
    *nPairs = np;

  // ---- 4. facet gather: ∂E/∂x_i (shift + facets), ∂E/∂w_i ----
  Kokkos::parallel_for(
      "nl.facets", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real ai[3] = {av(3 * i), av(3 * i + 1), av(3 * i + 2)};
        const Real Bi[9] = {bv(6 * i),     bv(6 * i + 1), bv(6 * i + 2),
                            bv(6 * i + 1), bv(6 * i + 3), bv(6 * i + 4),
                            bv(6 * i + 2), bv(6 * i + 4), bv(6 * i + 5)};
        const Real V = G.volume(i);
        Real fx[3], fw = 0;
        // the local integrand's shift: −V_i a_i − 2 B_i M_i
        for (int a = 0; a < 3; ++a) {
          Real bm = 0;
          for (int b = 0; b < 3; ++b)
            bm += Bi[3 * a + b] * mm(3 * i + b);
          fx[a] = -V * ai[a] - Real(2) * bm;
        }
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          const Real aj[3] = {av(3 * j), av(3 * j + 1), av(3 * j + 2)};
          const Real Bj[9] = {bv(6 * j),     bv(6 * j + 1), bv(6 * j + 2),
                              bv(6 * j + 1), bv(6 * j + 3), bv(6 * j + 4),
                              bv(6 * j + 2), bv(6 * j + 4), bv(6 * j + 5)};
          Real r[3], d2 = 0, c[3];
          for (int a = 0; a < 3; ++a) {
            r[a] = G.conn(3 * f + a);
            d2 += r[a] * r[a];
            c[a] = G.centroid(3 * f + a);
          }
          const Real d = Kokkos::sqrt(d2), A = G.area(f);
          // Φ(ξ) = α0 + α·ξ + ξᵀ𝔅ξ:  α0 = a_j·r − rᵀB_j r,  α = a_i − a_j + 2 B_j r,  𝔅 = B_i − B_j
          Real Bjr[3], al[3], Bb[9], al0 = 0;
          for (int a = 0; a < 3; ++a) {
            Bjr[a] = Bj[3 * a] * r[0] + Bj[3 * a + 1] * r[1] + Bj[3 * a + 2] * r[2];
            al0 += aj[a] * r[a] - r[a] * Bjr[a];
            al[a] = ai[a] - aj[a] + Real(2) * Bjr[a];
          }
          for (int e = 0; e < 9; ++e)
            Bb[e] = Bi[e] - Bj[e];
          // about the facet centroid: Φ(c), ∇Φ(c) = α + 2𝔅c
          Real Bc[3], phic = al0, gphi[3];
          for (int a = 0; a < 3; ++a) {
            Bc[a] = Bb[3 * a] * c[0] + Bb[3 * a + 1] * c[1] + Bb[3 * a + 2] * c[2];
            phic += (al[a] + Bc[a]) * c[a];
            gphi[a] = al[a] + Real(2) * Bc[a];
          }
          Real BS = 0;
          for (int e = 0; e < 9; ++e)
            BS += Bb[e] * G.second(9 * f + e);
          const Real I0 = A * phic + BS;
          for (int a = 0; a < 3; ++a) {
            Real Sg = 0, KB = 0;
            for (int b = 0; b < 3; ++b) {
              Sg += G.second(9 * f + 3 * a + b) * gphi[b];
              for (int cc = 0; cc < 3; ++cc)
                KB += G.third(10 * f + sym3Index(a, b, cc)) * Bb[3 * b + cc];
            }
            fx[a] += (c[a] * I0 + Sg + KB) / d;
          }
          fw += I0 / (Real(2) * d);
        }
        for (int a = 0; a < 3; ++a)
          gx(3 * i + a) += fx[a];
        if (hasW)
          gw(i) += fw;
      });
  return E;
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_NONLOCAL_ENERGY_HPP
