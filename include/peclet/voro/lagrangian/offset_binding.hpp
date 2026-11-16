/**
 * @file lagrangian/offset_binding.hpp
 * \brief The offset binding U_b of NHV-W (M7 DESIGN_NOTE_BINDING §18.5, with the floor amendment of
 * Decision Q-B1, 2026-11-14): a position-level generator–cell binding added to the partition
 * potential,
 *
 *   U_b = σ Σ_i (β₀ + β χ_i) h_i^{d−3} |o_i|²,   o_i = x_i − B_i,   d = 3: h_i^{d−3} = 1,
 *
 * B_i the current centroid of the power cell C_i (o_i = −cellCentroid_i, generator-relative), σ the
 * two-phase tension σ_01, β the band strength and β₀ the global floor. Band weight over the unlike
 * generators j (periodic minimum image r_ij = x_i − x_j, q = |r_ij|/r_b,i, r_b,i = 2.5 h_i,
 * h_i = (V⁰_i)^{1/3}):
 *
 *   W(q) = (1 − q)⁴(1 + 4q) (q < 1, else 0),  ψ_i = Σ_{j: ℓ_j ≠ ℓ_i} W(q_ij),
 *   χ_i = 1 − exp(−ψ_i/ψ₀),  ψ₀ = 0.1,  ∂ψ_i/∂x_i = Σ_j −20(1 − q_ij)³ r_ij/r_b,i² = −Σ_j
 * ∂ψ_i/∂x_j.
 *
 * Partials at fixed w (the caller's π solve then takes the summed right-hand side), with
 * a_i = 2σ(β₀ + βχ_i) o_i:
 *  1. direct    ∂_{x_i}U_b += a_i;
 *  2. weight    ∂_{x_k}U_b += Σ_i e_i ∂ψ_i/∂x_k, e_i = βσ|o_i|² e^{−ψ_i/ψ₀}/ψ₀ (pairs i–k only);
 *  3. centroid  −Σ_c a_c·dB_c = −Σ_f ∫_f Ψ_f u_f dA with Ψ_f(y) = a_c·(y − B_c)/V_c −
 *               a_c′·(y − B_c′)/V_c′ and the facet normal speed
 *               u_f = [(y − x_c)·dx_c − (y − x_c′)·dx_c′ + (dw_c − dw_c′)/2]/d_f, so that, per side
 *               c of facet f (from c's own copy of f: an atomic-free gather),
 *                 ∂_{x_c}U_b −= (1/d_f) ∫_f Ψ_f (y − x_c) dA,  ∂_{w_c}U_b −= (1/(2d_f)) ∫_f Ψ_f dA.
 *               In c's frame ξ = y − x_c, Ψ_f = α·ξ + γ with α = a_c/V_c − a_c′/V_c′ and
 *               γ = a_c·o_c/V_c − a_c′·(o_c′ − D)/V_c′, D = x̃_c′ − x_c (the facet's image): the
 *               integrands are quadratic, so the facet's area A, centroid ξ_f and central second
 *               moment S (the facet export) give them exactly: ∫Ψ = AΨ(ξ_f),
 *               ∫Ψξ = AΨ(ξ_f)ξ_f + Sα. Facets with neither side in the band (χ > 0; every cell
 *               when β₀ > 0) are skipped.
 * V⁰ enters only as constants (r_b,i), so there is no ∂V term. The current volume V_c is the
 * export's.
 *
 * Skip rule: β = β₀ = 0, or no tension (σ = 0), or all labels equal → return before any arithmetic;
 * nothing is added (single-phase runs stay bit-identical). With β₀ = 0 an empty band (no ψ_i > 0)
 * skips as well.
 *
 * Device: three cell-parallel passes (ψ and the per-cell terms; then the pair and facet gathers),
 * each cell writing only its own entries; the pair search is E_M's bin grid (nonlocal_energy.hpp,
 * detail::pairBins) at radius max_i r_b,i. Periodic boxes only; two types only. Single rank.
 */
#ifndef PECLET_VORO_LAGRANGIAN_OFFSET_BINDING_HPP
#define PECLET_VORO_LAGRANGIAN_OFFSET_BINDING_HPP

#include <algorithm>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/nonlocal_energy.hpp"  // detail::pairBins
#include "peclet/voro/lagrangian/operators.hpp"        // LView

namespace peclet::voro::lagrangian {

/// The binding's strengths and per-cell radii (bindingParams). Off (β = β₀ = 0) by default.
template <class Real>
struct BindingParams {
  Real beta = 0;          ///< band strength β
  Real beta0 = 0;         ///< global floor β₀
  Real psi0 = Real(0.1);  ///< ψ₀ of χ = 1 − exp(−ψ/ψ₀)
  LView<Real> rb;         ///< N: r_b,i = 2.5 (V⁰_i)^{1/3}
  Real rbMax = 0;         ///< max_i r_b,i (the pair-search radius)
  bool on() const { return beta != Real(0) || beta0 != Real(0); }
};

/// β, β₀ ≥ 0 and the radii r_b,i = 2.5 h_i, h_i = (V⁰_i)^{1/3} (§18.5 "Band weight").
template <class Real>
BindingParams<Real> bindingParams(const std::vector<double>& V0, double beta, double beta0) {
  if (!(beta >= 0) || !(beta0 >= 0))
    throw std::invalid_argument("voro: the offset binding needs beta >= 0 and beta0 >= 0");
  BindingParams<Real> p;
  p.beta = static_cast<Real>(beta);
  p.beta0 = static_cast<Real>(beta0);
  if (!p.on())
    return p;
  if (V0.empty())
    throw std::invalid_argument("voro: the offset binding needs the reference volumes");
  std::vector<Real> rb(V0.size());
  double rmax = 0;
  for (std::size_t i = 0; i < V0.size(); ++i) {
    rb[i] = static_cast<Real>(2.5 * std::cbrt(V0[i]));
    rmax = std::max(rmax, static_cast<double>(rb[i]));
  }
  p.rb = peclet::core::toDevice<Real>(rb, "bind.rb");
  p.rbMax = static_cast<Real>(rmax);
  return p;
}

/// The binding's per-call state (persistent across calls when held by the caller): its own
/// partials (∂_xU_b, ∂_wU_b, already added into the caller's gradient), χ, and the diagnostics of
/// the last call. `active` is false when the skip rule held (then nothing else is valid).
template <class Real>
struct BindingScratch {
  LView<Real> gx, gw;  ///< 3N, N: ∂_xU_b, ∂_wU_b at fixed w
  LView<Real> chi;     ///< N: χ_i
  LView<Real> a, e;    ///< 3N: a_i; N: e_i (dU_b/dψ_i)
  bool active = false;
  double energy = 0;              ///< U_b
  int nBand = 0;                  ///< cells with χ_i > 0
  double offMax = 0, offRms = 0;  ///< max and rms of |o_i|/h_i over the band (χ_i > 0)
};

namespace detail {
template <class Real>
KOKKOS_INLINE_FUNCTION Real minImage(Real d, Real L) {
  return d - L * Kokkos::round(d / L);
}
}  // namespace detail

/// U_b on the facets `F` (exported at `pos` and the current weights, matched; the full export:
/// centroid, second moment, cell volume and centroid) with `types` (N, two types) and `tension`
/// (2×2, σ = σ_01). ADDS ∂_xU_b into `gradX` (3N) and ∂_wU_b into `gradW` (N); returns U_b
/// (0 and nothing added under the skip rule). `walls`: the tessellation has walls (refused).
/// `scr` (optional): persistent scratch; it receives the binding's own partials and diagnostics.
template <class Real>
Real offsetBinding(const CellFacets<Real>& F, const LView<Real>& pos,
                   const Kokkos::View<int*, peclet::core::MemSpace>& types,
                   const LView<Real>& tension, int nTypes, const BindingParams<Real>& prm,
                   const Real L[3], const LView<Real>& gradX, const LView<Real>& gradW, bool walls,
                   BindingScratch<Real>* scr = nullptr) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  BindingScratch<Real> local;
  BindingScratch<Real>& S = scr ? *scr : local;
  S.active = false;
  S.energy = 0;
  S.nBand = 0;
  S.offMax = S.offRms = 0;
  // ---- the skip rule: β = β₀ = 0, no tension, or all labels equal ----
  if (!prm.on() || tension.extent(0) == 0)
    return Real(0);
  const int N = F.nCells;
  const Kokkos::View<int*, Mem> T = types;
  {
    int tmin = 0, tmax = 0;
    Kokkos::parallel_reduce(
        "bind.tmin", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i, int& acc) { acc = T(i) < acc ? T(i) : acc; },
        Kokkos::Min<int>(tmin));
    Kokkos::parallel_reduce(
        "bind.tmax", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i, int& acc) { acc = T(i) > acc ? T(i) : acc; },
        Kokkos::Max<int>(tmax));
    if (tmin == tmax)
      return Real(0);
  }
  if (nTypes != 2)
    throw std::runtime_error("voro: the offset binding takes exactly two types, got " +
                             std::to_string(nTypes));
  const Real sigma = peclet::core::toVector(tension)[1];
  if (sigma == Real(0))
    return Real(0);
  if (walls)
    throw std::runtime_error("voro: the offset binding is periodic-only (walls are not supported)");
  if (static_cast<int>(prm.rb.extent(0)) != N)
    throw std::logic_error("voro: the offset binding's radii do not match the cells");
  for (int a = 0; a < 3; ++a)
    if (!(prm.rbMax <= Real(0.5) * L[a]))
      throw std::runtime_error("voro: the offset binding's band radius " +
                               std::to_string(prm.rbMax) + " exceeds half the box (" +
                               std::to_string(0.5 * L[a]) + "); more cells needed");

  const std::size_t sN = static_cast<std::size_t>(N);
  auto grow = [](LView<Real>& v, std::size_t n, const char* label) {
    if (v.extent(0) != n)
      v = LView<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), n);
  };
  grow(S.gx, 3 * sN, "bind.gx");
  grow(S.gw, sN, "bind.gw");
  grow(S.chi, sN, "bind.chi");
  grow(S.a, 3 * sN, "bind.a");
  grow(S.e, sN, "bind.e");

  const detail::PairBins bins = detail::pairBins(pos, N, L, prm.rbMax);
  const Kokkos::Array<int, 3> NB = bins.nb;
  const Kokkos::View<int*, Mem> bin = bins.bin, cnt = bins.cnt, item = bins.item;
  const Kokkos::Array<Real, 3> L3 = {L[0], L[1], L[2]};
  const CellFacets<Real> G = F;
  const LView<Real> P = pos, rb = prm.rb, chi = S.chi, av = S.a, ev = S.e, bx = S.gx, bw = S.gw;
  const Real beta = prm.beta, beta0 = prm.beta0, ipsi0 = Real(1) / prm.psi0, rbMax = prm.rbMax;

  // ---- 1. per cell: ψ_i (unlike pairs within r_b,i), χ_i, a_i, e_i; U_b and the band stats ----
  Real Ub = 0, off2Sum = 0, off2Max = 0;
  int nBand = 0;
  Kokkos::parallel_reduce(
      "bind.psi", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& uAcc, Real& sAcc, Real& mAcc, int& nAcc) {
        const int ti = T(i);
        const Real rbi = rb(i), irb = Real(1) / rbi;
        Real psi = 0;
        int lo[3], hi[3];
        detail::binNeighbourRange(NB, bin(i), lo, hi);
        for (int oz = lo[2]; oz <= hi[2]; ++oz)
          for (int oy = lo[1]; oy <= hi[1]; ++oy)
            for (int ox = lo[0]; ox <= hi[0]; ++ox) {
              const int b = (ox + NB[0]) % NB[0] +
                            NB[0] * ((oy + NB[1]) % NB[1] + NB[1] * ((oz + NB[2]) % NB[2]));
              for (int p = cnt(b); p < cnt(b + 1); ++p) {
                const int j = item(p);
                if (T(j) == ti)
                  continue;
                Real r2 = 0;
                for (int a = 0; a < 3; ++a) {
                  const Real d = detail::minImage(P(3 * i + a) - P(3 * j + a), L3[a]);
                  r2 += d * d;
                }
                const Real q = Kokkos::sqrt(r2) * irb;
                if (q < Real(1)) {
                  const Real u = Real(1) - q, u2 = u * u;
                  psi += u2 * u2 * (Real(1) + Real(4) * q);
                }
              }
            }
        const Real ex = Kokkos::exp(-psi * ipsi0);
        const Real c = Real(1) - ex;
        chi(i) = c;
        Real o[3], o2 = 0;
        for (int a = 0; a < 3; ++a) {
          o[a] = -G.cellCentroid(3 * i + a);  // o_i = x_i − B_i
          o2 += o[a] * o[a];
        }
        const Real coef = sigma * (beta0 + beta * c);  // σ(β₀ + βχ_i) h_i^{d−3}, d = 3
        uAcc += coef * o2;
        for (int a = 0; a < 3; ++a)
          av(3 * i + a) = Real(2) * coef * o[a];
        ev(i) = beta * sigma * o2 * ex * ipsi0;
        if (c > Real(0)) {
          const Real hi2 = Real(6.25) * irb * irb;  // 1/h_i², h_i = r_b,i/2.5
          const Real s2 = o2 * hi2;
          sAcc += s2;
          mAcc = s2 > mAcc ? s2 : mAcc;
          ++nAcc;
        }
      },
      Ub, off2Sum, Kokkos::Max<Real>(off2Max), nBand);
  if (beta0 == Real(0) && nBand == 0)
    return Real(0);  // β₀ = 0 and an empty band: nothing (§18.5 skip rule)

  // ---- 2. per cell: direct, band-weight (pairs) and centroid (own facets) terms ----
  const bool hasFloor = beta0 > Real(0), weightTerm = beta > Real(0);
  Kokkos::parallel_for(
      "bind.grad", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int k) {
        Real g[3] = {av(3 * k), av(3 * k + 1), av(3 * k + 2)};  // 1. direct
        Real gwk = 0;
        // 2. band weight: Σ_j [e_k ∂ψ_k/∂x_k + e_j ∂ψ_j/∂x_k] over unlike j
        if (weightTerm) {
          const int tk = T(k);
          const Real rbk = rb(k), ek = ev(k), irb2k = Real(1) / (rbk * rbk);
          int lo[3], hi[3];
          detail::binNeighbourRange(NB, bin(k), lo, hi);
          for (int oz = lo[2]; oz <= hi[2]; ++oz)
            for (int oy = lo[1]; oy <= hi[1]; ++oy)
              for (int ox = lo[0]; ox <= hi[0]; ++ox) {
                const int b = (ox + NB[0]) % NB[0] +
                              NB[0] * ((oy + NB[1]) % NB[1] + NB[1] * ((oz + NB[2]) % NB[2]));
                for (int p = cnt(b); p < cnt(b + 1); ++p) {
                  const int j = item(p);
                  if (T(j) == tk)
                    continue;
                  Real r[3], r2 = 0;
                  for (int a = 0; a < 3; ++a) {
                    r[a] = detail::minImage(P(3 * k + a) - P(3 * j + a), L3[a]);
                    r2 += r[a] * r[a];
                  }
                  const Real dist = Kokkos::sqrt(r2);
                  if (!(dist < rbMax))
                    continue;
                  Real f = 0;
                  const Real qk = dist / rbk;
                  if (qk < Real(1)) {
                    const Real u = Real(1) - qk;
                    f -= Real(20) * ek * u * u * u * irb2k;
                  }
                  const Real rbj = rb(j), qj = dist / rbj;
                  if (qj < Real(1)) {
                    const Real u = Real(1) - qj;
                    f -= Real(20) * ev(j) * u * u * u / (rbj * rbj);
                  }
                  for (int a = 0; a < 3; ++a)
                    g[a] += f * r[a];
                }
              }
        }
        // 3. centroid: k's own facets with k or its neighbour in the band
        const bool inK = hasFloor || chi(k) > Real(0);
        const Real Vk = G.volume(k);
        Real ok[3];
        for (int a = 0; a < 3; ++a)
          ok[a] = -G.cellCentroid(3 * k + a);
        for (int f = G.facetOffset(k); f < G.facetOffset(k + 1); ++f) {
          const int c = G.nbr(f);
          if (!inK && !(chi(c) > Real(0)))
            continue;
          const Real Vc = G.volume(c), A = G.area(f);
          Real D[3], d2 = 0, al[3], gam = 0, xf[3];
          for (int a = 0; a < 3; ++a) {
            D[a] = G.conn(3 * f + a);
            d2 += D[a] * D[a];
            xf[a] = G.centroid(3 * f + a);
            const Real ak = av(3 * k + a) / Vk, ac = av(3 * c + a) / Vc;
            al[a] = ak - ac;
            gam += ak * ok[a] + ac * (G.cellCentroid(3 * c + a) + D[a]);  // −a_c′·(o_c′ − D)/V_c′
          }
          Real psiC = gam;  // Ψ at the facet centroid
          for (int a = 0; a < 3; ++a)
            psiC += al[a] * xf[a];
          const Real id = Real(1) / Kokkos::sqrt(d2);
          for (int a = 0; a < 3; ++a) {
            Real s = A * psiC * xf[a];
            for (int b = 0; b < 3; ++b)
              s += G.second(9 * f + 3 * a + b) * al[b];
            g[a] -= s * id;
          }
          gwk -= Real(0.5) * A * psiC * id;
        }
        for (int a = 0; a < 3; ++a)
          bx(3 * k + a) = g[a];
        bw(k) = gwk;
      });
  {
    const LView<Real> gX = gradX, gW = gradW;
    const bool hasW = gradW.extent(0) > 0;
    Kokkos::parallel_for(
        "bind.add", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int k) {
          for (int a = 0; a < 3; ++a)
            gX(3 * k + a) += bx(3 * k + a);
          if (hasW)
            gW(k) += bw(k);
        });
  }
  S.active = true;
  S.energy = static_cast<double>(Ub);
  S.nBand = nBand;
  S.offMax = std::sqrt(static_cast<double>(off2Max));
  S.offRms = nBand > 0 ? std::sqrt(static_cast<double>(off2Sum) / nBand) : 0.0;
  return Ub;
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_OFFSET_BINDING_HPP
