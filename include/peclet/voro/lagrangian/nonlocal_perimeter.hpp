/**
 * @file lagrangian/nonlocal_perimeter.hpp
 * \brief The set-based nonlocal capillary energy E_Γ of the Lagrangian power-cell method (M7
 * NHV-W, DESIGN_NOTE_STATICS §17.4–§17.5), opt-in (private `interface_energy = 'nonlocal_exact'`),
 * with its exact partials in positions AND weights:
 *
 *   E_Γ = σ C_Γ ∫_{Ω_a} ∫_{Ω_b} G_s(y − y′) dy dy′,   C_Γ = √(2π)/s,   s = κ_s h,
 *
 * G_s the untapered periodised Gaussian of width s (κ_s = 1, h = (Σ V⁰/N)^{1/3}), C_Γ normalised to
 * σ per area on a flat interface. Two phases a (the lower type index) and b; Γ = the unlike facets,
 * n̂_f from the a-cell toward the b-cell. Evaluated by the spectral direct sum over the half space
 * K⁺ of the box's modes with |k| ≤ k_max = √(2 ln(1/ε_G))/s (ε_G = 10⁻¹⁰): the truncated sum is
 * itself a set functional, and the partials below are its exact gradient for any cut-off.
 *
 *   Pass 1 (structure factor; parallel over k, Γ in fixed index order):
 *     S(k) = Σ_{f∈Γ} n̂_f e^{−ik·x_i} Φ_f(−k),  χ̂(k) = i k·S(k)/|k|²,  χ̂(0) = |Ω_a| = Σ_{i∈a} V⁰_i,
 *     E_Γ = σC_Γ [|Ω_a| − (|Ω_a|² + 2 Σ_{K⁺} Ĝ|χ̂|²)/|L|],   Ĝ(k) = exp(−s²|k|²/2).
 *   Pass 2 (κ moments; one team per directed facet, k strided over the team). With u = G_s*χ_a,
 *     u(y) = (1/|L|)[|Ω_a| + 2 Re Σ_{K⁺} Ĝχ̂(k) e^{ik·y}],  κ = σC_Γ[1 − 2u] (the density of δE_Γ =
 *     ∮_Γ κ δn for a growing into b), per directed unlike facet f of cell c toward c′ in c's frame
 *     (s_c = +1 on a, −1 on b, d_f = |x_{c′}′ − x_c|):
 *       I0_f = s_c σC_Γ [A_f − 2∫_f u],   I1_f = s_c σC_Γ [A_f ξ_c − 2∫_f u ξ],
 *       ∫_f u = (1/|L|)[|Ω_a| A_f + 2 Re Σ Ĝχ̂ e^{ik·x_c} Φ_f(k)]   (and ∫_f uξ with Φ1_f),
 *       ∂E/∂x_c = Σ_f I1_f / d_f,   ∂E/∂w_c = Σ_f I0_f / (2 d_f),   zero without an unlike facet.
 *   This is u_f of lagrangian/nonlocal_energy.hpp applied to an integrand fixed in space; each side
 *   is gathered from its own cell's facet (atomic-free; the two sides agree to round-off).
 *
 * Φ and Φ1 are the exact polygon transforms of lagrangian/polygon_transform.hpp. Every sum is in a
 * fixed order (Γ by index in pass 1, the mode sum of E sequential; in pass 2 team thread t of T
 * sums k = t, t + T, … in K⁺ order and the T partials are added in thread order, then each cell
 * adds its facets in index order), so the result does not depend on the thread count. T is a
 * constant per backend: 1 on a host backend (the plain K⁺-order sum), kPerimeterTeam on a device.
 *
 * Γ buffer (exportGamma): every directed unlike facet of every cell, from the cell's own geometry
 * (the resident power tessellation re-evaluated as lagrangian/facets.hpp does): the ordered
 * vertices ξ_v (generator-relative, counter-clockwise about n̂_f = r/|r|; reversed if their signed
 * area against n̂_f is negative), the polygon frame (centroid ξ_c, area A_f, h, D), the neighbour
 * and the connector r = x_{c′}′ − x_c, and the wrapped generator. Faces with fewer than three
 * vertices or A_f ≤ 10⁻¹⁴ h² are skipped (their exact contribution is O(A_f)). The a-side entries,
 * in index order, are Γ of pass 1.
 *
 * Periodic boxes only (walls throw: the wetting of the nonlocal perimeter is a separate pass), two
 * phases only. The static pressure has e = 0 (no explicit V⁰: P = −π, as for 'facet').
 */
#ifndef PECLET_VORO_LAGRANGIAN_NONLOCAL_PERIMETER_HPP
#define PECLET_VORO_LAGRANGIAN_NONLOCAL_PERIMETER_HPP

#include <chrono>
#include <cmath>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/operators.hpp"  // LView
#include "peclet/voro/lagrangian/polygon_transform.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/topology_store.hpp"

namespace peclet::voro::lagrangian {

/// The width, the mode cut-off and the normalisation of E_Γ (DESIGN_NOTE_STATICS §17.4–§17.5).
template <class Real>
struct PerimeterParams {
  Real h = 0;     ///< (Σ V⁰ / N)^{1/3}
  Real s = 0;     ///< κ_s h
  Real epsG = 0;  ///< ε_G: Ĝ(k_max) = ε_G
  Real kmax = 0;  ///< √(2 ln(1/ε_G)) / s
  Real Cg = 0;    ///< C_Γ = √(2π)/s
  bool valid() const { return s > Real(0) && kmax > Real(0) && Cg > Real(0); }
};

/// The E_Γ parameters from the reference volumes: h = (Σ V⁰/N)^{1/3} (as E_M), s = κ_s h,
/// k_max = √(2 ln(1/ε_G))/s, C_Γ = √(2π)/s.
template <class Real>
PerimeterParams<Real> perimeterParams(const std::vector<double>& V0, double kappaS = 1.0,
                                      double epsG = 1e-10) {
  if (V0.empty())
    throw std::invalid_argument(
        "voro: the nonlocal_exact interface energy needs the reference volumes");
  if (!(kappaS > 0))
    throw std::invalid_argument("voro: nonlocal width kappa_s must be > 0");
  if (!(epsG > 0 && epsG < 1))
    throw std::invalid_argument("voro: spectral tolerance eps_G must be in (0, 1)");
  double sum = 0;
  for (double v : V0)
    sum += v;
  const double pi = 3.14159265358979323846;
  const double h = std::cbrt(sum / static_cast<double>(V0.size())), s = kappaS * h;
  PerimeterParams<Real> p;
  p.h = static_cast<Real>(h);
  p.s = static_cast<Real>(s);
  p.epsG = static_cast<Real>(epsG);
  p.kmax = static_cast<Real>(std::sqrt(2.0 * std::log(1.0 / epsG)) / s);
  p.Cg = static_cast<Real>(std::sqrt(2.0 * pi) / s);
  return p;
}

/// The modes of the half space K⁺ with |k| ≤ k_max, in lexicographic (n₃, n₂, n₁) order:
/// k (3 per mode) and Ĝ(k) = exp(−s²|k|²/2).
template <class Real>
struct PerimeterModes {
  int n = 0;
  LView<Real> k, g;
};

template <class Real>
PerimeterModes<Real> perimeterModes(const Real L[3], const PerimeterParams<Real>& prm) {
  const double pi = 3.14159265358979323846;
  const double km = prm.kmax, km2 = km * km, s2 = double(prm.s) * double(prm.s);
  int M[3];
  double dk[3];
  for (int a = 0; a < 3; ++a) {
    dk[a] = 2.0 * pi / double(L[a]);
    M[a] = static_cast<int>(std::floor(km / dk[a]));
  }
  std::vector<double> kh, gh;
  for (int n3 = 0; n3 <= M[2]; ++n3)
    for (int n2 = (n3 == 0 ? 0 : -M[1]); n2 <= M[1]; ++n2)
      for (int n1 = (n3 == 0 && n2 == 0 ? 1 : -M[0]); n1 <= M[0]; ++n1) {
        const double k[3] = {dk[0] * n1, dk[1] * n2, dk[2] * n3};
        const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
        if (k2 > km2)
          continue;
        kh.insert(kh.end(), {k[0], k[1], k[2]});
        gh.push_back(std::exp(-0.5 * s2 * k2));
      }
  PerimeterModes<Real> m;
  m.n = static_cast<int>(gh.size());
  m.k = peclet::core::toDevice<Real>(std::vector<Real>(kh.begin(), kh.end()), "nlp.k");
  m.g = peclet::core::toDevice<Real>(std::vector<Real>(gh.begin(), gh.end()), "nlp.G");
  return m;
}

/// The Γ buffer (see the file comment): a CSR over the cells of their directed unlike facets.
/// Vertices are flat with a fixed stride of kMaxV vertices per facet (3·kMaxV reals).
template <class Real>
struct GammaFacets {
  using Mem = peclet::core::MemSpace;
  static constexpr int kMaxV = 28;  ///< ConvexCell::MAXFV
  int nCells = 0, nFacets = 0, nA = 0;
  Kokkos::View<int*, Mem> offset;  ///< nCells + 1
  Kokkos::View<int*, Mem> nbr;     ///< nFacets: the neighbour c′
  Kokkos::View<int*, Mem> nv;      ///< nFacets: vertex count
  LView<Real> xi;                  ///< 3·kMaxV·nFacets: ordered vertices, generator frame
  LView<Real> nrm;                 ///< 3·nFacets: n̂ = r/|r|
  LView<Real> conn;                ///< 3·nFacets: r = x_{c′}′ − x_c (min image)
  LView<Real> cen;                 ///< 3·nFacets: ξ_c (the polygon frame's area centroid)
  LView<Real> area, hgt, diam;     ///< nFacets: A_f, h = n̂·ξ_c, D
  LView<Real> xg;                  ///< 3·nFacets: the wrapped generator x_c
  Kokkos::View<int*, Mem> aIdx;    ///< nA: the a-side facets in index order (Γ of pass 1)
};

/// Timing and sizes of one evaluation (filled only when requested; the passes are fenced).
struct PerimeterStats {
  int nGamma = 0;                              ///< a-side Γ entries (unlike facets)
  int nDirected = 0;                           ///< directed unlike facets (both sides)
  int nModes = 0;                              ///< |K⁺|
  double tExport = 0, tPass1 = 0, tPass2 = 0;  ///< seconds
};

namespace detail {

/// Facet k of the loaded cell `c` (cell i) as a Γ entry, or false if it is not one: an interior
/// face toward a live cell of the other type, ≥ 3 vertices, A > aMin after orienting the vertices
/// counter-clockwise about n̂ = r/|r|. Fills the vertices (xi, stride 3·kMaxV), r, n̂ and the frame.
template <class Cell, class FL, class Real>
KOKKOS_INLINE_FUNCTION bool gammaFacet(const Cell& c, const FL& fl, int k, int i,
                                       const Kokkos::View<Real*, peclet::core::MemSpace>& P,
                                       const Kokkos::View<int*, peclet::core::MemSpace>& T,
                                       const Kokkos::View<Real*, peclet::core::MemSpace>& EV,
                                       bool hasEV, const Kokkos::Array<Real, 3>& L, Real aMin,
                                       Real* xi, int& m, int& j, Real r[3], Real n[3],
                                       PolygonFrame<Real>& Pf) {
  constexpr int MAXFV = Cell::MAXFV;
  if (!fl.isFace(k))
    return false;
  j = c.pnbr[k];
  if (j < 0 || (hasEV && !(EV(j) > Real(0))) || T(j) == T(i))
    return false;
  Real fx[MAXFV], fy[MAXFV], fz[MAXFV];
  m = fl.ordered(c, k, fx, fy, fz);
  if (m < 3)
    return false;
  Real d2 = 0;
  for (int a = 0; a < 3; ++a) {
    Real v = P(3 * j + a) - P(3 * i + a);
    const Real hL = Real(0.5) * L[a];
    v = v > hL ? v - L[a] : (v < -hL ? v + L[a] : v);
    r[a] = v;
    d2 += v * v;
  }
  if (!(d2 > Real(0)))
    return false;
  const Real dl = Kokkos::sqrt(d2);
  for (int a = 0; a < 3; ++a)
    n[a] = r[a] / dl;
  for (int q = 0; q < m; ++q) {
    xi[3 * q] = fx[q];
    xi[3 * q + 1] = fy[q];
    xi[3 * q + 2] = fz[q];
  }
  Pf = polygonFrame(xi, m, n);
  if (Pf.area < Real(0)) {  // clockwise about n̂: reverse the order
    for (int q = 0; q < m; ++q) {
      xi[3 * q] = fx[m - 1 - q];
      xi[3 * q + 1] = fy[m - 1 - q];
      xi[3 * q + 2] = fz[m - 1 - q];
    }
    Pf = polygonFrame(xi, m, n);
  }
  return Pf.area > aMin;
}

template <class Real>
KOKKOS_INLINE_FUNCTION PolygonFrame<Real> gammaFrame(const GammaFacets<Real>& G, int f) {
  PolygonFrame<Real> P;
  for (int a = 0; a < 3; ++a) {
    P.n[a] = G.nrm(3 * f + a);
    P.c[a] = G.cen(3 * f + a);
  }
  P.h = G.hgt(f);
  P.area = G.area(f);
  P.diam = G.diam(f);
  return P;
}

/// Pass 1 at mode m: χ̂(k) = i k·S(k)/|k|², S(k) = Σ_{f∈Γ} n̂_f e^{−ik·x_f} Φ_f(−k) (Γ in index
/// order).
template <class Real>
KOKKOS_INLINE_FUNCTION Kokkos::complex<Real> perimeterChi(const GammaFacets<Real>& G,
                                                          const LView<Real>& K, int m) {
  const Real k[3] = {K(3 * m), K(3 * m + 1), K(3 * m + 2)};
  const Real q[3] = {-k[0], -k[1], -k[2]};
  Kokkos::complex<Real> S[3] = {0, 0, 0};
  for (int g = 0; g < G.nA; ++g) {
    const int f = G.aIdx(g);
    const PolygonFrame<Real> Pf = gammaFrame(G, f);
    const Real* xi = G.xi.data() + 3 * GammaFacets<Real>::kMaxV * static_cast<std::size_t>(f);
    const Kokkos::complex<Real> phi = polygonPhi(xi, G.nv(f), Pf, q);
    const Real kx = k[0] * G.xg(3 * f) + k[1] * G.xg(3 * f + 1) + k[2] * G.xg(3 * f + 2);
    const Kokkos::complex<Real> t = detail::cis(-kx) * phi;
    for (int a = 0; a < 3; ++a)
      S[a] += Pf.n[a] * t;
  }
  const Kokkos::complex<Real> kS = k[0] * S[0] + k[1] * S[1] + k[2] * S[2];
  const Real k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
  return Kokkos::complex<Real>(-kS.imag() / k2, kS.real() / k2);  // i·kS/|k|²
}

/// Pass 2 for directed facet f: J0 = Re Σ_k A(k) e^{ik·x_c} Φ_f(k), J1 = Re Σ_k A(k) e^{ik·x_c}
/// Φ1_f(k), A = Ĝχ̂, over the modes m = m0, m0 + dm, … in K⁺ order (default: all of K⁺).
template <class Real>
KOKKOS_INLINE_FUNCTION void perimeterMoments(const GammaFacets<Real>& G, const LView<Real>& K,
                                             const LView<Real>& Are, const LView<Real>& Aim, int nK,
                                             int f, Real& J0, Real J1[3], int m0 = 0, int dm = 1) {
  const PolygonFrame<Real> Pf = gammaFrame(G, f);
  const Real* xi = G.xi.data() + 3 * GammaFacets<Real>::kMaxV * static_cast<std::size_t>(f);
  const int nv = G.nv(f);
  const Real x[3] = {G.xg(3 * f), G.xg(3 * f + 1), G.xg(3 * f + 2)};
  J0 = 0;
  J1[0] = J1[1] = J1[2] = 0;
  for (int m = m0; m < nK; m += dm) {
    const Real q[3] = {K(3 * m), K(3 * m + 1), K(3 * m + 2)};
    Kokkos::complex<Real> phi, phi1[3];
    polygonPhiPhi1(xi, nv, Pf, q, phi, phi1);
    const Kokkos::complex<Real> a = Kokkos::complex<Real>(Are(m), Aim(m)) *
                                    detail::cis(q[0] * x[0] + q[1] * x[1] + q[2] * x[2]);
    J0 += (a * phi).real();
    for (int d = 0; d < 3; ++d)
      J1[d] += (a * phi1[d]).real();
  }
}

/// Pass 2's team size on a device backend (threads per directed facet; a host backend uses 1).
inline constexpr int kPerimeterTeam = 128;

/// Per-facet I0, I1 (without the sign s_c) from J0, J1: σC_Γ[A − 2∫u], σC_Γ[Aξ_c − 2∫uξ].
template <class Real>
KOKKOS_INLINE_FUNCTION void perimeterFacetTerms(Real sC, Real Oa, Real iL, Real A, const Real c[3],
                                                Real J0, const Real J1[3], Real& I0, Real I1[3]) {
  const Real iu = iL * (Oa * A + Real(2) * J0);
  I0 = sC * (A - Real(2) * iu);
  for (int d = 0; d < 3; ++d) {
    const Real iux = iL * (Oa * A * c[d] + Real(2) * J1[d]);
    I1[d] = sC * (A * c[d] - Real(2) * iux);
  }
}

}  // namespace detail

/// The Γ buffer of the resident POWER tessellation (`store` at `pos`, `weight`): see the file
/// comment. `types` (N) with a = type 0, b = type 1; `engineVol` (optional) excludes buried
/// neighbours as exportCellFacets does; `h` sets the skip threshold A_f ≤ 10⁻¹⁴ h².
template <class Real, class Policy, int MAXP, int MAXT>
GammaFacets<Real> exportGamma(const TopologyStore<MAXP, MAXT>& store,
                              const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
                              const Kokkos::View<Real*, peclet::core::MemSpace>& weight, int N,
                              const Real L[3],
                              const Kokkos::View<int*, peclet::core::MemSpace>& types, Real h,
                              WallStore<Real> wall = {},
                              Kokkos::View<Real*, peclet::core::MemSpace> xRef = {},
                              Kokkos::View<Real*, peclet::core::MemSpace> engineVol = {}) {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  using Cell = ConvexCell<Real, MAXP, MAXT, false>;
  constexpr int KV = GammaFacets<Real>::kMaxV;
  static_assert(Cell::MAXFV <= KV, "Γ buffer vertex stride below the cell's face capacity");
  const TopologyStore<MAXP, MAXT> st = store;
  const Kokkos::View<Real*, Mem> P = pos, W = weight, XR = xRef, EV = engineVol;
  const Kokkos::View<int*, Mem> T = types;
  const bool hasEV = engineVol.extent(0) > 0;
  const WallStore<Real> Wl = wall;
  const Kokkos::Array<Real, 3> L3 = {L[0], L[1], L[2]};
  const Real aMin = Real(1e-14) * h * h;
  const std::size_t sN = static_cast<std::size_t>(N);

  GammaFacets<Real> G;
  G.nCells = N;
  G.offset = Kokkos::View<int*, Mem>("nlp.off", sN + 1);
  const auto off = G.offset;
  // ---- count ----
  Kokkos::parallel_for(
      "nlp.gammaCount", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Cell c;
        detail::loadCell<Policy>(st, Wl, P, W, XR, L3, i, c);
        const detail::CellFaceLists<Cell> fl(c);
        int cnt = 0;
        for (int k = 0; k < c.np; ++k) {
          Real xi[3 * Cell::MAXFV], r[3], n[3];
          int m, j;
          PolygonFrame<Real> Pf;
          if (detail::gammaFacet(c, fl, k, i, P, T, EV, hasEV, L3, aMin, xi, m, j, r, n, Pf))
            ++cnt;
        }
        off(i + 1) = cnt;
      });
  int nF = 0;
  Kokkos::parallel_scan(
      "nlp.gammaScan", N + 1,
      KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
        const int v = off(i);
        upd += v;
        if (fin)
          off(i) = upd;
      },
      nF);
  const std::size_t sF = static_cast<std::size_t>(nF);
  G.nFacets = nF;
  G.nbr = Kokkos::View<int*, Mem>("nlp.nbr", sF);
  G.nv = Kokkos::View<int*, Mem>("nlp.nv", sF);
  G.xi = LView<Real>("nlp.xi", 3 * KV * sF);
  G.nrm = LView<Real>("nlp.n", 3 * sF);
  G.conn = LView<Real>("nlp.r", 3 * sF);
  G.cen = LView<Real>("nlp.c", 3 * sF);
  G.area = LView<Real>("nlp.A", sF);
  G.hgt = LView<Real>("nlp.h", sF);
  G.diam = LView<Real>("nlp.D", sF);
  G.xg = LView<Real>("nlp.x", 3 * sF);
  const GammaFacets<Real> Gc = G;
  // ---- fill ----
  Kokkos::parallel_for(
      "nlp.gammaFill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Cell c;
        detail::loadCell<Policy>(st, Wl, P, W, XR, L3, i, c);
        const detail::CellFaceLists<Cell> fl(c);
        Real xw[3];
        for (int a = 0; a < 3; ++a) {
          const Real v = P(3 * i + a);
          xw[a] = v - L3[a] * Kokkos::floor(v / L3[a]);
        }
        int o = off(i);
        for (int k = 0; k < c.np; ++k) {
          Real xi[3 * Cell::MAXFV], r[3], n[3];
          int m, j;
          PolygonFrame<Real> Pf;
          if (!detail::gammaFacet(c, fl, k, i, P, T, EV, hasEV, L3, aMin, xi, m, j, r, n, Pf))
            continue;
          Gc.nbr(o) = j;
          Gc.nv(o) = m;
          const std::size_t b = 3 * KV * static_cast<std::size_t>(o);
          for (int q = 0; q < 3 * m; ++q)
            Gc.xi(b + q) = xi[q];
          for (int a = 0; a < 3; ++a) {
            Gc.nrm(3 * o + a) = Pf.n[a];
            Gc.conn(3 * o + a) = r[a];
            Gc.cen(3 * o + a) = Pf.c[a];
            Gc.xg(3 * o + a) = xw[a];
          }
          Gc.area(o) = Pf.area;
          Gc.hgt(o) = Pf.h;
          Gc.diam(o) = Pf.diam;
          ++o;
        }
      });
  // ---- the a-side list (owner of type 0), in index order ----
  Kokkos::View<int*, Mem> aOff("nlp.aOff", sN + 1);
  Kokkos::parallel_for(
      "nlp.aCount", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i) { aOff(i + 1) = T(i) == 0 ? off(i + 1) - off(i) : 0; });
  int nA = 0;
  Kokkos::parallel_scan(
      "nlp.aScan", N + 1,
      KOKKOS_LAMBDA(const int i, int& upd, const bool fin) {
        const int v = aOff(i);
        upd += v;
        if (fin)
          aOff(i) = upd;
      },
      nA);
  G.nA = nA;
  G.aIdx = Kokkos::View<int*, Mem>("nlp.aIdx", static_cast<std::size_t>(nA));
  const auto aIdx = G.aIdx;
  Kokkos::parallel_for(
      "nlp.aFill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        if (T(i) != 0)
          return;
        int o = aOff(i);
        for (int f = off(i); f < off(i + 1); ++f)
          aIdx(o++) = f;
      });
  return G;
}

/// E_Γ and its partials on a Γ buffer: `types` (N, two phases, a = 0), `sigma` = σ_ab, `V0` (N)
/// the reference volumes (|Ω_a| = Σ_{i∈a} V⁰_i), `prm` the width/cut-off, `L` the periodic box.
/// ADDS ∂E_Γ/∂x into `gradX` (3N) and ∂E_Γ/∂w into `gradW` (N; may be empty) on the cells with an
/// unlike facet only (the others are not touched) and returns E_Γ. `stats` (optional): sizes and
/// fenced pass timings.
template <class Real>
Real nonlocalPerimeterEnergy(const GammaFacets<Real>& G,
                             const Kokkos::View<int*, peclet::core::MemSpace>& types, Real sigma,
                             const LView<Real>& V0, const PerimeterParams<Real>& prm,
                             const Real L[3], const LView<Real>& gradX, const LView<Real>& gradW,
                             PerimeterStats* stats = nullptr) {
  using Exec = peclet::core::ExecSpace;
  using Clock = std::chrono::steady_clock;
  if (!prm.valid())
    throw std::logic_error("voro: nonlocalPerimeterEnergy: parameters not set (perimeterParams)");
  const int N = G.nCells;
  const Kokkos::View<int*, peclet::core::MemSpace> T = types;
  const LView<Real> v0 = V0, gx = gradX, gw = gradW;
  const bool hasW = gradW.extent(0) > 0;
  // |Ω_a| = Σ_{i∈a} V⁰_i, in index order
  Real Oa = 0;
  {
    LView<Real> oa("nlp.Oa", 1);
    Kokkos::parallel_for(
        "nlp.Oa", Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(const int) {
          Real s = 0;
          for (int i = 0; i < N; ++i)
            if (T(i) == 0)
              s += v0(i);
          oa(0) = s;
        });
    Oa = peclet::core::toVector(oa)[0];
  }
  const Real vol = L[0] * L[1] * L[2], iL = Real(1) / vol, sC = sigma * prm.Cg;
  const PerimeterModes<Real> M = perimeterModes(L, prm);
  const int nK = M.n;
  const std::size_t sK = static_cast<std::size_t>(nK);
  auto t0 = Clock::now();
  // ---- pass 1: A(k) = Ĝχ̂(k) and the mode sum Σ Ĝ|χ̂|² ----
  LView<Real> Are("nlp.Are", sK), Aim("nlp.Aim", sK), gc2("nlp.Gchi2", sK);
  {
    const GammaFacets<Real> Gc = G;
    const LView<Real> K = M.k, Gk = M.g;
    Kokkos::parallel_for(
        "nlp.pass1", Kokkos::RangePolicy<Exec>(0, nK), KOKKOS_LAMBDA(const int m) {
          const Kokkos::complex<Real> chi = detail::perimeterChi(Gc, K, m);
          const Real g = Gk(m);
          Are(m) = g * chi.real();
          Aim(m) = g * chi.imag();
          gc2(m) = g * (chi.real() * chi.real() + chi.imag() * chi.imag());
        });
  }
  Real E = 0;
  {
    LView<Real> e("nlp.E", 1);
    const Real oa = Oa, il = iL, sc = sC;
    Kokkos::parallel_for(
        "nlp.energy", Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(const int) {
          Real s = 0;
          for (int m = 0; m < nK; ++m)
            s += gc2(m);
          e(0) = sc * (oa - (oa * oa + Real(2) * s) * il);
        });
    E = peclet::core::toVector(e)[0];
  }
  auto t1 = Clock::now();
  // ---- pass 2: the κ moments per directed unlike facet (one team each), gathered per cell ----
  {
    using Team = Kokkos::TeamPolicy<Exec>;
    using Scratch =
        Kokkos::View<Real*, typename Exec::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    constexpr bool onHost =
        Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible;
    const int nT = onHost ? 1 : detail::kPerimeterTeam, nF = G.nFacets;
    const GammaFacets<Real> Gc = G;
    const LView<Real> K = M.k;
    const LView<Real> J("nlp.J", 4 * static_cast<std::size_t>(nF));
    Kokkos::parallel_for(
        "nlp.pass2", Team(nF, nT).set_scratch_size(0, Kokkos::PerTeam(Scratch::shmem_size(4 * nT))),
        KOKKOS_LAMBDA(const typename Team::member_type& tm) {
          const int f = tm.league_rank(), t = tm.team_rank(), n = tm.team_size();
          Real J0, J1[3];
          detail::perimeterMoments(Gc, K, Are, Aim, nK, f, J0, J1, t, n);
          const Scratch s(tm.team_scratch(0), 4 * n);
          s(4 * t) = J0;
          for (int d = 0; d < 3; ++d)
            s(4 * t + 1 + d) = J1[d];
          tm.team_barrier();
          if (t == 0)  // the team's partials in thread order (one partial on a host backend)
            for (int c = 0; c < 4; ++c) {
              Real v = s(c);
              for (int u = 1; u < n; ++u)
                v += s(4 * u + c);
              J(4 * f + c) = v;
            }
        });
    const Real oa = Oa, il = iL, sc = sC;
    Kokkos::parallel_for(
        "nlp.pass2gather", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          const int f0 = Gc.offset(i), f1 = Gc.offset(i + 1);
          if (f1 == f0)
            return;
          const Real sgn = T(i) == 0 ? Real(1) : Real(-1);
          Real fx[3] = {0, 0, 0}, fw = 0;
          for (int f = f0; f < f1; ++f) {
            Real I0, I1[3];
            const Real J0 = J(4 * f), J1[3] = {J(4 * f + 1), J(4 * f + 2), J(4 * f + 3)};
            const Real c[3] = {Gc.cen(3 * f), Gc.cen(3 * f + 1), Gc.cen(3 * f + 2)};
            detail::perimeterFacetTerms(sc, oa, il, Gc.area(f), c, J0, J1, I0, I1);
            const Real d = Kokkos::sqrt(Gc.conn(3 * f) * Gc.conn(3 * f) +
                                        Gc.conn(3 * f + 1) * Gc.conn(3 * f + 1) +
                                        Gc.conn(3 * f + 2) * Gc.conn(3 * f + 2));
            for (int a = 0; a < 3; ++a)
              fx[a] += sgn * I1[a] / d;
            fw += sgn * I0 / (Real(2) * d);
          }
          for (int a = 0; a < 3; ++a)
            gx(3 * i + a) += fx[a];
          if (hasW)
            gw(i) += fw;
        });
  }
  if (stats) {
    Kokkos::fence();
    auto t2 = Clock::now();
    stats->nGamma = G.nA;
    stats->nDirected = G.nFacets;
    stats->nModes = nK;
    stats->tPass1 = std::chrono::duration<double>(t1 - t0).count();
    stats->tPass2 = std::chrono::duration<double>(t2 - t1).count();
  }
  return E;
}

/// E_Γ on the N cells of the resident POWER tessellation `mt` (built at `pos`, `weight`): the Γ
/// export plus the two passes. `tension` (nTypes², nTypes = 2) gives σ = σ_01. Throws on walls and
/// on more than two types. ADDS into `gradX`, `gradW`; returns E_Γ.
template <class MT, class Real>
Real nonlocalPerimeter(MT& mt, int N, const LView<Real>& pos, const LView<Real>& weight,
                       const Real L[3], const Kokkos::View<int*, peclet::core::MemSpace>& types,
                       const LView<Real>& tension, int nTypes, const LView<Real>& V0,
                       const PerimeterParams<Real>& prm, const LView<Real>& gradX,
                       const LView<Real>& gradW, PerimeterStats* stats = nullptr) {
  using Policy = typename MT::PlanePolicy;
  static_assert(std::is_same_v<Policy, Power>, "nonlocalPerimeter needs a power tessellation");
  if (mt.wall.active())
    throw std::runtime_error(
        "voro: the nonlocal_exact interface energy is periodic-only (walls are not supported)");
  if (nTypes != 2)
    throw std::runtime_error(
        "voro: the nonlocal_exact interface energy takes exactly two types, got " +
        std::to_string(nTypes));
  if (static_cast<int>(V0.extent(0)) != N)
    throw std::logic_error("voro: the nonlocal_exact interface energy needs the reference volumes");
  const Real sigma = peclet::core::toVector(tension)[1];
  const auto t0 = std::chrono::steady_clock::now();
  const GammaFacets<Real> G = exportGamma<Real, Policy>(mt.store, pos, weight, N, L, types, prm.h,
                                                        mt.wall, mt.xRef, mt.vol);
  if (stats) {
    Kokkos::fence();
    stats->tExport = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return nonlocalPerimeterEnergy(G, types, sigma, V0, prm, L, gradX, gradW, stats);
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_NONLOCAL_PERIMETER_HPP
