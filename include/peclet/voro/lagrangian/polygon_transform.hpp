/**
 * @file lagrangian/polygon_transform.hpp
 * \brief Exact Fourier transforms of a planar polygon (M7 DESIGN_NOTE_STATICS §17.5): the
 * zeroth and first moments of e^{iq·ξ} over a facet P,
 *
 *   Φ_P(q)  = ∫_P e^{iq·ξ} dA,        Φ1_P(q) = ∫_P ξ e^{iq·ξ} dA      (q real, ξ ∈ ℝ³),
 *
 * the building blocks of the spectral nonlocal perimeter E_Γ (pass 1: S(k) from Φ; pass 2: the κ
 * moments from Φ and Φ1). Header-only, device-callable, no state.
 *
 * Polygon. nv vertices ξ_v, flat (3*v + a), counter-clockwise about the unit normal n̂, in the
 * generator frame (ξ = y − x_i, as lagrangian/facets.hpp). PolygonFrame carries n̂, a reference
 * point c in the plane (the area centroid ξ_c), h = n̂·c, the area A and D = max_v |ξ_v − c|;
 * polygonFrame() builds it. Split q = q_n n̂ + q_t (q_n = q·n̂).
 *
 * Two exact branches, chosen on |q_t| D:
 *
 *  - edge (|q_t| D ≥ 0.25). The in-plane divergence theorem with F = −i q_t e^{iq_t·ξ}/|q_t|²
 *    (∇_t·F = e^{iq_t·ξ}) turns P into its edges e: ξ_v → ξ_{v+1}, Δ_e = ξ_{v+1} − ξ_v, midpoint
 *    m_e, ℓ_e ν_e = Δ_e × n̂ (outward):
 *      Φ    = e^{iq_n h} (−i/|q_t|²) Σ_e (q_t·Δ_e×n̂) e^{iq_t·m_e} sinc(q_t·Δ_e/2),
 *      Φ1_t = P_t Σ_e (−i (q_t·Δ_e×n̂)/|q_t|²) ℓ_e⁻¹L1_e + (i q_t/|q_t|²) Φ        (from ∇_t·(ξ_α
 * F)), L1_e = ∫_e ξ e^{iq·ξ} dl = ℓ_e e^{iq·ξ_v} [ξ_v F₀(a) + Δ_e F₁(a)],   a = q·Δ_e, F₀ = (e^{ia}
 * − 1)/(ia),   F₁ = e^{ia}/(ia) + (e^{ia} − 1)/a²   (power series for |a| < 0.5);
 *  - series (|q_t| D < 0.25). Fan triangles T = (c, ξ_v, ξ_{v+1}), 2A_T = n̂·(η_v × η_{v+1}) with
 *    η = ξ − c, β = q_t·η_v, γ = q_t·η_{v+1}, and the exact moments of a linear form over a
 *    triangle, ∫_T (q_t·η)ⁿ dA = 2A_T n!/(n+2)! h_n(0, β, γ) (h_n the complete homogeneous
 *    symmetric polynomial) and ∫_T η (q_t·η)ⁿ dA = 2A_T n!/(n+3)! Σ_m η_m h_n(0, β, γ, α_m):
 *      Φ  = e^{iq·c} Σ_{n≤12} iⁿ/(n+2)! Σ_T 2A_T h_n(β, γ),
 *      Φ1 = e^{iq·c} Σ_{n≤12} iⁿ Σ_T 2A_T [c h_n(β,γ)/(n+2)! + (η_v h_n(β,γ,β) + η_{v+1}
 *           h_n(β,γ,γ))/(n+3)!].
 *    |β|, |γ| ≤ 1/4, so the first omitted term is below 10⁻¹⁸ A.
 *
 * In both branches the normal part of Φ1 is set exactly, n̂·Φ1 = h Φ (every ξ ∈ P has n̂·ξ = h),
 * and the in-plane part is projected. The two branches agree at the switch to round-off; the edge
 * branch's sum cancels its O(1/|q_t|) terms down to O(A), so it loses ~log10(Perimeter·D/A /
 * (|q_t| D)) digits — the switch at 0.25 keeps that to about one (gate G-S.1(a):
 * tests/kokkos/test_polygon_transform.cpp).
 */
#ifndef PECLET_VORO_LAGRANGIAN_POLYGON_TRANSFORM_HPP
#define PECLET_VORO_LAGRANGIAN_POLYGON_TRANSFORM_HPP

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

namespace peclet::voro::lagrangian {

/// Switches (DESIGN_NOTE_STATICS §17.5). Both sides of each switch are exact.
inline constexpr double kPolygonSeriesSwitch = 0.25;  // series branch iff |q_t| D < this
inline constexpr int kPolygonSeriesOrder = 12;        // Σ_{n=0}^{12}
inline constexpr double kEdgeSeriesSwitch = 0.5;      // F₀, F₁ by series iff |a| < this
inline constexpr int kEdgeSeriesTerms = 16;           // Σ_{n<16}
inline constexpr double kSincSeriesSwitch = 1e-3;     // sinc by series iff |x| < this

/// The plane data of a polygon: unit normal n̂ (vertices CCW about it), reference point c in the
/// plane (area centroid), h = n̂·c, area A, D = max_v |ξ_v − c|.
template <class Real>
struct PolygonFrame {
  Real n[3];
  Real c[3];
  Real h;
  Real area;
  Real diam;
};

namespace detail {

template <class Real>
KOKKOS_INLINE_FUNCTION Real dot3(const Real* a, const Real* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// e^{ix}
template <class Real>
KOKKOS_INLINE_FUNCTION Kokkos::complex<Real> cis(Real x) {
  return Kokkos::complex<Real>(Kokkos::cos(x), Kokkos::sin(x));
}

/// sin x / x, by its series for |x| < 10⁻³.
template <class Real>
KOKKOS_INLINE_FUNCTION Real sinc(Real x) {
  if (Kokkos::fabs(x) < Real(kSincSeriesSwitch)) {
    const Real x2 = x * x;
    return Real(1) - x2 / Real(6) * (Real(1) - x2 / Real(20));
  }
  return Kokkos::sin(x) / x;
}

/// F₀(a) = ∫₀¹ e^{iat} dt and F₁(a) = ∫₀¹ t e^{iat} dt.
template <class Real>
KOKKOS_INLINE_FUNCTION void edgeF01(Real a, Kokkos::complex<Real>& F0, Kokkos::complex<Real>& F1) {
  if (Kokkos::fabs(a) < Real(kEdgeSeriesSwitch)) {
    // Σ (ia)ⁿ/n! · [1/(n+1), 1/(n+2)]; p = aⁿ/n!, iⁿ cycles 1, i, −1, −i.
    Real f0r = 0, f0i = 0, f1r = 0, f1i = 0, p = 1;
    for (int n = 0; n < kEdgeSeriesTerms; ++n) {
      if (n > 0)
        p *= a / Real(n);
      const Real s = (n & 2) ? -p : p;
      const Real t0 = s / Real(n + 1), t1 = s / Real(n + 2);
      if (n & 1) {
        f0i += t0;
        f1i += t1;
      } else {
        f0r += t0;
        f1r += t1;
      }
    }
    F0 = Kokkos::complex<Real>(f0r, f0i);
    F1 = Kokkos::complex<Real>(f1r, f1i);
    return;
  }
  const Real sh = Kokkos::sin(a / Real(2));
  const Real cm1 = Real(-2) * sh * sh;  // cos a − 1, without the cancellation
  const Real sn = Kokkos::sin(a), cs = Kokkos::cos(a);
  const Real ia = Real(1) / a;
  // (e^{ia} − 1)/(ia) = −i (cm1 + i sn)/a
  F0 = Kokkos::complex<Real>(sn * ia, -cm1 * ia);
  // e^{ia}/(ia) + (e^{ia} − 1)/a² = −i (cs + i sn)/a + (cm1 + i sn)/a²
  F1 = Kokkos::complex<Real>(sn * ia + cm1 * ia * ia, -cs * ia + sn * ia * ia);
}

/// q_n = q·n̂, q_t = q − q_n n̂, |q_t|².
template <class Real>
KOKKOS_INLINE_FUNCTION Real splitQ(const PolygonFrame<Real>& P, const Real* q, Real* qt) {
  const Real qn = dot3(q, P.n);
  for (int a = 0; a < 3; ++a)
    qt[a] = q[a] - qn * P.n[a];
  return qn;
}

/// (q_t · Δ × n̂) for the edge ξ_v → ξ_w.
template <class Real>
KOKKOS_INLINE_FUNCTION Real edgeFlux(const Real* qt, const Real* d, const Real* n) {
  const Real cx = d[1] * n[2] - d[2] * n[1];
  const Real cy = d[2] * n[0] - d[0] * n[2];
  const Real cz = d[0] * n[1] - d[1] * n[0];
  return qt[0] * cx + qt[1] * cy + qt[2] * cz;
}

/// Φ, edge branch (|q_t| > 0).
template <class Real>
KOKKOS_INLINE_FUNCTION Kokkos::complex<Real> phiEdge(const Real* xi, int nv,
                                                     const PolygonFrame<Real>& P, const Real* q) {
  Real qt[3];
  const Real qn = splitQ(P, q, qt);
  const Real qt2 = dot3(qt, qt);
  Real sr = 0, si = 0;
  for (int v = 0; v < nv; ++v) {
    const Real* a = xi + 3 * v;
    const Real* b = xi + 3 * ((v + 1 == nv) ? 0 : v + 1);
    const Real d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const Real m[3] = {Real(0.5) * (a[0] + b[0]), Real(0.5) * (a[1] + b[1]),
                       Real(0.5) * (a[2] + b[2])};
    const Real w = edgeFlux(qt, d, P.n) * sinc(Real(0.5) * dot3(qt, d));
    const Real ph = dot3(qt, m);
    sr += w * Kokkos::cos(ph);
    si += w * Kokkos::sin(ph);
  }
  // (−i/|q_t|²)(sr + i si) = (si − i sr)/|q_t|²
  return cis(qn * P.h) * Kokkos::complex<Real>(si / qt2, -sr / qt2);
}

/// Φ and Φ1, edge branch (|q_t| > 0).
template <class Real>
KOKKOS_INLINE_FUNCTION void phiPhi1Edge(const Real* xi, int nv, const PolygonFrame<Real>& P,
                                        const Real* q, Kokkos::complex<Real>& phi,
                                        Kokkos::complex<Real>* phi1) {
  phi = phiEdge(xi, nv, P, q);
  Real qt[3];
  splitQ(P, q, qt);
  const Real qt2 = dot3(qt, qt);
  Kokkos::complex<Real> S[3] = {0, 0, 0};
  for (int v = 0; v < nv; ++v) {
    const Real* a = xi + 3 * v;
    const Real* b = xi + 3 * ((v + 1 == nv) ? 0 : v + 1);
    const Real d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    Kokkos::complex<Real> F0, F1;
    edgeF01(dot3(q, d), F0, F1);
    // (−i (q_t·Δ×n̂)/|q_t|²) e^{iq·ξ_v}: the ℓ_e of L1_e is absorbed in Δ×n̂.
    const Kokkos::complex<Real> w =
        cis(dot3(q, a)) * Kokkos::complex<Real>(0, -edgeFlux(qt, d, P.n) / qt2);
    for (int c = 0; c < 3; ++c)
      S[c] += w * (a[c] * F0 + d[c] * F1);
  }
  const Kokkos::complex<Real> Sn = S[0] * P.n[0] + S[1] * P.n[1] + S[2] * P.n[2];
  const Kokkos::complex<Real> iphi = Kokkos::complex<Real>(0, 1) * phi / qt2;
  for (int c = 0; c < 3; ++c)
    phi1[c] = S[c] - Sn * P.n[c] + iphi * qt[c] + phi * (P.h * P.n[c]);
}

/// Σ_{n≤12} iⁿ cₙ accumulated as (re, im): adds iⁿ·x to (re, im).
template <class Real>
KOKKOS_INLINE_FUNCTION void addIn(int n, Real x, Real& re, Real& im) {
  const Real s = (n & 2) ? -x : x;
  if (n & 1)
    im += s;
  else
    re += s;
}

/// Φ (and Φ1 when phi1 != nullptr), series branch.
template <class Real>
KOKKOS_INLINE_FUNCTION void phiPhi1Series(const Real* xi, int nv, const PolygonFrame<Real>& P,
                                          const Real* q, Kokkos::complex<Real>& phi,
                                          Kokkos::complex<Real>* phi1) {
  Real qt[3];
  splitQ(P, q, qt);
  const Real* c = P.c;
  Real pr = 0, pi = 0;  // Σ iⁿ/(n+2)! Σ_T 2A_T h_n(β,γ)
  Real tr[3] = {0, 0, 0},
       ti[3] = {0, 0, 0};  // Σ iⁿ/(n+3)! Σ_T 2A_T (η_v h_n(β,γ,β) + η_w h_n(β,γ,γ))
  for (int v = 0; v < nv; ++v) {
    const Real* a = xi + 3 * v;
    const Real* b = xi + 3 * ((v + 1 == nv) ? 0 : v + 1);
    const Real ev[3] = {a[0] - c[0], a[1] - c[1], a[2] - c[2]};
    const Real ew[3] = {b[0] - c[0], b[1] - c[1], b[2] - c[2]};
    const Real cr[3] = {ev[1] * ew[2] - ev[2] * ew[1], ev[2] * ew[0] - ev[0] * ew[2],
                        ev[0] * ew[1] - ev[1] * ew[0]};
    const Real A2 = dot3(P.n, cr);  // 2A_T, signed about n̂
    const Real be = dot3(qt, ev), ga = dot3(qt, ew);
    Real bn = 1, h2 = 0, h3b = 0, h3g = 0;        // βⁿ, h_n(β,γ), h_n(β,γ,β), h_n(β,γ,γ)
    Real f2 = Real(0.5), f3 = Real(1) / Real(6);  // 1/(n+2)!, 1/(n+3)!
    for (int n = 0; n <= kPolygonSeriesOrder; ++n) {
      if (n > 0) {
        bn *= be;
        f2 /= Real(n + 2);
        f3 /= Real(n + 3);
      }
      h2 = ga * h2 + bn;  // h_n(β,γ) = γ h_{n−1}(β,γ) + βⁿ
      addIn(n, A2 * h2 * f2, pr, pi);
      if (phi1) {
        h3b = h2 + be * h3b;  // h_n(β,γ,x) = h_n(β,γ) + x h_{n−1}(β,γ,x)
        h3g = h2 + ga * h3g;
        const Real k = A2 * f3;
        for (int d = 0; d < 3; ++d)
          addIn(n, k * (ev[d] * h3b + ew[d] * h3g), tr[d], ti[d]);
      }
    }
  }
  const Kokkos::complex<Real> e = cis(dot3(q, c));
  phi = e * Kokkos::complex<Real>(pr, pi);
  if (!phi1)
    return;
  Kokkos::complex<Real> V[3];
  for (int d = 0; d < 3; ++d)
    V[d] = e * Kokkos::complex<Real>(tr[d], ti[d]);
  // c Φ + V, with the normal part set to h Φ and the in-plane part projected.
  const Kokkos::complex<Real> Vn = V[0] * P.n[0] + V[1] * P.n[1] + V[2] * P.n[2];
  const Real cn = dot3(c, P.n);
  for (int d = 0; d < 3; ++d)
    phi1[d] = V[d] - Vn * P.n[d] + phi * (c[d] - cn * P.n[d] + P.h * P.n[d]);
}

template <class Real>
KOKKOS_INLINE_FUNCTION bool useSeries(const PolygonFrame<Real>& P, const Real* q) {
  Real qt[3];
  splitQ(P, q, qt);
  return Kokkos::sqrt(dot3(qt, qt)) * P.diam < Real(kPolygonSeriesSwitch);
}

}  // namespace detail

/// The frame of a polygon from its vertices (flat 3*nv, CCW about the unit normal n): area and
/// centroid from the fan about ξ_0 (signed about n), h = n·c, D = max_v |ξ_v − c|. A polygon of
/// zero area takes c = ξ_0.
template <class Real>
KOKKOS_INLINE_FUNCTION PolygonFrame<Real> polygonFrame(const Real* xi, int nv, const Real* n) {
  PolygonFrame<Real> P;
  for (int a = 0; a < 3; ++a)
    P.n[a] = n[a];
  Real A2 = 0, m[3] = {0, 0, 0};
  const Real* o = xi;
  for (int v = 1; v + 1 < nv; ++v) {
    const Real* a = xi + 3 * v;
    const Real* b = xi + 3 * (v + 1);
    const Real u[3] = {a[0] - o[0], a[1] - o[1], a[2] - o[2]};
    const Real w[3] = {b[0] - o[0], b[1] - o[1], b[2] - o[2]};
    const Real t = n[0] * (u[1] * w[2] - u[2] * w[1]) + n[1] * (u[2] * w[0] - u[0] * w[2]) +
                   n[2] * (u[0] * w[1] - u[1] * w[0]);
    A2 += t;
    for (int d = 0; d < 3; ++d)
      m[d] += t * (u[d] + w[d]);
  }
  P.area = Real(0.5) * A2;
  for (int d = 0; d < 3; ++d)
    P.c[d] = (A2 != Real(0)) ? o[d] + m[d] / (Real(3) * A2) : o[d];
  P.h = detail::dot3(P.n, P.c);
  Real D2 = 0;
  for (int v = 0; v < nv; ++v) {
    const Real e[3] = {xi[3 * v] - P.c[0], xi[3 * v + 1] - P.c[1], xi[3 * v + 2] - P.c[2]};
    D2 = Kokkos::fmax(D2, detail::dot3(e, e));
  }
  P.diam = Kokkos::sqrt(D2);
  return P;
}

/// Φ_P(q) = ∫_P e^{iq·ξ} dA (exact; branch on |q_t| D).
template <class Real>
KOKKOS_INLINE_FUNCTION Kokkos::complex<Real> polygonPhi(const Real* xi, int nv,
                                                        const PolygonFrame<Real>& P,
                                                        const Real* q) {
  if (detail::useSeries(P, q)) {
    Kokkos::complex<Real> phi;
    detail::phiPhi1Series<Real>(xi, nv, P, q, phi, nullptr);
    return phi;
  }
  return detail::phiEdge(xi, nv, P, q);
}

/// Φ_P(q) and Φ1_P(q) = ∫_P ξ e^{iq·ξ} dA (exact; branch on |q_t| D).
template <class Real>
KOKKOS_INLINE_FUNCTION void polygonPhiPhi1(const Real* xi, int nv, const PolygonFrame<Real>& P,
                                           const Real* q, Kokkos::complex<Real>& phi,
                                           Kokkos::complex<Real>* phi1) {
  if (detail::useSeries(P, q))
    detail::phiPhi1Series(xi, nv, P, q, phi, phi1);
  else
    detail::phiPhi1Edge(xi, nv, P, q, phi, phi1);
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_POLYGON_TRANSFORM_HPP
