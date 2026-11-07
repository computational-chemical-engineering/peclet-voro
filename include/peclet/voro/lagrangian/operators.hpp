/**
 * @file lagrangian/operators.hpp
 * \brief The cell operators of the Lagrangian power-cell method (M7 NHV-W, DESIGN_NOTE §2.1, §2.4,
 * §10.2) as matrix-free facet loops over CellFacets (lagrangian/facets.hpp), periodic box:
 *
 *   J_x = ∂V/∂x at fixed w:   (J_x v)_i = Σ_f (A/d) [ (c_f − x_i)·v_i + (x_j′ − c_f)·v_j ]
 *   H   = ∂V/∂w:              (H w)_i  = Σ_f A/(2d) (w_i − w_j)            (TPFA Laplacian)
 *   C   (FV divergence):      (C v)_i  = Σ_f (A/d) (c_f − x_i)·(v_j − v_i)  (= V_i tr G^s_i)
 *   ζ   (face-gradient Rayleigh matrix, R_in = ½ vᵀζv = Σ_f μ_f V_f |sym G_f|²):
 *       G^s_i = (1/V_i) Σ_f (v_j − v_i) ⊗ (A/d)(c_f − x_i),  Ḡ_f = ½(G^s_i + G^s_j),
 *       G_f  = Ḡ_f + [(v_j − v_i)/d − Ḡ_f n̂] ⊗ n̂,  V_f = A d/3,  μ_f = 2μ_iμ_j/(μ_i+μ_j),
 *   and the transposes J_xᵀ, Cᵀ.
 *
 * Velocities are (N,3) row-major (3i + a); a facet's quantities are taken in its cell's frame
 * (generator-relative centroid c_f, connector d = x_j′ − x_i, n̂ = d/|d|). Every operator is a
 * GATHER over a cell's own facets — the reciprocal facet's geometry is the own facet's,
 * re-expressed in the neighbour's frame (c − x_j′ = c_f − d) — so there are no atomics and the
 * result is deterministic. Since both sides of a facet are computed separately, Cᵀ is the transpose
 * of C (and J_xᵀ of J_x) to round-off, not bit-for-bit.
 *
 * Walls: the operators are periodic-only for now (the wall terms of C and of the Rayleigh function,
 * DESIGN_NOTE §2.4/§10.2, are not implemented); the binding refuses a walled tessellation.
 */
#ifndef PECLET_VORO_LAGRANGIAN_OPERATORS_HPP
#define PECLET_VORO_LAGRANGIAN_OPERATORS_HPP

#include <Kokkos_Core.hpp>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/lagrangian/facets.hpp"

namespace peclet::voro::lagrangian {

template <class Real>
using LView = Kokkos::View<Real*, peclet::core::MemSpace>;

/// out (N) = J_x v, v (3N).
template <class Real>
void applyVolumeJacobian(const CellFacets<Real>& F, const LView<Real>& v, const LView<Real>& out) {
  const CellFacets<Real> G = F;
  Kokkos::parallel_for(
      "lagrangian.Jx", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, G.nCells),
      KOKKOS_LAMBDA(const int i) {
        Real acc = 0;
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          const Real s = G.area(f) / Kokkos::sqrt(d2);
          for (int a = 0; a < 3; ++a) {
            const Real c = G.centroid(3 * f + a);
            acc += s * (c * v(3 * i + a) + (G.conn(3 * f + a) - c) * v(3 * j + a));
          }
        }
        out(i) = acc;
      });
}

/// out (3N) = J_xᵀ p, p (N):  (J_xᵀ p)_k = Σ_f (A/d)(c_f − x_k)(p_k − p_j).
template <class Real>
void applyVolumeJacobianT(const CellFacets<Real>& F, const LView<Real>& p, const LView<Real>& out) {
  const CellFacets<Real> G = F;
  Kokkos::parallel_for(
      "lagrangian.JxT", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, G.nCells),
      KOKKOS_LAMBDA(const int k) {
        Real acc[3] = {0, 0, 0};
        for (int f = G.facetOffset(k); f < G.facetOffset(k + 1); ++f) {
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          const Real s = G.area(f) / Kokkos::sqrt(d2) * (p(k) - p(G.nbr(f)));
          for (int a = 0; a < 3; ++a)
            acc[a] += s * G.centroid(3 * f + a);
        }
        for (int a = 0; a < 3; ++a)
          out(3 * k + a) = acc[a];
      });
}

/// out (N) = H w:  (H w)_i = Σ_f A/(2d) (w_i − w_j).
template <class Real>
void applyWeightHessian(const CellFacets<Real>& F, const LView<Real>& w, const LView<Real>& out) {
  const CellFacets<Real> G = F;
  Kokkos::parallel_for(
      "lagrangian.H", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, G.nCells),
      KOKKOS_LAMBDA(const int i) {
        Real acc = 0;
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          acc += G.area(f) / (Real(2) * Kokkos::sqrt(d2)) * (w(i) - w(G.nbr(f)));
        }
        out(i) = acc;
      });
}

/// out (N) = C v, v (3N).
template <class Real>
void applyDivergence(const CellFacets<Real>& F, const LView<Real>& v, const LView<Real>& out) {
  const CellFacets<Real> G = F;
  Kokkos::parallel_for(
      "lagrangian.C", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, G.nCells),
      KOKKOS_LAMBDA(const int i) {
        Real acc = 0;
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          const Real s = G.area(f) / Kokkos::sqrt(d2);
          for (int a = 0; a < 3; ++a)
            acc += s * G.centroid(3 * f + a) * (v(3 * j + a) - v(3 * i + a));
        }
        out(i) = acc;
      });
}

/// out (3N) = Cᵀ l, l (N):  (Cᵀ l)_k = Σ_f (A/d) [ (c_f − d_f) l_j − c_f l_k ].
template <class Real>
void applyDivergenceT(const CellFacets<Real>& F, const LView<Real>& l, const LView<Real>& out) {
  const CellFacets<Real> G = F;
  Kokkos::parallel_for(
      "lagrangian.CT", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, G.nCells),
      KOKKOS_LAMBDA(const int k) {
        Real acc[3] = {0, 0, 0};
        for (int f = G.facetOffset(k); f < G.facetOffset(k + 1); ++f) {
          const int j = G.nbr(f);
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          const Real s = G.area(f) / Kokkos::sqrt(d2);
          for (int a = 0; a < 3; ++a) {
            const Real c = G.centroid(3 * f + a);
            acc[a] += s * ((c - G.conn(3 * f + a)) * l(j) - c * l(k));
          }
        }
        for (int a = 0; a < 3; ++a)
          out(3 * k + a) = acc[a];
      });
}

/// gs (9N) = the linear-exact cell gradient G^s_i (row-major, ∂v_a/∂x_b at 9i + 3a + b).
template <class Real>
void cellGradient(const CellFacets<Real>& F, const LView<Real>& v, const LView<Real>& gs) {
  const CellFacets<Real> G = F;
  Kokkos::parallel_for(
      "lagrangian.Gs", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, G.nCells),
      KOKKOS_LAMBDA(const int i) {
        Real g[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          const Real s = G.area(f) / Kokkos::sqrt(d2);
          for (int a = 0; a < 3; ++a) {
            const Real dv = v(3 * j + a) - v(3 * i + a);
            for (int b = 0; b < 3; ++b)
              g[3 * a + b] += dv * s * G.centroid(3 * f + b);
          }
        }
        const Real iv = G.volume(i) > Real(0) ? Real(1) / G.volume(i) : Real(0);
        for (int e = 0; e < 9; ++e)
          gs(9 * i + e) = g[e] * iv;
      });
}

/// Scratch of applyDissipation (allocate once per mesh, reuse across applies).
template <class Real>
struct DissipationScratch {
  LView<Real> gs, gam;             // 9N each: G^s and the adjoint accumulator Γ
  DissipationScratch() = default;  // empty (no allocation)
  explicit DissipationScratch(int N)
      : gs("lagrangian.gs", 9 * static_cast<std::size_t>(N))
      , gam("lagrangian.gam", 9 * static_cast<std::size_t>(N)) {}
};

/// out (3N) = ζ v, the Hessian of R_in(v) = Σ_f μ_f V_f |sym G_f|² (so R_in = ½ vᵀζv and the
/// dissipation rate is vᵀζv). `mu` (N) is the cell viscosity; μ_f is its harmonic mean (μ for like
/// facets). Three gathers: G^s; per facet the stress S_f = 2μ_f V_f sym G_f with its direct
/// (v_j − v_i)/d adjoint and the Ḡ adjoint Γ_i = Σ_f ½(S_f − (S_f n̂)⊗n̂); then G^sᵀ Γ.
template <class Real>
void applyDissipation(const CellFacets<Real>& F, const LView<Real>& mu, const LView<Real>& v,
                      const LView<Real>& out, DissipationScratch<Real>& scr) {
  using Exec = peclet::core::ExecSpace;
  cellGradient(F, v, scr.gs);
  const CellFacets<Real> G = F;
  const LView<Real> gs = scr.gs, gam = scr.gam;
  Kokkos::parallel_for(
      "lagrangian.zeta.stress", Kokkos::RangePolicy<Exec>(0, G.nCells), KOKKOS_LAMBDA(const int i) {
        Real gm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, o[3] = {0, 0, 0};
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          const Real d = Kokkos::sqrt(d2);
          const Real n[3] = {G.conn(3 * f) / d, G.conn(3 * f + 1) / d, G.conn(3 * f + 2) / d};
          Real Gf[9];
          for (int e = 0; e < 9; ++e)
            Gf[e] = Real(0.5) * (gs(9 * i + e) + gs(9 * j + e));
          Real w[3];
          for (int a = 0; a < 3; ++a)
            w[a] = (v(3 * j + a) - v(3 * i + a)) / d -
                   (Gf[3 * a] * n[0] + Gf[3 * a + 1] * n[1] + Gf[3 * a + 2] * n[2]);
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              Gf[3 * a + b] += w[a] * n[b];
          const Real mi = mu(i), mj = mu(j);
          const Real muf = (mi + mj) > Real(0) ? Real(2) * mi * mj / (mi + mj) : Real(0);
          const Real k2 = Real(2) * muf * G.area(f) * d / Real(3);  // 2 μ_f V_f
          Real S[9];
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              S[3 * a + b] = k2 * Real(0.5) * (Gf[3 * a + b] + Gf[3 * b + a]);
          Real Sn[3];
          for (int a = 0; a < 3; ++a)
            Sn[a] = S[3 * a] * n[0] + S[3 * a + 1] * n[1] + S[3 * a + 2] * n[2];
          for (int a = 0; a < 3; ++a) {
            o[a] -= Sn[a] / d;
            for (int b = 0; b < 3; ++b)
              gm[3 * a + b] += Real(0.5) * (S[3 * a + b] - Sn[a] * n[b]);
          }
        }
        for (int e = 0; e < 9; ++e)
          gam(9 * i + e) = gm[e];
        for (int a = 0; a < 3; ++a)
          out(3 * i + a) = o[a];
      });
  Kokkos::parallel_for(
      "lagrangian.zeta.gather", Kokkos::RangePolicy<Exec>(0, G.nCells), KOKKOS_LAMBDA(const int k) {
        Real o[3] = {0, 0, 0};
        const Real ivk = G.volume(k) > Real(0) ? Real(1) / G.volume(k) : Real(0);
        for (int f = G.facetOffset(k); f < G.facetOffset(k + 1); ++f) {
          const int j = G.nbr(f);
          Real d2 = 0;
          for (int a = 0; a < 3; ++a)
            d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
          const Real s = G.area(f) / Kokkos::sqrt(d2);
          const Real ivj = G.volume(j) > Real(0) ? Real(1) / G.volume(j) : Real(0);
          Real gk[3], gj[3];  // (A/d)(c − x_k) and, in k's frame, (A/d)(c − x_j′)
          for (int b = 0; b < 3; ++b) {
            gk[b] = s * G.centroid(3 * f + b);
            gj[b] = s * (G.centroid(3 * f + b) - G.conn(3 * f + b));
          }
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              o[a] += gam(9 * j + 3 * a + b) * gj[b] * ivj - gam(9 * k + 3 * a + b) * gk[b] * ivk;
        }
        for (int a = 0; a < 3; ++a)
          out(3 * k + a) += o[a];
      });
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_OPERATORS_HPP
