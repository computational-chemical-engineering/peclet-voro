/**
 * @file lagrangian/operators.hpp
 * \brief The cell operators of the Lagrangian power-cell method (M7 NHV-W, DESIGN_NOTE §2.1, §2.4,
 * §10.2) as matrix-free facet loops over CellFacets (lagrangian/facets.hpp), periodic box with
 * optional fixed walls:
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
 * Walls (fixed, impermeable, no-slip; DESIGN_NOTE §2.4, §10.2). The wall facets of CellFacets
 * (outward unit normal n̂_w, area A^w, generator-relative centroid c^w) enter as
 *   C:  (C v)_i += Σ_w A^w n̂_w·(v_w − v_i), v_w = 0  — cancels the inner sum's implied wall flux,
 *       so 1ᵀC = 0 and Cᵀ1 = 0 hold with walls (C(const) = 0 only for a velocity along the walls);
 *   ζ:  R_w = ½ Σ_w μ_i A^w |P_w v_i|² / δ_w,  P_w = I − n̂_w n̂_wᵀ,  δ_w = n̂_w·(c^w − x_i) the
 *       generator–wall distance (the §2.4 wall rule, slip length 0), a per-cell diagonal block;
 *   J_x, H: no wall term (a fixed wall facet does not move with x or w: Neumann for H).
 * G^s stays the inner-facet gradient of §2.1 (the wall term is not part of it).
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
        for (int f = G.wallOffset(i); f < G.wallOffset(i + 1); ++f)
          for (int a = 0; a < 3; ++a)
            acc -= G.wallArea(f) * G.wallNormal(3 * f + a) * v(3 * i + a);
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
        for (int f = G.wallOffset(k); f < G.wallOffset(k + 1); ++f)
          for (int a = 0; a < 3; ++a)
            acc[a] -= G.wallArea(f) * G.wallNormal(3 * f + a) * l(k);
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

/// out (3N) = ζ v, the Hessian of R(v) = Σ_f μ_f V_f |sym G_f|² + R_w (so R = ½ vᵀζv and the
/// dissipation rate is vᵀζv; R_w the no-slip wall term, see the file comment). `mu` (N) is the cell
/// viscosity; μ_f is its harmonic mean (μ for like facets). Three gathers: G^s; per facet the
/// stress S_f = 2μ_f V_f sym G_f with its direct (v_j − v_i)/d adjoint and the Ḡ adjoint Γ_i = Σ_f
/// ½(S_f − (S_f n̂)⊗n̂); then G^sᵀ Γ.
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
        // R_w = ½ Σ_w μ_i A^w |P_w v_i|² / δ_w (no-slip wall facets)
        for (int f = G.wallOffset(i); f < G.wallOffset(i + 1); ++f) {
          const Real nw[3] = {G.wallNormal(3 * f), G.wallNormal(3 * f + 1),
                              G.wallNormal(3 * f + 2)};
          const Real dw = nw[0] * G.wallCentroid(3 * f) + nw[1] * G.wallCentroid(3 * f + 1) +
                          nw[2] * G.wallCentroid(3 * f + 2);
          if (!(dw > Real(0)))
            continue;
          const Real kw = mu(i) * G.wallArea(f) / dw;
          const Real vn = nw[0] * v(3 * i) + nw[1] * v(3 * i + 1) + nw[2] * v(3 * i + 2);
          for (int a = 0; a < 3; ++a)
            o[a] += kw * (v(3 * i + a) - vn * nw[a]);
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

/// The ζ-plan (M7 DESIGN_NOTE D14.5, §14.4.6 item 5): the geometric factors of applyDissipation,
/// precomputed once per geometry (and viscosity) so that each apply is three gathers with no
/// square root, division, harmonic mean or volume lookup. Per facet f of cell i (neighbour j):
///   gOwn_f = A (c_f − x_i)/(d V_i),  gNbr_f = A (c_f − x_j′)/(d V_j)  (j's factor of the same
///   facet, in i's frame), n̂_f, 1/d_f, k2_f = 2 μ_f V_f (V_f = A d/3);
/// per cell: gSum_i = Σ_f gOwn_f and the no-slip wall block W_i = Σ_w μ_i A^w/δ_w (I − n̂_w n̂_wᵀ)
/// (symmetric; xx, xy, xz, yy, yz, zz). V_i ≤ 0 gives zero factors, as in applyDissipation.
/// The apply agrees with applyDissipation to round-off, not bit for bit (the factors are
/// associated differently). Buffers grow with headroom and are reused: no allocation per build
/// in steady stepping. The plan keeps a reference to the facets it was built from.
template <class Real>
struct DissipationPlan {
  CellFacets<Real> F;
  LView<Real> gOwn, gNbr, nrm, invD, k2;  // 3, 3, 3, 1, 1 per facet
  LView<Real> gSum, wall;                 // 3, 6 per cell
  bool built() const { return F.nCells > 0; }

  void build(const CellFacets<Real>& facets, const LView<Real>& mu) {
    using Exec = peclet::core::ExecSpace;
    F = facets;
    const std::size_t nF = static_cast<std::size_t>(F.nFacets),
                      nC = static_cast<std::size_t>(F.nCells);
    grow(gOwn, 3 * nF, "zplan.gOwn");
    grow(gNbr, 3 * nF, "zplan.gNbr");
    grow(nrm, 3 * nF, "zplan.nrm");
    grow(invD, nF, "zplan.invD");
    grow(k2, nF, "zplan.k2");
    grow(gSum, 3 * nC, "zplan.gSum");
    grow(wall, 6 * nC, "zplan.wall");
    const CellFacets<Real> G = F;
    const LView<Real> M = mu, go = gOwn, gn = gNbr, nr = nrm, id = invD, kk = k2, gs = gSum,
                      wb = wall;
    Kokkos::parallel_for(
        "lagrangian.zplan", Kokkos::RangePolicy<Exec>(0, G.nCells), KOKKOS_LAMBDA(const int i) {
          const Real ivi = G.volume(i) > Real(0) ? Real(1) / G.volume(i) : Real(0);
          Real sum[3] = {0, 0, 0};
          for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
            const int j = G.nbr(f);
            Real d2 = 0;
            for (int a = 0; a < 3; ++a)
              d2 += G.conn(3 * f + a) * G.conn(3 * f + a);
            const Real d = Kokkos::sqrt(d2), s = G.area(f) / d;
            const Real ivj = G.volume(j) > Real(0) ? Real(1) / G.volume(j) : Real(0);
            for (int a = 0; a < 3; ++a) {
              const Real c = G.centroid(3 * f + a), q = G.conn(3 * f + a);
              go(3 * f + a) = s * c * ivi;
              gn(3 * f + a) = s * (c - q) * ivj;
              nr(3 * f + a) = q / d;
              sum[a] += s * c * ivi;
            }
            id(f) = Real(1) / d;
            const Real mi = M(i), mj = M(j);
            const Real muf = (mi + mj) > Real(0) ? Real(2) * mi * mj / (mi + mj) : Real(0);
            kk(f) = Real(2) * muf * G.area(f) * d / Real(3);
          }
          for (int a = 0; a < 3; ++a)
            gs(3 * i + a) = sum[a];
          Real w6[6] = {0, 0, 0, 0, 0, 0};
          for (int f = G.wallOffset(i); f < G.wallOffset(i + 1); ++f) {
            const Real nw[3] = {G.wallNormal(3 * f), G.wallNormal(3 * f + 1),
                                G.wallNormal(3 * f + 2)};
            const Real dw = nw[0] * G.wallCentroid(3 * f) + nw[1] * G.wallCentroid(3 * f + 1) +
                            nw[2] * G.wallCentroid(3 * f + 2);
            if (!(dw > Real(0)))
              continue;
            const Real kw = M(i) * G.wallArea(f) / dw;
            w6[0] += kw * (Real(1) - nw[0] * nw[0]);
            w6[1] -= kw * nw[0] * nw[1];
            w6[2] -= kw * nw[0] * nw[2];
            w6[3] += kw * (Real(1) - nw[1] * nw[1]);
            w6[4] -= kw * nw[1] * nw[2];
            w6[5] += kw * (Real(1) - nw[2] * nw[2]);
          }
          for (int e = 0; e < 6; ++e)
            wb(6 * i + e) = w6[e];
        });
  }

 private:
  static void grow(LView<Real>& v, std::size_t n, const char* label) {
    if (v.extent(0) < n)
      v = LView<Real>(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing),
                      n + n / 4 + 64);
  }
};

/// out (3N) = ζ v by the plan `P` (see DissipationPlan; the operator of applyDissipation). Three
/// gathers: G_i = Σ_f (v_j − v_i) ⊗ gOwn_f; per facet S_f = k2_f sym(Ḡ_f P_f + (v_j − v_i)/d ⊗ n̂)
/// with its direct adjoint −S_f n̂/d and Γ_i = Σ_f ½ S_f P_f, plus W_i v_i; then
/// Σ_f Γ_j gNbr_f − Γ_i gSum_i.
/// With `mass` (N) given, the last gather also applies the epilogue of the kicks' operators:
/// out = M v + a ζ v (+ `add`, 3N, when given) — M7 §14.4.5's vector update fused with the
/// operator pass (applyMassDissipation).
template <class Real>
void applyDissipation(const DissipationPlan<Real>& P, const LView<Real>& v, const LView<Real>& out,
                      DissipationScratch<Real>& scr, Real aZeta = Real(1),
                      const LView<Real>& mass = LView<Real>(),
                      const LView<Real>& add = LView<Real>()) {
  using Exec = peclet::core::ExecSpace;
  const CellFacets<Real> G = P.F;
  const LView<Real> gs = scr.gs, gam = scr.gam, go = P.gOwn, gn = P.gNbr, nr = P.nrm, id = P.invD,
                    kk = P.k2, gsum = P.gSum, wb = P.wall;
  const bool epi = mass.extent(0) > 0, withAdd = add.extent(0) > 0;
  Kokkos::parallel_for(
      "lagrangian.zplan.grad", Kokkos::RangePolicy<Exec>(0, G.nCells), KOKKOS_LAMBDA(const int i) {
        Real g[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        const Real vi[3] = {v(3 * i), v(3 * i + 1), v(3 * i + 2)};
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          const Real gf[3] = {go(3 * f), go(3 * f + 1), go(3 * f + 2)};
          for (int a = 0; a < 3; ++a) {
            const Real dv = v(3 * j + a) - vi[a];
            for (int b = 0; b < 3; ++b)
              g[3 * a + b] += dv * gf[b];
          }
        }
        for (int e = 0; e < 9; ++e)
          gs(9 * i + e) = g[e];
      });
  Kokkos::parallel_for(
      "lagrangian.zplan.stress", Kokkos::RangePolicy<Exec>(0, G.nCells),
      KOKKOS_LAMBDA(const int i) {
        Real gm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, o[3] = {0, 0, 0}, gi[9];
        const Real vi[3] = {v(3 * i), v(3 * i + 1), v(3 * i + 2)};
        for (int e = 0; e < 9; ++e)
          gi[e] = gs(9 * i + e);
        for (int f = G.facetOffset(i); f < G.facetOffset(i + 1); ++f) {
          const int j = G.nbr(f);
          const Real n[3] = {nr(3 * f), nr(3 * f + 1), nr(3 * f + 2)};
          const Real idf = id(f), k = kk(f);
          Real Gf[9];
          for (int e = 0; e < 9; ++e)
            Gf[e] = Real(0.5) * (gi[e] + gs(9 * j + e));
          Real w[3];
          for (int a = 0; a < 3; ++a)
            w[a] = (v(3 * j + a) - vi[a]) * idf -
                   (Gf[3 * a] * n[0] + Gf[3 * a + 1] * n[1] + Gf[3 * a + 2] * n[2]);
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              Gf[3 * a + b] += w[a] * n[b];
          Real S[9];
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              S[3 * a + b] = k * Real(0.5) * (Gf[3 * a + b] + Gf[3 * b + a]);
          Real Sn[3];
          for (int a = 0; a < 3; ++a)
            Sn[a] = S[3 * a] * n[0] + S[3 * a + 1] * n[1] + S[3 * a + 2] * n[2];
          for (int a = 0; a < 3; ++a) {
            o[a] -= Sn[a] * idf;
            for (int b = 0; b < 3; ++b)
              gm[3 * a + b] += Real(0.5) * (S[3 * a + b] - Sn[a] * n[b]);
          }
        }
        const Real* W = &wb(6 * i);
        o[0] += W[0] * vi[0] + W[1] * vi[1] + W[2] * vi[2];
        o[1] += W[1] * vi[0] + W[3] * vi[1] + W[4] * vi[2];
        o[2] += W[2] * vi[0] + W[4] * vi[1] + W[5] * vi[2];
        for (int e = 0; e < 9; ++e)
          gam(9 * i + e) = gm[e];
        for (int a = 0; a < 3; ++a)
          out(3 * i + a) = o[a];
      });
  Kokkos::parallel_for(
      "lagrangian.zplan.gather", Kokkos::RangePolicy<Exec>(0, G.nCells),
      KOKKOS_LAMBDA(const int k) {
        Real o[3] = {0, 0, 0};
        for (int f = G.facetOffset(k); f < G.facetOffset(k + 1); ++f) {
          const int j = G.nbr(f);
          const Real g[3] = {gn(3 * f), gn(3 * f + 1), gn(3 * f + 2)};
          for (int a = 0; a < 3; ++a)
            o[a] += gam(9 * j + 3 * a) * g[0] + gam(9 * j + 3 * a + 1) * g[1] +
                    gam(9 * j + 3 * a + 2) * g[2];
        }
        const Real s[3] = {gsum(3 * k), gsum(3 * k + 1), gsum(3 * k + 2)};
        if (!epi) {
          for (int c = 0; c < 3; ++c)
            out(3 * k + c) += o[c] - (gam(9 * k + 3 * c) * s[0] + gam(9 * k + 3 * c + 1) * s[1] +
                                      gam(9 * k + 3 * c + 2) * s[2]);
          return;
        }
        for (int c = 0; c < 3; ++c) {
          const Real z = out(3 * k + c) + o[c] -
                         (gam(9 * k + 3 * c) * s[0] + gam(9 * k + 3 * c + 1) * s[1] +
                          gam(9 * k + 3 * c + 2) * s[2]);
          Real r = mass(k) * v(3 * k + c) + aZeta * z;
          if (withAdd)
            r += add(3 * k + c);
          out(3 * k + c) = r;
        }
      });
}

/// out (3N) = M v + a ζ v (+ add) by the plan: the A = M + aζ apply of the kicks in three passes.
template <class Real>
void applyMassDissipation(const DissipationPlan<Real>& P, const LView<Real>& mass, Real a,
                          const LView<Real>& v, const LView<Real>& out,
                          DissipationScratch<Real>& scr, const LView<Real>& add = LView<Real>()) {
  applyDissipation(P, v, out, scr, a, mass, add);
}

}  // namespace peclet::voro::lagrangian

#endif  // PECLET_VORO_LAGRANGIAN_OPERATORS_HPP
