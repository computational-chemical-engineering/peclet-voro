/**
 * @file ale/operators.hpp
 * \brief Moving-grid stage 1, WO-S1.1 (docs/ale_design.md §A.3, §2.2, §2.3, §11 A7, A9): the bulk
 * operators of the moving power-cell solver as device kernels on a frame's face mesh.
 *
 * The specification of every kernel is the host reference `tests/python/ale_reference.py`; gate
 * G1.1 (`tools/ale_gates/G1_1.sh`) checks each one against it. Convention (§A.3, binding):
 *  - face f = (A, B), A the lower gid, n_f = conn_f/d_f the unit vector A → B,
 *    conn_f = x_B − x_A (minimum image); h_Af = (d² + w_A − w_B)/(2d), h_Bf = d − h_Af (signed);
 *    `faceCentroid` = c_f − x_A, and seen from B, c_f − x_B = (c_f − x_A) − conn_f (positions are
 *    never differenced across the box); q_f = c_f − x_A − h_Af n_f;
 *  - D is OUTFLOW: (DΦ)_i = Σ_f s_if Φ_f, s_if = +1 if i = A, −1 if i = B (no division by V);
 *    (Δ_n p)_f = p_B − p_A = −(Dᵀp)_f;
 *  - cell gradient G_i = (1/V_i) Σ_{f∋i} (A_f/d_f)(U_o − U_i) ⊗ (c_f − x_i), component (c,k) =
 *    ∂_kU_c stored at 9i + 3c + k;
 *  - flux map (I_F U)_f = A_f [n_f·(h_Bf U_A + h_Af U_B)/d_f + n_f·Ḡ_f q_f], Ḡ_f = ½(G_A + G_B);
 *  - C′ = D I_F; the pressure force on the cells is F_p = −I_FᵀΔ_n p = +C′ᵀp;
 *  - L_ρ = D diag(a_f) Dᵀ, a_f = A_f/(ρ_f d_f), 1/ρ_f = (h_Bf/ρ_A + h_Af/ρ_B)/d_f;
 *  - swept flux Φᵐ_f = (A_f/d_f)[(c_f − x_A)·ẋ_A − (c_f − x_B)·ẋ_B + (ẇ_A − ẇ_B)/2];
 *  - friction ½UᵀζU = Σ_f μ_f V_f |sym G_f|², G_f = Ḡ_f + [(U_B − U_A)/d_f − Ḡ_f n_f] ⊗ n_f,
 *    V_f = A_f d_f/3; ζU is the adjoint of U → G_f applied to σ_f = 2μ_f V_f sym G_f;
 *  - body force (§11 A7): F = ρ_iV_iḡ (cell values, `bodyForceCell`) + I_Fᵀγ, γ_f = ρ_A[ψ(x_A) −
 *    ψ(c_f)] + ρ_B[ψ(c_f) − ψ(x_B)] (`bodyForceFace`, or `bodyForceFaceUniform` for ψ = −g·x
 *    through differences). The sum is the caller's (no kernel for it).
 *
 * Kernel discipline (§2.2): every kernel is either K1, a RangePolicy over faces writing per-face
 * arrays without accumulation, or K2, a RangePolicy over the owned cells [0, nCells) looping the
 * cell's face list in its stored (ascending face) order and accumulating in registers. No atomic
 * floating-point accumulation and no scatter from faces to cells: the result is independent of
 * the thread schedule. Cell fields have length `nCombined` (×3 flat 3i + c for vectors, ×9 flat
 * 9i + 3c + k for gradients and moments); face fields have length `nFaces`. Scratch views are
 * passed in by the caller.
 *
 * Distributed use (WO-S1.4): a cell loop reads the other cell of a face, which may be a ghost
 * [nCells, nCombined). Inputs are exchanged by the caller; a derived cell field that the next
 * kernel reads at ghosts is handed to `exchange(field, components)` (the cell gradient, 9
 * components, after `cellGradient`; the moments T/V, 9 components, between the two passes of the
 * transposes). On one rank `exchange` is empty and not called.
 *
 * Only a periodic box is handled (every face interior); walls come with the wall work order.
 */
#ifndef PECLET_VORO_ALE_OPERATORS_HPP
#define PECLET_VORO_ALE_OPERATORS_HPP

#include <functional>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/fv/operators.hpp"

namespace peclet::voro::ale {

template <class Real>
using DV = fv::DV<Real>;

namespace detail {

template <class Real>
void requireInteriorOnly(const fv::FaceMesh<Real>& m, const char* who) {
  if (m.nInterior != m.nFaces)
    throw std::invalid_argument(std::string("ale::") + who +
                                ": boundary faces are not supported (periodic box only)");
}

/// Pass 2 of the transposes I_Fᵀ and ζ (the adjoint of the cell gradient applied to the moments
/// M = T/V): out_i ← sign · (out_i + Σ_{f∋i} (A_f/d_f)[−M_i (c_f − x_i) + M_o (c_f − x_o)]).
template <class Real>
void gradientAdjointPass(const fv::FaceMesh<Real>& m, const DV<Real>& M, const DV<Real>& out,
                         Real sign) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.gradAdjoint", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const bool isA = m.cellFaceSign(q) > 0;
          const int o = isA ? m.faceCellB(f) : m.faceCellA(f);
          const Real w = m.faceArea(f) / m.faceDist(f);
          Real ri[3], ro[3];
          for (int k = 0; k < 3; ++k) {
            const Real cA = m.centroid(f, k), cB = cA - m.conn(f, k);
            ri[k] = isA ? cA : cB;
            ro[k] = isA ? cB : cA;
          }
          for (int c = 0; c < 3; ++c) {
            Real mi = 0, mo = 0;
            for (int k = 0; k < 3; ++k) {
              mi += M(9 * i + 3 * c + k) * ri[k];
              mo += M(9 * o + 3 * c + k) * ro[k];
            }
            acc[c] += w * (-mi + mo);
          }
        }
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = sign * (out(3 * i + c) + acc[c]);
      });
}

/// sign · I_Fᵀφ in two cell gathers (§2.3; the form of `IFT_twopass` in the reference).
template <class Real>
void fluxMapTransposeSigned(const fv::FaceMesh<Real>& m, const DV<Real>& phi, const DV<Real>& out,
                            const DV<Real>& scratch9,
                            const std::function<void(const DV<Real>&, int)>& exchange, Real sign) {
  using Exec = peclet::core::ExecSpace;
  // Pass 1: the plain part Σ_f (A_f h_o/d_f) φ_f n_f (h_o of the other cell) and the moment
  // M_i = T_i/V_i, T_i = ½ Σ_{f∋i} φ_f A_f n_f ⊗ q_f.
  Kokkos::parallel_for(
      "ale.IFt1", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real v[3] = {0, 0, 0}, T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const Real ha = m.faceHa(f), hb = m.faceHb(f), area = m.faceArea(f);
          const Real hOther = m.cellFaceSign(q) > 0 ? hb : ha;
          const Real s = area * hOther / m.faceDist(f) * phi(f);
          const Real t = Real(0.5) * phi(f) * area;
          Real n[3], qf[3];
          for (int k = 0; k < 3; ++k) {
            n[k] = m.normal(f, k);
            qf[k] = m.centroid(f, k) - ha * n[k];
          }
          for (int c = 0; c < 3; ++c) {
            v[c] += s * n[c];
            for (int k = 0; k < 3; ++k)
              T[3 * c + k] += t * (n[c] * qf[k]);
          }
        }
        const Real vol = m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = v[c];
        for (int j = 0; j < 9; ++j)
          scratch9(9 * i + j) = T[j] / vol;
      });
  if (exchange)
    exchange(scratch9, 9);
  gradientAdjointPass(m, scratch9, out, sign);
}

}  // namespace detail

/// (DΦ)_i = Σ_{f∋i} s_if Φ_f, the outflow sum (no division by V_i). K2.
template <class Real>
void divergence(const fv::FaceMesh<Real>& m, const DV<Real>& phi, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "divergence");
  Kokkos::parallel_for(
      "ale.div", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc = 0;
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q)
          acc += m.cellFaceSign(q) * phi(m.cellFace(q));
        out(i) = acc;
      });
}

/// (Δ_n p)_f = p_B − p_A. K1.
template <class Real>
void faceDifference(const fv::FaceMesh<Real>& m, const DV<Real>& p, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "faceDifference");
  Kokkos::parallel_for(
      "ale.dn", Kokkos::RangePolicy<Exec>(0, m.nFaces),
      KOKKOS_LAMBDA(const int f) { out(f) = p(m.faceCellB(f)) - p(m.faceCellA(f)); });
}

/// G_i = (1/V_i) Σ_{f∋i} (A_f/d_f)(U_o − U_i) ⊗ (c_f − x_i), G (9·nCombined) at 9i + 3c + k =
/// ∂_kU_c. K2; then `exchange(G, 9)` (distributed).
template <class Real>
void cellGradient(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& G,
                  const std::function<void(const DV<Real>&, int)>& exchange = {}) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "cellGradient");
  Kokkos::parallel_for(
      "ale.grad", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real g[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const bool isA = m.cellFaceSign(q) > 0;
          const int o = isA ? m.faceCellB(f) : m.faceCellA(f);
          const Real w = m.faceArea(f) / m.faceDist(f);
          Real r[3];
          for (int k = 0; k < 3; ++k)
            r[k] = isA ? m.centroid(f, k) : m.centroid(f, k) - m.conn(f, k);
          for (int c = 0; c < 3; ++c) {
            const Real du = w * (U(3 * o + c) - U(3 * i + c));
            for (int k = 0; k < 3; ++k)
              g[3 * c + k] += du * r[k];
          }
        }
        const Real vol = m.cellVolume(i);
        for (int j = 0; j < 9; ++j)
          G(9 * i + j) = g[j] / vol;
      });
  if (exchange)
    exchange(G, 9);
}

/// (I_F U)_f = A_f [n_f·(h_Bf U_A + h_Af U_B)/d_f + n_f·Ḡ_f q_f], with G = cellGradient(U)
/// (valid at both cells of every face). K1.
template <class Real>
void fluxMap(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& G,
             const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "fluxMap");
  Kokkos::parallel_for(
      "ale.IF", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        const Real ha = m.faceHa(f), hb = m.faceHb(f);
        Real n[3], qf[3];
        for (int k = 0; k < 3; ++k) {
          n[k] = m.normal(f, k);
          qf[k] = m.centroid(f, k) - ha * n[k];
        }
        Real nUa = 0, nUb = 0, corr = 0;
        for (int c = 0; c < 3; ++c) {
          nUa += n[c] * U(3 * a + c);
          nUb += n[c] * U(3 * b + c);
          for (int k = 0; k < 3; ++k)
            corr += n[c] * (Real(0.5) * (G(9 * a + 3 * c + k) + G(9 * b + 3 * c + k))) * qf[k];
        }
        out(f) = m.faceArea(f) * ((hb * nUa + ha * nUb) / m.faceDist(f) + corr);
      });
}

/// I_Fᵀφ (3·nCombined), the adjoint of `fluxMap` in the Euclidean inner products: two cell
/// gathers (K2, K2) with the moments M = T/V in `scratch9` (9·nCombined) exchanged between them.
template <class Real>
void fluxMapTranspose(const fv::FaceMesh<Real>& m, const DV<Real>& phi, const DV<Real>& out,
                      const DV<Real>& scratch9,
                      const std::function<void(const DV<Real>&, int)>& exchange = {}) {
  detail::requireInteriorOnly(m, "fluxMapTranspose");
  detail::fluxMapTransposeSigned(m, phi, out, scratch9, exchange, Real(1));
}

/// C′U = D I_F U, with G = cellGradient(U); `scratchF` (nFaces) receives I_F U. K1, K2.
template <class Real>
void constraint(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& G,
                const DV<Real>& scratchF, const DV<Real>& out) {
  ale::fluxMap(m, U, G, scratchF);
  ale::divergence(m, scratchF, out);
}

/// The pressure force on the cells, +C′ᵀp = −I_FᵀΔ_n p (3·nCombined). `scratchF` (nFaces)
/// receives Δ_n p, `scratch9` (9·nCombined) the moments. K1, K2, K2.
template <class Real>
void pressureForce(const fv::FaceMesh<Real>& m, const DV<Real>& p, const DV<Real>& out,
                   const DV<Real>& scratchF, const DV<Real>& scratch9,
                   const std::function<void(const DV<Real>&, int)>& exchange = {}) {
  ale::faceDifference(m, p, scratchF);
  detail::fluxMapTransposeSigned(m, scratchF, out, scratch9, exchange, Real(-1));
}

/// L_ρ = D diag(a_f) Dᵀ as a matrix: row i lists the faces of cell i in ascending face order, one
/// off-diagonal entry −a_f per face with the column of the other cell (two faces between the same
/// pair of cells give two entries), `diag(i)` = Σ_{f∋i} a_f. `start` aliases the mesh's
/// `cellFaceOffset`.
template <class Real>
struct LaplacianMatrix {
  using Mem = peclet::core::MemSpace;
  int nRows = 0;
  Kokkos::View<int*, Mem> start;  ///< nRows + 1
  Kokkos::View<int*, Mem> col;    ///< start(nRows): the other cell of the face
  Kokkos::View<Real*, Mem> coef;  ///< start(nRows): −a_f
  Kokkos::View<Real*, Mem> diag;  ///< nRows: Σ a_f
};

/// Fill `L` with L_ρ of the mesh, a_f = A_f/(ρ_f d_f), 1/ρ_f = (h_Bf/ρ_A + h_Af/ρ_B)/d_f; `rho`
/// per combined cell. The views of `L` are (re)allocated only when their size does not match the
/// mesh. K2.
template <class Real>
void assembleLaplacian(const fv::FaceMesh<Real>& m, const DV<Real>& rho, LaplacianMatrix<Real>& L) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "assembleLaplacian");
  int nnz = 0;
  Kokkos::deep_copy(nnz, Kokkos::subview(m.cellFaceOffset, m.nCells));
  L.nRows = m.nCells;
  L.start = m.cellFaceOffset;
  if ((int)L.col.extent(0) != nnz)
    Kokkos::realloc(Kokkos::WithoutInitializing, L.col, nnz);
  if ((int)L.coef.extent(0) != nnz)
    Kokkos::realloc(Kokkos::WithoutInitializing, L.coef, nnz);
  if ((int)L.diag.extent(0) != m.nCells)
    Kokkos::realloc(Kokkos::WithoutInitializing, L.diag, m.nCells);
  auto col = L.col;
  auto coef = L.coef;
  auto diag = L.diag;
  Kokkos::parallel_for(
      "ale.Lrho.assemble", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real dsum = 0;
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const int a = m.faceCellA(f), b = m.faceCellB(f);
          const Real d = m.faceDist(f);
          const Real invRho = (m.faceHb(f) / rho(a) + m.faceHa(f) / rho(b)) / d;
          const Real af = m.faceArea(f) * invRho / d;
          col(q) = m.cellFaceSign(q) > 0 ? b : a;
          coef(q) = -af;
          dsum += af;
        }
        diag(i) = dsum;
      });
}

/// out = L x (CSR product; x per combined cell, exchanged by the caller). K2.
template <class Real>
void applyLaplacian(const LaplacianMatrix<Real>& L, const DV<Real>& x, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  auto start = L.start;
  auto col = L.col;
  auto coef = L.coef;
  auto diag = L.diag;
  Kokkos::parallel_for(
      "ale.Lrho.apply", Kokkos::RangePolicy<Exec>(0, L.nRows), KOKKOS_LAMBDA(const int i) {
        Real acc = diag(i) * x(i);
        for (int q = start(i); q < start(i + 1); ++q)
          acc += coef(q) * x(col(q));
        out(i) = acc;
      });
}

/// Φᵐ_f = (A_f/d_f)[(c_f − x_A)·ẋ_A − (c_f − x_B)·ẋ_B + (ẇ_A − ẇ_B)/2], positive when the face
/// moves along n_f. `xdot` 3 and `wdot` 1 per combined cell. K1.
template <class Real>
void sweptFlux(const fv::FaceMesh<Real>& m, const DV<Real>& xdot, const DV<Real>& wdot,
               const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "sweptFlux");
  Kokkos::parallel_for(
      "ale.swept", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        Real sa = 0, sb = 0;
        for (int c = 0; c < 3; ++c) {
          const Real cA = m.centroid(f, c), cB = cA - m.conn(f, c);
          sa += cA * xdot(3 * a + c);
          sb += cB * xdot(3 * b + c);
        }
        out(f) = m.faceArea(f) / m.faceDist(f) * (sa - sb + Real(0.5) * (wdot(a) - wdot(b)));
      });
}

/// ζU (3·nCombined) with G = cellGradient(U) and the face viscosity `muFace` (nFaces): the face
/// stress σ_f = 2μ_f V_f sym G_f (K1, into `scratchF12`, 12·nFaces: ½τ_f with τ_f = σ_f(I − n n),
/// then σ_f n_f/d_f), then the adjoint of U → G_f in two cell gathers (K2, K2) with the moments
/// M = T/V, T_i = Σ_{f∋i} ½τ_f, in `scratch9` (9·nCombined) exchanged between them.
template <class Real>
void friction(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& G,
              const DV<Real>& muFace, const DV<Real>& out, const DV<Real>& scratchF12,
              const DV<Real>& scratch9,
              const std::function<void(const DV<Real>&, int)>& exchange = {}) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "friction");
  Kokkos::parallel_for(
      "ale.zeta.face", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        const Real d = m.faceDist(f);
        Real n[3], gf[9];
        for (int k = 0; k < 3; ++k)
          n[k] = m.normal(f, k);
        for (int j = 0; j < 9; ++j)
          gf[j] = Real(0.5) * (G(9 * a + j) + G(9 * b + j));
        for (int c = 0; c < 3; ++c) {
          Real gn = 0;
          for (int k = 0; k < 3; ++k)
            gn += gf[3 * c + k] * n[k];
          const Real corr = (U(3 * b + c) - U(3 * a + c)) / d - gn;
          for (int k = 0; k < 3; ++k)
            gf[3 * c + k] += corr * n[k];
        }
        const Real vf = m.faceArea(f) * d / Real(3);
        const Real w = Real(2) * muFace(f) * vf;
        Real sig[9];
        for (int c = 0; c < 3; ++c)
          for (int k = 0; k < 3; ++k)
            sig[3 * c + k] = w * (Real(0.5) * (gf[3 * c + k] + gf[3 * k + c]));
        for (int c = 0; c < 3; ++c) {
          Real sn = 0;
          for (int k = 0; k < 3; ++k)
            sn += sig[3 * c + k] * n[k];
          for (int k = 0; k < 3; ++k)
            scratchF12(12 * f + 3 * c + k) = Real(0.5) * (sig[3 * c + k] - sn * n[k]);
          scratchF12(12 * f + 9 + c) = sn / d;
        }
      });
  Kokkos::parallel_for(
      "ale.zeta.T", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real v[3] = {0, 0, 0}, T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const Real s = m.cellFaceSign(q);
          for (int j = 0; j < 9; ++j)
            T[j] += scratchF12(12 * f + j);
          for (int c = 0; c < 3; ++c)
            v[c] -= s * scratchF12(12 * f + 9 + c);
        }
        const Real vol = m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = v[c];
        for (int j = 0; j < 9; ++j)
          scratch9(9 * i + j) = T[j] / vol;
      });
  if (exchange)
    exchange(scratch9, 9);
  detail::gradientAdjointPass(m, scratch9, out, Real(1));
}

/// γ_f = ρ_A[ψ(x_A) − ψ(c_f)] + ρ_B[ψ(c_f) − ψ(x_B)] (§11 A7) from the potential at the cells
/// (`psiCell`, per combined cell) and at the face centroids (`psiFace`, nFaces). K1.
template <class Real>
void bodyForceFace(const fv::FaceMesh<Real>& m, const DV<Real>& rho, const DV<Real>& psiCell,
                   const DV<Real>& psiFace, const DV<Real>& gamma) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "bodyForceFace");
  Kokkos::parallel_for(
      "ale.gamma", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        const Real pf = psiFace(f);
        gamma(f) = rho(a) * (psiCell(a) - pf) + rho(b) * (pf - psiCell(b));
      });
}

/// γ_f for the uniform field ψ = −g·x through differences: ψ(x_A) − ψ(c_f) = g·(c_f − x_A),
/// ψ(c_f) − ψ(x_B) = g·(conn_f − (c_f − x_A)). K1.
template <class Real>
void bodyForceFaceUniform(const fv::FaceMesh<Real>& m, const DV<Real>& rho, const Real g[3],
                          const DV<Real>& gamma) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "bodyForceFaceUniform");
  const Real g0 = g[0], g1 = g[1], g2 = g[2];
  Kokkos::parallel_for(
      "ale.gammaUniform", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const Real gv[3] = {g0, g1, g2};
        Real ga = 0, gb = 0;
        for (int c = 0; c < 3; ++c) {
          const Real cA = m.centroid(f, c);
          ga += gv[c] * cA;
          gb += gv[c] * (m.conn(f, c) - cA);
        }
        gamma(f) = rho(m.faceCellA(f)) * ga + rho(m.faceCellB(f)) * gb;
      });
}

/// The cell part of the body force, ρ_iV_iḡ (3·nCombined; owned cells written). K2.
template <class Real>
void bodyForceCell(const fv::FaceMesh<Real>& m, const DV<Real>& rho, const Real gbar[3],
                   const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  const Real g0 = gbar[0], g1 = gbar[1], g2 = gbar[2];
  Kokkos::parallel_for(
      "ale.bodyCell", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        const Real mass = rho(i) * m.cellVolume(i);
        out(3 * i) = mass * g0;
        out(3 * i + 1) = mass * g1;
        out(3 * i + 2) = mass * g2;
      });
}

}  // namespace peclet::voro::ale

#endif  // PECLET_VORO_ALE_OPERATORS_HPP
