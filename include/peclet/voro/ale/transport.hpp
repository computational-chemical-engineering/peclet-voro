/**
 * @file ale/transport.hpp
 * \brief Moving-grid stage 1, WO-S1.2b (docs/ale_design.md §4.1 S4, §2.3 "transport tendency",
 * §11 A15): the momentum transport of the predictor as device kernels on a frame's face mesh.
 *
 * The specification is the host reference `tests/python/ale_reference.py` (`transport_mf`,
 * `upwind_mf`, `reconstruction_jump_mf`, `centroid_correction_mf`, `transport_explicit_mf`); gate
 * G1.2 C0 checks each kernel against it. Convention as `ale/operators.hpp` (§A.3): face f = (A, B),
 * s_if = +1 for i = A and −1 for i = B, `faceCentroid` = c_f − x_A, c_f − x_B = (c_f − x_A) −
 * conn_f. Fm_f is the face mass flux ρ_f F_f of the transport flux F (positive A → B).
 *  - T, skew form (A15): (TU)_i = ½ Σ_{f∋i} s_if Fm_f U_o, o the other cell. Uᵀ T U = 0 for any
 *    Fm; with the diagonal ½(D Fm)_i U_i (zero when DF = 0, not carried) it is the central flux.
 *  - Q = Δᵀ W Δ, W = diag(½|Fm_f|), (ΔU)_f = U_A − U_B: (QU)_i = Σ_{f∋i} ½|Fm_f| (U_i − U_o).
 *  - (R − Δ)U on f: G_A (c_f − x_A) − G_B (c_f − x_B), the jump of the two linear reconstructions
 *    at c_f minus the jump of the cell values (zero for a linear field).
 *  - δ_f = ½[G_A (c_f − x_A) + G_B (c_f − x_B)]: (U_A + U_B)/2 + δ_f is the mean of the two
 *    linear reconstructions at c_f.
 *  - explicit transport (the predictor's right side carries τ times it), from U_ext:
 *    e6 = −Σ_f s_if Fm_f δ_f (the centroid correction of the central flux) and
 *    e8 = −Σ_f s_if ½|Fm_f| ((R − Δ)U)_f (the explicit part of the upwind split ΔᵀW R).
 *    The two are kept apart because the energy budget books them separately (B6, B8b).
 * G = cellGradient(U) of `ale/operators.hpp`, valid at both cells of every face.
 *
 * Kernel discipline (§2.2) as in `ale/operators.hpp`: K1 face loops write per-face arrays
 * (3·nFaces, flat 3f + c), K2 cell gathers loop the cell's face list in its stored order and
 * accumulate in registers; no atomics. Cell fields are 3·nCombined (flat 3i + c), written on the
 * owned cells.
 */
#ifndef PECLET_VORO_ALE_TRANSPORT_HPP
#define PECLET_VORO_ALE_TRANSPORT_HPP

#include <Kokkos_Core.hpp>

#include "peclet/voro/ale/operators.hpp"

namespace peclet::voro::ale {

/// (TU)_i = ½ Σ_{f∋i} s_if Fm_f U_o (skew form). K2.
template <class Real>
void transport(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& Fm,
               const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "transport");
  Kokkos::parallel_for(
      "ale.T", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const bool isA = m.cellFaceSign(q) > 0;
          const int o = isA ? m.faceCellB(f) : m.faceCellA(f);
          const Real h = Real(0.5) * m.cellFaceSign(q) * Fm(f);
          for (int c = 0; c < 3; ++c)
            acc[c] += h * U(3 * o + c);
        }
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = acc[c];
      });
}

/// (QU)_i = Σ_{f∋i} ½|Fm_f| (U_i − U_o) = (Δᵀ W Δ U)_i. K2.
template <class Real>
void upwind(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& Fm,
            const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "upwind");
  Kokkos::parallel_for(
      "ale.Q", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const int o = m.cellFaceSign(q) > 0 ? m.faceCellB(f) : m.faceCellA(f);
          const Real w = Real(0.5) * Kokkos::fabs(Fm(f));
          for (int c = 0; c < 3; ++c)
            acc[c] += w * (U(3 * i + c) - U(3 * o + c));
        }
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = acc[c];
      });
}

/// (T + Q)U in one gather (the predictor operator's transport part). K2.
template <class Real>
void transportUpwind(const fv::FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& Fm,
                     const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "transportUpwind");
  Kokkos::parallel_for(
      "ale.TQ", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const bool isA = m.cellFaceSign(q) > 0;
          const int o = isA ? m.faceCellB(f) : m.faceCellA(f);
          const Real fm = Fm(f);
          const Real h = Real(0.5) * m.cellFaceSign(q) * fm;
          const Real w = Real(0.5) * Kokkos::fabs(fm);
          for (int c = 0; c < 3; ++c)
            acc[c] += h * U(3 * o + c) + w * (U(3 * i + c) - U(3 * o + c));
        }
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = acc[c];
      });
}

/// ((R − Δ)U)_f = G_A (c_f − x_A) − G_B (c_f − x_B) (3·nFaces). K1.
template <class Real>
void reconstructionJump(const fv::FaceMesh<Real>& m, const DV<Real>& G, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "reconstructionJump");
  Kokkos::parallel_for(
      "ale.RmD", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        Real rA[3], rB[3];
        for (int k = 0; k < 3; ++k) {
          rA[k] = m.centroid(f, k);
          rB[k] = rA[k] - m.conn(f, k);
        }
        for (int c = 0; c < 3; ++c) {
          Real ga = 0, gb = 0;
          for (int k = 0; k < 3; ++k) {
            ga += G(9 * a + 3 * c + k) * rA[k];
            gb += G(9 * b + 3 * c + k) * rB[k];
          }
          out(3 * f + c) = ga - gb;
        }
      });
}

/// δ_f = ½[G_A (c_f − x_A) + G_B (c_f − x_B)] (3·nFaces). K1.
template <class Real>
void centroidCorrection(const fv::FaceMesh<Real>& m, const DV<Real>& G, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "centroidCorrection");
  Kokkos::parallel_for(
      "ale.delta", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        Real rA[3], rB[3];
        for (int k = 0; k < 3; ++k) {
          rA[k] = m.centroid(f, k);
          rB[k] = rA[k] - m.conn(f, k);
        }
        for (int c = 0; c < 3; ++c) {
          Real ga = 0, gb = 0;
          for (int k = 0; k < 3; ++k) {
            ga += G(9 * a + 3 * c + k) * rA[k];
            gb += G(9 * b + 3 * c + k) * rB[k];
          }
          out(3 * f + c) = Real(0.5) * (ga + gb);
        }
      });
}

/// out_i = sign · Σ_{f∋i} s_if q_f (3 components; q 3·nFaces): Δᵀq per component, i.e. the outflow
/// divergence of a face vector. K2.
template <class Real>
void faceScatter(const fv::FaceMesh<Real>& m, const DV<Real>& q3, const DV<Real>& out, Real sign) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "faceScatter");
  Kokkos::parallel_for(
      "ale.scatter3", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const Real s = m.cellFaceSign(q);
          for (int c = 0; c < 3; ++c)
            acc[c] += s * q3(3 * f + c);
        }
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = sign * acc[c];
      });
}

/// The explicit transport from U_ext (G = cellGradient(U_ext)): `outCentroid` = e6 = −Σ_f s_if
/// Fm_f δ_f and `outUpwind` = e8 = −Σ_f s_if ½|Fm_f| ((R − Δ)U)_f; their sum is
/// `transport_explicit_mf` of the reference. The face terms go through `scratch6F` (6·nFaces:
/// Fm δ at 3f + c, W(R − Δ)U at 3·nFaces + 3f + c). K1, K2, K2.
template <class Real>
void explicitTransport(const fv::FaceMesh<Real>& m, const DV<Real>& G, const DV<Real>& Fm,
                       const DV<Real>& outCentroid, const DV<Real>& outUpwind,
                       const DV<Real>& scratch6F) {
  using Exec = peclet::core::ExecSpace;
  detail::requireInteriorOnly(m, "explicitTransport");
  const int nF = m.nFaces;
  Kokkos::parallel_for(
      "ale.explicit.face", Kokkos::RangePolicy<Exec>(0, nF), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        const Real fm = Fm(f);
        const Real w = Real(0.5) * Kokkos::fabs(fm);
        Real rA[3], rB[3];
        for (int k = 0; k < 3; ++k) {
          rA[k] = m.centroid(f, k);
          rB[k] = rA[k] - m.conn(f, k);
        }
        for (int c = 0; c < 3; ++c) {
          Real ga = 0, gb = 0;
          for (int k = 0; k < 3; ++k) {
            ga += G(9 * a + 3 * c + k) * rA[k];
            gb += G(9 * b + 3 * c + k) * rB[k];
          }
          scratch6F(3 * f + c) = fm * (Real(0.5) * (ga + gb));
          scratch6F(3 * nF + 3 * f + c) = w * (ga - gb);
        }
      });
  const DV<Real> qCentroid = Kokkos::subview(scratch6F, std::make_pair(0, 3 * nF));
  const DV<Real> qUpwind = Kokkos::subview(scratch6F, std::make_pair(3 * nF, 6 * nF));
  ale::faceScatter(m, qCentroid, outCentroid, Real(-1));
  ale::faceScatter(m, qUpwind, outUpwind, Real(-1));
}

}  // namespace peclet::voro::ale

#endif  // PECLET_VORO_ALE_TRANSPORT_HPP
