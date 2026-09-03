/**
 * @file fv/operators.hpp
 * \brief Track C, rung C1: the discrete operators of the staggered COVOLUME scheme on a face mesh
 * (plan verdict V4) — face-normal fluxes as the primary velocity unknown, cell pressures, the
 * two-point flux (exact on Voronoi orthogonality), and the reconstructions.
 *
 *   divergence     (div u)_i   = (1/V_i) Σ_{f∈i} s_if u_f A_f            (u_f along the A→B normal)
 *   faceGradient   (grad p)_f  = (p_B − p_A)/d_f  (interior), (p_b − p_A)/h_A or 0 (boundary)
 *   laplacian      L p = div grad p                                       (= Σ A_f/d_f (p_j −
 * p_i)/V_i) greenGauss     ∇p_i = (1/V_i) Σ_f s_if p_f A_f n_f,  p_f distance-weighted between A
 * and B perot          u_i  = (1/V_i) Σ_f s_if u_f A_f (c_f − x_i)            (exact for a uniform
 * field) plus the discrete inner products that make div and grad ADJOINT to round-off: ⟨p, div u⟩_V
 * = −⟨grad p, u⟩_F,   ⟨a,b⟩_V = Σ V_i a_i b_i,   ⟨a,b⟩_F = Σ_interior A_f d_f a_f b_f.
 */
#ifndef PECLET_VORO_FV_OPERATORS_HPP
#define PECLET_VORO_FV_OPERATORS_HPP

#include <functional>
#include <Kokkos_Core.hpp>

#include "peclet/voro/fv/mesh.hpp"

namespace peclet::voro::fv {

template <class Real>
using DV = Kokkos::View<Real*, peclet::core::MemSpace>;

/// (div u)_i from face-normal fluxes u (nFaces; boundary faces included, along the outward normal).
template <class Real>
void divergence(const FaceMesh<Real>& m, const DV<Real>& u, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "fv.div", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc = 0;
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          acc += m.cellFaceSign(q) * u(f) * m.faceArea(f);
        }
        out(i) = acc / m.cellVolume(i);
      });
}

/// Two-point normal gradient on faces. Boundary faces: Dirichlet value pb(f − nInterior) if `pb` is
/// sized, else zero flux (Neumann).
template <class Real>
void faceGradient(const FaceMesh<Real>& m, const DV<Real>& p, const DV<Real>& out,
                  const DV<Real>& pb = DV<Real>{}) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  const bool dir = pb.extent(0) > 0;
  Kokkos::parallel_for(
      "fv.gradN", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f);
        if (f < nI) {
          out(f) = (p(m.faceCellB(f)) - p(a)) / m.faceDist(f);
        } else {
          out(f) = dir ? (pb(f - nI) - p(a)) / m.faceHa(f) : Real(0);
        }
      });
}

/// L p = div(grad p): the two-point Laplacian (Neumann on boundary faces unless `pb` is given).
template <class Real>
void laplacian(const FaceMesh<Real>& m, const DV<Real>& p, const DV<Real>& out,
               DV<Real>& scratchFace, const DV<Real>& pb = DV<Real>{}) {
  if ((int)scratchFace.extent(0) != m.nFaces)
    scratchFace = DV<Real>("fv.scratchFace", m.nFaces);
  faceGradient(m, p, scratchFace, pb);
  divergence(m, scratchFace, out);
}

/// Green–Gauss cell gradient (3N): distance-weighted face values, boundary faces use p_A.
template <class Real>
void greenGaussGradient(const FaceMesh<Real>& m, const DV<Real>& p, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  Kokkos::parallel_for(
      "fv.greenGauss", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real g[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const int a = m.faceCellA(f);
          Real pf;
          if (f < nI) {
            const int b = m.faceCellB(f);
            const Real ha = m.faceHa(f), hb = m.faceHb(f);
            pf = (hb * p(a) + ha * p(b)) / (ha + hb);  // linear along the connector
          } else {
            pf = p(a);
          }
          const Real w = m.cellFaceSign(q) * pf * m.faceArea(f);
          for (int c = 0; c < 3; ++c)
            g[c] += w * m.normal(f, c);
        }
        const Real iv = Real(1) / m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = g[c] * iv;
      });
}

/// Perot reconstruction of the cell velocity (3N) from face-normal fluxes.
template <class Real>
void perotVelocity(const FaceMesh<Real>& m, const DV<Real>& u, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "fv.perot", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real v[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const Real s = m.cellFaceSign(q);
          const Real w = s * u(f) * m.faceArea(f);
          // the face centroid relative to THIS cell's seed: A-relative as stored, or shifted by the
          // connector for the B side
          for (int c = 0; c < 3; ++c) {
            const Real cf = m.centroid(f, c) - (s < 0 ? m.conn(f, c) : Real(0));
            v[c] += w * cf;
          }
        }
        const Real iv = Real(1) / m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = v[c] * iv;
      });
}

/// Skewness correction of the Green–Gauss gradient. For a LINEAR field the plain Green–Gauss
/// gradient (face values at the connector foot) is (I − S) ∇p with the geometric tensor
/// S = (1/V) Σ_f s_f A_f t_f ⊗ n_f (t_f = the face centroid's tangential offset from the
/// connector; Gauss: Σ_f A_f c_f ⊗ n_f = V I). R = (I − S)⁻¹ makes it exact for linear fields on
/// any polyhedral cell; zero on a lattice / centroidal-symmetric cell.
template <class Real>
KOKKOS_INLINE_FUNCTION void cellSkewInverse(const FaceMesh<Real>& m, int i, Real R[9]) {
  Real A[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // I − S
  const Real iv = Real(1) / m.cellVolume(i);
  for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
    const int f = m.cellFace(q);  // all faces: the identity holds for the full polytope
    const Real w = m.cellFaceSign(q) * m.faceArea(f) * iv, ha = m.faceHa(f);
    for (int r = 0; r < 3; ++r) {
      const Real t = m.centroid(f, r) - ha * m.normal(f, r);
      for (int k = 0; k < 3; ++k)
        A[3 * r + k] -= w * t * m.normal(f, k);
    }
  }
  const Real det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
                   A[2] * (A[3] * A[7] - A[4] * A[6]);
  const Real id = Real(1) / det;
  R[0] = (A[4] * A[8] - A[5] * A[7]) * id;
  R[1] = (A[2] * A[7] - A[1] * A[8]) * id;
  R[2] = (A[1] * A[5] - A[2] * A[4]) * id;
  R[3] = (A[5] * A[6] - A[3] * A[8]) * id;
  R[4] = (A[0] * A[8] - A[2] * A[6]) * id;
  R[5] = (A[2] * A[3] - A[0] * A[5]) * id;
  R[6] = (A[3] * A[7] - A[4] * A[6]) * id;
  R[7] = (A[1] * A[6] - A[0] * A[7]) * id;
  R[8] = (A[0] * A[4] - A[1] * A[3]) * id;
}

/// Skew-corrected Green–Gauss gradient of a cell VECTOR field (3N → 9N, (∂_k U_c)_i at
/// 9i + 3c + k): the plain distance-weighted Green–Gauss gradient times R = (I − S)⁻¹
/// (cellSkewInverse) — exact for linear fields on every cell; boundary faces use the cell value.
template <class Real>
void vectorGreenGauss(const FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& out,
                      const DV<Real>& Uwall = DV<Real>{}) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  const bool hasW = Uwall.extent(0) > 0;
  Kokkos::parallel_for(
      "fv.vgg", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real g[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const Real s = m.cellFaceSign(q);
          const int a = m.faceCellA(f), b = m.faceCellB(f);
          Real Uf[3];
          if (f < nI) {
            const Real ha = m.faceHa(f), hb = m.faceHb(f);
            for (int c = 0; c < 3; ++c)
              Uf[c] = (hb * U(3 * a + c) + ha * U(3 * b + c)) / (ha + hb);
          } else {
            for (int c = 0; c < 3; ++c)
              Uf[c] = hasW ? Uwall(3 * (f - nI) + c) : U(3 * i + c);
          }
          const Real w = s * m.faceArea(f);
          for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
              g[3 * c + k] += w * Uf[c] * m.normal(f, k);
        }
        Real R[9];
        cellSkewInverse(m, i, R);
        const Real iv = Real(1) / m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          for (int k = 0; k < 3; ++k) {
            Real v = 0;
            for (int kk = 0; kk < 3; ++kk)
              v += R[3 * k + kk] * g[3 * c + kk];
            out(9 * i + 3 * c + k) = v * iv;
          }
      });
}

/// The centre→face CONSTRAINT interpolation T: u_f = U_f·n_f with U_f the distance-weighted
/// average of the two cell velocities (exact at the connector foot for linear fields). With
/// `gradU` (9N, vectorGreenGauss) the SKEW correction adds each cell's gradient times the
/// tangential offset t_f = c_f − h_A n_f of the face centroid from the connector, so U_f is the
/// value AT THE FACE CENTROID (second order on skewed faces). Boundary faces: U(A)·n.
template <class Real>
void projectToFaces(const FaceMesh<Real>& m, const DV<Real>& ucell, const DV<Real>& out,
                    const DV<Real>& gradU = DV<Real>{}, const DV<Real>& ub = DV<Real>{}) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  const bool skew = gradU.extent(0) > 0, hasB = ub.extent(0) > 0;
  Kokkos::parallel_for(
      "fv.project", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f);
        Real un = 0;
        if (f < nI) {
          const int b = m.faceCellB(f);
          const Real ha = m.faceHa(f), hb = m.faceHb(f), wa = hb / (ha + hb), wb = ha / (ha + hb);
          for (int c = 0; c < 3; ++c)
            un += (wa * ucell(3 * a + c) + wb * ucell(3 * b + c)) * m.normal(f, c);
          if (skew) {
            Real t[3];
            for (int k = 0; k < 3; ++k)
              t[k] = m.centroid(f, k) - ha * m.normal(f, k);
            for (int c = 0; c < 3; ++c) {
              Real dc = 0;
              for (int k = 0; k < 3; ++k)
                dc += (wa * gradU(9 * a + 3 * c + k) + wb * gradU(9 * b + 3 * c + k)) * t[k];
              un += dc * m.normal(f, c);
            }
          }
        } else if (hasB) {  // wall / inflow: the prescribed boundary flux
          un = ub(f - nI);
        } else {
          for (int c = 0; c < 3; ++c)
            un += ucell(3 * a + c) * m.normal(f, c);
        }
        out(f) = un;
      });
}

template <class Real>
Real dotCells(const FaceMesh<Real>& m, const DV<Real>& a, const DV<Real>& b) {
  using Exec = peclet::core::ExecSpace;
  Real s = 0;
  Kokkos::parallel_reduce(
      "fv.dotCells", Kokkos::RangePolicy<Exec>(0, m.nCells),
      KOKKOS_LAMBDA(const int i, Real& acc) { acc += m.cellVolume(i) * a(i) * b(i); }, s);
  return s;
}
template <class Real>
Real dotFaces(const FaceMesh<Real>& m, const DV<Real>& a, const DV<Real>& b) {
  using Exec = peclet::core::ExecSpace;
  Real s = 0;
  Kokkos::parallel_reduce(
      "fv.dotFaces", Kokkos::RangePolicy<Exec>(0, m.nInterior),
      KOKKOS_LAMBDA(const int f, Real& acc) { acc += m.faceArea(f) * m.faceDist(f) * a(f) * b(f); },
      s);
  return s;
}

/// Transpose of the constraint interpolation T (projectToFaces) in the inner products ⟨·,·⟩_V /
/// ⟨·,·⟩_F: ⟨T U, g⟩_F = ⟨U, Tᵀ g⟩_V for every cell field U and face field g. Plain part:
/// (T₀ᵀ g)_i = (1/V_i) Σ_{f∈i} A_f d_f w_if g_f n_f (w_if = the interpolation weight of cell i on
/// f; on a cubic lattice this IS the central difference = Green–Gauss). With `skew` the transpose
/// of the centroid correction T₁ ∘ GG is added (two passes: the per-cell moment
/// M_i,ck = Σ_f A_f d_f w_if g_f n_fc t_fk, then the Green–Gauss transpose through the faces).
/// Applied to g = the two-point face gradient of φ this is the cell pressure gradient that makes
/// the ABC cell correction the exact adjoint of the face constraint — flow's gauge-exact
/// gradient (`gpCenterGrad`, the transpose of centerToFace) on the Voronoi mesh: the pressure
/// does no work on the constraint manifold. `scratch9` (9N) is needed for the skew part.
template <class Real>
void faceInterpTranspose(const FaceMesh<Real>& m, const DV<Real>& g, const DV<Real>& out,
                         bool skew = false, const DV<Real>& scratch9 = DV<Real>{},
                         bool wallPrescribed = false,
                         const std::function<void(const DV<Real>&, int)>& exchange = {}) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  Kokkos::parallel_for(
      "fv.T0t", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real v[3] = {0, 0, 0}, M[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          if (f >= nI)
            continue;
          const Real ha = m.faceHa(f), hb = m.faceHb(f);
          const Real wi = (m.cellFaceSign(q) > 0 ? hb : ha) / (ha + hb);
          const Real w = m.faceArea(f) * (ha + hb) * wi * g(f);
          for (int c = 0; c < 3; ++c)
            v[c] += w * m.normal(f, c);
          if (skew) {
            Real t[3];
            for (int k = 0; k < 3; ++k)
              t[k] = m.centroid(f, k) - ha * m.normal(f, k);
            for (int c = 0; c < 3; ++c)
              for (int k = 0; k < 3; ++k)
                M[3 * c + k] += w * m.normal(f, c) * t[k];
          }
        }
        const Real iv = Real(1) / m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = v[c] * iv;
        if (skew) {  // (M_i R_i) / V_i — the transpose of the skew correction R (row index k)
          Real R[9];
          cellSkewInverse(m, i, R);
          for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) {
              Real v2 = 0;
              for (int kk = 0; kk < 3; ++kk)
                v2 += M[3 * c + kk] * R[3 * kk + k];
              scratch9(9 * i + 3 * c + k) = v2 * iv;
            }
        }
      });
  if (!skew)
    return;
  if (exchange)
    exchange(scratch9, 9);  // distributed: the neighbours' M/V (rung C5)
  // GGᵀ: coefficient of U_c(i) = Σ_{f∈i} w_if q_fc, q_fc = Σ_k A_f n_fk (M_A/V_A − M_B/V_B)_ck
  Kokkos::parallel_for(
      "fv.T1t", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real v[3] = {0, 0, 0};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          if (f >= nI) {  // vectorGreenGauss used the cell value on a boundary face (unless
                          // a wall velocity was prescribed: then the face value is a constant)
            if (wallPrescribed)
              continue;
            for (int c = 0; c < 3; ++c)
              for (int k = 0; k < 3; ++k)
                v[c] += m.faceArea(f) * m.normal(f, k) * scratch9(9 * i + 3 * c + k);
            continue;
          }
          const int a = m.faceCellA(f), b = m.faceCellB(f);
          const Real ha = m.faceHa(f), hb = m.faceHb(f);
          const Real wi = (m.cellFaceSign(q) > 0 ? hb : ha) / (ha + hb);
          for (int c = 0; c < 3; ++c) {
            Real qf = 0;
            for (int k = 0; k < 3; ++k)
              qf += m.faceArea(f) * m.normal(f, k) *
                    (scratch9(9 * a + 3 * c + k) - scratch9(9 * b + 3 * c + k));
            v[c] += wi * qf;
          }
        }
        const Real iv = Real(1) / m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) += v[c] * iv;
      });
}

/// Second-order wall gradient (3 × nBoundary): for each boundary face f of cell i, the normal
/// derivative AT THE WALL of the wall-anchored quadratic u(s) = U_wall + a s + b s² (s = distance
/// from the wall plane) least-squares fitted to the cell value and its interior neighbours, each
/// sample corrected for the tangential variation with the cell gradient gradU (9N; may be empty).
/// The two-point flux (U_i − U_wall)/h_A is the derivative at s = h_A/2 (the half-cell O(h) wall
/// shear that limits the sphere drag, C4); this is flow's wall-anchored quadratic
/// (centerToFaceWallAware) on the unstructured mesh. On a lattice wall cell the fit passes through
/// (wall, U_i, U_above) exactly, so Poiseuille is reproduced to round-off.
template <class Real>
void wallGradientLS(const FaceMesh<Real>& m, const DV<Real>& U, const DV<Real>& gradU,
                    const DV<Real>& Uwall, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior, nB = m.nFaces - nI;
  const bool hasG = gradU.extent(0) > 0, hasW = Uwall.extent(0) > 0;
  Kokkos::parallel_for(
      "fv.wallLS", Kokkos::RangePolicy<Exec>(0, nB), KOKKOS_LAMBDA(const int b) {
        const int f = nI + b, i = m.faceCellA(f);
        const Real ha = m.faceHa(f);
        Real n[3], Uw[3];
        for (int c = 0; c < 3; ++c) {
          n[c] = m.normal(f, c);
          Uw[c] = hasW ? Uwall(3 * b + c) : Real(0);
        }
        // normal equations of u − U_wall = a s + b s²: [S2 S3; S3 S4] [a; b] = [R1; R2]
        Real S2 = 0, S3 = 0, S4 = 0, R1[3] = {0, 0, 0}, R2[3] = {0, 0, 0};
        auto add = [&](Real s, const Real du[3]) {
          const Real s2 = s * s;
          S2 += s2;
          S3 += s2 * s;
          S4 += s2 * s2;
          for (int c = 0; c < 3; ++c) {
            R1[c] += s * du[c];
            R2[c] += s2 * du[c];
          }
        };
        Real du[3];
        for (int c = 0; c < 3; ++c)
          du[c] = U(3 * i + c) - Uw[c];
        add(ha, du);
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int g = m.cellFace(q);
          if (g >= nI)
            continue;
          const Real sg = m.cellFaceSign(q);
          const int j = sg > 0 ? m.faceCellB(g) : m.faceCellA(g);
          Real d[3], dn = 0;
          for (int c = 0; c < 3; ++c) {
            d[c] = sg * m.conn(g, c);  // x_j − x_i
            dn += d[c] * n[c];
          }
          const Real s = ha - dn;  // n points toward the wall: away from it is −n
          if (!(s > Real(0.05) * ha))
            continue;
          for (int c = 0; c < 3; ++c) {
            Real tang = 0;
            if (hasG)
              for (int k = 0; k < 3; ++k)
                tang += gradU(9 * i + 3 * c + k) * (d[k] - dn * n[k]);
            du[c] = U(3 * j + c) - tang - Uw[c];
          }
          add(s, du);
        }
        const Real det = S2 * S4 - S3 * S3;
        for (int c = 0; c < 3; ++c) {
          Real a;
          if (det > Real(1e-30) * S2 * S4)
            a = (S4 * R1[c] - S3 * R2[c]) / det;
          else
            a = (U(3 * i + c) - Uw[c]) / ha;  // no second abscissa: the two-point gradient
          out(3 * b + c) = a;
        }
      });
}

/// Conjugate gradients on −L p = f (matrix-free, Neumann/periodic: the mean of f is removed and
/// the mean of p pinned to zero — L is symmetric negative semi-definite with the constant null
/// space). Returns the iteration count; `resOut` the final relative residual. Unpreconditioned:
/// the coarse-grid / AMG preconditioner is track C2.
template <class Real>
int poissonCG(const FaceMesh<Real>& m, const DV<Real>& f, const DV<Real>& p, Real tol, int maxIter,
              Real& resOut) {
  using Exec = peclet::core::ExecSpace;
  const int N = m.nCells;
  DV<Real> r("cg.r", N), z("cg.z", N), q("cg.q", N), sf;
  Real Vtot = 0, fMean = 0;
  Kokkos::parallel_reduce(
      "cg.vtot", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) { acc += m.cellVolume(i); }, Vtot);
  Kokkos::parallel_reduce(
      "cg.fmean", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) { acc += m.cellVolume(i) * f(i); }, fMean);
  fMean /= Vtot;
  Kokkos::deep_copy(p, Real(0));
  // r = f − mean(f) − (−L p) = f − mean(f) at p = 0
  Kokkos::parallel_for(
      "cg.r0", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i) { r(i) = f(i) - fMean; });
  Kokkos::deep_copy(z, r);
  Real rr = dotCells(m, r, r);
  const Real r0 = Kokkos::sqrt(rr);
  if (!(r0 > Real(0))) {  // zero (or NaN) right-hand side: p = 0 is the solution
    resOut = 0;
    return 0;
  }
  int it = 0;
  for (; it < maxIter; ++it) {
    laplacian(m, z, q, sf);
    Kokkos::parallel_for(
        "cg.negq", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { q(i) = -q(i); });
    const Real zq = dotCells(m, z, q);
    if (!(zq > Real(0)))  // z in the null space (converged to round-off)
      break;
    const Real alpha = rr / zq;
    Kokkos::parallel_for(
        "cg.upd", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          p(i) += alpha * z(i);
          r(i) -= alpha * q(i);
        });
    const Real rrn = dotCells(m, r, r);
    resOut = Kokkos::sqrt(rrn) / (r0 > Real(0) ? r0 : Real(1));
    if (resOut < tol) {
      ++it;
      break;
    }
    const Real beta = rrn / rr;
    rr = rrn;
    Kokkos::parallel_for(
        "cg.z", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(const int i) { z(i) = r(i) + beta * z(i); });
  }
  // pin the mean of p to zero
  Real pMean = 0;
  Kokkos::parallel_reduce(
      "cg.pmean", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, Real& acc) { acc += m.cellVolume(i) * p(i); }, pMean);
  pMean /= Vtot;
  Kokkos::parallel_for(
      "cg.pin", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { p(i) -= pMean; });
  return it;
}

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_OPERATORS_HPP
