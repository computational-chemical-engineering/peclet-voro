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

/// Face-normal flux of a uniform (or cell-wise given) velocity field: u_f = U(A-side)·n_f.
template <class Real>
void projectToFaces(const FaceMesh<Real>& m, const DV<Real>& ucell, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  Kokkos::parallel_for(
      "fv.project", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        const int a = m.faceCellA(f);
        Real un = 0;
        if (f < nI) {  // distance-weighted average of the two cell velocities
          const int b = m.faceCellB(f);
          const Real ha = m.faceHa(f), hb = m.faceHb(f);
          for (int c = 0; c < 3; ++c)
            un += ((hb * ucell(3 * a + c) + ha * ucell(3 * b + c)) / (ha + hb)) * m.normal(f, c);
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
  int it = 0;
  for (; it < maxIter; ++it) {
    laplacian(m, z, q, sf);
    Kokkos::parallel_for(
        "cg.negq", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { q(i) = -q(i); });
    const Real zq = dotCells(m, z, q);
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
