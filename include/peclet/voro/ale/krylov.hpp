/**
 * @file ale/krylov.hpp
 * \brief Moving-grid stage 1, WO-S1.2b (docs/ale_design.md §4.2–§4.5): the device Krylov solvers
 * of the time step — preconditioned conjugate gradients (the L_ρ solves and the oracle projection
 * of mode R) and right-preconditioned restarted GMRES (the predictor, whose operator is not
 * symmetric: the transport T is skew, §11 A15).
 *
 * Both are written against `Operator`s on device vectors (`op(in, out)`, `prec(in, out)`), so that
 * a matrix-free operator and a multigrid V-cycle plug in alike. Vectors are `DV<Real>` of at least
 * `n` entries; the iteration runs over the first `n` (the owned entries). The vector updates are
 * elementwise kernels (bitwise reproducible); the inner products are `parallel_reduce` (§2.2 K5),
 * whose summation order is fixed for a fixed launch (backend and thread count) and differs between
 * launches, so the last bits of an iterate depend on the launch. The scalar recurrences (CG's α, β;
 * GMRES's Hessenberg matrix and Givens rotations) are host scalars (§2.1).
 *
 * Stopping tests are on the true or recursively updated residual r = b − A x, never on the
 * preconditioned one:
 *  - PCG: converged when |r|₂ ≤ rtol |b|₂, or |r|₂ ≤ atol, or max_i |r_i| w_i ≤ maxTol (each test
 *    active when its tolerance is positive; `w` empty means w_i = 1). The last is the cell-wise
 *    test of the compact projections (§4.4).
 *  - GMRES: |r|₂ ≤ rtol |b|₂ on the TRUE residual, checked at every restart (as the reference's
 *    `gmres` in tests/python/ale_reference_stepper.py); inside a cycle the Givens estimate is run
 *    to half the tolerance. On return the work's `r` holds b − A x of the returned x, which the
 *    caller books (the residual work and momentum of an inexact solve, §4.9, §11 A15).
 */
#ifndef PECLET_VORO_ALE_KRYLOV_HPP
#define PECLET_VORO_ALE_KRYLOV_HPP

#include <cmath>
#include <functional>
#include <Kokkos_Core.hpp>
#include <limits>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/fv/operators.hpp"

namespace peclet::voro::ale::krylov {

template <class Real>
using DV = fv::DV<Real>;

/// A linear map on device vectors, `op(in, out)`: the operator or the preconditioner of a solve. A
/// std::function and not a template parameter: nvcc forbids a host lambda type as the template
/// argument of a function that contains extended device lambdas.
template <class Real>
using Operator = std::function<void(const DV<Real>&, const DV<Real>&)>;

/// The outcome of one solve.
struct SolveReport {
  int iterations = 0;    ///< Krylov iterations (CG steps; GMRES inner steps over all cycles)
  int applications = 0;  ///< operator applications, residual evaluations included
  double residual0 = 0;  ///< |r|₂ of the initial guess
  double residual = 0;   ///< |r|₂ at return (GMRES: of the true residual)
  double rhsNorm = 0;    ///< |b|₂ (after the mean removal, when deflated)
  bool converged = false;
};

// ---- vector primitives (owned entries [0, n)) --------------------------------------------------
template <class Real>
Real dot(const DV<Real>& a, const DV<Real>& b, int n) {
  using Exec = peclet::core::ExecSpace;
  Real s = 0;
  Kokkos::parallel_reduce(
      "ale.kry.dot", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, Real& acc) { acc += a(i) * b(i); }, s);
  return s;
}

template <class Real>
Real norm2(const DV<Real>& a, int n) {
  return std::sqrt(dot(a, a, n));
}

/// max_i |r_i| w_i (w empty: w_i = 1).
template <class Real>
Real maxScaled(const DV<Real>& r, const DV<Real>& w, int n) {
  using Exec = peclet::core::ExecSpace;
  const bool hasW = w.extent(0) > 0;
  Real s = 0;
  Kokkos::parallel_reduce(
      "ale.kry.max", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, Real& acc) {
        const Real v = Kokkos::fabs(r(i)) * (hasW ? w(i) : Real(1));
        acc = v > acc ? v : acc;
      },
      Kokkos::Max<Real>(s));
  return s;
}

/// y ← y + a x.
template <class Real>
void axpy(const DV<Real>& y, Real a, const DV<Real>& x, int n) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.kry.axpy", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i) { y(i) += a * x(i); });
}

/// y ← x + b y.
template <class Real>
void xpby(const DV<Real>& y, const DV<Real>& x, Real b, int n) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.kry.xpby", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i) { y(i) = x(i) + b * y(i); });
}

/// out ← a − b.
template <class Real>
void difference(const DV<Real>& a, const DV<Real>& b, const DV<Real>& out, int n) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.kry.diff", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i) { out(i) = a(i) - b(i); });
}

/// x ← a x.
template <class Real>
void scale(const DV<Real>& x, Real a, int n) {
  using Exec = peclet::core::ExecSpace;
  Kokkos::parallel_for(
      "ale.kry.scale", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const int i) { x(i) *= a; });
}

/// x ← x − mean(x) (the plain mean over [0, n)); returns the removed mean.
template <class Real>
Real removeMean(const DV<Real>& x, int n) {
  using Exec = peclet::core::ExecSpace;
  Real s = 0;
  Kokkos::parallel_reduce(
      "ale.kry.sum", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, Real& acc) { acc += x(i); }, s);
  const Real mean = s / Real(n);
  Kokkos::parallel_for(
      "ale.kry.demean", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i) { x(i) -= mean; });
  return mean;
}

// ---- preconditioned conjugate gradients
// ----------------------------------------------------------
/// The stopping test of `pcg` (see the file comment); a test with a non-positive tolerance is off.
template <class Real>
struct PcgTest {
  Real rtol = 0;    ///< |r|₂ ≤ rtol |b|₂
  Real atol = 0;    ///< |r|₂ ≤ atol
  Real maxTol = 0;  ///< max_i |r_i| w_i ≤ maxTol
  DV<Real> weight;  ///< w of the max test (empty: 1)
};

/// Scratch of `pcg` (sized on first use, reallocated when n grows).
template <class Real>
struct PcgWork {
  DV<Real> r, z, d, q;
  void reserve(int n) {
    if ((int)r.extent(0) >= n)
      return;
    r = DV<Real>("ale.pcg.r", n);
    z = DV<Real>("ale.pcg.z", n);
    d = DV<Real>("ale.pcg.d", n);
    q = DV<Real>("ale.pcg.q", n);
  }
};

/// Solve A x = b by PCG from the initial guess in `x` (warm start; zero it for a cold start).
/// `deflate`: A is singular with the constants as null space (all-periodic L_ρ): the mean is
/// removed from the initial residual, from every preconditioned residual and from x on return.
/// A must be symmetric positive (semi)definite and `prec` symmetric positive definite.
template <class Real>
SolveReport pcg(const Operator<Real>& A, const Operator<Real>& M, const DV<Real>& b,
                const DV<Real>& x, const PcgTest<Real>& test, bool deflate, int maxIt,
                PcgWork<Real>& w, int n) {
  SolveReport rep;
  w.reserve(n);
  const DV<Real> r = w.r, z = w.z, d = w.d, q = w.q;
  // r = b − A x
  A(x, q);
  ++rep.applications;
  difference(b, q, r, n);
  Real bn2 = dot(b, b, n);
  if (deflate) {
    removeMean(r, n);
    Real bs = 0;
    {
      using Exec = peclet::core::ExecSpace;
      Kokkos::parallel_reduce(
          "ale.kry.sum", Kokkos::RangePolicy<Exec>(0, n),
          KOKKOS_LAMBDA(const int i, Real& acc) { acc += b(i); }, bs);
    }
    bn2 -= bs * bs / Real(n);  // |b − mean(b)|²
  }
  rep.rhsNorm = std::sqrt(bn2 > 0 ? bn2 : Real(0));
  auto done = [&](Real rn) {
    if (test.rtol > 0 && rn <= test.rtol * rep.rhsNorm)
      return true;
    if (test.atol > 0 && rn <= test.atol)
      return true;
    if (test.maxTol > 0 && maxScaled(r, test.weight, n) <= test.maxTol)
      return true;
    return false;
  };
  Real rn = norm2(r, n);
  rep.residual0 = rep.residual = rn;
  if (rn == Real(0) || done(rn)) {
    rep.converged = true;
    if (deflate)
      removeMean(x, n);
    return rep;
  }
  M(r, z);
  if (deflate)
    removeMean(z, n);
  Kokkos::deep_copy(Kokkos::subview(d, std::make_pair(0, n)),
                    Kokkos::subview(z, std::make_pair(0, n)));
  Real rz = dot(r, z, n);
  for (int it = 0; it < maxIt; ++it) {
    A(d, q);
    ++rep.applications;
    const Real dq = dot(d, q, n);
    if (!(dq > Real(0)))
      break;  // breakdown (A not positive on d)
    const Real alpha = rz / dq;
    axpy(x, alpha, d, n);
    axpy(r, -alpha, q, n);
    rep.iterations = it + 1;
    rn = norm2(r, n);
    rep.residual = rn;
    if (done(rn)) {
      rep.converged = true;
      break;
    }
    M(r, z);
    if (deflate)
      removeMean(z, n);
    const Real rzn = dot(r, z, n);
    const Real beta = rzn / rz;
    rz = rzn;
    xpby(d, z, beta, n);
  }
  if (deflate)
    removeMean(x, n);
  return rep;
}

// ---- right-preconditioned restarted GMRES
// ---------------------------------------------------------
/// Scratch of `gmres`: the Krylov basis (n × (restart + 1), one column per vector), the residual
/// `r` (b − A x of the returned x) and two work vectors.
template <class Real>
struct GmresWork {
  using Mem = peclet::core::MemSpace;
  Kokkos::View<Real**, Kokkos::LayoutLeft, Mem> V;
  DV<Real> r, w, z;
  Kokkos::View<Real*, Mem> y;
  int restart = 0;
  void reserve(int n, int m) {
    if ((int)V.extent(0) >= n && restart == m)
      return;
    restart = m;
    V = Kokkos::View<Real**, Kokkos::LayoutLeft, Mem>("ale.gmres.V", n, m + 1);
    r = DV<Real>("ale.gmres.r", n);
    w = DV<Real>("ale.gmres.w", n);
    z = DV<Real>("ale.gmres.z", n);
    y = Kokkos::View<Real*, Mem>("ale.gmres.y", m);
  }
  DV<Real> col(int j) const { return Kokkos::subview(V, Kokkos::ALL, j); }
};

/// Solve A x = b by GMRES(restart) with right preconditioning, x = P u, from the initial guess in
/// `x`. Modified Gram–Schmidt, Givens rotations. Returns when the true residual is ≤ rtol |b|₂ or
/// after `maxIt` inner iterations (not converged); `w.r` holds b − A x of the returned x.
template <class Real>
SolveReport gmres(const Operator<Real>& A, const Operator<Real>& P, const DV<Real>& b,
                  const DV<Real>& x, Real rtol, int restart, int maxIt, GmresWork<Real>& w, int n) {
  using Exec = peclet::core::ExecSpace;
  SolveReport rep;
  w.reserve(n, restart);
  const int m = restart;
  rep.rhsNorm = norm2(b, n);
  const Real tol = rtol * rep.rhsNorm;
  std::vector<Real> H((m + 1) * m), cs(m), sn(m), g(m + 1);
  auto h = [&](int i, int j) -> Real& { return H[i * m + j]; };
  const DV<Real> r = w.r, wv = w.w, z = w.z;
  bool first = true;
  while (true) {
    A(x, wv);
    ++rep.applications;
    difference(b, wv, r, n);
    const Real beta = norm2(r, n);
    if (first) {
      rep.residual0 = beta;
      first = false;
    }
    rep.residual = beta;
    if (beta <= tol) {
      rep.converged = true;
      return rep;
    }
    if (rep.iterations >= maxIt)
      return rep;
    // V_0 = r/β
    {
      const DV<Real> v0 = w.col(0);
      const Real ib = Real(1) / beta;
      Kokkos::parallel_for(
          "ale.gmres.v0", Kokkos::RangePolicy<Exec>(0, n),
          KOKKOS_LAMBDA(const int i) { v0(i) = r(i) * ib; });
    }
    std::fill(H.begin(), H.end(), Real(0));
    std::fill(g.begin(), g.end(), Real(0));
    g[0] = beta;
    int k = 0;
    for (int j = 0; j < m; ++j) {
      P(w.col(j), z);
      A(z, wv);
      ++rep.applications;
      for (int i = 0; i <= j; ++i) {
        const DV<Real> vi = w.col(i);
        h(i, j) = dot(wv, vi, n);
        axpy(wv, -h(i, j), vi, n);
      }
      h(j + 1, j) = norm2(wv, n);
      {
        const DV<Real> vj1 = w.col(j + 1);
        const Real hn = h(j + 1, j);
        const Real ih = hn > Real(0) ? Real(1) / hn : Real(1);
        Kokkos::parallel_for(
            "ale.gmres.vnext", Kokkos::RangePolicy<Exec>(0, n),
            KOKKOS_LAMBDA(const int i) { vj1(i) = wv(i) * ih; });
      }
      for (int i = 0; i < j; ++i) {
        const Real t = cs[i] * h(i, j) + sn[i] * h(i + 1, j);
        h(i + 1, j) = -sn[i] * h(i, j) + cs[i] * h(i + 1, j);
        h(i, j) = t;
      }
      const Real den = std::hypot(h(j, j), h(j + 1, j));
      cs[j] = h(j, j) / den;
      sn[j] = h(j + 1, j) / den;
      h(j, j) = den;
      h(j + 1, j) = 0;
      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];
      ++rep.iterations;
      k = j + 1;
      if (std::fabs(g[j + 1]) <= Real(0.5) * tol || h(j, j) == Real(0) || rep.iterations >= maxIt)
        break;
    }
    // y = H⁻¹ g (upper triangular, k × k), then x += P (V y)
    std::vector<Real> yh(k);
    for (int i = k - 1; i >= 0; --i) {
      Real s = g[i];
      for (int l = i + 1; l < k; ++l)
        s -= h(i, l) * yh[l];
      yh[i] = s / h(i, i);
    }
    {
      auto yHost = Kokkos::create_mirror_view(w.y);
      for (int i = 0; i < k; ++i)
        yHost(i) = yh[i];
      Kokkos::deep_copy(w.y, yHost);
      const auto V = w.V;
      const auto yd = w.y;
      Kokkos::parallel_for(
          "ale.gmres.Vy", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const int i) {
            Real s = 0;
            for (int l = 0; l < k; ++l)
              s += V(i, l) * yd(l);
            wv(i) = s;
          });
    }
    P(wv, z);
    axpy(x, Real(1), z, n);
  }
}

}  // namespace peclet::voro::ale::krylov

#endif  // PECLET_VORO_ALE_KRYLOV_HPP
