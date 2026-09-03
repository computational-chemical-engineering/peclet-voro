/**
 * @file fv/covolume.hpp
 * \brief Track C, rung C2a (Voronoi methods plan, verdict V4): the staggered COVOLUME
 * Navier–Stokes solver on the Voronoi face mesh — face-normal fluxes u_f as the velocity unknown,
 * cell pressures, exact projection with the two-point Laplacian L of C1.
 *
 * Momentum on the faces is the TRANSPOSE of the Perot reconstruction R (fluxes → cell vectors):
 *   R:   V_i U_i = Σ_{f∈i} s_if u_f A_f (c_f − x_i)
 *   Rᵀ:  d_f a_f = (c_f − x_A)·a_A − (c_f − x_B)·a_B             (interior faces)
 * so that ⟨R u, a⟩_V = ⟨u, Rᵀ a⟩_F to round-off (⟨a,b⟩_F = Σ A_f d_f a_f b_f). The cell-centred
 * tendency a_i is the finite-volume convection of the reconstructed field with the ARITHMETIC face
 * mean (on a Voronoi mesh the face is the bisector, h_A = h_B, so this is also the
 * distance-weighted mean) — skew-symmetric in ⟨·,·⟩_V for a divergence-free flux — plus the
 * two-point viscous term, which is symmetric negative. Hence, semi-discretely, d/dt ½⟨u,u⟩_F = −ν
 * Σ_f A_f/d_f |U_B − U_A|² ≤ 0, exactly zero in the inviscid limit, on ANY Voronoi mesh (skewness
 * included: the centroid offsets are carried by R and Rᵀ). The projection u ← u − Δt grad φ, L φ =
 * div u/Δt, is exact to the Poisson tolerance and is orthogonal to the divergence-free fluxes in
 * ⟨·,·⟩_F (adjointness of div and grad, C1), so it does not touch the energy either.
 *
 * Time integration: SSP-RK3, projection after every stage (the skew-symmetry needs div-free stage
 * fluxes). Pressure Poisson: unpreconditioned CG (C1) or PCG with core's smoothed-aggregation
 * GraphAMGDevice on the symmetric form −V L (the OT optimiser's graph Laplacian).
 *
 * Boundary faces (walls; C3): no-slip — zero flux, U = 0 at distance h_A for the viscous term.
 */
#ifndef PECLET_VORO_FV_COVOLUME_HPP
#define PECLET_VORO_FV_COVOLUME_HPP

#include <cstdio>
#include <functional>
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/graph_amg_device.hpp"
#include "peclet/voro/fv/dec.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/fv/operators.hpp"

namespace peclet::voro::fv {

/// Transpose of the Perot reconstruction: face-normal tendency from cell tendencies (3N).
template <class Real>
void perotTranspose(const FaceMesh<Real>& m, const DV<Real>& acell, const DV<Real>& out) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  Kokkos::parallel_for(
      "fv.perotT", Kokkos::RangePolicy<Exec>(0, m.nFaces), KOKKOS_LAMBDA(const int f) {
        if (f >= nI) {
          out(f) = Real(0);
          return;
        }
        const int a = m.faceCellA(f), b = m.faceCellB(f);
        Real s = 0;
        for (int c = 0; c < 3; ++c) {
          const Real cf = m.centroid(f, c);
          s += cf * acell(3 * a + c) - (cf - m.conn(f, c)) * acell(3 * b + c);
        }
        out(f) = s / m.faceDist(f);
      });
}

/// Cell-centred tendency of the reconstructed velocity U (3N) under the face flux u:
///   a_i = −(1/V_i) Σ_f s_if u_f A_f ½(U_i + U_j) + (ν/V_i) Σ_f A_f/d_f (U_j − U_i) + F_i
/// Boundary faces (walls, C3): the face velocity is the prescribed `Uwall` (3 × nBoundary; zero =
/// no-slip when not given) at distance h_A — two-point wall flux ν A_f (U_wall − U_i)/h_A, and the
/// convective flux F_f carries U_wall (F_f = 0 on an impermeable wall). `force` (3N) optional.
template <class Real>
void cellTendency(const FaceMesh<Real>& m, const DV<Real>& u, const DV<Real>& U, Real nu,
                  const DV<Real>& out, const DV<Real>& force = DV<Real>{}, Real convScale = Real(1),
                  const DV<Real>& Uwall = DV<Real>{}, const DV<Real>& wallGrad = DV<Real>{}) {
  using Exec = peclet::core::ExecSpace;
  const int nI = m.nInterior;
  const bool hasF = force.extent(0) > 0, hasW = Uwall.extent(0) > 0, hasWG = wallGrad.extent(0) > 0;
  Kokkos::parallel_for(
      "fv.tendency", Kokkos::RangePolicy<Exec>(0, m.nCells), KOKKOS_LAMBDA(const int i) {
        Real acc[3] = {0, 0, 0};
        const Real Ui[3] = {U(3 * i), U(3 * i + 1), U(3 * i + 2)};
        for (int q = m.cellFacesBegin(i); q < m.cellFacesEnd(i); ++q) {
          const int f = m.cellFace(q);
          const Real s = m.cellFaceSign(q);
          const Real F = convScale * s * u(f) * m.faceArea(f);  // outward volume flux
          if (f < nI) {
            const int j = (s > 0) ? m.faceCellB(f) : m.faceCellA(f);
            const Real w = nu * m.faceArea(f) / m.faceDist(f);
            for (int c = 0; c < 3; ++c) {
              const Real Uj = U(3 * j + c);
              acc[c] += -F * Real(0.5) * (Ui[c] + Uj) + w * (Uj - Ui[c]);
            }
          } else {  // wall at distance h_A with the prescribed velocity
            const Real w = nu * m.faceArea(f) / m.faceHa(f);
            for (int c = 0; c < 3; ++c) {
              const Real Uw = hasW ? Uwall(3 * (f - nI) + c) : Real(0);
              // viscous wall flux: −ν A ∂u/∂n at the wall (n outward), two-point or quadratic
              const Real visc =
                  hasWG ? -nu * m.faceArea(f) * wallGrad(3 * (f - nI) + c) : w * (Uw - Ui[c]);
              acc[c] += -F * Uw + visc;
            }
          }
        }
        const Real iv = Real(1) / m.cellVolume(i);
        for (int c = 0; c < 3; ++c)
          out(3 * i + c) = acc[c] * iv + (hasF ? force(3 * i + c) : Real(0));
      });
}

/// Pressure Poisson solver: −L φ = f, unpreconditioned CG (C1) or GraphAMGDevice-PCG on the
/// symmetric form K = −V L (K_ii = Σ A_f/d_f, K_ij = −A_f/d_f), b = V (f − mean f).
template <class Real>
struct PressureSolver {
  using Exec = peclet::core::ExecSpace;
  FaceMesh<Real> m;
  bool useAmg = false;
  Real tol = 1e-12;
  int maxIter = 5000;
  // Generalization (implicit diffusion, C2): K = extraDiag + lapScale·(−V L). Pressure: lapScale 1,
  // no extraDiag, `deflate` (mean removal + pin, singular Neumann). Velocity: extraDiag_i = V_i/Δt
  // + ν Σ_wall A_f/h_A (the two-point wall term), lapScale = ν, no deflation (SPD).
  Real lapScale = 1;
  bool deflate = true;
  DV<Real> extraDiag;
  int lastIters = 0;
  Real lastRes = 0;
  peclet::core::solver::GraphAMGDevice amg;
  // Distributed hooks (rung C5): `exchange(field, ncomp)` refreshes the ghost entries of a
  // cell field (sized m.nCombined) from their owners, `sum` is the global reduction. Unset =
  // single rank. With `exchange` set the PCG path is used (block-Jacobi AMG per rank when
  // useAmg, plain CG otherwise) — poissonCG has no hooks.
  std::function<void(const DV<Real>&, int)> exchange;
  std::function<Real(Real)> sum;
  Real localVtot_ = 0;
  /// Install the distributed hooks (after setup): the total volume of the mean deflation becomes
  /// the GLOBAL volume — with the rank-local one the deflated right-hand side is not mean-free
  /// and CG on the singular Neumann system drifts (the C5 bug found by the isolated-solve gate).
  void setHooks(std::function<void(const DV<Real>&, int)> ex, std::function<Real(Real)> s) {
    exchange = std::move(ex);
    sum = std::move(s);
    Vtot = sum ? sum(localVtot_) : localVtot_;
  }
  DV<Real> r, z, q, b, Kd;
  Kokkos::View<double*, peclet::core::MemSpace> rd, zd;  // AMG works in double
  Real Vtot = 0;

  /// Velocity-solve configuration: (V/Δt + ν Σ_wall A/h_A) u − ν V L u; `Uwall`-independent (the
  /// Dirichlet wall value enters the right-hand side). Rebuild when Δt or ν change.
  void setupVelocity(const FaceMesh<Real>& mesh, Real nu, Real dt, bool withAmg) {
    lapScale = nu;
    deflate = false;
    const FaceMesh<Real> mm = mesh;
    extraDiag = DV<Real>("ps.extraDiag", mesh.nCells);
    const DV<Real> ed = extraDiag;
    const int nI = mesh.nInterior;
    Kokkos::parallel_for(
        "ps.ediag", Kokkos::RangePolicy<Exec>(0, mesh.nCells), KOKKOS_LAMBDA(const int i) {
          Real d = mm.cellVolume(i) / dt;
          for (int t = mm.cellFacesBegin(i); t < mm.cellFacesEnd(i); ++t) {
            const int f = mm.cellFace(t);
            if (f >= nI)
              d += nu * mm.faceArea(f) / mm.faceHa(f);
          }
          ed(i) = d;
        });
    setup(mesh, withAmg);
  }
  void setup(const FaceMesh<Real>& mesh, bool withAmg) {
    m = mesh;
    useAmg = withAmg;
    const int N = m.nCells;
    const int NC = m.nCombined > 0 ? m.nCombined : N;
    r = DV<Real>("ps.r", N);
    z = DV<Real>("ps.z", NC);  // read at ghosts by applyK (distributed)
    q = DV<Real>("ps.q", N);
    b = DV<Real>("ps.b", N);
    {
      const FaceMesh<Real> mm = m;  // no `this` capture on the device
      Kokkos::parallel_reduce(
          "ps.vtot", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i, Real& acc) { acc += mm.cellVolume(i); }, Vtot);
    }
    if (sum)
      Vtot = sum(Vtot);
    localVtot_ = Vtot;
    if (!useAmg)
      return;
    rd = Kokkos::View<double*, peclet::core::MemSpace>("ps.rd", N);
    zd = Kokkos::View<double*, peclet::core::MemSpace>("ps.zd", N);
    // host CSR of K = −V L (interior faces only: Neumann walls contribute nothing). Distributed:
    // the owned block only (couplings to ghosts stay in the diagonal — block-Jacobi AMG).
    auto A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.faceCellA);
    auto B = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.faceCellB);
    auto Af = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.faceArea);
    auto df = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.faceDist);
    auto off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.cellFaceOffset);
    auto cf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.cellFace);
    auto cs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.cellFaceSign);
    peclet::core::solver::HostCsrOp K;
    K.n = N;
    K.diag.assign(N, 0.0);
    K.start.assign(N + 1, 0);
    std::vector<Real> edh;
    if (extraDiag.extent(0) > 0)
      edh = peclet::core::toVector(extraDiag);
    for (int i = 0; i < N; ++i) {
      if (!edh.empty())
        K.diag[i] += edh[i];
      for (int qq = off(i); qq < off(i + 1); ++qq) {
        const int f = cf(qq);
        if (f >= m.nInterior)
          continue;
        const int j = cs(qq) > 0 ? B(f) : A(f);
        const double w = lapScale * Af(f) / df(f);
        K.diag[i] += w;
        if (j >= N)
          continue;  // ghost neighbour: off-block
        K.nbr.push_back(j);
        K.coef.push_back(-w);
      }
      K.start[i + 1] = (peclet::core::Index)K.nbr.size();
    }
    peclet::core::solver::AmgParams prm;
    prm.ndofPerNode = 1;
    // K is singular (constant null space): smoother sweeps on the coarsest level instead of the
    // near-exact coarse CG, which divides by zero once the coarse residual lies in the null
    // space (seen on a 512-cell mesh, NaN).
    prm.coarseSweeps = deflate ? 8 : 0;  // the velocity matrix is SPD: exact coarse solve is fine
    amg.build(K, prm);
  }

  // q = K z
  void applyK(const DV<Real>& zz, const DV<Real>& qq) const {
    const FaceMesh<Real> mm = m;
    const int nI = mm.nInterior;
    const Real ls = lapScale;
    const DV<Real> ed = extraDiag;
    const bool hasEd = ed.extent(0) > 0;
    Kokkos::parallel_for(
        "ps.K", Kokkos::RangePolicy<Exec>(0, mm.nCells), KOKKOS_LAMBDA(const int i) {
          Real acc = 0;
          const Real zi = zz(i);
          for (int t = mm.cellFacesBegin(i); t < mm.cellFacesEnd(i); ++t) {
            const int f = mm.cellFace(t);
            if (f >= nI)
              continue;
            const int j = mm.cellFaceSign(t) > 0 ? mm.faceCellB(f) : mm.faceCellA(f);
            acc += ls * mm.faceArea(f) / mm.faceDist(f) * (zi - zz(j));
          }
          qq(i) = acc + (hasEd ? ed(i) * zi : Real(0));
        });
  }
  // dot over the OWNED cells (fields may be sized nCombined), globally reduced
  Real dot(const DV<Real>& a, const DV<Real>& bb) const {
    Real s = 0;
    Kokkos::parallel_reduce(
        "ps.dot", Kokkos::RangePolicy<Exec>(0, m.nCells),
        KOKKOS_LAMBDA(const int i, Real& acc) { acc += a(i) * bb(i); }, s);
    return sum ? sum(s) : s;
  }
  void precond(const DV<Real>& rr, const DV<Real>& zz) const {
    if (!useAmg) {  // zz may be longer than rr (ghost slots): copy the owned part
      Kokkos::deep_copy(Kokkos::subview(zz, std::make_pair(0, (int)rr.extent(0))), rr);
      return;
    }
    if constexpr (std::is_same_v<Real, double>) {
      if (zz.extent(0) == rr.extent(0))
        amg.apply(rr, zz);
      else
        amg.apply(rr, Kokkos::subview(zz, std::make_pair(0, (int)rr.extent(0))));
    } else {
      Kokkos::deep_copy(rd, rr);
      amg.apply(rd, zd);
      Kokkos::deep_copy(zz, zd);
    }
  }
  /// Solve −L p = f (mean of f removed, mean of p pinned to zero). Returns iterations.
  int solve(const DV<Real>& f, const DV<Real>& p) {
    const int N = m.nCells;
    if (!useAmg && !exchange && deflate)
      return lastIters = poissonCG(m, f, p, tol, maxIter, lastRes);
    const FaceMesh<Real> mm = m;
    const DV<Real> rr = r, zz = z, qq = q;
    const Real vt = Vtot;
    Real fMean = 0;
    if (deflate) {
      Kokkos::parallel_reduce(
          "ps.fmean", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i, Real& acc) { acc += mm.cellVolume(i) * f(i); }, fMean);
      if (sum)
        fMean = sum(fMean);
      fMean /= vt;
    }
    Kokkos::deep_copy(p, Real(0));
    const bool defl = deflate;
    Kokkos::parallel_for(
        "ps.b", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
          rr(i) = defl ? mm.cellVolume(i) * (f(i) - fMean) : f(i);  // velocity: f is the full RHS
        });
    const Real r0 = Kokkos::sqrt(dot(rr, rr));
    if (!(r0 > Real(0))) {  // zero right-hand side: p = 0
      lastRes = 0;
      return lastIters = 0;
    }
    precond(rr, zz);
    DV<Real> d("ps.d", zz.extent(0));
    Kokkos::deep_copy(d, zz);
    Real rz = dot(rr, zz);
    int it = 0;
    for (; it < maxIter; ++it) {
      if (exchange)
        exchange(d, 1);
      applyK(d, qq);
      const Real dq = dot(d, qq);
      if (!(dq > Real(0)))
        break;
      const Real alpha = rz / dq;
      Kokkos::parallel_for(
          "ps.upd", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
            p(i) += alpha * d(i);
            rr(i) -= alpha * qq(i);
          });
      lastRes = Kokkos::sqrt(dot(rr, rr)) / (r0 > Real(0) ? r0 : Real(1));
      if (lastRes < tol) {
        ++it;
        break;
      }
      precond(rr, zz);
      const Real rzn = dot(rr, zz);
      const Real beta = rzn / rz;
      rz = rzn;
      Kokkos::parallel_for(
          "ps.d", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i) { d(i) = zz(i) + beta * d(i); });
    }
    if (deflate) {
      Real pMean = 0;
      Kokkos::parallel_reduce(
          "ps.pmean", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(const int i, Real& acc) { acc += mm.cellVolume(i) * p(i); }, pMean);
      if (sum)
        pMean = sum(pMean);
      pMean /= vt;
      Kokkos::parallel_for(
          "ps.pin", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) { p(i) -= pMean; });
    }
    if (exchange)
      exchange(p, 1);
    return lastIters = it;
  }
};

/// The covolume NS solver state: face fluxes u, cell pressure p (from the last projection).
template <class Real>
struct CovolumeNS {
  using Exec = peclet::core::ExecSpace;
  FaceMesh<Real> m;
  Real nu = 0;
  Real convScale = 1;         // 0 = Stokes (diagnostic)
  DV<Real> u, p, force;       // nFaces, nCells, 3N (optional)
  DV<Real> Uwall, ub;         // 3 × nBoundary wall velocity, nBoundary wall flux (C3)
  bool wallQuadratic = true;  // second-order wall gradient (wallGradientLS) vs two-point
  DV<Real> wg, g9w;           // 3 × nBoundary wall gradient, 9N gradient scratch
  // C2a′: the DEC (Nicolaides) viscous term ν(grad div − curl curl) on the faces instead of the
  // Perot-reconstructed Rᵀ Δ₂ R. Enable with setDec(view) (needs the view's edge lengths,
  // `withAreaGrad`); periodic (wall-free) meshes only for now.
  bool viscousDEC = false;
  DecEdges<Real> dec;
  DV<Real> decOut, decDiv;
  DV<Real> U, a, k, u1, u2, div, gphi;  // workspace
  PressureSolver<Real> poisson;
  Real lastDiv = 0;

  void setup(const FaceMesh<Real>& mesh, Real viscosity, bool amg = false) {
    m = mesh;
    nu = viscosity;
    const int N = m.nCells, F = m.nFaces;
    u = DV<Real>("ns.u", F);
    p = DV<Real>("ns.p", N);
    U = DV<Real>("ns.U", 3 * N);
    a = DV<Real>("ns.a", 3 * N);
    k = DV<Real>("ns.k", F);
    u1 = DV<Real>("ns.u1", F);
    u2 = DV<Real>("ns.u2", F);
    div = DV<Real>("ns.div", N);
    gphi = DV<Real>("ns.gphi", F);
    setWallVelocity(DV<Real>{});
    poisson.setup(m, amg);
  }
  /// Prescribe the boundary-face velocity (3 × nBoundary; empty = no-slip): boundary fluxes
  /// u_f = U_wall·n; the viscous wall flux and the convective wall value read U_wall.
  void setWallVelocity(const DV<Real>& Uw) {
    const int nB = m.nFaces - m.nInterior, nI = m.nInterior;
    Uwall = Uw;
    ub = DV<Real>("fv.ub", nB);
    if (Uw.extent(0) == 0)
      return;
    const FaceMesh<Real> mm = m;
    const DV<Real> bb = ub;
    Kokkos::parallel_for(
        "fv.ub", Kokkos::RangePolicy<Exec>(0, nB), KOKKOS_LAMBDA(const int b) {
          Real un = 0;
          for (int c = 0; c < 3; ++c)
            un += Uw(3 * b + c) * mm.normal(nI + b, c);
          bb(b) = un;
        });
  }
  /// Impose the boundary fluxes on a face field.
  void applyWallFlux(const DV<Real>& uf) const {
    const int nB = m.nFaces - m.nInterior, nI = m.nInterior;
    const DV<Real> bb = ub;
    Kokkos::parallel_for(
        "fv.wall", Kokkos::RangePolicy<Exec>(0, nB),
        KOKKOS_LAMBDA(const int b) { uf(nI + b) = bb(b); });
  }
  /// Install the DEC viscous term from the published view (facet-edge CSR with edge lengths).
  void setDec(const TessellationView<Real>& view) {
    dec = buildDecEdges<Real>(view, m);
    decOut = DV<Real>("ns.decOut", m.nFaces);
    decDiv = DV<Real>("ns.decDiv", m.nCells);
    viscousDEC = true;
  }
  /// k = Rᵀ a(u): the face-normal tendency (no pressure).
  void rhs(const DV<Real>& uf, const DV<Real>& out) {
    if (viscousDEC) {  // convection (+ force) through Rᵀ, viscous term directly on the faces
      perotVelocity(m, uf, U);
      cellTendency(m, uf, U, Real(0), a, force, convScale, Uwall);
      perotTranspose(m, a, out);
      decLaplacian(m, dec, uf, decDiv, decOut);
      const DV<Real> dO = decOut;
      const Real nuL = nu;
      Kokkos::parallel_for(
          "ns.decvisc", Kokkos::RangePolicy<Exec>(0, m.nInterior),
          KOKKOS_LAMBDA(const int f) { out(f) += nuL * dO(f); });
      return;
    }
    perotVelocity(m, uf, U);
    if (wallQuadratic && m.nFaces > m.nInterior) {
      if (wg.extent(0) == 0) {
        wg = DV<Real>("ns.wg", 3 * (m.nFaces - m.nInterior));
        g9w = DV<Real>("ns.g9w", 9 * m.nCells);
      }
      vectorGreenGauss(m, U, g9w, Uwall);
      wallGradientLS(m, U, g9w, Uwall, wg);
      cellTendency(m, uf, U, nu, a, force, convScale, Uwall, wg);
    } else {
      cellTendency(m, uf, U, nu, a, force, convScale, Uwall);
    }
    perotTranspose(m, a, out);
  }
  /// Make uf divergence-free: L φ = div uf / dt, uf −= dt grad φ; p = φ.
  void project(const DV<Real>& uf, Real dt) {
    const DV<Real> dv = div, g = gphi;
    divergence(m, uf, dv);
    const Real s = -Real(1) / dt;
    Kokkos::parallel_for(
        "ns.rhsP", Kokkos::RangePolicy<Exec>(0, m.nCells),
        KOKKOS_LAMBDA(const int i) { dv(i) *= s; });
    poisson.solve(dv, p);
    faceGradient(m, p, g);
    Kokkos::parallel_for(
        "ns.corr", Kokkos::RangePolicy<Exec>(0, m.nFaces),
        KOKKOS_LAMBDA(const int f) { uf(f) -= dt * g(f); });
  }
  /// One SSP-RK3 step with a projection after every stage.
  void step(Real dt) {
    const int F = m.nFaces;
    const DV<Real> uu = u, kk = k, v1 = u1, v2 = u2;
    rhs(uu, kk);
    Kokkos::parallel_for(
        "ns.s1", Kokkos::RangePolicy<Exec>(0, F),
        KOKKOS_LAMBDA(const int f) { v1(f) = uu(f) + dt * kk(f); });
    project(v1, dt);
    rhs(v1, kk);
    Kokkos::parallel_for(
        "ns.s2", Kokkos::RangePolicy<Exec>(0, F), KOKKOS_LAMBDA(const int f) {
          v2(f) = Real(0.75) * uu(f) + Real(0.25) * (v1(f) + dt * kk(f));
        });
    project(v2, Real(0.25) * dt);
    rhs(v2, kk);
    Kokkos::parallel_for(
        "ns.s3", Kokkos::RangePolicy<Exec>(0, F),
        KOKKOS_LAMBDA(const int f) { uu(f) = (uu(f) + Real(2) * (v2(f) + dt * kk(f))) / Real(3); });
    project(uu, Real(2) / Real(3) * dt);
  }
  /// ½⟨u,u⟩_F — the energy the scheme conserves.
  Real kineticEnergy() const { return Real(0.5) * dotFaces(m, u, u); }
  /// max_i |div u|_i
  Real maxDivergence() {
    const DV<Real> dv = div;
    divergence(m, u, dv);
    Real mx = 0;
    Kokkos::parallel_reduce(
        "ns.maxdiv", Kokkos::RangePolicy<Exec>(0, m.nCells),
        KOKKOS_LAMBDA(const int i, Real& acc) {
          const Real v = Kokkos::fabs(dv(i));
          if (v > acc)
            acc = v;
        },
        Kokkos::Max<Real>(mx));
    return mx;
  }
};

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_COVOLUME_HPP
