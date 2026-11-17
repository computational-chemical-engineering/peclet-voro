/**
 * @file test_ale_operators.cpp
 * \brief Gate G1.1 of the moving-grid work order S1.1 (docs/ale_design.md §7 WO-S1.1, §11 A7, A9):
 * the bulk operators of `ale/operators.hpp` on frames of `ale::initialFrame` (power engine,
 * periodic unit box). Criteria fixed 2026-11-15 in the brief ALE_S1_1; "relative" is the max over
 * entries of |a − b| over the max over entries of |b|.
 *
 * Default run (the ctest): the identities that need no reference.
 *  K2 adjointness on the 8³ meshes M1, M2, M3 of the G0.3 export (same seeds), random fields:
 *     ⟨I_F U, φ⟩ − ⟨U, I_Fᵀφ⟩, ⟨C′U, p⟩ − ⟨U, C′ᵀp⟩, ⟨V, ζU⟩ − ⟨U, ζV⟩ (μ_f random in [1, 100]),
 *     ⟨x, L_ρ y⟩ − ⟨y, L_ρ x⟩ (density ratio 1000) ≤ 1e-14 of the smaller product of the norms.
 *  K3 linear and rigid exactness on 16³ meshes of type M1 and M2, U = a + Bx as cell values, on
 *     the cells and faces whose stencil does not cross the periodic wrap (faces with raw
 *     x_B − x_A ≠ conn, their cells and two further rings are excluded; ≥ 100 cells, ≥ 500
 *     faces): G_i = B, (I_F U)_f = A_f n_f·U(c_f) ≤ 1e-13 relative; trace-free B: C′U = 0 to
 *     1e-13 of max A_f |B| h; U = a + Ω × x: ζU = 0 to 1e-13 of max |ζW|, W random of equal norm.
 *  K4 V0.3 on the 16³ cubic lattice (zero weights): fluxMap/A_f against fv::projectToFaces,
 *     pressureForce against −V_i fv::faceInterpTranspose(fv::faceGradient(p)), divergence/V_i
 *     against fv::divergence(Φ/A_f), each ≤ 1e-14 relative; the faces other than the six lattice
 *     faces per cell are counted (and their largest area).
 *  K5 static body force (A9) on M1, M2, M3: (H1) ψ = cos(2πy/L), ρ = 1000, p_i = −ρψ(x_i):
 *     max_i |C′ᵀp + I_Fᵀγ|_i ≤ 1e-13 max_i |I_Fᵀγ|_i; (H2) bodyForceFaceUniform, g = (0.3, −1,
 *     0.2), ρ = 1000: Σ_i (I_Fᵀγ)_i = ρ(Σ_i V_i)g to 1e-13 relative; (H3) bodyForceCell = ρ_iV_iḡ
 *     to 1 ulp (ρ random in [1, 1000]).
 *  K6 L_ρ on M1, M2, M3 at ρ = 1 and ratio 1000: |row sum| ≤ 1e-13 diag; applyLaplacian against
 *     divergence(−a ⊙ faceDifference(x)) with a_f from the host formula ≤ 1e-14 relative.
 *
 *  test_ale_operators --apply DIR [--out ODIR] [--mesh M1|M2|M3|all]
 * K1, device side: for each mesh M exported by `test_ale_frame --export DIR`, rebuilds the frame
 * from `M.pos` / `M.weight` / `M.L`, asserts that every exported FaceMesh array is bitwise the
 * rebuilt one, reads the inputs `M.in_<name>` written by tests/python/test_ale_operators.py,
 * applies every kernel and writes `ODIR/M.out_<name>` (raw little-endian float64).
 *
 *  test_ale_operators --time N   wall time per kernel call on an N³ mesh of type M2 (no gate).
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/ale/frame.hpp"
#include "peclet/voro/ale/operators.hpp"
#include "peclet/voro/fv/operators.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/repair.hpp"

using Exec = peclet::core::ExecSpace;
constexpr int kP = peclet::voro::kMaxPlanes;
constexpr int kT = peclet::voro::kMaxTriangles;
using Engine = peclet::voro::MovingTessellation<Real, kP, kT, true>;
using FM = fv::FaceMesh<Real>;
using Frame = peclet::voro::ale::MeshFrame<Real>;
using FrameOpts = peclet::voro::ale::FrameOptions<Real>;
namespace ale = peclet::voro::ale;

namespace {

constexpr double kTwoPi = 6.283185307179586;

// ---- scenes: the generator of test_ale_frame's export (same seeds, same draw order) ----------
struct Scene {
  int n = 0, N = 0;
  Real L[3] = {1, 1, 1};
  Real h = 0;
  DV pos, weight;
};

Scene exportScene(int n, Real jitter) {
  Scene s;
  s.n = n;
  s.N = n * n * n;
  s.h = Real(1) / n;
  std::mt19937 rng(20261114u);
  std::uniform_real_distribution<Real> U(-1, 1), U01(0, 1);
  std::vector<Real> x(3 * s.N), w(s.N);
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const int c = i + n * (j + n * k);
        const int ijk[3] = {i, j, k};
        for (int d = 0; d < 3; ++d) {
          Real p = (ijk[d] + Real(0.5)) * s.h + jitter * s.h * U(rng);
          p = p < 0 ? p + 1 : (p >= 1 ? p - 1 : p);
          x[3 * c + d] = p;
        }
      }
  for (auto& v : w)
    v = Real(0.15) * s.h * s.h * U01(rng);
  s.pos = up(x, "pos");
  s.weight = up(w, "w");
  return s;
}

/// A cold frame of the power engine (zero weights included), as test_ale_frame's coldFrame.
Frame coldFrame(const Scene& s, const DV& pos, const DV& weight) {
  Engine mt;
  mt.alloc(s.N, s.L, Real(1e-4) * s.h, Real(0.25) * s.h, 4, s.N);
  mt.setWeights(weight);
  return ale::initialFrame(mt, pos, weight, s.L, FrameOpts{});
}

/// Mesh `M1` (jitter 0.3, zero weights), `M2` (the same seeds, weights) or `M3` (jitter 0.45).
struct Case {
  std::string name;
  Scene s;
  Frame fr;
};
Case makeCase(const std::string& type, int n) {
  Case c;
  c.name = type;
  c.s = exportScene(n, Real(type == "M3" ? 0.45 : (type == "LAT" ? 0.0 : 0.3)));
  const DV w = type == "M2" ? c.s.weight : DV("w0", c.s.N);
  c.fr = coldFrame(c.s, c.s.pos, w);
  if (c.fr.numEmptyCells > 0) {
    std::printf("SCENE %s n=%d FAIL numEmptyCells=%d\n", type.c_str(), n, c.fr.numEmptyCells);
    std::exit(1);
  }
  return c;
}

// ---- host helpers ------------------------------------------------------------------------------
std::vector<Real> randVec(std::size_t n, std::mt19937& rng, Real lo = -1, Real hi = 1) {
  std::uniform_real_distribution<Real> U(lo, hi);
  std::vector<Real> v(n);
  for (auto& x : v)
    x = U(rng);
  return v;
}
double dot(const std::vector<Real>& a, const std::vector<Real>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  return s;
}
double norm(const std::vector<Real>& a) {
  return std::sqrt(dot(a, a));
}
double maxAbs(const std::vector<Real>& a) {
  double m = 0;
  for (Real x : a)
    m = std::max(m, std::fabs((double)x));
  return m;
}
/// max |a − b| / max |b|
double relMax(const std::vector<Real>& a, const std::vector<Real>& b) {
  double d = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::fabs((double)a[i] - (double)b[i]));
  const double r = maxAbs(b);
  return r > 0 ? d / r : d;
}
std::vector<Real> sphereRho(const Scene& s, const std::vector<Real>& x) {
  std::vector<Real> rho(s.N);
  for (int i = 0; i < s.N; ++i) {
    double r2 = 0;
    for (int c = 0; c < 3; ++c) {
      const double d = x[3 * i + c] - 0.5 * s.L[c];
      r2 += d * d;
    }
    rho[i] = std::sqrt(r2) < 0.25 * s.L[0] ? Real(1) : Real(1000);
  }
  return rho;
}
const char* verdict(bool ok) {
  return ok ? "PASS" : "FAIL";
}

/// Scratch for the kernels of one mesh.
struct Work {
  DV G, sF, s9, sF12;
  explicit Work(const FM& m)
      : G("G", 9 * m.nCombined)
      , sF("sF", m.nFaces)
      , s9("s9", 9 * m.nCombined)
      , sF12("sF12", 12 * m.nFaces) {}
};

std::vector<Real> fluxMapOf(const FM& m, Work& w, const std::vector<Real>& U) {
  DV dU = up(U, "U"), out("IF", m.nFaces);
  ale::cellGradient(m, dU, w.G);
  ale::fluxMap(m, dU, w.G, out);
  return down(out);
}
std::vector<Real> fluxMapTOf(const FM& m, Work& w, const std::vector<Real>& phi) {
  DV out("IFt", 3 * m.nCombined);
  ale::fluxMapTranspose(m, up(phi, "phi"), out, w.s9);
  return down(out);
}
std::vector<Real> constraintOf(const FM& m, Work& w, const std::vector<Real>& U) {
  DV dU = up(U, "U"), out("Cp", m.nCombined);
  ale::cellGradient(m, dU, w.G);
  ale::constraint(m, dU, w.G, w.sF, out);
  return down(out);
}
std::vector<Real> pressureForceOf(const FM& m, Work& w, const std::vector<Real>& p) {
  DV out("CpT", 3 * m.nCombined);
  ale::pressureForce(m, up(p, "p"), out, w.sF, w.s9);
  return down(out);
}
std::vector<Real> frictionOf(const FM& m, Work& w, const std::vector<Real>& U,
                             const std::vector<Real>& mu) {
  DV dU = up(U, "U"), out("zeta", 3 * m.nCombined);
  ale::cellGradient(m, dU, w.G);
  ale::friction(m, dU, w.G, up(mu, "mu"), out, w.sF12, w.s9);
  return down(out);
}
std::vector<Real> gradientOf(const FM& m, Work& w, const std::vector<Real>& U) {
  ale::cellGradient(m, up(U, "U"), w.G);
  return down(w.G);
}

/// The host arrays of a mesh used by the checks.
struct HostMesh {
  int N = 0, F = 0;
  std::vector<int> A, B, off, cf;
  std::vector<Real> area, n, d, ha, hb, cA, conn, V, sign;
  explicit HostMesh(const FM& m)
      : N(m.nCells)
      , F(m.nFaces)
      , A(down(m.faceCellA))
      , B(down(m.faceCellB))
      , off(down(m.cellFaceOffset))
      , cf(down(m.cellFace))
      , area(down(m.faceArea))
      , n(down(m.faceNormal))
      , d(down(m.faceDist))
      , ha(down(m.faceHa))
      , hb(down(m.faceHb))
      , cA(down(m.faceCentroid))
      , conn(down(m.faceConn))
      , V(down(m.cellVolume))
      , sign(down(m.cellFaceSign)) {}
};

// ---- K2 ----------------------------------------------------------------------------------------
double adjRel(double lhs, double rhs, double p1, double p2) {
  return std::fabs(lhs - rhs) / std::min(p1, p2);
}

double checkK2(const Case& c) {
  const FM& m = c.fr.mesh;
  Work w(m);
  std::mt19937 rng(7001u);
  const int N = m.nCombined, F = m.nFaces;
  const auto U = randVec(3 * N, rng), V2 = randVec(3 * N, rng), phi = randVec(F, rng),
             p = randVec(N, rng), x = randVec(N, rng), y = randVec(N, rng),
             mu = randVec(F, rng, 1, 100);
  const auto IFU = fluxMapOf(m, w, U), IFTphi = fluxMapTOf(m, w, phi);
  const double e1 =
      adjRel(dot(IFU, phi), dot(U, IFTphi), norm(IFU) * norm(phi), norm(U) * norm(IFTphi));
  const auto CpU = constraintOf(m, w, U), CpTp = pressureForceOf(m, w, p);
  const double e2 = adjRel(dot(CpU, p), dot(U, CpTp), norm(CpU) * norm(p), norm(U) * norm(CpTp));
  const auto zU = frictionOf(m, w, U, mu), zV = frictionOf(m, w, V2, mu);
  const double e3 = adjRel(dot(V2, zU), dot(U, zV), norm(V2) * norm(zU), norm(U) * norm(zV));
  ale::LaplacianMatrix<Real> L;
  ale::assembleLaplacian(m, up(sphereRho(c.s, down(c.s.pos)), "rho"), L);
  DV Lx("Lx", N), Ly("Ly", N);
  ale::applyLaplacian(L, up(x, "x"), Lx);
  ale::applyLaplacian(L, up(y, "y"), Ly);
  const auto hLx = down(Lx), hLy = down(Ly);
  const double e4 = adjRel(dot(x, hLy), dot(y, hLx), norm(x) * norm(hLy), norm(y) * norm(hLx));
  const double e = std::max({e1, e2, e3, e4});
  std::printf("K2 %s %s IF=%.2e Cp=%.2e zeta=%.2e Lrho=%.2e\n", c.name.c_str(), verdict(e <= 1e-14),
              e1, e2, e3, e4);
  return e;
}

// ---- K3 ----------------------------------------------------------------------------------------
struct K3Result {
  double eG = 0, eIF = 0, eC = 0, eZ = 0;
  int cells = 0, faces = 0;
};

K3Result checkK3(const Case& c) {
  const FM& m = c.fr.mesh;
  const HostMesh hm(m);
  const auto x = down(c.s.pos);
  const int N = hm.N, F = hm.F;
  // wrapped faces, their cells, then two rings of neighbours
  std::vector<int> mark(N, 0);
  for (int f = 0; f < F; ++f) {
    bool wrapped = false;
    for (int k = 0; k < 3; ++k)
      wrapped |= (x[3 * hm.B[f] + k] - x[3 * hm.A[f] + k]) != hm.conn[3 * f + k];
    if (wrapped)
      mark[hm.A[f]] = mark[hm.B[f]] = 1;
  }
  for (int ring = 0; ring < 2; ++ring) {
    std::vector<int> next = mark;
    for (int f = 0; f < F; ++f)
      if (mark[hm.A[f]] || mark[hm.B[f]])
        next[hm.A[f]] = next[hm.B[f]] = 1;
    mark = next;
  }
  K3Result r;
  std::vector<int> useCell, useFace;
  for (int i = 0; i < N; ++i)
    if (!mark[i])
      useCell.push_back(i);
  for (int f = 0; f < F; ++f)
    if (!mark[hm.A[f]] && !mark[hm.B[f]])
      useFace.push_back(f);
  r.cells = (int)useCell.size();
  r.faces = (int)useFace.size();

  const Real a[3] = {0.3, -0.7, 1.1};
  const Real Bm[9] = {0.8, -0.3, 0.5, 0.2, -1.1, 0.7, -0.6, 0.4, 0.9};
  const Real tr = (Bm[0] + Bm[4] + Bm[8]) / 3;
  Real B0[9];
  for (int j = 0; j < 9; ++j)
    B0[j] = Bm[j] - ((j % 4 == 0) ? tr : Real(0));
  const Real Om[3] = {0.4, -0.9, 0.6};
  auto linear = [&](const Real* Bx, const Real* xx, Real* out) {
    for (int ci = 0; ci < 3; ++ci)
      out[ci] = a[ci] + Bx[3 * ci] * xx[0] + Bx[3 * ci + 1] * xx[1] + Bx[3 * ci + 2] * xx[2];
  };
  std::vector<Real> U(3 * N), U0(3 * N), Ur(3 * N);
  for (int i = 0; i < N; ++i) {
    const Real* xi = &x[3 * i];
    linear(Bm, xi, &U[3 * i]);
    linear(B0, xi, &U0[3 * i]);
    Ur[3 * i] = a[0] + Om[1] * xi[2] - Om[2] * xi[1];
    Ur[3 * i + 1] = a[1] + Om[2] * xi[0] - Om[0] * xi[2];
    Ur[3 * i + 2] = a[2] + Om[0] * xi[1] - Om[1] * xi[0];
  }
  Work w(m);
  const auto G = gradientOf(m, w, U);
  double dG = 0;
  for (int i : useCell)
    for (int j = 0; j < 9; ++j)
      dG = std::max(dG, std::fabs((double)G[9 * i + j] - Bm[j]));
  r.eG = dG / maxAbs(std::vector<Real>(Bm, Bm + 9));

  const auto IF = fluxMapOf(m, w, U);
  double dIF = 0, rIF = 0;
  for (int f : useFace) {
    Real xc[3], uc[3];
    for (int k = 0; k < 3; ++k)
      xc[k] = x[3 * hm.A[f] + k] + hm.cA[3 * f + k];
    linear(Bm, xc, uc);
    double ref = 0;
    for (int k = 0; k < 3; ++k)
      ref += hm.n[3 * f + k] * uc[k];
    ref *= hm.area[f];
    dIF = std::max(dIF, std::fabs(IF[f] - ref));
    rIF = std::max(rIF, std::fabs(ref));
  }
  r.eIF = dIF / rIF;

  const auto Cp = constraintOf(m, w, U0);
  double dC = 0;
  for (int i : useCell)
    dC = std::max(dC, std::fabs((double)Cp[i]));
  r.eC = dC / (maxAbs(hm.area) * maxAbs(std::vector<Real>(B0, B0 + 9)) * c.s.h);

  const std::vector<Real> mu(F, Real(1));
  const auto zr = frictionOf(m, w, Ur, mu);
  std::mt19937 rng(7003u);
  auto W = randVec(3 * N, rng);
  const double sc = norm(Ur) / norm(W);
  for (auto& v : W)
    v *= sc;
  const auto zW = frictionOf(m, w, W, mu);
  double dZ = 0, rZ = 0;
  for (int i : useCell)
    for (int k = 0; k < 3; ++k)
      dZ = std::max(dZ, std::fabs((double)zr[3 * i + k]));
  for (int i = 0; i < N; ++i)
    rZ = std::max(rZ, std::sqrt(zW[3 * i] * zW[3 * i] + zW[3 * i + 1] * zW[3 * i + 1] +
                                zW[3 * i + 2] * zW[3 * i + 2]));
  r.eZ = dZ / rZ;
  const bool ok = r.eG <= 1e-13 && r.eIF <= 1e-13 && r.eC <= 1e-13 && r.eZ <= 1e-13 &&
                  r.cells >= 100 && r.faces >= 500;
  std::printf("K3 %s-type n=%d %s G=%.2e IF=%.2e Cp=%.2e zetaRigid=%.2e cells=%d/%d faces=%d/%d\n",
              c.name.c_str(), c.s.n, verdict(ok), r.eG, r.eIF, r.eC, r.eZ, r.cells, N, r.faces, F);
  return r;
}

// ---- K4 ----------------------------------------------------------------------------------------
struct K4Result {
  double eIF = 0, eP = 0, eD = 0, maxExtraArea = 0;
  int extra = 0;
};

K4Result checkK4(const Case& c) {
  const FM& m = c.fr.mesh;
  const HostMesh hm(m);
  const int N = m.nCombined, F = m.nFaces;
  K4Result r;
  for (int f = 0; f < F; ++f) {
    int nz = 0;
    for (int k = 0; k < 3; ++k)
      nz += std::fabs(hm.conn[3 * f + k]) > 1e-9 * c.s.h ? 1 : 0;
    if (nz != 1) {
      ++r.extra;
      r.maxExtraArea = std::max(r.maxExtraArea, (double)hm.area[f]);
    }
  }
  std::mt19937 rng(7004u);
  const auto U = randVec(3 * N, rng), p = randVec(N, rng), Phi = randVec(F, rng);
  Work w(m);
  // fluxMap / A_f against projectToFaces
  auto IF = fluxMapOf(m, w, U);
  for (int f = 0; f < F; ++f)
    IF[f] /= hm.area[f];
  DV proj("proj", F);
  fv::projectToFaces(m, up(U, "U"), proj);
  r.eIF = relMax(IF, down(proj));
  // pressureForce against −V_i faceInterpTranspose(faceGradient(p))
  const auto Fp = pressureForceOf(m, w, p);
  DV gface("gface", F), cg("cg", 3 * N);
  fv::faceGradient(m, up(p, "p"), gface);
  fv::faceInterpTranspose(m, gface, cg);
  auto ref = down(cg);
  for (int i = 0; i < N; ++i)
    for (int k = 0; k < 3; ++k)
      ref[3 * i + k] *= -hm.V[i];
  r.eP = relMax(Fp, ref);
  // divergence / V_i against fv::divergence(Φ/A_f)
  DV dv("div", N), dfv("dfv", N);
  ale::divergence(m, up(Phi, "Phi"), dv);
  auto PhiA = Phi;
  for (int f = 0; f < F; ++f)
    PhiA[f] /= hm.area[f];
  fv::divergence(m, up(PhiA, "PhiA"), dfv);
  auto D = down(dv);
  for (int i = 0; i < N; ++i)
    D[i] /= hm.V[i];
  r.eD = relMax(D, down(dfv));
  const bool ok = r.eIF <= 1e-14 && r.eP <= 1e-14 && r.eD <= 1e-14;
  std::printf(
      "K4 lattice n=%d %s IF/A=%.2e CpT=%.2e div/V=%.2e faces=%d extraFaces=%d "
      "maxExtraArea=%.2e(h^2=%.2e)\n",
      c.s.n, verdict(ok), r.eIF, r.eP, r.eD, F, r.extra, r.maxExtraArea, (double)(c.s.h * c.s.h));
  return r;
}

// ---- K5 ----------------------------------------------------------------------------------------
struct K5Result {
  double h1 = 0, h2 = 0;
  long ulp = 0;
};

long ulpDistance(double a, double b) {
  if (a == b)
    return 0;
  std::int64_t ia, ib;
  std::memcpy(&ia, &a, 8);
  std::memcpy(&ib, &b, 8);
  if ((ia < 0) != (ib < 0))
    return std::numeric_limits<long>::max();
  return (long)std::llabs(ia - ib);
}

K5Result checkK5(const Case& c) {
  const FM& m = c.fr.mesh;
  const HostMesh hm(m);
  const int N = m.nCombined, F = m.nFaces;
  const auto x = down(c.s.pos);
  const Real k = Real(kTwoPi) / c.s.L[1];
  const Real rho0 = 1000;
  std::vector<Real> psiC(N), psiF(F), p(N), rhoU(N, rho0);
  for (int i = 0; i < N; ++i) {
    psiC[i] = std::cos(k * x[3 * i + 1]);
    p[i] = -rho0 * psiC[i];
  }
  for (int f = 0; f < F; ++f)
    psiF[f] = std::cos(k * (x[3 * hm.A[f] + 1] + hm.cA[3 * f + 1]));
  Work w(m);
  K5Result r;
  // H1
  DV gamma("gamma", F);
  const DV dRho = up(rhoU, "rho");
  ale::bodyForceFace(m, dRho, up(psiC, "psiC"), up(psiF, "psiF"), gamma);
  const auto Fp = pressureForceOf(m, w, p);
  const auto Fg = fluxMapTOf(m, w, down(gamma));
  double num = 0, den = 0;
  for (int i = 0; i < N; ++i) {
    double s2 = 0, g2 = 0;
    for (int q = 0; q < 3; ++q) {
      const double s = Fp[3 * i + q] + Fg[3 * i + q];
      s2 += s * s;
      g2 += (double)Fg[3 * i + q] * Fg[3 * i + q];
    }
    num = std::max(num, std::sqrt(s2));
    den = std::max(den, std::sqrt(g2));
  }
  r.h1 = num / den;
  // H2
  const Real g[3] = {0.3, -1, 0.2};
  ale::bodyForceFaceUniform(m, dRho, g, gamma);
  const auto Fu = fluxMapTOf(m, w, down(gamma));
  double vs = 0;
  for (int i = 0; i < N; ++i)
    vs += hm.V[i];
  std::vector<Real> sum(3, 0), ref(3);
  for (int i = 0; i < N; ++i)
    for (int q = 0; q < 3; ++q)
      sum[q] += Fu[3 * i + q];
  for (int q = 0; q < 3; ++q)
    ref[q] = rho0 * vs * g[q];
  r.h2 = relMax(sum, ref);
  // H3
  std::mt19937 rng(7005u);
  const auto rho = randVec(N, rng, 1, 1000);
  DV out("bodyCell", 3 * N);
  ale::bodyForceCell(m, up(rho, "rho"), g, out);
  const auto hb = down(out);
  for (int i = 0; i < N; ++i)
    for (int q = 0; q < 3; ++q) {
      const double expect = (double)rho[i] * hm.V[i] * g[q];
      r.ulp = std::max(r.ulp, ulpDistance(hb[3 * i + q], expect));
    }
  const bool ok = r.h1 <= 1e-13 && r.h2 <= 1e-13 && r.ulp <= 1;
  std::printf("K5 %s %s H1=%.2e H2=%.2e (sum=%.15e,%.15e,%.15e) H3ulp=%ld\n", c.name.c_str(),
              verdict(ok), r.h1, r.h2, (double)sum[0], (double)sum[1], (double)sum[2], r.ulp);
  return r;
}

// ---- K6 ----------------------------------------------------------------------------------------
struct K6Result {
  double rowsum = 0, apply = 0;
};

K6Result checkK6(const Case& c) {
  const FM& m = c.fr.mesh;
  const HostMesh hm(m);
  const int N = m.nCombined, F = m.nFaces;
  K6Result r;
  std::mt19937 rng(7006u);
  const auto x = randVec(N, rng);
  const std::vector<Real> rhoSets[2] = {std::vector<Real>(N, Real(1)),
                                        sphereRho(c.s, down(c.s.pos))};
  const char* tags[2] = {"ratio1", "ratio1000"};
  for (int t = 0; t < 2; ++t) {
    const auto& rho = rhoSets[t];
    ale::LaplacianMatrix<Real> L;
    ale::assembleLaplacian(m, up(rho, "rho"), L);
    const auto diag = down(L.diag), coef = down(L.coef);
    double rs = 0;
    for (int i = 0; i < N; ++i) {
      double s = diag[i];
      for (int q = hm.off[i]; q < hm.off[i + 1]; ++q)
        s += coef[q];
      rs = std::max(rs, std::fabs(s) / diag[i]);
    }
    DV Lx("Lx", N), dn("dn", F), comp("comp", N);
    ale::applyLaplacian(L, up(x, "x"), Lx);
    ale::faceDifference(m, up(x, "x"), dn);
    auto hdn = down(dn);
    for (int f = 0; f < F; ++f) {
      const double a =
          hm.area[f] * ((hm.hb[f] / rho[hm.A[f]] + hm.ha[f] / rho[hm.B[f]]) / hm.d[f]) / hm.d[f];
      hdn[f] = -a * hdn[f];
    }
    ale::divergence(m, up(hdn, "aDn"), comp);
    const double ap = relMax(down(Lx), down(comp));
    std::printf("K6 %s %s %s rowsum/diag=%.2e apply=%.2e\n", c.name.c_str(), tags[t],
                verdict(rs <= 1e-13 && ap <= 1e-14), rs, ap);
    r.rowsum = std::max(r.rowsum, rs);
    r.apply = std::max(r.apply, ap);
  }
  return r;
}

int runIdentities() {
  int rc = 0;
  std::vector<Case> cases;
  for (const char* t : {"M1", "M2", "M3"})
    cases.push_back(makeCase(t, 8));
  double k2 = 0;
  for (const Case& c : cases)
    k2 = std::max(k2, checkK2(c));
  const bool ok2 = k2 <= 1e-14;
  std::printf("K2 %s max=%.2e\n", verdict(ok2), k2);
  rc |= !ok2;

  K3Result k3;
  int minCells = 1 << 30, minFaces = 1 << 30;
  for (const char* t : {"M1", "M2"}) {
    const K3Result r = checkK3(makeCase(t, 16));
    k3.eG = std::max(k3.eG, r.eG);
    k3.eIF = std::max(k3.eIF, r.eIF);
    k3.eC = std::max(k3.eC, r.eC);
    k3.eZ = std::max(k3.eZ, r.eZ);
    minCells = std::min(minCells, r.cells);
    minFaces = std::min(minFaces, r.faces);
  }
  const bool ok3 = k3.eG <= 1e-13 && k3.eIF <= 1e-13 && k3.eC <= 1e-13 && k3.eZ <= 1e-13 &&
                   minCells >= 100 && minFaces >= 500;
  std::printf("K3 %s G=%.2e IF=%.2e Cp=%.2e zetaRigid=%.2e minCells=%d minFaces=%d\n", verdict(ok3),
              k3.eG, k3.eIF, k3.eC, k3.eZ, minCells, minFaces);
  rc |= !ok3;

  const K4Result k4 = checkK4(makeCase("LAT", 16));
  const bool ok4 = k4.eIF <= 1e-14 && k4.eP <= 1e-14 && k4.eD <= 1e-14;
  std::printf("K4 %s IF/A=%.2e CpT=%.2e div/V=%.2e extraFaces=%d maxExtraArea=%.2e\n", verdict(ok4),
              k4.eIF, k4.eP, k4.eD, k4.extra, k4.maxExtraArea);
  rc |= !ok4;

  K5Result k5;
  for (const Case& c : cases) {
    const K5Result r = checkK5(c);
    k5.h1 = std::max(k5.h1, r.h1);
    k5.h2 = std::max(k5.h2, r.h2);
    k5.ulp = std::max(k5.ulp, r.ulp);
  }
  const bool ok5 = k5.h1 <= 1e-13 && k5.h2 <= 1e-13 && k5.ulp <= 1;
  std::printf("K5 %s H1=%.2e H2=%.2e H3ulp=%ld\n", verdict(ok5), k5.h1, k5.h2, k5.ulp);
  rc |= !ok5;

  K6Result k6;
  for (const Case& c : cases) {
    const K6Result r = checkK6(c);
    k6.rowsum = std::max(k6.rowsum, r.rowsum);
    k6.apply = std::max(k6.apply, r.apply);
  }
  const bool ok6 = k6.rowsum <= 1e-13 && k6.apply <= 1e-14;
  std::printf("K6 %s rowsum/diag=%.2e apply=%.2e\n", verdict(ok6), k6.rowsum, k6.apply);
  rc |= !ok6;
  return rc;
}

// ---- --apply DIR: the device side of K1 --------------------------------------------------------
struct Arr {
  std::string name;
  std::vector<char> bytes;
};
template <class V>
Arr arrOf(const char* name, const V& v) {
  using T = typename V::non_const_value_type;
  Arr a;
  a.name = name;
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  const char* p = reinterpret_cast<const char*>(hv.data());
  a.bytes.assign(p, p + v.extent(0) * sizeof(T));
  return a;
}
/// The arrays test_ale_frame's export writes for a FaceMesh (its meshArrays), in the same order.
std::vector<Arr> meshArrays(const FM& m) {
  std::vector<Arr> out;
  Arr c;
  c.name = "counts";
  const long counts[6] = {m.nCells,    m.nFaces,   m.nInterior,
                          m.nCombined, m.nDropped, m.generatorDiag.numNegativeHeight};
  const Real sine = m.generatorDiag.maxNormalConnSine;
  c.bytes.assign(reinterpret_cast<const char*>(counts),
                 reinterpret_cast<const char*>(counts) + sizeof(counts));
  c.bytes.insert(c.bytes.end(), reinterpret_cast<const char*>(&sine),
                 reinterpret_cast<const char*>(&sine) + sizeof(Real));
  out.push_back(std::move(c));
  out.push_back(arrOf("faceCellA", m.faceCellA));
  out.push_back(arrOf("faceCellB", m.faceCellB));
  out.push_back(arrOf("faceFacet", m.faceFacet));
  out.push_back(arrOf("faceArea", m.faceArea));
  out.push_back(arrOf("faceNormal", m.faceNormal));
  out.push_back(arrOf("faceDist", m.faceDist));
  out.push_back(arrOf("faceHa", m.faceHa));
  out.push_back(arrOf("faceHb", m.faceHb));
  out.push_back(arrOf("faceCentroid", m.faceCentroid));
  out.push_back(arrOf("faceConn", m.faceConn));
  out.push_back(arrOf("cellFaceOffset", m.cellFaceOffset));
  out.push_back(arrOf("cellFace", m.cellFace));
  out.push_back(arrOf("cellFaceSign", m.cellFaceSign));
  out.push_back(arrOf("cellVolume", m.cellVolume));
  out.push_back(arrOf("cellCentroid", m.cellCentroid));
  return out;
}

bool readFile(const std::string& path, std::vector<char>& bytes) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp)
    return false;
  std::fseek(fp, 0, SEEK_END);
  const long sz = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  bytes.resize(sz);
  const std::size_t got = sz > 0 ? std::fread(bytes.data(), 1, sz, fp) : 0;
  std::fclose(fp);
  return (long)got == sz;
}
template <class T>
std::vector<T> readVec(const std::string& path, std::size_t expect) {
  std::vector<char> b;
  if (!readFile(path, b) || b.size() != expect * sizeof(T)) {
    std::fprintf(stderr, "cannot read %s (%zu values expected)\n", path.c_str(), expect);
    std::exit(2);
  }
  std::vector<T> v(expect);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}
void writeVec(const std::string& path, const std::vector<Real>& v) {
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(2);
  }
  if (!v.empty())
    std::fwrite(v.data(), sizeof(Real), v.size(), fp);
  std::fclose(fp);
}

int applyMesh(const std::string& dir, const std::string& odir, const std::string& name) {
  const std::string p = dir + "/" + name + ".";
  std::vector<char> cb;
  if (!readFile(p + "counts", cb) || cb.size() < 6 * sizeof(long)) {
    std::printf("APPLY %s FAIL cannot read %scounts\n", name.c_str(), p.c_str());
    return 1;
  }
  long counts[6];
  std::memcpy(counts, cb.data(), sizeof(counts));
  const int N = (int)counts[0], F = (int)counts[1];
  Scene s;
  s.N = N;
  s.n = (int)std::lround(std::cbrt((double)N));
  const auto Lv = readVec<Real>(p + "L", 3);
  for (int k = 0; k < 3; ++k)
    s.L[k] = Lv[k];
  s.h = s.L[0] / s.n;
  s.pos = up(readVec<Real>(p + "pos", 3 * (std::size_t)N), "pos");
  s.weight = up(readVec<Real>(p + "weight", N), "w");
  const Frame fr = coldFrame(s, s.pos, s.weight);
  const FM& m = fr.mesh;
  for (const Arr& a : meshArrays(m)) {
    std::vector<char> b;
    if (!readFile(p + a.name, b) || b != a.bytes) {
      std::size_t i = 0;
      while (i < std::min(a.bytes.size(), b.size()) && a.bytes[i] == b[i])
        ++i;
      std::printf(
          "APPLY %s FAIL frame array %s differs from the export at byte %zu (sizes %zu/%zu)\n",
          name.c_str(), a.name.c_str(), i, a.bytes.size(), b.size());
      return 1;
    }
  }
  const int NC = m.nCombined;
  auto in = [&](const char* nm, std::size_t n) { return readVec<Real>(p + "in_" + nm, n); };
  const DV U = up(in("U", 3 * (std::size_t)NC), "U"), pr = up(in("p", NC), "p"),
           phi = up(in("phi", F), "phi"), xdot = up(in("xdot", 3 * (std::size_t)NC), "xdot"),
           wdot = up(in("wdot", NC), "wdot"), rhoSphere = up(in("rhoSphere", NC), "rhoS"),
           rhoRand = up(in("rhoRand", NC), "rhoR"), mu = up(in("mu", F), "mu"),
           psiC = up(in("psiCell", NC), "psiC"), psiF = up(in("psiFace", F), "psiF");
  const auto g = in("g", 3), gbar = in("gbar", 3);
  Work w(m);
  const std::string o = odir + "/" + name + ".out_";
  DV cellN("cN", NC), cell3("c3", 3 * NC), face("f", F);
  ale::divergence(m, phi, cellN);
  writeVec(o + "div", down(cellN));
  ale::faceDifference(m, pr, face);
  writeVec(o + "dn", down(face));
  ale::cellGradient(m, U, w.G);
  writeVec(o + "grad", down(w.G));
  ale::fluxMap(m, U, w.G, face);
  writeVec(o + "IF", down(face));
  ale::fluxMapTranspose(m, phi, cell3, w.s9);
  writeVec(o + "IFT", down(cell3));
  ale::constraint(m, U, w.G, w.sF, cellN);
  writeVec(o + "Cp", down(cellN));
  ale::pressureForce(m, pr, cell3, w.sF, w.s9);
  writeVec(o + "CpT", down(cell3));
  ale::LaplacianMatrix<Real> L;
  {
    DV one("one", NC);
    Kokkos::deep_copy(one, Real(1));
    ale::assembleLaplacian(m, one, L);
    writeVec(o + "Ldiag_1", down(L.diag));
    ale::applyLaplacian(L, pr, cellN);
    writeVec(o + "Lapply_1", down(cellN));
    ale::assembleLaplacian(m, rhoSphere, L);
    writeVec(o + "Ldiag_1000", down(L.diag));
    ale::applyLaplacian(L, pr, cellN);
    writeVec(o + "Lapply_1000", down(cellN));
  }
  ale::sweptFlux(m, xdot, wdot, face);
  writeVec(o + "swept", down(face));
  {
    DV one("one", F);
    Kokkos::deep_copy(one, Real(1));
    ale::friction(m, U, w.G, one, cell3, w.sF12, w.s9);
    writeVec(o + "zeta_1", down(cell3));
    ale::friction(m, U, w.G, mu, cell3, w.sF12, w.s9);
    writeVec(o + "zeta_mu", down(cell3));
  }
  ale::bodyForceFace(m, rhoRand, psiC, psiF, face);
  writeVec(o + "gamma", down(face));
  ale::bodyForceFaceUniform(m, rhoRand, g.data(), face);
  writeVec(o + "gammaU", down(face));
  ale::bodyForceCell(m, rhoRand, gbar.data(), cell3);
  writeVec(o + "bodyCell", down(cell3));
  std::printf("APPLY %s ok N=%d nFaces=%d frame=bitwise-equal-to-export\n", name.c_str(), N, F);
  return 0;
}

// ---- --time N ----------------------------------------------------------------------------------
int timeKernels(int n) {
  const Case c = makeCase("M2", n);
  const FM& m = c.fr.mesh;
  const int NC = m.nCombined, F = m.nFaces;
  std::mt19937 rng(7007u);
  const DV U = up(randVec(3 * NC, rng), "U"), p = up(randVec(NC, rng), "p"),
           phi = up(randVec(F, rng), "phi"), mu = up(randVec(F, rng, 1, 100), "mu"),
           rho = up(randVec(NC, rng, 1, 1000), "rho"), xdot = up(randVec(3 * NC, rng), "xd"),
           wdot = up(randVec(NC, rng), "wd"), psiC = up(randVec(NC, rng), "psiC"),
           psiF = up(randVec(F, rng), "psiF");
  Work w(m);
  DV cellN("cN", NC), cell3("c3", 3 * NC), face("f", F);
  ale::LaplacianMatrix<Real> L;
  const Real g[3] = {0.3, -1, 0.2};
  const int reps = 50;
  auto time = [&](auto&& fn) {
    fn();
    Kokkos::fence();
    Kokkos::Timer t;
    for (int r = 0; r < reps; ++r)
      fn();
    Kokkos::fence();
    return t.seconds() / reps * 1e6;
  };
  ale::cellGradient(m, U, w.G);
  // timed in this order (the Laplacian is assembled before it is applied)
  const char* names[] = {"div",  "dn",     "grad",  "IF",   "IFT",   "Cp",     "CpT",
                         "Lasm", "Lapply", "swept", "zeta", "gamma", "gammaU", "bodyCell"};
  double us[14];
  us[0] = time([&] { ale::divergence(m, phi, cellN); });
  us[1] = time([&] { ale::faceDifference(m, p, face); });
  us[2] = time([&] { ale::cellGradient(m, U, w.G); });
  us[3] = time([&] { ale::fluxMap(m, U, w.G, face); });
  us[4] = time([&] { ale::fluxMapTranspose(m, phi, cell3, w.s9); });
  us[5] = time([&] { ale::constraint(m, U, w.G, w.sF, cellN); });
  us[6] = time([&] { ale::pressureForce(m, p, cell3, w.sF, w.s9); });
  us[7] = time([&] { ale::assembleLaplacian(m, rho, L); });
  us[8] = time([&] { ale::applyLaplacian(L, p, cellN); });
  us[9] = time([&] { ale::sweptFlux(m, xdot, wdot, face); });
  us[10] = time([&] { ale::friction(m, U, w.G, mu, cell3, w.sF12, w.s9); });
  us[11] = time([&] { ale::bodyForceFace(m, rho, psiC, psiF, face); });
  us[12] = time([&] { ale::bodyForceFaceUniform(m, rho, g, face); });
  us[13] = time([&] { ale::bodyForceCell(m, rho, g, cell3); });
  std::string line;
  char buf[64];
  for (int k = 0; k < 14; ++k) {
    std::snprintf(buf, sizeof buf, " %s=%.1f", names[k], us[k]);
    line += buf;
  }
  std::printf("TIME n=%d N=%d F=%d exec=%s concurrency=%d us/call:%s\n", n, NC, F, Exec::name(),
              Exec().concurrency(), line.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int rc = 0;
  {
    std::string applyDir, outDir, which = "all";
    int timeN = 0;
    for (int a = 1; a < argc; ++a) {
      const std::string s = argv[a];
      if (s == "--apply" && a + 1 < argc)
        applyDir = argv[++a];
      else if (s == "--out" && a + 1 < argc)
        outDir = argv[++a];
      else if (s == "--mesh" && a + 1 < argc)
        which = argv[++a];
      else if (s == "--time" && a + 1 < argc)
        timeN = std::atoi(argv[++a]);
    }
    if (!applyDir.empty()) {
      if (outDir.empty())
        outDir = applyDir;
      for (const char* mesh : {"M1", "M2", "M3"})
        if (which == "all" || which == mesh)
          rc |= applyMesh(applyDir, outDir, mesh);
      std::printf("=== S1.1 apply %s ===\n", rc ? "FAIL" : "PASS");
    } else if (timeN > 0) {
      rc = timeKernels(timeN);
    } else {
      std::printf("=== S1.1 operator identities: threads=%d exec=%s ===\n", Exec().concurrency(),
                  Exec::name());
      rc = runIdentities();
      std::printf("=== S1.1 identities %s ===\n", rc ? "FAIL" : "PASS");
    }
  }
  Kokkos::finalize();
  return rc;
}
