/**
 * @file test_ale_stepper.cpp
 * \brief Gate G1.2 of the moving-grid work order S1.2b (docs/ale_design.md §4.1–§4.5, §4.9, §11 A7,
 * A15): the device time step of `ale/stepper.hpp` for one phase on a static mesh, its L_ρ solver
 * (`ale/laplacian_solver.hpp`) and its transport kernels (`ale/transport.hpp`), on frames of
 * `ale::initialFrame` (power engine, periodic unit box, the mesh family of test_ale_frame's export:
 * Z = M1, jitter ±0.3h and zero weights; W = M2, the same seeds with weights in [0, 0.15h²]).
 * "Relative" is the max over entries of |a − b| over the max over entries of |b|.
 *
 * Default run (the ctest), 8³ and 12³, no reference needed:
 *  S1 T skew (⟨V, TU⟩ + ⟨U, TV⟩), Q symmetric and positive semidefinite, (T + Q) of the fused
 *     kernel = T + Q, the explicit transport zero on a constant field: each ≤ 1e-14 relative.
 *  S2 LaplacianSolver: PCG to 1e-10 on L_ρ (ρ = 1 and the sphere at ratio 1000): converged,
 *     |L x − b| ≤ 1e-10 |b| recomputed, mean of x zero.
 *  S3 W 8³ at rest, modes F and R, 10 steps: U and p exactly 0.
 *  S4 Taylor–Green, Z 8³, mode F, 10 steps: |closure| ≤ 1e-13 K, momentum drift with the residual's
 *     momentum booked ≤ 1e-13 Σ M|U|, max τ|DΦ₁|/V ≤ 1e-14.
 *
 *  test_ale_stepper --apply DIR --cells N [--mesh M1|M2|M3|all] [--out ODIR]
 * C0, device side: for each mesh <M>_N exported by `test_ale_frame --export DIR --cells N`,
 * rebuilds the frame, asserts every exported array bitwise, reads `in_*` written by
 * tests/python/test_ale_stepper.py and writes `ODIR/<M>_N.out_<name>` (raw float64).
 *
 *  test_ale_stepper --laplacian N   gate L: M1 at N³, sphere R = L/4 at the box center, ρ = 1
 * inside and 1 or 1000 outside; LaplacianSolver setup (time, levels, level sizes, device memory
 * added by the setup: the high-water mark of the Kokkos allocations in the default memory space
 * during `setup`, from Kokkos Tools allocate/deallocate callbacks) and PCG to 1e-10 from zero.
 *
 *  test_ale_stepper --trace DIR CASE --out ODIR [--steps K]
 * C1–C3: reads the start state written by `ale_reference_stepper.py trace` (DIR/CASE.meta, .U0,
 * .p0), rebuilds the frame of its mesh from DIR, runs K steps at the tight tolerances (predictor
 * 1e-13, Laplacian 1e-12) and writes ODIR/CASE.dev_{Ustar,F,U,p}_k and ODIR/CASE.dev_scalars.
 *
 *  test_ale_stepper --run CASE [--steps K] [--courant C] [--default-tol] [--stokes]   the runs of
 * gate G1.2 R0–R6 and T (--stokes: the transport switched off, same τ and steps), at the default
 * tolerances of §4.9 (C: the flux Courant number of the time step, 0.4 by default): one RESULT line
 * with the keys of the reference's run_case (and a few more), timings per stage. Cases:
 * <tg|abc>_<Z|W><n>_<F|R>, r0_<zero|unif>_<F|R> (W 8³), r3_tg_W16_F (2000 steps), r3_tg_W32_F (500
 * steps), r6i_W16, r6ii_rest_W16, r6ii_tg_W16, unif_Z16_<F|R> (uniform flow, τ = 1/400, 400 steps).
 *  ... --motion rest|follow|prescribed [--tend T]: the same run on a MOVING mesh (ale/
 * moving_stepper.hpp, §11 A18, brief ALE_S1_3A; the default, no --motion, is the static path,
 * unchanged); `prescribed` is ẋ = 0.3 (sin 2πy, sin 2πz, sin 2πx). --tend T: lround(T/τ) steps
 * (τ by the static rule of the case, from the start field's Φ). `rest` runs the static path beside
 * it in the same process and compares U and p bitwise after every step (gate V0.2). Extra RESULT
 * keys at the end: motion, gcl (max over the run of max_i |τ(DΦᵐ_c)_i − ΔV_i|/V_i), s3_restarts
 * (max over the run of the S3 correction's re-solves for its true residual), rebuilt,
 * frames, t_frames, t_setup (per step), B1_absmax (|B1|/K), mom_def (the booked defect momentum
 * over Σ M|U|), v02_maxdiff, v02_bitwise_steps (rest only).
 *  ... --dump-on-error DIR (diagnostic, brief ALE_S1_3A_R): the frames record their rounds; on a
 * frame error the last frames' positions, the weights and the forced sets are written to DIR
 * (dumpFrameError; replay: test_ale_frame --replay DIR), with the open cells' closure terms, the
 * rounds continued to 8 and a cold build of the target printed. Numerics unchanged.
 *
 *  test_ale_stepper --trace-moving DIR CASE [--motion M] [--steps K]   gate T of G1.3a: K = 5
 * moving steps (motion follow by default) at the tight tolerances of --trace; per step the frames
 * Fh, F1, the inputs and outputs and a line of scalars in DIR (see traceStep), for
 * `test_ale_stepper.py trace-moving`.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/ale/frame.hpp"
#include "peclet/voro/ale/laplacian_solver.hpp"
#include "peclet/voro/ale/moving_stepper.hpp"
#include "peclet/voro/ale/operators.hpp"
#include "peclet/voro/ale/stepper.hpp"
#include "peclet/voro/ale/transport.hpp"
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
namespace kry = peclet::voro::ale::krylov;

namespace {

constexpr double kTwoPi = 6.283185307179586;
/// --exact-rtol X (diagnostic): the relative tolerance of mode R in --run / --trace-moving
/// (StepOptions::exactRtol; default: the StepOptions default).
double g_exactRtol = -1;
/// --pred-rtol X (diagnostic): the predictor's relative tolerance (overrides --tight's 1e-13).
double g_predRtol = -1;

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

Frame coldFrame(const Scene& s, const DV& pos, const DV& weight) {
  Engine mt;
  mt.alloc(s.N, s.L, Real(1e-4) * s.h, Real(0.25) * s.h, 4, s.N);
  mt.setWeights(weight);
  return ale::initialFrame(mt, pos, s.L, FrameOpts{});
}

/// Mesh family Z (= M1), W (= M2) or M3 at n³.
/// With `keepEngine` the engine of the frame is kept (the moving step advances it); the frame is
/// the same either way.
struct Case {
  std::string name;
  Scene s;
  Frame fr;
  std::unique_ptr<Engine> engine;
};
Case makeCase(const std::string& type, int n, bool keepEngine = false) {
  Case c;
  c.name = type;
  const bool w = type == "W" || type == "M2";
  c.s = exportScene(n, Real(type == "M3" ? 0.45 : 0.3));
  const DV wt = w ? c.s.weight : DV("w0", c.s.N);
  if (keepEngine) {
    c.engine = std::make_unique<Engine>();
    c.engine->alloc(c.s.N, c.s.L, Real(1e-4) * c.s.h, Real(0.25) * c.s.h, 4, c.s.N);
    c.engine->setWeights(wt);
    c.fr = ale::initialFrame(*c.engine, c.s.pos, c.s.L, FrameOpts{});
  } else {
    c.fr = coldFrame(c.s, c.s.pos, wt);
  }
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
double dotH(const std::vector<Real>& a, const std::vector<Real>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  return s;
}
double normH(const std::vector<Real>& a) {
  return std::sqrt(dotH(a, a));
}
double maxAbs(const std::vector<Real>& a) {
  double m = 0;
  for (Real x : a)
    m = std::max(m, std::fabs(x));
  return m;
}
double relMax(const std::vector<Real>& a, const std::vector<Real>& b) {
  double d = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  const double s = maxAbs(b);
  return s > 0 ? d / s : d;
}
const char* verdict(bool ok) {
  return ok ? "PASS" : "FAIL";
}

DV constant(std::size_t n, Real v, const char* name) {
  DV d(name, n);
  Kokkos::deep_copy(d, v);
  return d;
}

/// ρ = 1 within R = L/4 of the box center, `outside` elsewhere (gate L; export_lrho.py's r1000).
std::vector<Real> sphereRho(const Scene& s, Real outside) {
  const auto x = down(s.pos);
  std::vector<Real> r(s.N);
  for (int i = 0; i < s.N; ++i) {
    double d2 = 0;
    for (int k = 0; k < 3; ++k) {
      const double dk = x[3 * i + k] - 0.5 * s.L[k];
      d2 += dk * dk;
    }
    r[i] = std::sqrt(d2) < 0.25 * s.L[0] ? Real(1) : outside;
  }
  return r;
}

// ---- velocity fields of the reference (ale_reference_stepper.py)
// ---------------------------------
std::vector<Real> taylorGreen(const std::vector<Real>& x, int N, double nu, double t) {
  std::vector<Real> u(3 * N);
  const double k = kTwoPi, e = std::exp(-2.0 * nu * k * k * t);
  for (int i = 0; i < N; ++i) {
    const double X = x[3 * i], Y = x[3 * i + 1];
    u[3 * i] = std::sin(k * X) * std::cos(k * Y) * e;
    u[3 * i + 1] = -std::cos(k * X) * std::sin(k * Y) * e;
    u[3 * i + 2] = 0;
  }
  return u;
}
std::vector<Real> abcFlow(const std::vector<Real>& x, int N, double nu, double t) {
  std::vector<Real> u(3 * N);
  const double k = kTwoPi, e = std::exp(-nu * k * k * t);
  for (int i = 0; i < N; ++i) {
    const double X = x[3 * i], Y = x[3 * i + 1], Z = x[3 * i + 2];
    u[3 * i] = (std::sin(k * Z) + std::cos(k * Y)) * e;
    u[3 * i + 1] = (std::sin(k * X) + std::cos(k * Z)) * e;
    u[3 * i + 2] = (std::sin(k * Y) + std::cos(k * X)) * e;
  }
  return u;
}

// ---- Kokkos Tools memory high-water mark (as core's test_graph_amg_parallel)
// ---------------------
struct MemTrack {
  static inline std::int64_t cur = 0, peak = 0;
  static bool mine(const Kokkos::Profiling::SpaceHandle h) {
    return std::strcmp(h.name, peclet::core::MemSpace::name()) == 0;
  }
  static void onAlloc(const Kokkos::Profiling::SpaceHandle h, const char*, const void*,
                      const std::uint64_t size) {
    if (mine(h)) {
      cur += (std::int64_t)size;
      peak = std::max(peak, cur);
    }
  }
  static void onFree(const Kokkos::Profiling::SpaceHandle h, const char*, const void*,
                     const std::uint64_t size) {
    if (mine(h))
      cur -= (std::int64_t)size;
  }
  static void install() {
    Kokkos::Tools::Experimental::set_allocate_data_callback(onAlloc);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(onFree);
  }
};

// ---- the frame data of a one-phase fluid -------------------------------------------------------
struct Fluid {
  DV rho, mu;
};
Fluid fluidOf(const FM& m, double rho, double nu) {
  Fluid f;
  f.rho = constant(m.nCombined, Real(rho), "rho");
  f.mu = constant(m.nFaces, Real(rho * nu), "muFace");
  return f;
}

// ================================================================== default run (the ctest)
int runIdentities() {
  int fails = 0;
  std::mt19937 rng(20261117u);
  // S1 transport identities
  {
    double skew = 0, symQ = 0, posQ = 1e300, fused = 0, cst = 0;
    for (const char* type : {"Z", "W"}) {
      Case c = makeCase(type, 8);
      const FM& m = c.fr.mesh;
      const int N = m.nCells, F = m.nFaces;
      const auto Uh = randVec(3 * N, rng), Vh = randVec(3 * N, rng), Fh = randVec(F, rng);
      const DV U = up(Uh, "U"), V = up(Vh, "V"), Fm = up(Fh, "Fm");
      DV a("a", 3 * N), b("b", 3 * N), t("t", 3 * N);
      ale::transport(m, U, Fm, a);
      ale::transport(m, V, Fm, b);
      const auto TU = down(a), TV = down(b);
      skew = std::max(skew, std::fabs(dotH(Vh, TU) + dotH(Uh, TV)) /
                                (normH(Vh) * normH(TU) + normH(Uh) * normH(TV)));
      ale::upwind(m, U, Fm, a);
      ale::upwind(m, V, Fm, b);
      const auto QU = down(a), QV = down(b);
      symQ = std::max(symQ, std::fabs(dotH(Vh, QU) - dotH(Uh, QV)) /
                                (normH(Vh) * normH(QU) + normH(Uh) * normH(QV)));
      posQ = std::min(posQ, dotH(Uh, QU) / (normH(Uh) * normH(QU)));
      ale::transportUpwind(m, U, Fm, t);
      auto sum = TU;
      for (std::size_t i = 0; i < sum.size(); ++i)
        sum[i] += QU[i];
      fused = std::max(fused, relMax(down(t), sum));
      // explicit transport of a constant field: G = 0, so e6 = e8 = 0 exactly up to the gradient's
      // rounding (Σ_f A_f/d_f (U_o − U_i) ⊗ r with U_o − U_i = 0 exactly)
      DV G("G", 9 * N), s6("s6", 6 * F), e6("e6", 3 * N), e8("e8", 3 * N);
      const DV C = constant(3 * N, Real(0.7), "C");
      ale::cellGradient(m, C, G);
      ale::explicitTransport(m, G, Fm, e6, e8, s6);
      cst = std::max({cst, maxAbs(down(e6)), maxAbs(down(e8))});
    }
    const bool ok = skew <= 1e-14 && symQ <= 1e-14 && posQ >= 0 && fused <= 1e-14 && cst == 0;
    fails += !ok;
    std::printf(
        "S1 %s skew(T)=%.2e sym(Q)=%.2e min uQu=%.2e fused(T+Q)=%.2e explicit(const)=%.1e\n",
        verdict(ok), skew, symQ, posQ, fused, cst);
  }
  // S2 Laplacian solver
  {
    bool ok = true;
    std::string txt;
    for (double ratio : {1.0, 1000.0}) {
      Case c = makeCase("Z", 12);
      const FM& m = c.fr.mesh;
      const int N = m.nCells;
      const DV rho = up(sphereRho(c.s, Real(ratio)), "rho");
      ale::LaplacianMatrix<Real> L;
      ale::assembleLaplacian(m, rho, L);
      ale::LaplacianSolver<Real> S;
      S.setup(L);
      auto bh = randVec(N, rng);
      double mean = 0;
      for (Real v : bh)
        mean += v;
      mean /= N;
      for (Real& v : bh)
        v -= mean;
      const DV b = up(bh, "b");
      DV x("x", N), Lx("Lx", N);
      kry::PcgTest<Real> test;
      test.rtol = 1e-10;
      const auto rep = S.solve(b, x, test);
      ale::applyLaplacian(L, x, Lx);
      auto r = down(Lx);
      for (int i = 0; i < N; ++i)
        r[i] = bh[i] - r[i];
      const auto xh = down(x);
      double xs = 0;
      for (Real v : xh)
        xs += v;
      const double res = normH(r) / normH(bh);
      const bool o = rep.converged && res <= 1.0001e-10 && std::fabs(xs) / N <= 1e-14 * maxAbs(xh);
      ok = ok && o;
      char buf[200];
      std::snprintf(buf, sizeof buf, " ratio%g:its=%d,res=%.1e,levels=%d,mean=%.0e", ratio,
                    rep.iterations, res, S.numLevels(), xs / N);
      txt += buf;
    }
    fails += !ok;
    std::printf("S2 %s%s\n", verdict(ok), txt.c_str());
  }
  // S3 rest stays exactly zero; S4 closure and momentum
  {
    bool ok3 = true;
    std::string t3;
    for (auto mode : {ale::Projection::Damped, ale::Projection::Exact}) {
      Case c = makeCase("W", 8);
      const FM& m = c.fr.mesh;
      Fluid fl = fluidOf(m, 1.0, 0.01);
      ale::FrameData<Real> fd;
      ale::setupFrame(fd, m, fl.rho, fl.mu);
      ale::FlowState<Real> st;
      st.allocate(m.nCombined);
      ale::Stepper<Real> stp;
      for (int k = 0; k < 10; ++k)
        stp.step(fd, st, Real(0.05), mode);
      const double mu = maxAbs(down(st.U)), mp = maxAbs(down(st.p));
      ok3 = ok3 && mu == 0 && mp == 0;
      char buf[100];
      std::snprintf(buf, sizeof buf, " %s:|U|=%.1e,|p|=%.1e",
                    mode == ale::Projection::Damped ? "F" : "R", mu, mp);
      t3 += buf;
    }
    fails += !ok3;
    std::printf("S3 %s%s\n", verdict(ok3), t3.c_str());
    Case c = makeCase("Z", 8);
    const FM& m = c.fr.mesh;
    Fluid fl = fluidOf(m, 1.0, 0.01);
    ale::FrameData<Real> fd;
    ale::setupFrame(fd, m, fl.rho, fl.mu);
    ale::FlowState<Real> st;
    st.allocate(m.nCombined);
    Kokkos::deep_copy(st.U, up(taylorGreen(down(c.s.pos), m.nCells, 0.01, 0.0), "tg"));
    ale::Stepper<Real> stp;
    double clo = 0, mom = 0, dv = 0, P0[3], dummy;
    {
      DV MU("MU", 3 * m.nCombined);
      ale::applyMass(m, fl.rho, st.U, MU);
      ale::detail::componentSums(MU, m.nCells, P0, &dummy);
    }
    double resMom[3] = {0, 0, 0};
    for (int k = 0; k < 10; ++k) {
      const auto r = stp.step(fd, st, Real(0.04), ale::Projection::Damped);
      clo = std::max(clo, std::fabs(r.closure) / r.K1);
      for (int d = 0; d < 3; ++d) {
        resMom[d] += r.residualMomentum[d];
        mom = std::max(mom, std::fabs(r.momentum[d] - P0[d] + resMom[d]) / r.absMomentum);
      }
      dv = std::max(dv, r.divPhi1);
    }
    const bool ok4 = clo <= 1e-13 && mom <= 1e-13 && dv <= 1e-14;
    fails += !ok4;
    std::printf("S4 %s closure/K=%.2e momentum(booked)=%.2e tau|DPhi1|/V=%.2e\n", verdict(ok4), clo,
                mom, dv);
  }
  return fails;
}

// ================================================================== file helpers
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

template <class V>
std::vector<char> bytesOf(const V& v) {
  using T = typename V::non_const_value_type;
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  const char* p = reinterpret_cast<const char*>(hv.data());
  return std::vector<char>(p, p + v.extent(0) * sizeof(T));
}

/// Rebuild the frame of an exported mesh `<dir>/<name>.*` and check the export's arrays bitwise
/// (the subset ale_reference.Mesh reads). Returns false (with a message) on a mismatch.
bool frameOfExport(const std::string& dir, const std::string& name, Scene& s, Frame& fr) {
  const std::string p = dir + "/" + name + ".";
  std::vector<char> cb;
  if (!readFile(p + "counts", cb) || cb.size() < 6 * sizeof(long)) {
    std::printf("FRAME %s FAIL cannot read %scounts\n", name.c_str(), p.c_str());
    return false;
  }
  long counts[6];
  std::memcpy(counts, cb.data(), sizeof(counts));
  const int N = (int)counts[0];
  s.N = N;
  s.n = (int)std::lround(std::cbrt((double)N));
  const auto Lv = readVec<Real>(p + "L", 3);
  for (int k = 0; k < 3; ++k)
    s.L[k] = Lv[k];
  s.h = s.L[0] / s.n;
  s.pos = up(readVec<Real>(p + "pos", 3 * (std::size_t)N), "pos");
  s.weight = up(readVec<Real>(p + "weight", N), "w");
  fr = coldFrame(s, s.pos, s.weight);
  const FM& m = fr.mesh;
  const std::vector<std::pair<const char*, std::vector<char>>> arrs = {
      {"faceCellA", bytesOf(m.faceCellA)},  {"faceCellB", bytesOf(m.faceCellB)},
      {"faceArea", bytesOf(m.faceArea)},    {"faceNormal", bytesOf(m.faceNormal)},
      {"faceDist", bytesOf(m.faceDist)},    {"faceHa", bytesOf(m.faceHa)},
      {"faceHb", bytesOf(m.faceHb)},        {"faceCentroid", bytesOf(m.faceCentroid)},
      {"faceConn", bytesOf(m.faceConn)},    {"cellFaceOffset", bytesOf(m.cellFaceOffset)},
      {"cellFace", bytesOf(m.cellFace)},    {"cellFaceSign", bytesOf(m.cellFaceSign)},
      {"cellVolume", bytesOf(m.cellVolume)}};
  for (const auto& [nm, bytes] : arrs) {
    std::vector<char> b;
    if (!readFile(p + nm, b) || b != bytes) {
      std::printf("FRAME %s FAIL array %s differs from the export\n", name.c_str(), nm);
      return false;
    }
  }
  return true;
}

// ================================================================== --apply (C0, device side)
int applyMesh(const std::string& dir, const std::string& odir, const std::string& name) {
  Scene s;
  Frame fr;
  if (!frameOfExport(dir, name, s, fr))
    return 1;
  const FM& m = fr.mesh;
  const int NC = m.nCombined, F = m.nFaces;
  const std::string p = dir + "/" + name + ".";
  auto in = [&](const char* nm, std::size_t n) { return readVec<Real>(p + "in_" + nm, n); };
  const DV U = up(in("U", 3 * (std::size_t)NC), "U"), Fm = up(in("Fm", F), "Fm"),
           mu = up(in("mu", F), "mu"), rho = up(in("rho", NC), "rho");
  const Real tau = in("tau", 1)[0];
  const std::string o = odir + "/" + name + ".out_";
  DV cell3("c3", 3 * NC), G("G", 9 * NC), face3("f3", 3 * F), s6("s6", 6 * F), e6("e6", 3 * NC),
      e8("e8", 3 * NC);
  ale::transport(m, U, Fm, cell3);
  writeVec(o + "T", down(cell3));
  ale::upwind(m, U, Fm, cell3);
  writeVec(o + "Q", down(cell3));
  ale::transportUpwind(m, U, Fm, cell3);
  writeVec(o + "TQ", down(cell3));
  ale::cellGradient(m, U, G);
  ale::reconstructionJump(m, G, face3);
  writeVec(o + "RmD", down(face3));
  ale::centroidCorrection(m, G, face3);
  writeVec(o + "delta", down(face3));
  ale::explicitTransport(m, G, Fm, e6, e8, s6);
  auto ex = down(e6);
  const auto e8h = down(e8);
  for (std::size_t i = 0; i < ex.size(); ++i)
    ex[i] += e8h[i];
  writeVec(o + "explicit", ex);
  {
    ale::FrameData<Real> fd;
    ale::setupFrame(fd, m, rho, mu);
    ale::Stepper<Real> stp;
    stp.reserve(m);
    stp.predictorOperator(fd, Fm, tau, U, cell3, Real(1));
    writeVec(o + "pred", down(cell3));
  }
  std::printf("APPLY %s ok N=%d nFaces=%d frame=bitwise-equal-to-export\n", name.c_str(), NC, F);
  return 0;
}

// ================================================================== --laplacian (gate L)
int laplacianGate(int n) {
  MemTrack::install();
  Case c = makeCase("Z", n);
  const FM& m = c.fr.mesh;
  const int N = m.nCells;
  std::printf("=== S1.2b L n=%d N=%d nFaces=%d exec=%s threads=%d ===\n", n, N, m.nFaces,
              Exec::name(), Exec().concurrency());
  for (double ratio : {1.0, 1000.0}) {
    const DV rho = up(sphereRho(c.s, Real(ratio)), "rho");
    ale::LaplacianMatrix<Real> L;
    ale::assembleLaplacian(m, rho, L);
    ale::LaplacianSolver<Real> S;
    double best = 1e30;
    std::int64_t peak = 0;
    for (int rep = 0; rep < 3; ++rep) {
      S = ale::LaplacianSolver<Real>();
      Kokkos::fence();
      const std::int64_t base = MemTrack::cur;
      MemTrack::peak = base;
      const auto t0 = std::chrono::steady_clock::now();
      S.setup(L);
      Kokkos::fence();
      best = std::min(best,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      peak = std::max(peak, MemTrack::peak - base);
    }
    std::mt19937 rng(7u);
    auto bh = randVec(N, rng);
    double mean = 0;
    for (Real v : bh)
      mean += v;
    mean /= N;
    for (Real& v : bh)
      v -= mean;
    const DV b = up(bh, "b");
    DV x("x", N);
    kry::PcgTest<Real> test;
    test.rtol = 1e-10;
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    const auto rep = S.solve(b, x, test);
    Kokkos::fence();
    const double ts = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string sizes;
    for (int l = 0; l < S.numLevels(); ++l)
      sizes += (l ? "," : "") + std::to_string((long long)S.levelSize(l));
    std::printf(
        "LAP n=%d ratio=%g iters=%d converged=%d res=%.2e levels=%d sizes=%s setup_ms=%.2f "
        "setup_peak_MiB=%.1f solve_ms=%.2f\n",
        n, ratio, rep.iterations, (int)rep.converged, rep.residual / rep.rhsNorm, S.numLevels(),
        sizes.c_str(), 1e3 * best, (double)peak / (1024.0 * 1024.0), 1e3 * ts);
  }
  return 0;
}

// ================================================================== the runs (R0–R6, T)
struct RunSpec {
  std::string name, fam = "Z", field = "tg", force;
  int n = 16, steps = -1, sample = 1;
  ale::Projection mode = ale::Projection::Damped;
  double rho = 1.0;
  double tauFixed = 0;  ///< > 0: τ of the case (unif_Z16_*), not the Courant rule
};

bool parseCase(const std::string& name, RunSpec& r) {
  r.name = name;
  if (name == "r0_zero_F" || name == "r0_zero_R" || name == "r0_unif_F" || name == "r0_unif_R") {
    r.fam = "W";
    r.n = 8;
    r.field = name.substr(3, 4);
    r.mode = name.back() == 'R' ? ale::Projection::Exact : ale::Projection::Damped;
    r.steps = r.field == "zero" ? 10 : 100;
    return true;
  }
  if (name == "unif_Z16_F" || name == "unif_Z16_R") {  // gate G1.3 b (brief ALE_S1_3A)
    r.fam = "Z", r.n = 16, r.field = "unif", r.steps = 400, r.tauFixed = 1.0 / 400;
    r.mode = name.back() == 'R' ? ale::Projection::Exact : ale::Projection::Damped;
    return true;
  }
  if (name == "r3_tg_W16_F") {
    r.fam = "W", r.n = 16, r.steps = 2000, r.sample = 100;
    return true;
  }
  if (name == "r3_tg_W32_F") {
    r.fam = "W", r.n = 32, r.steps = 500, r.sample = 25;
    return true;
  }
  if (name == "r6i_W16") {
    r.fam = "W", r.n = 16, r.field = "zero", r.steps = 100, r.rho = 1000.0, r.force = "psi_face";
    return true;
  }
  if (name == "r6ii_rest_W16") {
    r.fam = "W", r.n = 16, r.field = "zero", r.steps = 100, r.force = "gbar";
    return true;
  }
  if (name == "r6ii_tg_W16") {
    r.fam = "W", r.n = 16, r.field = "tg", r.steps = 100, r.force = "gbar";
    return true;
  }
  // <tg|abc>_<Z|W><n>_<F|R>
  const auto u1 = name.find('_'), u2 = name.rfind('_');
  if (u1 == std::string::npos || u2 == u1)
    return false;
  r.field = name.substr(0, u1);
  const std::string mesh = name.substr(u1 + 1, u2 - u1 - 1), mode = name.substr(u2 + 1);
  if ((r.field != "tg" && r.field != "abc") || mesh.size() < 2 ||
      (mesh[0] != 'Z' && mesh[0] != 'W') || (mode != "F" && mode != "R"))
    return false;
  r.fam = mesh.substr(0, 1);
  r.n = std::atoi(mesh.c_str() + 1);
  r.mode = mode == "R" ? ale::Projection::Exact : ale::Projection::Damped;
  return r.n > 0;
}

std::vector<Real> fieldOf(const std::string& f, const std::vector<Real>& x, int N, double nu,
                          double t) {
  if (f == "tg")
    return taylorGreen(x, N, nu, t);
  if (f == "abc")
    return abcFlow(x, N, nu, t);
  if (f == "unif") {
    std::vector<Real> u(3 * N);
    for (int i = 0; i < N; ++i)
      u[3 * i] = 0.3, u[3 * i + 1] = -0.7, u[3 * i + 2] = 0.5;
    return u;
  }
  return std::vector<Real>(3 * N, 0.0);
}

/// Body force of amendment A7 and the start pressure it implies (empty when none).
void bodyForce(const RunSpec& r, const FM& m, const Scene& s, const DV& rho, DV& body, DV& p0) {
  const int N = m.nCells, F = m.nFaces;
  if (r.force == "psi_face") {
    const auto x = down(s.pos), cA = down(m.faceCentroid);
    const auto A = down(m.faceCellA);
    const double k = kTwoPi / s.L[1];
    std::vector<Real> pc(N), pf(F), p(N);
    for (int i = 0; i < N; ++i) {
      pc[i] = std::cos(k * x[3 * i + 1]);
      p[i] = -r.rho * pc[i];
    }
    for (int f = 0; f < F; ++f)
      pf[f] = std::cos(k * (x[3 * A[f] + 1] + cA[3 * f + 1]));
    DV psiC = up(pc, "psiC"), psiF = up(pf, "psiF"), gam("gamma", F), s9("s9", 9 * N);
    ale::bodyForceFace(m, rho, psiC, psiF, gam);
    body = DV("body", 3 * N);
    ale::fluxMapTranspose(m, gam, body, s9);
    p0 = up(p, "p0");
  } else if (r.force == "gbar") {
    const Real g[3] = {0, 0, -1};
    body = DV("body", 3 * N);
    ale::bodyForceCell(m, rho, g, body);
  }
}

/// The prescribed mesh velocity of gate G1.3 b (brief ALE_S1_3A, box [0, 1)³): ẋ = 0.3 (sin 2πy,
/// sin 2πz, sin 2πx) at the generator's present position `pos`.
void prescribedVelocity(const DV& pos, int N, const DV& out) {
  Kokkos::parallel_for(
      "t.xdotP", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real k = Real(kTwoPi);
        out(3 * i) = Real(0.3) * Kokkos::sin(k * pos(3 * i + 1));
        out(3 * i + 1) = Real(0.3) * Kokkos::sin(k * pos(3 * i + 2));
        out(3 * i + 2) = Real(0.3) * Kokkos::sin(k * pos(3 * i));
      });
}

/// A frame as `<dir>/<name>.<array>` in the layout ale_reference.Mesh reads (test_ale_frame's
/// export: counts, the FaceMesh arrays, pos, weight, L).
void writeMesh(const std::string& dir, const std::string& name, const Frame& f, const Real L[3]) {
  const FM& m = f.mesh;
  const std::string p = dir + "/" + name + ".";
  auto put = [&](const char* nm, const std::vector<char>& b) {
    FILE* fp = std::fopen((p + nm).c_str(), "wb");
    if (!fp) {
      std::fprintf(stderr, "cannot write %s%s\n", p.c_str(), nm);
      std::exit(2);
    }
    if (!b.empty())
      std::fwrite(b.data(), 1, b.size(), fp);
    std::fclose(fp);
  };
  const long counts[6] = {m.nCells,    m.nFaces,   m.nInterior,
                          m.nCombined, m.nDropped, m.generatorDiag.numNegativeHeight};
  const Real sine = m.generatorDiag.maxNormalConnSine;
  std::vector<char> cb(reinterpret_cast<const char*>(counts),
                       reinterpret_cast<const char*>(counts) + sizeof(counts));
  cb.insert(cb.end(), reinterpret_cast<const char*>(&sine),
            reinterpret_cast<const char*>(&sine) + sizeof(Real));
  put("counts", cb);
  put("faceCellA", bytesOf(m.faceCellA));
  put("faceCellB", bytesOf(m.faceCellB));
  put("faceArea", bytesOf(m.faceArea));
  put("faceNormal", bytesOf(m.faceNormal));
  put("faceDist", bytesOf(m.faceDist));
  put("faceHa", bytesOf(m.faceHa));
  put("faceHb", bytesOf(m.faceHb));
  put("faceCentroid", bytesOf(m.faceCentroid));
  put("faceConn", bytesOf(m.faceConn));
  put("cellFaceOffset", bytesOf(m.cellFaceOffset));
  put("cellFace", bytesOf(m.cellFace));
  put("cellFaceSign", bytesOf(m.cellFaceSign));
  put("cellVolume", bytesOf(m.cellVolume));
  put("pos", bytesOf(f.pos));
  put("weight", bytesOf(f.weight));
  writeVec(p + "L", std::vector<Real>(L, L + 3));
}

std::vector<Real> downN(const DV& v, std::size_t n) {
  auto h = down(v);
  h.resize(n);
  return h;
}

/// --trace-moving: step k's frames Fh, F1 (`<case>.Fh_k`, `<case>.F1_k`), its inputs (Uⁿ, U_ext,
/// pⁿ, Vⁿ, ẋ: `<case>.in_<q>_k`), outputs (F on Fh, U*, U¹, p¹: `<case>.dev_<q>_k`) and one line
/// of scalars (k, ω, μ̂, gcl, closure, the B terms, dK, K¹, ΔP_def, residual momentum).
void traceStep(const std::string& dir, const std::string& cs, int k,
               const ale::MovingMesh<Real>& mm, const ale::MovingStepper<Real>& mstp,
               const std::vector<Real>& Un, const std::vector<Real>& pn,
               const std::vector<Real>& Vn, const ale::FlowState<Real>& st,
               const ale::MovingStepReport& r, FILE* sc) {
  const int N = mm.half.mesh.nCells;
  const std::string ks = std::to_string(k), p = dir + "/" + cs + ".";
  writeMesh(dir, cs + ".Fh_" + ks, mm.half, mm.L);
  writeMesh(dir, cs + ".F1_" + ks, mm.full, mm.L);
  writeVec(p + "in_U_" + ks, Un);
  writeVec(p + "in_Uext_" + ks, downN(mstp.base.ws.Uext, 3 * (std::size_t)N));
  writeVec(p + "in_p_" + ks, pn);
  writeVec(p + "in_Vn_" + ks, Vn);
  writeVec(p + "in_xdot_" + ks, downN(mstp.lastMeshVelocity(), 3 * (std::size_t)N));
  writeVec(p + "dev_F_" + ks, downN(mstp.lastTransportFlux(), mm.half.mesh.nFaces));
  writeVec(p + "dev_Ustar_" + ks, downN(mstp.lastUstar(), 3 * (std::size_t)N));
  writeVec(p + "dev_U_" + ks, downN(st.U, 3 * (std::size_t)N));
  writeVec(p + "dev_p_" + ks, downN(st.p, N));
  std::fprintf(sc,
               "%d %.17g %.17g %.17g %.17g B1=%.17g B2=%.17g B3=%.17g B4=%.17g B5=%.17g B6=%.17g "
               "B7=%.17g B8a=%.17g B8b=%.17g Bres=%.17g dK=%.17g K1=%.17g def=%.17g,%.17g,%.17g "
               "res=%.17g,%.17g,%.17g pred_apps=%d swept_its=%d rebuilt=%d,%d\n",
               k, r.omega, r.muHat, r.gcl, r.closure, r.B1, r.B2, r.B3, r.B4, r.B5, r.B6, r.B7,
               r.B8a, r.B8b, r.Bres, r.dK, r.K1, r.defectMomentum[0], r.defectMomentum[1],
               r.defectMomentum[2], r.residualMomentum[0], r.residualMomentum[1],
               r.residualMomentum[2], r.predictor.applications, r.swept.iterations, r.rebuiltHalf,
               r.rebuiltFull);
}

struct Stats {
  double closure = 0, excess = -1e300, mom = 0, momBooked = 0, divPhi1 = 0, divF = 0, cfl = 0,
         predRes = 0, maxU = 0, omegaMu = 0, muRatio = 1e300, unifDev = 0, gtDev = 0, meanDev = 0,
         meanDevBooked = 0, resOverDiss = 0;
  int muSamples = 0;
  long predIts = 0, predApps = 0, predAppsMax = 0, projIts = 0, projItsMax = 0, s3Its = 0,
       s6Its = 0, muIts = 0, muSolves = 0, steps = 0, notConverged = 0;
  double tS3 = 0, tPred = 0, tMu = 0, tS5 = 0, tS6 = 0, tBudget = 0, tSolveS3 = 0, tSolveS6 = 0;
  double Bsum[11] = {0};
};

// ---- --dump-on-error DIR (diagnostic, brief ALE_S1_3A_R) ---------------------------------------
/// The directory of --dump-on-error (empty: off).
std::string g_dumpDir;
/// The frames kept for the dump: the last kDumpChain frame positions (the oldest first).
constexpr std::size_t kDumpChain = 8;

void printOpenCells(const char* tag, const std::vector<ale::CellClosure>& oc) {
  for (const auto& c : oc)
    std::printf(
        "%s cell=%d own=%.3e dis=%.3e sum=%.6e own_rel=%.3e dis_rel=%.3e bar=%.3e worst_nbr=%ld "
        "worst_rel=%.3e facets=%d nonrecip=%d\n",
        tag, c.cell, c.own, c.dis, c.sum, c.own / c.sum, c.dis / c.sum, ale::kClosureFrac * c.sum,
        c.worstNbr, c.worstDis / c.sum, c.facets, c.nonRecip);
}

bool inList(const std::vector<int>& l, long i) {
  return std::binary_search(l.begin(), l.end(), (int)i);
}

/// A frame error of the moving step k (1-based): writes DIR/meta, DIR/weight (N), DIR/chain_<q>
/// (3N, q = 0 oldest; the last is the failing target, the one before it the last good frame),
/// DIR/forced_<r> (the slots forced in round r); prints the open cells of the failing frame with
/// their closure terms (OPEN), whether each and its neighbours were forced in each round
/// (OPENFORCED), the rounds continued on the same engine to 8 (MORE), and the open cells of a
/// cold build of the target (COLD).
void dumpFrameError(const std::string& dir, const std::string& name, int k, Case& c,
                    ale::MovingMesh<Real>& mm, std::vector<std::vector<Real>> chain) {
  const bool hFail = mm.half.numEmptyCells > 0 || mm.half.numOneSidedFacets > 0 ||
                     mm.half.reciprocity.numOpenCells > 0;
  Frame& f = hFail ? mm.half : mm.full;
  if (!hFail)
    chain.push_back(down(mm.half.pos));
  chain.push_back(down(f.pos));
  while (chain.size() > kDumpChain)
    chain.erase(chain.begin());
  Engine& eng = *c.engine;
  const int N = c.s.N;
  for (std::size_t q = 0; q < chain.size(); ++q)
    writeVec(dir + "/chain_" + std::to_string(q), chain[q]);
  writeVec(dir + "/weight", down(eng.weight));
  for (std::size_t r = 0; r < f.roundLog.forced.size(); ++r) {
    FILE* fp = std::fopen((dir + "/forced_" + std::to_string(r + 1)).c_str(), "w");
    for (int i : f.roundLog.forced[r])
      std::fprintf(fp, "%d\n", i);
    std::fclose(fp);
  }
  {
    FILE* fp = std::fopen((dir + "/meta").c_str(), "w");
    std::fprintf(fp,
                 "case=%s step=%d frame=%s N=%d L=%.17g h=%.17g tol_h=1e-4 skin_h=0.25 chain=%zu "
                 "rounds=%d numOpenCells=%d numOneSidedFacets=%d numEmptyCells=%d numRebuilt=%d "
                 "numRoundRebuilt=%d route=%d fellBack=%d pass1=%d pass1Raw=%d pass2=%d extra=%d "
                 "verifyPasses=%d exec=%s threads=%d\n",
                 name.c_str(), k, hFail ? "Fh" : "F1", N, (double)c.s.L[0], (double)c.s.h,
                 chain.size(), f.reciprocity.rounds, f.reciprocity.numOpenCells,
                 f.numOneSidedFacets, f.numEmptyCells, f.numRebuilt, f.reciprocity.numRoundRebuilt,
                 (int)f.stats.route, (int)f.stats.fellBack, f.stats.pass1, f.stats.pass1Raw,
                 f.stats.pass2, f.stats.extra, f.stats.verifyPasses, Exec::name(),
                 Exec().concurrency());
    std::fclose(fp);
  }
  std::printf(
      "DUMP dir=%s frame=%s step=%d rounds=%d route=%d fellBack=%d pass1=%d pass2=%d "
      "numRebuilt=%d numRoundRebuilt=%d open_per_publish=",
      dir.c_str(), hFail ? "Fh" : "F1", k, f.reciprocity.rounds, (int)f.stats.route,
      (int)f.stats.fellBack, f.stats.pass1, f.stats.pass2, f.numRebuilt,
      f.reciprocity.numRoundRebuilt);
  for (int o : f.roundLog.open)
    std::printf("%d,", o);
  std::printf(" forced_per_round=");
  for (const auto& l : f.roundLog.forced)
    std::printf("%zu,", l.size());
  std::printf("\n");
  const auto oc = ale::openCellsOf(eng, f.pos, f.weight, c.s.L);
  printOpenCells("OPEN", oc);
  for (const auto& o : oc)
    for (std::size_t r = 0; r < f.roundLog.forced.size(); ++r) {
      const auto& l = f.roundLog.forced[r];
      int nf = 0;
      for (long j : o.nbrs)
        nf += inList(l, j) ? 1 : 0;
      std::printf(
          "OPENFORCED cell=%d round=%zu cell_forced=%d worst_nbr_forced=%d nbrs_forced=%d/%zu\n",
          o.cell, r + 1, (int)inList(l, o.cell), (int)inList(l, o.worstNbr), nf, o.nbrs.size());
    }
  // the rounds continued on the same engine up to 8
  Frame g = f;
  const auto more = ale::continueRounds(eng, g, c.s.L, mm.frameOptions, 8);
  std::printf("MORE rounds_total=%d open_per_publish=", g.reciprocity.rounds);
  for (int o : more.open)
    std::printf("%d,", o);
  std::printf(" forced_per_round=");
  for (const auto& l : more.forced)
    std::printf("%zu,", l.size());
  std::printf(" oneSided=%d empty=%d\n", g.numOneSidedFacets, g.numEmptyCells);
  printOpenCells("MOREOPEN", ale::openCellsOf(eng, g.pos, g.weight, c.s.L));
  // a cold build of the target in a fresh engine
  Engine cold;
  cold.alloc(N, c.s.L, Real(1e-4) * c.s.h, Real(0.25) * c.s.h, 4, N);
  cold.setWeights(eng.weight);
  cold.rebuild(f.pos);
  cold.step(f.pos);
  const auto co = ale::openCellsOf(cold, f.pos, f.weight, c.s.L);
  std::printf("COLD open=%zu\n", co.size());
  printOpenCells("COLDOPEN", co);
  for (const auto& o : oc) {  // the open cell in the cold build
    const auto hr = down(eng.vol), hc = down(cold.vol);
    std::printf("COLDCELL cell=%d V_repair=%.17g V_cold=%.17g rel=%.3e\n", o.cell,
                (double)hr[o.cell], (double)hc[o.cell],
                std::fabs((double)hr[o.cell] - hc[o.cell]) / hc[o.cell]);
  }
}

int runCase(const std::string& name, int stepsOverride, bool tight, double courant, bool stokes,
            const std::string& motion = "none", double tend = -1,
            const std::string& traceDir = std::string()) {
  RunSpec r;
  if (!parseCase(name, r)) {
    std::printf("RESULT case=%s status=UNKNOWN_CASE\n", name.c_str());
    return 1;
  }
  const bool moving = motion != "none";
  ale::MeshMotion mot = ale::MeshMotion::Rest;
  if (motion == "follow")
    mot = ale::MeshMotion::Follow;
  else if (motion == "prescribed")
    mot = ale::MeshMotion::Prescribed;
  else if (moving && motion != "rest") {
    std::printf("RESULT case=%s status=UNKNOWN_MOTION motion=%s\n", name.c_str(), motion.c_str());
    return 1;
  }
  const double nu = 0.01;
  const auto tSetup0 = std::chrono::steady_clock::now();
  Case c = makeCase(r.fam, r.n, moving);
  const FM& m = c.fr.mesh;
  const int N = m.nCells;
  Fluid fl = fluidOf(m, r.rho, nu);
  ale::FrameData<Real> fd;
  ale::setupFrame(fd, m, fl.rho, fl.mu);
  ale::StepOptions opt;
  opt.transport = !stokes;
  if (tight) {
    opt.predictorRtol = 1e-13;
    opt.laplacianRtol = 1e-12;
  }
  if (g_exactRtol > 0)
    opt.exactRtol = g_exactRtol;
  if (g_predRtol > 0)
    opt.predictorRtol = g_predRtol;
  opt.timed = true;
  ale::Stepper<Real> stp(opt);
  stp.reserve(m);
  const auto xh = down(c.s.pos);
  const auto Vh = down(m.cellVolume);
  double Vsum = 0;
  for (Real v : Vh)
    Vsum += v;
  const double h = std::cbrt(Vsum / N);
  // start field and time step (run_case of the reference): the start field (Taylor–Green for the
  // fields at rest) projected exactly; τ from the flux Courant number `courant` (default 0.4) of
  // its compact flux
  auto U0h = fieldOf(r.field, xh, N, nu, 0.0);
  std::vector<Real> Uch = r.field == "zero" ? taylorGreen(xh, N, nu, 0.0) : U0h;
  DV Uc = up(Uch, "Uc"), dp0("dp0", N), t3("t3", 3 * N), t3b("t3b", 3 * N), sF("sF", m.nFaces),
     s9("s9", 9 * N), Phi("Phi", m.nFaces), phi("phi", N);
  const auto rep0 = stp.exactProjection(fd, Uc, Real(1), dp0);
  ale::pressureForce(m, dp0, t3, sF, s9);
  ale::applyInverseMass(m, fl.rho, t3, t3b);
  kry::axpy(Uc, Real(1), t3b, 3 * N);
  stp.compactFluxOf(fd, Uc, Real(1), phi, Phi);
  const double c1 = ale::detail::courant(m, Phi, 1.0);
  const int nst1 = (int)std::ceil(c1 / courant);
  const double tau = r.tauFixed > 0 ? r.tauFixed : 1.0 / nst1;
  const int nst = stepsOverride > 0 ? stepsOverride
                  : tend > 0        ? (int)std::lround(tend / tau)
                                    : (r.steps > 0 ? r.steps : nst1);
  DV body, pStart;
  bodyForce(r, m, c.s, fl.rho, body, pStart);
  double muEx = std::nan("");
  DV muv("muv", N);
  if (r.mode == ale::Projection::Damped)
    muEx = stp.muExact(fd, muv);
  ale::FlowState<Real> st;
  st.allocate(m.nCombined);
  if (pStart.extent(0) > 0)
    Kokkos::deep_copy(st.p, pStart);
  if (r.field == "tg" || r.field == "abc") {
    Kokkos::deep_copy(st.U, Uc);
    // start pressure from one trial step in mode R from p = 0 (the reference's run_case)
    ale::FlowState<Real> trial;
    trial.allocate(m.nCombined);
    Kokkos::deep_copy(trial.U, Uc);
    ale::Stepper<Real> ts(opt);
    ts.step(fd, trial, Real(tau), ale::Projection::Exact, body);
    Kokkos::deep_copy(st.p, trial.p);
  } else {
    Kokkos::deep_copy(st.U, up(U0h, "U0"));
  }
  Kokkos::fence();
  const double tSetup =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - tSetup0).count();
  // the moving mesh (A18): its engine, frame n and, at rest, the static path beside it (V0.2)
  ale::MovingStepper<Real> mstp(opt);
  ale::MovingMesh<Real> mm;
  ale::FlowState<Real> st0;
  DV xdotP;
  std::vector<std::vector<Real>> chain;  // --dump-on-error: the last frames' positions
  if (moving) {
    mm.start(c.fr, c.s.L);
    if (!g_dumpDir.empty()) {
      mm.frameOptions.recordRounds = true;
      chain.push_back(down(mm.pos));
    }
    if (mot == ale::MeshMotion::Prescribed)
      xdotP = DV("xdotP", 3 * N);
    if (mot == ale::MeshMotion::Rest) {
      st0.allocate(m.nCombined);
      Kokkos::deep_copy(st0.U, st.U);
      Kokkos::deep_copy(st0.p, st.p);
    }
  }
  if (!traceDir.empty()) {
    FILE* mf = std::fopen((traceDir + "/" + name + ".meta").c_str(), "w");
    std::fprintf(mf, "case=%s tau=%.17g rho=%.17g nu=%.17g mode=%s steps=%d motion=%s\n",
                 name.c_str(), tau, r.rho, nu, r.mode == ale::Projection::Exact ? "R" : "F", nst,
                 motion.c_str());
    std::fclose(mf);
  }
  FILE* traceScalars = traceDir.empty()
                           ? nullptr
                           : std::fopen((traceDir + "/" + name + ".dev_scalars").c_str(), "w");
  const auto U0 = down(st.U);
  double norm0 = 0, P0[3], absP0 = 0, Mtot = 0;
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d)
      norm0 += Vh[i] * U0[3 * i + d] * U0[3 * i + d];
  norm0 = std::sqrt(norm0);
  {
    ale::applyMass(m, fl.rho, st.U, t3);
    ale::detail::componentSums(t3, N, P0, &absP0);
    Mtot = r.rho * Vsum;
  }
  const double K0 = [&] {
    double k = 0;
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        k += 0.5 * r.rho * Vh[i] * U0[3 * i + d] * U0[3 * i + d];
    return k;
  }();
  std::printf(
      "## RUN %s mesh=%s%d N=%d F=%d field=%s mode=%s rho=%g nu=%g tau=%.6e steps=%d "
      "courant_start=%.4f beta=%.4f mu_exact=%.6f exact_proj_its=%d setup=%.2fs frame_setup=%.3fs "
      "exec=%s threads=%d tight=%d\n",
      name.c_str(), r.fam.c_str(), r.n, N, m.nFaces, r.field.c_str(),
      r.mode == ale::Projection::Exact ? "R" : "F", r.rho, nu, tau, nst, c1 * tau,
      tau * nu / (h * h), muEx, rep0.iterations, tSetup, fd.setupSeconds, Exec::name(),
      Exec().concurrency(), (int)tight);
  Stats S;
  double resMom[3] = {0, 0, 0}, Kprev = K0, defMom[3] = {0, 0, 0};
  double gcl = 0, B1max = 0, momDef = 0, v02diff = 0, tFrames = 0, tFrameSetup = 0;
  long rebuilt = 0, frames = 0, v02bitwise = 0;
  int s3Restarts = 0;
  const Real gbar[3] = {0, 0, -1};
  bool blowup = false;
  const auto tLoop0 = std::chrono::steady_clock::now();
  for (int k = 0; k < nst; ++k) {
    ale::MovingStepReport rr;
    std::vector<Real> trU, trP, trV;
    if (!traceDir.empty())
      trU = down(st.U), trP = down(st.p), trV = down(mm.volume);
    if (moving) {
      if (mot == ale::MeshMotion::Prescribed)
        prescribedVelocity(mm.pos, N, xdotP);
      try {
        rr = mstp.step(*c.engine, mm, st, fl.rho, Real(r.rho * nu), Real(tau), r.mode, mot, xdotP,
                       body);
      } catch (const std::exception& e) {
        if (!g_dumpDir.empty())
          dumpFrameError(g_dumpDir, name, k + 1, c, mm, chain);
        std::printf("RESULT case=%s status=FRAME_ERROR step=%d motion=%s what=\"%s\"\n",
                    name.c_str(), k + 1, motion.c_str(), e.what());
        return 1;
      }
      if (!g_dumpDir.empty()) {
        chain.push_back(down(mm.half.pos));
        chain.push_back(down(mm.full.pos));
        while (chain.size() > kDumpChain)
          chain.erase(chain.begin());
      }
      gcl = std::max(gcl, rr.gcl);
      s3Restarts = std::max(s3Restarts, rr.sweptRestarts);
      rebuilt += rr.rebuiltHalf + rr.rebuiltFull;
      frames += 2;
      tFrames += rr.timeFrames;
      tFrameSetup += rr.timeSetup;
      for (int d = 0; d < 3; ++d)
        defMom[d] += rr.defectMomentum[d];
      if (mot == ale::MeshMotion::Rest) {  // V0.2: the static path on the same start, same process
        stp.step(fd, st0, Real(tau), r.mode, body);
        const auto a = down(st.U), b = down(st0.U), pa = down(st.p), pb = down(st0.p);
        double dmax = 0;
        for (std::size_t q = 0; q < a.size(); ++q)
          dmax = std::max(dmax, std::fabs((double)a[q] - b[q]));
        for (std::size_t q = 0; q < pa.size(); ++q)
          dmax = std::max(dmax, std::fabs((double)pa[q] - pb[q]));
        const bool bit = std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)) == 0 &&
                         std::memcmp(pa.data(), pb.data(), pa.size() * sizeof(Real)) == 0;
        v02diff = std::max(v02diff, dmax);
        v02bitwise += bit ? 1 : 0;
      }
    } else {
      static_cast<ale::StepReport&>(rr) = stp.step(fd, st, Real(tau), r.mode, body);
    }
    if (!traceDir.empty())
      traceStep(traceDir, name, k + 1, mm, mstp, trU, trP, trV, st, rr, traceScalars);
    const double t = tau * (k + 1);
    const double Kref = std::max(rr.K1, Kprev);
    if (Kref > 0) {
      S.closure = std::max(S.closure, std::fabs(rr.closure) / Kref);
      S.excess = std::max(S.excess, (rr.dK - rr.unsignedSum) / Kref);
    }
    const double diss = std::fabs(rr.B4 + rr.B7 + rr.B8a);
    if (diss > 0)
      S.resOverDiss = std::max(S.resOverDiss, std::fabs(rr.Bres) / diss);
    Kprev = rr.K1;
    for (int d = 0; d < 3; ++d)
      resMom[d] += rr.residualMomentum[d];
    if (rr.absMomentum > 0 && body.extent(0) == 0)
      for (int d = 0; d < 3; ++d) {
        S.mom = std::max(S.mom, std::fabs(rr.momentum[d] - P0[d]) / rr.absMomentum);
        S.momBooked =
            std::max(S.momBooked,
                     std::fabs(rr.momentum[d] - P0[d] + resMom[d] - defMom[d]) / rr.absMomentum);
        momDef = std::max(momDef, std::fabs(defMom[d]) / rr.absMomentum);
      }
    if (moving && Kref > 0)
      B1max = std::max(B1max, std::fabs(rr.B1) / Kref);
    S.divPhi1 = std::max(S.divPhi1, rr.divPhi1);
    S.divF = std::max(S.divF, rr.divF);
    S.cfl = std::max(S.cfl, rr.cfl);
    S.predRes = std::max(S.predRes, rr.predictorResidual);
    S.predIts += rr.predictor.iterations;
    S.predApps += rr.predictor.applications;
    S.predAppsMax = std::max<long>(S.predAppsMax, rr.predictor.applications);
    S.projIts += rr.projection.iterations;
    S.projItsMax = std::max<long>(S.projItsMax, rr.projection.iterations);
    S.s3Its += rr.transportFlux.iterations;
    S.s6Its += rr.compactFlux.iterations;
    S.muIts += rr.muIterations;
    S.muSolves += rr.muSolves;
    S.notConverged += !rr.predictor.converged + !rr.projection.converged +
                      !rr.transportFlux.converged + !rr.compactFlux.converged;
    S.tS3 += rr.timeTransportFlux;
    S.tPred += rr.timePredictor;
    S.tMu += rr.timeMu;
    S.tS5 += rr.timeProjection;
    S.tS6 += rr.timeCompactFlux;
    S.tBudget += rr.timeBudget;
    S.tSolveS3 += rr.timeSolveTransportFlux;
    S.tSolveS6 += rr.timeSolveCompactFlux;
    const double Bs[11] = {rr.B1, rr.B2,  rr.B3,  rr.B4, rr.B5,  rr.B6,
                           rr.B7, rr.B8a, rr.B8b, rr.B9, rr.Bres};
    for (int q = 0; q < 11; ++q)
      S.Bsum[q] += Bs[q];
    ++S.steps;
    S.maxU = std::max(S.maxU, (double)kry::maxScaled(st.U, DV(), 3 * N));
    std::vector<Real> Uh;
    if (r.field == "unif" || (r.force == "gbar" && r.field == "zero"))
      Uh = down(st.U);
    if (r.mode == ale::Projection::Damped) {
      S.omegaMu = std::max(S.omegaMu, rr.omega * muEx);
      if ((k + 1) % r.sample == 0 || k == 0) {
        S.muRatio = std::min(S.muRatio, rr.muHat / muEx);
        ++S.muSamples;
      }
    }
    if (r.field == "unif") {
      const double nU = std::sqrt(0.3 * 0.3 + 0.7 * 0.7 + 0.5 * 0.5);
      double d = 0;
      for (int i = 0; i < 3 * N; ++i)
        d = std::max(d, std::fabs(Uh[i] - U0[i]));
      S.unifDev = std::max(S.unifDev, d / nU);
    }
    if (r.force == "gbar") {
      if (r.field == "zero") {
        double d = 0;
        for (int i = 0; i < N; ++i)
          for (int q = 0; q < 3; ++q)
            d = std::max(d, std::fabs(Uh[3 * i + q] - gbar[q] * t));
        S.gtDev = std::max(S.gtDev, d / t);
      }
      for (int q = 0; q < 3; ++q) {
        const double mean = (rr.momentum[q] - P0[q]) / Mtot;
        const double meanB = (rr.momentum[q] - P0[q] + resMom[q]) / Mtot;
        S.meanDev = std::max(S.meanDev, std::fabs(mean - gbar[q] * t) / t);
        S.meanDevBooked = std::max(S.meanDevBooked, std::fabs(meanB - gbar[q] * t) / t);
      }
    }
    if (r.sample > 1 && (k + 1) % r.sample == 0)
      std::printf("step n=%d t=%.4f K=%.6e closure/K=%.2e excess/K=%+.2e mu_hat=%.5f omega=%.4f\n",
                  k + 1, t, rr.K1, std::fabs(rr.closure) / std::max(rr.K1, 1e-300),
                  (rr.dK - rr.unsignedSum) / std::max(rr.K1, 1e-300), rr.muHat, rr.omega);
    if (k == 0)
      std::printf(
          "first_step omega=%.6f mu_hat=%.6f lambda=%.6f pred_apps=%d S3_its=%d S5_its=%d "
          "S6_its=%d mu_its=%d\n",
          rr.omega, rr.muHat, rr.lambda, rr.predictor.applications, rr.transportFlux.iterations,
          rr.projection.iterations, rr.compactFlux.iterations, rr.muIterations);
    if (!std::isfinite(rr.K1)) {
      blowup = true;
      std::printf("RESULT case=%s status=BLOWUP step=%d\n", name.c_str(), k + 1);
      break;
    }
    if (k == 4) {
      const double el =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - tLoop0).count();
      if (el / 5 * nst + tSetup > 7200.0) {
        std::printf("RESULT case=%s status=TIMEOUT wall_per_step=%.3f projected=%.0f\n",
                    name.c_str(), el / 5, el / 5 * nst + tSetup);
        return 1;
      }
    }
  }
  if (traceScalars)
    std::fclose(traceScalars);
  if (blowup)
    return 1;
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - tLoop0).count() / nst;
  const double ns = (double)S.steps;
  std::ostringstream o;
  o.precision(6);
  o << std::scientific;
  o << "RESULT case=" << name << " status=DONE N=" << N << " tau=" << tau << " steps=" << nst
    << " t=" << tau * nst << " closure=" << S.closure << " mom=" << S.mom
    << " mom_booked=" << S.momBooked << " divPhi1=" << S.divPhi1 << " divF=" << S.divF
    << " cfl=" << S.cfl << " excess=" << S.excess << " pred_its_mean=" << S.predIts / ns
    << " pred_apps_mean=" << S.predApps / ns << " pred_apps_max=" << (double)S.predAppsMax
    << " pred_res=" << S.predRes << " projR_its_mean=" << S.projIts / ns
    << " projR_its_max=" << (double)S.projItsMax << " s3_its_mean=" << S.s3Its / ns
    << " s5_its_mean=" << S.projIts / ns << " s6_its_mean=" << S.s6Its / ns
    << " mu_its_per_solve=" << (S.muSolves ? (double)S.muIts / S.muSolves : 0.0)
    << " not_converged=" << (double)S.notConverged << " res_over_diss=" << S.resOverDiss
    << " wall_per_step=" << wall << " setup=" << tSetup << " frame_setup=" << fd.setupSeconds
    << " t_s3=" << S.tS3 / ns << " t_pred=" << S.tPred / ns << " t_mu=" << S.tMu / ns
    << " t_s5=" << S.tS5 / ns << " t_s6=" << S.tS6 / ns << " t_budget=" << S.tBudget / ns
    << " t_s3_solve=" << S.tSolveS3 / ns << " t_s6_solve=" << S.tSolveS6 / ns
    << " mu_exact=" << muEx << " maxU=" << S.maxU << " K0=" << K0 << " K1=" << Kprev;
  if (r.mode == ale::Projection::Damped)
    o << " omega_mu=" << S.omegaMu << " mu_ratio=" << S.muRatio << " mu_samples=" << S.muSamples;
  double parOut = 0, perpOut = 0;
  if (r.field == "tg" || r.field == "abc") {
    const auto Uh = down(st.U);
    // on a moving mesh the exact field at the generators' present positions, the volumes of the
    // present frame
    const auto xe = moving ? down(mm.pos) : xh;
    const auto Ve = moving ? down(mm.volume) : Vh;
    const auto ex = fieldOf(r.field, xe, N, nu, tau * nst);
    const double dec = r.field == "tg" ? std::exp(-2.0 * nu * kTwoPi * kTwoPi * tau * nst)
                                       : std::exp(-nu * kTwoPi * kTwoPi * tau * nst);
    double e = 0, ue = 0, ee = 0;
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d) {
        const double q = Uh[3 * i + d] - ex[3 * i + d];
        e += Ve[i] * q * q;
        ue += Ve[i] * q * ex[3 * i + d];
        ee += Ve[i] * (double)ex[3 * i + d] * ex[3 * i + d];
      }
    const double err = std::sqrt(e) / norm0 / dec;
    const double errPar = (ue / ee) * std::sqrt(ee) / norm0 / dec;
    o << " err=" << err;
    parOut = errPar;
    perpOut = std::sqrt(std::max(err * err - errPar * errPar, 0.0));
  }
  if (r.field == "unif")
    o << " unif_dev=" << S.unifDev;
  if (r.force == "psi_face") {
    const double gmax = kTwoPi / c.s.L[1];
    o << " maxU_rel_bound=" << S.maxU / std::sqrt(gmax * h);
  }
  if (r.force == "gbar")
    o << " gt_dev=" << S.gtDev << " mean_dev=" << S.meanDev
      << " mean_dev_booked=" << S.meanDevBooked;
  o << " courant=" << courant;
  if (r.field == "tg" || r.field == "abc")
    o << " err_par=" << parOut << " err_perp=" << perpOut;
  o << " stokes=" << (stokes ? 1 : 0);
  // the moving mesh (brief ALE_S1_3A): gcl = max over the run of A18 item 3's measure
  o << " motion=" << motion << " gcl=" << gcl << " s3_restarts=" << s3Restarts
    << " rebuilt=" << (double)rebuilt << " frames=" << (double)frames
    << " t_frames=" << tFrames / ns << " t_setup=" << tFrameSetup / ns << " B1_absmax=" << B1max
    << " mom_def=" << momDef;
  if (moving && mot == ale::MeshMotion::Rest)
    o << " v02_maxdiff=" << v02diff << " v02_bitwise_steps=" << (double)v02bitwise;
  const char* bn[11] = {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8a", "B8b", "B9", "Bres"};
  std::printf("TERMS %s run_sum", name.c_str());
  for (int q = 0; q < 11; ++q)
    std::printf(" %s=%+.6e", bn[q], S.Bsum[q]);
  std::printf("\n%s\n", o.str().c_str());
  return 0;
}

// ================================================================== --trace (C1–C3)
std::map<std::string, std::string> readMeta(const std::string& path) {
  std::map<std::string, std::string> kv;
  std::vector<char> b;
  if (!readFile(path, b)) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(2);
  }
  std::istringstream is(std::string(b.begin(), b.end()));
  std::string tok;
  while (is >> tok) {
    const auto e = tok.find('=');
    if (e != std::string::npos)
      kv[tok.substr(0, e)] = tok.substr(e + 1);
  }
  return kv;
}

int traceCase(const std::string& dir, const std::string& cs, const std::string& odir, int steps) {
  auto meta = readMeta(dir + "/" + cs + ".meta");
  const std::string mesh = meta["mesh"];
  const double tau = std::stod(meta["tau"]), rho = std::stod(meta["rho"]),
               nu = std::stod(meta["nu"]);
  const auto mode = meta["mode"] == "R" ? ale::Projection::Exact : ale::Projection::Damped;
  if (steps <= 0)
    steps = std::stoi(meta["steps"]);
  Scene s;
  Frame fr;
  if (!frameOfExport(dir, mesh, s, fr))
    return 1;
  const FM& m = fr.mesh;
  const int N = m.nCells;
  Fluid fl = fluidOf(m, rho, nu);
  ale::FrameData<Real> fd;
  ale::setupFrame(fd, m, fl.rho, fl.mu);
  ale::StepOptions opt;
  opt.predictorRtol = 1e-13;
  opt.laplacianRtol = 1e-12;
  ale::Stepper<Real> stp(opt);
  ale::FlowState<Real> st;
  st.allocate(m.nCombined);
  Kokkos::deep_copy(st.U, up(readVec<Real>(dir + "/" + cs + ".U0", 3 * (std::size_t)N), "U0"));
  Kokkos::deep_copy(st.p, up(readVec<Real>(dir + "/" + cs + ".p0", N), "p0"));
  const std::string o = odir + "/" + cs + ".dev_";
  FILE* sc = std::fopen((o + "scalars").c_str(), "w");
  for (int k = 1; k <= steps; ++k) {
    const auto r = stp.step(fd, st, Real(tau), mode);
    writeVec(o + "Ustar_" + std::to_string(k), down(stp.lastUstar()));
    writeVec(o + "F_" + std::to_string(k), down(stp.lastTransportFlux()));
    writeVec(o + "U_" + std::to_string(k), down(st.U));
    writeVec(o + "p_" + std::to_string(k), down(st.p));
    std::fprintf(sc, "%d %.17g %.17g %d %d %d %d %.3e %.3e\n", k, r.omega, r.muHat,
                 r.predictor.applications, r.transportFlux.iterations, r.projection.iterations,
                 r.compactFlux.iterations, r.closure / std::max(r.K1, 1e-300), r.divPhi1);
    if (k == 1)
      std::printf("TRACE %s step 1 omega=%.6f mu_hat=%.6f pred_apps=%d\n", cs.c_str(), r.omega,
                  r.muHat, r.predictor.applications);
  }
  std::fclose(sc);
  std::printf("TRACE %s ok steps=%d exec=%s threads=%d\n", cs.c_str(), steps, Exec::name(),
              Exec().concurrency());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int rc = 0;
  {
    std::string applyDir, outDir, which = "all", runCaseName, traceDir, traceCaseName;
    std::string motion = "none", traceMovingDir, traceMovingCase;
    int cells = 16, lapN = 0, steps = -1;
    double courant = 0.4, tend = -1;
    bool tight = false, stokes = false;
    for (int a = 1; a < argc; ++a) {
      const std::string s = argv[a];
      if (s == "--apply" && a + 1 < argc)
        applyDir = argv[++a];
      else if (s == "--out" && a + 1 < argc)
        outDir = argv[++a];
      else if (s == "--mesh" && a + 1 < argc)
        which = argv[++a];
      else if (s == "--cells" && a + 1 < argc)
        cells = std::atoi(argv[++a]);
      else if (s == "--laplacian" && a + 1 < argc)
        lapN = std::atoi(argv[++a]);
      else if (s == "--run" && a + 1 < argc)
        runCaseName = argv[++a];
      else if (s == "--trace" && a + 2 < argc) {
        traceDir = argv[++a];
        traceCaseName = argv[++a];
      } else if (s == "--steps" && a + 1 < argc)
        steps = std::atoi(argv[++a]);
      else if (s == "--courant" && a + 1 < argc)
        courant = std::atof(argv[++a]);
      else if (s == "--tight")
        tight = true;
      else if (s == "--stokes")
        stokes = true;
      else if (s == "--motion" && a + 1 < argc)
        motion = argv[++a];
      else if (s == "--pred-rtol" && a + 1 < argc)
        g_predRtol = std::atof(argv[++a]);
      else if (s == "--exact-rtol" && a + 1 < argc)
        g_exactRtol = std::atof(argv[++a]);
      else if (s == "--dump-on-error" && a + 1 < argc)
        g_dumpDir = argv[++a];
      else if (s == "--tend" && a + 1 < argc)
        tend = std::atof(argv[++a]);
      else if (s == "--trace-moving" && a + 2 < argc) {
        traceMovingDir = argv[++a];
        traceMovingCase = argv[++a];
      }
    }
    if (!applyDir.empty()) {
      if (outDir.empty())
        outDir = applyDir;
      for (const char* mesh : {"M1", "M2", "M3"})
        if (which == "all" || which == mesh)
          rc |= applyMesh(applyDir, outDir, std::string(mesh) + "_" + std::to_string(cells));
      std::printf("=== S1.2b apply %s ===\n", rc ? "FAIL" : "PASS");
    } else if (lapN > 0) {
      rc = laplacianGate(lapN);
    } else if (!traceMovingDir.empty()) {
      // 5 steps at the tight tolerances of --trace (default motion: follow)
      rc = runCase(traceMovingCase, steps > 0 ? steps : 5, true, courant, stokes,
                   motion == "none" ? std::string("follow") : motion, -1, traceMovingDir);
    } else if (!runCaseName.empty()) {
      rc = runCase(runCaseName, steps, tight, courant, stokes, motion, tend);
    } else if (!traceDir.empty()) {
      if (outDir.empty())
        outDir = traceDir;
      rc = traceCase(traceDir, traceCaseName, outDir, steps);
    } else {
      std::printf("=== S1.2b stepper identities: threads=%d exec=%s ===\n", Exec().concurrency(),
                  Exec::name());
      rc = runIdentities();
      std::printf("=== S1.2b identities %s ===\n", rc ? "FAIL" : "PASS");
    }
  }
  Kokkos::finalize();
  return rc;
}
