/**
 * @file bench_mesh_optimizer.cpp
 * \brief Interstitial-packing volume-equalisation study: mesh THE PORE SPACE of a random sphere
 * packing with a Voronoi tessellation whose seeds sit only in the fluid, then drive the cell volumes
 * to be nearly equal by minimising E = Σ (V_i − V̄)² — and compare how efficiently the different
 * linear-solver / optimiser strategies do it.
 *
 * Pipeline: a peclet.dem periodic random-close packing (built by ../pack_bed.py → packing.txt) →
 * union-of-balls SDF (walls) → reject-sample N interstitial seeds → volume-variance minimisation.
 *
 * Two studies:
 *   PART 1 — inner linear solver for ONE Gauss-Newton step (the crux). Assemble g and the Gauss-
 *     Newton Hessian H once at the initial seeds, then solve H·dq = −g to a fixed tolerance with:
 *       plain CG · Jacobi-CG · symmetric-colored-GS-CG · single-level Chebyshev-CG ·
 *       AMG-CG (Jacobi smoother) · AMG-CG (Chebyshev smoother)   [AMG = peclet::core::solver::GraphAMG]
 *     reporting iterations + wall time, swept over N to expose the O(N) scaling (AMG flat, the
 *     single-level smoothers grow).
 *   PART 2 — end-to-end optimisation to a target volume spread: full Newton (Jacobi / colored-GS /
 *     AMG preconditioned CG, via meshVolumeOptimize) vs the first-order methods steepest descent and
 *     nonlinear (Polak–Ribière) CG. Wall time, outer iterations, final volume spread.
 *
 * Build: part of the voro Kokkos tests (PECLET_VORO_KOKKOS=ON). Run:
 *   ./bench_mesh_optimizer packing.txt [Npart-list...]    e.g.  ./bench_mesh_optimizer packing.txt 4000 16000 40000
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <Kokkos_Core.hpp>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "peclet/core/amr/momentum.hpp"      // greedyColoring / Coloring (colored-GS)
#include "peclet/core/solver/graph_amg.hpp"  // GraphAMG / HostCsrOp / amgPcg
#include "peclet/voro/mesh_optimizer.hpp"    // meshVolumeOptimize, Precond, buildTessellation, detail
#include "peclet/voro/sdf.hpp"

using Real = double;
using peclet::core::Index;
namespace solver = peclet::core::solver;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

namespace {

// Union-of-balls SDF with periodic min-image (box side L): sdf(x) = min_i(|x−c_i|_periodic − r_i);
// <0 inside a sphere, >0 in the fluid. Host provider (runs under the OpenMP exec space).
struct SdfSpheres {
  const Real* c;  // 3*n sphere centres, in [0,L)
  const Real* r;  // n radii
  int n;
  Real L;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    const Real Lh = Real(0.5) * L;
    Real m = Real(1e30);
    for (int i = 0; i < n; ++i) {
      Real dx = x - c[3 * i], dy = y - c[3 * i + 1], dz = z - c[3 * i + 2];
      dx -= L * Kokkos::round(dx / L);  // periodic min-image
      dy -= L * Kokkos::round(dy / L);
      dz -= L * Kokkos::round(dz / L);
      (void)Lh;
      const Real d = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) - r[i];
      if (d < m) m = d;
    }
    return m;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

struct Packing {
  std::vector<Real> c;  // 3*M
  std::vector<Real> r;  // M
  int M = 0;
  Real L = 1;
};

Packing readPacking(const char* path) {
  Packing p;
  std::FILE* f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open packing file '%s'\n", path);
    std::exit(1);
  }
  if (std::fscanf(f, "%d %lf", &p.M, &p.L) != 2) {
    std::fprintf(stderr, "bad packing header\n");
    std::exit(1);
  }
  p.c.resize(3 * p.M);
  p.r.resize(p.M);
  for (int i = 0; i < p.M; ++i)
    if (std::fscanf(f, "%lf %lf %lf %lf", &p.c[3 * i], &p.c[3 * i + 1], &p.c[3 * i + 2], &p.r[i]) !=
        4) {
      std::fprintf(stderr, "bad packing line %d\n", i);
      std::exit(1);
    }
  std::fclose(f);
  return p;
}

// Host union-SDF (periodic) for the rejection seeding — same formula as SdfSpheres::eval.
Real hostSdf(const Packing& p, Real x, Real y, Real z) {
  Real m = 1e30;
  for (int i = 0; i < p.M; ++i) {
    Real dx = x - p.c[3 * i], dy = y - p.c[3 * i + 1], dz = z - p.c[3 * i + 2];
    dx -= p.L * std::round(dx / p.L);
    dy -= p.L * std::round(dy / p.L);
    dz -= p.L * std::round(dz / p.L);
    m = std::min(m, std::sqrt(dx * dx + dy * dy + dz * dz) - p.r[i]);
  }
  return m;
}

// Reject-sample N seeds strictly inside the pore space (sdf > margin).
std::vector<Real> seedInterstitial(const Packing& p, int N, Real margin, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<Real> U(0.0, p.L);
  std::vector<Real> pos;
  pos.reserve(3 * N);
  long tried = 0;
  while ((int)(pos.size() / 3) < N) {
    Real x = U(rng), y = U(rng), z = U(rng);
    ++tried;
    if (hostSdf(p, x, y, z) > margin) {
      pos.push_back(x);
      pos.push_back(y);
      pos.push_back(z);
    }
  }
  std::printf("  seeded %d interstitial points (%.0f%% accept, margin=%.3g)\n", N,
              100.0 * N / (double)tried, margin);
  return pos;
}

// forward decls
struct Tess;
template <class Sdf>
Tess buildTess(const std::vector<Real>&, int, Real, int, const Sdf&,
               Kokkos::View<long*, peclet::core::MemSpace>&);

// --- tessellation build + assembly (mirrors meshVolumeOptimize's unweighted path) ------------------
struct Tess {
  std::vector<Real> vol, dvr;                 // cell volumes, per-facet ∂V/∂r (3*nFacet)
  std::vector<int> off, cnt;                  // per-cell facet offset / count
  std::vector<peclet::voro::gid_t> nbr;       // per-facet neighbour seed id
};

template <class Sdf>
Tess buildTess(const std::vector<Real>& pos, int N, Real L, int sw, const Sdf& sdf,
               Kokkos::View<long*, peclet::core::MemSpace>& gd) {
  using MemSpace = peclet::core::MemSpace;
  const Real Larr[3] = {L, L, L};
  Kokkos::View<Real*, MemSpace> dpos("bmo.pos", 3 * N), dw;
  Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(pos.data(), 3 * N));
  auto res = peclet::voro::buildTessellation<Real, false, Sdf>(dpos, dw, N, Larr, sw, N, gd, sdf,
                                                               true);
  Tess T;
  T.vol = peclet::voro::detail::toHostVec<Real>(res.view.cellVolume);
  T.dvr = peclet::voro::detail::toHostVec<Real>(res.view.facetConnect);
  T.off = peclet::voro::detail::toHostVecT<int>(res.view.cellFacetOffset);
  T.cnt = peclet::voro::detail::toHostVecT<int>(res.view.cellFacetCount);
  T.nbr = peclet::voro::detail::toHostVecT<peclet::voro::gid_t>(res.view.facetNeighbor);
  return T;
}

// Seed the pore space, then PRUNE seeds whose SDF-clipped cell is degenerate (V<=0) and re-tessellate
// until none remain (a seed in a tight pocket / too close to a wall gets no valid cell). This starts
// the optimiser from a clean nBad==0 state so all methods converge to the same minimum — the fair
// basis for the timing comparison. Over-seeds ~20% to land near the requested count after pruning.
template <class Sdf>
std::vector<Real> seedClean(const Packing& pk, int Nreq, Real margin, int sw, const Sdf& sdf,
                            Kokkos::View<long*, peclet::core::MemSpace>& gd, unsigned seed) {
  std::vector<Real> pos = seedInterstitial(pk, (int)(1.2 * Nreq), margin, seed);
  // Prune to a strictly-feasible set (every V > a margin fraction of the mean cell volume), so the
  // log-barrier optimiser can start from the interior. A plain V>0 cut leaves borderline cells that
  // flicker sign under the parallel tessellation and never fully clear; the margin removes them.
  for (int round = 0; round < 12; ++round) {
    int n = (int)(pos.size() / 3);
    Tess T = buildTess(pos, n, pk.L, sw, sdf, gd);
    double sV = 0;
    int m = 0;
    for (int c = 0; c < n; ++c)
      if (T.vol[c] > 0.0) {
        sV += T.vol[c];
        ++m;
      }
    const double thr = 0.25 * (m > 0 ? sV / m : 0.0);  // drop cells below 15% of the mean volume
    std::vector<Real> keep;
    keep.reserve(pos.size());
    for (int c = 0; c < n; ++c)
      if (T.vol[c] > thr) {
        keep.push_back(pos[3 * c]);
        keep.push_back(pos[3 * c + 1]);
        keep.push_back(pos[3 * c + 2]);
      }
    const int removed = n - (int)(keep.size() / 3);
    pos.swap(keep);
    if (removed == 0) break;
  }
  std::printf("  -> %d clean interstitial seeds after pruning small cells\n",
              (int)(pos.size() / 3));
  return pos;
}

// Sparse per-cell gradient stencil ∇V_c over the position DOFs (mirrors the optimiser).
void cellStencil(const Tess& T, int c, int N, std::vector<std::pair<int, double>>& st) {
  st.clear();
  const int nF = (int)T.nbr.size();
  double sx = 0, sy = 0, sz = 0;
  for (int f = T.off[c]; f < T.off[c] + T.cnt[c] && f < nF; ++f) {
    const long j = (long)T.nbr[f];
    if (j < 0 || j >= N) continue;
    const double dx = T.dvr[3 * f], dy = T.dvr[3 * f + 1], dz = T.dvr[3 * f + 2];
    st.emplace_back(3 * (int)j, dx);
    st.emplace_back(3 * (int)j + 1, dy);
    st.emplace_back(3 * (int)j + 2, dz);
    sx -= dx;
    sy -= dy;
    sz -= dz;
  }
  st.emplace_back(3 * c, sx);
  st.emplace_back(3 * c + 1, sy);
  st.emplace_back(3 * c + 2, sz);
}

// Objective E and gradient g (3N), no Hessian. nBad = cells with V<=0.
double gradEnergy(const Tess& T, int N, const std::vector<double>& vset, std::vector<double>& g,
                  long& nBad) {
  const double gamma = 1.0;
  g.assign(3 * N, 0.0);
  double E = 0;
  nBad = 0;
  std::vector<std::pair<int, double>> st;
  for (int c = 0; c < N; ++c) {
    const double e = T.vol[c] - vset[c];
    E += gamma * e * e;
    if (T.vol[c] <= 0.0) ++nBad;
    const double Rc = 2.0 * gamma * e;
    cellStencil(T, c, N, st);
    for (auto& [a, va] : st) g[a] += Rc * va;
  }
  return E;
}

// Assemble g (3N) and the Gauss-Newton Hessian H (HostCsrOp) at the current tessellation.
double assembleGH(const Tess& T, int N, const std::vector<double>& vset, std::vector<double>& g,
                  solver::HostCsrOp& H) {
  const double gamma = 1.0;
  const int nD = 3 * N;
  g.assign(nD, 0.0);
  std::vector<std::map<int, double>> row(nD);
  std::vector<std::pair<int, double>> st;
  double E = 0;
  for (int c = 0; c < N; ++c) {
    const double e = T.vol[c] - vset[c];
    E += gamma * e * e;
    const double Rc = 2.0 * gamma * e;
    cellStencil(T, c, N, st);
    for (auto& [a, va] : st) {
      g[a] += Rc * va;
      auto& ra = row[a];
      for (auto& [b, vb] : st) ra[b] += 2.0 * gamma * va * vb;
    }
  }
  H.n = nD;
  H.diag.assign(nD, 0.0);
  H.start.assign(nD + 1, 0);
  H.nbr.clear();
  H.coef.clear();
  for (int i = 0; i < nD; ++i) {
    for (auto& [j, v] : row[i]) {
      if (j == i)
        H.diag[i] = v;
      else {
        H.nbr.push_back(j);
        H.coef.push_back(v);
      }
    }
    H.start[i + 1] = (Index)H.nbr.size();
  }
  return E;
}

// PCG on H x = b with a generic preconditioner z = M⁻¹ r. Returns iterations; final rel-residual in
// *relres. tol is relative to ‖b − A x0‖ (x0 = 0).
int pcg(const solver::HostCsrOp& H, std::vector<double>& x, const std::vector<double>& b,
        const std::function<void(const std::vector<double>&, std::vector<double>&)>& prec,
        int maxit, double tol, double* relres) {
  const int n = (int)H.n;
  std::vector<double> r(b), z(n), p(n), Ap(n);
  std::fill(x.begin(), x.end(), 0.0);
  auto dot = [&](const std::vector<double>& u, const std::vector<double>& v) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += u[i] * v[i];
    return s;
  };
  const double r0 = std::sqrt(dot(r, r));
  if (r0 == 0) {
    *relres = 0;
    return 0;
  }
  prec(r, z);
  p = z;
  double rz = dot(r, z), rn = r0;
  int it = 0;
  for (; it < maxit; ++it) {
    H.apply(p, Ap);
    const double pAp = dot(p, Ap);
    if (pAp <= 0) break;
    const double a = rz / pAp;
    for (int i = 0; i < n; ++i) {
      x[i] += a * p[i];
      r[i] -= a * Ap[i];
    }
    rn = std::sqrt(dot(r, r));
    if (rn <= tol * r0) {
      ++it;
      break;
    }
    prec(r, z);
    const double rzn = dot(r, z);
    const double beta = rzn / rz;
    for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    rz = rzn;
  }
  *relres = rn / r0;
  return it;
}

// symmetric multicolour Gauss–Seidel preconditioner (forward+backward sweep, z=0 start).
struct SgsPrec {
  const solver::HostCsrOp* H;
  peclet::core::amr::Coloring col;
  std::vector<Index> colIdx;
  void operator()(const std::vector<double>& r, std::vector<double>& z) const {
    const int n = (int)H->n;
    z.assign(n, 0.0);
    auto sweep = [&](bool fwd) {
      for (int ci = 0; ci < col.nColors; ++ci) {
        const int cc = fwd ? ci : col.nColors - 1 - ci;
        for (Index t = col.hStart[cc]; t < col.hStart[cc + 1]; ++t) {
          const Index i = colIdx[t];
          double s = r[i];
          for (Index k = H->start[i]; k < H->start[i + 1]; ++k)
            s -= H->coef[k] * z[H->nbr[k]];
          if (std::fabs(H->diag[i]) > 1e-30) z[i] = s / H->diag[i];
        }
      }
    };
    sweep(true);
    sweep(false);
  }
};

// Volume variance ⟨(V − ⟨V⟩)²⟩ over the real cells — the equalisation objective; → 0 as cells equalise.
double volumeSpread(const Tess& T, int N) {
  double s = 0, s2 = 0;
  int m = 0;
  for (int c = 0; c < N; ++c) {
    if (T.vol[c] <= 0) continue;
    s += T.vol[c];
    s2 += T.vol[c] * T.vol[c];
    ++m;
  }
  const double mean = s / m;
  return std::max(0.0, s2 / m - mean * mean);
}

}  // namespace

int main(int argc, char** argv) {
  const char* pack = (argc > 1) ? argv[1] : "packing.txt";
  std::vector<int> Ns;
  for (int i = 2; i < argc; ++i) Ns.push_back(std::atoi(argv[i]));
  if (Ns.empty()) Ns = {4000, 16000, 40000};
  const int sw = 6;
  const double cgTol = 1e-8;
  const int cgMax = 3000;

  Kokkos::initialize(argc, argv);
  {
    Packing pk = readPacking(pack);
    const Real L = pk.L, boxVol = L * L * L;
    std::printf("packing: M=%d spheres  side=%.3f  meanR=%.4f\n", pk.M, L,
                std::accumulate(pk.r.begin(), pk.r.end(), 0.0) / pk.M);
    // device-visible sphere arrays for the SDF provider (host pointers ok under the OpenMP backend)
    SdfSpheres sdf{pk.c.data(), pk.r.data(), pk.M, L};
    Kokkos::View<long*, peclet::core::MemSpace> gd;
    const Real meanR = std::accumulate(pk.r.begin(), pk.r.end(), 0.0) / pk.M;
    const Real margin = 0.05 * meanR;

    // --------- diagnostic: is the SDF pore case STALLING or CONVERGED? Trace it, and compare to the
    // same seeds tessellated with NO walls (pure Voronoi, which equalises to ~0). --------------------
    bool trace = false;
    for (int i = 2; i < argc; ++i)
      if (std::string(argv[i]) == "--trace") trace = true;
    if (trace) {
      const int N = 3000;
      auto seeds = seedClean(pk, N, margin, sw, sdf, gd, 777u);
      const int Nc = (int)(seeds.size() / 3);
      std::vector<Real> vset(Nc, boxVol / Nc), noW;
      std::printf("\n---- NoSdf (no walls, pure Voronoi of the same seeds), Newton+AMG, verbose ----\n");
      std::vector<Real> p1 = seeds;
      peclet::voro::meshVolumeOptimize<Real, false, peclet::voro::NoSdf>(
          p1, noW, vset, (Real[3]){L, L, L}, Nc, sw, peclet::voro::NoSdf{}, 40, 1e-9, 400,
          peclet::voro::Precond::GraphAMG, true, 0.0, 0.7, /*freeEnergy=*/true);
      {
        Tess T = buildTess(p1, Nc, L, sw, peclet::voro::NoSdf{}, gd);
        std::printf("   NoSdf final variance = %.3e\n", volumeSpread(T, Nc));
      }
      // Graded reference volume V_ref = clamp(sdf, sLo, sHi)³ (relative; renormalised inside): small
      // cells hugging the walls (inflation layer), growing to a capped bulk size — and it makes the
      // naturally-small wall cells ON-TARGET so the log-barrier no longer fights them.
      std::vector<Real> vsetG(Nc);
      for (int i = 0; i < Nc; ++i) {
        double phi = hostSdf(pk, seeds[3 * i], seeds[3 * i + 1], seeds[3 * i + 2]);
        phi = std::min(0.35, std::max(0.06, phi));
        vsetG[i] = (Real)(phi * phi * phi);
      }
      (void)vsetG;
      std::printf("\n---- SDF (pore walls), FREE ENERGY −V_ref·log(V), steepest descent, verbose ----\n");
      std::vector<Real> p2 = seeds;
      peclet::voro::meshVolumeOptimize<Real, false, SdfSpheres>(
          p2, noW, vset, (Real[3]){L, L, L}, Nc, sw, sdf, 200, 1e-9, 400,
          peclet::voro::Precond::SteepestDescent, true, 0.0, 0.7, /*freeEnergy=*/true);
      {
        Tess T = buildTess(p2, Nc, L, sw, sdf, gd);
        std::printf("   SDF final variance = %.3e\n", volumeSpread(T, Nc));
      }
      Kokkos::finalize();
      return 0;
    }

    // =========================== PART 1 — inner solver for one Newton step =======================
    std::printf(
        "\n================ PART 1: one Gauss-Newton step, inner solve H·dq=−g to rel %.0e "
        "================\n",
        cgTol);
    std::printf("%8s | %-22s | %6s | %9s | %s\n", "N", "method", "iters", "solve[s]", "notes");
    std::printf("---------+------------------------+--------+-----------+-----------------------\n");
    for (int N : Ns) {
      auto pos = seedClean(pk, N, margin, sw, sdf, gd, 12345u);
      std::vector<double> vset(N, boxVol / N);
      Tess T = buildTess(pos, N, L, sw, sdf, gd);
      std::vector<double> g;
      solver::HostCsrOp H;
      auto ta = clk::now();
      assembleGH(T, N, vset, g, H);
      const double tAsm = secs(ta, clk::now());
      std::vector<double> b(3 * N);
      for (int i = 0; i < 3 * N; ++i) b[i] = -g[i];
      const int nD = 3 * N;

      auto run = [&](const char* name,
                     const std::function<void(const std::vector<double>&, std::vector<double>&)>& M) {
        std::vector<double> x(nD, 0.0);
        double rr;
        auto t0 = clk::now();
        int it = pcg(H, x, b, M, cgMax, cgTol, &rr);
        const double t = secs(t0, clk::now());
        std::printf("%8d | %-22s | %6d | %9.3f | rel=%.1e\n", N, name, it, t, rr);
      };
      auto runAmg = [&](const char* name, solver::AmgParams ap) {
        solver::GraphAMG M;
        auto tb = clk::now();
        M.build(H, ap);
        const double tbuild = secs(tb, clk::now());
        std::vector<double> x(nD, 0.0);
        auto t0 = clk::now();
        auto R = solver::amgPcg(H, x, b, M, cgMax, cgTol);
        const double t = secs(t0, clk::now());
        std::printf("%8d | %-22s | %6d | %9.3f | build %.2fs, %d lvls, opC %.2f\n", N, name, R.iters,
                    t, tbuild, M.numLevels(), M.operatorComplexity());
      };

      // plain CG (identity preconditioner)
      run("CG (no precond)", [&](const std::vector<double>& r, std::vector<double>& z) { z = r; });
      // Jacobi
      run("Jacobi-CG", [&](const std::vector<double>& r, std::vector<double>& z) {
        z.resize(nD);
        for (int i = 0; i < nD; ++i) z[i] = std::fabs(H.diag[i]) > 1e-30 ? r[i] / H.diag[i] : r[i];
      });
      // symmetric colored Gauss-Seidel
      {
        SgsPrec sgs;
        sgs.H = &H;
        sgs.col = peclet::core::amr::greedyColoring(H.start, H.nbr, (Index)nD);
        sgs.colIdx = peclet::voro::detail::toHostVecT<Index>(sgs.col.idx);
        run("colored-GS-CG", [&](const std::vector<double>& r, std::vector<double>& z) { sgs(r, z); });
      }
      // single-level Chebyshev polynomial preconditioner (no coarse grid: maxLevels=1 + smoother-only)
      {
        solver::AmgParams ap;
        ap.ndofPerNode = 3;
        ap.maxLevels = 1;
        ap.coarseSweeps = 1;
        ap.chebDegree = 2;
        runAmg("Chebyshev-CG (1 level)", ap);
      }
      // AMG with a damped-Jacobi (4th-kind degree-1) smoother
      {
        solver::AmgParams ap;
        ap.ndofPerNode = 3;
        ap.chebDegree = 1;
        ap.pre = ap.post = 2;
        runAmg("AMG-CG (Jacobi smooth)", ap);
      }
      // AMG (Chebyshev smoother) — the default multigrid
      {
        solver::AmgParams ap;
        ap.ndofPerNode = 3;
        ap.chebDegree = 2;
        runAmg("AMG-CG (Cheby smooth)", ap);
      }
      std::printf("%8d | %-22s |        | %9.3f |\n", N, "(assembly of H)", tAsm);
      std::printf(
          "---------+------------------------+--------+-----------+-----------------------\n");
    }

    // =========================== PART 2 — end-to-end optimisation ================================
    const int Nend = Ns.front();
    std::printf(
        "\n================ PART 2: end-to-end volume equalisation, N=%d ================\n", Nend);
    std::printf("target: reduce volume variance <(V-<V>)^2>; tol on ‖g‖_inf = 1e-9, max 60 steps\n");
    std::printf("%-26s | %6s | %10s | %12s\n", "method", "outer", "wall[s]", "variance");
    std::printf("---------------------------+--------+------------+----------------\n");

    auto pos0 = seedClean(pk, Nend, margin, sw, sdf, gd, 777u);
    std::vector<Real> vset(Nend, boxVol / Nend);
    {
      Tess T0 = buildTess(pos0, Nend, L, sw, sdf, gd);
      std::vector<double> g0;
      long nBad0 = 0;
      double E0 = gradEnergy(T0, Nend, vset, g0, nBad0);
      double gmax = 0;
      for (double v : g0) gmax = std::max(gmax, std::fabs(v));
      std::printf("%-26s | %6s | %10s | %12.3e  (nBad=%ld  E=%.3e  |g|inf=%.3e)\n",
                  "initial (random seeds)", "-", "-", volumeSpread(T0, Nend), nBad0, E0, gmax);
    }

    // Newton–Raphson with the three CG preconditioners (via the production optimiser).
    for (auto pc : {std::pair<const char*, peclet::voro::Precond>{"Newton  Jacobi-CG",
                                                                  peclet::voro::Precond::Jacobi},
                    {"Newton  colored-GS-CG", peclet::voro::Precond::ColoredGS},
                    {"Newton  AMG-CG", peclet::voro::Precond::GraphAMG}}) {
      std::vector<Real> pos = pos0, noW;
      auto t0 = clk::now();
      auto R = peclet::voro::meshVolumeOptimize<Real, false, SdfSpheres>(
          pos, noW, vset, (Real[3]){L, L, L}, Nend, sw, sdf, 60, 1e-9, 400, pc.second, false);
      const double t = secs(t0, clk::now());
      Tess Tf = buildTess(pos, Nend, L, sw, sdf, gd);
      std::printf("%-26s | %6d | %10.2f | %12.3e\n", pc.first, R.iters, t, volumeSpread(Tf, Nend));
    }

    // First-order methods (steepest descent + nonlinear Polak–Ribière CG) with Armijo line search.
    auto firstOrder = [&](const char* name, bool nlcg) {
      std::vector<Real> pos = pos0;
      Tess T = buildTess(pos, Nend, L, sw, sdf, gd);
      std::vector<double> g, gPrev, d(3 * Nend, 0.0);
      long nBadCur;
      double E = gradEnergy(T, Nend, vset, g, nBadCur);
      auto t0 = clk::now();
      int it = 0;
      const int maxIt = 400;
      for (; it < maxIt; ++it) {
        double gnorm = 0;
        for (double v : g) gnorm = std::max(gnorm, std::fabs(v));
        if (gnorm < 1e-9) break;
        if (!nlcg || it == 0) {
          for (int i = 0; i < 3 * Nend; ++i) d[i] = -g[i];
        } else {
          double num = 0, den = 0;
          for (int i = 0; i < 3 * Nend; ++i) {
            num += g[i] * (g[i] - gPrev[i]);
            den += gPrev[i] * gPrev[i];
          }
          const double beta = std::max(0.0, num / std::max(den, 1e-300));
          for (int i = 0; i < 3 * Nend; ++i) d[i] = -g[i] + beta * d[i];
        }
        double gDotD = 0;
        for (int i = 0; i < 3 * Nend; ++i) gDotD += g[i] * d[i];
        if (gDotD > 0) {  // not a descent direction — reset to steepest descent
          for (int i = 0; i < 3 * Nend; ++i) d[i] = -g[i];
          gDotD = 0;
          for (int i = 0; i < 3 * Nend; ++i) gDotD += g[i] * d[i];
        }
        const double spacing = std::cbrt(boxVol / Nend);
        double dmax = 0;
        for (double v : d) dmax = std::max(dmax, std::fabs(v));
        double alpha = std::min(1.0, 0.5 * spacing / std::max(dmax, 1e-30));
        bool acc = false;
        std::vector<Real> xt(3 * Nend);
        Tess Tt;
        std::vector<double> gt;
        double Et = E;
        for (int bt = 0; bt < 30; ++bt) {
          for (int i = 0; i < 3 * Nend; ++i) xt[i] = pos[i] + (Real)(alpha * d[i]);
          Tt = buildTess(xt, Nend, L, sw, sdf, gd);  // gd = the reusable Kokkos scratch view
          long nb;
          Et = gradEnergy(Tt, Nend, vset, gt, nb);
          if (std::isfinite(Et) && Et <= E + 1e-4 * alpha * gDotD) {
            acc = true;
            nBadCur = nb;
            break;
          }
          alpha *= 0.5;
        }
        if (!acc) break;
        pos = xt;
        T = Tt;
        gPrev = g;
        g = gt;
        E = Et;
      }
      const double t = secs(t0, clk::now());
      std::printf("%-26s | %6d | %10.2f | %12.3e\n", name, it, t, volumeSpread(T, Nend));
    };
    firstOrder("steepest descent", false);
    firstOrder("nonlinear CG (PR)", true);
  }
  Kokkos::finalize();
  return 0;
}
