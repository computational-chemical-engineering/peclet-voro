/**
 * @file benchmark_voronoi.cpp
 * @brief Wall-time benchmark comparing voronoi_dynamics against Voro++ for
 *        construction of a static 3-D periodic Voronoi tessellation.
 *
 * Two timing categories are measured for each library:
 *  - **tess**  : pure tessellation (neighbour list + half-plane cutting /
 *                container filling). No geometric properties are computed.
 *  - **full**  : tessellation + all geometric properties
 *                (voronoi_dynamics: CellGeometry::buildGeometry;
 *                 Voro++: compute_cell for every particle).
 *
 * Three point-set distributions are exercised:
 *  - **random_uniform**  : N uniform-random points in a unit cube [0,1)³
 *  - **cubic_lattice**   : simple-cubic lattice with N = n³ points + ε jitter
 *  - **sphere_surface**  : N uniform-random points on a sphere of radius 0.4
 *                           centred at (0.5, 0.5, 0.5) inside the unit cube
 *
 * Output (stdout): CSV with columns
 *     library, point_set, N, nthreads, reps, time_ms_mean, time_ms_std
 * Progress and metadata are written to stderr.
 *
 * Usage:
 *     ./benchmark_voronoi [--include-sphere] [output.csv]
 *
 * By default, the expensive sphere_surface dataset is skipped.
 * Pass --include-sphere to include it.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

// voronoi_dynamics public headers
#include <voronoi_dynamics/nbrlist.hpp>
#include <voronoi_dynamics/voronoi.hpp>

// Voro++ header
#include <voro++.hh>

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using Clock  = std::chrono::high_resolution_clock;
using Millis = std::chrono::duration<double, std::milli>;
using Pos3d  = vor::std::array<double, 3>;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/** @brief Return milliseconds elapsed since @p t0. */
static inline double elapsed_ms(Clock::time_point t0) {
  return Millis(Clock::now() - t0).count();
}

struct TimingResult {
  double mean_ms;  ///< arithmetic mean over all timed repetitions
  double std_ms;   ///< sample standard deviation (0 if reps == 1)
  int    reps;     ///< number of timed repetitions (warm-up not counted)
};

/**
 * @brief Time a callable @p fn over @p reps repetitions after one warm-up.
 * @tparam Func  Zero-argument callable.
 * @param fn     Function to benchmark.
 * @param reps   Number of timed repetitions (≥ 1).
 * @return       TimingResult with mean and std.
 */
template <typename Func>
static TimingResult time_function(Func&& fn, int reps) {
  // warm-up: ensures allocations, cache fills, and voro++ internal structures
  // are not counted in the first timed run.
  fn();

  std::vector<double> times(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    auto t0 = Clock::now();
    fn();
    times[static_cast<std::size_t>(i)] = elapsed_ms(t0);
  }

  const double mean = std::accumulate(times.begin(), times.end(), 0.0)
                      / static_cast<double>(reps);
  double var = 0.0;
  for (double t : times)
    var += (t - mean) * (t - mean);
  const double sd = (reps > 1) ? std::sqrt(var / static_cast<double>(reps - 1)) : 0.0;

  return {mean, sd, reps};
}

/** @brief Choose repetition count based on N so the total wall time stays bounded. */
static int nreps(int N) {
  if (N <=     500) return 20;
  if (N <=    2000) return 10;
  if (N <=   10000) return  5;
  if (N <=  100000) return  3;
  if (N <=  500000) return  2;
  return 1;
}

// ---------------------------------------------------------------------------
// Point-set generators
// ---------------------------------------------------------------------------

/**
 * @brief Generate N uniform-random points in [0, L)³.
 * @param N     Number of points.
 * @param L     Box side length.
 * @param seed  RNG seed.
 */
static std::vector<Pos3d> random_uniform(int N, double L, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, L);
  std::vector<Pos3d> pos(static_cast<std::size_t>(N));
  for (auto& p : pos) {
    p[0] = dist(rng);
    p[1] = dist(rng);
    p[2] = dist(rng);
  }
  return pos;
}

/**
 * @brief Generate N = n³ points on a simple-cubic lattice with ε jitter.
 *
 * The jitter (amplitude 1e-7 × spacing) breaks exact co-plane degeneracies
 * that would otherwise place vertices exactly on half-plane boundaries.
 * The Voronoi structure is practically identical to the perfect SC lattice.
 *
 * @param N     Desired number of points; the nearest perfect cube is used.
 * @param L     Box side length.
 * @param seed  RNG seed for jitter.
 * @return      Vector of n³ positions; size may differ slightly from N.
 */
static std::vector<Pos3d> cubic_lattice(int N, double L, uint64_t seed) {
  const int n  = static_cast<int>(std::round(std::cbrt(static_cast<double>(N))));
  const double dx = L / static_cast<double>(n);
  const double jitter_amp = 1.0e-7 * dx;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> jitter(-jitter_amp, jitter_amp);

  std::vector<Pos3d> pos;
  pos.reserve(static_cast<std::size_t>(n * n * n));
  for (int ix = 0; ix < n; ++ix)
    for (int iy = 0; iy < n; ++iy)
      for (int iz = 0; iz < n; ++iz) {
        Pos3d p;
        p[0] = (static_cast<double>(ix) + 0.5) * dx + jitter(rng);
        p[1] = (static_cast<double>(iy) + 0.5) * dx + jitter(rng);
        p[2] = (static_cast<double>(iz) + 0.5) * dx + jitter(rng);
        pos.push_back(p);
      }
  return pos;
}

/**
 * @brief Generate N uniform-random points on a sphere of radius @p R
 *        centred at the midpoint of the unit cube.
 *
 * Uses the Marsaglia normalisation trick: sample 3 independent standard
 * normals, normalise to the unit sphere, then scale and translate.
 *
 * @param N     Number of points.
 * @param L     Box side length (sphere radius = 0.4 × L).
 * @param seed  RNG seed.
 */
static std::vector<Pos3d> sphere_surface(int N, double L, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> ndist(0.0, 1.0);

  const double R  = 0.4 * L;
  const double cx = 0.5 * L;

  std::vector<Pos3d> pos(static_cast<std::size_t>(N));
  for (auto& p : pos) {
    double x, y, z, r;
    // Resample on extremely unlikely zero-vector draws
    do {
      x = ndist(rng);
      y = ndist(rng);
      z = ndist(rng);
      r = std::sqrt(x * x + y * y + z * z);
    } while (r < 1.0e-15);
    p[0] = cx + R * x / r;
    p[1] = cx + R * y / r;
    p[2] = cx + R * z / r;
  }
  return pos;
}

// ---------------------------------------------------------------------------
// voronoi_dynamics benchmark functions
// ---------------------------------------------------------------------------

/**
 * @brief voronoi_dynamics tessellation only (no CellGeometry).
 *
 * Replicates the logic of CellComplex::build but omits the buildGeometry call,
 * so only the NbrList setup and parallel CellMaker loop are timed.
 *
 * @param pos  Particle positions.
 * @param box  Periodic box.
 */
static void vd_tess_only(const std::vector<Pos3d>& pos, vor::Box<double>& box) {
  const auto& L = box.getL();
  const double density = static_cast<double>(pos.size()) / (L[0] * L[1] * L[2]);
  const double rcut    = 1.75 * std::pow(density, -1.0 / 3.0);

  vor::NbrList<vor::uint2, double> nbrList(&box);
  nbrList.setup(pos, rcut);

  const vor::Cuboid<double> cub(L);
  std::vector<vor::Cell<double>> cells(pos.size());

#pragma omp parallel
  {
    vor::CellMaker<double> maker;
#pragma omp for schedule(dynamic, 64)
    for (vor::uint2 i = 0; i < static_cast<vor::uint2>(pos.size()); ++i) {
      maker.build(i, pos, nbrList, cub);
      cells[i] = maker;
    }
  }

  nbrList.clear();  // free the cell-linked list (mirrors CellComplex::build)
}

/**
 * @brief voronoi_dynamics full build: tessellation + CellGeometry.
 *
 * Calls CellComplex::build which internally runs initNbrList, the parallel
 * CellMaker loop, and buildGeometry (connecting vectors, edge inverses,
 * volume derivatives).
 *
 * @param pos  Particle positions.
 * @param box  Periodic box.
 */
static void vd_full_build(const std::vector<Pos3d>& pos, vor::Box<double>& box) {
  vor::CellComplex<double> cx(&box);
  cx.build(pos);
  // CellComplex stores the result; destructs at end of scope.
}

// ---------------------------------------------------------------------------
// Voro++ benchmark functions
// ---------------------------------------------------------------------------

/**
 * @brief Voro++ tessellation only: container setup + particle insertion.
 *
 * No compute_cell calls are made; this measures the cost of allocating the
 * periodic container and binning all particles into the cell grid.
 *
 * @param pos  Particle positions.
 * @param Lx   Box length in x.
 * @param Ly   Box length in y (== Lx for a cube).
 * @param Lz   Box length in z (== Lx for a cube).
 */
static void voropp_tess_only(const std::vector<Pos3d>& pos,
                              double Lx, double Ly, double Lz) {
  const int n   = static_cast<int>(pos.size());
  const int nbl = std::max(1, static_cast<int>(std::cbrt(static_cast<double>(n) / 8.0)));

  // Orthogonal periodic box: bxy=bxz=byz=0
  voro::container_periodic con(Lx, 0, Ly, 0, 0, Lz, nbl, nbl, nbl, 8);
  for (int i = 0; i < n; ++i)
    con.put(i, pos[static_cast<std::size_t>(i)][0],
               pos[static_cast<std::size_t>(i)][1],
               pos[static_cast<std::size_t>(i)][2]);
  // Container destructs here, releasing all memory.
}

/**
 * @brief Voro++ full build: tessellation + compute_cell for all particles.
 *
 * Iterates over all particles with c_loop_all_periodic and calls
 * compute_cell, which computes the full Voronoi cell geometry.
 *
 * @param pos  Particle positions.
 * @param Lx   Box length in x.
 * @param Ly   Box length in y.
 * @param Lz   Box length in z.
 */
static void voropp_full_build(const std::vector<Pos3d>& pos,
                               double Lx, double Ly, double Lz) {
  const int n   = static_cast<int>(pos.size());
  const int nbl = std::max(1, static_cast<int>(std::cbrt(static_cast<double>(n) / 8.0)));

  voro::container_periodic con(Lx, 0, Ly, 0, 0, Lz, nbl, nbl, nbl, 8);
  for (int i = 0; i < n; ++i)
    con.put(i, pos[static_cast<std::size_t>(i)][0],
               pos[static_cast<std::size_t>(i)][1],
               pos[static_cast<std::size_t>(i)][2]);

  voro::voronoicell_neighbor c(con);
  voro::c_loop_all_periodic vl(con);
  if (vl.start())
    do {
      con.compute_cell(c, vl);
    } while (vl.inc());
}

// ---------------------------------------------------------------------------
// Single benchmark case runner
// ---------------------------------------------------------------------------

/**
 * @brief Run all four timing variants for a given particle configuration and
 *        print one CSV row per variant to @p out.
 *
 * @param out           Output file (stdout in typical use).
 * @param point_set     Label string (e.g., "random_uniform").
 * @param pos           Particle positions.
 * @param L             Box side length.
 * @param nthreads      Number of OpenMP threads (for metadata).
 * @param run_vd        If false, skip voronoi_dynamics timing (voro++ only).
 */
static void run_case(FILE*                      out,
                     const char*                point_set,
                     const std::vector<Pos3d>&  pos,
                     double                     L,
                     int                        nthreads,
                     bool                       run_vd = true) {
  const int N    = static_cast<int>(pos.size());
  const int reps = nreps(N);

  vor::std::array<double, 3> Lv;
  Lv[0] = Lv[1] = Lv[2] = L;
  vor::Box<double> box(Lv);

  fprintf(stderr, "  [%s  N=%-8d] ", point_set, N);
  fflush(stderr);

  // voronoi_dynamics timings (only if run_vd is true)
  if (run_vd) {
    auto t_vd_tess = time_function([&]() { vd_tess_only(pos, box);  }, reps);
    fprintf(stderr, "vd_tess ");  fflush(stderr);
    auto t_vd_full = time_function([&]() { vd_full_build(pos, box); }, reps);
    fprintf(stderr, "vd_full ");  fflush(stderr);
    fprintf(out, "voronoi_dynamics_tess,%s,%d,%d,%d,%.5f,%.5f\n",
      point_set, N, nthreads, reps, t_vd_tess.mean_ms, t_vd_tess.std_ms);
    fprintf(out, "voronoi_dynamics_full,%s,%d,%d,%d,%.5f,%.5f\n",
      point_set, N, nthreads, reps, t_vd_full.mean_ms, t_vd_full.std_ms);
  }

  // Voro++ timings (always run)
  auto t_vpp_tess = time_function([&]() { voropp_tess_only(pos, L, L, L); }, reps);
  fprintf(stderr, "vpp_tess ");  fflush(stderr);
  auto t_vpp_full = time_function([&]() { voropp_full_build(pos, L, L, L); }, reps);
  fprintf(stderr, "vpp_full  done\n");  fflush(stderr);

  fprintf(out, "voropp_tess,%s,%d,%d,%d,%.5f,%.5f\n",
    point_set, N, nthreads, reps, t_vpp_tess.mean_ms, t_vpp_tess.std_ms);
  fprintf(out, "voropp_full,%s,%d,%d,%d,%.5f,%.5f\n",
    point_set, N, nthreads, reps, t_vpp_full.mean_ms, t_vpp_full.std_ms);
  fflush(out);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  bool include_sphere = false;
  const char* out_name = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--include-sphere") == 0) {
      include_sphere = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::fprintf(stderr,
                   "Usage: %s [--include-sphere] [output.csv]\n"
                   "  --include-sphere   include sphere_surface benchmark cases\n"
                   "  output.csv         optional output CSV path (default: stdout)\n",
                   argv[0]);
      return 0;
    } else if (argv[i][0] == '-') {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      return 1;
    } else if (out_name == nullptr) {
      out_name = argv[i];
    } else {
      std::fprintf(stderr, "Unexpected positional argument: %s\n", argv[i]);
      return 1;
    }
  }

  // Optional: redirect CSV output to a named file
  FILE* out = stdout;
  if (out_name != nullptr && std::strcmp(out_name, "-") != 0) {
    out = std::fopen(out_name, "w");
    if (!out) {
      std::perror(out_name);
      return 1;
    }
  }

  // ── Thread count ────────────────────────────────────────────────────────
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif

  // ── Metadata header (CSV comments starting with '#') ────────────────────
  fprintf(out, "# voronoi_dynamics vs Voro++ – static tessellation benchmark\n");
  fprintf(out, "# Compiled: %s %s\n", __DATE__, __TIME__);
  fprintf(out, "# OpenMP threads: %d\n", nthreads);
  fprintf(out, "# Box: unit cube [0,1)^3\n");
  fprintf(out, "# Sphere radius: 0.4 (voronoi_dynamics NOT benchmarked: hollow interior creates >128-facet cells for any N)\n");
  fprintf(out, "# include_sphere: %s\n", include_sphere ? "true" : "false");
  fprintf(out, "# Columns: library,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");
  fprintf(out, "library,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");
  fflush(out);

  constexpr double L    = 1.0;
  constexpr uint64_t SEED = 20260327ULL;

  // ── random_uniform ───────────────────────────────────────────────────────
  fprintf(stderr, "\n=== random_uniform ===\n");
  for (int N : {100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000}) {
    auto pos = random_uniform(N, L, SEED);
    run_case(out, "random_uniform", pos, L, nthreads);
  }

  // ── cubic_lattice ────────────────────────────────────────────────────────
  // Sizes are exact cubes; actual N used is n³ (may differ slightly from target).
  fprintf(stderr, "\n=== cubic_lattice ===\n");
  for (int N : {125, 1000, 8000, 27000, 64000, 125000, 512000, 1000000}) {
    auto pos = cubic_lattice(N, L, SEED);
    run_case(out, "cubic_lattice", pos, L, nthreads);
  }

  // ── sphere_surface (optional) ────────────────────────────────────────────
  if (include_sphere) {
    // voronoi_dynamics is NOT benchmarked for sphere_surface.
    // The hollow sphere interior creates cells with more than 128 facets for
    // ALL tested N values (including N=100), overflowing the library's fixed-size
    // static arrays.  Voro++ (no cell-complexity limit) is timed for all sizes.
    fprintf(stderr, "\n=== sphere_surface (voronoi_dynamics skipped entirely) ===\n");
    constexpr int VD_SPHERE_MAX = 0;  // 0 = never run vd for sphere_surface
    // NOTE: Voro++ exhibits O(N^2) cost for the hollow sphere distribution
    // (empty interior causes each cell to check many more candidate cut planes).
    // N > 10000 would take > 1 hour; the study is therefore capped at N = 10000.
    for (int N : {100, 500, 1000, 5000, 10000}) {
      const bool run_vd = (N <= VD_SPHERE_MAX);
      if (!run_vd)
        fprintf(stderr, "    (voronoi_dynamics skipped: N > %d)\n", VD_SPHERE_MAX);
      run_case(out, "sphere_surface", sphere_surface(N, L, SEED), L, nthreads, run_vd);
    }
  } else {
    fprintf(stderr, "\n=== sphere_surface skipped (use --include-sphere to enable) ===\n");
  }

  fprintf(stderr, "\nBenchmark complete.\n");

  if (out != stdout)
    std::fclose(out);

  return 0;
}
