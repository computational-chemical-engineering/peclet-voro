/**
 * @file benchmark_full_build.cpp
 * @brief Benchmark basic static Voronoi build: vd CellComplex::build(false) vs Voro++ compute_cell.
 *
 * Comparison modes:
 *  - default: vorflow basic build vs Voro++ compute_cell
 *  - --vd-only: only vorflow basic build, useful for thread-scaling sweeps
 *
 * Datasets:
 *  - random_uniform
 *  - cubic_lattice
 *
 * Sphere case is intentionally excluded in this benchmark.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vorflow/voronoi.hpp>
#include <voro++.hh>

using Clock = std::chrono::high_resolution_clock;
using Millis = std::chrono::duration<double, std::milli>;
using Pos3d = std::array<double, 3>;

struct TimingResult {
  double mean_ms;
  double std_ms;
  int reps;
};

static inline double elapsed_ms(Clock::time_point t0) {
  return Millis(Clock::now() - t0).count();
}

template <typename Func>
static TimingResult time_function(Func&& fn, int reps) {
  fn();
  std::vector<double> times(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    auto t0 = Clock::now();
    fn();
    times[static_cast<std::size_t>(i)] = elapsed_ms(t0);
  }
  const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(reps);
  double var = 0.0;
  for (double t : times)
    var += (t - mean) * (t - mean);
  const double sd = (reps > 1) ? std::sqrt(var / static_cast<double>(reps - 1)) : 0.0;
  return {mean, sd, reps};
}

static int nreps(int N) {
  if (N <= 500)
    return 20;
  if (N <= 2000)
    return 10;
  if (N <= 10000)
    return 5;
  if (N <= 100000)
    return 3;
  if (N <= 500000)
    return 2;
  return 1;
}

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

static std::vector<Pos3d> cubic_lattice(int N, double L, uint64_t seed) {
  const int n = static_cast<int>(std::round(std::cbrt(static_cast<double>(N))));
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

static void voropp_compute_cell(const std::vector<Pos3d>& pos, double L) {
  const int n = static_cast<int>(pos.size());
  const int nbl = std::max(1, static_cast<int>(std::cbrt(static_cast<double>(n) / 8.0)));

  voro::container_periodic con(L, 0, L, 0, 0, L, nbl, nbl, nbl, 8);
  for (int i = 0; i < n; ++i)
    con.put(i,
            pos[static_cast<std::size_t>(i)][0],
            pos[static_cast<std::size_t>(i)][1],
            pos[static_cast<std::size_t>(i)][2]);

  voro::voronoicell_neighbor c(con);
  voro::c_loop_all_periodic vl(con);
  if (vl.start())
    do {
      con.compute_cell(c, vl);
    } while (vl.inc());
}

static void run_case(FILE* out,
                     const char* point_set,
                     const std::vector<Pos3d>& pos,
                     double L,
                     int nthreads,
                     int worker_count,
                     bool vd_only) {
  const int N = static_cast<int>(pos.size());
  const int reps = nreps(N);
  std::array<double, 3> Lv;
  Lv[0] = Lv[1] = Lv[2] = L;
  vor::Box<double> box(Lv);
  vor::CellComplex<double> cx(&box, static_cast<size_t>(std::max(worker_count, 0)));

  std::fprintf(stderr, "  [%s  N=%-8d] ", point_set, N);
  std::fflush(stderr);

  const auto t_vd = time_function([&]() { cx.build(pos, false); }, reps);
  std::fprintf(stderr, "vd_build_basic ");
  std::fflush(stderr);

  std::fprintf(out,
               "vorflow_build_basic,%s,%d,%d,%d,%.5f,%.5f\n",
               point_set,
               N,
               nthreads,
               reps,
               t_vd.mean_ms,
               t_vd.std_ms);

  if (!vd_only) {
    const auto t_vpp = time_function([&]() { voropp_compute_cell(pos, L); }, reps);
    std::fprintf(stderr, "vpp_compute_cell done\n");
    std::fflush(stderr);

    std::fprintf(out,
                 "voropp_compute_cell,%s,%d,%d,%d,%.5f,%.5f\n",
                 point_set,
                 N,
                 nthreads,
                 reps,
                 t_vpp.mean_ms,
                 t_vpp.std_ms);
  } else {
    std::fprintf(stderr, "done\n");
    std::fflush(stderr);
  }

  std::fflush(out);
}

int main(int argc, char* argv[]) {
  bool vd_only = false;
  const char* out_name = nullptr;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--vd-only") == 0)
      vd_only = true;
    else
      out_name = argv[i];
  }

  FILE* out = stdout;
  if (out_name != nullptr && std::strcmp(out_name, "-") != 0) {
    out = std::fopen(out_name, "w");
    if (!out) {
      std::perror(out_name);
      return 1;
    }
  }

#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
  const int worker_count = nthreads;
#else
  const int nthreads = 1;
  const int worker_count = 0;
#endif

  std::fprintf(out, "# Basic-build benchmark (no sphere)\n");
  std::fprintf(out, "# Compiled: %s %s\n", __DATE__, __TIME__);
  std::fprintf(out, "# OpenMP max threads at runtime: %d\n", nthreads);
  std::fprintf(out, "# Columns: library,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");
  std::fprintf(out, "library,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");

  constexpr double L = 1.0;
  constexpr uint64_t SEED = 20260330ULL;

  std::fprintf(stderr, "\n=== random_uniform ===\n");
  for (int N : {100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000})
    run_case(out, "random_uniform", random_uniform(N, L, SEED), L, nthreads, worker_count, vd_only);

  std::fprintf(stderr, "\n=== cubic_lattice ===\n");
  for (int N : {125, 1000, 8000, 27000, 64000, 125000, 512000, 1000000})
    run_case(out, "cubic_lattice", cubic_lattice(N, L, SEED), L, nthreads, worker_count, vd_only);

  std::fprintf(stderr, "\nBenchmark complete.\n");

  if (out != stdout)
    std::fclose(out);

  return 0;
}
