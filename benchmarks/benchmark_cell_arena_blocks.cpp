/**
 * @file benchmark_cell_arena_blocks.cpp
 * @brief Focused single-thread benchmark for CellArena reserve-capacity studies.
 *
 * Runs Voronoi build for random uniform points in a unit cube and reports timing
 * and CellArena memory footprint metrics for a single (N, reps, seed) setup.
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
#include <sys/resource.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <voronoi_dynamics/voronoi.hpp>

using Clock = std::chrono::high_resolution_clock;
using Millis = std::chrono::duration<double, std::milli>;
using Pos3d = vor::std::array<double, 3>;

struct TimingResult {
  double mean_ms;
  double std_ms;
  int reps;
};

struct ArenaBytes {
  std::size_t spans_size;
  std::size_t spans_capacity;
  std::size_t vertex_pos_size;
  std::size_t vertex_pos_capacity;
  std::size_t vertices_size;
  std::size_t vertices_capacity;
  std::size_t facets_size;
  std::size_t facets_capacity;
  std::size_t nbrs_size;
  std::size_t nbrs_capacity;

  std::size_t total_size() const {
    return spans_size + vertex_pos_size + vertices_size + facets_size + nbrs_size;
  }

  std::size_t total_capacity() const {
    return spans_capacity + vertex_pos_capacity + vertices_capacity + facets_capacity +
           nbrs_capacity;
  }
};

static inline double elapsed_ms(Clock::time_point t0) {
  return Millis(Clock::now() - t0).count();
}

template <typename Func>
static TimingResult time_function(Func &&fn, int reps) {
  fn();
  std::vector<double> times(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    auto t0 = Clock::now();
    fn();
    times[static_cast<std::size_t>(i)] = elapsed_ms(t0);
  }
  const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(reps);
  double var = 0.0;
  for (double t : times) {
    var += (t - mean) * (t - mean);
  }
  const double sd = (reps > 1) ? std::sqrt(var / static_cast<double>(reps - 1)) : 0.0;
  return {mean, sd, reps};
}

static std::vector<Pos3d> random_uniform(int n, double L, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, L);
  std::vector<Pos3d> pos(static_cast<std::size_t>(n));
  for (auto &p : pos) {
    p[0] = dist(rng);
    p[1] = dist(rng);
    p[2] = dist(rng);
  }
  return pos;
}

template <typename T>
static std::size_t vector_size_bytes(const std::vector<T> &v) {
  return v.size() * sizeof(T);
}

template <typename T>
static std::size_t vector_capacity_bytes(const std::vector<T> &v) {
  return v.capacity() * sizeof(T);
}

static ArenaBytes collect_arena_bytes(const vor::CellArena<double> &arena) {
  ArenaBytes out{};
  out.spans_size = vector_size_bytes(arena.spans());
  out.spans_capacity = vector_capacity_bytes(arena.spans());
  out.vertex_pos_size = vector_size_bytes(arena.vertexPos());
  out.vertex_pos_capacity = vector_capacity_bytes(arena.vertexPos());
  out.vertices_size = vector_size_bytes(arena.vertices());
  out.vertices_capacity = vector_capacity_bytes(arena.vertices());
  out.facets_size = vector_size_bytes(arena.facets());
  out.facets_capacity = vector_capacity_bytes(arena.facets());
  out.nbrs_size = vector_size_bytes(arena.nbrs());
  out.nbrs_capacity = vector_capacity_bytes(arena.nbrs());
  return out;
}

int main(int argc, char *argv[]) {
  int n = 10000;
  int reps = 12;
  std::uint64_t seed = 20260327ULL;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--n") == 0 && (i + 1) < argc) {
      n = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--reps") == 0 && (i + 1) < argc) {
      reps = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--seed") == 0 && (i + 1) < argc) {
      seed = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  if (n <= 0 || reps <= 0) {
    std::fprintf(stderr, "n and reps must be positive.\n");
    return 2;
  }

  constexpr double kL = 1.0;
  const auto pos = random_uniform(n, kL, seed);

  vor::std::array<double, 3> Lv;
  Lv[0] = kL;
  Lv[1] = kL;
  Lv[2] = kL;
  vor::Box<double> box(Lv);
  vor::CellComplex<double> complex(&box);

  const auto timing = time_function([&]() { complex.build(pos); }, reps);
  const auto &arena = complex.getCellArena();
  const ArenaBytes bytes = collect_arena_bytes(arena);

  const std::size_t total_size = bytes.total_size();
  const std::size_t total_capacity = bytes.total_capacity();
  const std::size_t slack = (total_capacity >= total_size) ? (total_capacity - total_size) : 0;
  const double slack_pct = (total_capacity > 0)
                               ? 100.0 * static_cast<double>(slack) / static_cast<double>(total_capacity)
                               : 0.0;

  struct rusage usage;
  long max_rss_kb = -1;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    max_rss_kb = usage.ru_maxrss;
  }

#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif

  std::printf("reserve_vertices,reserve_facets,N,reps,nthreads,time_ms_mean,time_ms_std,");
  std::printf("arena_size_bytes,arena_capacity_bytes,arena_slack_bytes,arena_slack_pct,");
  std::printf("spans_size_bytes,spans_capacity_bytes,");
  std::printf("vertex_pos_size_bytes,vertex_pos_capacity_bytes,");
  std::printf("vertices_size_bytes,vertices_capacity_bytes,");
  std::printf("facets_size_bytes,facets_capacity_bytes,");
  std::printf("nbrs_size_bytes,nbrs_capacity_bytes,max_rss_kb\n");

  std::printf("%u,%u,%d,%d,%d,%.6f,%.6f,",
              static_cast<unsigned>(vor::CellArena<double>::kReserveVerticesPerCell),
              static_cast<unsigned>(vor::CellArena<double>::kReserveFacetsPerCell),
              n,
              timing.reps,
              nthreads,
              timing.mean_ms,
              timing.std_ms);
  std::printf("%zu,%zu,%zu,%.6f,", total_size, total_capacity, slack, slack_pct);
  std::printf("%zu,%zu,", bytes.spans_size, bytes.spans_capacity);
  std::printf("%zu,%zu,", bytes.vertex_pos_size, bytes.vertex_pos_capacity);
  std::printf("%zu,%zu,", bytes.vertices_size, bytes.vertices_capacity);
  std::printf("%zu,%zu,", bytes.facets_size, bytes.facets_capacity);
  std::printf("%zu,%zu,%ld\n", bytes.nbrs_size, bytes.nbrs_capacity, max_rss_kb);

  return 0;
}
