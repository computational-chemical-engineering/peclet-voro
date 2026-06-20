/**
 * @file benchmark_nbrlist_setup_diagnostic.cpp
 * @brief Diagnostic benchmark for OpenMP/runtime overhead and NbrList::setup subphases.
 *
 * Focuses on:
 *  - random_uniform, N=10000
 *  - cubic_lattice, N=8000
 *
 * Benchmarks:
 *  - omp_empty_region
 *  - omp_three_barriers
 *  - setup_index_only
 *  - setup_histogram_only
 *  - setup_scatter_only
 *  - setup_full_decomposed
 *  - setup_library
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vorflow/nbrlist.hpp>

using Clock = std::chrono::high_resolution_clock;
using Millis = std::chrono::duration<double, std::milli>;
using Pos3d = std::array<double, 3>;
using UInt = vor::uint2;

struct TimingResult {
  double mean_ms;
  double std_ms;
  int reps;
};

static inline double elapsed_ms(Clock::time_point t0) {
  return Millis(Clock::now() - t0).count();
}

template <typename Func>
static TimingResult time_function(Func &&fn, int reps) {
  fn();
  std::vector<double> times(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
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

static constexpr int kDiagReps = 20;

static std::vector<Pos3d> random_uniform(int N, double L, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, L);
  std::vector<Pos3d> pos(static_cast<std::size_t>(N));
  for (auto &p : pos) {
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

static int runtime_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

static vor::Indx compute_grid_shape(double L, double rcut) {
  vor::Indx n;
  const UInt nk = static_cast<UInt>(std::floor(L / rcut));
  n[0] = nk;
  n[1] = nk;
  n[2] = nk;
  return n;
}

static inline UInt compute_cell_index(const Pos3d &pos,
                                      const std::array<double, 3> &invL,
                                      const vor::Indx &n) {
  UInt indx = 0;
  for (vor::uint0 k = 0; k < 3; ++k) {
    double r = pos[k] * invL[k];
    r -= std::floor(r);
    indx *= static_cast<UInt>(n[k]);
    indx += static_cast<UInt>(std::floor(r * n[k]));
  }
  return indx;
}

static void omp_empty_region() {
#pragma omp parallel
  {
  }
}

static void omp_three_barriers() {
#pragma omp parallel
  {
#pragma omp barrier
#pragma omp barrier
#pragma omp barrier
  }
}

static void setup_index_only(const std::vector<Pos3d> &pos,
                             const std::array<double, 3> &invL,
                             const vor::Indx &n) {
  std::vector<UInt> indx(pos.size());
#pragma omp parallel for
  for (UInt i = 0; i < pos.size(); ++i)
    indx[i] = compute_cell_index(pos[i], invL, n);
}

static void build_indices(const std::vector<Pos3d> &pos,
                          const std::array<double, 3> &invL,
                          const vor::Indx &n,
                          std::vector<UInt> &indx) {
  indx.resize(pos.size());
#pragma omp parallel for
  for (UInt i = 0; i < pos.size(); ++i)
    indx[i] = compute_cell_index(pos[i], invL, n);
}

static void setup_histogram_only(const std::vector<UInt> &indx, size_t numCells) {
  const size_t numPos = indx.size();
  const int numThreads = runtime_threads();

  if (numThreads <= 1) {
    std::vector<UInt> counts(numCells, 0);
    for (size_t i = 0; i < numPos; ++i)
      ++counts[indx[i]];
    return;
  }

  std::vector<UInt> localCounts(static_cast<size_t>(numThreads) * numCells, 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(numThreads)
  {
    const int tid = omp_get_thread_num();
    UInt *const counts = localCounts.data() + static_cast<size_t>(tid) * numCells;
    const size_t begin = (numPos * static_cast<size_t>(tid)) / static_cast<size_t>(numThreads);
    const size_t end = (numPos * static_cast<size_t>(tid + 1)) / static_cast<size_t>(numThreads);
    for (size_t i = begin; i < end; ++i)
      ++counts[indx[i]];
  }
#endif
}

static void make_offsets(const std::vector<UInt> &indx,
                         size_t numCells,
                         std::vector<UInt> &headCell,
                         std::vector<UInt> &localCounts) {
  const size_t numPos = indx.size();
  const int numThreads = runtime_threads();

  if (numThreads <= 1) {
    std::vector<UInt> counts(numCells, 0);
    for (size_t i = 0; i < numPos; ++i)
      ++counts[indx[i]];
    headCell.resize(numCells + 1);
    headCell[0] = 0;
    std::partial_sum(counts.begin(), counts.end(), headCell.begin() + 1);
    localCounts = headCell;
    return;
  }

  localCounts.assign(static_cast<size_t>(numThreads) * numCells, 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(numThreads)
  {
    const int tid = omp_get_thread_num();
    UInt *const counts = localCounts.data() + static_cast<size_t>(tid) * numCells;
    const size_t begin = (numPos * static_cast<size_t>(tid)) / static_cast<size_t>(numThreads);
    const size_t end = (numPos * static_cast<size_t>(tid + 1)) / static_cast<size_t>(numThreads);
    for (size_t i = begin; i < end; ++i)
      ++counts[indx[i]];
  }
#endif

  std::vector<UInt> counts(numCells, 0);
  for (size_t cell = 0; cell < numCells; ++cell) {
    UInt total = 0;
    for (int tid = 0; tid < numThreads; ++tid)
      total += localCounts[static_cast<size_t>(tid) * numCells + cell];
    counts[cell] = total;
  }

  headCell.resize(numCells + 1);
  headCell[0] = 0;
  std::partial_sum(counts.begin(), counts.end(), headCell.begin() + 1);

  for (size_t cell = 0; cell < numCells; ++cell) {
    UInt head = headCell[cell];
    for (int tid = 0; tid < numThreads; ++tid) {
      UInt &offset = localCounts[static_cast<size_t>(tid) * numCells + cell];
      const UInt count = offset;
      offset = head;
      head += count;
    }
  }
}

static void setup_scatter_only(const std::vector<Pos3d> &pos,
                               const std::vector<UInt> &indx,
                               const std::array<double, 3> &L,
                               const std::array<double, 3> &invL,
                               size_t numCells) {
  const size_t numPos = pos.size();
  const int numThreads = runtime_threads();
  std::vector<UInt> headCell;
  std::vector<UInt> localCounts;
  make_offsets(indx, numCells, headCell, localCounts);

  std::vector<vor::PosAndId<UInt, double> > cell2Pos(numPos);

  if (numThreads <= 1) {
    for (size_t i = 0; i < numPos; ++i) {
      const UInt head = localCounts[indx[i]]++;
      cell2Pos[head].id = static_cast<UInt>(i);
      cell2Pos[head].pos = pos[i];
      for (vor::uint0 k = 0; k < 3; ++k)
        cell2Pos[head].pos[k] -= L[k] * std::floor(cell2Pos[head].pos[k] * invL[k]);
    }
    return;
  }

#ifdef _OPENMP
#pragma omp parallel num_threads(numThreads)
  {
    const int tid = omp_get_thread_num();
    UInt *const offsets = localCounts.data() + static_cast<size_t>(tid) * numCells;
    const size_t begin = (numPos * static_cast<size_t>(tid)) / static_cast<size_t>(numThreads);
    const size_t end = (numPos * static_cast<size_t>(tid + 1)) / static_cast<size_t>(numThreads);
    for (size_t i = begin; i < end; ++i) {
      const UInt head = offsets[indx[i]]++;
      cell2Pos[head].id = static_cast<UInt>(i);
      cell2Pos[head].pos = pos[i];
      for (vor::uint0 k = 0; k < 3; ++k)
        cell2Pos[head].pos[k] -= L[k] * std::floor(cell2Pos[head].pos[k] * invL[k]);
    }
  }
#endif
}

static void setup_full_decomposed(const std::vector<Pos3d> &pos,
                                  const std::array<double, 3> &L,
                                  const std::array<double, 3> &invL,
                                  const vor::Indx &n,
                                  size_t numCells) {
  std::vector<UInt> indx;
  build_indices(pos, invL, n, indx);

  std::vector<UInt> headCell;
  std::vector<UInt> localCounts;
  make_offsets(indx, numCells, headCell, localCounts);

  std::vector<vor::PosAndId<UInt, double> > cell2Pos(pos.size());
  const int numThreads = runtime_threads();
  const size_t numPos = pos.size();

  if (numThreads <= 1) {
    for (size_t i = 0; i < numPos; ++i) {
      const UInt head = localCounts[indx[i]]++;
      cell2Pos[head].id = static_cast<UInt>(i);
      cell2Pos[head].pos = pos[i];
      for (vor::uint0 k = 0; k < 3; ++k)
        cell2Pos[head].pos[k] -= L[k] * std::floor(cell2Pos[head].pos[k] * invL[k]);
    }
    return;
  }

#ifdef _OPENMP
#pragma omp parallel num_threads(numThreads)
  {
    const int tid = omp_get_thread_num();
    UInt *const offsets = localCounts.data() + static_cast<size_t>(tid) * numCells;
    const size_t begin = (numPos * static_cast<size_t>(tid)) / static_cast<size_t>(numThreads);
    const size_t end = (numPos * static_cast<size_t>(tid + 1)) / static_cast<size_t>(numThreads);
    for (size_t i = begin; i < end; ++i) {
      const UInt head = offsets[indx[i]]++;
      cell2Pos[head].id = static_cast<UInt>(i);
      cell2Pos[head].pos = pos[i];
      for (vor::uint0 k = 0; k < 3; ++k)
        cell2Pos[head].pos[k] -= L[k] * std::floor(cell2Pos[head].pos[k] * invL[k]);
    }
  }
#endif
}

static void setup_library(const std::vector<Pos3d> &pos,
                          const std::array<double, 3> &L,
                          double rcut) {
  vor::Box<double> box(L);
  vor::NbrList<UInt, double> nbrList(&box);
  nbrList.setup(pos, rcut);
}

static void write_row(FILE *out,
                      const char *benchmark,
                      const char *dataset,
                      int N,
                      int nthreads,
                      const TimingResult &timing) {
  std::fprintf(out,
               "%s,%s,%d,%d,%d,%.5f,%.5f\n",
               benchmark,
               dataset,
               N,
               nthreads,
               timing.reps,
               timing.mean_ms,
               timing.std_ms);
}

static void run_case(FILE *out,
                     const char *dataset,
                     const std::vector<Pos3d> &pos,
                     double L,
                     int nthreads) {
  std::array<double, 3> Lv = {L, L, L};
  std::array<double, 3> invL = {1.0 / L, 1.0 / L, 1.0 / L};
  const double density = static_cast<double>(pos.size()) / (L * L * L);
  const double rcut = 1.75 * std::pow(density, -1.0 / 3.0);
  const vor::Indx n = compute_grid_shape(L, rcut);
  vor::Grid<UInt> grid;
  grid.init(n);
  const size_t numCells = static_cast<size_t>(grid.numCells());
  std::vector<UInt> indx;
  build_indices(pos, invL, n, indx);

  std::fprintf(stderr, "  [%s N=%-6zu threads=%-2d] ", dataset, pos.size(), nthreads);
  std::fflush(stderr);

  const TimingResult t_empty = time_function([&]() { omp_empty_region(); }, kDiagReps);
  std::fprintf(stderr, "empty ");
  std::fflush(stderr);

  const TimingResult t_barriers = time_function([&]() { omp_three_barriers(); }, kDiagReps);
  std::fprintf(stderr, "barriers ");
  std::fflush(stderr);

  const TimingResult t_index =
      time_function([&]() { setup_index_only(pos, invL, n); }, kDiagReps);
  std::fprintf(stderr, "index ");
  std::fflush(stderr);

  const TimingResult t_hist =
      time_function([&]() { setup_histogram_only(indx, numCells); }, kDiagReps);
  std::fprintf(stderr, "hist ");
  std::fflush(stderr);

  const TimingResult t_scatter =
      time_function([&]() { setup_scatter_only(pos, indx, Lv, invL, numCells); }, kDiagReps);
  std::fprintf(stderr, "scatter ");
  std::fflush(stderr);

  const TimingResult t_decomp =
      time_function([&]() { setup_full_decomposed(pos, Lv, invL, n, numCells); }, kDiagReps);
  std::fprintf(stderr, "decomp ");
  std::fflush(stderr);

  const TimingResult t_lib =
      time_function([&]() { setup_library(pos, Lv, rcut); }, kDiagReps);
  std::fprintf(stderr, "library done\n");
  std::fflush(stderr);

  write_row(out, "omp_empty_region", dataset, static_cast<int>(pos.size()), nthreads, t_empty);
  write_row(out, "omp_three_barriers", dataset, static_cast<int>(pos.size()), nthreads, t_barriers);
  write_row(out, "setup_index_only", dataset, static_cast<int>(pos.size()), nthreads, t_index);
  write_row(out, "setup_histogram_only", dataset, static_cast<int>(pos.size()), nthreads, t_hist);
  write_row(out, "setup_scatter_only", dataset, static_cast<int>(pos.size()), nthreads, t_scatter);
  write_row(out, "setup_full_decomposed", dataset, static_cast<int>(pos.size()), nthreads, t_decomp);
  write_row(out, "setup_library", dataset, static_cast<int>(pos.size()), nthreads, t_lib);
}

int main(int argc, char *argv[]) {
  const char *out_name = nullptr;
  if (argc >= 2)
    out_name = argv[1];

  FILE *out = stdout;
  if (out_name != nullptr && std::strcmp(out_name, "-") != 0) {
    out = std::fopen(out_name, "w");
    if (!out) {
      std::perror(out_name);
      return 1;
    }
  }

  const int nthreads = runtime_threads();

  std::fprintf(out, "# NbrList setup diagnostic benchmark\n");
  std::fprintf(out, "# Compiled: %s %s\n", __DATE__, __TIME__);
  std::fprintf(out, "# OpenMP max threads at runtime: %d\n", nthreads);
  std::fprintf(out, "# Cases: random_uniform N=10000, cubic_lattice N=8000\n");
  std::fprintf(out, "# Columns: benchmark,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");
  std::fprintf(out, "benchmark,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");

  constexpr double L = 1.0;
  constexpr uint64_t SEED = 20260329ULL;

  std::fprintf(stderr, "\n=== nbrlist diagnostic ===\n");
  run_case(out, "random_uniform", random_uniform(10000, L, SEED), L, nthreads);
  run_case(out, "cubic_lattice", cubic_lattice(8000, L, SEED), L, nthreads);

  if (out != stdout)
    std::fclose(out);
  return 0;
}
