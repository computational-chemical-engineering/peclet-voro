/**
 * @file benchmark_cell_creation_targeted.cpp
 * @brief Targeted cell-creation benchmark for low-N OpenMP scaling diagnostics.
 *
 * Focuses on:
 *  - random_uniform, N=10000
 *  - cubic_lattice, N=8000
 *
 * Benchmarks:
 *  - vd_setup_only
 *  - vd_cells_dynamic64
 *  - vd_cells_static
 *  - vd_total_dynamic64
 *  - vd_total_static
 *  - vd_total_dynamic64_second_of_two
 *  - vd_total_static_second_of_two
 *  - vd_total_dynamic64_persistent_second_of_two
 *  - vd_total_static_persistent_second_of_two
 *
 * Intended use:
 *  - build once with OpenMP enabled and sweep OMP_NUM_THREADS
 *  - optionally build once with OpenMP disabled for a no-OpenMP baseline
 */

#include <algorithm>
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

#include <vorflow/nbrlist.hpp>
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

enum class CellSchedule {
  Dynamic64,
  Static,
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

template <typename Func>
static TimingResult time_second_of_two(Func &&fn, int reps) {
  fn();
  fn();
  std::vector<double> times(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    fn();
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

static constexpr int kTargetReps = 12;

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

static double compute_rcut(const std::vector<Pos3d> &pos, const vor::Box<double> &box) {
  const auto &L = box.getL();
  const double density = static_cast<double>(pos.size()) / (L[0] * L[1] * L[2]);
  return 1.75 * std::pow(density, -1.0 / 3.0);
}

static void vd_setup_only(const std::vector<Pos3d> &pos, vor::Box<double> &box) {
  const double rcut = compute_rcut(pos, box);
  vor::NbrList<vor::uint2, double> nbr_list(&box);
  nbr_list.setup(pos, rcut);
}

static void vd_cells_only(const std::vector<Pos3d> &pos,
                          const vor::NbrList<vor::uint2, double> &nbr_list,
                          vor::Box<double> &box,
                          CellSchedule schedule) {
  const auto &L = box.getL();
  const vor::Cuboid<double> cub(L);
  std::vector<vor::Cell<double> > cells(pos.size());

  if (schedule == CellSchedule::Dynamic64) {
#pragma omp parallel
    {
      vor::ConstructionArena<double> arena;
      vor::CellMaker<double> maker(arena);
#pragma omp for schedule(dynamic, 64)
      for (vor::uint2 i = 0; i < static_cast<vor::uint2>(pos.size()); ++i) {
        maker.build(i, pos, nbr_list, cub);
        cells[i] = maker;
      }
    }
  } else {
#pragma omp parallel
    {
      vor::ConstructionArena<double> arena;
      vor::CellMaker<double> maker(arena);
#pragma omp for schedule(static)
      for (vor::uint2 i = 0; i < static_cast<vor::uint2>(pos.size()); ++i) {
        maker.build(i, pos, nbr_list, cub);
        cells[i] = maker;
      }
    }
  }
}

static void vd_total(const std::vector<Pos3d> &pos, vor::Box<double> &box, CellSchedule schedule) {
  const double rcut = compute_rcut(pos, box);
  vor::NbrList<vor::uint2, double> nbr_list(&box);
  nbr_list.setup(pos, rcut);
  vd_cells_only(pos, nbr_list, box, schedule);
  nbr_list.clear();
}

static void vd_cells_only_current_team(const std::vector<Pos3d> &pos,
                                       const vor::NbrList<vor::uint2, double> &nbr_list,
                                       const vor::Cuboid<double> &cub,
                                       CellSchedule schedule,
                                       std::vector<vor::Cell<double> > &cells,
                                       vor::CellMaker<double> &maker) {
  if (schedule == CellSchedule::Dynamic64) {
#pragma omp for schedule(dynamic, 64)
    for (vor::uint2 i = 0; i < static_cast<vor::uint2>(pos.size()); ++i) {
      maker.build(i, pos, nbr_list, cub);
      cells[i] = maker;
    }
  } else {
#pragma omp for schedule(static)
    for (vor::uint2 i = 0; i < static_cast<vor::uint2>(pos.size()); ++i) {
      maker.build(i, pos, nbr_list, cub);
      cells[i] = maker;
    }
  }
}

static void vd_total_current_team(const std::vector<Pos3d> &pos,
                                  double rcut,
                                  vor::NbrList<vor::uint2, double> &nbr_list,
                                  const vor::Cuboid<double> &cub,
                                  CellSchedule schedule,
                                  std::vector<vor::Cell<double> > &cells,
                                  vor::CellMaker<double> &maker) {
  nbr_list.setupCurrentTeam(pos, rcut);
  vd_cells_only_current_team(pos, nbr_list, cub, schedule, cells, maker);
#pragma omp barrier
#pragma omp single
  { nbr_list.clear(); }
#pragma omp barrier
}

static TimingResult time_persistent_second_of_two_total(const std::vector<Pos3d> &pos,
                                                        vor::Box<double> &box,
                                                        CellSchedule schedule,
                                                        int reps) {
#ifndef VORONOI_USE_OPENMP
  return time_second_of_two([&]() { vd_total(pos, box, schedule); }, reps);
#else
  const double rcut = compute_rcut(pos, box);
  const auto &L = box.getL();
  const vor::Cuboid<double> cub(L);
  vor::NbrList<vor::uint2, double> nbr_list(&box);
  std::vector<vor::Cell<double> > cells(pos.size());
  std::vector<double> times(static_cast<std::size_t>(reps));
  Clock::time_point t0;

#pragma omp parallel
  {
    vor::ConstructionArena<double> arena;
    vor::CellMaker<double> maker(arena);

    vd_total_current_team(pos, rcut, nbr_list, cub, schedule, cells, maker);
    vd_total_current_team(pos, rcut, nbr_list, cub, schedule, cells, maker);

    for (int i = 0; i < reps; ++i) {
      vd_total_current_team(pos, rcut, nbr_list, cub, schedule, cells, maker);

#pragma omp single
      { t0 = Clock::now(); }
#pragma omp barrier

      vd_total_current_team(pos, rcut, nbr_list, cub, schedule, cells, maker);

#pragma omp single
      { times[static_cast<std::size_t>(i)] = elapsed_ms(t0); }
#pragma omp barrier
    }
  }

  const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(reps);
  double var = 0.0;
  for (double t : times)
    var += (t - mean) * (t - mean);
  const double sd = (reps > 1) ? std::sqrt(var / static_cast<double>(reps - 1)) : 0.0;
  return {mean, sd, reps};
#endif
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
  std::array<double, 3> Lv;
  Lv[0] = Lv[1] = Lv[2] = L;
  vor::Box<double> box(Lv);

  const double rcut = compute_rcut(pos, box);
  vor::NbrList<vor::uint2, double> nbr_list(&box);
  nbr_list.setup(pos, rcut);

  std::fprintf(stderr, "  [%s N=%-6zu threads=%-2d] ", dataset, pos.size(), nthreads);
  std::fflush(stderr);

  const TimingResult t_setup = time_function([&]() { vd_setup_only(pos, box); }, kTargetReps);
  std::fprintf(stderr, "setup ");
  std::fflush(stderr);

  const TimingResult t_cells_dynamic =
      time_function([&]() { vd_cells_only(pos, nbr_list, box, CellSchedule::Dynamic64); }, kTargetReps);
  std::fprintf(stderr, "cells_dyn ");
  std::fflush(stderr);

  const TimingResult t_cells_static =
      time_function([&]() { vd_cells_only(pos, nbr_list, box, CellSchedule::Static); }, kTargetReps);
  std::fprintf(stderr, "cells_static ");
  std::fflush(stderr);

  const TimingResult t_total_dynamic =
      time_function([&]() { vd_total(pos, box, CellSchedule::Dynamic64); }, kTargetReps);
  std::fprintf(stderr, "total_dyn ");
  std::fflush(stderr);

  const TimingResult t_total_static =
      time_function([&]() { vd_total(pos, box, CellSchedule::Static); }, kTargetReps);
  std::fprintf(stderr, "total_static ");
  std::fflush(stderr);

  const TimingResult t_total_dynamic_second = time_second_of_two(
      [&]() { vd_total(pos, box, CellSchedule::Dynamic64); }, kTargetReps);
  std::fprintf(stderr, "total_dyn_second ");
  std::fflush(stderr);

  const TimingResult t_total_static_second =
      time_second_of_two([&]() { vd_total(pos, box, CellSchedule::Static); }, kTargetReps);
  std::fprintf(stderr, "total_static_second ");
  std::fflush(stderr);

  const TimingResult t_total_dynamic_persistent_second =
      time_persistent_second_of_two_total(pos, box, CellSchedule::Dynamic64, kTargetReps);
  std::fprintf(stderr, "total_dyn_persistent_second ");
  std::fflush(stderr);

  const TimingResult t_total_static_persistent_second =
      time_persistent_second_of_two_total(pos, box, CellSchedule::Static, kTargetReps);
  std::fprintf(stderr, "total_static done\n");
  std::fflush(stderr);

  write_row(out, "vd_setup_only", dataset, static_cast<int>(pos.size()), nthreads, t_setup);
  write_row(out, "vd_cells_dynamic64", dataset, static_cast<int>(pos.size()), nthreads, t_cells_dynamic);
  write_row(out, "vd_cells_static", dataset, static_cast<int>(pos.size()), nthreads, t_cells_static);
  write_row(out, "vd_total_dynamic64", dataset, static_cast<int>(pos.size()), nthreads, t_total_dynamic);
  write_row(out, "vd_total_static", dataset, static_cast<int>(pos.size()), nthreads, t_total_static);
  write_row(out, "vd_total_dynamic64_second_of_two", dataset, static_cast<int>(pos.size()), nthreads,
            t_total_dynamic_second);
  write_row(out, "vd_total_static_second_of_two", dataset, static_cast<int>(pos.size()), nthreads,
            t_total_static_second);
  write_row(out, "vd_total_dynamic64_persistent_second_of_two", dataset, static_cast<int>(pos.size()),
            nthreads, t_total_dynamic_persistent_second);
  write_row(out, "vd_total_static_persistent_second_of_two", dataset, static_cast<int>(pos.size()),
            nthreads, t_total_static_persistent_second);
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

#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif

  std::fprintf(out, "# Targeted cell-creation benchmark\n");
  std::fprintf(out, "# Compiled: %s %s\n", __DATE__, __TIME__);
  std::fprintf(out, "# OpenMP max threads at runtime: %d\n", nthreads);
  std::fprintf(out, "# Cases: random_uniform N=10000, cubic_lattice N=8000\n");
  std::fprintf(out, "# Columns: benchmark,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");
  std::fprintf(out, "benchmark,point_set,N,nthreads,reps,time_ms_mean,time_ms_std\n");

  constexpr double L = 1.0;
  constexpr uint64_t SEED = 20260328ULL;

  std::fprintf(stderr, "\n=== targeted benchmark ===\n");
  run_case(out, "random_uniform", random_uniform(10000, L, SEED), L, nthreads);
  run_case(out, "cubic_lattice", cubic_lattice(8000, L, SEED), L, nthreads);

  if (out != stdout)
    std::fclose(out);

  return 0;
}
