/**
 * @file benchmark_update_thread_scaling.cpp
 * @brief Large-N thread-scaling study for the default update versus full rebuild.
 *
 * This benchmark is intentionally performance-focused:
 *  - random positions in a unit periodic box
 *  - Gaussian velocities with sigma = N^(-1/3), so dt is dimensionless on the scale of a cell
 *  - advances the system for a small number of timesteps
 *  - measures per-step runtime and non-convex fraction before the update
 *  - compares full build and the default asynchronous sweep update
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <vorflow/voronoi.hpp>

namespace {

using Clock = std::chrono::high_resolution_clock;
using Millis = std::chrono::duration<double, std::milli>;
using Real = double;
using Pos3 = std::array<Real, 3>;

enum MethodKind {
  kFullBuild = 0,
  kUpdate = 1,
};

struct MethodSpec {
  const char* key = "";
  MethodKind kind = kFullBuild;
};

struct StepStats {
  int step = 0;
  double update_ms = 0.0;
  int non_convex_before = 0;
  double non_convex_fraction_before = 0.0;
  int rebuild_candidates = 0;
  int local_rebuild_cells = 0;
  int full_rebuild_cells = 0;
  int repair_iterations = 0;
  int repair_proposals_total = 0;
};

struct MethodResult {
  std::string method_key;
  int worker_count = 0;
  int initial_non_convex_at_build = 0;
  std::vector<StepStats> step_stats;
};

static inline double elapsed_ms(Clock::time_point t0) {
  return Millis(Clock::now() - t0).count();
}

std::vector<Pos3> RandomUniformPositions(int n, Real box_len, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<Real> uniform_dist(0.0, box_len);
  std::vector<Pos3> pos(static_cast<std::size_t>(n));
  for (Pos3& p : pos) {
    p[0] = uniform_dist(rng);
    p[1] = uniform_dist(rng);
    p[2] = uniform_dist(rng);
  }
  return pos;
}

std::vector<Pos3> RandomNormalVelocities(int n, Real sigma, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<Real> normal_dist(0.0, sigma);
  std::vector<Pos3> vel(static_cast<std::size_t>(n));
  for (Pos3& v : vel) {
    v[0] = normal_dist(rng);
    v[1] = normal_dist(rng);
    v[2] = normal_dist(rng);
  }
  return vel;
}

void AdvectAndWrap(std::vector<Pos3>& pos, const std::vector<Pos3>& vel, Real dt, vor::Box<Real>& box) {
  for (std::size_t i = 0; i < pos.size(); ++i) {
    pos[i][0] += vel[i][0] * dt;
    pos[i][1] += vel[i][1] * dt;
    pos[i][2] += vel[i][2] * dt;
  }
  box.putInBox(pos);
}

int CountNonConvexCells(vor::CellComplex<Real>& complex, const std::vector<Pos3>& pos,
                        const vor::Box<Real>& box) {
  std::vector<vor::CellGeometry<Real> >& geoms = complex.getGeoms();
  int non_convex = 0;
  for (std::size_t i = 0; i < geoms.size(); ++i) {
    geoms[i].computeConnectingVectors(pos, box);
    geoms[i].computeEdgeInv();
    geoms[i].updateVertexPos();
    if (!geoms[i].isConvex())
      ++non_convex;
  }
  return non_convex;
}

MethodResult RunStudy(const MethodSpec& method, int num_particles, int num_steps, Real box_len, Real dt,
                      const std::vector<Pos3>& pos0, const std::vector<Pos3>& vel, int worker_count) {
  Pos3 L;
  L[0] = box_len;
  L[1] = box_len;
  L[2] = box_len;
  vor::Box<Real> box(L);
  std::vector<Pos3> pos = pos0;

  MethodResult result;
  result.method_key = method.key;
  result.worker_count = worker_count;

  vor::CellComplex<Real> complex(&box, static_cast<size_t>(std::max(worker_count, 0)));
  complex.build(pos);
  result.initial_non_convex_at_build = CountNonConvexCells(complex, pos, box);
  result.step_stats.reserve(static_cast<std::size_t>(num_steps));

  for (int step = 1; step <= num_steps; ++step) {
    AdvectAndWrap(pos, vel, dt, box);

    const int non_convex_before = CountNonConvexCells(complex, pos, box);
    const auto t0 = Clock::now();
    switch (method.kind) {
      case kFullBuild:
        complex.build(pos);
        break;
      case kUpdate:
        complex.update(pos);
        break;
    }
    const double update_ms = elapsed_ms(t0);

    const vor::CellComplexUpdateStats& update_stats = complex.getLastUpdateStats();
    StepStats s;
    s.step = step;
    s.update_ms = update_ms;
    s.non_convex_before = non_convex_before;
    s.non_convex_fraction_before =
        (num_particles > 0) ? static_cast<double>(non_convex_before) / static_cast<double>(num_particles) : 0.0;
    s.rebuild_candidates = static_cast<int>(update_stats.num_rebuild_candidates);
    s.local_rebuild_cells = static_cast<int>(update_stats.num_local_rebuild_cells);
    s.full_rebuild_cells = static_cast<int>(update_stats.num_full_rebuild_cells);
    s.repair_iterations = static_cast<int>(update_stats.num_repair_iterations);
    s.repair_proposals_total = static_cast<int>(update_stats.num_repair_proposals_total);
    result.step_stats.push_back(s);
  }

  return result;
}

}  // namespace

int main(int argc, char** argv) {
  int num_particles = 100000;
  int num_steps = 2;
  const Real box_len = 1.0;
  Real dt = 1.0e-3;
  std::uint64_t pos_seed = 20260329ULL;
  std::uint64_t vel_seed = 20260330ULL;
  int worker_count = 1;

  if (argc > 1)
    num_particles = std::atoi(argv[1]);
  if (argc > 2)
    num_steps = std::atoi(argv[2]);
  if (argc > 3)
    dt = std::atof(argv[3]);
  if (argc > 4)
    pos_seed = static_cast<std::uint64_t>(std::strtoull(argv[4], nullptr, 10));
  if (argc > 5)
    vel_seed = static_cast<std::uint64_t>(std::strtoull(argv[5], nullptr, 10));
  if (argc > 6)
    worker_count = std::atoi(argv[6]);

  if (num_particles <= 0 || num_steps <= 0 || dt <= 0.0 || worker_count <= 0) {
    std::fprintf(stderr, "Usage: %s [N=100000] [steps=2] [dt=1e-3] [pos_seed] [vel_seed] [worker_count=1]\n",
                 argv[0]);
    return 1;
  }

  const Real velocity_sigma = 1.0 / std::cbrt(static_cast<Real>(num_particles));
  const std::vector<Pos3> pos0 = RandomUniformPositions(num_particles, box_len, pos_seed);
  const std::vector<Pos3> vel = RandomNormalVelocities(num_particles, velocity_sigma, vel_seed);

  const MethodSpec methods[] = {
      {"fullBuild", kFullBuild},
      {"update", kUpdate},
  };

  std::printf("# benchmark_update_thread_scaling\n");
  std::printf("# Compiled: %s %s\n", __DATE__, __TIME__);
  std::printf("# N=%d steps=%d dt=%.8g velocity_sigma=%.16e pos_seed=%llu vel_seed=%llu worker_count=%d\n",
              num_particles, num_steps, dt, velocity_sigma, static_cast<unsigned long long>(pos_seed),
              static_cast<unsigned long long>(vel_seed), worker_count);
  std::printf("method,N,steps,dt,velocity_sigma,pos_seed,vel_seed,worker_count,initial_non_convex_at_build,"
              "step,update_ms,non_convex_before,non_convex_fraction_before,rebuild_candidates,"
              "local_rebuild_cells,full_rebuild_cells,repair_iterations,repair_proposals_total\n");

  for (const MethodSpec& method : methods) {
    const MethodResult result = RunStudy(method, num_particles, num_steps, box_len, dt, pos0, vel, worker_count);
    for (const StepStats& s : result.step_stats) {
      std::printf("%s,%d,%d,%.16e,%.16e,%llu,%llu,%d,%d,%d,%.16e,%d,%.16e,%d,%d,%d,%d,%d\n",
                  result.method_key.c_str(), num_particles, num_steps, dt, velocity_sigma,
                  static_cast<unsigned long long>(pos_seed), static_cast<unsigned long long>(vel_seed),
                  result.worker_count, result.initial_non_convex_at_build, s.step, s.update_ms,
                  s.non_convex_before, s.non_convex_fraction_before, s.rebuild_candidates,
                  s.local_rebuild_cells, s.full_rebuild_cells, s.repair_iterations,
                  s.repair_proposals_total);
    }
  }

  return 0;
}
