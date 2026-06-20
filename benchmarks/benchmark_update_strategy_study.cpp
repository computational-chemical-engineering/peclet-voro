/**
 * @file benchmark_update_strategy_study.cpp
 * @brief Parameter study comparing the default update against full rebuild.
 *
 * The benchmark:
 *  - builds an initial tessellation for 1e4 random particles
 *  - assigns normally distributed velocities
 *  - advances the system for several timesteps at a chosen dt
 *  - records per-step statistics and timing for each method
 *  - compares each final tessellation against a clean static rebuild
 *
 * Output:
 *  CSV rows, one per timestep per method, with repeated final-comparison metrics
 *  so a report script can aggregate results across multiple dt runs.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
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

struct CellSignature {
  vor::uint1 num_vertices = 0;
  vor::uint1 num_facets = 0;
  std::vector<vor::uint2> nbrs_sorted;
};

struct StepStats {
  int step = 0;
  double update_ms = 0.0;
  int non_convex_before = 0;
  int non_convex_after = 0;
  int rebuild_candidates = 0;
  int local_rebuild_cells = 0;
  int empty_after_local_rebuild = 0;
  int full_rebuild_cells = 0;
  int repair_iterations = 0;
  int repair_proposals_total = 0;
  int repair_target_groups_total = 0;
  int repair_cells_changed_total = 0;
  int repair_direct_attempts = 0;
  int repair_direct_successes = 0;
  int repair_indirect_candidates = 0;
  int repair_batch_calls = 0;
  int repair_batch_changes = 0;
  int topology_changed_cells = 0;
  int no_nbr_cells_after = 0;
  Real max_convex_violation_after = 0.0;
};

struct MethodSpec {
  const char* key = "";
  const char* label = "";
  MethodKind kind = kFullBuild;
};

struct MethodResult {
  std::string method_key;
  int worker_count = 0;
  int initial_non_convex_at_build = 0;
  std::vector<StepStats> step_stats;
  int final_signature_mismatch_cells = 0;
  int final_non_reciprocal_pairs = 0;
  Real final_max_abs_volume_diff = 0.0;
  Real final_max_rel_volume_diff = 0.0;
  Real final_total_volume_update = 0.0;
  Real final_total_volume_rebuild = 0.0;
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

Real MaxConvexViolation(vor::CellComplex<Real>& complex, const std::vector<Pos3>& pos,
                        const vor::Box<Real>& box) {
  std::vector<vor::CellGeometry<Real> >& geoms = complex.getGeoms();
  Real max_violation = 0.0;
  for (std::size_t i = 0; i < geoms.size(); ++i) {
    geoms[i].computeConnectingVectors(pos, box);
    geoms[i].computeEdgeInv();
    geoms[i].updateVertexPos();
    const Real violation = geoms[i].maxConvexViolation();
    if (violation > max_violation)
      max_violation = violation;
  }
  return max_violation;
}

std::vector<CellSignature> BuildSignaturesByCellId(const vor::CellComplex<Real>& complex,
                                                   int num_cells) {
  const vor::CellArena<Real>& arena = complex.getCellArena();
  std::vector<CellSignature> sigs(static_cast<std::size_t>(num_cells));
  for (std::size_t i = 0; i < arena.numCells(); ++i) {
    const vor::uint2 id = arena.cellId(i);
    if (id >= static_cast<vor::uint2>(num_cells))
      continue;

    CellSignature sig;
    sig.num_vertices = arena.cellNumVertices(i);
    sig.num_facets = arena.cellNumFacets(i);
    const vor::uint2* nbr_ptr = arena.cellNbrData(i);
    for (vor::uint1 f = 0; f < sig.num_facets; ++f) {
      const vor::uint2 nbr = nbr_ptr[f];
      if (nbr != vor::noNbr)
        sig.nbrs_sorted.push_back(nbr);
    }
    std::sort(sig.nbrs_sorted.begin(), sig.nbrs_sorted.end());
    sigs[id] = sig;
  }
  return sigs;
}

int CountNoNeighborCells(const vor::CellComplex<Real>& complex) {
  const vor::CellArena<Real>& arena = complex.getCellArena();
  int count = 0;
  for (std::size_t i = 0; i < arena.numCells(); ++i) {
    const vor::uint2* nbr_ptr = arena.cellNbrData(i);
    const vor::uint1 nf = arena.cellNumFacets(i);
    bool has_no_nbr = false;
    for (vor::uint1 f = 0; f < nf; ++f) {
      if (nbr_ptr[f] == vor::noNbr) {
        has_no_nbr = true;
        break;
      }
    }
    if (has_no_nbr)
      ++count;
  }
  return count;
}

int CountNonReciprocalPairs(const vor::CellComplex<Real>& complex, int num_cells) {
  const vor::CellArena<Real>& arena = complex.getCellArena();
  int final_non_reciprocal_pairs = 0;
  std::vector<std::size_t> id_to_idx(static_cast<std::size_t>(num_cells), SIZE_MAX);
  for (std::size_t i = 0; i < arena.numCells(); ++i) {
    const vor::uint2 cid = arena.cellId(i);
    if (cid < static_cast<vor::uint2>(num_cells))
      id_to_idx[cid] = i;
  }
  for (std::size_t i = 0; i < arena.numCells(); ++i) {
    const vor::uint2 cid = arena.cellId(i);
    if (cid >= static_cast<vor::uint2>(num_cells))
      continue;
    const vor::uint2* nbr_ptr = arena.cellNbrData(i);
    const vor::uint1 nf = arena.cellNumFacets(i);
    for (vor::uint1 f = 0; f < nf; ++f) {
      const vor::uint2 n = nbr_ptr[f];
      if (n == vor::noNbr || n >= static_cast<vor::uint2>(num_cells))
        continue;
      const std::size_t nidx = id_to_idx[n];
      if (nidx == SIZE_MAX)
        continue;
      const vor::uint2* nbr2 = arena.cellNbrData(nidx);
      const vor::uint1 nf2 = arena.cellNumFacets(nidx);
      bool found = false;
      for (vor::uint1 g = 0; g < nf2; ++g) {
        if (nbr2[g] == cid) {
          found = true;
          break;
        }
      }
      if (!found)
        ++final_non_reciprocal_pairs;
    }
  }
  return final_non_reciprocal_pairs;
}

void ComputeVolumeDiffStatsByCellId(vor::CellComplex<Real>& a, vor::CellComplex<Real>& b,
                                    int num_cells, Real* max_abs_diff, Real* max_rel_diff,
                                    Real* total_vol_a, Real* total_vol_b) {
  *max_abs_diff = 0.0;
  *max_rel_diff = 0.0;
  *total_vol_a = 0.0;
  *total_vol_b = 0.0;

  const vor::CellArena<Real>& arena_a = a.getCellArena();
  const vor::CellArena<Real>& arena_b = b.getCellArena();
  std::vector<vor::CellGeometry<Real> >& ga = a.getGeoms();
  std::vector<vor::CellGeometry<Real> >& gb = b.getGeoms();

  std::vector<std::array<Real, 2> > vol_by_id_a(static_cast<std::size_t>(num_cells), {{0.0, 0.0}});
  std::vector<std::array<Real, 2> > vol_by_id_b(static_cast<std::size_t>(num_cells), {{0.0, 0.0}});

  for (std::size_t i = 0; i < arena_a.numCells(); ++i) {
    const vor::uint2 id = arena_a.cellId(i);
    if (id >= static_cast<vor::uint2>(num_cells))
      continue;
    ga[i].computeVolume();
    vol_by_id_a[id][0] = ga[i].getVolume();
    vol_by_id_a[id][1] = 1.0;
    *total_vol_a += ga[i].getVolume();
  }

  for (std::size_t i = 0; i < arena_b.numCells(); ++i) {
    const vor::uint2 id = arena_b.cellId(i);
    if (id >= static_cast<vor::uint2>(num_cells))
      continue;
    gb[i].computeVolume();
    vol_by_id_b[id][0] = gb[i].getVolume();
    vol_by_id_b[id][1] = 1.0;
    *total_vol_b += gb[i].getVolume();
  }

  for (int id = 0; id < num_cells; ++id) {
    if (vol_by_id_a[static_cast<std::size_t>(id)][1] == 0.0 ||
        vol_by_id_b[static_cast<std::size_t>(id)][1] == 0.0) {
      continue;
    }
    const Real va = vol_by_id_a[static_cast<std::size_t>(id)][0];
    const Real vb = vol_by_id_b[static_cast<std::size_t>(id)][0];
    const Real abs_diff = std::fabs(va - vb);
    const Real rel_diff = (std::fabs(vb) > 0.0) ? abs_diff / std::fabs(vb) : abs_diff;
    if (abs_diff > *max_abs_diff)
      *max_abs_diff = abs_diff;
    if (rel_diff > *max_rel_diff)
      *max_rel_diff = rel_diff;
  }
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

  vor::CellComplex<Real> complex_incremental(&box, static_cast<size_t>(std::max(worker_count, 0)));
  complex_incremental.build(pos);
  result.initial_non_convex_at_build = CountNonConvexCells(complex_incremental, pos, box);
  result.step_stats.reserve(static_cast<std::size_t>(num_steps));

  for (int step = 1; step <= num_steps; ++step) {
    AdvectAndWrap(pos, vel, dt, box);

    const int non_convex_before = CountNonConvexCells(complex_incremental, pos, box);
    const std::vector<CellSignature> sig_before =
        BuildSignaturesByCellId(complex_incremental, num_particles);

    const auto t0 = Clock::now();
    if (method.kind == kFullBuild)
      complex_incremental.build(pos);
    else
      complex_incremental.update(pos);
    const double update_ms = elapsed_ms(t0);

    const int non_convex_after = CountNonConvexCells(complex_incremental, pos, box);
    const int no_nbr_cells_after = CountNoNeighborCells(complex_incremental);
    const Real max_convex_violation_after = MaxConvexViolation(complex_incremental, pos, box);
    const vor::CellComplexUpdateStats& update_stats = complex_incremental.getLastUpdateStats();
    const std::vector<CellSignature> sig_after =
        BuildSignaturesByCellId(complex_incremental, num_particles);

    int changed_cells = 0;
    for (std::size_t i = 0; i < sig_before.size(); ++i) {
      if (sig_before[i].num_vertices != sig_after[i].num_vertices ||
          sig_before[i].num_facets != sig_after[i].num_facets ||
          sig_before[i].nbrs_sorted != sig_after[i].nbrs_sorted) {
        ++changed_cells;
      }
    }

    StepStats s;
    s.step = step;
    s.update_ms = update_ms;
    s.non_convex_before = non_convex_before;
    s.non_convex_after = non_convex_after;
    s.rebuild_candidates = static_cast<int>(update_stats.num_rebuild_candidates);
    s.local_rebuild_cells = static_cast<int>(update_stats.num_local_rebuild_cells);
    s.empty_after_local_rebuild = static_cast<int>(update_stats.num_empty_after_local_rebuild);
    s.full_rebuild_cells = static_cast<int>(update_stats.num_full_rebuild_cells);
    s.repair_iterations = static_cast<int>(update_stats.num_repair_iterations);
    s.repair_proposals_total = static_cast<int>(update_stats.num_repair_proposals_total);
    s.repair_target_groups_total = static_cast<int>(update_stats.num_repair_target_groups_total);
    s.repair_cells_changed_total = static_cast<int>(update_stats.num_repair_cells_changed_total);
    s.repair_direct_attempts = static_cast<int>(update_stats.num_repair_direct_attempts);
    s.repair_direct_successes = static_cast<int>(update_stats.num_repair_direct_successes);
    s.repair_indirect_candidates = static_cast<int>(update_stats.num_repair_indirect_candidates);
    s.repair_batch_calls = static_cast<int>(update_stats.num_repair_batch_calls);
    s.repair_batch_changes = static_cast<int>(update_stats.num_repair_batch_changes);
    s.topology_changed_cells = changed_cells;
    s.no_nbr_cells_after = no_nbr_cells_after;
    s.max_convex_violation_after = max_convex_violation_after;
    result.step_stats.push_back(s);
  }

  vor::CellComplex<Real> complex_clean(&box, static_cast<size_t>(std::max(worker_count, 0)));
  complex_clean.build(pos);

  const std::vector<CellSignature> sig_incremental =
      BuildSignaturesByCellId(complex_incremental, num_particles);
  const std::vector<CellSignature> sig_clean =
      BuildSignaturesByCellId(complex_clean, num_particles);
  for (std::size_t i = 0; i < sig_incremental.size(); ++i) {
    if (sig_incremental[i].num_vertices != sig_clean[i].num_vertices ||
        sig_incremental[i].num_facets != sig_clean[i].num_facets ||
        sig_incremental[i].nbrs_sorted != sig_clean[i].nbrs_sorted) {
      ++result.final_signature_mismatch_cells;
    }
  }

  result.final_non_reciprocal_pairs = CountNonReciprocalPairs(complex_incremental, num_particles);
  ComputeVolumeDiffStatsByCellId(complex_incremental, complex_clean, num_particles,
                                 &result.final_max_abs_volume_diff, &result.final_max_rel_volume_diff,
                                 &result.final_total_volume_update, &result.final_total_volume_rebuild);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  int num_particles = 10000;
  int num_steps = 6;
  const Real box_len = 1.0;
  Real dt = 1.0e-3;
  Real velocity_sigma = 1.0;
  std::uint64_t pos_seed = 20260329ULL;
  std::uint64_t vel_seed = 20260330ULL;
  int worker_count = 0;

  if (argc > 1)
    num_particles = std::atoi(argv[1]);
  if (argc > 2)
    num_steps = std::atoi(argv[2]);
  if (argc > 3)
    dt = std::atof(argv[3]);
  if (argc > 4)
    velocity_sigma = std::atof(argv[4]);
  if (argc > 5)
    pos_seed = static_cast<std::uint64_t>(std::strtoull(argv[5], nullptr, 10));
  if (argc > 6)
    vel_seed = static_cast<std::uint64_t>(std::strtoull(argv[6], nullptr, 10));
  if (argc > 7)
    worker_count = std::atoi(argv[7]);

  if (num_particles <= 0 || num_steps <= 0 || dt <= 0.0 || velocity_sigma < 0.0 || worker_count < 0) {
    std::fprintf(stderr,
                 "Usage: %s [N=10000] [steps=6] [dt=1e-3] [velocity_sigma=1.0] [pos_seed] [vel_seed] [worker_count=0]\n",
                 argv[0]);
    return 1;
  }

  const std::vector<Pos3> pos0 = RandomUniformPositions(num_particles, box_len, pos_seed);
  const std::vector<Pos3> vel = RandomNormalVelocities(num_particles, velocity_sigma, vel_seed);

  const MethodSpec methods[] = {
      {"fullBuild", "fullBuild", kFullBuild},
      {"update", "update", kUpdate},
  };

  std::printf("# benchmark_update_strategy_study\n");
  std::printf("# Compiled: %s %s\n", __DATE__, __TIME__);
  std::printf("# N=%d steps=%d dt=%.8g velocity_sigma=%.6f pos_seed=%llu vel_seed=%llu worker_count=%d\n",
              num_particles, num_steps, dt, velocity_sigma,
              static_cast<unsigned long long>(pos_seed),
              static_cast<unsigned long long>(vel_seed), worker_count);
  std::printf(
      "method,N,steps,dt,velocity_sigma,pos_seed,vel_seed,worker_count,initial_non_convex_at_build,step,update_ms,"
      "non_convex_before,non_convex_after,rebuild_candidates,local_rebuild_cells,"
      "empty_after_local_rebuild,full_rebuild_cells,repair_iterations,repair_proposals_total,"
      "repair_target_groups_total,repair_cells_changed_total,repair_direct_attempts,"
      "repair_direct_successes,repair_indirect_candidates,repair_batch_calls,repair_batch_changes,"
      "topology_changed_cells,no_nbr_cells_after,max_convex_violation_after,"
      "final_signature_mismatch_cells,final_non_reciprocal_pairs,final_max_abs_volume_diff,"
      "final_max_rel_volume_diff,final_total_volume_update,final_total_volume_rebuild\n");

  for (const MethodSpec& method : methods) {
    const MethodResult result = RunStudy(method, num_particles, num_steps, box_len, dt, pos0, vel, worker_count);
    for (const StepStats& s : result.step_stats) {
      std::printf(
          "%s,%d,%d,%.16e,%.16e,%llu,%llu,%d,%d,%d,%.16e,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.16e,%d,%d,%.16e,%.16e,%.16e,%.16e\n",
          result.method_key.c_str(), num_particles, num_steps, dt, velocity_sigma,
          static_cast<unsigned long long>(pos_seed), static_cast<unsigned long long>(vel_seed),
          result.worker_count, result.initial_non_convex_at_build, s.step, s.update_ms, s.non_convex_before,
          s.non_convex_after, s.rebuild_candidates, s.local_rebuild_cells,
          s.empty_after_local_rebuild, s.full_rebuild_cells, s.repair_iterations,
          s.repair_proposals_total, s.repair_target_groups_total, s.repair_cells_changed_total,
          s.repair_direct_attempts, s.repair_direct_successes, s.repair_indirect_candidates,
          s.repair_batch_calls, s.repair_batch_changes, s.topology_changed_cells,
          s.no_nbr_cells_after, s.max_convex_violation_after, result.final_signature_mismatch_cells,
          result.final_non_reciprocal_pairs, result.final_max_abs_volume_diff,
          result.final_max_rel_volume_diff, result.final_total_volume_update,
          result.final_total_volume_rebuild);
    }
  }

  return 0;
}
