/**
 * @file test_moving_points_repair_stats.cpp
 * @brief Moving-points tessellation repair study for 1e3 particles.
 *
 * Workflow:
 * 1. Initialize random particle positions and normally distributed velocities.
 * 2. Build an initial Voronoi tessellation.
 * 3. Advance particles linearly for several timesteps with periodic wrapping.
 * 4. At each step, measure non-convex cells before and after update.
 * 5. Compare final incrementally-updated tessellation against a clean rebuild
 *    at the final positions.
 *
 * The executable prints a per-step statistics table and exits non-zero if the
 * final tessellation comparison fails configured tolerances.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include <vorflow/voronoi.hpp>

namespace {

using Real = double;
using Pos3 = std::array<Real, 3>;

/**
 * @brief Compact per-cell topology signature used for final comparisons.
 */
struct CellSignature {
  vor::uint1 num_vertices = 0;
  vor::uint1 num_facets = 0;
  std::vector<vor::uint2> nbrs_sorted;
};

/**
 * @brief Per-timestep diagnostics for the moving-points run.
 */
struct StepStats {
  int step = 0;
  int non_convex_before = 0;
  int non_convex_after = 0;
  int convex_fixed_cells = 0;
  int rebuild_candidates = 0;
  int local_rebuild_cells = 0;
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

/**
 * @brief Create random positions uniformly in a cubic periodic box.
 */
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

/**
 * @brief Create normally distributed velocities with zero mean.
 */
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

/**
 * @brief Move points linearly for one timestep and wrap periodically.
 */
void AdvectAndWrap(std::vector<Pos3>& pos, const std::vector<Pos3>& vel, Real dt,
                   vor::Box<Real>& box) {
  for (std::size_t i = 0; i < pos.size(); ++i) {
    pos[i][0] += vel[i][0] * dt;
    pos[i][1] += vel[i][1] * dt;
    pos[i][2] += vel[i][2] * dt;
  }
  box.putInBox(pos);
}

/**
 * @brief Count non-convex cells for a given point set and current tessellation.
 */
int CountNonConvexCells(vor::CellComplex<Real>& complex, const std::vector<Pos3>& pos,
                        const vor::Box<Real>& box) {
  std::vector<vor::CellGeometry<Real> >& geoms = complex.getGeoms();
  int non_convex = 0;
  for (std::size_t i = 0; i < geoms.size(); ++i) {
    geoms[i].computeConnectingVectors(pos, box);
    geoms[i].computeEdgeInv();
    geoms[i].updateVertexPos();
    if (!geoms[i].isConvex()) {
      ++non_convex;
    }
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
    if (violation > max_violation) {
      max_violation = violation;
    }
  }
  return max_violation;
}

/**
 * @brief Build per-cell signatures from the packed CellArena representation.
 */
std::vector<CellSignature> BuildSignaturesByCellId(const vor::CellComplex<Real>& complex,
                                                   int num_cells) {
  const vor::CellArena<Real>& arena = complex.getCellArena();
  std::vector<CellSignature> sigs(static_cast<std::size_t>(num_cells));

  for (std::size_t i = 0; i < arena.numCells(); ++i) {
    const vor::uint2 id = arena.cellId(i);
    if (id >= static_cast<vor::uint2>(num_cells)) {
      continue;
    }

    CellSignature sig;
    sig.num_vertices = arena.cellNumVertices(i);
    sig.num_facets = arena.cellNumFacets(i);

    const vor::uint2* nbr_ptr = arena.cellNbrData(i);
    for (vor::uint1 f = 0; f < sig.num_facets; ++f) {
      const vor::uint2 nbr = nbr_ptr[f];
      if (nbr != vor::noNbr) {
        sig.nbrs_sorted.push_back(nbr);
      }
    }
    std::sort(sig.nbrs_sorted.begin(), sig.nbrs_sorted.end());
    sigs[id] = sig;
  }

  return sigs;
}

/**
 * @brief Count cells with at least one facet lacking a neighbor id.
 */
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
    if (has_no_nbr) {
      ++count;
    }
  }
  return count;
}

/**
 * @brief Compute max absolute and relative per-cell volume differences.
 */
void ComputeVolumeDiffStatsByCellId(vor::CellComplex<Real>& a, vor::CellComplex<Real>& b,
                                    int num_cells, Real* max_abs_diff, Real* max_rel_diff) {
  *max_abs_diff = 0.0;
  *max_rel_diff = 0.0;

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
  }

  for (std::size_t i = 0; i < arena_b.numCells(); ++i) {
    const vor::uint2 id = arena_b.cellId(i);
    if (id >= static_cast<vor::uint2>(num_cells))
      continue;
    gb[i].computeVolume();
    vol_by_id_b[id][0] = gb[i].getVolume();
    vol_by_id_b[id][1] = 1.0;
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

    if (abs_diff > *max_abs_diff) {
      *max_abs_diff = abs_diff;
    }
    if (rel_diff > *max_rel_diff) {
      *max_rel_diff = rel_diff;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  int num_particles = 1000;
  int num_steps = 8;
  const Real box_len = 1.0;
  Real dt = 0.01;
  Real velocity_sigma = 1.0;
  std::uint64_t pos_seed = 20260327ULL;
  std::uint64_t vel_seed = 20260328ULL;
  Real max_rel_vol_tol = 1e-12;
  bool require_correct = false;
  bool compare_final = true;

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
    max_rel_vol_tol = std::atof(argv[7]);
  if (argc > 8)
    require_correct = (std::atoi(argv[8]) != 0);
  if (argc > 9)
    compare_final = (std::atoi(argv[9]) != 0);

  if (num_particles <= 0 || num_steps <= 0 || dt <= 0.0 || velocity_sigma < 0.0 ||
      max_rel_vol_tol < 0.0) {
    std::fprintf(stderr,
                 "Usage: %s [N=1000] [steps=8] [dt=0.01] [velocity_sigma=1.0] "
                 "[pos_seed] [vel_seed] [max_rel_vol_tol=1e-12] [require_correct=0|1] "
                 "[compare_final=0|1]\n",
                 argv[0]);
    return 1;
  }

  Pos3 L;
  L[0] = box_len;
  L[1] = box_len;
  L[2] = box_len;
  vor::Box<Real> box(L);

  std::vector<Pos3> pos = RandomUniformPositions(num_particles, box_len, pos_seed);
  const std::vector<Pos3> vel = RandomNormalVelocities(num_particles, velocity_sigma, vel_seed);

  vor::CellComplex<Real> complex_incremental(&box);
  complex_incremental.build(pos);

  std::vector<StepStats> step_stats;
  step_stats.reserve(static_cast<std::size_t>(num_steps));

  int cumulative_non_convex_before = 0;
  int cumulative_topology_changed = 0;

  std::printf("# Moving-points Voronoi repair test\n");
  std::printf("# N=%d, steps=%d, dt=%.6f, velocity_sigma=%.6f\n", num_particles, num_steps, dt,
              velocity_sigma);
  std::printf("# seeds: pos=%llu vel=%llu\n", static_cast<unsigned long long>(pos_seed),
              static_cast<unsigned long long>(vel_seed));
  std::printf("step,non_convex_before,non_convex_after,convex_fixed_cells,rebuild_candidates,");
  std::printf("local_rebuild_cells,full_rebuild_cells,repair_iterations,repair_proposals_total,");
  std::printf("repair_target_groups_total,repair_cells_changed_total,repair_direct_attempts,");
  std::printf("repair_direct_successes,repair_indirect_candidates,repair_batch_calls,");
  std::printf("repair_batch_changes,topology_changed_cells,no_nbr_cells_after,");
  std::printf("max_convex_violation_after\n");

  for (int step = 1; step <= num_steps; ++step) {
    AdvectAndWrap(pos, vel, dt, box);

    const int non_convex_before = CountNonConvexCells(complex_incremental, pos, box);

    const std::vector<CellSignature> sig_before =
        BuildSignaturesByCellId(complex_incremental, num_particles);

    complex_incremental.update(pos);

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
    s.non_convex_before = non_convex_before;
    s.non_convex_after = non_convex_after;
    s.convex_fixed_cells = non_convex_before - non_convex_after;
    s.rebuild_candidates = static_cast<int>(update_stats.num_rebuild_candidates);
    s.local_rebuild_cells = static_cast<int>(update_stats.num_local_rebuild_cells);
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
    step_stats.push_back(s);

    cumulative_non_convex_before += non_convex_before;
    cumulative_topology_changed += changed_cells;

    std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.16e\n", s.step,
                s.non_convex_before, s.non_convex_after, s.convex_fixed_cells, s.rebuild_candidates,
                s.local_rebuild_cells, s.full_rebuild_cells, s.repair_iterations,
                s.repair_proposals_total, s.repair_target_groups_total,
                s.repair_cells_changed_total, s.repair_direct_attempts, s.repair_direct_successes,
                s.repair_indirect_candidates, s.repair_batch_calls, s.repair_batch_changes,
                s.topology_changed_cells, s.no_nbr_cells_after, s.max_convex_violation_after);
  }

  int signature_mismatch_cells = 0;
  int non_reciprocal_pairs = 0;
  Real max_abs_vol_diff = 0.0;
  Real max_rel_vol_diff = 0.0;
  bool final_topology_match = true;
  bool final_volume_match = true;
  if (compare_final) {
    vor::CellComplex<Real> complex_clean(&box);
    complex_clean.build(pos);

    const std::vector<CellSignature> sig_incremental =
        BuildSignaturesByCellId(complex_incremental, num_particles);
    const std::vector<CellSignature> sig_clean =
        BuildSignaturesByCellId(complex_clean, num_particles);

    for (std::size_t i = 0; i < sig_incremental.size(); ++i) {
      if (sig_incremental[i].num_vertices != sig_clean[i].num_vertices ||
          sig_incremental[i].num_facets != sig_clean[i].num_facets ||
          sig_incremental[i].nbrs_sorted != sig_clean[i].nbrs_sorted) {
        ++signature_mismatch_cells;
        std::printf("# MISMATCH cell %zu: incr(V=%u F=%u nbrs=", i, sig_incremental[i].num_vertices,
                    sig_incremental[i].num_facets);
        for (auto n : sig_incremental[i].nbrs_sorted)
          std::printf("%u ", n);
        std::printf(") clean(V=%u F=%u nbrs=", sig_clean[i].num_vertices, sig_clean[i].num_facets);
        for (auto n : sig_clean[i].nbrs_sorted)
          std::printf("%u ", n);
        std::printf(")\n");
        std::vector<vor::uint2> missing, extra;
        std::set_difference(sig_clean[i].nbrs_sorted.begin(), sig_clean[i].nbrs_sorted.end(),
                            sig_incremental[i].nbrs_sorted.begin(),
                            sig_incremental[i].nbrs_sorted.end(), std::back_inserter(missing));
        std::set_difference(sig_incremental[i].nbrs_sorted.begin(),
                            sig_incremental[i].nbrs_sorted.end(), sig_clean[i].nbrs_sorted.begin(),
                            sig_clean[i].nbrs_sorted.end(), std::back_inserter(extra));
        if (!missing.empty()) {
          std::printf("#   missing from incr: ");
          for (auto n : missing)
            std::printf("%u ", n);
          std::printf("\n");
        }
        if (!extra.empty()) {
          std::printf("#   extra in incr: ");
          for (auto n : extra)
            std::printf("%u ", n);
          std::printf("\n");
        }
      }
    }

    {
      const vor::CellArena<Real>& arena = complex_incremental.getCellArena();
      std::vector<std::size_t> id_to_idx(static_cast<std::size_t>(num_particles), SIZE_MAX);
      for (std::size_t i = 0; i < arena.numCells(); ++i) {
        const vor::uint2 cid = arena.cellId(i);
        if (cid < static_cast<vor::uint2>(num_particles))
          id_to_idx[cid] = i;
      }
      for (std::size_t i = 0; i < arena.numCells(); ++i) {
        const vor::uint2 cid = arena.cellId(i);
        if (cid >= static_cast<vor::uint2>(num_particles))
          continue;
        const vor::uint2* nbr_ptr = arena.cellNbrData(i);
        const vor::uint1 nf = arena.cellNumFacets(i);
        for (vor::uint1 f = 0; f < nf; ++f) {
          const vor::uint2 n = nbr_ptr[f];
          if (n == vor::noNbr || n >= static_cast<vor::uint2>(num_particles))
            continue;
          std::size_t nidx = id_to_idx[n];
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
          if (!found) {
            std::printf("# NON-RECIPROCAL: cell %u has nbr %u, but %u does not have %u\n", cid, n,
                        n, cid);
            ++non_reciprocal_pairs;
          }
        }
      }
    }

    ComputeVolumeDiffStatsByCellId(complex_incremental, complex_clean, num_particles,
                                   &max_abs_vol_diff, &max_rel_vol_diff);
    final_topology_match = (signature_mismatch_cells == 0);
    final_volume_match = (max_rel_vol_diff <= max_rel_vol_tol);
  }

  const bool all_steps_convex_after =
      std::all_of(step_stats.begin(), step_stats.end(),
                  [](const StepStats& s) { return s.non_convex_after == 0; });
  const bool no_no_nbr_after =
      std::all_of(step_stats.begin(), step_stats.end(),
                  [](const StepStats& s) { return s.no_nbr_cells_after == 0; });

  const bool final_ok = compare_final ? (final_topology_match && final_volume_match) : true;

  std::printf("\n# Summary\n");
  std::printf("cumulative_non_convex_before=%d\n", cumulative_non_convex_before);
  std::printf("cumulative_topology_changed_cells=%d\n", cumulative_topology_changed);
  if (compare_final) {
    std::printf("final_signature_mismatch_cells=%d\n", signature_mismatch_cells);
    std::printf("final_non_reciprocal_pairs=%d\n", non_reciprocal_pairs);
    std::printf("final_max_abs_volume_diff=%.16e\n", max_abs_vol_diff);
    std::printf("final_max_rel_volume_diff=%.16e\n", max_rel_vol_diff);
    std::printf("final_tessellation_correct=%s\n", final_ok ? "yes" : "no");
  } else {
    std::printf("final_comparison_skipped=yes\n");
  }

  if (compare_final && !final_ok && require_correct) {
    std::fprintf(stderr,
                 "FAIL: final tessellation check failed. "
                 "(convex_after_all_steps=%d, no_nbr_after_all_steps=%d, mismatch_cells=%d, "
                 "max_rel_vol=%.3e)\n",
                 all_steps_convex_after ? 1 : 0, no_no_nbr_after ? 1 : 0, signature_mismatch_cells,
                 max_rel_vol_diff);
    return 1;
  }
  if (compare_final && !final_ok) {
    std::fprintf(stderr,
                 "WARNING: incremental and clean final tessellations differ. "
                 "Set require_correct=1 to make this a hard failure.\n");
  }

  return 0;
}
