#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <voronoi_dynamics/voronoi.hpp>

int main() {
  using real_t = double;

  std::array<real_t, 3> L = {1.0, 1.0, 1.0};
  vor::Box<real_t> box(L);

  const int num_particles = 256;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<real_t> uniform(0.0, 1.0);

  std::vector<std::array<real_t, 3> > pos(static_cast<std::size_t>(num_particles));
  for (std::array<real_t, 3> &p : pos) {
    p[0] = uniform(rng);
    p[1] = uniform(rng);
    p[2] = uniform(rng);
  }

  const real_t density = static_cast<real_t>(num_particles) / (L[0] * L[1] * L[2]);
  const real_t rcut = 1.75 * std::pow(density, -1.0 / 3.0);

  vor::NbrList<vor::uint2, real_t> nbr_list(&box);
  nbr_list.setup(pos, rcut);

  const vor::Cuboid<real_t> cub(L);
  vor::resetCellMakerTelemetry();

  for (vor::uint2 i = 0; i < static_cast<vor::uint2>(pos.size()); ++i) {
    vor::ConstructionArena<real_t> arena(8, 6);
    vor::CellMaker<real_t> maker(arena);
    maker.build(i, pos, nbr_list, cub);
  }

  const vor::CellMakerTelemetry &telemetry = vor::cellMakerTelemetry();
  const uint64_t vertex_growth = telemetry.vertex_growth_events.load(std::memory_order_relaxed);
  const uint64_t facet_growth = telemetry.facet_growth_events.load(std::memory_order_relaxed);
  const uint64_t peak_vertex_capacity =
      telemetry.peak_vertex_capacity.load(std::memory_order_relaxed);
  const uint64_t peak_facet_capacity =
      telemetry.peak_facet_capacity.load(std::memory_order_relaxed);

  std::printf("vertex_growth=%llu facet_growth=%llu peak_vertex_capacity=%llu "
              "peak_facet_capacity=%llu\n",
              static_cast<unsigned long long>(vertex_growth),
              static_cast<unsigned long long>(facet_growth),
              static_cast<unsigned long long>(peak_vertex_capacity),
              static_cast<unsigned long long>(peak_facet_capacity));

  if (vertex_growth == 0 && facet_growth == 0) {
    std::fprintf(stderr, "FAIL: expected CellMaker growth events with tiny construction arenas.\n");
    return 1;
  }

  if (peak_vertex_capacity <= 8 && peak_facet_capacity <= 6) {
    std::fprintf(stderr, "FAIL: capacities never exceeded tiny arena defaults.\n");
    return 1;
  }

  return 0;
}
