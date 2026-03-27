/**
 * @file cell_topology_stats.cpp
 * @brief Build a Voronoi tessellation for random particles and dump per-cell
 *        topology statistics (vertices, edges, faces).
 *
 * Edges are computed from Euler's relation for convex polyhedra:
 *   E = V + F - 2.
 *
 * Usage:
 *   ./cell_topology_stats [output_csv] [N] [seed]
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <voronoi_dynamics/voronoi.hpp>

using Pos3d = vor::Array<double, 3>;

namespace {

std::vector<Pos3d> RandomUniformPositions(int n, double L, uint64_t seed) {
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

struct SummaryStats {
  int min_val = 0;
  int max_val = 0;
  double mean = 0.0;
  double std = 0.0;
};

SummaryStats ComputeSummary(const std::vector<int> &vals) {
  SummaryStats stats;
  if (vals.empty()) {
    return stats;
  }

  auto minmax = std::minmax_element(vals.begin(), vals.end());
  stats.min_val = *minmax.first;
  stats.max_val = *minmax.second;

  const double n = static_cast<double>(vals.size());
  const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
  stats.mean = sum / n;

  double var = 0.0;
  for (int v : vals) {
    const double dv = static_cast<double>(v) - stats.mean;
    var += dv * dv;
  }
  stats.std = std::sqrt(var / n);
  return stats;
}

}  // namespace

int main(int argc, char *argv[]) {
  const char *output_csv = (argc > 1) ? argv[1] : "cell_topology_1e4_random.csv";
  const int N = (argc > 2) ? std::atoi(argv[2]) : 10000;
  const uint64_t seed = (argc > 3) ? static_cast<uint64_t>(std::strtoull(argv[3], nullptr, 10))
                                   : 20260327ULL;

  if (N <= 0) {
    std::fprintf(stderr, "Error: N must be positive, got %d\n", N);
    return 1;
  }

  constexpr double kBoxLength = 1.0;
  auto pos = RandomUniformPositions(N, kBoxLength, seed);

  vor::Array<double, 3> L;
  L[0] = kBoxLength;
  L[1] = kBoxLength;
  L[2] = kBoxLength;
  vor::Box<double> box(L);
  vor::CellComplex<double> complex(&box);

  std::fprintf(stderr, "Building Voronoi tessellation for N=%d random particles...\n", N);
  complex.build(pos);
  const auto &arena = complex.getCellArena();

  std::vector<int> vertices;
  std::vector<int> edges;
  std::vector<int> faces;
  vertices.reserve(arena.numCells());
  edges.reserve(arena.numCells());
  faces.reserve(arena.numCells());

  FILE *fp = std::fopen(output_csv, "w");
  if (fp == nullptr) {
    std::perror(output_csv);
    return 1;
  }

  std::fprintf(fp, "cell_id,vertices,edges,faces\n");
  for (std::size_t i = 0; i < arena.numCells(); ++i) {
    const int v = static_cast<int>(arena.cellNumVertices(i));
    const int f = static_cast<int>(arena.cellNumFacets(i));
    const int e = v + f - 2;
    vertices.push_back(v);
    edges.push_back(e);
    faces.push_back(f);
    std::fprintf(fp, "%u,%d,%d,%d\n", static_cast<unsigned>(arena.cellId(i)), v, e, f);
  }
  std::fclose(fp);

  const SummaryStats vs = ComputeSummary(vertices);
  const SummaryStats es = ComputeSummary(edges);
  const SummaryStats fs = ComputeSummary(faces);

  std::fprintf(stderr, "Saved per-cell topology CSV: %s\n", output_csv);
  std::fprintf(stderr, "Vertices: min=%d max=%d mean=%.3f std=%.3f\n",
               vs.min_val, vs.max_val, vs.mean, vs.std);
  std::fprintf(stderr, "Edges:    min=%d max=%d mean=%.3f std=%.3f\n",
               es.min_val, es.max_val, es.mean, es.std);
  std::fprintf(stderr, "Faces:    min=%d max=%d mean=%.3f std=%.3f\n",
               fs.min_val, fs.max_val, fs.mean, fs.std);

  return 0;
}