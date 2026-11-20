/**
 * @file sdf_voronoi_vtu.cpp
 * @brief Interstitial Voronoi tessellation of a sphere packing, with the packing's SDF as a wall,
 *        written as a ParaView unstructured grid (VTU, one VTK_POLYHEDRON per cell).
 *
 * Reads `packing.txt` (produced by pack_and_seed.py): a sphere packing + a set of seed points that
 * live in the interstitial space (outside every sphere). For each seed it builds the Voronoi cell
 * (bisectors against the other seeds, closest-first with the security-radius early-out), clips it to
 * the domain box, and clips it against the union-of-balls SDF so the sphere surfaces act as curved
 * walls (a few tangent-plane facets per sphere it touches). The clipped cells are emitted as solid
 * polyhedra to `cells.vtu`, coloured by cell volume + a boundary flag.
 *
 * Usage:  sdf_voronoi_vtu <packing.txt> <cells.vtu>
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/sdf.hpp"
#include "polyhedra_vtu.hpp"

using Real = double;
using Cell = peclet::voro::ConvexCell<Real, 128, 256>;

namespace {

// Union-of-balls SDF: sdf(x) = min_i(|x - c_i| - r_i)  (< 0 inside any ball, > 0 in the fluid,
// gradient points outward into the fluid — the suite convention). Host provider (eval + gradH).
struct SdfSpheres {
  const Real* c;  // 3*n centres
  const Real* r;  // n radii
  int n;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real m = Real(1e30);
    for (int i = 0; i < n; ++i) {
      const Real dx = x - c[3 * i], dy = y - c[3 * i + 1], dz = z - c[3 * i + 2];
      const Real d = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) - r[i];
      if (d < m) m = d;
    }
    return m;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

// The interstitial cell of seed i: built against all the other seeds (closest first, for the
// security radius' early-out), clipped to the domain AABB (box planes, -1) and to the sphere walls.
void buildSeedCell(Cell& c, const std::vector<Real>& seed, int nSeed, int i, Real lo, Real hi,
                   const SdfSpheres& sdf, std::vector<std::pair<Real, int>>& ord) {
  const Real sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
  ord.clear();
  for (int j = 0; j < nSeed; ++j) {
    if (j == i) continue;
    const Real dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
    ord.emplace_back(dx * dx + dy * dy + dz * dz, j);
  }
  std::sort(ord.begin(), ord.end());
  std::vector<Real> rx(ord.size()), ry(ord.size()), rz(ord.size());
  std::vector<int> ids(ord.size());
  for (size_t k = 0; k < ord.size(); ++k) {
    const int j = ord[k].second;
    rx[k] = seed[3 * j] - sx;
    ry[k] = seed[3 * j + 1] - sy;
    rz[k] = seed[3 * j + 2] - sz;
    ids[k] = j;
  }
  const Real Lbig[3] = {2 * (hi - lo), 2 * (hi - lo), 2 * (hi - lo)};  // box far away; walls added
  peclet::voro::buildConvexCell(c, Lbig, rx.data(), ry.data(), rz.data(), ids.data(),
                                (int)ord.size());
  const Real axes[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  const Real off6[6] = {hi - sx, sx - lo, hi - sy, sy - lo, hi - sz, sz - lo};
  for (int a = 0; a < 6; ++a) c.clip(axes[a], off6[a], -1);
  const Real seedW[3] = {sx, sy, sz};
  peclet::voro::clipCellAgainstSdf<Real, 128, 256, false>(c, seedW, sdf);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <packing.txt> <cells.vtu>\n", argv[0]);
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  int nSph = 0, nSeed = 0;
  Real lo = 0, hi = 1;
  in >> nSph >> nSeed >> lo >> hi;
  std::vector<Real> sc(3 * nSph), sr(nSph), seed(3 * nSeed);
  for (int i = 0; i < nSph; ++i) in >> sc[3 * i] >> sc[3 * i + 1] >> sc[3 * i + 2] >> sr[i];
  for (int i = 0; i < nSeed; ++i) in >> seed[3 * i] >> seed[3 * i + 1] >> seed[3 * i + 2];

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    SdfSpheres sdf{sc.data(), sr.data(), nSph};

    PolyhedraVtu<Real> vtu;
    std::vector<Real> cellVol;  // scalar for colouring
    std::vector<std::pair<Real, int>> ord;  // neighbours by distance²
    ord.reserve(nSeed);
    int nBuilt = 0;

    for (int i = 0; i < nSeed; ++i) {
      Cell c;
      buildSeedCell(c, seed, nSeed, i, lo, hi, sdf, ord);
      if (c.empty() || c.overflow) continue;
      if (!vtu.append(c, seed[3 * i], seed[3 * i + 1], seed[3 * i + 2])) continue;
      cellVol.push_back(c.volumePerVertex());
      ++nBuilt;
    }

    vtu.write(argv[2], {{"volume", &cellVol}});
    const long nPts = vtu.numPoints(), nCells = vtu.numCells();
    std::printf("wrote %s: %ld cells, %ld points (%d/%d seeds tessellated)\n", argv[2], nCells, nPts,
                nBuilt, nSeed);
  }
  Kokkos::finalize();
  return rc;
}
