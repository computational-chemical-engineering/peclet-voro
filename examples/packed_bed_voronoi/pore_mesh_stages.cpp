/**
 * @file pore_mesh_stages.cpp
 * \brief Four stages of meshing the interstitial pore space of a periodic sphere packing with a
 * volume-controlled Voronoi tessellation, each exported as a VTU (clipped VTK_POLYHEDRON cells,
 * coloured by volume, V/V_ref, and a wall-touch flag) for the visual study in the peclet-examples
 * draft "pore-mesh-voronoi". Reads a `packing.txt` (from pack_bed.py) for the spheres; seeds and
 * optimises internally.
 *
 *   stage 1  random      — uniform interstitial seeds, raw Voronoi (no optimisation)
 *   stage 2  equalized   — stage-1 seeds relaxed toward a UNIFORM target volume
 *   stage 3  graded-seed — seeds placed with density ∝ 1/V_ref, V_ref = clamp(sdf, sLo, sHi)³
 *                          (dense near the walls — an inflation layer, coarser in the bulk), raw
 *   stage 4  graded-relax— stage-3 seeds relaxed toward that graded V_ref
 *
 * Everything is periodic (min-image), matching peclet::voro::meshVolumeOptimize. Relaxation uses
 * steepest descent on the wall-aware volume gradient (the method that moves; see the bench study).
 *
 * Build: examples/packed_bed_voronoi/CMakeLists.txt. Run: ./pore_mesh_stages packing.txt outdir N
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <vector>

#include "peclet/voro/mesh_optimizer.hpp"  // meshVolumeOptimize, Precond, ConvexCell, buildConvexCell, clip
#include "polyhedra_vtu.hpp"

using Real = double;
using Cell = peclet::voro::ConvexCell<Real, 128, 256>;

namespace {

// Periodic union-of-balls SDF (min-image): sdf<0 inside a sphere, >0 in the fluid.
struct SdfSpheres {
  const Real* c;
  const Real* r;
  int n;
  Real L;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real y, Real z) const {
    Real m = Real(1e30);
    for (int i = 0; i < n; ++i) {
      Real dx = x - c[3 * i], dy = y - c[3 * i + 1], dz = z - c[3 * i + 2];
      dx -= L * Kokkos::round(dx / L);
      dy -= L * Kokkos::round(dy / L);
      dz -= L * Kokkos::round(dz / L);
      const Real d = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) - r[i];
      if (d < m) m = d;
    }
    return m;
  }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-4); }
};

Real hostSdf(const std::vector<Real>& sc, const std::vector<Real>& sr, int M, Real L, Real x, Real y,
             Real z) {
  Real m = 1e30;
  for (int i = 0; i < M; ++i) {
    Real dx = x - sc[3 * i], dy = y - sc[3 * i + 1], dz = z - sc[3 * i + 2];
    dx -= L * std::round(dx / L);
    dy -= L * std::round(dy / L);
    dz -= L * std::round(dz / L);
    m = std::min(m, std::sqrt(dx * dx + dy * dy + dz * dz) - sr[i]);
  }
  return m;
}

// Graded reference "cell size" V_ref(x) = clamp(sdf, sLo, sHi)³ — small at the walls, capped in bulk.
constexpr Real kSlo = 0.06, kShi = 0.35;
Real vrefOf(Real phi) {
  const Real s = std::min(kShi, std::max(kSlo, phi));
  return s * s * s;
}

// The interstitial cell of seed i: built against its 80 nearest periodic (min-image) neighbours
// and clipped to the sphere walls. `ord` is scratch.
void buildSeedCell(Cell& c, const std::vector<Real>& seed, int N, int i, const SdfSpheres& sdf,
                   Real L, std::vector<std::pair<Real, int>>& ord) {
  const Real Lh = 0.5 * L;
  const Real big = 4 * L;
  const Real sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
  ord.clear();
  for (int j = 0; j < N; ++j) {
    if (j == i) continue;
    Real dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
    dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
    dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
    dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
    ord.emplace_back(dx * dx + dy * dy + dz * dz, j);
  }
  std::sort(ord.begin(), ord.end());
  const int M = std::min((int)ord.size(), 80);
  std::vector<Real> rx(M), ry(M), rz(M);
  std::vector<int> ids(M);
  for (int k = 0; k < M; ++k) {
    const int j = ord[k].second;
    Real dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
    dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
    dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
    dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
    rx[k] = dx; ry[k] = dy; rz[k] = dz; ids[k] = j;
  }
  const Real Lbig[3] = {big, big, big};
  peclet::voro::buildConvexCell(c, Lbig, rx.data(), ry.data(), rz.data(), ids.data(), M);
  const Real seedW[3] = {sx, sy, sz};
  peclet::voro::clipCellAgainstSdf<Real, 128, 256, false>(c, seedW, sdf);
}

// Reconstruct every interstitial cell (periodic min-image neighbours + SDF clip) and write a VTU of
// clipped polyhedra, coloured by volume, V/V_ref (rel), and a wall-touch flag.
void writeVtu(const std::vector<Real>& seed, int N, const SdfSpheres& sdf, Real L,
              const std::vector<Real>& vref, const char* fname) {
  PolyhedraVtu<Real> vtu;
  std::vector<Real> cellVol, cellRel, cellVref;
  std::vector<std::pair<Real, int>> ord;
  for (int i = 0; i < N; ++i) {
    Cell c;
    buildSeedCell(c, seed, N, i, sdf, L, ord);
    if (c.empty() || c.overflow) continue;
    if (!vtu.append(c, seed[3 * i], seed[3 * i + 1], seed[3 * i + 2])) continue;
    const Real V = c.volumePerVertex();
    cellVol.push_back(V);
    cellVref.push_back(vref.empty() ? 1.0 : vref[i]);
  }
  // rel = V / V_ref, with V_ref renormalised to the actual total cell volume (Σ V_ref = Σ V), so
  // rel = 1 is exactly on target (the optimiser renormalises internally the same way).
  cellRel.resize(cellVol.size(), 1.0);
  if (!vref.empty()) {
    double sV = 0, sR = 0;
    for (std::size_t k = 0; k < cellVol.size(); ++k) { sV += cellVol[k]; sR += cellVref[k]; }
    const double scale = (sR > 0) ? sV / sR : 1.0;
    for (std::size_t k = 0; k < cellVol.size(); ++k)
      cellRel[k] = cellVol[k] / (cellVref[k] * scale);
  }
  vtu.write(fname, {{"volume", &cellVol}, {"rel", &cellRel}});
  std::printf("  wrote %s: %ld cells\n", fname, vtu.numCells());
}

// Reject-sample seeds; `graded` weights the acceptance by 1/V_ref (density ∝ 1/cell-size).
std::vector<Real> seedPoints(const std::vector<Real>& sc, const std::vector<Real>& sr, int M, Real L,
                             int Nreq, Real margin, bool graded, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<Real> U(0.0, L), U01(0.0, 1.0);
  std::vector<Real> pos;
  const Real vrefMin = vrefOf(kSlo);  // densest acceptance reference
  while ((int)(pos.size() / 3) < Nreq) {
    Real x = U(rng), y = U(rng), z = U(rng);
    const Real phi = hostSdf(sc, sr, M, L, x, y, z);
    if (phi <= margin) continue;
    if (graded) {
      const Real accept = vrefMin / vrefOf(phi);  // ∝ 1/V_ref, ∈ (0,1]
      if (U01(rng) > accept) continue;
    }
    pos.push_back(x); pos.push_back(y); pos.push_back(z);
  }
  return pos;
}

// Drop cells below a volume margin so the relaxation starts feasible (no collapsed cells).
std::vector<Real> prune(std::vector<Real> pos, const SdfSpheres& sdf, Real L, int sw,
                        Kokkos::View<long*, peclet::core::MemSpace>& gd) {
  for (int round = 0; round < 10; ++round) {
    const int n = (int)(pos.size() / 3);
    const Real Larr[3] = {L, L, L};
    Kokkos::View<Real*, peclet::core::MemSpace> dp("p", 3 * n), dw;
    Kokkos::deep_copy(dp, Kokkos::View<const Real*, Kokkos::HostSpace>(pos.data(), 3 * n));
    auto res = peclet::voro::buildTessellation<Real, false, SdfSpheres>(dp, dw, n, Larr, sw, n, gd,
                                                                        sdf, true);
    auto vol = peclet::voro::detail::toHostVec<Real>(res.view.cellVolume);
    double s = 0;
    int m = 0;
    for (int c = 0; c < n; ++c)
      if (vol[c] > 0) { s += vol[c]; ++m; }
    const double thr = 0.2 * (m ? s / m : 0);
    std::vector<Real> keep;
    for (int c = 0; c < n; ++c)
      if (vol[c] > thr) {
        keep.push_back(pos[3 * c]); keep.push_back(pos[3 * c + 1]); keep.push_back(pos[3 * c + 2]);
      }
    const int removed = n - (int)(keep.size() / 3);
    pos.swap(keep);
    if (!removed) break;
  }
  return pos;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <packing.txt> <outdir> [N=4000]\n", argv[0]);
    return 2;
  }
  const std::string outdir = argv[2];
  const int Nreq = (argc > 3) ? std::atoi(argv[3]) : 4000;

  std::ifstream in(argv[1]);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  int M = 0;
  Real L = 1;
  in >> M >> L;
  std::vector<Real> sc(3 * M), sr(M);
  for (int i = 0; i < M; ++i) in >> sc[3 * i] >> sc[3 * i + 1] >> sc[3 * i + 2] >> sr[i];
  std::printf("packing: M=%d spheres, L=%.3f\n", M, L);

  Kokkos::initialize(argc, argv);
  {
    SdfSpheres sdf{sc.data(), sr.data(), M, L};
    Kokkos::View<long*, peclet::core::MemSpace> gd;
    const int sw = 6;
    const Real Larr[3] = {L, L, L};
    double meanR = 0;
    for (int i = 0; i < M; ++i) meanR += sr[i];
    meanR /= M;
    const Real margin = 0.05 * meanR;
    std::vector<Real> noW, empty;

    // ---- stage 1: uniform random interstitial seeds, raw Voronoi ----
    std::printf("stage 1: random uniform seeds\n");
    auto s1 = prune(seedPoints(sc, sr, M, L, (int)(1.25 * Nreq), margin, false, 1u), sdf, L, sw, gd);
    const int N1 = (int)(s1.size() / 3);
    writeVtu(s1, N1, sdf, L, empty, (outdir + "/stage1_random.vtu").c_str());

    // ---- stage 2: relax stage-1 toward a UNIFORM V_ref ----
    std::printf("stage 2: uniform-target relaxation\n");
    std::vector<Real> s2 = s1, vsetU(N1, L * L * L / N1);
    peclet::voro::meshVolumeOptimize<Real, false, SdfSpheres>(s2, noW, vsetU, Larr, N1, sw, sdf, 80,
                                                              1e-9, 400, peclet::voro::Precond::SteepestDescent);
    writeVtu(s2, N1, sdf, L, vsetU, (outdir + "/stage2_equalized.vtu").c_str());

    // ---- stage 3: graded seeding (density ∝ 1/V_ref), raw ----
    std::printf("stage 3: graded seeding (density ∝ 1/V_ref = 1/sdf^3)\n");
    auto s3 = prune(seedPoints(sc, sr, M, L, (int)(1.25 * Nreq), margin, true, 3u), sdf, L, sw, gd);
    const int N3 = (int)(s3.size() / 3);
    std::vector<Real> vset3(N3);
    for (int i = 0; i < N3; ++i)
      vset3[i] = vrefOf(hostSdf(sc, sr, M, L, s3[3 * i], s3[3 * i + 1], s3[3 * i + 2]));
    writeVtu(s3, N3, sdf, L, vset3, (outdir + "/stage3_graded_seed.vtu").c_str());

    // ---- stage 4: relax stage-3 toward the graded V_ref ----
    std::printf("stage 4: graded relaxation\n");
    std::vector<Real> s4 = s3;
    peclet::voro::meshVolumeOptimize<Real, false, SdfSpheres>(s4, noW, vset3, Larr, N3, sw, sdf, 80,
                                                              1e-9, 400, peclet::voro::Precond::SteepestDescent);
    std::vector<Real> vset4(N3);
    for (int i = 0; i < N3; ++i)
      vset4[i] = vrefOf(hostSdf(sc, sr, M, L, s4[3 * i], s4[3 * i + 1], s4[3 * i + 2]));
    writeVtu(s4, N3, sdf, L, vset4, (outdir + "/stage4_graded_relaxed.vtu").c_str());

    // spheres for the plotter
    std::ofstream sp(outdir + "/spheres.txt");
    sp << M << ' ' << L << '\n';
    for (int i = 0; i < M; ++i)
      sp << sc[3 * i] << ' ' << sc[3 * i + 1] << ' ' << sc[3 * i + 2] << ' ' << sr[i] << '\n';
  }
  Kokkos::finalize();
  return 0;
}
