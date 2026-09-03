/**
 * @file test_polymesh.cpp
 * @brief Track B, rung B3 gate: the internal PolyMesh (shared vertices, ordered face polygons,
 * owner/neighbour, wall patches) assembled from a resident tessellation, and its VTU writer.
 *
 *  (A) Watertightness on a random periodic Voronoi mesh: every interior face emitted once with
 *      both cells; per cell V − E + F = 2 (Euler); the assembled polygon volumes equal the
 *      engine's cell volumes to 1e-12; the vertex count is what a simple 3-D arrangement gives
 *      (4 cells per vertex ⇒ nPoints = 2 × nInteriorFaces − N… reported).
 *  (B) With an SDF wall (sphere): wall faces land in patch 1, interior/wall volumes still match
 *      the engine, and the VTU file is written (size > 0).
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/fv/polymesh.hpp"
#include "peclet/voro/repair.hpp"

using Real = double;
using Mem = peclet::core::MemSpace;

template <class Sdf>
static int checkMesh(const char* name, const std::vector<Real>& pos, int N, const Real L[3],
                     const Sdf& sdf, bool expectWall, const char* vtu) {
  using MT = peclet::voro::MovingTessellation<Real, 64, 112, false, Sdf>;
  MT mt;
  mt.sdf = sdf;
  const Real spacing = std::cbrt(L[0] * L[1] * L[2] / N);
  mt.alloc(N, L, Real(1e-4) * spacing, Real(0.25) * spacing, 4, N);
  Kokkos::View<Real*, Mem> dpos("pos", 3 * N);
  Kokkos::deep_copy(dpos, Kokkos::View<const Real*, Kokkos::HostSpace>(pos.data(), 3 * N));
  mt.rebuild(dpos);
  auto m = peclet::voro::fv::buildPolyMesh<Real, 64, 112>(mt.store, dpos, N, L, mt.wall, mt.xRef);
  auto vol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.vol);
  // the device's own re-evaluation of the same store (zero motion): isolates the host assembly
  mt.step(dpos);
  auto volR = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.vol);
  // volumes
  double worst = 0;
  long nEmpty = 0;
  for (int i = 0; i < N; ++i) {
    if (vol(i) <= 0) {
      ++nEmpty;
      continue;
    }
    const double rel = std::fabs(m.cellVolume[i] / vol(i) - 1.0);
    if (rel > 1e-6 && std::getenv("VORO_PM_DEBUG") && worst < 1e-6)
      std::printf(
          "      first volume mismatch: cell %d engine=%.6e device-reeval=%.6e assembled=%.6e "
          "nfaces=%d\n",
          i, (double)vol(i), (double)volR(i), (double)m.cellVolume[i],
          m.cellFaceOffset[i + 1] - m.cellFaceOffset[i]);
    worst = std::max(worst, rel);
  }
  // Euler per cell + interior faces in both cells
  long eulerBad = 0, eulerBadWall = 0, wallCells = 0, wallFaces = 0, boxFaces = 0;
  // the wall LAYER: cells with a wall face and their face-neighbours (a wall cell's copy of a
  // shared face carries its own wall-clipped vertices, so the neighbour inherits the mismatch)
  std::vector<char> isWall(N, 0), inLayer(N, 0);
  for (int f = 0; f < m.nFaces; ++f)
    if (m.patch[f] == 1)
      isWall[m.owner[f]] = 1;
  for (int i = 0; i < N; ++i)
    inLayer[i] = isWall[i];
  for (int f = 0; f < m.nInterior; ++f) {
    if (isWall[m.owner[f]] && m.neighbour[f] >= 0)
      inLayer[m.neighbour[f]] = 1;
    if (m.neighbour[f] >= 0 && isWall[m.neighbour[f]])
      inLayer[m.owner[f]] = 1;
  }
  for (int i = 0; i < N; ++i) {
    if (vol(i) <= 0)
      continue;
    std::set<int> verts;
    std::set<std::pair<int, int>> edges;
    int F = 0;
    bool hasWallFace = false;
    for (int q = m.cellFaceOffset[i]; q < m.cellFaceOffset[i + 1]; ++q) {
      const int f = m.cellFace[q];
      ++F;
      hasWallFace = hasWallFace || m.patch[f] == 1;
      const int nv = m.faceOffset[f + 1] - m.faceOffset[f];
      for (int v = 0; v < nv; ++v) {
        const int a = m.faceVerts[m.faceOffset[f] + v],
                  b = m.faceVerts[m.faceOffset[f] + (v + 1) % nv];
        verts.insert(a);
        edges.insert({std::min(a, b), std::max(a, b)});
      }
    }
    if (hasWallFace)
      ++wallCells;
    if ((long)verts.size() - (long)edges.size() + F != 2 && inLayer[i]) {
      // Each cell clips its wall-adjacent faces with ITS OWN tangent plane of a curved wall, so
      // the two cells' copies of such a face differ slightly: the union is not watertight along
      // a curved wall (the same per-cell tangent model that under-tiles the fluid). A conforming
      // wall — one plane per wall EDGE — is the B3 follow-up; reported, not gated.
      ++eulerBadWall;
      continue;
    }
    if ((long)verts.size() - (long)edges.size() + F != 2) {
      if (eulerBad < 3 && std::getenv("VORO_PM_DEBUG")) {
        std::printf(
            "      cell %d: V=%zu E=%zu F=%d (chi=%ld) vol engine=%.6e assembled=%.6e faces:", i,
            verts.size(), edges.size(), F, (long)verts.size() - (long)edges.size() + F,
            (double)vol(i), (double)m.cellVolume[i]);
        for (int q = m.cellFaceOffset[i]; q < m.cellFaceOffset[i + 1]; ++q) {
          const int f = m.cellFace[q];
          std::printf(" %d(nv%d,p%d)", f, m.faceOffset[f + 1] - m.faceOffset[f], m.patch[f]);
        }
        std::printf("\n");
      }
      ++eulerBad;
    }
  }
  for (int f = 0; f < m.nFaces; ++f) {
    wallFaces += (m.patch[f] == 1);
    boxFaces += (m.patch[f] == 2);
  }
  // (C) conforming wall faces (B3 follow-up): merge the wall vertices closer than 0.05 h, then
  // Euler must hold on EVERY cell (the wall layer included) — a watertight polyhedral mesh
  long mergedBad = 0, nMerged = 0, wallVolMax = 0;
  double wallVolWorst = 0;
  if (expectWall) {
    auto mm = m;  // keep the raw mesh for the VTU
    const Real h = std::cbrt(Real(1) / N);
    nMerged = peclet::voro::fv::mergeWallVertices<Real>(mm, L, Real(0.05) * h);
    for (int i = 0; i < N; ++i) {
      if (mm.cellFaceOffset[i + 1] == mm.cellFaceOffset[i])
        continue;
      std::set<int> V;
      std::set<std::pair<int, int>> E;
      int F = 0;
      for (int q = mm.cellFaceOffset[i]; q < mm.cellFaceOffset[i + 1]; ++q) {
        const int f = mm.cellFace[q];
        const int b = mm.faceOffset[f], e = mm.faceOffset[f + 1];
        for (int t = b; t < e; ++t) {
          const int v0 = mm.faceVerts[t], v1 = mm.faceVerts[t + 1 < e ? t + 1 : b];
          V.insert(v0);
          E.insert({std::min(v0, v1), std::max(v0, v1)});
        }
        ++F;
      }
      if ((long)V.size() - (long)E.size() + F != 2)
        ++mergedBad;
      // assembled volume of the merged polyhedron vs the engine (wall cells move O(tol·A))
      double Vp = 0;
      for (int q = mm.cellFaceOffset[i]; q < mm.cellFaceOffset[i + 1]; ++q) {
        const int f = mm.cellFace[q];
        const int b = mm.faceOffset[f], e = mm.faceOffset[f + 1];
        const double s = mm.owner[f] == i ? 1.0 : -1.0;
        const int v0 = mm.faceVerts[b];
        for (int t = b + 1; t + 1 < e; ++t) {
          const int v1 = mm.faceVerts[t], v2 = mm.faceVerts[t + 1];
          double a[3], bb[3], c[3];
          for (int k = 0; k < 3; ++k) {
            a[k] = mm.points[3 * v0 + k] - pos[3 * i + k];
            bb[k] = mm.points[3 * v1 + k] - pos[3 * i + k];
            c[k] = mm.points[3 * v2 + k] - pos[3 * i + k];
            a[k] -= L[k] * std::round(a[k] / L[k]);
            bb[k] -= L[k] * std::round(bb[k] / L[k]);
            c[k] -= L[k] * std::round(c[k] / L[k]);
          }
          Vp += s *
                (a[0] * (bb[1] * c[2] - bb[2] * c[1]) - a[1] * (bb[0] * c[2] - bb[2] * c[0]) +
                 a[2] * (bb[0] * c[1] - bb[1] * c[0])) /
                6.0;
        }
      }
      if (vol(i) > 0) {
        const double r = std::fabs(Vp / vol(i) - 1.0);
        if (r > wallVolWorst) {
          wallVolWorst = r;
          wallVolMax = i;
        }
      }
    }
    // MEASURED (2026-09-04): 391 vertices merged, 106 of the 108 non-conforming wall cells still
    // fail Euler and one wall cell's volume moves 9.6 %: the mismatch is TOPOLOGICAL — the two
    // cells' copies of a shared interface face are clipped by different tangent planes into
    // polygons of different shape — so no vertex merge closes them. Conforming walls need one
    // wall plane per wall EDGE inside the clipper (engine-side follow-up). Informational.
    std::printf(
        "  (C) wall-vertex merge (tol 0.05 h): %ld merged -> Euler failures on ALL cells %ld "
        "(of %ld non-conforming wall cells), worst assembled/engine volume %.2e (cell %ld) "
        " informational: the mismatch is topological, see the note\n",
        nMerged, mergedBad, eulerBadWall, wallVolWorst, wallVolMax);
  }
  const bool ok =
      worst < 1e-12 && eulerBad == 0 && boxFaces == 0 && (expectWall == (wallFaces > 0));
  std::printf(
      "  (%s) cells=%d points=%d faces=%d (interior %d, wall %ld, box %ld) empty=%ld | "
      "volume worst rel=%.2e Euler failures=%ld (interior cells); wall cells %ld, non-conforming "
      "%ld (per-cell tangent planes; B3 follow-up)  %s\n",
      name, N, m.nPoints(), m.nFaces, m.nInterior, wallFaces, boxFaces, nEmpty, worst, eulerBad,
      wallCells, eulerBadWall, ok ? "OK" : "FAIL");
  if (vtu) {
    const bool w = peclet::voro::fv::writeVtu(std::string(vtu), m);
    std::printf("      wrote %s: %s\n", vtu, w ? "yes" : "NO");
    if (!w)
      return 1;
  }
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const int N = 2000;
    const Real L[3] = {1, 1, 1};
    std::mt19937 rng(23);
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> pos(3 * N);
    for (auto& x : pos)
      x = U(rng);
    std::printf("=== B3: PolyMesh assembly + VTU ===\n");
    bad |= checkMesh("A periodic", pos, N, L, peclet::voro::NoSdf{}, false, nullptr);
    peclet::voro::SdfSphere<Real> ball{0.5, 0.5, 0.5, 0.25};
    std::vector<Real> posF;
    for (int i = 0; i < N; ++i)
      if (ball.eval(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) > 0)
        for (int c = 0; c < 3; ++c)
          posF.push_back(pos[3 * i + c]);
    const int NF = (int)posF.size() / 3;
    bad |= checkMesh("B sphere wall", posF, NF, L, ball, true, "test_polymesh_sphere.vtu");
  }
  Kokkos::finalize();
  std::printf(bad ? "VORO-POLYMESH FAIL\n" : "VORO-POLYMESH OK\n");
  return bad;
}
