/**
 * @file test_pervertex_geometry.cpp
 * \brief Acceptance criteria for the vertex-local, SORT-FREE geometry (design note §7).
 *
 * Validates ConvexCell::volumePerVertex / facetAreasPerVertex (flag/divergence construction, no
 * vertex ordering, no adjacency) against the ordered reference (volume()/faceOrdered area), the
 * internal divergence identity, label invariance, obtuse/skinny robustness, and a finite-difference
 * gradient check. FP64. Cells are built by clipping random nearest neighbours (incl. an anisotropic
 * batch that produces feet-outside-face cells). Host-side, single thread.
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "vorflow/device/convex_cell.hpp"

using real_t = double;
using Cell = vor::device::ConvexCell<real_t, 64, 128>;

// build one cell at site `i` from its K nearest neighbours (periodic min-image, security break)
static bool buildCell(Cell& c, int i, const std::vector<real_t>& pos, int N, real_t L,
                      const std::vector<int>& nbr) {
  const real_t sx = pos[3 * i], sy = pos[3 * i + 1], sz = pos[3 * i + 2], Lh = 0.5 * L;
  c.initBox(L, L, L);
  real_t secR2 = 2.0 * c.maxVertexRsq();
  for (int g : nbr) {
    real_t rx = pos[3 * g] - sx, ry = pos[3 * g + 1] - sy, rz = pos[3 * g + 2] - sz;
    rx = rx > Lh ? rx - L : (rx < -Lh ? rx + L : rx);
    ry = ry > Lh ? ry - L : (ry < -Lh ? ry + L : ry);
    rz = rz > Lh ? rz - L : (rz < -Lh ? rz + L : rz);
    const real_t off = 0.5 * (rx * rx + ry * ry + rz * rz);
    if (off >= secR2) break;
    const real_t n[3] = {rx, ry, rz};
    if (c.clip(n, off, g)) secR2 = 2.0 * c.maxVertexRsq();
    if (c.overflow) return false;
  }
  return !c.overflow;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int N = 60000;
    const real_t L = 1.0, spacing = std::cbrt(1.0 / N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<real_t> U(0, 1);

    // two batches: isotropic, and anisotropic (squashed z -> skinny/obtuse cells, feet outside faces)
    for (int batch = 0; batch < 2; ++batch) {
      const real_t sz_scale = (batch == 0) ? 1.0 : 0.35;  // squash z in batch 1
      std::vector<real_t> pos(3 * N);
      for (int i = 0; i < N; ++i) {
        pos[3 * i] = U(rng); pos[3 * i + 1] = U(rng); pos[3 * i + 2] = U(rng) * sz_scale + 0.5 * (1 - sz_scale);
      }
      // neighbour lists (brute-force grid)
      const int dim = std::max(1, (int)std::floor(1.0 / spacing));
      const real_t csz = 1.0 / dim;
      std::vector<std::vector<int>> grid(dim * dim * dim);
      auto gid = [&](int x, int y, int z) {
        return (x + dim) % dim + dim * (((y + dim) % dim) + dim * ((z + dim) % dim));
      };
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz), gz = (int)(pos[3 * i + 2] / csz);
        grid[gid(std::min(gx, dim - 1), std::min(gy, dim - 1), std::min(gz, dim - 1))].push_back(i);
      }

      double maxV = 0, maxDiv = 0, maxArea = 0, maxAreaAbs = 0, maxLabel = 0, maxDVforce = 0;
      long nchecked = 0, nfaceChecked = 0, nGradTot = 0, nGradPass = 0;
      double cellScale = spacing * spacing;  // typical face area ~ spacing²
      const int sw = 3;
      std::vector<std::array<real_t, 2>> tmp;
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz), gz = (int)(pos[3 * i + 2] / csz);
        gx = std::min(gx, dim - 1); gy = std::min(gy, dim - 1); gz = std::min(gz, dim - 1);
        tmp.clear();
        for (int dz = -sw; dz <= sw; ++dz)
          for (int dy = -sw; dy <= sw; ++dy)
            for (int dx = -sw; dx <= sw; ++dx)
              for (int j : grid[gid(gx + dx, gy + dy, gz + dz)]) {
                if (j == i) continue;
                real_t rx = pos[3 * j] - pos[3 * i], ry = pos[3 * j + 1] - pos[3 * i + 1],
                       rz = pos[3 * j + 2] - pos[3 * i + 2];
                rx -= std::round(rx); ry -= std::round(ry); rz -= std::round(rz);
                tmp.push_back({rx * rx + ry * ry + rz * rz, (real_t)j});
              }
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        std::vector<int> nbr;
        for (int t = 0; t < (int)tmp.size() && t < 64; ++t) nbr.push_back((int)tmp[t][1]);

        Cell c;
        if (!buildCell(c, i, pos, N, L, nbr)) continue;
        const double Vref = c.volume();
        if (Vref <= 0) continue;
        ++nchecked;

        // (1) volume: vertex-local vs ordered reference
        const double Vpv = c.volumePerVertex();
        maxV = std::max(maxV, std::fabs(Vpv - Vref) / Vref);

        // areas + first moments (vertex-local scatter)
        double area[64], mx[64], my[64], mz[64];
        for (int k = 0; k < c.np; ++k) { area[k] = mx[k] = my[k] = mz[k] = 0.0; }
        c.facetMomentsPerVertex(area, mx, my, mz);

        // (4) divergence identity  V == (1/3) Σ_f |n_f| A_f   (|n_f| = pd/|pn|)
        double Vdiv = 0;
        for (int k = 0; k < c.np; ++k) {
          const double pl = std::sqrt(c.pn[k][0] * c.pn[k][0] + c.pn[k][1] * c.pn[k][1] + c.pn[k][2] * c.pn[k][2]);
          if (pl > 0) Vdiv += (c.pd[k] / pl) * area[k];
        }
        Vdiv /= 3.0;
        maxDiv = std::max(maxDiv, std::fabs(Vdiv - Vpv) / Vref);

        // (3) per-facet areas vs ordered polygon area
        real_t fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
        for (int k = 0; k < c.np; ++k) {
          const int m = c.faceOrdered(k, fx, fy, fz);
          if (m < 3) continue;
          real_t A[3];
          Cell::polyAreaVec(fx, fy, fz, m, A);
          const double aref = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
          maxAreaAbs = std::max(maxAreaAbs, std::fabs(area[k] - aref) / cellScale);  // abs, scale-normed
          if (aref > 0.05 * cellScale) {  // skip near-degenerate faces (relative metric blows up there)
            maxArea = std::max(maxArea, std::fabs(area[k] - aref) / aref);
            ++nfaceChecked;
          }
        }

        // (7) volume gradient / FORCE: per-vertex dV vs the oracle-validated facetGeometry.dv
        real_t aV[3], dvRef[3], cn[3];
        for (int k = 6; k < c.np; ++k) {
          if (area[k] < 0.05 * cellScale) continue;
          if (!c.facetGeometry(k, aV, dvRef, cn)) continue;
          const double rl = std::sqrt(c.pn[k][0] * c.pn[k][0] + c.pn[k][1] * c.pn[k][1] + c.pn[k][2] * c.pn[k][2]);
          const double s = area[k] / rl;
          const double cxk = mx[k] / area[k], cyk = my[k] / area[k], czk = mz[k] / area[k];
          const double dvx = s * (c.pn[k][0] - cxk), dvy = s * (c.pn[k][1] - cyk), dvz = s * (c.pn[k][2] - czk);
          const double err = std::sqrt((dvx - dvRef[0]) * (dvx - dvRef[0]) + (dvy - dvRef[1]) * (dvy - dvRef[1]) +
                                       (dvz - dvRef[2]) * (dvz - dvRef[2]));
          const double mag = std::sqrt(dvRef[0] * dvRef[0] + dvRef[1] * dvRef[1] + dvRef[2] * dvRef[2]);
          if (mag > 1e-9) maxDVforce = std::max(maxDVforce, err / mag);
        }

        // (5) label invariance: cyclically rotate every triangle's plane triple
        Cell c2 = c;
        for (int t = 0; t < c2.nt; ++t)
          if (c2.alive[t]) {
            unsigned char a = c2.t0[t]; c2.t0[t] = c2.t1[t]; c2.t1[t] = c2.t2[t]; c2.t2[t] = a;
          }
        maxLabel = std::max(maxLabel, std::fabs(c2.volumePerVertex() - Vpv) / Vref);

        // (6) gradient: dV/d(pd_f) finite-diff vs analytic A_f/|pn_f|  (one real face per cell)
        int kf = -1;
        for (int k = 6; k < c.np; ++k)
          if (area[k] > 1e-6) { kf = k; break; }
        if (kf >= 0) {
          const double pl = std::sqrt(c.pn[kf][0] * c.pn[kf][0] + c.pn[kf][1] * c.pn[kf][1] + c.pn[kf][2] * c.pn[kf][2]);
          const double eps = 1e-8;  // small step; topology stays fixed away from combinatorial events
          auto Vat = [&](double dpd) {
            Cell cc = c;
            cc.pd[kf] += dpd;
            for (int t = 0; t < cc.nt; ++t) if (cc.alive[t]) cc.computeVertex(t);
            return cc.volumePerVertex();
          };
          const double dVdpd = (Vat(eps) - Vat(-eps)) / (2 * eps);
          const double analytic = area[kf] / pl;  // dV/dh=A_f, dh/dpd=1/|pn|
          ++nGradTot;
          if (std::fabs(dVdpd - analytic) / std::fabs(analytic) < 1e-4) ++nGradPass;  // smooth cells
        }
      }
      std::printf("batch %d (%s): cells=%ld faces=%ld\n", batch, batch ? "anisotropic/obtuse" : "isotropic",
                  nchecked, nfaceChecked);
      const double gradFrac = nGradTot ? (double)nGradPass / nGradTot : 1.0;
      std::printf("  (1) Vpv vs ordered       max rel = %.3e\n", maxV);
      std::printf("  (3) areas vs ordered     max rel(non-degenerate) = %.3e   max abs/scale = %.3e\n",
                  maxArea, maxAreaAbs);
      std::printf("  (4) divergence identity  max rel = %.3e\n", maxDiv);
      std::printf("  (5) label invariance     max rel = %.3e\n", maxLabel);
      std::printf("  (7) dV(force) vs oracle  max rel = %.3e\n", maxDVforce);
      std::printf("  (6) gradient (FD<1e-4)   %.4f%% of cells (rest are at topology events)\n",
                  100.0 * gradFrac);
      const double tol = 1e-9;
      if (maxV > tol || maxArea > tol || maxDiv > tol || maxLabel > 1e-12 || maxAreaAbs > tol || maxDVforce > 1e-9 ||
          gradFrac < 0.98) {
        std::printf("  FAIL\n");
        rc = 1;
      } else {
        std::printf("  PASS\n");
      }
    }
  }
  Kokkos::finalize();
  return rc;
}
