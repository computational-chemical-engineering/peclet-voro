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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "vorflow/device/convex_cell.hpp"
#include "vorflow/device/plane_policy.hpp"

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
    if (off >= secR2)
      break;
    const real_t n[3] = {rx, ry, rz};
    if (c.clip(n, off, g))
      secR2 = 2.0 * c.maxVertexRsq();
    if (c.overflow)
      return false;
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

    // two batches: isotropic, and anisotropic (squashed z -> skinny/obtuse cells, feet outside
    // faces)
    for (int batch = 0; batch < 2; ++batch) {
      const real_t sz_scale = (batch == 0) ? 1.0 : 0.35;  // squash z in batch 1
      std::vector<real_t> pos(3 * N);
      for (int i = 0; i < N; ++i) {
        pos[3 * i] = U(rng);
        pos[3 * i + 1] = U(rng);
        pos[3 * i + 2] = U(rng) * sz_scale + 0.5 * (1 - sz_scale);
      }
      // neighbour lists (brute-force grid)
      const int dim = std::max(1, (int)std::floor(1.0 / spacing));
      const real_t csz = 1.0 / dim;
      std::vector<std::vector<int>> grid(dim * dim * dim);
      auto gid = [&](int x, int y, int z) {
        return (x + dim) % dim + dim * (((y + dim) % dim) + dim * ((z + dim) % dim));
      };
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz),
            gz = (int)(pos[3 * i + 2] / csz);
        grid[gid(std::min(gx, dim - 1), std::min(gy, dim - 1), std::min(gz, dim - 1))].push_back(i);
      }

      double maxV = 0, maxDiv = 0, maxArea = 0, maxAreaAbs = 0, maxLabel = 0, maxDVforce = 0,
             maxMerged = 0;
      double maxGrad = 0, maxGradVol = 0;  // (9) geomVolumeGrad vs closed-form dV/dn
      double maxAreaVec = 0, maxGradA = 0,
             maxAreaVol = 0;  // (10) geomVolumeArea vs facetGeometry/closed form
      double maxAreaGather = 0, maxdAfd = 0, maxAreaJacAD = 0;  // (11) geomFull dA/dn
      long ndAtot = 0, ndApass = 0, nFull = 0;
      double maxChain = 0;
      long nChainTot = 0, nChainPass = 0, nChainCells = 0;  // (12) Voronoi chainToDofs
      long nchecked = 0, nfaceChecked = 0, nGradTot = 0, nGradPass = 0;
      double cellScale = spacing * spacing;  // typical face area ~ spacing²
      const int sw = 3;
      std::vector<std::array<real_t, 2>> tmp;
      for (int i = 0; i < N; ++i) {
        int gx = (int)(pos[3 * i] / csz), gy = (int)(pos[3 * i + 1] / csz),
            gz = (int)(pos[3 * i + 2] / csz);
        gx = std::min(gx, dim - 1);
        gy = std::min(gy, dim - 1);
        gz = std::min(gz, dim - 1);
        tmp.clear();
        for (int dz = -sw; dz <= sw; ++dz)
          for (int dy = -sw; dy <= sw; ++dy)
            for (int dx = -sw; dx <= sw; ++dx)
              for (int j : grid[gid(gx + dx, gy + dy, gz + dz)]) {
                if (j == i)
                  continue;
                real_t rx = pos[3 * j] - pos[3 * i], ry = pos[3 * j + 1] - pos[3 * i + 1],
                       rz = pos[3 * j + 2] - pos[3 * i + 2];
                rx -= std::round(rx);
                ry -= std::round(ry);
                rz -= std::round(rz);
                tmp.push_back({rx * rx + ry * ry + rz * rz, (real_t)j});
              }
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        std::vector<int> nbr;
        for (int t = 0; t < (int)tmp.size() && t < 64; ++t)
          nbr.push_back((int)tmp[t][1]);

        Cell c;
        if (!buildCell(c, i, pos, N, L, nbr))
          continue;
        const double Vref = c.volume();
        if (Vref <= 0)
          continue;
        ++nchecked;

        // (1) volume: vertex-local vs ordered reference
        const double Vpv = c.volumePerVertex();
        maxV = std::max(maxV, std::fabs(Vpv - Vref) / Vref);

        // areas + first moments (vertex-local scatter)
        double area[64], mx[64], my[64], mz[64];
        for (int k = 0; k < c.np; ++k) {
          area[k] = mx[k] = my[k] = mz[k] = 0.0;
        }
        c.facetMomentsPerVertex(area, mx, my, mz);

        // merged G1+G2 kernel must match the separate calls
        {
          double vg = 0, ag[64], gmx[64], gmy[64], gmz[64];
          for (int k = 0; k < c.np; ++k) {
            ag[k] = gmx[k] = gmy[k] = gmz[k] = 0.0;
          }
          c.geometryPerVertex(vg, ag, gmx, gmy, gmz);
          maxMerged = std::max(maxMerged, std::fabs(vg - Vpv) / Vref);
          for (int k = 0; k < c.np; ++k)
            maxMerged = std::max(maxMerged, std::fabs(ag[k] - area[k]) / cellScale);
        }

        // (9) geomVolumeGrad: areas-free reverse-scatter dV/dn vs the closed form (2·area·n −
        // m)/|n|
        {
          double vg = 0, dgx[64], dgy[64], dgz[64];
          for (int k = 0; k < c.np; ++k) {
            dgx[k] = dgy[k] = dgz[k] = 0.0;
          }
          c.geomVolumeGrad(vg, dgx, dgy, dgz);
          maxGradVol = std::max(maxGradVol, std::fabs(vg - Vpv) / Vref);
          for (int k = 0; k < c.np; ++k) {
            if (area[k] < 0.05 * cellScale)
              continue;  // skip near-degenerate facets (mag → 0)
            const double invn = 1.0 / std::sqrt(c.nn[k]);
            const double cfx = (2.0 * area[k] * c.n[k][0] - mx[k]) * invn;
            const double cfy = (2.0 * area[k] * c.n[k][1] - my[k]) * invn;
            const double cfz = (2.0 * area[k] * c.n[k][2] - mz[k]) * invn;
            const double err =
                std::sqrt((dgx[k] - cfx) * (dgx[k] - cfx) + (dgy[k] - cfy) * (dgy[k] - cfy) +
                          (dgz[k] - cfz) * (dgz[k] - cfz));
            const double mag = std::sqrt(cfx * cfx + cfy * cfy + cfz * cfz);
            if (mag > 1e-9)
              maxGrad = std::max(maxGrad, err / mag);
          }
        }

        // (10) geomVolumeArea: areas-free sqrt-free area-vectors + dV/dn vs facetGeometry / closed
        // form
        {
          double varea = 0, avx[64], avy[64], avz[64], dgx[64], dgy[64], dgz[64];
          for (int k = 0; k < c.np; ++k) {
            avx[k] = avy[k] = avz[k] = dgx[k] = dgy[k] = dgz[k] = 0.0;
          }
          c.geomVolumeArea(varea, avx, avy, avz, dgx, dgy, dgz);
          maxAreaVol = std::max(maxAreaVol, std::fabs(varea - Vpv) / Vref);
          real_t aV[3], dvRef[3], cn[3];
          for (int k = 6; k < c.np; ++k) {
            if (area[k] < 0.05 * cellScale)
              continue;
            // area-vector vs facetGeometry's oriented polygon area vector
            if (c.facetGeometry(k, aV, dvRef, cn)) {
              const double aerr = std::sqrt((avx[k] - aV[0]) * (avx[k] - aV[0]) +
                                            (avy[k] - aV[1]) * (avy[k] - aV[1]) +
                                            (avz[k] - aV[2]) * (avz[k] - aV[2]));
              const double amag =
                  std::sqrt((double)aV[0] * aV[0] + (double)aV[1] * aV[1] + (double)aV[2] * aV[2]);
              if (amag > 1e-9)
                maxAreaVec = std::max(maxAreaVec, aerr / amag);
            }
            // dV/dn vs closed form (2·area·n − m)/|n|
            const double invn = 1.0 / std::sqrt(c.nn[k]);
            const double cfx = (2.0 * area[k] * c.n[k][0] - mx[k]) * invn;
            const double cfy = (2.0 * area[k] * c.n[k][1] - my[k]) * invn;
            const double cfz = (2.0 * area[k] * c.n[k][2] - mz[k]) * invn;
            const double err =
                std::sqrt((dgx[k] - cfx) * (dgx[k] - cfx) + (dgy[k] - cfy) * (dgy[k] - cfy) +
                          (dgz[k] - cfz) * (dgz[k] - cfz));
            const double mag = std::sqrt(cfx * cfx + cfy * cfy + cfz * cfz);
            if (mag > 1e-9)
              maxGradA = std::max(maxGradA, err / mag);
          }
        }

        // (11) geomFull: gather the per-triangle area-Jacobian geomVolumeAreaGrad into the full
        // dA_k/dn_l and (on a sample) finite-difference check it. Expensive, so cap the sample per
        // batch.
        if (nFull < 60) {
          ++nFull;
          const int np = c.np;
          std::vector<double> dA((size_t)np * np * 3, 0.0), Ag(np, 0.0);
          for (int t = 0; t < c.nt; ++t) {
            if (!c.alive[t])
              continue;
            int pl[3];
            double cb[3], gr[3][3][3];
            c.geomVolumeAreaGrad(t, pl, cb, gr);
            {  // analytic vs forward-AD oracle (must match to machine precision)
              int plA[3];
              double cbA[3], grA[3][3][3];
              c.geomVolumeAreaGradAD(t, plA, cbA, grA);
              double sc = 0;
              for (int ii = 0; ii < 3; ++ii)
                sc = std::max(sc, std::fabs(cb[ii]));
              sc = std::max(sc, 1e-30);
              for (int ii = 0; ii < 3; ++ii)
                for (int jj = 0; jj < 3; ++jj)
                  for (int dd = 0; dd < 3; ++dd)
                    maxAreaJacAD = std::max(
                        maxAreaJacAD, std::fabs(gr[ii][jj][dd] - grA[ii][jj][dd]) / (sc / spacing));
            }
            for (int ii = 0; ii < 3; ++ii) {
              Ag[pl[ii]] += cb[ii];
              for (int jj = 0; jj < 3; ++jj)
                for (int cc = 0; cc < 3; ++cc)
                  dA[((size_t)pl[ii] * np + pl[jj]) * 3 + cc] += gr[ii][jj][cc];
            }
          }
          for (int k = 0; k < np;
               ++k)  // gathered area must reproduce facetAreasPerVertex's area[k]
            maxAreaGather = std::max(maxAreaGather, std::fabs(Ag[k] - area[k]) / cellScale);
          // finite-difference dA_k/dn_l vs the gathered analytic, over real faces
          const double eps = 1e-7;
          for (int l = 6; l < np; ++l) {
            if (area[l] < 0.05 * cellScale)
              continue;
            for (int cc = 0; cc < 3; ++cc) {
              auto areasAt = [&](double dd) {
                Cell cp = c;
                cp.n[l][cc] += dd;
                cp.nn[l] =
                    cp.n[l][0] * cp.n[l][0] + cp.n[l][1] * cp.n[l][1] + cp.n[l][2] * cp.n[l][2];
                for (int t = 0; t < cp.nt; ++t)
                  if (cp.alive[t])
                    cp.computeVertex(t);
                std::vector<double> ar(cp.np, 0.0);
                cp.facetAreasPerVertex(ar.data());
                return ar;
              };
              auto ap = areasAt(eps), am = areasAt(-eps);
              for (int k = 6; k < np; ++k) {
                if (area[k] < 0.05 * cellScale)
                  continue;
                const double fd = (ap[k] - am[k]) / (2 * eps);
                const double an = dA[((size_t)k * np + l) * 3 + cc];
                if (std::fabs(fd) > 0.02 * spacing) {  // only significant couplings (FD-resolvable)
                  ++ndAtot;
                  if (std::fabs(an - fd) / std::fabs(fd) < 2e-3)
                    ++ndApass;
                  maxdAfd = std::max(maxdAfd, std::fabs(an - fd) / std::fabs(fd));
                }
              }
            }
          }
        }

        // (12) Phase 3: Voronoi policy + chainToDofs — dV/dp_self (geomVolumeGrad's dV/dn chained
        // to the own seed) vs a finite difference of the rebuilt cell's volume. End-to-end
        // geometry→DOF check.
        if (nChainCells < 40) {
          ++nChainCells;
          double vg = 0, dgx[64], dgy[64], dgz[64];
          for (int k = 0; k < c.np; ++k)
            dgx[k] = dgy[k] = dgz[k] = 0.0;
          c.geomVolumeGrad(vg, dgx, dgy, dgz);
          double fSelf[3], fnx[64], fny[64], fnz[64];
          vor::device::chainToDofs<vor::device::Voronoi>(c, dgx, dgy, dgz, fSelf, fnx, fny, fnz);
          const double fmag =
              std::sqrt(fSelf[0] * fSelf[0] + fSelf[1] * fSelf[1] + fSelf[2] * fSelf[2]);
          const double denom = std::max(fmag, cellScale);
          const double eps = 1e-7;
          for (int cc = 0; cc < 3;
               ++cc) {  // FD of V wrt the self seed, rebuilding from the same neighbours
            std::vector<real_t> pp = pos;
            pp[3 * i + cc] += eps;
            Cell cpp;
            const bool okp = buildCell(cpp, i, pp, N, L, nbr);
            pp[3 * i + cc] -= 2 * eps;
            Cell cpm;
            const bool okm = buildCell(cpm, i, pp, N, L, nbr);
            if (!okp || !okm)
              continue;
            const double fd = (cpp.volumePerVertex() - cpm.volumePerVertex()) / (2 * eps);
            ++nChainTot;
            const double rel = std::fabs(fd - fSelf[cc]) / denom;
            if (rel < 2e-3)
              ++nChainPass;  // FD-limited; misses are topology events
            maxChain = std::max(maxChain, rel);
          }
        }

        // (4) divergence identity  V == (1/3) Σ_f |n_f| A_f   (|n_f| = sqrt(nn) = seed->plane
        // distance)
        double Vdiv = 0;
        for (int k = 0; k < c.np; ++k)
          Vdiv += std::sqrt(c.nn[k]) * area[k];
        Vdiv /= 3.0;
        maxDiv = std::max(maxDiv, std::fabs(Vdiv - Vpv) / Vref);

        // (3) per-facet areas vs ordered polygon area
        real_t fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
        for (int k = 0; k < c.np; ++k) {
          const int m = c.faceOrdered(k, fx, fy, fz);
          if (m < 3)
            continue;
          real_t A[3];
          Cell::polyAreaVec(fx, fy, fz, m, A);
          const double aref = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
          maxAreaAbs =
              std::max(maxAreaAbs, std::fabs(area[k] - aref) / cellScale);  // abs, scale-normed
          if (aref >
              0.05 * cellScale) {  // skip near-degenerate faces (relative metric blows up there)
            maxArea = std::max(maxArea, std::fabs(area[k] - aref) / aref);
            ++nfaceChecked;
          }
        }

        // (7) volume gradient / FORCE: per-vertex dV vs the oracle-validated facetGeometry.dv
        real_t aV[3], dvRef[3], cn[3];
        for (int k = 6; k < c.np; ++k) {
          if (area[k] < 0.05 * cellScale)
            continue;
          if (!c.facetGeometry(k, aV, dvRef, cn))
            continue;
          // connector r = 2·n (foot point); force dV/dr = (area/|r|)(r − centroid)
          const double rx = 2.0 * c.n[k][0], ry = 2.0 * c.n[k][1], rz = 2.0 * c.n[k][2];
          const double rl = std::sqrt(rx * rx + ry * ry + rz * rz);
          const double s = area[k] / rl;
          const double cxk = mx[k] / area[k], cyk = my[k] / area[k], czk = mz[k] / area[k];
          const double dvx = s * (rx - cxk), dvy = s * (ry - cyk), dvz = s * (rz - czk);
          const double err =
              std::sqrt((dvx - dvRef[0]) * (dvx - dvRef[0]) + (dvy - dvRef[1]) * (dvy - dvRef[1]) +
                        (dvz - dvRef[2]) * (dvz - dvRef[2]));
          const double mag =
              std::sqrt(dvRef[0] * dvRef[0] + dvRef[1] * dvRef[1] + dvRef[2] * dvRef[2]);
          if (mag > 1e-9)
            maxDVforce = std::max(maxDVforce, err / mag);
        }

        // (5) label invariance: cyclically rotate every triangle's plane triple
        Cell c2 = c;
        for (int t = 0; t < c2.nt; ++t)
          if (c2.alive[t]) {
            unsigned char a = c2.t0[t];
            c2.t0[t] = c2.t1[t];
            c2.t1[t] = c2.t2[t];
            c2.t2[t] = a;
          }
        maxLabel = std::max(maxLabel, std::fabs(c2.volumePerVertex() - Vpv) / Vref);

        // (6) gradient: dV/dh_f finite-diff vs analytic A_f  (one real face per cell). Perturb
        // plane kf along its normal by dh: scale the foot point n[kf] by (h+dh)/h, set nn =
        // (h+dh)².
        int kf = -1;
        for (int k = 6; k < c.np; ++k)
          if (area[k] > 1e-6) {
            kf = k;
            break;
          }
        if (kf >= 0) {
          const double h = std::sqrt(c.nn[kf]);  // |n_kf| = seed->plane distance
          const double eps =
              1e-8;  // small step; topology stays fixed away from combinatorial events
          auto Vat = [&](double dh) {
            Cell cc = c;
            const double sc = (h + dh) / h;
            cc.n[kf][0] *= sc;
            cc.n[kf][1] *= sc;
            cc.n[kf][2] *= sc;
            cc.nn[kf] = (h + dh) * (h + dh);
            for (int t = 0; t < cc.nt; ++t)
              if (cc.alive[t])
                cc.computeVertex(t);
            return cc.volumePerVertex();
          };
          const double dVdh = (Vat(eps) - Vat(-eps)) / (2 * eps);
          const double analytic = area[kf];  // dV/dh = A_f
          ++nGradTot;
          if (std::fabs(dVdh - analytic) / std::fabs(analytic) < 1e-4)
            ++nGradPass;  // smooth cells
        }
      }
      std::printf("batch %d (%s): cells=%ld faces=%ld\n", batch,
                  batch ? "anisotropic/obtuse" : "isotropic", nchecked, nfaceChecked);
      const double gradFrac = nGradTot ? (double)nGradPass / nGradTot : 1.0;
      std::printf("  (1) Vpv vs ordered       max rel = %.3e\n", maxV);
      std::printf(
          "  (3) areas vs ordered     max rel(non-degenerate) = %.3e   max abs/scale = %.3e\n",
          maxArea, maxAreaAbs);
      std::printf("  (4) divergence identity  max rel = %.3e\n", maxDiv);
      std::printf("  (5) label invariance     max rel = %.3e\n", maxLabel);
      std::printf("  (7) dV(force) vs oracle  max rel = %.3e\n", maxDVforce);
      std::printf("  (8) merged kernel match  max rel = %.3e\n", maxMerged);
      std::printf("  (9) geomVolumeGrad dV/dn  max rel = %.3e   (vol match = %.3e)\n", maxGrad,
                  maxGradVol);
      std::printf(
          "  (10) geomVolumeArea       areaVec rel = %.3e   dV/dn rel = %.3e   (vol match = "
          "%.3e)\n",
          maxAreaVec, maxGradA, maxAreaVol);
      const double dAfrac = ndAtot ? (double)ndApass / ndAtot : 1.0;
      std::printf(
          "  (11) geomFull dA/dn vs FD  %.4f%% of %ld couplings (max rel %.2e)   gather %.3e   "
          "analytic-vs-AD %.3e\n",
          100.0 * dAfrac, ndAtot, maxdAfd, maxAreaGather, maxAreaJacAD);
      const double chainFrac = nChainTot ? (double)nChainPass / nChainTot : 1.0;
      std::printf("  (12) Voronoi chain dV/dp_self vs FD  %.4f%% of %ld comps (max rel %.2e)\n",
                  100.0 * chainFrac, nChainTot, maxChain);
      std::printf("  (6) gradient (FD<1e-4)   %.4f%% of cells (rest are at topology events)\n",
                  100.0 * gradFrac);
      const double tol = 1e-9;
      if (maxV > tol || maxArea > tol || maxDiv > tol || maxLabel > 1e-12 || maxAreaAbs > tol ||
          maxDVforce > 1e-9 || maxMerged > 1e-9 || maxGrad > 1e-9 || maxGradVol > 1e-9 ||
          maxAreaVec > tol || maxGradA > 1e-9 || maxAreaVol > 1e-9 || maxAreaGather > tol ||
          dAfrac < 0.95 || maxAreaJacAD > 1e-9 || chainFrac < 0.95 || gradFrac < 0.98) {
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
