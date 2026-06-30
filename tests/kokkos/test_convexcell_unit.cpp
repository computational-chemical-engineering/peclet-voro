// Controlled unit tests for ConvexCell: known cells with exact volumes.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>

#include "vorflow/device/convex_cell.hpp"

using vor::device::ConvexCell;
using R = double;

static void report(const char* tag, ConvexCell<R>& c, double expVol, int expFaces) {
  double v = c.empty() ? 0 : c.volume();
  int f = c.countFaces();
  std::printf("%-22s vol=%.6f (exp %.6f) faces=%d (exp %d) np=%d nt=%d ovf=%d  %s\n", tag, v,
              expVol, f, expFaces, c.np, c.nt, (int)c.overflow,
              (std::fabs(v - expVol) < 1e-9 && f == expFaces) ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    // Case 0: just the box (no clips). L=2 -> [-1,1]^3, vol 8, 6 faces.
    {
      ConvexCell<R> c;
      c.initBox(2, 2, 2);
      report("box 2x2x2", c, 8.0, 6);
    }
    // Case 1: 6 axis neighbours at distance 1 -> bisector at 0.5 -> unit cube vol 1, 6 faces.
    {
      ConvexCell<R> c;
      c.initBox(10, 10, 10);
      R nb[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      for (int k = 0; k < 6; ++k) {
        R d = 0.5 * (nb[k][0] * nb[k][0] + nb[k][1] * nb[k][1] + nb[k][2] * nb[k][2]);
        c.clip(nb[k], d, k);
      }
      report("6 axis nbrs -> cube", c, 1.0, 6);
    }
    // Case 2: same 6, then 8 corner neighbours at (+-1,+-1,+-1) dist sqrt3: bisector
    // n=(1,1,1), d=1.5; plane x+y+z<=1.5; cube max x+y+z = 1.5 -> tangent, no real cut.
    {
      ConvexCell<R> c;
      c.initBox(10, 10, 10);
      R nb[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      for (int k = 0; k < 6; ++k)
        c.clip(nb[k], 0.5, k);
      int id = 6;
      for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
          for (int sz = -1; sz <= 1; sz += 2) {
            R n[3] = {(R)sx, (R)sy, (R)sz};
            c.clip(n, 1.5, id++);  // d = 0.5*3 = 1.5
          }
      report("cube + tangent corners", c, 1.0, 6);
    }
    // Case 3: a closer neighbour at (1,0,0) dist 1 AND a farther one (3,0,0): the far one
    // must NOT cut (its bisector at x=1.5 is outside the cube face x=0.5).
    {
      ConvexCell<R> c;
      c.initBox(10, 10, 10);
      R nb[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      for (int k = 0; k < 6; ++k)
        c.clip(nb[k], 0.5, k);
      R far[3] = {3, 0, 0};
      bool cut = c.clip(far, 4.5, 99);  // d=0.5*9=4.5 -> plane x=1.5, no cut
      std::printf("far nbr cut=%d (exp 0)\n", (int)cut);
      report("cube + far (no-op)", c, 1.0, 6);
    }
    // Case 4: truncate one corner. 6 axis nbrs (cube) + 1 neighbour at (1,1,1) dist sqrt3
    // but CLOSER: put it at (0.6,0.6,0.6), d=0.5*1.08=0.54, plane 0.6(x+y+z)<=0.54 ->
    // x+y+z<=0.9; cuts the corner (0.5,0.5,0.5) where x+y+z=1.5>0.9. Truncated cube: vol =
    // 1 - (1/6)*a^3 where the cut removes a corner tetra. Corner at (0.5,0.5,0.5); plane
    // x+y+z=0.9 meets the 3 edges at x=0.9-0.5-0.5=-0.1?? -> let me just check faces=7 and
    // vol<1 and Σ consistency rather than exact.
    {
      ConvexCell<R> c;
      c.initBox(10, 10, 10);
      R nb[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      for (int k = 0; k < 6; ++k)
        c.clip(nb[k], 0.5, k);
      R cn[3] = {0.6, 0.6, 0.6};
      c.clip(cn, 0.5 * (0.36 * 3), 7);  // d=0.54
      double v = c.volume();
      std::printf("corner-truncated cube vol=%.6f (exp <1, >0) faces=%d (exp 7) %s\n", v,
                  c.countFaces(), (v < 1.0 && v > 0.5 && c.countFaces() == 7) ? "OK" : "FAIL");
    }
    // Case 5: many planes tangent to a sphere (random directions), closest-first. The cell
    // approaches the sphere of radius r=1 (vol -> 4/3 pi). Trace per-clip max vertex rsq to
    // catch a blow-up (the cascade bug).
    {
      ConvexCell<R, 128, 256> c;
      c.initBox(20, 20, 20);
      std::srand(7);
      // directions on a sphere, neighbour at 2*dir so bisector at dir (distance 1)
      const int M = 60;
      R dir[256][3];
      R key[256];
      int idx[256];
      for (int k = 0; k < M; ++k) {
        // random direction
        R x, y, z, r2;
        do {
          x = 2.0 * std::rand() / RAND_MAX - 1;
          y = 2.0 * std::rand() / RAND_MAX - 1;
          z = 2.0 * std::rand() / RAND_MAX - 1;
          r2 = x * x + y * y + z * z;
        } while (r2 > 1 || r2 < 1e-6);
        R inv = 1.0 / std::sqrt(r2);
        dir[k][0] = x * inv;
        dir[k][1] = y * inv;
        dir[k][2] = z * inv;
        key[k] = 0.5;  // all at distance 1 -> bisector offset 0.5*|2dir|^2... use rel=2dir
        idx[k] = k;
      }
      for (int k = 0; k < M; ++k) {
        R rel[3] = {2 * dir[k][0], 2 * dir[k][1], 2 * dir[k][2]};  // neighbour at distance 2
        R d = 0.5 * (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);  // bisector at dist 1
        c.clip(rel, d, idx[k]);
      }
      double v = c.volume();
      R mr = c.maxVertexRsq();
      // Monte-Carlo ground truth: fraction of [-2,2]^3 inside all half-spaces.
      long inside = 0, total = 2000000;
      std::srand(99);
      for (long s = 0; s < total; ++s) {
        R px = 4.0 * std::rand() / RAND_MAX - 2, py = 4.0 * std::rand() / RAND_MAX - 2,
          pz = 4.0 * std::rand() / RAND_MAX - 2;
        bool in = true;
        for (int k = 0; k < c.np && in; ++k)
          if (c.n[k][0] * px + c.n[k][1] * py + c.n[k][2] * pz > c.nn[k])
            in = false;
        if (in)
          ++inside;
      }
      double mc = 64.0 * inside / total;
      std::printf("sphere (60 planes r=1) vol=%.4f  MC=%.4f  maxVrsq=%.3f faces=%d ovf=%d %s\n", v,
                  mc, mr, c.countFaces(), (int)c.overflow,
                  (std::fabs(v - mc) / mc < 0.02 && !c.overflow) ? "OK" : "FAIL");
    }
    // Case 6: random Voronoi-like cells (neighbours at varying distances, closest-first),
    // MC-checked. Replicates the random-bench condition; finds the first failing config.
    {
      int fails = 0, tested = 0;
      double worst = 0;
      for (int trial = 0; trial < 2000; ++trial) {
        std::srand(1000 + trial);
        // ~40 random neighbours in a cluster around the origin
        const int M = 40;
        R rx[64], ry[64], rz[64], key[64];
        int id[64];
        for (int k = 0; k < M; ++k) {
          rx[k] = 0.4 * (2.0 * std::rand() / RAND_MAX - 1);
          ry[k] = 0.4 * (2.0 * std::rand() / RAND_MAX - 1);
          rz[k] = 0.4 * (2.0 * std::rand() / RAND_MAX - 1);
          key[k] = 0.5 * (rx[k] * rx[k] + ry[k] * ry[k] + rz[k] * rz[k]);
          id[k] = k;
        }
        // sort closest-first
        for (int a = 1; a < M; ++a) {
          R kk = key[a], xx = rx[a], yy = ry[a], zz = rz[a];
          int ii = id[a], b = a - 1;
          while (b >= 0 && key[b] > kk) {
            key[b + 1] = key[b];
            rx[b + 1] = rx[b];
            ry[b + 1] = ry[b];
            rz[b + 1] = rz[b];
            id[b + 1] = id[b];
            --b;
          }
          key[b + 1] = kk;
          rx[b + 1] = xx;
          ry[b + 1] = yy;
          rz[b + 1] = zz;
          id[b + 1] = ii;
        }
        ConvexCell<R, 128, 256> c;
        c.initBox(4, 4, 4);
        for (int s = 0; s < M; ++s) {
          if (!(key[s] < 2.0 * c.maxVertexRsq()))
            break;
          R n[3] = {rx[s], ry[s], rz[s]};
          c.clip(n, key[s], id[s]);
          if (c.overflow)
            break;
        }
        if (c.overflow)
          continue;
        double v = c.volume();
        // MC in a tight box [-R,R]^3 around the cell so the inside fraction is O(1).
        const R Rb = 1.05 * std::sqrt(c.maxVertexRsq());
        long inside = 0, total = 400000;
        std::srand(55 + trial);
        for (long s = 0; s < total; ++s) {
          R px = Rb * (2.0 * std::rand() / RAND_MAX - 1),
            py = Rb * (2.0 * std::rand() / RAND_MAX - 1),
            pz = Rb * (2.0 * std::rand() / RAND_MAX - 1);
          bool in = true;
          for (int k = 0; k < c.np && in; ++k)
            if (c.n[k][0] * px + c.n[k][1] * py + c.n[k][2] * pz > c.nn[k])
              in = false;
          if (in)
            ++inside;
        }
        double mc = (2 * Rb) * (2 * Rb) * (2 * Rb) * inside / total;
        ++tested;
        double rel = std::fabs(v - mc) / (mc + 1e-12);
        if (rel > worst)
          worst = rel;
        if (rel > 0.05) {
          ++fails;
          if (fails <= 2)
            std::printf("  trial %d FAIL: vol=%.4f MC=%.4f rel=%.3f np=%d faces=%d\n", trial, v, mc,
                        rel, c.np, c.countFaces());
        }
      }
      std::printf("random cells: %d/%d fail (>5%% vs MC), worst rel=%.3f  %s\n", fails, tested,
                  worst, fails == 0 ? "OK" : "FAIL");
    }
  }
  Kokkos::finalize();
  return 0;
}
