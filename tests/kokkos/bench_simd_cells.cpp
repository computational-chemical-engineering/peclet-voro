// bench_simd_cells.cpp — SIMD "cells-as-lanes" prototype for the per-vertex geometry kernel.
//
// Measures the per-core throughput ceiling of vectorising the ConvexCell per-vertex VOLUME kernel
// (volumePerVertex) across cells (SoA, one cell per SIMD lane) vs the current scalar one-cell-at-a-time
// form. Faithfully reproduces the per-triangle arithmetic from
// include/vorflow/device/convex_cell.hpp::volumePerVertex (xprod/dot3/det3/edgeFoot + the D<0 canonical
// swap, here a branchless blend). Plane normals are PRE-RESOLVED per triangle (n1,n2,n3) — i.e. the
// per-cell `n[t0[t]]` index indirection is hoisted out, which is exactly the SoA layout cells-as-lanes
// needs (no per-lane gather). Standalone: no Kokkos (the bootstrapped Kokkos was built without
// KOKKOS_ARCH_AVX2, so its simd falls back to scalar); compile with -march=native for real AVX2.
//
//   g++ -O3 -march=native -fopenmp -std=c++20 bench_simd_cells.cpp -o bench_simd_cells
//   ./bench_simd_cells [N cells] [nt triangles/cell] [reps]
//
// Reports scalar vs cells-as-lanes cells/s (single thread) and the SIMD speedup, for FP64 and FP32.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// ---- scalar reference: one cell at a time, exactly the volumePerVertex inner body (AoS) ----------
template <typename Real>
static inline Real cell_volume_scalar(const Real* n1, const Real* n2c, const Real* n3c, const Real* vv,
                                      int nt) {
  Real vol = 0;
  for (int t = 0; t < nt; ++t) {
    Real a1[3] = {n1[3 * t], n1[3 * t + 1], n1[3 * t + 2]};
    Real a2[3] = {n2c[3 * t], n2c[3 * t + 1], n2c[3 * t + 2]};
    Real a3[3] = {n3c[3 * t], n3c[3 * t + 1], n3c[3 * t + 2]};
    auto cross = [](const Real a[3], const Real b[3], Real o[3]) {
      o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
    };
    auto dot = [](const Real a[3], const Real b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    Real c1[3], c2[3], c3[3];
    cross(a2, a3, c1); cross(a3, a1, c2); cross(a1, a2, c3);
    Real D = dot(a1, c1);
    if (D < Real(0)) {
      for (int a = 0; a < 3; ++a) { Real t1 = a2[a]; a2[a] = a3[a]; a3[a] = t1; t1 = c2[a]; c2[a] = c3[a]; c3[a] = t1; }
    }
    const Real v[3] = {vv[3 * t], vv[3 * t + 1], vv[3 * t + 2]};
    auto edgeFoot = [&](const Real ck[3], Real f[3]) {
      Real cc = dot(ck, ck);
#ifdef NODIV
      Real s = dot(v, ck);  // diagnostic: skip the reciprocal to isolate divide cost
#else
      Real s = (cc > Real(0)) ? dot(v, ck) / cc : Real(0);
#endif
      f[0] = v[0] - s * ck[0]; f[1] = v[1] - s * ck[1]; f[2] = v[2] - s * ck[2];
    };
    auto det3 = [&](const Real a[3], const Real b[3], const Real c[3]) {
      Real bc[3]; cross(b, c, bc); return dot(a, bc);
    };
#ifdef V1DIV
    // 1-divide identity (the convex_cell.hpp fix): det3(e,edgeFoot(v,ck),v)=(v·ck)(e·(v×ck))/|ck|²,
    // three terms folded over the common denominator g1·g2·g3.
    Real vc1[3], vc2[3], vc3[3];
    cross(v, c1, vc1); cross(v, c2, vc2); cross(v, c3, vc3);
    Real e[3];
    e[0]=a1[0]-a2[0]; e[1]=a1[1]-a2[1]; e[2]=a1[2]-a2[2]; Real numc3 = dot(v,c3)*dot(e,vc3);
    e[0]=a2[0]-a3[0]; e[1]=a2[1]-a3[1]; e[2]=a2[2]-a3[2]; Real numc1 = dot(v,c1)*dot(e,vc1);
    e[0]=a3[0]-a1[0]; e[1]=a3[1]-a1[1]; e[2]=a3[2]-a1[2]; Real numc2 = dot(v,c2)*dot(e,vc2);
    Real g1=dot(c1,c1), g2=dot(c2,c2), g3=dot(c3,c3), den=g1*g2*g3;
    vol += (den > Real(0)) ? (numc3*g1*g2 + numc1*g2*g3 + numc2*g1*g3) / den : Real(0);
#else
    Real f12[3], f23[3], f31[3];
    edgeFoot(c3, f12); edgeFoot(c1, f23); edgeFoot(c2, f31);
    Real e[3], d = 0;
    e[0] = a1[0] - a2[0]; e[1] = a1[1] - a2[1]; e[2] = a1[2] - a2[2]; d += det3(e, f12, v);
    e[0] = a2[0] - a3[0]; e[1] = a2[1] - a3[1]; e[2] = a2[2] - a3[2]; d += det3(e, f23, v);
    e[0] = a3[0] - a1[0]; e[1] = a3[1] - a1[1]; e[2] = a3[2] - a1[2]; d += det3(e, f31, v);
    vol += d;
#endif
  }
  return vol * (Real(1) / Real(6));
}

// ---- cells-as-lanes: SoA arrays [t][component][cell], inner loop over cells = SIMD lanes -----------
// Layout: component arrays of length nt*N, indexed (t*N + i). The innermost (contiguous) index is the
// cell i, so `#pragma omp simd` over i vectorises across cells. The D<0 swap becomes a branchless blend.
template <typename Real>
struct SoA {
  int N, nt;
  std::vector<Real> n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, vx, vy, vz, vol;
  SoA(int N_, int nt_) : N(N_), nt(nt_),
    n1x(N_*nt_), n1y(N_*nt_), n1z(N_*nt_), n2x(N_*nt_), n2y(N_*nt_), n2z(N_*nt_),
    n3x(N_*nt_), n3y(N_*nt_), n3z(N_*nt_), vx(N_*nt_), vy(N_*nt_), vz(N_*nt_), vol(N_) {}
};

template <typename Real>
static void cell_volume_simd(const SoA<Real>& s) {
  const int N = s.N, nt = s.nt;
  Real* __restrict vol = const_cast<Real*>(s.vol.data());
  const Real* __restrict n1x = s.n1x.data(); const Real* __restrict n1y = s.n1y.data(); const Real* __restrict n1z = s.n1z.data();
  const Real* __restrict n2x = s.n2x.data(); const Real* __restrict n2y = s.n2y.data(); const Real* __restrict n2z = s.n2z.data();
  const Real* __restrict n3x = s.n3x.data(); const Real* __restrict n3y = s.n3y.data(); const Real* __restrict n3z = s.n3z.data();
  const Real* __restrict vxa = s.vx.data(); const Real* __restrict vya = s.vy.data(); const Real* __restrict vza = s.vz.data();
#pragma omp parallel
  {
    int nth = omp_get_num_threads(), tid = omp_get_thread_num();
    int lo = (int)((long)N * tid / nth), hi = (int)((long)N * (tid + 1) / nth);
#pragma omp simd
    for (int i = lo; i < hi; ++i) vol[i] = 0;
    for (int t = 0; t < nt; ++t) {
      const int b = t * N;
#pragma omp simd
      for (int i = lo; i < hi; ++i) {
      Real a1x = n1x[b+i], a1y = n1y[b+i], a1z = n1z[b+i];
      Real a2x = n2x[b+i], a2y = n2y[b+i], a2z = n2z[b+i];
      Real a3x = n3x[b+i], a3y = n3y[b+i], a3z = n3z[b+i];
      // cross products c1=a2xa3, c2=a3xa1, c3=a1xa2
      Real c1x = a2y*a3z - a2z*a3y, c1y = a2z*a3x - a2x*a3z, c1z = a2x*a3y - a2y*a3x;
      Real c2x = a3y*a1z - a3z*a1y, c2y = a3z*a1x - a3x*a1z, c2z = a3x*a1y - a3y*a1x;
      Real c3x = a1y*a2z - a1z*a2y, c3y = a1z*a2x - a1x*a2z, c3z = a1x*a2y - a1y*a2x;
      Real D = a1x*c1x + a1y*c1y + a1z*c1z;
      // canonical D>0: blend-swap (a2,a3) and (c2,c3) when D<0
      Real m = (D < Real(0)) ? Real(1) : Real(0);
      Real t2x = a2x + m*(a3x-a2x), t3x = a3x + m*(a2x-a3x); a2x=t2x; a3x=t3x;
      Real t2y = a2y + m*(a3y-a2y), t3y = a3y + m*(a2y-a3y); a2y=t2y; a3y=t3y;
      Real t2z = a2z + m*(a3z-a2z), t3z = a3z + m*(a2z-a3z); a2z=t2z; a3z=t3z;
      Real u2x = c2x + m*(c3x-c2x), u3x = c3x + m*(c2x-c3x); c2x=u2x; c3x=u3x;
      Real u2y = c2y + m*(c3y-c2y), u3y = c3y + m*(c2y-c3y); c2y=u2y; c3y=u3y;
      Real u2z = c2z + m*(c3z-c2z), u3z = c3z + m*(c2z-c3z); c2z=u2z; c3z=u3z;
      Real vX = vxa[b+i], vY = vya[b+i], vZ = vza[b+i];
#ifdef V1DIV
      // 1-divide identity (the convex_cell.hpp fix): (v·ck)(e·(v×ck))/|ck|², folded over g1·g2·g3.
      Real vc1x=vY*c1z-vZ*c1y, vc1y=vZ*c1x-vX*c1z, vc1z=vX*c1y-vY*c1x;
      Real vc2x=vY*c2z-vZ*c2y, vc2y=vZ*c2x-vX*c2z, vc2z=vX*c2y-vY*c2x;
      Real vc3x=vY*c3z-vZ*c3y, vc3y=vZ*c3x-vX*c3z, vc3z=vX*c3y-vY*c3x;
      Real numc3 = (vX*c3x+vY*c3y+vZ*c3z) * ((a1x-a2x)*vc3x+(a1y-a2y)*vc3y+(a1z-a2z)*vc3z);
      Real numc1 = (vX*c1x+vY*c1y+vZ*c1z) * ((a2x-a3x)*vc1x+(a2y-a3y)*vc1y+(a2z-a3z)*vc1z);
      Real numc2 = (vX*c2x+vY*c2y+vZ*c2z) * ((a3x-a1x)*vc2x+(a3y-a1y)*vc2y+(a3z-a1z)*vc2z);
      Real g1=c1x*c1x+c1y*c1y+c1z*c1z, g2=c2x*c2x+c2y*c2y+c2z*c2z, g3=c3x*c3x+c3y*c3y+c3z*c3z, den=g1*g2*g3;
      Real d = (den > Real(0)) ? (numc3*g1*g2 + numc1*g2*g3 + numc2*g1*g3) / den : Real(0);
#else
      // edgeFoot(v, ck) = v - (v.ck/ck.ck) ck ; guard ck.ck>0
      auto foot = [&](Real kx, Real ky, Real kz, Real& fx, Real& fy, Real& fz) {
        Real cc = kx*kx + ky*ky + kz*kz;
#ifdef NODIV
        Real s = (vX*kx + vY*ky + vZ*kz);  // diagnostic: skip the reciprocal
#else
        Real s = (cc > Real(0)) ? (vX*kx + vY*ky + vZ*kz) / cc : Real(0);
#endif
        fx = vX - s*kx; fy = vY - s*ky; fz = vZ - s*kz;
      };
      Real f12x,f12y,f12z, f23x,f23y,f23z, f31x,f31y,f31z;
      foot(c3x,c3y,c3z, f12x,f12y,f12z);
      foot(c1x,c1y,c1z, f23x,f23y,f23z);
      foot(c2x,c2y,c2z, f31x,f31y,f31z);
      // det3(e, f, v) = e . (f x v)
      auto det = [&](Real ex,Real ey,Real ez, Real fx,Real fy,Real fz) {
        Real bx = fy*vZ - fz*vY, by = fz*vX - fx*vZ, bz = fx*vY - fy*vX;
        return ex*bx + ey*by + ez*bz;
      };
      Real d = det(a1x-a2x,a1y-a2y,a1z-a2z, f12x,f12y,f12z)
             + det(a2x-a3x,a2y-a3y,a2z-a3z, f23x,f23y,f23z)
             + det(a3x-a1x,a3y-a1y,a3z-a1z, f31x,f31y,f31z);
#endif
      vol[i] += d;
      }
    }
#pragma omp simd
    for (int i = lo; i < hi; ++i) vol[i] *= Real(1) / Real(6);
  }  // omp parallel
}

template <typename Real>
static void run(const char* tag, int N, int nt, int reps) {
  // synthesize plausible non-degenerate triangles (arithmetic intensity is what matters)
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  SoA<Real> s(N, nt);
  std::vector<Real> aos(N * nt * 12);  // AoS for the scalar path: per cell, [nt][12]
  for (int i = 0; i < N; ++i)
    for (int t = 0; t < nt; ++t) {
      Real vals[12];
      for (int k = 0; k < 12; ++k) vals[k] = (Real)(0.3 + 0.7 * std::fabs(U(rng))) * (Real)((k % 3) ? 1 : 1);
      // ensure non-tiny normals
      s.n1x[t*N+i]=vals[0]; s.n1y[t*N+i]=vals[1]; s.n1z[t*N+i]=vals[2];
      s.n2x[t*N+i]=vals[3]; s.n2y[t*N+i]=vals[4]; s.n2z[t*N+i]=vals[5];
      s.n3x[t*N+i]=vals[6]; s.n3y[t*N+i]=vals[7]; s.n3z[t*N+i]=vals[8];
      s.vx[t*N+i]=vals[9]; s.vy[t*N+i]=vals[10]; s.vz[t*N+i]=vals[11];
      for (int k = 0; k < 12; ++k) aos[(i*nt + t)*12 + k] = vals[k];
    }
  // scalar (AoS, one cell at a time)
  volatile double sink = 0;
  cell_volume_simd(s);  // warm
  double tsc = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = clk::now();
    double acc = 0;
#pragma omp parallel for schedule(static) reduction(+:acc)
    for (int i = 0; i < N; ++i) {
      const Real* base = &aos[(i*nt)*12];
      // de-interleave into n1/n2/n3/v contiguous-per-triangle for the scalar body
      // (cheap; mirrors reading the cell's resident arrays)
      Real n1[3*64], n2[3*64], n3[3*64], vv[3*64];  // nt<=64
      for (int t = 0; t < nt; ++t) {
        const Real* p = &base[t*12];
        n1[3*t]=p[0]; n1[3*t+1]=p[1]; n1[3*t+2]=p[2];
        n2[3*t]=p[3]; n2[3*t+1]=p[4]; n2[3*t+2]=p[5];
        n3[3*t]=p[6]; n3[3*t+1]=p[7]; n3[3*t+2]=p[8];
        vv[3*t]=p[9]; vv[3*t+1]=p[10]; vv[3*t+2]=p[11];
      }
      acc += (double)cell_volume_scalar<Real>(n1, n2, n3, vv, nt);
    }
    auto t1 = clk::now();
    tsc = std::min(tsc, secs(t0, t1)); sink += acc;
  }
  // simd (cells-as-lanes)
  double tsi = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = clk::now();
    cell_volume_simd(s);
    auto t1 = clk::now();
    tsi = std::min(tsi, secs(t0, t1));
    double acc = 0; for (int i = 0; i < N; ++i) acc += (double)s.vol[i]; sink += acc;
  }
  printf("%-6s  scalar %8.1f Mcell/s   cells-as-lanes %8.1f Mcell/s   speedup %.2fx   (sink %.3e)\n",
         tag, N / tsc / 1e6, N / tsi / 1e6, tsc / tsi, (double)sink);
}

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 8192;
  int nt = argc > 2 ? atoi(argv[2]) : 24;
  int reps = argc > 3 ? atoi(argv[3]) : 300;
  if (nt > 64) nt = 64;
  printf("per-vertex VOLUME kernel, N=%d cells, nt=%d tri/cell, reps=%d, threads=%d%s\n", N, nt, reps,
         omp_get_max_threads(),
#if defined(NODIV)
         "  [NODIV diagnostic]");
#elif defined(V1DIV)
         "  [V1DIV: 1-divide convex_cell.hpp fix]");
#else
         "  [3-divide edgeFoot baseline]");
#endif
  run<double>("FP64", N, nt, reps);
  run<float>("FP32", N, nt, reps);
  return 0;
}
