/**
 * @file fv_test_util.hpp
 * @brief Shared helpers of the track-C solver tests: host<->device vectors, the face mesh of a seed
 * set, jittered lattices, Lloyd relaxation (B1), the Taylor–Green vortex and its face fluxes.
 */
#ifndef PECLET_VORO_TESTS_FV_TEST_UTIL_HPP
#define PECLET_VORO_TESTS_FV_TEST_UTIL_HPP

#include <cmath>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/energy/lloyd.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/fv/operators.hpp"
#include "peclet/voro/tessellator.hpp"

using Real = double;
using Mem = peclet::core::MemSpace;
using DV = Kokkos::View<Real*, Mem>;
namespace fv = peclet::voro::fv;

template <class T>
static std::vector<T> down(const Kokkos::View<T*, Mem>& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  return std::vector<T>(h.data(), h.data() + h.extent(0));
}
static DV up(const std::vector<Real>& h, const char* name) {
  DV d(name, h.size());
  Kokkos::deep_copy(d, Kokkos::View<const Real*, Kokkos::HostSpace>(h.data(), h.size()));
  return d;
}
static fv::FaceMesh<Real> meshOf(const std::vector<Real>& pos, int N, const Real L[3]) {
  DV dpos = up(pos, "pos"), dw;
  Kokkos::View<long*, Mem> gd;
  auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(dpos, dw, N, L, 4, N,
                                                                               gd, {}, true);
  auto aux = peclet::voro::buildAuxMaps(res.view);
  return fv::buildFaceMesh(res.view, aux);
}
static std::vector<Real> lattice(int n, Real jitter, std::mt19937& rng) {
  std::vector<Real> pos(3 * n * n * n);
  std::uniform_real_distribution<Real> J(-jitter, jitter);
  const Real h = Real(1) / n;
  int i = 0;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x, ++i) {
        pos[3 * i] = (x + 0.5 + J(rng)) * h;
        pos[3 * i + 1] = (y + 0.5 + J(rng)) * h;
        pos[3 * i + 2] = (z + 0.5 + J(rng)) * h;
      }
  return pos;
}
// Lloyd relaxation x <- centroid (the B1 descent), `steps` sweeps.
static void lloydRelax(std::vector<Real>& pos, int N, const Real L[3], int steps) {
  for (int it = 0; it < steps; ++it) {
    DV dpos = up(pos, "pos"), dw, cent("cent", 3 * N);
    Kokkos::View<long*, Mem> gd;
    auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(
        dpos, dw, N, L, 4, N, gd, {}, true, -1, {}, {}, {}, {}, {}, {}, 0, nullptr, {}, false, {},
        {}, 0, Real(0), false, /*withMoments=*/true);
    peclet::voro::energy::cellCentroids<Real>(res.view, dpos, cent);
    pos = down(cent);
    for (int i = 0; i < 3 * N; ++i)
      pos[i] -= L[i % 3] * std::floor(pos[i] / L[i % 3]);
  }
}
// Taylor–Green (2D vortex, z-independent): exact 3D NS solution, decay exp(−2 ν k² t), k = 2π.
static void tgv(Real x, Real y, Real t, Real nu, Real& u, Real& v) {
  const Real k = 2 * M_PI, e = std::exp(-2 * nu * k * k * t);
  u = std::sin(k * x) * std::cos(k * y) * e;
  v = -std::cos(k * x) * std::sin(k * y) * e;
}
// exact face-normal flux at the face centroids (absolute position x_A + c_f)
static std::vector<Real> tgvFlux(const fv::FaceMesh<Real>& m, const std::vector<Real>& pos, Real t,
                                 Real nu) {
  auto A = down(m.faceCellA);
  auto C = down(m.faceCentroid), Nn = down(m.faceNormal);
  std::vector<Real> uf(m.nFaces);
  for (int f = 0; f < m.nFaces; ++f) {
    const int a = A[f];
    const Real x = pos[3 * a] + C[3 * f], y = pos[3 * a + 1] + C[3 * f + 1];
    Real u, v;
    tgv(x, y, t, nu, u, v);
    uf[f] = u * Nn[3 * f] + v * Nn[3 * f + 1];
  }
  return uf;
}
static Real relErrF(const fv::FaceMesh<Real>& m, const DV& u, const std::vector<Real>& ex) {
  auto uh = down(u), A = down(m.faceArea), d = down(m.faceDist);
  double e = 0, n = 0;
  for (int f = 0; f < m.nInterior; ++f) {
    e += A[f] * d[f] * (uh[f] - ex[f]) * (uh[f] - ex[f]);
    n += A[f] * d[f] * ex[f] * ex[f];
  }
  return std::sqrt(e / n);
}
// exact cell velocity at the seeds (3N)
static std::vector<Real> tgvCell(const std::vector<Real>& pos, int N, Real t, Real nu) {
  std::vector<Real> U(3 * N, Real(0));
  for (int i = 0; i < N; ++i)
    tgv(pos[3 * i], pos[3 * i + 1], t, nu, U[3 * i], U[3 * i + 1]);
  return U;
}
static Real relErrV(const fv::FaceMesh<Real>& m, const DV& U, const std::vector<Real>& ex) {
  auto Uh = down(U), V = down(m.cellVolume);
  double e = 0, n = 0;
  for (int i = 0; i < m.nCells; ++i)
    for (int c = 0; c < 3; ++c) {
      e += V[i] * (Uh[3 * i + c] - ex[3 * i + c]) * (Uh[3 * i + c] - ex[3 * i + c]);
      n += V[i] * ex[3 * i + c] * ex[3 * i + c];
    }
  return std::sqrt(e / n);
}
// area-weighted face skewness |c_f - h_A n_f| / d_f
static Real faceSkewness(const fv::FaceMesh<Real>& m) {
  auto C = down(m.faceCentroid), Nn = down(m.faceNormal), ha = down(m.faceHa), d = down(m.faceDist),
       Af = down(m.faceArea);
  double sk = 0, at = 0;
  for (int f = 0; f < m.nInterior; ++f) {
    double t2 = 0;
    for (int c = 0; c < 3; ++c) {
      const double t = C[3 * f + c] - ha[f] * Nn[3 * f + c];
      t2 += t * t;
    }
    sk += Af[f] * std::sqrt(t2) / d[f];
    at += Af[f];
  }
  return sk / at;
}

#endif  // PECLET_VORO_TESTS_FV_TEST_UTIL_HPP
