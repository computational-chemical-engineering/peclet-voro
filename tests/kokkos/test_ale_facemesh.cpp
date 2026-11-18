/**
 * @file test_ale_facemesh.cpp
 * \brief Gate G0.1 of the moving-grid design note (docs/ale_design.md §7 WO-S0.1, §11 A1/A2): the
 * opt-in `FaceMeshOptions` of `fv::buildFaceMesh`.
 *
 * On an n³ periodic power mesh (seeds jittered ±0.3h, weights uniform in [0, 0.15h²], fixed seed)
 * and the n³ Voronoi mesh of the same seeds, both published by `reevalPublish`:
 *  (b) all four options on (gid = a fixed permutation of the slots): per cell |Σ A_f n_f| ≤ 1e-13
 *      A_cell and ‖Σ A_f n_f ⊗ (c_f − x_i) − V_i I‖_F ≤ 1e-13 V_i; per face |h_A + h_B − d| ≤
 *      2 ulp(d), the centroid against a polygon oracle (area-weighted triangle fan over
 *      `ConvexCell::faceOrdered`) ≤ 1e-13 h, the mesh normal's |n × conn|/d ≤ 4 ulp, the published
 *      polygon normal's |n × conn|/d ≤ 1e-12 on A_f > 1e-2 h² (other area bins: INFO); the cell
 *      centroid against the oracle's pyramids ≤ 1e-13 h. Also: the gid-oriented mesh WITHOUT the
 *      deterministic order (A ≠ emitter on about half the faces) equals the deterministic one face
 *      by face to 1e-12 (relative to h).
 *  (c′) two builds from the same published view with all options on are bitwise equal.
 *  (d) `deterministicOrder` alone equals the default builder bitwise in every array (CSR order
 *      bitwise at one thread, per-cell sets otherwise: the default order is schedule-dependent).
 * `--dump DIR` writes every FaceMesh array of the deterministic configurations as raw bytes for
 * gate (c) (tools/ale_gates/G0_1.sh compares them across thread counts and runs). Usage:
 *   test_ale_facemesh [n=16] [--dump DIR] [--time]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/transpose.hpp"

using real_t = double;
using Mem = peclet::core::MemSpace;
using Exec = peclet::core::ExecSpace;
constexpr int kP = peclet::voro::kMaxPlanes;
constexpr int kT = peclet::voro::kMaxTriangles;

namespace {

template <class V>
auto toHost(const V& v) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
}

struct Scene {
  int n = 0, N = 0;
  real_t L[3] = {1, 1, 1};
  real_t h = 0;
  Kokkos::View<real_t*, Mem> pos, weight;
  Kokkos::View<std::int64_t*, Mem> gidPerm;
};

Scene makeScene(int n) {
  Scene s;
  s.n = n;
  s.N = n * n * n;
  s.h = real_t(1) / n;
  std::mt19937 rng(20261114u);
  std::uniform_real_distribution<real_t> U(-1, 1), U01(0, 1);
  std::vector<real_t> x(3 * s.N), w(s.N);
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const int c = i + n * (j + n * k);
        const int ijk[3] = {i, j, k};
        for (int d = 0; d < 3; ++d) {
          real_t p = (ijk[d] + real_t(0.5)) * s.h + real_t(0.3) * s.h * U(rng);
          p = p < 0 ? p + 1 : (p >= 1 ? p - 1 : p);
          x[3 * c + d] = p;
        }
      }
  for (auto& v : w)
    v = real_t(0.15) * s.h * s.h * U01(rng);
  std::vector<std::int64_t> perm(s.N);
  std::iota(perm.begin(), perm.end(), std::int64_t(0));
  std::shuffle(perm.begin(), perm.end(), rng);
  s.pos = Kokkos::View<real_t*, Mem>("pos", 3 * s.N);
  s.weight = Kokkos::View<real_t*, Mem>("w", s.N);
  s.gidPerm = Kokkos::View<std::int64_t*, Mem>("gid", s.N);
  Kokkos::deep_copy(s.pos, Kokkos::View<const real_t*, Kokkos::HostSpace>(x.data(), 3 * s.N));
  Kokkos::deep_copy(s.weight, Kokkos::View<const real_t*, Kokkos::HostSpace>(w.data(), s.N));
  Kokkos::deep_copy(s.gidPerm,
                    Kokkos::View<const std::int64_t*, Kokkos::HostSpace>(perm.data(), s.N));
  return s;
}

// The oracle: per published facet the polygon (faceOrdered) area vector and area-weighted
// triangle-fan centroid, and per cell the pyramid volume and first moment, all seed-relative.
struct Oracle {
  Kokkos::View<real_t*, Mem> facetCen;  // 3*nF
  Kokkos::View<real_t*, Mem> cellCen;   // 3*N
};

template <bool Weighted>
struct Built {
  using Policy = std::conditional_t<Weighted, peclet::voro::Power, peclet::voro::Voronoi>;
  peclet::voro::MovingTessellation<real_t, kP, kT, Weighted> mt;
  peclet::voro::TessellationView<real_t> view;
  peclet::voro::AuxMaps<real_t> aux;
  Oracle oracle;
};

template <bool Weighted>
void build(const Scene& s, Built<Weighted>& b) {
  using Policy = typename Built<Weighted>::Policy;
  const real_t tol = real_t(1e-4) * s.h, skin = real_t(0.25) * s.h;
  b.mt.alloc(s.N, s.L, tol, skin, 4, s.N);
  Kokkos::View<real_t*, Mem> w;
  if constexpr (Weighted) {
    b.mt.setWeights(s.weight);
    w = s.weight;
  }
  b.mt.rebuild(s.pos);
  b.view = peclet::voro::reevalPublish<real_t, kP, kT, Policy>(
      b.mt.store, s.pos, b.mt.vol, s.N, s.L, b.mt.wall, b.mt.xRef, false, false, w);
  b.aux = peclet::voro::buildAuxMaps(b.view);

  // oracle kernel: re-load each cell exactly as reevalPublish does and walk its faces
  using Cell = peclet::voro::ConvexCell<real_t, kP, kT, false>;
  const int N = s.N;
  const int nF = b.view.numFacets();
  b.oracle.facetCen = Kokkos::View<real_t*, Mem>("oracle.fc", 3 * (size_t)nF);
  b.oracle.cellCen = Kokkos::View<real_t*, Mem>("oracle.cc", 3 * (size_t)N);
  auto FC = b.oracle.facetCen;
  auto CC = b.oracle.cellCen;
  auto st = b.mt.store;
  auto P = s.pos;
  auto W = w;
  auto view = b.view;
  const real_t Lx = s.L[0], Ly = s.L[1], Lz = s.L[2];
  Kokkos::parallel_for(
      "oracle", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        Cell c;
        st.load(i, c, Lx, Ly, Lz);
        // capture P and W outside the constexpr-if (nvcc extended-lambda rule)
        const real_t* wPtr = W.data();
        const real_t* pPtr = P.data();
        const real_t sx = pPtr[3 * i], sy = pPtr[3 * i + 1], sz = pPtr[3 * i + 2];
        (void)wPtr;
        if constexpr (Policy::kHasWeightDof)
          c.template reevalGeometry<Policy>(sx, sy, sz, pPtr, Lx, wPtr[i], wPtr);
        else
          c.reevalGeometry(sx, sy, sz, pPtr, Lx);
        int idx = 0;
        const int b0 = view.facetBegin(i);
        real_t vol = 0, mx = 0, my = 0, mz = 0;
        for (int k = 0; k < c.np; ++k) {
          int cnt = 0;
          for (int t = 0; t < c.nt; ++t)
            if (c.alive[t] && (c.t0[t] == k || c.t1[t] == k || c.t2[t] == k))
              ++cnt;
          if (cnt < 3)
            continue;
          const int f = b0 + idx;
          ++idx;
          real_t fx[Cell::MAXFV], fy[Cell::MAXFV], fz[Cell::MAXFV];
          const int m = c.faceOrdered(k, fx, fy, fz);
          real_t A[3] = {0, 0, 0};
          if (m >= 3)
            Cell::polyAreaVec(fx, fy, fz, m, A);
          // outward = the published area vector's side
          if (A[0] * view.area(f, 0) + A[1] * view.area(f, 1) + A[2] * view.area(f, 2) < 0) {
            A[0] = -A[0];
            A[1] = -A[1];
            A[2] = -A[2];
          }
          const real_t a = Kokkos::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
          real_t cx = 0, cy = 0, cz = 0, asum = 0;
          if (a > 0) {
            const real_t nx = A[0] / a, ny = A[1] / a, nz = A[2] / a;
            for (int q = 1; q + 1 < m; ++q) {  // signed triangle areas along the face normal
              const real_t ux = fx[q] - fx[0], uy = fy[q] - fy[0], uz = fz[q] - fz[0];
              const real_t vx = fx[q + 1] - fx[0], vy = fy[q + 1] - fy[0], vz = fz[q + 1] - fz[0];
              const real_t at = real_t(0.5) * ((uy * vz - uz * vy) * nx + (uz * vx - ux * vz) * ny +
                                               (ux * vy - uy * vx) * nz);
              const real_t sgn = at < 0 ? real_t(-1) : real_t(1);  // CW or CCW fan
              cx += sgn * at * (fx[0] + fx[q] + fx[q + 1]);
              cy += sgn * at * (fy[0] + fy[q] + fy[q + 1]);
              cz += sgn * at * (fz[0] + fz[q] + fz[q + 1]);
              asum += sgn * at;
            }
            cx /= 3 * asum;
            cy /= 3 * asum;
            cz /= 3 * asum;
            const real_t hf = nx * cx + ny * cy + nz * cz;  // signed pyramid height
            vol += hf * a / 3;
            mx += real_t(0.25) * hf * a * cx;
            my += real_t(0.25) * hf * a * cy;
            mz += real_t(0.25) * hf * a * cz;
          }
          FC(3 * f) = cx;
          FC(3 * f + 1) = cy;
          FC(3 * f + 2) = cz;
        }
        CC(3 * i) = vol != 0 ? mx / vol : 0;
        CC(3 * i + 1) = vol != 0 ? my / vol : 0;
        CC(3 * i + 2) = vol != 0 ? mz / vol : 0;
      });
  Kokkos::fence();
}

using FM = peclet::voro::fv::FaceMesh<real_t>;
using FMO = peclet::voro::fv::FaceMeshOptions<real_t>;

template <class V>
bool sameBits(const V& a, const V& b) {
  if (a.extent(0) != b.extent(0))
    return false;
  auto ha = toHost(a);
  auto hb = toHost(b);
  return a.extent(0) == 0 ||
         std::memcmp(ha.data(), hb.data(), a.extent(0) * sizeof(typename V::value_type)) == 0;
}

// (d): the deterministic builder against the default one, every array.
int checkDefaultEquivalence(const char* tag, const FM& m0, const FM& m1, bool serial) {
  int bad = 0;
  std::string first;
  auto chk = [&](const char* name, bool ok) {
    if (!ok && first.empty())
      first = name;
    bad += ok ? 0 : 1;
  };
  chk("nFaces", m0.nFaces == m1.nFaces && m0.nInterior == m1.nInterior);
  chk("nDropped", m0.nDropped == m1.nDropped);
  chk("faceCellA", sameBits(m0.faceCellA, m1.faceCellA));
  chk("faceCellB", sameBits(m0.faceCellB, m1.faceCellB));
  chk("faceFacet", sameBits(m0.faceFacet, m1.faceFacet));
  chk("faceArea", sameBits(m0.faceArea, m1.faceArea));
  chk("faceNormal", sameBits(m0.faceNormal, m1.faceNormal));
  chk("faceDist", sameBits(m0.faceDist, m1.faceDist));
  chk("faceHa", sameBits(m0.faceHa, m1.faceHa));
  chk("faceHb", sameBits(m0.faceHb, m1.faceHb));
  chk("faceCentroid", sameBits(m0.faceCentroid, m1.faceCentroid));
  chk("faceConn", sameBits(m0.faceConn, m1.faceConn));
  chk("cellFaceOffset", sameBits(m0.cellFaceOffset, m1.cellFaceOffset));
  if (serial) {
    chk("cellFace", sameBits(m0.cellFace, m1.cellFace));
    chk("cellFaceSign", sameBits(m0.cellFaceSign, m1.cellFaceSign));
  } else if (m0.cellFace.extent(0) == m1.cellFace.extent(0)) {  // per-cell sets
    auto off = toHost(m0.cellFaceOffset);
    auto c0 = toHost(m0.cellFace);
    auto c1 = toHost(m1.cellFace);
    auto s0 = toHost(m0.cellFaceSign);
    auto s1 = toHost(m1.cellFaceSign);
    bool ok = true;
    for (int i = 0; i < m0.nCells && ok; ++i) {
      std::vector<std::pair<int, real_t>> e0, e1;
      for (int q = off(i); q < off(i + 1); ++q) {
        e0.emplace_back(c0(q), -s0(q));
        e1.emplace_back(c1(q), -s1(q));
      }
      std::sort(e0.begin(), e0.end());
      std::sort(e1.begin(), e1.end());
      ok = e0 == e1;
    }
    chk("cellFace(set)", ok);
  } else {
    chk("cellFace(size)", false);
  }
  std::printf("RESULT d %-7s %s arrays_differing=%d%s%s (%s CSR)\n", tag, bad ? "FAIL" : "PASS",
              bad, bad ? " first=" : "", first.c_str(), serial ? "bitwise" : "per-cell set");
  return bad ? 1 : 0;
}

// (b): identities and oracles on the all-options mesh.
template <bool Weighted>
int checkGeometry(const char* tag, const Scene& s, const Built<Weighted>& b, const FM& m,
                  const FM& mNoDet) {
  const real_t h = s.h;
  auto A = toHost(m.faceCellA);
  auto B = toHost(m.faceCellB);
  auto FF = toHost(m.faceFacet);
  auto AR = toHost(m.faceArea);
  auto NM = toHost(m.faceNormal);
  auto DS = toHost(m.faceDist);
  auto HA = toHost(m.faceHa);
  auto HB = toHost(m.faceHb);
  auto CE = toHost(m.faceCentroid);
  auto CN = toHost(m.faceConn);
  auto off = toHost(m.cellFaceOffset);
  auto CF = toHost(m.cellFace);
  auto CS = toHost(m.cellFaceSign);
  auto CC = toHost(m.cellCentroid);
  auto V = toHost(m.cellVolume);
  auto cellOf = toHost(b.aux.cellOfFacet);
  auto ofc = toHost(b.oracle.facetCen);
  auto occ = toHost(b.oracle.cellCen);

  auto VA = toHost(b.view.facetArea);  // the published area vectors, 3*f + c
  double clos = 0, mom = 0, hsum = 0, cen = 0, sine = 0, meshSine = 0, ccen = 0;
  double cenNeg = 0, closNeg = 0, momNeg = 0;
  int nNeg = 0, nAntiNormal = 0;
  double sineBin[5] = {0, 0, 0, 0, 0};
  std::vector<char> touchesNeg(m.nCells, 0);
  const real_t aMin = real_t(1e-10) * h * h;
  for (int g = 0; g < m.nFaces; ++g) {
    const int f = FF(g);
    const int e = cellOf(f);
    real_t c[3] = {ofc(3 * f), ofc(3 * f + 1), ofc(3 * f + 2)};  // emitter-relative
    if (g < m.nInterior && A(g) != e)
      for (int d = 0; d < 3; ++d)
        c[d] += CN(3 * g + d);
    double ec = 0;
    for (int d = 0; d < 3; ++d)
      ec = std::max(ec, std::abs(CE(3 * g + d) - c[d]));
    cen = std::max(cen, ec / h);
    if (g >= m.nInterior)
      continue;
    const real_t d = DS(g);
    const real_t ulp = std::nextafter(d, std::numeric_limits<real_t>::infinity()) - d;
    hsum = std::max(hsum, std::abs(HA(g) + HB(g) - d) / ulp);
    const real_t nx = NM(3 * g), ny = NM(3 * g + 1), nz = NM(3 * g + 2);
    const real_t cx = CN(3 * g), cy = CN(3 * g + 1), cz = CN(3 * g + 2);
    if (nx * cx + ny * cy + nz * cz <= 0)
      ++nAntiNormal;
    if (AR(g) > aMin) {
      const real_t tx = ny * cz - nz * cy, ty = nz * cx - nx * cz, tz = nx * cy - ny * cx;
      meshSine = std::max(meshSine, std::sqrt(double(tx * tx + ty * ty + tz * tz)) / d);
      // the published polygon normal against the connector
      const real_t iar = real_t(1) / AR(g);
      const real_t px = VA(3 * f) * iar, py = VA(3 * f + 1) * iar, pz = VA(3 * f + 2) * iar;
      const real_t ux = py * cz - pz * cy, uy = pz * cx - px * cz, uz = px * cy - py * cx;
      const double sn = std::sqrt(ux * ux + uy * uy + uz * uz) / d;
      sine = std::max(sine, sn);
      for (int q = 0; q < 5; ++q)  // by area class A_f/h² > 1e-10, 1e-8, ..., 1e-2
        if (AR(g) > aMin * std::pow(100.0, q))
          sineBin[q] = std::max(sineBin[q], sn);
    }
    if (HA(g) < 0 || HB(g) < 0) {
      ++nNeg;
      cenNeg = std::max(cenNeg, ec / h);
      touchesNeg[A(g)] = 1;
      if (B(g) >= 0 && B(g) < m.nCells)
        touchesNeg[B(g)] = 1;
    }
  }
  for (int i = 0; i < m.nCells; ++i) {
    real_t sa[3] = {0, 0, 0}, M[3][3] = {{0}}, acell = 0;
    for (int q = off(i); q < off(i + 1); ++q) {
      const int g = CF(q);
      const real_t sg = CS(q);
      real_t r[3] = {CE(3 * g), CE(3 * g + 1), CE(3 * g + 2)};
      if (sg < 0)
        for (int d = 0; d < 3; ++d)
          r[d] -= CN(3 * g + d);
      acell += AR(g);
      for (int a = 0; a < 3; ++a) {
        const real_t na = sg * NM(3 * g + a) * AR(g);
        sa[a] += na;
        for (int k = 0; k < 3; ++k)
          M[a][k] += na * r[k];
      }
    }
    double fro = 0;
    for (int a = 0; a < 3; ++a)
      for (int k = 0; k < 3; ++k) {
        const double x = M[a][k] - (a == k ? V(i) : 0);
        fro += x * x;
      }
    const double ci = std::sqrt(sa[0] * sa[0] + sa[1] * sa[1] + sa[2] * sa[2]) / acell;
    const double mi = std::sqrt(fro) / V(i);
    clos = std::max(clos, ci);
    mom = std::max(mom, mi);
    if (touchesNeg[i]) {
      closNeg = std::max(closNeg, ci);
      momNeg = std::max(momNeg, mi);
    }
    for (int d = 0; d < 3; ++d)
      ccen = std::max(ccen, std::abs(CC(3 * i + d) - occ(3 * i + d)) / h);
  }
  // the gid-oriented mesh without the deterministic order, face by face (key: the cell pair)
  double noDet = 0;
  int noDetFlipped = 0, noDetMissing = 0;
  {
    auto A2 = toHost(mNoDet.faceCellA);
    auto B2 = toHost(mNoDet.faceCellB);
    auto FF2 = toHost(mNoDet.faceFacet);
    auto NM2 = toHost(mNoDet.faceNormal);
    auto AR2 = toHost(mNoDet.faceArea);
    auto CE2 = toHost(mNoDet.faceCentroid);
    auto CN2 = toHost(mNoDet.faceConn);
    auto HA2 = toHost(mNoDet.faceHa);
    auto CC2 = toHost(mNoDet.cellCentroid);
    std::vector<std::pair<std::int64_t, int>> key(m.nInterior), key2(mNoDet.nInterior);
    for (int g = 0; g < m.nInterior; ++g)
      key[g] = {std::int64_t(A(g)) * s.N + B(g), g};
    for (int g = 0; g < mNoDet.nInterior; ++g) {
      key2[g] = {std::int64_t(A2(g)) * s.N + B2(g), g};
      noDetFlipped += cellOf(FF2(g)) != A2(g) ? 1 : 0;
    }
    std::sort(key.begin(), key.end());
    std::sort(key2.begin(), key2.end());
    if (key.size() != key2.size())
      noDetMissing = 1;
    for (std::size_t q = 0; q < key.size() && !noDetMissing; ++q) {
      if (key[q].first != key2[q].first) {
        ++noDetMissing;
        break;
      }
      const int g = key[q].second, g2 = key2[q].second;
      for (int d = 0; d < 3; ++d) {
        // area vectors (a twin facet's unit normal carries the view's round-off on tiny faces)
        noDet =
            std::max(noDet, std::abs(AR(g) * NM(3 * g + d) - AR2(g2) * NM2(3 * g2 + d)) / (h * h));
        noDet = std::max(noDet, std::abs(CE(3 * g + d) - CE2(3 * g2 + d)) / h);
        noDet = std::max(noDet, std::abs(CN(3 * g + d) - CN2(3 * g2 + d)) / h);
      }
      noDet = std::max(noDet, std::abs(HA(g) - HA2(g2)) / h);
    }
    for (int i = 0; i < m.nCells; ++i)
      for (int d = 0; d < 3; ++d)
        noDet = std::max(noDet, std::abs(CC(3 * i + d) - CC2(3 * i + d)) / h);
  }
  const bool okDiag =
      m.generatorDiag.maxNormalConnSine == sine && m.generatorDiag.numNegativeHeight == nNeg;
  // the builder's own guarantees decide the exit code; |n x conn|/d measures the PUBLISHED area
  // vector's orientation (round-off of the polygon normal on small faces) and is judged by the
  // gate script from its own line
  const double meshSineUlp = meshSine / std::numeric_limits<real_t>::epsilon();
  const bool pass = clos <= 1e-13 && mom <= 1e-13 && hsum <= 2 && cen <= 1e-13 && ccen <= 1e-13 &&
                    meshSineUlp <= 4 && okDiag && noDet <= 1e-12 && noDetMissing == 0 &&
                    noDetFlipped > 0 && nAntiNormal == 0;
  std::printf(
      "RESULT b %-7s %s faces=%d closure=%.2e moment=%.2e hsum_ulp=%.1f centroid=%.2e "
      "mesh_sine=%.2e (%.1f ulp) pub_sine=%.2e cellcentroid=%.2e diag=%s negh=%d "
      "negh_centroid=%.2e negh_closure=%.2e "
      "negh_moment=%.2e nodet=%.2e nodet_flipped=%d nodet_missing=%d anti_normal=%d\n",
      tag, pass ? "PASS" : "FAIL", m.nFaces, clos, mom, hsum, cen, meshSine, meshSineUlp, sine,
      ccen, okDiag ? "match" : "MISMATCH", nNeg, cenNeg, closNeg, momNeg, noDet, noDetFlipped,
      noDetMissing, nAntiNormal);
  std::printf(
      "RESULT b_sine %-7s %s published-normal max |n x conn|/d = %.2e on A_f > 1e-2 h^2 "
      "(criterion 1e-12)\n",
      tag, sineBin[4] <= 1e-12 ? "PASS" : "MISS", sineBin[4]);
  std::printf(
      "INFO b %-7s published-normal max |n x conn|/d for A_f/h^2 > 1e-10: %.2e  > 1e-8: %.2e  > "
      "1e-6: %.2e  "
      "> 1e-4: %.2e  > 1e-2: %.2e\n",
      tag, sineBin[0], sineBin[1], sineBin[2], sineBin[3], sineBin[4]);
  return pass ? 0 : 1;
}

void dumpView(const std::string& path, const char* data, std::size_t bytes) {
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(2);
  }
  if (bytes)
    std::fwrite(data, 1, bytes, fp);
  std::fclose(fp);
}

template <class V>
void dumpArr(const std::string& dir, const std::string& tag, const char* name, const V& v) {
  auto hv = toHost(v);
  dumpView(dir + "/" + tag + "." + name, reinterpret_cast<const char*>(hv.data()),
           v.extent(0) * sizeof(typename V::value_type));
}

void dumpMesh(const std::string& dir, const std::string& tag, const FM& m) {
  const int counts[4] = {m.nFaces, m.nInterior, m.nDropped, m.generatorDiag.numNegativeHeight};
  dumpView(dir + "/" + tag + ".counts", reinterpret_cast<const char*>(counts), sizeof(counts));
  dumpArr(dir, tag, "faceCellA", m.faceCellA);
  dumpArr(dir, tag, "faceCellB", m.faceCellB);
  dumpArr(dir, tag, "faceFacet", m.faceFacet);
  dumpArr(dir, tag, "faceArea", m.faceArea);
  dumpArr(dir, tag, "faceNormal", m.faceNormal);
  dumpArr(dir, tag, "faceDist", m.faceDist);
  dumpArr(dir, tag, "faceHa", m.faceHa);
  dumpArr(dir, tag, "faceHb", m.faceHb);
  dumpArr(dir, tag, "faceCentroid", m.faceCentroid);
  dumpArr(dir, tag, "faceConn", m.faceConn);
  dumpArr(dir, tag, "cellFaceOffset", m.cellFaceOffset);
  dumpArr(dir, tag, "cellFace", m.cellFace);
  dumpArr(dir, tag, "cellFaceSign", m.cellFaceSign);
  dumpArr(dir, tag, "cellVolume", m.cellVolume);
  dumpArr(dir, tag, "cellCentroid", m.cellCentroid);
}

template <bool Weighted>
int runCase(const char* tag, const Scene& s, const std::string& dumpDir, bool timeIt) {
  Built<Weighted> b;
  build(s, b);
  const bool serial = Exec().concurrency() == 1;
  int rc = 0;
  // (d) deterministic order alone vs the default builder
  const FM m0 = peclet::voro::fv::buildFaceMesh(b.view, b.aux);
  FMO det;
  det.deterministicOrder = true;
  const FM m1 = peclet::voro::fv::buildFaceMesh(b.view, b.aux, -1, det);
  rc |= checkDefaultEquivalence(tag, m0, m1, serial);
  // (b) all four options (gid permuted), and the same without the deterministic order
  FMO all;
  all.deterministicOrder = all.gidOrientation = all.generatorGeometry = all.cellCentroids = true;
  all.gid = s.gidPerm;
  all.pos = s.pos;
  if (Weighted)
    all.weight = s.weight;
  for (int d = 0; d < 3; ++d)
    all.L[d] = s.L[d];
  const FM m2 = peclet::voro::fv::buildFaceMesh(b.view, b.aux, -1, all);
  FMO noDet = all;
  noDet.deterministicOrder = false;
  const FM m3 = peclet::voro::fv::buildFaceMesh(b.view, b.aux, -1, noDet);
  rc |= checkGeometry(tag, s, b, m2, m3);
  {  // (c′) a second build from the same view, every array bitwise
    const FM m4 = peclet::voro::fv::buildFaceMesh(b.view, b.aux, -1, all);
    int nd = m2.nFaces == m4.nFaces && m2.nInterior == m4.nInterior && m2.nDropped == m4.nDropped &&
                     m2.generatorDiag.numNegativeHeight == m4.generatorDiag.numNegativeHeight
                 ? 0
                 : 1;
    nd += sameBits(m2.faceCellA, m4.faceCellA) ? 0 : 1;
    nd += sameBits(m2.faceCellB, m4.faceCellB) ? 0 : 1;
    nd += sameBits(m2.faceFacet, m4.faceFacet) ? 0 : 1;
    nd += sameBits(m2.faceArea, m4.faceArea) ? 0 : 1;
    nd += sameBits(m2.faceNormal, m4.faceNormal) ? 0 : 1;
    nd += sameBits(m2.faceDist, m4.faceDist) ? 0 : 1;
    nd += sameBits(m2.faceHa, m4.faceHa) ? 0 : 1;
    nd += sameBits(m2.faceHb, m4.faceHb) ? 0 : 1;
    nd += sameBits(m2.faceCentroid, m4.faceCentroid) ? 0 : 1;
    nd += sameBits(m2.faceConn, m4.faceConn) ? 0 : 1;
    nd += sameBits(m2.cellFaceOffset, m4.cellFaceOffset) ? 0 : 1;
    nd += sameBits(m2.cellFace, m4.cellFace) ? 0 : 1;
    nd += sameBits(m2.cellFaceSign, m4.cellFaceSign) ? 0 : 1;
    nd += sameBits(m2.cellVolume, m4.cellVolume) ? 0 : 1;
    nd += sameBits(m2.cellCentroid, m4.cellCentroid) ? 0 : 1;
    std::printf("RESULT c_sameview %-7s %s %d\n", tag, nd ? "MISS" : "PASS", nd);
  }
  if (!dumpDir.empty()) {  // (c)
    dumpMesh(dumpDir, std::string(tag) + ".det", m1);
    dumpMesh(dumpDir, std::string(tag) + ".all", m2);
    dumpArr(dumpDir, std::string(tag) + ".view", "area", b.view.facetArea);
    dumpArr(dumpDir, std::string(tag) + ".view", "nbr", b.view.facetNeighbor);
  }
  if (timeIt) {  // builder wall time, options vs default (item 2 of the brief; no threshold)
    const int reps = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      (void)peclet::voro::fv::buildFaceMesh(b.view, b.aux);
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      (void)peclet::voro::fv::buildFaceMesh(b.view, b.aux, -1, all);
    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      (void)peclet::voro::fv::buildFaceMesh(b.view, b.aux, -1, det);
    auto t3 = std::chrono::steady_clock::now();
    const double ms = 1e3 / reps;
    std::printf("TIME %-7s threads=%d default=%.3f ms all_options=%.3f ms det_only=%.3f ms\n", tag,
                Exec().concurrency(), std::chrono::duration<double>(t1 - t0).count() * ms,
                std::chrono::duration<double>(t2 - t1).count() * ms,
                std::chrono::duration<double>(t3 - t2).count() * ms);
  }
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    int n = 16;
    std::string dumpDir;
    bool timeIt = false;
    for (int a = 1; a < argc; ++a) {
      if (!std::strcmp(argv[a], "--dump") && a + 1 < argc)
        dumpDir = argv[++a];
      else if (!std::strcmp(argv[a], "--time"))
        timeIt = true;
      else
        n = std::atoi(argv[a]);
    }
    const Scene s = makeScene(n);
    std::printf("test_ale_facemesh: n=%d N=%d threads=%d\n", n, s.N, Exec().concurrency());
    rc |= runCase<true>("power", s, dumpDir, timeIt);
    rc |= runCase<false>("voronoi", s, dumpDir, timeIt);
    std::printf("test_ale_facemesh: %s\n", rc ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return rc;
}
