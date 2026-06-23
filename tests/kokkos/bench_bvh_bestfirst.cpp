/**
 * @file bench_bvh_bestfirst.cpp
 * \brief F4 / Option-2 — best-first BVH traversal with a cell-dependent stop.
 *
 * ArborX builds the BVH (its strength); we drive a custom **best-first** traversal over its
 * nodes (a per-cell min-heap of pending nodes keyed by distance-to-seed) and clip the ConvexCell
 * as we descend, **stopping the moment the closest pending node is beyond the security radius**
 * (2·sqrt(maxVertexRsq)) — no fixed K, no over/under-gather, robust to non-uniform point sets.
 * This is what ArborX's public API cannot do (its `nearest` stop is fixed-K, callback post-hoc).
 *
 * ArborX node access is confined to the `BvhAccess` adapter (the only place touching ArborX
 * `detail/HappyTreeFriends` — a one-file change if ArborX's internals move). Periodic via boundary
 * ghosts. Validates Σvol == box; reports throughput vs the F4 fixed-K kNN (6.7 M/s). FP32.
 */
#include <ArborX.hpp>
#include <detail/ArborX_HappyTreeFriends.hpp>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"

using real_t = float;
using Exec = Kokkos::DefaultExecutionSpace;
using Mem = Exec::memory_space;
using Point = ArborX::Point<3, float>;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
static constexpr int CC_MAXP = 64, CC_MAXT = 112, HCAP = 128;

/// Thin adapter over ArborX's internal node accessors — the ONLY place coupled to
/// ArborX::Details::HappyTreeFriends. Node index space: leaves [0,size), root = size,
/// internals [size, 2*size). One file to change if ArborX's internals move.
template <class BVH>
struct BvhAccess {
  using HTF = ArborX::Details::HappyTreeFriends;
  static KOKKOS_INLINE_FUNCTION int root(BVH const& b) { return HTF::getRoot(b); }
  static KOKKOS_INLINE_FUNCTION bool isLeaf(BVH const& b, int n) { return HTF::isLeaf(b, n); }
  static KOKKOS_INLINE_FUNCTION int leftChild(BVH const& b, int n) { return HTF::getLeftChild(b, n); }
  static KOKKOS_INLINE_FUNCTION int rightChild(BVH const& b, int n) { return HTF::getRightChild(b, n); }
  static KOKKOS_INLINE_FUNCTION int leafIndex(BVH const& b, int n) {
    return (int)HTF::getValue(b, n).index;
  }
  static KOKKOS_INLINE_FUNCTION void leafPoint(BVH const& b, int n, float p[3]) {
    auto const ix = HTF::getIndexable(b, n);
    p[0] = ix[0]; p[1] = ix[1]; p[2] = ix[2];
  }
  // squared distance from seed to node n's bounding volume (point for a leaf, AABB for internal)
  static KOKKOS_INLINE_FUNCTION float dist2(BVH const& b, int n, const float s[3]) {
    if (isLeaf(b, n)) {
      float p[3]; leafPoint(b, n, p);
      const float dx = p[0] - s[0], dy = p[1] - s[1], dz = p[2] - s[2];
      return dx * dx + dy * dy + dz * dz;
    }
    auto const& bv = HTF::getInternalBoundingVolume(b, n);
    float d = 0;
    for (int k = 0; k < 3; ++k) {
      const float lo = bv.minCorner()[k], hi = bv.maxCorner()[k];
      const float e = s[k] < lo ? lo - s[k] : (s[k] > hi ? s[k] - hi : 0.f);
      d += e * e;
    }
    return d;
  }
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = argc > 1 ? std::atoi(argv[1]) : 1000000;
    const float L = 1.0f, spacing = std::cbrt(1.0f / N), margin = 5.0f * spacing;
    const float boxL = 10.0f * spacing;  // tight initial cell box (±5 spacings)
    std::printf("backend: %s  N=%d\n", Exec::name(), N);
    std::mt19937 rng(99 + N);
    std::uniform_real_distribution<float> U(0, 1);
    std::vector<float> px(N), py(N), pz(N);
    for (int i = 0; i < N; ++i) { px[i] = U(rng); py[i] = U(rng); pz[i] = U(rng); }
    std::vector<Point> hpts;
    std::vector<int> horig;
    for (int i = 0; i < N; ++i) { hpts.push_back({px[i], py[i], pz[i]}); horig.push_back(i); }
    for (int i = 0; i < N; ++i)
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            if (!dx && !dy && !dz) continue;
            const float x = px[i] + dx * L, y = py[i] + dy * L, z = pz[i] + dz * L;
            if (x < -margin || x > L + margin || y < -margin || y > L + margin || z < -margin ||
                z > L + margin)
              continue;
            hpts.push_back({x, y, z});
            horig.push_back(i);
          }
    const int M = (int)hpts.size();
    std::printf("ghosts: %d (%.1f%%)\n", M - N, 100.0 * (M - N) / N);
    Kokkos::View<Point*, Mem> pts(Kokkos::view_alloc("pts", Kokkos::WithoutInitializing), M);
    Kokkos::View<int*, Mem> orig(Kokkos::view_alloc("orig", Kokkos::WithoutInitializing), M);
    {
      auto hp = Kokkos::create_mirror_view(pts);
      auto ho = Kokkos::create_mirror_view(orig);
      for (int i = 0; i < M; ++i) { hp(i) = hpts[i]; ho(i) = horig[i]; }
      Kokkos::deep_copy(pts, hp); Kokkos::deep_copy(orig, ho);
    }

    Exec space;
    Kokkos::View<real_t*, Mem> cellVol("vol", N);
    Kokkos::View<int*, Mem> cellFaces("faces", N), cellOvf("ovf", N);
    double tBuild = 0, tTrav = 0;

    auto run = [&](bool timing) {
      auto t0 = clk::now();
      ArborX::BoundingVolumeHierarchy bvh{space, ArborX::Experimental::attach_indices(pts)};
      space.fence();
      auto t1 = clk::now();
      using BVH = decltype(bvh);
      auto ptsL = pts; auto origL = orig;
      Kokkos::parallel_for(
          "bestfirst", Kokkos::RangePolicy<Exec>(space, 0, N), KOKKOS_LAMBDA(int i) {
            using A = BvhAccess<BVH>;
            const float s[3] = {ptsL(i)[0], ptsL(i)[1], ptsL(i)[2]};
            vor::device::ConvexCell<real_t, CC_MAXP, CC_MAXT> c;
            // Seed a TIGHT box (a few spacings), not the full domain: the security radius
            // (4·maxVertexRsq) then starts ~(L/box)² smaller, so push-prune bites from the
            // first node instead of after ~6 clips. boxL is a safe over-estimate of the cell
            // diameter; if the final cell still touches a box face the result is rejected.
            c.initBox(boxL, boxL, boxL);
            float secR = 4.0f * (float)c.maxVertexRsq();  // cached; only changes on a clip
            // per-thread min-heap of (dist2, node) — globally-closest-first, so the cell shrinks
            // optimally and secR prunes the most. (A stack DFS with the same radius was measured
            // slower: more nodes visited despite cheaper per-node cost.)
            float hd[HCAP]; int hn[HCAP]; int hs = 0;
            {  // push root
              const int rt = A::root(bvh);
              hd[0] = A::dist2(bvh, rt, s); hn[0] = rt; hs = 1;
            }
            bool ovf = false;
            while (hs > 0) {
              // pop-min
              const float d2 = hd[0]; const int node = hn[0];
              --hs; hd[0] = hd[hs]; hn[0] = hn[hs];
              for (int p = 0;;) {  // sift down
                int l = 2 * p + 1, r = l + 1, m = p;
                if (l < hs && hd[l] < hd[m]) m = l;
                if (r < hs && hd[r] < hd[m]) m = r;
                if (m == p) break;
                float td = hd[p]; hd[p] = hd[m]; hd[m] = td;
                int tn = hn[p]; hn[p] = hn[m]; hn[m] = tn;
                p = m;
              }
              // security stop: nothing closer than the popped node can cut (min-heap => done)
              if (d2 >= secR) break;
              if (A::isLeaf(bvh, node)) {
                const int id = A::leafIndex(bvh, node);
                if (id == i) continue;  // self (real or own ghost)
                float p[3]; A::leafPoint(bvh, node, p);
                const real_t r[3] = {p[0] - s[0], p[1] - s[1], p[2] - s[2]};
                const real_t off = real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                if (c.clip(r, off, id)) secR = 4.0f * (float)c.maxVertexRsq();  // shrank
                if (c.overflow) { ovf = true; break; }
              } else {
                const int ch[2] = {A::leftChild(bvh, node), A::rightChild(bvh, node)};
                for (int k = 0; k < 2; ++k) {
                  float nd = A::dist2(bvh, ch[k], s);
                  if (nd >= secR) continue;  // push-prune: child can't contribute a cutter
                  if (hs >= HCAP) { ovf = true; break; }
                  int p = hs++;  // push, sift up
                  hd[p] = nd; hn[p] = ch[k];
                  while (p > 0) {
                    int par = (p - 1) / 2;
                    if (hd[par] <= hd[p]) break;
                    float td = hd[par]; hd[par] = hd[p]; hd[p] = td;
                    int tn = hn[par]; hn[par] = hn[p]; hn[p] = tn;
                    p = par;
                  }
                }
                if (ovf) break;
              }
            }
            cellVol(i) = (c.overflow || ovf) ? real_t(0) : c.volume();  // G1, timed
            if (!timing) {
              cellFaces(i) = (c.overflow || ovf) ? 0 : c.countFaces();
              cellOvf(i) = (c.overflow || ovf) ? 1 : 0;
              real_t sg = 0;
              for (int k = 0; k < c.np; ++k) {
                if (c.pnbr[k] < 0) continue;
                real_t a[3], d[3], cv[3];
                if (c.facetGeometry(k, a, d, cv)) sg += d[0];
              }
              cellVol(i) += real_t(0) * sg;
            }
          });
      space.fence();
      auto t2 = clk::now();
      tBuild = secs(t0, t1); tTrav = secs(t1, t2);
    };

    run(false);
    double volSum = 0; long faceSum = 0, ovf = 0;
    {
      auto hv = Kokkos::create_mirror_view(cellVol);
      auto hf = Kokkos::create_mirror_view(cellFaces);
      auto ho = Kokkos::create_mirror_view(cellOvf);
      Kokkos::deep_copy(hv, cellVol); Kokkos::deep_copy(hf, cellFaces); Kokkos::deep_copy(ho, cellOvf);
      for (int i = 0; i < N; ++i) { volSum += hv(i); faceSum += hf(i); ovf += ho(i); }
    }
    double best = 1e30, bb = 0, bt = 0;
    for (int rep = 0; rep < 3; ++rep) {
      run(true);
      if (tBuild + tTrav < best) { best = tBuild + tTrav; bb = tBuild; bt = tTrav; }
    }
    std::printf("Σvol/box err=%.2e  faces/cell=%.2f  overflow=%ld\n", std::fabs(volSum - 1.0),
                (double)faceSum / N, ovf);
    std::printf("BVH-build %.1f k/s   best-first traverse+clip %.1f k/s\n", N / bb / 1e3,
                N / bt / 1e3);
    std::printf("TOTAL %.1f kcells/s   [F4 fixed-K=64: 6727,  grid: 5418]\n", N / best / 1e3);
  }
  Kokkos::finalize();
  return 0;
}
