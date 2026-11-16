/**
 * @file test_determinism.cpp
 * \brief Thread-count determinism of the published tessellation, and the cold build's
 *        partition invariants on moving power seeds.
 *
 * Prints the SHA-256 of every published array; the companion script `determinism_check.sh` runs
 * this binary at several OpenMP thread counts and requires the hashes to be equal. Scene: an n³
 * periodic lattice jittered ±0.3h; power weights uniform in [0, 0.15h²] (fixed seed).
 *
 *  cold:  for Voronoi and power, the cold build's own view (buildTessellation, force geometry
 *         on) and the resident engine's publish (MovingTessellation::rebuild → reevalPublish →
 *         buildAuxMaps → buildFaceMesh) — one `HASH cold <policy> <array> <sha>` line per array.
 *  move:  `frames` steps of u = U0 (sin 2πy, sin 2πz, sin 2πx), explicit Euler, U0·dt = 0.05h/√3,
 *         wrapped into [0,1); the engine repairs (`step`) and publishes every frame; one
 *         `FRAME <policy> k <sha>` line per frame over the published arrays + the rebuilt mask.
 *  guard: every frame of the power run, a cold build of that frame's positions: the volume sum
 *         against the box, facets without a reciprocal (cell i has a facet to j, j none to i),
 *         and every empty cell published with zero volume and no facets.
 *
 * Usage: test_determinism [--n N] [--frames F] [--every K]
 * Exit status: nonzero if a guard check fails (hashes are compared by the script, not here).
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"
#include "peclet/voro/transpose.hpp"

constexpr int kP = peclet::voro::kMaxPlanes;
constexpr int kT = peclet::voro::kMaxTriangles;
template <bool W>
using Engine = peclet::voro::MovingTessellation<Real, kP, kT, W>;

namespace {

// ---- SHA-256 (FIPS 180-4), enough to fingerprint byte arrays ----------------------------------
class Sha256 {
 public:
  Sha256() { reset(); }
  void reset() {
    static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, h0, sizeof(h_));
    len_ = 0;
    nbuf_ = 0;
  }
  void update(const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    len_ += n;
    while (n > 0) {
      const std::size_t take = std::min<std::size_t>(64 - nbuf_, n);
      std::memcpy(buf_ + nbuf_, p, take);
      nbuf_ += take;
      p += take;
      n -= take;
      if (nbuf_ == 64) {
        block(buf_);
        nbuf_ = 0;
      }
    }
  }
  std::string hex() {
    const uint64_t bits = len_ * 8;
    const unsigned char one = 0x80, zero = 0;
    update(&one, 1);
    while (nbuf_ != 56)
      update(&zero, 1);
    unsigned char lb[8];
    for (int k = 0; k < 8; ++k)
      lb[k] = static_cast<unsigned char>(bits >> (56 - 8 * k));
    update(lb, 8);
    char out[65];
    for (int k = 0; k < 8; ++k)
      std::snprintf(out + 8 * k, 9, "%08x", h_[k]);
    return std::string(out, 64);
  }

 private:
  static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
  void block(const unsigned char* b) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t w[64];
    for (int t = 0; t < 16; ++t)
      w[t] = (uint32_t(b[4 * t]) << 24) | (uint32_t(b[4 * t + 1]) << 16) |
             (uint32_t(b[4 * t + 2]) << 8) | uint32_t(b[4 * t + 3]);
    for (int t = 16; t < 64; ++t) {
      const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
      const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
      w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    uint32_t a = h_[0], bb = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6],
             h = h_[7];
    for (int t = 0; t < 64; ++t) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + S1 + ch + K[t] + w[t];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t mj = (a & bb) ^ (a & c) ^ (bb & c);
      const uint32_t t2 = S0 + mj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = bb;
      bb = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += bb;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }
  uint32_t h_[8];
  uint64_t len_;
  unsigned char buf_[64];
  std::size_t nbuf_;
};

/// A named list of device arrays, hashed one by one and as a whole.
struct Digest {
  std::vector<std::pair<std::string, std::string>> parts;  // (name, sha)
  Sha256 all;
  template <class T>
  void add(const std::string& name, const Kokkos::View<T*, Mem>& v) {
    const auto h = down(v);
    Sha256 one;
    one.update(h.data(), h.size() * sizeof(T));
    all.update(h.data(), h.size() * sizeof(T));
    parts.emplace_back(name, one.hex());
  }
};

struct Scene {
  int n = 0, N = 0;
  Real L[3] = {1, 1, 1};
  Real h = 0;
  DV pos, weight;
};

/// The S0.2 scene (test_ale_frame on branch ale-s0): same generator, same draw order.
Scene makeScene(int n) {
  Scene s;
  s.n = n;
  s.N = n * n * n;
  s.h = Real(1) / n;
  std::mt19937 rng(20261114u);
  std::uniform_real_distribution<Real> U(-1, 1), U01(0, 1);
  std::vector<Real> x(3 * s.N), w(s.N);
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const int c = i + n * (j + n * k);
        const int ijk[3] = {i, j, k};
        for (int d = 0; d < 3; ++d) {
          Real p = (ijk[d] + Real(0.5)) * s.h + Real(0.3) * s.h * U(rng);
          p = p < 0 ? p + 1 : (p >= 1 ? p - 1 : p);
          x[3 * c + d] = p;
        }
      }
  for (auto& v : w)
    v = Real(0.15) * s.h * s.h * U01(rng);
  s.pos = up(x, "pos");
  s.weight = up(w, "w");
  return s;
}

template <bool W>
void allocEngine(Engine<W>& mt, const Scene& s) {
  mt.alloc(s.N, s.L, Real(1e-4) * s.h, Real(0.25) * s.h, 4, s.N);
  if constexpr (W)
    mt.setWeights(s.weight);
}

template <bool W>
peclet::voro::TessellatorResult<Real> coldBuild(const Scene& s, const DV& pos) {
  return peclet::voro::buildTessellation<Real, W, peclet::voro::NoSdf>(pos, W ? s.weight : DV(),
                                                                      s.N, s.L);
}

void addView(Digest& d, const std::string& pre, const peclet::voro::TessellationView<Real>& v) {
  d.add(pre + "cellFacetOffset", v.cellFacetOffset);
  d.add(pre + "cellFacetCount", v.cellFacetCount);
  d.add(pre + "cellSeedId", v.cellSeedId);
  d.add(pre + "cellVolume", v.cellVolume);
  d.add(pre + "facetNeighbor", v.facetNeighbor);
  d.add(pre + "facetArea", v.facetArea);
  d.add(pre + "facetConnect", v.facetConnect);
  d.add(pre + "facetConnVec", v.facetConnVec);
}

void addMesh(Digest& d, const fv::FaceMesh<Real>& m) {
  d.add("mesh.faceCellA", m.faceCellA);
  d.add("mesh.faceCellB", m.faceCellB);
  d.add("mesh.faceFacet", m.faceFacet);
  d.add("mesh.faceArea", m.faceArea);
  d.add("mesh.faceNormal", m.faceNormal);
  d.add("mesh.faceDist", m.faceDist);
  d.add("mesh.faceHa", m.faceHa);
  d.add("mesh.faceHb", m.faceHb);
  d.add("mesh.faceCentroid", m.faceCentroid);
  d.add("mesh.faceConn", m.faceConn);
  d.add("mesh.cellFaceOffset", m.cellFaceOffset);
  d.add("mesh.cellFace", m.cellFace);
  d.add("mesh.cellFaceSign", m.cellFaceSign);
}

/// The resident engine's publish: store → reevalPublish → aux maps → face mesh.
template <bool W>
void addPublish(Digest& d, Engine<W>& mt, const Scene& s, const DV& pos) {
  auto view = peclet::voro::reevalPublish<Real, kP, kT>(mt.store, pos, mt.vol, s.N, s.L, mt.wall,
                                                        mt.xRef);
  auto aux = peclet::voro::buildAuxMaps(view);
  auto mesh = fv::buildFaceMesh(view, aux);
  addView(d, "pub.", view);
  d.add("aux.recip", aux.recip);
  d.add("aux.cellOfFacet", aux.cellOfFacet);
  addMesh(d, mesh);
  d.add("store.np", mt.store.np);
  d.add("store.nt", mt.store.nt);
  d.add("store.pnbr", mt.store.pnbr);
  d.add("store.tri", mt.store.tri);
}

template <bool W>
void coldHashes(const Scene& s, const char* tag) {
  Digest d;
  auto res = coldBuild<W>(s, s.pos);
  addView(d, "raw.", res.view);
  d.add("raw.status", res.status);
  Engine<W> mt;
  allocEngine<W>(mt, s);
  mt.rebuild(s.pos);
  addPublish<W>(d, mt, s, s.pos);
  for (const auto& p : d.parts)
    std::printf("HASH cold %s %s %s\n", tag, p.first.c_str(), p.second.c_str());
  std::printf("HASH cold %s ALL %s\n", tag, d.all.hex().c_str());
}

/// Partition invariants of one cold build (host-side).
struct GuardStats {
  double volRel = 0;      // |Σ V − L³| / L³
  long nonRecip = 0;      // facets i→j (|A| > 0) with no facet j→i
  double nonRecipA = 0;   // the largest such |A|, in h²
  long empty = 0;         // cells with status kEmpty
  long buried = 0;        // power seeds outside their own cell (kBuried)
  long emptyBad = 0;      // empty cells published with a volume or facets
};

GuardStats guard(const Scene& s, const peclet::voro::TessellatorResult<Real>& res) {
  GuardStats g;
  const auto vol = down(res.view.cellVolume);
  const auto off = down(res.view.cellFacetOffset);
  const auto cnt = down(res.view.cellFacetCount);
  const auto nbr = down(res.view.facetNeighbor);
  const auto area = down(res.view.facetArea);
  const auto st = down(res.status);
  double sum = 0;
  for (int i = 0; i < s.N; ++i)
    sum += vol[i];
  g.volRel = std::fabs(sum - 1.0);
  std::vector<std::vector<long>> nb(s.N);
  for (int i = 0; i < s.N; ++i) {
    for (int f = off[i]; f < off[i] + cnt[i]; ++f)
      nb[i].push_back(static_cast<long>(nbr[f]));
    std::sort(nb[i].begin(), nb[i].end());
    if (st[i] & peclet::voro::kEmpty) {
      ++g.empty;
      if (vol[i] != Real(0) || cnt[i] != 0)
        ++g.emptyBad;
    }
    if (st[i] & peclet::voro::kBuried)
      ++g.buried;
  }
  const double h2 = s.h * s.h;
  for (int i = 0; i < s.N; ++i)
    for (int f = off[i]; f < off[i] + cnt[i]; ++f) {
      const long j = static_cast<long>(nbr[f]);
      if (j < 0)
        continue;
      const double a = std::sqrt(area[3 * f] * area[3 * f] + area[3 * f + 1] * area[3 * f + 1] +
                                 area[3 * f + 2] * area[3 * f + 2]);
      if (a == 0)
        continue;
      if (!std::binary_search(nb[j].begin(), nb[j].end(), static_cast<long>(i))) {
        ++g.nonRecip;
        g.nonRecipA = std::max(g.nonRecipA, a / h2);
      }
    }
  return g;
}

template <bool W>
int moving(const Scene& s, const char* tag, int frames, int every, bool doGuard) {
  Engine<W> mt;
  allocEngine<W>(mt, s);
  DV pos("pos", 3 * s.N);
  Kokkos::deep_copy(pos, s.pos);
  mt.rebuild(pos);
  const Real twoPi = Real(2) * Real(M_PI);
  const Real step = Real(0.05) * s.h / std::sqrt(Real(3));  // U0·dt
  int bad = 0;
  GuardStats worst;
  long totalRebuilt = 0;
  int fell = 0, gateRebuilds = 0;
  for (int k = 1; k <= frames; ++k) {
    {
      auto P = pos;
      Kokkos::parallel_for(
          "det.advect", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, s.N),
          KOKKOS_LAMBDA(const int i) {
            const Real x = P(3 * i), y = P(3 * i + 1), z = P(3 * i + 2);
            Real q[3] = {x + step * Kokkos::sin(twoPi * y), y + step * Kokkos::sin(twoPi * z),
                         z + step * Kokkos::sin(twoPi * x)};
            for (int d = 0; d < 3; ++d) {
              q[d] = q[d] < Real(0) ? q[d] + Real(1) : (q[d] >= Real(1) ? q[d] - Real(1) : q[d]);
              P(3 * i + d) = q[d];
            }
          });
    }
    const auto st = mt.step(pos);
    fell += st.fellBack ? 1 : 0;
    gateRebuilds += st.route == peclet::voro::RepairStats::kRebuildGate ? 1 : 0;
    long nReb = 0;
    {
      auto rb = mt.rebuilt;
      Kokkos::parallel_reduce(
          "det.nRebuilt", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, s.N),
          KOKKOS_LAMBDA(const int i, long& a) { a += rb(i); }, nReb);
    }
    totalRebuilt += nReb;
    if (k % every == 0 || k == frames) {
      Digest d;
      d.add("pos", pos);
      d.add("rebuilt", mt.rebuilt);
      addPublish<W>(d, mt, s, pos);
      std::printf("FRAME %s %d %s rebuilt=%ld\n", tag, k, d.all.hex().c_str(), nReb);
    }
    if (doGuard) {
      const auto g = guard(s, coldBuild<W>(s, pos));
      const bool ok = g.volRel <= 1e-12 && g.nonRecip == 0 && g.emptyBad == 0;
      if (!ok) {
        ++bad;
        std::printf(
            "GUARD %s frame=%d MISS volRel=%.3e nonRecip=%ld maxNonRecipA/h2=%.3e empty=%ld "
            "buried=%ld emptyBad=%ld\n",
            tag, k, g.volRel, g.nonRecip, g.nonRecipA, g.empty, g.buried, g.emptyBad);
      }
      worst.volRel = std::max(worst.volRel, g.volRel);
      worst.nonRecip = std::max(worst.nonRecip, g.nonRecip);
      worst.nonRecipA = std::max(worst.nonRecipA, g.nonRecipA);
      worst.empty = std::max(worst.empty, g.empty);
      worst.buried = std::max(worst.buried, g.buried);
      worst.emptyBad = std::max(worst.emptyBad, g.emptyBad);
    }
  }
  std::printf("RESULT move %s n=%d frames=%d totalRebuilt=%ld fellBack=%d gateRebuilds=%d\n", tag,
              s.n, frames, totalRebuilt, fell, gateRebuilds);
  if (doGuard)
    std::printf(
        "RESULT guard %s %s framesMiss=%d maxVolRel=%.3e maxNonRecip=%ld maxNonRecipA/h2=%.3e "
        "maxEmpty=%ld maxBuried=%ld maxEmptyBad=%ld\n",
        tag, bad == 0 ? "PASS" : "FAIL", bad, worst.volRel, worst.nonRecip, worst.nonRecipA,
        worst.empty, worst.buried, worst.emptyBad);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  int n = 16, frames = 40, every = 1;
  bool doGuard = true;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--n") && a + 1 < argc)
      n = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--frames") && a + 1 < argc)
      frames = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--every") && a + 1 < argc)
      every = std::max(1, std::atoi(argv[++a]));
    else if (!std::strcmp(argv[a], "--no-guard"))
      doGuard = false;
  }
  int rc = 0;
  Kokkos::initialize(argc, argv);
  {
    std::printf("test_determinism: n=%d frames=%d threads=%d\n", n, frames,
                peclet::core::ExecSpace().concurrency());
    const Scene s = makeScene(n);
    coldHashes<false>(s, "voronoi");
    coldHashes<true>(s, "power");
    moving<false>(s, "voronoi", frames, every, false);
    rc |= moving<true>(s, "power", frames, every, doGuard);
  }
  Kokkos::finalize();
  return rc;
}
