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
 * Usage: test_determinism [--n N] [--frames F] [--every K] [--no-guard] [--detail]
 *        [--dump K FILE] [--time]  (--detail: per-array hashes on every FRAME line too; --dump: the
 * power run's positions then weights at frame K, raw doubles; --time: wall times only) Exit status:
 * nonzero if a guard check fails (hashes are compared by the script, not here).
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

template <class T>
Kokkos::View<T*, Mem> up(const std::vector<T>& h, const char* name) {
  Kokkos::View<T*, Mem> d(name, h.size());
  Kokkos::deep_copy(d, Kokkos::View<const T*, Kokkos::HostSpace>(h.data(), h.size()));
  return d;
}

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

/// The view's facets in CELL order (cell 0's facets, then cell 1's, ...), each cell's in its own
/// order: invariant under where the CSR places each cell, so it compares a finish-order CSR with
/// a cell-ordered one.
void addByCell(Digest& d, const std::string& pre, const peclet::voro::TessellationView<Real>& v) {
  const auto off = down(v.cellFacetOffset), cnt = down(v.cellFacetCount);
  const auto nbr = down(v.facetNeighbor);
  const auto a = down(v.facetArea), c = down(v.facetConnect), q = down(v.facetConnVec);
  std::vector<Real> rec;
  for (std::size_t i = 0; i < off.size(); ++i)
    for (int f = off[i]; f < off[i] + cnt[i]; ++f) {
      rec.push_back(static_cast<Real>(nbr[f]));
      for (int k = 0; k < 3; ++k) {
        rec.push_back(a[3 * f + k]);
        rec.push_back(c[3 * f + k]);
        rec.push_back(q[3 * f + k]);
      }
    }
  d.add(pre + "byCell", up(rec, "rec"));
}

/// The facet-edge area-Jacobian CSR in cell order with cell-relative facet indices (placement
/// invariant, like addByCell).
void addEdgesByCell(Digest& d, const std::string& pre,
                    const peclet::voro::TessellationView<Real>& v) {
  const auto off = down(v.cellFacetOffset), cnt = down(v.cellFacetCount);
  const auto eo = down(v.facetEdgeOffset), ec = down(v.facetEdgeCount);
  const auto ef = down(v.edgeFacet);
  const auto eg = down(v.edgeAreaGrad), el = down(v.edgeLength), m2 = down(v.facetMoment2);
  std::vector<Real> rec;
  for (std::size_t i = 0; i < off.size(); ++i)
    for (int f = off[i]; f < off[i] + cnt[i]; ++f) {
      rec.push_back(m2[f]);
      rec.push_back(static_cast<Real>(ec[f]));
      for (int e = eo[f]; e < eo[f] + ec[f]; ++e) {
        rec.push_back(static_cast<Real>(ef[e] - off[i]));
        for (int k = 0; k < 3; ++k)
          rec.push_back(eg[3 * e + k]);
        rec.push_back(el[e]);
      }
    }
  d.add(pre + "edgesByCell", up(rec, "rec"));
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

/// The resident topology, live slots only (the store is allocated without initialization, so
/// the slots past np / nt hold whatever was there).
void addStore(Digest& d, const peclet::voro::TopologyStore<kP, kT>& st) {
  const auto np = down(st.np), nt = down(st.nt);
  const auto pnbr = down(st.pnbr);
  const auto tri = down(st.tri);
  std::vector<int> p;
  std::vector<unsigned> t;
  for (std::size_t i = 0; i < np.size(); ++i) {
    for (int k = 0; k < np[i]; ++k)
      p.push_back(pnbr[i * kP + k]);
    for (int k = 0; k < nt[i]; ++k)
      t.push_back(tri[i * kT + k]);
  }
  d.add("store.np", st.np);
  d.add("store.nt", st.nt);
  d.add("store.pnbr", Kokkos::View<int*, Mem>(up<int>(p, "pnbr")));
  d.add("store.tri", Kokkos::View<unsigned*, Mem>(up<unsigned>(t, "tri")));
}

/// The resident engine's publish: store → reevalPublish → aux maps → face mesh.
template <bool W>
void addPublish(Digest& d, Engine<W>& mt, const Scene& s, const DV& pos) {
  auto view =
      peclet::voro::reevalPublish<Real, kP, kT>(mt.store, pos, mt.vol, s.N, s.L, mt.wall, mt.xRef);
  auto aux = peclet::voro::buildAuxMaps(view);
  auto mesh = fv::buildFaceMesh(view, aux);
  addView(d, "pub.", view);
  d.add("aux.recip", aux.recip);
  d.add("aux.cellOfFacet", aux.cellOfFacet);
  addMesh(d, mesh);
  addStore(d, mt.store);
}

template <bool W>
void coldHashes(const Scene& s, const char* tag) {
  Digest d;
  auto res = coldBuild<W>(s, s.pos);
  addView(d, "raw.", res.view);
  d.add("raw.status", res.status);
  addByCell(d, "raw.", res.view);
  {  // the opt-in facet-edge area-Jacobian CSR and facet moments of the cold build
    auto ag = peclet::voro::buildTessellation<Real, W, peclet::voro::NoSdf>(
        s.pos, W ? s.weight : DV(), s.N, s.L, peclet::voro::kSearchWindow, -1, {}, {}, true, -1, {},
        {}, {}, {}, {}, {}, 0, nullptr, {}, /*withAreaGrad=*/true, {}, {}, 0, Real(0), false,
        /*withMoments=*/true);
    addView(d, "rawAG.", ag.view);
    addByCell(d, "rawAG.", ag.view);
    addEdgesByCell(d, "rawAG.", ag.view);
    d.add("rawAG.facetEdgeOffset", ag.view.facetEdgeOffset);
    d.add("rawAG.facetEdgeCount", ag.view.facetEdgeCount);
    d.add("rawAG.edgeFacet", ag.view.edgeFacet);
    d.add("rawAG.edgeAreaGrad", ag.view.edgeAreaGrad);
    d.add("rawAG.edgeLength", ag.view.edgeLength);
    d.add("rawAG.facetMoment2", ag.view.facetMoment2);
  }
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
  double volRel = 0;     // |Σ V − L³| / L³
  long nonRecip = 0;     // facets i→j (|A| > 0) with no facet j→i
  double nonRecipA = 0;  // the largest such |A|, in h²
  long empty = 0;        // cells with status kEmpty
  long buried = 0;       // power seeds outside their own cell (kBuried)
  long emptyBad = 0;     // empty cells published with a volume or facets
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

/// One frame of the S0.2 motion: x += U0·dt (sin 2πy, sin 2πz, sin 2πx), U0·dt = 0.05h/√3,
/// wrapped into [0,1).
void advect(const Scene& s, const DV& pos) {
  const Real twoPi = Real(2) * Real(M_PI);
  const Real step = Real(0.05) * s.h / std::sqrt(Real(3));
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

template <bool W>
int moving(const Scene& s, const char* tag, int frames, int every, bool doGuard, bool detail,
           int dumpFrame = -1, const std::string& dumpPath = "") {
  Engine<W> mt;
  allocEngine<W>(mt, s);
  DV pos("pos", 3 * s.N);
  Kokkos::deep_copy(pos, s.pos);
  mt.rebuild(pos);
  int bad = 0;
  GuardStats worst;
  long totalRebuilt = 0;
  int fell = 0, gateRebuilds = 0;
  for (int k = 1; k <= frames; ++k) {
    advect(s, pos);
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
      if (detail)
        for (const auto& p : d.parts)
          std::printf("FRAME %s %d %s %s\n", tag, k, p.first.c_str(), p.second.c_str());
    }
    if (k == dumpFrame && !dumpPath.empty()) {  // positions then weights, raw doubles
      const auto hp = down(pos), hw = down(s.weight);
      if (FILE* fp = std::fopen(dumpPath.c_str(), "wb")) {
        std::fwrite(hp.data(), sizeof(Real), hp.size(), fp);
        std::fwrite(hw.data(), sizeof(Real), hw.size(), fp);
        std::fclose(fp);
      }
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

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

/// Wall times (ms) of the cold build (buildTessellation with force geometry), the engine's
/// rebuild, and per frame of the moving run its repair (`step`) and its publish (reevalPublish →
/// aux maps → face mesh): medians over `reps` builds and over frames 11..`frames`.
template <bool W>
void timing(const Scene& s, const char* tag, int reps, int frames) {
  Kokkos::Timer t;
  std::vector<double> cold, reb, rep, pub;
  for (int r = 0; r < reps + 2; ++r) {
    Kokkos::fence();
    t.reset();
    auto res = coldBuild<W>(s, s.pos);
    Kokkos::fence();
    if (r >= 2)
      cold.push_back(1e3 * t.seconds());
  }
  Engine<W> mt;
  allocEngine<W>(mt, s);
  for (int r = 0; r < reps + 2; ++r) {
    Kokkos::fence();
    t.reset();
    mt.rebuild(s.pos);
    Kokkos::fence();
    if (r >= 2)
      reb.push_back(1e3 * t.seconds());
  }
  DV pos("pos", 3 * s.N);
  Kokkos::deep_copy(pos, s.pos);
  mt.rebuild(pos);
  for (int k = 1; k <= frames; ++k) {
    advect(s, pos);
    Kokkos::fence();
    t.reset();
    mt.step(pos);
    Kokkos::fence();
    const double tr = 1e3 * t.seconds();
    t.reset();
    auto view = peclet::voro::reevalPublish<Real, kP, kT>(mt.store, pos, mt.vol, s.N, s.L, mt.wall,
                                                          mt.xRef);
    auto aux = peclet::voro::buildAuxMaps(view);
    auto mesh = fv::buildFaceMesh(view, aux);
    Kokkos::fence();
    if (k > 10) {
      rep.push_back(tr);
      pub.push_back(1e3 * t.seconds());
    }
  }
  std::printf("TIME %s n=%d threads=%d cold=%.3f rebuild=%.3f repair=%.3f publish=%.3f ms\n", tag,
              s.n, peclet::core::ExecSpace().concurrency(), median(cold), median(reb), median(rep),
              median(pub));
}

}  // namespace

int main(int argc, char** argv) {
  int n = 16, frames = 40, every = 1;
  bool doGuard = true, detail = false, timeIt = false;
  int dumpFrame = -1;
  std::string dumpPath;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--n") && a + 1 < argc)
      n = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--frames") && a + 1 < argc)
      frames = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--every") && a + 1 < argc)
      every = std::max(1, std::atoi(argv[++a]));
    else if (!std::strcmp(argv[a], "--no-guard"))
      doGuard = false;
    else if (!std::strcmp(argv[a], "--detail"))
      detail = true;
    else if (!std::strcmp(argv[a], "--time"))
      timeIt = true;
    else if (!std::strcmp(argv[a], "--dump") && a + 2 < argc) {
      dumpFrame = std::atoi(argv[++a]);
      dumpPath = argv[++a];
    }
  }
  int rc = 0;
  Kokkos::initialize(argc, argv);
  {
    std::printf("test_determinism: n=%d frames=%d threads=%d\n", n, frames,
                peclet::core::ExecSpace().concurrency());
    const Scene s = makeScene(n);
    if (timeIt) {
      timing<false>(s, "voronoi", 11, frames);
      timing<true>(s, "power", 11, frames);
    } else {
      coldHashes<false>(s, "voronoi");
      coldHashes<true>(s, "power");
      moving<false>(s, "voronoi", frames, every, false, detail);
      rc |= moving<true>(s, "power", frames, every, doGuard, detail, dumpFrame, dumpPath);
    }
  }
  Kokkos::finalize();
  return rc;
}
