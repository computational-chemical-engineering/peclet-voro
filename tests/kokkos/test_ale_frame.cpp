/**
 * @file test_ale_frame.cpp
 * \brief Gates V0.1, G0.2 and F of the moving-grid work order S0.2 (docs/ale_design.md §7
 * WO-S0.2): the frame pipeline `ale::initialFrame` / `ale::advanceFrame` (repair → publish → face
 * mesh), the forced worklist `MovingTessellation::step(pos, forced, n)` and the changed set
 * `changedMask()`.
 *
 * Seeds: an n³ periodic lattice jittered ±0.3h, power weights uniform in [0, 0.15h²] (fixed seed).
 *  (a) V0.1 (a): Voronoi and power engines; initialFrame, then `frames` × advanceFrame at unchanged
 *      positions. Every FaceMesh array of every frame bitwise equal to frame 0; changedMask all
 *      zero in every frame.
 *  (b) V0.1 (b): CollocatedNS on the n³ Voronoi mesh of a lattice jittered ±0.2h, TGV start,
 *      `steps` steps at dt = 0.2h in the (B) configuration of test_collocated_ns (RK3, ν = 0,
 *      AMG, Poisson tol 1e-13) and the (D) one (implicit diffusion, ν = 0.01, tol 1e-11). Run 1
 *      keeps the mesh of initialFrame; run 2 replaces it before every step by the mesh of
 *      advanceFrame at unchanged positions (ns.m = mesh; poisson.setup; viscDt = -1). U, p, uf
 *      bitwise equal after the last step.
 *  (f) F: power engine at rest, step(pos, forced, 50 random slots): changedMask ⊇ the 50; the
 *      result equals a cold build (neighbour sets exact, volumes ≤ 1e-13 relative); step(pos)
 *      afterwards changes nothing.
 *  (g) G0.2: power engine; every generator advected by u = U0 (sin 2πy/L, sin 2πz/L, sin 2πx/L)
 *      with explicit Euler, U0·dt = 0.05h/√3 (|Δx| ≤ 0.05h), wrapped into [0, L); `frames`
 *      frames, with reciprocity enforcement off and on: (i) |ΣV − L³|/L³ ≤ 1e-13 every frame;
 *      (ii) total rebuilt cells ≥ 1000 (with --full; > 0 otherwise); (iii) max non-reciprocal
 *      facet area per frame (off: recorded; on: ≤ 1e-12 h² every frame); (iv) at every
 *      frames/5-th frame the repaired state equals a cold build (neighbour sets exact, volumes ≤
 *      1e-9 relative). `--dump DIR` writes the last frame's FaceMesh arrays for (v)
 *      (tools/ale_gates/G0_2.sh compares them across OMP 1, 8, 8).
 * Builder options: `--opts default` (all off), `frame` (ale::frameMeshOptions), `det`
 * (deterministicOrder only); `auto` (the default) = default at one thread, else frame for (a)
 * and det for (b). (g) always uses the frame options.
 * Usage: test_ale_frame [n=16] [--part a|b|f|g|all] [--full] [--opts auto|default|frame|det]
 *                       [--dump DIR] [--time]
 * Without --full (the ctest): (a) 20 frames, (b) 20 steps, (g) 50 frames.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "fv_test_util.hpp"
#include "peclet/voro/ale/frame.hpp"
#include "peclet/voro/dynamic_validate.hpp"
#include "peclet/voro/fv/collocated.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"

using Exec = peclet::core::ExecSpace;
constexpr int kP = peclet::voro::kMaxPlanes;
constexpr int kT = peclet::voro::kMaxTriangles;
template <bool W>
using Engine = peclet::voro::MovingTessellation<Real, kP, kT, W>;
using FM = fv::FaceMesh<Real>;
using FMO = fv::FaceMeshOptions<Real>;
using Frame = peclet::voro::ale::MeshFrame<Real>;
using FrameOpts = peclet::voro::ale::FrameOptions<Real>;
namespace ale = peclet::voro::ale;

namespace {

struct Scene {
  int n = 0, N = 0;
  Real L[3] = {1, 1, 1};
  Real h = 0;
  DV pos, weight;
};

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
void allocEngine(Engine<W>& mt, int N, const Real L[3], Real h, const DV& weight) {
  mt.alloc(N, L, Real(1e-4) * h, Real(0.25) * h, 4, N);
  if constexpr (W)
    mt.setWeights(weight);
}

DV copyOf(const DV& v) {
  DV c("copy", v.extent(0));
  Kokkos::deep_copy(c, v);
  return c;
}

// ---- FaceMesh arrays as raw bytes
// ----------------------------------------------------------------
struct Arr {
  std::string name;
  std::size_t elem = 1;
  std::vector<char> bytes;
};

template <class V>
Arr arrOf(const char* name, const V& v) {
  using T = typename V::non_const_value_type;
  Arr a;
  a.name = name;
  a.elem = sizeof(T);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  const char* p = reinterpret_cast<const char*>(hv.data());
  a.bytes.assign(p, p + v.extent(0) * sizeof(T));
  return a;
}

std::vector<Arr> meshArrays(const FM& m) {
  std::vector<Arr> out;
  Arr c;
  c.name = "counts";
  c.elem = sizeof(long);
  const long counts[6] = {m.nCells,    m.nFaces,   m.nInterior,
                          m.nCombined, m.nDropped, m.generatorDiag.numNegativeHeight};
  const Real sine = m.generatorDiag.maxNormalConnSine;
  c.bytes.assign(reinterpret_cast<const char*>(counts),
                 reinterpret_cast<const char*>(counts) + sizeof(counts));
  c.bytes.insert(c.bytes.end(), reinterpret_cast<const char*>(&sine),
                 reinterpret_cast<const char*>(&sine) + sizeof(Real));
  out.push_back(std::move(c));
  out.push_back(arrOf("faceCellA", m.faceCellA));
  out.push_back(arrOf("faceCellB", m.faceCellB));
  out.push_back(arrOf("faceFacet", m.faceFacet));
  out.push_back(arrOf("faceArea", m.faceArea));
  out.push_back(arrOf("faceNormal", m.faceNormal));
  out.push_back(arrOf("faceDist", m.faceDist));
  out.push_back(arrOf("faceHa", m.faceHa));
  out.push_back(arrOf("faceHb", m.faceHb));
  out.push_back(arrOf("faceCentroid", m.faceCentroid));
  out.push_back(arrOf("faceConn", m.faceConn));
  out.push_back(arrOf("cellFaceOffset", m.cellFaceOffset));
  out.push_back(arrOf("cellFace", m.cellFace));
  out.push_back(arrOf("cellFaceSign", m.cellFaceSign));
  out.push_back(arrOf("cellVolume", m.cellVolume));
  out.push_back(arrOf("cellCentroid", m.cellCentroid));
  return out;
}

/// First difference between two array sets: "" when bitwise equal, else "<array>[<index>]".
std::string firstDiff(const std::vector<Arr>& a, const std::vector<Arr>& b) {
  for (std::size_t k = 0; k < a.size(); ++k) {
    const Arr &x = a[k], &y = b[k];
    if (x.bytes == y.bytes)
      continue;
    const std::size_t n = std::min(x.bytes.size(), y.bytes.size());
    std::size_t i = 0;
    while (i < n && x.bytes[i] == y.bytes[i])
      ++i;
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s[%zu] (sizes %zu/%zu)", x.name.c_str(), i / x.elem,
                  x.bytes.size() / x.elem, y.bytes.size() / y.elem);
    return buf;
  }
  return "";
}

void writeFile(const std::string& path, const std::vector<char>& bytes) {
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(2);
  }
  if (!bytes.empty())
    std::fwrite(bytes.data(), 1, bytes.size(), fp);
  std::fclose(fp);
}

void dumpArrays(const std::string& dir, const std::string& tag, const std::vector<Arr>& arrs) {
  for (const Arr& a : arrs)
    writeFile(dir + "/" + tag + "." + a.name, a.bytes);
}

int nonzero(const Kokkos::View<int*, Mem>& m, int n) {
  int c = 0;
  Kokkos::parallel_reduce(
      "t.nz", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int i, int& a) { a += m(i) ? 1 : 0; }, c);
  return c;
}

FMO optionsOf(const std::string& o) {
  if (o == "frame")
    return ale::frameMeshOptions<Real>();
  FMO m;
  if (o == "det")
    m.deterministicOrder = true;
  return m;  // "default": all off
}

// ---- (a) V0.1 (a)
// --------------------------------------------------------------------------------
template <bool W>
int partA(const char* tag, const Scene& s, int frames, const std::string& opts) {
  Engine<W> mt;
  allocEngine<W>(mt, s.N, s.L, s.h, s.weight);
  FrameOpts fo;
  fo.mesh = optionsOf(opts);
  const DV w = W ? s.weight : DV();
  const Frame f0 = ale::initialFrame(mt, s.pos, w, s.L, fo);
  const auto ref = meshArrays(f0.mesh);
  int badFrames = 0, changedFrames = 0, rounds = f0.reciprocity.rounds;
  std::string first;
  for (int k = 1; k <= frames; ++k) {
    const Frame f = ale::advanceFrame(mt, s.pos, w, s.L, fo);
    const std::string d = firstDiff(ref, meshArrays(f.mesh));
    if (!d.empty()) {
      if (badFrames == 0)
        first = "frame " + std::to_string(k) + " " + d;
      ++badFrames;
    }
    const int nz = nonzero(mt.changedMask(), s.N);
    if (nz != 0 || f.numRebuilt != 0 || f.coldBuild)
      ++changedFrames;
    rounds += f.reciprocity.rounds;
  }
  const bool pass = badFrames == 0 && changedFrames == 0;
  std::printf(
      "RESULT a %-7s n=%d opts=%s frames=%d %s arrays=%zu mismatchFrames=%d changedFrames=%d "
      "rounds=%d nDropped=%d%s%s\n",
      tag, s.n, opts.c_str(), frames, pass ? "PASS" : "FAIL", ref.size(), badFrames, changedFrames,
      rounds, f0.mesh.nDropped, first.empty() ? "" : " first=", first.c_str());
  return pass ? 0 : 1;
}

// ---- (b) V0.1 (b)
// --------------------------------------------------------------------------------
struct NsConfig {
  const char* name;
  Real nu;
  bool implicitDiffusion;
  Real tol;
};

int partB(int n, int steps, const std::string& opts, const std::string& dumpDir) {
  const Real L[3] = {1, 1, 1};
  const int N = n * n * n;
  const Real h = Real(1) / n, dt = Real(0.2) * h;
  std::mt19937 rng(17);
  const std::vector<Real> pos = lattice(n, 0.2, rng);
  const DV dpos = up(pos, "pos");
  FrameOpts fo;
  fo.mesh = optionsOf(opts);
  const NsConfig cfgs[2] = {{"B_rk3", 0.0, false, 1e-13}, {"D_implicit", 0.01, true, 1e-11}};
  int rc = 0;
  for (const NsConfig& c : cfgs) {
    Engine<false> mt;
    allocEngine<false>(mt, N, L, h, DV());
    const Frame f0 = ale::initialFrame(mt, dpos, DV(), L, fo);
    auto configure = [&](fv::CollocatedNS<Real>& ns) {
      ns.setup(f0.mesh, c.nu, true);
      ns.implicitDiffusion = c.implicitDiffusion;
      ns.poisson.tol = c.tol;
      ns.initialize(up(tgvCell(pos, N, 0, c.nu), "U0"));
    };
    fv::CollocatedNS<Real> ns1, ns2;
    configure(ns1);
    configure(ns2);
    for (int s = 0; s < steps; ++s)
      ns1.step(dt);
    int changed = 0, meshDiff = 0;
    const auto ref = meshArrays(f0.mesh);
    for (int s = 0; s < steps; ++s) {
      const Frame f = ale::advanceFrame(mt, dpos, DV(), L, fo);
      changed += f.numRebuilt;
      if (!firstDiff(ref, meshArrays(f.mesh)).empty())
        ++meshDiff;
      ns2.m = f.mesh;
      ns2.poisson.setup(ns2.m, true);
      ns2.viscDt = -1;
      ns2.step(dt);
    }
    std::vector<Arr> r1 = {arrOf("U", ns1.U), arrOf("p", ns1.p), arrOf("uf", ns1.uf)};
    std::vector<Arr> r2 = {arrOf("U", ns2.U), arrOf("p", ns2.p), arrOf("uf", ns2.uf)};
    const std::string d = firstDiff(r1, r2);
    const bool pass = d.empty();
    std::printf("RESULT b %-10s n=%d opts=%s steps=%d %s E=%.15e meshDiffSteps=%d rebuilt=%d%s%s\n",
                c.name, n, opts.c_str(), steps, pass ? "PASS" : "FAIL", (double)ns2.kineticEnergy(),
                meshDiff, changed, pass ? "" : " first=", d.c_str());
    if (!dumpDir.empty()) {
      dumpArrays(dumpDir, std::string("b_") + c.name + "_run1", r1);
      dumpArrays(dumpDir, std::string("b_") + c.name + "_run2", r2);
    }
    rc |= pass ? 0 : 1;
  }
  return rc;
}

// ---- cold-build comparison
// -------------------------------------------------------------------------
template <bool W>
peclet::voro::TessellatorResult<Real> coldBuild(const DV& pos, const DV& weight, int N,
                                                const Real L[3]) {
  Kokkos::View<long*, Mem> gd;
  return peclet::voro::buildTessellation<Real, W, peclet::voro::NoSdf>(pos, W ? weight : DV(), N, L,
                                                                       4, N, gd, {}, false);
}

template <bool W>
peclet::voro::OracleDiff againstCold(Engine<W>& mt, const peclet::voro::TessellationView<Real>& v) {
  peclet::voro::OracleDiff d;
  peclet::voro::compareVolumes(mt.vol, v, d);
  peclet::voro::compareNeighbours(mt.store.np, mt.store.nt, mt.store.pnbr, mt.store.tri, kP, kT, v,
                                  d);
  return d;
}

/// Localize a neighbour-set difference (host): per differing cell, the cold-build-only neighbours
/// with the cold facet area and its reverse facet, and the repair-only neighbours with the area of
/// their face in the frame's mesh (areas in units of h²; -1 = absent).
template <bool W>
void explainDiff(const char* tag, int k, Engine<W>& mt,
                 const peclet::voro::TessellationView<Real>& v, const FM& mesh, int N, Real h,
                 int maxCells = 12) {
  auto off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.cellFacetOffset);
  auto cnt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.cellFacetCount);
  auto nbr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.facetNeighbor);
  auto area = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.facetArea);
  auto np = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mt.store.np);
  auto nt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mt.store.nt);
  auto pn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mt.store.pnbr);
  auto tri = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mt.store.tri);
  auto fA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.faceCellA);
  auto fB = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.faceCellB);
  auto fAr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.faceArea);
  const double h2 = (double)h * h;
  auto coldArea = [&](int i, long j) {
    for (int g = off(i); g < off(i) + cnt(i); ++g)
      if ((long)nbr(g) == j)
        return std::sqrt(area(3 * g) * area(3 * g) + area(3 * g + 1) * area(3 * g + 1) +
                         area(3 * g + 2) * area(3 * g + 2)) /
               h2;
    return -1.0;
  };
  auto meshArea = [&](int i, int j) {
    for (int f = 0; f < mesh.nInterior; ++f)
      if ((fA(f) == i && fB(f) == j) || (fA(f) == j && fB(f) == i))
        return (double)fAr(f) / h2;
    return -1.0;
  };
  auto storeFaces = [&](int i) {
    std::vector<int> out;
    for (int p = 6; p < np(i); ++p) {
      const int j = pn((size_t)i * kP + p);
      if (j < 0)
        continue;
      int c = 0;
      for (int t = 0; t < nt(i); ++t) {
        const unsigned w = tri((size_t)i * kT + t);
        if (!((w >> 24) & 1u))
          continue;
        if ((int)(w & 0xffu) == p || (int)((w >> 8) & 0xffu) == p || (int)((w >> 16) & 0xffu) == p)
          ++c;
      }
      if (c >= 3)
        out.push_back(j);
    }
    return out;
  };
  int shown = 0;
  for (int i = 0; i < N && shown < maxCells; ++i) {
    const std::vector<int> st = storeFaces(i);
    std::vector<long> cold;
    for (int g = off(i); g < off(i) + cnt(i); ++g)
      if ((long)nbr(g) >= 0)
        cold.push_back((long)nbr(g));
    std::string msg;
    for (long j : cold)
      if (std::find(st.begin(), st.end(), (int)j) == st.end()) {
        char b[200];
        std::snprintf(b, sizeof b, " coldOnly j=%ld A_ij/h2=%.3e A_ji/h2=%.3e meshA/h2=%.3e", j,
                      coldArea(i, j), coldArea((int)j, i), meshArea(i, (int)j));
        msg += b;
      }
    for (int j : st)
      if (std::find(cold.begin(), cold.end(), (long)j) == cold.end()) {
        char b[200];
        std::snprintf(b, sizeof b, " repairOnly j=%d meshA/h2=%.3e coldA_ji/h2=%.3e", j,
                      meshArea(i, j), coldArea(j, i));
        msg += b;
      }
    if (!msg.empty()) {
      std::printf("DIFF %s k=%d cell=%d%s\n", tag, k, i, msg.c_str());
      ++shown;
    }
  }
}

// ---- (f) forced worklist
// ---------------------------------------------------------------------------
int partF(const Scene& s) {
  Engine<true> mt;
  allocEngine<true>(mt, s.N, s.L, s.h, s.weight);
  mt.rebuild(s.pos);
  constexpr int kForced = 50;
  std::vector<int> slots(s.N);
  std::iota(slots.begin(), slots.end(), 0);
  std::mt19937 rng(5u);
  std::shuffle(slots.begin(), slots.end(), rng);
  slots.resize(kForced);
  Kokkos::View<int*, Mem> forced("forced", kForced);
  Kokkos::deep_copy(forced, Kokkos::View<const int*, Kokkos::HostSpace>(slots.data(), kForced));
  const auto st = mt.step(s.pos, forced, kForced);
  const bool cold = st.route == peclet::voro::RepairStats::kRebuildGate || st.fellBack;
  auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mt.changedMask());
  int covered = 0, total = 0;
  for (int j : slots)
    covered += hm(j) ? 1 : 0;
  for (int i = 0; i < s.N; ++i)
    total += hm(i) ? 1 : 0;
  const auto cold0 = coldBuild<true>(s.pos, s.weight, s.N, s.L);
  const auto d = againstCold<true>(mt, cold0.view);
  const auto st2 = mt.step(s.pos);
  const int after = nonzero(mt.changedMask(), s.N);
  const bool cold2 = st2.route == peclet::voro::RepairStats::kRebuildGate || st2.fellBack;
  const bool pass = !cold && covered == kForced && d.changedNbrFrac == 0 && d.missedNbr == 0 &&
                    d.maxVolRelErr <= 1e-13 && after == 0 && !cold2;
  std::printf(
      "RESULT f n=%d %s forced=%d covered=%d changed=%d route=%d fellBack=%d nbrDiffFrac=%.3e "
      "missedNbr=%ld maxVolRel=%.3e afterChanged=%d afterRoute=%d\n",
      s.n, pass ? "PASS" : "FAIL", kForced, covered, total, (int)st.route, (int)st.fellBack,
      d.changedNbrFrac, d.missedNbr, d.maxVolRelErr, after, (int)st2.route);
  return pass ? 0 : 1;
}

// ---- (g) G0.2
// ----------------------------------------------------------------------------------------
void advect(const DV& pos, int N, Real L, Real step) {
  const Real k = Real(6.283185307179586) / L;  // 2π/L
  Kokkos::parallel_for(
      "t.advect", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        const Real x = pos(3 * i), y = pos(3 * i + 1), z = pos(3 * i + 2);
        Real q[3] = {x + step * std::sin(k * y), y + step * std::sin(k * z),
                     z + step * std::sin(k * x)};
        for (int d = 0; d < 3; ++d) {
          Real p = q[d];
          if (p >= L)
            p -= L;
          if (p < 0) {
            p += L;
            if (p >= L)
              p = 0;
          }
          pos(3 * i + d) = p;
        }
      });
}

double sumVolume(const FM& m) {
  double s = 0;
  auto V = m.cellVolume;
  Kokkos::parallel_reduce(
      "t.sumV", Kokkos::RangePolicy<Exec>(0, m.nCells),
      KOKKOS_LAMBDA(const int i, double& a) { a += (double)V(i); }, s);
  return s;
}

int partG(const Scene& s, int frames, bool full, bool timeIt, const std::string& dumpDir) {
  int rc = 0;
  const int every = std::max(1, frames / 5);
  const double h2 = (double)s.h * s.h;
  const double boxVol = (double)s.L[0] * s.L[1] * s.L[2];
  const Real step = Real(0.05) * s.h / std::sqrt(Real(3));
  for (int enforce = 0; enforce < 2; ++enforce) {
    Engine<true> mt;
    allocEngine<true>(mt, s.N, s.L, s.h, s.weight);
    DV pos = copyOf(s.pos);
    FrameOpts fo;
    fo.enforceReciprocity = enforce == 1;
    fo.timed = timeIt;
    Frame f = ale::initialFrame(mt, pos, s.weight, s.L, fo);
    double maxSumRel = 0, maxNonRecip = 0, sumT[5] = {0, 0, 0, 0, 0}, wall = 0;
    long totalRebuilt = 0, maxNonRecipCount = 0, negTotal = 0, nbrDiffCells = 0;
    int maxRebuilt = 0, framesNonRecip = 0, framesOver = 0, roundsTotal = 0, roundsMax = 0;
    int routes[3] = {0, 0, 0}, fellBack = 0, negMax = 0, checks = 0, checksBad = 0;
    double maxVolRel = 0;
    for (int k = 1; k <= frames; ++k) {
      advect(pos, s.N, s.L[0], step);
      const auto t0 = std::chrono::steady_clock::now();
      f = ale::advanceFrame(mt, pos, s.weight, s.L, fo);
      if (timeIt) {
        Kokkos::fence();
        wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        sumT[0] += f.timings.repair;
        sumT[1] += f.timings.publish;
        sumT[2] += f.timings.auxMaps;
        sumT[3] += f.timings.faceMesh;
        sumT[4] += f.timings.reciprocity;
      }
      const double sumRel = std::abs(sumVolume(f.mesh) - boxVol) / boxVol;
      maxSumRel = std::max(maxSumRel, sumRel);
      totalRebuilt += f.numRebuilt;
      maxRebuilt = std::max(maxRebuilt, f.numRebuilt);
      const auto& r = f.reciprocity;
      if (r.nNonRecip > 0)
        ++framesNonRecip;
      if (r.maxNonRecipArea > r.areaThreshold)
        ++framesOver;
      maxNonRecip = std::max(maxNonRecip, r.maxNonRecipArea);
      maxNonRecipCount = std::max(maxNonRecipCount, r.nNonRecip);
      roundsTotal += r.rounds;
      roundsMax = std::max(roundsMax, r.rounds);
      routes[(int)f.stats.route] += 1;
      fellBack += f.stats.fellBack ? 1 : 0;
      const int neg = f.mesh.generatorDiag.numNegativeHeight;
      negTotal += neg;
      negMax = std::max(negMax, neg);
      std::printf(
          "FRAME enforce=%d k=%d rebuilt=%d cold=%d route=%d fellBack=%d nNonRecip=%ld "
          "nDropped=%d maxNonRecipArea/h2=%.3e rounds=%d negH=%d sumVrel=%.2e\n",
          enforce, k, f.numRebuilt, (int)f.coldBuild, (int)f.stats.route, (int)f.stats.fellBack,
          r.nNonRecip, r.nDropped, r.maxNonRecipArea / h2, r.rounds, neg, sumRel);
      if (k % every == 0) {
        const auto cold = coldBuild<true>(pos, s.weight, s.N, s.L);
        const auto d = againstCold<true>(mt, cold.view);
        ++checks;
        const long diffCells = std::lround(d.changedNbrFrac * s.N);
        nbrDiffCells += diffCells;
        maxVolRel = std::max(maxVolRel, d.maxVolRelErr);
        const bool ok = diffCells == 0 && d.missedNbr == 0 && d.maxVolRelErr <= 1e-9;
        checksBad += ok ? 0 : 1;
        std::printf("CHECK enforce=%d k=%d %s nbrDiffCells=%ld missedNbr=%ld maxVolRel=%.3e\n",
                    enforce, k, ok ? "PASS" : "FAIL", diffCells, d.missedNbr, d.maxVolRelErr);
        {
          // localization: the cold build's volume sum, buried (emptied) power seeds, empty cells
          const auto cst = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cold.status);
          auto cv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cold.view.cellVolume);
          auto mv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mt.vol);
          double cs = 0;
          int buried = 0, emptyCold = 0, emptyRep = 0;
          for (int i = 0; i < s.N; ++i) {
            cs += cv(i);
            buried += (cst(i) & peclet::voro::kBuried) ? 1 : 0;
            emptyCold += cv(i) <= 0 ? 1 : 0;
            emptyRep += mv(i) <= 0 ? 1 : 0;
          }
          std::printf(
              "PROBE enforce=%d k=%d coldSumVrel=%.3e buried=%d emptyCold=%d emptyRepair=%d\n",
              enforce, k, std::abs(cs - boxVol) / boxVol, buried, emptyCold, emptyRep);
        }
        if (!ok)
          explainDiff<true>(enforce ? "enforce=1" : "enforce=0", k, mt, cold.view, f.mesh, s.N,
                            s.h);
      }
    }
    const bool okI = maxSumRel <= 1e-13;
    const bool okII = full ? totalRebuilt >= 1000 : totalRebuilt > 0;
    const bool okIII = enforce == 0 || framesOver == 0;
    const bool okIV = checksBad == 0 && checks > 0;
    const bool pass = okI && okII && okIII && okIV;
    rc |= pass ? 0 : 1;
    std::printf(
        "RESULT g enforce=%d n=%d frames=%d %s i=%s maxSumVrel=%.2e ii=%s totalRebuilt=%ld "
        "meanRebuilt=%.1f maxRebuilt=%d iii=%s maxNonRecipArea/h2=%.3e framesNonRecip=%d "
        "framesOverThreshold=%d maxNonRecipCount=%ld roundsTotal=%d roundsMax=%d iv=%s "
        "checks=%d maxVolRel=%.2e nbrDiffCells=%ld routes(two/dil/gate)=%d/%d/%d fellBack=%d "
        "negH(total/maxPerFrame)=%ld/%d\n",
        enforce, s.n, frames, pass ? "PASS" : "FAIL", okI ? "PASS" : "FAIL", maxSumRel,
        okII ? "PASS" : "FAIL", totalRebuilt, (double)totalRebuilt / frames, maxRebuilt,
        enforce == 0 ? "INFO" : (okIII ? "PASS" : "FAIL"), maxNonRecip / h2, framesNonRecip,
        framesOver, maxNonRecipCount, roundsTotal, roundsMax, okIV ? "PASS" : "FAIL", checks,
        maxVolRel, nbrDiffCells, routes[0], routes[1], routes[2], fellBack, negTotal, negMax);
    if (timeIt)
      std::printf(
          "TIME g enforce=%d threads=%d per-frame ms: repair %.2f publish %.2f auxMaps %.2f "
          "faceMesh %.2f reciprocity %.2f total %.2f\n",
          enforce, Exec().concurrency(), 1e3 * sumT[0] / frames, 1e3 * sumT[1] / frames,
          1e3 * sumT[2] / frames, 1e3 * sumT[3] / frames, 1e3 * sumT[4] / frames,
          1e3 * wall / frames);
    if (!dumpDir.empty())
      dumpArrays(dumpDir, "g_enforce" + std::to_string(enforce), meshArrays(f.mesh));
  }
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int rc = 0;
  {
    int n = 16;
    std::string part = "all", opts = "auto", dumpDir;
    bool full = false, timeIt = false;
    int gframes = 0;
    for (int a = 1; a < argc; ++a) {
      const std::string s = argv[a];
      if (s == "--part" && a + 1 < argc)
        part = argv[++a];
      else if (s == "--opts" && a + 1 < argc)
        opts = argv[++a];
      else if (s == "--dump" && a + 1 < argc)
        dumpDir = argv[++a];
      else if (s == "--gframes" && a + 1 < argc)
        gframes = std::atoi(argv[++a]);
      else if (s == "--full")
        full = true;
      else if (s == "--time")
        timeIt = true;
      else if (s.rfind("--", 0) != 0)
        n = std::atoi(s.c_str());
    }
    const bool serial = Exec().concurrency() == 1;
    const std::string optsA = opts == "auto" ? (serial ? "default" : "frame") : opts;
    const std::string optsB = opts == "auto" ? (serial ? "default" : "det") : opts;
    std::printf("=== S0.2 frame pipeline: n=%d part=%s threads=%d %s ===\n", n, part.c_str(),
                Exec().concurrency(), full ? "full" : "short");
    const bool all = part == "all";
    if (all || part == "a" || part == "f" || part == "g") {
      const Scene s = makeScene(n);
      if (all || part == "a") {
        rc |= partA<false>("voronoi", s, full ? 100 : 20, optsA);
        rc |= partA<true>("power", s, full ? 100 : 20, optsA);
      }
      if (all || part == "f")
        rc |= partF(s);
      if (all || part == "g")
        rc |= partG(s, gframes > 0 ? gframes : (full ? 500 : 50), full, timeIt, dumpDir);
    }
    if (all || part == "b")
      rc |= partB(n, full ? 100 : 20, optsB, dumpDir);
    std::printf("=== S0.2 %s ===\n", rc ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return rc;
}
