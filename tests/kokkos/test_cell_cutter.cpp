/**
 * @file test_cell_cutter.cpp
 * \brief Phase-2 acceptance: device-callable cutter vs legacy topology.
 *
 * For a random seed battery, builds every Voronoi cell with the device
 * ScratchCell cutter (in a Kokkos parallel_for over cells, one cell per work
 * item) and checks against the legacy CellComplex:
 *   - per-cell volume matches to ~1e-9 (space-filling sum == box);
 *   - facet count matches; Euler 2F - V == 4;
 *   - the neighbour SET (which seeds share a face) matches exactly.
 * Topology may legitimately differ only at degeneracies; on general-position
 * random seeds it is identical.
 *
 * Neighbours are passed in seed-relative coordinates, minimal-imaged and sorted
 * by ascending rSqHalf on the host (the cutter's security early-out needs that
 * order — exactly the legacy CompareNbrDist sort). No physics.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/device/cell_cutter.hpp"
#include "vorflow/voronoi.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {

constexpr int MAXF = 128;

int buildAndCheck(int N, Vec3 L, unsigned seed) {
  const real_t boxVol = L[0] * L[1] * L[2];
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * U(rng);

  vor::Box<real_t> box(L);
  vor::CellComplex<real_t> complex(&box);
  complex.build(pos);
  std::vector<vor::Cell<real_t> > legacy;
  complex.materializeCells(legacy);
  const std::vector<vor::CellGeometry<real_t> >& geom = complex.getGeoms();

  // Host: per-cell minimal-image neighbour lists, sorted by ascending rSqHalf.
  std::vector<int> off(N + 1, 0);
  std::vector<real_t> rx, ry, rz;
  std::vector<int> nid;
  for (int i = 0; i < N; ++i) {
    std::vector<std::pair<real_t, int> > order;  // (rSqHalf, j)
    order.reserve(N - 1);
    std::vector<Vec3> rel(N);
    for (int j = 0; j < N; ++j) {
      if (j == i)
        continue;
      Vec3 d = {pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]};
      box.makeShortestDistance(d);
      rel[j] = d;
      order.emplace_back(0.5 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), j);
    }
    std::sort(order.begin(), order.end());
    for (auto& pr : order) {
      int j = pr.second;
      rx.push_back(rel[j][0]);
      ry.push_back(rel[j][1]);
      rz.push_back(rel[j][2]);
      nid.push_back(j);
    }
    off[i + 1] = (int)nid.size();
  }

  // Upload neighbour CSR to device.
  auto toDev = [](const std::vector<int>& h, const char* l) {
    Kokkos::View<int*, tpx::MemSpace> d(
        Kokkos::view_alloc(std::string(l), Kokkos::WithoutInitializing), h.size());
    auto hv = Kokkos::create_mirror_view(d);
    for (size_t k = 0; k < h.size(); ++k)
      hv(k) = h[k];
    Kokkos::deep_copy(d, hv);
    return d;
  };
  auto toDevR = [](const std::vector<real_t>& h, const char* l) {
    Kokkos::View<real_t*, tpx::MemSpace> d(
        Kokkos::view_alloc(std::string(l), Kokkos::WithoutInitializing), h.size());
    auto hv = Kokkos::create_mirror_view(d);
    for (size_t k = 0; k < h.size(); ++k)
      hv(k) = h[k];
    Kokkos::deep_copy(d, hv);
    return d;
  };
  auto dOff = toDev(off, "off");
  auto dRx = toDevR(rx, "rx");
  auto dRy = toDevR(ry, "ry");
  auto dRz = toDevR(rz, "rz");
  auto dId = toDev(nid, "id");

  Kokkos::View<real_t*, tpx::MemSpace> dVol("vol", N);
  Kokkos::View<int*, tpx::MemSpace> dNf("nf", N);
  Kokkos::View<int*, tpx::MemSpace> dNv("nv", N);
  Kokkos::View<int*, tpx::MemSpace> dStatus("status", N);
  Kokkos::View<int*, tpx::MemSpace> dNbr("nbr", (size_t)N * MAXF);

  const real_t Lx = L[0], Ly = L[1], Lz = L[2];
  Kokkos::parallel_for(
      "cutter.build", Kokkos::RangePolicy<tpx::ExecSpace>(0, N), KOKKOS_LAMBDA(const int i) {
        vor::device::ScratchCell<real_t> c;
        const real_t Larr[3] = {Lx, Ly, Lz};
        const int b = dOff(i), e = dOff(i + 1);
        vor::device::CutStatus st = vor::device::buildVoronoiCell<real_t, false>(
            c, Larr, &dRx(b), &dRy(b), &dRz(b), &dId(b), nullptr, e - b, real_t(0));
        dStatus(i) = (int)st;
        dVol(i) = c.volume();
        dNf(i) = c.countFacets();
        dNv(i) = c.countVertices();
        int k = 0;
        for (int f = 0; f < c.numAllocF; ++f)
          if (c.aliveF[f] && k < MAXF)
            dNbr((size_t)i * MAXF + k++) = c.fnbr[f];
        for (; k < MAXF; ++k)
          dNbr((size_t)i * MAXF + k) = -1;
      });
  Kokkos::fence();

  auto vol = Kokkos::create_mirror_view(dVol);
  auto nf = Kokkos::create_mirror_view(dNf);
  auto nv = Kokkos::create_mirror_view(dNv);
  auto status = Kokkos::create_mirror_view(dStatus);
  auto nbr = Kokkos::create_mirror_view(dNbr);
  Kokkos::deep_copy(vol, dVol);
  Kokkos::deep_copy(nf, dNf);
  Kokkos::deep_copy(nv, dNv);
  Kokkos::deep_copy(status, dStatus);
  Kokkos::deep_copy(nbr, dNbr);

  int fail = 0;
  double totalVol = 0;
  int volMis = 0, facetMis = 0, eulerMis = 0, setMis = 0, ovf = 0;
  for (int i = 0; i < N; ++i) {
    if (status(i) != 0)
      ++ovf;
    totalVol += vol(i);
    if (std::fabs(vol(i) - geom[i].getVolume()) / geom[i].getVolume() > 1e-9)
      ++volMis;
    if (nf(i) != legacy[i].numFacets())
      ++facetMis;
    if (2 * nf(i) - nv(i) != 4)
      ++eulerMis;
    // neighbour-set comparison (sorted, sentinels dropped).
    std::vector<int> dn, ln;
    for (int k = 0; k < MAXF; ++k)
      if (nbr((size_t)i * MAXF + k) >= 0)
        dn.push_back(nbr((size_t)i * MAXF + k));
    for (vor::uint1 f = 0; f < legacy[i].numFacets(); ++f) {
      vor::uint2 g = legacy[i].getNbr(f);
      if (g != vor::noNbr && g != vor::boundaryNbr)
        ln.push_back((int)g);
    }
    std::sort(dn.begin(), dn.end());
    std::sort(ln.begin(), ln.end());
    if (dn != ln)
      ++setMis;
  }
  double volErr = std::fabs(totalVol - boxVol) / boxVol;
  if (ovf) {
    std::fprintf(stderr, "  [N=%d] %d overflow cells\n", N, ovf);
    ++fail;
  }
  if (volMis) {
    std::fprintf(stderr, "  [N=%d] %d per-cell volume mismatches\n", N, volMis);
    ++fail;
  }
  if (facetMis) {
    std::fprintf(stderr, "  [N=%d] %d facet-count mismatches\n", N, facetMis);
    ++fail;
  }
  if (eulerMis) {
    std::fprintf(stderr, "  [N=%d] %d Euler violations\n", N, eulerMis);
    ++fail;
  }
  if (setMis) {
    std::fprintf(stderr, "  [N=%d] %d neighbour-set mismatches\n", N, setMis);
    ++fail;
  }
  if (volErr > 1e-9) {
    std::fprintf(stderr, "  [N=%d] space-filling err %.2e\n", N, volErr);
    ++fail;
  }
  std::printf("  [N=%d] volErr=%.2e volMis=%d facetMis=%d setMis=%d euler=%d ovf=%d %s\n", N,
              volErr, volMis, facetMis, setMis, eulerMis, ovf, fail == 0 ? "PASS" : "FAIL");
  return fail;
}

// Power (Laguerre) diagram: device Weighted cutter vs legacy PowerCellComplex.
// Small weight spread keeps every seed's power-cell non-empty (1:1 with seeds)
// while still shifting facets onto the radical plane, so the device radical
// offset is validated against the legacy weighted build.
int buildAndCheckPower(int N, Vec3 L, unsigned seed) {
  const real_t boxVol = L[0] * L[1] * L[2];
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  std::vector<real_t> w(N);
  for (int i = 0; i < N; ++i) {
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * U(rng);
    w[i] = 0.01 * U(rng);  // small spread => all cells present
  }

  vor::Box<real_t> box(L);
  vor::CellComplex<real_t, true> complex(&box);
  complex.setWeights(w);
  complex.build(pos);
  std::vector<vor::Cell<real_t> > legacy;
  complex.materializeCells(legacy);
  const std::vector<vor::CellGeometry<real_t> >& geom = complex.getGeoms();
  // Some seeds may have no power cell (buried under heavier neighbours); legacy
  // present cells tile the whole box. Map seed id -> legacy cell index (or -1).
  std::vector<int> idToLegacy(N, -1);
  for (int li = 0; li < (int)legacy.size(); ++li)
    idToLegacy[(int)legacy[li].getID()] = li;

  // Neighbour CSR carries the neighbour weight; no security early-out for Power.
  std::vector<int> off(N + 1, 0), nid;
  std::vector<real_t> rx, ry, rz, wn;
  for (int i = 0; i < N; ++i) {
    std::vector<std::pair<real_t, int> > order;
    std::vector<Vec3> rel(N);
    for (int j = 0; j < N; ++j) {
      if (j == i)
        continue;
      Vec3 d = {pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]};
      box.makeShortestDistance(d);
      rel[j] = d;
      order.emplace_back(0.5 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), j);
    }
    std::sort(order.begin(), order.end());
    for (auto& pr : order) {
      int j = pr.second;
      rx.push_back(rel[j][0]);
      ry.push_back(rel[j][1]);
      rz.push_back(rel[j][2]);
      nid.push_back(j);
      wn.push_back(w[j]);
    }
    off[i + 1] = (int)nid.size();
  }

  auto toI = [](const std::vector<int>& h, const char* l) {
    Kokkos::View<int*, tpx::MemSpace> d(
        Kokkos::view_alloc(std::string(l), Kokkos::WithoutInitializing), h.size());
    auto hv = Kokkos::create_mirror_view(d);
    for (size_t k = 0; k < h.size(); ++k)
      hv(k) = h[k];
    Kokkos::deep_copy(d, hv);
    return d;
  };
  auto toR = [](const std::vector<real_t>& h, const char* l) {
    Kokkos::View<real_t*, tpx::MemSpace> d(
        Kokkos::view_alloc(std::string(l), Kokkos::WithoutInitializing), h.size());
    auto hv = Kokkos::create_mirror_view(d);
    for (size_t k = 0; k < h.size(); ++k)
      hv(k) = h[k];
    Kokkos::deep_copy(d, hv);
    return d;
  };
  auto dOff = toI(off, "off");
  auto dRx = toR(rx, "rx"), dRy = toR(ry, "ry"), dRz = toR(rz, "rz"), dWn = toR(wn, "wn");
  auto dId = toI(nid, "id");
  auto dWself = toR(w, "wself");

  Kokkos::View<real_t*, tpx::MemSpace> dVol("vol", N);
  Kokkos::View<int*, tpx::MemSpace> dNf("nf", N), dStatus("st", N);
  Kokkos::View<int*, tpx::MemSpace> dNbr("nbr", (size_t)N * MAXF);
  const real_t Lx = L[0], Ly = L[1], Lz = L[2];
  Kokkos::parallel_for(
      "cutter.power", Kokkos::RangePolicy<tpx::ExecSpace>(0, N), KOKKOS_LAMBDA(const int i) {
        vor::device::ScratchCell<real_t> c;
        const real_t Larr[3] = {Lx, Ly, Lz};
        const int b = dOff(i), e = dOff(i + 1);
        vor::device::CutStatus st = vor::device::buildVoronoiCell<real_t, true>(
            c, Larr, &dRx(b), &dRy(b), &dRz(b), &dId(b), &dWn(b), e - b, dWself(i));
        dStatus(i) = (int)st;
        dVol(i) = c.volume();
        dNf(i) = c.countFacets();
        int k = 0;
        for (int f = 0; f < c.numAllocF; ++f)
          if (c.aliveF[f] && k < MAXF)
            dNbr((size_t)i * MAXF + k++) = c.fnbr[f];
        for (; k < MAXF; ++k)
          dNbr((size_t)i * MAXF + k) = -1;
      });
  Kokkos::fence();

  auto vol = Kokkos::create_mirror_view(dVol);
  auto nf = Kokkos::create_mirror_view(dNf);
  auto nbr = Kokkos::create_mirror_view(dNbr);
  auto dStatusHost = Kokkos::create_mirror_view(dStatus);
  Kokkos::deep_copy(vol, dVol);
  Kokkos::deep_copy(nf, dNf);
  Kokkos::deep_copy(nbr, dNbr);
  Kokkos::deep_copy(dStatusHost, dStatus);

  int fail = 0, volMis = 0, facetMis = 0, setMis = 0, emptyMis = 0;
  double totalVol = 0;           // over legacy-present seeds; these tile the box
  for (int i = 0; i < N; ++i) {  // i == seed id
    const int li = idToLegacy[i];
    if (li < 0) {
      // Buried seed: legacy has no cell here; the device should agree (empty).
      if (dStatusHost(i) == 0 && vol(i) > 1e-12)
        ++emptyMis;
      continue;
    }
    if (dStatusHost(i) != 0) {  // legacy has a cell, device must too
      ++emptyMis;
      continue;
    }
    totalVol += vol(i);
    if (std::fabs(vol(i) - geom[li].getVolume()) / geom[li].getVolume() > 1e-9)
      ++volMis;
    if (nf(i) != legacy[li].numFacets())
      ++facetMis;
    std::vector<int> dn, ln;
    for (int k = 0; k < MAXF; ++k)
      if (nbr((size_t)i * MAXF + k) >= 0)
        dn.push_back(nbr((size_t)i * MAXF + k));
    for (vor::uint1 f = 0; f < legacy[li].numFacets(); ++f) {
      vor::uint2 g = legacy[li].getNbr(f);
      if (g != vor::noNbr && g != vor::boundaryNbr)
        ln.push_back((int)g);
    }
    std::sort(dn.begin(), dn.end());
    std::sort(ln.begin(), ln.end());
    if (dn != ln)
      ++setMis;
  }
  double volErr = std::fabs(totalVol - boxVol) / boxVol;
  if (volMis || facetMis || setMis || emptyMis || volErr > 1e-9) {
    std::fprintf(stderr, "  [power N=%d] volMis=%d facetMis=%d setMis=%d emptyMis=%d volErr=%.2e\n",
                 N, volMis, facetMis, setMis, emptyMis, volErr);
    ++fail;
  }
  std::printf(
      "  [power N=%d] cells=%zu volErr=%.2e volMis=%d facetMis=%d setMis=%d emptyMis=%d %s\n", N,
      legacy.size(), volErr, volMis, facetMis, setMis, emptyMis, fail == 0 ? "PASS" : "FAIL");
  return fail;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    failures += buildAndCheck(500, {1.0, 1.0, 1.0}, 1);
    failures += buildAndCheck(1000, {1.0, 1.0, 1.0}, 7);
    failures += buildAndCheck(800, {2.0, 1.0, 0.5}, 13);
    failures += buildAndCheckPower(500, {1.0, 1.0, 1.0}, 21);
    failures += buildAndCheckPower(800, {1.0, 1.0, 1.0}, 29);
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "cell_cutter PASS" : "cell_cutter FAIL");
  return failures;
}
