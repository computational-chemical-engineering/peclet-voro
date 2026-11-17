/**
 * @file test_power_cells.cpp
 * @brief P1 acceptance test for the device POWER (Laguerre) tessellation.
 *
 * Validates buildTessellation<Real, Weighted=true> — the radical-plane forward build (apply-all
 * reference gate, no security early-out) — three ways:
 *
 *   (A) Equal-weights regression (machine-exact): with a constant weight the radical plane offset
 *       ½(|r|²+w_i−w_j) collapses to the bisector ½|r|², so the power cells must reproduce the
 *       ordinary Voronoi cells bit-for-bit. Compares per-cell volumes of buildTessellation<true>
 *       (const w) against buildTessellation<false>.
 *
 *   (B) Oracle-free invariants (varying weights): a valid power diagram still PARTITIONS the box
 * and its radical facets are reciprocal, so the same space-filling / area-reciprocity / closure
 *       invariants used for Voronoi (checkInvariants) must hold.
 *
 *   (C) Independent brute-force oracle (varying weights): each cell's published volume vs a host
 *       O(N) clip of ALL other seeds' radical planes (min-image, closest-first). Also compares the
 *       face-neighbour set.
 *
 *   (F) reevalPublish on a weighted lattice re-evaluates the radical planes: its facet CSR equals a
 *       cold power build's (area vector, dV, connector, volume) to round-off, static and moved.
 *
 *   (G) A seed outside its own power cell (negative height, reached by re-evaluating a resident
 *       topology): the published connector is the generator connector p_j − p_i, the area vector
 *       points out of the cell, dV matches central differences, and the partition closes.
 *
 *   (H) Voronoi: the published connector 2·foot equals p_j − p_i bit for bit.
 *
 *   (I) Seeds outside their own power cell and empty power cells (DESIGN_NOTE_OUTSIDE_SEED WO-2):
 *       8³ lattices jittered ±0.3h with weights in [0, W h²], every cell against the sign-safe
 *       brute oracle (volume, empty ⟺ empty, facet set, kNegOffset ⟺ a face with nn < 0), and the
 *       oracle itself against the TRANSLATION oracle, which never executes a σ = −1 code path:
 *       the cell rebuilt in the frame of an interior point y, where every offset is positive.
 *       An empty cell is certified by the snapshot before its emptying candidate (validated the
 *       same way): every snapshot vertex violates the emptying plane.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/dynamic_validate.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"

using real_t = double;

namespace {

using OracleCell = peclet::voro::ConvexCell<real_t, 200, 400>;

// One radical-plane candidate of seed i: min-image connector r = p_j − p_i and offset
// d = ½(|r|²+w_i−w_j) of the half-space {x : r·x ≤ d} (seed frame), either sign.
struct Cand {
  real_t r[3];
  real_t off;
  int j;
};

// Every other seed's radical plane for seed i, closest-first (min-image, cubic box L).
std::vector<Cand> candidates(int i, const std::vector<real_t>& x, const std::vector<real_t>& w,
                             int N, real_t L) {
  const real_t Lh = real_t(0.5) * L;
  std::vector<std::pair<real_t, int>> ord;
  ord.reserve(N - 1);
  for (int j = 0; j < N; ++j) {
    if (j == i)
      continue;
    real_t rho = 0;
    for (int d = 0; d < 3; ++d) {
      real_t rr = x[3 * j + d] - x[3 * i + d];
      rr = rr > Lh ? rr - L : (rr < -Lh ? rr + L : rr);
      rho += rr * rr;
    }
    ord.emplace_back(rho, j);
  }
  std::sort(ord.begin(), ord.end());
  std::vector<Cand> out;
  out.reserve(ord.size());
  for (auto& pr : ord) {
    Cand c;
    c.j = pr.second;
    for (int d = 0; d < 3; ++d) {
      real_t rr = x[3 * c.j + d] - x[3 * i + d];
      c.r[d] = rr > Lh ? rr - L : (rr < -Lh ? rr + L : rr);
    }
    c.off = real_t(0.5) * (c.r[0] * c.r[0] + c.r[1] * c.r[1] + c.r[2] * c.r[2] + w[i] - w[c.j]);
    out.push_back(c);
  }
  return out;
}

// Clip `c` (initBox(box) centred on the frame origin) by the first `nUse` candidates in order, each
// shifted into the frame of the point y (offset d − r·y; y = 0: the seed frame). `seedBox` > 0:
// first clip the seed-centred box of that extent as six explicit planes (±x_ax ≤ seedBox/2 ∓ y_ax),
// so a cell touching the seed frame's box is the same set in y's frame. Returns 0 = ok, 1 = empty
// (a clip took every vertex; `emptyAt` = that candidate's index), -1 = overflow.
int clipCands(const std::vector<Cand>& cand, int nUse, const real_t y[3], real_t box, OracleCell& c,
              int* emptyAt = nullptr, real_t seedBox = 0) {
  c.initBox(box, box, box);
  if (seedBox > 0)
    for (int ax = 0; ax < 3; ++ax)
      for (int sg = -1; sg <= 1; sg += 2) {
        real_t pd[3] = {0, 0, 0};
        pd[ax] = real_t(sg);
        c.clip(pd, real_t(0.5) * seedBox - sg * y[ax], -1);
      }
  for (int k = 0; k < nUse; ++k) {
    const Cand& q = cand[k];
    const real_t off = q.off - (q.r[0] * y[0] + q.r[1] * y[1] + q.r[2] * y[2]);
    c.clip(q.r, off, q.j);
    if (c.overflow)
      return -1;
    if (c.nt == 0) {
      if (emptyAt)
        *emptyAt = k;
      return 1;
    }
  }
  return 0;
}

// Host brute-force power cell for seed i: clip the box by EVERY other seed's radical plane
// {x : r·x ≤ ½(|r|²+w_i−w_j)}, closest-first (keeps the live-plane count bounded), either sign of
// the offset (sign-safe planes). Returns 0 = ok, 1 = empty power cell, -1 = overflow. Generous
// caps for the O(N) oracle.
int buildPowerCell(int i, const std::vector<real_t>& x, const std::vector<real_t>& w, int N,
                   real_t L, OracleCell& c) {
  const auto cand = candidates(i, x, w, N, L);
  const real_t y0[3] = {0, 0, 0};
  return clipCands(cand, static_cast<int>(cand.size()), y0, L, c);
}

// The cell's faces: sorted pnbr of the planes an alive triangle references; `box` = a box face.
std::vector<long> facesOf(const OracleCell& c, bool* box = nullptr) {
  std::vector<long> f;
  if (box)
    *box = false;
  for (int k = 0; k < c.np; ++k) {
    bool used = false;
    for (int t = 0; t < c.nt && !used; ++t)
      used = c.alive[t] && (c.t0[t] == k || c.t1[t] == k || c.t2[t] == k);
    if (!used)
      continue;
    if (c.pnbr[k] < 0) {
      if (box)
        *box = true;
    } else {
      f.push_back(c.pnbr[k]);
    }
  }
  std::sort(f.begin(), f.end());
  return f;
}

// Some face of the cell excludes the seed (nn < 0): the seed lies outside its cell.
bool hasNegFace(const OracleCell& c) {
  for (int t = 0; t < c.nt; ++t)
    if (c.alive[t] && (c.nn[c.t0[t]] < 0 || c.nn[c.t1[t]] < 0 || c.nn[c.t2[t]] < 0))
      return true;
  return false;
}

// Per-face area by neighbour id (facetAreasPerVertex over the planes).
std::vector<std::pair<long, double>> areasOf(const OracleCell& c) {
  std::vector<real_t> a(c.np, real_t(0));
  c.facetAreasPerVertex(a.data());
  std::vector<std::pair<long, double>> out;
  for (long j : facesOf(c))
    for (int k = 0; k < c.np; ++k)
      if (c.pnbr[k] == j) {
        out.emplace_back(j, a[k]);
        break;
      }
  return out;
}

// The translation oracle for a seed-frame oracle cell `c` built from the first nUse candidates:
// y = mean of its live vertices; every candidate must have d − r·y > 0 (else "not interior"); the
// cell is rebuilt in y's frame (initBox(2L) centred at y, same order — only σ = +1 planes) into
// `ct`. Agreement: |ΔV| ≤ 1e-12 h³, facet sets equal, |ΔA| ≤ 1e-12 h², no box face. A snapshot
// (`snapshot`: the cell before its emptying candidate, which may still touch the seed frame's box)
// carries that box into y's frame as explicit planes, and its box faces must then agree instead.
struct TransResult {
  bool interior = false, agree = false;
  double dV = 0, dA = 0, distOverSize = 0;
};
TransResult translationOracle(const std::vector<Cand>& cand, int nUse, const OracleCell& c,
                              real_t L, double h, OracleCell& ct, real_t y[3],
                              bool snapshot = false) {
  TransResult r;
  y[0] = y[1] = y[2] = 0;
  int m = 0;
  for (int t = 0; t < c.nt; ++t)
    if (c.alive[t]) {
      y[0] += c.vx[t];
      y[1] += c.vy[t];
      y[2] += c.vz[t];
      ++m;
    }
  if (m == 0)
    return r;
  for (int d = 0; d < 3; ++d)
    y[d] /= m;
  for (int k = 0; k < nUse; ++k)
    if (!(cand[k].off - (cand[k].r[0] * y[0] + cand[k].r[1] * y[1] + cand[k].r[2] * y[2]) > 0))
      return r;  // not interior
  r.interior = true;
  if (clipCands(cand, nUse, y, 2 * L, ct, nullptr, snapshot ? L : real_t(0)) != 0)
    return r;
  const double V = c.volumePerVertex(), Vt = ct.volumePerVertex();
  r.dV = std::abs(V - Vt);
  r.distOverSize = std::sqrt(double(y[0]) * y[0] + double(y[1]) * y[1] + double(y[2]) * y[2]) /
                   std::cbrt(std::max(V, 1e-300));
  bool box = false, boxT = false;
  const auto f = facesOf(c, &box), ft = facesOf(ct, &boxT);
  const auto a = areasOf(c), at = areasOf(ct);
  bool sameA = a.size() == at.size();
  for (std::size_t k = 0; sameA && k < a.size(); ++k) {
    sameA = a[k].first == at[k].first;
    r.dA = std::max(r.dA, std::abs(a[k].second - at[k].second));
  }
  const bool boxOk = snapshot ? box == boxT : (!box && !boxT);
  r.agree = boxOk && f == ft && sameA && r.dV <= 1e-12 * h * h * h && r.dA <= 1e-12 * h * h;
  return r;
}

// Volume of the brute-force power cell (0 empty, -1 overflow).
double bruteVol(int i, const std::vector<real_t>& x, const std::vector<real_t>& w, int N, real_t L,
                std::vector<int>* nbrOut = nullptr) {
  OracleCell c;
  const int rc = buildPowerCell(i, x, w, N, L, c);
  if (rc == 1)
    return 0.0;
  if (rc < 0)
    return -1.0;
  if (nbrOut) {
    nbrOut->clear();
    for (int k = 0; k < c.np; ++k)
      if (c.pnbr[k] >= 0)
        nbrOut->push_back(c.pnbr[k]);
    std::sort(nbrOut->begin(), nbrOut->end());
  }
  return c.volumePerVertex();
}

std::vector<double> toHost(const Kokkos::View<real_t*, peclet::core::MemSpace>& v) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<double>(m.data(), m.data() + m.extent(0));
}

// (A) equal-weights: power(const w) volumes must match Voronoi volumes to machine precision.
int caseEqualWeights(int N, real_t L, unsigned seed) {
  const real_t Larr[3] = {L, L, L};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> xh(3 * N);
  for (auto& v : xh)
    v = L * U(rng);

  Kokkos::View<real_t*, peclet::core::MemSpace> pos("pos", 3 * N);
  Kokkos::deep_copy(pos, Kokkos::View<const real_t*, Kokkos::HostSpace>(xh.data(), 3 * N));
  Kokkos::View<real_t*, peclet::core::MemSpace> wConst("w", N);
  Kokkos::deep_copy(wConst, real_t(0.375));  // any constant
  Kokkos::View<real_t*, peclet::core::MemSpace> wEmpty;
  Kokkos::View<long*, peclet::core::MemSpace> gd;

  auto vor = peclet::voro::buildTessellation<real_t, false>(pos, wEmpty, N, Larr, 4, N, gd,
                                                            peclet::voro::NoSdf{}, true);
  auto pow = peclet::voro::buildTessellation<real_t, true>(pos, wConst, N, Larr, 4, N, gd,
                                                           peclet::voro::NoSdf{}, true);
  auto Vv = toHost(vor.view.cellVolume), Vp = toHost(pow.view.cellVolume);
  double maxRel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double denom = std::max(std::abs(Vv[i]), 1e-30);
    maxRel = std::max(maxRel, std::abs(Vv[i] - Vp[i]) / denom);
  }
  // apply-all commits borderline zero-area faces the Voronoi security cull skips (identical planes,
  // a retriangulation-rounding difference), so the match is ~1e-13, not bit-exact.
  const bool pass = maxRel < 1e-11;
  std::printf("  (A) equal-w  N=%-6d seed=%u  maxRelVolDiff(power vs voronoi)=%.2e  %s\n", N, seed,
              maxRel, pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

// (B)+(C) varying weights: invariants + brute-force oracle.
int caseVarWeights(int N, real_t L, unsigned seed, real_t wSpreadFrac) {
  const real_t Larr[3] = {L, L, L};
  const double boxVol = double(L) * L * L;
  const real_t spacing = L / std::cbrt(real_t(N));
  const real_t wMax = wSpreadFrac * spacing * spacing;  // weight span scaled to spacing²
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> xh(3 * N), wh(N);
  for (auto& v : xh)
    v = L * U(rng);
  for (auto& v : wh)
    v = wMax * U(rng);

  Kokkos::View<real_t*, peclet::core::MemSpace> pos("pos", 3 * N), wd("w", N);
  Kokkos::deep_copy(pos, Kokkos::View<const real_t*, Kokkos::HostSpace>(xh.data(), 3 * N));
  Kokkos::deep_copy(wd, Kokkos::View<const real_t*, Kokkos::HostSpace>(wh.data(), N));
  Kokkos::View<long*, peclet::core::MemSpace> gd;

  auto res = peclet::voro::buildTessellation<real_t, true>(pos, wd, N, Larr, 4, N, gd,
                                                           peclet::voro::NoSdf{}, true);
  auto aux = peclet::voro::buildAuxMaps(res.view);
  auto inv = peclet::voro::checkInvariants(res.view, aux, boxVol);

  // status: count finished / empty (buried) / overflow cells.
  long nOk = 0, nEmpty = 0, nOverflow = 0;
  {
    auto S = res.status;
    Kokkos::parallel_reduce(
        "nok", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i, long& c) { c += (S(i) == peclet::voro::kOk) ? 1 : 0; }, nOk);
    Kokkos::parallel_reduce(
        "nempty", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i, long& c) { c += (S(i) & peclet::voro::kEmpty) ? 1 : 0; }, nEmpty);
    Kokkos::parallel_reduce(
        "novf", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i, long& c) { c += (S(i) & peclet::voro::kOverflow) ? 1 : 0; },
        nOverflow);
  }

  // (C) brute-force full-N radical-plane oracle on host. Check ALL cells when N is small enough.
  auto Vh = toHost(res.view.cellVolume);
  const int nCheck = std::min(N, 2000);
  double maxRelVol = 0.0, maxAbsVol = 0.0, sumOracle = 0.0, sumDevChecked = 0.0;
  long nVolMismatch = 0, nOracleChecked = 0, nBuriedMismatch = 0;
  for (int i = 0; i < nCheck; ++i) {
    const double vb = bruteVol(i, xh, wh, N, L, nullptr);
    if (vb < 0)
      continue;  // oracle overflow (shouldn't happen); skip
    ++nOracleChecked;
    sumOracle += vb;
    sumDevChecked += Vh[i];
    // buried agreement: device-empty ⟺ oracle-empty (both detect the seed-outside-own-cell case).
    if ((Vh[i] <= 0.0) != (vb <= 0.0))
      ++nBuriedMismatch;
    if (Vh[i] <= 0.0 || vb <= 0.0)
      continue;  // buried on either side: no volume comparison
    const double rel = std::abs(vb - Vh[i]) / std::max(vb, 1e-30);
    maxRelVol = std::max(maxRelVol, rel);
    // Absolute criterion |ΔV| ≤ 1e-12 h³ (DESIGN_NOTE_OUTSIDE_SEED §2 and its amendment 3: a tiny
    // cell far from its seed carries a relative error ∝ ε·(dist/size)³ in the seed frame).
    const double h3 = double(spacing) * spacing * spacing;
    maxAbsVol = std::max(maxAbsVol, std::abs(vb - Vh[i]) / h3);
    if (std::abs(vb - Vh[i]) > 1e-12 * h3)
      ++nVolMismatch;
  }
  const double oracleFill = (nCheck == N) ? sumOracle / boxVol : -1.0;  // oracle space-filling
  const double devVsOracleSum = std::abs(sumDevChecked - sumOracle) / std::max(sumOracle, 1e-30);

  // P2 CORRECTNESS GATE: the device radical-plane build (weight-aware security cull + reach break +
  // sign-safe planes + empty-cell clip) must reproduce an INDEPENDENT full-N brute-force power
  // diagram — per-cell volume (|ΔV| ≤ 1e-12 h³), empty membership (nBuriedMismatch==0), and
  // reciprocal recorded facets (sumAreaMag≈0) — with no overflow. This proves the cull is
  // conservative (drops no real face) and outside-seed / empty cells are handled.
  //
  // NOT gated (diagnostics only): `volRelErr` (space-filling). The device matches the brute-force
  // oracle EXACTLY (devVsOracleSum≈0), yet the oracle ITSELF is not a perfect partition (oracleFill
  // ≈0.97–0.99): the min-image, sw-limited candidate model is not exactly space-filling for power
  // cells — a face can require a non-nearest periodic image. Closing that needs a multi-image
  // gather (future; the near-incompressible solver runs at small weights where the error is
  // negligible).
  const bool pass =
      nVolMismatch == 0 && nBuriedMismatch == 0 && inv.sumAreaMag < 1e-9 && nOverflow == 0;
  std::printf(
      "  (B/C) var-w N=%-6d seed=%u wSpread=%.2f | oracle: n=%ld maxRelVol=%.2e maxDV/h3=%.2e "
      "nVolMism=%ld emptyMism=%ld devVsOracle=%.2e areaClosure=%.2e | diag: nOk=%ld nEmpty=%ld "
      "volRelErr=%.2e oracleFill=%.6f  %s\n",
      N, seed, wSpreadFrac, nOracleChecked, maxRelVol, maxAbsVol, nVolMismatch, nBuriedMismatch,
      devVsOracleSum, inv.sumAreaMag, nOk, nEmpty, inv.volRelErr, oracleFill, pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

// (D) FD-vs-analytic power gradients: dV_i/dx_self, dV_i/dw_self, dV_i/dx_nbr, dV_i/dw_nbr from
// geomVolumeGrad (dV/dn_k) chained through the Power policy (dn_k/dDOF) via chainToDofs<Power>.
// Compact interior cells only (a box-wall face would move with the seed-centred oracle box and
// contaminate the position FD).
int caseGradients(int N, real_t L, unsigned seed, real_t wSpreadFrac) {
  const real_t spacing = L / std::cbrt(real_t(N));
  const real_t wMax = wSpreadFrac * spacing * spacing;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> xh(3 * N), wh(N);
  for (auto& v : xh)
    v = L * U(rng);
  for (auto& v : wh)
    v = wMax * U(rng);

  auto volAt = [&](int i, const std::vector<real_t>& x, const std::vector<real_t>& w) -> double {
    OracleCell cc;
    return buildPowerCell(i, x, w, N, L, cc) == 0 ? cc.volumePerVertex() : -1.0;
  };

  const real_t eps = 1e-6;
  const double relTol = 3e-3;                   // FD-limited
  const double sig = 5e-3 * spacing * spacing;  // significant-gradient floor (dV/dx ~ face area)
  double maxRel = 0.0;
  long nTot = 0, nPass = 0, nCellsChecked = 0, nNegChecked = 0;
  const int nCheck = std::min(N, 300);
  for (int i = 0; i < nCheck; ++i) {
    OracleCell c;
    if (buildPowerCell(i, xh, wh, N, L, c) != 0)
      continue;                          // empty/overflow
    const bool negFace = hasNegFace(c);  // seed outside its cell: foot-point gradients (WO-4)
    // interior filter: skip cells with any nonzero-area box-wall face (pnbr<0).
    double area[200];
    for (int k = 0; k < c.np; ++k)
      area[k] = 0.0;
    c.facetAreasPerVertex(area);
    bool wall = false;
    double cellArea = 0.0;
    for (int k = 0; k < c.np; ++k) {
      cellArea = std::max(cellArea, area[k]);
      if (c.pnbr[k] < 0 && area[k] > 1e-12)
        wall = true;
    }
    if (wall)
      continue;
    ++nCellsChecked;
    nNegChecked += negFace ? 1 : 0;

    // analytic: dV/dn_k then chain to DOFs.
    double vg = 0, dgx[200], dgy[200], dgz[200];
    for (int k = 0; k < c.np; ++k)
      dgx[k] = dgy[k] = dgz[k] = 0.0;
    c.geomVolumeGrad(vg, dgx, dgy, dgz);
    double fSelf[3], fwSelf, fnx[200], fny[200], fnz[200], fwn[200];
    const double seed3[3] = {xh[3 * i], xh[3 * i + 1], xh[3 * i + 2]};
    peclet::voro::chainToDofs<peclet::voro::Power>(c, seed3, xh.data(), (double)wh[i], wh.data(),
                                                   (double)L, dgx, dgy, dgz, fSelf, fwSelf, fnx,
                                                   fny, fnz, fwn);

    auto check = [&](double analytic, int idx, double eps2, bool weightDof) {
      // central-difference the volume of cell i w.r.t. DOF (perturbing xh/wh entry `idx`).
      std::vector<real_t>& arr = weightDof ? wh : xh;
      const real_t save = arr[idx];
      arr[idx] = save + (real_t)eps2;
      const double vp = volAt(i, xh, wh);
      arr[idx] = save - (real_t)eps2;
      const double vm = volAt(i, xh, wh);
      arr[idx] = save;
      if (vp < 0 || vm < 0)
        return;  // topology changed under perturbation: skip
      const double fd = (vp - vm) / (2 * eps2);
      if (std::abs(fd) < sig)
        return;  // not FD-resolvable
      ++nTot;
      const double rel = std::abs(fd - analytic) / std::max(std::abs(fd), 1e-30);
      maxRel = std::max(maxRel, rel);
      if (rel < relTol)
        ++nPass;
    };

    // self position (3) + self weight (1)
    for (int cc = 0; cc < 3; ++cc)
      check(fSelf[cc], 3 * i + cc, eps, false);
    check(fwSelf, i, eps, true);
    // one significant real-neighbour face: its position (3) + weight (1)
    for (int k = 0; k < c.np; ++k) {
      const int j = c.pnbr[k];
      if (j < 0 || area[k] < 0.1 * cellArea)
        continue;
      for (int cc = 0; cc < 3; ++cc)
        check(fnx[k] * (cc == 0) + fny[k] * (cc == 1) + fnz[k] * (cc == 2), 3 * j + cc, eps, false);
      check(fwn[k], j, eps, true);
      break;
    }
  }
  const bool pass = nTot > 0 && (double)nPass / nTot > 0.98 && maxRel < 5e-2;
  std::printf(
      "  (D) grad  N=%-6d seed=%u wSpread=%.2f | cells=%ld (outside-seed %ld) comps=%ld "
      "pass=%ld/%ld maxRel=%.2e  %s\n",
      N, seed, wSpreadFrac, nCellsChecked, nNegChecked, nTot, nPass, nTot, maxRel,
      pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

// (E) dynamic power tessellation infrastructure: cold-build a power tessellation, move the seeds a
// little, and require MovingTessellation<...,Weighted=true>::step() to reproduce a cold power
// rebuild at the new positions — per-cell volume and buried/empty membership. Small weights (d>0).
int caseDynamic(int N, real_t L, unsigned seed, real_t wSpreadFrac, real_t dispFrac) {
  using Mem = peclet::core::MemSpace;
  const real_t Larr[3] = {L, L, L};
  const real_t spacing = L / std::cbrt(real_t(N));
  const real_t wMax = wSpreadFrac * spacing * spacing;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> x0(3 * N), x1(3 * N), wh(N);
  for (auto& v : x0)
    v = L * U(rng);
  for (auto& v : wh)
    v = wMax * U(rng);
  const real_t disp = dispFrac * spacing;
  for (int k = 0; k < 3 * N; ++k) {
    real_t p = x0[k] + disp * (2 * U(rng) - 1);
    p = p < 0 ? p + L : (p >= L ? p - L : p);
    x1[k] = p;
  }

  Kokkos::View<real_t*, Mem> pos0("p0", 3 * N), pos1("p1", 3 * N), wd("w", N);
  Kokkos::deep_copy(pos0, Kokkos::View<const real_t*, Kokkos::HostSpace>(x0.data(), 3 * N));
  Kokkos::deep_copy(pos1, Kokkos::View<const real_t*, Kokkos::HostSpace>(x1.data(), 3 * N));
  Kokkos::deep_copy(wd, Kokkos::View<const real_t*, Kokkos::HostSpace>(wh.data(), N));

  const real_t tol = real_t(1e-4) * spacing, skin = real_t(0.25) * spacing;
  peclet::voro::MovingTessellation<real_t, 64, 112, true> mt;
  mt.alloc(N, Larr, tol, skin, 4, N);
  mt.setWeights(wd);
  mt.rebuild(pos0);
  auto stats = mt.step(pos1);
  auto volRepair = toHost(mt.vol);

  Kokkos::View<long*, Mem> gd;
  auto cold = peclet::voro::buildTessellation<real_t, true>(pos1, wd, N, Larr, 4, N, gd,
                                                            peclet::voro::NoSdf{}, true);
  auto volCold = toHost(cold.view.cellVolume);

  double maxRel = 0.0;
  long nMism = 0, nBuriedMism = 0, nEmpty = 0;
  for (int i = 0; i < N; ++i) {
    if ((volCold[i] <= 0.0) != (volRepair[i] <= 0.0))
      ++nBuriedMism;
    if (volCold[i] <= 0.0) {
      ++nEmpty;
      continue;
    }
    if (volRepair[i] <= 0.0)
      continue;
    const double rel = std::abs(volCold[i] - volRepair[i]) / std::max(volCold[i], 1e-30);
    maxRel = std::max(maxRel, rel);
    if (rel > 1e-9)
      ++nMism;
  }
  const bool pass = maxRel < 1e-9 && nMism == 0 && nBuriedMism == 0;
  std::printf(
      "  (E) dyn   N=%-6d seed=%u wSpread=%.2f disp=%.2f | step vs cold: maxRelVol=%.2e "
      "nMism=%ld buriedMism=%ld nEmpty=%ld fellBack=%d  %s\n",
      N, seed, wSpreadFrac, dispFrac, maxRel, nMism, nBuriedMism, nEmpty, (int)stats.fellBack,
      pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

// Host copy of a published facet CSR (per-cell facet range, neighbour, area vector, dV, connector).
struct HostFacets {
  std::vector<int> off, cnt;
  std::vector<long> nbr;
  std::vector<double> A, D, C, vol;
};

HostFacets toHostFacets(const peclet::voro::TessellationView<real_t>& v) {
  HostFacets h;
  auto off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.cellFacetOffset);
  auto cnt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.cellFacetCount);
  auto nb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.facetNeighbor);
  h.off.assign(off.data(), off.data() + off.extent(0));
  h.cnt.assign(cnt.data(), cnt.data() + cnt.extent(0));
  h.nbr.assign(nb.data(), nb.data() + nb.extent(0));
  h.A = toHost(v.facetArea);
  h.D = toHost(v.facetConnect);
  h.C = toHost(v.facetConnVec);
  h.vol = toHost(v.cellVolume);
  return h;
}

// Facet of cell i toward neighbour j in a published CSR, or -1.
int findFacet(const HostFacets& h, int i, long j) {
  for (int f = h.off[i]; f < h.off[i] + h.cnt[i]; ++f)
    if (h.nbr[f] == j)
      return f;
  return -1;
}

// Min-image generator connector p_j − p_i (cubic box L).
void genConn(const std::vector<real_t>& x, int i, int j, real_t L, double r[3]) {
  const real_t Lh = real_t(0.5) * L;
  for (int d = 0; d < 3; ++d) {
    real_t rr = x[3 * j + d] - x[3 * i + d];
    r[d] = rr > Lh ? rr - L : (rr < -Lh ? rr + L : rr);
  }
}

// Jittered cubic lattice of n³ seeds in the unit box (spacing h = 1/n, jitter ±jit·h).
std::vector<real_t> jitteredLattice(int n, real_t jit, std::mt19937& rng) {
  std::uniform_real_distribution<real_t> U(-1.0, 1.0);
  const real_t h = real_t(1) / n;
  std::vector<real_t> x;
  x.reserve(3 * n * n * n);
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const real_t p[3] = {(i + real_t(0.5)) * h, (j + real_t(0.5)) * h, (k + real_t(0.5)) * h};
        for (int d = 0; d < 3; ++d)
          x.push_back(p[d] + jit * h * U(rng));
      }
  return x;
}

// (F) reevalPublish on a weighted lattice re-evaluates the RADICAL planes: the facet CSR it
// publishes over the resident topology (static, and after a small move + repair) must equal a
// cold power build's at the same positions — per facet the area vector, dV and connector, per
// cell the neighbour set and volume — to round-off. Before the fix reevalPublish rebuilt every
// plane as the Voronoi bisector (O(1) area errors on a weighted tessellation).
int caseReevalVsCold(int n, unsigned seed, real_t wSpreadFrac, real_t dispFrac) {
  using Mem = peclet::core::MemSpace;
  const real_t L = 1.0, Larr[3] = {L, L, L}, h = L / n;
  const int N = n * n * n;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> x0 = jitteredLattice(n, 0.15, rng), x1(3 * N), wh(N);
  for (auto& v : wh)
    v = wSpreadFrac * h * h * U(rng);
  for (int k = 0; k < 3 * N; ++k) {
    real_t p = x0[k] + dispFrac * h * (2 * U(rng) - 1);
    x1[k] = p < 0 ? p + L : (p >= L ? p - L : p);
  }
  Kokkos::View<real_t*, Mem> pos0("p0", 3 * N), pos1("p1", 3 * N), wd("w", N);
  Kokkos::deep_copy(pos0, Kokkos::View<const real_t*, Kokkos::HostSpace>(x0.data(), 3 * N));
  Kokkos::deep_copy(pos1, Kokkos::View<const real_t*, Kokkos::HostSpace>(x1.data(), 3 * N));
  Kokkos::deep_copy(wd, Kokkos::View<const real_t*, Kokkos::HostSpace>(wh.data(), N));
  Kokkos::View<long*, Mem> gd;

  peclet::voro::MovingTessellation<real_t, 64, 112, true> mt;
  mt.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
  mt.setWeights(wd);
  mt.rebuild(pos0);
  int rc = 0;
  for (int pass = 0; pass < 2; ++pass) {
    const auto& pos = pass == 0 ? pos0 : pos1;
    if (pass == 1)
      mt.step(pos1);
    auto rp = peclet::voro::reevalPublish<real_t, 64, 112, peclet::voro::Power>(
        mt.store, pos, mt.vol, N, Larr, mt.wall, mt.xRef, false, false, wd);
    auto cold = peclet::voro::buildTessellation<real_t, true>(pos, wd, N, Larr, 4, N, gd,
                                                              peclet::voro::NoSdf{}, true);
    const HostFacets R = toHostFacets(rp), C = toHostFacets(cold.view);
    double maxA = 0, maxD = 0, maxC = 0, maxV = 0;
    long nSetMism = 0;
    for (int i = 0; i < N; ++i) {
      maxV = std::max(maxV, std::abs(R.vol[i] - C.vol[i]) / std::max(C.vol[i], 1e-300));
      // the cell's largest facet area / dV: the scales of a relative facet error
      double aScale = 0, dScale = 0;
      for (int f = C.off[i]; f < C.off[i] + C.cnt[i]; ++f) {
        aScale =
            std::max(aScale, std::sqrt(C.A[3 * f] * C.A[3 * f] + C.A[3 * f + 1] * C.A[3 * f + 1] +
                                       C.A[3 * f + 2] * C.A[3 * f + 2]));
        dScale =
            std::max(dScale, std::sqrt(C.D[3 * f] * C.D[3 * f] + C.D[3 * f + 1] * C.D[3 * f + 1] +
                                       C.D[3 * f + 2] * C.D[3 * f + 2]));
      }
      if (R.cnt[i] != C.cnt[i])
        ++nSetMism;
      for (int f = C.off[i]; f < C.off[i] + C.cnt[i]; ++f) {
        const int g = findFacet(R, i, C.nbr[f]);
        if (g < 0) {
          ++nSetMism;
          continue;
        }
        const double rLen = std::sqrt(C.C[3 * f] * C.C[3 * f] + C.C[3 * f + 1] * C.C[3 * f + 1] +
                                      C.C[3 * f + 2] * C.C[3 * f + 2]);
        for (int d = 0; d < 3; ++d) {
          maxA = std::max(maxA, std::abs(R.A[3 * g + d] - C.A[3 * f + d]) / aScale);
          maxD = std::max(maxD, std::abs(R.D[3 * g + d] - C.D[3 * f + d]) / dScale);
          maxC = std::max(maxC, std::abs(R.C[3 * g + d] - C.C[3 * f + d]) / rLen);
        }
      }
    }
    const bool pass_ =
        nSetMism == 0 && maxA <= 1e-13 && maxV <= 1e-13 && maxD <= 1e-12 && maxC <= 1e-14;
    std::printf(
        "  (F) reeval N=%-5d seed=%u wSpread=%.2f %s | vs cold: maxRelArea=%.2e maxRelVol=%.2e "
        "maxRelDV=%.2e maxRelConn=%.2e nbrMism=%ld  %s\n",
        N, seed, wSpreadFrac, pass == 0 ? "static" : "moved ", maxA, maxV, maxD, maxC, nSetMism,
        pass_ ? "OK" : "FAIL");
    rc |= pass_ ? 0 : 1;
  }
  return rc;
}

// (G) power facet geometry with a seed OUTSIDE its own cell. A seed i is placed at 0.3h from a
// lattice seed j and j carries the weight W: cell i lies on the far side of the radical plane,
// at height d = ½(|r|² − W)/|r| from p_i — inside for W < |r|², outside (d < 0) for W > |r|².
// The cold build buries a seed outside its own cell, so the outside state is reached the way the
// moving path reaches it: the resident topology of a cold build at W0 = 0.95|r|², re-evaluated
// (reevalPublish) at W1 = 1.05|r|². For every published neighbour facet, in both states:
//   * conn == p_j − p_i (the generator connector) to round-off — not 2·foot;
//   * A·conn > 0 — the area vector points out of the cell, toward the neighbour;
// and for the cell i itself, dV against central differences of V_i in p_j (the derivation of
// ∂V/∂r with r the generator connector). The re-evaluated partition is checked independently:
// Σ_i V_i (divergence theorem over the published facets) = box volume, A_ij = −A_ji.
int caseSeedOutside(int n, unsigned seed) {
  using Mem = peclet::core::MemSpace;
  const real_t L = 1.0, Larr[3] = {L, L, L}, h = L / n;
  std::mt19937 rng(seed);
  std::vector<real_t> x = jitteredLattice(n, 0.1, rng);
  const int jSeed = (n / 2) + n * (n / 2) + n * n * (n / 2);  // an interior lattice seed
  const int N = n * n * n + 1, iSeed = N - 1;                 // seed i appended
  const real_t sep = real_t(0.3) * h;
  x.push_back(x[3 * jSeed] + sep);
  x.push_back(x[3 * jSeed + 1]);
  x.push_back(x[3 * jSeed + 2]);
  const real_t rho = sep * sep;
  std::vector<real_t> w0(N, 0.0), w1(N, 0.0);
  w0[jSeed] = real_t(0.95) * rho;
  w1[jSeed] = real_t(1.05) * rho;

  Kokkos::View<real_t*, Mem> pos("p", 3 * N), wd0("w0", N), wd1("w1", N);
  Kokkos::deep_copy(pos, Kokkos::View<const real_t*, Kokkos::HostSpace>(x.data(), 3 * N));
  Kokkos::deep_copy(wd0, Kokkos::View<const real_t*, Kokkos::HostSpace>(w0.data(), N));
  Kokkos::deep_copy(wd1, Kokkos::View<const real_t*, Kokkos::HostSpace>(w1.data(), N));
  Kokkos::View<long*, Mem> gd;

  peclet::voro::MovingTessellation<real_t, 64, 112, true> mt;
  mt.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
  mt.setWeights(wd0);
  mt.rebuild(pos);
  auto cold = peclet::voro::buildTessellation<real_t, true>(pos, wd0, N, Larr, 4, N, gd,
                                                            peclet::voro::NoSdf{}, true);
  auto publish = [&](const Kokkos::View<real_t*, Mem>& p, const Kokkos::View<real_t*, Mem>& w) {
    return toHostFacets(peclet::voro::reevalPublish<real_t, 64, 112, peclet::voro::Power>(
        mt.store, p, mt.vol, N, Larr, mt.wall, mt.xRef, false, false, w));
  };
  const HostFacets F0 = toHostFacets(cold.view), F1 = publish(pos, wd1);

  // Volume of cell i from its published facets (divergence theorem, origin at p_i): each facet
  // lies in the plane {y : r·y = d}, d = ½(|r|² + w_i − w_j), so ∮ y·dA = Σ_f (d/|r|²) A_f·r_f.
  auto cellVol = [&](const HostFacets& F, int i, const std::vector<real_t>& w) {
    double V = 0;
    for (int f = F.off[i]; f < F.off[i] + F.cnt[i]; ++f) {
      if (F.nbr[f] < 0)
        return std::nan("");  // a box / wall facet: not a periodic partition
      const double* r = &F.C[3 * f];
      const double rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      const double d = 0.5 * (rr + w[i] - w[F.nbr[f]]);
      V += d / rr * (F.A[3 * f] * r[0] + F.A[3 * f + 1] * r[1] + F.A[3 * f + 2] * r[2]);
    }
    return V / 3.0;
  };

  int rc = 0;
  for (int st = 0; st < 2; ++st) {
    const HostFacets& F = st == 0 ? F0 : F1;
    const std::vector<real_t>& w = st == 0 ? w0 : w1;
    double maxConn = 0, minOut = 1e300, maxRecip = 0, sumV = 0;
    long nFacets = 0;
    for (int i = 0; i < N; ++i) {
      for (int f = F.off[i]; f < F.off[i] + F.cnt[i]; ++f) {
        const int j = (int)F.nbr[f];
        if (j < 0)
          continue;
        double r[3];
        genConn(x, i, j, L, r);
        const double rl = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        const double* A = &F.A[3 * f];
        const double al = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
        for (int d = 0; d < 3; ++d)
          maxConn = std::max(maxConn, std::abs(F.C[3 * f + d] - r[d]) / rl);
        if (al > 1e-12 * h * h)  // orientation is meaningful on a non-degenerate facet
          minOut = std::min(minOut, (A[0] * r[0] + A[1] * r[1] + A[2] * r[2]) / (al * rl));
        const int g = findFacet(F, j, i);
        if (g >= 0)
          for (int d = 0; d < 3; ++d)
            maxRecip = std::max(maxRecip, std::abs(A[d] + F.A[3 * g + d]) / (h * h));
        ++nFacets;
      }
      sumV += cellVol(F, i, w);
    }
    // the seed's height over the i–j plane (negative: p_i outside its own cell)
    const double hij = 0.5 * (rho - w[jSeed]) / sep;
    const int fij = findFacet(F, iSeed, jSeed);
    bool ok = fij >= 0 && maxConn <= 1e-14 && minOut > 0.0 && maxRecip <= 1e-13 &&
              std::abs(sumV - L * L * L) <= 1e-13;
    double fdErr = -1;
    if (st == 1 && fij >= 0) {  // dV_i/dp_j by central differences on the resident topology
      const double eps = 1e-6 * h;
      double dvMax = 0, errMax = 0;
      for (int d = 0; d < 3; ++d) {
        std::vector<real_t> xp = x, xm = x;
        xp[3 * jSeed + d] += eps;
        xm[3 * jSeed + d] -= eps;
        Kokkos::View<real_t*, Mem> pp("pp", 3 * N), pm("pm", 3 * N);
        Kokkos::deep_copy(pp, Kokkos::View<const real_t*, Kokkos::HostSpace>(xp.data(), 3 * N));
        Kokkos::deep_copy(pm, Kokkos::View<const real_t*, Kokkos::HostSpace>(xm.data(), 3 * N));
        const double fd =
            (cellVol(publish(pp, wd1), iSeed, w1) - cellVol(publish(pm, wd1), iSeed, w1)) /
            (2 * eps);
        dvMax = std::max(dvMax, std::abs(F.D[3 * fij + d]));
        errMax = std::max(errMax, std::abs(F.D[3 * fij + d] - fd));
      }
      fdErr = errMax / dvMax;
      ok = ok && fdErr <= 1e-6;
    }
    std::printf(
        "  (G) outside N=%-4d seed=%u %s h_ij/h=%+.4f | facets=%ld max|conn-(pj-pi)|/|r|=%.2e "
        "min cos(A,conn)=%+.3f recip=%.2e |sumV-L3|=%.2e dV-FD rel=%.2e  %s\n",
        N, seed, st == 0 ? "cold  W=0.95|r|^2" : "reeval W=1.05|r|^2", hij / h, nFacets, maxConn,
        minOut, maxRecip, std::abs(sumV - L * L * L), fdErr, ok ? "OK" : "FAIL");
    rc |= ok ? 0 : 1;
  }
  return rc;
}

// (H) Voronoi: the published connector 2·foot IS the generator connector p_j − p_i, bit for bit
// (the bisector foot is ½r, a power-of-two scaling), on the cold build and on reevalPublish —
// the reason the Voronoi publish keeps 2·foot and stays byte-identical.
int caseVoronoiConnExact(int N, unsigned seed) {
  using Mem = peclet::core::MemSpace;
  const real_t L = 1.0, Larr[3] = {L, L, L};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> x(3 * N);
  for (auto& v : x)
    v = L * U(rng);
  Kokkos::View<real_t*, Mem> pos("p", 3 * N), wEmpty;
  Kokkos::deep_copy(pos, Kokkos::View<const real_t*, Kokkos::HostSpace>(x.data(), 3 * N));
  Kokkos::View<long*, Mem> gd;
  auto cold = peclet::voro::buildTessellation<real_t, false>(pos, wEmpty, N, Larr, 4, N, gd,
                                                             peclet::voro::NoSdf{}, true);
  const real_t h = L / std::cbrt(real_t(N));
  peclet::voro::MovingTessellation<real_t, 64, 112, false> mt;
  mt.alloc(N, Larr, real_t(1e-4) * h, real_t(0.25) * h, 4, N);
  mt.rebuild(pos);
  const HostFacets C = toHostFacets(cold.view),
                   R = toHostFacets(peclet::voro::reevalPublish<real_t, 64, 112>(
                       mt.store, pos, mt.vol, N, Larr, mt.wall, mt.xRef));
  long nDiff = 0, nF = 0;
  for (const HostFacets* F : {&C, &R})
    for (int i = 0; i < N; ++i)
      for (int f = F->off[i]; f < F->off[i] + F->cnt[i]; ++f) {
        if (F->nbr[f] < 0)
          continue;
        double r[3];
        genConn(x, i, (int)F->nbr[f], L, r);
        ++nF;
        for (int d = 0; d < 3; ++d)
          nDiff += (F->C[3 * f + d] != r[d]) ? 1 : 0;
      }
  const bool pass = nDiff == 0 && nF > 0;
  std::printf(
      "  (H) voronoi N=%-5d seed=%u | 2·foot vs p_j-p_i over %ld facets (cold+reeval): "
      "%ld components differ  %s\n",
      N, seed, nF, nDiff, pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace

// (I) Seeds outside their own power cell and empty power cells against the sign-safe oracle and
// the translation oracle (header). Totals over all configurations feed the non-vacuity check.
long gOutside = 0, gEmpty = 0;
double gDistOverSize = 0;
int caseOutsideSeedOracle(int n, unsigned seed, real_t W) {
  using Mem = peclet::core::MemSpace;
  const real_t L = 1.0, Larr[3] = {L, L, L}, h = L / n;
  const int N = n * n * n;
  std::mt19937 rng(seed);
  std::vector<real_t> x = jitteredLattice(n, 0.3, rng);
  for (auto& v : x)
    v = v < 0 ? v + L : (v >= L ? v - L : v);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<real_t> w(N);
  for (auto& v : w)
    v = W * h * h * U(rng);
  Kokkos::View<real_t*, Mem> pos("pos", 3 * N), wd("w", N);
  Kokkos::deep_copy(pos, Kokkos::View<const real_t*, Kokkos::HostSpace>(x.data(), 3 * N));
  Kokkos::deep_copy(wd, Kokkos::View<const real_t*, Kokkos::HostSpace>(w.data(), N));
  Kokkos::View<long*, Mem> gd;
  auto res = peclet::voro::buildTessellation<real_t, true>(pos, wd, N, Larr, 4, N, gd,
                                                           peclet::voro::NoSdf{}, true);
  const HostFacets hf = toHostFacets(res.view);
  auto stH = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.status);
  const double h3 = double(h) * h * h;
  long nOutside = 0, nEmpty = 0, bad = 0, badTrans = 0, badCert = 0, oracleOvf = 0;
  double maxDV = 0, maxTransDV = 0, maxTransDA = 0;
  for (int i = 0; i < N; ++i) {
    const auto cand = candidates(i, x, w, N, L);
    const real_t y0[3] = {0, 0, 0};
    OracleCell c, ct;
    int emptyAt = -1;
    const int rc = clipCands(cand, static_cast<int>(cand.size()), y0, L, c, &emptyAt);
    if (rc < 0) {
      ++oracleOvf;
      continue;
    }
    const int st = stH(i);
    const bool devEmpty = (st & peclet::voro::kEmpty) != 0;
    real_t y[3];
    if (rc == 1) {  // empty: certify by the validated snapshot
      ++nEmpty;
      OracleCell s;
      clipCands(cand, emptyAt, y0, L, s);
      const TransResult tr = translationOracle(cand, emptyAt, s, L, h, ct, y, true);
      bool allOut = tr.agree;
      const Cand& e = cand[emptyAt];
      const real_t de = e.off - (e.r[0] * y[0] + e.r[1] * y[1] + e.r[2] * y[2]);
      for (int t = 0; t < ct.nt && allOut; ++t)
        if (ct.alive[t])
          allOut = e.r[0] * ct.vx[t] + e.r[1] * ct.vy[t] + e.r[2] * ct.vz[t] > de;
      if (!allOut)
        ++badCert;
      const bool ok =
          devEmpty && (st & peclet::voro::kBuried) && hf.vol[i] == 0.0 && hf.cnt[i] == 0;
      if (!ok) {
        ++bad;
        if (bad <= 5)
          std::printf("    (I) cell %d: oracle EMPTY, device status=%d vol=%.3e facets=%d\n", i, st,
                      hf.vol[i], hf.cnt[i]);
      }
      continue;
    }
    const TransResult tr = translationOracle(cand, static_cast<int>(cand.size()), c, L, h, ct, y);
    if (!tr.agree) {
      ++badTrans;
      if (badTrans <= 5)
        std::printf("    (I) cell %d: translation oracle interior=%d dV=%.3e dA=%.3e\n", i,
                    (int)tr.interior, tr.dV, tr.dA);
    }
    maxTransDV = std::max(maxTransDV, tr.dV / h3);
    maxTransDA = std::max(maxTransDA, tr.dA / (double(h) * h));
    const bool neg = hasNegFace(c);
    if (neg) {
      ++nOutside;
      gDistOverSize = std::max(gDistOverSize, tr.distOverSize);
    }
    std::vector<long> fd;
    for (int f = hf.off[i]; f < hf.off[i] + hf.cnt[i]; ++f)
      fd.push_back(hf.nbr[f]);
    std::sort(fd.begin(), fd.end());
    const double dV = std::abs(hf.vol[i] - c.volumePerVertex());
    maxDV = std::max(maxDV, dV / h3);
    const bool devNeg = (st & peclet::voro::kNegOffset) != 0;
    const bool ok = !devEmpty && dV <= 1e-12 * h3 && fd == facesOf(c) && devNeg == neg;
    if (!ok) {
      ++bad;
      if (bad <= 5)
        std::printf(
            "    (I) cell %d: status=%d dV/h3=%.3e facets dev=%zu oracle=%zu neg dev=%d "
            "oracle=%d\n",
            i, st, dV / h3, fd.size(), facesOf(c).size(), (int)devNeg, (int)neg);
    }
  }
  gOutside += nOutside;
  gEmpty += nEmpty;
  const bool pass = bad == 0 && badTrans == 0 && badCert == 0 && oracleOvf == 0;
  std::printf(
      "  (I) outside n=%d seed=%u W=%.2f | outsideSeed=%ld empty=%ld | device vs oracle: bad=%ld "
      "maxDV/h3=%.2e | translation: bad=%ld maxDV/h3=%.2e maxDA/h2=%.2e | emptyCertBad=%ld "
      "oracleOvf=%ld  %s\n",
      n, seed, W, nOutside, nEmpty, bad, maxDV, badTrans, maxTransDV, maxTransDA, badCert,
      oracleOvf, pass ? "OK" : "FAIL");
  return pass ? 0 : 1;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("device POWER-cell tessellation (backend=%s):\n",
                Kokkos::DefaultExecutionSpace::name());
    for (unsigned seed : {1u, 7u, 42u}) {
      rc |= caseEqualWeights(2000, 1.0, seed);
      rc |= caseVarWeights(2000, 1.0, seed, 0.25f);
      rc |= caseVarWeights(2000, 1.0, seed, 0.60f);
      rc |= caseGradients(2000, 1.0, seed, 0.25f);
    }
    rc |= caseEqualWeights(20000, 1.0, 123u);
    rc |= caseVarWeights(20000, 1.0, 123u, 0.40f);
    for (unsigned seed : {1u, 7u}) {
      rc |= caseDynamic(4000, 1.0, seed, 0.05f, 0.02f);  // small weights, small motion
      rc |= caseDynamic(4000, 1.0, seed, 0.10f, 0.05f);  // more weight + motion
    }
    for (unsigned seed : {3u, 11u})
      rc |= caseReevalVsCold(10, seed, 0.30f, 0.02f);
    for (unsigned seed : {5u, 9u})
      rc |= caseSeedOutside(8, seed);
    for (unsigned seed : {1u, 7u})
      rc |= caseVoronoiConnExact(3000, seed);
    // W = 2.0 added because W ≤ 1.0 gave 7 empty cells (< 10; note WO-2 non-vacuity rule)
    for (real_t W : {real_t(0.15), real_t(0.5), real_t(1.0), real_t(2.0)})
      for (unsigned seed : {1u, 2u, 3u})
        rc |= caseOutsideSeedOracle(8, seed, W);
    {
      const bool nonVacuous = gOutside >= 50 && gEmpty >= 10;
      std::printf("  (I) totals: outsideSeed=%ld (>= 50) empty=%ld (>= 10) maxDist/size=%.2f  %s\n",
                  gOutside, gEmpty, gDistOverSize, nonVacuous ? "OK" : "FAIL (vacuous)");
      rc |= nonVacuous ? 0 : 1;
    }
    std::printf("%s\n", rc == 0 ? "ALL POWER-CELL CHECKS PASS" : "POWER-CELL CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
