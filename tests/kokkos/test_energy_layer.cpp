/**
 * @file test_energy_layer.cpp
 * @brief Rung A3 gate (Voronoi methods plan §3): the published facet-edge AREA JACOBIANS and
 * the shared energy layer built on them.
 *
 *  (A) Published ∂A_f/∂n_l == the per-cell reconstruction (geomVolumeAreaGrad gathered over a
 *      rebuilt ConvexCell, the pre-A3 path) block for block, to round-off; the published area
 *      magnitudes match |facetArea|.
 *  (B) Interfacial energy gradient on the published view == the reconstruction oracle
 *      (detail::interfaceGradReconstruct) to round-off, and FD-exact on random DOFs.
 *  (C) Power (weighted) interfacial gradient: FD-exact in positions AND weights.
 *  (D) Wall energy σ_s Σ|A_wall| against a FLAT wall: FD-exact (the seed-foot chain is exact
 *      for a flat wall).
 *  (E) Volume energy Σ(V/Vref−1)² gradient: FD-exact (neighbour facets; with the flat wall too).
 * FD checks are central differences with topology-stable steps; a DOF whose step flips a face
 * is skipped (reported), never counted as a pass.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/energy/interface.hpp"
#include "peclet/voro/energy/volume.hpp"
#include "peclet/voro/energy/wall.hpp"
#include "peclet/voro/mesh_optimizer.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tessellator.hpp"

using Real = double;
using Mem = peclet::core::MemSpace;
using DV = Kokkos::View<Real*, Mem>;
using peclet::voro::NoSdf;
using peclet::voro::Power;
using peclet::voro::Voronoi;

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

// A flat wall: solid where x < xw (φ = x − xw, |∇φ| = 1).
struct FlatWall {
  Real xw;
  KOKKOS_INLINE_FUNCTION Real eval(Real x, Real, Real) const { return x - xw; }
  KOKKOS_INLINE_FUNCTION Real gradH() const { return Real(1e-5); }
};

// Energy of a configuration under the chosen term (rebuilds the tessellation). kind: 0 =
// interface, 1 = wall, 2 = volume target. Also returns the force and a topology signature.
template <class Policy, class Sdf>
static double energyForce(int kind, const std::vector<Real>& x, const std::vector<Real>& w,
                          const std::vector<int>& type, Real sigma, const Real L[3], int N,
                          const Sdf& sdf, std::vector<Real>& f, std::vector<Real>& fw,
                          std::vector<int>* sig = nullptr) {
  constexpr bool W = Policy::kHasWeightDof;
  DV dpos = up(x, "pos"), dw = W ? up(w, "w") : DV{};
  Kokkos::View<long*, Mem> gd;
  auto res = peclet::voro::buildTessellation<Real, W, Sdf>(
      dpos, dw, N, L, 4, N, gd, sdf, true, -1, {}, {}, {}, {}, {}, {}, 0, nullptr, {},
      /*withAreaGrad=*/true, {}, {}, 0, Real(0), /*withWallFD=*/true);
  DV force("force", 3 * N), forceW("forceW", W ? N : 0);
  Kokkos::deep_copy(force, Real(0));
  if (W)
    Kokkos::deep_copy(forceW, Real(0));
  double E = 0;
  if (kind == 0) {
    Kokkos::View<int*, Mem> dtype("type", N);
    Kokkos::deep_copy(dtype, Kokkos::View<const int*, Kokkos::HostSpace>(type.data(), N));
    DV ten("ten", 4);
    Kokkos::deep_copy(ten, Real(0));
    Kokkos::deep_copy(Kokkos::subview(ten, 1), sigma);
    Kokkos::deep_copy(Kokkos::subview(ten, 2), sigma);
    E = peclet::voro::energy::interfaceEnergyForce<Real, Policy, Sdf>(res.view, dtype, ten, 2, dpos,
                                                                      dw, L[0], force, forceW, sdf);
  } else if (kind == 1) {  // per-species wall tension: σ_s(0) = 0, σ_s(1) = sigma (wetting)
    Kokkos::View<int*, Mem> dtype("type", N);
    Kokkos::deep_copy(dtype, Kokkos::View<const int*, Kokkos::HostSpace>(type.data(), N));
    DV ss("ss", 2);
    Kokkos::deep_copy(ss, Real(0));
    Kokkos::deep_copy(Kokkos::subview(ss, 1), sigma);
    E = peclet::voro::energy::wallEnergyForce<Real, Policy, Sdf>(res.view, dtype, ss, sdf, dpos, dw,
                                                                 L[0], force, forceW);
  } else {
    auto vol = down(res.view.cellVolume);
    const Real Vref = (L[0] * L[1] * L[2]) / N;
    std::vector<Real> de(N);
    for (int i = 0; i < N; ++i) {
      const Real r = vol[i] / Vref - Real(1);
      E += r * r;
      de[i] = peclet::voro::energy::dTargetVolume<Real>(vol[i], Vref);
    }
    peclet::voro::energy::volumeGradientForce<Real, Policy, Sdf>(res.view, up(de, "de"), dpos, dw,
                                                                 L[0], force, forceW, sdf);
  }
  f = down(force);
  fw = W ? down(forceW) : std::vector<Real>{};
  if (sig) {  // topology signature: per-cell facet counts (a flip changes one)
    auto cnt = down(res.view.cellFacetCount);
    sig->assign(cnt.begin(), cnt.end());
  }
  return E;
}

// Central-difference check of the force on `nProbe` random DOFs (positions, and weights if W).
template <class Policy, class Sdf>
static bool fdCheck(const char* name, int kind, std::vector<Real> x, std::vector<Real> w,
                    const std::vector<int>& type, Real sigma, const Real L[3], int N,
                    const Sdf& sdf, std::mt19937& rng, int nProbe, double tolRel) {
  constexpr bool W = Policy::kHasWeightDof;
  std::vector<Real> f, fw;
  std::vector<int> sig0;
  energyForce<Policy>(kind, x, w, type, sigma, L, N, sdf, f, fw, &sig0);
  const Real h = 1e-6;
  double worst = 0;
  int checked = 0, skipped = 0, tried = 0;
  std::uniform_int_distribution<int> pick(0, N - 1);
  const int nDof = W ? 4 : 3;
  while (checked < nProbe && tried < 40 * nProbe) {
    ++tried;
    const int i = pick(rng), d = tried % nDof;
    const double an = (d < 3) ? f[3 * i + d] : fw[i];
    if (std::fabs(an) < 1e-6)
      continue;  // unresolvable by FD at this step
    std::vector<Real> xp = x, xm = x, wp = w, wm = w;
    if (d < 3) {
      xp[3 * i + d] += h;
      xm[3 * i + d] -= h;
    } else {
      wp[i] += h;
      wm[i] -= h;
    }
    std::vector<Real> fp, fwp;
    std::vector<int> sp, sm;
    const double Ep = energyForce<Policy>(kind, xp, wp, type, sigma, L, N, sdf, fp, fwp, &sp);
    const double Em = energyForce<Policy>(kind, xm, wm, type, sigma, L, N, sdf, fp, fwp, &sm);
    if (sp != sig0 || sm != sig0) {
      ++skipped;
      continue;
    }
    const double fd = (Ep - Em) / (2 * h);
    const double rel = std::fabs(fd - an) / std::max(std::fabs(an), 1e-30);
    if (rel > tolRel)
      std::printf("      probe cell %d dof %d: analytic=%.6e fd=%.6e rel=%.2e (E=%.6e)\n", i, d, an,
                  fd, rel, Ep);
    worst = std::max(worst, rel);
    ++checked;
  }
  // a term that acts on few cells (wetting: the contact line only) resolves fewer probes
  const bool ok = checked >= 3 && worst < tolRel;
  std::printf("  %-34s checked=%d skipped=%d worstRel=%.2e  %s\n", name, checked, skipped, worst,
              ok ? "OK" : "FAIL");
  return ok;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int bad = 0;
  {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1500;
    const Real L[3] = {1, 1, 1};
    std::mt19937 rng(7);
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> pos(3 * N), w(N);
    for (auto& v : pos)
      v = U(rng);
    const Real spacing = std::cbrt(1.0 / N);
    for (auto& v : w)
      v = U(rng) * (0.05 * spacing) * (0.05 * spacing);
    std::vector<int> type(N);
    for (int i = 0; i < N; ++i) {
      const Real dx = pos[3 * i] - 0.5, dy = pos[3 * i + 1] - 0.5, dz = pos[3 * i + 2] - 0.5;
      type[i] = (dx * dx + dy * dy + dz * dz < 0.3 * 0.3) ? 1 : 0;
    }
    std::printf("=== A3: published area Jacobians + energy layer (N=%d) ===\n", N);

    // ---- (A) published ∂A/∂n vs the per-cell reconstruction -------------------------------
    {
      DV dpos = up(pos, "pos"), dw;
      Kokkos::View<long*, Mem> gd;
      auto res = peclet::voro::buildTessellation<Real, false, NoSdf>(
          dpos, dw, N, L, 4, N, gd, NoSdf{}, true, -1, {}, {}, {}, {}, {}, {}, 0, nullptr, {},
          true);
      auto off = down(res.view.cellFacetOffset), cnt = down(res.view.cellFacetCount);
      auto nbr = down(res.view.facetNeighbor);
      auto area = down(res.view.facetArea);
      auto eoff = down(res.view.facetEdgeOffset), ecnt = down(res.view.facetEdgeCount);
      auto ef = down(res.view.edgeFacet);
      auto eg = down(res.view.edgeAreaGrad);
      auto st = down(res.status);
      using RCell = peclet::voro::ConvexCell<Real, 128, 256>;
      double worstG = 0, worstA = 0, scale = 0;
      long blocks = 0, missing = 0, extra = 0, cellsChecked = 0;
      const Real Lh = 0.5;
      for (int c = 0; c < N; c += 3) {
        if (st[c] != 0)
          continue;
        // reconstruct (the pre-A3 path)
        std::vector<Real> rx, ry, rz;
        std::vector<int> ids;
        std::vector<std::array<Real, 4>> nb;
        for (int f = off[c]; f < off[c] + cnt[c]; ++f) {
          const int j = nbr[f];
          if (j < 0 || j >= N)
            continue;
          Real r[3];
          for (int d = 0; d < 3; ++d) {
            Real rr = pos[3 * j + d] - pos[3 * c + d];
            rr = rr > Lh ? rr - 1 : (rr < -Lh ? rr + 1 : rr);
            r[d] = rr;
          }
          nb.push_back({r[0] * r[0] + r[1] * r[1] + r[2] * r[2], r[0], r[1], r[2]});
          ids.push_back(j);
        }
        const int M = (int)ids.size();
        std::vector<int> ord(M);
        for (int i = 0; i < M; ++i)
          ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return nb[a][0] < nb[b][0]; });
        rx.resize(M);
        ry.resize(M);
        rz.resize(M);
        std::vector<int> id2(M);
        for (int i = 0; i < M; ++i) {
          rx[i] = nb[ord[i]][1];
          ry[i] = nb[ord[i]][2];
          rz[i] = nb[ord[i]][3];
          id2[i] = ids[ord[i]];
        }
        RCell cell;
        peclet::voro::buildConvexCell(cell, L, rx.data(), ry.data(), rz.data(), id2.data(), M);
        if (cell.empty() || cell.overflow)
          continue;
        const int np = cell.np;
        std::vector<double> Ag(np, 0.0), dA((size_t)np * np * 3, 0.0);
        for (int t = 0; t < cell.nt; ++t) {
          if (!cell.alive[t])
            continue;
          int pl[3];
          double cb[3], gr[3][3][3];
          cell.geomVolumeAreaGrad(t, pl, cb, gr);
          for (int ii = 0; ii < 3; ++ii) {
            Ag[pl[ii]] += cb[ii];
            for (int jj = 0; jj < 3; ++jj)
              for (int cc = 0; cc < 3; ++cc)
                dA[((size_t)pl[ii] * np + pl[jj]) * 3 + cc] += gr[ii][jj][cc];
          }
        }
        // plane index of neighbour id j in the reconstructed cell
        auto planeOf = [&](int j) {
          for (int k = 0; k < np; ++k)
            if (cell.pnbr[k] == j)
              return k;
          return -1;
        };
        ++cellsChecked;
        for (int f = off[c]; f < off[c] + cnt[c]; ++f) {
          const int j = nbr[f];
          if (j < 0)
            continue;
          const int k = planeOf(j);
          if (k < 0) {
            ++missing;
            continue;
          }
          const double Apub =
              std::sqrt(area[3 * f] * area[3 * f] + area[3 * f + 1] * area[3 * f + 1] +
                        area[3 * f + 2] * area[3 * f + 2]);
          worstA = std::max(worstA, std::fabs(Apub - Ag[k]));
          scale = std::max(scale, Ag[k]);
          // every published block must match; every reconstruction block with nonzero norm that
          // is a face-face pair must be published
          std::vector<char> seen(np, 0);
          for (int e = eoff[f]; e < eoff[f] + ecnt[f]; ++e) {
            const int fp = ef[e];
            const int jp = nbr[fp];
            const int l = (jp < 0) ? -1 : planeOf(jp);
            if (l < 0) {
              ++missing;
              continue;
            }
            seen[l] = 1;
            for (int cc = 0; cc < 3; ++cc)
              worstG =
                  std::max(worstG, std::fabs(eg[3 * e + cc] - dA[((size_t)k * np + l) * 3 + cc]));
            ++blocks;
          }
          for (int l = 0; l < np; ++l) {
            if (seen[l] || cell.pnbr[l] < 0)
              continue;
            double nrm = 0;
            for (int cc = 0; cc < 3; ++cc)
              nrm += std::fabs(dA[((size_t)k * np + l) * 3 + cc]);
            if (nrm > 1e-12 * std::max(scale, 1e-30) && Ag[l] > 1e-12)
              ++extra;  // a face-face dependence the CSR does not carry
          }
        }
      }
      std::printf(
          "  (A) published dA/dn vs reconstruction: cells=%ld blocks=%ld worstAbs=%.2e "
          "(scale %.2e) |A| worst=%.2e missing=%ld unpublished=%ld  %s\n",
          cellsChecked, blocks, worstG, scale, worstA, missing, extra,
          (worstG < 1e-10 * scale && worstA < 1e-12 * scale && missing == 0 && extra == 0)
              ? "OK"
              : "FAIL");
      if (!(worstG < 1e-10 * scale && worstA < 1e-12 * scale && missing == 0 && extra == 0))
        bad = 1;
    }

    // ---- (B) interface gradient: published path vs reconstruction oracle + FD ---------------
    {
      std::vector<Real> f, fw;
      const double Enew = energyForce<Voronoi>(0, pos, w, type, 1.0, L, N, NoSdf{}, f, fw);
      std::vector<double> gOld;
      double Eold;
      peclet::voro::detail::interfaceGradReconstruct<Real>(pos, type, 1.0, L, N, 4, NoSdf{}, gOld,
                                                           Eold);
      double worst = 0, gmax = 0;
      for (int i = 0; i < 3 * N; ++i) {
        worst = std::max(worst, std::fabs(gOld[i] - (double)f[i]));
        gmax = std::max(gmax, std::fabs(gOld[i]));
      }
      const bool ok = std::fabs(Enew - Eold) < 1e-12 * Eold && worst < 1e-10 * gmax;
      std::printf(
          "  (B) interface: E new=%.10e old=%.10e  |g_new-g_old| worst=%.2e (gmax %.2e)  %s\n",
          Enew, Eold, worst, gmax, ok ? "OK" : "FAIL");
      if (!ok)
        bad = 1;
      if (!fdCheck<Voronoi>("(B) interface FD (Voronoi)", 0, pos, w, type, 1.0, L, N, NoSdf{}, rng,
                            12, 1e-5))
        bad = 1;
    }
    // ---- (C) power: positions + weights ------------------------------------------------------
    if (!fdCheck<Power>("(C) interface FD (Power x,w)", 0, pos, w, type, 1.0, L, N, NoSdf{}, rng,
                        12, 1e-5))
      bad = 1;
    // ---- (D) wall energy against a flat wall --------------------------------------------------
    {
      FlatWall wall{Real(0.3)};
      std::vector<Real> posW = pos;
      for (int i = 0; i < N; ++i)
        if (posW[3 * i] < 0.3 + 0.2 * spacing)
          posW[3 * i] = 0.3 + 0.2 * spacing + 0.8 * spacing * U(rng);  // keep seeds in the fluid
      std::vector<int> typeW(N);  // species 1 = a blob centred ON the wall (a sessile drop)
      for (int i = 0; i < N; ++i) {
        const Real dx = posW[3 * i] - 0.3, dy = posW[3 * i + 1] - 0.5, dz = posW[3 * i + 2] - 0.5;
        typeW[i] = (dx * dx + dy * dy + dz * dz < 0.25 * 0.25) ? 1 : 0;
      }
      {
        std::vector<Real> f, fw;
        const double Ew = energyForce<Voronoi>(1, posW, w, typeW, 1.0, L, N, wall, f, fw);
        double fmax = 0;
        for (Real v : f)
          fmax = std::max(fmax, (double)std::fabs(v));
        DV dpos = up(posW, "pos"), dw;
        Kokkos::View<long*, Mem> gd;
        auto res = peclet::voro::buildTessellation<Real, false, FlatWall>(
            dpos, dw, N, L, 4, N, gd, wall, true, -1, {}, {}, {}, {}, {}, {}, 0, nullptr, {}, true);
        auto nbr = down(res.view.facetNeighbor);
        long nWall = 0;
        for (int v : nbr)
          nWall += (v == peclet::voro::kBoundaryFacet);
        std::printf(
            "  (D) wall energy: E=%.6e max|f|=%.3e wall facets=%ld of %zu (area-grad edges %d)\n",
            Ew, fmax, nWall, nbr.size(), res.view.numEdges());
      }
      if (!fdCheck<Voronoi>("(D) wetting energy FD (flat wall)", 1, posW, w, typeW, 1.0, L, N, wall,
                            rng, 12, 1e-5))
        bad = 1;
      // ---- (E) volume target energy, with and without the wall ----------------------------
      if (!fdCheck<Voronoi>("(E) volume energy FD", 2, pos, w, type, 1.0, L, N, NoSdf{}, rng, 12,
                            1e-5))
        bad = 1;
      if (!fdCheck<Voronoi>("(E) volume energy FD (flat wall)", 2, posW, w, type, 1.0, L, N, wall,
                            rng, 12, 1e-5))
        bad = 1;
      // ---- (G) rung A1 force half: a CURVED wall (sphere, tangent clip) with the FD wall part;
      // the sagitta clip's polygon cross term is the open item (see test_sdf_policy (D2)) ------
      peclet::voro::TangentOnly<peclet::voro::SdfSphere<Real>> ball{
          {Real(0.5), Real(0.5), Real(0.5), Real(0.25)}};
      std::vector<Real> posS = pos;
      for (int i = 0; i < N; ++i) {  // keep seeds out of the ball (fluid only)
        Real d[3] = {posS[3 * i] - Real(0.5), posS[3 * i + 1] - Real(0.5),
                     posS[3 * i + 2] - Real(0.5)};
        const Real r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (r < 0.25 + 0.2 * spacing) {
          const Real rn = 0.25 + 0.2 * spacing + 0.8 * spacing * U(rng);
          for (int c = 0; c < 3; ++c)
            posS[3 * i + c] = Real(0.5) + d[c] / std::max(r, Real(1e-12)) * rn;
        }
      }
      std::vector<int> typeS(N);  // species 1 = a cap on the sphere (sessile drop, curved wall)
      for (int i = 0; i < N; ++i) {
        const Real dx = posS[3 * i] - 0.5, dy = posS[3 * i + 1] - 0.5, dz = posS[3 * i + 2] - 0.8;
        typeS[i] = (dx * dx + dy * dy + dz * dz < 0.2 * 0.2) ? 1 : 0;
      }
      if (!fdCheck<Voronoi>("(G) volume energy FD (sphere wall)", 2, posS, w, typeS, 1.0, L, N,
                            ball, rng, 12, 1e-5))
        bad = 1;
      if (!fdCheck<Voronoi>("(G) wetting energy FD (sphere wall)", 1, posS, w, typeS, 1.0, L, N,
                            ball, rng, 12, 1e-5))
        bad = 1;
    }
  }
  // ---- (F) the incremental path: reevalPublish(withAreaGrad) == cold build, cell by cell -------
  {
    const int N = 3000;
    const Real L[3] = {1, 1, 1};
    std::mt19937 rng(21);
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> pos(3 * N);
    for (auto& v : pos)
      v = U(rng);
    DV dpos = up(pos, "pos"), dw;
    Kokkos::View<long*, Mem> gd;
    auto cold = peclet::voro::buildTessellation<Real, false, NoSdf>(
        dpos, dw, N, L, 4, N, gd, NoSdf{}, true, -1, {}, {}, {}, {}, {}, {}, 0, nullptr, {}, true);
    peclet::voro::MovingTessellation<Real, 64, 112, false, NoSdf> mt;
    const Real spacing = std::cbrt(1.0 / N);
    mt.alloc(N, L, Real(1e-4) * spacing, Real(0.25) * spacing, 4, N);
    mt.rebuild(dpos);
    auto rp =
        peclet::voro::reevalPublish<Real, 64, 112>(mt.store, dpos, mt.vol, N, L, {}, {}, true);
    auto cOff = down(cold.view.cellFacetOffset), cCnt = down(cold.view.cellFacetCount);
    auto cNbr = down(cold.view.facetNeighbor);
    auto cEOff = down(cold.view.facetEdgeOffset), cECnt = down(cold.view.facetEdgeCount);
    auto cEF = down(cold.view.edgeFacet);
    auto cEG = down(cold.view.edgeAreaGrad);
    auto rOff = down(rp.cellFacetOffset), rCnt = down(rp.cellFacetCount);
    auto rNbr = down(rp.facetNeighbor);
    auto rEOff = down(rp.facetEdgeOffset), rECnt = down(rp.facetEdgeCount);
    auto rEF = down(rp.edgeFacet);
    auto rEG = down(rp.edgeAreaGrad);
    // Facets are matched by NEIGHBOUR id (the counting-sort grid can order a cell's planes
    // differently between two builds, which permutes the facet CSR within the cell without
    // changing it); edge partners likewise.
    long mism = 0, blocks = 0, mCnt = 0, mNbr = 0, mECnt = 0, mPart = 0;
    double worst = 0, scale = 0;
    for (int i = 0; i < N; ++i) {
      if (cCnt[i] != rCnt[i]) {
        ++mism;
        ++mCnt;
        continue;
      }
      std::map<int, int> rOf;  // nbr -> reeval facet index
      for (int k = 0; k < rCnt[i]; ++k)
        rOf[rNbr[rOff[i] + k]] = rOff[i] + k;
      for (int k = 0; k < cCnt[i]; ++k) {
        const int fc = cOff[i] + k;
        auto it = rOf.find(cNbr[fc]);
        if (it == rOf.end()) {
          ++mism;
          ++mNbr;
          continue;
        }
        const int fr = it->second;
        if (cECnt[fc] != rECnt[fr]) {
          ++mism;
          ++mECnt;
          continue;
        }
        std::map<int, int> rSlot;  // partner nbr -> reeval edge index
        for (int e = 0; e < rECnt[fr]; ++e)
          rSlot[rNbr[rEF[rEOff[fr] + e]]] = rEOff[fr] + e;
        for (int e = 0; e < cECnt[fc]; ++e) {
          const int ec = cEOff[fc] + e;
          auto jt = rSlot.find(cNbr[cEF[ec]]);
          if (jt == rSlot.end()) {
            ++mism;
            ++mPart;
            continue;
          }
          const int er = jt->second;
          for (int cc = 0; cc < 3; ++cc) {
            worst = std::max(worst, std::fabs(cEG[3 * ec + cc] - rEG[3 * er + cc]));
            scale = std::max(scale, std::fabs(cEG[3 * ec + cc]));
          }
          ++blocks;
        }
      }
    }
    const bool ok = mism == 0 && blocks > 0 && worst < 1e-10 * scale;
    std::printf(
        "  (F) reevalPublish area-Jacobian CSR == cold build: blocks=%ld mismatches=%ld "
        "(facetCount %ld, nbr %ld, edgeCount %ld, partner %ld) worstAbs=%.2e (scale %.2e)  %s\n",
        blocks, mism, mCnt, mNbr, mECnt, mPart, worst, scale, ok ? "OK" : "FAIL");
    if (!ok)
      bad = 1;
  }
  Kokkos::finalize();
  std::printf(bad ? "VORO-ENERGY FAIL\n" : "VORO-ENERGY OK\n");
  return bad;
}
