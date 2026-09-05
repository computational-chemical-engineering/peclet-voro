/**
 * @file test_flow_mpi.cpp
 * @brief Track C, rung C5 gate (Voronoi methods plan): the distributed collocated solver.
 *
 * A jittered 12^3 lattice in the periodic unit box, decomposed by VoronoiHalo's ORB; every rank
 * builds (a) its owned cells with the gathered ghosts and steps the distributed CollocatedNS
 * (GhostExchange hooks), and (b) the whole problem single-rank as the reference. Decaying
 * Taylor–Green, 20 SSP-RK3 steps at CFL 0.2. Gates (the flow standard):
 *   np = 1: bit-exact to the single-rank reference (same code path, no ghosts; OMP_NUM_THREADS=1
 *           keeps the face CSR order deterministic);
 *   np = 2, 4: owned cell velocities within 1e-12 of the reference (measured 3e-15: the
 *           block-Jacobi AMG changes only the PCG iterates), the global kinetic energy within
 *           1e-13, the face divergence ≤ 1e-11 on every rank, no cell flagged;
 *   (A) the distributed pressure solve in isolation: true residual == recursive residual, K·1 = 0,
 *       K symmetric, ghost order consistent.
 * Usage: mpirun -np P test_flow_mpi [n=12] [steps=20]
 */
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <map>
#include <random>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/fv/collocated.hpp"
#include "peclet/voro/fv/distributed.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/mpi/voronoi_halo.hpp"
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
static void tgv(Real x, Real y, Real& u, Real& v) {
  const Real k = 2 * M_PI;
  u = std::sin(k * x) * std::cos(k * y);
  v = -std::cos(k * x) * std::sin(k * y);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    int rank = 0, np = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    const int n = argc > 1 ? std::atoi(argv[1]) : 12;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 20;
    const Real L = 1, h = L / n, nu = 0.01, dt = 0.2 * h;
    const int N = n * n * n;
    // identical global seed set on every rank
    std::vector<Real> gpos(3 * N);
    {
      std::mt19937 rng(77);
      std::uniform_real_distribution<Real> J(-0.2 * h, 0.2 * h);
      int i = 0;
      for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
          for (int x = 0; x < n; ++x, ++i) {
            gpos[3 * i] = std::fmod((x + 0.5) * h + J(rng) + L, L);
            gpos[3 * i + 1] = std::fmod((y + 0.5) * h + J(rng) + L, L);
            gpos[3 * i + 2] = std::fmod((z + 0.5) * h + J(rng) + L, L);
          }
    }
    std::vector<Real> U0g(3 * N, 0.0);
    for (int i = 0; i < N; ++i)
      tgv(gpos[3 * i], gpos[3 * i + 1], U0g[3 * i], U0g[3 * i + 1]);
    const Real Lb[3] = {L, L, L};

    // ---- reference: the whole problem on this rank -------------------------------------------
    const bool covol = std::getenv("FLOW_MPI_COVOLUME") != nullptr;
    std::vector<Real> Uref;
    Real Eref = 0;
    if (covol) {  // covolume reference: face fluxes, compared through the Perot cell velocity
      DV dpos = up(gpos, "pos"), dw;
      Kokkos::View<long*, Mem> gd;
      auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(
          dpos, dw, N, Lb, 4, N, gd, {}, true);
      auto aux = peclet::voro::buildAuxMaps(res.view);
      auto m = fv::buildFaceMesh(res.view, aux);
      fv::CovolumeNS<Real> cv;
      cv.setup(m, nu, true);
      cv.poisson.tol = 1e-12;
      fv::projectToFaces(m, up(U0g, "U0"), cv.u);
      cv.project(cv.u, 1.0);
      for (int s = 0; s < steps; ++s)
        cv.step(dt);
      DV Uc("Uc", 3 * N);
      fv::perotVelocity(m, cv.u, Uc);
      Uref = down(Uc);
      Eref = cv.kineticEnergy();
    } else {
      DV dpos = up(gpos, "pos"), dw;
      Kokkos::View<long*, Mem> gd;
      auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(
          dpos, dw, N, Lb, 4, N, gd, {}, true);
      auto aux = peclet::voro::buildAuxMaps(res.view);
      auto m = fv::buildFaceMesh(res.view, aux);
      fv::CollocatedNS<Real> co;
      co.setup(m, nu, true);
      if (std::getenv("FLOW_MPI_SKEW") && std::atoi(std::getenv("FLOW_MPI_SKEW")) == 0)
        co.skewCorrected = false;
      co.poisson.tol = 1e-12;
      co.implicitDiffusion = std::getenv("FLOW_MPI_IMPLICIT") != nullptr;
      co.initialize(up(U0g, "U0"));
      for (int s = 0; s < steps; ++s)
        co.step(dt);
      Uref = down(co.U);
      Eref = co.kineticEnergy();
    }

    // ---- distributed --------------------------------------------------------------------------
    peclet::voro::mpi::VoronoiHalo<Real> halo;
    halo.init({0, 0, 0}, {L, L, L}, {4, 4, 4}, {true, true, true}, MPI_COMM_WORLD);
    std::vector<std::array<Real, 3>> ownedPos;
    std::vector<long> ownedGid;
    std::vector<Real> ownedW;
    for (int i = 0; i < N; ++i) {
      std::array<Real, 3> x{gpos[3 * i], gpos[3 * i + 1], gpos[3 * i + 2]};
      if (halo.ownerOf(x) == rank) {
        ownedPos.push_back(x);
        ownedGid.push_back(i);
        ownedW.push_back(0);
      }
    }
    const Real rcut = 4.0 * h;  // beyond the reach of any jittered-lattice cell
    auto g = halo.gather(ownedPos, ownedGid, ownedW, rcut);
    const int nOwned = g.nOwned, nComb = (int)g.pos.size();
    std::vector<Real> cpos(3 * nComb), U0(3 * nComb, 0.0);
    for (int i = 0; i < nComb; ++i)
      for (int c = 0; c < 3; ++c)
        cpos[3 * i + c] = g.pos[i][c];
    for (int i = 0; i < nComb; ++i)  // the initial field everywhere (ghosts included)
      tgv(cpos[3 * i], cpos[3 * i + 1], U0[3 * i], U0[3 * i + 1]);
    DV dpos = up(cpos, "cpos"), dw;
    Kokkos::View<long*, Mem> gd;
    auto res = peclet::voro::buildTessellation<Real, false, peclet::voro::NoSdf>(
        dpos, dw, nComb, Lb, 4, N, gd, {}, true, nOwned);
    int flagged = 0;
    {
      auto st = down(res.status);
      for (int i = 0; i < nOwned; ++i)
        flagged += st[i] != 0 ? 1 : 0;
    }
    auto aux = peclet::voro::buildAuxMaps(res.view);
    auto m = fv::buildFaceMesh(res.view, aux, nOwned);
    fv::GhostExchange<Real> ex;
    ex.init(halo, nOwned);
    {  // the device-packed exchange must equal the host reference path bitwise
      std::vector<Real> rh(3 * nComb);
      std::mt19937 rr(99 + rank);
      std::uniform_real_distribution<Real> Ur(-1, 1);
      for (int i = 0; i < 3 * nOwned; ++i)
        rh[i] = Ur(rr);
      DV fa = up(rh, "fa"), fb = up(rh, "fb");
      ex.devicePack = true;
      ex.exchange(fa, 3);
      ex.devicePack = false;
      ex.exchange(fb, 3);
      ex.devicePack = true;
      auto ha = down(fa), hb = down(fb);
      int diff = 0;
      for (int i = 0; i < 3 * nComb; ++i)
        diff += ha[i] != hb[i];
      int diffG = 0;
      MPI_Allreduce(&diff, &diffG, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (rank == 0)
        std::printf("  (X) device-packed exchange == host path: %d differing entries  %s\n", diffG,
                    diffG == 0 ? "OK" : "FAIL");
      if (diffG != 0)
        bad = 1;
    }
    Real E = 0, divMax = 0;
    std::vector<Real> Ud;
    int pcgIters = 0;
    fv::CollocatedNS<Real> co;
    const bool useAmg =
        std::getenv("FLOW_MPI_AMG") == nullptr || std::atoi(std::getenv("FLOW_MPI_AMG")) != 0;
    bool solveOk = true;
    fv::CovolumeNS<Real> cvd;
    if (covol) {
      cvd.setup(m, nu, useAmg);
      cvd.setExchange(ex);
      cvd.poisson.tol = 1e-12;
      DV U0d = up(U0, "U0");
      if (true)
        ex.exchange(U0d, 3);
      fv::projectToFaces(m, U0d, cvd.u);
      cvd.project(cvd.u, 1.0);
      for (int s = 0; s < steps; ++s)
        cvd.step(dt);
      E = cvd.kineticEnergy();
      divMax = cvd.maxDivergence();
      DV Uc("Uc", 3 * (m.nCombined > 0 ? m.nCombined : m.nCells));
      fv::perotVelocity(m, cvd.u, Uc);
      Ud = down(Uc);
      pcgIters = cvd.poisson.lastIters;
    }
    co.setup(m, nu, useAmg);
    if (!covol) {
      co.setExchange(ex);
      if (std::getenv("FLOW_MPI_SKEW") && std::atoi(std::getenv("FLOW_MPI_SKEW")) == 0)
        co.skewCorrected = false;
      co.poisson.tol = 1e-12;
      co.implicitDiffusion = std::getenv("FLOW_MPI_IMPLICIT") != nullptr;
      {  // (A) the distributed pressure solve in isolation: random right-hand side — the recursive
         // residual must equal the TRUE residual (b − K p with the ghosts of p refreshed), K·1 = 0,
         // K symmetric, the ghost order consistent. (This gate found the rank-local total volume in
         // the mean deflation: with it the deflated RHS is not globally mean-free and CG diverges.)
        std::vector<Real> fh(nComb, 0.0);
        std::mt19937 rng(5 + rank);
        std::uniform_real_distribution<Real> Ur(-1, 1);
        for (int i = 0; i < nOwned; ++i)
          fh[i] = Ur(rng);
        DV f = up(fh, "f"), pp("pp", nComb), q("q", nOwned);
        co.poisson.solve(f, pp);
        co.poisson.applyK(pp, q);
        auto qh = down(q), vol = down(m.cellVolume), ph = down(pp);
        double fm = 0, vt = 0;
        for (int i = 0; i < nOwned; ++i) {
          fm += vol[i] * fh[i];
          vt += vol[i];
        }
        double fmG = 0, vtG = 0;
        MPI_Allreduce(&fm, &fmG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&vt, &vtG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        fmG /= vtG;
        double r2 = 0, b2 = 0;
        for (int i = 0; i < nOwned; ++i) {
          const double b = vol[i] * (fh[i] - fmG), r = b - qh[i];
          r2 += r * r;
          b2 += b * b;
        }
        double r2G = 0, b2G = 0;
        MPI_Allreduce(&r2, &r2G, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&b2, &b2G, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // are the ghost p values equal to the owners' values? compare through a second exchange
        DV pp2("pp2", nComb);
        Kokkos::deep_copy(pp2, pp);
        ex.exchange(pp2, 1);
        auto p2 = down(pp2);
        double gd = 0;
        for (int j = nOwned; j < nComb; ++j)
          gd = std::max(gd, std::fabs(p2[j] - ph[j]));
        // ghost order: forward the owned gids and compare with the gathered ghost gids
        std::vector<double> og(nOwned), gg(nComb - nOwned);
        for (int i = 0; i < nOwned; ++i)
          og[i] = (double)ownedGid[i];
        halo.forward(og.data(), gg.data());
        int badGid = 0;
        for (int j = 0; j < nComb - nOwned; ++j)
          badGid += ((long)gg[j] != g.gid[nOwned + j]) ? 1 : 0;
        // symmetry of K: <d1, K d2> vs <d2, K d1> (global)
        std::vector<Real> d1h(nComb), d2h(nComb);
        for (int i = 0; i < nOwned; ++i) {
          d1h[i] = Ur(rng);
          d2h[i] = Ur(rng);
        }
        DV d1 = up(d1h, "d1"), d2 = up(d2h, "d2"), k1("k1", nOwned), k2("k2", nOwned);
        ex.exchange(d1, 1);
        ex.exchange(d2, 1);
        co.poisson.applyK(d1, k1);
        co.poisson.applyK(d2, k2);
        auto k1h = down(k1), k2h = down(k2);
        double s12 = 0, s21 = 0;
        for (int i = 0; i < nOwned; ++i) {
          s12 += d1h[i] * k2h[i];
          s21 += d2h[i] * k1h[i];
        }
        double s12G = 0, s21G = 0;
        MPI_Allreduce(&s12, &s12G, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&s21, &s21G, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double k1Gmax = 0;
        {  // K·1 (must vanish) and the Rayleigh quotient of random vectors (must be ≥ 0)
          std::vector<Real> oneh(nComb, 1.0);
          DV one = up(oneh, "one"), k1v("k1v", nOwned);
          co.poisson.applyK(one, k1v);
          auto kh = down(k1v);
          double k1max = 0;
          for (int i = 0; i < nOwned; ++i)
            k1max = std::max(k1max, std::fabs(kh[i]));
          double k1G = 0;
          MPI_Allreduce(&k1max, &k1G, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
          double rqMin = 1e30;
          for (int t = 0; t < 5; ++t) {
            std::vector<Real> dh(nComb, 0.0);
            for (int i = 0; i < nOwned; ++i)
              dh[i] = Ur(rng);
            DV dv = up(dh, "dv"), kv("kv", nOwned);
            ex.exchange(dv, 1);
            co.poisson.applyK(dv, kv);
            auto kvh = down(kv);
            double num = 0, den = 0;
            for (int i = 0; i < nOwned; ++i) {
              num += dh[i] * kvh[i];
              den += dh[i] * dh[i];
            }
            double numG = 0, denG = 0;
            MPI_Allreduce(&num, &numG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&den, &denG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            rqMin = std::min(rqMin, numG / denG);
          }
          k1Gmax = k1G;
          if (rank == 0 && std::getenv("FLOW_MPI_TRACE"))
            std::printf(
                "    trace: |K 1|_max = %.3e, min Rayleigh quotient of 5 random vectors %.3e\n",
                k1G, rqMin);
        }
        // interface neighbour indices in range?
        auto Bf = down(m.faceCellB);
        int badB = 0;
        for (int f = 0; f < m.nInterior; ++f)
          badB += (Bf[f] < 0 || Bf[f] >= nComb) ? 1 : 0;
        {  // mesh conditioning: face distances / coefficients, interior vs interface
          auto dfh = down(m.faceDist), Afh = down(m.faceArea);
          double dmin[2] = {1e9, 1e9}, cmax[2] = {0, 0};
          int cntI[2] = {0, 0};
          for (int f = 0; f < m.nInterior; ++f) {
            const int k = Bf[f] >= nOwned ? 1 : 0;
            dmin[k] = std::min(dmin[k], (double)dfh[f]);
            cmax[k] = std::max(cmax[k], (double)(Afh[f] / dfh[f]));
            ++cntI[k];
          }
          if (std::getenv("FLOW_MPI_TRACE"))
            std::printf(
                "    rank %d: owned-owned faces %d (min d %.3e, max A/d %.3e), interface faces %d "
                "(min d %.3e, max A/d %.3e), sym diff %.3e\n",
                rank, cntI[0], dmin[0], cmax[0], cntI[1], dmin[1], cmax[1], s12G - s21G);
          // per owned cell: number of faces and the diagonal Σ A/d; the weakest cells
          auto off = down(m.cellFaceOffset), cf = down(m.cellFace);
          std::vector<double> diag(nOwned, 0.0);
          int minF = 1 << 30;
          for (int i = 0; i < nOwned; ++i) {
            minF = std::min(minF, off[i + 1] - off[i]);
            for (int t = off[i]; t < off[i + 1]; ++t) {
              const int f = cf[t];
              if (f < m.nInterior)
                diag[i] += Afh[f] / dfh[f];
            }
          }
          std::vector<double> sd = diag;
          std::sort(sd.begin(), sd.end());
          if (std::getenv("FLOW_MPI_TRACE"))
            std::printf(
                "    rank %d: dropped facets %d, min faces/cell %d, diag min %.3e median %.3e "
                "max %.3e; #cells with diag < 1e-6*median: %d\n",
                rank, m.nDropped, minF, sd[0], sd[nOwned / 2], sd[nOwned - 1],
                (int)std::count_if(sd.begin(), sd.end(),
                                   [&](double v) { return v < 1e-6 * sd[nOwned / 2]; }));
        }
        const double trueRes = std::sqrt(r2G / b2G);
        solveOk = trueRes < 1e-10 && badGid == 0 && badB == 0 &&
                  std::fabs(s12G - s21G) < 1e-12 * std::fabs(s12G) && k1Gmax < 1e-12;
        if (rank == 0)
          std::printf(
              "  (A) distributed pressure solve: %d PCG iterations, recursive residual %.2e, "
              "TRUE residual %.2e; |K 1| %.1e, symmetry %.1e, ghost gid mismatches %d  %s\n",
              co.poisson.lastIters, co.poisson.lastRes, trueRes, k1Gmax, s12G - s21G, badGid,
              solveOk ? "OK" : "FAIL");
      }
      co.initialize(up(U0, "U0"));
      const bool trace = std::getenv("FLOW_MPI_TRACE") != nullptr;
      if (trace) {  // collective (the max hook all-reduces)
        const Real d0 = co.maxFaceDivergence();
        if (rank == 0)
          std::printf("    trace: after initialize div %.2e (iters %d res %.1e)\n", d0,
                      co.poisson.lastIters, co.poisson.lastRes);
      }
      for (int s = 0; s < steps; ++s) {
        co.step(dt);
        if (trace && s < 3) {
          const Real ds = co.maxFaceDivergence();
          if (rank == 0)
            std::printf("    trace: after step %d div %.2e (iters %d res %.1e)\n", s, ds,
                        co.poisson.lastIters, co.poisson.lastRes);
        }
      }
      {  // DIAGNOSTIC: the two copies of each interface flux, and the global mean divergence
        auto A = down(m.faceCellA), B = down(m.faceCellB);
        auto uf = down(covol ? cvd.u : co.uf), Af = down(m.faceArea), vol = down(m.cellVolume);
        DV dvv("dvv", m.nCells);
        fv::divergence(m, covol ? cvd.u : co.uf, dvv);
        auto dh = down(dvv);
        double sv = 0;
        for (int i = 0; i < m.nCells; ++i)
          sv += vol[i] * dh[i];
        double svG = 0;
        MPI_Allreduce(&sv, &svG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        std::vector<double> rec;  // gidA, gidB, flux*area (A->B positive)
        for (int f = 0; f < m.nInterior; ++f)
          if (B[f] >= nOwned) {
            rec.push_back((double)g.gid[A[f]]);
            rec.push_back((double)g.gid[B[f]]);
            rec.push_back(uf[f] * Af[f]);
          }
        int cnt = (int)rec.size();
        std::vector<int> cnts(np), disp(np);
        MPI_Gather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        int tot = 0;
        for (int r = 0; r < np; ++r) {
          disp[r] = tot;
          tot += cnts[r];
        }
        std::vector<double> all(rank == 0 ? tot : 0);
        MPI_Gatherv(rec.data(), cnt, MPI_DOUBLE, all.data(), cnts.data(), disp.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
          std::map<std::pair<long, long>, double> flux;
          double mism = 0, fmax = 0;
          int pairs = 0;
          for (int k = 0; k < tot; k += 3) {
            const long ga = (long)all[k], gb = (long)all[k + 1];
            const double F = all[k + 2];
            fmax = std::max(fmax, std::fabs(F));
            auto key = std::make_pair(std::min(ga, gb), std::max(ga, gb));
            const double Fo = ga < gb ? F : -F;  // oriented low -> high
            auto it = flux.find(key);
            if (it == flux.end()) {
              flux[key] = Fo;
            } else {
              mism = std::max(mism, std::fabs(it->second - Fo));
              ++pairs;
            }
          }
          if (std::getenv("FLOW_MPI_TRACE"))
            std::printf(
                "    DIAG: interface faces %d (paired %d), max |flux_A - flux_B| = %.2e (max "
                "|flux| %.2e), global sum V div = %.2e\n",
                tot / 3, pairs, mism, fmax, svG);
        }
      }
      E = co.kineticEnergy();
      divMax = co.maxFaceDivergence();
      Ud = down(co.U);
      pcgIters = co.poisson.lastIters;
    }
    double dmax = 0, umax = 0;
    for (int i = 0; i < nOwned; ++i) {
      const long gi = ownedGid[i];
      for (int c = 0; c < 3; ++c) {
        dmax = std::max(dmax, std::fabs(Ud[3 * i + c] - Uref[3 * gi + c]));
        umax = std::max(umax, std::fabs(Uref[3 * gi + c]));
      }
    }
    double dmaxG = 0, umaxG = 0;
    int flaggedG = 0, ghostsG = 0, nGh = halo.numGhost();
    MPI_Allreduce(&dmax, &dmaxG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&umax, &umaxG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&flagged, &flaggedG, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nGh, &ghostsG, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const double rel = dmaxG / umaxG, eRel = std::fabs(E / Eref - 1);
    // MEASURED (2026-09-03): np=2/4 agree with the single-rank reference to 3e-15 (velocity) and
    // 2e-16 (energy) — the block-Jacobi AMG only changes the PCG iterates (28/36 vs 11 iterations)
    // implicit variant (FLOW_MPI_IMPLICIT): three velocity PCG solves per step at 1e-12 with the
    // block-Jacobi preconditioner — measured 2e-11 / 2e-14 at np = 2/4
    const bool impl = std::getenv("FLOW_MPI_IMPLICIT") != nullptr;
    // np = 1 is bit-exact to the single-rank reference on the HOST backends (OpenMP / Serial:
    // identical kernels, identical summation order). On a DEVICE backend the two runs differ at
    // round-off and are not even reproducible run-to-run (measured on CUDA, 2026-09-05: 7.2e-16 /
    // 8.7e-16 in two runs of the same binary; the tessellator's facet CSR is assembled with
    // atomics, so per-cell sums land in a different order each launch) — the suite-wide policy
    // (core tests, 2026-07-23): bit-exact on OpenMP/Serial, round-off tolerance on CUDA/HIP.
    const bool deviceBackend =
        !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
    const double tolU = np == 1 ? (deviceBackend ? 1e-13 : 0.0) : (impl ? 1e-9 : 1e-12),
                 tolE = np == 1 ? (deviceBackend ? 1e-14 : 0.0) : (impl ? 1e-12 : 1e-13);
    const bool ok = solveOk && rel <= tolU && eRel <= tolE && divMax < 1e-11 && flaggedG == 0;
    if (rank == 0)
      std::printf(
          "  (B) C5 distributed collocated TGV (%s): np=%d n=%d steps=%d ghosts=%d | max |U - "
          "U_ref| / "
          "max|U| = %.3e (gate %.0e), E/E_ref - 1 = %.3e (gate %.0e), max face div %.1e, flagged "
          "cells %d, PCG iters %d  %s\n",
          covol ? "covolume RK3" : (impl ? "implicit diffusion" : "RK3"), np, n, steps, ghostsG,
          rel, tolU, eRel, tolE, divMax, flaggedG, co.poisson.lastIters, ok ? "OK" : "FAIL");
    if (rank == 0)
      std::printf("    (last pressure solve: %d iterations, relative residual %.2e, amg %d)\n",
                  co.poisson.lastIters, co.poisson.lastRes, (int)useAmg);
    if (!ok)
      bad = 1;
  }
  Kokkos::finalize();
  MPI_Finalize();
  return bad;
}
