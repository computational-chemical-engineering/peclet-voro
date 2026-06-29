/**
 * @file device/repair.hpp
 * \brief Part II, Phase 2 — the two-pass gather repair (the moving-point per-step update).
 *
 * `MovingTessellation` keeps the resident topology + geometry of a moving point set and updates it each
 * step WITHOUT a full cold rebuild, falling back to the cold build only when repair cannot close. Per
 * step (`step(pos)`), following dynamic_update_decision_and_plan.md §1/§3:
 *
 *   1. build the per-step grid (buildTessGrid — cheap, memory-bound);
 *   2. re-evaluate every cell on the new positions over its stored topology (ConvexCell::reevalGeometry)
 *      and run the convexity/in-sphere certificate, collecting per cell: its own flag (lost-face), the
 *      violated-plane partner seeds (the gaining cells of a flip), and the Verlet-skin movers (the
 *      insertion trigger the certificate cannot see);
 *   3. **Pass 1** — subset-gather (cold-build kernel over an index list) the set
 *      {flagged ∪ partners ∪ skin-movers}: those cells get exact, freshly-clipped topology + volume;
 *   4. **Pass 2** — subset-gather the NEW face-neighbours of the Pass-1 cells that were not themselves
 *      rebuilt (chiefly a far-mover's new neighbours, which never flag — §1c);
 *   5. **verify** — re-run the certificate; if clean, publish; else a bounded number of extra
 *      certificate-driven passes, then fall back to a full cold rebuild (always exact).
 *
 * For separated events this is exact in two passes; the verify + bounded extra passes + rebuild
 * fallback make it exact in general. No adaptive per-backend gate, no single-face repair, no
 * ConnectivityArena, no SDF boundary trigger, no BVH (those are later phases). One Kokkos path for
 * CUDA/HIP/OpenMP/Serial. Voronoi only (Power deferred — see the static_asserts downstream).
 *
 * Single-domain: the stored `pnbr` are local indices (== global), stable across steps. The distributed
 * (MPI) driver keeps its store keyed by global id and remaps around this primitive each step.
 */
#ifndef VORFLOW_DEVICE_REPAIR_HPP
#define VORFLOW_DEVICE_REPAIR_HPP

#include <string>
#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"
#include "vorflow/device/convex_cell.hpp"
#include "vorflow/device/subset_gather.hpp"
#include "vorflow/device/tess_grid.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/device/topology_store.hpp"

namespace vor {
namespace device {

/// Per-step repair telemetry.
struct RepairStats {
  /// Which path the Phase-3 adaptive gate took this step.
  enum Route { kTwoPass = 0, kDilated = 1, kRebuildGate = 2 };
  int pass1 = 0;       ///< cells gathered in Pass 1 (flagged ∪ partners ∪ skin-movers, after dilation)
  int pass1Raw = 0;    ///< flagged count BEFORE dilation (the gate signal: pass1Raw/nProc = churn)
  int pass2 = 0;       ///< cells gathered in Pass 2 (new face-neighbours of the movers)
  int extra = 0;       ///< cells gathered across the verify extra-passes
  int surgical = 0;    ///< Pass-1 cells repaired surgically (Phase 4, no grid gather)
  int verifyPasses = 0;///< number of verify iterations run
  Route route = kTwoPass;
  bool fellBack = false;///< true if the cold-build fallback was triggered (repair did not close)
};

/// Resident moving-point tessellation with two-pass gather repair. MAXP/MAXT must match
/// CellBuilder::kMaxP / kMaxT (64 / 112).
template <class Real, int MAXP = 64, int MAXT = 112>
struct MovingTessellation {
  using Mem = tpx::MemSpace;
  using Exec = tpx::ExecSpace;
  using Cell = ConvexCell<Real, MAXP, MAXT>;
  using Store = TopologyStore<MAXP, MAXT>;

  int N = 0;       ///< total cells in the resident arrays (single-domain: all; MPI: owned+ghost)
  int nProc = 0;   ///< cells this instance MAINTAINS (single-domain: N; MPI: owned, [0,nProc)). The grid
                   ///< + gather candidates always span all N (ghosts are cut candidates, not maintained).
  int sw = 4, densityCount = -1;
  Real L[3] = {1, 1, 1};
  Real tol = 0;   ///< certificate tolerance (absolute distance); ~1e-4·spacing FP64, ~2e-3·spacing FP32
  Real skin = 0;  ///< Verlet skin width (absolute); a particle moving > skin/2 is a Pass-1 mover
  int verifyCap = 2;  ///< max verify extra-passes before falling back to a full rebuild

  // ---- Phase 3: adaptive three-way gate (decided after the free certificate, before any gather) ----
  bool useGate = true;        ///< high-churn → rebuild routing (the "never slower than rebuild" guard)
  bool useDilation = false;   ///< dense-cluster regional dilation branch. DEFAULT OFF: on Poisson/lattice
                              ///< workloads there are no genuine clusters, the verify loop already closes
                              ///< the rare cascade, and the per-step count/mark scan is pure overhead. Turn
                              ///< on for strongly clustered inputs (the regime it is meant for).
  double churnThresh = 0.55;  ///< flagged-fraction above which a full rebuild is cheaper than repairing
                              ///< (per-backend, set in alloc — GPU rebuild is cheap so it gives up sooner).
  double dilateMaxChurn = 0.15;///< only consider dilation when GLOBAL churn is low (a real local cluster,
                              ///< not a uniformly-moderate field where every neighbourhood looks "dense").
  int clusterNbhd = 18;       ///< a cell whose 27-grid-cell neighbourhood has > this many flagged seeds is
                              ///< "dense" (≈ a region far above the field average) and gets dilated.
  // ---- Phase 4: single-face surgical repair (no grid gather; gated, default OFF — see note) ----
  // NEGATIVE RESULT (measured): on this engine surgical re-clip is BOTH slower and not robustly exact,
  // so it stays off. (1) Slower: the cold gather clips closest-first with the security-radius early-out
  // (the cell tightens fast, few live triangles per clip), whereas re-clipping the stored+partner
  // candidates unsorted keeps the cell large and costs more O(#tri) horizon scans — the gather wins.
  // (2) Not exact: a surgically-rebuilt cell that GAINS a neighbour absent from its candidate set stays
  // convex, so the certificate can't see the gain (§1c) and the verify loop can't catch it; the small
  // per-step error then COMPOUNDS in the resident store (maxRelV grew to ~0.5 over a sweep). Making it
  // exact needs the maintained 1-/2-ring candidate set (the ConnectivityArena the plan parks). Kept as
  // the implemented experiment behind this flag; the production path is the Phase-3 gated two-pass gather.
  bool surgical = false;     ///< repair flip cells by re-clipping known candidates instead of gathering
  static constexpr int kExtraCap = 8;  ///< per-cell capacity for partner-discovered new neighbours

  Store store;
  Kokkos::View<Real*, Mem> vol;   // N : cell volumes (the published geometry scalar)
  Kokkos::View<Real*, Mem> xRef;  // 3N : positions at the last (re)build (Verlet reference)
  // scratch
  Kokkos::View<int*, Mem> mask, mask2, mover, rebuilt, wl1, wl2, wlM;
  Kokkos::View<int*, Mem> cellFlag;        // per linear grid-cell flagged-seed count (dilation)
  Kokkos::View<int*, Mem> extraNbr, extraCnt;  // Phase-4 partner-discovered candidate neighbours
  Kokkos::View<int, Mem> counter;
  int gridNcell = 0;

  /// @param nProc_  cells to maintain (default n = single-domain). For the distributed driver pass the
  ///                owned count: cells [0,nProc) are repaired, [nProc,N) are ghost cut-candidates only.
  void alloc(int n, const Real Lbox[3], Real tol_, Real skin_, int sw_ = 4, int densityCount_ = -1,
             int nProc_ = -1) {
    N = n; nProc = (nProc_ < 0) ? n : nProc_;
    sw = sw_; densityCount = densityCount_ < 0 ? n : densityCount_;
    tol = tol_; skin = skin_;
    for (int k = 0; k < 3; ++k) L[k] = Lbox[k];
    store.alloc(n);
    using Kokkos::view_alloc; using Kokkos::WithoutInitializing;
    vol = Kokkos::View<Real*, Mem>("mt.vol", n);
    xRef = Kokkos::View<Real*, Mem>(view_alloc(std::string("mt.xRef"), WithoutInitializing), (size_t)n * 3);
    mask = Kokkos::View<int*, Mem>("mt.mask", n);
    mask2 = Kokkos::View<int*, Mem>("mt.mask2", n);
    mover = Kokkos::View<int*, Mem>("mt.mover", n);
    rebuilt = Kokkos::View<int*, Mem>("mt.rebuilt", n);
    wl1 = Kokkos::View<int*, Mem>(view_alloc(std::string("mt.wl1"), WithoutInitializing), n);
    wl2 = Kokkos::View<int*, Mem>(view_alloc(std::string("mt.wl2"), WithoutInitializing), n);
    wlM = Kokkos::View<int*, Mem>(view_alloc(std::string("mt.wlM"), WithoutInitializing), n);
    extraNbr = Kokkos::View<int*, Mem>(view_alloc(std::string("mt.extraNbr"), WithoutInitializing),
                                       (size_t)n * kExtraCap);
    extraCnt = Kokkos::View<int*, Mem>("mt.extraCnt", n);
    counter = Kokkos::View<int, Mem>("mt.counter");
    // Per-backend gate threshold: rebuild is cheap + clip-bound on GPU (repair loses past ~50% flagged),
    // expensive on the CPU paths (repair pays out to ~85%). Hand-set from the cross-device sweep; the
    // plan's Phase-3 tuning would fit these from logged (signals vs verify-clean) per backend.
    constexpr bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, Mem>::accessible;
    churnThresh = host ? 0.70 : 0.50;
  }

  /// Full cold (re)build: the production tessellator emitting the resident topology + volume, then
  /// reset the Verlet reference. This is the oracle, the fallback, and the initial build.
  void rebuild(const Kokkos::View<Real*, Mem>& pos) {
    Kokkos::View<Real*, Mem> wd; Kokkos::View<long*, Mem> gd;
    const int nBuild = (nProc == N) ? -1 : nProc;  // build only the owned cells; ghosts are candidates
    auto res = buildTessellation<Real, false>(pos, wd, N, L, sw, densityCount, gd, NoSdf{},
                                              /*withForceGeom=*/false, nBuild, store.np, store.nt,
                                              store.pnbr, store.tri);
    Kokkos::deep_copy(vol, res.view.cellVolume);
    Kokkos::deep_copy(xRef, pos);
  }

  /// Compact the nonzero entries of `m[0..nProc)` into `wl`, returning the count.
  int compact(const Kokkos::View<int*, Mem>& m, const Kokkos::View<int*, Mem>& wl) {
    Kokkos::deep_copy(counter, 0);
    auto M = m; auto W = wl; auto c = counter;
    Kokkos::parallel_for("mt.compact", Kokkos::RangePolicy<Exec>(0, nProc),
                         KOKKOS_LAMBDA(int i) { if (M(i)) W(Kokkos::atomic_fetch_add(&c(), 1)) = i; });
    int h = 0; Kokkos::deep_copy(h, counter); return h;
  }

  /// Re-eval every cell on `pos` over its stored topology, write its volume, and classify it:
  ///   `outMask(i)=1`  if the cell is inconsistent (lost-face) OR is a violated-plane partner of some
  ///                   flagging cell (the flip side — fully covered by Pass 1) OR is a skin-mover;
  ///   `outMover(i)=1` if the cell moved > skin/2 since its last (re)build (the INSERTION trigger —
  ///                   only these need their new neighbours expanded in Pass 2, §1c).
  /// `useSkin` gates the mover trigger (off during verify passes — movers are a once-per-step trigger
  /// already folded into Pass 1, and the verify only needs to re-close residual flips).
  void certify(const Kokkos::View<Real*, Mem>& pos, const Kokkos::View<int*, Mem>& outMask,
               const Kokkos::View<int*, Mem>& outMover, bool useSkin) {
    Kokkos::deep_copy(outMask, 0);
    if (useSkin) Kokkos::deep_copy(outMover, 0);
    // Phase-4: in the Pass-1 certify, also record partner-discovered new neighbours for the surgical
    // path. The gaining cells of a flip are the violated-plane partners; each pair gains the other, so
    // scatter every other partner of a flagging cell onto each partner's extra-candidate list (no dedup
    // — duplicates are harmless no-op clips; overflow past kExtraCap just truncates and the cell falls
    // back to a full gather at verify). Only in Pass 1 (useSkin), and only when surgical is enabled.
    const bool doExtra = surgical && useSkin;
    if (doExtra) Kokkos::deep_copy(extraCnt, 0);
    auto EN = extraNbr; auto EC = extraCnt;
    constexpr int kCap = kExtraCap;
    auto P = pos; auto XR = xRef; auto Vv = vol; auto M = outMask; auto Mv = outMover;
    Store st = store;
    const Real Lx = L[0], Ly = L[1], Lz = L[2];
    const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
    const Real tolL = tol, half2 = Real(0.25) * skin * skin;
    const bool skinOn = useSkin;
    const int nP_ = nProc;
    Kokkos::parallel_for(
        "mt.certify", Kokkos::RangePolicy<Exec>(0, nProc), KOKKOS_LAMBDA(int i) {
          Cell c; st.load(i, c, Lx, Ly, Lz);
          c.reevalGeometry(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), Lx);
          Vv(i) = c.volumePerVertex();
          int partners[32]; int nP = 0;
          const bool ok = c.isSelfConsistent(tolL, partners, 32, nP);
          bool mover = false;
          if (skinOn) {
            Real dx = P(3 * i) - XR(3 * i), dy = P(3 * i + 1) - XR(3 * i + 1), dz = P(3 * i + 2) - XR(3 * i + 2);
            dx = dx > Lxh ? dx - Lx : (dx < -Lxh ? dx + Lx : dx);
            dy = dy > Lyh ? dy - Ly : (dy < -Lyh ? dy + Ly : dy);
            dz = dz > Lzh ? dz - Lz : (dz < -Lzh ? dz + Lz : dz);
            mover = (dx * dx + dy * dy + dz * dz) > half2;
            if (mover) Mv(i) = 1;
          }
          if (!ok || mover) M(i) = 1;
          // only mark partners we MAINTAIN (owned); a ghost partner is the owning rank's responsibility.
          for (int q = 0; q < nP; ++q) if (partners[q] < nP_) Kokkos::atomic_exchange(&M(partners[q]), 1);
          if (doExtra) {  // each pair of gaining cells gains the other -> seed the surgical candidate lists
            for (int q = 0; q < nP; ++q) {
              const int cell = partners[q];
              if (cell < 0 || cell >= nP_) continue;
              for (int r = 0; r < nP; ++r) {
                if (r == q) continue;
                const int nb = partners[r];
                if (nb < 0) continue;
                const int slot = Kokkos::atomic_fetch_add(&EC(cell), 1);
                if (slot < kCap) EN((size_t)cell * kCap + slot) = nb;
              }
            }
          }
        });
  }

  /// Scoped re-certify: reeval + certificate over ONLY the `n` cells in `list` (refresh their vol), set
  /// outMask(i)=1 for any inconsistent + OR-mark their partners. The verify uses this instead of a full
  /// re-certify: after the gathers, every NON-gathered cell's reeval is bit-identical to the Pass-1
  /// certify (its store + its neighbours' positions are unchanged), and every gained edge has BOTH
  /// endpoints gathered (a flip's partner endpoints in Pass 1, a mover's new neighbours in Pass 2), so
  /// only the gathered cells can differ — re-checking just them is provably equivalent to a full
  /// re-certify at a fraction of the cost (this is what lets the small-displacement speedup rise instead
  /// of being pinned by a second full-N reeval).
  void certifyList(const Kokkos::View<Real*, Mem>& pos, const Kokkos::View<int*, Mem>& list, int n,
                   const Kokkos::View<int*, Mem>& outMask) {
    Kokkos::deep_copy(outMask, 0);
    if (n <= 0) return;
    auto P = pos; auto Vv = vol; auto M = outMask; Store st = store;
    const Real Lx = L[0], Ly = L[1], Lz = L[2]; const Real tolL = tol; const int nP_ = nProc;
    Kokkos::parallel_for("mt.certifyList", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int s) {
      const int i = list(s);
      Cell c; st.load(i, c, Lx, Ly, Lz);
      c.reevalGeometry(P(3 * i), P(3 * i + 1), P(3 * i + 2), P.data(), Lx);
      Vv(i) = c.volumePerVertex();
      int partners[32]; int nP = 0;
      if (!c.isSelfConsistent(tolL, partners, 32, nP)) M(i) = 1;
      for (int q = 0; q < nP; ++q) if (partners[q] < nP_) Kokkos::atomic_exchange(&M(partners[q]), 1);
    });
  }

  /// Gather the `n` cells listed in `wl[0..n)` off `grid` (updating store + vol in place), mark them
  /// rebuilt, and RESET their Verlet reference to the current position (their neighbourhood is now
  /// fresh, so the skin/2 mover test for these cells restarts from here — proper per-cell Verlet).
  /// Reuses the per-step grid.
  void gatherSet(const TessGrid<Real>& grid, const Kokkos::View<int*, Mem>& wl, int n,
                 const Kokkos::View<Real*, Mem>& pos) {
    if (n <= 0) return;
    subsetGather<Real, false>(grid, wl, n, store.np, store.nt, store.pnbr, store.tri, vol, NoSdf{},
                              /*withForceGeom=*/false);
    auto W = wl; auto rb = rebuilt; auto P = pos; auto XR = xRef;
    Kokkos::parallel_for("mt.markRebuilt", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int s) {
      const int i = W(s);
      rb(i) = 1;
      XR(3 * i) = P(3 * i); XR(3 * i + 1) = P(3 * i + 1); XR(3 * i + 2) = P(3 * i + 2);
    });
  }

  /// Collect into `outMask` the face-neighbours (planes with ≥3 incident live triangles) of the `n`
  /// cells in `wl[0..n)` that have NOT already been rebuilt — the Pass-2 / expansion set.
  void collectNewNbrs(const Kokkos::View<int*, Mem>& wl, int n, const Kokkos::View<int*, Mem>& outMask) {
    Kokkos::deep_copy(outMask, 0);
    if (n <= 0) return;
    Store st = store; auto rb = rebuilt; auto M = outMask; const int nP_ = nProc;
    Kokkos::parallel_for(
        "mt.collectNbrs", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int s) {
          const int i = wl(s);
          const int np = st.np(i), nt = st.nt(i);
          for (int k = 6; k < np; ++k) {
            const int j = st.pnbr((size_t)i * MAXP + k);
            if (j < 0 || j >= nP_ || rb(j)) continue;  // only expand neighbours we MAINTAIN (owned)
            int cnt = 0;  // face = ≥3 incident live triangles
            for (int t = 0; t < nt; ++t) {
              const unsigned w = st.tri((size_t)i * MAXT + t);
              if (!((w >> 24) & 1u)) continue;
              const int a = (int)(w & 0xffu), b = (int)((w >> 8) & 0xffu), cc = (int)((w >> 16) & 0xffu);
              if (a == k || b == k || cc == k) ++cnt;
            }
            if (cnt >= 3) Kokkos::atomic_exchange(&M(j), 1);
          }
        });
  }

  /// Phase-4 single-face surgical repair: rebuild each flip cell in wl[0..n) by re-clipping the box with
  /// its KNOWN candidate set — stored neighbours ∪ partner-discovered extras (extraNbr) — at the current
  /// positions, with NO grid gather (the clip is order-independent and only commits planes that cut, so
  /// stale/lost-face neighbours are harmless no-ops). Updates store + vol, marks rebuilt, resets the
  /// Verlet reference. A cell whose candidate set is incomplete comes out wrong and is caught + re-gathered
  /// by the verify loop (the exact fallback) — so surgical is safe by construction. Used only for FLIP
  /// cells; far-mover insertions keep the full gather (their new neighbours aren't in any candidate list).
  void surgicalRepair(const Kokkos::View<int*, Mem>& wl, int n, const Kokkos::View<Real*, Mem>& pos) {
    if (n <= 0) return;
    auto W = wl; auto P = pos; auto rb = rebuilt; auto XR = xRef; auto Vv = vol;
    auto EN = extraNbr; auto EC = extraCnt; constexpr int kCap = kExtraCap;
    Store st = store;
    const Real Lx = L[0], Ly = L[1], Lz = L[2];
    const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
    Kokkos::parallel_for("mt.surgical", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int s) {
      const int i = W(s);
      Cell c; c.initBox(Lx, Ly, Lz);
      const Real sx = P(3 * i), sy = P(3 * i + 1), sz = P(3 * i + 2);
      const int npi = st.np(i);
      for (int k = 6; k < npi && !c.overflow; ++k) {  // stored neighbours, current positions
        const int j = st.pnbr((size_t)i * MAXP + k);
        if (j < 0) continue;
        Real rx = P(3 * j) - sx, ry = P(3 * j + 1) - sy, rz = P(3 * j + 2) - sz;
        rx = rx > Lxh ? rx - Lx : (rx < -Lxh ? rx + Lx : rx);
        ry = ry > Lyh ? ry - Ly : (ry < -Lyh ? ry + Ly : ry);
        rz = rz > Lzh ? rz - Lz : (rz < -Lzh ? rz + Lz : rz);
        const Real nrm[3] = {rx, ry, rz};
        c.clip(nrm, Real(0.5) * (rx * rx + ry * ry + rz * rz), j);
      }
      const int ne = EC(i) < kCap ? EC(i) : kCap;
      for (int e = 0; e < ne && !c.overflow; ++e) {  // partner-discovered new neighbours
        const int j = EN((size_t)i * kCap + e);
        if (j < 0 || j == i) continue;
        Real rx = P(3 * j) - sx, ry = P(3 * j + 1) - sy, rz = P(3 * j + 2) - sz;
        rx = rx > Lxh ? rx - Lx : (rx < -Lxh ? rx + Lx : rx);
        ry = ry > Lyh ? ry - Ly : (ry < -Lyh ? ry + Ly : ry);
        rz = rz > Lzh ? rz - Lz : (rz < -Lzh ? rz + Lz : rz);
        const Real nrm[3] = {rx, ry, rz};
        c.clip(nrm, Real(0.5) * (rx * rx + ry * ry + rz * rz), j);
      }
      if (!c.overflow) { st.save(i, c); Vv(i) = c.volumePerVertex(); }
      rb(i) = 1;
      XR(3 * i) = sx; XR(3 * i + 1) = sy; XR(3 * i + 2) = sz;
    });
  }

  /// Dense-cluster dilation (Phase 3): bin the flagged cells into the linear grid, then add every owned
  /// cell whose 27-grid-cell neighbourhood holds > clusterNbhd flagged seeds to the Pass-1 mask. This
  /// over-covers a local cascade in ONE regional gather instead of paying a kernel-launch per verify
  /// iteration. Over-coverage is always safe (a gathered cell is just re-clipped correctly). Returns
  /// true if it added any cell. Uses the linear grid index (independent of the grid's Morton ordering).
  bool dilate(const TessGrid<Real>& grid, const Kokkos::View<Real*, Mem>& pos) {
    const int dimx = grid.dimx, dimy = grid.dimy, dimz = grid.dimz;
    const int ncell = dimx * dimy * dimz;
    if ((int)cellFlag.extent(0) < ncell)
      cellFlag = Kokkos::View<int*, Mem>(Kokkos::view_alloc(std::string("mt.cellFlag"),
                                                            Kokkos::WithoutInitializing), ncell);
    Kokkos::deep_copy(cellFlag, 0);
    const Real icx = grid.icx, icy = grid.icy, icz = grid.icz;
    auto P = pos; auto M = mask; auto CF = cellFlag;
    Kokkos::parallel_for("mt.dilate.count", Kokkos::RangePolicy<Exec>(0, nProc), KOKKOS_LAMBDA(int i) {
      if (!M(i)) return;
      const int gx = ((int)Kokkos::floor(P(3 * i) * icx) % dimx + dimx) % dimx;
      const int gy = ((int)Kokkos::floor(P(3 * i + 1) * icy) % dimy + dimy) % dimy;
      const int gz = ((int)Kokkos::floor(P(3 * i + 2) * icz) % dimz + dimz) % dimz;
      Kokkos::atomic_inc(&CF(gx + gy * dimx + gz * dimx * dimy));
    });
    const int thr = clusterNbhd;
    long added = 0;
    Kokkos::parallel_reduce(
        "mt.dilate.mark", Kokkos::RangePolicy<Exec>(0, nProc),
        KOKKOS_LAMBDA(int i, long& a) {
          if (M(i)) return;
          const int gx = ((int)Kokkos::floor(P(3 * i) * icx) % dimx + dimx) % dimx;
          const int gy = ((int)Kokkos::floor(P(3 * i + 1) * icy) % dimy + dimy) % dimy;
          const int gz = ((int)Kokkos::floor(P(3 * i + 2) * icz) % dimz + dimz) % dimz;
          int sum = 0;
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const int nx = (gx + dx + dimx) % dimx, ny = (gy + dy + dimy) % dimy, nz = (gz + dz + dimz) % dimz;
                sum += CF(nx + ny * dimx + nz * dimx * dimy);
              }
          if (sum > thr) { M(i) = 1; ++a; }
        },
        added);
    return added > 0;
  }

  /// One moving-point update step. Updates store + vol in place from `pos`.
  RepairStats step(const Kokkos::View<Real*, Mem>& pos) {
    RepairStats s;
    auto grid = buildTessGrid<Real, false>(pos, Kokkos::View<Real*, Mem>(), N, L, sw, densityCount,
                                           Kokkos::View<long*, Mem>());
    Kokkos::deep_copy(rebuilt, 0);

    // Pass 1: detect (flagged ∪ partners ∪ skin-movers). flagged-commons + their violated-plane partners
    // (the gaining cells) fully cover an isolated flip — no Pass 2 for flips.
    certify(pos, mask, mover, /*useSkin=*/true);
    int n1 = compact(mask, wl1);
    s.pass1Raw = n1;

    // Small-displacement fast path: the certificate found every cell consistent and no skin-mover (n1
    // counts flagged ∪ partners ∪ movers). By the §1 completeness argument a stale topology always
    // flags some cell, so n1==0 ⇒ the stored topology is still Voronoi on the new positions — the
    // re-eval already refreshed every volume, so publish with NO gather and NO verify pass. This is the
    // "almost no cells invalidated" regime: the per-step cost collapses to one re-eval + the grid.
    if (n1 == 0) return s;

    // Phase-3 adaptive gate (decided here, after the FREE certificate, before any gather):
    //   high global churn  -> a full rebuild is cheaper than repairing this many cells;
    //   dense local cluster -> dilate the Pass-1 set by a grid-cell buffer (one regional gather);
    //   else (sparse)       -> the two-pass repair below.
    if (useGate && (double)n1 / (nProc > 0 ? nProc : 1) > churnThresh) {
      rebuild(pos);
      s.route = RepairStats::kRebuildGate;
      return s;
    }
    if (useGate && useDilation && n1 > 0 && (double)n1 / nProc < dilateMaxChurn && dilate(grid, pos)) {
      n1 = compact(mask, wl1);
      s.route = RepairStats::kDilated;
    }
    s.pass1 = n1;
    // Pass 1 gather. Phase 4: FLIP cells (flagged non-movers) are repaired SURGICALLY (re-clip known
    // candidates, no grid gather); far-mover insertions keep the full gather (their new neighbours are
    // in no candidate list). Default (surgical off): the whole Pass-1 set goes through the full gather.
    int nM = compact(mover, wlM);
    if (surgical) {
      auto Msk = mask; auto Mv = mover; auto M2 = mask2;
      Kokkos::parallel_for("mt.surgMask", Kokkos::RangePolicy<Exec>(0, nProc),
                           KOKKOS_LAMBDA(int i) { M2(i) = (Msk(i) && !Mv(i)) ? 1 : 0; });
      int nS = compact(mask2, wl1);  // wl1's Pass-1 list no longer needed -> reuse for the flip list
      s.surgical = nS;
      surgicalRepair(wl1, nS, pos);     // flips: no gather
      gatherSet(grid, wlM, nM, pos);    // movers: full gather
    } else {
      gatherSet(grid, wl1, n1, pos);
    }

    // Pass 2: ONLY the new face-neighbours of the skin-MOVERS (the insertion side §1c — those neighbours
    // never flag and are never partners). Expanding every Pass-1 cell would re-clip ~all neighbours of
    // every flip cluster for nothing; movers are the only cells whose rebuild reveals an unflagged gainer.
    collectNewNbrs(wlM, nM, mask2);
    int n2 = compact(mask2, wl2);
    s.pass2 = n2;
    gatherSet(grid, wl2, n2, pos);

    // Verify (scoped): re-certify only the GATHERED cells (every non-gathered cell is provably unchanged
    // — see certifyList). Clean ⇒ publish; else gather the still-flagged ∪ partners (residual coupled
    // flips / tol-degenerate cells) and re-verify, bounded.
    for (int vp = 0; vp < verifyCap + 1; ++vp) {
      const int ng = compact(rebuilt, wl1);   // all cells gathered so far this step
      certifyList(pos, wl1, ng, mask);
      int nv = compact(mask, wl2);
      s.verifyPasses = vp + 1;
      if (nv == 0) return s;            // clean — done
      if (vp == verifyCap) break;       // out of budget — fall back
      s.extra += nv;
      gatherSet(grid, wl2, nv, pos);
    }

    // Still dirty after the verify budget: cold rebuild (always exact). Rare; the data point that says
    // "for this step, rebuild was cheaper" — the adaptive gate (Phase 3) will pre-empt it up front.
    rebuild(pos);
    s.fellBack = true;
    return s;
  }
};

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_REPAIR_HPP
