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
  int pass1 = 0;       ///< cells gathered in Pass 1 (flagged ∪ partners ∪ skin-movers)
  int pass2 = 0;       ///< cells gathered in Pass 2 (new face-neighbours of Pass-1 cells)
  int extra = 0;       ///< cells gathered across the verify extra-passes
  int verifyPasses = 0;///< number of verify iterations run
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

  int N = 0, sw = 4, densityCount = -1;
  Real L[3] = {1, 1, 1};
  Real tol = 0;   ///< certificate tolerance (absolute distance); ~1e-4·spacing FP64, ~2e-3·spacing FP32
  Real skin = 0;  ///< Verlet skin width (absolute); a particle moving > skin/2 is a Pass-1 mover
  int verifyCap = 2;  ///< max verify extra-passes before falling back to a full rebuild

  Store store;
  Kokkos::View<Real*, Mem> vol;   // N : cell volumes (the published geometry scalar)
  Kokkos::View<Real*, Mem> xRef;  // 3N : positions at the last (re)build (Verlet reference)
  // scratch
  Kokkos::View<int*, Mem> mask, mask2, mover, rebuilt, wl1, wl2, wlM;
  Kokkos::View<int, Mem> counter;

  void alloc(int n, const Real Lbox[3], Real tol_, Real skin_, int sw_ = 4, int densityCount_ = -1) {
    N = n; sw = sw_; densityCount = densityCount_ < 0 ? n : densityCount_;
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
    counter = Kokkos::View<int, Mem>("mt.counter");
  }

  /// Full cold (re)build: the production tessellator emitting the resident topology + volume, then
  /// reset the Verlet reference. This is the oracle, the fallback, and the initial build.
  void rebuild(const Kokkos::View<Real*, Mem>& pos) {
    Kokkos::View<Real*, Mem> wd; Kokkos::View<long*, Mem> gd;
    auto res = buildTessellation<Real, false>(pos, wd, N, L, sw, densityCount, gd, NoSdf{},
                                              /*withForceGeom=*/false, -1, store.np, store.nt,
                                              store.pnbr, store.tri);
    Kokkos::deep_copy(vol, res.view.cellVolume);
    Kokkos::deep_copy(xRef, pos);
  }

  /// Compact `m` (nonzero entries) into `wl`, returning the count.
  int compact(const Kokkos::View<int*, Mem>& m, const Kokkos::View<int*, Mem>& wl) {
    Kokkos::deep_copy(counter, 0);
    auto M = m; auto W = wl; auto c = counter;
    Kokkos::parallel_for("mt.compact", Kokkos::RangePolicy<Exec>(0, N),
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
    auto P = pos; auto XR = xRef; auto Vv = vol; auto M = outMask; auto Mv = outMover;
    Store st = store;
    const Real Lx = L[0], Ly = L[1], Lz = L[2];
    const Real Lxh = Real(0.5) * Lx, Lyh = Real(0.5) * Ly, Lzh = Real(0.5) * Lz;
    const Real tolL = tol, half2 = Real(0.25) * skin * skin;
    const bool skinOn = useSkin;
    Kokkos::parallel_for(
        "mt.certify", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(int i) {
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
          for (int q = 0; q < nP; ++q) Kokkos::atomic_exchange(&M(partners[q]), 1);
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
    Store st = store; auto rb = rebuilt; auto M = outMask;
    Kokkos::parallel_for(
        "mt.collectNbrs", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int s) {
          const int i = wl(s);
          const int np = st.np(i), nt = st.nt(i);
          for (int k = 6; k < np; ++k) {
            const int j = st.pnbr((size_t)i * MAXP + k);
            if (j < 0 || rb(j)) continue;
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

  /// One moving-point update step. Updates store + vol in place from `pos`.
  RepairStats step(const Kokkos::View<Real*, Mem>& pos) {
    RepairStats s;
    auto grid = buildTessGrid<Real, false>(pos, Kokkos::View<Real*, Mem>(), N, L, sw, densityCount,
                                           Kokkos::View<long*, Mem>());
    Kokkos::deep_copy(rebuilt, 0);

    // Pass 1: detect (flagged ∪ partners ∪ skin-movers) and gather them. flagged-commons + their
    // violated-plane partners (the gaining cells) fully cover an isolated flip — no Pass 2 for flips.
    certify(pos, mask, mover, /*useSkin=*/true);
    int n1 = compact(mask, wl1);
    s.pass1 = n1;
    gatherSet(grid, wl1, n1, pos);

    // Pass 2: ONLY the new face-neighbours of the skin-MOVERS (the insertion side §1c — those neighbours
    // never flag and are never partners). Expanding every Pass-1 cell would re-clip ~all neighbours of
    // every flip cluster for nothing; movers are the only cells whose rebuild reveals an unflagged gainer.
    int nM = compact(mover, wlM);
    collectNewNbrs(wlM, nM, mask2);
    int n2 = compact(mask2, wl2);
    s.pass2 = n2;
    gatherSet(grid, wl2, n2, pos);

    // Verify: re-run the certificate over all cells (no skin — movers already handled). Clean ⇒ publish;
    // else gather the still-flagged ∪ partners (residual coupled flips) and re-verify, bounded.
    for (int vp = 0; vp < verifyCap + 1; ++vp) {
      certify(pos, mask, mover, /*useSkin=*/false);
      int nv = compact(mask, wl1);
      s.verifyPasses = vp + 1;
      if (nv == 0) return s;            // clean — done
      if (vp == verifyCap) break;       // out of budget — fall back
      s.extra += nv;
      gatherSet(grid, wl1, nv, pos);
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
