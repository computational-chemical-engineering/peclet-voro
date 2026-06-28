/**
 * @file device/dynamic_validate.hpp
 * \brief Part-II Phase-0 validators: per-step invariants + oracle diff for the moving-point updater.
 *
 * You cannot trust any incremental update result without (a) cheap per-step invariants that a corrupt
 * update violates, and (b) a periodic comparison against the cold-build oracle (the same
 * buildTessellation that is also the rebuild fallback). This header provides both, on device, in one
 * Kokkos path. They are diagnostic — call them every step in debug / every k steps in production; the
 * production hot loop pays nothing unless it opts in.
 *
 * Invariants (checkInvariants, reads a published TessellationView + its AuxMaps reciprocal map):
 *   - volRelErr   : |Σ cellVolume − boxVolume| / boxVolume                 (space-filling).
 *   - maxAreaAsym : max over interior facets of |A(g) + A(recip(g))| / |A(g)|  (A_ij = −A_ji exactly:
 *                   the momentum-conservation certificate — any pairwise force ∝ A then sums to zero).
 *   - sumAreaMag  : |Σ_g A(g)| over ALL facets (interior pairs cancel, periodic box has no true
 *                   boundary ⇒ ≈0); the global form of the same conservation.
 *   - sumForceMag : |Σ_g dV(g)| over all facets (the published volume-gradient/force vectors).
 *   - nNonRecip   : interior facets (neighbour in [0,N)) with no reciprocal — a topology-symmetry leak.
 *
 * Oracle diff (oracleDiff, current store/volume vs a fresh buildTessellation view):
 *   - meanVolRelErr,maxVolRelErr,volMismatch : per-cell volume error vs the oracle.
 *   - changedNbrFrac : fraction of cells whose stored neighbour SET differs from the oracle's.
 *   - missedNbr      : cells with a TRUE (oracle) neighbour absent from the stored set (real staleness).
 *
 * Core header: Kokkos + transport-core + the published view/aux + TopologyStore. No physics.
 */
#ifndef VORFLOW_DEVICE_DYNAMIC_VALIDATE_HPP
#define VORFLOW_DEVICE_DYNAMIC_VALIDATE_HPP

#include <cmath>
#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"
#include "vorflow/device/transpose.hpp"  // AuxMaps, buildAuxMaps
#include "vorflow/tessellation_view.hpp"

namespace vor {
namespace device {

/// Per-step invariant residuals (all should be ~0 / ~machine-precision for a correct tessellation).
struct Invariants {
  double volRelErr = 0;     ///< |Σvol − boxVol| / boxVol
  double maxAreaAsym = 0;   ///< max interior facet |A(g)+A(recip)| / |A(g)|
  double sumAreaMag = 0;    ///< |Σ_g A(g)| over all facets
  double sumForceMag = 0;   ///< |Σ_g dV(g)| over all facets
  long nNonRecip = 0;       ///< interior facets with no reciprocal
};

/// Compute the per-step invariants of a published TessellationView. `aux` is its reciprocal map
/// (buildAuxMaps(view)); pass the box volume Lx*Ly*Lz.
template <class Real>
Invariants checkInvariants(const TessellationView<Real>& view, const AuxMaps<Real>& aux,
                           double boxVolume) {
  using Exec = tpx::ExecSpace;
  const int N = view.numCells();
  const int nF = view.numFacets();
  Invariants inv;

  // Σ cellVolume.
  double sumV = 0;
  Kokkos::parallel_reduce(
      "inv.sumVol", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, double& a) { a += (double)view.volume(i); }, sumV);
  inv.volRelErr = boxVolume > 0 ? std::abs(sumV - boxVolume) / boxVolume : std::abs(sumV);

  // Reciprocal area antisymmetry + global area/force sums + non-reciprocal interior facet count.
  auto recip = aux.recip;
  double maxAsym = 0;
  Kokkos::parallel_reduce(
      "inv.areaAsym", Kokkos::RangePolicy<Exec>(0, nF),
      KOKKOS_LAMBDA(const int g, double& m) {
        const int h = recip(g);
        if (h < 0) return;
        double e = 0, a2 = 0;
        for (int c = 0; c < 3; ++c) {
          const double ag = view.area(g, c), ah = view.area(h, c);
          const double s = ag + ah;
          e += s * s;
          a2 += ag * ag;
        }
        const double rel = a2 > 0 ? std::sqrt(e / a2) : std::sqrt(e);
        if (rel > m) m = rel;
      },
      Kokkos::Max<double>(maxAsym));
  inv.maxAreaAsym = maxAsym;

  double sax = 0, say = 0, saz = 0, sfx = 0, sfy = 0, sfz = 0;
  Kokkos::parallel_reduce(
      "inv.sumArea", Kokkos::RangePolicy<Exec>(0, nF),
      KOKKOS_LAMBDA(const int g, double& ax, double& ay, double& az, double& fx, double& fy,
                    double& fz) {
        ax += (double)view.area(g, 0); ay += (double)view.area(g, 1); az += (double)view.area(g, 2);
        fx += (double)view.connect(g, 0); fy += (double)view.connect(g, 1); fz += (double)view.connect(g, 2);
      },
      sax, say, saz, sfx, sfy, sfz);
  inv.sumAreaMag = std::sqrt(sax * sax + say * say + saz * saz);
  inv.sumForceMag = std::sqrt(sfx * sfx + sfy * sfy + sfz * sfz);

  long nonRecip = 0;
  Kokkos::parallel_reduce(
      "inv.nonRecip", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, long& a) {
        for (int g = view.facetBegin(i); g < view.facetEnd(i); ++g) {
          const int j = (int)view.facetNbr(g);
          if (j >= 0 && j < N && recip(g) < 0) ++a;
        }
      },
      nonRecip);
  inv.nNonRecip = nonRecip;
  return inv;
}

/// Result of comparing the current updated state against a fresh cold-build oracle.
struct OracleDiff {
  double meanVolRelErr = 0;
  double maxVolRelErr = 0;
  long volMismatch = 0;     ///< cells with vol rel err > 1e-3
  double changedNbrFrac = 0;///< fraction of cells whose stored neighbour set differs from the oracle
  long missedNbr = 0;       ///< cells with a TRUE (oracle) neighbour absent from the stored set
};

/// Compare a current per-cell volume array against the oracle view's volumes (cell i == seed i).
template <class Real>
void compareVolumes(const Kokkos::View<Real*, tpx::MemSpace>& volCurrent,
                    const TessellationView<Real>& oracle, OracleDiff& out) {
  using Exec = tpx::ExecSpace;
  const int N = oracle.numCells();
  double sum = 0, mx = 0;
  long mm = 0;
  Kokkos::parallel_reduce(
      "diff.vol", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, double& s, double& m, long& c) {
        const double vo = (double)oracle.volume(i);
        const double r = vo > 0 ? std::abs((double)volCurrent(i) - vo) / vo : 0;
        s += r;
        if (r > m) m = r;
        if (r > 1e-3) ++c;
      },
      sum, Kokkos::Max<double>(mx), mm);
  out.meanVolRelErr = sum / (N > 0 ? N : 1);
  out.maxVolRelErr = mx;
  out.volMismatch = mm;
}

/// Compare the stored topology against the oracle's neighbour sets, on FACE neighbours (a stored
/// plane k counts only if ≥3 live triangles are incident — the raw `pnbr` also keeps non-face cutting
/// planes, which are not neighbours). changedNbrFrac = cells whose face-neighbour SET differs (either
/// direction); missedNbr = cells missing a TRUE neighbour (genuine staleness — what the repair must
/// drive to zero). Needs the packed triangles (tri at stride MAXT) + nt to identify the faces.
template <class Real>
void compareNeighbours(const Kokkos::View<int*, tpx::MemSpace>& storeNp,
                       const Kokkos::View<int*, tpx::MemSpace>& storeNt,
                       const Kokkos::View<int*, tpx::MemSpace>& storePnbr,
                       const Kokkos::View<unsigned*, tpx::MemSpace>& storeTri, int MAXP, int MAXT,
                       const TessellationView<Real>& oracle, OracleDiff& out) {
  using Exec = tpx::ExecSpace;
  const int N = oracle.numCells();
  long changed = 0, missed = 0;
  Kokkos::parallel_reduce(
      "diff.nbr", Kokkos::RangePolicy<Exec>(0, N),
      KOKKOS_LAMBDA(const int i, long& chg, long& mis) {
        const int snp = storeNp(i), snt = storeNt(i);
        bool diff = false, miss = false;
        // every true (oracle) neighbour must be a store FACE neighbour (≥3 incident live triangles).
        for (int g = oracle.facetBegin(i); g < oracle.facetEnd(i); ++g) {
          const int j = (int)oracle.facetNbr(g);
          if (j < 0) continue;
          bool inStore = false;
          for (int k = 6; k < snp && !inStore; ++k) {
            if (storePnbr((size_t)i * MAXP + k) != j) continue;
            int cnt = 0;
            for (int t = 0; t < snt; ++t) {
              const unsigned w = storeTri((size_t)i * MAXT + t);
              if (!((w >> 24) & 1u)) continue;
              const int t0 = (int)(w & 0xffu), t1 = (int)((w >> 8) & 0xffu), t2 = (int)((w >> 16) & 0xffu);
              if (t0 == k || t1 == k || t2 == k) ++cnt;
            }
            if (cnt >= 3) inStore = true;
          }
          if (!inStore) { diff = true; miss = true; }
        }
        // every store FACE neighbour must be a true (oracle) neighbour (set differs the other way).
        for (int k = 6; k < snp && !diff; ++k) {
          const int j = storePnbr((size_t)i * MAXP + k);
          if (j < 0) continue;
          int cnt = 0;
          for (int t = 0; t < snt; ++t) {
            const unsigned w = storeTri((size_t)i * MAXT + t);
            if (!((w >> 24) & 1u)) continue;
            const int t0 = (int)(w & 0xffu), t1 = (int)((w >> 8) & 0xffu), t2 = (int)((w >> 16) & 0xffu);
            if (t0 == k || t1 == k || t2 == k) ++cnt;
          }
          if (cnt < 3) continue;  // not a face
          bool inOracle = false;
          for (int g = oracle.facetBegin(i); g < oracle.facetEnd(i); ++g)
            if ((int)oracle.facetNbr(g) == j) { inOracle = true; break; }
          if (!inOracle) diff = true;
        }
        if (diff) ++chg;
        if (miss) ++mis;
      },
      changed, missed);
  out.changedNbrFrac = (double)changed / (N > 0 ? N : 1);
  out.missedNbr = missed;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_DYNAMIC_VALIDATE_HPP
