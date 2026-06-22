/**
 * @file device/cell_cutter.hpp
 * \brief Device-callable Voronoi/Power cell cutter (migration Phase 2).
 *
 * A self-contained, KOKKOS_FUNCTION port of the legacy half-edge plane-cut
 * (CellMaker::cutCell2) onto a fixed-capacity scratch cell — no std::vector, no
 * arena, no telemetry — so a whole cell is constructed in registers/scratch and
 * only the finished cell is published (plan §2.3). The algorithm is ported
 * faithfully: same cuboid seed, same closest-first cut order with the
 * security-radius early-out, same sign-change edge trace and DFS removal, so on
 * general-position inputs it reproduces the legacy topology bit-for-bit.
 *
 * Capacity is 128 because the half-edge label packs facet/vertex into 7-bit
 * fields (sentinel index 127 == "dummy"), exactly as the legacy code. Overflow
 * is reported (status), not aborted; the caller re-runs such cells on a larger
 * path (Phase 3 fallback).
 *
 * Weighting policy is a compile-time template parameter (Unweighted | Power),
 * matching legacy CellMaker<Real, Weighted>; the only difference is the plane
 * offset (radical plane for Power). The cut predicate is evaluated in `Real`
 * (double in practice); a deterministic global-id tie-break keeps cross-rank
 * decisions identical (plan §2.6).
 *
 * Core header: Kokkos only, no physics include.
 */
#ifndef VORFLOW_DEVICE_CELL_CUTTER_HPP
#define VORFLOW_DEVICE_CELL_CUTTER_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>

namespace vor {
namespace device {

#ifdef VORFLOW_CUTTER_PROFILE
// Host-only per-cell profiling counters (Phase 0 cutter profiler). Defined in
// bench_cutter.cpp; fully absent unless the profiler target sets the macro.
struct CutterCounters {
  long cuts = 0, cdistCalls = 0, traceSteps = 0, dfsSteps = 0;
  long findRsqMaxCalls = 0, findRsqMaxScans = 0, exhaustiveCalls = 0, exhaustiveScans = 0;
};
extern CutterCounters g_cc;
#define VF_CC(stmt) stmt
#else
#define VF_CC(stmt)
#endif

using lbl_t = std::uint16_t;  ///< half-edge label (facet:7 | vertex:7 | edge:2)

// Bit layout identical to vor::makeLabel/getFacet/getVertex/getEdge, but
// KOKKOS_INLINE_FUNCTION so it is callable on every backend.
KOKKOS_INLINE_FUNCTION lbl_t makeLabel(int facet, int vertex, int edge) {
  return static_cast<lbl_t>((facet << 9) | (vertex << 2) | edge);
}
KOKKOS_INLINE_FUNCTION int getFacet(lbl_t l) {
  return (l >> 9) & 0x7f;
}
KOKKOS_INLINE_FUNCTION int getVertex(lbl_t l) {
  return (l >> 2) & 0x7f;
}
KOKKOS_INLINE_FUNCTION int getEdge(lbl_t l) {
  return l & 0x3;
}

/// Weighting policies (compile-time, no per-cut branch in the hot path).
struct Unweighted {};
struct Power {};

/// Cut status returned by build().
enum class CutStatus : int { Ok = 0, Overflow = 1, Empty = 2 };

/**
 * Fixed-capacity scratch cell. Holds the half-edge topology under construction
 * plus the small work stacks cutCell2 needs. One instance per cell; sized for
 * the 128 label cap. Trivially default-constructible (POD-like) so it can live
 * in registers, team scratch, or a per-thread stack.
 */
template <class Real>
struct ScratchCell {
  static constexpr int CAP = 128;
  static constexpr int VDUM = CAP - 1;  // dummy vertex sentinel (127)
  static constexpr int FDUM = CAP - 1;  // dummy facet sentinel (127)

  // Vertex data.
  Real vpos[CAP][3];
  lbl_t vlab[CAP][3];  // three outgoing edge labels per (3-valent) vertex
  Real rsq[CAP];       // squared distance of vertex from seed (security radius)
  Real dist[CAP];      // cached signed plane distance
  int knownGen[CAP];   // generation tag for the dist cache

  // Facet data.
  lbl_t flab[CAP];    // one entry half-edge label per facet
  int fnbr[CAP];      // neighbour seed id (gid); negative => boundary
  Real pvec[CAP][3];  // cut-plane vector (relPos to neighbour)
  Real poff[CAP];     // cut-plane offset

  // Slot allocators (alive flags + LIFO free stacks + high-water marks).
  bool aliveV[CAP];
  bool aliveF[CAP];
  int freeV[CAP];
  int freeF[CAP];
  int freeTopV, freeTopF;
  int numAllocV, numAllocF;

  int gen;      // dist-cache generation counter
  int vRsqMax;  // vertex with the largest rsq (drives the security radius)

  // Work stacks (cutCell2 scratch).
  int newV[CAP];
  int facetPrev[CAP];
  int vStack[CAP];

  // ---- slot allocator ----
  // Only the n live slots need initialising: a slot past the high-water mark is
  // never read until getFreeV/getFreeF first allocates it (which sets aliveV/aliveF),
  // and all iteration is bounded by numAllocV/numAllocF — so the old O(CAP) clear was
  // ~120 wasted writes per axis per cell.
  KOKKOS_INLINE_FUNCTION void resetV(int n) {
    for (int i = 0; i < n; ++i)
      aliveV[i] = true;
    numAllocV = n;
    freeTopV = 0;
  }
  KOKKOS_INLINE_FUNCTION void resetF(int n) {
    for (int i = 0; i < n; ++i)
      aliveF[i] = true;
    numAllocF = n;
    freeTopF = 0;
  }
  KOKKOS_INLINE_FUNCTION int getFreeV() {
    if (freeTopV > 0) {
      int s = freeV[--freeTopV];
      aliveV[s] = true;
      return s;
    }
    if (numAllocV < VDUM) {  // keep index 127 reserved as the dummy sentinel
      aliveV[numAllocV] = true;
      return numAllocV++;
    }
    return -1;  // overflow
  }
  KOKKOS_INLINE_FUNCTION int getFreeF() {
    if (freeTopF > 0) {
      int s = freeF[--freeTopF];
      aliveF[s] = true;
      return s;
    }
    if (numAllocF < FDUM) {
      aliveF[numAllocF] = true;
      return numAllocF++;
    }
    return -1;
  }
  KOKKOS_INLINE_FUNCTION void releaseV(int i) {
    aliveV[i] = false;
    freeV[freeTopV++] = i;
  }
  KOKKOS_INLINE_FUNCTION void releaseF(int i) {
    aliveF[i] = false;
    freeF[freeTopF++] = i;
  }
  KOKKOS_INLINE_FUNCTION bool isFreeV(int i) const { return !aliveV[i]; }
  KOKKOS_INLINE_FUNCTION bool isFreeF(int i) const { return !aliveF[i]; }
  KOKKOS_INLINE_FUNCTION int firstAliveV() const {
    for (int i = 0; i < numAllocV; ++i)
      if (aliveV[i])
        return i;
    return VDUM;
  }
  KOKKOS_INLINE_FUNCTION bool emptyV() const {
    for (int i = 0; i < numAllocV; ++i)
      if (aliveV[i])
        return false;
    return true;
  }

  // ---- geometry helpers ----
  KOKKOS_INLINE_FUNCTION void computeRsq(int i) {
    rsq[i] = vpos[i][0] * vpos[i][0] + vpos[i][1] * vpos[i][1] + vpos[i][2] * vpos[i][2];
    if (rsq[i] > rsq[vRsqMax])
      vRsqMax = i;
  }
  KOKKOS_INLINE_FUNCTION void computeAllRsq() {
    vRsqMax = firstAliveV();
    for (int i = 0; i < numAllocV; ++i)
      if (aliveV[i])
        computeRsq(i);
  }
  KOKKOS_INLINE_FUNCTION void findRsqMax() {
    VF_CC(++g_cc.findRsqMaxCalls);
    VF_CC(g_cc.findRsqMaxScans += numAllocV);
    Real m = 0;
    for (int i = 0; i < numAllocV; ++i)
      if (aliveV[i] && rsq[i] > m) {
        m = rsq[i];
        vRsqMax = i;
      }
  }
  KOKKOS_INLINE_FUNCTION void resetDist() { ++gen; }
  // Cached signed distance of vertex i to the current plane (pv, off).
  KOKKOS_INLINE_FUNCTION Real cdist(int i, const Real pv[3], Real off) {
    VF_CC(++g_cc.cdistCalls);
    if (knownGen[i] != gen) {
      dist[i] = vpos[i][0] * pv[0] + vpos[i][1] * pv[1] + vpos[i][2] * pv[2] - off;
      knownGen[i] = gen;
    }
    return dist[i];
  }
  KOKKOS_INLINE_FUNCTION void computeAllDist(const Real pv[3], Real off) {
    VF_CC(++g_cc.exhaustiveCalls);
    VF_CC(g_cc.exhaustiveScans += numAllocV);
    for (int i = 0; i < numAllocV; ++i)
      if (aliveV[i]) {
        dist[i] = vpos[i][0] * pv[0] + vpos[i][1] * pv[1] + vpos[i][2] * pv[2] - off;
        knownGen[i] = gen;
      }
  }

  KOKKOS_INLINE_FUNCTION int getNextLabelCCW(lbl_t label) const {
    int facetMasked = label & 0xfe00;
    lbl_t rev = vlab[getVertex(label)][getEdge(label)];
    int vertMasked = rev & 0x01fc;
    int edge = getEdge(rev);
    edge = (edge == 0 ? 2 : edge - 1);
    return facetMasked | vertMasked | edge;
  }

  /// Seed the cell with the big cuboid of extent L (centred on the seed).
  /// Topology copied verbatim from legacy Cuboid<real_t>.
  KOKKOS_INLINE_FUNCTION void initCuboid(const Real L[3]) {
    resetV(8);
    resetF(6);
    gen = 0;
    for (int i = 0; i < CAP; ++i)
      knownGen[i] = 0;
    for (int i = 0; i < 6; ++i) {
      fnbr[i] = -1;  // noNbr
      poff[i] = 0;
      pvec[i][0] = pvec[i][1] = pvec[i][2] = 0;
    }
    const Real hx = Real(0.5) * L[0], hy = Real(0.5) * L[1], hz = Real(0.5) * L[2];
    const Real px[8] = {-hx, hx, -hx, hx, -hx, hx, -hx, hx};
    const Real py[8] = {-hy, -hy, hy, hy, -hy, -hy, hy, hy};
    const Real pz[8] = {-hz, -hz, -hz, -hz, hz, hz, hz, hz};
    for (int i = 0; i < 8; ++i) {
      vpos[i][0] = px[i];
      vpos[i][1] = py[i];
      vpos[i][2] = pz[i];
    }
    flab[0] = makeLabel(0, 0, 0);
    vlab[0][0] = makeLabel(2, 1, 2);
    vlab[1][1] = makeLabel(4, 5, 2);
    vlab[5][1] = makeLabel(5, 4, 0);
    vlab[4][2] = makeLabel(1, 0, 1);
    flab[1] = makeLabel(1, 0, 1);
    vlab[0][1] = makeLabel(0, 4, 2);
    vlab[4][1] = makeLabel(5, 6, 0);
    vlab[6][2] = makeLabel(3, 2, 1);
    vlab[2][0] = makeLabel(2, 0, 2);
    flab[2] = makeLabel(2, 0, 2);
    vlab[0][2] = makeLabel(1, 2, 0);
    vlab[2][2] = makeLabel(3, 3, 0);
    vlab[3][2] = makeLabel(4, 1, 0);
    vlab[1][2] = makeLabel(0, 0, 0);
    flab[3] = makeLabel(3, 3, 0);
    vlab[3][0] = makeLabel(2, 2, 2);
    vlab[2][1] = makeLabel(1, 6, 2);
    vlab[6][1] = makeLabel(5, 7, 0);
    vlab[7][2] = makeLabel(4, 3, 1);
    flab[4] = makeLabel(4, 1, 0);
    vlab[1][0] = makeLabel(2, 3, 2);
    vlab[3][1] = makeLabel(3, 7, 2);
    vlab[7][1] = makeLabel(5, 5, 0);
    vlab[5][2] = makeLabel(0, 1, 1);
    flab[5] = makeLabel(5, 4, 0);
    vlab[4][0] = makeLabel(0, 5, 1);
    vlab[5][0] = makeLabel(4, 7, 1);
    vlab[7][0] = makeLabel(3, 6, 1);
    vlab[6][0] = makeLabel(1, 4, 1);
    computeAllRsq();
  }

  /// Faithful port of CellMaker::cutCell2. Clips the cell by the half-space
  /// {x : x·pv - off <= 0}, attaching the new facet to neighbour `nbr`.
  /// Returns true if the cell was modified. On allocation overflow sets *ovf.
  KOKKOS_INLINE_FUNCTION bool cutCell2(const Real pv[3], Real off, int nbr, bool* ovf) {
    VF_CC(++g_cc.cuts);
    resetDist();
    if (emptyV())
      return false;
    int v1 = firstAliveV();

    bool found = false;
    lbl_t edgeStart = 0;
    if (cdist(v1, pv, off) > 0) {
      int k = 0;
      while (dist[v1] > 0 && k < 3) {
        k = 0;
        while (k < 3 && cdist(getVertex(vlab[v1][k]), pv, off) >= dist[v1])
          ++k;
        if (k < 3) {
          edgeStart = static_cast<lbl_t>((v1 << 2) | k);
          v1 = getVertex(vlab[v1][k]);
        }
      }
      if (dist[v1] <= 0) {
        found = true;
      } else {
        // Round-off fallback: exhaustive search for a sign change.
        computeAllDist(pv, off);
        Real distMax = 0;
        for (int i = 0; i < numAllocV; ++i) {
          if (!aliveV[i] || dist[i] > 0)
            continue;
          for (int k2 = 0; k2 < 3; ++k2) {
            int nv = getVertex(vlab[i][k2]);
            Real nd = cdist(nv, pv, off);
            if (nd > 0) {
              found = true;
              Real newDist = nd - dist[i];
              if (newDist > distMax) {
                edgeStart = vlab[i][k2];
                distMax = newDist;
              }
            }
          }
        }
      }
      if (!found) {  // whole cell on the cut side
        resetV(0);
        resetF(0);
        return true;
      }
    } else {
      int k = 0;
      while (dist[v1] <= 0 && k < 3) {
        k = 0;
        while (k < 3 && cdist(getVertex(vlab[v1][k]), pv, off) <= dist[v1])
          ++k;
        if (k < 3) {
          edgeStart = vlab[v1][k];
          v1 = getVertex(edgeStart);
        }
      }
      found = (dist[v1] > 0);
    }
    if (!found)
      return false;

    // Trace the sign-change loop, inserting new vertices on cut edges.
    lbl_t label = getReverse(edgeStart);
    lbl_t labelRev = getReverse(label);
    const lbl_t lDummy = makeLabel(FDUM, VDUM, 3);
    const int fDummyShifted = (FDUM << 9);
    int nNewV = 0;
    int nFacetPrev = 0;
    int v = getVertex(label);
    int e = getEdge(label);
    int fPrev = getFacet(labelRev);
    int vRev = getVertex(labelRev);
    int eRev = getEdge(labelRev);
    do {
      facetPrev[nFacetPrev++] = fPrev;
      vlab[vRev][eRev] = lDummy;
      int vNew = getFreeV();
      if (vNew < 0) {
        *ovf = true;
        return true;
      }
      newV[nNewV++] = vNew;
      {
        Real dvr = cdist(vRev, pv, off);
        Real dv = cdist(v, pv, off);
        Real lambda = dvr / (dvr - dv);
        for (int k = 0; k < 3; ++k)
          vpos[vNew][k] = lambda * vpos[v][k] + (Real(1) - lambda) * vpos[vRev][k];
      }
      vlab[vNew][0] = label;
      vlab[v][e] = makeLabel(fPrev, vNew, 0);
      flab[fPrev] = vlab[v][e];
      do {
        VF_CC(++g_cc.traceSteps);
        v = vRev;
        e = (eRev == 0 ? 2 : eRev - 1);
        label = labelRev;
        labelRev = vlab[v][e];
        vRev = getVertex(labelRev);
        eRev = getEdge(labelRev);
        if (vRev == VDUM)
          break;
        fPrev = getFacet(vlab[vRev][eRev]);
        vlab[vRev][eRev] = static_cast<lbl_t>(fDummyShifted | (vlab[vRev][eRev] & 0x01ff));
      } while (cdist(vRev, pv, off) > 0);
      int vSwap = vRev, eSwap = eRev;
      vRev = v;
      eRev = e;
      label = labelRev;
      v = vSwap;
      e = eSwap;
    } while (v != VDUM);

    // New facet interconnecting the new vertices.
    {
      int facetNew = getFreeF();
      if (facetNew < 0) {
        *ovf = true;
        return true;
      }
      int imin = nNewV - 1;
      for (int i = 0; i < nNewV; ++i) {
        int iplus = (i + 1 == nNewV) ? 0 : i + 1;
        int vNew = newV[i];
        vlab[vNew][1] = makeLabel(facetPrev[i], newV[imin], 2);
        vlab[vNew][2] = makeLabel(facetNew, newV[iplus], 1);
        imin = i;
      }
      flab[facetNew] = makeLabel(facetNew, newV[0], 1);
      fnbr[facetNew] = nbr;
      pvec[facetNew][0] = pv[0];
      pvec[facetNew][1] = pv[1];
      pvec[facetNew][2] = pv[2];
      poff[facetNew] = off;
    }
    for (int i = 0; i < nNewV; ++i)
      computeRsq(newV[i]);

    // Remove cut-away vertices/facets by DFS from edgeStart's tail vertex.
    bool largestDeleted = false;
    {
      int sv = getVertex(edgeStart);
      releaseV(sv);
      if (sv == vRsqMax)
        largestDeleted = true;
      int top = 0;
      vStack[top++] = sv;
      while (top > 0) {
        VF_CC(++g_cc.dfsSteps);
        int vv = vStack[--top];
        for (int k = 0; k < 3; ++k) {
          int vNxt = getVertex(vlab[vv][k]);
          if (vNxt == VDUM)
            continue;
          int facet = getFacet(vlab[vv][k]);
          if (facet != FDUM && !isFreeF(facet))
            releaseF(facet);
          if (isFreeV(vNxt))
            continue;
          releaseV(vNxt);
          if (vNxt == vRsqMax)
            largestDeleted = true;
          vStack[top++] = vNxt;
        }
      }
    }
    if (largestDeleted)
      findRsqMax();
    return true;
  }

  KOKKOS_INLINE_FUNCTION lbl_t getReverse(lbl_t label) const {
    return vlab[getVertex(label)][getEdge(label)];
  }

  /// Cell volume in seed-relative coordinates (origin = seed). Uses the planar
  /// identity V = (1/3) Σ_facets p0·A_f with A_f = ½ Σ (pi × p{i+1}).
  KOKKOS_INLINE_FUNCTION Real volume() {
    Real vol = 0;
    for (int f = 0; f < numAllocF; ++f) {
      if (!aliveF[f])
        continue;
      lbl_t start = flab[f];
      int v0 = getVertex(start);
      Real ax = 0, ay = 0, az = 0;
      lbl_t cur = start;
      // Walk the facet polygon, accumulate the area vector ½ Σ pi × p{i+1}.
      do {
        lbl_t nxt = getNextLabelCCW(cur);
        int a = getVertex(cur);
        int b = getVertex(nxt);
        ax += vpos[a][1] * vpos[b][2] - vpos[a][2] * vpos[b][1];
        ay += vpos[a][2] * vpos[b][0] - vpos[a][0] * vpos[b][2];
        az += vpos[a][0] * vpos[b][1] - vpos[a][1] * vpos[b][0];
        cur = nxt;
      } while (cur != start);
      vol += vpos[v0][0] * (Real(0.5) * ax) + vpos[v0][1] * (Real(0.5) * ay) +
             vpos[v0][2] * (Real(0.5) * az);
    }
    return vol * (Real(1) / Real(3));
  }

  // Per-facet published geometry (filled by computeGeometry from the half-edge
  // mesh) — the facet area vector and the volume gradient dV the physics forces
  // consume. (The dual edge tensor "edgeInv" is no longer stored per cell — it was a
  // 9 KB CAP×3×3 member that is purely transient; vertexEdgeInv recomputes a vertex's
  // 3×3 block on demand, shrinking the scratch cell ~9 KB. The result is bit-identical:
  // edgeInv[vc][e] = pvec[facet(L)] × pvec[facet(reverse(L))] holds in both the direct
  // and the antisymmetric case since −(b×a)=a×b and FP multiply is commutative.)
  Real fArea[CAP][3];  // facet area vector
  Real fdV[CAP][3];    // facet volume gradient (legacy m_dV)

  /// Normalised dual edge tensor of vertex vc (its 3 edges × 3 components). Recomputed
  /// on demand instead of stored; see the note above.
  KOKKOS_INLINE_FUNCTION void vertexEdgeInv(int vc, Real eInv[3][3]) const {
    for (int e = 0; e < 3; ++e) {
      lbl_t L = vlab[vc][e];
      const Real* a = pvec[getFacet(L)];
      const Real* b = pvec[getFacet(getReverse(L))];
      eInv[e][0] = a[1] * b[2] - a[2] * b[1];
      eInv[e][1] = a[2] * b[0] - a[0] * b[2];
      eInv[e][2] = a[0] * b[1] - a[1] * b[0];
    }
    int indxF = getFacet(vlab[vc][2]);
    Real vol = pvec[indxF][0] * eInv[0][0] + pvec[indxF][1] * eInv[0][1] +
               pvec[indxF][2] * eInv[0][2];
    for (int m = 0; m < 3; ++m)
      for (int c = 0; c < 3; ++c)
        eInv[m][c] /= vol;
  }

  /// Faithful port of CellGeometry::computeEdgeInv + diffVolume: fills fArea, fdV
  /// from the in-scratch half-edge mesh. connV[f] is the plane vector pvec[f]
  /// (= pos[nbr]-pos[seed] for both Voronoi and Power); m_rSq[f] is poff[f].
  KOKKOS_INLINE_FUNCTION void computeGeometry() {
    for (int f = 0; f < numAllocF; ++f)
      if (aliveF[f])
        for (int c = 0; c < 3; ++c) {
          fArea[f][c] = 0;
          fdV[f][c] = 0;
        }
    const int eOpp[3] = {2, 0, 1};
    for (int vc = 0; vc < numAllocV; ++vc) {
      if (!aliveV[vc])
        continue;
      int vN[3], ff[3];
      vN[0] = getVertex(vlab[vc][0]);
      ff[2] = getFacet(vlab[vc][0]);
      vN[1] = getVertex(vlab[vc][1]);
      ff[0] = getFacet(vlab[vc][1]);
      vN[2] = getVertex(vlab[vc][2]);
      ff[1] = getFacet(vlab[vc][2]);
      Real dv[3][3];
      for (int c = 0; c < 3; ++c) {
        dv[0][c] = vpos[vN[0]][c] - vpos[vN[1]][c];
        dv[1][c] = vpos[vN[1]][c] - vpos[vN[2]][c];
        dv[2][c] = vpos[vN[2]][c] - vpos[vN[0]][c];
      }
      Real eInv[3][3];
      vertexEdgeInv(vc, eInv);
      Real dVertex[3][3][3];
      for (int j = 0; j < 3; ++j)
        for (int ii = 0; ii < 3; ++ii)
          for (int l = 0; l < 3; ++l)
            dVertex[j][ii][l] = eInv[eOpp[ii]][l] * (pvec[ff[ii]][j] - vpos[vc][j]);
      for (int m = 0; m < 3; ++m) {
        Real dA[3];
        dA[0] = vpos[vc][1] * dv[m][2] - vpos[vc][2] * dv[m][1];
        dA[1] = vpos[vc][2] * dv[m][0] - vpos[vc][0] * dv[m][2];
        dA[2] = vpos[vc][0] * dv[m][1] - vpos[vc][1] * dv[m][0];
        for (int c = 0; c < 3; ++c)
          fArea[ff[m]][c] += Real(0.25) * dA[c];
        for (int j = 0; j < 3; ++j) {
          for (int ii = 0; ii < 3; ++ii) {
            Real ddA[3];
            ddA[0] = dVertex[j][ii][1] * dv[m][2] - dVertex[j][ii][2] * dv[m][1];
            ddA[1] = dVertex[j][ii][2] * dv[m][0] - dVertex[j][ii][0] * dv[m][2];
            ddA[2] = dVertex[j][ii][0] * dv[m][1] - dVertex[j][ii][1] * dv[m][0];
            for (int c = 0; c < 3; ++c)
              fdV[ff[ii]][j] += pvec[ff[m]][c] * ddA[c] / Real(12);
          }
          fdV[ff[m]][j] += dA[j] / Real(24);
        }
      }
    }
  }

  /// Gradient of facet f's area² w.r.t. the plane positions of f and its
  /// edge-adjacent facets (faithful port of CellGeometry::gradFacetAreaSq), for
  /// the interface-tension force. Requires computeGeometry() first. outF[m] are
  /// cell-local facet slots (outF[0]==f), outG[m] the gradient; numF entries.
  KOKKOS_INLINE_FUNCTION void gradFacetAreaSq(int indxFacet, int outF[], Real outG[][3],
                                              int& numF) {
    lbl_t labels[40];
    int numV = 0;
    lbl_t lstart = flab[indxFacet];
    labels[numV++] = lstart;
    lbl_t lnext = static_cast<lbl_t>(getNextLabelCCW(lstart));
    while (lnext != lstart && numV < 40) {
      labels[numV++] = lnext;
      lnext = static_cast<lbl_t>(getNextLabelCCW(lnext));
    }
    numF = numV + 1;
    outF[0] = indxFacet;
    for (int i = 0; i < numV; ++i)
      outF[i + 1] = getFacet(getReverse(labels[i]));
    for (int i = 0; i < numF; ++i)
      for (int c = 0; c < 3; ++c)
        outG[i][c] = 0;
    const Real* A = fArea[indxFacet];
    int nbrPrev = numV, nbrNext = 1;
    for (int i = 0; i < numV; ++i) {
      int vc = getVertex(labels[i]);
      int e0 = getEdge(labels[i]);
      int e1 = (e0 == 2 ? 0 : e0 + 1);
      int e2 = (e0 == 0 ? 2 : e0 - 1);
      int f0 = getFacet(vlab[vc][e1]);
      int f1 = getFacet(vlab[vc][e2]);
      int f2 = getFacet(vlab[vc][e0]);
      int v0 = getVertex(vlab[vc][e0]);
      int v1 = getVertex(vlab[vc][e1]);
      Real dv[3];
      for (int c = 0; c < 3; ++c)
        dv[c] = vpos[v0][c] - vpos[v1][c];
      Real dvA[3] = {dv[1] * A[2] - dv[2] * A[1], dv[2] * A[0] - dv[0] * A[2],
                     dv[0] * A[1] - dv[1] * A[0]};
      Real eInv[3][3];
      vertexEdgeInv(vc, eInv);
      Real s = eInv[e0][0] * dvA[0] + eInv[e0][1] * dvA[1] + eInv[e0][2] * dvA[2];
      for (int c = 0; c < 3; ++c)
        outG[nbrPrev][c] += (pvec[f1][c] - vpos[vc][c]) * s;
      s = eInv[e1][0] * dvA[0] + eInv[e1][1] * dvA[1] + eInv[e1][2] * dvA[2];
      for (int c = 0; c < 3; ++c)
        outG[nbrNext][c] += (pvec[f2][c] - vpos[vc][c]) * s;
      s = eInv[e2][0] * dvA[0] + eInv[e2][1] * dvA[1] + eInv[e2][2] * dvA[2];
      for (int c = 0; c < 3; ++c)
        outG[0][c] += (pvec[f0][c] - vpos[vc][c]) * s;
      nbrPrev = nbrNext;
      ++nbrNext;
    }
  }

  /// Outward area vector of facet f (½ Σ pi × p{i+1} over its polygon loop).
  KOKKOS_INLINE_FUNCTION void facetAreaVec(int f, Real out[3]) {
    Real ax = 0, ay = 0, az = 0;
    lbl_t start = flab[f];
    lbl_t cur = start;
    do {
      lbl_t nxt = getNextLabelCCW(cur);
      int a = getVertex(cur);
      int b = getVertex(nxt);
      ax += vpos[a][1] * vpos[b][2] - vpos[a][2] * vpos[b][1];
      ay += vpos[a][2] * vpos[b][0] - vpos[a][0] * vpos[b][2];
      az += vpos[a][0] * vpos[b][1] - vpos[a][1] * vpos[b][0];
      cur = nxt;
    } while (cur != start);
    out[0] = Real(0.5) * ax;
    out[1] = Real(0.5) * ay;
    out[2] = Real(0.5) * az;
  }

  KOKKOS_INLINE_FUNCTION int countFacets() const {
    int n = 0;
    for (int i = 0; i < numAllocF; ++i)
      if (aliveF[i])
        ++n;
    return n;
  }
  KOKKOS_INLINE_FUNCTION int countVertices() const {
    int n = 0;
    for (int i = 0; i < numAllocV; ++i)
      if (aliveV[i])
        ++n;
    return n;
  }
};

/**
 * Build one Voronoi/Power cell from a neighbour list in seed-relative
 * coordinates. Faithful to CellMaker::processNbrs: closest-first cut order with
 * the unweighted security-radius early-out.
 *
 * REQUIREMENT: for the Unweighted policy the neighbours must be sorted by
 * ascending rSqHalf (= ½|rel|²) so the early-out is valid — exactly the legacy
 * std::sort(CompareNbrDist). The Power policy applies every candidate (a distant
 * heavy seed can still claim a facet), so order only affects degenerate ties,
 * which the global-id ordering of the input resolves deterministically.
 *
 * @tparam Weighted  false => Voronoi (offset = rSqHalf); true => Power/Laguerre
 *                   (offset = rSqHalf + ½(wSelf - wNbr), the radical plane).
 */
template <class Real, bool Weighted>
KOKKOS_INLINE_FUNCTION CutStatus buildVoronoiCell(ScratchCell<Real>& c, const Real L[3],
                                                  const Real* relx, const Real* rely,
                                                  const Real* relz, const int* ids,
                                                  const Real* wNbr, int nNbr, Real wSelf) {
  c.initCuboid(L);
  bool ovf = false;
  for (int n = 0; n < nNbr; ++n) {
    const Real rx = relx[n], ry = rely[n], rz = relz[n];
    const Real rSqHalf = Real(0.5) * (rx * rx + ry * ry + rz * rz);
    Real off;
    if constexpr (Weighted)
      off = rSqHalf + Real(0.5) * (wSelf - wNbr[n]);
    else
      off = rSqHalf;
    if constexpr (!Weighted) {
      // Security radius: once the plane offset exceeds twice the cell's largest
      // squared vertex radius, no remaining (farther) seed can cut. Break.
      if (!(off < Real(2) * c.rsq[c.vRsqMax]))
        break;
    }
    const Real pv[3] = {rx, ry, rz};
    c.cutCell2(pv, off, ids[n], &ovf);
    if (ovf)
      return CutStatus::Overflow;
  }
  if (c.emptyV())
    return CutStatus::Empty;
  return CutStatus::Ok;
}

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_CELL_CUTTER_HPP
