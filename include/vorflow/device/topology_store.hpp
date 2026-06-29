/**
 * @file device/topology_store.hpp
 * \brief Compact, resident per-cell ConvexCell topology — the Part II (moving-points) persistence layer.
 *
 * The cold build (`buildTessellation` / `buildConvexCell`) discards each cell's ConvexCell after emitting
 * the facet CSR. For the moving-particle loop we instead PERSIST the topology so a step can re-evaluate
 * geometry on the fixed topology (`ConvexCell::reevalGeometry`) without re-gathering/re-clipping. Storing the
 * whole ~3 KB cell per seed is wasteful and memory-bound; the topology alone is tiny and recompute-cheap:
 *
 *   - np, nt                  : plane / triangle high-water marks (the 6 box planes are implicit, re-seeded
 *                               by `ConvexCell::initBoxPlanes`, so only planes [6,np) carry a neighbour id);
 *   - pnbr[N*MAXP]            : neighbour seed id per plane (<0 ⇒ bounding box);
 *   - tri[N*MAXT]             : packed dual triangle (t0 | t1<<8 | t2<<16 | alive<<24).
 *
 * To reload: `initBoxPlanes` → set np/nt → copy pnbr[6..np) and the packed triangles → `reevalGeometry`
 * (which rebuilds the neighbour planes from the current seed positions and recomputes every vertex). This is
 * the same compact layout measured in tests/kokkos/bench_incremental.cpp (the 29→65 Mc/s re-eval win).
 *
 * Header-only, Kokkos only. The struct holds Views (reference-counted handles) so it captures by value into
 * a device lambda; save()/load() are KOKKOS_INLINE_FUNCTION and templated on the ConvexCell instantiation.
 */
#ifndef VORFLOW_DEVICE_TOPOLOGY_STORE_HPP
#define VORFLOW_DEVICE_TOPOLOGY_STORE_HPP

#include <string>
#include <Kokkos_Core.hpp>

#include "tpx/common/view.hpp"

namespace vor {
namespace device {

/// Resident compact topology for N cells of ConvexCell<Real, MAXP, MAXT>. Captured by value into kernels.
template <int MAXP, int MAXT>
struct TopologyStore {
  using MemSpace = tpx::MemSpace;
  int N = 0;
  Kokkos::View<int*, MemSpace> np;        // N
  Kokkos::View<int*, MemSpace> nt;        // N
  Kokkos::View<int*, MemSpace> pnbr;      // N*MAXP : neighbour seed id per plane (<0 = box)
  Kokkos::View<unsigned*, MemSpace> tri;  // N*MAXT : t0 | t1<<8 | t2<<16 | alive<<24
  Kokkos::View<unsigned*, MemSpace> poke; // N*MAXT : per-triangle 3 edge-opposite plane indices (local
                                          // certificate; allocated only when allocPoke() is called)

  void alloc(int n) {
    using Kokkos::view_alloc;
    using Kokkos::WithoutInitializing;
    N = n;
    np = Kokkos::View<int*, MemSpace>("topo.np", n);
    nt = Kokkos::View<int*, MemSpace>("topo.nt", n);
    pnbr = Kokkos::View<int*, MemSpace>(view_alloc(std::string("topo.pnbr"), WithoutInitializing),
                                        (size_t)n * MAXP);
    tri = Kokkos::View<unsigned*, MemSpace>(view_alloc(std::string("topo.tri"), WithoutInitializing),
                                            (size_t)n * MAXT);
  }

  /// Allocate the per-triangle poke-plane store (the local-Delaunay certificate; opt-in so callers that
  /// don't use it pay no memory). N*MAXT unsigned ≈ MAXT·4 bytes/cell.
  void allocPoke() {
    poke = Kokkos::View<unsigned*, MemSpace>(
        Kokkos::view_alloc(std::string("topo.poke"), Kokkos::WithoutInitializing), (size_t)N * MAXT);
  }

  /// Persist cell `c` at slot i (call after the cell is finalised — clipped + complete).
  template <class Cell>
  KOKKOS_INLINE_FUNCTION void save(int i, const Cell& c) const {
    np(i) = c.np;
    nt(i) = c.nt;
    for (int k = 0; k < c.np; ++k) pnbr((size_t)i * MAXP + k) = c.pnbr[k];
    for (int t = 0; t < c.nt; ++t)
      tri((size_t)i * MAXT + t) = (unsigned)c.t0[t] | ((unsigned)c.t1[t] << 8) |
                                  ((unsigned)c.t2[t] << 16) | ((c.alive[t] ? 1u : 0u) << 24);
  }

  /// Reload the stored topology of slot i into a fresh cell, ready for reevalGeometry(): seeds the six box
  /// planes (initBoxPlanes), restores np/nt/overflow, the neighbour ids of planes [6,np), and the triangles.
  /// The neighbour PLANE equations and all vertices are (re)computed by the following reevalGeometry() call.
  template <class Cell, class Real>
  KOKKOS_INLINE_FUNCTION void load(int i, Cell& c, Real L0, Real L1, Real L2) const {
    c.initBoxPlanes(L0, L1, L2);
    const int npi = np(i), nti = nt(i);
    c.np = npi;
    c.nt = nti;
    c.overflow = false;
    for (int k = 6; k < npi; ++k) c.pnbr[k] = pnbr((size_t)i * MAXP + k);
    for (int t = 0; t < nti; ++t) {
      const unsigned w = tri((size_t)i * MAXT + t);
      c.t0[t] = (unsigned char)(w & 0xffu);
      c.t1[t] = (unsigned char)((w >> 8) & 0xffu);
      c.t2[t] = (unsigned char)((w >> 16) & 0xffu);
      c.alive[t] = ((w >> 24) & 1u) != 0u;
    }
  }
};

}  // namespace device
}  // namespace vor

#endif  // VORFLOW_DEVICE_TOPOLOGY_STORE_HPP
