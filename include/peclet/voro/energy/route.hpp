/**
 * @file energy/route.hpp
 * \brief Rung A3: route a cell's per-facet geometry gradient to the seed DOFs on the PUBLISHED
 * view — the one combiner every energy term uses (no cell reconstruction).
 *
 * Given, for cell i, g_k = ∂E/∂n_k for each published facet k (SoA, local facet index), the
 * neighbour facets go through the plane policy's chain (Voronoi: ∓½ g; Power: the radical-plane
 * Jacobians from the current positions + weights) onto the cell's own seed and the neighbour
 * seed, the SDF wall facets are aggregated and routed through the seed-foot wall chain
 * (energy/wall.hpp) onto the own seed, and the box facets carry no DOF dependence. This is
 * chainToDofs + addSdfWallForce expressed on TessellationView data.
 *
 * Deterministic combine, no floating-point atomics: an energy kernel writes each contribution to
 * its own slot (RouteBuffers: the neighbour term of facet g at g, the own-seed term at the cell),
 * and gatherRoute then has every target cell add its incoming terms in ascending SOURCE-cell
 * order — the facets of one source in their local order, the target's own term at its own index.
 * That is the order the serial (one-thread) atomic scatter summed them in, so one thread is
 * bit-identical to it and any thread count gives the same bits. The incoming lists are a
 * transpose of facetNbr (integer count + fill, then an insertion sort per target: the idiom of
 * nl.binSort and tess.binSort), not the reciprocal-facet map, so a facet without a reciprocal still
 * delivers its term.
 */
#ifndef PECLET_VORO_ENERGY_ROUTE_HPP
#define PECLET_VORO_ENERGY_ROUTE_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/sdf.hpp"
#include "peclet/voro/tess_grid.hpp"  // scratchView
#include "peclet/voro/tessellation_view.hpp"

namespace peclet::voro::energy {

namespace detail {

/// Facets one cell may publish (CellBuilder::MAXF_TMP); bounds the per-thread gradient array.
constexpr int kMaxLocalFacets = 50;

/// Persistent backing arrays of the combiner (scratchView slots), held by a caller that routes
/// every step (PotentialScratch); null scratch allocates per call.
template <class Real>
struct RouteScratch {
  using Mem = peclet::core::MemSpace;
  Kokkos::View<Real*, Mem> fn, fwn, fs, fws, fx;
  Kokkos::View<unsigned char*, Mem> on, own;
  Kokkos::View<int*, Mem> start, fill, item, src;
};

/// One routing pass's contribution slots. `fn` / `fwn`: facet g's term on its neighbour seed
/// (valid iff on(g)); `fs` / `fws`: cell i's own-seed term (own(i) bit 1); `fx`: an own-seed term a
/// caller adds after the route — the exact wall part (own(i) bit 2). `fwn` / `fws` are empty
/// unless the policy has a weight DOF and the caller passed forceW.
template <class Real>
struct RouteBuffers {
  using Mem = peclet::core::MemSpace;
  Kokkos::View<Real*, Mem> fn, fwn, fs, fws, fx;
  Kokkos::View<unsigned char*, Mem> on, own;

  /// The caller's extra own-seed term of cell i (one call per cell, after routeCellGradient).
  KOKKOS_INLINE_FUNCTION void addOwn(int i, Real x, Real y, Real z) const {
    fx(3 * i) = x;
    fx(3 * i + 1) = y;
    fx(3 * i + 2) = z;
    own(i) |= 2;
  }
};

/// Slots for one routing pass over `view` (zeroed flags; values are written before they are read).
template <class Real, class Policy>
RouteBuffers<Real> beginRoute(const TessellationView<Real>& view,
                              const Kokkos::View<Real*, peclet::core::MemSpace>& forceW,
                              RouteScratch<Real>* scr = nullptr) {
  const std::size_t N = static_cast<std::size_t>(view.numCells());
  const std::size_t nF = static_cast<std::size_t>(view.numFacets());
  const std::size_t nW = (Policy::kHasWeightDof && forceW.extent(0) > 0) ? 1 : 0;
  RouteBuffers<Real> rb;
  rb.fn = scratchView(scr ? &scr->fn : nullptr, 3 * nF, "route.fn", false);
  rb.fwn = scratchView(scr ? &scr->fwn : nullptr, nW * nF, "route.fwn", false);
  rb.fs = scratchView(scr ? &scr->fs : nullptr, 3 * N, "route.fs", false);
  rb.fws = scratchView(scr ? &scr->fws : nullptr, nW * N, "route.fws", false);
  rb.fx = scratchView(scr ? &scr->fx : nullptr, 3 * N, "route.fx", false);
  rb.on = scratchView(scr ? &scr->on : nullptr, nF, "route.on", true);
  rb.own = scratchView(scr ? &scr->own : nullptr, N, "route.own", true);
  return rb;
}

/// Route cell i's per-facet gradient `gl` (3 per local facet, nf facets) into the slots of `rb`.
/// `pos` / `weight` / `L` feed the weighted policy's chain (unused by Voronoi; pass empty views).
template <class Real, class Policy, class Sdf>
KOKKOS_INLINE_FUNCTION void routeCellGradient(
    const TessellationView<Real>& view, int i, const Real* gl, int nf,
    const Kokkos::View<Real*, peclet::core::MemSpace>& pos,
    const Kokkos::View<Real*, peclet::core::MemSpace>& weight, Real L, const Sdf& sdf,
    const RouteBuffers<Real>& rb) {
  const int N = view.numCells();
  const int b = view.facetBegin(i);
  const Real Lh = Real(0.5) * L;
  Real fSelf[3] = {Real(0), Real(0), Real(0)};
  Real fwSelf = Real(0);
  double gw[3] = {0.0, 0.0, 0.0};
  Real sx = Real(0), sy = Real(0), sz = Real(0);
  if constexpr (Policy::kHasWeightDof) {
    sx = pos(3 * i);
    sy = pos(3 * i + 1);
    sz = pos(3 * i + 2);
  }
  for (int k = 0; k < nf; ++k) {
    const Real g[3] = {gl[3 * k], gl[3 * k + 1], gl[3 * k + 2]};
    if (g[0] == Real(0) && g[1] == Real(0) && g[2] == Real(0))
      continue;
    const int j = view.facetNbr(b + k);
    if (j == kBoundaryFacet) {
      gw[0] += g[0];
      gw[1] += g[1];
      gw[2] += g[2];
      continue;
    }
    if (j < 0 || j >= N)
      continue;  // box facet: no DOF dependence
    Real r[3] = {Real(0), Real(0), Real(0)}, rho = Real(0), cc = Real(0);
    if constexpr (Policy::kHasWeightDof) {
      const Real s[3] = {sx, sy, sz};
      for (int d = 0; d < 3; ++d) {
        Real rr = pos(3 * j + d) - s[d];
        rr = rr > Lh ? rr - L : (rr < -Lh ? rr + L : rr);
        r[d] = rr;
      }
      rho = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      cc = weight(i) - weight(j);
    }
    Real fs[3], fn[3], fws = Real(0), fwn = Real(0);
    Policy::template chain<Real>(g, r, rho, cc, fs, fn, fws, fwn);
    fSelf[0] += fs[0];
    fSelf[1] += fs[1];
    fSelf[2] += fs[2];
    fwSelf += fws;
    const int f = b + k;
    rb.fn(3 * f) = fn[0];
    rb.fn(3 * f + 1) = fn[1];
    rb.fn(3 * f + 2) = fn[2];
    if constexpr (Policy::kHasWeightDof)
      if (rb.fwn.extent(0) > 0)
        rb.fwn(f) = fwn;
    rb.on(f) = 1;
  }
  if constexpr (!std::is_same_v<Sdf, NoSdf>) {
    double fw[3] = {0.0, 0.0, 0.0};
    Real px = pos.extent(0) > 0 ? pos(3 * i) : Real(0);
    Real py = pos.extent(0) > 0 ? pos(3 * i + 1) : Real(0);
    Real pz = pos.extent(0) > 0 ? pos(3 * i + 2) : Real(0);
    sdfWallChain<Real>(sdf, px, py, pz, gw, fw);
    fSelf[0] += (Real)fw[0];
    fSelf[1] += (Real)fw[1];
    fSelf[2] += (Real)fw[2];
  } else {
    (void)sdf;
  }
  rb.fs(3 * i) = fSelf[0];
  rb.fs(3 * i + 1) = fSelf[1];
  rb.fs(3 * i + 2) = fSelf[2];
  if constexpr (Policy::kHasWeightDof)
    if (rb.fws.extent(0) > 0)
      rb.fws(i) = fwSelf;
  rb.own(i) |= 1;
}

/// Add the routed slots of `rb` into `force` (3N) and `forceW` (N, when rb carries weight slots):
/// each target cell sums its incoming facet terms in ascending (source cell, source facet) order,
/// its own terms (route, then the caller's extra) at its own index — the serial scatter's order.
template <class Real>
void gatherRoute(const TessellationView<Real>& view, const RouteBuffers<Real>& rb,
                 const Kokkos::View<Real*, peclet::core::MemSpace>& force,
                 const Kokkos::View<Real*, peclet::core::MemSpace>& forceW,
                 RouteScratch<Real>* scr = nullptr) {
  using Exec = peclet::core::ExecSpace;
  const int N = view.numCells();
  const std::size_t nF = static_cast<std::size_t>(view.numFacets());
  const auto start = scratchView(scr ? &scr->start : nullptr, static_cast<std::size_t>(N) + 1,
                                 "route.start", true);
  const auto fill =
      scratchView(scr ? &scr->fill : nullptr, static_cast<std::size_t>(N), "route.fill", true);
  const auto item = scratchView(scr ? &scr->item : nullptr, nF, "route.item", false);
  const auto src = scratchView(scr ? &scr->src : nullptr, nF, "route.src", false);
  const RouteBuffers<Real> B = rb;
  const bool withW = rb.fws.extent(0) > 0;

  // ---- transpose of the routed facets: incoming list per target (integer count + fill) ----
  Kokkos::parallel_for(
      "energy.route.count", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        if (!(B.own(i) & 1))
          return;  // not routed: no facet slot set
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f)
          if (B.on(f))
            Kokkos::atomic_inc(&start(view.facetNbr(f) + 1));
      });
  Kokkos::parallel_scan(
      "energy.route.scan", N + 1, KOKKOS_LAMBDA(const int t, int& upd, const bool fin) {
        upd += start(t);
        if (fin)
          start(t) = upd;
      });
  Kokkos::parallel_for(
      "energy.route.fill", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int i) {
        if (!(B.own(i) & 1))
          return;
        for (int f = view.facetBegin(i); f < view.facetEnd(i); ++f)
          if (B.on(f)) {
            const int t = view.facetNbr(f);
            const int p = start(t) + Kokkos::atomic_fetch_add(&fill(t), 1);
            item(p) = f;
            src(p) = i;
          }
      });

  // ---- per target: fixed order (insertion sort by source cell, then facet), then sum ----
  Kokkos::parallel_for(
      "energy.route.gather", Kokkos::RangePolicy<Exec>(0, N), KOKKOS_LAMBDA(const int t) {
        const int p0 = start(t), p1 = start(t + 1);
        for (int p = p0 + 1; p < p1; ++p) {
          const int vf = item(p), vs = src(p);
          int q = p - 1;
          while (q >= p0 && (src(q) > vs || (src(q) == vs && item(q) > vf))) {
            item(q + 1) = item(q);
            src(q + 1) = src(q);
            --q;
          }
          item(q + 1) = vf;
          src(q + 1) = vs;
        }
        const unsigned char own = B.own(t);
        if (p0 == p1 && own == 0)
          return;
        Real a[3] = {force(3 * t), force(3 * t + 1), force(3 * t + 2)};
        Real aw = withW ? forceW(t) : Real(0);
        int p = p0;
        for (; p < p1 && src(p) <= t; ++p) {  // sources below t, and t's own self-image facets
          const int f = item(p);
          for (int c = 0; c < 3; ++c)
            a[c] += B.fn(3 * f + c);
          if (withW)
            aw += B.fwn(f);
        }
        if (own & 1) {
          for (int c = 0; c < 3; ++c)
            a[c] += B.fs(3 * t + c);
          if (withW)
            aw += B.fws(t);
        }
        if (own & 2)
          for (int c = 0; c < 3; ++c)
            a[c] += B.fx(3 * t + c);
        for (; p < p1; ++p) {
          const int f = item(p);
          for (int c = 0; c < 3; ++c)
            a[c] += B.fn(3 * f + c);
          if (withW)
            aw += B.fwn(f);
        }
        for (int c = 0; c < 3; ++c)
          force(3 * t + c) = a[c];
        if (withW)
          forceW(t) = aw;
      });
}

}  // namespace detail
}  // namespace peclet::voro::energy

#endif  // PECLET_VORO_ENERGY_ROUTE_HPP
