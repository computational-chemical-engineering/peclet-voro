/**
 * @file params.hpp
 * @brief The named engine defaults of peclet.voro — one header-level home for every capacity and
 *        default the library and its Python bindings (`peclet.voro.defaults`) drive the engine
 *        with (QUALITY_PLAN G.7). A template default or a default argument elsewhere in
 *        include/peclet/voro/ names one of these; it never repeats the literal.
 *
 * Layer 0 (no voro includes).
 */
#ifndef PECLET_VORO_PARAMS_HPP
#define PECLET_VORO_PARAMS_HPP

namespace peclet::voro {

/// ConvexCell capacity of the resident (moving-point) tessellation: planes and dual triangles
/// (= primal vertices) per cell — the production layout and the topology store's strides
/// (README "Early wall clip"). The plane cap must fit a byte (ConvexCell's static_assert).
constexpr int kMaxPlanes = 64;
constexpr int kMaxTriangles = 112;

/// Capacity of the pore-space (SDF-walled interstitial) cell reconstruction: wall-cut cells
/// carry more planes than a Poisson–Voronoi cell.
constexpr int kPoreMaxPlanes = 128;
constexpr int kPoreMaxTriangles = 256;

/// Gather window of the tessellator's counting-sort grid: blocks per axis around a seed
/// (coverage = window · block size); the moving-point path and the cold build share it.
constexpr int kSearchWindow = 4;

/// Moving-point repair: the certificate tolerance and the Verlet skin, both as fractions of the
/// mean spacing cbrt(V/N) (MovingTessellation::alloc takes the absolute values).
constexpr double kCertificateTolerance = 1e-4;
constexpr double kSkin = 0.25;

/// Mesh optimisers: the CG iteration cap of the Gauss–Newton solve, and the per-iteration decay
/// of the pore-space optimiser's log-barrier weight.
constexpr int kOptimizerCgIters = 300;
constexpr double kBarrierDecay = 0.7;

}  // namespace peclet::voro

#endif  // PECLET_VORO_PARAMS_HPP
