/// @file
/// @brief Internal helpers shared by the translation units of the `peclet.voro` nanobind module.
///
/// Not an installed header: `src/` only.  Everything here is host-side glue (array conversion,
/// the domain/mode contract checks, the device SDF scene holder, the typed optimiser results) —
/// no device kernel is instantiated by including it.
#ifndef PECLET_VORO_BINDINGS_COMMON_HPP
#define PECLET_VORO_BINDINGS_COMMON_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <Kokkos_Core.hpp>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/python/kokkos_teardown.hpp"
#include "peclet/core/python/ndarray_interop.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/sdf.hpp"

namespace nb = nanobind;
using real_t = double;
using DView = Kokkos::View<real_t*, peclet::core::MemSpace>;

// A NAMED namespace on purpose: Tess / Flow / Sim have virtual methods (Releasable), and under
// hipcc/lld a virtual class in an anonymous namespace leaves its vtable unemitted in the host
// object ("undefined hidden symbol: vtable for (anonymous namespace)::Tess"). Same fix as
// core/python/amr_bindings.cpp.
namespace peclet::voro::pybind {

// --------------------------------------------------------------------------------------------------
// The named defaults (QUALITY_PLAN G.7 "TessellationParams"): every literal the bound surface
// drives the engine with, in one place. `peclet.voro.defaults` exposes them and the docstrings
// below are generated from them.
// --------------------------------------------------------------------------------------------------
namespace defaults {
// The engine-level defaults live in include/peclet/voro/params.hpp (the template defaults and
// default arguments of the headers name them); re-exported here, never repeated: the ConvexCell
// capacities of the resident tessellation and of the pore-space reconstruction, the grid gather
// window, the repair tolerance + Verlet skin (fractions of the mean spacing cbrt(V/N)), the
// optimisers' CG cap and the log-barrier decay.
using peclet::voro::kBarrierDecay;
using peclet::voro::kCertificateTolerance;
using peclet::voro::kMaxPlanes;
using peclet::voro::kMaxTriangles;
using peclet::voro::kOptimizerCgIters;
using peclet::voro::kPoreMaxPlanes;
using peclet::voro::kPoreMaxTriangles;
using peclet::voro::kSearchWindow;
using peclet::voro::kSkin;
// SDF geometry: the central-difference step of the SDF gradient; the wall re-gather skin
// (fraction of the mean spacing) of the 'skin' wall mode.
constexpr double kSdfGradientStep = 1e-5;
constexpr double kWallSkin = 0.0;
// Mesh optimisers (Gauss-Newton on Σ(V/V_ref − 1)², the interface minimiser), bound surface only:
// the optimiser's own search window, iteration caps, gradient tolerance, the pore-space CG cap,
// and the interface tension.
constexpr int kOptimizerSearchWindow = 5;
constexpr int kPoreSearchWindow = 6;
constexpr int kOptimizerMaxIter = 60;
constexpr int kPoreMaxIter = 80;
constexpr double kOptimizerTolerance = 1e-9;
constexpr int kPoreCgIters = 400;
constexpr double kInterfaceSigma = 1.0;
// The host pore-cell oracle's cap on its counting-sort grid resolution (bins per axis).
constexpr int kPoreMaxBins = 96;
// Distributed repair driver (validated by tests/kokkos_mpi/bench_repair_mpi at np = 1, 2, 4): the
// ghost cutoff, in mean spacings, and the ORB granularity per axis.
constexpr double kDistributedRcut = 3.5;
constexpr long kDistributedCells = 16;
}  // namespace defaults

// A docstring assembled at module init from the defaults (nanobind keeps the pointer; the deque
// keeps the storage stable).
inline const char* doc(std::string s) {
  static std::deque<std::string> keep;
  keep.push_back(std::move(s));
  return keep.back().c_str();
}
inline std::string fmt(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}
inline std::string fmt(int v) {
  return std::to_string(v);
}

// (N,3) c-contiguous array -> flat row-major host vector of length 3N.
inline std::vector<real_t> flatten3(nb::ndarray<real_t, nb::c_contig> a) {
  if (a.ndim() != 2 || a.shape(1) != 3)
    throw std::runtime_error("expected an (N,3) array");
  const real_t* p = a.data();
  return std::vector<real_t>(p, p + static_cast<std::size_t>(a.shape(0)) * 3);
}

// (N,) array -> host vector of length N.
inline std::vector<real_t> flatten1(nb::ndarray<real_t, nb::c_contig> a) {
  return peclet::core::python::ndarray_to_vector<real_t>(nb::ndarray<>(a));
}

// The union-of-spheres wall SDF is cubic-periodic (one L), so the pore-space functions take the
// suite-wide `extent` triple and check it is a cube.
inline real_t cubicExtent(std::array<real_t, 3> e, const char* fn) {
  if (!(e[0] > real_t(0)) || e[0] != e[1] || e[1] != e[2])
    throw std::invalid_argument(std::string("voro: ") + fn +
                                "(extent=...) must be a cubic box (Lx == Ly == Lz > 0) — the "
                                "union-of-spheres wall SDF is cubic-periodic.");
  return e[0];
}

// ---- SDF geometry from Python (rung A0) ---------------------------------------------------------
using NoSdfT = peclet::voro::NoSdf;
using SceneT = peclet::voro::SdfScene<real_t>;
using Mem = peclet::core::MemSpace;

// A device-resident core shape scene + the SdfScene provider over it. The node table comes from the
// flat node encoding (3 int32 + 16 float64 per node) that peclet.core.geom.Scene.encode() returns
// (the same arrays dem.add_analytic_wall takes); the encoding carries no sampled grids, so this is
// the ANALYTIC vocabulary (primitives + CSG + transforms).
struct SceneHolder {
  Kokkos::View<peclet::core::geom::ShapeNode<real_t>*, Mem> nodes;
  Kokkos::View<peclet::core::geom::GridDesc<real_t>*, Mem> grids;
  Kokkos::View<float*, Mem> pool;
  SceneT scene;
  bool set = false;
  void clear() {
    nodes = {};
    grids = {};
    pool = {};
    scene = SceneT{};
    set = false;
  }
};

inline SceneHolder makeSceneHolder(nb::ndarray<int, nb::c_contig> node_ints,
                                   nb::ndarray<real_t, nb::c_contig> node_reals, int root,
                                   real_t grad_h) {
  using namespace peclet::core::geom;
  std::vector<int> ni(node_ints.data(), node_ints.data() + node_ints.size());
  std::vector<real_t> nr(node_reals.data(), node_reals.data() + node_reals.size());
  SceneBuilder<real_t> b = SceneBuilder<real_t>::decode(ni, nr, {}, {}, {}, {});
  const auto& nodes = b.nodes();
  if (nodes.empty())
    throw std::runtime_error("set_geometry: empty node table");
  if (root < 0 || root >= static_cast<int>(nodes.size()))
    throw std::runtime_error("set_geometry: root node index out of range");
  SceneHolder h;
  h.nodes = Kokkos::View<ShapeNode<real_t>*, Mem>("scene.nodes", nodes.size());
  {
    auto hn = Kokkos::create_mirror_view(h.nodes);
    for (std::size_t i = 0; i < nodes.size(); ++i)
      hn(i) = nodes[i];
    Kokkos::deep_copy(h.nodes, hn);
  }
  const auto& grids = b.grids();
  const auto& pool = b.samples();
  h.grids = Kokkos::View<GridDesc<real_t>*, Mem>("scene.grids", grids.empty() ? 1 : grids.size());
  h.pool = Kokkos::View<float*, Mem>("scene.pool", pool.empty() ? 1 : pool.size());
  if (!grids.empty()) {
    auto hg = Kokkos::create_mirror_view(h.grids);
    for (std::size_t i = 0; i < grids.size(); ++i)
      hg(i) = grids[i];
    Kokkos::deep_copy(h.grids, hg);
  }
  if (!pool.empty())
    Kokkos::deep_copy(h.pool,
                      Kokkos::View<const float*, Kokkos::HostSpace>(pool.data(), pool.size()));
  h.scene = SceneT{h.nodes, h.grids, h.pool, static_cast<int>(nodes.size()), root, grad_h};
  h.set = true;
  return h;
}

// The suite-wide domain contract of this engine (docs/NAMING.md 1.1): `extent` is the box SIZE;
// the tessellator's periodic box always starts at the origin and is periodic on all three axes, so
// `origin` and `periodic` exist to be CHECKED rather than stored — a caller who writes what every
// other code in the suite writes gets an error naming the limitation instead of a silently
// ignored argument.
inline void checkDomain(const char* cls, std::array<real_t, 3> extent, std::array<real_t, 3> origin,
                        std::array<bool, 3> periodic) {
  for (int a = 0; a < 3; ++a) {
    if (!(extent[a] > real_t(0)))
      throw std::invalid_argument(std::string("voro: ") + cls +
                                  ".set_domain(extent=...) needs three positive lengths.");
    if (origin[a] != real_t(0))
      throw std::invalid_argument(std::string("voro: ") + cls +
                                  ".set_domain(origin=...) must be (0, 0, 0) — the tessellator's "
                                  "periodic box is anchored at the origin. Shift your points "
                                  "instead.");
    if (!periodic[a])
      throw std::invalid_argument(std::string("voro: ") + cls +
                                  ".set_domain(periodic=...) must be (True, True, True) — this "
                                  "engine has no non-periodic axis. Use a wall SDF (set_geometry) "
                                  "to bound the domain.");
  }
}

// Wall re-gather policy of the incremental step, as a string mode.
constexpr const char* kWallModeList = "'exact', 'skin'";
inline bool parseWallMode(const std::string& mode) {
  if (mode == "exact")
    return true;
  if (mode == "skin")
    return false;
  throw std::invalid_argument("voro: set_wall_mode(mode='" + mode +
                              "') is not a wall mode; accepted: " + kWallModeList + ".");
}

inline nb::ndarray<nb::numpy, real_t> toNumpy3(std::vector<real_t> v) {
  const std::size_t N = v.size() / 3;
  return peclet::core::python::vector_to_ndarray(std::move(v), {N, std::size_t(3)}, {3, 1});
}
inline nb::ndarray<nb::numpy, real_t> toNumpy1(std::vector<real_t> v) {
  const std::size_t N = v.size();
  return peclet::core::python::vector_to_ndarray(std::move(v), {N}, {1});
}

// --------------------------------------------------------------------------------------------------
// Typed results of the mesh optimisers (no dict-of-strings, no field reuse across meanings).
// --------------------------------------------------------------------------------------------------
struct OptimizeResult {
  nb::object positions;  ///< (N,3) float64
  nb::object weights;    ///< (N,) float64 when the power weights were optimised, else None
  int iters = 0;
  double max_vol_err = 0, mean_vol_err = 0;
  bool converged = false;
  int num_empty = 0;
};

struct InterfaceResult {
  nb::object positions;  ///< (N,3) float64
  double energy = 0, energy_ratio = 1;
  int iters = 0;
  bool converged = false;
};

// ---- the per-subsystem binding registrars (one translation unit each) ---------------------------
// The module is split by subsystem so that nvcc compiles the device instantiations of the
// tessellator, the FV solvers, the moving-cell fluid, the mesh optimisers and the pore-space
// family CONCURRENTLY under `make -j` instead of serially inside one object.  Each TU owns its
// classes and registers them here; `voro_bindings.cpp` holds only NB_MODULE.
void bindOptimizers(nb::module_& m);
void bindPore(nb::module_& m);
void bindTessellation(nb::module_& m);
void bindFlow(nb::module_& m);
void bindSimulation(nb::module_& m);
#ifdef PECLET_VORO_MPI
void bindMpi(nb::module_& m);
#endif

}  // namespace peclet::voro::pybind

#endif  // PECLET_VORO_BINDINGS_COMMON_HPP
