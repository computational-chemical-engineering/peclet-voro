/**
 * @file voro_bindings.cpp
 * @brief Kokkos nanobind Python module `peclet.voro`.
 *
 * Drives the production device path — multicore CPU (OpenMP), or GPU (CUDA/HIP), selected by the
 * Kokkos backend the extension was built against — from Python. The PUBLIC surface (what a user of
 * the method needs to set up, run and read out a computation; suite/docs/QUALITY_PLAN.md D2):
 *
 *  - @ref Tess "peclet.voro.Tessellation" — the bare moving-particle (power-)Voronoi tessellator,
 *    optionally SDF-clipped: a cold build plus the incremental two-pass *repair* update (the fast
 *    per-step path for moving points); volumes, neighbour / wall counts, and the energy layer.
 *  - @ref Flow "peclet.voro.FlowSolver" — the static collocated / covolume Navier–Stokes solvers on
 *    the face mesh of a resident tessellation.
 *  - @ref Sim "peclet.voro.Simulation" — the device-native compressible-Euler / Navier–Stokes
 *    moving-cell fluid (velocity-Verlet over the tessellation).
 *  - `optimize_volume_mesh`, `minimize_interface` — the mesh optimisers on a periodic box.
 *  - `peclet.voro.pore_mesh` (a lazily imported submodule) carries the SDF-walled pore-space
 *    family bound here (`optimize_pore_mesh`, `sdf_voronoi_cells`, `sdf_voronoi_section`).
 *  - under `PECLET_VORO_MPI`: `VoronoiHalo` (ghost gather) and `DistributedTessellation` (the
 *    distributed repair driver, `mpi/distributed_moving.hpp`).
 *
 * Every instrument, ablation switch and validity report lives on the object's `diagnostics`
 * sub-object (`t.diagnostics.build_report()`), one per class, holding a reference to its owner.
 *
 * String modes are validated against the accepted set and the error lists it; call-order
 * requirements are state checks that name the correct order; the literals the engine is driven
 * with are the named `defaults` below (also `peclet.voro.defaults`), and the docstrings are
 * generated from them so they cannot drift.
 *
 * Particle data crosses the boundary as NumPy arrays: positions/velocities are `(N,3)` float64,
 * scalars (masses, viscosities, volumes) are `(N,)`. Arrays move through the shared
 * `peclet::core::python` bridge (core): returned arrays are backed by host buffers (no extra device
 * copy).
 *
 * Kokkos teardown follows the suite-wide pattern of peclet/core/python/kokkos_teardown.hpp: Kokkos
 * is initialized at import; every bound Tessellation / FlowSolver / Simulation is a `Releasable`
 * (registered on construction, release() drops its Views) and every zero-copy capsule of the shared
 * bridge is one too; the module's single `atexit` hook (also `peclet.voro.finalize()`) releases all
 * of them and THEN calls Kokkos::finalize, so an object still referenced at interpreter exit
 * (script globals, a Jupyter/Quarto kernel) can no longer be destroyed after finalize -- which is a
 * Kokkos::abort (SIGABRT / exit 134, on OpenMP as on CUDA). Nothing needs `del` before exit.
 *
 * Example
 * -------
 * @code{.py}
 *   import numpy as np, peclet.voro
 *   rng = np.random.default_rng(0)
 *   pos = rng.random((100_000, 3))            # uniform points in the unit box
 *   t = peclet.voro.Tessellation()
 *   t.set_domain(extent=(1.0, 1.0, 1.0))
 *   t.build(pos)                              # cold tessellation
 *   vol = t.get_volumes()                     # (N,) cell volumes; sum ~= box volume
 *   for _ in range(50):                       # move + repair each step (faster than rebuilding)
 *       pos = (pos + 1e-4 * rng.standard_normal(pos.shape)) % 1.0
 *       stats = t.step(pos)                   # {'flagged','pass1','pass2','rebuilt','fell_back'}
 *   nbr = t.get_neighbor_counts()             # (N,) Voronoi neighbours per cell
 * @endcode
 */
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
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/energy/interface.hpp"
#include "peclet/voro/energy/lloyd.hpp"
#include "peclet/voro/energy/tension.hpp"
#include "peclet/voro/energy/volume.hpp"
#include "peclet/voro/energy/wall.hpp"
#include "peclet/voro/fv/collocated.hpp"
#include "peclet/voro/fv/covolume.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/mesh_optimizer.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/physics/simulation.hpp"
#include "peclet/voro/pore_cells.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/topology_store.hpp"

#ifdef PECLET_VORO_MPI
#include <mpi.h>
#include <nanobind/stl/tuple.h>

#include "peclet/voro/mpi/distributed_moving.hpp"
#include "peclet/voro/mpi/voronoi_halo.hpp"
#endif

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
const char* doc(std::string s) {
  static std::deque<std::string> keep;
  keep.push_back(std::move(s));
  return keep.back().c_str();
}
std::string fmt(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}
std::string fmt(int v) {
  return std::to_string(v);
}

// (N,3) c-contiguous array -> flat row-major host vector of length 3N.
std::vector<real_t> flatten3(nb::ndarray<real_t, nb::c_contig> a) {
  if (a.ndim() != 2 || a.shape(1) != 3)
    throw std::runtime_error("expected an (N,3) array");
  const real_t* p = a.data();
  return std::vector<real_t>(p, p + static_cast<std::size_t>(a.shape(0)) * 3);
}

// (N,) array -> host vector of length N.
std::vector<real_t> flatten1(nb::ndarray<real_t, nb::c_contig> a) {
  return peclet::core::python::ndarray_to_vector<real_t>(nb::ndarray<>(a));
}

// The union-of-spheres wall SDF is cubic-periodic (one L), so the pore-space functions take the
// suite-wide `extent` triple and check it is a cube.
real_t cubicExtent(std::array<real_t, 3> e, const char* fn) {
  if (!(e[0] > real_t(0)) || e[0] != e[1] || e[1] != e[2])
    throw std::invalid_argument(std::string("voro: ") + fn +
                                "(extent=...) must be a cubic box (Lx == Ly == Lz > 0) — the "
                                "union-of-spheres wall SDF is cubic-periodic.");
  return e[0];
}

// The CG preconditioner / descent method of the mesh optimisers, as a string mode.
constexpr const char* kMethodList = "'jacobi', 'colored_gs', 'graphamg', 'steepest'";
peclet::voro::Precond parseMethod(const std::string& m, const char* fn) {
  if (m == "jacobi")
    return peclet::voro::Precond::Jacobi;
  if (m == "colored_gs")
    return peclet::voro::Precond::ColoredGS;
  if (m == "graphamg")
    return peclet::voro::Precond::GraphAMG;
  if (m == "steepest")
    return peclet::voro::Precond::SteepestDescent;
  throw std::invalid_argument(std::string("voro: ") + fn + "(method='" + m +
                              "') is not a method; accepted: " + kMethodList + ".");
}

// ---- pore-space meshing helpers (SDF-walled interstitial Voronoi + geometry export) ----------
using PoreCell =
    peclet::voro::ConvexCell<real_t, defaults::kPoreMaxPlanes, defaults::kPoreMaxTriangles>;

// Build a periodic union-of-balls SDF from (M,3) centers + (M,) radii; the Views must outlive its
// use.
peclet::voro::SdfSpheres<real_t> makeSpheresSdf(nb::ndarray<real_t, nb::c_contig> centers,
                                                nb::ndarray<real_t, nb::c_contig> radii, real_t L,
                                                DView& cenHold, DView& radHold) {
  const int M = (int)radii.shape(0);
  auto cflat = flatten3(centers);
  if ((int)(cflat.size() / 3) != M)
    throw std::runtime_error("sphere_centers (M,3) and sphere_radii (M,) must agree on M");
  cenHold = DView("sph.cen", 3 * M);
  radHold = DView("sph.rad", M);
  Kokkos::deep_copy(cenHold, Kokkos::View<const real_t*, Kokkos::HostSpace>(cflat.data(), 3 * M));
  Kokkos::deep_copy(radHold, Kokkos::View<const real_t*, Kokkos::HostSpace>(radii.data(), M));
  return peclet::voro::SdfSpheres<real_t>{cenHold, radHold, M, L};
}
// The same union-of-balls SDF with HOST views, for the host-serial pore-cell oracle (its clip runs
// on the host, so on a CUDA build the device-view variant would abort on the first eval).
using HView = Kokkos::View<real_t*, Kokkos::HostSpace>;
using HostSpheresSdf = peclet::voro::SdfSpheres<real_t, Kokkos::HostSpace>;
HostSpheresSdf makeSpheresSdfHost(nb::ndarray<real_t, nb::c_contig> centers,
                                  nb::ndarray<real_t, nb::c_contig> radii, real_t L, HView& cenHold,
                                  HView& radHold) {
  const int M = (int)radii.shape(0);
  auto cflat = flatten3(centers);
  if ((int)(cflat.size() / 3) != M)
    throw std::runtime_error("sphere_centers (M,3) and sphere_radii (M,) must agree on M");
  cenHold = HView("sph.cen.host", 3 * M);
  radHold = HView("sph.rad.host", M);
  std::copy(cflat.begin(), cflat.end(), cenHold.data());
  std::copy(radii.data(), radii.data() + M, radHold.data());
  return HostSpheresSdf{cenHold, radHold, M, L};
}

// The host-serial reconstruction of the SDF-clipped interstitial Voronoi cell of any seed — the
// TEST ORACLE of the device path (pore_cells.hpp; python/test_voro.py test_pore_cells). Builds a
// periodic counting-sort grid once; each build() walks Chebyshev shells of bins outward, after
// each shell rebuilding the ConvexCell (far box, min-image neighbours closest-first) from
// everything gathered so far, until the shell radius certifies the cell: every seed within R·hbin
// has been gathered and a seed cuts only if it is closer than twice the cell's reach, so
// (R·hbin)² ≥ 4·rSqMax closes it (the same bisector certificate the tessellator's worklist
// uses). If the walk reaches the min-image half box uncertified, every seed is gathered. Then the
// SDF clip. (Its predecessor gathered a fixed 80 nearest seeds — the device port's gate showed
// that truncation missing planes on 8 of 805 cells, up to 2.5e-3 in volume.)
struct PoreReconstructor {
  const real_t* seed;
  int N;
  real_t L, Lh, big, hbin;
  int nb;
  HostSpheresSdf sdf;
  std::vector<int> binStart, binItem;
  mutable std::vector<std::pair<real_t, int>> ord;
  mutable std::vector<real_t> rx, ry, rz;
  mutable std::vector<int> ids;

  int binOf(real_t x) const {
    int b = (int)std::floor(x / hbin) % nb;
    return b < 0 ? b + nb : b;
  }
  int cellOf(int i) const {
    return binOf(seed[3 * i]) + nb * (binOf(seed[3 * i + 1]) + nb * binOf(seed[3 * i + 2]));
  }
  PoreReconstructor(const std::vector<real_t>& s, real_t L_, HostSpheresSdf sdf_)
      : seed(s.data()), N((int)(s.size() / 3)), L(L_), Lh(0.5 * L_), big(4 * L_), sdf(sdf_) {
    nb = std::max(1, std::min((int)std::cbrt((double)N / 2.0 + 1.0), defaults::kPoreMaxBins));
    hbin = L / nb;
    const int nbin = nb * nb * nb;
    binStart.assign(nbin + 1, 0);
    for (int i = 0; i < N; ++i)
      ++binStart[cellOf(i) + 1];
    for (int b = 0; b < nbin; ++b)
      binStart[b + 1] += binStart[b];
    binItem.resize(N);
    std::vector<int> cur(binStart.begin(), binStart.end());
    for (int i = 0; i < N; ++i)
      binItem[cur[cellOf(i)]++] = i;
  }
  // Gather bin (gx,gy,gz) (raw, wrapped here) into `ord` as (dist², j), min-image, j != i.
  void gatherBin(int i, real_t sx, real_t sy, real_t sz, int gx, int gy, int gz) const {
    gx = ((gx % nb) + nb) % nb;
    gy = ((gy % nb) + nb) % nb;
    gz = ((gz % nb) + nb) % nb;
    const int b = gx + nb * (gy + nb * gz);
    for (int t = binStart[b]; t < binStart[b + 1]; ++t) {
      const int j = binItem[t];
      if (j == i)
        continue;
      real_t dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
      dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
      dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
      dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
      ord.emplace_back(dx * dx + dy * dy + dz * dz, j);
    }
  }
  // Rebuild `c` from everything in `ord`, closest-first against the far box.
  void rebuild(real_t sx, real_t sy, real_t sz, PoreCell& c) const {
    std::sort(ord.begin(), ord.end());
    const int M = (int)ord.size();
    rx.resize(M);
    ry.resize(M);
    rz.resize(M);
    ids.resize(M);
    for (int k = 0; k < M; ++k) {
      const int j = ord[k].second;
      real_t dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
      dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
      dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
      dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
      rx[k] = dx;
      ry[k] = dy;
      rz[k] = dz;
      ids[k] = j;
    }
    const real_t Lbig[3] = {big, big, big};
    peclet::voro::buildConvexCell(c, Lbig, rx.data(), ry.data(), rz.data(), ids.data(), M);
  }
  bool build(int i, PoreCell& c) const {
    const real_t sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
    ord.clear();
    const int bx = binOf(sx), by = binOf(sy), bz = binOf(sz);
    const int Rmax = (nb - 1) / 2;  // beyond it a bin would be visited twice (periodic wrap)
    for (int R = 0;; ++R) {
      if (R > Rmax) {  // uncertified at the min-image half box: take every seed
        ord.clear();
        for (int gz = 0; gz < nb; ++gz)
          for (int gy = 0; gy < nb; ++gy)
            for (int gx = 0; gx < nb; ++gx)
              gatherBin(i, sx, sy, sz, gx, gy, gz);
        rebuild(sx, sy, sz, c);
        break;
      }
      for (int dz2 = -R; dz2 <= R; ++dz2)
        for (int dy2 = -R; dy2 <= R; ++dy2)
          for (int dx2 = -R; dx2 <= R; ++dx2) {
            int cheb = std::abs(dx2);
            cheb = std::max(cheb, std::abs(dy2));
            cheb = std::max(cheb, std::abs(dz2));
            if (cheb != R)
              continue;
            gatherBin(i, sx, sy, sz, bx + dx2, by + dy2, bz + dz2);
          }
      rebuild(sx, sy, sz, c);
      if (c.overflow)
        return false;
      const real_t rh = (real_t)R * hbin;  // every seed within rh is in `ord`
      if (rh * rh >= real_t(4) * c.maxVertexRsq())
        break;
    }
    const real_t seedW[3] = {sx, sy, sz};
    peclet::voro::clipCellAgainstSdf<real_t, defaults::kPoreMaxPlanes, defaults::kPoreMaxTriangles,
                                     false>(c, seedW, sdf);
    return !(c.empty() || c.overflow);
  }
};

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

SceneHolder makeSceneHolder(nb::ndarray<int, nb::c_contig> node_ints,
                            nb::ndarray<real_t, nb::c_contig> node_reals, int root, real_t grad_h) {
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
void checkDomain(const char* cls, std::array<real_t, 3> extent, std::array<real_t, 3> origin,
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
bool parseWallMode(const std::string& mode) {
  if (mode == "exact")
    return true;
  if (mode == "skin")
    return false;
  throw std::invalid_argument("voro: set_wall_mode(mode='" + mode +
                              "') is not a wall mode; accepted: " + kWallModeList + ".");
}

nb::ndarray<nb::numpy, real_t> toNumpy3(std::vector<real_t> v) {
  const std::size_t N = v.size() / 3;
  return peclet::core::python::vector_to_ndarray(std::move(v), {N, std::size_t(3)}, {3, 1});
}
nb::ndarray<nb::numpy, real_t> toNumpy1(std::vector<real_t> v) {
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
OptimizeResult makeOptimizeResult(std::vector<real_t> pos, std::optional<std::vector<real_t>> w,
                                  const peclet::voro::OtResult& R) {
  OptimizeResult r;
  r.positions = nb::cast(toNumpy3(std::move(pos)));
  r.weights = w ? nb::cast(toNumpy1(std::move(*w))) : nb::none();
  r.iters = R.iters;
  r.max_vol_err = R.maxVolErr;
  r.mean_vol_err = R.meanVolErr;
  r.converged = R.converged;
  r.num_empty = (int)R.nEmpty;
  return r;
}

struct InterfaceResult {
  nb::object positions;  ///< (N,3) float64
  double energy = 0, energy_ratio = 1;
  int iters = 0;
  bool converged = false;
};

// --------------------------------------------------------------------------------------------------
// Tessellation: the bare moving-point (power-)Voronoi tessellator, optionally SDF-clipped.
// The engine is chosen at build() from what was set: {Voronoi, Power} x {no geometry, SdfScene}.
// --------------------------------------------------------------------------------------------------
template <bool W, class S>
using MT =
    peclet::voro::MovingTessellation<real_t, defaults::kMaxPlanes, defaults::kMaxTriangles, W, S>;
using MtVariant =
    std::variant<MT<false, NoSdfT>, MT<true, NoSdfT>, MT<false, SceneT>, MT<true, SceneT>>;

nb::dict repairStatsDict(const peclet::voro::RepairStats& st) {
  nb::dict d;
  d["flagged"] = st.pass1Raw;  // cells the certificate flagged
  d["pass1"] = st.pass1;       // cells gathered in Pass 1
  d["pass2"] = st.pass2;       // cells gathered in Pass 2
  d["rebuilt"] =
      (st.route == peclet::voro::RepairStats::kRebuildGate);  // gate routed to a full rebuild
  d["fell_back"] = st.fellBack;                               // verify failed -> cold rebuild
  d["extra"] = st.extra;                 // cells gathered across the verify extra-passes
  d["surgical"] = st.surgical;           // Pass-1 cells repaired surgically (no grid gather)
  d["verify_passes"] = st.verifyPasses;  // number of verify iterations run
  d["wall_flagged"] = st.wallFlagged;    // cells flagged by the SDF boundary watch
  return d;
}

class Tess : public peclet::core::python::Releasable {
 public:
  Tess() = default;

  void set_domain(std::array<real_t, 3> extent, std::array<real_t, 3> origin,
                  std::array<bool, 3> periodic) {
    checkDomain("Tessellation", extent, origin, periodic);
    L_ = extent;
  }
  std::array<real_t, 3> extent() const { return L_; }
  void set_tolerance(real_t frac) {
    if (!(frac > real_t(0)))
      throw std::invalid_argument("voro: set_tolerance(frac) needs frac > 0.");
    tolFrac_ = frac;
  }
  void set_geometry(nb::ndarray<int, nb::c_contig> node_ints,
                    nb::ndarray<real_t, nb::c_contig> node_reals, int root, real_t grad_h) {
    scene_ = makeSceneHolder(node_ints, node_reals, root, grad_h);
  }
  void clear_geometry() { scene_.clear(); }
  void set_wall_mode(const std::string& mode, real_t skin_frac) {
    wallExact_ = parseWallMode(mode);
    if (skin_frac < real_t(0))
      throw std::invalid_argument("voro: set_wall_mode(skin_frac=...) needs skin_frac >= 0.");
    wallSkinFrac_ = skin_frac;
  }
  void set_weights(nb::ndarray<real_t, nb::c_contig> w) {
    if (w.ndim() != 1)
      throw std::runtime_error("set_weights(): expected an (N,) array");
    wHost_ = flatten1(w);
    weighted_ = true;
    wDirty_ = true;
  }
  void clear_weights() {
    wHost_.clear();
    weighted_ = false;
    wDirty_ = false;
  }

  // diagnostics: ablation switches of the repair (both certificates are complete; the gate is the
  // "never slower than a cold build" guard) and the validity counts of the last cold build.
  void set_local_certificate(bool on) { localCert_ = on; }
  void set_gate(bool on) { useGate_ = on; }
  void set_profile(bool on) {
    profile_ = on;
    std::visit([&](auto& mt) { mt.profile = on; }, mt_);
  }
  nb::dict build_report() {
    auto r = std::visit([](auto& mt) { return mt.report(); }, mt_);
    nb::dict d;
    d["buried"] = r.buried;
    d["reach_exceeded"] = r.reachExceeded;
    d["empty"] = r.empty;
    d["overflow"] = r.overflow;
    d["incomplete"] = r.incomplete;
    d["over_buffer_rebuilds"] = r.overBufferRebuilds;
    return d;
  }

  // Cold build: (re)allocate the resident tessellation for N points and build it from scratch.
  void build(nb::ndarray<real_t, nb::c_contig> a, bool strict) {
    std::vector<real_t> p = flatten3(a);
    N_ = static_cast<int>(p.size() / 3);
    if (N_ <= 0)
      throw std::runtime_error("voro: build() needs at least one point");
    const double boxVol = static_cast<double>(L_[0]) * L_[1] * L_[2];
    const real_t spacing = static_cast<real_t>(std::cbrt(boxVol / N_));
    if (weighted_ && static_cast<int>(wHost_.size()) != N_)
      throw std::runtime_error("build(): weights (N,) must match the particle count");
    pos_ = peclet::core::toDevice<real_t>(p, "pos");
    if (weighted_) {
      weight_ = peclet::core::toDevice<real_t>(wHost_, "w");
      wDirty_ = false;
    }
    if (!weighted_ && !scene_.set)
      mt_.template emplace<MT<false, NoSdfT>>();
    else if (weighted_ && !scene_.set)
      mt_.template emplace<MT<true, NoSdfT>>();
    else if (!weighted_)
      mt_.template emplace<MT<false, SceneT>>();
    else
      mt_.template emplace<MT<true, SceneT>>();
    std::visit(
        [&](auto& mt) {
          using T = std::decay_t<decltype(mt)>;
          mt.localCert = localCert_;
          mt.useGate = useGate_;
          mt.profile = profile_;
          if constexpr (T::kHasSdf) {
            mt.sdf = scene_.scene;
            mt.wallExact = wallExact_;
            mt.wallSkin = wallSkinFrac_ * spacing;
          }
          if constexpr (std::is_same_v<typename T::PlanePolicy, peclet::voro::Power>)
            mt.setWeights(weight_);
          mt.alloc(N_, L_.data(), tolFrac_ * spacing, real_t(defaults::kSkin) * spacing,
                   defaults::kSearchWindow, N_);
          mt.rebuild(pos_);
        },
        mt_);
    auto r = std::visit([](auto& mt) { return mt.report(); }, mt_);
    if (r.buried > 0 || r.reachExceeded > 0 || r.overflow > 0) {
      const std::string msg =
          "peclet.voro: the tessellation is not a guaranteed-exact partition: " +
          std::to_string(r.buried) +
          " buried power cell(s) (seed outside its own cell, emptied), " +
          std::to_string(r.reachExceeded) + " cell(s) with a search reach beyond half the box " +
          "(min-image invalid), " + std::to_string(r.overflow) +
          " overflowed cell(s). See diagnostics.build_report().";
      if (strict)
        throw std::runtime_error(msg);
      nb::module_::import_("warnings").attr("warn")(msg);
    }
  }

  // Incremental repair: update the resident tessellation to new positions (same N) without a full
  // rebuild. Returns the per-step work stats.
  nb::dict step(nb::ndarray<real_t, nb::c_contig> a) {
    if (N_ == 0)
      throw std::runtime_error(
          "voro: Tessellation.step() before build() — call build(positions) first.");
    std::vector<real_t> p = flatten3(a);
    if (static_cast<int>(p.size() / 3) != N_)
      throw std::runtime_error(
          "voro: step(positions) got " + std::to_string(p.size() / 3) + " points but build() had " +
          std::to_string(N_) +
          " — the count is fixed by build(); call build() again to change it.");
    pos_ = peclet::core::toDevice<real_t>(p, "pos");
    if (weighted_ && wDirty_) {  // weights changed since the last build/step: refresh in place
      if (static_cast<int>(wHost_.size()) != N_)
        throw std::runtime_error("step(): weights (N,) must match the particle count");
      Kokkos::deep_copy(
          weight_, Kokkos::View<const real_t*, Kokkos::HostSpace>(wHost_.data(), wHost_.size()));
      wDirty_ = false;
    }
    auto st = std::visit([&](auto& mt) { return mt.step(pos_); }, mt_);
    return repairStatsDict(st);
  }

  nb::ndarray<nb::numpy, real_t> get_volumes() {
    const std::size_t N = static_cast<std::size_t>(N_);
    DView vol = std::visit([](auto& mt) { return mt.vol; }, mt_);
    auto v = Kokkos::subview(vol, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(v), {N}, {1});
  }

  // Per-cell Voronoi neighbour (= face) count, recomputed from the resident topology store.
  nb::ndarray<nb::numpy, int> get_neighbor_counts() {
    using Cell =
        peclet::voro::ConvexCell<real_t, defaults::kMaxPlanes, defaults::kMaxTriangles, false>;
    const int N = N_;
    Kokkos::View<int*, peclet::core::MemSpace> cnt("nbr", N);
    auto st = std::visit([](auto& mt) { return mt.store; }, mt_);
    auto C = cnt;
    const real_t Lx = L_[0], Ly = L_[1], Lz = L_[2];
    Kokkos::parallel_for(
        "peclet.voro.nbrcount", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i) {
          Cell c;
          st.load(i, c, Lx, Ly, Lz);
          C(i) = c.countFaces();
        });
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(cnt),
                                                   {static_cast<std::size_t>(N)}, {1});
  }

  // Rung A3: energies + forces of the RESIDENT tessellation (after build/step) on the published
  // view — interfacial Σσ(t_i,t_j)A, wetting Σσ_s(t_i)A_wall, volume Σe_i(V_i) via a
  // caller-supplied e'(V_i) — routed to the seed positions (and power weights). No rebuild:
  // reevalPublish over the store with the facet-edge area-Jacobian CSR.
  nb::dict energy_forces(nb::ndarray<int, nb::c_contig> types,
                         nb::ndarray<real_t, nb::c_contig> tension,
                         std::optional<nb::ndarray<real_t, nb::c_contig>> sigma_wall,
                         std::optional<nb::ndarray<real_t, nb::c_contig>> dEdV, real_t lloyd,
                         real_t facet_tension) {
    if (N_ == 0)
      throw std::runtime_error(
          "voro: Tessellation.energy_forces() before build() — call build(positions) first.");
    const int N = N_;
    if ((int)types.shape(0) != N)
      throw std::runtime_error("energy_forces(): types must be (N,)");
    int nT = 0;
    for (int i = 0; i < N; ++i)
      nT = std::max(nT, types.data()[i] + 1);
    if ((int)tension.size() != nT * nT)
      throw std::runtime_error(
          "energy_forces(): tension must be (nTypes, nTypes) with nTypes = max(types)+1");
    Kokkos::View<int*, Mem> dtype("en.type", N);
    Kokkos::deep_copy(dtype, Kokkos::View<const int*, Kokkos::HostSpace>(types.data(), N));
    DView dten("en.tension", (size_t)nT * nT);
    Kokkos::deep_copy(
        dten, Kokkos::View<const real_t*, Kokkos::HostSpace>(tension.data(), (size_t)nT * nT));
    DView dsw, dde;
    if (sigma_wall) {
      if ((int)sigma_wall->size() != nT)
        throw std::runtime_error("energy_forces(): sigma_wall must be (nTypes,)");
      dsw = DView("en.sw", nT);
      Kokkos::deep_copy(dsw,
                        Kokkos::View<const real_t*, Kokkos::HostSpace>(sigma_wall->data(), nT));
    }
    if (dEdV) {
      if ((int)dEdV->size() != N)
        throw std::runtime_error("energy_forces(): dEdV must be (N,)");
      dde = DView("en.dEdV", N);
      Kokkos::deep_copy(dde, Kokkos::View<const real_t*, Kokkos::HostSpace>(dEdV->data(), N));
    }
    DView force("en.force", 3 * (size_t)N), forceW("en.forceW", weighted_ ? N : 0);
    Kokkos::deep_copy(force, real_t(0));
    if (weighted_)
      Kokkos::deep_copy(forceW, real_t(0));
    real_t eIf = 0, eWall = 0, eLloyd = 0, eTen = 0;
    const real_t Larr[3] = {L_[0], L_[1], L_[2]};
    std::visit(
        [&](auto& mt) {
          using T = std::decay_t<decltype(mt)>;
          using Policy = typename T::PlanePolicy;
          auto view =
              peclet::voro::reevalPublish<real_t, defaults::kMaxPlanes, defaults::kMaxTriangles>(
                  mt.store, pos_, mt.vol, N, Larr, mt.wall, mt.xRef, /*withAreaGrad=*/true,
                  /*withMoments=*/lloyd != 0);
          if (lloyd != 0)
            eLloyd = peclet::voro::energy::lloydEnergyForce<real_t>(view, pos_, lloyd, force);
          if (facet_tension != 0)
            eTen = peclet::voro::energy::facetTensionEnergyForce<real_t, Policy>(
                view, facet_tension, pos_, weight_, L_[0], force, forceW, mt.sdf);
          eIf = peclet::voro::energy::interfaceEnergyForce<real_t, Policy>(
              view, dtype, dten, nT, pos_, weight_, L_[0], force, forceW, mt.sdf);
          if (dsw.extent(0) > 0)
            eWall = peclet::voro::energy::wallEnergyForce<real_t, Policy>(
                view, dtype, dsw, mt.sdf, pos_, weight_, L_[0], force, forceW);
          if (dde.extent(0) > 0)
            peclet::voro::energy::volumeGradientForce<real_t, Policy>(view, dde, pos_, weight_,
                                                                      L_[0], force, forceW, mt.sdf);
        },
        mt_);
    nb::dict d;
    d["interface_energy"] = eIf;
    d["wall_energy"] = eWall;
    d["lloyd_energy"] = eLloyd;
    d["tension_energy"] = eTen;
    d["force"] = peclet::core::python::vector_to_ndarray(
        peclet::core::toVector(force), {static_cast<std::size_t>(N), std::size_t(3)}, {3, 1});
    if (weighted_)
      d["force_w"] = peclet::core::python::vector_to_ndarray(peclet::core::toVector(forceW),
                                                             {static_cast<std::size_t>(N)}, {1});
    return d;
  }

  // The face mesh of the resident tessellation (track C): reevalPublish over the store, then the
  // reciprocal map and the owner/neighbour face records (fv/mesh.hpp). Used by FlowSolver.
  peclet::voro::fv::FaceMesh<real_t> face_mesh() {
    if (N_ == 0)
      throw std::runtime_error(
          "voro: FlowSolver needs a built Tessellation — call tessellation.build(positions) "
          "first.");
    const int N = N_;
    const real_t Larr[3] = {L_[0], L_[1], L_[2]};
    return std::visit(
        [&](auto& mt) {
          auto view =
              peclet::voro::reevalPublish<real_t, defaults::kMaxPlanes, defaults::kMaxTriangles>(
                  mt.store, pos_, mt.vol, N, Larr, mt.wall, mt.xRef);
          auto aux = peclet::voro::buildAuxMaps(view);
          return peclet::voro::fv::buildFaceMesh(view, aux);
        },
        mt_);
  }

  // Per-cell number of resident SDF wall planes (0 everywhere without geometry).
  nb::ndarray<nb::numpy, int> get_wall_counts() {
    const std::size_t N = static_cast<std::size_t>(N_);
    std::vector<int> v = std::visit(
        [&](auto& mt) {
          using T = std::decay_t<decltype(mt)>;
          if constexpr (T::kHasSdf)
            return peclet::core::toVector(mt.wall.cnt);
          else
            return std::vector<int>(N, 0);
        },
        mt_);
    return peclet::core::python::vector_to_ndarray(std::move(v), {N}, {1});
  }

  int num_particles() const { return N_; }

  // Drop every Kokkos View (teardown registry: runs before Kokkos::finalize at shutdown).
  void release() noexcept override {
    mt_.template emplace<MT<false, NoSdfT>>();
    pos_ = DView{};
    weight_ = DView{};
    scene_.clear();
    N_ = 0;
  }

 private:
  std::array<real_t, 3> L_{1, 1, 1};
  real_t tolFrac_ = defaults::kCertificateTolerance;
  bool localCert_ = true, useGate_ = true, profile_ = false;
  bool wallExact_ = true;
  real_t wallSkinFrac_ = defaults::kWallSkin;
  bool weighted_ = false, wDirty_ = false;
  std::vector<real_t> wHost_;
  int N_ = 0;
  DView pos_, weight_;
  SceneHolder scene_;
  MtVariant mt_;
};

// The diagnostics tier of Tessellation: a view onto the owner (no state of its own).
struct TessDiagnostics {
  Tess* t;
};

// --------------------------------------------------------------------------------------------------
// FlowSolver (track C, rungs C2/C3/C5): the static Navier–Stokes solvers on the face mesh of a
// resident Tessellation — the collocated solver (peclet.flow's approximate projection with the
// skew-corrected adjoint constraint pair, the default) or the staggered covolume solver. The mesh
// is frozen at construction (rebuild the FlowSolver after moving seeds).
// --------------------------------------------------------------------------------------------------
constexpr const char* kLayoutList = "'collocated', 'covolume'";

class Flow : public peclet::core::python::Releasable {
 public:
  Flow(Tess& t, real_t nu, const std::string& layout) : layout_(layout) {
    if (layout != "collocated" && layout != "covolume")
      throw std::invalid_argument("voro: FlowSolver(layout='" + layout +
                                  "') is not a layout; accepted: " + kLayoutList + ".");
    if (!(nu >= real_t(0)))
      throw std::invalid_argument("voro: FlowSolver(viscosity=...) needs viscosity >= 0.");
    m_ = t.face_mesh();
    if (layout == "collocated") {
      co_ = std::make_unique<peclet::voro::fv::CollocatedNS<real_t>>();
      co_->setup(m_, nu, /*amg=*/true);
    } else {
      cv_ = std::make_unique<peclet::voro::fv::CovolumeNS<real_t>>();
      cv_->setup(m_, nu, /*amg=*/true);
    }
  }
  // Drop the solvers and the face mesh (Kokkos Views) BEFORE Kokkos::finalize at shutdown.
  void release() noexcept override {
    co_.reset();
    cv_.reset();
    m_ = peclet::voro::fv::FaceMesh<real_t>{};
  }
  int num_cells() const { return m_.nCells; }
  int num_faces() const { return m_.nFaces; }
  int num_wall_faces() const { return m_.nFaces - m_.nInterior; }
  void set_body_force(std::array<real_t, 3> g) {
    const int N = m_.nCells;
    DView f("flow.force", 3 * (size_t)N);
    auto h = Kokkos::create_mirror_view(f);
    for (int i = 0; i < N; ++i) {
      h(3 * i) = g[0];
      h(3 * i + 1) = g[1];
      h(3 * i + 2) = g[2];
    }
    Kokkos::deep_copy(f, h);
    if (co_)
      co_->force = f;
    else
      cv_->force = f;
  }
  void set_stokes(bool on) {
    if (co_)
      co_->convScale = on ? 0 : 1;
    else
      cv_->convScale = on ? 0 : 1;
  }
  void set_pressure_tolerance(real_t tol) {
    if (!(tol > real_t(0)))
      throw std::invalid_argument("voro: set_pressure_tolerance(tol) needs tol > 0.");
    (co_ ? co_->poisson : cv_->poisson).tol = tol;
  }
  void set_implicit_diffusion(bool on) {
    if (co_)
      co_->implicitDiffusion = on;
    else
      throw std::runtime_error(
          "voro: set_implicit_diffusion() is a collocated-layout step; this FlowSolver has "
          "layout='covolume'.");
  }
  // diagnostics: the measured-worse alternatives (README: the plain constraint pair, the two-point
  // wall flux) kept for ablation.
  void set_skew_corrected(bool on) {
    if (co_)
      co_->skewCorrected = on;
    else
      throw std::runtime_error(
          "voro: set_skew_corrected() is a collocated-layout switch; this FlowSolver has "
          "layout='covolume'.");
  }
  void set_wall_gradient_quadratic(bool on) {
    if (co_)
      co_->wallQuadratic = on;
    else
      cv_->wallQuadratic = on;
  }
  void set_wall_velocity(nb::ndarray<real_t, nb::c_contig> Uw) {
    const int nB = m_.nFaces - m_.nInterior;
    if (Uw.ndim() != 2 || (int)Uw.shape(0) != nB || Uw.shape(1) != 3)
      throw std::runtime_error("set_wall_velocity(): expected (num_wall_faces, 3) = (" +
                               std::to_string(nB) + ", 3)");
    DView d("flow.Uwall", 3 * (size_t)nB);
    Kokkos::deep_copy(d, Kokkos::View<const real_t*, Kokkos::HostSpace>(Uw.data(), 3 * (size_t)nB));
    if (co_)
      co_->setWallVelocity(d);
    else
      cv_->setWallVelocity(d);
  }
  // Set the cell velocity (N,3); the collocated solver projects it once, the covolume solver
  // takes the face-normal components of the distance-weighted face average.
  void set_velocity(nb::ndarray<real_t, nb::c_contig> U) {
    const int N = m_.nCells;
    if (U.ndim() != 2 || (int)U.shape(0) != N || U.shape(1) != 3)
      throw std::runtime_error("set_velocity(): expected (num_cells, 3) = (" + std::to_string(N) +
                               ", 3)");
    DView d("flow.U0", 3 * (size_t)N);
    Kokkos::deep_copy(d, Kokkos::View<const real_t*, Kokkos::HostSpace>(U.data(), 3 * (size_t)N));
    if (co_) {
      co_->initialize(d);
    } else {
      peclet::voro::fv::projectToFaces(m_, d, cv_->u, DView{}, cv_->ub);
      cv_->project(cv_->u, real_t(1));
    }
  }
  // docs/NAMING.md 1.5: the time step is configured with set_dt and read back as dt; step(n)
  // advances n steps of it and raises if none was set.
  void set_dt(real_t dt) {
    if (!(dt > real_t(0)))
      throw std::invalid_argument("voro: set_dt(dt) needs dt > 0.");
    dt_ = dt;
  }
  real_t dt() const { return dt_; }
  void step(int n) {
    if (!(dt_ > real_t(0)))
      throw std::invalid_argument("voro: step() has no time step — call set_dt(dt) first.");
    if (n < 0)
      throw std::invalid_argument("voro: step(num_steps) needs num_steps >= 0.");
    for (int i = 0; i < n; ++i) {
      if (co_)
        co_->step(dt_);
      else
        cv_->step(dt_);
    }
  }
  nb::ndarray<nb::numpy, real_t> get_velocities() {
    const std::size_t N = m_.nCells;
    std::vector<real_t> v;
    if (co_) {
      v = peclet::core::toVector(co_->U);
    } else {
      DView Uc("flow.Uc", 3 * N);
      peclet::voro::fv::perotVelocity(m_, cv_->u, Uc);
      v = peclet::core::toVector(Uc);
    }
    return peclet::core::python::vector_to_ndarray(std::move(v), {N, std::size_t(3)}, {3, 1});
  }
  nb::ndarray<nb::numpy, real_t> get_pressure() {
    const std::size_t N = m_.nCells;
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(co_ ? co_->p : cv_->p),
                                                   {N}, {1});
  }
  nb::ndarray<nb::numpy, real_t> get_volumes() {
    const std::size_t N = m_.nCells;
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(m_.cellVolume), {N}, {1});
  }
  real_t kinetic_energy() { return co_ ? co_->kineticEnergy() : cv_->kineticEnergy(); }
  real_t max_divergence() { return co_ ? co_->maxFaceDivergence() : cv_->maxDivergence(); }
  int pressure_iterations() const { return (co_ ? co_->poisson : cv_->poisson).lastIters; }
  std::string layout() const { return layout_; }

 private:
  std::string layout_;
  real_t dt_{0};
  peclet::voro::fv::FaceMesh<real_t> m_;
  std::unique_ptr<peclet::voro::fv::CollocatedNS<real_t>> co_;
  std::unique_ptr<peclet::voro::fv::CovolumeNS<real_t>> cv_;
};

struct FlowDiagnostics {
  Flow* f;
};

// --------------------------------------------------------------------------------------------------
// Simulation: device-native compressible-Euler / Navier-Stokes Voronoi fluid dynamics.
// --------------------------------------------------------------------------------------------------
class Sim : public peclet::core::python::Releasable {
 public:
  Sim() = default;

  // Drop all Kokkos Views (so they free BEFORE Kokkos::finalize at shutdown).
  void release() noexcept override {
    sim_.template emplace<EE<NoSdfT>>();
    dmass_ = DView{};
    scene_.clear();
    pos_.clear();
    vel_.clear();
    mass_.clear();
    visc_.clear();
    bulk_.clear();
    inited_ = false;
  }

  // The state setters configure init(); after init() the device state is resident and they would
  // silently do nothing, so they raise with the order instead.
  void set_domain(std::array<real_t, 3> extent, std::array<real_t, 3> origin,
                  std::array<bool, 3> periodic) {
    beforeInit("set_domain");
    checkDomain("Simulation", extent, origin, periodic);
    L_ = extent;
  }
  std::array<real_t, 3> extent() const { return L_; }
  void set_positions(nb::ndarray<real_t, nb::c_contig> a) {
    beforeInit("set_positions");
    pos_ = flatten3(a);
  }
  void set_velocities(nb::ndarray<real_t, nb::c_contig> a) {
    beforeInit("set_velocities");
    vel_ = flatten3(a);
  }
  void set_masses(nb::ndarray<real_t, nb::c_contig> a) {
    beforeInit("set_masses");
    if (a.ndim() != 1)
      throw std::runtime_error("set_masses(): expected an (N,) array");
    mass_ = flatten1(a);
  }
  void set_pressure(real_t p) {
    beforeInit("set_pressure");
    pressEq_ = p;
  }
  void set_viscosities(nb::ndarray<real_t, nb::c_contig> a) {
    beforeInit("set_viscosities");
    if (a.ndim() != 1)
      throw std::runtime_error("set_viscosities(): expected an (N,) array");
    visc_ = flatten1(a);
  }
  void set_bulk_viscosities(nb::ndarray<real_t, nb::c_contig> a) {
    beforeInit("set_bulk_viscosities");
    if (a.ndim() != 1)
      throw std::runtime_error("set_bulk_viscosities(): expected an (N,) array");
    bulk_ = flatten1(a);
  }
  // diagnostics: the opt-in incremental-repair path (a performance path, off by default).
  void set_repair(bool on) {
    beforeInit("diagnostics.set_repair");
    repair_ = on;
  }
  void set_profile(bool on) {
    profile_ = on;
    std::visit([&](auto& s) { s.setProfile(on); }, sim_);
  }
  void set_geometry(nb::ndarray<int, nb::c_contig> node_ints,
                    nb::ndarray<real_t, nb::c_contig> node_reals, int root, real_t grad_h) {
    beforeInit("set_geometry");
    scene_ = makeSceneHolder(node_ints, node_reals, root, grad_h);
  }
  void clear_geometry() {
    beforeInit("clear_geometry");
    scene_.clear();
  }

  void init() {
    const int N = static_cast<int>(mass_.size());
    if (N == 0)
      throw std::runtime_error("voro: Simulation.init() needs set_masses(masses) first.");
    if (static_cast<int>(pos_.size() / 3) != N)
      throw std::runtime_error("voro: Simulation.init(): set_positions gave " +
                               std::to_string(pos_.size() / 3) + " points but set_masses " +
                               std::to_string(N) + " — every per-particle array must be (N,·).");
    if (vel_.empty())
      vel_.assign(3 * (size_t)N, real_t(0));  // no set_velocities: start at rest
    if (static_cast<int>(vel_.size() / 3) != N)
      throw std::runtime_error("voro: Simulation.init(): set_velocities gave " +
                               std::to_string(vel_.size() / 3) + " rows for " + std::to_string(N) +
                               " particles.");
    if (!visc_.empty() && static_cast<int>(visc_.size()) != N)
      throw std::runtime_error("voro: Simulation.init(): set_viscosities must be (N,).");
    if (!bulk_.empty() && static_cast<int>(bulk_.size()) != N)
      throw std::runtime_error("voro: Simulation.init(): set_bulk_viscosities must be (N,).");
    for (int i = 0; i < N; ++i)
      if (!(mass_[i] > real_t(0)))
        throw std::invalid_argument("voro: Simulation.init(): every mass must be > 0 (particle " +
                                    std::to_string(i) + ").");
    std::vector<real_t> invm(N);
    for (int i = 0; i < N; ++i)
      invm[i] = real_t(1) / mass_[i];
    if (scene_.set)
      sim_.template emplace<EE<SceneT>>();
    else
      sim_.template emplace<EE<NoSdfT>>();
    dmass_ = peclet::core::toDevice<real_t>(
        mass_, "mass");  // resident; kinetic-energy reads it each call (E4b)
    std::visit(
        [&](auto& s) {
          using T = std::decay_t<decltype(s)>;
          s.setRepair(repair_);
          s.setProfile(profile_);
          if constexpr (std::is_same_v<T, EE<SceneT>>)
            s.setSdf(scene_.scene);
          s.init(peclet::core::toDevice<real_t>(pos_, "pos"),
                 peclet::core::toDevice<real_t>(vel_, "vel"),
                 peclet::core::toDevice<real_t>(invm, "im"), L_, pressEq_);
          if (!visc_.empty()) {
            if (bulk_.empty())
              bulk_.assign(N, 0.0);
            s.setViscous(peclet::core::toDevice<real_t>(visc_, "visc"),
                         peclet::core::toDevice<real_t>(bulk_, "bulk"));
          }
        },
        sim_);
    inited_ = true;
  }

  // The time step is configured with `set_dt` and read back as `dt`, like every other stepper in
  // the suite (docs/NAMING.md 1.5); `step(n)` advances n steps of it and raises if none was set.
  void set_dt(real_t dt) {
    if (!(dt > real_t(0)))
      throw std::invalid_argument("voro: set_dt(dt) needs dt > 0.");
    dt_ = dt;
  }
  real_t dt() const { return dt_; }
  void step(int nsteps) {
    if (!inited_)
      throw std::runtime_error(
          "voro: Simulation.step() before init() — set the state, call "
          "init(), then set_dt(dt) and step(n).");
    if (!(dt_ > real_t(0)))
      throw std::invalid_argument("voro: step() has no time step — call set_dt(dt) first.");
    if (nsteps < 0)
      throw std::invalid_argument("voro: step(num_steps) needs num_steps >= 0.");
    std::visit([&](auto& s) { s.step(nsteps, dt_); }, sim_);
  }

  nb::ndarray<nb::numpy, real_t> get_positions() {
    return from3(std::visit([](auto& s) { return s.positions(); }, sim_));
  }
  nb::ndarray<nb::numpy, real_t> get_velocities() {
    return from3(std::visit([](auto& s) { return s.velocities(); }, sim_));
  }
  nb::ndarray<nb::numpy, real_t> get_forces() {
    return from3(std::visit([](auto& s) { return s.force(); }, sim_));
  }
  real_t kinetic_energy() {
    return std::visit([&](auto& s) { return s.kineticEnergy(dmass_); }, sim_);
  }
  real_t internal_energy() {
    return std::visit([](auto& s) { return s.internalEnergy(); }, sim_);
  }
  real_t time() {
    return std::visit([](auto& s) { return s.time(); }, sim_);
  }
  int num_particles() const {
    return std::visit([](auto& s) { return s.numParticles(); }, sim_);
  }

  nb::ndarray<nb::numpy, real_t> get_volumes() {
    const std::size_t N = static_cast<std::size_t>(num_particles());
    DView cv = std::visit([](auto& s) { return s.view().cellVolume; }, sim_);
    auto cell = Kokkos::subview(cv, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(cell), {N}, {1});
  }

  nb::ndarray<nb::numpy, int> get_neighbor_counts() {
    // Per-cell facet (neighbour) count is the explicit cellFacetCount view. NOTE: the device
    // cellFacetOffset is a per-cell *base* into the facet buffer in cell-finish order, NOT a CSR
    // prefix sum (see tessellation_view.hpp), so off(i+1)-off(i) is meaningless — read the count.
    const std::size_t N = static_cast<std::size_t>(num_particles());
    Kokkos::View<int*, Mem> fc = std::visit([](auto& s) { return s.view().cellFacetCount; }, sim_);
    auto cnt = Kokkos::subview(fc, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(cnt), {N}, {1});
  }

 private:
  template <class S>
  using EE = peclet::voro::physics::ExplicitEuler<real_t, S>;
  using EeVariant = std::variant<EE<NoSdfT>, EE<SceneT>>;

  void beforeInit(const char* what) const {
    if (inited_)
      throw std::runtime_error(std::string("voro: Simulation.") + what +
                               "() after init() — the particle state is resident on the device "
                               "once init() ran; set everything, then call init() (a new "
                               "Simulation for a new state).");
  }

  // Flat (3N,) host-or-device view -> (N,3) float64 numpy array (single D2H, no host loop — S2a).
  static nb::ndarray<nb::numpy, real_t> from3(const DView& d) {
    const std::size_t N = static_cast<std::size_t>(d.extent(0)) / 3;
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(d), {N, std::size_t(3)},
                                                   {3, 1});
  }

  real_t dt_{0};
  std::array<real_t, 3> L_{1, 1, 1};
  real_t pressEq_ = 0;
  bool repair_ = false, profile_ = false;
  bool inited_ = false;
  std::vector<real_t> pos_, vel_, mass_, visc_, bulk_;
  DView dmass_;  // device-resident masses, uploaded once in init() (E4b)
  SceneHolder scene_;
  EeVariant sim_;
};

struct SimDiagnostics {
  Sim* s;
};

#ifdef PECLET_VORO_MPI
void ensureMpi() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
  }
}
using Vec3 = std::array<real_t, 3>;
std::vector<Vec3> toVec3(nb::ndarray<real_t, nb::c_contig> a, const char* fn) {
  if (a.ndim() != 2 || a.shape(1) != 3)
    throw std::runtime_error(std::string(fn) + ": positions must be (N,3)");
  const std::size_t N = a.shape(0);
  const real_t* p = a.data();
  std::vector<Vec3> v(N);
  for (std::size_t i = 0; i < N; ++i)
    v[i] = Vec3{p[3 * i], p[3 * i + 1], p[3 * i + 2]};
  return v;
}
nb::ndarray<nb::numpy, real_t> fromVec3(const std::vector<Vec3>& v) {
  std::vector<real_t> op(3 * v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    op[3 * i] = v[i][0];
    op[3 * i + 1] = v[i][1];
    op[3 * i + 2] = v[i][2];
  }
  return toNumpy3(std::move(op));
}

// --------------------------------------------------------------------------------------------------
// VoronoiHalo: distributed (MPI) ghost-gather for the multi-rank Voronoi tessellation.
//
// A Voronoi cell is fully determined by its local neighbourhood, so the distributed tessellation is
// one ORB block decomposition + one ghost exchange (no iteration): each rank owns a block, gathers
// every seed within `rcut`, tessellates its owned+ghost subset with the SINGLE-RANK `Tessellation`
// building only the first `n_owned` cells, and keeps those cells — they are bit-identical to the
// serial cells. This binds `peclet::voro::mpi::VoronoiHalo<double>`; drive it from mpi4py.
// --------------------------------------------------------------------------------------------------
class VHalo {
 public:
  VHalo(std::array<long, 3> cells, std::array<real_t, 3> extent, std::array<real_t, 3> origin,
        std::array<bool, 3> periodic) {
    for (int a = 0; a < 3; ++a) {
      if (cells[a] < 1)
        throw std::invalid_argument("voro: VoronoiHalo(cells=...) needs three counts >= 1.");
      if (!(extent[a] > real_t(0)))
        throw std::invalid_argument("voro: VoronoiHalo(extent=...) needs three positive lengths.");
    }
    ensureMpi();
    halo_.init(origin, extent, cells, periodic, MPI_COMM_WORLD);
  }

  int rank() const { return halo_.rank(); }
  int num_ranks() const { return halo_.size(); }

  // Per-point mask (N,) int32: 1 if this rank owns the point, else 0.
  nb::ndarray<nb::numpy, int32_t> owned_mask(nb::ndarray<real_t, nb::c_contig> a) {
    auto p = toVec3(a, "owned_mask");
    const int r = halo_.rank();
    std::vector<int32_t> m(p.size());
    for (std::size_t i = 0; i < p.size(); ++i)
      m[i] = (halo_.ownerOf(p[i]) == r) ? 1 : 0;
    return peclet::core::python::vector_to_ndarray(std::move(m), {p.size()}, {1});
  }

  int owner_of(Vec3 x) { return halo_.ownerOf(x); }

  // Gather ghost seeds within rcut. Returns (pos (M,3), gid (M,), weight (M,), n_owned); the first
  // n_owned rows are this rank's owned seeds, the rest are gathered ghosts (periodic images incl.).
  nb::tuple gather(nb::ndarray<real_t, nb::c_contig> pos, nb::ndarray<int64_t, nb::c_contig> gid,
                   double rcut, std::optional<nb::ndarray<real_t, nb::c_contig>> weight) {
    auto ownedPos = toVec3(pos, "gather");
    const std::size_t N = ownedPos.size();
    if (gid.ndim() != 1 || gid.shape(0) != N)
      throw std::runtime_error("gather: gids must be (N,) int64, one per position");
    if (weight && (weight->ndim() != 1 || weight->shape(0) != N))
      throw std::runtime_error("gather: weights must be (N,) float64, one per position");
    if (!(rcut > 0))
      throw std::invalid_argument("voro: gather(rcut=...) needs rcut > 0.");
    std::vector<long> ownedGid(N);
    std::vector<real_t> ownedW(N, real_t(0));
    for (std::size_t i = 0; i < N; ++i) {
      ownedGid[i] = static_cast<long>(gid.data()[i]);
      if (weight)
        ownedW[i] = weight->data()[i];
    }
    auto g = halo_.gather(ownedPos, ownedGid, ownedW, rcut);
    const std::size_t M = g.pos.size();
    std::vector<int64_t> og(M);
    std::vector<real_t> ow(M);
    for (std::size_t i = 0; i < M; ++i) {
      og[i] = static_cast<int64_t>(g.gid[i]);
      ow[i] = g.weight[i];
    }
    return nb::make_tuple(fromVec3(g.pos),
                          peclet::core::python::vector_to_ndarray(std::move(og), {M}, {1}),
                          toNumpy1(std::move(ow)), g.nOwned);
  }

  // Position-only halo refresh (Verlet fast path): re-forward the CURRENT owned positions onto the
  // topology of the last gather(); returns the combined owned+ghost positions (M,3) in that order.
  nb::ndarray<nb::numpy, real_t> refresh_positions(nb::ndarray<real_t, nb::c_contig> pos) {
    auto ownedPos = toVec3(pos, "refresh_positions");
    std::vector<Vec3> out;
    halo_.refreshPositions(ownedPos, out);
    return fromVec3(out);
  }

 private:
  peclet::voro::mpi::VoronoiHalo<real_t> halo_;
};

// --------------------------------------------------------------------------------------------------
// DistributedTessellation: the library-level distributed repair driver
// (peclet::voro::mpi::DistributedMovingTessellation): VoronoiHalo + the device MovingTessellation
// under the distributed Verlet-skin invariant — the moving-point fast path under MPI, which the
// VoronoiHalo recipe (cold build per step) does not give. Owned cells are [0, num_owned) of the
// combined (owned + ghost) tessellation this rank holds.
// --------------------------------------------------------------------------------------------------
template <class S>
using DMT = peclet::voro::mpi::DistributedMovingTessellation<real_t, defaults::kMaxPlanes,
                                                             defaults::kMaxTriangles, S>;
using DmtVariant = std::variant<DMT<NoSdfT>, DMT<SceneT>>;

class DTess : public peclet::core::python::Releasable {
 public:
  DTess(std::array<long, 3> cells, std::array<real_t, 3> extent, std::array<real_t, 3> origin,
        std::array<bool, 3> periodic, real_t rcut, real_t skin, real_t tolerance)
      : cells_(cells), L_(extent), rcutFrac_(rcut), skinFrac_(skin), tolFrac_(tolerance) {
    checkDomain("DistributedTessellation", extent, origin, periodic);
    for (int a = 0; a < 3; ++a)
      if (cells[a] < 1)
        throw std::invalid_argument(
            "voro: DistributedTessellation(cells=...) needs three counts >= 1.");
    if (!(rcut > 0) || !(skin > 0) || !(tolerance > 0))
      throw std::invalid_argument(
          "voro: DistributedTessellation(rcut, skin, tolerance) need three positive fractions of "
          "the mean spacing.");
    ensureMpi();
    halo_.init(origin, extent, cells, periodic, MPI_COMM_WORLD);
  }

  int rank() const { return halo_.rank(); }
  int num_ranks() const { return halo_.size(); }

  nb::ndarray<nb::numpy, int32_t> owned_mask(nb::ndarray<real_t, nb::c_contig> a) {
    auto p = toVec3(a, "owned_mask");
    const int r = halo_.rank();
    std::vector<int32_t> m(p.size());
    for (std::size_t i = 0; i < p.size(); ++i)
      m[i] = (halo_.ownerOf(p[i]) == r) ? 1 : 0;
    return peclet::core::python::vector_to_ndarray(std::move(m), {p.size()}, {1});
  }

  void set_geometry(nb::ndarray<int, nb::c_contig> node_ints,
                    nb::ndarray<real_t, nb::c_contig> node_reals, int root, real_t grad_h) {
    beforeEstablish("set_geometry");
    scene_ = makeSceneHolder(node_ints, node_reals, root, grad_h);
  }
  void clear_geometry() {
    beforeEstablish("clear_geometry");
    scene_.clear();
  }
  void set_wall_mode(const std::string& mode, real_t skin_frac) {
    beforeEstablish("set_wall_mode");
    wallExact_ = parseWallMode(mode);
    if (skin_frac < real_t(0))
      throw std::invalid_argument("voro: set_wall_mode(skin_frac=...) needs skin_frac >= 0.");
    wallSkinFrac_ = skin_frac;
  }

  // Collective: establish the tessellation from this rank's owned seeds (all ranks call it).
  void establish(nb::ndarray<real_t, nb::c_contig> pos, nb::ndarray<int64_t, nb::c_contig> gid,
                 std::optional<nb::ndarray<real_t, nb::c_contig>> weight) {
    auto ownedPos = toVec3(pos, "establish");
    const std::size_t N = ownedPos.size();
    if (gid.ndim() != 1 || gid.shape(0) != N)
      throw std::runtime_error("establish: gids must be (N,) int64, one per position");
    if (weight && (weight->ndim() != 1 || weight->shape(0) != N))
      throw std::runtime_error("establish: weights must be (N,) float64, one per position");
    std::vector<long> ownedGid(N);
    std::vector<real_t> ownedW(N, real_t(0));
    for (std::size_t i = 0; i < N; ++i) {
      ownedGid[i] = static_cast<long>(gid.data()[i]);
      if (weight)
        ownedW[i] = weight->data()[i];
    }
    long nLocal = static_cast<long>(N), nGlobal = 0;
    MPI_Allreduce(&nLocal, &nGlobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (nGlobal <= 0)
      throw std::runtime_error("voro: establish() needs at least one point over all ranks");
    const double boxVol = static_cast<double>(L_[0]) * L_[1] * L_[2];
    const real_t spacing = static_cast<real_t>(std::cbrt(boxVol / nGlobal));
    if (scene_.set)
      dmt_.template emplace<DMT<SceneT>>();
    else
      dmt_.template emplace<DMT<NoSdfT>>();
    std::visit(
        [&](auto& d) {
          using T = std::decay_t<decltype(d)>;
          d.init({0, 0, 0}, L_, cells_, {true, true, true}, rcutFrac_ * spacing,
                 skinFrac_ * spacing, tolFrac_ * spacing, MPI_COMM_WORLD, defaults::kSearchWindow,
                 static_cast<int>(nGlobal));
          if constexpr (std::is_same_v<T, DMT<SceneT>>)
            d.setSdf(scene_.scene);
          d.setWallMode(wallExact_, wallSkinFrac_ * spacing);
          d.establish(ownedPos, ownedGid, ownedW);
        },
        dmt_);
    established_ = true;
  }

  // Collective every step: advance to the new owned positions (same ownership as establish).
  nb::dict step(nb::ndarray<real_t, nb::c_contig> pos) {
    if (!established_)
      throw std::runtime_error(
          "voro: DistributedTessellation.step() before establish() — call "
          "establish(positions, gids) first.");
    auto ownedPos = toVec3(pos, "step");
    if (static_cast<int>(ownedPos.size()) != num_owned())
      throw std::runtime_error("voro: step(positions) got " + std::to_string(ownedPos.size()) +
                               " points but establish() had " + std::to_string(num_owned()) +
                               " — ownership is fixed by establish(); call it again to change it.");
    // StepStats is a nested type of each instantiation, so the visitor returns a plain pair.
    auto [regathered, repair] = std::visit(
        [&](auto& d) {
          auto st = d.step(ownedPos);
          return std::pair<bool, peclet::voro::RepairStats>(st.regathered, st.repair);
        },
        dmt_);
    nb::dict d = repairStatsDict(regathered ? peclet::voro::RepairStats{} : repair);
    d["regathered"] = regathered;  // this step re-gathered the ghosts + cold-rebuilt
    return d;
  }

  int num_owned() const {
    return std::visit([](auto& d) { return d.nOwned(); }, dmt_);
  }
  int num_combined() const {
    return std::visit([](auto& d) { return d.nCombined(); }, dmt_);
  }
  long num_regathers() const {
    return std::visit([](auto& d) { return d.numRegathers(); }, dmt_);
  }
  void set_profile(bool on) {
    std::visit([&](auto& d) { d.setProfile(on); }, dmt_);
  }

  nb::ndarray<nb::numpy, real_t> get_volumes() {
    const std::size_t N = static_cast<std::size_t>(num_owned());
    DView vol = std::visit([](auto& d) { return d.tess().vol; }, dmt_);
    auto v = Kokkos::subview(vol, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(v), {N}, {1});
  }
  nb::ndarray<nb::numpy, int> get_neighbor_counts() {
    using Cell =
        peclet::voro::ConvexCell<real_t, defaults::kMaxPlanes, defaults::kMaxTriangles, false>;
    const int N = num_owned();
    Kokkos::View<int*, peclet::core::MemSpace> cnt("nbr", N);
    auto st = std::visit([](auto& d) { return d.tess().store; }, dmt_);
    auto C = cnt;
    const real_t Lx = L_[0], Ly = L_[1], Lz = L_[2];
    Kokkos::parallel_for(
        "peclet.voro.dnbrcount", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i) {
          Cell c;
          st.load(i, c, Lx, Ly, Lz);
          C(i) = c.countFaces();
        });
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(cnt),
                                                   {static_cast<std::size_t>(N)}, {1});
  }
  nb::ndarray<nb::numpy, int> get_wall_counts() {
    const std::size_t N = static_cast<std::size_t>(num_owned());
    std::vector<int> v = std::visit(
        [&](auto& d) {
          using T = std::decay_t<decltype(d)>;
          if constexpr (std::is_same_v<T, DMT<SceneT>>) {
            auto all = peclet::core::toVector(d.tess().wall.cnt);
            all.resize(N);
            return all;
          } else {
            return std::vector<int>(N, 0);
          }
        },
        dmt_);
    return peclet::core::python::vector_to_ndarray(std::move(v), {N}, {1});
  }
  // Global ids of the combined (owned + ghost) seeds after the last (re)gather, (num_combined,).
  nb::ndarray<nb::numpy, int64_t> get_combined_gids() {
    const auto& g =
        std::visit([](auto& d) -> const std::vector<long>& { return d.combinedGid(); }, dmt_);
    std::vector<int64_t> v(g.begin(), g.end());
    const std::size_t M = v.size();
    return peclet::core::python::vector_to_ndarray(std::move(v), {M}, {1});
  }

  void release() noexcept override {
    dmt_.template emplace<DMT<NoSdfT>>();
    scene_.clear();
    established_ = false;
  }

 private:
  void beforeEstablish(const char* what) const {
    if (established_)
      throw std::runtime_error(std::string("voro: DistributedTessellation.") + what +
                               "() after establish() — it configures the next establish(); call "
                               "establish() again afterwards.");
  }

  std::array<long, 3> cells_;
  std::array<real_t, 3> L_;
  real_t rcutFrac_, skinFrac_, tolFrac_;
  bool wallExact_ = true;
  real_t wallSkinFrac_ = defaults::kWallSkin;
  bool established_ = false;
  SceneHolder scene_;
  peclet::voro::mpi::VoronoiHalo<real_t> halo_;  // for owned_mask before establish()
  DmtVariant dmt_;
};

struct DTessDiagnostics {
  DTess* d;
};
#endif  // PECLET_VORO_MPI

}  // namespace peclet::voro::pybind

using namespace peclet::voro::pybind;

NB_MODULE(_voro, m) {
  using namespace defaults;
  m.attr("__doc__") =
      "peclet.voro (device/Kokkos): moving-particle Voronoi tessellation and dynamics.\n\n"
      "Classes: Tessellation (cold build + incremental repair, volumes, neighbour counts, energy\n"
      "forces), FlowSolver (static Navier-Stokes on the face mesh), Simulation (moving-cell\n"
      "compressible-Euler / Navier-Stokes fluid); functions optimize_volume_mesh, "
      "minimize_interface;\n"
      "the pore-space family under peclet.voro.pore_mesh; VoronoiHalo and DistributedTessellation\n"
      "when built with MPI. Every instrument lives on the object's `diagnostics`. Arrays are "
      "NumPy:\n"
      "positions/velocities (N,3) float64, scalars (N,). The backend (Serial/OpenMP/CUDA/HIP) is\n"
      "fixed at build time; see peclet.voro.execution_space. peclet.voro.defaults lists the named\n"
      "defaults the engine is driven with.";
  // Kokkos init + the release-then-finalize atexit hook + finalize() + execution_space: the
  // suite-wide teardown pattern (file comment; peclet/core/python/kokkos_teardown.hpp). Every
  // Tess/Flow/Sim is a Releasable, so the registry releases them in one sweep before finalize.
  peclet::core::python::install(m);

  {
    nb::dict d;
    d["max_planes"] = kMaxPlanes;
    d["max_triangles"] = kMaxTriangles;
    d["pore_max_planes"] = kPoreMaxPlanes;
    d["pore_max_triangles"] = kPoreMaxTriangles;
    d["certificate_tolerance"] = kCertificateTolerance;
    d["skin"] = kSkin;
    d["search_window"] = kSearchWindow;
    d["sdf_gradient_step"] = kSdfGradientStep;
    d["wall_skin"] = kWallSkin;
    d["optimizer_search_window"] = kOptimizerSearchWindow;
    d["pore_search_window"] = kPoreSearchWindow;
    d["optimizer_max_iter"] = kOptimizerMaxIter;
    d["pore_max_iter"] = kPoreMaxIter;
    d["optimizer_tolerance"] = kOptimizerTolerance;
    d["optimizer_cg_iters"] = kOptimizerCgIters;
    d["pore_cg_iters"] = kPoreCgIters;
    d["barrier_decay"] = kBarrierDecay;
    d["interface_sigma"] = kInterfaceSigma;
    d["pore_max_bins"] = kPoreMaxBins;
    d["distributed_rcut"] = kDistributedRcut;
    d["distributed_cells"] = kDistributedCells;
    m.attr("defaults") = d;
  }

  // ---- typed results ---------------------------------------------------------------------------
  nb::class_<OptimizeResult>(m, "OptimizeResult",
                             "Result of optimize_volume_mesh / pore_mesh.optimize_pore_mesh.")
      .def_ro("positions", &OptimizeResult::positions, "The optimised seeds (N,3) float64.")
      .def_ro("weights", &OptimizeResult::weights,
              "The optimised power weights (N,) float64, or None when use_weights=False.")
      .def_ro("iters", &OptimizeResult::iters, "Gauss-Newton / descent iterations run.")
      .def_ro("max_vol_err", &OptimizeResult::max_vol_err,
              "max_i |V_i / V_ref,i - 1| at the returned seeds.")
      .def_ro("mean_vol_err", &OptimizeResult::mean_vol_err,
              "mean_i |V_i / V_ref,i - 1| at the returned seeds.")
      .def_ro("converged", &OptimizeResult::converged, "True if the gradient fell below tol.")
      .def_ro("num_empty", &OptimizeResult::num_empty,
              "Seeds whose cell is empty at the returned seeds (0 for a valid mesh).")
      .def("__repr__", [](const OptimizeResult& r) {
        return "OptimizeResult(iters=" + std::to_string(r.iters) +
               ", max_vol_err=" + fmt(r.max_vol_err) + ", mean_vol_err=" + fmt(r.mean_vol_err) +
               ", converged=" + (r.converged ? "True" : "False") +
               ", num_empty=" + std::to_string(r.num_empty) + ")";
      });
  nb::class_<InterfaceResult>(m, "InterfaceResult", "Result of minimize_interface.")
      .def_ro("positions", &InterfaceResult::positions, "The minimised seeds (N,3) float64.")
      .def_ro("energy", &InterfaceResult::energy,
              "Final interfacial energy E = sum sigma A_ij over faces between different types.")
      .def_ro("energy_ratio", &InterfaceResult::energy_ratio,
              "energy / the energy of the input seeds.")
      .def_ro("iters", &InterfaceResult::iters, "Descent iterations run.")
      .def_ro("converged", &InterfaceResult::converged, "True if the gradient fell below tol.")
      .def("__repr__", [](const InterfaceResult& r) {
        return "InterfaceResult(energy=" + fmt(r.energy) + ", energy_ratio=" + fmt(r.energy_ratio) +
               ", iters=" + std::to_string(r.iters) +
               ", converged=" + (r.converged ? "True" : "False") + ")";
      });

  // ---- mesh optimisers on a periodic box ------------------------------------------------------
  m.def(
      "optimize_volume_mesh",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> vset_in,
         std::array<real_t, 3> extent, int search_window, int max_iter, real_t tol, int cg_iters,
         bool use_weights, const std::string& method) {
        auto pos = flatten3(pos_in);
        auto vset = flatten1(vset_in);
        const int N = (int)vset.size();
        if ((int)(pos.size() / 3) != N)
          throw std::runtime_error(
              "optimize_volume_mesh: positions (N,3) and target_volumes (N,) must agree on N");
        for (int a = 0; a < 3; ++a)
          if (!(extent[a] > real_t(0)))
            throw std::invalid_argument(
                "voro: optimize_volume_mesh(extent=...) needs three positive lengths.");
        const real_t Larr[3] = {extent[0], extent[1], extent[2]};
        const auto prec = parseMethod(method, "optimize_volume_mesh");
        peclet::voro::OtResult R;
        std::vector<real_t> w;
        if (use_weights) {
          w.assign(N, 0.0);
          R = peclet::voro::meshVolumeOptimize<real_t, true>(pos, w, vset, Larr, N, search_window,
                                                             peclet::voro::NoSdf{}, max_iter, tol,
                                                             cg_iters, prec, false);
        } else {
          std::vector<real_t> noW;
          R = peclet::voro::meshVolumeOptimize<real_t, false>(pos, noW, vset, Larr, N,
                                                              search_window, peclet::voro::NoSdf{},
                                                              max_iter, tol, cg_iters, prec, false);
        }
        return makeOptimizeResult(std::move(pos),
                                  use_weights ? std::optional(std::move(w)) : std::nullopt, R);
      },
      nb::arg("positions"), nb::arg("target_volumes"), nb::arg("extent"), nb::kw_only(),
      nb::arg("search_window") = kOptimizerSearchWindow, nb::arg("max_iter") = kOptimizerMaxIter,
      nb::arg("tol") = kOptimizerTolerance, nb::arg("cg_iters") = kOptimizerCgIters,
      nb::arg("use_weights") = false, nb::arg("method") = "jacobi",
      doc("Move seeds (N,3) — and optionally the power weights — to minimise "
          "sum (V_i / V_ref,i - 1)^2 by\n"
          "damped Gauss-Newton (Newton-Raphson + CG) on the periodic box `extent` (Lx, Ly, Lz).\n"
          "target_volumes (N,) are the per-cell reference volumes V_ref (renormalised to the box\n"
          "volume). method: the CG preconditioner, one of " +
          std::string(kMethodList) +
          " (default 'jacobi'; 'graphamg'\n"
          "is the O(N) choice at large N, 'steepest' is plain descent). search_window (default " +
          fmt(kOptimizerSearchWindow) + "), max_iter (" + fmt(kOptimizerMaxIter) + "), tol (" +
          fmt(kOptimizerTolerance) + ")\nand cg_iters (" + fmt(kOptimizerCgIters) +
          ") are peclet.voro.defaults. Returns an OptimizeResult (positions, weights, iters,\n"
          "max_vol_err, mean_vol_err, converged, num_empty). Pure Voronoi (use_weights=False)\n"
          "reaches equal/graded volumes well; weights add fuller volume control but are limited "
          "by\nthe periodic tessellation's ~1% min-image floor."));

  m.def(
      "minimize_interface",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<int, nb::c_contig> type_in,
         std::array<real_t, 3> extent, real_t sigma, int search_window, int max_iter, real_t tol) {
        auto pos = flatten3(pos_in);
        const int N = (int)type_in.shape(0);
        if ((int)(pos.size() / 3) != N)
          throw std::runtime_error(
              "minimize_interface: positions (N,3) and types (N,) must agree on N");
        for (int a = 0; a < 3; ++a)
          if (!(extent[a] > real_t(0)))
            throw std::invalid_argument(
                "voro: minimize_interface(extent=...) needs three positive lengths.");
        std::vector<int> type(type_in.data(), type_in.data() + N);
        const real_t Larr[3] = {extent[0], extent[1], extent[2]};
        auto R = peclet::voro::interfaceMinimize<real_t>(
            pos, type, sigma, Larr, N, search_window, peclet::voro::NoSdf{}, max_iter, tol, false);
        InterfaceResult r;
        r.positions = nb::cast(toNumpy3(std::move(pos)));
        r.energy = R.energy;
        r.energy_ratio = R.energyRatio;
        r.iters = R.iters;
        r.converged = R.converged;
        return r;
      },
      nb::arg("positions"), nb::arg("types"), nb::arg("extent"), nb::kw_only(),
      nb::arg("sigma") = kInterfaceSigma, nb::arg("search_window") = kOptimizerSearchWindow,
      nb::arg("max_iter") = kOptimizerMaxIter, nb::arg("tol") = kOptimizerTolerance,
      doc("Surface-Evolver-style interfacial-tension minimiser: move seeds (N,3) on the periodic "
          "box\n`extent` to minimise the total area of faces between cells of different integer "
          "type (N,),\nE = sum sigma A_ij (sigma default " +
          fmt(kInterfaceSigma) +
          "). Steepest descent with a trust-region line search on\nthe (non-smooth) interfacial "
          "energy; search_window / max_iter / tol default to " +
          fmt(kOptimizerSearchWindow) + " / " + fmt(kOptimizerMaxIter) + " / " +
          fmt(kOptimizerTolerance) +
          ".\nReturns an InterfaceResult (positions, energy, energy_ratio = final/initial, iters, "
          "converged)."));

  // ---- pore-space (SDF-walled) family: bound here, exposed by peclet.voro.pore_mesh ------------
  m.def(
      "optimize_pore_mesh",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> vref_in,
         nb::ndarray<real_t, nb::c_contig> sph_c, nb::ndarray<real_t, nb::c_contig> sph_r,
         std::array<real_t, 3> extent, int search_window, int max_iter, real_t tol, int cg_iters,
         const std::string& method, real_t mu_barrier, bool free_energy) {
        auto pos = flatten3(pos_in);
        auto vref = flatten1(vref_in);
        const int N = (int)vref.size();
        if ((int)(pos.size() / 3) != N)
          throw std::runtime_error(
              "optimize_pore_mesh: positions (N,3) and target_volumes (N,) must agree on N");
        const real_t L = cubicExtent(extent, "optimize_pore_mesh");
        const real_t Larr[3] = {L, L, L};
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        const auto prec = parseMethod(method, "optimize_pore_mesh");
        std::vector<real_t> noW;
        auto R = peclet::voro::meshVolumeOptimize<real_t, false, peclet::voro::SdfSpheres<real_t>>(
            pos, noW, vref, Larr, N, search_window, sdf, max_iter, tol, cg_iters, prec, false,
            mu_barrier, (real_t)kBarrierDecay, free_energy);
        return makeOptimizeResult(std::move(pos), std::nullopt, R);
      },
      nb::arg("positions"), nb::arg("target_volumes"), nb::arg("sphere_centers"),
      nb::arg("sphere_radii"), nb::arg("extent"), nb::kw_only(),
      nb::arg("search_window") = kPoreSearchWindow, nb::arg("max_iter") = kPoreMaxIter,
      nb::arg("tol") = kOptimizerTolerance, nb::arg("cg_iters") = kPoreCgIters,
      nb::arg("method") = "graphamg", nb::arg("mu_barrier") = 0.0, nb::arg("free_energy") = false,
      doc("Relax interstitial seeds (N,3) so their SDF-clipped Voronoi cell volumes approach the "
          "per-cell\ntarget_volumes (N,), with the sphere packing (sphere_centers (M,3), "
          "sphere_radii (M,)) as\nperiodic walls in the cubic box `extent` (Lx == Ly == Lz). "
          "method: one of " +
          std::string(kMethodList) +
          " (default\n'graphamg'; 'steepest' is plain descent). free_energy=True uses "
          "E = -sum V_ref log V (pressure\nV_ref/V, resists collapse); mu_barrier > 0 adds a "
          "log-barrier that decays by " +
          fmt(kBarrierDecay) +
          " per iteration.\nsearch_window / max_iter / tol / cg_iters default to " +
          fmt(kPoreSearchWindow) + " / " + fmt(kPoreMaxIter) + " / " + fmt(kOptimizerTolerance) +
          " / " + fmt(kPoreCgIters) +
          ". Returns an\nOptimizeResult. Experimental (pore-space meshing; see the "
          "pore-mesh-voronoi example)."));

  // The pore-space export: the device path (pore_cells.hpp — the tessellator's own gather at the
  // pore capacity, count -> scan -> fill in seed order) under the public names, and the host-serial
  // PoreReconstructor under the `_host` names as its test oracle (python/test_voro.py
  // test_pore_cells compares them). Both share the per-cell export rule + layout
  // (peclet::voro::detail::poreCellCount / poreCellFill).
  auto poreCellsDict = [](std::vector<real_t>&& pts, std::vector<int64_t>&& faces,
                          std::vector<int64_t>&& faceOff, std::vector<real_t>&& vol,
                          std::vector<int32_t>&& boundary, std::vector<int32_t>&& cellSeed,
                          long numOverflow, long numIncomplete) {
    const std::size_t nPts = pts.size() / 3, nCells = vol.size();
    nb::dict d;
    d["points"] = peclet::core::python::vector_to_ndarray(std::move(pts), {nPts, 3}, {3, 1});
    d["faces"] = peclet::core::python::vector_to_ndarray(std::move(faces), {faces.size()}, {1});
    d["face_offsets"] =
        peclet::core::python::vector_to_ndarray(std::move(faceOff), {faceOff.size()}, {1});
    d["volume"] = peclet::core::python::vector_to_ndarray(std::move(vol), {nCells}, {1});
    d["boundary"] = peclet::core::python::vector_to_ndarray(std::move(boundary), {nCells}, {1});
    d["seed"] = peclet::core::python::vector_to_ndarray(std::move(cellSeed), {nCells}, {1});
    d["num_overflow"] = numOverflow;
    d["num_incomplete"] = numIncomplete;
    return d;
  };
  auto poreSectionDict = [](std::vector<real_t>&& verts, std::vector<int64_t>&& off,
                            std::vector<real_t>&& vol, std::vector<int32_t>&& cellSeed,
                            long numOverflow, long numIncomplete) {
    const std::size_t nV = verts.size() / 3, nP = vol.size();
    nb::dict d;
    d["verts"] = peclet::core::python::vector_to_ndarray(std::move(verts), {nV, 3}, {3, 1});
    d["offsets"] = peclet::core::python::vector_to_ndarray(std::move(off), {off.size()}, {1});
    d["volume"] = peclet::core::python::vector_to_ndarray(std::move(vol), {nP}, {1});
    d["seed"] = peclet::core::python::vector_to_ndarray(std::move(cellSeed), {nP}, {1});
    d["num_overflow"] = numOverflow;
    d["num_incomplete"] = numIncomplete;
    return d;
  };
  m.def(
      "sdf_voronoi_cells",
      [poreCellsDict](nb::ndarray<real_t, nb::c_contig> pos_in,
                      nb::ndarray<real_t, nb::c_contig> sph_c,
                      nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent,
                      int search_window) {
        auto seed = flatten3(pos_in);
        const int N = (int)(seed.size() / 3);
        const real_t L = cubicExtent(extent, "sdf_voronoi_cells");
        if (N <= 0)
          throw std::invalid_argument(
              "voro: sdf_voronoi_cells(positions) needs at least one seed.");
        if (search_window < 1)
          throw std::invalid_argument("voro: sdf_voronoi_cells(search_window=...) needs >= 1.");
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        auto pos = peclet::core::toDevice<real_t>(seed, "pore.pos");
        const real_t Larr[3] = {L, L, L};
        auto r = peclet::voro::buildPoreCells<real_t, peclet::voro::SdfSpheres<real_t>>(
            pos, N, Larr, sdf, search_window);
        using peclet::voro::detail::toHostVecT;
        return poreCellsDict(toHostVecT<real_t>(r.points), toHostVecT<int64_t>(r.faces),
                             toHostVecT<int64_t>(r.faceOffset), toHostVecT<real_t>(r.volume),
                             toHostVecT<int32_t>(r.boundary), toHostVecT<int32_t>(r.seed),
                             r.numOverflow, r.numIncomplete);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      nb::kw_only(), nb::arg("search_window") = kPoreSearchWindow,
      doc("Reconstruct the SDF-clipped interstitial Voronoi cells (cubic periodic box `extent`, "
          "the\nspheres as walls) on the device — the tessellator's own gather, one thread per "
          "cell — and\nreturn their polyhedra as flat arrays (VTK_POLYHEDRON layout) in seed "
          "order: 'points' (Np,3),\n'faces' + 'face_offsets' (per-cell face lists, global point "
          "ids, each face CCW about its\noutward normal), 'volume' (Nc,), 'boundary' (Nc, 1 "
          "where the cell touches a sphere wall),\n'seed' (Nc,). Seeds inside a sphere have no "
          "cell; 'num_overflow' counts cells skipped for\nexceeding the cell capacity ("
          "peclet.voro.defaults max_planes / max_triangles) and 'num_incomplete' the cells "
          "whose\ngather window "
          "(search_window grid blocks per axis, default " +
          fmt(kPoreSearchWindow) + ") did not close — raise\nsearch_window if it is not 0."));
  m.def(
      "sdf_voronoi_section",
      [poreSectionDict](
          nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> sph_c,
          nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent,
          std::array<real_t, 3> point, std::array<real_t, 3> normal, int search_window) {
        auto seed = flatten3(pos_in);
        const int N = (int)(seed.size() / 3);
        const real_t L = cubicExtent(extent, "sdf_voronoi_section");
        if (N <= 0)
          throw std::invalid_argument(
              "voro: sdf_voronoi_section(positions) needs at least one seed.");
        if (search_window < 1)
          throw std::invalid_argument("voro: sdf_voronoi_section(search_window=...) needs >= 1.");
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        auto pos = peclet::core::toDevice<real_t>(seed, "pore.pos");
        const real_t Larr[3] = {L, L, L};
        auto r = peclet::voro::buildPoreSection<real_t, peclet::voro::SdfSpheres<real_t>>(
            pos, N, Larr, sdf, point.data(), normal.data(), search_window);
        using peclet::voro::detail::toHostVecT;
        return poreSectionDict(toHostVecT<real_t>(r.verts), toHostVecT<int64_t>(r.offset),
                               toHostVecT<real_t>(r.volume), toHostVecT<int32_t>(r.seed),
                               r.numOverflow, r.numIncomplete);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      nb::arg("point"), nb::arg("normal"), nb::kw_only(),
      nb::arg("search_window") = kPoreSearchWindow,
      doc("Cross-section of the SDF-clipped interstitial Voronoi mesh (cubic periodic box "
          "`extent`) by\nthe plane through `point` with `normal`, on the device: every cell cut "
          "directly\n(ConvexCell::sectionPolygon, from the dual edges, so it tiles the plane "
          "exactly where a\nface-by-face slice drops facets). Returns 'verts' (Nv,3, world "
          "coords, all on the plane, CCW\nabout the normal) + 'offsets' (Npoly+1, per-polygon "
          "vertex ranges) + 'volume' (Npoly, the 3-D\ncell volume) + 'seed' (Npoly, the seed "
          "index), in seed order, plus 'num_overflow' /\n'num_incomplete' as in "
          "sdf_voronoi_cells (search_window default " +
          fmt(kPoreSearchWindow) +
          "). For a z=z0 slice pass\npoint=(0,0,z0), normal=(0,0,1) "
          "and plot verts[:, :2]."));
  m.def(
      "_sdf_voronoi_cells_host",
      [poreCellsDict](nb::ndarray<real_t, nb::c_contig> pos_in,
                      nb::ndarray<real_t, nb::c_contig> sph_c,
                      nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent) {
        auto seed = flatten3(pos_in);
        const int N = (int)(seed.size() / 3);
        const real_t L = cubicExtent(extent, "_sdf_voronoi_cells_host");
        HView cenH, radH;
        auto sdf = makeSpheresSdfHost(sph_c, sph_r, L, cenH, radH);
        std::vector<real_t> pts, vol;
        std::vector<int64_t> faces, faceOff(1, 0);
        std::vector<int32_t> boundary, cellSeed;
        PoreReconstructor rec(seed, L, sdf);
        long numOverflow = 0;
        for (int i = 0; i < N; ++i) {
          PoreCell c;
          if (!rec.build(i, c)) {
            numOverflow += c.overflow ? 1 : 0;
            continue;
          }
          int np = 0, ne = 0;
          bool wall = false;
          if (!peclet::voro::detail::poreCellCount(c, np, ne, wall))
            continue;
          const int64_t ptBase = (int64_t)(pts.size() / 3), fBase = (int64_t)faces.size();
          pts.resize(pts.size() + 3 * (std::size_t)np);
          faces.resize(faces.size() + (std::size_t)ne);
          const real_t s[3] = {seed[3 * i], seed[3 * i + 1], seed[3 * i + 2]};
          peclet::voro::detail::poreCellFill(c, s, ptBase, fBase, pts.data(), faces.data());
          faceOff.push_back((int64_t)faces.size());
          vol.push_back(c.volumePerVertex());
          boundary.push_back(wall ? 1 : 0);
          cellSeed.push_back(i);
        }
        return poreCellsDict(std::move(pts), std::move(faces), std::move(faceOff), std::move(vol),
                             std::move(boundary), std::move(cellSeed), numOverflow, 0);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      "The host-serial reconstruction of sdf_voronoi_cells (PoreReconstructor: the 80 nearest "
      "seeds\nby a Chebyshev shell walk, closest-first clip against a far box, then the SDF "
      "clip), kept as\nthe TEST ORACLE of the device path; same result layout. Not part of the "
      "API.");
  m.def(
      "_sdf_voronoi_section_host",
      [poreSectionDict](nb::ndarray<real_t, nb::c_contig> pos_in,
                        nb::ndarray<real_t, nb::c_contig> sph_c,
                        nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent,
                        std::array<real_t, 3> point, std::array<real_t, 3> normal) {
        auto seed = flatten3(pos_in);
        const real_t L = cubicExtent(extent, "_sdf_voronoi_section_host");
        HView cenH, radH;
        auto sdf = makeSpheresSdfHost(sph_c, sph_r, L, cenH, radH);
        PoreReconstructor rec(seed, L, sdf);
        std::vector<real_t> verts, vol;
        std::vector<int64_t> off(1, 0);
        std::vector<int32_t> cellSeed;
        real_t spx[PoreCell::MAXSV], spy[PoreCell::MAXSV], spz[PoreCell::MAXSV];
        long numOverflow = 0;
        for (int i = 0; i < rec.N; ++i) {
          const real_t sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
          PoreCell c;
          if (!rec.build(i, c)) {
            numOverflow += c.overflow ? 1 : 0;
            continue;
          }
          const real_t p0[3] = {point[0] - sx, point[1] - sy, point[2] - sz};  // plane, cell frame
          const real_t u3[3] = {normal[0], normal[1], normal[2]};
          const int mm = c.sectionPolygon(p0, u3, spx, spy, spz);
          if (mm < 3)
            continue;
          for (int k = 0; k < mm; ++k) {
            verts.push_back(sx + spx[k]);
            verts.push_back(sy + spy[k]);
            verts.push_back(sz + spz[k]);
          }
          off.push_back((int64_t)(verts.size() / 3));
          vol.push_back(c.volumePerVertex());
          cellSeed.push_back(i);
        }
        return poreSectionDict(std::move(verts), std::move(off), std::move(vol),
                               std::move(cellSeed), numOverflow, 0);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      nb::arg("point"), nb::arg("normal"),
      "The host-serial reconstruction of sdf_voronoi_section, kept as the TEST ORACLE of the "
      "device\npath; same result layout. Not part of the API.");

  // ---- Tessellation -----------------------------------------------------------------------------
  nb::class_<TessDiagnostics>(
      m, "TessellationDiagnostics",
      "Instruments and ablation switches of a Tessellation (reached as `t.diagnostics`).")
      .def(
          "build_report", [](TessDiagnostics& d) { return d.t->build_report(); },
          "Validity counts of the last build: {'buried', 'reach_exceeded', 'empty', 'overflow',\n"
          "'incomplete'} — all zero for a guaranteed-exact partition (build() already warns, or\n"
          "raises with strict=True, when they are not) — and 'over_buffer_rebuilds', the number "
          "of\n"
          "times the build's facet/edge over-buffer estimate was exceeded and the build pass "
          "re-run\n"
          "at the exact demand (0 normally; each one doubles that build's cost).")
      .def(
          "set_profile", [](TessDiagnostics& d, bool on) { d.t->set_profile(on); },
          nb::arg("on") = true,
          "Print the cold build's timing (grid / build / CSR), the worklist size, the over-buffer\n"
          "rebuilds and the max facets per cell on stderr (default off). Takes effect at the next\n"
          "build().")
      .def(
          "set_local_certificate",
          [](TessDiagnostics& d, bool on) { d.t->set_local_certificate(on); }, nb::arg("on"),
          "Ablation: the cheap O(nt) Lawson local certificate (default True) vs the brute "
          "O(nt*np)\n"
          "form for detecting which cells changed. Both are complete; local is faster. Takes "
          "effect\nat the next build().")
      .def(
          "set_gate", [](TessDiagnostics& d, bool on) { d.t->set_gate(on); }, nb::arg("on"),
          "Ablation: the adaptive gate (default True) that routes high-churn steps straight to a\n"
          "full rebuild — the 'never much slower than a cold build' guard. Takes effect at the "
          "next\nbuild().");

  nb::class_<Tess>(
      m, "Tessellation",
      "Moving-particle (power-)Voronoi tessellator on the device path, optionally clipped by an\n"
      "SDF solid.\n\n"
      "Build a tessellation once (`build`) then advance it cheaply as the points move (`step`) —\n"
      "the incremental two-pass repair is several times faster than rebuilding for the small\n"
      "per-step displacements typical of CFD/DEM, and falls back to a full rebuild (via an\n"
      "adaptive gate) when displacements are large, so it is never much slower than a cold\n"
      "build. Periodic box anchored at the origin. Single domain (one process); see\n"
      "DistributedTessellation for the MPI driver. Instruments: `diagnostics`.")
      .def(nb::init<>())
      .def("set_domain", &Tess::set_domain, nb::arg("extent"),
           nb::arg("origin") = std::array<real_t, 3>{0, 0, 0},
           nb::arg("periodic") = std::array<bool, 3>{true, true, true},
           "Set the periodic box before `build`: `extent` is the box SIZE (Lx, Ly, Lz). The "
           "suite-wide spelling (suite/docs/NAMING.md 1.1), the same call "
           "`dem.Simulation.set_domain` takes. "
           "`origin` must be (0, 0, 0) and `periodic` (True, True, True) — this engine's box is "
           "anchored at the origin and periodic on every axis; both are checked rather than "
           "ignored, so a caller who writes the suite-wide form gets an error naming the "
           "limitation.")
      .def_prop_ro("extent", &Tess::extent,
                   "The box size (Lx, Ly, Lz) — read-only; set it with `set_domain`.")
      .def("set_tolerance", &Tess::set_tolerance, nb::arg("frac") = kCertificateTolerance,
           doc("Certificate tolerance of the repair as a fraction of the mean inter-particle "
               "spacing\n(default " +
               fmt(kCertificateTolerance) +
               " = peclet.voro.defaults['certificate_tolerance']). A vertex poking past a "
               "stored\nplane by more than this flags the cell for repair; smaller is stricter "
               "(closer to\nmachine-exact) at marginally higher cost. Takes effect at the next "
               "build()."))
      .def("set_geometry", &Tess::set_geometry, nb::arg("node_ints"), nb::arg("node_reals"),
           nb::arg("root") = 0, nb::arg("grad_h") = kSdfGradientStep,
           doc("Clip the cells by an SDF solid given as a core shape scene in the flat node "
               "encoding\n(node_ints int32 (3 per node), node_reals float64 (16 per node)) — "
               "exactly "
               "what\npeclet.core.geom.Scene.encode() returns and dem.add_analytic_wall takes; "
               "`root` is the\ntree root to evaluate. Suite sign convention: sdf < 0 inside the "
               "solid. Seeds inside the\nsolid get no cell (volume 0); cells reaching into it gain "
               "wall facets. Applies to the\nnext `build` and is carried through every `step` "
               "(wall planes are resident; a boundary\nwatch re-clips cells at the wall). `grad_h` "
               "(default " +
               fmt(kSdfGradientStep) +
               ") is the central-difference step for the\nSDF gradient. Analytic vocabulary only "
               "(no sampled grids through this path yet)."))
      .def("clear_geometry", &Tess::clear_geometry,
           "Drop the SDF geometry (takes effect at the next `build`).")
      .def("set_wall_mode", &Tess::set_wall_mode, nb::arg("mode"), nb::arg("skin_frac") = kWallSkin,
           doc("Wall re-gather policy for `step`. mode: one of " + std::string(kWallModeList) +
               ". 'exact' (the default)\nre-clips every wall-clipped cell that moved, so the "
               "incremental result equals a cold\nrebuild; 'skin' keeps a cell's stale tangent "
               "planes until it moved more than skin_frac x\nthe mean spacing (default " +
               fmt(kWallSkin) +
               "; cheaper, not exact by construction). Takes effect at the\nnext build()."))
      .def(
          "set_weights", &Tess::set_weights, nb::arg("weights"),
          "Per-seed POWER (Laguerre) weights (N,) float64: the cells become the power diagram\n"
          "(radical planes) instead of the Voronoi diagram. Takes effect at the next `build`;\n"
          "call again before a `step` to update the weights alongside the positions. Exact in the\n"
          "small-weight regime (see the docs).")
      .def("clear_weights", &Tess::clear_weights,
           "Back to the unweighted Voronoi diagram (next `build`).")
      .def("build", &Tess::build, nb::arg("positions"), nb::arg("strict") = false,
           "Cold-build the (power-)Voronoi tessellation of `positions` (N,3) from scratch and make "
           "it resident, clipped by the geometry from `set_geometry` if any.\n"
           "Sets the particle count N for subsequent `step` calls. Warns (raises if strict=True)\n"
           "when the result is not a guaranteed-exact partition: buried power cells (a seed "
           "outside\n"
           "its own cell — never for w = r² of non-overlapping spheres), a search reach beyond "
           "half\n"
           "the box, or overflowed cells; see `diagnostics.build_report()`.")
      .def(
          "step", &Tess::step, nb::arg("positions"),
          "Incrementally repair the resident tessellation to new `positions` (N,3, same N as "
          "`build`;\nraises before `build`).\n"
          "Returns a dict of per-step work stats: 'flagged' (cells the certificate flagged), "
          "'pass1'\n"
          "and 'pass2' (cells re-gathered in each pass), 'extra' (cells gathered across verify "
          "extra-passes),\n"
          "'surgical' (Pass-1 cells repaired surgically), 'verify_passes' (verify iterations run), "
          "'rebuilt'\n"
          "(True if the gate routed this step to a full rebuild), 'fell_back' (True if the verify "
          "failed and\n"
          "a cold rebuild was forced), 'wall_flagged' (cells the SDF boundary watch re-clipped).")
      .def("get_volumes", &Tess::get_volumes,
           "Per-particle Voronoi cell volume (N,) float64 (a copy). Sums to the box volume "
           "(space-filling).")
      .def("get_neighbor_counts", &Tess::get_neighbor_counts,
           "Per-particle Voronoi neighbour count (N,) int32 (a copy) — the number of faces of each "
           "cell (wall facets included).")
      .def("get_wall_counts", &Tess::get_wall_counts,
           "Per-particle number of resident SDF wall planes (N,) int32 (a copy); all zero without "
           "geometry.")
      .def(
          "energy_forces", &Tess::energy_forces, nb::arg("types"), nb::arg("tension"),
          nb::arg("sigma_wall") = nb::none(), nb::arg("dEdV") = nb::none(), nb::arg("lloyd") = 0.0,
          nb::arg("facet_tension") = 0.0,
          "Energies and their exact gradients on the RESIDENT cells (after build/step), no "
          "rebuild:\n"
          "  interfacial  E = Σ σ(t_i,t_j) A_ij over facets between different `types` (N,) int32,\n"
          "               with the symmetric `tension` table (nTypes, nTypes) float64;\n"
          "  wetting      E = Σ σ_wall(t_i) A_wall,i over SDF wall facets, if `sigma_wall` "
          "(nTypes,)\n"
          "               is given (a uniform wall tension is a constant — only the species\n"
          "               difference does work, which is what sets the contact angle);\n"
          "  volume       Σ e_i(V_i) for a caller-supplied e'(V_i) = `dEdV` (N,) (e.g. "
          "2(V/Vref−1)/Vref);\n"
          "  centroidal   `lloyd` · Σ ∫_cell |y − x_i|² (Lloyd/CVT; gradient 2V(x−c) drives seeds "
          "to\n"
          "               their centroids — the skewness the grid solver's two-point operators "
          "need gone);\n"
          "  roundness    `facet_tension` · Σ A_f over all interior faces.\n"
          "Returns {'interface_energy', 'wall_energy', 'lloyd_energy', 'tension_energy', 'force' "
          "(N,3) = dE/dx,\n'force_w' (N,) = dE/dw when weights are set}. Descend along −force to "
          "minimise.")
      .def_prop_ro("num_particles", &Tess::num_particles,
                   "Particle count N set by the last `build` (0 before it).")
      .def_prop_ro(
          "diagnostics", [](Tess& t) { return TessDiagnostics{&t}; }, nb::keep_alive<0, 1>(),
          "The diagnostics tier: build_report(), set_local_certificate(), set_gate(), "
          "set_profile().");

  // ---- FlowSolver -------------------------------------------------------------------------------
  nb::class_<FlowDiagnostics>(
      m, "FlowSolverDiagnostics",
      "Ablation switches of a FlowSolver (reached as `f.diagnostics`): the measured-worse "
      "alternatives kept for comparison.")
      .def(
          "set_skew_corrected", [](FlowDiagnostics& d, bool on) { d.f->set_skew_corrected(on); },
          nb::arg("on"),
          "Collocated only: the centroid-consistent constraint pair (default True; the plain pair "
          "drops\nto first order on skewed meshes — README, rung C2b).")
      .def(
          "set_wall_gradient_quadratic",
          [](FlowDiagnostics& d, bool on) { d.f->set_wall_gradient_quadratic(on); }, nb::arg("on"),
          "Wall viscous flux from the wall-anchored least-squares quadratic (default True; exact "
          "for\nPoiseuille) instead of the two-point (U_i - U_wall)/h_A (-13 % on the sphere "
          "drag).");

  nb::class_<Flow>(
      m, "FlowSolver",
      "Static Navier–Stokes solver on the face mesh of a resident (built) Tessellation (Voronoi "
      "methods plan, track C). layout='collocated' (default): peclet.flow's approximate projection "
      "with the skew-corrected adjoint constraint pair — second order on unstructured Voronoi "
      "meshes; layout='covolume': the staggered covolume scheme (exact energy conservation, first "
      "order on unstructured meshes). Walls come from the tessellation's SDF geometry (no-slip "
      "unless set_wall_velocity). SSP-RK3 with a projection per stage; GraphAMG-PCG pressure "
      "solve. The mesh is frozen at construction — build a new FlowSolver after moving the seeds. "
      "Instruments: `diagnostics`.")
      .def(nb::init<Tess&, real_t, const std::string&>(), nb::arg("tessellation"),
           nb::arg("viscosity"), nb::arg("layout") = "collocated",
           doc("FlowSolver(tessellation, viscosity, layout='collocated'): layout is one of " +
               std::string(kLayoutList) + "."))
      .def_prop_ro("num_cells", &Flow::num_cells,
                   "Number of cells of the face mesh (= the tessellation's particle count).")
      .def_prop_ro("num_faces", &Flow::num_faces,
                   "Number of faces of the face mesh: interior faces first, then the wall faces.")
      .def_prop_ro(
          "num_wall_faces", &Flow::num_wall_faces,
          "Number of SDF wall faces (the trailing block of the faces); 0 without geometry.")
      .def_prop_ro("layout", &Flow::layout,
                   "The solver layout this instance was built with: 'collocated' or 'covolume'.")
      .def("set_body_force", &Flow::set_body_force, nb::arg("force"),
           "Uniform body force per unit mass (fx, fy, fz) applied to every cell (a pressure "
           "gradient drive, gravity).")
      .def("set_stokes", &Flow::set_stokes, nb::arg("on"),
           "Drop the convective term (creeping flow).")
      .def("set_pressure_tolerance", &Flow::set_pressure_tolerance, nb::arg("tol"),
           "Relative residual at which the pressure PCG stops (default set by the solver).")
      .def("set_implicit_diffusion", &Flow::set_implicit_diffusion, nb::arg("on"),
           "Collocated only (raises on covolume): flow's semi-implicit step (explicit convection, "
           "backward-Euler viscous solve, approximate projection) — no diffusive dt limit, first "
           "order in time.")
      .def("set_wall_velocity", &Flow::set_wall_velocity, nb::arg("U"),
           "Prescribed velocity on the wall faces, (num_wall_faces, 3).")
      .def("set_velocity", &Flow::set_velocity, nb::arg("U"),
           "Initial cell velocity (num_cells, 3); projected once.")
      .def("set_dt", &Flow::set_dt, nb::arg("dt"),
           "Set the time step (suite/docs/NAMING.md 1.5); `step` uses it.")
      .def_prop_ro("dt", &Flow::dt, "The stored time step (0 until `set_dt`).")
      .def("step", &Flow::step, nb::arg("num_steps"),
           "Advance `num_steps` steps of the stored time step (`set_dt`); raises if none was set.")
      .def("get_velocities", &Flow::get_velocities,
           "Cell velocity (num_cells, 3) float64 (a copy; the covolume layout reconstructs it "
           "from the face fluxes).")
      .def("get_pressure", &Flow::get_pressure, "Cell pressure (num_cells,) float64 (a copy).")
      .def("get_volumes", &Flow::get_volumes,
           "Cell volume (num_cells,) float64 (a copy) — the face mesh's, i.e. the tessellation's.")
      .def("kinetic_energy", &Flow::kinetic_energy,
           "Total kinetic energy ½ Σ V_i |U_i|² over the cells (a device reduction).")
      .def("max_divergence", &Flow::max_divergence,
           "Max over the cells of the discrete divergence of the transporting face flux — "
           "round-off after a projection.")
      .def_prop_ro("pressure_iterations", &Flow::pressure_iterations,
                   "PCG iteration count of the last pressure solve.")
      .def_prop_ro(
          "diagnostics", [](Flow& f) { return FlowDiagnostics{&f}; }, nb::keep_alive<0, 1>(),
          "The diagnostics tier: set_skew_corrected(), set_wall_gradient_quadratic().");

  // ---- Simulation -------------------------------------------------------------------------------
  nb::class_<SimDiagnostics>(m, "SimulationDiagnostics",
                             "Performance-path switches of a Simulation (reached as "
                             "`s.diagnostics`).")
      .def(
          "set_repair", [](SimDiagnostics& d, bool on) { d.s->set_repair(on); },
          nb::arg("on") = true,
          "Opt-in (default off): use the incremental moving-point repair + reeval-published force "
          "geometry each step instead of a full rebuild. Before init().")
      .def(
          "set_profile", [](SimDiagnostics& d, bool on) { d.s->set_profile(on); },
          nb::arg("on") = true,
          "Print each cold build's timing and over-buffer report on stderr (default off).");

  nb::class_<Sim>(
      m, "Simulation",
      "Device-native compressible-Euler / Navier-Stokes Voronoi fluid simulation.\n\n"
      "Velocity-Verlet dynamics of a moving-particle Voronoi fluid: pressure forces from an\n"
      "EOS plus an optional per-particle viscous (Navier-Stokes) term, with the tessellation\n"
      "repaired each step on the device. Set the particle state, `init`, `set_dt`, then `step`;\n"
      "the state setters raise after `init` (the state is then resident on the device).\n"
      "Instruments: `diagnostics`.")
      .def(nb::init<>())
      .def("set_domain", &Sim::set_domain, nb::arg("extent"),
           nb::arg("origin") = std::array<real_t, 3>{0, 0, 0},
           nb::arg("periodic") = std::array<bool, 3>{true, true, true},
           "Set the periodic box before `init`: `extent` is the box SIZE (Lx, Ly, Lz). The "
           "suite-wide spelling (suite/docs/NAMING.md 1.1), the same call "
           "`dem.Simulation.set_domain` takes. "
           "`origin` must be (0, 0, 0) and `periodic` (True, True, True) — this engine's box is "
           "anchored at the origin and periodic on every axis; both are checked rather than "
           "ignored, so a caller who writes the suite-wide form gets an error naming the "
           "limitation.")
      .def_prop_ro("extent", &Sim::extent,
                   "The box size (Lx, Ly, Lz) — read-only; set it with `set_domain`.")
      .def("set_positions", &Sim::set_positions, nb::arg("positions"),
           "Initial particle positions (N,3) float64 (before init).")
      .def("set_velocities", &Sim::set_velocities, nb::arg("velocities"),
           "Initial particle velocities (N,3) float64 (before init; at rest if not set).")
      .def("set_masses", &Sim::set_masses, nb::arg("masses"),
           "Particle masses (N,) float64, all > 0 (before init; sets N).")
      .def("set_pressure", &Sim::set_pressure, nb::arg("pressure"),
           "Equation-of-state pressure constant (the stiffness of the barotropic EOS; before "
           "init).")
      .def("set_viscosities", &Sim::set_viscosities, nb::arg("viscosities"),
           "Per-particle shear viscosity (N,) — enables the viscous Navier-Stokes term (before "
           "init).")
      .def("set_bulk_viscosities", &Sim::set_bulk_viscosities, nb::arg("viscosities"),
           "Per-particle bulk viscosity (N,) float64 (defaults to zero if unset; before init).")
      .def(
          "set_geometry", &Sim::set_geometry, nb::arg("node_ints"), nb::arg("node_reals"),
          nb::arg("root") = 0, nb::arg("grad_h") = kSdfGradientStep,
          "SDF solid walls for the fluid (same flat node encoding as Tessellation.set_geometry).\n"
          "The cells are clipped by the solid; the EOS pressure acts on the wall facets (the wall\n"
          "pushes back). Before init().")
      .def("clear_geometry", &Sim::clear_geometry, "Drop the SDF geometry (before init()).")
      .def("init", &Sim::init,
           "Build the first tessellation and forces from the particle state set above (checks "
           "that every per-particle array has the N of set_masses).")
      .def("set_dt", &Sim::set_dt, nb::arg("dt"),
           "Set the time step. The suite-wide way to configure a stepper "
           "(suite/docs/NAMING.md 1.5) — `flow.Solver`, `dem.Simulation` and "
           "`peclet.core.amr.Flow` all "
           "take `set_dt`.")
      .def_prop_ro("dt", &Sim::dt, "The stored time step (0 until `set_dt`).")
      .def("step", &Sim::step, nb::arg("num_steps"),
           "Advance the velocity-Verlet dynamics by `num_steps` steps of the stored time step "
           "(`set_dt`); raises before init() and if no dt was set.")
      .def("get_positions", &Sim::get_positions, "Current particle positions (N,3) float64.")
      .def("get_velocities", &Sim::get_velocities, "Current particle velocities (N,3) float64.")
      .def("get_forces", &Sim::get_forces,
           "Current per-particle force (N,3) float64 — the pressure (EOS) force plus the optional\n"
           "viscous Navier-Stokes term, as used by the last velocity-Verlet kick. Useful for\n"
           "force-field analysis, equilibrium/convergence checks, and coupling.")
      .def_prop_ro("num_particles", &Sim::num_particles, "Particle count N.")
      .def("kinetic_energy", &Sim::kinetic_energy,
           "Total kinetic energy ½ Σ m_i |v_i|² (a device reduction).")
      .def("internal_energy", &Sim::internal_energy, "Total internal (EOS) energy (scalar).")
      .def_prop_ro("time", &Sim::time, "Current simulation time.")
      .def("get_volumes", &Sim::get_volumes,
           "Per-particle Voronoi cell volume (N,) float64 (a copy).")
      .def("get_neighbor_counts", &Sim::get_neighbor_counts,
           "Per-particle Voronoi neighbour (facet) count (N,) int32 (a copy).")
      .def_prop_ro(
          "diagnostics", [](Sim& s) { return SimDiagnostics{&s}; }, nb::keep_alive<0, 1>(),
          "The diagnostics tier: set_repair(), set_profile().");

#ifdef PECLET_VORO_MPI
  // ---- VoronoiHalo (distributed) ----------------------------------------------------------------
  nb::class_<VHalo>(m, "VoronoiHalo",
                    "Distributed (MPI) ghost-gather for the multi-rank Voronoi tessellation.\n\n"
                    "ORB block-decomposes a periodic box across MPI ranks and gathers, for each "
                    "rank, every seed\n"
                    "within a cutoff `rcut` of its owned block (periodic images included). The "
                    "recipe: select this\n"
                    "rank's owned seeds with `owned_mask`, `gather(...)` the owned+ghost set, "
                    "tessellate it with the\n"
                    "single-rank `Tessellation` building only the first `n_owned` cells, and keep "
                    "those cells — they\n"
                    "are bit-identical to a serial full-box tessellation (each owned cell has all "
                    "its neighbours\n"
                    "present). `rcut` must exceed the largest owned-cell interaction distance (a "
                    "few mean spacings).\n"
                    "For moving points use DistributedTessellation (the repair driver over this "
                    "halo).\n"
                    "Auto-initialises MPI (MPI_COMM_WORLD). Drive it from mpi4py.")
      .def(nb::init<std::array<long, 3>, std::array<real_t, 3>, std::array<real_t, 3>,
                    std::array<bool, 3>>(),
           nb::arg("cells"), nb::kw_only(), nb::arg("extent"),
           nb::arg("origin") = std::array<real_t, 3>{0, 0, 0},
           nb::arg("periodic") = std::array<bool, 3>{true, true, true},
           "Build the ORB decomposition of the box [origin, origin+extent) on `cells` ORB cells "
           "per axis\nwith per-axis `periodic` flags, over MPI_COMM_WORLD (the suite-wide domain "
           "quartet, as\npeclet.core.mpi.ParticleHalo).")
      .def_prop_ro("rank", &VHalo::rank, "This rank's MPI index.")
      .def_prop_ro("num_ranks", &VHalo::num_ranks, "Number of MPI ranks.")
      .def("owned_mask", &VHalo::owned_mask, nb::arg("positions"),
           "Mask (N,) int32 over the given positions (N,3): 1 where this rank owns the point, else "
           "0.")
      .def("owner_of", &VHalo::owner_of, nb::arg("point"),
           "Owning rank of a single point (x, y, z).")
      .def("gather", &VHalo::gather, nb::arg("positions"), nb::arg("gids"), nb::arg("rcut"),
           nb::arg("weights") = nb::none(),
           "Gather ghost seeds within `rcut` of this rank's owned seeds. Inputs: the owned "
           "positions\n(N,3) float64, their global ids (N,) int64, the cutoff, and optional power "
           "weights (N,)\nfloat64 (zeros if None). Returns a tuple (pos (M,3) float64, gid (M,) "
           "int64, weight (M,)\nfloat64, n_owned): rows [0,n_owned) are the owned seeds, "
           "[n_owned,M) the gathered ghosts\n(with their owners' global ids/weights).")
      .def("refresh_positions", &VHalo::refresh_positions, nb::arg("positions"),
           "Position-only halo refresh (Verlet fast path): re-forward the current owned positions\n"
           "(N,3) onto the topology of the last `gather`, returning the combined owned+ghost "
           "positions\n(M,3) in the same order as that gather (no re-decomposition / ghost "
           "re-selection).");

  // ---- DistributedTessellation (distributed repair driver)
  // ----------------------------------------
  nb::class_<DTessDiagnostics>(m, "DistributedTessellationDiagnostics",
                               "Instruments of a DistributedTessellation (reached as "
                               "`d.diagnostics`).")
      .def_prop_ro(
          "num_regathers", [](DTessDiagnostics& d) { return d.d->num_regathers(); },
          "Number of collective re-gather + cold-rebuild events since construction (establish "
          "counts as one).")
      .def(
          "set_profile", [](DTessDiagnostics& d, bool on) { d.d->set_profile(on); },
          nb::arg("on") = true,
          "Print each rank's cold-build timing and over-buffer report on stderr (default off).");

  nb::class_<DTess>(
      m, "DistributedTessellation",
      doc("Distributed (MPI) moving-point Voronoi tessellation: VoronoiHalo's ORB decomposition + "
          "ghost\ngather composed with the device incremental repair under the distributed "
          "Verlet-skin\ninvariant (peclet::voro::mpi::DistributedMovingTessellation, gated by "
          "tests/kokkos_mpi at\nnp = 1, 2, 4). Each rank owns the seeds `owned_mask` selects; "
          "`establish` gathers the ghosts\nwithin rcut and cold-builds; every `step` refreshes the "
          "ghost positions on the established\ntopology and repairs locally — until any rank's "
          "owned displacement since the last gather\nexceeds skin/2, when ALL ranks re-gather and "
          "rebuild (a collective decision). The owned\ncells [0, num_owned) equal a cold rebuild "
          "of the same combined positions to the certificate\ntolerance. rcut, skin and "
          "tolerance are fractions of the mean spacing cbrt(V/N_global)\n(defaults " +
          fmt(kDistributedRcut) + ", " + fmt(kSkin) + ", " + fmt(kCertificateTolerance) +
          "). Collective calls: establish, step. Auto-initialises MPI\n(MPI_COMM_WORLD). "
          "Instruments: `diagnostics`."))
      .def(nb::init<std::array<long, 3>, std::array<real_t, 3>, std::array<real_t, 3>,
                    std::array<bool, 3>, real_t, real_t, real_t>(),
           nb::arg("cells") =
               std::array<long, 3>{kDistributedCells, kDistributedCells, kDistributedCells},
           nb::kw_only(), nb::arg("extent"), nb::arg("origin") = std::array<real_t, 3>{0, 0, 0},
           nb::arg("periodic") = std::array<bool, 3>{true, true, true},
           nb::arg("rcut") = kDistributedRcut, nb::arg("skin") = kSkin,
           nb::arg("tolerance") = kCertificateTolerance,
           doc("DistributedTessellation(cells=(" + fmt((int)kDistributedCells) +
               ",)*3, *, extent, "
               "origin=(0,0,0), periodic=(True,)*3,\nrcut=" +
               fmt(kDistributedRcut) + ", skin=" + fmt(kSkin) +
               ", tolerance=" + fmt(kCertificateTolerance) +
               "): `cells` is the ORB granularity per axis; `extent`\nthe box size; origin and "
               "periodic are checked as in Tessellation.set_domain."))
      .def_prop_ro("rank", &DTess::rank, "This rank's MPI index.")
      .def_prop_ro("num_ranks", &DTess::num_ranks, "Number of MPI ranks.")
      .def("owned_mask", &DTess::owned_mask, nb::arg("positions"),
           "Mask (N,) int32 over the given positions (N,3): 1 where this rank owns the point.")
      .def("set_geometry", &DTess::set_geometry, nb::arg("node_ints"), nb::arg("node_reals"),
           nb::arg("root") = 0, nb::arg("grad_h") = kSdfGradientStep,
           "Replicated SDF solid (every rank passes the same scene), as Tessellation.set_geometry. "
           "Before establish().")
      .def("clear_geometry", &DTess::clear_geometry, "Drop the SDF geometry (before establish()).")
      .def("set_wall_mode", &DTess::set_wall_mode, nb::arg("mode"),
           nb::arg("skin_frac") = kWallSkin,
           doc("Wall re-gather policy, as Tessellation.set_wall_mode: mode one of " +
               std::string(kWallModeList) + ". Before establish()."))
      .def("establish", &DTess::establish, nb::arg("positions"), nb::arg("gids"),
           nb::arg("weights") = nb::none(),
           "Collective: gather the ghosts of this rank's owned seeds (positions (N,3), global ids "
           "(N,)\nint64, optional weights (N,)) and cold-build the combined tessellation. Call "
           "once, and\nagain whenever ownership changes.")
      .def("step", &DTess::step, nb::arg("positions"),
           "Collective: advance to the new owned positions (N,3, same N and ownership as "
           "establish).\nReturns the Tessellation.step stats dict plus 'regathered' (True when "
           "this step took the\nre-gather + cold-rebuild path; the repair stats are then zero).")
      .def_prop_ro("num_owned", &DTess::num_owned, "This rank's owned cell count.")
      .def_prop_ro("num_combined", &DTess::num_combined,
                   "Owned + ghost seed count of this rank's tessellation after the last gather.")
      .def("get_volumes", &DTess::get_volumes,
           "Owned-cell volumes (num_owned,) float64 (a copy), in establish() order.")
      .def("get_neighbor_counts", &DTess::get_neighbor_counts,
           "Owned-cell neighbour counts (num_owned,) int32 (a copy).")
      .def("get_wall_counts", &DTess::get_wall_counts,
           "Owned-cell resident SDF wall plane counts (num_owned,) int32 (a copy).")
      .def("get_combined_gids", &DTess::get_combined_gids,
           "Global ids (num_combined,) int64 of the owned + ghost seeds after the last gather "
           "(owned\nfirst, in establish() order).")
      .def_prop_ro(
          "diagnostics", [](DTess& d) { return DTessDiagnostics{&d}; }, nb::keep_alive<0, 1>(),
          "The diagnostics tier: num_regathers, set_profile().");
#endif
}
