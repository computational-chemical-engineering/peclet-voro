/**
 * @file voro_bindings.cpp
 * @brief Kokkos nanobind Python module `peclet.voro`.
 *
 * Drives the production device path — multicore CPU (OpenMP), or GPU (CUDA/HIP), selected by the
 * Kokkos backend the extension was built against — from Python. Two surfaces:
 *
 *  - @ref Tess "peclet.voro.Tessellation" — the bare moving-particle Voronoi tessellator: a cold build
 *    plus the incremental two-pass *repair* update (the fast per-step path for moving points),
 * exposing per-cell volumes and neighbour counts. This is the core primitive all the geometry work
 * builds on.
 *  - @ref Sim "peclet.voro.Simulation" — a device-native compressible-Euler / Navier–Stokes Voronoi
 * fluid simulation (velocity-Verlet over the tessellation) on top of that primitive.
 *
 * Particle data crosses the boundary as NumPy arrays: positions/velocities are `(N,3)` float64,
 * scalars (masses, viscosities, volumes) are `(N,)`. Arrays move through the shared `peclet::core::python`
 * bridge (core): returned arrays are backed by host buffers (no extra device copy).
 *
 * Kokkos is initialized at import and finalized via a Python `atexit` hook (with every live
 * object's Views released first — required on CUDA). Call `peclet.voro.finalize()` for deterministic
 * teardown.
 *
 * Example
 * -------
 * @code{.py}
 *   import numpy as np, peclet.voro
 *   rng = np.random.default_rng(0)
 *   pos = rng.random((100_000, 3))            # uniform points in the unit box
 *   t = peclet.voro.Tessellation()
 *   t.set_box((1.0, 1.0, 1.0))
 *   t.build(pos)                              # cold tessellation
 *   vol = t.volumes()                         # (N,) cell volumes; sum ~= box volume
 *   for _ in range(50):                       # move + repair each step (faster than rebuilding)
 *       pos = (pos + 1e-4 * rng.standard_normal(pos.shape)) % 1.0
 *       stats = t.step(pos)                   # {'flagged','pass1','pass2','rebuilt','fell_back'}
 *   nbr = t.neighbor_counts()                 # (N,) Voronoi neighbours per cell
 * @endcode
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <optional>
#include <set>
#include <type_traits>
#include <variant>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/python/ndarray_interop.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/energy/interface.hpp"
#include "peclet/voro/energy/lloyd.hpp"
#include "peclet/voro/energy/tension.hpp"
#include "peclet/voro/energy/volume.hpp"
#include "peclet/voro/energy/wall.hpp"
#include "peclet/voro/mesh_optimizer.hpp"
#include "peclet/voro/physics/simulation.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/topology_store.hpp"

#ifdef PECLET_VORO_MPI
#include <mpi.h>

#include <nanobind/stl/tuple.h>

#include "peclet/voro/mpi/voronoi_halo.hpp"
#endif

namespace nb = nanobind;
using real_t = double;
using DView = Kokkos::View<real_t*, peclet::core::MemSpace>;

namespace {

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

// ---- pore-space meshing helpers (SDF-walled interstitial Voronoi + geometry export) --------------
using PoreCell = peclet::voro::ConvexCell<real_t, 128, 256>;

// Build a periodic union-of-balls SDF from (M,3) centres + (M,) radii; the Views must outlive its use.
peclet::voro::SdfSpheres<real_t> makeSpheresSdf(nb::ndarray<real_t, nb::c_contig> centres,
                                                nb::ndarray<real_t, nb::c_contig> radii, real_t L,
                                                DView& cenHold, DView& radHold) {
  const int M = (int)radii.shape(0);
  auto cflat = flatten3(centres);
  cenHold = DView("sph.cen", 3 * M);
  radHold = DView("sph.rad", M);
  Kokkos::deep_copy(cenHold, Kokkos::View<const real_t*, Kokkos::HostSpace>(cflat.data(), 3 * M));
  Kokkos::deep_copy(radHold, Kokkos::View<const real_t*, Kokkos::HostSpace>(radii.data(), M));
  return peclet::voro::SdfSpheres<real_t>{cenHold, radHold, M, L};
}

// Ordered alive-triangle indices for face k (watertight by triangle index). Port of the one in
// examples/packed_bed_voronoi/pore_mesh_stages.cpp.
int faceOrderedIdx(const PoreCell& c, int k, int out[PoreCell::MAXFV]) {
  int m = 0;
  real_t fx[PoreCell::MAXFV], fy[PoreCell::MAXFV], fz[PoreCell::MAXFV];
  for (int t = 0; t < c.nt; ++t) {
    if (!c.alive[t]) continue;
    if (c.t0[t] != k && c.t1[t] != k && c.t2[t] != k) continue;
    if (m < PoreCell::MAXFV) {
      out[m] = t;
      fx[m] = c.vx[t];
      fy[m] = c.vy[t];
      fz[m] = c.vz[t];
      ++m;
    }
  }
  if (m < 3) return m;
  const real_t nx = c.n[k][0], ny = c.n[k][1], nz = c.n[k][2];
  const real_t nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (nlen == real_t(0)) return 0;
  const real_t un[3] = {nx / nlen, ny / nlen, nz / nlen};
  real_t e1[3];
  if (std::fabs(un[0]) <= std::fabs(un[1]) && std::fabs(un[0]) <= std::fabs(un[2])) {
    e1[0] = 0; e1[1] = -un[2]; e1[2] = un[1];
  } else if (std::fabs(un[1]) <= std::fabs(un[2])) {
    e1[0] = -un[2]; e1[1] = 0; e1[2] = un[0];
  } else {
    e1[0] = -un[1]; e1[1] = un[0]; e1[2] = 0;
  }
  const real_t e1l = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  e1[0] /= e1l; e1[1] /= e1l; e1[2] /= e1l;
  const real_t e2[3] = {un[1] * e1[2] - un[2] * e1[1], un[2] * e1[0] - un[0] * e1[2],
                        un[0] * e1[1] - un[1] * e1[0]};
  real_t cx = 0, cy = 0, cz = 0;
  for (int i = 0; i < m; ++i) { cx += fx[i]; cy += fy[i]; cz += fz[i]; }
  cx /= m; cy /= m; cz /= m;
  real_t ang[PoreCell::MAXFV];
  for (int i = 0; i < m; ++i) {
    const real_t dx = fx[i] - cx, dy = fy[i] - cy, dz = fz[i] - cz;
    const real_t pu = dx * e1[0] + dy * e1[1] + dz * e1[2];
    const real_t pv = dx * e2[0] + dy * e2[1] + dz * e2[2];
    const real_t s = std::fabs(pu) + std::fabs(pv);
    const real_t tt = (s > real_t(0)) ? pv / s : real_t(0);
    ang[i] = (pu < real_t(0)) ? (real_t(2) - tt) : (pv < real_t(0) ? real_t(4) + tt : tt);
  }
  for (int i = 1; i < m; ++i) {
    const real_t ka = ang[i];
    const int ki = out[i];
    int j = i - 1;
    while (j >= 0 && ang[j] > ka) { ang[j + 1] = ang[j]; out[j + 1] = out[j]; --j; }
    ang[j + 1] = ka; out[j + 1] = ki;
  }
  return m;
}

// Reconstructs the SDF-clipped interstitial Voronoi cell of any seed. Builds a periodic counting-sort
// grid once; each build() gathers the ~80 nearest seeds via an O(1) Chebyshev shell walk (stop once the
// 80th nearest is provably found), builds the ConvexCell against a far box, and clips it to the SDF.
// Shared by sdf_voronoi_cells (polyhedra) and sdf_voronoi_section (plane cross-section).
struct PoreReconstructor {
  const real_t* seed;
  int N;
  real_t L, Lh, big, hbin;
  int nb;
  peclet::voro::SdfSpheres<real_t> sdf;
  std::vector<int> binStart, binItem;
  mutable std::vector<std::pair<real_t, int>> ord;

  int binOf(real_t x) const {
    int b = (int)std::floor(x / hbin) % nb;
    return b < 0 ? b + nb : b;
  }
  int cellOf(int i) const {
    return binOf(seed[3 * i]) + nb * (binOf(seed[3 * i + 1]) + nb * binOf(seed[3 * i + 2]));
  }
  PoreReconstructor(const std::vector<real_t>& s, real_t L_, peclet::voro::SdfSpheres<real_t> sdf_)
      : seed(s.data()), N((int)(s.size() / 3)), L(L_), Lh(0.5 * L_), big(4 * L_), sdf(sdf_) {
    nb = std::max(1, std::min((int)std::cbrt((double)N / 2.0 + 1.0), 96));
    hbin = L / nb;
    const int nbin = nb * nb * nb;
    binStart.assign(nbin + 1, 0);
    for (int i = 0; i < N; ++i) ++binStart[cellOf(i) + 1];
    for (int b = 0; b < nbin; ++b) binStart[b + 1] += binStart[b];
    binItem.resize(N);
    std::vector<int> cur(binStart.begin(), binStart.end());
    for (int i = 0; i < N; ++i) binItem[cur[cellOf(i)]++] = i;
  }
  bool build(int i, PoreCell& c) const {
    const real_t sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
    ord.clear();
    const int Kwant = 80, bx = binOf(sx), by = binOf(sy), bz = binOf(sz);
    for (int R = 0; R <= nb; ++R) {
      for (int dz2 = -R; dz2 <= R; ++dz2)
        for (int dy2 = -R; dy2 <= R; ++dy2)
          for (int dx2 = -R; dx2 <= R; ++dx2) {
            int cheb = std::abs(dx2);
            cheb = std::max(cheb, std::abs(dy2));
            cheb = std::max(cheb, std::abs(dz2));
            if (cheb != R) continue;
            const int gx = ((bx + dx2) % nb + nb) % nb, gy = ((by + dy2) % nb + nb) % nb,
                      gz = ((bz + dz2) % nb + nb) % nb;
            const int b = gx + nb * (gy + nb * gz);
            for (int t = binStart[b]; t < binStart[b + 1]; ++t) {
              const int j = binItem[t];
              if (j == i) continue;
              real_t dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
              dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
              dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
              dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
              ord.emplace_back(dx * dx + dy * dy + dz * dz, j);
            }
          }
      if ((int)ord.size() >= Kwant) {
        std::nth_element(ord.begin(), ord.begin() + (Kwant - 1), ord.end());
        const real_t rh = (real_t)R * hbin;
        if (rh * rh >= ord[Kwant - 1].first) break;
      }
    }
    std::sort(ord.begin(), ord.end());
    const int M = std::min((int)ord.size(), 80);
    real_t rx[80], ry[80], rz[80];
    int ids[80];
    for (int k = 0; k < M; ++k) {
      const int j = ord[k].second;
      real_t dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
      dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
      dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
      dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
      rx[k] = dx; ry[k] = dy; rz[k] = dz; ids[k] = j;
    }
    const real_t Lbig[3] = {big, big, big};
    peclet::voro::buildConvexCell(c, Lbig, rx, ry, rz, ids, M);
    const real_t seedW[3] = {sx, sy, sz};
    peclet::voro::clipCellAgainstSdf<real_t, 128, 256, false>(c, seedW, sdf);
    return !(c.empty() || c.overflow);
  }
};

// --------------------------------------------------------------------------------------------------
// Tessellation: the bare moving-particle Voronoi tessellator (cold build + incremental repair).
// --------------------------------------------------------------------------------------------------
// ---- SDF geometry from Python (rung A0)
// ----------------------------------------------------------
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

// --------------------------------------------------------------------------------------------------
// Tessellation: the bare moving-point (power-)Voronoi tessellator, optionally SDF-clipped.
// The engine is chosen at build() from what was set: {Voronoi, Power} x {no geometry, SdfScene}.
// --------------------------------------------------------------------------------------------------
template <bool W, class S>
using MT = peclet::voro::MovingTessellation<real_t, 64, 112, W, S>;
using MtVariant =
    std::variant<MT<false, NoSdfT>, MT<true, NoSdfT>, MT<false, SceneT>, MT<true, SceneT>>;

class Tess {
 public:
  Tess() { live().insert(this); }
  ~Tess() { live().erase(this); }

  void set_box(std::array<real_t, 3> L) { L_ = L; }
  void set_tolerance(real_t frac) { tolFrac_ = frac; }
  void set_local_certificate(bool on) { localCert_ = on; }
  void set_gate(bool on) { useGate_ = on; }
  void set_geometry(nb::ndarray<int, nb::c_contig> node_ints,
                    nb::ndarray<real_t, nb::c_contig> node_reals, int root, real_t grad_h) {
    scene_ = makeSceneHolder(node_ints, node_reals, root, grad_h);
  }
  void clear_geometry() { scene_.clear(); }
  void set_wall_mode(bool exact, real_t skin_frac) {
    wallExact_ = exact;
    wallSkinFrac_ = skin_frac;
  }
  void set_weights(nb::ndarray<real_t, nb::c_contig> w) {
    wHost_ = flatten1(w);
    weighted_ = true;
    wDirty_ = true;
  }
  void clear_weights() {
    wHost_.clear();
    weighted_ = false;
    wDirty_ = false;
  }

  // A2a: validity diagnostics of the last cold build.
  nb::dict build_report() {
    auto r = std::visit([](auto& mt) { return mt.report(); }, mt_);
    nb::dict d;
    d["buried"] = r.buried;
    d["reach_exceeded"] = r.reachExceeded;
    d["empty"] = r.empty;
    d["overflow"] = r.overflow;
    d["incomplete"] = r.incomplete;
    return d;
  }

  // Cold build: (re)allocate the resident tessellation for N points and build it from scratch.
  void build(nb::ndarray<real_t, nb::c_contig> a, bool strict) {
    std::vector<real_t> p = flatten3(a);
    N_ = static_cast<int>(p.size() / 3);
    const double boxVol = static_cast<double>(L_[0]) * L_[1] * L_[2];
    const real_t spacing = static_cast<real_t>(std::cbrt(boxVol / (N_ > 0 ? N_ : 1)));
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
          if constexpr (T::kHasSdf) {
            mt.sdf = scene_.scene;
            mt.wallExact = wallExact_;
            mt.wallSkin = wallSkinFrac_ * spacing;
          }
          if constexpr (std::is_same_v<typename T::PlanePolicy, peclet::voro::Power>)
            mt.setWeights(weight_);
          mt.alloc(N_, L_.data(), tolFrac_ * spacing, real_t(0.25) * spacing, 4, N_);
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
          " overflowed cell(s). See voro/docs/power_large_weights_plan.md.";
      if (strict)
        throw std::runtime_error(msg);
      nb::module_::import_("warnings").attr("warn")(msg);
    }
  }

  // Incremental repair: update the resident tessellation to new positions (same N) without a full
  // rebuild. Returns the per-step work stats. Positions must be the same count as the last build().
  nb::dict step(nb::ndarray<real_t, nb::c_contig> a) {
    std::vector<real_t> p = flatten3(a);
    if (static_cast<int>(p.size() / 3) != N_)
      throw std::runtime_error("step(): particle count differs from build(); call build() first");
    pos_ = peclet::core::toDevice<real_t>(p, "pos");
    if (weighted_ && wDirty_) {  // weights changed since the last build/step: refresh in place
      if (static_cast<int>(wHost_.size()) != N_)
        throw std::runtime_error("step(): weights (N,) must match the particle count");
      Kokkos::deep_copy(
          weight_, Kokkos::View<const real_t*, Kokkos::HostSpace>(wHost_.data(), wHost_.size()));
      wDirty_ = false;
    }
    auto st = std::visit([&](auto& mt) { return mt.step(pos_); }, mt_);
    nb::dict d;
    d["flagged"] = st.pass1Raw;  // cells the certificate flagged
    d["pass1"] = st.pass1;       // cells gathered in Pass 1
    d["pass2"] = st.pass2;       // cells gathered in Pass 2
    d["rebuilt"] =
        (st.route == peclet::voro::RepairStats::kRebuildGate);  // gate routed to a full rebuild
    d["fell_back"] = st.fellBack;                              // verify failed -> cold rebuild
    d["extra"] = st.extra;                  // cells gathered across the verify extra-passes
    d["surgical"] = st.surgical;            // Pass-1 cells repaired surgically (no grid gather)
    d["verify_passes"] = st.verifyPasses;   // number of verify iterations run
    d["wall_flagged"] = st.wallFlagged;     // cells flagged by the SDF boundary watch
    return d;
  }

  nb::ndarray<nb::numpy, real_t> volumes() {
    const std::size_t N = static_cast<std::size_t>(N_);
    DView vol = std::visit([](auto& mt) { return mt.vol; }, mt_);
    auto v = Kokkos::subview(vol, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(v), {N}, {1});
  }

  // Per-cell Voronoi neighbour (= face) count, recomputed from the resident topology store.
  nb::ndarray<nb::numpy, int> neighbor_counts() {
    using Cell = peclet::voro::ConvexCell<real_t, 64, 112, false>;
    const int N = N_;
    Kokkos::View<int*, peclet::core::MemSpace> cnt("nbr", N);
    auto st = std::visit([](auto& mt) { return mt.store; }, mt_);
    auto C = cnt;
    const real_t Lx = L_[0], Ly = L_[1], Lz = L_[2];
    Kokkos::parallel_for(
        "peclet.voro.nbrcount", Kokkos::RangePolicy<peclet::core::ExecSpace>(0, N), KOKKOS_LAMBDA(int i) {
          Cell c;
          st.load(i, c, Lx, Ly, Lz);
          C(i) = c.countFaces();
        });
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(cnt), {static_cast<std::size_t>(N)}, {1});
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
          auto view = peclet::voro::reevalPublish<real_t, 64, 112>(
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

  // Per-cell number of resident SDF wall planes (0 everywhere without geometry).
  nb::ndarray<nb::numpy, int> wall_counts() {
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

  void release() {
    mt_.template emplace<MT<false, NoSdfT>>();
    pos_ = DView{};
    weight_ = DView{};
    scene_.clear();
    N_ = 0;
  }
  static std::set<Tess*>& live() {
    static std::set<Tess*> s;
    return s;
  }
  static void releaseAll() {
    for (Tess* t : live())
      t->release();
  }

 private:
  std::array<real_t, 3> L_{1, 1, 1};
  real_t tolFrac_ = 1e-4;  // certificate tolerance as a fraction of the mean spacing
  bool localCert_ = true, useGate_ = true;
  bool wallExact_ = true;
  real_t wallSkinFrac_ = 0;
  bool weighted_ = false, wDirty_ = false;
  std::vector<real_t> wHost_;
  int N_ = 0;
  DView pos_, weight_;
  SceneHolder scene_;
  MtVariant mt_;
};

// --------------------------------------------------------------------------------------------------
// Simulation: device-native compressible-Euler / Navier-Stokes Voronoi fluid dynamics.
// --------------------------------------------------------------------------------------------------
class Sim {
 public:
  Sim() { live().insert(this); }
  ~Sim() { live().erase(this); }

  // Drop all Kokkos Views (so they free BEFORE Kokkos::finalize at shutdown).
  void release() {
    sim_.template emplace<EE<NoSdfT>>();
    dmass_ = DView{};
    scene_.clear();
    pos_.clear();
    vel_.clear();
    mass_.clear();
    visc_.clear();
    bulk_.clear();
  }
  static std::set<Sim*>& live() {
    static std::set<Sim*> s;
    return s;
  }
  static void releaseAll() {
    for (Sim* d : live())
      d->release();
  }

  void set_box(std::array<real_t, 3> L) { L_ = L; }
  void set_positions(nb::ndarray<real_t, nb::c_contig> a) { pos_ = flatten3(a); }
  void set_velocities(nb::ndarray<real_t, nb::c_contig> a) { vel_ = flatten3(a); }
  void set_masses(nb::ndarray<real_t, nb::c_contig> a) { mass_ = flatten1(a); }
  void set_pressure(real_t p) { pressEq_ = p; }
  void set_viscosities(nb::ndarray<real_t, nb::c_contig> a) { visc_ = flatten1(a); }
  void set_bulk_viscosities(nb::ndarray<real_t, nb::c_contig> a) { bulk_ = flatten1(a); }
  // Opt-in incremental-repair path (E1 scaffolding, default off). Set before init().
  void set_repair(bool on) { repair_ = on; }
  void set_geometry(nb::ndarray<int, nb::c_contig> node_ints,
                    nb::ndarray<real_t, nb::c_contig> node_reals, int root, real_t grad_h) {
    scene_ = makeSceneHolder(node_ints, node_reals, root, grad_h);
  }
  void clear_geometry() { scene_.clear(); }

  void init() {
    const int N = static_cast<int>(mass_.size());
    std::vector<real_t> invm(N);
    for (int i = 0; i < N; ++i)
      invm[i] = real_t(1) / mass_[i];
    if (scene_.set)
      sim_.template emplace<EE<SceneT>>();
    else
      sim_.template emplace<EE<NoSdfT>>();
    dmass_ =
        peclet::core::toDevice<real_t>(mass_, "mass");  // resident; kinetic-energy reads it each call (E4b)
    std::visit(
        [&](auto& s) {
          using T = std::decay_t<decltype(s)>;
          s.setRepair(repair_);
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
  }

  void step(int nsteps, real_t dt) {
    std::visit([&](auto& s) { s.step(nsteps, dt); }, sim_);
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
  real_t get_kinetic_energy() {
    return std::visit([&](auto& s) { return s.kineticEnergy(dmass_); }, sim_);
  }
  real_t get_internal_energy() {
    return std::visit([](auto& s) { return s.internalEnergy(); }, sim_);
  }
  real_t get_time() {
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

  nb::ndarray<nb::numpy, int> get_num_neighbors() {
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

  // Flat (3N,) host-or-device view -> (N,3) float64 numpy array (single D2H, no host loop — S2a).
  static nb::ndarray<nb::numpy, real_t> from3(const DView& d) {
    const std::size_t N = static_cast<std::size_t>(d.extent(0)) / 3;
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(d), {N, std::size_t(3)}, {3, 1});
  }

  std::array<real_t, 3> L_{1, 1, 1};
  real_t pressEq_ = 0;
  bool repair_ = false;
  std::vector<real_t> pos_, vel_, mass_, visc_, bulk_;
  DView dmass_;  // device-resident masses, uploaded once in init() (E4b)
  SceneHolder scene_;
  EeVariant sim_;
};

#ifdef PECLET_VORO_MPI
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
  using Vec3 = std::array<real_t, 3>;

  VHalo(std::array<real_t, 3> origin, std::array<real_t, 3> size, std::array<long, 3> gsize,
        std::array<bool, 3> periodic) {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) {
      int argc = 0;
      char** argv = nullptr;
      MPI_Init(&argc, &argv);
    }
    halo_.init(origin, size, gsize, periodic, MPI_COMM_WORLD);
  }

  int rank() const { return halo_.rank(); }
  int size() const { return halo_.size(); }

  // Per-point mask (N,) int32: 1 if this rank owns the point, else 0.
  nb::ndarray<nb::numpy, int32_t> owned_mask(nb::ndarray<real_t, nb::c_contig> a) {
    if (a.ndim() != 2 || a.shape(1) != 3)
      throw std::runtime_error("owned_mask: expected an (N,3) array");
    const std::size_t N = a.shape(0);
    const real_t* p = a.data();
    const int r = halo_.rank();
    std::vector<int32_t> m(N);
    for (std::size_t i = 0; i < N; ++i)
      m[i] = (halo_.ownerOf(Vec3{p[3 * i], p[3 * i + 1], p[3 * i + 2]}) == r) ? 1 : 0;
    return peclet::core::python::vector_to_ndarray(std::move(m), {N}, {1});
  }

  // Owning rank of a single point.
  int owner_of(real_t x, real_t y, real_t z) { return halo_.ownerOf(Vec3{x, y, z}); }

  // Gather ghost seeds within rcut. Returns (pos (M,3), gid (M,), weight (M,), n_owned); the first
  // n_owned rows are this rank's owned seeds, the rest are gathered ghosts (periodic images incl.).
  nb::tuple gather(nb::ndarray<real_t, nb::c_contig> pos, nb::ndarray<int64_t, nb::c_contig> gid,
                   nb::ndarray<real_t, nb::c_contig> weight, double rcut) {
    if (pos.ndim() != 2 || pos.shape(1) != 3)
      throw std::runtime_error("gather: positions must be (N,3)");
    const std::size_t N = pos.shape(0);
    if (gid.shape(0) != N || weight.shape(0) != N)
      throw std::runtime_error("gather: gid/weight length must match positions");
    const real_t* pp = pos.data();
    const int64_t* gp = gid.data();
    const real_t* wp = weight.data();
    std::vector<Vec3> ownedPos(N);
    std::vector<long> ownedGid(N);
    std::vector<real_t> ownedW(N);
    for (std::size_t i = 0; i < N; ++i) {
      ownedPos[i] = Vec3{pp[3 * i], pp[3 * i + 1], pp[3 * i + 2]};
      ownedGid[i] = static_cast<long>(gp[i]);
      ownedW[i] = wp[i];
    }
    auto g = halo_.gather(ownedPos, ownedGid, ownedW, rcut);
    const std::size_t M = g.pos.size();
    std::vector<real_t> op(3 * M);
    std::vector<int64_t> og(M);
    std::vector<real_t> ow(M);
    for (std::size_t i = 0; i < M; ++i) {
      op[3 * i] = g.pos[i][0];
      op[3 * i + 1] = g.pos[i][1];
      op[3 * i + 2] = g.pos[i][2];
      og[i] = static_cast<int64_t>(g.gid[i]);
      ow[i] = g.weight[i];
    }
    return nb::make_tuple(
        peclet::core::python::vector_to_ndarray(std::move(op), {M, std::size_t(3)}, {3, 1}),
        peclet::core::python::vector_to_ndarray(std::move(og), {M}, {1}),
        peclet::core::python::vector_to_ndarray(std::move(ow), {M}, {1}), g.nOwned);
  }

  // Position-only halo refresh (Verlet fast path): re-forward the CURRENT owned positions onto the
  // topology of the last gather(); returns the combined owned+ghost positions (M,3) in that order.
  nb::ndarray<nb::numpy, real_t> refresh_positions(nb::ndarray<real_t, nb::c_contig> pos) {
    if (pos.ndim() != 2 || pos.shape(1) != 3)
      throw std::runtime_error("refresh_positions: positions must be (N,3)");
    const std::size_t N = pos.shape(0);
    const real_t* pp = pos.data();
    std::vector<Vec3> ownedPos(N), out;
    for (std::size_t i = 0; i < N; ++i)
      ownedPos[i] = Vec3{pp[3 * i], pp[3 * i + 1], pp[3 * i + 2]};
    halo_.refreshPositions(ownedPos, out);
    const std::size_t M = out.size();
    std::vector<real_t> op(3 * M);
    for (std::size_t i = 0; i < M; ++i) {
      op[3 * i] = out[i][0];
      op[3 * i + 1] = out[i][1];
      op[3 * i + 2] = out[i][2];
    }
    return peclet::core::python::vector_to_ndarray(std::move(op), {M, std::size_t(3)}, {3, 1});
  }

 private:
  peclet::voro::mpi::VoronoiHalo<real_t> halo_;
};
#endif  // PECLET_VORO_MPI

}  // namespace

NB_MODULE(_voro, m) {
  m.attr("__doc__") =
      "peclet.voro (device/Kokkos): moving-particle Voronoi dynamics on the device path.\n\n"
      "Classes: Tessellation (bare cold build + incremental repair, volumes, neighbour counts);\n"
      "Simulation (compressible-Euler / Navier-Stokes Voronoi fluid). Arrays are NumPy: "
      "positions/\n"
      "velocities (N,3) float64, scalars (N,). The backend (Serial/OpenMP/CUDA) is fixed at build\n"
      "time; see peclet.voro.execution_space.";
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  // Teardown order matters on CUDA: releaseAll() drops every live object's Views FIRST (so none
  // outlive finalize -> no "deallocated after Kokkos::finalize"), THEN Kokkos::finalize() runs from
  // a Python atexit hook while the CUDA driver is still up (so no cudaErrorCudartUnloading). Doing
  // only one of the two aborts on CUDA. Returned arrays are backed by host std::vectors (no device
  // Views), so they need no special handling.
  auto shutdown = []() {
    Tess::releaseAll();
    Sim::releaseAll();
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  };
  m.def("finalize", shutdown,
        "Release every live Tessellation/Simulation and finalize Kokkos (deterministic teardown; "
        "also "
        "run automatically at interpreter exit).");
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());
  nb::module_::import_("atexit").attr("register")(nb::cpp_function(shutdown));

  // ---- mesh optimiser ---------------------------------------------------------------------------
  m.def(
      "optimize_volume_mesh",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> vset_in,
         real_t L, int sw, int max_newton, real_t tol, int cg_iters, bool use_weights,
         bool colored_gs) {
        auto pos = flatten3(pos_in);
        auto vset = flatten1(vset_in);
        const int N = (int)vset.size();
        const real_t Larr[3] = {L, L, L};
        const auto prec =
            colored_gs ? peclet::voro::Precond::ColoredGS : peclet::voro::Precond::Jacobi;
        peclet::voro::OtResult R;
        std::vector<real_t> w;
        if (use_weights) {
          w.assign(N, 0.0);
          R = peclet::voro::meshVolumeOptimize<real_t, true>(pos, w, vset, Larr, N, sw,
                                                             peclet::voro::NoSdf{}, max_newton, tol,
                                                             cg_iters, prec, false);
        } else {
          std::vector<real_t> noW;
          R = peclet::voro::meshVolumeOptimize<real_t, false>(pos, noW, vset, Larr, N, sw,
                                                              peclet::voro::NoSdf{}, max_newton, tol,
                                                              cg_iters, prec, false);
        }
        nb::dict d;
        d["positions"] = peclet::core::python::vector_to_ndarray(
            std::move(pos), {static_cast<std::size_t>(N), 3}, {3, 1});
        if (use_weights)
          d["weights"] = peclet::core::python::vector_to_ndarray(
              std::move(w), {static_cast<std::size_t>(N)}, {1});
        d["iters"] = R.iters;
        d["max_vol_err"] = R.maxVolErr;
        d["mean_vol_err"] = R.meanVolErr;
        d["converged"] = R.converged;
        d["n_empty"] = R.nEmpty;
        return d;
      },
      nb::arg("positions"), nb::arg("vset"), nb::arg("L") = 1.0, nb::arg("sw") = 5,
      nb::arg("max_newton") = 60, nb::arg("tol") = 1e-9, nb::arg("cg_iters") = 300,
      nb::arg("use_weights") = false, nb::arg("colored_gs") = false,
      "Move seeds (N,3) — and optionally the power weights — to minimise Σ(V_i − vset_i)² by damped\n"
      "Gauss-Newton (Newton–Raphson + CG with a Jacobi or colored-Gauss-Seidel preconditioner).\n"
      "vset (N,) are the target cell volumes (renormalised to the box volume). Returns a dict with\n"
      "the updated 'positions' (and 'weights' if use_weights), plus iters/max_vol_err/converged.\n"
      "Pure Voronoi (use_weights=False) reaches equal/graded volumes well; weights add fuller volume\n"
      "control but are limited by the periodic tessellation's ~1% min-image floor.");

  // ---- pore-space (SDF-walled) mesh optimiser + geometry export ---------------------------------
  m.def(
      "optimize_pore_mesh",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> vref_in,
         nb::ndarray<real_t, nb::c_contig> sph_c, nb::ndarray<real_t, nb::c_contig> sph_r, real_t L,
         int sw, int max_iter, real_t tol, int cg_iters, const std::string& method, real_t mu_barrier,
         bool free_energy) {
        auto pos = flatten3(pos_in);
        auto vref = flatten1(vref_in);
        const int N = (int)vref.size();
        const real_t Larr[3] = {L, L, L};
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        peclet::voro::Precond prec = peclet::voro::Precond::GraphAMG;
        if (method == "steepest") prec = peclet::voro::Precond::SteepestDescent;
        else if (method == "jacobi") prec = peclet::voro::Precond::Jacobi;
        else if (method == "colored_gs") prec = peclet::voro::Precond::ColoredGS;
        std::vector<real_t> noW;
        auto R = peclet::voro::meshVolumeOptimize<real_t, false, peclet::voro::SdfSpheres<real_t>>(
            pos, noW, vref, Larr, N, sw, sdf, max_iter, tol, cg_iters, prec, false, mu_barrier,
            (real_t)0.7, free_energy);
        nb::dict d;
        d["positions"] = peclet::core::python::vector_to_ndarray(
            std::move(pos), {static_cast<std::size_t>(N), 3}, {3, 1});
        d["iters"] = R.iters;
        d["max_vol_err"] = R.maxVolErr;
        d["converged"] = R.converged;
        d["n_empty"] = R.nEmpty;
        return d;
      },
      nb::arg("positions"), nb::arg("vref"), nb::arg("sphere_centres"), nb::arg("sphere_radii"),
      nb::arg("L"), nb::arg("sw") = 6, nb::arg("max_iter") = 80, nb::arg("tol") = 1e-9,
      nb::arg("cg_iters") = 400, nb::arg("method") = "graphamg", nb::arg("mu_barrier") = 0.0,
      nb::arg("free_energy") = false,
      "Relax interstitial seeds (N,3) so their SDF-clipped Voronoi cell volumes approach the per-cell\n"
      "targets vref (N,), with the sphere packing (sphere_centres (M,3), sphere_radii (M,)) as periodic\n"
      "walls. method: 'graphamg'|'jacobi'|'colored_gs' (Gauss-Newton CG) or 'steepest' (descent).\n"
      "free_energy=True uses E=-Σ V_ref·log V (pressure V_ref/V, resists collapse); mu_barrier>0 adds a\n"
      "log-barrier. EXPERIMENTAL (pore-space meshing; see the pore-mesh-voronoi example).");

  m.def(
      "sdf_voronoi_cells",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> sph_c,
         nb::ndarray<real_t, nb::c_contig> sph_r, real_t L) {
        auto seed = flatten3(pos_in);
        const int N = (int)(seed.size() / 3);
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        // Reconstruct each interstitial cell (periodic min-image neighbours + SDF clip) and pack the
        // clipped polyhedra as flat arrays (VTK_POLYHEDRON layout) for host-side slicing/plotting.
        std::vector<real_t> px, py, pz, vol;
        std::vector<int64_t> faces, faceOff(1, 0);
        std::vector<int32_t> boundary, cellSeed;
        PoreReconstructor rec(seed, L, sdf);
        for (int i = 0; i < N; ++i) {
          const real_t sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
          PoreCell c;
          if (!rec.build(i, c)) continue;
          const int64_t base = (int64_t)px.size();
          std::vector<int> triToPt(c.nt, -1);
          int np = 0;
          for (int t = 0; t < c.nt; ++t) {
            if (!c.alive[t]) continue;
            triToPt[t] = np++;
            px.push_back(sx + c.vx[t]); py.push_back(sy + c.vy[t]); pz.push_back(sz + c.vz[t]);
          }
          if (np < 4) continue;
          std::vector<std::vector<int64_t>> cellFaces;
          bool wall = false;
          for (int k = 0; k < c.np; ++k) {
            int fidx[PoreCell::MAXFV];
            const int m = faceOrderedIdx(c, k, fidx);
            if (m < 3) continue;
            std::vector<int64_t> face;
            for (int q = 0; q < m; ++q)
              if (triToPt[fidx[q]] >= 0) face.push_back(base + triToPt[fidx[q]]);
            if ((int)face.size() >= 3) {
              cellFaces.push_back(std::move(face));
              if (c.pnbr[k] == peclet::voro::kBoundaryFacet) wall = true;
            }
          }
          if (cellFaces.size() < 4) continue;
          faces.push_back((int64_t)cellFaces.size());
          for (auto& f : cellFaces) {
            faces.push_back((int64_t)f.size());
            for (int64_t id : f) faces.push_back(id);
          }
          faceOff.push_back((int64_t)faces.size());
          vol.push_back(c.volumePerVertex());
          boundary.push_back(wall ? 1 : 0);
          cellSeed.push_back(i);
        }
        const std::size_t nPts = px.size(), nCells = vol.size();
        std::vector<real_t> pts(3 * nPts);
        for (std::size_t p = 0; p < nPts; ++p) { pts[3 * p] = px[p]; pts[3 * p + 1] = py[p]; pts[3 * p + 2] = pz[p]; }
        nb::dict d;
        d["points"] = peclet::core::python::vector_to_ndarray(std::move(pts), {nPts, 3}, {3, 1});
        d["faces"] = peclet::core::python::vector_to_ndarray(std::move(faces), {faces.size()}, {1});
        d["face_offsets"] =
            peclet::core::python::vector_to_ndarray(std::move(faceOff), {faceOff.size()}, {1});
        d["volume"] = peclet::core::python::vector_to_ndarray(std::move(vol), {nCells}, {1});
        d["boundary"] = peclet::core::python::vector_to_ndarray(std::move(boundary), {nCells}, {1});
        d["seed"] = peclet::core::python::vector_to_ndarray(std::move(cellSeed), {nCells}, {1});
        return d;
      },
      nb::arg("positions"), nb::arg("sphere_centres"), nb::arg("sphere_radii"), nb::arg("L"),
      "Reconstruct the SDF-clipped interstitial Voronoi cells and return their polyhedra as flat\n"
      "arrays (VTK_POLYHEDRON layout): 'points' (Np,3), 'faces' + 'face_offsets' (per-cell face lists,\n"
      "global point ids), 'volume' (Nc,), 'boundary' (Nc, 1 where the cell touches a sphere wall).");

  m.def(
      "sdf_voronoi_section",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> sph_c,
         nb::ndarray<real_t, nb::c_contig> sph_r, real_t L,
         std::array<real_t, 3> origin, std::array<real_t, 3> normal) {
        auto seed = flatten3(pos_in);
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        PoreReconstructor rec(seed, L, sdf);
        // Cut every reconstructed cell by the plane {x : (x-origin)·normal = 0} and collect the
        // convex section polygons (robust: ConvexCell::sectionPolygon works from the dual edges, so it
        // tiles the cross-section exactly). Vertices returned in WORLD 3-D (all on the plane).
        std::vector<real_t> verts, vol;
        std::vector<int64_t> off(1, 0);
        std::vector<int32_t> cellSeed;
        real_t spx[PoreCell::MAXSV], spy[PoreCell::MAXSV], spz[PoreCell::MAXSV];
        for (int i = 0; i < rec.N; ++i) {
          const real_t sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
          PoreCell c;
          if (!rec.build(i, c)) continue;
          const real_t p0[3] = {origin[0] - sx, origin[1] - sy, origin[2] - sz};  // plane in cell frame
          const real_t u3[3] = {normal[0], normal[1], normal[2]};
          const int mm = c.sectionPolygon(p0, u3, spx, spy, spz);
          if (mm < 3) continue;
          for (int k = 0; k < mm; ++k) {
            verts.push_back(sx + spx[k]);
            verts.push_back(sy + spy[k]);
            verts.push_back(sz + spz[k]);
          }
          off.push_back((int64_t)(verts.size() / 3));
          vol.push_back(c.volumePerVertex());
          cellSeed.push_back(i);
        }
        const std::size_t nV = verts.size() / 3, nP = vol.size();
        nb::dict d;
        d["verts"] = peclet::core::python::vector_to_ndarray(std::move(verts), {nV, 3}, {3, 1});
        d["offsets"] = peclet::core::python::vector_to_ndarray(std::move(off), {off.size()}, {1});
        d["volume"] = peclet::core::python::vector_to_ndarray(std::move(vol), {nP}, {1});
        d["seed"] = peclet::core::python::vector_to_ndarray(std::move(cellSeed), {nP}, {1});
        return d;
      },
      nb::arg("positions"), nb::arg("sphere_centres"), nb::arg("sphere_radii"), nb::arg("L"),
      nb::arg("origin"), nb::arg("normal"),
      "Cross-section of the SDF-clipped interstitial Voronoi mesh by the plane through `origin` with\n"
      "`normal`: cut every cell directly (ConvexCell::sectionPolygon, robust — works from the dual\n"
      "edges, so it tiles the plane exactly where a face-by-face slice drops facets). Returns 'verts'\n"
      "(Nv,3, world coords, all on the plane) + 'offsets' (Npoly+1, per-polygon vertex ranges) +\n"
      "'volume' (Npoly, the 3-D cell volume) + 'seed' (Npoly, the seed index). For a z=z0 slice pass\n"
      "origin=(0,0,z0), normal=(0,0,1) and plot verts[:, :2].");

  m.def(
      "minimize_interface",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<int, nb::c_contig> type_in,
         real_t sigma, real_t L, int sw, int max_iter, real_t tol) {
        auto pos = flatten3(pos_in);
        const int N = (int)type_in.shape(0);
        std::vector<int> type(type_in.data(), type_in.data() + N);
        const real_t Larr[3] = {L, L, L};
        auto R = peclet::voro::interfaceMinimize<real_t>(pos, type, sigma, Larr, N, sw,
                                                         peclet::voro::NoSdf{}, max_iter, tol, false);
        nb::dict d;
        d["positions"] = peclet::core::python::vector_to_ndarray(
            std::move(pos), {static_cast<std::size_t>(N), 3}, {3, 1});
        d["energy"] = R.maxVolErr;         // final interfacial energy
        d["energy_ratio"] = R.meanVolErr;  // E_final / E_initial
        d["iters"] = R.iters;
        d["converged"] = R.converged;
        return d;
      },
      nb::arg("positions"), nb::arg("types"), nb::arg("sigma") = 1.0, nb::arg("L") = 1.0,
      nb::arg("sw") = 5, nb::arg("max_iter") = 60, nb::arg("tol") = 1e-9,
      "Surface-Evolver-style interfacial-tension minimiser: move seeds (N,3) to minimise the total\n"
      "area of faces between cells of different integer type (N,), E = Σ σ A_ij. Steepest descent\n"
      "with a trust-region line search on the (non-smooth) interfacial energy. Returns a dict with\n"
      "the updated 'positions', final 'energy', 'energy_ratio' (final/initial), and iters.");

  // ---- Tessellation -----------------------------------------------------------------------------
  nb::class_<Tess>(
      m, "Tessellation",
      "Moving-particle Voronoi tessellator on the device path.\n\n"
      "Build a tessellation once (`build`) then advance it cheaply as the points move\n"
      "(`step`) — the incremental two-pass repair is several times faster than rebuilding\n"
      "for the small per-step displacements typical of CFD/DEM, and falls back to a full\n"
      "rebuild (via an adaptive gate) when displacements are large, so it is never much\n"
      "slower than a cold build. Periodic cubic box. Single domain (one process).")
      .def(nb::init<>())
      .def("set_box", &Tess::set_box, nb::arg("L"),
           "Set the periodic box edge lengths (Lx, Ly, Lz). Call before `build`.")
      .def("set_tolerance", &Tess::set_tolerance, nb::arg("frac") = 1e-4,
           "Certificate tolerance as a fraction of the mean inter-particle spacing (default 1e-4). "
           "A\n"
           "vertex poking past a stored plane by more than this flags the cell for repair; smaller "
           "is\n"
           "stricter (closer to machine-exact) at marginally higher cost.")
      .def("set_local_certificate", &Tess::set_local_certificate, nb::arg("on") = true,
           "Use the cheap O(nt) Lawson local certificate (default True) instead of the brute "
           "O(nt*np)\n"
           "form for detecting which cells changed. Both are complete; local is faster.")
      .def("set_gate", &Tess::set_gate, nb::arg("on") = true,
           "Enable the adaptive gate (default True) that routes high-churn steps straight to a "
           "full\n"
           "rebuild — the 'never much slower than a cold build' guard.")
      .def(
          "set_geometry", &Tess::set_geometry, nb::arg("node_ints"), nb::arg("node_reals"),
          nb::arg("root") = 0, nb::arg("grad_h") = 1e-5,
          "Clip the cells by an SDF solid given as a core shape scene in the flat node encoding\n"
          "(node_ints int32 (3 per node), node_reals float64 (16 per node)) — exactly what\n"
          "peclet.core.geom.Scene.encode() returns and dem.add_analytic_wall takes; `root` is the\n"
          "tree root to evaluate. Suite sign convention: sdf < 0 inside the solid. Seeds inside "
          "the\n"
          "solid get no cell (volume 0); cells reaching into it gain wall facets. Applies to the\n"
          "next `build` and is carried through every `step` (wall planes are resident; a boundary\n"
          "watch re-clips cells at the wall). `grad_h` is the central-difference step for the\n"
          "SDF gradient. Analytic vocabulary only (no sampled grids through this path yet).")
      .def("clear_geometry", &Tess::clear_geometry,
           "Drop the SDF geometry (takes effect at the next `build`).")
      .def(
          "set_wall_mode", &Tess::set_wall_mode, nb::arg("exact") = true,
          nb::arg("skin_frac") = 0.0,
          "Wall re-gather policy for `step` (default exact=True): re-clip every wall-clipped cell\n"
          "that moved, so the incremental result equals a cold rebuild. exact=False keeps a "
          "cell's\n"
          "stale tangent planes until it moved more than skin_frac × mean spacing (cheaper, not\n"
          "exact by construction).")
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
           "the box, or overflowed cells; see `build_report()`.")
      .def("build_report", &Tess::build_report,
           "Validity counts of the last build: {'buried', 'reach_exceeded', 'empty', 'overflow',\n"
           "'incomplete'} — all zero for a guaranteed-exact partition.")
      .def(
          "step", &Tess::step, nb::arg("positions"),
          "Incrementally repair the resident tessellation to new `positions` (N,3, same N as "
          "`build`).\n"
          "Returns a dict of per-step work stats: 'flagged' (cells the certificate flagged), "
          "'pass1'\n"
          "and 'pass2' (cells re-gathered in each pass), 'extra' (cells gathered across verify "
          "extra-passes),\n"
          "'surgical' (Pass-1 cells repaired surgically), 'verify_passes' (verify iterations run), "
          "'rebuilt'\n"
          "(True if the gate routed this step to a full rebuild), 'fell_back' (True if the verify "
          "failed and\n"
          "a cold rebuild was forced).")
      .def("volumes", &Tess::volumes,
           "Per-particle Voronoi cell volume (N,) float64. Sums to the box volume (space-filling).")
      .def("neighbor_counts", &Tess::neighbor_counts,
           "Per-particle Voronoi neighbour count (N,) int32 — the number of faces of each cell "
           "(wall facets included).")
      .def("wall_counts", &Tess::wall_counts,
           "Per-particle number of resident SDF wall planes (N,) int32; all zero without geometry.")
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
          "  centroidal   `lloyd` · Σ ∫_cell |y − x_i|² (Lloyd/CVT; gradient 2V(x−c) drives seeds "
          "to\n"
          "               their centroids — the skewness the grid solver's two-point operators "
          "need gone);\n"
          "  roundness    `facet_tension` · Σ A_f over all interior faces.\n"
          "2(V/Vref−1)/Vref).\n"
          "Returns {'interface_energy', 'wall_energy', 'force' (N,3) = dE/dx, 'force_w' (N,) = "
          "dE/dw\n"
          "when weights are set}. Descend along −force to minimise.")
      .def_prop_ro("num_particles", &Tess::num_particles,
                   "Particle count N set by the last `build`.");

  // ---- Simulation -------------------------------------------------------------------------------
  nb::class_<Sim>(
      m, "Simulation",
      "Device-native compressible-Euler / Navier-Stokes Voronoi fluid simulation.\n\n"
      "Velocity-Verlet dynamics of a moving-particle Voronoi fluid: pressure forces from an\n"
      "EOS plus an optional per-particle viscous (Navier-Stokes) term, with the tessellation\n"
      "repaired each step on the device. Set the particle state, `init`, then `step`.")
      .def(nb::init<>())
      .def("set_box", &Sim::set_box, nb::arg("L"),
           "Set the periodic box edge lengths (Lx, Ly, Lz).")
      .def("set_positions", &Sim::set_positions, nb::arg("positions"),
           "Initial particle positions (N,3) float64.")
      .def("set_velocities", &Sim::set_velocities, nb::arg("velocities"),
           "Initial particle velocities (N,3) float64.")
      .def("set_masses", &Sim::set_masses, nb::arg("masses"), "Particle masses (N,) float64.")
      .def("set_pressure", &Sim::set_pressure, nb::arg("pressure"),
           "Equation-of-state pressure constant (the stiffness of the barotropic EOS).")
      .def("set_viscosities", &Sim::set_viscosities, nb::arg("viscosities"),
           "Per-particle shear viscosity (N,) — enables the viscous Navier-Stokes term.")
      .def("set_repair", &Sim::set_repair, nb::arg("on") = true,
           "Opt-in (default off): use the incremental moving-point repair + reeval-published force "
           "geometry each step instead of a full rebuild. Call before init().")
      .def("set_bulk_viscosities", &Sim::set_bulk_viscosities, nb::arg("viscosities"),
           "Per-particle bulk viscosity (N,) float64 (defaults to zero if unset).")
      .def(
          "set_geometry", &Sim::set_geometry, nb::arg("node_ints"), nb::arg("node_reals"),
          nb::arg("root") = 0, nb::arg("grad_h") = 1e-5,
          "SDF solid walls for the fluid (same flat node encoding as Tessellation.set_geometry).\n"
          "The cells are clipped by the solid; the EOS pressure acts on the wall facets (the wall\n"
          "pushes back). Call before init().")
      .def("clear_geometry", &Sim::clear_geometry, "Drop the SDF geometry (before init()).")
      .def("init", &Sim::init,
           "Build the first tessellation and forces from the particle state set above.")
      .def("step", &Sim::step, nb::arg("num_steps"), nb::arg("dt"),
           "Advance the velocity-Verlet dynamics by `num_steps` steps of size `dt`.")
      .def("get_positions", &Sim::get_positions, "Current particle positions (N,3) float64.")
      .def("get_velocities", &Sim::get_velocities, "Current particle velocities (N,3) float64.")
      .def("get_forces", &Sim::get_forces,
           "Current per-particle force (N,3) float64 — the pressure (EOS) force plus the optional\n"
           "viscous Navier-Stokes term, as used by the last velocity-Verlet kick. Useful for\n"
           "force-field analysis, equilibrium/convergence checks, and coupling.")
      .def_prop_ro("num_particles", &Sim::num_particles, "Particle count N.")
      .def("get_kinetic_energy", &Sim::get_kinetic_energy, "Total kinetic energy (scalar).")
      .def("get_internal_energy", &Sim::get_internal_energy,
           "Total internal (EOS) energy (scalar).")
      .def("get_time", &Sim::get_time, "Current simulation time (scalar).")
      .def("get_volumes", &Sim::get_volumes, "Per-particle Voronoi cell volume (N,) float64.")
      .def("get_num_neighbors", &Sim::get_num_neighbors,
           "Per-particle Voronoi neighbour (facet) count (N,) int32.");

#ifdef PECLET_VORO_MPI
  // ---- VoronoiHalo (distributed) ----------------------------------------------------------------
  nb::class_<VHalo>(
      m, "VoronoiHalo",
      "Distributed (MPI) ghost-gather for the multi-rank Voronoi tessellation.\n\n"
      "ORB block-decomposes a periodic box across MPI ranks and gathers, for each rank, every seed\n"
      "within a cutoff `rcut` of its owned block (periodic images included). The recipe: select this\n"
      "rank's owned seeds with `owned_mask`, `gather(...)` the owned+ghost set, tessellate it with the\n"
      "single-rank `Tessellation` building only the first `n_owned` cells, and keep those cells — they\n"
      "are bit-identical to a serial full-box tessellation (each owned cell has all its neighbours\n"
      "present). `rcut` must exceed the largest owned-cell interaction distance (a few mean spacings).\n"
      "Auto-initialises MPI (MPI_COMM_WORLD). Drive it from mpi4py.")
      .def(nb::init<std::array<real_t, 3>, std::array<real_t, 3>, std::array<long, 3>,
                    std::array<bool, 3>>(),
           nb::arg("origin"), nb::arg("size"), nb::arg("gsize"), nb::arg("periodic"),
           "Build the ORB decomposition of the box [origin, origin+size) on `gsize` ORB cells with\n"
           "per-axis `periodic` flags, over MPI_COMM_WORLD.")
      .def("rank", &VHalo::rank, "This rank's MPI index.")
      .def("size", &VHalo::size, "Number of MPI ranks.")
      .def("owned_mask", &VHalo::owned_mask, nb::arg("positions"),
           "Mask (N,) int32 over the given positions (N,3): 1 where this rank owns the point, else 0.")
      .def("owner_of", &VHalo::owner_of, nb::arg("x"), nb::arg("y"), nb::arg("z"),
           "Owning rank of a single point (x, y, z).")
      .def("gather", &VHalo::gather, nb::arg("owned_pos"), nb::arg("owned_gid"),
           nb::arg("owned_weight"), nb::arg("rcut"),
           "Gather ghost seeds within `rcut` of this rank's owned seeds. Inputs: owned_pos (N,3)\n"
           "float64, owned_gid (N,) int64, owned_weight (N,) float64. Returns a tuple\n"
           "(pos (M,3) float64, gid (M,) int64, weight (M,) float64, n_owned): rows [0,n_owned) are the\n"
           "owned seeds, [n_owned,M) the gathered ghosts (with their owners' global ids/weights).")
      .def("refresh_positions", &VHalo::refresh_positions, nb::arg("owned_pos"),
           "Position-only halo refresh (Verlet fast path): re-forward the current owned positions\n"
           "(N,3) onto the topology of the last `gather`, returning the combined owned+ghost positions\n"
           "(M,3) in the same order as that gather (no re-decomposition / ghost re-selection).");
#endif
}
