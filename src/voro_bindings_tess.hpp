/// @file
/// @brief `peclet.voro.Tessellation` — the bound class over the four MovingTessellation variants.
///
/// The class lives in a header because `Flow` (voro_flow.cpp) and the distributed drivers
/// (voro_mpi.cpp) name it.  `face_mesh()` is DECLARED here and DEFINED in voro_tessellation.cpp:
/// it is the only method a second TU calls, and its body instantiates reevalPublish /
/// buildAuxMaps / buildFaceMesh for all four variants — i.e. the whole point of the split would
/// be lost if it were inline (QUALITY_PLAN G.8, the same discipline as flow's
/// `flow_solver_staggered.cpp`).
#ifndef PECLET_VORO_BINDINGS_TESS_HPP
#define PECLET_VORO_BINDINGS_TESS_HPP

#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/energy/interface.hpp"
#include "peclet/voro/energy/lloyd.hpp"
#include "peclet/voro/energy/tension.hpp"
#include "peclet/voro/energy/volume.hpp"
#include "peclet/voro/energy/wall.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "peclet/voro/reeval_tessellation.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/topology_store.hpp"
#include "voro_bindings_common.hpp"

namespace peclet::voro::pybind {

// --------------------------------------------------------------------------------------------------
// Tessellation: the bare moving-point (power-)Voronoi tessellator, optionally SDF-clipped.
// The engine is chosen at build() from what was set: {Voronoi, Power} x {no geometry, SdfScene}.
// --------------------------------------------------------------------------------------------------
template <bool W, class S>
using MT =
    peclet::voro::MovingTessellation<real_t, defaults::kMaxPlanes, defaults::kMaxTriangles, W, S>;
using MtVariant =
    std::variant<MT<false, NoSdfT>, MT<true, NoSdfT>, MT<false, SceneT>, MT<true, SceneT>>;

inline nb::dict repairStatsDict(const peclet::voro::RepairStats& st) {
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
  peclet::voro::fv::FaceMesh<real_t> face_mesh();

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
}  // namespace peclet::voro::pybind

#endif  // PECLET_VORO_BINDINGS_TESS_HPP
