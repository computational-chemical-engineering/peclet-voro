/// @file
/// @brief The distributed drivers: `VoronoiHalo` and `DistributedTessellation`.
#include <mpi.h>
#include <nanobind/stl/tuple.h>

#include "peclet/voro/mpi/distributed_moving.hpp"
#include "peclet/voro/mpi/voronoi_halo.hpp"
#include "voro_bindings_tess.hpp"

namespace peclet::voro::pybind {
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

void bindMpi(nb::module_& m) {
  using namespace defaults;
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
}
#endif  // PECLET_VORO_MPI
}  // namespace peclet::voro::pybind
