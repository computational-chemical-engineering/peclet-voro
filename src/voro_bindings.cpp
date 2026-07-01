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

#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <set>
#include <vector>

#include "peclet/core/common/view.hpp"
#include "peclet/core/python/ndarray_interop.hpp"
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/topology_store.hpp"
#include "peclet/voro/physics/simulation.hpp"

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

// --------------------------------------------------------------------------------------------------
// Tessellation: the bare moving-particle Voronoi tessellator (cold build + incremental repair).
// --------------------------------------------------------------------------------------------------
class Tess {
 public:
  Tess() { live().insert(this); }
  ~Tess() { live().erase(this); }

  void set_box(std::array<real_t, 3> L) { L_ = L; }
  void set_tolerance(real_t frac) { tolFrac_ = frac; }
  void set_local_certificate(bool on) { localCert_ = on; }
  void set_gate(bool on) { useGate_ = on; }

  // Cold build: (re)allocate the resident tessellation for N points and build it from scratch.
  void build(nb::ndarray<real_t, nb::c_contig> a) {
    std::vector<real_t> p = flatten3(a);
    N_ = static_cast<int>(p.size() / 3);
    const double boxVol = static_cast<double>(L_[0]) * L_[1] * L_[2];
    const real_t spacing = static_cast<real_t>(std::cbrt(boxVol / (N_ > 0 ? N_ : 1)));
    mt_.localCert = localCert_;
    mt_.useGate = useGate_;
    mt_.alloc(N_, L_.data(), tolFrac_ * spacing, real_t(0.25) * spacing, 4, N_);
    pos_ = peclet::core::toDevice<real_t>(p, "pos");
    mt_.rebuild(pos_);
  }

  // Incremental repair: update the resident tessellation to new positions (same N) without a full
  // rebuild. Returns the per-step work stats. Positions must be the same count as the last build().
  nb::dict step(nb::ndarray<real_t, nb::c_contig> a) {
    std::vector<real_t> p = flatten3(a);
    if (static_cast<int>(p.size() / 3) != N_)
      throw std::runtime_error("step(): particle count differs from build(); call build() first");
    pos_ = peclet::core::toDevice<real_t>(p, "pos");
    auto st = mt_.step(pos_);
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
    return d;
  }

  nb::ndarray<nb::numpy, real_t> volumes() {
    const std::size_t N = static_cast<std::size_t>(N_);
    auto v = Kokkos::subview(mt_.vol, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(v), {N}, {1});
  }

  // Per-cell Voronoi neighbour (= face) count, recomputed from the resident topology store.
  nb::ndarray<nb::numpy, int> neighbor_counts() {
    using Cell = peclet::voro::ConvexCell<real_t, 64, 112, false>;
    const int N = N_;
    Kokkos::View<int*, peclet::core::MemSpace> cnt("nbr", N);
    auto st = mt_.store;
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

  int num_particles() const { return N_; }

  void release() {
    mt_ = peclet::voro::MovingTessellation<real_t, 64, 112>{};
    pos_ = DView{};
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
  int N_ = 0;
  DView pos_;
  peclet::voro::MovingTessellation<real_t, 64, 112> mt_;
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
    sim_ = peclet::voro::physics::ExplicitEuler<real_t>{};
    dmass_ = DView{};
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
  void set_repair(bool on) { sim_.setRepair(on); }

  void init() {
    const int N = static_cast<int>(mass_.size());
    std::vector<real_t> invm(N);
    for (int i = 0; i < N; ++i)
      invm[i] = real_t(1) / mass_[i];
    sim_.init(peclet::core::toDevice<real_t>(pos_, "pos"), peclet::core::toDevice<real_t>(vel_, "vel"),
              peclet::core::toDevice<real_t>(invm, "im"), L_, pressEq_);
    dmass_ =
        peclet::core::toDevice<real_t>(mass_, "mass");  // resident; kinetic-energy reads it each call (E4b)
    if (!visc_.empty()) {
      if (bulk_.empty())
        bulk_.assign(N, 0.0);
      sim_.setViscous(peclet::core::toDevice<real_t>(visc_, "visc"), peclet::core::toDevice<real_t>(bulk_, "bulk"));
    }
  }

  void step(int nsteps, real_t dt) { sim_.step(nsteps, dt); }

  nb::ndarray<nb::numpy, real_t> get_positions() { return from3(sim_.positions()); }
  nb::ndarray<nb::numpy, real_t> get_velocities() { return from3(sim_.velocities()); }
  nb::ndarray<nb::numpy, real_t> get_forces() { return from3(sim_.force()); }
  real_t get_kinetic_energy() { return sim_.kineticEnergy(dmass_); }
  real_t get_internal_energy() { return sim_.internalEnergy(); }
  real_t get_time() { return sim_.time(); }
  int num_particles() const { return sim_.numParticles(); }

  nb::ndarray<nb::numpy, real_t> get_volumes() {
    const std::size_t N = static_cast<std::size_t>(sim_.numParticles());
    auto cell = Kokkos::subview(sim_.view().cellVolume, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(cell), {N}, {1});
  }

  nb::ndarray<nb::numpy, int> get_num_neighbors() {
    // Per-cell facet (neighbour) count is the explicit cellFacetCount view. NOTE: the device
    // cellFacetOffset is a per-cell *base* into the facet buffer in cell-finish order, NOT a CSR
    // prefix sum (see tessellation_view.hpp), so off(i+1)-off(i) is meaningless — read the count.
    const std::size_t N = static_cast<std::size_t>(sim_.numParticles());
    auto cnt = Kokkos::subview(sim_.view().cellFacetCount, Kokkos::make_pair(std::size_t(0), N));
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(cnt), {N}, {1});
  }

 private:
  // Flat (3N,) host-or-device view -> (N,3) float64 numpy array (single D2H, no host loop — S2a).
  static nb::ndarray<nb::numpy, real_t> from3(const DView& d) {
    const std::size_t N = static_cast<std::size_t>(d.extent(0)) / 3;
    return peclet::core::python::vector_to_ndarray(peclet::core::toVector(d), {N, std::size_t(3)}, {3, 1});
  }

  std::array<real_t, 3> L_{1, 1, 1};
  real_t pressEq_ = 0;
  std::vector<real_t> pos_, vel_, mass_, visc_, bulk_;
  DView dmass_;  // device-resident masses, uploaded once in init() (E4b)
  peclet::voro::physics::ExplicitEuler<real_t> sim_;
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
      .def("build", &Tess::build, nb::arg("positions"),
           "Cold-build the Voronoi tessellation of `positions` (N,3) from scratch and make it "
           "resident.\n"
           "Sets the particle count N for subsequent `step` calls.")
      .def("step", &Tess::step, nb::arg("positions"),
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
           "Per-particle Voronoi neighbour count (N,) int32 — the number of faces of each cell.")
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
      .def("set_box", &Sim::set_box, nb::arg("L"), "Set the periodic box edge lengths (Lx, Ly, Lz).")
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
