/**
 * @file vorflow_bindings.cpp
 * @brief Device-native (Kokkos) nanobind Python module `vorflow`.
 *
 * Drives the production device path — multicore CPU (OpenMP), or GPU (CUDA/HIP), selected by the
 * Kokkos backend the extension was built against — from Python. Two surfaces:
 *
 *  - @ref Tess "vorflow.Tessellation" — the bare moving-particle Voronoi tessellator: a cold build
 *    plus the incremental two-pass *repair* update (the fast per-step path for moving points),
 * exposing per-cell volumes and neighbour counts. This is the core primitive all the geometry work
 * builds on.
 *  - @ref Sim "vorflow.Simulation" — a device-native compressible-Euler / Navier–Stokes Voronoi
 * fluid simulation (velocity-Verlet over the tessellation) on top of that primitive.
 *
 * Particle data crosses the boundary as NumPy arrays: positions/velocities are `(N,3)` float64,
 * scalars (masses, viscosities, volumes) are `(N,)`. Arrays move through the shared `tpx::python`
 * bridge (transport-core): returned arrays are backed by host buffers (no extra device copy).
 *
 * Kokkos is initialized at import and finalized via a Python `atexit` hook (with every live
 * object's Views released first — required on CUDA). Call `vorflow.finalize()` for deterministic
 * teardown.
 *
 * Example
 * -------
 * @code{.py}
 *   import numpy as np, vorflow
 *   rng = np.random.default_rng(0)
 *   pos = rng.random((100_000, 3))            # uniform points in the unit box
 *   t = vorflow.Tessellation()
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

#include "tpx/common/view.hpp"
#include "tpx/python/ndarray_interop.hpp"
#include "vorflow/device/convex_cell.hpp"
#include "vorflow/device/repair.hpp"
#include "vorflow/device/topology_store.hpp"
#include "vorflow/physics/device_simulation.hpp"

namespace nb = nanobind;
using real_t = double;
using DView = Kokkos::View<real_t*, tpx::MemSpace>;

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
  return tpx::python::ndarray_to_vector<real_t>(nb::ndarray<>(a));
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
    pos_ = tpx::toDevice<real_t>(p, "pos");
    mt_.rebuild(pos_);
  }

  // Incremental repair: update the resident tessellation to new positions (same N) without a full
  // rebuild. Returns the per-step work stats. Positions must be the same count as the last build().
  nb::dict step(nb::ndarray<real_t, nb::c_contig> a) {
    std::vector<real_t> p = flatten3(a);
    if (static_cast<int>(p.size() / 3) != N_)
      throw std::runtime_error("step(): particle count differs from build(); call build() first");
    pos_ = tpx::toDevice<real_t>(p, "pos");
    auto st = mt_.step(pos_);
    nb::dict d;
    d["flagged"] = st.pass1Raw;  // cells the certificate flagged
    d["pass1"] = st.pass1;       // cells gathered in Pass 1
    d["pass2"] = st.pass2;       // cells gathered in Pass 2
    d["rebuilt"] =
        (st.route == vor::device::RepairStats::kRebuildGate);  // gate routed to a full rebuild
    d["fell_back"] = st.fellBack;                              // verify failed -> cold rebuild
    return d;
  }

  nb::ndarray<nb::numpy, real_t> volumes() {
    const std::size_t N = static_cast<std::size_t>(N_);
    auto v = Kokkos::subview(mt_.vol, Kokkos::make_pair(std::size_t(0), N));
    return tpx::python::vector_to_ndarray(tpx::toVector(v), {N}, {1});
  }

  // Per-cell Voronoi neighbour (= face) count, recomputed from the resident topology store.
  nb::ndarray<nb::numpy, int> neighbor_counts() {
    using Cell = vor::device::ConvexCell<real_t, 64, 112, false>;
    const int N = N_;
    Kokkos::View<int*, tpx::MemSpace> cnt("nbr", N);
    auto st = mt_.store;
    auto C = cnt;
    const real_t Lx = L_[0], Ly = L_[1], Lz = L_[2];
    Kokkos::parallel_for(
        "vorflow.nbrcount", Kokkos::RangePolicy<tpx::ExecSpace>(0, N), KOKKOS_LAMBDA(int i) {
          Cell c;
          st.load(i, c, Lx, Ly, Lz);
          C(i) = c.countFaces();
        });
    return tpx::python::vector_to_ndarray(tpx::toVector(cnt), {static_cast<std::size_t>(N)}, {1});
  }

  int num_particles() const { return N_; }

  void release() {
    mt_ = vor::device::MovingTessellation<real_t, 64, 112>{};
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
  vor::device::MovingTessellation<real_t, 64, 112> mt_;
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
    sim_ = vor::physics::ExplicitEulerDevice<real_t>{};
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

  void set_l(std::array<real_t, 3> L) { L_ = L; }
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
    sim_.init(tpx::toDevice<real_t>(pos_, "pos"), tpx::toDevice<real_t>(vel_, "vel"),
              tpx::toDevice<real_t>(invm, "im"), L_, pressEq_);
    dmass_ =
        tpx::toDevice<real_t>(mass_, "mass");  // resident; kinetic-energy reads it each call (E4b)
    if (!visc_.empty()) {
      if (bulk_.empty())
        bulk_.assign(N, 0.0);
      sim_.setViscous(tpx::toDevice<real_t>(visc_, "visc"), tpx::toDevice<real_t>(bulk_, "bulk"));
    }
  }

  void step(int nsteps, real_t dt) { sim_.step(nsteps, dt); }

  nb::ndarray<nb::numpy, real_t> get_positions() { return from3(sim_.positions()); }
  nb::ndarray<nb::numpy, real_t> get_velocities() { return from3(sim_.velocities()); }
  real_t get_kinetic_energy() { return sim_.kineticEnergy(dmass_); }
  real_t get_internal_energy() { return sim_.internalEnergy(); }
  real_t get_time() { return sim_.time(); }

  nb::ndarray<nb::numpy, real_t> get_volumes() {
    const std::size_t N = static_cast<std::size_t>(sim_.numParticles());
    auto cell = Kokkos::subview(sim_.view().cellVolume, Kokkos::make_pair(std::size_t(0), N));
    return tpx::python::vector_to_ndarray(tpx::toVector(cell), {N}, {1});
  }

  nb::ndarray<nb::numpy, int> get_num_neighbors() {
    // Per-cell facet (neighbour) count is the explicit cellFacetCount view. NOTE: the device
    // cellFacetOffset is a per-cell *base* into the facet buffer in cell-finish order, NOT a CSR
    // prefix sum (see tessellation_view.hpp), so off(i+1)-off(i) is meaningless — read the count.
    const std::size_t N = static_cast<std::size_t>(sim_.numParticles());
    auto cnt = Kokkos::subview(sim_.view().cellFacetCount, Kokkos::make_pair(std::size_t(0), N));
    return tpx::python::vector_to_ndarray(tpx::toVector(cnt), {N}, {1});
  }

 private:
  // Flat (3N,) host-or-device view -> (N,3) float64 numpy array (single D2H, no host loop — S2a).
  static nb::ndarray<nb::numpy, real_t> from3(const DView& d) {
    const std::size_t N = static_cast<std::size_t>(d.extent(0)) / 3;
    return tpx::python::vector_to_ndarray(tpx::toVector(d), {N, std::size_t(3)}, {3, 1});
  }

  std::array<real_t, 3> L_{1, 1, 1};
  real_t pressEq_ = 0;
  std::vector<real_t> pos_, vel_, mass_, visc_, bulk_;
  DView dmass_;  // device-resident masses, uploaded once in init() (E4b)
  vor::physics::ExplicitEulerDevice<real_t> sim_;
};

}  // namespace

NB_MODULE(vorflow, m) {
  m.attr("__doc__") =
      "vorflow (device/Kokkos): moving-particle Voronoi dynamics on the device path.\n\n"
      "Classes: Tessellation (bare cold build + incremental repair, volumes, neighbour counts);\n"
      "Simulation (compressible-Euler / Navier-Stokes Voronoi fluid). Arrays are NumPy: "
      "positions/\n"
      "velocities (N,3) float64, scalars (N,). The backend (Serial/OpenMP/CUDA) is fixed at build\n"
      "time; see vorflow.execution_space.";
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
           "and 'pass2' (cells re-gathered in each pass), 'rebuilt' (True if the gate routed this "
           "step\n"
           "to a full rebuild), 'fell_back' (True if the verify failed and a cold rebuild was "
           "forced).")
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
      .def("set_l", &Sim::set_l, nb::arg("L"), "Set the periodic box edge lengths (Lx, Ly, Lz).")
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
      .def("get_kinetic_energy", &Sim::get_kinetic_energy, "Total kinetic energy (scalar).")
      .def("get_internal_energy", &Sim::get_internal_energy,
           "Total internal (EOS) energy (scalar).")
      .def("get_time", &Sim::get_time, "Current simulation time (scalar).")
      .def("get_volumes", &Sim::get_volumes, "Per-particle Voronoi cell volume (N,) float64.")
      .def("get_num_neighbors", &Sim::get_num_neighbors,
           "Per-particle Voronoi neighbour (facet) count (N,) int32.");
}
