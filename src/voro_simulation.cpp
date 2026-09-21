/// @file
/// @brief `peclet.voro.Simulation`: the device-native moving-cell compressible fluid.
#include "peclet/voro/physics/simulation.hpp"
#include "voro_bindings_common.hpp"

namespace peclet::voro::pybind {

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

void bindSimulation(nb::module_& m) {
  using namespace defaults;
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
      .def(nb::init<>(),
           "Simulation(): takes no arguments. Configure with `set_domain` (required),\n"
           "`set_positions` / `set_masses` (required) and optionally `set_velocities` / "
           "`set_pressure` /\n"
           "`set_viscosities` / `set_bulk_viscosities` / `set_geometry` before the one `init()` "
           "call\n"
           "that uploads the state to the device and builds the first tessellation.")
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
}
}  // namespace peclet::voro::pybind
