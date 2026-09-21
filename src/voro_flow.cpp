/// @file
/// @brief `peclet.voro.FlowSolver`: the collocated / covolume Navier-Stokes solvers on the face
/// mesh.
#include "peclet/voro/fv/collocated.hpp"
#include "peclet/voro/fv/covolume.hpp"
#include "peclet/voro/fv/mesh.hpp"
#include "voro_bindings_tess.hpp"

namespace peclet::voro::pybind {

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

void bindFlow(nb::module_& m) {
  using namespace defaults;
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
}
}  // namespace peclet::voro::pybind
