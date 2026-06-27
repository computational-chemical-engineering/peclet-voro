// vorflow -- device (Kokkos) Python bindings: drive the device-native
// moving-particle Voronoi simulation (the de-legacy path) from Python.
//
// Exposes ExplicitEulerDevice (compressible Euler, optional viscous Navier-Stokes)
// running entirely on the device path (multicore CPU via OpenMP, or GPU). Particle
// arrays are numpy: positions/velocities (N,3) float64, masses/viscosities (N,).
// This is the device counterpart of the legacy python/ module and shares the name
// `vorflow`; Kokkos is initialized at import and finalized via atexit (release the
// Simulation before interpreter exit, or call vorflow.finalize()).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <Kokkos_Core.hpp>
#include <set>
#include <vector>

#include "tpx/common/view.hpp"
#include "vorflow/physics/device_simulation.hpp"

namespace py = pybind11;
using real_t = double;
using DView = Kokkos::View<real_t*, tpx::MemSpace>;

namespace {

DView upload(const std::vector<real_t>& h, const char* l) {
  DView d(Kokkos::view_alloc(std::string(l), Kokkos::WithoutInitializing), h.size());
  auto hv = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < h.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d, hv);
  return d;
}

std::vector<real_t> flatten3(py::array_t<real_t, py::array::c_style | py::array::forcecast> a) {
  auto r = a.unchecked<2>();
  if (r.shape(1) != 3)
    throw std::runtime_error("expected an (N,3) array");
  std::vector<real_t> v(static_cast<size_t>(r.shape(0)) * 3);
  for (py::ssize_t i = 0; i < r.shape(0); ++i)
    for (int k = 0; k < 3; ++k)
      v[3 * i + k] = r(i, k);
  return v;
}

template <class V>
std::vector<real_t> flatten1(py::array_t<V, py::array::c_style | py::array::forcecast> a) {
  auto r = a.template unchecked<1>();
  std::vector<real_t> v(r.shape(0));
  for (py::ssize_t i = 0; i < r.shape(0); ++i)
    v[i] = static_cast<real_t>(r(i));
  return v;
}

// Device-driven compressible-Euler / Navier-Stokes simulation.
class Sim {
 public:
  Sim() { live().insert(this); }
  ~Sim() { live().erase(this); }

  // Drop all Kokkos Views (so they free BEFORE Kokkos::finalize at shutdown).
  void release() {
    sim_ = vor::physics::ExplicitEulerDevice<real_t>{};
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
  void set_positions(py::array_t<real_t> a) { pos_ = flatten3(a); }
  void set_velocities(py::array_t<real_t> a) { vel_ = flatten3(a); }
  void set_masses(py::array_t<real_t> a) { mass_ = flatten1<real_t>(a); }
  void set_pressure(real_t p) { pressEq_ = p; }
  void set_viscosities(py::array_t<real_t> a) { visc_ = flatten1<real_t>(a); }
  void set_bulk_viscosities(py::array_t<real_t> a) { bulk_ = flatten1<real_t>(a); }

  void init() {
    const int N = static_cast<int>(mass_.size());
    std::vector<real_t> invm(N);
    for (int i = 0; i < N; ++i)
      invm[i] = real_t(1) / mass_[i];
    sim_.init(upload(pos_, "pos"), upload(vel_, "vel"), upload(invm, "im"), L_, pressEq_);
    if (!visc_.empty()) {
      if (bulk_.empty())
        bulk_.assign(N, 0.0);
      sim_.setViscous(upload(visc_, "visc"), upload(bulk_, "bulk"));
    }
  }

  void step(int nsteps, real_t dt) { sim_.step(nsteps, dt); }

  py::array_t<real_t> get_positions() { return from3(sim_.positions()); }
  py::array_t<real_t> get_velocities() { return from3(sim_.velocities()); }
  real_t get_kinetic_energy() { return sim_.kineticEnergy(upload(mass_, "m")); }
  real_t get_internal_energy() { return sim_.internalEnergy(); }
  real_t get_time() { return sim_.time(); }

  py::array_t<real_t> get_volumes() {
    auto v = Kokkos::create_mirror_view(sim_.view().cellVolume);
    Kokkos::deep_copy(v, sim_.view().cellVolume);
    const int N = sim_.numParticles();
    py::array_t<real_t> a(N);
    auto r = a.mutable_unchecked<1>();
    for (int i = 0; i < N; ++i)
      r(i) = v(i);
    return a;
  }

  py::array_t<int> get_num_neighbors() {
    auto off = Kokkos::create_mirror_view(sim_.view().cellFacetOffset);
    Kokkos::deep_copy(off, sim_.view().cellFacetOffset);
    const int N = sim_.numParticles();
    py::array_t<int> a(N);
    auto r = a.mutable_unchecked<1>();
    for (int i = 0; i < N; ++i)
      r(i) = off(i + 1) - off(i);
    return a;
  }

 private:
  static py::array_t<real_t> from3(const DView& d) {
    auto h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(h, d);
    const py::ssize_t N = static_cast<py::ssize_t>(d.extent(0) / 3);
    py::array_t<real_t> a({N, py::ssize_t(3)});
    auto r = a.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < N; ++i)
      for (int k = 0; k < 3; ++k)
        r(i, k) = h(3 * i + k);
    return a;
  }

  std::array<real_t, 3> L_{1, 1, 1};
  real_t pressEq_ = 0;
  std::vector<real_t> pos_, vel_, mass_, visc_, bulk_;
  vor::physics::ExplicitEulerDevice<real_t> sim_;
};

}  // namespace

PYBIND11_MODULE(vorflow, m) {
  m.doc() = "vorflow (device/Kokkos): moving-particle Voronoi dynamics on the device path.";
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  auto shutdown = []() {
    Sim::releaseAll();  // free Views before Kokkos shuts down
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  };
  m.def("finalize", shutdown, "Finalize Kokkos (call after releasing all Simulations).");
  py::module_::import("atexit").attr("register")(py::cpp_function(shutdown));
  m.attr("execution_space") = py::str(Kokkos::DefaultExecutionSpace::name());

  py::class_<Sim>(m, "Simulation",
                        "Device-native compressible-Euler / Navier-Stokes Voronoi simulation.")
      .def(py::init<>())
      .def("set_l", &Sim::set_l, py::arg("L"), "Set the periodic box (Lx,Ly,Lz).")
      .def("set_positions", &Sim::set_positions, py::arg("positions"), "Positions (N,3).")
      .def("set_velocities", &Sim::set_velocities, py::arg("velocities"), "Velocities (N,3).")
      .def("set_masses", &Sim::set_masses, py::arg("masses"), "Masses (N,).")
      .def("set_pressure", &Sim::set_pressure, py::arg("pressure"), "EOS pressure constant.")
      .def("set_viscosities", &Sim::set_viscosities, py::arg("viscosities"),
           "Per-particle shear viscosity (N,) — enables the viscous Navier-Stokes term.")
      .def("set_bulk_viscosities", &Sim::set_bulk_viscosities, py::arg("viscosities"),
           "Per-particle bulk viscosity (N,).")
      .def("init", &Sim::init, "Build the first tessellation and forces.")
      .def("step", &Sim::step, py::arg("num_steps"), py::arg("dt"),
           "Advance the velocity-Verlet dynamics by num_steps steps of size dt.")
      .def("get_positions", &Sim::get_positions, "Return positions (N,3).")
      .def("get_velocities", &Sim::get_velocities, "Return velocities (N,3).")
      .def("get_kinetic_energy", &Sim::get_kinetic_energy, "Total kinetic energy.")
      .def("get_internal_energy", &Sim::get_internal_energy, "Total internal (EOS) energy.")
      .def("get_time", &Sim::get_time, "Current simulation time.")
      .def("get_volumes", &Sim::get_volumes, "Per-particle Voronoi cell volume (N,).")
      .def("get_num_neighbors", &Sim::get_num_neighbors,
           "Per-particle Voronoi neighbour (facet) count (N,).");
}
