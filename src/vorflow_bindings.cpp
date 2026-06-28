// vorflow -- device (Kokkos) Python bindings: drive the device-native
// moving-particle Voronoi simulation (the de-legacy path) from Python.
//
// Exposes the device-native compressible-Euler / Navier-Stokes Voronoi simulation
// (ExplicitEulerDevice) running entirely on the device path (multicore CPU via OpenMP,
// or GPU). Particle arrays are numpy: positions/velocities (N,3) float64, masses/
// viscosities (N,). nanobind module; Kokkos is initialized at import and left initialized
// for the interpreter's lifetime (call vorflow.finalize() for deterministic teardown).
//
// Arrays cross the boundary through the shared tpx::python bridge (transport-core): host
// fields move into the returned numpy array's backing store with no extra copy.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <array>
#include <Kokkos_Core.hpp>
#include <set>
#include <vector>

#include "tpx/common/view.hpp"
#include "tpx/python/ndarray_interop.hpp"
#include "vorflow/physics/device_simulation.hpp"

namespace nb = nanobind;
using real_t = double;
using DView = Kokkos::View<real_t*, tpx::MemSpace>;

namespace {

// (N,3) c-contiguous array -> flat row-major host vector of length 3N.
std::vector<real_t> flatten3(nb::ndarray<real_t, nb::c_contig> a) {
  if (a.ndim() != 2 || a.shape(1) != 3) throw std::runtime_error("expected an (N,3) array");
  const real_t* p = a.data();
  return std::vector<real_t>(p, p + static_cast<std::size_t>(a.shape(0)) * 3);
}

// (N,) array -> host vector of length N.
std::vector<real_t> flatten1(nb::ndarray<real_t, nb::c_contig> a) {
  return tpx::python::ndarray_to_vector<real_t>(nb::ndarray<>(a));
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
  void set_positions(nb::ndarray<real_t, nb::c_contig> a) { pos_ = flatten3(a); }
  void set_velocities(nb::ndarray<real_t, nb::c_contig> a) { vel_ = flatten3(a); }
  void set_masses(nb::ndarray<real_t, nb::c_contig> a) { mass_ = flatten1(a); }
  void set_pressure(real_t p) { pressEq_ = p; }
  void set_viscosities(nb::ndarray<real_t, nb::c_contig> a) { visc_ = flatten1(a); }
  void set_bulk_viscosities(nb::ndarray<real_t, nb::c_contig> a) { bulk_ = flatten1(a); }

  void init() {
    const int N = static_cast<int>(mass_.size());
    std::vector<real_t> invm(N);
    for (int i = 0; i < N; ++i)
      invm[i] = real_t(1) / mass_[i];
    sim_.init(tpx::toDevice<real_t>(pos_, "pos"), tpx::toDevice<real_t>(vel_, "vel"),
              tpx::toDevice<real_t>(invm, "im"), L_, pressEq_);
    if (!visc_.empty()) {
      if (bulk_.empty())
        bulk_.assign(N, 0.0);
      sim_.setViscous(tpx::toDevice<real_t>(visc_, "visc"), tpx::toDevice<real_t>(bulk_, "bulk"));
    }
  }

  void step(int nsteps, real_t dt) { sim_.step(nsteps, dt); }

  nb::ndarray<nb::numpy, real_t> get_positions() { return from3(sim_.positions()); }
  nb::ndarray<nb::numpy, real_t> get_velocities() { return from3(sim_.velocities()); }
  real_t get_kinetic_energy() { return sim_.kineticEnergy(tpx::toDevice<real_t>(mass_, "m")); }
  real_t get_internal_energy() { return sim_.internalEnergy(); }
  real_t get_time() { return sim_.time(); }

  nb::ndarray<nb::numpy, real_t> get_volumes() {
    auto v = Kokkos::create_mirror_view(sim_.view().cellVolume);
    Kokkos::deep_copy(v, sim_.view().cellVolume);
    const std::size_t N = static_cast<std::size_t>(sim_.numParticles());
    std::vector<real_t> out(N);
    for (std::size_t i = 0; i < N; ++i) out[i] = v(i);
    return tpx::python::vector_to_ndarray(std::move(out), {N}, {1});
  }

  nb::ndarray<nb::numpy, int> get_num_neighbors() {
    auto off = Kokkos::create_mirror_view(sim_.view().cellFacetOffset);
    Kokkos::deep_copy(off, sim_.view().cellFacetOffset);
    const std::size_t N = static_cast<std::size_t>(sim_.numParticles());
    std::vector<int> out(N);
    for (std::size_t i = 0; i < N; ++i) out[i] = off(i + 1) - off(i);
    return tpx::python::vector_to_ndarray(std::move(out), {N}, {1});
  }

 private:
  // Flat (3N,) host-or-device view -> (N,3) float64 numpy array (host copy; matches the prior path).
  static nb::ndarray<nb::numpy, real_t> from3(const DView& d) {
    auto h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(h, d);
    const std::size_t M = static_cast<std::size_t>(d.extent(0));
    std::vector<real_t> out(M);
    for (std::size_t i = 0; i < M; ++i) out[i] = h(i);
    const std::size_t N = M / 3;
    return tpx::python::vector_to_ndarray(std::move(out), {N, std::size_t(3)}, {3, 1});
  }

  std::array<real_t, 3> L_{1, 1, 1};
  real_t pressEq_ = 0;
  std::vector<real_t> pos_, vel_, mass_, visc_, bulk_;
  vor::physics::ExplicitEulerDevice<real_t> sim_;
};

}  // namespace

NB_MODULE(vorflow, m) {
  m.attr("__doc__") = "vorflow (device/Kokkos): moving-particle Voronoi dynamics on the device path.";
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  // finalize() frees every live Simulation's Views then finalizes Kokkos (deterministic teardown).
  // No atexit hook: a Simulation (or a returned array's owning capsule) can outlive it, and finalizing
  // first aborts ("deallocated after Kokkos::finalize"). Matches sdflow/dem/tpx_amr.
  m.def("finalize",
        []() {
          Sim::releaseAll();
          if (Kokkos::is_initialized() && !Kokkos::is_finalized())
            Kokkos::finalize();
        },
        "Release all live Simulations and finalize Kokkos (deterministic teardown).");
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());

  nb::class_<Sim>(m, "Simulation",
                  "Device-native compressible-Euler / Navier-Stokes Voronoi simulation.")
      .def(nb::init<>())
      .def("set_l", &Sim::set_l, nb::arg("L"), "Set the periodic box (Lx,Ly,Lz).")
      .def("set_positions", &Sim::set_positions, nb::arg("positions"), "Positions (N,3).")
      .def("set_velocities", &Sim::set_velocities, nb::arg("velocities"), "Velocities (N,3).")
      .def("set_masses", &Sim::set_masses, nb::arg("masses"), "Masses (N,).")
      .def("set_pressure", &Sim::set_pressure, nb::arg("pressure"), "EOS pressure constant.")
      .def("set_viscosities", &Sim::set_viscosities, nb::arg("viscosities"),
           "Per-particle shear viscosity (N,) — enables the viscous Navier-Stokes term.")
      .def("set_bulk_viscosities", &Sim::set_bulk_viscosities, nb::arg("viscosities"),
           "Per-particle bulk viscosity (N,).")
      .def("init", &Sim::init, "Build the first tessellation and forces.")
      .def("step", &Sim::step, nb::arg("num_steps"), nb::arg("dt"),
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
