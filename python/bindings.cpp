// vordyn -- pybind11 bindings for voronoi_dynamics (the first Python surface for the library).
//
// Exposes the moving-particle Voronoi simulation hierarchy (Simulation -> ExplicitEuler ->
// NavierStokes -> {Incompressible, IntfDyn}) for real_t = double, driven from Python like the suite's
// other compiled solvers (cfd-gpu `pnm_backend`, packing-gpu `demgpu`). Particle arrays are numpy:
// positions/velocities (N,3) float64, masses/types/volumes (N,). Verb names match the suite
// convention (set_positions / get_positions / step / ...). See ../../docs/CONVENTIONS.md s.6.
//
// Build: configure with -DVORONOI_BUILD_PYTHON=ON (see python/CMakeLists.txt).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <vector>

#include "voronoi_dynamics/simulation.hpp"
#include "voronoi_dynamics/voronoi.hpp"

namespace py = pybind11;
using real_t = double;

// ---- numpy <-> std conversions ---------------------------------------------------------------
static std::vector<std::array<real_t, 3>> to_vec3(py::array_t<real_t, py::array::c_style | py::array::forcecast> a) {
  auto r = a.unchecked<2>();
  if (r.shape(1) != 3)
    throw std::runtime_error("expected an (N,3) array");
  std::vector<std::array<real_t, 3>> v(r.shape(0));
  for (py::ssize_t i = 0; i < r.shape(0); ++i)
    v[i] = {r(i, 0), r(i, 1), r(i, 2)};
  return v;
}

static py::array_t<real_t> from_vec3(const std::vector<std::array<real_t, 3>>& v) {
  py::array_t<real_t> a({static_cast<py::ssize_t>(v.size()), static_cast<py::ssize_t>(3)});
  auto r = a.mutable_unchecked<2>();
  for (std::size_t i = 0; i < v.size(); ++i) {
    r(i, 0) = v[i][0];
    r(i, 1) = v[i][1];
    r(i, 2) = v[i][2];
  }
  return a;
}

template <class T>
static std::vector<T> to_vec1(py::array_t<T, py::array::c_style | py::array::forcecast> a) {
  auto r = a.template unchecked<1>();
  std::vector<T> v(r.shape(0));
  for (py::ssize_t i = 0; i < r.shape(0); ++i)
    v[i] = r(i);
  return v;
}

template <class T>
static py::array_t<T> from_vec1(const std::vector<T>& v) {
  py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
  auto r = a.template mutable_unchecked<1>();
  for (std::size_t i = 0; i < v.size(); ++i)
    r(i) = v[i];
  return a;
}

// Per-particle tessellation output (cells are stored by cell index; scatter to particle order via
// the cell's id). `volume` true -> per-cell Voronoi volume; false -> per-cell neighbour (facet) count.
static py::array_t<real_t> cell_scalar(vor::Simulation<real_t>& s, bool volume) {
  auto& cc = s.getCellComplex();
  const std::size_t np = s.getPositions().size();
  const std::size_t nc = cc.getGeometryArena().numCells();
  std::vector<real_t> out(np, real_t(0));
  for (std::size_t i = 0; i < nc; ++i) {
    const auto id = cc.getCellView(i).getID();
    if (static_cast<std::size_t>(id) >= np)
      continue;
    out[id] = volume ? cc.getGeometryView(i).getVolume()
                     : static_cast<real_t>(cc.getCellView(i).numFacets());
  }
  return from_vec1(out);
}

PYBIND11_MODULE(vordyn, m) {
  m.doc() = "voronoi_dynamics: dynamic 3D Voronoi tessellation of moving particles (Python surface)";

  using Sim = vor::Simulation<real_t>;
  py::class_<Sim>(m, "Simulation")
      .def("set_l", [](Sim& s, std::array<real_t, 3> L) { s.setL(L); }, py::arg("L"),
           "Set the (cubic/Lees-Edwards) box dimensions")
      .def("set_mass_density", &Sim::setMassDensity, py::arg("density"))
      .def("set_masses", [](Sim& s, py::array_t<real_t> a) { s.setMasses(to_vec1<real_t>(a)); },
           py::arg("masses"))
      .def("set_positions", [](Sim& s, py::array_t<real_t> a) { s.setPositions(to_vec3(a)); },
           py::arg("positions"), "Set particle positions, (N,3)")
      .def("set_velocities", [](Sim& s, py::array_t<real_t> a) { s.setVelocities(to_vec3(a)); },
           py::arg("velocities"))
      .def("set_time", &Sim::setTime, py::arg("time"))
      .def("get_positions", [](Sim& s) { return from_vec3(s.getPositions()); })
      .def("get_velocities", [](Sim& s) { return from_vec3(s.getVelocities()); })
      .def("get_time", &Sim::getTime)
      .def("get_kinetic_energy", &Sim::getKineticEnergy)
      .def("get_internal_energy", &Sim::getInternalEnergy)
      .def("put_in_box", &Sim::putInBox, "Wrap particle positions back into the periodic box")
      .def("init", &Sim::init, "Build the initial tessellation; returns False on failure")
      .def("step", &Sim::step, py::arg("num_steps"), py::arg("dt"),
           "Advance the dynamics by num_steps timesteps of size dt")
      .def("get_volumes", [](Sim& s) { return cell_scalar(s, true); },
           "Per-particle Voronoi cell volume, (N,) in particle order (0 if no cell)")
      .def("get_num_neighbors", [](Sim& s) { return cell_scalar(s, false); },
           "Per-particle Voronoi neighbour (facet) count, (N,) in particle order");

  py::class_<vor::ExplicitEuler<real_t>, Sim>(m, "ExplicitEuler")
      .def(py::init<>())
      .def("set_types", [](vor::ExplicitEuler<real_t>& s,
                           py::array_t<std::uint8_t> a) { s.setTypes(to_vec1<std::uint8_t>(a)); },
           py::arg("types"), "Per-particle phase/type label (uint8)")
      .def("set_pressure", &vor::ExplicitEuler<real_t>::setPressure, py::arg("pressure"));

  py::class_<vor::NavierStokes<real_t>, vor::ExplicitEuler<real_t>>(m, "NavierStokes")
      .def(py::init<>())
      .def("set_viscosity", &vor::NavierStokes<real_t>::setViscosity, py::arg("viscosity"))
      .def("set_bulk_viscosity", &vor::NavierStokes<real_t>::setBulkViscosity, py::arg("viscosity"))
      .def("set_viscosities",
           [](vor::NavierStokes<real_t>& s, py::array_t<real_t> a) { s.setViscosities(to_vec1<real_t>(a)); },
           py::arg("viscosities"))
      .def("set_bulk_viscosities",
           [](vor::NavierStokes<real_t>& s, py::array_t<real_t> a) { s.setBulkViscosities(to_vec1<real_t>(a)); },
           py::arg("viscosities"))
      .def("set_ext_force_dens", &vor::NavierStokes<real_t>::setExtForceDens, py::arg("force_density"));

  py::class_<vor::Incompressible<real_t>, vor::NavierStokes<real_t>>(m, "Incompressible")
      .def(py::init<>())
      .def("build_constraint_matrix", &vor::Incompressible<real_t>::buildConstraintMatrix);

  py::class_<vor::IntfDyn<real_t>, vor::NavierStokes<real_t>>(m, "IntfDyn")
      .def(py::init<>())
      .def("set_intf_tension", &vor::IntfDyn<real_t>::setIntfTension, py::arg("tension"),
           py::arg("i"), py::arg("j"), "Set interface tension between phases i and j")
      .def("get_intf_energy", &vor::IntfDyn<real_t>::getIntfEnergy);
}
