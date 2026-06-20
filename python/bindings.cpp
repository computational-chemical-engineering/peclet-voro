// vordyn -- pybind11 bindings for vorflow (the first Python surface for the library).
//
// Exposes the moving-particle Voronoi simulation hierarchy (Simulation -> ExplicitEuler ->
// NavierStokes -> {Incompressible, IntfDyn}) for real_t = double, driven from Python like the
// suite's other compiled solvers (cfd-gpu `pnm_backend`, packing-gpu `demgpu`). Particle arrays are
// numpy: positions/velocities (N,3) float64, masses/types/volumes (N,). Verb names match the suite
// convention (set_positions / get_positions / step / ...). See ../../docs/CONVENTIONS.md s.6.
//
// Build: configure with -DVORONOI_BUILD_PYTHON=ON (see python/CMakeLists.txt).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vorflow/simulation.hpp"
#include "vorflow/voronoi.hpp"

namespace py = pybind11;
using real_t = double;

// ---- numpy <-> std conversions ---------------------------------------------------------------
static std::vector<std::array<real_t, 3>> to_vec3(
    py::array_t<real_t, py::array::c_style | py::array::forcecast> a) {
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
// the cell's id). `volume` true -> per-cell Voronoi volume; false -> per-cell neighbour (facet)
// count.
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
  m.doc() =
      "vorflow: dynamic 3D Voronoi tessellation of moving particles (Python surface).\n\n"
      "Drive a moving-particle Voronoi simulation from Python: set the periodic box and the\n"
      "per-particle state (positions, velocities, masses), call init() to build the first\n"
      "tessellation, then step() the dynamics. The solver hierarchy is\n"
      "  Simulation -> ExplicitEuler -> NavierStokes -> {Incompressible, IntfDyn}\n"
      "with real_t = double. Particle arrays are numpy: positions/velocities (N,3) float64,\n"
      "masses/types/volumes (N,). Verb names follow the suite convention (set_*/get_*/step).";

  using Sim = vor::Simulation<real_t>;
  py::class_<Sim>(m, "Simulation",
                  "Base moving-particle Voronoi simulation: owns the periodic box, the particle\n"
                  "state and the cell complex, and exposes the per-cell tessellation observables.")
      .def(
          "set_l", [](Sim& s, std::array<real_t, 3> L) { s.setL(L); }, py::arg("L"),
          "Set the (cubic / Lees-Edwards) box dimensions L = (Lx, Ly, Lz).")
      .def("set_mass_density", &Sim::setMassDensity, py::arg("density"),
           "Set the uniform mass density used to assign particle masses when none are given.")
      .def(
          "set_masses", [](Sim& s, py::array_t<real_t> a) { s.setMasses(to_vec1<real_t>(a)); },
          py::arg("masses"), "Set per-particle masses, (N,).")
      .def(
          "set_positions", [](Sim& s, py::array_t<real_t> a) { s.setPositions(to_vec3(a)); },
          py::arg("positions"), "Set particle positions, (N,3).")
      .def(
          "set_velocities", [](Sim& s, py::array_t<real_t> a) { s.setVelocities(to_vec3(a)); },
          py::arg("velocities"), "Set particle velocities, (N,3).")
      .def("set_time", &Sim::setTime, py::arg("time"), "Set the current simulation time.")
      .def(
          "get_positions", [](Sim& s) { return from_vec3(s.getPositions()); },
          "Return particle positions, (N,3).")
      .def(
          "get_velocities", [](Sim& s) { return from_vec3(s.getVelocities()); },
          "Return particle velocities, (N,3).")
      .def("get_time", &Sim::getTime, "Return the current simulation time.")
      .def("get_kinetic_energy", &Sim::getKineticEnergy,
           "Return the total kinetic energy 1/2 sum m_i |v_i|^2.")
      .def("get_internal_energy", &Sim::getInternalEnergy,
           "Return the total internal (EOS) energy of the current tessellation.")
      .def("put_in_box", &Sim::putInBox, "Wrap particle positions back into the periodic box.")
      .def("init", &Sim::init, "Build the initial tessellation; returns False on failure.")
      .def("step", &Sim::step, py::arg("num_steps"), py::arg("dt"),
           "Advance the dynamics by num_steps timesteps of size dt.")
      .def(
          "get_volumes", [](Sim& s) { return cell_scalar(s, true); },
          "Per-particle Voronoi cell volume, (N,) in particle order (0 if no cell).")
      .def(
          "get_num_neighbors", [](Sim& s) { return cell_scalar(s, false); },
          "Per-particle Voronoi neighbour (facet) count, (N,) in particle order.")
      .def(
          "get_neighbor_pairs",
          [](Sim& s) {
            auto& cc = s.getCellComplex();
            const std::size_t np = s.getPositions().size();
            const std::size_t nc = cc.getGeometryArena().numCells();
            std::vector<int> flat;  // (M,2) row-major: each owned face (cell_id, neighbour_id)
            for (std::size_t i = 0; i < nc; ++i) {
              const auto cv = cc.getCellView(i);
              const auto id = cv.getID();
              if (static_cast<std::size_t>(id) >= np)
                continue;
              const unsigned nf = cv.numFacets();
              for (unsigned j = 0; j < nf; ++j) {
                const auto nb = cv.getNbr(j);
                if (static_cast<std::size_t>(nb) < np) {  // skip boundary/sentinel markers
                  flat.push_back(static_cast<int>(id));
                  flat.push_back(static_cast<int>(nb));
                }
              }
            }
            const py::ssize_t m = static_cast<py::ssize_t>(flat.size() / 2);
            py::array_t<int> a({m, py::ssize_t(2)});
            std::memcpy(a.mutable_data(), flat.data(), flat.size() * sizeof(int));
            return a;
          },
          "Voronoi connectivity as an (M,2) edge list (particle_id, neighbour_particle_id) over "
          "faces");

  py::class_<vor::ExplicitEuler<real_t>, Sim>(
      m, "ExplicitEuler",
      "Compressible-Euler dynamics: velocity-Verlet integration of the per-cell pressure\n"
      "force from an isothermal EOS (press_i = pressEq * volAvg / vol_i).")
      .def(py::init<>())
      .def(
          "set_types",
          [](vor::ExplicitEuler<real_t>& s, py::array_t<std::uint8_t> a) {
            s.setTypes(to_vec1<std::uint8_t>(a));
          },
          py::arg("types"), "Set the per-particle phase/type label (uint8).")
      .def("set_pressure", &vor::ExplicitEuler<real_t>::setPressure, py::arg("pressure"),
           "Set the equilibrium pressure constant of the EOS.")
      // force/integrate split (for distributed scheme-C drivers): the caller does the Verlet
      // kick/drift and inserts a halo force exchange between recompute_forces() and the kick.
      .def(
          "get_forces", [](vor::ExplicitEuler<real_t>& s) { return from_vec3(s.getForces()); },
          "Return the per-particle force, (N,3).")
      .def(
          "set_forces",
          [](vor::ExplicitEuler<real_t>& s, py::array_t<real_t> a) { s.getForces() = to_vec3(a); },
          py::arg("forces"),
          "Overwrite the per-particle force, (N,3) — used by distributed drivers that exchange\n"
          "forces over a halo between recompute_forces() and the Verlet kick.")
      .def("recompute_forces", &vor::ExplicitEuler<real_t>::recomputeForces,
           "Incrementally update the tessellation from current positions and recompute forces.");

  py::class_<vor::NavierStokes<real_t>, vor::ExplicitEuler<real_t>>(
      m, "NavierStokes",
      "Viscous Navier-Stokes dynamics: ExplicitEuler plus a viscous stress assembled from the\n"
      "per-cell velocity gradient.")
      .def(py::init<>())
      .def("set_viscosity", &vor::NavierStokes<real_t>::setViscosity, py::arg("viscosity"),
           "Set the uniform shear viscosity.")
      .def("set_bulk_viscosity", &vor::NavierStokes<real_t>::setBulkViscosity, py::arg("viscosity"),
           "Set the uniform bulk viscosity.")
      .def(
          "set_viscosities",
          [](vor::NavierStokes<real_t>& s, py::array_t<real_t> a) {
            s.setViscosities(to_vec1<real_t>(a));
          },
          py::arg("viscosities"), "Set per-particle shear viscosities, (N,).")
      .def(
          "set_bulk_viscosities",
          [](vor::NavierStokes<real_t>& s, py::array_t<real_t> a) {
            s.setBulkViscosities(to_vec1<real_t>(a));
          },
          py::arg("viscosities"), "Set per-particle bulk viscosities, (N,).")
      .def("set_ext_force_dens", &vor::NavierStokes<real_t>::setExtForceDens,
           py::arg("force_density"), "Set a uniform external body-force density (e.g. gravity).");

  py::class_<vor::Incompressible<real_t>, vor::NavierStokes<real_t>>(
      m, "Incompressible", "Divergence-free (pressure-projection) variant of the viscous solver.")
      .def(py::init<>())
      .def("build_constraint_matrix", &vor::Incompressible<real_t>::buildConstraintMatrix,
           "Assemble the elliptic pressure constraint matrix for the projection step.");

  py::class_<vor::IntfDyn<real_t>, vor::NavierStokes<real_t>>(
      m, "IntfDyn", "Multiphase dynamics with interface (surface) tension between particle phases.")
      .def(py::init<>())
      .def("set_intf_tension", &vor::IntfDyn<real_t>::setIntfTension, py::arg("tension"),
           py::arg("i"), py::arg("j"), "Set the interface tension between phases i and j.")
      .def("get_intf_energy", &vor::IntfDyn<real_t>::getIntfEnergy,
           "Return the total interfacial energy of the current configuration.");
}
