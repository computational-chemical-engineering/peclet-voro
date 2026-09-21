/**
 * @file voro_bindings.cpp
 * @brief Kokkos nanobind Python module `peclet.voro`.
 *
 * Drives the production device path — multicore CPU (OpenMP), or GPU (CUDA/HIP), selected by the
 * Kokkos backend the extension was built against — from Python. The PUBLIC surface (what a user of
 * the method needs to set up, run and read out a computation; suite/docs/QUALITY_PLAN.md D2):
 *
 *  - @ref Tess "peclet.voro.Tessellation" — the bare moving-particle (power-)Voronoi tessellator,
 *    optionally SDF-clipped: a cold build plus the incremental two-pass *repair* update (the fast
 *    per-step path for moving points); volumes, neighbour / wall counts, and the energy layer.
 *  - @ref Flow "peclet.voro.FlowSolver" — the static collocated / covolume Navier–Stokes solvers on
 *    the face mesh of a resident tessellation.
 *  - @ref Sim "peclet.voro.Simulation" — the device-native compressible-Euler / Navier–Stokes
 *    moving-cell fluid (velocity-Verlet over the tessellation).
 *  - `optimize_volume_mesh`, `minimize_interface` — the mesh optimisers on a periodic box.
 *  - `peclet.voro.pore_mesh` (a lazily imported submodule) carries the SDF-walled pore-space
 *    family bound here (`optimize_pore_mesh`, `sdf_voronoi_cells`, `sdf_voronoi_section`).
 *  - under `PECLET_VORO_MPI`: `VoronoiHalo` (ghost gather) and `DistributedTessellation` (the
 *    distributed repair driver, `mpi/distributed_moving.hpp`).
 *
 * Every instrument, ablation switch and validity report lives on the object's `diagnostics`
 * sub-object (`t.diagnostics.build_report()`), one per class, holding a reference to its owner.
 *
 * String modes are validated against the accepted set and the error lists it; call-order
 * requirements are state checks that name the correct order; the literals the engine is driven
 * with are the named `defaults` below (also `peclet.voro.defaults`), and the docstrings are
 * generated from them so they cannot drift.
 *
 * Particle data crosses the boundary as NumPy arrays: positions/velocities are `(N,3)` float64,
 * scalars (masses, viscosities, volumes) are `(N,)`. Arrays move through the shared
 * `peclet::core::python` bridge (core): returned arrays are backed by host buffers (no extra device
 * copy).
 *
 * Kokkos teardown follows the suite-wide pattern of peclet/core/python/kokkos_teardown.hpp: Kokkos
 * is initialized at import; every bound Tessellation / FlowSolver / Simulation is a `Releasable`
 * (registered on construction, release() drops its Views) and every zero-copy capsule of the shared
 * bridge is one too; the module's single `atexit` hook (also `peclet.voro.finalize()`) releases all
 * of them and THEN calls Kokkos::finalize, so an object still referenced at interpreter exit
 * (script globals, a Jupyter/Quarto kernel) can no longer be destroyed after finalize -- which is a
 * Kokkos::abort (SIGABRT / exit 134, on OpenMP as on CUDA). Nothing needs `del` before exit.
 *
 * Example
 * -------
 * @code{.py}
 *   import numpy as np, peclet.voro
 *   rng = np.random.default_rng(0)
 *   pos = rng.random((100_000, 3))            # uniform points in the unit box
 *   t = peclet.voro.Tessellation()
 *   t.set_domain(extent=(1.0, 1.0, 1.0))
 *   t.build(pos)                              # cold tessellation
 *   vol = t.get_volumes()                     # (N,) cell volumes; sum ~= box volume
 *   for _ in range(50):                       # move + repair each step (faster than rebuilding)
 *       pos = (pos + 1e-4 * rng.standard_normal(pos.shape)) % 1.0
 *       stats = t.step(pos)                   # {'flagged','pass1','pass2','rebuilt','fell_back'}
 *   nbr = t.get_neighbor_counts()             # (N,) Voronoi neighbours per cell
 * @endcode
 */
#include "voro_bindings_common.hpp"

using namespace peclet::voro::pybind;

NB_MODULE(_voro, m) {
  using namespace defaults;
  m.attr("__doc__") =
      "peclet.voro (device/Kokkos): moving-particle Voronoi tessellation and dynamics.\n\n"
      "Classes: Tessellation (cold build + incremental repair, volumes, neighbour counts, energy\n"
      "forces), FlowSolver (static Navier-Stokes on the face mesh), Simulation (moving-cell\n"
      "compressible-Euler / Navier-Stokes fluid); functions optimize_volume_mesh, "
      "minimize_interface;\n"
      "the pore-space family under peclet.voro.pore_mesh; VoronoiHalo and DistributedTessellation\n"
      "when built with MPI. Every instrument lives on the object's `diagnostics`. Arrays are "
      "NumPy:\n"
      "positions/velocities (N,3) float64, scalars (N,). The backend (Serial/OpenMP/CUDA/HIP) is\n"
      "fixed at build time; see peclet.voro.execution_space. peclet.voro.defaults lists the named\n"
      "defaults the engine is driven with.";
  // Kokkos init + the release-then-finalize atexit hook + finalize() + execution_space: the
  // suite-wide teardown pattern (file comment; peclet/core/python/kokkos_teardown.hpp). Every
  // Tess/Flow/Sim is a Releasable, so the registry releases them in one sweep before finalize.
  peclet::core::python::install(m);

  {
    nb::dict d;
    d["max_planes"] = kMaxPlanes;
    d["max_triangles"] = kMaxTriangles;
    d["pore_max_planes"] = kPoreMaxPlanes;
    d["pore_max_triangles"] = kPoreMaxTriangles;
    d["certificate_tolerance"] = kCertificateTolerance;
    d["skin"] = kSkin;
    d["search_window"] = kSearchWindow;
    d["sdf_gradient_step"] = kSdfGradientStep;
    d["wall_skin"] = kWallSkin;
    d["optimizer_search_window"] = kOptimizerSearchWindow;
    d["pore_search_window"] = kPoreSearchWindow;
    d["optimizer_max_iter"] = kOptimizerMaxIter;
    d["pore_max_iter"] = kPoreMaxIter;
    d["optimizer_tolerance"] = kOptimizerTolerance;
    d["optimizer_cg_iters"] = kOptimizerCgIters;
    d["pore_cg_iters"] = kPoreCgIters;
    d["barrier_decay"] = kBarrierDecay;
    d["interface_sigma"] = kInterfaceSigma;
    d["pore_max_bins"] = kPoreMaxBins;
    d["distributed_rcut"] = kDistributedRcut;
    d["distributed_cells"] = kDistributedCells;
    m.attr("defaults") = d;
  }

  // ---- typed results ---------------------------------------------------------------------------
  nb::class_<OptimizeResult>(m, "OptimizeResult",
                             "Result of optimize_volume_mesh / pore_mesh.optimize_pore_mesh.")
      .def_ro("positions", &OptimizeResult::positions, "The optimised seeds (N,3) float64.")
      .def_ro("weights", &OptimizeResult::weights,
              "The optimised power weights (N,) float64, or None when use_weights=False.")
      .def_ro("iters", &OptimizeResult::iters, "Gauss-Newton / descent iterations run.")
      .def_ro("max_vol_err", &OptimizeResult::max_vol_err,
              "max_i |V_i / V_ref,i - 1| at the returned seeds.")
      .def_ro("mean_vol_err", &OptimizeResult::mean_vol_err,
              "mean_i |V_i / V_ref,i - 1| at the returned seeds.")
      .def_ro("converged", &OptimizeResult::converged, "True if the gradient fell below tol.")
      .def_ro("num_empty", &OptimizeResult::num_empty,
              "Seeds whose cell is empty at the returned seeds (0 for a valid mesh).")
      .def("__repr__", [](const OptimizeResult& r) {
        return "OptimizeResult(iters=" + std::to_string(r.iters) +
               ", max_vol_err=" + fmt(r.max_vol_err) + ", mean_vol_err=" + fmt(r.mean_vol_err) +
               ", converged=" + (r.converged ? "True" : "False") +
               ", num_empty=" + std::to_string(r.num_empty) + ")";
      });
  nb::class_<InterfaceResult>(m, "InterfaceResult", "Result of minimize_interface.")
      .def_ro("positions", &InterfaceResult::positions, "The minimised seeds (N,3) float64.")
      .def_ro("energy", &InterfaceResult::energy,
              "Final interfacial energy E = sum sigma A_ij over faces between different types.")
      .def_ro("energy_ratio", &InterfaceResult::energy_ratio,
              "energy / the energy of the input seeds.")
      .def_ro("iters", &InterfaceResult::iters, "Descent iterations run.")
      .def_ro("converged", &InterfaceResult::converged, "True if the gradient fell below tol.")
      .def("__repr__", [](const InterfaceResult& r) {
        return "InterfaceResult(energy=" + fmt(r.energy) + ", energy_ratio=" + fmt(r.energy_ratio) +
               ", iters=" + std::to_string(r.iters) +
               ", converged=" + (r.converged ? "True" : "False") + ")";
      });

  bindOptimizers(m);
  bindPore(m);
  bindTessellation(m);
  bindFlow(m);
  bindSimulation(m);
#ifdef PECLET_VORO_MPI
  bindMpi(m);
#endif
}
