/// @file
/// @brief `peclet.voro.Tessellation`: the out-of-line face_mesh() + the class registration.
#include "voro_bindings_tess.hpp"

namespace peclet::voro::pybind {

// The one out-of-line member (see voro_bindings_tess.hpp).
peclet::voro::fv::FaceMesh<real_t> Tess::face_mesh() {
  if (N_ == 0)
    throw std::runtime_error(
        "voro: FlowSolver needs a built Tessellation — call tessellation.build(positions) "
        "first.");
  const int N = N_;
  const real_t Larr[3] = {L_[0], L_[1], L_[2]};
  return std::visit(
      [&](auto& mt) {
        auto view =
            peclet::voro::reevalPublish<real_t, defaults::kMaxPlanes, defaults::kMaxTriangles>(
                mt.store, pos_, mt.vol, N, Larr, mt.wall, mt.xRef);
        auto aux = peclet::voro::buildAuxMaps(view);
        return peclet::voro::fv::buildFaceMesh(view, aux);
      },
      mt_);
}

void bindTessellation(nb::module_& m) {
  using namespace defaults;
  // ---- Tessellation -----------------------------------------------------------------------------
  nb::class_<TessDiagnostics>(
      m, "TessellationDiagnostics",
      "Instruments and ablation switches of a Tessellation (reached as `t.diagnostics`).")
      .def(
          "build_report", [](TessDiagnostics& d) { return d.t->build_report(); },
          "Validity counts of the last build: {'buried', 'reach_exceeded', 'empty', 'overflow',\n"
          "'incomplete'} — all zero for a guaranteed-exact partition (build() already warns, or\n"
          "raises with strict=True, when they are not) — and 'over_buffer_rebuilds', the number "
          "of\n"
          "times the build's facet/edge over-buffer estimate was exceeded and the build pass "
          "re-run\n"
          "at the exact demand (0 normally; each one doubles that build's cost).")
      .def(
          "set_profile", [](TessDiagnostics& d, bool on) { d.t->set_profile(on); },
          nb::arg("on") = true,
          "Print the cold build's timing (grid / build / CSR), the worklist size, the over-buffer\n"
          "rebuilds and the max facets per cell on stderr (default off). Takes effect at the next\n"
          "build().")
      .def(
          "set_local_certificate",
          [](TessDiagnostics& d, bool on) { d.t->set_local_certificate(on); }, nb::arg("on"),
          "Ablation: the cheap O(nt) Lawson local certificate (default True) vs the brute "
          "O(nt*np)\n"
          "form for detecting which cells changed. Both are complete; local is faster. Takes "
          "effect\nat the next build().")
      .def(
          "set_gate", [](TessDiagnostics& d, bool on) { d.t->set_gate(on); }, nb::arg("on"),
          "Ablation: the adaptive gate (default True) that routes high-churn steps straight to a\n"
          "full rebuild — the 'never much slower than a cold build' guard. Takes effect at the "
          "next\nbuild().");

  nb::class_<Tess>(
      m, "Tessellation",
      "Moving-particle (power-)Voronoi tessellator on the device path, optionally clipped by an\n"
      "SDF solid.\n\n"
      "Build a tessellation once (`build`) then advance it cheaply as the points move (`step`) —\n"
      "the incremental two-pass repair is several times faster than rebuilding for the small\n"
      "per-step displacements typical of CFD/DEM, and falls back to a full rebuild (via an\n"
      "adaptive gate) when displacements are large, so it is never much slower than a cold\n"
      "build. Periodic box anchored at the origin. Single domain (one process); see\n"
      "DistributedTessellation for the MPI driver. Instruments: `diagnostics`.")
      .def(nb::init<>(),
           "Tessellation(): takes no arguments. Configure with `set_domain` (required) and "
           "optionally `set_tolerance` / `set_geometry` / `set_wall_mode` / `set_weights` before "
           "the first `build(positions)`.")
      .def("set_domain", &Tess::set_domain, nb::arg("extent"),
           nb::arg("origin") = std::array<real_t, 3>{0, 0, 0},
           nb::arg("periodic") = std::array<bool, 3>{true, true, true},
           "Set the periodic box before `build`: `extent` is the box SIZE (Lx, Ly, Lz). The "
           "suite-wide spelling (suite/docs/NAMING.md 1.1), the same call "
           "`dem.Simulation.set_domain` takes. "
           "`origin` must be (0, 0, 0) and `periodic` (True, True, True) — this engine's box is "
           "anchored at the origin and periodic on every axis; both are checked rather than "
           "ignored, so a caller who writes the suite-wide form gets an error naming the "
           "limitation.")
      .def_prop_ro("extent", &Tess::extent,
                   "The box size (Lx, Ly, Lz) — read-only; set it with `set_domain`.")
      .def("set_tolerance", &Tess::set_tolerance, nb::arg("frac") = kCertificateTolerance,
           doc("Certificate tolerance of the repair as a fraction of the mean inter-particle "
               "spacing\n(default " +
               fmt(kCertificateTolerance) +
               " = peclet.voro.defaults['certificate_tolerance']). A vertex poking past a "
               "stored\nplane by more than this flags the cell for repair; smaller is stricter "
               "(closer to\nmachine-exact) at marginally higher cost. Takes effect at the next "
               "build()."))
      .def("set_geometry", &Tess::set_geometry, nb::arg("node_ints"), nb::arg("node_reals"),
           nb::arg("root") = 0, nb::arg("grad_h") = kSdfGradientStep,
           doc("Clip the cells by an SDF solid given as a core shape scene in the flat node "
               "encoding\n(node_ints int32 (3 per node), node_reals float64 (16 per node)) — "
               "exactly "
               "what\npeclet.core.geom.Scene.encode() returns and dem.add_analytic_wall takes; "
               "`root` is the\ntree root to evaluate. Suite sign convention: sdf < 0 inside the "
               "solid. Seeds inside the\nsolid get no cell (volume 0); cells reaching into it gain "
               "wall facets. Applies to the\nnext `build` and is carried through every `step` "
               "(wall planes are resident; a boundary\nwatch re-clips cells at the wall). `grad_h` "
               "(default " +
               fmt(kSdfGradientStep) +
               ") is the central-difference step for the\nSDF gradient. Analytic vocabulary only "
               "(no sampled grids through this path yet)."))
      .def("clear_geometry", &Tess::clear_geometry,
           "Drop the SDF geometry (takes effect at the next `build`).")
      .def("set_wall_mode", &Tess::set_wall_mode, nb::arg("mode"), nb::arg("skin_frac") = kWallSkin,
           doc("Wall re-gather policy for `step`. mode: one of " + std::string(kWallModeList) +
               ". 'exact' (the default)\nre-clips every wall-clipped cell that moved, so the "
               "incremental result equals a cold\nrebuild; 'skin' keeps a cell's stale tangent "
               "planes until it moved more than skin_frac x\nthe mean spacing (default " +
               fmt(kWallSkin) +
               "; cheaper, not exact by construction). Takes effect at the\nnext build()."))
      .def(
          "set_weights", &Tess::set_weights, nb::arg("weights"),
          "Per-seed POWER (Laguerre) weights (N,) float64: the cells become the power diagram\n"
          "(radical planes) instead of the Voronoi diagram. Takes effect at the next `build`;\n"
          "call again before a `step` to update the weights alongside the positions. Exact in the\n"
          "small-weight regime (see the docs).")
      .def("clear_weights", &Tess::clear_weights,
           "Back to the unweighted Voronoi diagram (next `build`).")
      .def("build", &Tess::build, nb::arg("positions"), nb::arg("strict") = false,
           "Cold-build the (power-)Voronoi tessellation of `positions` (N,3) from scratch and make "
           "it resident, clipped by the geometry from `set_geometry` if any.\n"
           "Sets the particle count N for subsequent `step` calls. Warns (raises if strict=True)\n"
           "when the result is not a guaranteed-exact partition: buried power cells (a seed "
           "outside\n"
           "its own cell — never for w = r² of non-overlapping spheres), a search reach beyond "
           "half\n"
           "the box, or overflowed cells; see `diagnostics.build_report()`.")
      .def(
          "step", &Tess::step, nb::arg("positions"),
          "Incrementally repair the resident tessellation to new `positions` (N,3, same N as "
          "`build`;\nraises before `build`).\n"
          "Returns a dict of per-step work stats: 'flagged' (cells the certificate flagged), "
          "'pass1'\n"
          "and 'pass2' (cells re-gathered in each pass), 'extra' (cells gathered across verify "
          "extra-passes),\n"
          "'surgical' (Pass-1 cells repaired surgically), 'verify_passes' (verify iterations run), "
          "'rebuilt'\n"
          "(True if the gate routed this step to a full rebuild), 'fell_back' (True if the verify "
          "failed and\n"
          "a cold rebuild was forced), 'wall_flagged' (cells the SDF boundary watch re-clipped).")
      .def("get_volumes", &Tess::get_volumes,
           "Per-particle Voronoi cell volume (N,) float64 (a copy). Sums to the box volume "
           "(space-filling).")
      .def("get_neighbor_counts", &Tess::get_neighbor_counts,
           "Per-particle Voronoi neighbour count (N,) int32 (a copy) — the number of faces of each "
           "cell (wall facets included).")
      .def("get_wall_counts", &Tess::get_wall_counts,
           "Per-particle number of resident SDF wall planes (N,) int32 (a copy); all zero without "
           "geometry.")
      .def(
          "energy_forces", &Tess::energy_forces, nb::arg("types"), nb::arg("tension"),
          nb::arg("sigma_wall") = nb::none(), nb::arg("dEdV") = nb::none(), nb::arg("lloyd") = 0.0,
          nb::arg("facet_tension") = 0.0,
          "Energies and their exact gradients on the RESIDENT cells (after build/step), no "
          "rebuild:\n"
          "  interfacial  E = Σ σ(t_i,t_j) A_ij over facets between different `types` (N,) int32,\n"
          "               with the symmetric `tension` table (nTypes, nTypes) float64;\n"
          "  wetting      E = Σ σ_wall(t_i) A_wall,i over SDF wall facets, if `sigma_wall` "
          "(nTypes,)\n"
          "               is given (a uniform wall tension is a constant — only the species\n"
          "               difference does work, which is what sets the contact angle);\n"
          "  volume       Σ e_i(V_i) for a caller-supplied e'(V_i) = `dEdV` (N,) (e.g. "
          "2(V/Vref−1)/Vref);\n"
          "  centroidal   `lloyd` · Σ ∫_cell |y − x_i|² (Lloyd/CVT; gradient 2V(x−c) drives seeds "
          "to\n"
          "               their centroids — the skewness the grid solver's two-point operators "
          "need gone);\n"
          "  roundness    `facet_tension` · Σ A_f over all interior faces.\n"
          "Returns {'interface_energy', 'wall_energy', 'lloyd_energy', 'tension_energy', 'force' "
          "(N,3) = dE/dx,\n'force_w' (N,) = dE/dw when weights are set}. Descend along −force to "
          "minimise.")
      .def_prop_ro("num_particles", &Tess::num_particles,
                   "Particle count N set by the last `build` (0 before it).")
      .def_prop_ro(
          "diagnostics", [](Tess& t) { return TessDiagnostics{&t}; }, nb::keep_alive<0, 1>(),
          "The diagnostics tier: build_report(), set_local_certificate(), set_gate(), "
          "set_profile().");
}
}  // namespace peclet::voro::pybind
