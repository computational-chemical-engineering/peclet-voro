/// @file
/// @brief The Lagrangian power-cell method's surface (M7 NHV-W): the Tessellation's facet export
/// and weight solve (Tess members declared in voro_bindings_tess.hpp, defined here so the export,
/// the operators and the weight solve instantiate in this TU only) and `FacetOperators`.
#include <optional>

#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/operators.hpp"
#include "peclet/voro/lagrangian/weight_solve.hpp"
#include "voro_bindings_tess.hpp"

namespace peclet::voro::pybind {

namespace lag = peclet::voro::lagrangian;

lag::CellFacets<real_t> Tess::cell_facets() {
  if (N_ == 0)
    throw std::runtime_error(
        "voro: the facet export needs a built Tessellation — call "
        "build(positions) first.");
  if (weighted_ && wDirty_)
    throw std::runtime_error(
        "voro: set_weights() was called after the last build/step — the "
        "resident cells do not reflect those weights; build or step first.");
  const real_t Larr[3] = {L_[0], L_[1], L_[2]};
  return std::visit(
      [&](auto& mt) {
        using T = std::decay_t<decltype(mt)>;
        return lag::exportCellFacets<real_t, typename T::PlanePolicy>(
            mt.store, pos_, weight_, N_, Larr, mt.wall, mt.xRef, mt.vol);
      },
      mt_);
}

nb::dict Tess::get_facets() {
  const auto F = cell_facets();
  const auto off = peclet::core::toVector(F.facetOffset);
  const auto nbr = peclet::core::toVector(F.nbr);
  const auto area = peclet::core::toVector(F.area);
  const auto cen = peclet::core::toVector(F.centroid);
  const auto conn = peclet::core::toVector(F.conn);
  const auto sec = peclet::core::toVector(F.second);
  const auto x = peclet::core::toVector(pos_);
  std::vector<int> vi, vj;
  std::vector<real_t> va, vc, vd, vs;
  for (int i = 0; i < N_; ++i)
    for (int f = off[i]; f < off[i + 1]; ++f) {
      const int j = nbr[f];
      if (j == i)
        throw std::runtime_error(
            "voro: get_facets(): a cell borders its own periodic image — the box is too small "
            "for the point count.");
      if (j < i)
        continue;  // the facet is exported once, from its lower cell
      vi.push_back(i);
      vj.push_back(j);
      va.push_back(area[f]);
      for (int a = 0; a < 3; ++a) {
        vc.push_back(x[3 * i + a] + cen[3 * f + a]);
        vd.push_back(conn[3 * f + a]);
      }
      for (int e = 0; e < 9; ++e)
        vs.push_back(sec[9 * f + e]);
    }
  const std::size_t n = vi.size();
  nb::dict d;
  d["i"] = peclet::core::python::vector_to_ndarray(std::move(vi), {n}, {1});
  d["j"] = peclet::core::python::vector_to_ndarray(std::move(vj), {n}, {1});
  d["area"] = peclet::core::python::vector_to_ndarray(std::move(va), {n}, {1});
  d["centroid"] = peclet::core::python::vector_to_ndarray(std::move(vc), {n, 3}, {3, 1});
  d["dvec"] = peclet::core::python::vector_to_ndarray(std::move(vd), {n, 3}, {3, 1});
  d["second_moment"] = peclet::core::python::vector_to_ndarray(std::move(vs), {n, 3, 3}, {9, 3, 1});
  return d;
}

nb::dict Tess::get_wall_facets() {
  const auto F = cell_facets();
  const auto off = peclet::core::toVector(F.wallOffset);
  const auto x = peclet::core::toVector(pos_);
  auto area = peclet::core::toVector(F.wallArea);
  auto cen = peclet::core::toVector(F.wallCentroid);
  auto nrm = peclet::core::toVector(F.wallNormal);
  std::vector<int> vi(area.size());
  for (int i = 0; i < N_; ++i)
    for (int f = off[i]; f < off[i + 1]; ++f) {
      vi[f] = i;
      for (int a = 0; a < 3; ++a)
        cen[3 * f + a] += x[3 * i + a];
    }
  const std::size_t n = area.size();
  nb::dict d;
  d["i"] = peclet::core::python::vector_to_ndarray(std::move(vi), {n}, {1});
  d["area"] = peclet::core::python::vector_to_ndarray(std::move(area), {n}, {1});
  d["centroid"] = peclet::core::python::vector_to_ndarray(std::move(cen), {n, 3}, {3, 1});
  d["normal"] = peclet::core::python::vector_to_ndarray(std::move(nrm), {n, 3}, {3, 1});
  return d;
}

nb::dict Tess::get_cell_moments() {
  const auto F = cell_facets();
  const auto x = peclet::core::toVector(pos_);
  auto vol = peclet::core::toVector(F.volume);
  auto cen = peclet::core::toVector(F.cellCentroid);
  for (std::size_t q = 0; q < cen.size(); ++q)
    cen[q] += x[q];
  nb::dict d;
  d["volume"] = toNumpy1(std::move(vol));
  d["centroid"] = toNumpy3(std::move(cen));
  return d;
}

nb::ndarray<nb::numpy, real_t> Tess::get_weights() {
  if (!weighted_)
    return toNumpy1(std::vector<real_t>(static_cast<std::size_t>(N_), real_t(0)));
  return toNumpy1(wHost_);
}

nb::dict Tess::solve_weights(nb::ndarray<real_t, nb::c_contig> volumes, real_t tol, int max_iter) {
  if (N_ == 0 || !weighted_)
    throw std::runtime_error(
        "voro: solve_weights() needs a POWER tessellation: call set_weights(initial guess, e.g. "
        "zeros) and build(positions) first.");
  std::vector<real_t> v0 = flatten1(volumes);
  if (static_cast<int>(v0.size()) != N_)
    throw std::runtime_error("voro: solve_weights(volumes) must be (N,)");
  double s0 = 0, s = 0;
  for (real_t v : v0) {
    if (!(v > 0))
      throw std::invalid_argument("voro: solve_weights(volumes) must be positive");
    s0 += v;
  }
  for (real_t v : peclet::core::toVector(Kokkos::subview(
           std::visit([](auto& mt) { return mt.vol; }, mt_), std::make_pair(0, N_))))
    s += v;
  if (std::fabs(s0 / s - 1.0) > 1e-10)
    throw std::invalid_argument(
        "voro: solve_weights(volumes) must sum to the tessellated volume (" + std::to_string(s) +
        "), got " + std::to_string(s0) + " — the weight shift is the only free constant.");
  if (wDirty_) {
    Kokkos::deep_copy(weight_,
                      Kokkos::View<const real_t*, Kokkos::HostSpace>(wHost_.data(), wHost_.size()));
    wDirty_ = false;
  }
  lag::WeightSolveParams prm;
  prm.tol = tol;
  prm.maxIter = max_iter;
  const real_t Larr[3] = {L_[0], L_[1], L_[2]};
  lag::WeightSolveResult R = std::visit(
      [&](auto& mt) -> lag::WeightSolveResult {
        using T = std::decay_t<decltype(mt)>;
        if constexpr (std::is_same_v<typename T::PlanePolicy, peclet::voro::Power>)
          return lag::solveWeights(mt, pos_, weight_, v0, Larr, prm);
        else
          throw std::logic_error("voro: weighted Tessellation without a power engine");
      },
      mt_);
  wHost_ = peclet::core::toVector(weight_);
  nb::list hist;
  for (const auto& st : R.history) {
    nb::dict h;
    h["max_rel_err"] = st.maxRelErr;
    h["alpha"] = st.alpha;
    h["pcg_iterations"] = st.pcgIters;
    h["pcg_rel_residual"] = st.pcgRelRes;
    h["halvings"] = st.halvings;
    hist.append(h);
  }
  nb::dict d;
  d["iterations"] = R.iters;
  d["max_rel_err"] = R.maxRelErr;
  d["converged"] = R.converged;
  d["roundoff_floor"] = R.roundoffFloor;
  d["history"] = hist;
  return d;
}

// --------------------------------------------------------------------------------------------------
// FacetOperators: J_x, J_xᵀ, H, C, Cᵀ, G^s and ζ of a snapshot of the resident cells.
// --------------------------------------------------------------------------------------------------
class FacetOperators : public peclet::core::python::Releasable {
 public:
  FacetOperators(Tess& t, std::optional<nb::ndarray<real_t, nb::c_contig>> viscosity) {
    init(t, viscosity ? flatten1(*viscosity) : std::vector<real_t>{real_t(1)});
  }
  FacetOperators(Tess& t, real_t viscosity) { init(t, std::vector<real_t>{viscosity}); }

  nb::ndarray<nb::numpy, real_t> apply(const std::string& op, nb::ndarray<real_t, nb::c_contig> x) {
    const std::size_t N = static_cast<std::size_t>(F_.nCells);
    std::vector<real_t> h = flatten1(x);
    const bool vecIn = op == "volume_jacobian" || op == "divergence" || op == "dissipation" ||
                       op == "cell_gradient";
    const bool scalIn = op == "volume_jacobian_t" || op == "weight_hessian" || op == "divergence_t";
    if (!vecIn && !scalIn)
      throw std::invalid_argument("voro: FacetOperators.apply(op='" + op +
                                  "') is not an operator; " + kOps);
    if (h.size() != (vecIn ? 3 * N : N))
      throw std::runtime_error("voro: FacetOperators.apply('" + op + "') takes an array of " +
                               std::string(vecIn ? "(N,3)" : "(N,)") + " values");
    DView in = peclet::core::toDevice<real_t>(h, "fo.in");
    if (op == "volume_jacobian" || op == "divergence" || op == "weight_hessian") {
      DView out("fo.out", N);
      if (op == "volume_jacobian")
        lag::applyVolumeJacobian(F_, in, out);
      else if (op == "divergence")
        lag::applyDivergence(F_, in, out);
      else
        lag::applyWeightHessian(F_, in, out);
      return toNumpy1(peclet::core::toVector(out));
    }
    if (op == "cell_gradient") {
      DView out("fo.out", 9 * N);
      lag::cellGradient(F_, in, out);
      return peclet::core::python::vector_to_ndarray(peclet::core::toVector(out), {N, 3, 3},
                                                     {9, 3, 1});
    }
    DView out("fo.out", 3 * N);
    if (op == "volume_jacobian_t")
      lag::applyVolumeJacobianT(F_, in, out);
    else if (op == "divergence_t")
      lag::applyDivergenceT(F_, in, out);
    else
      lag::applyDissipation(F_, mu_, in, out, scr_);
    return toNumpy3(peclet::core::toVector(out));
  }

  int num_cells() const { return F_.nCells; }
  int num_facets() const { return F_.nFacets; }

  void release() noexcept override {
    F_ = lag::CellFacets<real_t>{};
    mu_ = DView{};
    scr_ = lag::DissipationScratch<real_t>{};
  }

  // A scalar viscosity arrives as a one-element vector.
  void init(Tess& t, std::vector<real_t> mu) {
    if (t.has_geometry())
      throw std::runtime_error(
          "voro: FacetOperators are periodic-only for now (the wall terms of C and of the "
          "Rayleigh function are not implemented); clear_geometry() and rebuild.");
    F_ = t.cell_facets();
    const int N = F_.nCells;
    if (static_cast<int>(mu.size()) == 1)
      mu.assign(static_cast<std::size_t>(N), mu[0]);
    if (static_cast<int>(mu.size()) != N)
      throw std::runtime_error("voro: FacetOperators(viscosity=) must be a scalar or (N,)");
    for (real_t m : mu)
      if (!(m > real_t(0)))
        throw std::invalid_argument("voro: FacetOperators(viscosity=) must be positive");
    mu_ = peclet::core::toDevice<real_t>(mu, "fo.mu");
    scr_ = lag::DissipationScratch<real_t>(N);
  }

  static constexpr const char* kOps =
      "accepted: 'volume_jacobian', 'volume_jacobian_t', 'weight_hessian', 'divergence', "
      "'divergence_t', 'cell_gradient', 'dissipation'.";

 private:
  lag::CellFacets<real_t> F_;
  DView mu_;
  lag::DissipationScratch<real_t> scr_;
};

void bindLagrangian(nb::module_& m) {
  nb::class_<FacetOperators>(
      m, "FacetOperators",
      "The cell operators of the Lagrangian power-cell method on a SNAPSHOT of a built\n"
      "Tessellation's cells (periodic box; Voronoi or power): matrix-free facet loops on the\n"
      "device. Velocities are (N,3), cell scalars (N,). `apply(op, x)` with op one of\n"
      "  'volume_jacobian'   J_x v = ∂V/∂x|_w · v                      (N,3) -> (N,)\n"
      "  'volume_jacobian_t' J_xᵀ p                                    (N,)  -> (N,3)\n"
      "  'weight_hessian'    H w = ∂V/∂w · w = Σ A/(2d)(w_i − w_j)     (N,)  -> (N,)\n"
      "  'divergence'        C v = Σ A (c − x_i)·(v_j − v_i)/d         (N,3) -> (N,)\n"
      "  'divergence_t'      Cᵀ l                                      (N,)  -> (N,3)\n"
      "  'cell_gradient'     G^s_i, ∂v_a/∂x_b                          (N,3) -> (N,3,3)\n"
      "  'dissipation'       ζ v, the Hessian of the face-gradient Rayleigh function\n"
      "                      R = Σ_f μ_f (A d/3) |sym G_f|² = ½ vᵀζv  (N,3) -> (N,3)\n"
      "`viscosity` (scalar or (N,), default 1) is the cell viscosity; facets take the harmonic\n"
      "mean. The snapshot does not follow later build/step calls — construct a new one.")
      .def(nb::init<Tess&, real_t>(), nb::arg("tessellation"), nb::arg("viscosity"))
      .def(nb::init<Tess&, std::optional<nb::ndarray<real_t, nb::c_contig>>>(),
           nb::arg("tessellation"), nb::arg("viscosity") = nb::none())
      .def("apply", &FacetOperators::apply, nb::arg("op"), nb::arg("x"),
           "Apply operator `op` to `x` (see the class doc); returns a new array.")
      .def_prop_ro("num_cells", &FacetOperators::num_cells)
      .def_prop_ro("num_facets", &FacetOperators::num_facets,
                   "Directed interior facets (each interior facet counted from both cells).");
}
}  // namespace peclet::voro::pybind
