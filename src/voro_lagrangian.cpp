/// @file
/// @brief The Lagrangian power-cell method's surface (M7 NHV-W): the Tessellation's facet export
/// and weight solve (Tess members declared in voro_bindings_tess.hpp, defined here so the export,
/// the operators and the weight solve instantiate in this TU only) and `FacetOperators`.
#include <cmath>
#include <limits>
#include <memory>
#include <optional>

#include "peclet/voro/lagrangian/facets.hpp"
#include "peclet/voro/lagrangian/integrator.hpp"
#include "peclet/voro/lagrangian/operators.hpp"
#include "peclet/voro/lagrangian/potential.hpp"
#include "peclet/voro/lagrangian/weight_solve.hpp"
#include "voro_bindings_tess.hpp"
// after Kokkos_Core (MORTON_HD -> KOKKOS_FUNCTION)
#include "morton/morton.hpp"

namespace peclet::voro::pybind {

namespace lag = peclet::voro::lagrangian;

namespace {
/// The record order of a per-facet export in the caller's labels: facets grouped by the caller
/// index of their exporting cell `ci` (ascending, stable — the identity when nothing was
/// relabelled).
std::vector<std::size_t> callerRecordOrder(const std::vector<int>& ci, bool relabelled) {
  std::vector<std::size_t> o(ci.size());
  for (std::size_t q = 0; q < o.size(); ++q)
    o[q] = q;
  if (relabelled)
    std::stable_sort(o.begin(), o.end(),
                     [&](std::size_t a, std::size_t b) { return ci[a] < ci[b]; });
  return o;
}
/// The Morton (Z) order of the points `x` (3N) in the box [0, L): order[k] is the index of the k-th
/// point along the curve (21 bits per axis, x fastest; ties keep their index order).
std::vector<int> mortonOrder(const std::vector<real_t>& x, const std::array<real_t, 3>& L) {
  using Code = morton::Morton3D21;
  const std::size_t N = x.size() / 3;
  const double scale = static_cast<double>(Code::coord_max) + 1.0;
  std::vector<Code::code_type> key(N);
  for (std::size_t i = 0; i < N; ++i) {
    Code::coord_type q[3];
    for (int a = 0; a < 3; ++a) {
      const double f = std::floor(static_cast<double>(x[3 * i + a]) / L[a] * scale);
      q[a] = static_cast<Code::coord_type>(
          std::min(std::max(f, 0.0), static_cast<double>(Code::coord_max)));
    }
    key[i] = Code::encode(q[0], q[1], q[2]).code();
  }
  std::vector<int> order(N);
  for (std::size_t i = 0; i < N; ++i)
    order[i] = static_cast<int>(i);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
  return order;
}

template <class T>
std::vector<T> gatherRecords(const std::vector<T>& v, const std::vector<std::size_t>& o,
                             std::size_t s) {
  std::vector<T> out(v.size());
  for (std::size_t q = 0; q < o.size(); ++q)
    for (std::size_t c = 0; c < s; ++c)
      out[s * q + c] = v[s * o[q] + c];
  return out;
}
}  // namespace

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
  // caller labels (the identity unless a LagrangianFlow relabelled the cells)
  const auto lab = [&](int k) { return perm_.empty() ? k : perm_[static_cast<std::size_t>(k)]; };
  std::vector<int> vi, vj;
  std::vector<real_t> va, vc, vd, vs;
  for (int i = 0; i < N_; ++i)
    for (int f = off[i]; f < off[i + 1]; ++f) {
      const int j = nbr[f];
      if (j == i)
        throw std::runtime_error(
            "voro: get_facets(): a cell borders its own periodic image — the box is too small "
            "for the point count.");
      if (lab(j) < lab(i))
        continue;  // the facet is exported once, from its lower cell (caller labels)
      vi.push_back(lab(i));
      vj.push_back(lab(j));
      va.push_back(area[f]);
      for (int a = 0; a < 3; ++a) {
        vc.push_back(x[3 * i + a] + cen[3 * f + a]);
        vd.push_back(conn[3 * f + a]);
      }
      for (int e = 0; e < 9; ++e)
        vs.push_back(sec[9 * f + e]);
    }
  const std::size_t n = vi.size();
  if (!perm_.empty()) {  // records grouped by the caller's lower cell, as without relabelling
    const auto o = callerRecordOrder(vi, true);
    vi = gatherRecords(vi, o, 1);
    vj = gatherRecords(vj, o, 1);
    va = gatherRecords(va, o, 1);
    vc = gatherRecords(vc, o, 3);
    vd = gatherRecords(vd, o, 3);
    vs = gatherRecords(vs, o, 9);
  }
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
      vi[f] = perm_.empty() ? i : perm_[static_cast<std::size_t>(i)];
      for (int a = 0; a < 3; ++a)
        cen[3 * f + a] += x[3 * i + a];
    }
  const std::size_t n = area.size();
  if (!perm_.empty()) {  // records grouped by the caller's cell index
    const auto o = callerRecordOrder(vi, true);
    vi = gatherRecords(vi, o, 1);
    area = gatherRecords(area, o, 1);
    cen = gatherRecords(cen, o, 3);
    nrm = gatherRecords(nrm, o, 3);
  }
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
  d["volume"] = toNumpy1(cellsToCaller(perm_, std::move(vol), 1));
  d["centroid"] = toNumpy3(cellsToCaller(perm_, std::move(cen), 3));
  return d;
}

nb::ndarray<nb::numpy, real_t> Tess::get_weights() {
  if (!weighted_)
    return toNumpy1(std::vector<real_t>(static_cast<std::size_t>(N_), real_t(0)));
  return toNumpy1(cellsToCaller(perm_, wHost_, 1));
}

nb::dict Tess::solve_weights(nb::ndarray<real_t, nb::c_contig> volumes, real_t tol, int max_iter) {
  if (N_ == 0 || !weighted_)
    throw std::runtime_error(
        "voro: solve_weights() needs a POWER tessellation: call set_weights(initial guess, e.g. "
        "zeros) and build(positions) first.");
  std::vector<real_t> v0 = cellsToInternal(perm_, flatten1(volumes), 1);
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

namespace {
/// The validated `interface_energy` mode: 'facet' (the raw unlike-facet area, the default) or
/// 'nonlocal' (the moment-blob energy E_M, DESIGN_NOTE §12.3.3).
lag::InterfaceEnergy interfaceEnergyMode(const std::string& m, const char* where) {
  if (m == "facet")
    return lag::InterfaceEnergy::Facet;
  if (m == "nonlocal")
    return lag::InterfaceEnergy::Nonlocal;
  throw std::invalid_argument(std::string("voro: ") + where +
                              ": interface_energy must be 'facet' or 'nonlocal', got '" + m + "'");
}
}  // namespace

nb::dict Tess::reduced_potential(nb::ndarray<int, nb::c_contig> types,
                                 std::optional<nb::ndarray<real_t, nb::c_contig>> tension,
                                 std::optional<nb::ndarray<real_t, nb::c_contig>> sigma_wall,
                                 std::optional<nb::ndarray<real_t, nb::c_contig>> density,
                                 std::optional<std::array<real_t, 3>> gravity,
                                 real_t reference_density, real_t tol,
                                 const std::string& interface_energy,
                                 std::optional<nb::ndarray<real_t, nb::c_contig>> volumes) {
  if (N_ == 0 || !weighted_)
    throw std::runtime_error(
        "voro: reduced_potential() needs a built POWER tessellation (set_weights, build, "
        "usually solve_weights first): U~ is reduced through the weights.");
  const int N = N_;
  if (static_cast<int>(types.size()) != N)
    throw std::runtime_error("voro: reduced_potential(types) must be (N,)");
  int nT = 0;
  for (int i = 0; i < N; ++i) {
    if (types.data()[i] < 0)
      throw std::invalid_argument("voro: reduced_potential(types) must be >= 0");
    nT = std::max(nT, types.data()[i] + 1);
  }
  auto table = [&](const std::optional<nb::ndarray<real_t, nb::c_contig>>& a, std::size_t n,
                   const char* what) {
    if (!a)
      return DView{};
    if (a->size() < n)
      throw std::runtime_error(std::string("voro: reduced_potential(") + what +
                               ") is too short for max(types)+1 = " + std::to_string(nT) +
                               " types");
    return peclet::core::toDevice<real_t>(std::vector<real_t>(a->data(), a->data() + n), what);
  };
  const std::size_t sT = static_cast<std::size_t>(nT);
  if (tension && tension->size() != sT * sT)
    throw std::runtime_error("voro: reduced_potential(tension) must be (nTypes, nTypes)");
  const DView dten = table(tension, sT * sT, "tension");
  const DView dsw = table(sigma_wall, sT, "sigma_wall");
  const DView drho = table(density, sT, "density");
  if (density.has_value() != gravity.has_value())
    throw std::invalid_argument(
        "voro: reduced_potential() takes `density` and `gravity` together (facet gravity)");
  const lag::CellFacets<real_t> F = cell_facets();
  Kokkos::View<int*, Mem> dtype = peclet::core::toDevice<int>(
      cellsToInternal(perm_, std::vector<int>(types.data(), types.data() + N), 1), "rp.type");
  DView gx("rp.dUdx", 3 * static_cast<std::size_t>(N)), gw("rp.dUdw", N);
  lag::PotentialSpec<real_t> spec;
  spec.types = dtype;
  spec.nTypes = nT;
  spec.tension = dten;
  spec.sigmaWall = dsw;
  spec.density = drho;
  if (gravity)
    for (int a = 0; a < 3; ++a)
      spec.gravity[a] = (*gravity)[a];
  spec.rhoRef = reference_density;
  spec.interfaceEnergy = interfaceEnergyMode(interface_energy, "reduced_potential");
  if (spec.interfaceEnergy == lag::InterfaceEnergy::Nonlocal) {
    // V⁰ of E_M: the given reference volumes, else the current cell volumes
    std::vector<double> V0 =
        volumes
            ? cellsToInternal(
                  perm_, std::vector<double>(volumes->data(), volumes->data() + volumes->size()), 1)
            : peclet::core::toVector(F.volume);
    if (static_cast<int>(V0.size()) != N)
      throw std::runtime_error("voro: reduced_potential(volumes) must be (N,)");
    spec.refVolume = peclet::core::toDevice<real_t>(V0, "rp.V0");
    spec.nonlocal = lag::nonlocalParams<real_t>(V0);
  } else if (volumes) {
    throw std::invalid_argument(
        "voro: reduced_potential(volumes) is the V0 of interface_energy='nonlocal' only");
  }
  const real_t Larr[3] = {L_[0], L_[1], L_[2]};
  const lag::PotentialParts e = std::visit(
      [&](auto& mt) -> lag::PotentialParts {
        using T = std::decay_t<decltype(mt)>;
        if constexpr (std::is_same_v<typename T::PlanePolicy, peclet::voro::Power>)
          return lag::partitionPotential(mt, F, pos_, weight_, Larr, spec, gx, gw);
        else
          throw std::logic_error("voro: weighted Tessellation without a power engine");
      },
      mt_);
  lag::WeightSolveParams prm;
  prm.pcgTol = tol;
  DView pi("rp.pi", N), grad("rp.grad", 3 * static_cast<std::size_t>(N));
  const lag::AdjointResult R = lag::reducedGradient(F, gx, gw, pi, grad, prm);
  nb::dict d;
  d["interface_energy"] = e.interface;
  d["wall_energy"] = e.wall;
  d["gravity_energy"] = e.gravity;
  d["energy"] = e.total();
  d["gradient"] = toNumpy3(cellsToCaller(perm_, peclet::core::toVector(grad), 3));
  d["gradient_x"] = toNumpy3(cellsToCaller(perm_, peclet::core::toVector(gx), 3));
  d["gradient_w"] = toNumpy1(cellsToCaller(perm_, peclet::core::toVector(gw), 1));
  d["pi"] = toNumpy1(cellsToCaller(perm_, peclet::core::toVector(pi), 1));
  d["pcg_iterations"] = R.pcgIters;
  d["pcg_rel_residual"] = R.pcgRelRes;
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
    DView in = peclet::core::toDevice<real_t>(cellsToInternal(perm_, std::move(h), vecIn ? 3 : 1),
                                              "fo.in");
    if (op == "volume_jacobian" || op == "divergence" || op == "weight_hessian") {
      DView out("fo.out", N);
      if (op == "volume_jacobian")
        lag::applyVolumeJacobian(F_, in, out);
      else if (op == "divergence")
        lag::applyDivergence(F_, in, out);
      else
        lag::applyWeightHessian(F_, in, out);
      return toNumpy1(cellsToCaller(perm_, peclet::core::toVector(out), 1));
    }
    if (op == "cell_gradient") {
      DView out("fo.out", 9 * N);
      lag::cellGradient(F_, in, out);
      return peclet::core::python::vector_to_ndarray(
          cellsToCaller(perm_, peclet::core::toVector(out), 9), {N, 3, 3}, {9, 3, 1});
    }
    DView out("fo.out", 3 * N);
    if (op == "volume_jacobian_t")
      lag::applyVolumeJacobianT(F_, in, out);
    else if (op == "divergence_t")
      lag::applyDivergenceT(F_, in, out);
    else
      lag::applyDissipation(F_, mu_, in, out, scr_);
    return toNumpy3(cellsToCaller(perm_, peclet::core::toVector(out), 3));
  }

  int num_cells() const { return F_.nCells; }
  int num_facets() const { return F_.nFacets; }

  void release() noexcept override {
    F_ = lag::CellFacets<real_t>{};
    mu_ = DView{};
    scr_ = lag::DissipationScratch<real_t>{};
    perm_.clear();
  }

  // A scalar viscosity arrives as a one-element vector.
  void init(Tess& t, std::vector<real_t> mu) {
    F_ = t.cell_facets();
    perm_ = t.perm_;  // the snapshot's cell order (apply() maps through it)
    const int N = F_.nCells;
    if (static_cast<int>(mu.size()) == 1)
      mu.assign(static_cast<std::size_t>(N), mu[0]);
    else
      mu = cellsToInternal(perm_, std::move(mu), 1);
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
  std::vector<int> perm_;  ///< Tess::perm_ at the snapshot (empty: the caller's order)
};

// --------------------------------------------------------------------------------------------------
// LagrangianFlow: the NHV-W time integrator (lagrangian/integrator.hpp) driving a built POWER
// Tessellation in place (its positions, weights and resident cells move with the flow).
// --------------------------------------------------------------------------------------------------
class LagrangianFlow : public peclet::core::python::Releasable {
  template <class S>
  using Integ = lag::Integrator<MT<true, S>>;

 public:
  LagrangianFlow(Tess& t, nb::ndarray<int, nb::c_contig> types,
                 std::optional<nb::ndarray<real_t, nb::c_contig>> volumes)
      : t_(&t) {
    if (t.N_ == 0 || !t.weighted_)
      throw std::runtime_error(
          "voro: LagrangianFlow needs a built POWER Tessellation (set_weights, e.g. zeros, then "
          "build(positions)).");
    const int N = t.N_;
    if (static_cast<int>(types.size()) != N)
      throw std::runtime_error("voro: LagrangianFlow(types) must be (N,)");
    // per-cell state is held in the Tessellation's internal cell order (Tess::perm_)
    types_ = cellsToInternal(t.perm_, std::vector<int>(types.data(), types.data() + N), 1);
    relabels_ = t.relabels_;
    nT_ = 0;
    for (int ty : types_) {
      if (ty < 0)
        throw std::invalid_argument("voro: LagrangianFlow(types) must be >= 0");
      nT_ = std::max(nT_, ty + 1);
    }
    if (volumes) {
      V0_ = cellsToInternal(t.perm_, flatten1(*volumes), 1);
      if (static_cast<int>(V0_.size()) != N)
        throw std::runtime_error("voro: LagrangianFlow(volumes) must be (N,)");
    } else {
      DView vol = std::visit([](auto& mt) { return mt.vol; }, t.mt_);
      V0_ = peclet::core::toVector(Kokkos::subview(vol, std::make_pair(0, N)));
    }
    double s0 = 0, s = 0;
    for (real_t v : V0_) {
      if (!(v > 0))
        throw std::invalid_argument("voro: LagrangianFlow(volumes) must be positive");
      s0 += v;
    }
    DView vol = std::visit([](auto& mt) { return mt.vol; }, t.mt_);
    for (real_t v : peclet::core::toVector(Kokkos::subview(vol, std::make_pair(0, N))))
      s += v;
    if (std::fabs(s0 / s - 1.0) > 1e-10)
      throw std::invalid_argument(
          "voro: LagrangianFlow(volumes) must sum to the tessellated volume");
    rho_.assign(nT_, 1.0);
    mu_.assign(nT_, 0.0);
    vel_.assign(3 * static_cast<std::size_t>(N), 0.0);
  }

  // ---- physical inputs (per type; marking the state for a re-preparation) ----
  void set_density(nb::ndarray<real_t, nb::c_contig> v) {
    rho_ = perType(v, "density");
    for (double r : rho_)
      if (!(r > 0))
        throw std::invalid_argument("voro: LagrangianFlow.set_density() must be positive");
    dirty_ = true;
  }
  void set_viscosity(nb::ndarray<real_t, nb::c_contig> v) {
    mu_ = perType(v, "viscosity");
    for (double m : mu_)
      if (!(m >= 0))
        throw std::invalid_argument("voro: LagrangianFlow.set_viscosity() must be >= 0");
    dirty_ = true;
  }
  void set_tension(nb::ndarray<real_t, nb::c_contig> v) {
    if (v.size() != static_cast<std::size_t>(nT_ * nT_))
      throw std::runtime_error("voro: LagrangianFlow.set_tension() must be (nTypes, nTypes)");
    tension_.assign(v.data(), v.data() + v.size());
    dirty_ = true;
  }
  void set_interface_energy(const std::string& mode) {
    interfaceEnergy_ = interfaceEnergyMode(mode, "LagrangianFlow.set_interface_energy()");
    dirty_ = true;
  }
  void set_wall_tension(nb::ndarray<real_t, nb::c_contig> v) {
    sigmaWall_ = perType(v, "wall_tension");
    dirty_ = true;
  }
  void set_gravity(std::array<real_t, 3> g, std::optional<real_t> reference_density) {
    gravity_ = g;
    hasGravity_ = g[0] != 0 || g[1] != 0 || g[2] != 0;
    rhoRef_ = reference_density;
    dirty_ = true;
  }
  void set_body_force(std::array<real_t, 3> amplitude, std::array<real_t, 3> wavevector) {
    bodyAmp_ = amplitude;
    bodyWave_ = wavevector;
    dirty_ = true;
  }
  void set_dt(real_t dt) {
    if (!(dt >= 0))
      throw std::invalid_argument("voro: LagrangianFlow.set_dt() must be >= 0 (0: no cap)");
    dtUser_ = dt;
  }
  void set_cfl(real_t advective, real_t capillary) {
    if (!(advective >= 0) || !(capillary >= 0))
      throw std::invalid_argument("voro: LagrangianFlow.set_cfl() constants must be >= 0");
    cfl_ = advective;
    cap_ = capillary;
    dirty_ = true;  // the θ rule reads the step bound
  }
  void set_kick_constraint(const std::string& mode) {
    namespace lg = peclet::voro::lagrangian;
    if (mode == "midpoint")
      kick1_ = lg::KickConstraint::kMidpoint;
    else if (mode == "start")
      kick1_ = lg::KickConstraint::kStart;
    else if (mode == "pressure")
      kick1_ = lg::KickConstraint::kPressure;
    else
      throw std::invalid_argument(
          "voro: LagrangianFlow.set_kick_constraint(mode) takes 'midpoint' (C of kick 1 at x^n + "
          "dt/2 v^n, option S), 'start' (at x^n) or 'pressure' (no constraint: the last "
          "pressures in kick 1; the default)");
    dirty_ = true;
  }
  // M7 WO4a solver trials (private): the assembled Schur preconditioner, the π-solve tolerance
  void set_solver_options(bool schurAssembled, real_t adjointTol, real_t saddleTol) {
    if (!(adjointTol > 0) || !(saddleTol > 0))
      throw std::invalid_argument(
          "voro: LagrangianFlow._set_solver_options(adjoint_tol > 0, saddle_tol > 0)");
    schurAssembled_ = schurAssembled;
    adjointTol_ = adjointTol;
    saddleTol_ = saddleTol;
    withIntegIfAny([&](auto& I) { I.setSolverOptions(schurAssembled_, adjointTol_, saddleTol_); });
  }
  // M7 WO-PERF trials (private): the saddle method, B's rotational chi, AMG reuse
  void set_perf_options(const std::string& method, real_t rotational, int amgReuse, real_t warmTol,
                        real_t newtonTol, int relabel, const std::string& gauge) {
    namespace lg = peclet::voro::lagrangian;
    if (method == "schur")
      method_ = lg::SaddleMethod::kSchurCG;
    else if (method == "coupled")
      method_ = lg::SaddleMethod::kCoupled;
    else if (method == "projection")
      method_ = lg::SaddleMethod::kProjection;
    else
      throw std::invalid_argument(
          "voro: LagrangianFlow._set_perf_options(method in 'schur', 'coupled', 'projection')");
    lg::GaugeMode gm;
    if (gauge == "trial")
      gm = lg::GaugeMode::kTrial;
    else if (gauge == "drift")
      gm = lg::GaugeMode::kDrift;
    else
      throw std::invalid_argument(
          "voro: LagrangianFlow._set_perf_options(gauge in 'trial', 'drift')");
    if (!(rotational >= 0) || amgReuse < 0 || !(warmTol > 0) || !(newtonTol > 0) || relabel < -1)
      throw std::invalid_argument(
          "voro: LagrangianFlow._set_perf_options(rotational >= 0, amg_reuse >= 0, tolerances > "
          "0, relabel >= -1)");
    relabelEvery_ = relabel;
    rotational_ = rotational;
    amgReuse_ = amgReuse;
    warmTol_ = warmTol;
    newtonTol_ = newtonTol;
    gaugeMode_ = gm;
    perfSet_ = true;
    withIntegIfAny([&](auto& I) {
      I.setPerfOptions(method_, rotational_, amgReuse_, warmTol_, newtonTol_);
      I.setGaugeMode(gaugeMode_);
    });
  }
  void set_gauge(int every, real_t strain, real_t alpha_max) {
    if (every < 0 || !(strain >= 0) || !(alpha_max >= 0))
      throw std::invalid_argument(
          "voro: LagrangianFlow.set_gauge() takes every >= 0, strain >= 0, alpha_max >= 0");
    gaugeEvery_ = every;
    strain_ = strain;
    alphaMax_ = alpha_max;
    withIntegIfAny([&](auto& I) { I.setGauge(gaugeOn_ ? gaugeEvery_ : 0, strain_, alphaMax_); });
  }
  // M7 WO-PERF3 accuracy knobs (provisional names, not public API): relative tolerances in (0, 1)
  static void checkRelTol(const char* name, real_t tol) {
    if (!(tol > 0 && tol < 1))
      throw std::invalid_argument(std::string("voro: LagrangianFlow.") + name +
                                  "(tol) takes a relative tolerance 0 < tol < 1");
  }
  void set_kick_tolerance(real_t tol) {
    checkRelTol("_set_kick_tolerance", tol);
    kickTol_ = tol;
    withIntegIfAny([&](auto& I) { I.setKickTolerance(kickTol_); });
  }
  void set_volume_tolerance(real_t tol) {
    checkRelTol("_set_volume_tolerance", tol);
    volumeTol_ = tol;
    withIntegIfAny([&](auto& I) { I.setVolumeTolerance(volumeTol_); });
  }
  void set_gauge_enabled(bool on) {
    gaugeOn_ = on;
    withIntegIfAny([&](auto& I) { I.setGauge(gaugeOn_ ? gaugeEvery_ : 0, strain_, alphaMax_); });
  }
  void set_wall_velocity(std::array<real_t, 3> offset,
                         std::optional<nb::ndarray<real_t, nb::c_contig>> gradient) {
    wallU_ = offset;
    wallG_.fill(0);
    if (gradient) {
      if (gradient->size() != 9)
        throw std::runtime_error("voro: LagrangianFlow.set_wall_velocity(gradient) must be (3,3)");
      std::copy(gradient->data(), gradient->data() + 9, wallG_.begin());
    }
    dirty_ = true;
  }
  /// The t = 0 relaxation R0 (Integrator::relax, DESIGN_NOTE §12.3.1). Raises when a closing
  /// check fails (the run must not start from that state).
  nb::dict relax(int max_iterations, real_t tol, int window, real_t kappa, real_t eta) {
    if (max_iterations < 0 || !(tol > 0) || window < 1 || !(kappa > 0) || !(eta > 0))
      throw std::invalid_argument(
          "voro: LagrangianFlow.relax() takes max_iterations >= 0, tol > 0, window >= 1, "
          "kappa > 0, eta > 0");
    prepare();
    lag::RelaxParams rp;
    rp.maxIter = max_iterations;
    rp.tol = tol;
    rp.window = window;
    rp.kappa = kappa;
    rp.eta = eta;
    lag::RelaxReport r;
    withInteg([&](auto& I) { r = I.relax(rp); });
    t_->wHost_ = peclet::core::toVector(t_->weight_);
    t_->wDirty_ = false;
    if (!r.passed)
      throw std::runtime_error(
          "voro: LagrangianFlow.relax(): a closing check failed (OT converged " +
          std::to_string(r.otConverged) + " in " + std::to_string(r.otIterations) +
          " iterations; max |V - V0|/V0 = " + std::to_string(r.maxVolumeError) +
          " (<= 1e-12); max |s|/d = " + std::to_string(r.maxShift) +
          " (<= 0.15); min margin = " + std::to_string(r.minMargin) + " (> 0))");
    nb::dict d;
    d["iterations"] = r.iterations;
    d["converged"] = r.converged;
    d["energy"] = toNumpy1(std::vector<real_t>(r.energy.begin(), r.energy.end()));
    d["unlike_area_history"] = toNumpy1(std::vector<real_t>(r.area.begin(), r.area.end()));
    d["ot_iterations"] = r.otIterations;
    d["max_volume_error"] = r.maxVolumeError;
    d["max_shift"] = r.maxShift;
    d["min_margin"] = r.minMargin;
    d["unlike_area"] = r.unlikeArea;
    return d;
  }
  void set_velocities(nb::ndarray<real_t, nb::c_contig> v) {
    std::vector<real_t> h = flatten3(v);
    if (h.size() != vel_.size())
      throw std::runtime_error("voro: LagrangianFlow.set_velocities() must be (N,3)");
    checkOrder();
    pull();
    vel_ = cellsToInternal(t_->perm_, std::move(h), 3);
    dirty_ = true;
  }

  // (the getters return the caller's cell order: an O(N) gather through Tess::perm_)
  nb::ndarray<nb::numpy, real_t> get_velocities() {
    checkOrder();
    pull();
    return toNumpy3(cellsToCaller(t_->perm_, vel_, 3));
  }
  nb::ndarray<nb::numpy, real_t> get_positions() {
    checkOrder();
    return toNumpy3(cellsToCaller(t_->perm_, peclet::core::toVector(t_->pos_), 3));
  }
  nb::ndarray<nb::numpy, real_t> get_pi() {
    auto v = fromInteg([](auto& I) { return I.pi(); });
    return toNumpy1(cellsToCaller(t_->perm_, std::move(v), 1));
  }
  nb::ndarray<nb::numpy, real_t> get_static_pressure() {
    auto v = fromInteg([](auto& I) { return I.staticPressure(); });
    return toNumpy1(cellsToCaller(t_->perm_, std::move(v), 1));
  }
  nb::ndarray<nb::numpy, real_t> get_lambda() {
    auto v = fromInteg([](auto& I) { return I.lambda(); });
    return toNumpy1(cellsToCaller(t_->perm_, std::move(v), 1));
  }
  nb::ndarray<nb::numpy, real_t> get_potential_gradient() {
    auto v = fromInteg([](auto& I) { return I.gradient(); });
    return toNumpy3(cellsToCaller(t_->perm_, std::move(v), 3));
  }

  nb::dict energies() {
    prepare();
    nb::dict d;
    withInteg([&](auto& I) {
      d["kinetic"] = I.kinetic();
      d["potential"] = I.potential().total();
      d["interface"] = I.potential().interface;
      d["wall"] = I.potential().wall;
      d["gravity"] = I.potential().gravity;
      d["heat"] = I.heat();
      d["constraint_work"] = I.constraintWork();
      d["body_work"] = I.bodyWork();
      d["wrap_offset"] = I.wrapOffset();
      d["gauge_potential"] = I.gaugePotential();
      d["gauge_kinetic"] = I.gaugeKinetic();
      d["unlike_area"] = I.unlikeArea();
    });
    return d;
  }

  /// n steps; per-step records as arrays.
  nb::dict step(int n) {
    prepare();
    std::vector<lag::StepReport> reps;
    reps.reserve(n);
    for (int k = 0; k < n; ++k) {
      if (relabelEvery_ > 0 && sinceRelabel_ >= relabelEvery_)
        resort();
      withInteg([&](auto& I) {
        const double tau = I.stableStep(cfl_, cap_, dtUser_);
        if (!std::isfinite(tau))
          throw std::runtime_error(
              "voro: LagrangianFlow.step(): nothing limits the step (no velocity, no tension) "
              "— call set_dt().");
        reps.push_back(I.step(tau));
        time_ = I.time();
      });
      ++sinceRelabel_;
    }
    lag::detail::hostCopy(t_->weight_, t_->wHost_);  // no staging allocation per step
    t_->wDirty_ = false;
    lastDt_ = reps.empty() ? lastDt_ : reps.back().dt;
    auto col = [&](auto get) {
      std::vector<real_t> c;
      c.reserve(reps.size());
      for (const auto& r : reps)
        c.push_back(static_cast<real_t>(get(r)));
      return toNumpy1(std::move(c));
    };
    nb::dict d;
    d["dt"] = col([](const auto& r) { return r.dt; });
    d["time"] = col([](const auto& r) { return r.time; });
    d["kinetic"] = col([](const auto& r) { return r.kinetic; });
    d["potential"] = col([](const auto& r) { return r.potential; });
    d["heat"] = col([](const auto& r) { return r.heat; });
    d["constraint_work"] = col([](const auto& r) { return r.constraintWork; });
    d["body_work"] = col([](const auto& r) { return r.bodyWork; });
    d["wrap_offset"] = col([](const auto& r) { return r.wrapOffset; });
    d["max_speed"] = col([](const auto& r) { return r.maxSpeed; });
    d["max_shift"] = col([](const auto& r) { return r.maxShift; });
    d["max_shift_unlike"] = col([](const auto& r) { return r.maxShiftUnlike; });
    d["kick1_outer"] = col([](const auto& r) { return r.kick1.outer; });
    d["kick1_inner"] = col([](const auto& r) { return r.kick1.inner; });
    d["kick2_outer"] = col([](const auto& r) { return r.kick2.outer; });
    d["kick2_inner"] = col([](const auto& r) { return r.kick2.inner; });
    d["constraint_residual"] =
        col([](const auto& r) { return std::max(r.kick1.constraint, r.kick2.constraint); });
    d["visc_number"] = col([](const auto& r) { return r.kick2.viscNumber; });
    d["warm_iterations"] = col([](const auto& r) { return r.warmIters; });
    d["newton_iterations"] = col([](const auto& r) { return r.newtonIters; });
    d["repairs"] = col([](const auto& r) { return r.evaluations; });
    d["drift_evaluations"] = col([](const auto& r) { return r.driftEvaluations; });
    d["ot_r0"] = col([](const auto& r) { return r.otR0; });
    d["carry_rho"] = col([](const auto& r) { return r.carryRho; });
    d["ot_r0_lin"] = col([](const auto& r) { return r.otR0Lin; });
    d["carry_drops"] = col([](const auto& r) { return r.carryDrops; });
    d["cold_rebuilds"] = col([](const auto& r) { return r.coldRebuilds; });
    d["weight_error"] = col([](const auto& r) { return r.weightErr; });
    d["adjoint_iterations"] = col([](const auto& r) { return r.adjIters; });
    d["halvings"] = col([](const auto& r) { return r.halvings; });
    d["kick_restart"] = col([](const auto& r) { return r.kickRestart; });
    d["unmatched_facets"] = col([](const auto& r) { return r.unmatched; });
    d["repair_fallbacks"] = col([](const auto& r) { return r.repairFallbacks; });
    d["weight_restarts"] = col([](const auto& r) { return r.coldStarts; });
    d["repair_mismatch"] = col([](const auto& r) { return r.repairMismatch; });
    d["unmatched_area"] = col([](const auto& r) { return r.unmatchedArea; });
    d["t_kick1"] = col([](const auto& r) { return r.tKick1; });
    d["t_kick2"] = col([](const auto& r) { return r.tKick2; });
    d["t_repair"] = col([](const auto& r) { return r.tRepair; });
    d["t_weights"] = col([](const auto& r) { return r.tWeights; });
    d["t_potential"] = col([](const auto& r) { return r.tPotential; });
    d["t_total"] = col([](const auto& r) { return r.tTotal; });
    d["t_facets"] = col([](const auto& r) { return r.tFacets; });
    d["t_energy"] = col([](const auto& r) { return r.tEnergy; });
    d["t_amg_h"] = col([](const auto& r) { return r.tAmgH; });
    d["t_adjoint"] = col([](const auto& r) { return r.tAdjoint; });
    d["kick1_t_setup"] = col([](const auto& r) { return r.kick1.tSetup; });
    d["kick1_t_inner"] = col([](const auto& r) { return r.kick1.tInner; });
    d["kick1_t_precond"] = col([](const auto& r) { return r.kick1.tPrecond; });
    d["kick1_t_solve"] = col([](const auto& r) { return r.kick1.tTotal; });
    d["kick2_t_setup"] = col([](const auto& r) { return r.kick2.tSetup; });
    d["kick2_t_inner"] = col([](const auto& r) { return r.kick2.tInner; });
    d["kick2_t_precond"] = col([](const auto& r) { return r.kick2.tPrecond; });
    d["kick2_t_solve"] = col([](const auto& r) { return r.kick2.tTotal; });
    d["gauge_tries"] = col([](const auto& r) { return r.gaugeTries; });
    d["gauge_accepted"] = col([](const auto& r) { return r.gaugeAccepted ? 1 : 0; });
    d["gauge_normal"] = col([](const auto& r) { return r.gaugeNormal; });
    d["gauge_projected"] = col([](const auto& r) { return r.gaugeProjected; });
    d["gauge_capped"] = col([](const auto& r) { return r.gaugeCapped; });
    d["gauge_margin_rejects"] = col([](const auto& r) { return r.gaugeMarginRejects; });
    d["min_margin"] = col([](const auto& r) { return r.minMargin; });
    d["gauge_interface_cells"] = col([](const auto& r) { return r.gaugeInterface; });
    d["gauge_potential"] = col([](const auto& r) { return r.gaugePotential; });
    d["gauge_kinetic"] = col([](const auto& r) { return r.gaugeKinetic; });
    d["gauge_alpha_max"] = col([](const auto& r) { return r.gaugeAlphaMax; });
    d["t_gauge"] = col([](const auto& r) { return r.tGauge; });
    d["gauge_solve_fails"] = col([](const auto& r) { return r.gaugeSolveFails; });
    d["gauge_energy_rejects"] = col([](const auto& r) { return r.gaugeEnergyRejects; });
    d["gauge_trial_dU"] = col([](const auto& r) { return r.gaugeTrialDU; });
    d["gauge_applied"] = col([](const auto& r) { return r.gaugeApplied; });
    d["gauge_redos"] = col([](const auto& r) { return r.gaugeRedos; });
    d["gauge_active"] = col([](const auto& r) { return r.gaugeActive; });
    d["gauge_factor"] = col([](const auto& r) { return r.gaugeFactor; });
    d["gauge_beta"] = col([](const auto& r) { return r.gaugeBeta; });
    d["gauge_weight_residual"] = col([](const auto& r) { return r.gaugeWeightRes; });
    // the step's energy ledger by stage (diagnostics, M7 WO-EB; StepReport::ebK1 ff.)
    d["eb_k1"] = col([](const auto& r) { return r.ebK1; });
    d["eb_heat1"] = col([](const auto& r) { return r.ebHeat1; });
    d["eb_qc1"] = col([](const auto& r) { return r.ebQc1; });
    d["eb_work1"] = col([](const auto& r) { return r.ebWork1; });
    d["eb_grad1"] = col([](const auto& r) { return r.ebGrad1; });
    d["eb_k2"] = col([](const auto& r) { return r.ebK2; });
    d["eb_heat2"] = col([](const auto& r) { return r.ebHeat2; });
    d["eb_qc2"] = col([](const auto& r) { return r.ebQc2; });
    d["eb_work2"] = col([](const auto& r) { return r.ebWork2; });
    d["eb_grad2"] = col([](const auto& r) { return r.ebGrad2; });
    d["eb_dU"] = col([](const auto& r) { return r.ebDU; });
    d["eb_trap"] = col([](const auto& r) { return r.ebTrap; });
    d["eb_gd1"] = col([](const auto& r) { return r.ebGd1; });
    d["eb_phi1"] = col([](const auto& r) { return r.ebPhi1; });
    d["eb_psic1"] = col([](const auto& r) { return r.ebPsiC1; });
    d["eb_gd2"] = col([](const auto& r) { return r.ebGd2; });
    d["eb_phi2"] = col([](const auto& r) { return r.ebPhi2; });
    d["eb_psic2"] = col([](const auto& r) { return r.ebPsiC2; });
    withInteg([&](auto& I) {
      d["gauge_potential_total"] = I.gaugePotential();
      d["gauge_kinetic_total"] = I.gaugeKinetic();
    });
    return d;
  }

  real_t time() const { return time_; }
  real_t dt() const { return lastDt_; }
  real_t theta() {
    prepare();
    return theta_;
  }
  int num_cells() const { return static_cast<int>(types_.size()); }

  void release() noexcept override {
    i0_.reset();
    i1_.reset();
  }

 private:
  std::vector<double> perType(nb::ndarray<real_t, nb::c_contig> v, const char* what) const {
    if (v.size() != static_cast<std::size_t>(nT_))
      throw std::runtime_error(std::string("voro: LagrangianFlow.set_") + what +
                               "() takes one value per type (" + std::to_string(nT_) + ")");
    return std::vector<double>(v.data(), v.data() + v.size());
  }

  template <class F>
  void withInteg(F&& f) {
    if (i0_)
      f(*i0_);
    else if (i1_)
      f(*i1_);
    else
      throw std::logic_error("voro: LagrangianFlow has no integrator");
  }
  template <class F>
  void withIntegIfAny(F&& f) {
    if (i0_)
      f(*i0_);
    else if (i1_)
      f(*i1_);
  }
  template <class F>
  std::vector<real_t> fromInteg(F&& f) {
    prepare();
    std::vector<real_t> out;
    withInteg([&](auto& I) { out = peclet::core::toVector(f(I)); });
    return out;
  }

  /// Copy the integrator's velocity (and the run's accumulators) to the host.
  void pull() {
    if (!i0_ && !i1_)
      return;
    withInteg([&](auto& I) {
      vel_ = peclet::core::toVector(I.velocity());
      time_ = I.time();
      acc_ = {I.heat(),       I.constraintWork(), I.bodyWork(),
              I.wrapOffset(), I.gaugePotential(), I.gaugeKinetic()};
      gaugeFactor_ = I.gaugeFactor();
    });
  }

  /// The Tessellation's cell order is the one this flow's per-cell state was gathered in.
  void checkOrder() const {
    if (t_->relabels_ != relabels_)
      throw std::runtime_error(
          "voro: LagrangianFlow: its Tessellation was rebuilt with a new point count or driven by "
          "another LagrangianFlow — build a new LagrangianFlow.");
  }

  /// Store the cells in the Morton order of their current positions (cache locality of the facet
  /// operators, AMG and the repair; the caller's order is kept by Tess::perm_). Permutes this
  /// flow's per-cell host state and the Tessellation (cold rebuild), marking the integrator for a
  /// re-preparation. Returns the gather order, empty when the cells already are in that order.
  std::vector<int> relabelCells() {
    Tess& t = *t_;
    ordered_ = true;
    sinceRelabel_ = 0;
    std::vector<int> order = mortonOrder(peclet::core::toVector(t.pos_), t.L_);
    bool identity = true;
    for (std::size_t k = 0; k < order.size() && identity; ++k)
      identity = order[k] == static_cast<int>(k);
    if (identity)
      return {};
    pull();  // the integrator's state is now on the host; it is rebuilt in the new order
    i0_.reset();
    i1_.reset();
    types_ = cellsToInternal(order, std::move(types_), 1);
    V0_ = cellsToInternal(order, std::move(V0_), 1);
    vel_ = cellsToInternal(order, std::move(vel_), 3);
    t.relabel(order);
    relabels_ = t.relabels_;
    dirty_ = true;
    return order;
  }

  /// The periodic re-sort of a running flow: relabel, re-prepare, and carry λ₂ (kick 1's warm
  /// start), the gauge's step count, the drift gauge's pending δⁿ and the 'pressure' kick's
  /// history (λⁿ⁻¹, v^{n−½}, τ_prev) into the new integrator.
  void resort() {
    std::vector<real_t> lam, delta, lamPrev, vHalf;
    long count = 0;
    double tauPrev = 0;
    withInteg([&](auto& I) {
      lam = peclet::core::toVector(I.lambda());
      count = I.stepCount();
      delta = I.pendingDelta();
      auto h = I.kickHistory();
      lamPrev = std::move(h.lamPrev);
      vHalf = std::move(h.vHalf);
      tauPrev = h.tauPrev;
    });
    const std::vector<int> order = relabelCells();
    if (order.empty())
      return;
    prepare();
    lam = cellsToInternal(order, std::move(lam), 1);
    if (!delta.empty())
      delta = cellsToInternal(order, std::move(delta), 3);
    if (!lamPrev.empty()) {
      lamPrev = cellsToInternal(order, std::move(lamPrev), 1);
      vHalf = cellsToInternal(order, std::move(vHalf), 3);
    }
    withInteg([&](auto& I) {
      I.setWarmStart(lam, count);
      I.setPendingDelta(delta);
      I.setKickHistory({std::move(lamPrev), std::move(vHalf), tauPrev});
    });
  }

  /// (Re)build the integrator when an input, the tessellation's positions or its weights changed.
  void prepare() {
    Tess& t = *t_;
    checkOrder();
    if ((i0_ || i1_) && !dirty_ && !t.wDirty_ && t.pos_.data() == posData_)
      return;
    pull();
    if (!ordered_ && relabelEvery_ >= 0)
      relabelCells();  // the first preparation stores the cells in Morton order
    if (t.wDirty_) {
      Kokkos::deep_copy(t.weight_, Kokkos::View<const real_t*, Kokkos::HostSpace>(t.wHost_.data(),
                                                                                  t.wHost_.size()));
      t.wDirty_ = false;
    }
    const int N = t.N_;
    lag::PotentialSpec<real_t> spec;
    Kokkos::View<int*, Mem> dtype("lf.type", N);
    Kokkos::deep_copy(dtype, Kokkos::View<const int*, Kokkos::HostSpace>(types_.data(), N));
    spec.types = dtype;
    spec.nTypes = nT_;
    if (!tension_.empty())
      spec.tension = peclet::core::toDevice<real_t>(tension_, "lf.tension");
    spec.interfaceEnergy = interfaceEnergy_;
    if (interfaceEnergy_ == lag::InterfaceEnergy::Nonlocal) {
      spec.refVolume = peclet::core::toDevice<real_t>(V0_, "lf.V0");
      spec.nonlocal = lag::nonlocalParams<real_t>(V0_);
    }
    if (!sigmaWall_.empty())
      spec.sigmaWall = peclet::core::toDevice<real_t>(sigmaWall_, "lf.sigmaWall");
    if (hasGravity_) {
      spec.density = peclet::core::toDevice<real_t>(rho_, "lf.density");
      for (int a = 0; a < 3; ++a)
        spec.gravity[a] = gravity_[a];
      if (rhoRef_)
        spec.rhoRef = *rhoRef_;
      else if (t.scene_.set)
        spec.rhoRef = 0;  // walls ⟂ g: the partition potential
      else {              // periodic in g: ρ̄ = Σ m / |Ω|
        double m = 0, v = 0;
        for (int i = 0; i < N; ++i) {
          m += rho_[types_[i]] * V0_[i];
          v += V0_[i];
        }
        spec.rhoRef = m / v;
      }
    }
    lag::FlowParams fp;
    fp.density = rho_;
    fp.viscosity = mu_;
    for (int a = 0; a < 3; ++a) {
      fp.bodyAmp[a] = bodyAmp_[a];
      fp.bodyWave[a] = bodyWave_[a];
      fp.wallVelocity[a] = wallU_[a];
    }
    for (int e = 0; e < 9; ++e)
      fp.wallGradient[e] = wallG_[e];
    const real_t Larr[3] = {t.L_[0], t.L_[1], t.L_[2]};
    i0_.reset();
    i1_.reset();
    std::visit(
        [&](auto& mt) {
          using T = std::decay_t<decltype(mt)>;
          if constexpr (std::is_same_v<typename T::PlanePolicy, peclet::voro::Power>) {
            // hold the reference volumes first (a cold weight solve) if they are not held yet
            const auto V = peclet::core::toVector(Kokkos::subview(mt.vol, std::make_pair(0, N)));
            double err = 0;
            for (int i = 0; i < N; ++i)
              err = std::max(err, std::fabs(V[i] - V0_[i]) / V0_[i]);
            if (err > 1e-12) {
              const auto R = lag::solveWeights(mt, t.pos_, t.weight_, V0_, Larr);
              if (!R.converged)
                throw std::runtime_error(
                    "voro: LagrangianFlow: the weight solve for the reference volumes failed "
                    "(max |V - V0|/V0 = " +
                    std::to_string(R.maxRelErr) + ")");
              t.wHost_ = peclet::core::toVector(t.weight_);
            }
            auto I = std::make_unique<lag::Integrator<T>>(mt, t.pos_, t.weight_, Larr, V0_, spec,
                                                          fp, vel_);
            // θ = 1 for the whole run if max ν τ/h² > 1 at the step bound (DESIGN_NOTE §2.7)
            double tauMax = dtUser_ > 0 ? dtUser_ : std::numeric_limits<double>::infinity();
            if (cap_ > 0)
              tauMax = std::min(tauMax, I->capillaryStep(cap_));
            theta_ = (std::isfinite(tauMax) && I->maxNuOverH2() * tauMax > 1.0) ? 1.0 : 0.5;
            I->setTheta(theta_);
            I->setKickConstraint(kick1_);
            I->setTime(time_);
            I->setAccumulators(acc_[0], acc_[1], acc_[2], acc_[3], acc_[4], acc_[5]);
            I->setGauge(gaugeOn_ ? gaugeEvery_ : 0, strain_, alphaMax_);
            if (adjointTol_ > 0)
              I->setSolverOptions(schurAssembled_, adjointTol_, saddleTol_);
            if (perfSet_) {
              I->setPerfOptions(method_, rotational_, amgReuse_, warmTol_, newtonTol_);
              I->setGaugeMode(gaugeMode_);
            }
            I->setGaugeFactor(gaugeFactor_);
            if (kickTol_ > 0)
              I->setKickTolerance(kickTol_);
            if (volumeTol_ > 0)
              I->setVolumeTolerance(volumeTol_);
            if constexpr (T::kHasSdf)
              i1_ = std::move(I);
            else
              i0_ = std::move(I);
          } else {
            throw std::logic_error("voro: weighted Tessellation without a power engine");
          }
        },
        t.mt_);
    posData_ = t.pos_.data();
    dirty_ = false;
  }

  Tess* t_;
  std::vector<int> types_;
  int nT_ = 0;
  std::vector<double> V0_, rho_, mu_, tension_, sigmaWall_;
  lag::InterfaceEnergy interfaceEnergy_ = lag::InterfaceEnergy::Facet;
  std::vector<real_t> vel_;
  std::array<real_t, 3> gravity_{0, 0, 0}, bodyAmp_{0, 0, 0}, bodyWave_{0, 0, 0};
  bool hasGravity_ = false, schurAssembled_ = false;
  peclet::voro::lagrangian::KickConstraint kick1_ =
      peclet::voro::lagrangian::KickConstraint::kPressure;
  real_t adjointTol_ = 0, saddleTol_ = 0;  // 0: not set (the integrator's defaults)
  peclet::voro::lagrangian::SaddleMethod method_ = peclet::voro::lagrangian::SaddleMethod::kCoupled;
  real_t rotational_ = 1;
  int amgReuse_ = 10;
  real_t warmTol_ = 1e-4, newtonTol_ = 1e-3;
  bool perfSet_ = false;  // false: the integrator's defaults
  peclet::voro::lagrangian::GaugeMode gaugeMode_ = peclet::voro::lagrangian::GaugeMode::kTrial;
  real_t gaugeFactor_ = 1;  // the drift gauge's f_G, carried across a re-preparation
  // Morton relabelling: -1 off, 0 once at the first preparation, K > 0 also every K steps. K = 100
  // (M7 WO-PERF2): a re-sort costs ~0.7 step (N 32768); a frozen order loses 66 % of its locality
  // over 180 Taylor-Green steps (gamma 6), the step time within noise of K = 20 over that run
  int relabelEvery_ = 100, sinceRelabel_ = 0;
  bool ordered_ = false;
  unsigned relabels_ = 0;  ///< Tess::relabels_ the per-cell state was gathered at
  std::optional<real_t> rhoRef_;
  real_t dtUser_ = 0, cfl_ = 0.2, cap_ = 0.5, time_ = 0, lastDt_ = 0, theta_ = 0.5;
  std::array<double, 6> acc_{0, 0, 0, 0, 0, 0};
  std::array<real_t, 3> wallU_{0, 0, 0};
  std::array<real_t, 9> wallG_{0, 0, 0, 0, 0, 0, 0, 0, 0};
  real_t strain_ = 3, alphaMax_ = 0.5;  // M7 WO-G1 (D16.6): was 0.2
  int gaugeEvery_ = 1;
  bool gaugeOn_ = true;                 // _set_gauge_enabled (every_ kept while off)
  real_t kickTol_ = 0, volumeTol_ = 0;  // accuracy knobs; 0: the integrator's defaults
  bool dirty_ = true;
  const real_t* posData_ = nullptr;
  std::unique_ptr<Integ<NoSdfT>> i0_;
  std::unique_ptr<Integ<SceneT>> i1_;
};

void bindLagrangian(nb::module_& m) {
  nb::class_<FacetOperators>(
      m, "FacetOperators",
      "The cell operators of the Lagrangian power-cell method on a SNAPSHOT of a built\n"
      "Tessellation's cells (periodic box, with or without SDF walls; Voronoi or power):\n"
      "matrix-free facet loops on the device. Walls are fixed, impermeable and no-slip: C\n"
      "gains -A_w n_w.v_i per wall facet, the Rayleigh function 1/2 mu A_w |P_w v_i|^2 / delta_w\n"
      "(delta_w the generator-wall distance); J_x and H have no wall term.\n"
      "Velocities are (N,3), cell scalars (N,). `apply(op, x)` with op one of\n"
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

  nb::class_<LagrangianFlow>(
      m, "LagrangianFlow",
      "The NHV-W time integrator of the Lagrangian power-cell two-phase method (M7): drives a\n"
      "built POWER Tessellation in place. Each cell carries the mass rho_t V0 of its type t,\n"
      "the velocity v and keeps its volume V0 (weights solved every step, each Newton iterate an\n"
      "incremental repair). One step: kick 1 (friction fused, theta-midpoint; velocity\n"
      "constraint C v = 0 by Schur-complement CG with AMG on the two-point Laplacian), drift,\n"
      "weight solve, pi and grad U~ (the reduced potential: interface tension, wetting, facet\n"
      "gravity), kick 2. Energy bookkeeping: kinetic K, potential U~, heat Q, constraint work "
      "Q_c,\n"
      "body-force work W; K + U~ + Q + Q_c - W is conserved to O(dt^2). Physical inputs only:\n"
      "per-type density / viscosity, the tension table, wall tension, gravity, a body\n"
      "acceleration amp*sin(k.x), and the step: dt = min(cfl*min h/|v|, capillary*sqrt(rho\n"
      "h^3/sigma), set_dt cap). The gauge G'' after kick 2 (set_gauge); no topology bookkeeping "
      "beyond the repair.\n"
      "`volumes` (N,): the reference volumes (default: the current ones).")
      .def(nb::init<Tess&, nb::ndarray<int, nb::c_contig>,
                    std::optional<nb::ndarray<real_t, nb::c_contig>>>(),
           nb::arg("tessellation"), nb::arg("types"), nb::arg("volumes") = nb::none(),
           nb::keep_alive<1, 2>())
      .def("set_density", &LagrangianFlow::set_density, nb::arg("density"),
           "Density per type (nTypes,), > 0 (default 1).")
      .def("set_viscosity", &LagrangianFlow::set_viscosity, nb::arg("viscosity"),
           "Dynamic viscosity per type (nTypes,), >= 0 (default 0: inviscid).")
      .def("set_tension", &LagrangianFlow::set_tension, nb::arg("tension"),
           "Interface tension table (nTypes, nTypes) on unlike facets.")
      .def("set_interface_energy", &LagrangianFlow::set_interface_energy, nb::arg("mode"),
           "The capillary energy that carries the tension table: 'facet' (the raw unlike-facet\n"
           "area, the default) or 'nonlocal' (the moment-blob Gaussian nonlocal perimeter E_M,\n"
           "C^{1,1} in the positions: width s = h, cutoff 5s, h = (sum V0 / N)^(1/3)).")
      .def("set_wall_tension", &LagrangianFlow::set_wall_tension, nb::arg("wall_tension"),
           "Solid-fluid tension per type (nTypes,) on wall facets (wetting).")
      .def("set_gravity", &LagrangianFlow::set_gravity, nb::arg("gravity"),
           nb::arg("reference_density") = nb::none(),
           "Facet gravity (the partition potential). reference_density default: 0 with walls\n"
           "(walls perpendicular to g), the mean density in a periodic box.")
      .def("set_body_force", &LagrangianFlow::set_body_force, nb::arg("amplitude"),
           nb::arg("wavevector"), "Body acceleration a(x) = amplitude * sin(wavevector . x).")
      .def("set_dt", &LagrangianFlow::set_dt, nb::arg("dt"), "Upper bound of the step (0: none).")
      .def("set_cfl", &LagrangianFlow::set_cfl, nb::arg("advective") = 0.2,
           nb::arg("capillary") = 0.5,
           "Step-size constants C_cfl, C_sigma (0 switches a limit off).")
      .def("set_kick_constraint", &LagrangianFlow::set_kick_constraint, nb::arg("mode"),
           "Where kick 1's velocity constraint is taken: 'midpoint' (x^n + dt/2 v^n with the\n"
           "weights solved there; DESIGN_NOTE option S: second order in the energy), 'start'\n"
           "(x^n; first order), or 'pressure' (the default: no constraint and no saddle; the\n"
           "last pressures 3/4 lambda^n + 1/4 lambda^(n-1) act in kick 1's implicit solve, one\n"
           "saddle per step, second order; restart steps - the first, and the retry after a\n"
           "step halving - use option S; DESIGN_NOTE D14.1).")
      .def(
          "_set_perf_options", &LagrangianFlow::set_perf_options, nb::arg("method") = "coupled",
          nb::arg("rotational") = 1.0, nb::arg("amg_reuse") = 10, nb::arg("warm_tol") = 1e-3,
          nb::arg("newton_tol") = 1e-3, nb::arg("relabel") = 100, nb::arg("gauge") = "trial",
          "M7 WO-PERF solver trials (not public API): the kick's saddle method ('schur' nested\n"
          "Schur CG, 'coupled' block-preconditioned FGMRES, 'projection' rotational incremental\n"
          "pressure correction with chi = `rotational`), and AMG hierarchy reuse over at most\n"
          "`amg_reuse` further setups (0: rebuilt every setup); the relative tolerances of the "
          "warm-start\n"
          "and Newton H solves (the Newton one is inert under the Eisenstat-Walker forcing, the\n"
          "default since M7 WO-PERF3 P3.2); the cell relabelling (`relabel`: -1 off, 0 Morton "
          "order at the\n"
          "first preparation, K > 0 also re-sorted every K steps (default 100); the caller's "
          "order is kept); the gauge's displacement (`gauge`: 'trial' G''s trial OT after kick 2,\n"
          "'drift' added in the next step's drift, DESIGN_NOTE §14.4.4 with G''s activation).")
      .def(
          "_set_kick_tolerance", &LagrangianFlow::set_kick_tolerance, nb::arg("tol"),
          "M7 WO-PERF3 accuracy knob (provisional, not public API): eta_K, the relative tolerance\n"
          "of the kick's velocity/constraint solves, |C v|_inf <= tol |C| v_ref and |r_v|_2 <=\n"
          "tol |b|_2 (default 1e-9). 0 < tol < 1.")
      .def("_set_volume_tolerance", &LagrangianFlow::set_volume_tolerance, nb::arg("tol"),
           "M7 WO-PERF3 accuracy knob (provisional, not public API): the per-step volume solve's\n"
           "max_i |V_i - V0_i| / V0_i (default 1e-12). 0 < tol < 1.")
      .def("_set_gauge_enabled", &LagrangianFlow::set_gauge_enabled, nb::arg("on"),
           "M7 WO-PERF3 accuracy knob (provisional, not public API): the gauge on (default) or\n"
           "off, keeping set_gauge's cadence and parameters.")
      .def("_set_solver_options", &LagrangianFlow::set_solver_options, nb::arg("schur_assembled"),
           nb::arg("adjoint_tol"), nb::arg("saddle_tol") = 1e-9,
           "M7 WO4a solver trials (not public API): AMG on the assembled C M^-1 C^T as the Schur\n"
           "preconditioner, the relative tolerance of the pi (adjoint) solve, and the saddle's\n"
           "constraint tolerance |C v|_inf <= saddle_tol |C| v_ref (relative; M7 WO-PERF3 P3.2).")
      .def(
          "set_gauge", &LagrangianFlow::set_gauge, nb::arg("every") = 1, nb::arg("strain") = 3.0,
          nb::arg("alpha_max") = 0.5,
          "The gauge G'' (M7 DESIGN_NOTE §12.3.2) after kick 2 every `every` steps (0: off):\n"
          "strain-activated, Lloyd-only and ascent-free. alpha_i = min(alpha_max, strain * dt *\n"
          "max |sym grad v| over the cell and its face neighbours); skipped, the state untouched,\n"
          "when max alpha < 1e-6. Interface cells move tangentially to their power-cell centroid\n"
          "(plus the normal part when an unlike facet has |s|/d > 0.25), contact-line cells stay,\n"
          "the rest move to the centroid: delta = alpha (b - x); each cell's ascent component\n"
          "along grad U~ removed, |delta| capped at half the distance to its own facet planes;\n"
          "weights re-solved, rejected if U~ rises or a seed leaves its cell (3 halvings), then\n"
          "v projected onto C v = 0 in the M-metric. Its dU, dK are counted separately.")
      .def("set_wall_velocity", &LagrangianFlow::set_wall_velocity, nb::arg("offset"),
           nb::arg("gradient") = nb::none(),
           "No-slip wall velocity u_w(y) = offset + gradient @ y at the wall facets (tangential\n"
           "part; e.g. Couette between z-walls: gradient[0, 2] = shear rate). Default 0.")
      .def("relax", &LagrangianFlow::relax, nb::arg("max_iterations") = 5000, nb::arg("tol") = 1e-6,
           nb::arg("window") = 250, nb::arg("kappa") = 100.0, nb::arg("eta") = 0.1,
           "The t = 0 relaxation R0 (M7 DESIGN_NOTE §12.3.1), from a Lloyd mesh whose volumes\n"
           "are the reference volumes: steepest descent of F = E_sigma + (kappa sigma h^2/2)\n"
           "sum ((V - V0)/V0)^2 on VORONOI cells (w = 0), step -(eta/kappa) grad F / sigma, until\n"
           "|dF|/F < tol over `window` iterations or `max_iterations`; then one OT solve to V0.\n"
           "Raises if a closing check fails (max |V - V0|/V0 <= 1e-12, max |s|/d <= 0.15, every\n"
           "generator strictly inside its cell). Returns iterations, converged, energy and\n"
           "unlike_area_history (per window), ot_iterations, max_volume_error, max_shift,\n"
           "min_margin and unlike_area (the solved power cells). Call at t = 0 only.")
      .def("set_velocities", &LagrangianFlow::set_velocities, nb::arg("velocities"))
      .def("get_velocities", &LagrangianFlow::get_velocities)
      .def("get_positions", &LagrangianFlow::get_positions)
      .def("get_pi", &LagrangianFlow::get_pi,
           "pi of the current state (H pi = -dU/dw; the Laplace and hydrostatic part).")
      .def("_get_static_pressure", &LagrangianFlow::get_static_pressure,
           "The static pressure P = -(pi - e) of the current state (M7 D15.2): the total volume\n"
           "derivative dU~/dV0, e = dU/dV0 at fixed (x, w) (E_M only; facet energies: P = -pi).\n"
           "Up to a uniform constant; the reported pressure is P - lambda. Output only.")
      .def("get_lambda", &LagrangianFlow::get_lambda,
           "lambda of the last kick 2 (the dynamic pressure multiplier); p = -(pi + lambda).")
      .def("get_potential_gradient", &LagrangianFlow::get_potential_gradient,
           "grad U~ of the current state (N,3).")
      .def("energies", &LagrangianFlow::energies, "The current energies and accumulators (dict).")
      .def("step", &LagrangianFlow::step, nb::arg("n") = 1,
           "Advance n steps; returns per-step records as arrays (energies after the step,\n"
           "solver iterations, residuals, max facet shift |s|/d, timings).")
      .def_prop_ro("time", &LagrangianFlow::time)
      .def_prop_ro("dt", &LagrangianFlow::dt, "The last step's size.")
      .def_prop_ro("theta", &LagrangianFlow::theta)
      .def_prop_ro("num_cells", &LagrangianFlow::num_cells);
}
}  // namespace peclet::voro::pybind
