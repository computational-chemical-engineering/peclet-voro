/// @file
/// @brief `optimize_volume_mesh` and `minimize_interface`: the mesh optimisers on a periodic box.
#include "voro_bindings_optim.hpp"

namespace peclet::voro::pybind {

void bindOptimizers(nb::module_& m) {
  using namespace defaults;
  // ---- mesh optimisers on a periodic box ------------------------------------------------------
  m.def(
      "optimize_volume_mesh",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> vset_in,
         std::array<real_t, 3> extent, int search_window, int max_iter, real_t tol, int cg_iters,
         bool use_weights, const std::string& method) {
        auto pos = flatten3(pos_in);
        auto vset = flatten1(vset_in);
        const int N = (int)vset.size();
        if ((int)(pos.size() / 3) != N)
          throw std::runtime_error(
              "optimize_volume_mesh: positions (N,3) and target_volumes (N,) must agree on N");
        for (int a = 0; a < 3; ++a)
          if (!(extent[a] > real_t(0)))
            throw std::invalid_argument(
                "voro: optimize_volume_mesh(extent=...) needs three positive lengths.");
        const real_t Larr[3] = {extent[0], extent[1], extent[2]};
        const auto prec = parseMethod(method, "optimize_volume_mesh");
        peclet::voro::OtResult R;
        std::vector<real_t> w;
        if (use_weights) {
          w.assign(N, 0.0);
          R = peclet::voro::meshVolumeOptimize<real_t, true>(pos, w, vset, Larr, N, search_window,
                                                             peclet::voro::NoSdf{}, max_iter, tol,
                                                             cg_iters, prec, false);
        } else {
          std::vector<real_t> noW;
          R = peclet::voro::meshVolumeOptimize<real_t, false>(pos, noW, vset, Larr, N,
                                                              search_window, peclet::voro::NoSdf{},
                                                              max_iter, tol, cg_iters, prec, false);
        }
        return makeOptimizeResult(std::move(pos),
                                  use_weights ? std::optional(std::move(w)) : std::nullopt, R);
      },
      nb::arg("positions"), nb::arg("target_volumes"), nb::arg("extent"), nb::kw_only(),
      nb::arg("search_window") = kOptimizerSearchWindow, nb::arg("max_iter") = kOptimizerMaxIter,
      nb::arg("tol") = kOptimizerTolerance, nb::arg("cg_iters") = kOptimizerCgIters,
      nb::arg("use_weights") = false, nb::arg("method") = "jacobi",
      doc("Move seeds (N,3) — and optionally the power weights — to minimise "
          "sum (V_i / V_ref,i - 1)^2 by\n"
          "damped Gauss-Newton (Newton-Raphson + CG) on the periodic box `extent` (Lx, Ly, Lz).\n"
          "target_volumes (N,) are the per-cell reference volumes V_ref (renormalised to the box\n"
          "volume). method: the CG preconditioner, one of " +
          std::string(kMethodList) +
          " (default 'jacobi'; 'graphamg'\n"
          "is the O(N) choice at large N, 'steepest' is plain descent). search_window (default " +
          fmt(kOptimizerSearchWindow) + "), max_iter (" + fmt(kOptimizerMaxIter) + "), tol (" +
          fmt(kOptimizerTolerance) + ")\nand cg_iters (" + fmt(kOptimizerCgIters) +
          ") are peclet.voro.defaults. Returns an OptimizeResult (positions, weights, iters,\n"
          "max_vol_err, mean_vol_err, converged, num_empty). Pure Voronoi (use_weights=False)\n"
          "reaches equal/graded volumes well; weights add fuller volume control but are limited "
          "by\nthe periodic tessellation's ~1% min-image floor."));

  m.def(
      "minimize_interface",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<int, nb::c_contig> type_in,
         std::array<real_t, 3> extent, real_t sigma, int search_window, int max_iter, real_t tol) {
        auto pos = flatten3(pos_in);
        const int N = (int)type_in.shape(0);
        if ((int)(pos.size() / 3) != N)
          throw std::runtime_error(
              "minimize_interface: positions (N,3) and types (N,) must agree on N");
        for (int a = 0; a < 3; ++a)
          if (!(extent[a] > real_t(0)))
            throw std::invalid_argument(
                "voro: minimize_interface(extent=...) needs three positive lengths.");
        std::vector<int> type(type_in.data(), type_in.data() + N);
        const real_t Larr[3] = {extent[0], extent[1], extent[2]};
        auto R = peclet::voro::interfaceMinimize<real_t>(
            pos, type, sigma, Larr, N, search_window, peclet::voro::NoSdf{}, max_iter, tol, false);
        InterfaceResult r;
        r.positions = nb::cast(toNumpy3(std::move(pos)));
        r.energy = R.energy;
        r.energy_ratio = R.energyRatio;
        r.iters = R.iters;
        r.converged = R.converged;
        return r;
      },
      nb::arg("positions"), nb::arg("types"), nb::arg("extent"), nb::kw_only(),
      nb::arg("sigma") = kInterfaceSigma, nb::arg("search_window") = kOptimizerSearchWindow,
      nb::arg("max_iter") = kOptimizerMaxIter, nb::arg("tol") = kOptimizerTolerance,
      doc("Surface-Evolver-style interfacial-tension minimiser: move seeds (N,3) on the periodic "
          "box\n`extent` to minimise the total area of faces between cells of different integer "
          "type (N,),\nE = sum sigma A_ij (sigma default " +
          fmt(kInterfaceSigma) +
          "). Steepest descent with a trust-region line search on\nthe (non-smooth) interfacial "
          "energy; search_window / max_iter / tol default to " +
          fmt(kOptimizerSearchWindow) + " / " + fmt(kOptimizerMaxIter) + " / " +
          fmt(kOptimizerTolerance) +
          ".\nReturns an InterfaceResult (positions, energy, energy_ratio = final/initial, iters, "
          "converged)."));
}
}  // namespace peclet::voro::pybind
