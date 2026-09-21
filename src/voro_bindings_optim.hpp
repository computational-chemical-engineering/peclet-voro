/// @file
/// @brief The two mesh-optimiser helpers shared by `voro_optimizers.cpp` and `voro_pore.cpp`.
#ifndef PECLET_VORO_BINDINGS_OPTIM_HPP
#define PECLET_VORO_BINDINGS_OPTIM_HPP

#include "peclet/voro/mesh_optimizer.hpp"
#include "voro_bindings_common.hpp"

namespace peclet::voro::pybind {

// The CG preconditioner / descent method of the mesh optimisers, as a string mode.
constexpr const char* kMethodList = "'jacobi', 'colored_gs', 'graphamg', 'steepest'";
inline peclet::voro::Precond parseMethod(const std::string& m, const char* fn) {
  if (m == "jacobi")
    return peclet::voro::Precond::Jacobi;
  if (m == "colored_gs")
    return peclet::voro::Precond::ColoredGS;
  if (m == "graphamg")
    return peclet::voro::Precond::GraphAMG;
  if (m == "steepest")
    return peclet::voro::Precond::SteepestDescent;
  throw std::invalid_argument(std::string("voro: ") + fn + "(method='" + m +
                              "') is not a method; accepted: " + kMethodList + ".");
}

inline OptimizeResult makeOptimizeResult(std::vector<real_t> pos,
                                         std::optional<std::vector<real_t>> w,
                                         const peclet::voro::OtResult& R) {
  OptimizeResult r;
  r.positions = nb::cast(toNumpy3(std::move(pos)));
  r.weights = w ? nb::cast(toNumpy1(std::move(*w))) : nb::none();
  r.iters = R.iters;
  r.max_vol_err = R.maxVolErr;
  r.mean_vol_err = R.meanVolErr;
  r.converged = R.converged;
  r.num_empty = (int)R.nEmpty;
  return r;
}

}  // namespace peclet::voro::pybind

#endif  // PECLET_VORO_BINDINGS_OPTIM_HPP
