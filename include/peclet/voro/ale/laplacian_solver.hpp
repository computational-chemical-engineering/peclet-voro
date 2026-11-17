/**
 * @file ale/laplacian_solver.hpp
 * \brief Moving-grid stage 1, WO-S1.2b (docs/ale_design.md §4.5 with §11 A10): the L_ρ solver of
 * the time step — the matrix of `assembleLaplacian` handed to core's `GraphAMGDevice` with the
 * parallel MIS-2 setup on the device and no label (A10), one V-cycle preconditioning a device PCG
 * (`krylov::pcg`) with mean removal when singular, warm start, and the stopping test of the caller
 * (relative for mode F and μ̂, cell-wise absolute for the compact projections, §4.4, §4.5 item 7).
 *
 * The setup is a call per frame (`setup`), on the device: the matrix's int CSR is widened to
 * core's `DeviceCsr` (Index columns) and `GraphAMGDevice::build(DeviceCsr, prm)` runs the MIS-2
 * aggregation, the Galerkin products and the smoother set-up there. The parameters are those of
 * gate G0.4 (`runFile` of core's test_graph_amg_parallel): `Aggregation::Mis2`,
 * `CoarseSolve::Dense`, the rest at core's defaults. No multigrid of voro's own, no change in core.
 *
 * Single rank (WO-S1.4 adds the distributed hooks): `n` = the owned rows; a periodic box makes L_ρ
 * singular with the constants as null space, which `singular` (default true) deflates.
 */
#ifndef PECLET_VORO_ALE_LAPLACIAN_SOLVER_HPP
#define PECLET_VORO_ALE_LAPLACIAN_SOLVER_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "peclet/core/solver/graph_amg_device.hpp"
#include "peclet/core/solver/graph_amg_setup.hpp"
#include "peclet/voro/ale/krylov.hpp"
#include "peclet/voro/ale/operators.hpp"

namespace peclet::voro::ale {

/// The AMG parameters of the L_ρ solver (those of gate G0.4).
inline peclet::core::solver::AmgParams laplacianAmgParams() {
  peclet::core::solver::AmgParams prm;
  prm.aggregation = peclet::core::solver::Aggregation::Mis2;
  prm.coarseSolve = peclet::core::solver::CoarseSolve::Dense;
  return prm;
}

template <class Real>
class LaplacianSolver {
  static_assert(std::is_same_v<Real, double>,
                "ale::LaplacianSolver: core's GraphAMGDevice is FP64");

 public:
  using Exec = peclet::core::ExecSpace;
  using Index = peclet::core::Index;

  int maxIterations = 1000;
  bool singular = true;  ///< deflate the constants (all-periodic box)

  /// Build the hierarchy for the matrix `L` of this frame (device setup, every frame). `L` is
  /// referenced, not copied: it must outlive the solver's use for this frame.
  void setup(const LaplacianMatrix<Real>& L,
             const peclet::core::solver::AmgParams& prm = laplacianAmgParams()) {
    L_ = L;
    n_ = L.nRows;
    peclet::core::solver::DeviceCsr d;
    d.n = n_;
    const int nnz = (int)L.col.extent(0);
    d.diag = peclet::core::View<double>("ale.lap.diag", n_);
    d.start = peclet::core::View<Index>("ale.lap.start", n_ + 1);
    d.nbr = peclet::core::View<Index>("ale.lap.nbr", nnz);
    d.coef = peclet::core::View<double>("ale.lap.coef", nnz);
    Kokkos::deep_copy(d.diag, L.diag);
    Kokkos::deep_copy(d.coef, L.coef);
    const auto start = L.start;
    const auto col = L.col;
    const auto ds = d.start;
    const auto dn = d.nbr;
    Kokkos::parallel_for(
        "ale.lap.widenStart", Kokkos::RangePolicy<Exec>(0, n_ + 1),
        KOKKOS_LAMBDA(const int i) { ds(i) = start(i); });
    Kokkos::parallel_for(
        "ale.lap.widenCol", Kokkos::RangePolicy<Exec>(0, nnz),
        KOKKOS_LAMBDA(const int q) { dn(q) = col(q); });
    amg_ = peclet::core::solver::GraphAMGDevice();
    amg_.build(d, prm);
  }

  /// z = one V-cycle of the hierarchy applied to r (owned entries); the mean of z is NOT removed.
  void precondition(const DV<Real>& r, const DV<Real>& z) const {
    if ((int)r.extent(0) == n_ && (int)z.extent(0) == n_) {
      amg_.apply(r, z);
      return;
    }
    amg_.apply(Kokkos::subview(r, std::make_pair(0, n_)),
               Kokkos::subview(z, std::make_pair(0, n_)));
  }

  /// out = L x.
  void apply(const DV<Real>& x, const DV<Real>& out) const { applyLaplacian(L_, x, out); }

  /// Solve L x = b by AMG-PCG from the initial guess in `x` (warm start; zero it for a cold
  /// start). With `singular`, the mean of b is ignored and x is returned with zero mean.
  krylov::SolveReport solve(const DV<Real>& b, const DV<Real>& x,
                            const krylov::PcgTest<Real>& test) {
    const auto& self = *this;
    auto op = [&self](const DV<Real>& in, const DV<Real>& out) { self.apply(in, out); };
    auto prec = [&self](const DV<Real>& in, const DV<Real>& out) { self.precondition(in, out); };
    last_ = krylov::pcg<Real>(op, prec, b, x, test, singular, maxIterations, work_, n_);
    return last_;
  }

  int size() const { return n_; }
  int numLevels() const { return amg_.numLevels(); }
  Index levelSize(int l) const { return amg_.size(l); }
  const krylov::SolveReport& lastReport() const { return last_; }
  const peclet::core::solver::GraphAMGDevice& hierarchy() const { return amg_; }

 private:
  LaplacianMatrix<Real> L_;
  int n_ = 0;
  peclet::core::solver::GraphAMGDevice amg_;
  krylov::PcgWork<Real> work_;
  krylov::SolveReport last_;
};

}  // namespace peclet::voro::ale

#endif  // PECLET_VORO_ALE_LAPLACIAN_SOLVER_HPP
