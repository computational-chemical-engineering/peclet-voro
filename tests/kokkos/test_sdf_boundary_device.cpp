/**
 * @file test_sdf_boundary_device.cpp
 * \brief Acceptance: device SDF boundary clip == legacy SignedDistanceBoundary.
 *
 * Tessellates a seed set around a solid with the device tessellator + an SDF
 * provider, and checks it against the legacy CellComplex driven by a
 * SignedDistanceBoundary adapter over the SAME geometry (value = the provider's
 * eval, gradient = the same central difference). For every seed: a fluid seed's
 * clipped-cell volume matches the legacy cell, a seed inside the solid has no
 * cell in either, and the clipped fluid cells tile box \ solid.
 *
 * Cut planes sit at sdf = 0 with normal ∇sdf — the requested geometry mechanism,
 * on the device path over the suite's tpx::geom. Both providers are covered:
 *   - analytic (SdfSphere / SdfBox), and
 *   - a sampled grid round-tripped through VTI (SdfGrid from tpx::geom::readVti).
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <vector>

#include "tpx/common/view.hpp"
#include "tpx/geom/grid_sdf.hpp"
#include "tpx/geom/sdf.hpp"
#include "tpx/geom/vti_io.hpp"
#include "vorflow/device/sdf.hpp"
#include "vorflow/device/tessellator.hpp"
#include "vorflow/voronoi.hpp"

using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {

// Legacy boundary adapter over anything with eval(x,y,z)/gradH() (an analytic
// device provider, or a host tpx::geom::GridSdf). gradient() is the SAME central
// difference the device uses, so the legacy and device clips see identical geometry.
template <class EvalSdf>
struct Adapter : vor::SignedDistanceBoundary<real_t> {
  EvalSdf s;
  explicit Adapter(const EvalSdf& d) : s(d) {}
  real_t value(const Vec3& x) const override { return s.eval(x[0], x[1], x[2]); }
  Vec3 gradient(const Vec3& x) const override {
    const real_t h = s.gradH();
    return {(s.eval(x[0] + h, x[1], x[2]) - s.eval(x[0] - h, x[1], x[2])) / (2 * h),
            (s.eval(x[0], x[1] + h, x[2]) - s.eval(x[0], x[1] - h, x[2])) / (2 * h),
            (s.eval(x[0], x[1], x[2] + h) - s.eval(x[0], x[1], x[2] - h)) / (2 * h)};
  }
};

// Host eval shim for a tpx::geom::GridSdf with the same gradH() the device SdfGrid uses.
struct HostGrid {
  const tpx::geom::GridSdf* g;
  real_t eval(real_t x, real_t y, real_t z) const { return g->eval({x, y, z}); }
  real_t gradH() const {
    real_t m = std::min({g->spacing[0], g->spacing[1], g->spacing[2]});
    return real_t(0.25) * m;
  }
};

template <class DevSdf>
int runCase(const char* tag, int N, Vec3 L, unsigned seed, const DevSdf& sdf,
            const vor::SignedDistanceBoundary<real_t>& bnd, double solidVol, double fluidTol) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::vector<Vec3> pos(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * U(rng);

  // Legacy reference: clip against the boundary. Cells are index-aligned (cell i ==
  // seed i for a dense build); a seed inside the solid is emptied in place (zero
  // volume), so volume == 0 marks "no fluid cell".
  vor::Box<real_t> box(L);
  vor::CellComplex<real_t> legacy(&box);
  legacy.setBoundary(&bnd);
  legacy.build(pos);
  const std::vector<vor::CellGeometry<real_t> >& lg = legacy.getGeoms();
  std::map<int, real_t> refVol;
  for (int i = 0; i < N && i < (int)lg.size(); ++i)
    if (lg[i].getVolume() > real_t(0))
      refVol[i] = lg[i].getVolume();

  // Device tessellation with the SDF clip.
  Kokkos::View<real_t*, tpx::MemSpace> dPos(
      Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), (size_t)N * 3);
  {
    auto h = Kokkos::create_mirror_view(dPos);
    for (int i = 0; i < N; ++i)
      for (int k = 0; k < 3; ++k)
        h(3 * i + k) = pos[i][k];
    Kokkos::deep_copy(dPos, h);
  }
  Kokkos::View<real_t*, tpx::MemSpace> dW("w", N);
  const real_t Larr[3] = {L[0], L[1], L[2]};
  auto res = vor::device::buildTessellation<real_t, false, DevSdf>(
      dPos, dW, N, Larr, /*sw=*/4, /*densityCount=*/N, /*gid=*/{}, sdf);
  auto vol = Kokkos::create_mirror_view(res.view.cellVolume);
  auto st = Kokkos::create_mirror_view(res.status);
  Kokkos::deep_copy(vol, res.view.cellVolume);
  Kokkos::deep_copy(st, res.status);

  const double boxVol = L[0] * L[1] * L[2];
  const double meanVol = (boxVol - solidVol) / std::max<size_t>(1, refVol.size());
  int volMis = 0, emptyMis = 0;
  double devFluid = 0, legacyFluid = 0, maxRel = 0;
  const real_t tol = 1e-6;  // relative to the mean fluid-cell volume
  for (int i = 0; i < N; ++i) {
    const bool devEmpty = (st(i) & vor::device::kEmpty) != 0;
    auto it = refVol.find(i);
    if (it == refVol.end()) {  // legacy: seed in solid -> no cell
      if (!devEmpty)
        ++emptyMis;
      continue;
    }
    if (devEmpty) {
      ++emptyMis;
      continue;
    }
    devFluid += vol(i);
    legacyFluid += it->second;
    real_t rel = std::fabs(vol(i) - it->second) / meanVol;
    if (rel > maxRel)
      maxRel = rel;
    if (rel > tol)
      ++volMis;
  }
  const double fluidErr = std::fabs(devFluid - (boxVol - solidVol)) / (boxVol - solidVol);

  int fail = 0;
  if (volMis || emptyMis) {
    std::fprintf(stderr, "  [%s] volMis=%d emptyMis=%d\n", tag, volMis, emptyMis);
    ++fail;
  }
  if (legacyFluid > 0 && std::fabs(devFluid - legacyFluid) / legacyFluid > 1e-12) {
    std::fprintf(stderr, "  [%s] fluid total dev=%.12f vs legacy=%.12f\n", tag, devFluid,
                 legacyFluid);
    ++fail;
  }
  if (fluidErr > fluidTol) {
    std::fprintf(stderr, "  [%s] fluid-fill err %.3e vs analytic\n", tag, fluidErr);
    ++fail;
  }
  std::printf("  [%s] N=%d fluidCells=%zu volMis=%d emptyMis=%d maxRelVol=%.2e fluidErr=%.2e  %s\n",
              tag, N, refVol.size(), volMis, emptyMis, maxRel, fluidErr,
              fail == 0 ? "PASS" : "FAIL");
  return fail;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    const Vec3 L = {1.0, 1.0, 1.0};

    // (1) Analytic solid ball (curved -> multi-plane clip).
    vor::device::SdfSphere<real_t> sphere{0.5, 0.5, 0.5, 0.25};
    const double ballVol = 4.0 / 3.0 * M_PI * 0.25 * 0.25 * 0.25;
    Adapter<vor::device::SdfSphere<real_t> > sphereBnd(sphere);
    failures += runCase("sphere", 3000, L, 5, sphere, sphereBnd, ballVol, 0.05);

    // (2) Analytic solid box (planar faces incl. creased edges/corners).
    vor::device::SdfBox<real_t> sbox{0.5, 0.5, 0.5, 0.2, 0.2, 0.2};
    const double boxSolid = 0.4 * 0.4 * 0.4;
    Adapter<vor::device::SdfBox<real_t> > boxBnd(sbox);
    failures += runCase("box", 3000, L, 11, sbox, boxBnd, boxSolid, 0.05);

    // (3) VTI / sampled grid: bake the ball onto a grid, round-trip through VTI,
    //     and evaluate it on device — the path real geometry takes.
    tpx::geom::Sphere shp{{0.5, 0.5, 0.5}, 0.25};
    const int n = 64;
    const double sp = 1.0 / (n - 1);
    tpx::geom::GridSdf grid = tpx::geom::sample(shp, {n, n, n}, {0.0, 0.0, 0.0}, {sp, sp, sp});
    const std::string path = "/tmp/vorflow_sphere_sdf.vti";
    tpx::geom::writeVti(path, grid);
    tpx::geom::GridSdf grid2 = tpx::geom::readVti(path);

    vor::device::SdfGrid<real_t> dgrid;
    {
      Kokkos::View<float*, tpx::MemSpace> vals(
          Kokkos::view_alloc(std::string("sdfvals"), Kokkos::WithoutInitializing),
          grid2.values.size());
      auto hv = Kokkos::create_mirror_view(vals);
      for (size_t i = 0; i < grid2.values.size(); ++i)
        hv(i) = grid2.values[i];
      Kokkos::deep_copy(vals, hv);
      dgrid.values = vals;
      dgrid.nx = (int)grid2.dims[0];
      dgrid.ny = (int)grid2.dims[1];
      dgrid.nz = (int)grid2.dims[2];
      dgrid.ox = grid2.origin[0];
      dgrid.oy = grid2.origin[1];
      dgrid.oz = grid2.origin[2];
      dgrid.sx = grid2.spacing[0];
      dgrid.sy = grid2.spacing[1];
      dgrid.sz = grid2.spacing[2];
    }
    Adapter<HostGrid> gridBnd(HostGrid{&grid2});
    // The trilinear grid sphere has more faceting error vs the analytic ball, so a
    // looser fluid-fill bound; the device-vs-legacy match stays machine precision.
    failures += runCase("vti-grid", 3000, L, 5, dgrid, gridBnd, ballVol, 0.05);
  }
  Kokkos::finalize();
  std::printf("%s\n", failures == 0 ? "sdf_boundary_device PASS" : "sdf_boundary_device FAIL");
  return failures;
}
