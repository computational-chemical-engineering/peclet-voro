/**
 * @file test_incremental_skin.cpp
 * \brief Phase-5 acceptance: incremental update tracks full rebuild + skin hook.
 *
 * Moves a seed set along a trajectory, advancing one CellComplex by the
 * incremental update() and comparing it, every step, against a freshly built
 * reference complex: per-cell volumes must agree and the box must stay filled.
 * Also exercises SkinRefresh — the Verlet-style hook the distributed scheme-C
 * driver (Phase 6) keys its halo re-gather off.
 *
 * No Boost (std::mt19937). Legacy CPU path (the incremental update stays on CPU
 * by design; the GPU does full rebuilds).
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>
#include <vorflow/skin_refresh.hpp>
#include <vorflow/voronoi.hpp>

using std::vector;
using real_t = double;
using Vec3 = std::array<real_t, 3>;

namespace {

// Compare two complexes by per-cell volume keyed on seed id. Returns max rel err.
real_t compareVolumes(vor::CellComplex<real_t>& a, vor::CellComplex<real_t>& b) {
  vector<vor::Cell<real_t> > ca, cb;
  a.materializeCells(ca);
  b.materializeCells(cb);
  const vector<vor::CellGeometry<real_t> >& ga = a.getGeoms();
  const vector<vor::CellGeometry<real_t> >& gb = b.getGeoms();
  std::map<vor::uint2, real_t> volB;
  for (size_t i = 0; i < cb.size(); ++i)
    volB[cb[i].getID()] = gb[i].getVolume();
  real_t worst = 0;
  for (size_t i = 0; i < ca.size(); ++i) {
    auto it = volB.find(ca[i].getID());
    if (it == volB.end())
      return 1e30;
    real_t va = ga[i].getVolume();
    real_t rel = std::fabs(va - it->second) / it->second;
    if (rel > worst)
      worst = rel;
  }
  return worst;
}

}  // namespace

int main() {
  const int N = 800;
  const Vec3 L = {1.0, 1.0, 1.0};
  const real_t boxVol = L[0] * L[1] * L[2];
  std::mt19937 rng(2024);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  std::normal_distribution<real_t> G(0.0, 1.0);

  vector<Vec3> pos(N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < 3; ++d)
      pos[i][d] = L[d] * U(rng);

  vor::Box<real_t> box(L);
  vor::CellComplex<real_t> incr(&box);
  incr.build(pos);

  const real_t skin = 0.1;  // ~ mean spacing (spacing ~0.108 here)
  vor::SkinRefresh<real_t> refresh(skin);
  refresh.reset(pos);

  const real_t step = 0.01;  // per-step rms displacement
  const int T = 40;
  int failures = 0, rebuildEvents = 0;
  real_t worstVol = 0, worstFill = 0;

  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d)
        pos[i][d] += step * G(rng);
    box.putInBox(pos);

    incr.update(pos);  // incremental

    // Fresh reference each step.
    vor::CellComplex<real_t> ref(&box);
    ref.build(pos);

    real_t volErr = compareVolumes(incr, ref);
    if (volErr > worstVol)
      worstVol = volErr;

    const vector<vor::CellGeometry<real_t> >& gi = incr.getGeoms();
    real_t total = 0;
    for (size_t i = 0; i < gi.size(); ++i)
      total += gi[i].getVolume();
    real_t fill = std::fabs(total - boxVol) / boxVol;
    if (fill > worstFill)
      worstFill = fill;

    // Skin hook: rebuild reference when drift exceeds skin/2.
    if (refresh.needsRebuild(pos, L)) {
      ++rebuildEvents;
      refresh.reset(pos);
    }
  }

  if (worstVol > 1e-9) {
    std::fprintf(stderr, "FAIL incremental vs rebuild: worst per-cell vol rel err %.3e\n",
                 worstVol);
    ++failures;
  }
  if (worstFill > 1e-9) {
    std::fprintf(stderr, "FAIL space-filling: worst %.3e\n", worstFill);
    ++failures;
  }
  if (rebuildEvents == 0) {
    std::fprintf(stderr, "FAIL skin hook never fired over %d steps (motion too small?)\n", T);
    ++failures;
  }

  std::printf(
      "incremental-vs-rebuild: steps=%d worstVolErr=%.2e worstFill=%.2e skinRebuilds=%d  %s\n", T,
      worstVol, worstFill, rebuildEvents, failures == 0 ? "PASS" : "FAIL");
  std::printf("%s\n", failures == 0 ? "incremental_skin PASS" : "incremental_skin FAIL");
  return failures;
}
