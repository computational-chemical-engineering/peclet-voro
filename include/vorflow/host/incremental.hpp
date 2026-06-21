/**
 * @file host/incremental.hpp
 * \brief CPU incremental tessellation on the new device-callable cutter (Phase 6).
 *
 * The suite architecture keeps the incremental update on the CPU (the GPU does a
 * full rebuild each step). This host driver replaces the legacy
 * CellComplex::update: a full build gathers each cell's neighbour candidates
 * (one security radius + skin deep) and caches them; while the Verlet skin holds
 * (no seed has moved more than skin/2, SkinRefresh) an update just re-cuts every
 * cell from its cached candidates at the new positions — no neighbour re-search.
 * The cached list still contains every true neighbour, so the result is identical
 * to a full rebuild. When the skin is crossed, it re-gathers (full build).
 *
 * It calls the same ScratchCell cutter as the GPU path, so CPU-incremental and
 * GPU-full-rebuild share one cell-construction kernel. Host-only (no Kokkos
 * parallel needed — the cutter is plain inline code); OpenMP-parallel over cells.
 */
#ifndef VORFLOW_HOST_INCREMENTAL_HPP
#define VORFLOW_HOST_INCREMENTAL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "vorflow/device/cell_cutter.hpp"
#include "vorflow/skin_refresh.hpp"

namespace vor {
namespace host {

template <class Real>
class Incremental {
 public:
  using Vec3 = std::array<Real, 3>;

  /// Full build: gather each cell's candidates (within rcut) and cache them.
  /// @param skin  Verlet skin; the cached lists stay valid until a seed moves > skin/2.
  void build(const std::vector<Vec3>& pos, const Vec3& L, Real rcut, Real skin) {
    L_ = L;
    rcut_ = rcut;
    skin_.setSkin(skin);
    const int N = static_cast<int>(pos.size());
    cand_.assign(N, {});
    const Real cut = rcut + skin;  // cache a little extra so the skin stays covered
    const Real cut2 = cut * cut;
    // Brute-force gather (the test sizes are small); a grid would replace this at
    // scale. Cache every seed whose minimal image is within rcut+skin.
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        if (j == i)
          continue;
        Real d2 = 0;
        for (int k = 0; k < 3; ++k) {
          Real d = pos[j][k] - pos[i][k];
          d -= L[k] * std::round(d / L[k]);
          d2 += d * d;
        }
        if (d2 <= cut2)
          cand_[i].push_back(j);
      }
    }
    skin_.reset(pos);
    retessellate(pos);
  }

  /// Update to new positions. Returns true if a full re-gather was triggered.
  bool update(const std::vector<Vec3>& pos) {
    if (skin_.needsRebuild(pos, L_)) {
      build(pos, L_, rcut_, skin_.skin());
      return true;
    }
    retessellate(pos);  // reuse cached candidates
    return false;
  }

  const std::vector<Real>& volumes() const { return vol_; }
  const std::vector<int>& facetCount() const { return nf_; }

 private:
  void retessellate(const std::vector<Vec3>& pos) {
    const int N = static_cast<int>(pos.size());
    vol_.assign(N, Real(0));
    nf_.assign(N, 0);
    const Real Larr[3] = {L_[0], L_[1], L_[2]};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int i = 0; i < N; ++i) {
      const std::vector<int>& cand = cand_[i];
      const int nc = static_cast<int>(cand.size());
      std::vector<Real> rx(nc), ry(nc), rz(nc), key(nc);
      std::vector<int> idx(nc);
      for (int n = 0; n < nc; ++n) {
        int j = cand[n];
        Real dx = pos[j][0] - pos[i][0], dy = pos[j][1] - pos[i][1], dz = pos[j][2] - pos[i][2];
        dx -= L_[0] * std::round(dx / L_[0]);
        dy -= L_[1] * std::round(dy / L_[1]);
        dz -= L_[2] * std::round(dz / L_[2]);
        rx[n] = dx;
        ry[n] = dy;
        rz[n] = dz;
        key[n] = Real(0.5) * (dx * dx + dy * dy + dz * dz);
        idx[n] = n;
      }
      std::sort(idx.begin(), idx.end(), [&](int a, int b) { return key[a] < key[b]; });
      std::vector<Real> sx(nc), sy(nc), sz(nc);
      std::vector<int> sid(nc);
      for (int n = 0; n < nc; ++n) {
        sx[n] = rx[idx[n]];
        sy[n] = ry[idx[n]];
        sz[n] = rz[idx[n]];
        sid[n] = cand[idx[n]];
      }
      device::ScratchCell<Real> c;
      device::buildVoronoiCell<Real, false>(c, Larr, sx.data(), sy.data(), sz.data(), sid.data(),
                                            nullptr, nc, Real(0));
      vol_[i] = c.volume();
      nf_[i] = c.countFacets();
    }
  }

  Vec3 L_{};
  Real rcut_ = 0;
  SkinRefresh<Real> skin_;
  std::vector<std::vector<int>> cand_;
  std::vector<Real> vol_;
  std::vector<int> nf_;
};

}  // namespace host
}  // namespace vor

#endif  // VORFLOW_HOST_INCREMENTAL_HPP
