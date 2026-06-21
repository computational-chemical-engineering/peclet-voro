/**
 * @file host/interface_force.hpp
 * \brief Multiphase interface-tension force on the new cutter (de-legacy Phase 3).
 *
 * Faithful port of IntfDyn::computeIntfForces. For each cell of phase t_i, for each
 * facet to a LOWER-phase neighbour, the gradient of that facet's area (gradFacetAreaSq
 * on the live ScratchCell mesh) is turned into a surface-tension force on the seeds
 * of the facet's edge-adjacent neighbours and the cell itself:
 *   df = -γ(t_i,t_j) · ∂A²/∂x / (2·A) ;  F[nbr2] += df ;  F[self] -= df.
 *
 * Host driver: rebuilds each cell with the cutter (the half-edge mesh + edgeInv are
 * needed), serial accumulation (the scatter to neighbours overlaps between cells).
 * The GPU version would fold this into the build kernel where the cell is live.
 */
#ifndef VORFLOW_HOST_INTERFACE_FORCE_HPP
#define VORFLOW_HOST_INTERFACE_FORCE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "vorflow/device/cell_cutter.hpp"

namespace vor {
namespace host {

/// Accumulate the interface-tension force (size N) for a multiphase seed set.
/// tension is row-major nTypes×nTypes; types is per seed.
template <class Real>
std::vector<std::array<Real, 3>> interfaceForce(const std::vector<std::array<Real, 3>>& pos,
                                                const std::vector<int>& types,
                                                const std::vector<Real>& tension, int nTypes,
                                                const std::array<Real, 3>& L, Real rcut) {
  const int N = static_cast<int>(pos.size());
  std::vector<std::array<Real, 3>> force(N, {0, 0, 0});
  const Real Larr[3] = {L[0], L[1], L[2]};
  const Real cut2 = rcut * rcut;
  std::vector<Real> rx, ry, rz, key;
  std::vector<int> ids, idx;
  for (int i = 0; i < N; ++i) {
    if (types[i] == 0)
      continue;
    // gather candidates within rcut (minimal image)
    rx.clear();
    ry.clear();
    rz.clear();
    ids.clear();
    for (int j = 0; j < N; ++j) {
      if (j == i)
        continue;
      Real dx = pos[j][0] - pos[i][0], dy = pos[j][1] - pos[i][1], dz = pos[j][2] - pos[i][2];
      dx -= L[0] * std::round(dx / L[0]);
      dy -= L[1] * std::round(dy / L[1]);
      dz -= L[2] * std::round(dz / L[2]);
      if (dx * dx + dy * dy + dz * dz <= cut2) {
        rx.push_back(dx);
        ry.push_back(dy);
        rz.push_back(dz);
        ids.push_back(j);
      }
    }
    const int nc = static_cast<int>(ids.size());
    key.resize(nc);
    idx.resize(nc);
    for (int n = 0; n < nc; ++n) {
      key[n] = Real(0.5) * (rx[n] * rx[n] + ry[n] * ry[n] + rz[n] * rz[n]);
      idx[n] = n;
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return key[a] < key[b]; });
    std::vector<Real> sx(nc), sy(nc), sz(nc);
    std::vector<int> sid(nc);
    for (int n = 0; n < nc; ++n) {
      sx[n] = rx[idx[n]];
      sy[n] = ry[idx[n]];
      sz[n] = rz[idx[n]];
      sid[n] = ids[idx[n]];
    }
    device::ScratchCell<Real> c;
    device::buildVoronoiCell<Real, false>(c, Larr, sx.data(), sy.data(), sz.data(), sid.data(),
                                          nullptr, nc, Real(0));
    if (c.emptyV())
      continue;
    c.computeGeometry();
    for (int f = 0; f < c.numAllocF; ++f) {
      if (!c.aliveF[f])
        continue;
      const int nbr = c.fnbr[f];
      if (nbr < 0 || nbr >= N)
        continue;
      if (types[nbr] >= types[i])
        continue;  // count each interface once
      const Real* A = c.fArea[f];
      const Real area = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
      if (area <= Real(0))
        continue;
      const Real gamma = tension[types[i] * nTypes + types[nbr]];
      int outF[41];
      Real outG[41][3];
      int numF = 0;
      c.gradFacetAreaSq(f, outF, outG, numF);
      for (int m = 0; m < numF; ++m) {
        const int nbr2 = c.fnbr[outF[m]];
        if (nbr2 < 0 || nbr2 >= N)
          continue;
        for (int k = 0; k < 3; ++k) {
          const Real df = -gamma * outG[m][k] / (Real(2) * area);
          force[nbr2][k] += df;
          force[i][k] -= df;
        }
      }
    }
  }
  return force;
}

}  // namespace host
}  // namespace vor

#endif  // VORFLOW_HOST_INTERFACE_FORCE_HPP
