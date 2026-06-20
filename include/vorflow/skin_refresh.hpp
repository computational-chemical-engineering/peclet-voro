/**
 * @file skin_refresh.hpp
 * \brief Verlet-style skin hook for incremental / distributed refresh (Phase 5).
 *
 * The incremental tessellation (and, in the distributed case, the owner<->ghost
 * halo) may reuse its topology across steps as long as no seed has moved far
 * enough for a Voronoi neighbour to cross the gathered skin. This host utility
 * tracks the reference positions captured at the last (re)build and reports when
 * the maximum seed displacement exceeds skin/2 — the standard neighbour-list
 * criterion (two seeds can close the skin between them, hence half each).
 *
 * It is the hook the scheme-C distributed driver (Phase 6) keys its halo
 * re-gather off, and the trigger for a full incremental resync. Host-only, core
 * side (no physics, no engine include — operates on plain position arrays + L).
 */
#ifndef VORFLOW_SKIN_REFRESH_HPP
#define VORFLOW_SKIN_REFRESH_HPP

#include <array>
#include <cmath>
#include <vector>

namespace vor {

template <class Real>
class SkinRefresh {
 public:
  using Vec3 = std::array<Real, 3>;

  SkinRefresh() = default;
  explicit SkinRefresh(Real skin) : m_skin(skin) {}

  void setSkin(Real skin) { m_skin = skin; }
  Real skin() const { return m_skin; }

  /// Capture the reference configuration (called right after a (re)build).
  void reset(const std::vector<Vec3>& pos) { m_ref = pos; }

  bool hasReference() const { return !m_ref.empty(); }

  /// Largest per-seed displacement from the reference (minimal image in box L).
  Real maxDisplacement(const std::vector<Vec3>& pos, const Vec3& L) const {
    Real worst = 0;
    const std::size_t n = pos.size() < m_ref.size() ? pos.size() : m_ref.size();
    for (std::size_t i = 0; i < n; ++i) {
      Real d2 = 0;
      for (int k = 0; k < 3; ++k) {
        Real d = pos[i][k] - m_ref[i][k];
        d -= L[k] * std::round(d / L[k]);  // minimal image
        d2 += d * d;
      }
      if (d2 > worst)
        worst = d2;
    }
    return std::sqrt(worst);
  }

  /// True when the configuration has drifted enough to require a rebuild, i.e.
  /// max displacement > skin/2 (or there is no reference yet, or the seed count
  /// changed).
  bool needsRebuild(const std::vector<Vec3>& pos, const Vec3& L) const {
    if (m_ref.size() != pos.size())
      return true;
    return maxDisplacement(pos, L) > Real(0.5) * m_skin;
  }

 private:
  Real m_skin = 0;
  std::vector<Vec3> m_ref;
};

}  // namespace vor

#endif  // VORFLOW_SKIN_REFRESH_HPP
