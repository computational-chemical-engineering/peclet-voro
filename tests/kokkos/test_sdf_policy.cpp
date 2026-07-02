/**
 * @file test_sdf_policy.cpp
 * @brief P5 acceptance test for the differentiable SDF wall force (Effort 2, Option A).
 *
 * A cell clipped by an SDF solid gains wall facets (pnbr == kBoundaryFacet). addSdfWallForce models
 * each as the tangent plane at the seed's foot point on sdf=0 and adds its dV/dseed contribution
 * (J_wallᵀ g) to the particle self-force from chainToDofs. We finite-difference the volume of a
 * cell (neighbour bisectors + SDF wall) w.r.t. the seed and compare to the analytic force:
 *
 *   (A) tilted FLAT wall  — φ linear ⇒ H=0, one facet ⇒ the model is EXACT (match ~1e-6);
 *   (B) SPHERE            — curved ⇒ the seed-foot model is FIRST-ORDER; reported, and required only
 *                           to be "close" (the clip approximates the curve by several facets).
 *
 * Interior cells only: the cell must be fully bounded by neighbour planes + the wall (no seed-box
 * face, which would move with the seed and contaminate the FD).
 */
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/plane_policy.hpp"
#include "peclet/voro/sdf.hpp"

using real_t = double;
using Cell = peclet::voro::ConvexCell<real_t, 128, 256>;

namespace {

// A tilted planar wall: fluid where n·x > d (unit n ⇒ a true SDF, H=0).
struct FlatWall {
  real_t nx, ny, nz, d;
  KOKKOS_INLINE_FUNCTION real_t eval(real_t x, real_t y, real_t z) const {
    return nx * x + ny * y + nz * z - d;
  }
  KOKKOS_INLINE_FUNCTION real_t gradH() const { return real_t(1e-4); }
};

// Build a cell for `seed` from the given absolute neighbour positions (Voronoi bisectors), then clip
// by the SDF. Returns false if the cell is empty, overflowed, has no wall facet, or is not fully
// enclosed (a nonzero-area seed-box face remains).
template <class Sdf>
bool buildWallCell(const real_t seed[3], const std::vector<real_t>& nbr, const Sdf& sdf, Cell& c) {
  c.initBox(100.0, 100.0, 100.0);  // large box: its faces must not survive on an interior cell
  const int M = (int)nbr.size() / 3;
  for (int j = 0; j < M; ++j) {
    real_t r[3] = {nbr[3 * j] - seed[0], nbr[3 * j + 1] - seed[1], nbr[3 * j + 2] - seed[2]};
    const real_t off = real_t(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    c.clip(r, off, j);
    if (c.overflow) return false;
  }
  peclet::voro::clipCellAgainstSdf<real_t, 128, 256, false>(c, seed, sdf);
  if (c.empty() || c.overflow) return false;
  double area[128];
  for (int k = 0; k < c.np; ++k) area[k] = 0.0;
  c.facetAreasPerVertex(area);
  bool hasWall = false;
  for (int k = 0; k < c.np; ++k) {
    if (c.pnbr[k] == -1 && area[k] > 1e-12) return false;  // seed-box face: not interior
    if (c.pnbr[k] == peclet::voro::kBoundaryFacet && area[k] > 1e-12) hasWall = true;
  }
  return hasWall;
}

// Topology signature: sorted list of face-plane neighbour ids (real + wall + box), so a face
// gain/loss between two nearby seeds is detectable (an FD across it must be skipped).
std::vector<int> topoSig(const Cell& c) {
  double area[128];
  for (int k = 0; k < c.np; ++k) area[k] = 0.0;
  const_cast<Cell&>(c).facetAreasPerVertex(area);
  std::vector<int> sig;
  for (int k = 0; k < c.np; ++k)
    if (area[k] > 1e-10) sig.push_back(c.pnbr[k]);
  std::sort(sig.begin(), sig.end());
  return sig;
}

// FD-vs-analytic wall force over many random interior cells against `sdf`. Returns worst relative
// error on FD-resolvable components; sets nCells to the number checked.
template <class Sdf>
double sweep(const Sdf& sdf, real_t seedPhiTarget, unsigned seed, int trials, double tolPass,
             long& nCells, long& nComp, long& nPass) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real_t> U(0.0, 1.0);
  // FD-resolvable floor: below this the central-difference round-off (~vol_eps/2eps) swamps the
  // relative error, so only well-resolved force components are compared.
  const real_t eps = 1e-6, sigFloor = 1e-4, relFloor = 1e-30;
  double maxRel = 0.0;
  nCells = 0;
  nComp = 0;
  nPass = 0;
  for (int t = 0; t < trials; ++t) {
    // seed a bit above the surface: march from a random point to φ≈seedPhiTarget along ∇φ.
    real_t s[3] = {2 * U(rng) - 1, 2 * U(rng) - 1, 2 * U(rng) - 1};
    for (int it = 0; it < 40; ++it) {
      real_t g[3];
      peclet::voro::sdfGradient<real_t>(sdf, s[0], s[1], s[2], g);
      const real_t gn = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
      if (gn < 1e-9) break;
      const real_t phi = sdf.eval(s[0], s[1], s[2]);
      for (int d = 0; d < 3; ++d) s[d] += (seedPhiTarget - phi) * g[d] / (gn * gn);
    }
    // random neighbours in a ball around the seed, kept in the fluid (φ>0.1).
    std::vector<real_t> nbr;
    const int want = 90;
    for (int a = 0; a < want; ++a) {
      real_t d[3] = {2 * U(rng) - 1, 2 * U(rng) - 1, 2 * U(rng) - 1};
      const real_t rr = 0.8 + 1.4 * U(rng);
      const real_t nn = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) + 1e-12;
      real_t p[3] = {s[0] + rr * d[0] / nn, s[1] + rr * d[1] / nn, s[2] + rr * d[2] / nn};
      if (sdf.eval(p[0], p[1], p[2]) > 0.1) {
        nbr.push_back(p[0]);
        nbr.push_back(p[1]);
        nbr.push_back(p[2]);
      }
    }
    Cell c;
    if (!buildWallCell(s, nbr, sdf, c)) continue;
    ++nCells;

    // analytic: dV/dn_k -> particle self-force (Voronoi chain) + SDF wall self-force.
    double vg = 0, dgx[128], dgy[128], dgz[128];
    for (int k = 0; k < c.np; ++k) dgx[k] = dgy[k] = dgz[k] = 0.0;
    c.geomVolumeGrad(vg, dgx, dgy, dgz);
    double fSelf[3], fwSelf, fnx[128], fny[128], fnz[128], fwn[128];
    peclet::voro::chainToDofs<peclet::voro::Voronoi>(c, s, (const double*)nullptr, 0.0,
                                                     (const double*)nullptr, 1e30, dgx, dgy, dgz,
                                                     fSelf, fwSelf, fnx, fny, fnz, fwn);
    peclet::voro::addSdfWallForce(c, s, sdf, dgx, dgy, dgz, fSelf);

    const std::vector<int> sig0 = topoSig(c);
    for (int cc = 0; cc < 3; ++cc) {
      real_t sp[3] = {s[0], s[1], s[2]};
      Cell cp, cm;
      sp[cc] = s[cc] + eps;
      if (!buildWallCell(sp, nbr, sdf, cp)) continue;
      sp[cc] = s[cc] - eps;
      if (!buildWallCell(sp, nbr, sdf, cm)) continue;
      if (topoSig(cp) != sig0 || topoSig(cm) != sig0) continue;  // face gain/loss: FD invalid, skip
      const double fd = (cp.volumePerVertex() - cm.volumePerVertex()) / (2 * eps);
      if (std::abs(fd) < sigFloor) continue;
      ++nComp;
      const double rel = std::abs(fd - fSelf[cc]) / std::max(std::abs(fd), relFloor);
      maxRel = std::max(maxRel, rel);
      if (rel < tolPass) ++nPass;
    }
  }
  return maxRel;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("differentiable SDF wall force (Effort 2, Option A):\n");
    // (A) flat wall: exact.
    {
      const real_t n0[3] = {0.3, -0.4, 0.8660254};  // unit-ish; normalise
      const real_t nn = std::sqrt(n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2]);
      FlatWall wall{n0[0] / nn, n0[1] / nn, n0[2] / nn, -0.4};  // fluid n·x > -0.4
      long nc = 0, ncomp = 0, npass = 0;
      const double mr = sweep(wall, 0.4, 12345u, 300, 1e-5, nc, ncomp, npass);
      // exact model (H=0): every well-resolved component matches; allow a rare near-degenerate FD
      // artifact but no systematic error (maxRel far below a sign/scale mistake).
      const bool pass = nc > 20 && ncomp > 20 && (double)npass / ncomp > 0.99 && mr < 1e-2;
      std::printf("  (A) flat wall   cells=%ld comps=%ld pass(<1e-5)=%ld/%ld maxRel=%.2e  %s\n", nc,
                  ncomp, npass, ncomp, mr, pass ? "OK" : "FAIL");
      rc |= pass ? 0 : 1;
    }
    // (B) sphere: first-order (curved). Report; require only "close".
    {
      peclet::voro::SdfSphere<real_t> ball{0.0, 0.0, 0.0, 3.0};  // solid ball radius 3, seed outside
      long nc = 0, ncomp = 0, npass = 0;
      const double mr = sweep(ball, 0.5, 999u, 400, 1e-2, nc, ncomp, npass);
      // first-order surrogate on a curved wall: the great majority within 1e-2, none wildly off.
      const bool pass = nc > 20 && ncomp > 20 && (double)npass / ncomp > 0.9 && mr < 0.2;
      std::printf("  (B) sphere      cells=%ld comps=%ld pass(<1e-2)=%ld/%ld maxRel=%.2e "
                  "(first-order)  %s\n",
                  nc, ncomp, npass, ncomp, mr, pass ? "OK" : "FAIL");
      rc |= pass ? 0 : 1;
    }
    std::printf("%s\n", rc == 0 ? "SDF WALL-FORCE CHECKS PASS" : "SDF WALL-FORCE CHECKS FAILED");
  }
  Kokkos::finalize();
  return rc;
}
