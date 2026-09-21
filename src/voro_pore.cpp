/// @file
/// @brief The SDF-walled pore-space family exposed by `peclet.voro.pore_mesh`.
#include "peclet/voro/convex_cell.hpp"
#include "peclet/voro/pore_cells.hpp"
#include "voro_bindings_optim.hpp"

namespace peclet::voro::pybind {

// ---- pore-space meshing helpers (SDF-walled interstitial Voronoi + geometry export) ----------
using PoreCell =
    peclet::voro::ConvexCell<real_t, defaults::kPoreMaxPlanes, defaults::kPoreMaxTriangles>;

// Build a periodic union-of-balls SDF from (M,3) centers + (M,) radii; the Views must outlive its
// use.
inline peclet::voro::SdfSpheres<real_t> makeSpheresSdf(nb::ndarray<real_t, nb::c_contig> centers,
                                                       nb::ndarray<real_t, nb::c_contig> radii,
                                                       real_t L, DView& cenHold, DView& radHold) {
  const int M = (int)radii.shape(0);
  auto cflat = flatten3(centers);
  if ((int)(cflat.size() / 3) != M)
    throw std::runtime_error("sphere_centers (M,3) and sphere_radii (M,) must agree on M");
  cenHold = DView("sph.cen", 3 * M);
  radHold = DView("sph.rad", M);
  Kokkos::deep_copy(cenHold, Kokkos::View<const real_t*, Kokkos::HostSpace>(cflat.data(), 3 * M));
  Kokkos::deep_copy(radHold, Kokkos::View<const real_t*, Kokkos::HostSpace>(radii.data(), M));
  return peclet::voro::SdfSpheres<real_t>{cenHold, radHold, M, L};
}
// The same union-of-balls SDF with HOST views, for the host-serial pore-cell oracle (its clip runs
// on the host, so on a CUDA build the device-view variant would abort on the first eval).
using HView = Kokkos::View<real_t*, Kokkos::HostSpace>;
using HostSpheresSdf = peclet::voro::SdfSpheres<real_t, Kokkos::HostSpace>;
inline HostSpheresSdf makeSpheresSdfHost(nb::ndarray<real_t, nb::c_contig> centers,
                                         nb::ndarray<real_t, nb::c_contig> radii, real_t L,
                                         HView& cenHold, HView& radHold) {
  const int M = (int)radii.shape(0);
  auto cflat = flatten3(centers);
  if ((int)(cflat.size() / 3) != M)
    throw std::runtime_error("sphere_centers (M,3) and sphere_radii (M,) must agree on M");
  cenHold = HView("sph.cen.host", 3 * M);
  radHold = HView("sph.rad.host", M);
  std::copy(cflat.begin(), cflat.end(), cenHold.data());
  std::copy(radii.data(), radii.data() + M, radHold.data());
  return HostSpheresSdf{cenHold, radHold, M, L};
}

// The host-serial reconstruction of the SDF-clipped interstitial Voronoi cell of any seed — the
// TEST ORACLE of the device path (pore_cells.hpp; python/test_voro.py test_pore_cells). Builds a
// periodic counting-sort grid once; each build() walks Chebyshev shells of bins outward, after
// each shell rebuilding the ConvexCell (far box, min-image neighbours closest-first) from
// everything gathered so far, until the shell radius certifies the cell: every seed within R·hbin
// has been gathered and a seed cuts only if it is closer than twice the cell's reach, so
// (R·hbin)² ≥ 4·rSqMax closes it (the same bisector certificate the tessellator's worklist
// uses). If the walk reaches the min-image half box uncertified, every seed is gathered. Then the
// SDF clip. (Its predecessor gathered a fixed 80 nearest seeds — the device port's gate showed
// that truncation missing planes on 8 of 805 cells, up to 2.5e-3 in volume.)
struct PoreReconstructor {
  const real_t* seed;
  int N;
  real_t L, Lh, big, hbin;
  int nb;
  HostSpheresSdf sdf;
  std::vector<int> binStart, binItem;
  mutable std::vector<std::pair<real_t, int>> ord;
  mutable std::vector<real_t> rx, ry, rz;
  mutable std::vector<int> ids;

  int binOf(real_t x) const {
    int b = (int)std::floor(x / hbin) % nb;
    return b < 0 ? b + nb : b;
  }
  int cellOf(int i) const {
    return binOf(seed[3 * i]) + nb * (binOf(seed[3 * i + 1]) + nb * binOf(seed[3 * i + 2]));
  }
  PoreReconstructor(const std::vector<real_t>& s, real_t L_, HostSpheresSdf sdf_)
      : seed(s.data()), N((int)(s.size() / 3)), L(L_), Lh(0.5 * L_), big(4 * L_), sdf(sdf_) {
    nb = std::max(1, std::min((int)std::cbrt((double)N / 2.0 + 1.0), defaults::kPoreMaxBins));
    hbin = L / nb;
    const int nbin = nb * nb * nb;
    binStart.assign(nbin + 1, 0);
    for (int i = 0; i < N; ++i)
      ++binStart[cellOf(i) + 1];
    for (int b = 0; b < nbin; ++b)
      binStart[b + 1] += binStart[b];
    binItem.resize(N);
    std::vector<int> cur(binStart.begin(), binStart.end());
    for (int i = 0; i < N; ++i)
      binItem[cur[cellOf(i)]++] = i;
  }
  // Gather bin (gx,gy,gz) (raw, wrapped here) into `ord` as (dist², j), min-image, j != i.
  void gatherBin(int i, real_t sx, real_t sy, real_t sz, int gx, int gy, int gz) const {
    gx = ((gx % nb) + nb) % nb;
    gy = ((gy % nb) + nb) % nb;
    gz = ((gz % nb) + nb) % nb;
    const int b = gx + nb * (gy + nb * gz);
    for (int t = binStart[b]; t < binStart[b + 1]; ++t) {
      const int j = binItem[t];
      if (j == i)
        continue;
      real_t dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
      dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
      dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
      dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
      ord.emplace_back(dx * dx + dy * dy + dz * dz, j);
    }
  }
  // Rebuild `c` from everything in `ord`, closest-first against the far box.
  void rebuild(real_t sx, real_t sy, real_t sz, PoreCell& c) const {
    std::sort(ord.begin(), ord.end());
    const int M = (int)ord.size();
    rx.resize(M);
    ry.resize(M);
    rz.resize(M);
    ids.resize(M);
    for (int k = 0; k < M; ++k) {
      const int j = ord[k].second;
      real_t dx = seed[3 * j] - sx, dy = seed[3 * j + 1] - sy, dz = seed[3 * j + 2] - sz;
      dx -= dx > Lh ? L : (dx < -Lh ? -L : 0);
      dy -= dy > Lh ? L : (dy < -Lh ? -L : 0);
      dz -= dz > Lh ? L : (dz < -Lh ? -L : 0);
      rx[k] = dx;
      ry[k] = dy;
      rz[k] = dz;
      ids[k] = j;
    }
    const real_t Lbig[3] = {big, big, big};
    peclet::voro::buildConvexCell(c, Lbig, rx.data(), ry.data(), rz.data(), ids.data(), M);
  }
  bool build(int i, PoreCell& c) const {
    const real_t sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
    ord.clear();
    const int bx = binOf(sx), by = binOf(sy), bz = binOf(sz);
    const int Rmax = (nb - 1) / 2;  // beyond it a bin would be visited twice (periodic wrap)
    for (int R = 0;; ++R) {
      if (R > Rmax) {  // uncertified at the min-image half box: take every seed
        ord.clear();
        for (int gz = 0; gz < nb; ++gz)
          for (int gy = 0; gy < nb; ++gy)
            for (int gx = 0; gx < nb; ++gx)
              gatherBin(i, sx, sy, sz, gx, gy, gz);
        rebuild(sx, sy, sz, c);
        break;
      }
      for (int dz2 = -R; dz2 <= R; ++dz2)
        for (int dy2 = -R; dy2 <= R; ++dy2)
          for (int dx2 = -R; dx2 <= R; ++dx2) {
            int cheb = std::abs(dx2);
            cheb = std::max(cheb, std::abs(dy2));
            cheb = std::max(cheb, std::abs(dz2));
            if (cheb != R)
              continue;
            gatherBin(i, sx, sy, sz, bx + dx2, by + dy2, bz + dz2);
          }
      rebuild(sx, sy, sz, c);
      if (c.overflow)
        return false;
      const real_t rh = (real_t)R * hbin;  // every seed within rh is in `ord`
      if (rh * rh >= real_t(4) * c.maxVertexRsq())
        break;
    }
    const real_t seedW[3] = {sx, sy, sz};
    peclet::voro::clipCellAgainstSdf<real_t, defaults::kPoreMaxPlanes, defaults::kPoreMaxTriangles,
                                     false>(c, seedW, sdf);
    return !(c.empty() || c.overflow);
  }
};

void bindPore(nb::module_& m) {
  using namespace defaults;
  // ---- pore-space (SDF-walled) family: bound here, exposed by peclet.voro.pore_mesh ------------
  m.def(
      "optimize_pore_mesh",
      [](nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> vref_in,
         nb::ndarray<real_t, nb::c_contig> sph_c, nb::ndarray<real_t, nb::c_contig> sph_r,
         std::array<real_t, 3> extent, int search_window, int max_iter, real_t tol, int cg_iters,
         const std::string& method, real_t mu_barrier, bool free_energy) {
        auto pos = flatten3(pos_in);
        auto vref = flatten1(vref_in);
        const int N = (int)vref.size();
        if ((int)(pos.size() / 3) != N)
          throw std::runtime_error(
              "optimize_pore_mesh: positions (N,3) and target_volumes (N,) must agree on N");
        const real_t L = cubicExtent(extent, "optimize_pore_mesh");
        const real_t Larr[3] = {L, L, L};
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        const auto prec = parseMethod(method, "optimize_pore_mesh");
        std::vector<real_t> noW;
        auto R = peclet::voro::meshVolumeOptimize<real_t, false, peclet::voro::SdfSpheres<real_t>>(
            pos, noW, vref, Larr, N, search_window, sdf, max_iter, tol, cg_iters, prec, false,
            mu_barrier, (real_t)kBarrierDecay, free_energy);
        return makeOptimizeResult(std::move(pos), std::nullopt, R);
      },
      nb::arg("positions"), nb::arg("target_volumes"), nb::arg("sphere_centers"),
      nb::arg("sphere_radii"), nb::arg("extent"), nb::kw_only(),
      nb::arg("search_window") = kPoreSearchWindow, nb::arg("max_iter") = kPoreMaxIter,
      nb::arg("tol") = kOptimizerTolerance, nb::arg("cg_iters") = kPoreCgIters,
      nb::arg("method") = "graphamg", nb::arg("mu_barrier") = 0.0, nb::arg("free_energy") = false,
      doc("Relax interstitial seeds (N,3) so their SDF-clipped Voronoi cell volumes approach the "
          "per-cell\ntarget_volumes (N,), with the sphere packing (sphere_centers (M,3), "
          "sphere_radii (M,)) as\nperiodic walls in the cubic box `extent` (Lx == Ly == Lz). "
          "method: one of " +
          std::string(kMethodList) +
          " (default\n'graphamg'; 'steepest' is plain descent). free_energy=True uses "
          "E = -sum V_ref log V (pressure\nV_ref/V, resists collapse); mu_barrier > 0 adds a "
          "log-barrier that decays by " +
          fmt(kBarrierDecay) +
          " per iteration.\nsearch_window / max_iter / tol / cg_iters default to " +
          fmt(kPoreSearchWindow) + " / " + fmt(kPoreMaxIter) + " / " + fmt(kOptimizerTolerance) +
          " / " + fmt(kPoreCgIters) +
          ". Returns an\nOptimizeResult. Experimental (pore-space meshing; see the "
          "pore-mesh-voronoi example)."));

  // The pore-space export: the device path (pore_cells.hpp — the tessellator's own gather at the
  // pore capacity, count -> scan -> fill in seed order) under the public names, and the host-serial
  // PoreReconstructor under the `_host` names as its test oracle (python/test_voro.py
  // test_pore_cells compares them). Both share the per-cell export rule + layout
  // (peclet::voro::detail::poreCellCount / poreCellFill).
  auto poreCellsDict = [](std::vector<real_t>&& pts, std::vector<int64_t>&& faces,
                          std::vector<int64_t>&& faceOff, std::vector<real_t>&& vol,
                          std::vector<int32_t>&& boundary, std::vector<int32_t>&& cellSeed,
                          long numOverflow, long numIncomplete) {
    const std::size_t nPts = pts.size() / 3, nCells = vol.size();
    nb::dict d;
    d["points"] = peclet::core::python::vector_to_ndarray(std::move(pts), {nPts, 3}, {3, 1});
    d["faces"] = peclet::core::python::vector_to_ndarray(std::move(faces), {faces.size()}, {1});
    d["face_offsets"] =
        peclet::core::python::vector_to_ndarray(std::move(faceOff), {faceOff.size()}, {1});
    d["volume"] = peclet::core::python::vector_to_ndarray(std::move(vol), {nCells}, {1});
    d["boundary"] = peclet::core::python::vector_to_ndarray(std::move(boundary), {nCells}, {1});
    d["seed"] = peclet::core::python::vector_to_ndarray(std::move(cellSeed), {nCells}, {1});
    d["num_overflow"] = numOverflow;
    d["num_incomplete"] = numIncomplete;
    return d;
  };
  auto poreSectionDict = [](std::vector<real_t>&& verts, std::vector<int64_t>&& off,
                            std::vector<real_t>&& vol, std::vector<int32_t>&& cellSeed,
                            long numOverflow, long numIncomplete) {
    const std::size_t nV = verts.size() / 3, nP = vol.size();
    nb::dict d;
    d["verts"] = peclet::core::python::vector_to_ndarray(std::move(verts), {nV, 3}, {3, 1});
    d["offsets"] = peclet::core::python::vector_to_ndarray(std::move(off), {off.size()}, {1});
    d["volume"] = peclet::core::python::vector_to_ndarray(std::move(vol), {nP}, {1});
    d["seed"] = peclet::core::python::vector_to_ndarray(std::move(cellSeed), {nP}, {1});
    d["num_overflow"] = numOverflow;
    d["num_incomplete"] = numIncomplete;
    return d;
  };
  m.def(
      "sdf_voronoi_cells",
      [poreCellsDict](nb::ndarray<real_t, nb::c_contig> pos_in,
                      nb::ndarray<real_t, nb::c_contig> sph_c,
                      nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent,
                      int search_window) {
        auto seed = flatten3(pos_in);
        const int N = (int)(seed.size() / 3);
        const real_t L = cubicExtent(extent, "sdf_voronoi_cells");
        if (N <= 0)
          throw std::invalid_argument(
              "voro: sdf_voronoi_cells(positions) needs at least one seed.");
        if (search_window < 1)
          throw std::invalid_argument("voro: sdf_voronoi_cells(search_window=...) needs >= 1.");
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        auto pos = peclet::core::toDevice<real_t>(seed, "pore.pos");
        const real_t Larr[3] = {L, L, L};
        auto r = peclet::voro::buildPoreCells<real_t, peclet::voro::SdfSpheres<real_t>>(
            pos, N, Larr, sdf, search_window);
        using peclet::voro::detail::toHostVecT;
        return poreCellsDict(toHostVecT<real_t>(r.points), toHostVecT<int64_t>(r.faces),
                             toHostVecT<int64_t>(r.faceOffset), toHostVecT<real_t>(r.volume),
                             toHostVecT<int32_t>(r.boundary), toHostVecT<int32_t>(r.seed),
                             r.numOverflow, r.numIncomplete);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      nb::kw_only(), nb::arg("search_window") = kPoreSearchWindow,
      doc("Reconstruct the SDF-clipped interstitial Voronoi cells (cubic periodic box `extent`, "
          "the\nspheres as walls) on the device — the tessellator's own gather, one thread per "
          "cell — and\nreturn their polyhedra as flat arrays (VTK_POLYHEDRON layout) in seed "
          "order: 'points' (Np,3),\n'faces' + 'face_offsets' (per-cell face lists, global point "
          "ids, each face CCW about its\noutward normal), 'volume' (Nc,), 'boundary' (Nc, 1 "
          "where the cell touches a sphere wall),\n'seed' (Nc,). Seeds inside a sphere have no "
          "cell; 'num_overflow' counts cells skipped for\nexceeding the cell capacity ("
          "peclet.voro.defaults max_planes / max_triangles) and 'num_incomplete' the cells "
          "whose\ngather window "
          "(search_window grid blocks per axis, default " +
          fmt(kPoreSearchWindow) + ") did not close — raise\nsearch_window if it is not 0."));
  m.def(
      "sdf_voronoi_section",
      [poreSectionDict](
          nb::ndarray<real_t, nb::c_contig> pos_in, nb::ndarray<real_t, nb::c_contig> sph_c,
          nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent,
          std::array<real_t, 3> point, std::array<real_t, 3> normal, int search_window) {
        auto seed = flatten3(pos_in);
        const int N = (int)(seed.size() / 3);
        const real_t L = cubicExtent(extent, "sdf_voronoi_section");
        if (N <= 0)
          throw std::invalid_argument(
              "voro: sdf_voronoi_section(positions) needs at least one seed.");
        if (search_window < 1)
          throw std::invalid_argument("voro: sdf_voronoi_section(search_window=...) needs >= 1.");
        DView cenH, radH;
        auto sdf = makeSpheresSdf(sph_c, sph_r, L, cenH, radH);
        auto pos = peclet::core::toDevice<real_t>(seed, "pore.pos");
        const real_t Larr[3] = {L, L, L};
        auto r = peclet::voro::buildPoreSection<real_t, peclet::voro::SdfSpheres<real_t>>(
            pos, N, Larr, sdf, point.data(), normal.data(), search_window);
        using peclet::voro::detail::toHostVecT;
        return poreSectionDict(toHostVecT<real_t>(r.verts), toHostVecT<int64_t>(r.offset),
                               toHostVecT<real_t>(r.volume), toHostVecT<int32_t>(r.seed),
                               r.numOverflow, r.numIncomplete);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      nb::arg("point"), nb::arg("normal"), nb::kw_only(),
      nb::arg("search_window") = kPoreSearchWindow,
      doc("Cross-section of the SDF-clipped interstitial Voronoi mesh (cubic periodic box "
          "`extent`) by\nthe plane through `point` with `normal`, on the device: every cell cut "
          "directly\n(ConvexCell::sectionPolygon, from the dual edges, so it tiles the plane "
          "exactly where a\nface-by-face slice drops facets). Returns 'verts' (Nv,3, world "
          "coords, all on the plane, CCW\nabout the normal) + 'offsets' (Npoly+1, per-polygon "
          "vertex ranges) + 'volume' (Npoly, the 3-D\ncell volume) + 'seed' (Npoly, the seed "
          "index), in seed order, plus 'num_overflow' /\n'num_incomplete' as in "
          "sdf_voronoi_cells (search_window default " +
          fmt(kPoreSearchWindow) +
          "). For a z=z0 slice pass\npoint=(0,0,z0), normal=(0,0,1) "
          "and plot verts[:, :2]."));
  m.def(
      "_sdf_voronoi_cells_host",
      [poreCellsDict](nb::ndarray<real_t, nb::c_contig> pos_in,
                      nb::ndarray<real_t, nb::c_contig> sph_c,
                      nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent) {
        auto seed = flatten3(pos_in);
        const int N = (int)(seed.size() / 3);
        const real_t L = cubicExtent(extent, "_sdf_voronoi_cells_host");
        HView cenH, radH;
        auto sdf = makeSpheresSdfHost(sph_c, sph_r, L, cenH, radH);
        std::vector<real_t> pts, vol;
        std::vector<int64_t> faces, faceOff(1, 0);
        std::vector<int32_t> boundary, cellSeed;
        PoreReconstructor rec(seed, L, sdf);
        long numOverflow = 0;
        for (int i = 0; i < N; ++i) {
          PoreCell c;
          if (!rec.build(i, c)) {
            numOverflow += c.overflow ? 1 : 0;
            continue;
          }
          int np = 0, ne = 0;
          bool wall = false;
          if (!peclet::voro::detail::poreCellCount(c, np, ne, wall))
            continue;
          const int64_t ptBase = (int64_t)(pts.size() / 3), fBase = (int64_t)faces.size();
          pts.resize(pts.size() + 3 * (std::size_t)np);
          faces.resize(faces.size() + (std::size_t)ne);
          const real_t s[3] = {seed[3 * i], seed[3 * i + 1], seed[3 * i + 2]};
          peclet::voro::detail::poreCellFill(c, s, ptBase, fBase, pts.data(), faces.data());
          faceOff.push_back((int64_t)faces.size());
          vol.push_back(c.volumePerVertex());
          boundary.push_back(wall ? 1 : 0);
          cellSeed.push_back(i);
        }
        return poreCellsDict(std::move(pts), std::move(faces), std::move(faceOff), std::move(vol),
                             std::move(boundary), std::move(cellSeed), numOverflow, 0);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      "The host-serial reconstruction of sdf_voronoi_cells (PoreReconstructor: the 80 nearest "
      "seeds\nby a Chebyshev shell walk, closest-first clip against a far box, then the SDF "
      "clip), kept as\nthe TEST ORACLE of the device path; same result layout. Not part of the "
      "API.");
  m.def(
      "_sdf_voronoi_section_host",
      [poreSectionDict](nb::ndarray<real_t, nb::c_contig> pos_in,
                        nb::ndarray<real_t, nb::c_contig> sph_c,
                        nb::ndarray<real_t, nb::c_contig> sph_r, std::array<real_t, 3> extent,
                        std::array<real_t, 3> point, std::array<real_t, 3> normal) {
        auto seed = flatten3(pos_in);
        const real_t L = cubicExtent(extent, "_sdf_voronoi_section_host");
        HView cenH, radH;
        auto sdf = makeSpheresSdfHost(sph_c, sph_r, L, cenH, radH);
        PoreReconstructor rec(seed, L, sdf);
        std::vector<real_t> verts, vol;
        std::vector<int64_t> off(1, 0);
        std::vector<int32_t> cellSeed;
        real_t spx[PoreCell::MAXSV], spy[PoreCell::MAXSV], spz[PoreCell::MAXSV];
        long numOverflow = 0;
        for (int i = 0; i < rec.N; ++i) {
          const real_t sx = seed[3 * i], sy = seed[3 * i + 1], sz = seed[3 * i + 2];
          PoreCell c;
          if (!rec.build(i, c)) {
            numOverflow += c.overflow ? 1 : 0;
            continue;
          }
          const real_t p0[3] = {point[0] - sx, point[1] - sy, point[2] - sz};  // plane, cell frame
          const real_t u3[3] = {normal[0], normal[1], normal[2]};
          const int mm = c.sectionPolygon(p0, u3, spx, spy, spz);
          if (mm < 3)
            continue;
          for (int k = 0; k < mm; ++k) {
            verts.push_back(sx + spx[k]);
            verts.push_back(sy + spy[k]);
            verts.push_back(sz + spz[k]);
          }
          off.push_back((int64_t)(verts.size() / 3));
          vol.push_back(c.volumePerVertex());
          cellSeed.push_back(i);
        }
        return poreSectionDict(std::move(verts), std::move(off), std::move(vol),
                               std::move(cellSeed), numOverflow, 0);
      },
      nb::arg("positions"), nb::arg("sphere_centers"), nb::arg("sphere_radii"), nb::arg("extent"),
      nb::arg("point"), nb::arg("normal"),
      "The host-serial reconstruction of sdf_voronoi_section, kept as the TEST ORACLE of the "
      "device\npath; same result layout. Not part of the API.");
}
}  // namespace peclet::voro::pybind
