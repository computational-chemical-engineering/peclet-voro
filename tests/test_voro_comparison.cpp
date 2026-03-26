/**
 * @file test_voro_comparison.cpp
 * \brief Tests that compare voronoi_dynamics against voro++ for a periodic box.
 *
 * Two test scenarios are covered:
 *   1. Static tessellation: build a Voronoi diagram for a fixed random point
 *      set and verify that per-particle volumes agree between the two libraries.
 *   2. Moving points: build an initial tessellation, displace the particles,
 *      rebuild with voronoi_dynamics (via CellComplex::build) and compare the
 *      resulting volumes against a fresh voro++ tessellation of the same
 *      final positions.
 *
 * The voro++ library must be pre-built in the voro/src/ subfolder so that
 * voro/src/libvoro++.a is available at link time.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include <voronoi_dynamics/voronoi.hpp>
#include <boost/random.hpp>

// voro++ headers
#include "voro/src/voro++.hh"

using std::vector;
using vor::Array;
using vor::uint2;
using vor::Box;
using vor::CellComplex;
using vor::CellGeometry;

// ---------------------------------------------------------------------------
// Helper: build a voro++ periodic tessellation and return per-particle volumes
// sorted by particle id.
// ---------------------------------------------------------------------------
static vector<double> voropp_volumes(const vector<Array<double, 3> > &pos,
                                     double Lx, double Ly, double Lz)
{
    const int n = static_cast<int>(pos.size());
    // Choose grid dimensions: roughly n^(1/3) blocks per side
    int nblk = std::max(1, static_cast<int>(std::cbrt(static_cast<double>(n) / 8.0)));
    int nx = nblk, ny = nblk, nz = nblk;

    // Orthogonal periodic box: bxy=bxz=byz=0
    voro::container_periodic con(Lx, 0, Ly, 0, 0, Lz, nx, ny, nz, 8);
    for (int i = 0; i < n; ++i)
        con.put(i, pos[i][0], pos[i][1], pos[i][2]);

    vector<double> vols(n, 0.0);
    voro::voronoicell_neighbor c(con);
    voro::c_loop_all_periodic vl(con);
    if (vl.start()) do {
        if (con.compute_cell(c, vl))
            vols[vl.pid()] = c.volume();
    } while (vl.inc());

    return vols;
}

// ---------------------------------------------------------------------------
// Helper: build a voronoi_dynamics tessellation and return per-particle volumes.
// ---------------------------------------------------------------------------
static vector<double> vordyn_volumes(const vector<Array<double, 3> > &pos,
                                     double Lx, double Ly, double Lz)
{
    typedef double real_t;
    const int n = static_cast<int>(pos.size());

    Array<real_t, 3> L;
    L[0] = Lx; L[1] = Ly; L[2] = Lz;
    vor::Box<real_t> box(L);
    CellComplex<real_t> cx(&box);
    cx.build(pos);

    vector<CellGeometry<real_t> > &geoms = cx.getGeoms();
    vector<double> vols(n);
    for (int i = 0; i < n; ++i) {
        geoms[i].computeVolume();
        vols[i] = geoms[i].getVolume();
    }
    return vols;
}

// ---------------------------------------------------------------------------
// Test 1: static tessellation comparison
// ---------------------------------------------------------------------------
static int testStaticComparison(int n, double Lx, double Ly, double Lz,
                                unsigned int seed, double tol)
{
    typedef double real_t;
    typedef boost::mt19937 rng_type;
    typedef boost::uniform_01<real_t> dist_type;
    typedef boost::variate_generator<rng_type &, dist_type> gen_type;

    rng_type rng(seed);
    gen_type gen(rng, dist_type());

    vector<Array<real_t, 3> > pos(n);
    for (int i = 0; i < n; ++i) {
        pos[i][0] = Lx * gen();
        pos[i][1] = Ly * gen();
        pos[i][2] = Lz * gen();
    }

    vector<double> vd = vordyn_volumes(pos, Lx, Ly, Lz);
    vector<double> vp = voropp_volumes(pos, Lx, Ly, Lz);

    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    for (int i = 0; i < n; ++i) {
        double ae = std::fabs(vd[i] - vp[i]);
        double re = (vp[i] > 0.0) ? ae / vp[i] : ae;
        if (ae > maxAbsErr) maxAbsErr = ae;
        if (re > maxRelErr) maxRelErr = re;
    }

    printf("  n=%d  max|Δvol|=%.2e  max|Δvol|/vol=%.2e\n", n, maxAbsErr, maxRelErr);

    if (maxRelErr > tol) {
        fprintf(stderr,
                "FAIL: max relative volume error %.2e exceeds tolerance %.2e\n",
                maxRelErr, tol);
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Test 2: moving-points comparison
// Displace particles by small random amounts, rebuild with voronoi_dynamics
// and compare against a fresh voro++ tessellation of the final positions.
// ---------------------------------------------------------------------------
static int testMovingComparison(int n, double Lx, double Ly, double Lz,
                                unsigned int seed, double tol)
{
    typedef double real_t;
    typedef boost::mt19937 rng_type;
    typedef boost::uniform_01<real_t> dist_type;
    typedef boost::variate_generator<rng_type &, dist_type> gen_type;

    rng_type rng(seed);
    gen_type gen(rng, dist_type());

    // Initial positions
    vector<Array<real_t, 3> > pos(n);
    for (int i = 0; i < n; ++i) {
        pos[i][0] = Lx * gen();
        pos[i][1] = Ly * gen();
        pos[i][2] = Lz * gen();
    }

    Array<real_t, 3> L;
    L[0] = Lx; L[1] = Ly; L[2] = Lz;
    vor::Box<real_t> box(L);
    CellComplex<real_t> cx(&box);

    // Initial build
    cx.build(pos);

    // Move particles: small fraction of mean inter-particle spacing
    double dx = 0.05 * std::cbrt(Lx * Ly * Lz / static_cast<double>(n));
    rng_type rng2(seed + 1000);
    gen_type gen2(rng2, dist_type());
    for (int i = 0; i < n; ++i) {
        pos[i][0] += dx * (gen2() - 0.5);
        pos[i][1] += dx * (gen2() - 0.5);
        pos[i][2] += dx * (gen2() - 0.5);
    }
    // Wrap back into periodic box
    box.putInBox(pos);

    // Rebuild with voronoi_dynamics
    cx.build(pos);

    vector<CellGeometry<real_t> > &geoms = cx.getGeoms();
    vector<double> vd(n);
    for (int i = 0; i < n; ++i) {
        geoms[i].computeVolume();
        vd[i] = geoms[i].getVolume();
    }

    // Compare against fresh voro++ tessellation of final positions
    vector<double> vp = voropp_volumes(pos, Lx, Ly, Lz);

    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    for (int i = 0; i < n; ++i) {
        double ae = std::fabs(vd[i] - vp[i]);
        double re = (vp[i] > 0.0) ? ae / vp[i] : ae;
        if (ae > maxAbsErr) maxAbsErr = ae;
        if (re > maxRelErr) maxRelErr = re;
    }

    printf("  n=%d  max|Δvol|=%.2e  max|Δvol|/vol=%.2e\n", n, maxAbsErr, maxRelErr);

    if (maxRelErr > tol) {
        fprintf(stderr,
                "FAIL: max relative volume error after move %.2e exceeds tolerance %.2e\n",
                maxRelErr, tol);
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    int failures = 0;

    // ------------------------------------------------------------------
    // Static tessellation tests
    // ------------------------------------------------------------------
    printf("=== Test group 1: static tessellation (voronoi_dynamics vs voro++) ===\n");

    printf("Test 1a: 200 particles, unit cube\n");
    failures += testStaticComparison(200, 1.0, 1.0, 1.0, 42, 1e-8);

    printf("Test 1b: 500 particles, unit cube\n");
    failures += testStaticComparison(500, 1.0, 1.0, 1.0, 123, 1e-8);

    printf("Test 1c: 1000 particles, non-cubic box (2x1x0.5)\n");
    // The 2x1x0.5 box has a thin z-dimension; particles near the z-boundary
    // interact with many periodic images and small floating-point differences
    // between the two implementations can reach ~1e-6 in relative volume.
    failures += testStaticComparison(1000, 2.0, 1.0, 0.5, 456, 1e-5);

    printf("Test 1d: 2000 particles, unit cube\n");
    failures += testStaticComparison(2000, 1.0, 1.0, 1.0, 789, 1e-8);

    // ------------------------------------------------------------------
    // Moving-points tests
    // ------------------------------------------------------------------
    printf("\n=== Test group 2: moving points (voronoi_dynamics rebuild vs voro++) ===\n");

    printf("Test 2a: 200 particles moved, unit cube\n");
    failures += testMovingComparison(200, 1.0, 1.0, 1.0, 42, 1e-8);

    printf("Test 2b: 500 particles moved, unit cube\n");
    failures += testMovingComparison(500, 1.0, 1.0, 1.0, 123, 1e-8);

    printf("Test 2c: 1000 particles moved, non-cubic box (2x1x0.5)\n");
    failures += testMovingComparison(1000, 2.0, 1.0, 0.5, 456, 1e-8);

    printf("Test 2d: 2000 particles moved, unit cube\n");
    failures += testMovingComparison(2000, 1.0, 1.0, 1.0, 789, 1e-8);

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    if (failures == 0) {
        printf("\nAll tests PASSED\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
