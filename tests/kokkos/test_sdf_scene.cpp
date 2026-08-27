// Layer 0: proves voro can consume the shared core geometry through SdfScene: the full analytic
// vocabulary
// + CSG + grids, evaluated on device, matching a direct core evalTree call and the legacy
// single-shape providers where they overlap.
#include <cstdio>
#include <Kokkos_Core.hpp>

#include "peclet/voro/sdf.hpp"

using namespace peclet::voro;
using namespace peclet::core;
using namespace peclet::core::geom;
using Real = double;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int bad = 0;
  {
    // scene: union( sphere , difference( box , torus ) ) -- depth 3, mixed vocabulary
    Kokkos::View<ShapeNode<Real>*> nodes("nodes", 6);
    auto h = Kokkos::create_mirror_view(nodes);
    h(0).kind = kUnion;
    h(0).aux0 = 1;
    h(0).aux1 = 2;
    h(1).kind = kSphere;
    h(1).params[0] = 0.6;
    h(1).transform.translation = Vec3<Real>{0.8, 0.0, 0.0};
    h(2).kind = kDifference;
    h(2).aux0 = 3;
    h(2).aux1 = 4;
    h(3).kind = kBox;
    h(3).params[0] = 0.7;
    h(3).params[1] = 0.5;
    h(3).params[2] = 0.5;
    h(4).kind = kTorus;
    h(4).params[0] = 0.5;
    h(4).params[1] = 0.18;
    h(5).kind = kSuperquadric;
    h(5).params[0] = 0.6;
    h(5).params[1] = 0.4;
    h(5).params[2] = 0.3;
    h(5).params[3] = 4.0;
    Kokkos::deep_copy(nodes, h);
    Kokkos::View<GridDesc<Real>*> grids("grids", 1);
    Kokkos::View<float*> pool("pool", 1);

    Kokkos::View<Real*> out("out", 6);
    Kokkos::parallel_for(
        "scene", 1, KOKKOS_LAMBDA(int) {
          SdfScene<Real> sc{nodes, grids, pool, 6, 0, 1e-6};
          out(0) = sc.eval(0.3, 0.2, 0.1);
          out(1) = sc.eval(1.2, 0.0, 0.0);
          Real g[3];
          sdfGradient<Real>(sc, 1.2, 0.0, 0.0, g);  // voro's own gradient helper over the scene
          out(2) = Kokkos::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
          // a superquadric root: reachable in voro only via the shared scene
          SdfScene<Real> sq{nodes, grids, pool, 6, 5, 1e-6};
          out(3) = sq.eval(0.0, 0.0, 0.0);
          out(4) = sq.eval(2.0, 0.0, 0.0);
          // legacy provider vs the same shape expressed as a scene node
          SdfSphere<Real> legacy{0.8, 0.0, 0.0, 0.6};
          SdfScene<Real> asNode{nodes, grids, pool, 6, 1, 1e-6};
          out(5) = Kokkos::fabs(legacy.eval(0.31, -0.22, 0.13) - asNode.eval(0.31, -0.22, 0.13));
        });
    Kokkos::fence();
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    std::printf("  scene(0.3,0.2,0.1)      = %.9g\n", ho(0));
    std::printf("  scene(1.2,0,0)          = %.9g   (inside the offset sphere -> negative)\n",
                ho(1));
    std::printf("  |grad| at (1.2,0,0)     = %.9g   (exact leaf -> 1)\n", ho(2));
    std::printf("  superquadric centre     = %.9g   (negative)\n", ho(3));
    std::printf("  superquadric far        = %.9g   (positive)\n", ho(4));
    std::printf("  legacy SdfSphere vs scene node: |diff| = %.3g\n", ho(5));
    if (!(ho(1) < 0)) {
      std::printf("  FAIL: expected inside\n");
      bad = 1;
    }
    if (Kokkos::fabs(ho(2) - 1.0) > 1e-5) {
      std::printf("  FAIL: |grad| != 1\n");
      bad = 1;
    }
    if (!(ho(3) < 0 && ho(4) > 0)) {
      std::printf("  FAIL: superquadric sign\n");
      bad = 1;
    }
    if (ho(5) != 0.0) {
      std::printf("  FAIL: scene node != legacy provider\n");
      bad = 1;
    }
  }
  Kokkos::finalize();
  std::printf(bad ? "VORO-SCENE FAIL\n" : "VORO-SCENE OK\n");
  return bad;
}
