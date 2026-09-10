"""peclet.voro.scenes — small SDF scene helpers for the tessellator's ``set_geometry``.

The tessellator takes a core shape scene in the flat node encoding (``node_ints`` (n,3) int32,
``node_reals`` (n,16) float64, a root index) — what :meth:`peclet.core.geom.SceneBuilder.encode`
returns. These helpers build the two encodings the pore-space examples need without importing
``peclet.core``: the CSG union of solid spheres (a packed bed as walls) and its numpy evaluation.
"""

import numpy as np

__all__ = ["sphere_union_scene", "sphere_union_sdf"]


def sphere_union_scene(centers, radii):
    """Flat scene encoding ``(node_ints (n,3) int32, node_reals (n,16) float64, root)`` of the CSG
    union of solid spheres — pass it to :meth:`Tessellation.set_geometry` as
    ``t.set_geometry(*sphere_union_scene(centers, radii))``. ``centers`` (M,3), ``radii`` (M,)."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    if len(radii) != len(centers):
        raise ValueError(f"sphere_union_scene: {len(centers)} centers but {len(radii)} radii")
    if len(radii) == 0:
        raise ValueError("sphere_union_scene: needs at least one sphere")
    m = len(radii)
    ints, reals = [], []
    for c, r in zip(centers, radii):  # leaves
        ints.append([1, -1, -1])  # kSphere
        row = np.zeros(16)
        row[0] = r
        row[8:11] = c
        row[11:15] = (0.0, 0.0, 0.0, 1.0)
        row[15] = 1.0
        reals.append(row)
    root = 0
    for k in range(1, m):  # union chain: node m+k-1 = union(previous root, leaf k)
        ints.append([32, root, k])  # kUnion
        row = np.zeros(16)
        row[11:15] = (0.0, 0.0, 0.0, 1.0)
        row[15] = 1.0
        reals.append(row)
        root = len(ints) - 1
    return (np.ascontiguousarray(np.array(ints, dtype=np.int32)),
            np.ascontiguousarray(np.array(reals, dtype=np.float64)), root)


def sphere_union_sdf(points, centers, radii, extent):
    """Signed distance of ``points`` (N,3) to the periodic union of spheres (``centers`` (M,3),
    ``radii`` (M,)) in the box ``extent`` (Lx, Ly, Lz): ``min_i(|x - c_i|_minimage - r_i)``, < 0
    inside a sphere, > 0 in the fluid. The numpy twin of the scene from
    :func:`sphere_union_scene`."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    L = np.asarray(extent, dtype=np.float64).reshape(3)
    d = points[:, None, :] - centers[None, :, :]
    d -= L * np.round(d / L)
    return (np.linalg.norm(d, axis=2) - radii[None, :]).min(axis=1)
