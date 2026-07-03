#!/usr/bin/env python3
"""
Step 3: a z = z0 cross-section of the interstitial Voronoi grid.

Reads cells.vtu (the SDF-walled Voronoi from sdf_voronoi_vtu) and packing.npz, slices the unstructured
grid with a plane using VTK, and draws the cell cross-sections (coloured by cell volume) with the
solid spheres' cross-section discs overlaid — the interstitial tessellation walled by the packing.

    python3 plot_cross_section.py            # z0 = 0, writes cross_section.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
import vtk
from vtk.util.numpy_support import vtk_to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))


def slice_cells(vtu_path, z0):
    """Cross-section each convex Voronoi cell at z=z0, computed from its faces (robust to VTK's
    polyhedron watertightness checks): collect where cell face-edges cross the plane, then order the
    crossings into a convex polygon. Returns per-cell (polygon xy, volume)."""
    vtk.vtkObject.GlobalWarningDisplayOff()
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    g = reader.GetOutput()
    vol = vtk_to_numpy(g.GetCellData().GetArray("volume"))

    polys, vals = [], []
    for ci in range(g.GetNumberOfCells()):
        cell = g.GetCell(ci)                 # vtkPolyhedron for type 42
        nf = cell.GetNumberOfFaces()
        pts = []
        for fj in range(nf):
            face = cell.GetFace(fj)
            npf = face.GetNumberOfPoints()
            P = np.array([face.GetPoints().GetPoint(k) for k in range(npf)])
            for k in range(npf):             # edges of the face polygon
                a, b = P[k], P[(k + 1) % npf]
                da, db = a[2] - z0, b[2] - z0
                if (da > 0) != (db > 0):     # edge crosses the plane
                    t = da / (da - db)
                    pts.append(a[:2] + t * (b[:2] - a[:2]))
        if len(pts) < 3:
            continue
        pts = np.array(pts)
        c = pts.mean(0)                       # order CCW around the centroid (cell is convex)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        polys.append(pts)
        vals.append(vol[ci])
    return polys, np.asarray(vals)


def main():
    z0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    data = np.load(os.path.join(HERE, "packing.npz"))
    centres, radii, half = data["centres"], data["radii"], float(data["half"])
    polys, vals = slice_cells(os.path.join(HERE, "cells.vtu"), z0)
    print(f"z0={z0}: {len(polys)} cell cross-sections")

    fig, ax = plt.subplots(figsize=(8, 8))
    # Voronoi cell cross-sections, coloured by 3D cell volume.
    pc = PolyCollection(polys, array=vals, cmap="viridis", edgecolors="k", linewidths=0.3, alpha=0.9)
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Voronoi cell volume")

    # solid spheres intersecting the plane -> grey discs.
    for c, r in zip(centres, radii):
        dz = z0 - c[2]
        if abs(dz) < r:
            ax.add_patch(Circle((c[0], c[1]), np.sqrt(r * r - dz * dz),
                                 facecolor="0.6", edgecolor="0.3", lw=0.6, zorder=3))

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_title(f"Interstitial Voronoi of a sphere packing (SDF walls), cross-section z = {z0:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    out = os.path.join(HERE, "cross_section.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
