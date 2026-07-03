#!/usr/bin/env python3
"""Cross-section visualisation of the four pore-mesh stages (pore_mesh_stages).

Slices each stage's clipped-Voronoi VTU at z=z0, draws the cell cross-sections coloured by cell
volume (log scale, shared) with the solid spheres overlaid, and adds a wall zoom to see exactly what
happens to the cells against a sphere. Also colours the two RELAXED stages by V/V_ref (equalisation
quality). Usage:  python3 plot_stages.py <outdir> [z0]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm, TwoSlopeNorm
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def slice_cells(vtu_path, z0, array="volume"):
    vtk.vtkObject.GlobalWarningDisplayOff()
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    g = reader.GetOutput()
    val = vtk_to_numpy(g.GetCellData().GetArray(array))
    bnd = vtk_to_numpy(g.GetCellData().GetArray("boundary"))
    polys, vals, wall = [], [], []
    for ci in range(g.GetNumberOfCells()):
        cell = g.GetCell(ci)
        pts = []
        for fj in range(cell.GetNumberOfFaces()):
            face = cell.GetFace(fj)
            npf = face.GetNumberOfPoints()
            P = np.array([face.GetPoints().GetPoint(k) for k in range(npf)])
            for k in range(npf):
                a, b = P[k], P[(k + 1) % npf]
                da, db = a[2] - z0, b[2] - z0
                if (da > 0) != (db > 0):
                    t = da / (da - db)
                    pts.append(a[:2] + t * (b[:2] - a[:2]))
        if len(pts) < 3:
            continue
        pts = np.array(pts)
        c = pts.mean(0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        polys.append(pts)
        vals.append(val[ci])
        wall.append(bnd[ci])
    return polys, np.asarray(vals), np.asarray(wall)


def read_spheres(path):
    with open(path) as f:
        M, L = f.readline().split()
        M, L = int(M), float(L)
        c = np.array([[float(x) for x in f.readline().split()] for _ in range(M)])
    return c[:, :3], c[:, 3], L


def draw(ax, polys, vals, norm, cmap, spheres, radii, L, z0, title, window=None, edgelw=0.25):
    pc = PolyCollection(polys, array=np.asarray(vals), cmap=cmap, norm=norm,
                        edgecolors="k", linewidths=edgelw, alpha=0.92)
    ax.add_collection(pc)
    for c, r in zip(spheres, radii):
        for sx in (0, -L, L):  # nearest periodic images in x/y for a clean window
            for sy in (0, -L, L):
                dz = z0 - c[2]
                if abs(dz) < r:
                    ax.add_patch(Circle((c[0] + sx, c[1] + sy), np.sqrt(r * r - dz * dz),
                                        facecolor="0.55", edgecolor="0.25", lw=0.5, zorder=3))
    if window:
        ax.set_xlim(window[0], window[1]); ax.set_ylim(window[2], window[3])
    else:
        ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return pc


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    spheres, radii, L = read_spheres(os.path.join(outdir, "spheres.txt"))
    z0 = float(sys.argv[2]) if len(sys.argv) > 2 else L / 2
    stages = [
        ("stage1_random.vtu", "1 · random seeds (raw Voronoi)"),
        ("stage2_equalized.vtu", "2 · uniform-target relaxation"),
        ("stage3_graded_seed.vtu", "3 · graded seeding  V_ref=sdf³  (raw)"),
        ("stage4_graded_relaxed.vtu", "4 · graded relaxation"),
    ]
    sl = [slice_cells(os.path.join(outdir, f), z0) for f, _ in stages]

    # shared log colour scale over all cell volumes
    allv = np.concatenate([v for _, v, _ in sl])
    vmin, vmax = np.percentile(allv[allv > 0], [3, 97])
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    pc = None
    for ax, (polys, vals, _), (_, title) in zip(axes.ravel(), sl, stages):
        pc = draw(ax, polys, vals, norm, "viridis", spheres, radii, L, z0, title)
    cb = fig.colorbar(pc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Voronoi cell volume (log)")
    fig.suptitle(f"Pore-space Voronoi mesh — 4 stages, cross-section z={z0:.2f}", fontsize=13)
    p = os.path.join(outdir, "stages_volume.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)

    # wall zoom: pick the sphere nearest z0 and window around it; stage 1 vs stage 4
    cross = [(i, np.sqrt(r * r - (z0 - c[2]) ** 2)) for i, (c, r) in enumerate(zip(spheres, radii))
             if abs(z0 - c[2]) < r]
    i0 = max(cross, key=lambda t: t[1])[0]
    cx, cy = spheres[i0, 0], spheres[i0, 1]
    w = 1.6
    win = (cx - w, cx + w, cy - w, cy + w)
    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, k, title in [(ax2[0], 0, "random seeds"), (ax2[1], 3, "graded seeding + relaxation")]:
        polys, vals, _ = sl[k]
        draw(ax, polys, vals, norm, "viridis", spheres, radii, L, z0, title, window=win, edgelw=0.5)
    fig2.suptitle("Zoom at a wall — cells against a sphere (colour = volume)", fontsize=13)
    p = os.path.join(outdir, "stages_wall_zoom.png")
    fig2.savefig(p, dpi=140, bbox_inches="tight"); print("wrote", p)

    # equalisation quality: relaxed stages coloured by V/V_ref (1 = on target)
    fig3, ax3 = plt.subplots(1, 2, figsize=(15, 7.5))
    rnorm = TwoSlopeNorm(vmin=1 / 4, vcenter=1.0, vmax=4.0)
    pc3 = None
    for ax, f, title in [(ax3[0], "stage2_equalized.vtu", "uniform target"),
                         (ax3[1], "stage4_graded_relaxed.vtu", "graded target sdf³")]:
        polys, vals, _ = slice_cells(os.path.join(outdir, f), z0, array="rel")
        pc3 = draw(ax, polys, vals, rnorm, "coolwarm", spheres, radii, L, z0, title)
    cb = fig3.colorbar(pc3, ax=ax3, fraction=0.025, pad=0.02)
    cb.set_label("V / V_ref   (1 = on target)")
    fig3.suptitle("Equalisation quality after relaxation (blue=too small, red=too big)", fontsize=13)
    p = os.path.join(outdir, "stages_rel.png")
    fig3.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    main()
