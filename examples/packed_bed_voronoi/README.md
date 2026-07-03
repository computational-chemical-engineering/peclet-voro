# Packed-bed interstitial Voronoi (with SDF walls)

An end-to-end example across the suite: grow a **sphere packing** with `peclet.dem`, scatter points in
the **interstitial** space, tessellate them into a **Voronoi grid whose walls are the packing's SDF**
(the sphere surfaces clip the cells), write a ParaView **unstructured grid**, and plot a cross-section.

```
 pack_and_seed.py ──► packing.txt / packing.npz ──► sdf_voronoi_vtu ──► cells.vtu ──► plot_cross_section.py ──► cross_section.png
   (peclet.dem)            spheres + seeds          (voro headers)      (polyhedra)         (vtk + matplotlib)
```

## Pipeline

1. **`pack_and_seed.py`** — `dem.Simulation` settles `n` equal spheres into a non-overlapping
   *periodic* packing at a target solid fraction (periodicity confines them to the box; a plain
   domain does not). Random points are scattered and only those in the fluid (union-of-balls
   `sdf > margin`, i.e. outside every sphere) are kept. Writes `packing.txt` (for the C++ tool) and
   `packing.npz` (for the plot).

2. **`sdf_voronoi_vtu.cpp`** — for each interstitial seed, builds the Voronoi cell against the other
   seeds (`buildConvexCell`, closest-first with the security-radius early-out), clips it to the
   domain box, and clips it against the **union-of-balls SDF** (`clipCellAgainstSdf`) so the sphere
   surfaces act as curved walls (a few tangent-plane facets per touched sphere — the seed-foot
   Option-A model). Emits the clipped cells as `VTK_POLYHEDRON` cells to `cells.vtu`, with per-cell
   `volume` and a `boundary` flag (cell touches a sphere wall).

3. **`plot_cross_section.py`** — slices `cells.vtu` at `z = z0`, drawing each convex cell's
   cross-section (coloured by 3-D cell volume) with the solid spheres' section discs overlaid.

## Run

```bash
# 1. build the C++ tessellator (once), against the bootstrapped host Kokkos prefix
cmake -B build -DCMAKE_PREFIX_PATH="$PWD/../../../extern/install/host-openmp" -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. pack + seed (dem module built in ../../../dem/build; needs numpy)
PYTHONPATH=../../../dem/build python3 pack_and_seed.py

# 3. tessellate -> cells.vtu   (open this in ParaView)
OMP_PROC_BIND=false ./build/sdf_voronoi_vtu packing.txt cells.vtu

# 4. cross-section figure -> cross_section.png   (needs vtk + matplotlib)
python3 plot_cross_section.py 0.0
```

Open `cells.vtu` in ParaView (colour by `volume`, threshold on `boundary`, or `Slice`/`Clip` it).

## Notes

- **Walls are approximate.** The SDF clip approximates each curved sphere by a few flat tangent
  facets (the first-order Option-A model), so clipped cells sit slightly *inside* the true sphere
  surface — visible as thin gaps between the cells and the section discs, and a ~10% under-fill of
  the fluid volume (also seed-margin shell). Increase the clip resolution or seed density to tighten.
- The tessellator is a self-contained host program over the `voro` headers (`ConvexCell`,
  `buildConvexCell`, `clipCellAgainstSdf`); it does not need the Python module. It is `O(N²)` in the
  seed count (fine for a few thousand seeds).
