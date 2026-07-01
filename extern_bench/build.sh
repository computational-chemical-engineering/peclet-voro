#!/usr/bin/env bash
# Reproducible CPU head-to-head: geogram VBW::ConvexCell vs our vor::ConvexCell.
# Clones geogram (shallow), compiles its ConvexCell standalone, builds bench_geogram.
set -e
cd "$(dirname "$0")"
[ -d geogram ] || git clone --depth 1 https://github.com/BrunoLevy/geogram.git geogram
GEO=geogram/src/lib
# voro++ (voronoicell::plane clip-only path) from the voro build's FetchContent deps
VOROSRC=$(ls -d ../build/*/_deps/voropp-src/src 2>/dev/null | head -1)
VOROLIB=$(ls ../build/*/_deps/voropp-build/libvoro++.a 2>/dev/null | head -1)
FLAGS="-O3 -std=c++17 -fopenmp -DNDEBUG -DSTANDALONE_CONVEX_CELL -DGEOGRAM_API= -include prelude.h"
g++ $FLAGS -I $GEO -c $GEO/geogram/voronoi/convex_cell.cpp -o geo_cc.o
g++ $FLAGS -I $GEO -I ../include -I shim -I "$VOROSRC" bench_geogram.cpp geo_cc.o "$VOROLIB" -o bench_geogram
echo "built ./bench_geogram  — run: OMP_NUM_THREADS=1 ./bench_geogram 1000000"
