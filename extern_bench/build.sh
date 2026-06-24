#!/usr/bin/env bash
# Reproducible CPU head-to-head: geogram VBW::ConvexCell vs our vor::ConvexCell.
# Clones geogram (shallow), compiles its ConvexCell standalone, builds bench_geogram.
set -e
cd "$(dirname "$0")"
[ -d geogram ] || git clone --depth 1 https://github.com/BrunoLevy/geogram.git geogram
GEO=geogram/src/lib
FLAGS="-O3 -std=c++17 -fopenmp -DNDEBUG -DSTANDALONE_CONVEX_CELL -DGEOGRAM_API= -include prelude.h"
g++ $FLAGS -I $GEO -c $GEO/geogram/voronoi/convex_cell.cpp -o geo_cc.o
g++ $FLAGS -I $GEO -I ../include -I shim bench_geogram.cpp geo_cc.o -o bench_geogram
echo "built ./bench_geogram  — run: OMP_NUM_THREADS=1 ./bench_geogram 1000000"
