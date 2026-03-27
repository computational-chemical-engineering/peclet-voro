# Static Voronoi Tessellation Performance Study
## Protocol Document — version 1.0, 2026-03-27

This document describes every step needed to reproduce the benchmark on any
Linux machine with a C++17 compiler, CMake ≥ 3.16, Boost ≥ 1.65, and Python 3.

---

## 1. Purpose

Compare the wall time for constructing a static, three-dimensional, periodic
Voronoi tessellation using two implementations:

| Implementation | Repository | Notes |
|---|---|---|
| **voronoi_dynamics** | `computational-chemical-engineering/voronoi_dynamics` | Header-only C++17; incremental half-plane cutting + OpenMP |
| **Voro++** | `chr1shr/voro` (tag: `master`) | Widely used reference library |

Two timing categories are measured for each library:

| Category | voronoi_dynamics | Voro++ |
|---|---|---|
| **tess** | `NbrList::setup` + parallel `CellMaker` loop | `container_periodic::put` for all N particles |
| **full** | **tess** + `CellComplex::buildGeometry` (connecting vectors, edge inverses, ∂V) | **tess** + `compute_cell` for every particle |

Three point-set distributions are exercised:

| Distribution | Description | N range (vd / voro++) |
|---|---|---|
| `random_uniform` | N uniform-random points in [0,1)³ | 100–1 000 000 / same |
| `cubic_lattice` | Simple-cubic lattice (N = n³) + 10⁻⁷ × spacing jitter | 125–1 000 000 / same |
| `sphere_surface` | N Marsaglia-normalised points on a sphere of radius 0.4, centred at (0.5,0.5,0.5) | **none** / 100–10 000 |

> **Two library-specific limitations for sphere_surface:**
>
> 1. **voronoi_dynamics** is not benchmarked at all for this distribution.
>    The hollow sphere interior creates cells that accumulate more than 128
>    facets even at N = 100, overflowing the library's fixed-size static
>    arrays (`maxNumVertices = maxNumFacets = 128`).
>
> 2. **Voro++** exhibits super-linear (empirically O(N²)) scaling for this
>    distribution: the empty sphere interior forces each cell to check many
>    more candidate cut planes, so cost per cell grows with N.  At N = 10 000
>    the single run already takes ~14 s; N = 50 000 would require > 5 min per
>    repetition.  The study caps Voro++ sphere tests at N = 10 000.

---

## 2. Prerequisites

### 2.1 System packages

```bash
# Ubuntu / Debian
sudo apt-get install -y \
    build-essential cmake git \
    libboost-dev libomp-dev python3 python3-pip

# Fedora / RHEL
sudo dnf install -y \
    gcc-c++ cmake git boost-devel libomp-devel python3 python3-pip
```

### 2.2 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib numpy scipy
```

### 2.3 Verify

```bash
cmake --version      # >= 3.16
g++ --version        # >= 9  (C++17)
python3 --version    # >= 3.9
```

---

## 3. Build

```bash
# Clone (or update) the repository
git clone https://github.com/computational-chemical-engineering/voronoi_dynamics.git
cd voronoi_dynamics

# Configure — Release mode + benchmarks ON, tests OFF (faster configure)
cmake -S . -B build_bench \
      -DCMAKE_BUILD_TYPE=Release \
      -DVORONOI_BUILD_TESTS=OFF \
      -DVORONOI_BUILD_BENCHMARKS=ON

# Build (Voro++ is fetched and compiled automatically)
cmake --build build_bench --target benchmark_voronoi -j$(nproc)
```

### 3.1 Compile flags applied

The following flags are set **per-target** on `benchmark_voronoi` and on the
Voro++ static library via `target_compile_options`:

| Flag | Effect |
|---|---|
| `-O3` | Full optimization beyond `-O2`: auto-vectorisation, loop transforms |
| `-march=native` | Emit instructions for the exact CPU (may enable AVX, AVX-512, etc.) |
| `-mtune=native` | Schedule for the CPU's micro-architecture |
| `-ffast-math` | Allow reassociation, fused multiply-add, NaN/Inf elision |
| `-funroll-loops` | Unroll hot loops for throughput |
| `-DNDEBUG` | Disable `assert()` and debug checks |

> **Reproducibility note.** `-march=native` generates CPU-specific code.
> Binaries compiled on one machine may not run (or may run with degraded
> performance) on a different CPU.  Report the exact CPU model alongside
> results (printed in the CSV header).

---

## 4. Run the Benchmark

```bash
cd build_bench/benchmarks
mkdir -p results
./benchmark_voronoi results/benchmark_results.csv
```

Progress is written to **stderr**; the CSV is written to the file argument.
Typical wall time: 10–15 minutes on a modern workstation (dominated by large
Voro++ full-build runs at N = 10⁶).

Redirect stderr to a log if desired:
```bash
./benchmark_voronoi results/benchmark_results.csv 2> results/benchmark.log
```

### 4.1 Repetition schedule

| N | reps (timed) | warm-up |
|---|---|---|
| ≤ 500 | 20 | 1 |
| ≤ 2 000 | 10 | 1 |
| ≤ 10 000 | 5 | 1 |
| ≤ 100 000 | 3 | 1 |
| ≤ 500 000 | 2 | 1 |
| > 500 000 | 1 | 1 |

### 4.2 CSV output format

```
# <comment lines beginning with #>
library,point_set,N,nthreads,reps,time_ms_mean,time_ms_std
voronoi_dynamics_tess,random_uniform,1000,48,5,42.5,1.2
...
```

---

## 5. Generate the Report

```bash
cd voronoi_dynamics/benchmarks        # run from the repository root/benchmarks
python3 analyze_results.py \
        ../build_bench/benchmarks/results/benchmark_results.csv \
        --output-dir ../build_bench/benchmarks/results
```

This produces:
- `results/figures/*.png` — timing plots
- `results/BENCHMARK_REPORT.md` — markdown report with embedded figures

---

## 6. Expected Output Structure

```
build_bench/
  benchmarks/
    results/
      benchmark_results.csv     # raw timing data
      benchmark.log             # stderr progress (optional)
      figures/
        timing_random_uniform.png
        timing_cubic_lattice.png
        timing_sphere_surface.png
        timing_per_particle.png
        speedup_ratio.png
        geometry_fraction.png
      BENCHMARK_REPORT.md
```

---

## 7. Reporting System Information

To include system information in the report, run:

```bash
echo "# System info" > results/sysinfo.txt
uname -srvmo                          >> results/sysinfo.txt
lscpu | grep -E "Model name|Socket|Core|Thread|MHz|cache" >> results/sysinfo.txt
g++ --version                         >> results/sysinfo.txt
cmake --version | head -1             >> results/sysinfo.txt
cat /proc/cpuinfo | grep "flags" | head -1 | tr ' ' '\n' | grep -E "avx|sse|fma" | sort -u >> results/sysinfo.txt
```

---

## 8. Interpreting Results

- **`voronoi_dynamics_tess`** vs **`voropp_tess`**: raw tessellation cost
  (data structure construction, no geometric properties).
- **`voronoi_dynamics_full`** vs **`voropp_full`**: end-to-end cost including
  cell geometry (volumes, area vectors, gradient operators for vd; full cell
  structure for voro++).
- **Geometry overhead** = (`full` − `tess`) / `full` ×100%: fraction of time
  spent on geometric property computation.
- **Speedup** = `voropp_full` / `voronoi_dynamics_full`: > 1 means vd is
  faster; < 1 means voro++ is faster.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| CMake cannot find Boost | Boost headers not on standard path | Set `-DBOOST_ROOT=/path/to/boost` |
| FetchContent fails (voro++) | No internet during configure | Run on a machine with internet access; or pre-clone voro++ and use `FetchContent_Declare(... SOURCE_DIR /path)` |
| Benchmark crashes on sphere_surface | N > 100 passed to voronoi_dynamics | Expected; the code skips vd for sphere N > 100 automatically |
| Very long run time | N = 10⁶ Voro++ full | Normal; ~15 s/rep. Reduce reps by editing `nreps()` in `benchmark_voronoi.cpp` |
| Results differ between runs | OS scheduling noise | Isolate cores with `taskset` or `numactl`; increase reps for small N |
