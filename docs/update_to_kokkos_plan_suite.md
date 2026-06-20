# Kokkos+MPI migration — suite-aligned revision

> **Status (2026-06-21): all 8 phases (0–7) implemented on branch `suite-integration`.**
> Device cutter + full tessellator reproduce the legacy Voronoi *and* Power diagrams
> bit-exactly (volume/neighbour-set/Euler), the atomic-free physics force matches
> legacy to ~3e-16, the incremental path tracks rebuild to ~1e-15, and the
> transport-core distributed path matches serial owned-cells at np=1,2,4. The
> Kokkos work runs on the OpenMP backend (CI builds Kokkos 4.5 from source); CUDA/HIP
> backends and at-scale GPU granularity tuning are the remaining hardware-gated work.
> Submodule commits are on `suite-integration` (not pushed; umbrella pointer not bumped).

This document **overlays** [`update_to_kokkos_plan.md`](update_to_kokkos_plan.md), which remains
the detailed reference for the engine port (layered architecture, `TessellationView`, the
`Weighting` policy, the invariant oracle, the cutter/scratch design). That plan was written as
if vorflow were standalone. vorflow lives in the **`peclet`** suite, so this revision replaces
the plan's *from-scratch* MPI/decomposition/transport work with the suite's already-built,
already-validated infrastructure, and aligns the build/test on the suite conventions used by
`sdflow` and `dem`.

Read this together with the original; where they conflict, **this document wins** on
infrastructure/build/MPI, the original wins on engine internals.

## Why this revision

`transport-core/` already provides — and has already been validated *for vorflow specifically*
— the exact MPI infrastructure the original plan's §4/§6 proposes to build:

- [`mpi/validate_voronoi.py`](../mpi/validate_voronoi.py): owned+ghost tessellation across ranks
  via transport-core's `tpx_mpi` shim (ORB decomposition + Lagrangian migration + ghost gather)
  matches the serial full-box tessellation to **~1e-15** at np=1/2/4.
- [`mpi/validate_voronoi_scheme_c.py`](../mpi/validate_voronoi_scheme_c.py): persistent
  owner→ghost halo with force-forwarding (`tpx_mpi.Halo.forward`) — the original's §2.5
  "atomic-free gather" and §4 "incremental + MPI" — at a 3–4× speedup over per-step re-gather.
- Design recorded in [`distributed_voronoi.md`](distributed_voronoi.md).

So the C++ migration consumes transport-core's halo classes (the same ones that Python shim
wraps), exactly as **`dem`** (the Lagrangian sibling) already does. The job reduces to: (1) port
the serial cutter + arenas to Kokkos device code; (2) wire the distributed C++ path to
transport-core.

## A. Reuse the suite instead of rebuilding (the main change)

| Original section | Original intent | Reuse from suite |
|---|---|---|
| §4 Partition (block / RCB) | build a partitioner | `tpx::decomp::BlockDecomposer<3>` (ORB) |
| §4 Ghost-seed halo | build ghost exchange | `tpx::halo::ParticleMigrator<3>` + `tpx::halo::ParticleHalo<3>` |
| §4 GPU-aware exchange shim | build a transport shim | `tpx::halo::DeviceParticleHaloKokkos<3>` (host-staged; `TPX_GPU_AWARE_MPI` opt-in) |
| §2.5 atomic-free gather/transpose | design force forwarding | `ParticleHalo::forward`/`reverse` (reverse = atomic-accumulate) |
| §2.6 deterministic tie-break | per-rank determinism | `globalId` carried by the migrator drives the tie-break |
| §2.7 ArborX neighbor policy | optional ArborX | ArborX v2.1 already bootstrapped + consumed by `dem` |
| boundary SDF / obstacles (future) | — | `tpx::geom::{Sdf,GridSdf}` + VTI I/O |
| Kokkos `View`/`toDevice` | roll our own | `tpx::View<T>`, `tpx::Field3D<T>`, `tpx::toDevice` |

Header locations: `../../transport-core/include/tpx/{decomp,halo,geom,common}/`.

**Layering / dependency rule.** transport-core sits *below* the tessellation core (pure
decomposition/halo/geometry, zero Voronoi knowledge), so depending on it does not violate the
original §1 one-way rule. The graph becomes
`physics → TessellationView → engine → core → {data, transport-core}`.

## B. Build & test on suite rails (replaces the original Phase-0 CMake/CI sketch)

Mirror `dem` (the Lagrangian, ArborX-using, MPI-gated analog):

- **Deps:** `tools/bootstrap_deps.sh <nvidia-cuda|host-openmp|lumi-hip>` builds pinned Kokkos
  5.1.1 + ArborX v2.1 into `extern/install/<backend>`; configure with `cmake --preset <backend>`
  (`suite/CMakePresets.json`). Hard build dependency.
- **CMake:** `find_package(Kokkos CONFIG REQUIRED)` + `find_package(ArborX CONFIG REQUIRED)` +
  `find_package(pybind11 CONFIG REQUIRED)`; `pybind11_add_module(vorflow NO_EXTRAS …)`; link
  `ArborX::ArborX Kokkos::kokkos`. Reuse `suite/cmake/SuiteKokkos.cmake`.
- **MPI gate (the `DEM_MPI` pattern):** `if(VORFLOW_MPI) find_package(MPI REQUIRED COMPONENTS CXX)`,
  add `-I../transport-core/include`, define `VORFLOW_MPI=1`, link `MPI::MPI_CXX`. Single-rank
  module stays byte-identical; MPI entry points behind `#ifdef VORFLOW_MPI` in the bindings.
- **Device kernels** in header-only `.hpp` compiled through the Kokkos launch compiler (never
  `.cu`); one pybind11 TU; drive from Python.
- **Tests:** standalone `tests/kokkos_mpi/CMakeLists.txt` (own `find_package`), np=1,2,4
  (closed + periodic), distributed == single-rank — mirror `dem/tests/kokkos_mpi/`.
- **Style/CI:** keep vorflow's `.clang-format`/`.clang-tidy`/`.github/workflows/ci.yml`; add the
  backend build matrix.

## C. Phase deltas

Phases 0–5 and 7 keep the original's deliverables/acceptance; only the infra/build notes above
apply. The one structural replacement:

**Phase 6 — Distributed C++ path via transport-core (REPLACES the original from-scratch §6).**
Add `include/vorflow/mpi_halo.hpp::KokkosVoronoiHalo`, modeled on
`dem/src/mpi_halo.hpp::KokkosParticleHalo`, wrapping `BlockDecomposer<3>` + `ParticleMigrator<3>`
+ `ParticleHalo<3>` + `DeviceParticleHaloKokkos<3>`. A `stepMpi` header orchestration (modeled on
`dem/src/sim.hpp::demStepMpi`): migrate owned seeds → gather ghosts within `rcut` (2-ring for
dynamics, per `distributed_voronoi.md`) → device tessellate owned+ghost → publish
`TessellationView` → forces → `forward` owner forces onto ghosts (scheme C) → integrate →
`forwardPositions` (periodic wrap built in). Both re-gather and force-forward modes;
`rebuild_every` skin control as in the prototype. Bindings expose
`init_mpi`/`enable_mpi_step`/`step_mpi`/`rank`/`num_ghost` behind `#ifdef VORFLOW_MPI`.
*Accept:* owned-cell tessellation + dynamics identical single- vs multi-rank (invariants + the
existing `mpi/validate_voronoi*.py` tolerances, now against the C++ module); new `tests/kokkos_mpi`
passes np=1,2,4; weak/strong scaling on LUMI-G.

## Verification

1. Per-backend build: `tools/bootstrap_deps.sh host-openmp && cmake --preset host-openmp`
   (then `nvidia-cuda`) → `build/vorflow.*.so`.
2. Invariant oracle (original §5) on Serial/OpenMP/CUDA — the cross-platform truth.
3. Legacy parity vs the frozen `data/*.dat` seed sets.
4. Distributed: new `tests/kokkos_mpi` np=1,2,4; cross-check existing
   `mpirun python mpi/validate_voronoi*.py` against the new C++ module.
5. Throughput: cells/sec (NVIDIA, MI250X) + scaling on LUMI-G; CI perf guard.
