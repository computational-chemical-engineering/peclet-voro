# Implementation Plan — Kokkos + MPI Voronoi/Power Tessellation Core

> **Suite note:** this plan predates vorflow joining the `peclet` suite. Read it together with
> [`update_to_kokkos_plan_suite.md`](update_to_kokkos_plan_suite.md), which reuses
> `transport-core` for decomposition/halo/transport (replacing the from-scratch §4/§6 here) and
> aligns the build/test on the suite conventions. That overlay wins on infrastructure/MPI/build;
> this document remains the reference for the engine internals.

**Scope of this plan.** Migrate the tessellation engine (cell creation + maintenance) to a performance-portable Kokkos + MPI design that scales to very large systems. The tessellation is treated as a **standalone, reusable library**; physics is a downstream consumer behind a stable interface. First milestone is a correct, fast, scalable Voronoi/Power-cell builder — not the physics.

**Working assumptions** (correct me where wrong): seeds may be polydisperse and the **power (Laguerre) diagram is a first-class case**, not an afterthought; targets are LUMI-G (MI250X) and NVIDIA, plus multicore CPU; the CPU keeps the incremental update, the GPU does full rebuild each step; double precision is used for the cut predicate, storage precision is configurable.

---

## 1. Architecture: layered modules, one-way dependencies

```
            +-------------------------------------------------+
 physics →  |  Physics modules (Euler, NS, IntfDyn, future…)  |   (separate libs)
            +-------------------------------------------------+
                                  | reads only
                                  v
            +-------------------------------------------------+
 tess    →  |  TessellationView  (read-only device API)       |   stable interface
            +-------------------------------------------------+
                                  ^ publishes
            +-------------------------------------------------+
 engine  →  |  Tessellation engine                            |
            |   - neighbor search (grid | ArborX)  [policy]   |
            |   - full-rebuild driver (device)                |
            |   - incremental driver (host)                   |
            |   - CSR write-out (parallel_scan)               |
            +-------------------------------------------------+
                                  | calls
                                  v
            +-------------------------------------------------+
 core    →  |  Cell cutter  (KOKKOS_FUNCTION, scratch Cell)   |
            |   + Weighting policy (Unweighted | Power)        |
            |   + topology-oriented robust predicates         |
            +-------------------------------------------------+
            +-------------------------------------------------+
 data    →  |  Kokkos Views / CSR arenas / mirrors            |
            +-------------------------------------------------+
            +-------------------------------------------------+
 dd      →  |  Domain decomposition + halo (MPI)              |
            +-------------------------------------------------+
```

**Dependency rule (load-bearing).** Arrows point one direction only: `physics → TessellationView → engine → core → data`. The core geometry never includes a physics header. This is what makes the tessellation reusable for non-physics applications (packing/microstructure analysis, mesh generation, etc.). Enforce it in CI with an include-graph check, not just convention.

---

## 2. Core design decisions to bake in

### 2.1 Physics decoupling: a read-only `TessellationView`
Physics kernels never touch the half-edge internals. They `parallel_for` over a published, data-oriented view that exposes per-cell and per-facet quantities from the CSR arenas:

- `numCells()`, `cellSeedId(i)`
- `cellVolume(i)`
- facet range `[facetBegin(i), facetEnd(i))`
- per facet: `facetNeighbor(f)`, `facetAreaVec(f)`, `facetCentroid(f)`, `connectingVector(f)` (the `dV`), and the transpose handle into `NbrsToFacets`
- optionally per-vertex positions for modules that need them (gradient schemes)

A physics module declares which fields it needs (a compile-time field set) so the engine can skip publishing unused geometry. New physics = new consumer of this view; zero changes to the core.

### 2.2 Weighted/Power as a compile-time policy, not a runtime branch
Keep the existing `Weighted` split but promote it to a `Weighting` policy class injected into the cutter:
- `Unweighted`: bisector plane offset = midpoint.
- `Power`: offset shifted by `(w_i − w_j)` (radical plane). Security radius and halo criterion become **power-distance**, not Euclidean (a distant heavy seed can still claim a facet).
Policy is a template parameter so the hot cutter has no per-cut branch; both instantiations compile from one source. Leave room for a future `Anisotropic` policy by keeping the plane definition behind the policy interface.

### 2.3 Construct in scratch (AoS) → publish to global (CSR)
The cell *under construction* stays compact AoS in team scratch / registers — half-edge traversal is random local pointer-walking and belongs in LDS, not strided global memory. Only the **finished** cell is written to global CSR, sized by an exclusive scan over per-cell facet/vertex counts. Do **not** store the persistent topology at the 128-cap fixed stride: mean cells are ~15 facets, so cap-padding is ~8× wasted bandwidth on every downstream gather. The existing `TopologyArena`/`ConnectivityArena`/`GeometryArena` are the SoA/CSR target; they become `Kokkos::View`s.

### 2.4 Execution granularity — resolve by measurement (Phase 3)
The plane-cut sequence mutating the half-edge structure is inherently serial; the **neighbor security-radius scan and signed-distance predicate evaluation are parallel**. Three candidates, to benchmark, not pre-decide:
- **thread-per-cell**: simplest, divergent, register/scratch-heavy (cap drives occupancy).
- **warp/team-per-cell**: cooperatively evaluate candidate planes and the radius scan; apply cuts serially. More scratch, less divergence, parallelism inside the cell.
- **complexity-binned**: pre-classify cells by neighbor count, launch homogeneous batches to cut divergence.

Size scratch from the empirical 99.9th-percentile facet/vertex count, **not** the 128 cap, with an overflow fallback (rare overflow cells re-run in a second pass with larger scratch, or on host). Make scratch size a tunable.

### 2.5 Atomic-free forces via gather + transpose
- Pairwise facet forces (pressure, viscous): each cell gathers its own facets into its own output; the shared facet contribution negates across `i↔j` (Newton's third law). Computes each facet twice, zero atomics — a win over contention on GPU.
- The interface-tension force couples a facet to *other facets of the same cell* and lands on multiple neighbors; its atomic-free form is the **transpose**, which is exactly what `NbrsToFacets` provides. Build/maintain that map and gather through it. Verify it stays cheap to refresh under the incremental update.

### 2.6 Robustness + cross-platform determinism
Keep the topology-oriented (Sugihara-style) predicates — they remove the need for divergent adaptive-precision exact arithmetic, which is ideal for SIMT. **But** the method guarantees a *valid* cell, not the *same* cell: near-degenerate sign decisions can flip between host/device or between ranks under FMA/rounding differences, yielding different-but-valid topologies. For MPI this is a correctness issue at rank boundaries (a shared facet must be decided identically on both sides). Mitigations to bake in:
- Evaluate the cut predicate in **double**, with `-ffp-contract=off` (or a consistent contraction setting) on every backend and rank.
- Make the degeneracy tie-break a deterministic function of seed *global ids*, not of memory order, so two ranks looking at the same pair of seeds always decide the same way.
- Cross-validate by **geometric invariants**, not facet-for-facet topology (see §5).

### 2.7 Neighbor search as a policy
- **Cell-linked grid** (counting sort via `parallel_scan`) for near-monodisperse seeds — your OpenMP path is already this pattern.
- **ArborX** for strongly polydisperse / power cells where interaction radius varies (consistent with your earlier ArborX decision; reuse its distributed tree for the MPI partition in §4).
Inject behind a `NeighborSearch` policy so the engine code is identical for both.

---

## 3. Data structure specification

All persistent state lives in `Kokkos::View`s with explicit `MemorySpace` and `Layout`; host access via mirrors. Default `LayoutLeft` on device so the leading (cell/facet) index is the coalesced one.

**Seeds (SoA):**
- `pos`  : `View<real_t*[3]>`
- `weight` : `View<real_t*>` (Power only; empty View otherwise)
- `globalId` : `View<gid_t*>` (stable across ranks; drives deterministic tie-break)
- `active` : `View<uint8_t*>`

**Topology CSR (published):**
- `cellFacetOffset` : `View<uint*>` (size nCells+1, exclusive scan of facet counts)
- `facetNeighbor`   : `View<gid_t*>` (CSR-packed)
- `facetAreaVec`    : `View<real_t*[3]>`
- `facetConnect`    : `View<real_t*[3]>`  (`dV`)
- `cellVolume`      : `View<real_t*>`
- vertex CSR (`cellVertexOffset`, `vertexPos`) published only when a consumer requests it

**Scratch Cell (per team, transient):** fixed-cap AoS mirror of the current `Cell` (vertex pos, half-edge `Vertex` labels, facet labels, neighbor ids, plane vecs/offsets) in `ScratchMemorySpace`. This is the only place the 128-cap appears.

**Transpose map:** `NbrsToFacets` as CSR (`View`s) for the gather-form interface force and for sparse-matrix assembly.

Provide a thin `legacy ↔ View` converter so the existing CPU code can populate/validate the new layout during transition (used heavily in Phases 0–2).

---

## 4. MPI / domain decomposition

**Key property to exploit:** a Voronoi/power cell is fully determined by its local neighborhood, so with a correct halo the tessellation needs **one halo exchange, no iteration**. That is the scalability lever.

- **Partition.** Start with a regular block grid for uniform density; use recursive coordinate bisection (or ArborX distributed-tree partitioning) for non-uniform density. Partition on seed count first; revisit with cost-weighting in Phase 7 (construction cost ∝ facet count ∝ local disorder, not seed count).
- **Halo width.** = local **security/power radius**, derived per-boundary from local density and (for Power) the maximum weight spread. Import ghost seeds (pos, weight, globalId) from neighbor ranks within that radius.
- **Build.** Tessellate owned + ghost cells; owned cells are exact given a correct halo. Boundary facets are consistent across ranks **iff** the predicate is deterministic in global ids (§2.6).
- **Incremental + MPI.** Static tessellation needs one exchange; the CPU incremental path refreshes the halo only when boundary seeds move beyond a skin distance (Verlet-list style), keeping communication amortized.
- **GPU-aware exchange.** Pack ghost data into contiguous device buffers and exchange device-to-device; keep the NVIDIA vs AMD GPU-aware-MPI config differences behind a small transport shim.

**Acceptance for the MPI layer:** owned-cell tessellation is identical single-rank vs multi-rank (invariants + boundary facet reciprocity across ranks), and weak/strong scaling curves are produced on LUMI-G.

---

## 5. Invariants and test harness (the backbone)

Codify these as automated checks; they are the cross-platform/cross-rank oracle since exact topology may legitimately differ:

1. **Space-filling:** Σ cellVolume = box volume (periodic) within tol.
2. **Euler characteristic:** per cell, V − E + F = 2.
3. **Facet reciprocity:** facet `i→j` exists ⇔ `j→i` exists; area vectors negate; connecting vectors consistent.
4. **Area closure:** Σ facetAreaVec over a cell = 0.
5. **Power/Voronoi membership:** every published facet's plane is the correct (radical/bisector) plane for its seed pair; no seed lies inside another's cell.
6. **Golden master:** geometric invariants match the legacy CPU build within tol on a frozen seed set (uniform, lattice/degenerate, strongly polydisperse, clustered).

Harness runs on every backend (Serial, OpenMP, CUDA, HIP) and at 1, 2, N ranks.

---

## 6. Phased plan with acceptance criteria

**Phase 0 — Scaffolding & oracle.**
Deliverables: CMake + Kokkos (+ optional ArborX/Trilinos) + MPI build; CI matrix (Serial/OpenMP/CUDA/HIP × 1/N ranks); legacy code wrapped; golden outputs captured; invariants §5 implemented as tests; include-graph dependency check (§1).
Accept: invariants pass on legacy outputs; CI green on all backends (kernels still legacy/host).

**Phase 1 — Data layer.**
Deliverables: all Views/CSR arenas (§3); `legacy ↔ View` converter; `TessellationView` API surface (no kernels yet).
Accept: round-trip legacy → Views → legacy preserves invariants exactly; `TessellationView` queries reproduce legacy per-cell/per-facet values.

**Phase 2 — Device-callable cutter.**
Deliverables: cut/clip ported to `KOKKOS_FUNCTION` on scratch Cell; `Weighting` policy (Unweighted, Power); deterministic global-id tie-break; double-precision predicate.
Accept: single-cell construction matches legacy topology on Serial/OpenMP for the §5 seed battery; runs on CUDA/HIP producing invariant-valid cells; Power case validated against an independent radical-plane reference.

**Phase 3 — Full-rebuild tessellation on device.**
Deliverables: neighbor search policy (grid + ArborX); `TeamPolicy` construction with scratch; CSR write-out via `parallel_scan`; the §2.4 granularity benchmark resolved and documented; scratch sizing from empirical distribution + overflow fallback.
Accept: full-system tessellation passes invariants on GPU; cells/sec baseline reported on NVIDIA and MI250X; throughput scales with N; overflow fallback exercised by a stress case.

**Phase 4 — Physics interface + reference module.**
Deliverables: atomic-free gather force kernel over `TessellationView`; `NbrsToFacets` transpose for the interface force; one reference physics (Euler pressure) as the first consumer.
Accept: forces match legacy within tol; **zero atomics** in the kernel (verify by inspection/tooling); the physics module compiles against `TessellationView` only (dependency check passes).

**Phase 5 — CPU incremental update.**
Deliverables: incremental worklist driver on host (OpenMP/Threads) calling the **same** cutter; halo-skin refresh hook.
Accept: incremental result matches full-rebuild over a moving-seed trajectory (invariants + geometric match each step); measured crossover points (incremental-vs-full on CPU; CPU-incremental-vs-GPU-full).

**Phase 6 — MPI domain decomposition.**
Deliverables: partition (block/RCB/ArborX); power-aware halo sizing + ghost import; owned+ghost build; cross-rank boundary consistency check; GPU-aware halo transport shim.
Accept: owned-cell tessellation identical single- vs multi-rank (invariants + cross-rank reciprocity); weak + strong scaling on LUMI-G.

**Phase 7 — Optimization & hardening.**
Deliverables: profiling (occupancy/roofline), Layout tuning, scratch/team retuning, cost-weighted load balancing, mixed-precision storage tuning, telemetry replacing the legacy atomics counters.
Accept: documented occupancy/bandwidth targets met; scaling curves; perf regression guard added to CI.

---

## 7. Open questions to resolve by measurement (not up front)

- Construction granularity: thread- vs warp/team- vs binned-per-cell (Phase 3).
- Real facet/vertex distribution → scratch size and occupancy tradeoff.
- Grid vs ArborX crossover as a function of polydispersity.
- Incremental-vs-full and CPU-vs-GPU crossovers (Phase 5) — these decide the production execution policy per regime.
- Load-balance metric: seed count vs facet-count-weighted partition (Phase 7).
- Halo refresh cadence / skin distance for the MPI + incremental combination.

---

## 8. Sequencing summary

Correctness oracle (0) → layout (1) → portable cutter (2) → **first real milestone: GPU full rebuild (3)** → modular physics boundary proven (4) → CPU incremental (5) → scale-out (6) → optimize (7). Phases 0–3 deliver the standalone, reusable, GPU-fast tessellation library you asked to prioritize; 4 proves the physics decoupling; 5–7 add maintenance, scale, and speed.