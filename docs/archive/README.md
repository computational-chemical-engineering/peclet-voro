# Archive — design notes, plans and campaign records (`peclet.voro`)

These are **dated design notes, phased plans, benchmark records and session handoffs**, kept for the
record and moved out of the live documentation set on 2026-09-08 (suite `docs/QUALITY_PLAN.md`
decision D7: *docs describe the code that exists*). Each was written to steer a piece of work that
has since landed, been superseded, or been folded into the README; the authority on how the engine
behaves today is the code under `include/peclet/voro/` plus the live documents one level up —
[`../../README.md`](../../README.md) (the engine, its data structure, the Python API and every
measured rung), [`../architecture.dox`](../architecture.dox) (the Doxygen architecture page),
[`../distributed_voronoi.md`](../distributed_voronoi.md) (the MPI path) and
[`../performance_report.md`](../performance_report.md) (the current performance/accuracy report).

**Nothing here is maintained.** File paths, `file:line` citations, throughput tables, "status" boxes
and "next step" sections are snapshots of their date — several describe the retired half-edge
(`voronoi.hpp` / `CellComplex` / `ScratchCell`) engine or the `vordyn` module name as if current, and
their cross-references to each other now resolve inside this directory. Re-read the source before
acting on any of them. Nothing here is deleted either.

| note | date | what it is |
|---|---|---|
| [dynamic_update_decision_and_plan.md](dynamic_update_decision_and_plan.md) | 2026-06 | The phased plan for the moving-point update loop (phases 0–4): the decision, the two-pass repair design, the gate templates. Executed — `repair.hpp`'s `MovingTessellation` is the result. |
| [dynamic_update_phase01_results.md](dynamic_update_phase01_results.md) | 2026-06-29 | Measurement record for phase 0 (validators + benchmark foundation) and phase 1 (repair primitives, `tess_grid.hpp`). |
| [dynamic_update_phase2_results.md](dynamic_update_phase2_results.md) | 2026-06-29 | Two-pass gather repair vs cold build across four device configurations, as a function of δ/h. |
| [dynamic_update_phase34_results.md](dynamic_update_phase34_results.md) | 2026-06-29 | Phase 3 (the adaptive three-way churn gate, now production) and phase 4 (surgical repair) results. |
| [dynamic_update_repair_sweep.md](dynamic_update_repair_sweep.md) | 2026-06-29 | The displacement sweep behind the gate thresholds — speedup and re-gathered-cell counts vs δ/h, per backend. Figures live in `../figs/`. |
| [free_surface_design.md](free_surface_design.md) | 2026-07 | Design-only note (no engine code) for a free-surface layer on the power-cell machinery. |
| [full_port_plan.md](full_port_plan.md) | 2026-06/07 | The de-legacy port plan: remove the header-only half-edge engine and ship one device path as `voro` (retiring `vordyn`). Done — the legacy engine was deleted in `0d4f3b8`. |
| [performance.md](performance.md) | 2026-06/07 | The optimisation ledger and profile analysis of the tessellator, written while the half-edge "ScratchCell" path was still production. Superseded by `../performance_report.md`; it is the comparison that led to adopting ConvexCell. |
| [power_cell_solver_spec.md](power_cell_solver_spec.md) | 2026-06 | Specification for a near-incompressible power-cell fluid solver from a variational (Lagrangian) principle. Not implemented as written. |
| [power_large_weights_plan.md](power_large_weights_plan.md) | 2026-09-03 | Rung A2 design note: power diagrams beyond the small-weight regime (buried cells, the general `(û, d)` half-space, multi-image gather). A2a landed; A2b/A2c open. |
| [sdf_geometry_plan.md](sdf_geometry_plan.md) | 2026-09-03 | The plan for SDF-defined geometry on the tessellation (rungs A0/A1: moving-point clipping, the resident wall store, second-order sagitta placement). Landed — the README's "Curved walls" section is the live description. |
| [update_and_repair_redesign.md](update_and_repair_redesign.md) | 2026-06 | The earlier wave-BFS / `ConnectivityArena` repair design, superseded by the two-pass gather repair. |
| [voronoi_build_plan.md](voronoi_build_plan.md) | 2026-06 | The actionable plan for the GPU build engine ahead of moving particles — phases, decisions, gates. |
| [voronoi_coldbuild_benchmark_report.md](voronoi_coldbuild_benchmark_report.md) | 2026-06-27 | Cold-build benchmark report (serial / multicore / GPU / MPI + voro++, geogram, the Liu-2020 GPU code), with its concurrent-load caveat. |
| [voronoi_cold_tessellation_benchmark.md](voronoi_cold_tessellation_benchmark.md) | 2026-06-26 | Cold tessellation throughput vs N against voro++ and the SOTA GPU reference. |
| [voronoi_construct_ledger.md](voronoi_construct_ledger.md) | 2026-06 | Running ledger of every attempt to push the single-thread ConvexCell construct toward SOTA, with the measured result of each. |
| [voronoi_cpu_migration_discussion.md](voronoi_cpu_migration_discussion.md) | 2026-06/07 | Briefing note for adapting the GPU per-vertex ConvexCell engine to multicore CPU. |
| [voronoi_dynamic_update_study.md](voronoi_dynamic_update_study.md) | 2026-06/07 | Controlled comparison (S0–S6) of moving-seed update strategies and the convexity-certificate idea. |
| [voronoi_gpu_research_program.md](voronoi_gpu_research_program.md) | 2026-06/07 | Research note reframing the problem around the per-step update, with the phase-0 findings and the phased experimental program. |
| [voronoi_neighbor_update_overview.md](voronoi_neighbor_update_overview.md) | 2026-06/07 | Broad handoff on neighbour finding and moving-point updating: the implemented method, the legacy CPU strategies, a literature survey, and avenues. |
| [voronoi_pervertex_geometry_report.md](voronoi_pervertex_geometry_report.md) | 2026-06-25 | Findings for the vertex-local, sort-free ConvexCell geometry (divergence theorem, no `atan2`) — now the production kernel. |
| [voronoi_simd_cells_prototype.md](voronoi_simd_cells_prototype.md) | 2026-06-26 | Bounded CPU prototype: cells-as-lanes SIMD for the per-vertex geometry kernel; divide-bound, fast reciprocal is the lever. |
| [voronoi_worklist_gather_project.md](voronoi_worklist_gather_project.md) | 2026-06-26 | The voro++-style worklist gather project brief and its closing numbers. Done — the gather is the default on both backends. |
