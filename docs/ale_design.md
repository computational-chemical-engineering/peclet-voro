# Two-phase flow on moving power cells in `peclet.voro`: design note (phase A)

Status: design, 2026-11-14, branch `ale-design`. No production code exists yet. This note is the
contract for the work orders of §7. A worker who executes a work order reads this note, the
standing rules (`peclet-papers/work/M7/ALE_RULES.md`) and the formula sections of
`peclet-papers/work/M7/ALE_DERIVATION.md` that the work order's brief quotes. **The worker does not
derive anything.** Where this note says "per the derivation", the brief must quote the formula. If
it does not, the worker stops.

Sources: the orchestrator's digest `peclet-papers/work/M7/briefs/ALE_PHASE_A_DIGEST.md` (2026-11-14),
the update of the same day (the 2D dynamic gate of the closure has run, and the mesh energy is now
open; see §4.6 and Q-F9), the voro and core code at `819d9d6`, and the suite contracts
(`docs/CONVENTIONS.md`, `NAMING.md`, `DECISIONS.md`). Tags as in the rules: [derived], [measured]
(2D/3D Python oracle, digest), [estimate].

---

## 0. Decisions at a glance

| # | Decision | Rejected alternative (reason in the section) |
|---|---|---|
| D1 | Geometry lives in **frames**. A frame is one repair, one publish and one full face-mesh **rebuild**. There are two full frames per step (n+½, n+1), plus volumes and centroids of frame n. | Incremental face-mesh update with stable face ids (§1.5) |
| D2 | A new solver in a new layer-5 directory `include/peclet/voro/ale/`. `CollocatedNS`, `FlowSolver` and covolume are not modified. | Extending `CollocatedNS` (§6) |
| D3 | The ALE operators follow the formulation: the Voronoi gradient G_i and Ḡ_f = ½(G_i+G_j). They are new kernels, not the existing GG-skew kernels. | Reusing `projectToFaces`/`faceInterpTranspose` (§A.3, Q-OPS) |
| D4 | Ghost state is **one cell ring**, selected by distance as now. Intermediates (G, transpose moments) are exchanged. | A two-ring (or four-ring) cell halo (§3.3) |
| D5 | Interface faces get **one keyed exchange per frame**. A face survives only if both sides have it (reciprocity), and its geometry comes from the side with the lower gid. All faces are oriented lower gid → higher gid. | Recomputing on both sides without a check (§3.5) |
| D6 | Dual triangles are decided by **one owner rank** (the owner of the triangle's minimum-gid point) and then exchanged. All per-point sums are gathers in canonical key order. | Each rank deriving its own list (§1.7) |
| D7 | A pair is migrated **atomically**: both generators and the point go to the owner of x_k. | Independent ownership of the two generators (§3.7) |
| D8 | The pressure Laplacian is an assembled CSR per frame. It is preconditioned by a **device, deterministic, phase-respecting unsmoothed-aggregation AMG rebuilt every frame**, with a global one-dof-per-rank coarse level. | Host GraphAMG setup per frame; numeric-only refresh; block-Jacobi without a coarse level (§4.5) |
| D9 | μ̂ is updated by **one warm-started power iteration per step**. | A few cold iterations per step (§4.3) |
| D10 | The predictor applies ζ matrix-free with Jacobi-PCG. A per-component scalar AMG is used when the diffusion number exceeds 2. | Assembling ζ (stencil radius 4) (§4.2) |
| D11 | No floating-point atomics in any ALE kernel. Runs are bitwise reproducible at fixed np and backend (a gate). Results agree across np to a tolerance. | Atomic scatter-add (§2.2) |
| D12 | Births and deaths (stage 3) reuse free slots under an active mask, with a forced worklist in the repair. Overlaps are differences of volumes and first moments at frozen positions, processed in rounds. | Polyhedron intersection; compaction per event (§5) |
| D13 | FP64 throughout: state, geometry and tessellation. | FP32 geometry (the gates sit at 1e-13) |
| D14 | The front force is the **total derivative of a per-point energy W** with a one-ring stencil. The form of the mesh-energy term is pluggable (umbrella χ = 1 is one implementation, and its form is OPEN, F9). There is a slot for a tangential rule velocity ż. | Hard-wiring the umbrella (§4.6) |

---

## A. Problem, scope, constraints

### A.1 What is built
This builds incompressible one- and two-phase Navier–Stokes in 3D on a power diagram whose generators may move.
- The velocity U_i (cell momentum ρ_iV_iU_i) and the pressure p_i live at the generators.
- No unknown lives on faces. Every face quantity is recomputed from cell and front data after each re-tessellation.
- Like facets carry the relative flux. Unlike facets are material.
- The interface is a **paired front**: a point x_k with normal n_k and distance ε_k owns two equal-weight generators x_k ∓ (ε_k/2)n_k, one in each phase.
- The surface energy is the smooth E_s on the dual triangles of the front, plus a mesh energy.
- The pressure coupling is the ABC approximate projection (mode F), **never Rhie–Chow**.
- Forces act inside the predictor.

Stages:
- **S1**: one phase on a moving mesh. The fixed grid is the limit of no motion.
- **S2**: paired front, two phases, bulk cells following the fluid.
- **S3**: stationary bulk crossed by the front, with cell death and rebirth.

**Out of scope, with room left:** implicit surface motion (F10); SDF solids and the contact line (F8); a free surface as a void phase; point insertion and removal (F6); paper text; re-deriving the method. **Not touched:** anything on `m7*` branches or in `lagrangian/` (§6.4).

### A.2 Invariants (what must hold, and how tightly)
| Invariant | Tolerance | Where it is enforced |
|---|---|---|
| Existing voro results (`python/state_hash.py`, all ctests, `bench_dynamic_update --gates`) | bitwise unchanged | every WO touching shared code |
| Eulerian limit V0 (§7, V0.1/V0.2/V0.3) | bitwise / 1e-14 | S0.2, S1.3, S1.1 |
| Adjointness ⟨C′U,p⟩ = ⟨U,C′ᵀp⟩; ζ symmetric | ≤ 1e-14 relative | S1.1 |
| I_F exact for linear and rigid fields, per face | ≤ 1e-13 relative | S1.1 |
| GCL: τ(DΦᵐ)_i = V_i^{n+1} − V_i^n | ≤ 1e-14 V_i | S1.3 |
| Uniform flow through topology changes | ≤ 1e-12 relative | S1.3 |
| Energy budget per step, all booked terms included | ≤ 1e-13 K (or W) | S1.2, S1.3, S2.4 |
| Phase volume per drop; total momentum | ≤ 1e-13 relative | S2.3, S2.4, S3 |
| Bitwise run-to-run at fixed np and backend | bitwise | every WO (gate) |
| np 1 vs np N, one step from identical state / 200 steps | ≤ 1e-12 / ≤ 1e-9 | S1.4, S2.x |

The 1e-9 bound over many steps is set by voro's tessellation equivalence contract: a 1e-9 volume
tolerance plus an exact neighbor set (register: "NOT bit-exactness"). Repair schedules differ with np
inside the certificate band.

### A.3 Formula sheet and sign convention (binding)
Notation:
- **Faces:** f = (A, B) is **oriented from the lower gid to the higher gid**, with n_f a unit vector A → B.
- **Geometry:** d_f = |x_B − x_A| (minimum image); h_Af = (d_f² + w_A − w_B)/(2d_f); h_Bf = d_f − h_Af. c_f is the centroid. q_f = c_f − x_A − h_Af n_f, and the same q_f holds seen from B.
- **Signs:** s_if = +1 if i = A and −1 if i = B.
- **Divergence D is OUTFLOW:** (DΦ)_i = Σ_f s_if Φ_f. This matches voro's `divergence` and `cellFaceSign`.
- **Face difference:** (Δ_n p)_f = p_B − p_A = −(Dᵀp)_f.
- **Cell gradient:** G_i = (1/V_i) Σ_{f∋i} (A_f/d_f)(U_o − U_i) ⊗ (c_f − x_i), with o the other cell. Component (c,k) is ∂_kU_c, stored at 9i+3c+k.
- **Flux map:** (I_F U)_f = A_f [ n_f·(h_Bf U_A + h_Af U_B)/d_f + Σ_{c,k} n_c (Ḡ_f)_{ck} q_k ], with Ḡ_f = ½(G_A + G_B).
- **Constraint:** C′ = D I_F. The physical pressure force on cells is **F_p = −I_Fᵀ Δ_n p = +C′ᵀp**.
- **Translation from the digest:** the digest writes "−C′ᵀp" and δp = +ωL⁺C′U*/τ. In its convention Dᵀ is our Δ_n. Every formula carries over with C′ → −C′. In ours the update is δp = −ω L_ρ⁺ C′U*/τ and U^{n+1} = U* + τ M⁻¹C′ᵀδp. The energy identities do not depend on the convention.
- **The sign is fixed by gate G1.1-h** (hydrostatic balance), not by reading.
- **Swept flux** [derived; verified 6e-9]: Φᵐ_f = (A_f/d_f)[(c_f − x_A)·ẋ_A − (c_f − x_B)·ẋ_B + (ẇ_A − ẇ_B)/2]. It is positive when the face moves along n_f, so cell A grows and τDΦᵐ = ΔV. The relative flux is F = Φ − Φᵐ on LIKE faces and F = 0 on unlike faces.
- **Laplacian:** L_ρ = D diag(a_f) Dᵀ with a_f = A_f/(ρ_f d_f) and 1/ρ_f = (h_Bf/ρ_A + h_Af/ρ_B)/d_f. It is symmetric positive semidefinite.
- **Friction:** ½UᵀζU = Σ_f μ_f V_f |sym G_f|², harmonic μ_f on unlike faces. The face gradient G_f and the face volume V_f are taken from the derivation (Q-FRIC).
- **Transport:** flux form with the mean of the two linear reconstructions at c_f. The upwind part is split as ΔᵀWΔ implicit plus ΔᵀW(R − Δ)U explicit, with W = diag(½ρ_f|F_f|), (ΔU)_f = U_A − U_B, and R the jump of the reconstructions at c_f.
- **Front, per point k:**
  - A_k(ṙ_k − a_k·ṅ_k) = (I_F U)_k on the pair facet, with ṙ_k = n_k·ẋ_k and a_k the centroid offset.
  - Point motion: ẋ_k = ṙ_k n_k + P_k^t ½(U_k⁻ + U_k⁺) + T_k ż_k. The ż slot is default 0 (§4.6).
  - Side facets take Φ := Φᵐ, which gives I_F″ and C″ = D I_F″.
  - Normals are slaved, n = n(x), with S = dn/dx.
  - Force: f = W_x + SᵀW_n + (B_x + B_nS)ᵀD_Γᵀp, with W = E_s + Q (digest notation; translate the sign of the pressure term by the rule above; gate G2.2's rest-state Laplace pressure fixes it).

Orientation of the pair facet: n_f = ±n_k. Kernels use σ_k = sign(n_f·n_k) and never assume A = genMinus.

---

## 1. Data layout and ownership

### 1.1 Index spaces
| Entity | Local index | Global identity | Persistence |
|---|---|---|---|
| Cell (generator) | slot: owned [0, nOwnedSlots), ghosts [nOwnedSlots, nCombined) | `gid` int64, never reused | persistent; migrated |
| Front point | owned [0, nOwnedPts), ghosts after | `pgid` int64 | persistent; migrated |
| Face | per frame: interior/interface [0, nInterior), boundary after | key (gid_lo, gid_hi) used only for matching and diagnostics | **none** (derived per frame) |
| Dual triangle | per frame | key = sorted (pgid_a, pgid_b, pgid_c) | **none** (derived per frame) |
| Drop | global id d ∈ [0, nDrops) | same | persistent (replicated table) |

Gids are int64 in the ALE layer. The tessellation's `gid_t` (int32) is not used for identity: its
slots are local, and stage-3 births exhaust int32 within one long run (1e5 births per step over 1e5
steps). In S1 and S2 every owned slot is active. S3 adds `active` (int8) with free-slot reuse (§5).

### 1.2 Cell state (device SoA, flat `3*i+c` as in voro today)
Persistent and migrated:
- x(3), w; U(3), U_prev(3); p
- phase (int8)
- gid (int64)
- point (int32 local point slot or −1), side (int8: −1 for genMinus/phase 1, +1 for genPlus/phase 0, 0 for bulk)
- drop (int32, the drop id for phase-1 cells, else −1)
- μ̂ vector (1)
- warm starts δp_prev, φ_prev, φᵐ_prev (1 each)

Derived per frame (never migrated): V, cell centroid (3), G (9), scratch.

The x and w of pair cells are **written from point state** by one kernel (x_k ∓ ε_k/2 n_k; equal
weight w_k stored per point). They are never integrated on their own.

### 1.3 Front state (device SoA)
Persistent and migrated:
- x_k(3), n_k(3) (the slaved normal at the last frame, needed to place generators)
- ε_k, w_k
- pgid
- genMinus/genPlus (local slots) together with their gids (for re-linking after migration)
- drop (int32)
- ż_k(2) (default 0)

Derived per frame:
- tangent basis T_k(3×2) built deterministically from n_k
- pair-facet A_k, a_k(3)
- point→triangle CSR, ring CSR
- rates scratch

### 1.4 Frames
A frame (`ale::MeshFrame<Real>`) holds:

| Field | Content |
|---|---|
| snapshot | x, w of the combined cells (owned and ghost) |
| published view | `reevalPublish` output; a **transient** buffer, reused, freed after the face mesh is built |
| FaceMesh | deterministic order, gid orientation, generator geometry, published centroids (§1.5) |
| faceKind (int8) | §1.6 |
| cell data | cellVolume, cellCentroid(3) |
| faceInvRho | only when there are two phases |
| L_ρ | CSR values: coef per cell-face entry plus diag. The structure is the cell→face CSR |
| AMG | the hierarchy (§4.5) |
| front (S2) | sorted triangle list, point→triangle CSR, ring CSR |
| stats | RepairStats, topology-change count, reciprocity defect, side-facet keys (junction diagnostic) |

Lifecycle per step:
- Two full frame buffers live (Fh at n+½, F1 at n+1).
- Of frame n only V^n and the centroids are kept. At the end of the step F1 becomes Fn by swapping handles, and the freed buffer receives the next Fh.
- The pass loop of F10 rebuilds Fh and F1 once per pass.

### 1.5 Face mesh that follows the repair: **rebuild** every frame
**Decision: rebuild** the FaceMesh from the published view in every frame. Reasons:
- All generators move, so every face's area, normal, centroid and distance change in every frame anyway.
- There is no persistent face unknown.
- At production size some topology changes somewhere in every step.
- The structural part of the rebuild is about three passes over facets, against the re-evaluation that touches every facet anyway [estimate; measured by gate P4].

Incremental maintenance (holes, compaction, face ids stable across ranks) would be complexity paid forever for at most that structural part.

**Changed set.** `MovingTessellation` exports its per-cell rebuilt flag (today the internal `rebuilt`, `repair.hpp:249`) as a read-only view `changedMask`, next to the existing `RepairStats`. It is used for:
- diagnostics (topology-change and junction-switch counts);
- verification of stage-3 event rounds;
- a restricted publish in stage 3, built only if gate P4 fails there (WO-P3).

**Builder extensions** (`fv/mesh.hpp`, opt-in through a `FaceMeshOptions` struct). The defaults reproduce today's arrays bitwise:
1. **Deterministic order** (no atomics):
   - Faces are enumerated by (interior before boundary, emitting owned cell slot, facet position within that cell).
   - An owned cell emits a face to an owned neighbor only if it has the lower gid. It always emits interface and boundary faces.
   - Each cell's face list is **sorted by ascending face index**.
   - Counts come from per-cell loops and offsets from scans.
   - For `reevalPublish` views (cell-major) and owner-facet orientation this reproduces today's serial order exactly (proved by V0.1).
2. **Orientation:** owner-facet (today) or **lower gid → higher gid** (ALE). With gid orientation an interface face may have a ghost as A; no ALE kernel assumes A is owned.
3. **Generator geometry:** conn = minimum-image(x_B − x_A), d = |conn|, h_A = (d² + w_A − w_B)/(2d), h_B = d − h_A. Today's d = |2n| and h = |n| hold for Voronoi cells only (the file says "pass positions when that matters"), so they are wrong for power cells.
   - Diagnostic: |n_f × conn|/d ≤ 1e-10 on faces with A_f > 1e-10 h². Power planes are perpendicular to the connector, so a larger value flags a degenerate facet.
4. **Published centroids:**
   - `reevalPublish(..., withFacetCentroids, withCellCentroids)` computes the facet centroid (relative to the owner) and the cell first moment in `ConvexCell::reevalGeometry` by the triangle fan, in the same pass.
   - The builder takes c_f from it. Today's c = r − dV·|r|/|A| assumes r = x_B − x_A, which holds for Voronoi cells only.
   - The cell first moment is also needed by the mesh rule (centroidal regularization) and by the S3 remap.

### 1.6 Face classification; junctions
faceKind is computed per frame by a face kernel from phase and point; the ghost values arrive by the cell exchange.
- LIKE: same phase; carries F.
- PAIR: the two generators of one point; material.
- SIDE: unlike and not PAIR, i.e. pair|pair of different points or pair|bulk; Φ := Φᵐ.
- WALL: boundary.
- FREE: later, toward the void phase.
- **Unlike bulk|bulk is a breakdown** (F6 clearance). It is counted, and if the count is above 0 the step fails loudly (default; Q-F6).

A junction (an edge where four cells of two pairs meet) has **no data structure**. Side facets appear and vanish there. Junction switches are counted as the size of the symmetric difference of the sorted SIDE keys of F1 and Fn (a diagnostic).

### 1.7 Dual triangles (deterministic and identical on all ranks)
1. **Detection** runs over owned pair cells and their diagram edges, which are the edges of the ConvexCell dual-triangle topology.
   - It emits triangles whose minimum-pgid point is owned here, as (key, three local point slots).
   - Integer atomics are used for the append only.
   - The 3D definition of which edges give triangles is the oracle's (`a2f_front3d.py`; Q-TRI).
2. Sort by key and remove duplicates.
3. Exchange owned triangles in the front halo (§3.4) and merge the received ones. Sort by key again.
4. Build the point→triangle CSR in key order, and the ring CSR (unique ring neighbors from triangle edges, sorted by pgid).
5. Check: every triangle edge with an owned endpoint occurs in exactly two triangles (closed manifold). A violation fails the step (non-manifold front; F6).

Each triangle is decided by exactly one rank and copied verbatim, so the lists agree on every rank
that holds them. A flip of a dual triangle is an energy step (F3). It is booked by comparing the
key sets of Fn and F1 (counted; the energy step is reported).

### 1.8 Phase and density
Phase is int8 per cell. A device table holds (ρ_a, μ_a, flags) per phase (flags: `void` reserved for the free surface). ρ_i = ρ[phase_i]; nothing is stored per cell. With one phase, `twoPhase = false`: kernels skip faceInvRho, and 1/ρ multiplies as a global scale. This keeps the one-phase path free of the h_A + h_B ≠ d rounding of 1/ρ_f.

### 1.9 Conventions (binding)
- Phase 0 is continuous and phase 1 dispersed (the drop).
- n_k points out of phase 1. genMinus = x_k − (ε_k/2)n_k lies in phase 1.
- Positions are wrapped into [0, L); all differences are minimum-image.
- SDF is negative in the solid (CONVENTIONS).
- Python arrays are (N, 3).

---

## 2. Device residency and kernel structure

### 2.1 Host is allowed only for
- MPI on host-staged packed buffers, the existing `GhostExchange` pattern (GPU-aware MPI is an independent later step);
- the redundant global coarse factor and solve (np-sized, §4.5);
- scalars: dt, iteration and event counts, convergence flags;
- Python copies on `get_*`;
- one-time setup (seeding, the first cold build);
- the regather topology selection, until WO-S1.6 moves it to the device before stage 1 closes.

**Never on the host:** any per-step loop over cells, faces, points or triangles, the AMG setup, the face mesh, the front kernels. Host oracles live in tests only.

### 2.2 Kernel patterns (the only five)
| Pattern | Policy | Writes | Rule |
|---|---|---|---|
| K1 face loop | RangePolicy over frame faces | per-face arrays | reads A and B (owned or ghost); no accumulation |
| K2 cell gather | RangePolicy over owned slots | per-cell outputs | loops the cell's face list in ascending face order; accumulates in registers |
| K3 point gather | over owned points | per-point outputs | loops point→triangle CSR (key order) or point→unlike-face list (face order) |
| K4 triangle loop | over the frame's triangles | per-triangle partials | consumed by K3 only |
| K5 reduction | `parallel_reduce` (global); per-drop through the drop CSR (team per drop) | scalars | then the MPI reduction |

**No atomic floating-point accumulation anywhere** (D11). Integer atomics for appends are allowed only if a sort follows wherever order matters. Whether `parallel_reduce` and team reductions are run-to-run deterministic on each backend for a fixed launch is **to measure** (gate G-DET). Where it fails, that reduction gets a two-pass blocked sum with a fixed block size.

### 2.3 Operator kernels (stage 1)
| Operator | Pattern | Exchanges before / inside |
|---|---|---|
| G (9N) | K2 over i's faces | U (3) before; G (9) after, to ghosts |
| I_F | K1 | needs U and G at A and B |
| D, Δ_n | K2 / K1 | — |
| I_Fᵀ (and C′ᵀ = I_FᵀDᵀ) | two-pass K2. Pass 1: plain part and T_i/V_i with T_i = ½Σ_{f∋i} g_f A_f n_f⊗q_f. Exchange T/V (9). Pass 2: Σ_{f∋m}(A_f/d_f)[−(T_m/V_m)(c_f−x_m) + (T_o/V_o)(c_f−x_o)] | 9 inside |
| L_ρ x | CSR SpMV, K2 | x (1) |
| ζU | G → face stress σ_f = 2μ_fV_f sym G_f (K1) → transpose as I_Fᵀ | U(3), G(9), moments(9) |
| Φᵐ | K1 from rates | rates (4) once per frame |
| transport tendency | K1 (momentum flux plus explicit upwind) → K2 | shares G |
| predictor operator | M + τζ/2 + τΔᵀWΔ⊗I₃ | as ζ, plus U |

The structure of I_Fᵀ is today's `faceInterpTranspose` (two passes, moment exchange); its formula is new (Voronoi gradient).

### 2.4 Front kernels
- E_s per triangle is a template on the scalar type: `S energy(const S x[3][3], const S n[3][3])`.
  - Gradients use a header-local forward-mode dual number with 18 directions (`KOKKOS_INLINE_FUNCTION` operators, no dependency).
  - Hessian-vector products for F10 use nested duals.
- n(x) and S are explicit: a normalized sum of flat triangle normals, with a hand derivative.
- The mesh-energy term is a **per-point kernel interface**: Q_k(x_k, ring positions, ring slaved normals) with a one-ring stencil, value and gradient through the same dual type. The umbrella (χσ|x̄_k − x_k|²) is one implementation (D14).
- The cost of the 18-direction duals is **to measure**. A hand reverse mode is built only if the front exceeds 10% of step time (P2).

### 2.5 Memory [estimate; measured by P6, target ≤ 6 KB per cell]
| Item | Size per cell |
|---|---|
| state | ~0.2 KB |
| TopologyStore (64 planes, 112 triangles, poke4) | ~1.2 KB |
| two full frames | ~2.2 KB |
| transient view | ~1.5 KB |
| AMG | ~0.4 KB |
| solver vectors | ~0.5 KB |
| **total** | **~6 KB** |

That is about 6 GB per 1e6 cells.

---

## 3. MPI contract

### 3.1 Ownership
- core's `BlockDecomposer` decomposes the domain by ORB.
- A bulk cell belongs to owner(x_i). **A pair cell and its point belong to owner(x_k)** (D7).
- The generator on the far side may lie up to ε/2 outside the block. The ghost width accounts for it.

### 3.2 Ghost layers (by distance) and their invariants
**Cells:** ghosts are all generators within r_cut of an owned generator (VoronoiHalo).
- Invariant, checked every frame: r_cut ≥ 2R_max + ε_max/2 + skin. R_max is the global maximum over owned cells of the maximum vertex distance from the generator.
- If the invariant is violated, regather with a larger r_cut (collective).
- This holds ring 1 of every owned cell, which is what the tessellation and all one-ring operators need.

**Front points:** ghosts are all points within R_f = max(r_cut + ε_max, 2L_edge,max + skin) of an owned point, a second particle halo of the same core classes over points.
- Invariant, checked every frame: every triangle that touches an owned point or its ring has all three vertices present. If not, regather.

### 3.3 Halo depth per operator
| Needed at an owned cell, face or point | Supplied by |
|---|---|
| tessellation of the owned cell | ghost layer r_cut (3.2) |
| G_i, transport, ζ (local part) | ring 1 plus the U exchange |
| I_F on an interface face (G of the ghost) | **G exchange**: computed by the ghost's owner, so no ring 2 is needed |
| I_Fᵀ, C′ᵀ, ζ transpose | moment exchange (T/V or the stress moment) |
| L_ρ SpMV, faceKind, densities | ring 1 plus the scalar exchange |
| front force on an owned point | points in ring ≤ 2 (front halo); triangles touching ring ≤ 1 (triangle exchange); p of cells at unlike faces of ring-1 pairs (ring 1 cells) |
| per-drop volume, start pressure, volume shift | per-drop partials, then `MPI_Allreduce`(nDrops) |

Why one ring plus exchanges (D4) and not two rings: computing G on ring-1 ghosts needs their full
neighbor lists, so ring-1 ghosts would have to be tessellated, which needs ring-2 and ring-3
candidates. The transposed operators (I_Fᵀ, ζ) have stencil radius 4, so a no-exchange halo would
be four rings deep. In 3D at 1e6 cells per rank that is about 25% extra cells, all re-tessellated
every frame [estimate]. Exchanging the intermediates adds latency instead (counted in §3.4). The
front is the opposite case: it is two-dimensional and sparse, so redundant computation over two
front rings is negligible. The front therefore uses the deep halo, needs no reverse communication,
and needs one forward exchange of point positions per force evaluation.

### 3.4 Exchange schedule per step (default composition, §4.1)
| When | What | Size per ghost or entity |
|---|---|---|
| per frame (×2) | x, w of cells (device pack, per-ghost periodic shift stored at regather) | 4 |
| per frame (×2) | **interface-face record** (§3.5) | ~11 doubles plus 2 int64 |
| per frame (×2, S2) | point state (x, n, ε, w, ż); owned triangles to halo neighbors | ~11; 3 int64 |
| per frame | rates ẋ, ẇ; phase, point (S2) | 4; 2 |
| per G / I_F | U (3), G (9) | — |
| per transpose | moments (9) | — |
| per PCG iteration (Laplacian) | x (1), 2 allreduce, 1 allgather(np) for the coarse level | — |
| per PCG iteration (predictor) | U (3), G (9), moments (9), 2 allreduce | — |
| per step | dt (min), μ̂ Rayleigh quotients, per-drop sums (nDrops), energy diagnostics | — |

Overlap of interior work with the exchanges (owned cells split into interior and boundary sets) is a
measured optimization (P5), not part of the first implementation.

### 3.5 Faces on rank boundaries (D5)
Per frame, after the publish:
1. Each rank sends to owner(B) one record per interface face: key (gid_lo, gid_hi), present-flag, and the geometry computed **from the lower-gid side's facet**: A_f, n_f, c_f relative to x_lo, d_f, h_lo.
2. The rank that owns gid_lo is authoritative. Both ranks sort their records by key and merge them.
3. A face exists iff both sides have it; otherwise both drop it, as the serial builder drops non-reciprocal facets.
4. Both sides evaluate every face formula in the lo→hi orientation from bitwise-identical inputs (geometry from the record; A and B cell data from exchanges).

So Φ_f is **bitwise antisymmetric across ranks**, and volume, momentum and phase volume close as in serial. The dropped area is booked into `reciprocityDefect`.

Within a rank, non-reciprocal facets with area > α h² (α = 1e-12) force a rebuild of both cells through the forced worklist (§5.1). This is needed only if gate G0.2 measures such facets; the default is on.

### 3.6 Per-drop scalars and connected components
- Drop labels are **persistent per cell and per point**. Seeding sets them by the inside test of the known shape.
- They are recomputed only at events that can change connectivity: births in S3, and later insertion and removal, coalescence and breakup.
- Recomputation is label propagation of the minimum gid over LIKE faces within phase 1, with halo label exchanges, until a global no-change flag. This is rare, so its latency is acceptable.
- Per step, each rank sums its owned cells per drop through the drop CSR (K5), followed by `MPI_Allreduce(SUM, nDrops)`.
- The drop volume is a **sum over cells**, not a surface integral. This is safe for drops wrapped across periodic boundaries.
- V_d^target is a replicated table.

### 3.7 Migration, regather, rebalancing
**Trigger** (global `MPI_Allreduce(MAX)`; this avoids the deadlock recorded in `distributed_moving.hpp` when ranks disagree):
- an owned generator moved more than skin/2 since the last regather, or
- a §3.2 invariant is violated, or
- the rebalance condition holds (max/mean rank weight > 1.10).

**Collective sequence:**
1. ORB, weighted when rebalancing. Weights: bulk cell 1, pair cell w_pair (default 2, measured), point w_pt.
2. **Migrate the payload**: cells §1.2, points §1.3. A pair moves as one unit with its point. Packing on the device, MPI host-staged.
3. Regather ghosts for cells and points.
4. Rebuild the gid → slot map (sorted gids plus binary search on the device; deterministic).
5. Re-link genMinus/genPlus and point by gid.
6. **Cold-rebuild** the tessellation (today's behavior, correct).
7. Rebuild the drop CSR.

**Warm topology migration** (TopologyStore records migrate with pnbr translated through gids; cells left unresolved go to the forced worklist) is WO-P3. It is built only if regather plus cold rebuild exceeds 15% of the amortized step time.

### 3.8 Determinism and np-equivalence
- At fixed np and backend everything is bitwise reproducible (G-DET).
- Across np the only sources of difference are:
  - global reductions (Krylov round-off): 1e-12 after one step from identical frames;
  - repair schedules that differ with np inside the certificate band: 1e-9 over 200 steps.
- Np-equivalence is a tolerance gate, never bitwise.
- A divergent topology decision needs a round-off-sized perturbation to land on a flip within round-off of zero. The probability is about 1e-13/(CFL·h) per event [estimate], so tolerance gates over 1e4 events are expected to hold.

---

## 4. Solver composition

### 4.1 Step sequence (default; §3.2 of the digest, our sign convention)
| Stage | Frame | Content | Solves |
|---|---|---|---|
| S1 rates | Fn | U_ext = 1.5Uⁿ − 0.5Uⁿ⁻¹ (the first step is done twice, per the derivation); bulk rates (ẋ, ẇ) = MeshRule(Fn, U_ext, I_F U_ext) (Q-MESH); point rates from K_F(U_ext, ż) plus the tangential rule; pair generator rates from point rates; x^{n+½}, x^{n+1} | maybe one mesh-rate Laplacian (Q-MESH) |
| S2 frames | — | repair n → ½ → Fh; repair ½ → 1 → F1; interface exchange each | — |
| S3 transport fluxes | Fh | Φᵐ, then the compact correction to τDΦᵐ = V^{n+1} − V^n; Φ on LIKE faces (Q-FRAME); F = Φ − Φᵐ | L_ρ (Fh) |
| S4 predictor | Fh | (M_L + τζ/2 + τΔᵀWΔ)U* = (M_R − τζ/2)Uⁿ + τ[transport + C′ᵀpⁿ + K_Fᵀf + body] (M_L, M_R per the derivation, Q-MASS) | PCG (cells) |
| S5 projection | F1 | μ̂ update; δp = −ω L_ρ⁺ C″U*/τ, ω = min(1, 1/μ̂); U^{n+1} = U* + τM⁻¹C″ᵀδp; p += δp | L_ρ (F1) plus one loose solve |
| S6 compact flux | F1 | Φ_1 = I_F″ U^{n+1}; L_ρ φ = DΦ_1; Φ_1 corrected; DΦ_1 = 0 to tolerance | L_ρ (F1) |
| S7 front finalize | F1 | per-drop normal shift (Newton ≤ 3, each a front-local repair and publish of F1); labels | — |
| S8 rotate | — | Fn ← F1; Uⁿ⁻¹ ← Uⁿ | — |

The default totals per step are three to four Laplacian solves, one predictor PCG, two frame builds
and two AMG setups. The digest's §6 says "L_ρ twice"; if the derivation merges S3's correction or
the μ̂ solve into another solve, the count drops and nothing in the layout changes (Q-FRAME). The
stages are pure functions of (frames, state), so a pass loop over S1–S5 (F10) wraps them without
restructuring.

### 4.2 Predictor (S4)
- The operator is symmetric positive definite and matrix-free (ζ as in §2.3; ΔᵀWΔ assembled per frame on the cell→face CSR).
- PCG uses a 3×3 block-Jacobi preconditioner from the frame's diagonal blocks, which are SPD. Any SPD diagonal is admissible; the two-point part μ_fA_f/d_f is the default diagonal of ζ.
- When the maximum diffusion number β = τμ_i/(ρ_ih²) exceeds 2, the preconditioner switches to a scalar AggregationAMG per component on M/τ + μ_fA_f/d_f + W. Low Reynolds number and the gas phase at air–water ratio reach this. The switch is automatic and overridable at the diagnostics tier, and changes results only within the solver tolerance.
- The initial guess is U_ext.

### 4.3 Mode F and μ̂ (S5)
- μ̂ = λ_max(C″M⁻¹C″ᵀ, L_ρ) by **one power iteration per step**, warm-started from the persistent μ̂ vector (a cell field, migrated):
  - v ← L_ρ⁺(C″M⁻¹C″ᵀv), with the solve at relative tolerance 1e-2;
  - Rayleigh quotient λ = vᵀC″M⁻¹C″ᵀv / vᵀL_ρv;
  - μ̂ = max(λ, 0.9 μ̂_prev).
- Stability needs ωμ ≤ 4/3, i.e. μ̂ ≥ 0.75μ [derived]. The Rayleigh quotient is a lower bound, so G1.2 gates the tracking error: μ̂ ≥ 0.9 μ_exact.
- The first step uses 20 cold iterations from a hash-seeded ±1 vector (voro register: a well-mixed ±1 seed).
- M⁻¹ is behind an interface `applyInverseMass(r) → z`, default diagonal. F10 replaces it with Ã⁻¹ (§4.7).
- Mode R (PCG on C″M⁻¹C″ᵀ preconditioned by one AggregationAMG V-cycle of L_ρ) exists as the **oracle and the start-pressure solver only**, behind `diagnostics.projection = 'exact'`.

### 4.4 Compact flux projection and swept-flux correction (S3, S6)
- Both are L_ρ solves with absolute stopping tests tied to the invariant:
  - S6: max_i τ|(DΦ_1)_i| ≤ 1e-14 V_i;
  - S3: max_i |τ(DΦᵐ)_i − ΔV_i| ≤ 1e-14 V_i.
- They are warm-started from φ_prev and φᵐ_prev.
- Following flow's register, a pressure-solver change is judged on this divergence contract and never on per-step ratios.

### 4.5 Laplacian solver (`fv/aggregation_amg.hpp` plus `ale::LaplacianSolver`)
**Fine level.** The CSR from the frame (row i = i's face list, ascending face order; coef a_f; diag Σa_f plus Dirichlet terms). It is an SpMV, not a face loop: about 180 B per row against about 100 B per face times 7.5 faces [estimate].

**Local hierarchy** (owned rows; ghost couplings stay in the diagonal, as in today's block-Jacobi), rebuilt on the device **every frame** (D8):
1. **Strength:** only LIKE faces between owned cells, kept if |a_ij| ≥ θ√(d_id_j) with θ = 0.05 (voro register). Unlike faces never join an aggregate. Phases are material in S1 and S2, so aggregates stay aligned with the density jump for any ratio.
2. **Deterministic distance-2 MIS:**
   - priority π_i = splitmix64(gid_i), ties broken by gid;
   - Luby rounds, each two passes for the maximum over the 2-ring of undecided nodes and two passes to mark nodes within distance 2 of new roots;
   - repeat until nothing is undecided.
3. **Aggregates:**
   - each root is an aggregate;
   - each distance-1 node joins its adjacent root of highest priority;
   - each remaining node joins the aggregated strong neighbor with the largest a_ij (ties by priority);
   - nodes left over are singletons.
4. **Galerkin with piecewise-constant P:** the coarse graph Laplacian has off-diagonal (I,J) = Σ_{i∈I, j∈J} a_ij and diag I = Σ_{i∈I} d_i + Σ_{i≠j∈I} a_ij.
   - Assembly: sort the 64-bit keys (I·2³² + J) carrying the fine-entry index.
   - Within each equal-key segment, sort by fine index (an insertion sort, segments ≤ ~20), then sum in that order. This is deterministic.
   - Recurse until n_c ≤ 64 or the coarsening ratio drops below 1.5.
5. **Smoother:** fourth-kind Chebyshev, degree 2, pre and post (voro register: "4th-kind Chebyshev, not colored GS"). λ_max comes from 10 power iterations with a hash-seeded ±1 vector.
   - Coarsest local level: dense Cholesky of A_c, regularized as A_c + (tr A_c/n_c²)·11ᵀ when singular (np = 1 with all-Neumann or periodic).
6. **Global coarse level (np > 1, Nicolaides):**
   - Coarse matrix: E_rs = −Σ a_f over interface faces between ranks r and s; E_rr = Σ over r's interface faces.
   - **Additive:** B = B_loc + Z E⁺ Zᵀ. One `MPI_Allgather` of np residual sums per apply; every rank solves the same E y = s redundantly, mean-zero when singular.
   - E is assembled and factorized densely at regather events and at the first frame, and reused in between. A stale E is still an SPD preconditioner component.
   - Limit: np ≤ 512 (factor about 4.5e7 flops per event). Beyond that, Q-NP.
7. **Outer solver:** PCG with mean removal when singular (today's `PressureSolver` logic), warm start, absolute or relative tests per caller.

Why rebuild every frame and not refresh: flow's register already rejected a numeric-only GraphAMG
refresh (complexity, and a re-setup trigger is needed once interfaces move). Here the setup costs
about 25 face passes against about 200 per solve [estimate; measured by P2], so a refresh would save
roughly 8% of a step at the price of triggers, staleness checks and patch rules for births and
deaths. Why not the host GraphAMG: its setup is host-sequential (`graph_amg_device.hpp:3-8`), which
the device-first rule forbids per frame. Why a coarse level: block-Jacobi with exact local solves
has κ ~ 1/(Hh) [derived], so under weak scaling CG iterations grow as np^{1/3} (×2 at np 8, ×4 at
np 64). The coarse space removes that growth. Unsmoothed aggregation converges more slowly than
smoothed aggregation; G0.4 bounds the penalty at 1.5×. If that misses, try a scaled coarse
correction α ∈ {1.5, 1.8}, which stays symmetric for α < 2; device SpGEMM for smoothed aggregation
is out.

### 4.6 Front force, kinematics, volume shift, start pressure (S2)
**Kinematics map** K_F: (U, ż) ↦ point velocities.
- Normal: from the pair-facet flux, A_k(ṙ_k − a_k·ṅ_k) = (I_F U)_k with ṅ = Sẋ.
- Tangential: P^t ½(U⁻+U⁺) + T_k ż_k.
- The implicit ṅ makes it a front-local linear system. It is solved by Jacobi fixed-point iteration (contraction about |a_k|/R) to 1e-15 relative, at most 50 iterations, and the step fails loudly when the iteration does not converge.
- The adjoint K_Fᵀ is solved the same way on the transposed system.
- Side-facet rows of I_F″ are Φᵐ of the generators, which are functions of the point velocities. Consequently I_F″ = I_F on LIKE and PAIR faces, plus the side rows through K_F.
- **Forces on cells are K_Fᵀ f** (adjoint pairing: work on points equals work on cells). They enter the predictor as one cell force vector.
- If the derivation lags ṅ (Q-KIN), the system becomes explicit and the iteration exits after one sweep.

**Force (D14, update of 2026-11-14).** f is the total derivative of W = E_s + Q, where Q = Σ_k Q_k is a per-point mesh energy. Each Q_k has a **one-ring stencil and may depend on the slaved normals**. The form is **OPEN (F9 mesh energy)**: the umbrella at χ = 1 acts as a physical front elasticity and raises the mode-2 frequency by +18% / +8% at R/h 5 / 7.5, about 4.6(h/R)² [measured 2D, A5u]. The design fixes only the interface:
- `MeshEnergy::value/grad(point k, ring)` evaluated in K3/K4 form;
- the total-derivative assembly f = W_x + SᵀW_n + (B_x + B_nS)ᵀD_Γᵀp (sign translated per §A.3), the same for any Q;
- the stencil of the force on a point is two front rings for every admissible Q (one ring of Q times one ring of S).

**Tangential motion, two variants (both kept):**
1. **Rule-set** (today's 2D scheme): the point slides toward the facet centroid by β a_k per step, β = 0.5. ż ≡ 0, and the slide is a position update in S1/S7.
2. **Rate** ż_k ∈ ℝ²: a front unknown with dissipation η|ż|².
   - It adds a source to the constraint, C″U = −D_Γ(B_x + B_nS)T ż.
   - It adds η|ż|² to the energy budget.
   - Data slot: ż(2) and η per point. K_F already takes ż.
   - If ż is solved together with U, the predictor gains 2P unknowns (a cell–point block system, still SPD with η > 0).
   - The default is variant 1. Variant 2 enters with the decision on F9 or F12.

**Volume shift (S7).**
- One Newton step per drop: s_d = (V_d^target − V_d)/V′_d, where V′_d is the swept-volume derivative of the drop's unlike faces for ẋ_gen = n_k, computed from the Φᵐ formula.
- Then a front-local repair and publish, and a check.
- Iterate while max_d |V_d − V_d^target|/V_d > 1e-14, at most 3 times.

**Start pressure:** a mode-R solve once at initialization, from the acceleration balance (`p0=acc`).

### 4.7 Room for the implicit surface (F10)
| Need | Provision |
|---|---|
| front stiffness K | matrix-free Hessian-vector products of W plus the B-term by nested duals in the K3/K4 kernels |
| Ã = M + (τ²/4)GᵀKG | `applyInverseMass` implemented by an inner MINRES on the 3P point unknowns (K is indefinite tangentially) |
| linearization at the old front | the front state is copied at S1 |
| repeated passes | the stage functions are pure, so the pass loop over S1–S5 rebuilds Fh and F1 per pass |

Each pass costs two frame builds and the solves. The layout does not change.

### 4.8 Time step
τ = min over three limits, each taken as a global MPI minimum:
- the Courant number of F: c_F = τ max_i Σ_f max(s_if F_f, 0)/V_i ≤ 0.5;
- the capillary step: τ ≤ τ0 = 0.3(ρ̄h³/(πσ))^½ with ρ̄ = ½(ρ0+ρ1) (Q-CAP);
- the small-cell limit (F5 open; default none beyond Courant).

`max_stable_dt()` reports the minimum. `set_dt` is the user's choice.

### 4.9 Tolerances (defaults; G1.2 checks that they are adequate)
- Predictor: relative 1e-12.
- Mode F Laplacian: relative 1e-8.
- μ̂: one solve at relative 1e-2.
- Compact and swept-flux corrections: absolute (§4.4).
- The energy-budget diagnostic **includes the residual work of every inexact solve**, so closure is independent of the tolerances. Separately, the residual terms must be ≤ 1e-10 of the dissipation.

---

## 5. Cell birth and death (S3) and the remap contract

### 5.1 Tessellation support (WO-S3.1; the forced worklist lands earlier, in S0.2)
- `MovingTessellation` allocates with **capacity** and gets an `active` mask. Inactive slots are skipped when binning in `buildTessGrid` and in the certificate. With no mask passed, today's code path runs bitwise unchanged.
- `step(pos, forced)`: a device list of slots that the gather rebuilds in Pass 1 whatever their certificate says.
- **Death of g:** set active(g) = 0 and force g's former neighbors, read from its stored topology.
- **Birth of b:** take a free slot (free list per rank) and set active. Force b, the power-nearest cell containing x_b, and that cell's ring. Existing verify passes catch anything missed, gated against a cold rebuild.
- **gid of a newborn:** gidHighWater + `MPI_Exscan`(per-rank count) + the local index in canonical order (sorted by the triggering point's pgid). Then gidHighWater += the global count.

### 5.2 Remap
- Events are processed **at frozen positions** between S6 and S7, in **rounds**: deaths first, then births. Within a round the events are an independent set with no shared ring-1 neighbors; conflicts move to the next round.
- In each round the overlaps come from differences, with no polyhedron intersection:
  - death of g: overlap(g, j) = V_j^new − V_j^old and the first moment m(g, j) = M1_j^new − M1_j^old for each former neighbor j;
  - birth of b: overlap(j, b) = V_j^old − V_j^new and the same for first moments.
- These are exact because only one generator changes in the neighborhood within a round.
- **Contract:** each round yields an overlap list (old slot, new slot, overlap volume, first moment(3)). A remap consumes it:
  - momentum by the constant (conservative) rule, so K does not rise (Jensen);
  - the first moments are carried so that the linear remap (F4) can be added without changing the contract.
- MPI: gather form. The receiving owner computes its new momentum from its own volume change and the dead ghost's ρU, so no reverse communication is needed. Ghost V^old and V^new are exchanged (2), so both sides use the same overlap value.
- An overlap that crosses phases (a dying phase-0 cell giving volume to a phase-1 cell) is a formulation question (Q-REMAP). Default: transfer in the receiver's phase, book the cross-phase mass, and let the S7 volume shift restore the drop.

---

## 6. Reuse, extension, addition, untouched

### 6.1 Reused unchanged
- `ConvexCell`, `Power` plane policy, `TopologyStore`, `buildAuxMaps`, `FaceMesh` (struct), `GhostExchange` (cell fields; also instantiated for points)
- `VoronoiHalo` (topology build at regather until WO-S1.6)
- core `BlockDecomposer`, `ParticleMigrator`/`ParticleHaloTopology`, weighted ORB

### 6.2 Extended (opt-in; defaults bitwise unchanged, proved by `state_hash.py` and all ctests)
| File | Extension | WO |
|---|---|---|
| `fv/mesh.hpp` | `FaceMeshOptions`: deterministic order, gid orientation, generator geometry, published centroids | S0.1 |
| `reeval_tessellation.hpp`, `convex_cell.hpp` | facet centroids, cell first moments | S0.1 |
| `tessellation_view.hpp` | optional `facetCentroid`, `cellCentroid` views (layer 0) | S0.1 |
| `repair.hpp`, `tess_grid.hpp` | `changedMask` export; forced worklist; capacity plus active mask | S0.2, S3.1 |
| `tools/check_include_graph.sh` | `ale/*` → layer 5 | S1.1 |

### 6.3 Added
- `fv/aggregation_amg.hpp` (layer 4, generic). It stays in voro until a second consumer appears, then moves to `core/solver` (Q-AMG).
- `ale/frame.hpp`, `ale/operators.hpp`, `ale/laplacian_solver.hpp`, `ale/stepper.hpp`, `ale/front.hpp`, `ale/front_energy.hpp`, `ale/remap.hpp`, `ale/distributed.hpp` (layer 5).
- One new binding TU `src/voro_ale.cpp` with `bindAle`, following voro's one-TU-per-subsystem rule. **Check that the unity build still works.**
- Tests: `tests/kokkos/test_ale_*.cpp`, `tests/kokkos_mpi/test_ale_mpi.cpp`, `python/test_ale.py`, and the Python 3D operator reference `tests/python/ale_reference.py` (an oracle; host only).

### 6.4 Must not be touched
- Any `m7*` branch or worktree.
- `lagrangian/polygon_transform.hpp`, `lagrangian/operators.hpp` (they exist only on m7; not a dependency).
- `CollocatedNS`, covolume, `FlowSolver`, `Simulation`, `DistributedMovingTessellation` (bitwise and API).
- core (no core change in this plan; Q-AMG, Q-HALO).

---

## 7. Work orders and gates

Every gate runs through a script that prints one summary line and writes its log to a file. Stated
pass criteria are fixed before the run. Meshes:
- 16³, 32³, 64³ periodic, jittered (±0.3h) power seeds;
- weights uniform in [0, 0.15h²] where a gate says "power" (h = (L³/N)^{1/3});
- host OpenMP for gates, with CUDA repeated where marked.

### Stage 0: frame infrastructure (no new numerics)
- **WO-S0.1 Face-mesh options and published centroids.** Gate G0.1:
  - (a) all existing ctests pass and `state_hash.py` is unchanged;
  - (b) on 32³ power meshes:
    - |Σ_f A_f n_f| ≤ 1e-13 A_cell;
    - ‖Σ_f A_f n_f⊗(c_f − x_i) − V_iI‖ ≤ 1e-13 V_i;
    - |h_A + h_B − d| ≤ 2 ulp(d);
    - facet centroid vs a host polygon oracle ≤ 1e-13 h;
    - |n_f × conn|/d ≤ 1e-12 on faces with A_f > 1e-10h²;
  - (c) FaceMesh bitwise identical at OMP 1 vs 8 and across two runs (CUDA: two runs);
  - (d) deterministic order with owner-facet orientation is bitwise equal to the default builder at OMP 1 on `reevalPublish` views.
- **WO-S0.2 Frame pipeline, forced worklist, changed set.** Gate **V0.1 (Eulerian limit, first gate of the campaign)**, then G0.2:
  - V0.1 (a): a repair with zero displacement, then publish and build, reproduces the FaceMesh bitwise for 100 frames, all arrays.
  - V0.1 (b): `CollocatedNS` (the `stepImplicit` and `stepRK3` configurations of `test_collocated_ns`), on 16³ Voronoi cells with a TGV initial field, with the mesh replaced every step by the re-derived one. SHA-256 of (U, p, uf) equals the fixed-mesh run after 100 steps, at OMP 1 (default builder) and OMP 8 (deterministic option).
  - G0.2 runs generators advected by a smooth periodic field for 500 frames with ≥ 1000 topology changes:
    - Σ V_i = L³ to ≤ 1e-13 relative in every frame;
    - the maximum non-reciprocal facet area is **measured** and recorded; with enforcement on it must be ≤ 1e-12 h²;
    - at every 100th frame the repaired tessellation equals a cold rebuild (exact neighbor sets, volumes ≤ 1e-9).
- **WO-S0.3 3D operator reference (Python, dense, from an exported FaceMesh).** Gate G0.3 on three meshes with N ≈ 500:
  - adjointness ≤ 1e-14;
  - linear and rigid exactness ≤ 1e-13;
  - ζ symmetric ≤ 1e-14 and ζ(rigid) ≤ 1e-13;
  - L_ρ symmetric positive semidefinite with null space the constants;
  - swept flux vs a finite-difference ΔV is third order (error ratio ∈ [6, 10] when τ halves).
- **WO-S0.4 AggregationAMG and LaplacianSolver (serial).** Gate G0.4, on 32³ and 64³ power meshes:
  - iterations to relative 1e-10 ≤ 1.5× PCG with host GraphAMG (SA) on the same matrices;
  - growth from 32³ to 64³ ≤ 20%;
  - at density ratio 1000 (sphere R = L/4, phase-respecting aggregation): iterations ≤ 1.5× the ratio-1 count;
  - setup time ≤ one solve;
  - bitwise across OMP 1/8 and across runs.

### Stage 1: one phase, moving mesh (not blocked)
- **WO-S1.1 Bulk operators on the device.** Gate G1.1:
  - vs the S0.3 reference ≤ 1e-13 relative on random fields, three meshes;
  - adjointness ≤ 1e-14;
  - per-face linear and rigid exactness ≤ 1e-13;
  - ζ(rigid) ≤ 1e-13;
  - **V0.3:** on a 16³ cubic lattice with zero weights, I_F/A_f = `projectToFaces`, C′ᵀp = −V⊙`cellGrad`(p), and (DΦ)/V = `divergence`, all to ≤ 1e-14 relative;
  - **G1.1-h:** hydrostatic rest with body force g at ρ = 1000: C′ᵀp + ρVg = 0 to ≤ 1e-13 relative, and |U| ≤ 1e-12 √(gh) over 100 steps. This fixes the sign.
- **WO-S1.2 Static-mesh stepper** (predictor, mode F, μ̂, compact projection, energy-budget diagnostic, mode R oracle). Gate G1.2:
  - 2D Taylor–Green extruded on 3D power meshes and ABC flow (A = B = C = 1), both exact NS solutions, at Re 100: the L2 velocity error at t = 1 has order ≥ 1.8 between 32³ and 64³ (cell Re ≤ 5);
  - energy budget ≤ 1e-13 K per step;
  - mode F stable over 2000 steps at weight spread 0.15h²: K never rises beyond the booked terms, μ̂ ≥ 0.9 μ_exact at 20 sampled steps (μ_exact from 200 cold iterations), ω μ_exact ≤ 4/3;
  - mode F vs mode R at Re 100: ≤ 2% in the error norm (2D: 1.3%).
- **WO-S1.3 Moving mesh** (MeshRule per Q-MESH, frames, swept flux and correction). Gates **V0.2** first, then G1.3:
  - V0.2: the stepper with zero mesh rates is bitwise identical to the static path for 100 steps (Φᵐ ≡ 0, the correction is a no-op, repairs are no-ops).
  - G1.3:
    - uniform flow ≤ 1e-12 after ≥ 1000 topology changes (2D: 4e-13);
    - GCL ≤ 1e-14 V_i per cell per step;
    - energy budget ≤ 1e-13 K;
    - momentum ≤ 1e-13;
    - Taylor–Green and ABC order ≥ 1.8 on the moving mesh at cell Re ≤ 5;
    - stable at weight spread 0.15h².
- **WO-S1.4 MPI stage 1** (frames, interface-face exchange, migration plus cold regather, distributed AMG with the coarse level, distributed μ̂). Gate G1.4, np 1/2/4/8:
  - one step from identical state: owned fields ≤ 1e-12;
  - 200 steps with topology changes: ≤ 1e-9, while conservation per run is as serial (≤ 1e-13);
  - interface Φ bitwise antisymmetric;
  - weak scaling at a fixed 32³ per rank, np 1 → 8: Laplacian iterations grow ≤ 1.2× (without the coarse level expect about 2×: record both);
  - after a regather, the tessellation equals a cold rebuild.
- **WO-S1.5 Python surface, stage 1, and profiling diagnostics.** Gate G1.5:
  - API per §10 checked against NAMING;
  - a `state_hash`-style SHA of a 50-step run is reproducible bitwise;
  - per-stage timers sum to the step within 5%.
- **WO-S1.6 Device regather** (ghost and point selection on the device from block boxes). Gate: bitwise-identical ghost sets vs the host `VoronoiHalo`; regather time is measured (P1). Required before stage 1 closes.

### Stage 2: paired front
- **WO-S2.1 Front data, seeding, pair-atomic migration, front halo, drop labels and CSR** (not blocked). Gate G2.1:
  - a sphere and an ellipsoid (axes 1.5, 1, 1/1.5) seeded from physical center and radius;
  - pairing persists through 50 regathers at np 1–8;
  - every pair's generators sit on one rank;
  - drop CSR volumes = serial ≤ 1e-13.
- **WO-S2.2 Triangles, normals, E_s, mesh-energy interface (umbrella χ = 1 as the oracle's), gradients, B-term; statics driver** (not blocked). Gate G2.2:
  - the triangle list equals the oracle's as a set on the `a2f_front3d.py` configurations;
  - at identical inputs E_s, Q, the normals and all gradients match the oracle to ≤ 1e-12 relative;
  - finite-difference check ≤ 1e-8;
  - started from the oracle's initial configurations, the statics reproduce Laplace +0.68/+0.27/+0.12% (geometric normals), roughness and the pair-facet range to the oracle's printed digits, with rest residual ≤ 1e-12 σh;
  - triangle lists are identical at np 1/2/4.
- **WO-S2.3 Two-phase bulk** (faceKind, faceInvRho, harmonic μ, I_F″/C″, K_F and K_Fᵀ, ż slot), with σ = 0 dynamics (not blocked). Gate G2.3:
  - phase volume per drop ≤ 1e-13 drift over 1000 steps (2D: 4e-15);
  - momentum ≤ 1e-13;
  - pair-constraint residual ≤ 1e-13;
  - hydrostatic rest with gravity at density ratio 1000: |U| ≤ 1e-12 √(gR);
  - np 1/2/4 within the G1.4 tolerances.
- **WO-S2.4 Front force in the predictor, volume shift, start pressure. BLOCKED on F9 (mesh energy).** The 2D dynamic closure gate has passed (rest 3e-14 u_σ, conservation 1e-15, energy never rises, mismatch 1e-5 to 1e-4 of the dissipation). The block is now the form of Q. Gate G2.4, with u_σ = √(σ/(ρR)):
  - rest drop at R/h 8: |U| ≤ 1e-12 u_σ over 1000 steps;
  - K + W never above its start beyond the booked F3 flip steps, which are reported;
  - drop volume and momentum ≤ 1e-13;
  - l = 2 oscillation: frequency within 2% of Lamb's (with the viscous correction) at R/h 8, and the error decreases R/h 5 → 8 → 12;
  - density ratio 1000 with viscosity ratio 55: conservation as above, stable over 2000 steps, frequency within 3%;
  - np 1/2/4 within tolerance.

### Stage 3: stationary bulk
- **WO-S3.1 Active mask, free slots, forced births and deaths, gid allocation** (not blocked). Gate G3.1:
  - after 1e4 random events the repaired tessellation equals a cold rebuild (exact neighbor sets, 1e-9);
  - `bench_dynamic_update --gates` and `state_hash` unchanged.
- **WO-S3.2 Remap contract** (rounds, overlap list, conservative remap) (not blocked). Gate G3.2:
  - Σ overlaps = V_old ≤ 1e-13;
  - first moments ≤ 1e-13;
  - momentum ≤ 1e-13;
  - ΔK ≤ 0 at every event (Jensen) up to 1e-14 K.
- **WO-S3.3 Death at 1.0h and rebirth at 1.25h around the moving front; carried drop. BLOCKED as S2.4.** Gate G3.3:
  - drop volume and momentum ≤ 1e-13 through ≥ 500 events;
  - K does not rise at any event;
  - oscillation frequency within 2% (2D: −0.8% at R/h 7.5).

### Performance (parallel to the stages; numbers in §8)
- WO-P1 profiling harness (in S1.5).
- WO-P2 roofline and weak scaling after S1.4.
- WO-P3 measured options: warm migration, restricted publish (S3), exchange overlap, CG with one reduction per iteration. Each is built only if its trigger in §8 fires.

---

## 8. Performance design against the yardstick (all **to measure** unless derived)

The method needs three to four Laplacian solves, a matrix-free PCG and two re-tessellations per
step, so it cannot match a structured-grid solver per cell-step. The SOTA directive is applied **per
component**:

| Id | Component | Target | Trigger if missed |
|---|---|---|---|
| P1 | per-stage profile at 1e6 cells per GPU and per 16-core CPU rank | recorded, one line per stage | — |
| P2 | Laplacian | iterations ≤ 1.5× SA (host GraphAMG; hypre BoomerAMG where available); setup ≤ one solve; SpMV and Chebyshev at ≥ 50% of the measured STREAM bandwidth of the device | scaled coarse correction; kernel tuning |
| P3 | front kernels | ≤ 10% of the step | hand reverse mode |
| P4 | tessellation per frame (repair + publish + face mesh) | ≤ 30% of the step in S1/S2; in S3 the same | restricted publish (WO-P3) |
| P5 | weak scaling, CPU node, np 1→8 | step efficiency ≥ 80%; Laplacian iteration growth ≤ 1.2× | exchange/compute overlap; coarse variants |
| P6 | memory | ≤ 6 KB per cell | drop the transient view after the face-mesh build (already planned) |
| P7 | regather plus cold rebuild, amortized | ≤ 15% of step time | warm migration (WO-P3) |
| P8 | GPU host syncs (each scalar `parallel_reduce` result fences) | recorded | CG with one reduction per iteration |

Each scalar CG reduction is a host sync on the device backends (Kokkos Programming Guide,
"Parallel dispatch / parallel_reduce": a reduction into a host scalar does not return before the
result is available). About 100–200 per step at about 10 µs each, 1–2 ms per step [estimate] — P8.
Kernel claims (fusion, team vs range policy, layout) are not made here. Per the brief they are
measured in P2 against the Kokkos documentation before they are adopted. `Kokkos::sort`
(Kokkos_Sort.hpp) and execution-space instances are used only where the installed Kokkos version is
confirmed to provide them (check `extern/install/<backend>` first; `sort_by_key` is
version-dependent).

Cost model [estimate]: one predictor PCG iteration ≈ 5 face passes; one Laplacian PCG iteration ≈ 1
SpMV + a V-cycle ≈ 6–8 SpMV. At 1e6 cells per GPU that is about 50–60 ms per Laplacian solve
without warm starts. Warm starts (cells persist, so the previous increment maps to the current cell)
are expected to remove 30–50% [estimate; measured in P2].

---

## 9. Risks and open questions (each with a default; labeled fact or preference)

| Id | Question | Needs | Default (so the work proceeds) |
|---|---|---|---|
| Q-F9 | Form of the mesh energy Q (umbrella at χ = 1 stiffens mode 2 by +18% / +8%) | fact (orchestrator's derivation) | interface §4.6; umbrella as the oracle's reference implementation; **S2.4 and S3.3 blocked** |
| Q-ż | Tangential motion: rule-set slide vs rate ż with η | fact, with F12 | rule-set (variant 1); slot for ż kept |
| Q-OPS | The digest says "this is voro's collocated solver with skewCorrected". It is not: today's kernels use the skew-corrected Green–Gauss gradient with R = (I−S)⁻¹ and distance-weighted Ḡ_f, while the digest uses the Voronoi gradient and ½/½. They agree only on a centroidal lattice. | fact (which formula the 2D oracle `a4` used) | the digest's formula (where the measured evidence lives); V0 is redefined as V0.1/V0.2/V0.3 |
| Q-V0 | Meaning of "bit-identical to the existing collocated solver at zero motion" | preference | V0.1 (existing solver on re-derived frames, bitwise), V0.2 (inertness of motion in the new stepper, bitwise), V0.3 (lattice agreement 1e-14). Whole-step identity is impossible: CN, Rayleigh friction and the upwind split differ |
| Q-MESH | The bulk mesh rule (ẋ, ẇ from U_ext and its fluxes): formula, and whether it needs a solve | fact | brief S1.3 quotes it; a solve, if any, uses the unit-density Laplacian of Fn |
| Q-FRAME | Frame of the transport flux (Fh) and whether S3's correction and the μ̂ solve are separate solves | fact | §4.1 as written (three to four Laplacian solves) |
| Q-MASS | Left and right masses in the predictor on a moving mesh | fact | inputs M_L, M_R; brief quotes them; the uniform-flow gate checks them |
| Q-FRIC | 3D G_f and V_f of the Rayleigh function | fact | G_f = Ḡ_f + ((U_B−U_A)/d − Ḡ_f n)⊗n, V_f = A_f d_f/3; the brief must confirm |
| Q-KIN | ṅ in the pair constraint: slaved (implicit) or lagged | fact | implicit, by fixed-point iteration |
| Q-TRI | 3D rule for which diagram edges give dual triangles (and at four-cell junctions) | fact (a2f oracle) | sorted pair-triple of each edge whose three cells belong to three distinct pairs; gate G2.2 is set equality with the oracle |
| Q-F6 | An unlike bulk|bulk face (clearance breakdown) | preference | fail the step loudly |
| Q-REMAP | Overlaps that cross phases in S3 | fact (a5q) | receiver's phase; book the mass; S7 restores the drop |
| Q-CAP | Density in the capillary step | fact | ρ̄ = ½(ρ0+ρ1) |
| Q-AMG | Where the device AMG lives | preference (release process) | voro `fv/`, moved to core when a second consumer appears |
| Q-HALO | Does core or dem already have a device-resident particle halo and migrator with payload? | fact | if not, write it in `ale/distributed.hpp` on core's block boxes (WO-S1.6) |
| Q-NP | Global coarse level beyond 512 ranks | preference (scope) | out of scope until such a run is planned |
| Q-NAME | Public names (§10) | preference (USER: public names) | as §10 |
| Q-DOC | Place of this note (voro's rule: plans go to `docs/archive/`) | preference | `docs/ale_design.md` while the campaign is live (brief); moved to `docs/archive/` when S3 closes and replaced by a description of the implemented method |

**Risks:**
1. The certificate band may produce non-reciprocal facets far above round-off, which would break uniform flow and phase volume at 1e-8. G0.2 measures this first, and the forced-rebuild enforcement is the mitigation.
2. Unsmoothed aggregation may miss the 1.5× bound (G0.4). Mitigation: the scaled coarse correction.
3. Regather frequency with mesh following the fluid: at CFL 0.5 and skin 2h, about every two steps [estimate]. P7 measures it, and warm migration is the remedy.
4. The 18-direction dual-number front may be expensive (P3).
5. Three to four Laplacian solves per step dominate. Warm starts and Q-FRAME may reduce them.
6. Np-equivalence at 1e-9 relies on rare divergent flips (§3.8). If G1.4 misses, compare integrated quantities and log the first divergent topology event.
7. F3 flip energy steps in 3D (1e-6 to 5e-5 E_s) make "K + W never rises" a booked gate, not a raw one.
8. F12 (first order in τ) and Decision 8 (high cell Re) change only the orchestration and operator choices, not the layout.

**Where the design would change:**
- Free normals (P1) or minimizing normals (P2) would add a front linear solve per force evaluation (and an adjoint solve for P2). Both use the K3/K4 machinery and the inner MINRES of §4.7; no layout change.
- A rate ż solved implicitly turns the predictor into a cell–point block system (§4.6).
- Implicit surface: §4.7.

---

## 10. Public Python surface (physical inputs only; names per `docs/NAMING.md`)

```python
s = peclet.voro.ALEFlowSolver(positions, extent, *, weights=None, comm=None)   # (N,3) m; m; m²; mpi4py comm
s.set_phases(density=(rho0, rho1), dynamic_viscosity=(mu0, mu1))  # kg/m³, Pa·s; one phase: 1-tuples
s.set_surface_tension(sigma)                     # N/m (flow's name)
s.set_body_force((fx, fy, fz))                   # same meaning as FlowSolver.set_body_force
s.set_mesh_motion('fluid' | 'stationary')        # validated string; 'stationary' = stage 3
d = s.add_drop(center=(x, y, z), radius=R)       # phase 1 inside; front spacing derived from the cells
s.set_velocities(U)                              # (N,3) m/s, copied in
s.set_dt(dt); s.step(n=1); s.max_stable_dt()
# properties: time, dt, num_cells, num_faces, num_front_points, num_drops, front_spacing (derived, read-only)
# copies: get_positions(), get_weights(), get_velocities(), get_pressures(), get_volumes(), get_phases(),
#         get_front_positions(), get_front_normals(), get_front_triangles()
# computed: kinetic_energy(), surface_energy(), momentum(), drop_volumes()
# diagnostics tier: projection ('damped' | 'exact' = oracle), mu_estimate, pressure_iterations,
#   predictor_iterations, topology_changes, junction_switches, triangle_flips, reciprocity_defect,
#   energy_budget(), set_profile(True), timings()
```

- No public parameter takes a cell size, ε, h, β, χ or a tolerance. The resolution is the number of generators; ε = h is derived.
- Mesh-energy choices (F9) and ż/η enter at the diagnostics tier until the decision on F9 is recorded.
- `dynamic_viscosity` is chosen because `FlowSolver(viscosity=)` is kinematic, and the canon allows one spelling per concept (Q-NAME).
- New names are added to NAMING.md §2 when they land.

**Register entries to add when each lands** (`docs/decisions/voro.md`): D1, D3, D4, D5, D6, D8 (with the flow rejection of numeric-only refresh cited as consistent), D9, D11, D12, D13, D14, and the V0 interpretation. Each records its rejected alternative from §0.

---

## 11. Amendments after phase A (orchestrator; each one replaces the text it names)

### A1 (2026-11-14): centroids and the weighted publish (replaces §1.5 item 4 and the rows of §6.2 for `convex_cell.hpp`, `reeval_tessellation.hpp`, `tessellation_view.hpp` under S0.1)
Facts found when stage 0 started:
- `ConvexCell::reevalGeometry` re-evaluates planes and vertices only. The facet geometry is `facetGeometry`
  (area vector, dV, connector), called in the fill kernel of `reevalPublish`.
- For every plane policy the published connector is r = 2·foot and dV = (|A|/|r|)(r − c), with c the facet
  centroid relative to the owner seed. So the builder's present formula c = conn − dV·|conn|/|A| gives the
  centroid for power cells as well. What is Voronoi-only in today's builder is d = |conn| and h_B = d − h_A.
- On main `reevalPublish` was Voronoi-only. Commit 0516c00 (cherry-pick of 4219392, session `voronoi`) adds the
  plane policy and `weight`; the Voronoi default is byte-identical.

Decision: S0.1 changes `fv/mesh.hpp` only.
- The facet centroid stays the builder's formula. It is verified against a host polygon oracle on power meshes
  (gate G0.1 (b)). Only if that fails by more than 1e-13 h does S0.1 publish centroids from
  `facetMomentsPerVertex` through `reevalPublish`; that step needs a new decision first.
- The cell centroid is computed in the builder from the faces of the cell (pyramids with apex at the seed:
  V_i (c_i − x_i) = ¼ Σ_f h_f A_f (c_f − x_i), h_f = n_f·(c_f − x_i) with n_f outward of cell i) and stored in an
  optional `FaceMesh::cellCentroid` (3·N, relative to the seed), allocated only on request.
- No view is added to `TessellationView`; nothing is added to `convex_cell.hpp` or `reeval_tessellation.hpp`.

### A2 (2026-11-14): inputs of the builder options
`FaceMeshOptions` carries: `deterministicOrder` (bool), `gidOrientation` (bool), `generatorGeometry` (bool),
`cellCentroids` (bool), `gid` (int64 per combined cell; empty means the slot index), `pos` (3 per combined cell),
`weight` (per combined cell; empty means zero), `L[3]` (box, for the minimum image). With `generatorGeometry` the
face normal points from A to B (n_f·conn > 0), and `faceCentroid` is relative to the seed of A whichever cell
emitted the face (emitter B: c − x_A = (c − x_B) + conn). All options off reproduces today's arrays bitwise.

### A3 (2026-11-14): multigrid (replaces "no core change" of §6.4 for S0.4 only; to be written out with that work order)
Core branch `amgpar` has a deterministic device setup (MIS-2 aggregation). S0.4 is re-scoped to a phase-respecting
option on it (strength graph cut at unlike faces) after agreement with session `voronoi`; no second multigrid in
`fv/` unless that fails.

### A4 (2026-11-14): the face normal with generator geometry; what gate G0.1 can ask of the builder
First run of G0.1 (e7fe644, bbd9501, a91807b): (a) and (d) pass; all identities of (b) pass (closure 3e-15,
moment 1.4e-14, h_A + h_B − d 1 ulp, facet and cell centroid 4e-16 / 2e-15 h) except |n_f × conn|/d: 1.5e-7 on
faces with A_f > 1e-10 h² (1.2e-13 for A_f > 1e-2 h²): round-off of the published polygon normal of sliver
facets. (c) holds on the power mesh (32 arrays) and misses on the Voronoi mesh, because the published Voronoi
view itself differs between two runs at 8 threads (facets permuted within cells, volumes at 1e-16).

Decision:
- With `generatorGeometry` the face normal **is** conn/d (radical planes are perpendicular to the connector); the
  published area vector gives the magnitude A_f only. The angle between the published normal and conn stays a
  reported diagnostic (`maxNormalConnSine`), with no threshold on sliver faces.
- Gate G0.1 is recorded as FAIL for (b, sine) and (c, Voronoi). Its re-run after this change is G0.1′: (a), (d)
  unchanged; (b′) the same identities with n_f = conn/d, and the published-normal sine ≤ 1e-12 on faces with
  A_f > 1e-2 h² only; (c′) builder output bitwise equal for two builds from the **same** view in one process at 8
  threads (Voronoi and power), and bitwise equal across OMP 1 / 8 / 8 on the power mesh. The reproducibility of
  the Voronoi view across runs is not a property of the builder; it is reported and passed to session `voronoi`.
- The ALE solver publishes through the Power policy only (zero weights included); its reproducibility under
  repair is gated in S0.2.

### A5 (2026-11-14): what gate G0.2 found, and its re-run G0.2′
First run (fcf8a6b; logs `build_s0/gate_logs/S0_2.*`): V0.1 (a) and (b), F and base PASS. G0.2 FAIL on (i), (iii),
(iv), (v), from three separate causes:
1. **Empty power cells.** With fixed random weights (up to 0.15 h²) and generators carried 25 h by the flow,
   generators come close enough that a power cell is empty (from frame 152). The cold build then loses volume
   (3.5e-5 to 1.9e-3) and leaves one-sided facets up to 2 h² toward the empty cell; forcing cells cannot remove
   them. This is a defect of the tessellation for empty power cells (passed to session `voronoi`) and a case the
   stage 0–2 solver must not enter: its cells keep equal weights unless a later stage changes that with its own
   gate. Until the frame pipeline has a rule for it, a frame with an empty cell is an error that `advanceFrame`
   reports (count of cells with zero volume or no facet).
2. **Degenerate facets.** Repair and cold build differ only in facets of area ≤ 2.4e-10 h² (volumes agree to
   1.5e-12). "Exact neighbor sets" is not a property a certificate with a tolerance band can have.
3. **The repair depends on the thread schedule.** The set of rebuilt cells differs between 1 and 8 threads from
   frame 18 and between two runs at 8 threads at frame 452; volumes and face order then differ. §3.8 "bitwise at
   fixed np and backend" does not hold for the repair engine as it is with more than one thread.

G0.2′ (criteria fixed before the re-run): variant Z, power engine with all weights zero; variant W, weights as
before, evaluated only up to the last frame without an empty cell (the first frame with one is reported).
(i) Σ V = L³ to 1e-13 in every evaluated frame; (ii) ≥ 1000 rebuilt cells; (iii) with enforcement on, max
non-reciprocal facet area ≤ 1e-12 h²; (iv′) against a cold rebuild every 100 frames: volumes ≤ 1e-9 relative, and
every facet present in only one of the two has area ≤ 1e-8 h² (the largest such area is reported); (v′) two runs at
one thread are bitwise identical in every FaceMesh array of the last frame; between 1 and 8 threads and between
two runs at 8 threads the cell volumes of the last frame agree to 1e-9 relative (reported: the largest difference
and the first frame at which the rebuilt counts differ). §3.8 is amended accordingly: bitwise reproducibility of a
moving mesh is claimed at one thread only, until the repair engine is made independent of the schedule.

### A6 (2026-11-14): what gate G0.3 found; the pressure force of a linear pressure; hydrostatic balance
First run of G0.3 (61541cf; `build_s0/gate_logs/G0_3.py.log`): R0–R4 PASS (matrix against matrix-free 2e-15,
adjointness 1e-15, linear exactness 3e-14, ζ(rigid) 6e-16, L_ρ symmetric with null space the constants at density
ratio 1 and 1000). Two criteria of the brief were wrong:
- **R5** asked C′ᵀp = −V_i ∇p per cell for a linear p. That is not a property of these operators: the per-cell
  defect is 0.3–0.6 on jittered meshes, in 3D and in the 2D oracle alike (`G0_3.r5_2d.log`), while the sum over
  cells is exact (1e-15). C′ᵀ is the adjoint of a consistent divergence and is consistent only in the weak sense.
  R5′: the sum over cells equals −(Σ V_i) ∇p to 1e-13; the per-cell defect is reported.
- **R6** compared τDΦᵐ of the old mesh with V(τ) − V(0), which is second order by construction (ratios 4.0).
  R6′: against the central difference (V(τ) − V(−τ))/2, i.e. the swept flux of the mid mesh as the step uses it:
  ratios 6.8–8.2 on the three meshes, third order.
**Consequence for stage 1:** a body force ρ V_i g applied per cell cannot be balanced by a pressure through C′ᵀ on
an irregular mesh, so gate G1.1-h as written (C′ᵀp + ρVg = 0 to 1e-13) cannot pass. The body force has to enter
through the same adjoint map as the pressure (a face quantity under I_Fᵀ), inside the predictor. Its form with a
density jump is an open formulation question (F13 in the campaign state); G1.1-h is rewritten when that is settled.

### A7 (2026-11-14): the body force (replaces the body-force term of §4.2 and gate G1.1-h of WO-S1.1)
Derivation D4 (campaign file `briefs/ALE_D4.md`, checked independently and measured in the 2D oracle): a force
given as cell values with a gradient part leaves 18–21 % of itself after the projection on power cells with
weights (3 % at zero weights), independent of h. Rule:
- Write the body force per mass as g = ḡ − ∇ψ_per, with ḡ the mean part along periodic directions and ψ_per
  single-valued (in a box closed along g, all of the gravity potential is ψ_per).
- F = ρ_iV_iḡ (cell values) + I_Fᵀγ, γ_f = ρ_A[ψ_per(x_A) − ψ_per(c_f)] + ρ_B[ψ_per(c_f) − ψ_per(x_B)], inside the predictor.
- Potential energy for the budget: Σ_i ρ_i ∫_cell ψ dV (for uniform g: the cell centroids). With two phases the
  budget has a defect on turning unlike facets (not yet measured).
- **G1.1-h, rewritten:** (i) periodic box, ψ_per = cos(2πy/L) at ρ = 1000, face form: C′ᵀp + F = 0 to 1e-13 with
  p_i = −ρψ(x_i), and |U| ≤ 1e-12 (|∇ψ|h)^½ over 100 steps on a jittered mesh with weights; (ii) uniform ḡ along a
  periodic direction, cell form, from a Taylor–Green start: the velocity equals that of the run without ḡ plus ḡt
  to 1e-12 relative over 100 steps; (iii) control, reported not gated: (i) with the force as cell values (the
  expected drift is the D4 number). Walls and the hydrostatic column in a closed box come with the wall work order.

### A8 (2026-11-14): the mesh energy of the front (lifts the block on WO-S2.4 and WO-S3.3; Frank's decision, desk card `ale-mesh-energy-2`)
The mesh energy is the full umbrella Q = χσ Σ_k |x̄_k − x_k|² (x̄_k the mean of the dual-triangle neighbors of point
k), with χ = 0.7 in 3D (0.1 in the 2D oracle), normals the geometric normals of the flat dual triangles, force the
total derivative (D2). It stays a pluggable per-point energy (§4.6). Booked error, second order in h/R: l = 2
frequency about +2.3 % at R/h 5 and +1.1 % at R/h 8 (stiffness for normal point motion +4.7 / +2.2 %), Laplace
pressure +0.28 / +0.12 %. Rejected by their gates (campaign log, 2026-11-14: A2i, R1, A2j, A2k): the umbrella along
the surface only (no rest state: an l = 1 slide of the points), a weaker full umbrella (χ ≤ 0.5), an anchor of the
surface centroid. No further work on the mesh energy until 3D runs in time exist; the gates of WO-S2.4 take their
thresholds from the booked error above, not from the 2D scheme.

### A9 (2026-11-15): what WO-S1.1 can gate of the body force; scope of WO-S1.1 (replaces G1.1-h of A7 for S1.1 and the last two rows of §2.3 for S1.1)
Gate G1.1-h of A7 asks for runs of 100 steps, and WO-S1.1 has no stepper. Decision:
- WO-S1.1 gates the static part: (H1) ψ = cos(2πy/L) at ρ = 1000 with p_i = −ρψ(x_i): C′ᵀp + I_Fᵀγ = 0 to 1e-13 of
  max|I_Fᵀγ| on the three meshes; (H2) uniform g through differences: Σ_i (I_Fᵀγ)_i = ρ(Σ_i V_i)g to 1e-13; (H3) the cell
  form ρ_iV_iḡ. With one phase γ = −ρΔ_nψ, so H1 holds by construction and checks the implementation; H2 fixes the
  physical sign.
- The time parts of A7 ((i) |U| over 100 steps, (ii) free fall from a Taylor–Green start, (iii) the control with
  cell values) move unchanged to gate G1.2 of WO-S1.2.
- WO-S1.1 covers D, Δ_n, G, I_F, I_Fᵀ, C′, C′ᵀ, L_ρ (assembly and product), Φᵐ, ζ (μ_f a face input) and the body
  force. The transport tendency and the predictor operator move to WO-S1.2: the reference
  (`tests/python/ale_reference.py`) does not have them, and it gets them first.
- L_ρ is assembled as a struct of views in `ale/operators.hpp` (row i = the faces of cell i in ascending face
  order, one entry per face, diagonal Σ a_f).

### A10 (2026-11-16): the multigrid does not respect the phases (replaces item 1 of §4.5 "Unlike faces never join an aggregate", A3, and gate G0.4)
Gate G0.4 was run on a branch of core `amgpar` (`ale-amg`, local: 6f6c6d0, 3994738, 62155c6) with a per-node label
that cuts the strength graph and the smoothing filter. Operator: L_ρ of mesh M1 at 32³ and 64³ (zero weights),
sphere R = L/4 with ρ = 1 inside and 1000 outside; PCG to 1e-10, registered parameters. Logs
`build_s0/gate_logs/G0_4.*`.
- The option does what it says (no aggregate and no prolongator entry across labels on any level; inert with one
  label; default path byte-identical; hierarchy identical at 1, 2, 4, 8 threads; setup 0.17 s against 0.22 s per
  solve at 64³).
- It does not help. Iterations at 32³ / 64³: ratio 1 without label 12 / 15; ratio 1000 without label 17 / 26;
  ratio 1000 with label 19 / 29 (plain cut: 19 / 33, with 3298 cells left as singletons); ratio 1 with the label
  15 / 26. **G0.4 MISS** on growth with size (1.53 and 1.73 against 1.2) and on the device criterion.
- Reading [estimate]: with 1/ρ_f = (h_B/ρ_A + h_A/ρ_B)/d the interface face is the strongest connection of a heavy
  cell, so the algebraically smooth error is continuous across the interface and a transfer cut there removes
  what the coarse space needs.
Decision: `ale::LaplacianSolver` uses the MIS-2 setup of core `amgpar` without a label. No `fv/aggregation_amg.hpp`
and no core change for the ALE solver. Open, carried to the performance gates: the unlabeled count grows from
32³ to 64³ by 1.25 (ratio 1) and 1.53 (ratio 1000); on CUDA the setup of amgpar at b21d4bd asks for 31.5 GiB at
32³ and its level check against the host order is not bitwise (both passed to session `voronoi`).

### A11 (2026-11-17): the initial frame is a cold build followed by one repair step at rest (gate G1.1 K8, work order S1.1b)
- Finding [measured, 16³, `build_s0/gate_logs/S1_1b.gpu_diag_r{1,2}.log`]: on CUDA the frame at zero motion was not
  bitwise. After a cold `rebuild()` the engine's `vol` comes from the clipping kernel of the tessellator; after any
  `step()` it comes from the re-evaluation in the certificate kernel. nvcc contracts a·b + c into fused
  multiply-adds differently in the two kernels: `cellVolume` differs in 1106 of 4096 cells (Voronoi) and 1118 of
  4096 (power), by at most 6 ulp (7.1e-16 relative); `cellCentroid` follows because it is divided by the volume;
  the other 14 FaceMesh arrays are bitwise. Frames 2–100 equal frame 1 (a fixed point after one step), and two GPU
  runs are identical. With `--fmad=false` the defect is gone (`S1_1b.gpu_nofma_*.log`); GCC with
  `-ffp-contract=fast -march=native` contracts both kernels alike (`S1_1b.hostfma_*.log`).
- Rule: `ale::initialFrame` calls `engine.step(pos)` once after `engine.rebuild(pos)`. The step rebuilds nothing;
  it leaves in the engine the volume of the kernel every later step evaluates, so the frames at rest equal the
  initial frame bitwise on every backend (V0.1). No compiler flag, no change to the repair engine. Cost: one
  certificate pass, once per run.
- Rejected: (1) the same re-evaluation at the end of `MovingTessellation::rebuild()` (patch tested in a scratch
  copy, `S1_1b.repair_patch.diff`, passes the same gates): it charges every cold build of every user of the engine
  and needs guards for SDF engines and empty cells, for a property (repair = cold, bitwise) that the engine's own
  contract does not require. (2) Products that cannot be contracted in `convex_cell.hpp`: touches the clipping hot
  loop. (3) `--fmad=false` in the build: a flag that changes results for every user.
- Gates after the change (`build_s0/gate_logs/S1_1c.*`): G1.1 PASS with K8 = PASS (GPU: `test_ale_frame` and
  `test_ale_operators` pass, export bitwise, G0.3′ PASS, K1 6.13e-16); K0–K7, G0.3′ and the state hash unchanged
  to the digit on the host; base 46/46; the S0.2′ line unchanged (V0.1a 8/8, V0.1b 4/4, F PASS; the recorded miss
  without enforcement stays). On the GPU V0.1 (a) and (b) pass at 100 frames / steps for both engines.
- Open: a frame that follows a cold build inside `advanceFrame` (rebuild gate, fallback) still holds the
  clipping-kernel volume, so on CUDA the next frame at rest may differ from it by a few ulp once. It occurs only
  after motion and is not part of the Eulerian limit; S0.5 decides whether those frames get the same step. With an
  SDF engine or an empty power cell the step at rest is not established to be inert (S1.1b report, "Not done").

### A12 (2026-11-17): a frame is reciprocal or an error (work order S0.5; replaces the trigger of §3.5 and criterion (iii) of G0.2′, closes the open item of A11)
- **Trigger:** every one-sided facet with area > 0 forces a rebuild of both cells (was: area > 1e-12 h²). At most
  two rounds, as before.
- **Error of the frame:** `MeshFrame::numOneSidedFacets` counts the one-sided facets above 1e-14 h² that remain after
  the rounds; the contract is that of `numEmptyCells`. The bar: a dropped facet of area a leaves a volume-rate
  defect of up to a/h² of V_i per step at Courant number 1 [derived], and the geometric conservation bar of
  stage 1 is 1e-14 V_i.
- **Weights:** the frame snapshots the weights of the engine; `initialFrame` and `advanceFrame` take no weights.
- **Step at rest after a cold build inside `advanceFrame`** (rebuild gate, fallback, a reciprocity round): the
  same step as in `initialFrame` (A11), so every published frame holds the volume of the certificate kernel.
  Known side effect, accepted: the step at rest builds the adjacency that the rebuild gate skips, so the next
  step uses the local certificate where it used the brute one, and its count `pass1Raw` changes (1407 → 1400 on
  the host, → 1399 on the GPU; power 1394 → 1393); route, changed set and volumes are bitwise the same. Rejected:
  switching the local certificate off around the step at rest (untested, and it moves the change of kernel to a
  later frame on CUDA).
- **Gate G0.5** (`build_s0/gate_logs/S0_5.*`, head 174461d): H1 PASS (zero weights, 32³, 500 frames, 1 and 8
  threads: no one-sided facet left, largest area left 0, one round always enough); H3 PASS (`S0.2' PASS`, the
  runs with enforcement gate (iv′), the runs without are printed as INFO); H4 PASS (V0.1, F, 46/46, G0.3′);
  H6 PASS (frame after a cold build = the next two frames at rest, bitwise in all 16 arrays, host and GPU;
  `G1.1 PASS`, K0–K7 unchanged to the digit); H5: 146 ms per frame at 32³, 8 threads (171 ms before at the same
  load), 8.4 cells rebuilt per frame by forcing (max 26), never a second round.
- **H2 MISS, open:** the face mesh of a repaired frame does not close: |Σ_f s_if A_f n_f| > 1e-13 Σ_f A_f in up to
  184 of 32768 cells in 499 of 500 frames, largest 6.3e-5 (weights: 3.7e-5), with and without enforcement. Each
  cell fails on its own published facets too (2.7e-5); a cold frame at the same positions closes to 9e-15; the
  cell volumes agree with the cold build to 1e-10. So it is not a one-sided facet. Uniform flow is then not
  divergence-free on a moving mesh (C′e = e·Σ_f s_if A_f n_f), which gates G1.3 forbid at 1e-12: **S1.3 is blocked
  on this; S1.2 (static mesh, frames at rest) is not.** Diagnosis: work order S0.6.

### A13 (2026-11-17): the free-fall gate on a static mesh (replaces (ii) of gate G1.1-h in A7 for S1.2; (ii) itself moves to G1.3)
A7 (ii) asks that a Taylor–Green start under a uniform ḡ equals the run without ḡ plus ḡt to 1e-12. On a static
mesh that cannot hold: the fall carries the pattern through the mesh, and the discrete derivative of the pattern
along ḡ is not zero on an irregular mesh, even where the continuous one is [derived]. On a static mesh the exact
statements are (ii′): from rest the velocity is ḡt to 1e-12 relative, and from the Taylor–Green start the mean
velocity Σ M U/Σ M is ḡt to 1e-12 relative (transport, friction and pressure conserve momentum). The original
(ii) is a statement of Galilean invariance and is gated in S1.3 with a mesh that falls with ḡt.

### A14 (2026-11-17): a frame closes or it is an error (work order S0.6; closes H2 of gate G0.5 in A12, extends §3.5)
**Finding (gate G0.6, logs `build_s0/gate_logs/S0_6.*`, all measured).** The repair keeps the topology of a cell
when its certificate holds within 1e-4 h. A kept cell can then carry a vertex 4.5e-5–8.6e-5 h outside one of its
planes: an inverted short edge, seen as two reflex corners on two facets. The facet's vertices are ordered by
angle around the facet centroid (`faceOrderedIdx`); for such a polygon that order is not the order of the
dual-triangle adjacency, and the published area vector is wrong to first order in the tolerance (up to 6.3e-5 of
the cell surface at 32³; 2.89e-5 / 7.79e-7 / 1.41e-7 at tolerance 1e-4 / 1e-5 / 1e-6 h at 16³), while the per-vertex
volume is right to 1e-10. With the adjacency order the kept cell still disagrees with its neighbours' facets to
second order (2.5e-9 at 16³), so a reorder alone does not reach 1e-13. Refuted, 0 cases each: a facet skipped as
degenerate, a vertex evaluated twice, stale positions. voro main b93f0e5 does not change it. The GPU shows the
same numbers.

**Rule.** `publish()` tests every cell i: |Σ_g A⃗_g| + Σ_g |A⃗_g + A⃗_recip(g)| > 1e-13 Σ_g |A⃗_g| (its own closure plus
the mismatch with the other side of each facet: the bound of its closure in the face mesh). With
`FrameOptions::enforceClosure` (default on, independent of `enforceReciprocity`) such cells are forced in the
rounds of A12 and rebuilt from their clip. `ReciprocityReport::numOpenCells` counts the cells above the bar after
the rounds. **`numOpenCells > 0` after two rounds is an error of the frame**, as `numOneSidedFacets > 0` is: the
time step must not use that frame. The repair engine, its route and the cold build are unchanged; the bar is not
a tolerance of the repair and changing the certificate tolerance is not the fix.

**Numbers (32³, 500 frames, 8 threads, zero weights and weights 0.15 h²).** No cell above 1e-13 in any frame,
with reciprocity enforcement on and off; largest closure 5.03e-14 (zero weights), 1.05e-14 (weights); volumes
against the cold build 8.6e-16 (1.1e-10 before). 114 cells forced per frame, a second round in 275 of 500
frames, never a third. `S0.2' PASS`, `G0.3' PASS`, `G1.1 PASS` with K0–K7 unchanged to the character; the frame
at rest is unchanged bitwise (no round at rest). GPU at 16³: 0 frames above the bar, largest 9.9e-15.

**Open.** A timing on a quiet host (149.6 ms per frame against 146 ms, measured at different times; +10 to +22 %
in two same-load pairs at load 35–45); it is taken with the cost number of S1.3. The root (a kept cell whose
facet polygon is not convex) is in the repair and in `convex_cell.hpp`, which are session `voronoi`'s code; it
has the finding. Across ranks the forcing has the same gap as A12's (review F5), before S1.4.

### A15 (2026-11-17): the reference time step on a static mesh (work order S1.2a, gate G1.2r; fixes what S1.2b is checked against)
**What exists.** `tests/python/ale_reference.py` (appended: transport matrix T, upwind part Q = ΔᵀWΔ, R − Δ, centroid
correction δ, predictor operator, each as a matrix and as a face loop), `tests/python/ale_reference_stepper.py`
(the step of §4.1 for one phase on a static mesh, modes F and R, the energy budget), `tools/ale_gates/G1_2r.sh`.
Meshes Z = M1 (jitter ±0.3 h, zero weights), W = M2 (weights uniform in [0, 0.15 h²]), 8³–32³ (`--cells n --mesh`).

**Bindings found while writing it (they hold for the device stepper):**
- The predictor operator is not symmetric (T is skew): a Krylov method for nonsymmetric systems. The reference
  uses GMRES with a 3×3 block-Jacobi preconditioner from U_ext: 9.4 iterations (mean) at 32³.
- T is the skew form ½ Σ_f s_if ρF_f U_o; its energy term B2 is zero to rounding because DF = 0 after the compact
  projection. The diagonal ½(DF)U of the 2D oracle is not carried.
- **An inexact predictor solve changes momentum by the momentum of its residual.** That is booked, as §4.9 books
  the residual work: the conservation gate is on the momentum with the residual's part booked. Measured: drift
  2.2e-13 of Σ M|U| at relative tolerance 1e-13, 2.2e-14 at 1e-14, 6e-17 with the residual booked.
- Momentum is judged as the drift from its start value: a field sampled at the generators of an irregular mesh
  has Σ M U ≠ 0 (9e-5 for Taylor–Green at 16³).
- τ = 1/ceil(c₁/0.4), c₁ the flux Courant number of the start field at τ = 1.

**Gate G1.2r as fixed before the run: FAIL (R2, R4, R6 (ii′) miss).** Logs `build_s0/gate_logs/S1_2a.*`, all measured.
| id | result |
|---|---|
| R0 PASS | rest exactly 0; uniform flow 2.9e-14 |
| R1 PASS | budget closure 2.4e-17 K (Z and W 16³, modes F and R) |
| R2 MISS | momentum drift 2.2e-13 against 1e-13 (see above: the solve tolerance tested at its own size); τ\|DΦ₁\|/V 1.4e-16 |
| R3 PASS | 2000 steps, K never rises beyond the unsigned terms (−9.6e-8 K); μ̂ ≥ 0.972 μ_exact; ω μ_exact ≤ 0.9999; μ_exact Z 0.898 / 0.986 / 1.009, W 0.920 / 1.000 / 1.022 at 8³ / 16³ / 32³ |
| R4 MISS at 16³ | error of mode F against mode R: 3.4 % at 16³ (bound 2 %), 1.1 % at 32³ |
| R5 PASS | Taylor–Green, Re 100, Z: 1.71e-1 / 3.33e-2 / 1.03e-2, order 2.36, 1.69; ABC, Z: 1.92e-1 / 3.19e-2 / 5.59e-3, order 2.59, 2.51. W: TG 1.72e-1 / 3.53e-2 / 1.07e-2 (1.72), ABC 1.92e-1 / 3.57e-2 / 6.70e-3 (2.41) |
| R6 | (i) hydrostatic, face form: 2.5e-15 PASS. (ii′) fall from rest 1.12e-12 against 1e-12 MISS (1.4e-13 at tolerance 1e-14, 7e-15 with a direct solve); mean velocity from the Taylor–Green start 5.0e-14 PASS. Reported: Taylor–Green plus ḡ against the run without ḡ plus ḡt differs by 7.4e-2 on a static mesh (A13); the same potential force given as cell values leaves 2.6e-2 of the free acceleration (D4). |
| R7 | 32³, one thread: 1.30 s per step in mode F, 12.2 s in mode R (99 CG iterations at every resolution) |

**Corrected criteria (orchestrator, fixed 2026-11-17 before S1.2b; R2 and R6 (ii′) were wrong criteria, shown by
the controls; R4 stays a recorded miss).** R2′: momentum drift with the residual's momentum booked ≤ 1e-13 Σ M|U|
(measured 6e-17). R6 (ii′)′, fall from rest: |U − ḡt|/(|ḡ|t) ≤ 20 × the predictor's relative tolerance after 100
steps (measured 11 × at 1e-13, 14 × at 1e-14). R4: reported at 16³ and 32³, bound 2 % at 32³ only.

**What the numbers say.** At Re 100 (cell Reynolds 6 → 3) the 3D step is between order 1.7 (Taylor–Green) and 2.5
(ABC), with and without weights; weights cost 4 % (Taylor–Green) and 20 % (ABC) of accuracy at 32³. The first
order with weights of decision 8 is a statement at cell Reynolds 80 and is not tested here. Mode F is 9 times
cheaper per step than mode R at 32³ and differs from it by 1 % of the error there.

**Open for S1.2b.** The reference estimates μ̂ with the exact L_ρ solve; §4.3's estimate with a 1e-2 solve is
tested on the device. The form without the explicit upwind part was not run (the stop rule did not fire).

### A16 (2026-11-18): the device time step on a static mesh (work orders S1.2b, S1.2c; gate G1.2; replaces §4.2 "PCG" and the solver file of §4.5)
**What exists.** `ale/transport.hpp` (T in skew form, Q, R − Δ, δ, explicit transport), `ale/krylov.hpp` (device PCG
and right-preconditioned GMRES, matrix-free), `ale/laplacian_solver.hpp` (`LaplacianSolver`: the matrix of
`assembleLaplacian` handed to core's `GraphAMGDevice` with the MIS-2 device setup and no label, PCG with mean
removal, warm start, relative or absolute test), `ale/stepper.hpp` (stages S1–S6 as functions of frame and state,
modes F and R, the budget of energy and momentum with the residuals booked, one pair `applyMass` /
`applyInverseMass`), `tests/kokkos/test_ale_stepper.cpp` with `tests/python/test_ale_stepper.py`,
`tools/ale_gates/G1_2.sh`. The reference got a `trace` subcommand; its judge applies R2′, R6 (ii′)′ and R4 at 32³.

**Bindings (they hold for the stages that follow):**
- The predictor is solved by GMRES(30), right-preconditioned with the scalar diagonal of §4.2; 11 operator
  applications per step at 32³ and 64³. (§4.2's "PCG" is void: the operator is not symmetric, A15.)
- There is no `fv/aggregation_amg.hpp`. The MIS-2 setup lives in core (`solver/graph_amg_setup.hpp`), today only
  on core branch `amgpar` (9e0f881). Until that is on core main, `ale-s0` is built with
  `-DPECLET_VENDOR_SIBLINGS=ON -DFETCHCONTENT_SOURCE_DIR_PECLET_SIB_PECLET-CORE=<checkout of amgpar>` (and the
  same for morton); no source line of voro depends on the path. **`ale-s0` cannot land before `amgpar` has.**
- Reductions (dot products, budget) are plain `parallel_reduce`: bitwise equal run to run at a fixed launch on
  both backends, 6e-16 between 1 and 8 threads. A step is therefore reproducible, not thread-count independent.
- The 20 cold iterations of μ̂ use the 1e-2 solve as well. Mode R's absolute floor is 4ε times a bound of |C′||U|.
- The comparison with the reference is made at ω = 1 (mesh Z at 16³ in mode F, mesh W in mode R), because the
  reference seeds μ̂ with numpy's generator; μ̂ itself is gated in R3.

**Gate G1.2 as fixed before the run: FAIL (R5 (b) misses; B missed in the first run).** Logs
`build_s2/gate_logs/S1_2b.*`, `S1_2c.*`; all measured, host (OpenMP, 8 threads) and GPU unless stated.
| id | result |
|---|---|
| B **MISS** | host: G1.1 with K1–K6 equal to the record, battery 46/46 with the same hash, `S0.2'`, `G0.3'` PASS: amgpar's headers change no digit. GPU: `G0.3'` and G1.1 K8 PASS; three first measurements miss (below) |
| L PASS | PCG to 1e-10, L_ρ of M1 with the sphere of G0.4, 32³ / 64³: density ratio 1 12 / 15, ratio 1000 16 / 26; GPU the same counts and level sizes; GPU setup at 64³ 609 MiB; setup 224 ms host, 110 ms GPU |
| C0 PASS | new operators against the reference 4.6e-16 (host), 5.0e-16 (GPU) |
| C1 PASS | one step: F 1.2e-15, U* 1.9e-14, U¹ 1.8e-14, p 4.5e-14 |
| C2 PASS | ten steps: U 1.1e-13, p 2.1e-13; W in mode R 2.6e-13 |
| C3 PASS | GPU against host 2.0e-15; repeated runs bitwise; 1 against 8 threads 6.1e-16 |
| R0 PASS | rest exactly 0; uniform flow 2.9e-15 |
| R1 PASS | budget closure 5.5e-17 K at the default tolerances; residual work 2.6e-13 of the dissipation |
| R2′ PASS | momentum with the residual booked 6.0e-16 (1.7e-12 without); τ\|DΦ₁\|/V 9.8e-15 |
| R3 PASS | 2000 steps (W 16³) and 500 (W 32³): K never rises; μ̂ ≥ 0.951 μ_exact with the 1e-2 solve; ω μ_exact ≤ 1.023; μ_exact Z / W 0.897 / 0.919, 0.986 / 0.999, 1.009 / 1.023, 1.014 / 1.026 at 8³ … 64³ |
| R4 PASS | mode F against mode R 1.08 % at 32³ (3.43 % at 16³) |
| R5 (a) PASS | 16³ equal to the reference in every printed digit, 32³ to 4.6e-5 |
| R5 (b) **MISS** | error at 16³ / 32³ / 64³ and order: TG Z 3.329e-2 / 1.029e-2 / 3.409e-3 (1.69, 1.59); ABC Z 3.191e-2 / 5.586e-3 / 1.727e-3 (2.51, 1.69); TG W 3.532e-2 / 1.073e-2 / 3.577e-3 (1.72, 1.58); ABC W 3.565e-2 / 6.701e-3 / 2.05e-3 (2.41, 1.71); asked ≥ 1.8 between 32³ and 64³ |
| R6 PASS | (i) 5.3e-16; (ii′)′ fall from rest 1.49e-11 against 2e-11; mean velocity from the TG start 1.2e-15 |
| T PASS | per step, mode F: 32³ host 91 ms, GPU 19 ms; 64³ host 972 ms (predictor 555), GPU 102 ms (predictor 47, each Laplacian solve 11–13); predictor 10.6 / 11.3 applications; Laplacian 8.5 / 9.0 / 8.5 iterations (S3 / S5 / S6, 64³). Measured at load 19 to 110 with the GPU shared: upper figures |

**B on the GPU (first runs there; none comes from S1.2b, none is diagnosed).** `test_ale_facemesh.cpp` did not
compile with nvcc until 9b58bce, so G0.1′ and the whole-tree `G0_2.sh` had never run on CUDA.
- `test_ale_facemesh`, case b (power and Voronoi): `diag=MISMATCH`, every other number of the case at rounding
  (closure 6e-16, moment 3e-15); the host passes. Candidate [estimate]: contraction, as A11.
- `S0.2'`: (i)–(iv), empty cells, V0.1 a and b PASS; **(v′) fails: two identical launches of the 500-frame moving
  mesh differ on CUDA** (sha differs, volumes to 9.3e-16), Z and W. On the host two launches at one thread agree
  bitwise. A static mesh is bitwise on the GPU (C3). This bears on V0.2 and G1.5 of the moving mesh.
- `python/state_hash.py` aborts on CUDA (host access to the device View `sph.cen`): shared voro code, not ALE.
- `G1_1.sh:142` fixes the battery at 46 ctests; with `test_ale_stepper` there are 47.

**The order (S1.2c, criterion D-τ fixed before the run: error at Courant 0.2 over error at 0.4, at 64³, ≤ 0.80
means time, ≥ 0.95 means space).** GPU, mode F, Courant 0.4 / 0.2 / 0.1:
| case | error | ratio 0.2 / 0.4 |
|---|---|---|
| TG Z 32³ | 1.02943e-2 / 1.00771e-2 / 1.00249e-2 | 0.979 |
| TG Z 64³ | 3.40907e-3 / 3.37037e-3 / 3.36086e-3 | 0.989 |
| ABC Z 32³ | 5.58572e-3 / 5.58041e-3 / 5.57907e-3 | 0.999 |
| ABC Z 64³ | 1.72739e-3 / 1.72604e-3 / 1.72570e-3 | 0.999 |
| TG W 64³ | 3.57698e-3 / 3.55073e-3 / 3.54428e-3 | 0.993 |
**The first-order part is spatial.** The step is second order in τ here (successive differences fall by 4.0 to
4.2), and at Courant 0.4 the time error is 1.5 % (TG) and 0.1 % (ABC) of the error at 64³. Extrapolated to τ = 0
the order between 32³ and 64³ is 1.58 (TG) and 1.69 (ABC) [derived from the table]. What in space is not shown by
these runs. Candidate [estimate]: the lumped mass with the adjoint pressure force (decision 8 of the campaign: on
these meshes a cell-valued gradient leaves 9 % of itself after projection at every h, zero weights, 3D statics),
which the consistent mass removed in the 2D oracle at the price of momentum and an energy theorem. Not tested: a
lattice (jitter 0), the friction and the transport operators one by one.

**What this fixes for S1.3.** The stepper's stages are the ones S1.3 wraps with two frames. The order bound of
G1.3 (≥ 1.8 on the moving mesh) will miss for the same reason unless the bulk scheme changes; S1.3 gates the
moving mesh against the static one at equal resolution, and the order is carried as a recorded miss of the bulk
scheme, with decision 8.

### A17 (2026-11-18): the GPU base of the ALE tests; what each bulk operator leaves on exact fields (work orders S1.2d, S1.2e; gates G1.2d, D-h)

**S1.2d (GPU base).** `test_ale_facemesh` case b compared the builder's `maxNormalConnSine` (device) with a host
recomputation by `==`. On CUDA the two differ by 0.005 ε (power) and 0.069 ε (Voronoi), and by exactly 0 with
`--fmad=false` and on the host [measured]: nvcc contracts the cross product, as in A11. The test now compares to
16 ε and prints both values; the builder is unchanged. `ctest -LE bench` is 47/47 on the host and on CUDA
(`G1_1.sh` counts 47). Left open on CUDA, both outside this branch's code: two launches of the 500-frame moving
mesh differ (volumes 9.3e-16; fixed on `repair-det`, not on main), and `python/state_hash.py` aborts.

**S1.2e (a-priori test, `tests/python/ale_apriori.py`).** Exact Taylor–Green and ABC fields (ρ = 1, ν = 0.01,
t = 0) at the generators go into the matrix-free reference operators of the static step. With G = ∇p/ρ = −(u·∇)u
at the points and P_h the discrete projection (M-orthogonal, onto ker C′), the truncation of the step is exactly
T′ + G′ + Q′ + Z′ + U′ with T′ = P_h(a_T − G) (central transport with the centroid correction), G′ = P_h G (a
gradient carried as cell values), Q′ = P_h a_Q (upwind part with the reconstruction jump), Z′ = P_h(a_Z − νΔu)
(friction), U′ = (P_h − 1)νΔu. Each part is measured by its cell norm, by its rate along the solution
coh = ⟨X, u⟩/‖u‖², and by its content in the Fourier modes max|k_i| ≤ 2·2π with the part along u removed. Meshes:
a cubic lattice built in Python, Z (M1) and W (M2), at 16³, 32³, 64³. Table: `build_s2/gate_logs/S1_2e.table`.

- **Lattice control (E1 PASS):** raw residues of transport, friction and pressure force fall by 0.248–0.255 per
  doubling (second order), the upwind part by 0.126 (third); T′ and G′ vanish to rounding (on a lattice the
  transport of these fields is a discrete gradient).
- **Norms on Z, Taylor–Green, 16³ / 32³ / 64³** (ABC on Z at 64³: T′ 1.5 %, G′ 9.2 %, Q′ 0.8 %, Z′ 33 %): T′ 6.3 / 3.2 / 1.6 % and
  Q′ 2.9 / 1.4 / 0.7 % of ‖G‖ (first order); **G′ 12.1 / 9.9 / 9.3 % (W: 12.0 % at 64³): does not fall**, the
  numbers of decision 8 (D5) on a dynamic field; **Z′ 17 / 20 / 30 % of ‖νΔu‖: grows.**
- **Why the friction residue grows [measured twice, mechanism derived]:** ζ is the adjoint of U → G_f applied to
  the face stress. For a uniform stress σ its two-point (normal) part vanishes by closure (3e-15), its tangential
  part does not: the force on cell i is σ·b_i with a mesh vector b_i, Σ_i b_i = 0, ‖b_i/V_i‖ = 0.0421/h on Z
  and 0.0469/h on W (16³, 32³, 64³: 0.04212, 0.04195, 0.04192). The cell weight of the tangential stress,
  Σ_f ½V_f(I − n nᵀ), is not ⅔V_i I per cell because the face centroids are offset from the connectors (q_f ≠ 0).
  For Taylor–Green this predicts a residue 2.65e-3 n; the fit raw_Z² = A + B n² gives 2.6e-3 n. So the friction
  force is consistent only in the weak sense, like the pressure force (A6, decision 8): one family of defect, the
  adjoint of a linearly exact reconstruction is not linearly exact where q_f ≠ 0.
- **Rates along u (Taylor–Green, Z):** friction +5.10e-2 / +1.32e-2 / +3.35e-3 (second order; 1.3 × the
  lattice's), upwind −4.79e-2 / −6.28e-3 / −7.93e-4 (third order), all others ≤ 1.1e-5 at 64³. Total +2.78e-3 /
  +6.97e-3 / +2.56e-3 against the run's errors 3.33e-2 / 1.03e-2 / 3.41e-3 (A16): at 32³ and 64³ the amplitude
  error is a h² term and a h³ term of opposite sign (ratio 0.367, the error's 0.331; criterion E3 PASS). **No
  operator is first order along the solution.**
- **Not attributed (criterion E4 MISS):** for ABC the total rate (+4.2e-4 at 64³) is a quarter of the error
  (1.73e-3). In both flows a part (error² − rate²)^½ remains, 7.6e-3 / 2.25e-3 (TG) and 5.57e-3 / 1.67e-3 (ABC) at
  32³ / 64³, of order 1.7. The low-mode content does not predict it: it is 3.8e-2 ‖u‖ at every h, and it is all in
  G′. That content is an identity, not a free forcing: P_h is an orthogonal projector, so the component of G′
  along G is ‖G′‖²/‖G‖² = 0.86 % (Z), 1.43 % (W) of G exactly [derived; measured 0.862 %, 1.431 %], tied to mesh
  noise ten times larger. What the mesh-scale parts (G′, Z′) leave in the solution after damping is a response, and
  an a-priori test does not measure it.

**What this fixes.** R5 (b) of G1.2 is not a first-order operator. It is (i) the cancellation above and (ii) a
response to mesh-scale residues of the two adjoint forces, not split between them. The falsifier for friction is a
Stokes decay ladder on the stepper (transport off; exact solution the decaying Taylor–Green field, p = 0). Not
measured: the pressure error; a moving mesh; two phases.

### A18 (2026-11-18): the time step on a moving mesh, one phase (work order S1.3a; answers Q-MESH, Q-FRAME, Q-MASS of §9; replaces the S3 and S4 rows of §4.1 for stage 1 and the gates of WO-S1.3)
One phase, uniform ρ and μ, periodic box, no body force. Masses are lumped: Mⁿ = ρVⁿ, M¹ = ρV¹, ΔM = M¹ − Mⁿ.
1. **Mesh rule (Q-MESH), evaluated in S1 from U_ext.** `Rest`: ẋ = 0, ẇ = 0. `Follow`: ẋ_i = U_ext,i, ẇ = 0.
   `Prescribed`: ẋ given by the caller per step (a view), ẇ = 0. Weights never change in stage 1.
   x^h = xⁿ + ½τẋ, x¹ = xⁿ + τẋ (positions wrapped into the periodic box as `test_ale_frame` does).
2. **Frames (S2).** Fh = `advanceFrame` at x^h, then F1 = `advanceFrame` at x¹, each followed by `setupFrame`.
   Both exist before S3. A frame error (empty cell, one-sided facet, open cell) is an error of the step: no
   retry, no smaller step. Of frame n only Vⁿ is kept. U_ext = Uⁿ at the first step, as in the static step.
3. **S3 on Fh.** Φᵐ = `sweptFlux`(Fh, ẋ, ẇ). Correction to the exact volume change, with the frame's L_ρ and
   a_f, the same operator pair as `compactFluxOf`: solve L_ρ ψ = D Φᵐ − (V¹ − Vⁿ)/τ in the sign convention of
   `compactFluxOf`, Φᵐ_c = Φᵐ − (its correction), so that max_i |τ(DΦᵐ_c)_i − (V¹_i − Vⁿ_i)| ≤ `compactTol`·V_i
   (1e-14; V_i of Fh). A right-hand side that is exactly zero returns ψ = 0 exactly, without iterating. Then
   Φ = `compactFluxOf`(Fh, U_ext) as now, F = Φ − Φᵐ_c, Fm = ρF. The Courant number reported is that of F.
   (With one density the frame's L_ρ is the unit-density Laplacian up to a factor; which one two phases need is
   open and is not decided here.)
4. **S4 on Fh.** With S, Q, e6, e8 from Fm and ζ, C′ᵀ from Fh (the existing kernels):
   (M¹ − ¼ΔM + ½τ(S + Q + ζ)) U* = (Mⁿ + ¼ΔM − ½τ(S + Q + ζ)) Uⁿ + τ(e6 + e8 + C′ᵀpⁿ).
   This is the flux-form transport of the oracle, (T x)_i = Σ_f s_if Fm_f ½(x_i + x_o) = (S x)_i + ½(D Fm)_i x_i,
   with (D Fm)_i replaced by its value −ΔM_i/τ under item 3. Written so, a mesh at rest gives ΔM = 0 exactly and
   the static operator bit for bit. The preconditioner diagonal uses M¹ − ¼ΔM where it used M.
5. **S5, S6 on F1** exactly as the static step on the frame F1 with the mass M¹ (μ̂, mode F or R, compact flux).
6. **S8.** Vⁿ ← V¹; Uⁿ⁻¹ ← Uⁿ; the engine stays at x¹.
7. **Budget.** K = ½U·MU with the mass of its frame. K* − Kⁿ gains exactly one term over the static budget:
   **B1 = +⅛ Σ_i ΔM_i |U*_i − Uⁿ_i|²** (derived: multiply item 4 by Ū = ½(U* + Uⁿ); S is skew). The other terms
   keep their formulas with Fh's operators in S4 and F1's in S5. If `B1` in the code is not identically zero on
   a static mesh today, stop and report.
8. **Momentum.** Per step the scheme conserves Σ_i M_i U_i up to the predictor's residual (booked today, R2′) and
   up to the defect of item 3 and of DΦ = 0, which changes the momentum by
   ΔP_def = +½ Σ_i ρ[τ(DF)_i + (V¹_i − Vⁿ_i)] Ū_i per step (derived: column sums of S are −½ D Fm). Book it
   as the residual's momentum is booked (`mom_booked` keeps its meaning: drift minus everything booked).
9. **Not in this work order:** a body force on a moving mesh (a non-empty `body` in the moving step is an
   error; the mean part needs ½(Mⁿ + M¹), work order S1.3c), power weights that change, two phases, MPI, Python
   bindings, public names.

**Gates (fixed before any run; brief `ALE_S1_3A`).** V0.2: `--motion rest` against the static path, U and p bit for
bit after each of 100 steps (Z and W, host and GPU). T: each of 5 steps against a Python reference written from
the items above, ≤ 1e-12. G1.3 a: volume change, item 3's measure ≤ 1e-14. b: uniform flow under the prescribed
motion ẋ = 0.3 (sin 2πy, sin 2πz, sin 2πx), ≤ 1e-12 |U| after ≥ 1000 rebuilt cells. c: budget closure ≤ 1e-13 K
per step with B1. d: momentum with the booked parts ≤ 1e-13 Σ M|U|. The order bound of §7 is replaced (A16): the
moving mesh is gated against the static one at equal resolution in work order S1.3b, error ratio at t = 0.5
≤ 5 (follow) and ≤ 3.5 (prescribed) at 32³ and 64³, order ≥ the static order − 0.3 [2D oracle, A4e: ratios
3.5–4.3 and 2.4–2.6]. Stability at weight spread 0.15 h² is not gated on a moving mesh: a power cell that empties
is a frame error (A5).

### A19 (2026-11-18): the Stokes ladder; the friction operator carries the order-1.7 part (work order S1.2g, gates D-S, D-S′; closes the open item of A17)
`StepOptions::transport = false` sets the mass flux of S4 to zero (Stokes flow; exact solution the decaying
Taylor–Green field, p = 0; time step of the full run). The runs print the error along the exact field (`err_par`,
signed) and orthogonal to it (`err_perp`). Commit e84a8ac; logs `build_s2/gate_logs/S1_2g.*`; without the option
every digit on record is reproduced (host, GPU, reference); device against reference in Stokes flow ≤ 3.8e-6.

| TG, GPU | error | along u | orthogonal |
|---|---|---|---|
| full, Z 32³ / 64³ | 1.029e-2 / 3.409e-3 | 6.84e-3 / 2.407e-3 | 7.70e-3 / 2.414e-3 |
| Stokes, Z 16³ / 32³ / 64³ | 5.857e-2 / 1.530e-2 / 4.326e-3 | 5.223e-2 / 1.363e-2 / 3.680e-3 | 2.651e-2 / 6.95e-3 / 2.274e-3 |
| full, W 32³ / 64³ | 1.073e-2 / 3.577e-3 | 6.37e-3 / 2.318e-3 | 8.63e-3 / 2.724e-3 |
| Stokes, W 16³ / 32³ / 64³ | 5.985e-2 / 1.568e-2 / 4.519e-3 | 5.292e-2 / 1.382e-2 / 3.787e-3 | 2.797e-2 / 7.40e-3 / 2.466e-3 |

- **D-S′ (measured): friction carries it.** Orthogonal part, Stokes / full, at 32³ / 64³: 0.90 / 0.94 (Z),
  0.86 / 0.91 (W), against ≥ ½. D-S (with the a-priori rate of A17): 1.01 / 1.22 (Z), 1.02 / 1.21 (W). They agree.
- Orders in Stokes flow, 16³ → 32³ → 64³: error 1.94, 1.82 (Z), 1.93, 1.79 (W); along u 1.94, 1.89 (Z); orthogonal
  1.93, 1.61 (Z), 1.92, 1.59 (W). The part along u is second order (the h² rate of A17; measured 3 % and 10 %
  above the a-priori rate). The orthogonal part loses order with refinement.
- The transport side adds little: if the two parts add in quadrature it is 3.3e-3 / 0.81e-3 (Z) at 32³ / 64³
  [estimate], 43 % / 34 % of the orthogonal part, and it falls faster than the friction part.

**What this fixes.** R5 (b) of G1.2 is the friction operator: its adjoint force is consistent only in the weak
sense (A17: a uniform stress σ gives the force σ·b_i, ‖b_i/V_i‖ = 0.042/h), and the solution's response to that
mesh-scale force is the part of order 1.6. The cell-valued gradient of the transport (decision 8) is second in
size. A repair of the friction force is a formulation question (F14 of the campaign state): no operator changes
before a checked derivation. Not measured: ABC in Stokes flow is not an exact solution with p = 0 and was not
run; the pressure error; a moving mesh.
