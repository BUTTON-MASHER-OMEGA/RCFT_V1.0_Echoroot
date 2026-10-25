# 01 — Architectural Hierarchy and Mathematical Interfaces

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/`  
**Date of Audit:** October 24, 2026  
**Source Commit SHAs:** RCF `c419353214ee83d83376747064912544aee257b9`, RCMT `b7eecd0dfd8080864b34441af86329f819cba65a`

---

## 1. Architectural Relationship: RCF vs. RCMT

The research program operates under a strict two-tier architecture governed by the RCF Compression Protocol (RCP-1.2, `RCF/rcp-1.2.md`):

### 1.1 The Upstream Kernel: RCF
The **Relational Coherence Framework (RCF)** is an eliminative, constraint-first mathematical scaffold (`RCF/README.md#L14-L49`). It does not specify physical dynamics, fundamental entities, or equations of motion. Instead, it defines admissibility conditions: any relational structure that fails to persist under increasing constraint is discarded without rescue clauses.

RCF operates at the level of:
- Pure adjacency and boundary/coboundary operations ($\partial, \delta$) (`RCF/core/geometry/boundary_ops.md`).
- Discrete exterior calculus on abstract complexes.
- Spectral and algebraic invariants (`RCF/core/algebra/`).
- Ten non-negotiable core invariants (`RCF/core/S∞.RCF.Invariants.v1.1.md`).

### 1.2 The Downstream Physics Layer: RCMT
**Relational Coherence Mode Theory (RCMT)** is a physics-specific theory compiled *downstream* against the RCF core (`RCMT/README.md#L1-L20`). RCMT takes the relational primitives of RCF and investigates whether physical phenomena—such as spacetime metric, Lorentz invariance, gauge symmetries, quantum measurement, mass, and decoherence—can emerge purely through mode selection on discrete relational structures.

**Crucial Architectural Distinction:**
- RCF asks: *What relational structures survive constraint?*
- RCMT asks: *Can standard physical phenomena be faithfully modeled as the survivor modes of those relational structures?*

Under the anti-backflow rules of RCP-1.2 (`RCF/rcp-1.2.md#L157-L165`), RCMT is strictly forbidden from modifying upstream RCF invariants or elevating its own physical interpretations into mathematical requirements of RCF.

---

## 2. The RCMT Pipeline Architecture

The flow of operations in RCMT is formally specified in `RCMT/physics_interface/pipeline/overview.md#L10-L100`:

```text
┌────────────────────────┐
│ 1. Upstream Substrate  │  Pure relational units, directions, coherence relations,
│                        │  Upstream Time Index (ordering only). NO metric, NO Lorentz frame.
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 2. Local Coherence     │  Mismatch pruning, directional alignment, local smoothing,
│    Operations (LCO)    │  basin stabilization. Strictly local action.
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 3. Coherent Relational │  Locally consistent, basin-stable, mismatch-free state.
│    State               │  Still geometry-free, metric-free, Lorentz-free.
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 4. Projection Layer    │  First representational operation. Dual-lane attachment:
│                        │  - Mode-space lane: Sp(8) (unbroken)
│                        │  - Geometric lane: SO(9) -> SO(5) x SO(3,1)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 5. Projected Geometric │  Lorentzian metric, causal light cones, coordinate time,
│    Frame               │  effective local curvature.
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 6. Global Invariants   │  Nontrivial topology detected across closed paths:
│                        │  Twist (frame mismatch), Holonomy, Winding (linkage).
└────────────────────────┘
```

### Key Principles of the Pipeline:
1. **Spacetime is Downstream:** Upstream has no continuous coordinates, no background metric, and no geometric duration. Coordinate time and Lorentz signatures ($SO(3,1)$) exist only as representational features at Stage 5 (`RCMT/physics_interface/pipeline/overview.md#L45-L60`).
2. **LCO is Strictly Local:** The Local Coherence Operator cannot act globally. It smoothes local neighborhoods and prunes local mismatches.
3. **Global Invariants Resist LCO:** Global properties (Twist, Winding, Holonomy) are locally trivial but globally persistent (`RCMT/global/invariants.md#L20-L40`). Because LCO acts locally, it cannot remove global topological obstruction.

---

## 3. The Dual-Lane Symmetry Architecture

In `RCMT/local/constraints/symmetry_chain.md` and `RCMT/local/invariants.md`, RCMT establishes a **dual-lane symmetry requirement**:

$$\text{Symmetry Structure} = \text{Mode-Space Lane} \;\times\; \text{Geometric Lane}$$

1. **Mode-Space Lane:** Symplectic structure ($Sp(8)$ or $Sp(8, \mathbb{R})$) that remains strictly unbroken across all admissible operations, carrying unresolved phase relations and stochastic information.
2. **Geometric Lane:** Candidate structure for downstream geometric projection, requiring a local Lorentzian structure compatible with $SO(3,1)$ and compact internal symmetry ($SO(4)$) at the low-energy projection target.

### Current Candidate Survivor Implementations:
$$\mathbf{Candidate\ A:} \quad Sp(8) \times SO(9)$$
$$\mathbf{Candidate\ B:} \quad Sp(8) \times SO(8)$$

### Architectural Status of the Symmetry Candidates:
- **Provisional Candidates, Not Fixed Embeddings:** Neither candidate is an established mathematical decomposition or core ontological identity of RCMT (`RCMT/local/constraints/symmetry_chain.md#L55-L75`). Both are currently subject to comparative constraint testing.
- **Historical Chain Relegated:** The previously explored chain $E_8 \longrightarrow SO(10) \longrightarrow Sp(8) \times SO(9)$ is now formally classified as **historical exploratory work** rather than a current architectural dependency (`RCMT/local/constraints/symmetry_chain.md#L90-L100`). This resolves the Lie algebra dimensional incompatibility ($\dim \mathfrak{so}(10) = 45 < 72$) noted in prior audits.
- **Projection Boundary:** Neither $SO(9)$ nor $SO(8)$ is asserted to contain $SO(3,1)$ through a conventional compact-group symmetry-breaking chain. The emergence of Lorentzian structure is an explicit **projection-level requirement** whose mathematical realization remains to be verified.
- **Clarified Compactness Constraints:** Under the updated Finite Structure Invariant (`RCMT/local/invariants.md#L90-L105`), the geometric lane is subject to compactness constraints (except for local Lorentz structure at projection), while the symplectic mode-space lane is governed by its own preservation constraints and is **not** required to be compact. This resolves the previous mathematical conflict regarding non-compact $Sp(8,\mathbb{R})$ carrier symmetry.

---

## 4. The Role of SMG v2.1 (Standard Model of Geometry)

The Standard Model of Geometry (SMG v2.1, `RCMT/smg_v2.1/`) is positioned between the upstream physics interface and downstream phenomenology:

```text
RCF Geometry Stack / Physics Interfaces
                 │
                 ▼ (Read-Only Imports: Chromatic-12, 36 composite modes, gamma plateaus)
             [ SMG v2.1 ]
                 │
                 ▼ (Local Thesis Claims & Translations)
Downstream Standard Model Phenomenology (Quarks, Leptons, Gauge Bosons)
```

### Governing Contract (`RCMT/smg_v2.1/smg_contract.md`):
- **Imported Structure (Read-Only):** Chromatic-12 indexing, three gamma-plateau groupings, 36 composite modes, and sectorization records from `gcm_interface` and `physics_interface`.
- **SMG-Local Thesis Claims:**
  - Asserts *Dual Closure* as its model-local organizing principle (`RCMT/smg_v2.1/claims/dual_closure.md`).
  - Maps Chromatic-12 to the Standard Model $8 + 3 + 1$ gauge structure ($SU(3) \times SU(2) \times U(1)$) (`RCMT/smg_v2.1/smg_contract.md#L40-L55`).
  - Maps the three gamma plateaus to three fermion generations (`RCMT/smg_v2.1/dictionary/gamma_generations.md`).
- **Anti-Backflow Boundary:** SMG's particle assignments and semantic interpretations are model-local labels. They are not upstream RCF facts. Downstream models may opt into SMG v2.1, but non-adopting models are not bound by SMG assertions (`RCMT/smg_v2.1/smg_contract.md#L65-L85`).

---

## 5. The Role of OPS (Operational Phase Space)

The Operational Phase Space (OPS) project (`Projects/Operational_Phase_Space/`) serves as the computational proving ground where algebraic and geometric hypotheses are subjected to discrete algebraic testing:

1. **Carrier Regularity (SIM14):** Tests whether a finite 8-state carrier $X_8$ with canonical symplectic form $\Omega$ and frozen cycle $C_8$ can support a transitive, regular group action without continuous approximations. Result: proved the $V_4$ Orbit-Bridge Regularity Theorem.
2. **Coxeter & Polytope Embedding (SIM15):** Tests whether the operational symmetry group $H_{192} = C_2^3 \rtimes S_4$ embeds into standard exceptional reflection groups. Result: discovered a unique embedding into $W(H_4)$ with a threefold outer normalizer ($N_{576}/H_{192} \cong C_3$ triality) and identified the 25 conjugate architectures $\mathcal{L}_{25}$ with the 25 native 24-cells in the 600-cell.
3. **Metric Rigidity (SIM16):** Evaluates whether an invariant metric $G$ compatible with $J$ ($J^2 = -I, G\Omega G = \Omega, G > 0$) can be selected by graph locality. Result: proved that strong $C_8$ adjacency forces an exact 1-parameter hyperbolic metric family $G(t) = \cosh(t) I + \sinh(t) |\Omega|$.

---

## 6. Document Authority and Precedence

The repository's document precedence hierarchy is strictly ordered:

$$\mathbf{RCF\ Core\ Invariants} > \mathbf{RCF\ Contracts\ (RCP)} > \mathbf{RCMT\ Physics\ Interface} > \mathbf{RCMT\ Local\ Constraints} > \mathbf{SMG\ v2.1\ / Simulations}$$

When a conflict occurs:
1. Upstream invariants overrule downstream interpretations.
2. An analytic proof or exhaustive combinatorial verification overrules authorial narrative prose.
3. Computational results in OPS or `lisi-smg-test` establish admissibility or falsification within their tested scope, but cannot redefine upstream contracts.
