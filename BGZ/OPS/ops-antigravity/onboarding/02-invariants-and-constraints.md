# 02 — Invariants, Admissibility Constraints, and Exclusions

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/`  
**Date of Audit:** October 24, 2026  
**Source Commit SHAs:** RCF `c419353214ee83d83376747064912544aee257b9`, RCMT `b7eecd0dfd8080864b34441af86329f819cba65a`

---

## 1. The RCF Invariant Kernel

The highest normative authority in the repository is the **RCF Invariant Shard v1.1** (`RCF/core/S∞.RCF.Invariants.v1.1.md`). It defines ten irreducible, pre-theoretical constraints that govern all upstream and downstream modules. Any module that violates these invariants is non-admissible under RCF.

| Invariant Number | Invariant Name | Authoritative Formulation | Operational Meaning & Architectural Impact |
|:---|:---|:---|:---|
| **1** | **Persistence Under Constraint** (`#L26-L36`) | *Structures that persist under increasing relational constraint are meaningful; those that do not are eliminated.* | RCF does not construct entities or postulate fundamental particles. It filters survivability. Meaning is operational persistence under pressure. |
| **2** | **Emergence via Elimination** (`#L38-L46`) | *Structure emerges through the elimination of unstable relations, not through additive construction.* | Reject axiomatic building blocks. Only relational structures that survive exhaustive constraint pruning are retained. |
| **3** | **Relational Primacy** (`#L48-L56`) | *No entity possesses intrinsic meaning outside its relations.* | Identity is relational. No fundamental subject/object distinction. Background space and isolated point particles are forbidden. |
| **4** | **Spectral Dominance** (`#L58-L66`) | *Persistent structure is encoded spectrally, not configurationally.* | Microscopic states and configurations are transient. Spectra (eigenvalues, graph spectra, invariant operators) persist. |
| **5** | **Constraint Over Dynamics** (`#L68-L76`) | *RCF specifies admissibility conditions, not equations of motion.* | Rejects fundamental dynamical equations (e.g. Lagrangian mechanics, Hamiltonian evolution). Time evolution is treated as constraint variation. |
| **6** | **Minimal Free Parameters** (`#L78-L86`) | *Valid RCF-compatible structures minimize tunable constants.* | Structural ratios and integers must survive parameter elimination. Arbitrary fine-tuning or continuous parameter fitting is inadmissible. |
| **7** | **Non-Privileged Observers** (`#L88-L96`) | *Observation does not create reality; it constrains relational resolution.* | No observer is ontologically special. Observer and system are interchangeable under relabeling; measurement is boundary constraint resolution. |
| **8** | **Self-Applicability** (`#L98-L106`) | *The framework must be applicable to itself without contradiction.* | RCF concepts, documents, and rules must survive their own filters. Special exemptions for the framework itself cause immediate failure. |
| **9** | **I-Allowance** (`#L108-L118`) | *No complex system collapses to a single mode without residual alternatives.* | Dominant modes stabilize; residual alternative bands ($I$-bands) preserve adaptability and prevent brittle deadlocks. |
| **10** | **Invariant Closure** (`#L120-L130`) | *These invariants are jointly sufficient, mutually reinforcing, and minimal.* | Any compatible extension must preserve all 10 invariants, introduce no contradiction, and remain eliminable without core damage. |

---

## 2. Downstream RCMT Physics Constraints

In `RCMT/README.md`, `RCMT/global/constraints.md`, and `RCMT/local/constraints/`, RCMT adopts downstream physical constraints compiled against RCF:

### 2.1 Adjacency-First Invariant
- **Rule:** Discrete graph adjacency and boundary topology precede any metric, distance function, or coordinate manifold (`RCMT/local/constraints/geometry.md`).
- **Impact:** Spacetime cannot be postulated as a smooth 4-manifold. The manifold is an effective low-energy continuum approximation of an underlying discrete combinatorial complex.

### 2.2 Dual-Lane Symmetry Invariant
- **Rule:** Symmetry must partition into an unbroken symplectic mode-space lane ($Sp(8)$) and a representational geometric lane (currently evaluated via competing candidates $SO(9)$ vs. $SO(8)$) (`RCMT/local/invariants.md#L25-L40`, `RCMT/local/constraints/symmetry_chain.md#L45-L75`).
- **Impact:** Spacetime geometry and quantum-like mode space are strictly segregated. The symplectic lane never undergoes upstream symmetry breaking, preserving phase and probability measures across all physical processes. The historical $E_8 \to SO(10)$ cascade is formally archived as exploratory work.

### 2.3 Finite Structure Invariant (Clarified Formulation)
- **Rule:** All symmetry structures used in RCMT must admit finite-dimensional, adjacency-compatible representations (`RCMT/local/invariants.md#L90-L105`).
- **Specific Lane Scope:**
  - The **geometric lane** is subject to compactness constraints (except for local Lorentz structure $SO(3,1)$ introduced at the projection interface).
  - The **symplectic mode-space lane** ($Sp(8)$ / $Sp(8,\mathbb{R})$) is governed by its own preservation and admissibility constraints and is **not required to be compact solely by this invariant**.
- **Impact:** Infinite Kaluza-Klein towers, string excitations, and non-compact global geometric groups are prohibited, while the non-compact real symplectic carrier symmetry is mathematically reconciled.

### 2.4 Mode-Space Projection Invariant
- **Rule:** Continuous spacetime coordinates, Lorentzian metric signatures ($SO(3,1)$), and causal light cones exist exclusively at the **Projection Layer** (`RCMT/physics_interface/pipeline/overview.md#L45-L65`).
- **Impact:** There is no "upstream time." Time as duration, light-cone causality, and metric intervals are downstream representational features. Upstream contains only an ordered relational index (the Upstream Time Index).

### 2.5 Local Curvature Constraint
- **Rule:** Curvature must be local and representational; global non-compact curvature groups are rejected (`RCMT/local/constraints/symmetry_chain.md#L80-L95`).
- **Impact:** Gravity is modeled as projected mode-resistance and local relational mismatch, not as an ontologically fundamental warping of a smooth continuum.

---

## 3. Explicit Prohibitions and Documented Exclusions

RCF and RCMT explicitly reject several standard physics and mathematical moves to maintain framework stability (`RCF/README.md#L69-L82`, `RCF/rcp-1.2.md#L157-L165`):

```text
               ╔═══════════════════════════════════════════════╗
               ║             WHAT RCF / RCMT REFUSES           ║
               ╚═══════════════════════════════════════════════╝
 ❌ NO Postulation of Fundamental Particles  (Entities must emerge via elimination)
 ❌ NO Dynamics-First Formulations           (No fundamental differential equations)
 ❌ NO Metric-First Spacetime                (Adjacency precedes metric)
 ❌ NO Upstream Continuous Time              (Time is downstream projection)
 ❌ NO Privileged Observers                  (Observer is merely boundary constraint)
 ❌ NO Non-Compact Global Symmetries         (No infinite Kaluza-Klein towers)
 ❌ NO Continuous Parameter Fitting          (Structural ratios must be parameter-free)
 ❌ NO Rescue Clauses or Patching             (Failing models are discarded)
 ❌ NO Metaphysical or Mystical Claims       (Strictly anti-numerology and anti-dogma)
```

### Documented Reasons for Exclusions:
1. **No Fundamental Entities:** Postulating primitives (e.g. fundamental strings, point quarks) creates arbitrary starting points and invites untestable ontology. RCF demands that persistent entities emerge as irreducible boundary modes.
2. **No Continuous Metric Upstream:** Continuous manifolds introduce infinities (ultraviolet divergences) and assume geometric distances prior to relational interactions.
3. **No Non-Compact Global Symmetry:** Non-compact global groups generate infinite-volume state spaces that violate finite admissibility and combinatorial computability.
4. **No Rescue Clauses:** If a hypothesis (e.g. static geometric selection of a $3+1$ particle partition, or an unconstrained complex structure) fails its explicit test, it must be recorded as a failure rather than modified post-hoc.

---

## 4. Anti-Backflow Boundaries and Precedence Rules

Under **RCP-1.2** (`RCF/rcp-1.2.md#L157-L195`), every document participates in an authoritative directionality contract:

```text
Upstream Invariants (RCF Core)
         │  ▲
         │  │ ── PROHIBITED: Downstream redefinition, expansion,
         │  │    or promotion of interpretations upstream (Anti-Backflow)
         ▼  │
Downstream Physics (RCMT)
         │  ▲
         │  │ ── PROHIBITED: Model-local claims (e.g. SMG Dual Closure)
         │  │    acting as upstream physical requirements
         ▼  │
Domain Bridges (SMG v2.1) & Computational Laboratories (OPS, Lisi-SMG)
```

### Mandatory Document Contract Rules:
1. Every canonical file must declare a valid **Document Contract** (`owner`, `type`, `scope`, `authority`, `upstream dependencies`, `status`, `purpose`, `anti-backflow`).
2. **Authority vs. Dependency:** Consuming an upstream module (`upstream dependencies`) does not confer its authority (`authority`). Downstream documents maintain downstream authority only.
3. **Conflict Resolution:** When body prose contradicts the document contract or upstream contracts, the upstream contract governs unconditionally (`RCF/rcp-1.2.md#L180-L192`). Conflicting body prose fails validation and must be discarded.
