# 00 — Repository Map and Operational Reconnaissance

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/`  
**Date of Audit:** October 24, 2026  
**Theory Root Path:** `C:\Users\blaur\projects\theory`  
**Git Commit SHAs:**
- `RCF/`: `57af06b5ac7c73b66839f70ebba3293399a1c880` (Updated following D1-D11 cleanup pass)
- `RCMT/`: `666ed97aaa1d11f82d0a9881e99bb0c9cae2d225` (Updated following D1-D11 cleanup pass)
- `Projects/Operational_Phase_Space/`: Non-git input directory (cleaned executable scripts, `recorded_output/`, and `ARCHIVE_MANIFEST.json`)
- `Projects/lisi-smg-test/`: Non-git input directory (supplemented with graph-certified `120cell_vertices.repaired.txt` and regression suite)

---

## 1. High-Level Workspace Architecture

The `theory/` repository tree houses four primary research units representing different strata of the research program:

```text
theory/
├── RCF/                                # [Upstream Root] Relational Coherence Framework (Normative Kernel)
│   ├── core/                           # Invariants, abstract algebra, discrete calculus, geometry, RCD
│   ├── core-adjacent/                  # Pre-formal and scaffolding modules
│   ├── modules/                        # Downstream interfaces (gcm, gco, lco, kinematics, temporal)
│   ├── src/python/                     # Tested Python reference implementation (rcf_core)
│   ├── rcp-1.1.md / rcp-1.2.md         # RCF Compression Protocols (document contracts, anti-backflow)
│   ├── rcf_compressed.md               # Compressed canonical shard
│   └── README.md                       # Framework identity and charter
│
├── RCMT/                               # [Downstream Physics] Relational Coherence Mode Theory
│   ├── global/                         # Global topological invariants (twist, holonomy, winding)
│   ├── local/                          # Local constraints (cosmology, decoherence, lco, symmetry chain)
│   ├── modules/                        # Mode selection, memory, and coupling modules
│   ├── physics_interface/              # Sealed contracts: Upstream -> LCO -> Coherent State -> Projection
│   ├── smg_v2.1/                       # Standard Model of Geometry translation bridge
│   └── README.md                       # Physics-specific scope and architectural boundaries
│
└── Projects/
    ├── Operational_Phase_Space/        # [Computational Laboratory] OPS SIM14, SIM15, SIM16 scripts & notes
    │   ├── SIM14.0.py - SIM14.8.py     # Symplectic carrier, M1/M2 bifurcation, torsor & Fano incidence
    │   ├── SIM15.0.py - SIM15.6.py     # H4 Coxeter survivability, W(D4) triality, 24-cell identification
    │   ├── SIM16.0.py - SIM16.4.py     # J-complex exact rigidity, hyperbolic metric family, transport audits
    │   ├── Vocabulary_Packet.md        # Authoritative vocabulary freeze for SIM14
    │   ├── V4_Orbit-Bridge_...md       # Analytic proof of V4 Orbit-Bridge Regularity Theorem
    │   └── SIM14/15 Discussions        # Detailed research deliberation records
    │
    ├── lisi-smg-test/                  # [Integration Study] 24-cell -> 600-cell -> 120-cell -> SMG study
    │   ├── reference_geometry/         # Raw polytope adjacency tables (600-cell, 120-cell)
    │   ├── references/                 # Lisi paper and smg_v2.1 concatenation
    │   ├── src/ & tests/               # Executable verification pipeline
    │   ├── artifacts/                  # Cryptographically hashed (SHA-256) JSON execution artifacts
    │   ├── REPORT_PHASE1 - PHASE6D     # Phase-by-phase mathematical audits and verification reports
    │   └── summary.md                  # Comprehensive synthesis of geometry recovery vs. missing LCO
    │
    ├── ops-antigravity/                # [Assigned Agent Workspace] Antigravity audit and onboarding output
    │   ├── onboarding/                 # Deliverables 00 through 07
    │   └── research/                   # Working scratch space, reproduction harnesses, and test logs
    │
    └── ops-codex/                      # [Independent Peer Workspace] Uninspected per onboarding isolation rule
```

---

## 2. Inventory and Purpose of Core Directories

### 2.1 `RCF/` — Relational Coherence Framework (Kernel)
- **Role:** Pure mathematical and constraint-first foundation. It is pre-physical, non-mechanistic, and eliminative.
- **Authority:** Highest authority in the project hierarchy.
- **Key Documents:**
  - `README.md`: Defines RCF as a constraint filter rather than a generative physics theory; lays out what RCF refuses to do (`RCF/README.md#L69-L82`).
  - `rcp-1.2.md`: Canonical document contract standard specifying mandatory metadata (`owner`, `type`, `scope`, `authority`, `upstream dependencies`, `status`, `purpose`, `anti-backflow`) (`RCF/rcp-1.2.md#L30-L50`).
  - `core/S∞.RCF.Invariants.v1.1.md`: The root invariant shard defining the 10 irreducible RCF invariants (`RCF/core/S∞.RCF.Invariants.v1.1.md#L20-L109`).
  - `src/python/`: Python package `rcf_core` containing simplicial geometry, discrete boundary/coboundary operators, and topological data structures. Validated by 13 unit tests (`pytest`).

### 2.2 `RCMT/` — Relational Coherence Mode Theory
- **Role:** Downstream physical theory compiling against RCF. It models physical phenomena (mass, decoherence, measurement, gauge structure) through relational mode selection under boundary conditions.
- **Authority:** Subordinate to RCF. Cannot alter or backflow into RCF invariants.
- **Key Documents:**
  - `README.md`: Declares the dual-lane architecture ($SO(3,1) \times SO(4)$ geometric lane and $Sp(8)$ mode-space lane) and defines the downstream boundary (`RCMT/README.md#L1-L80`).
  - `physics_interface/pipeline/overview.md`: Defines the end-to-end operation flow: Upstream substrate $\to$ Local Coherence Operations (LCO) $\to$ Coherent relational state $\to$ Projection layer $\to$ Projected geometric frame $\to$ Global invariants (`RCMT/physics_interface/pipeline/overview.md#L10-L100`).
  - `local/constraints/symmetry_chain.md`: Documents the active candidate symmetry structures ($Sp(8) \times SO(9)$ and $Sp(8) \times SO(8)$), designating the historical $E_8 \to SO(10)$ cascade as exploratory background and establishing the projection boundary (`RCMT/local/constraints/symmetry_chain.md#L45-L105`).
  - `global/invariants.md`: Classifies persistent global topological invariants (twist, winding, holonomy) that resist local LCO smoothing.
  - `smg_v2.1/`: The "Standard Model of Geometry" translation layer mapping discrete mathematical survivors to Standard Model particles and gauge groups. Governed by `smg_v2.1/smg_contract.md`.

### 2.3 `Projects/Operational_Phase_Space/` (OPS)
- **Role:** Rigorous computational laboratory testing discrete phase space models, group actions, metric rigidity, and Coxeter embeddings.
- **Key Documents & Suites:**
  - `Vocabulary_Packet.md`: Authoritative vocabulary specification freezing terms for the 8-state carrier $X_8$, cycle $C_8$, projection partition $\Pi$, involutions $p$ and $h$, symplectic form $\Omega$, and orbit bridges (`Projects/Operational_Phase_Space/Vocabulary_Packet.md#L1-L220`).
  - `V4_Orbit-Bridge_Regularity_Theorem.md`: Analytic theorem and proof establishing that a commuting independent involution $s$ produces a regular $(\mathbb{Z}_2)^3$ action on an 8-element set if and only if $s$ exchanges the two free $V_4$ orbits (`Projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#L1-L150`).
  - `SIM14.0 - SIM14.8`: The symplectic $C_8$ carrier program. Explores matchings $M_1$ and $M_2$, regular torsor actions, and stabilizer $H_{192} = C_2^3 \rtimes S_4$.
  - `SIM15.0 - SIM15.6`: External symmetry stress-testing in $W(H_4)$ (order 14,400). Discovers the unique $H_{192}$ class, threefold normalizer $N_{576}$ ($N/L \cong C_3$ triality), 25 conjugate architectures $\mathcal{L}_{25}$, and exact identification with the 25 native 24-cells in the 600-cell.
  - `SIM16.0 - SIM16.4`: Metric rigidity and $J$-complex compatibility. Proves that under strong $C_8$ adjacency constraints, the signed local metric is rigidly confined to a 1-parameter hyperbolic family $G(t) = \cosh(t) I + \sinh(t) |\Omega|$.

### 2.4 `Projects/lisi-smg-test/`
- **Role:** Integration and falsification study evaluating whether Garrett Lisi's 24-cell particle assignment embeds into the 600-cell, transports to the 120-cell, and maps onto SMG v2.1.
- **Key Documents:**
  - `task.md` & `PLAN.md`: Stated objective and falsification guidelines.
  - `REPORT_PHASE3A.md`: Maximal clique search in 600-cell compatibility graph; proves existence of exactly 25 inscribed 24-cells.
  - `REPORT_PHASE5.md`: Provenance-hidden analysis of transported 120-cell structure (24 dodecahedral fibers, 12 antipodal pairs, 96 bridge edges, order-96 automorphism group).
  - `REPORT_PHASE6.md`: Hard falsification result: no full structure-preserving map between transported geometry and SMG v2.1 exists; only an assumption-dependent forgetful projection.
  - `REPORT_PHASE6B.md` & `REPORT_PHASE6D.md`: Geometry intrinsically forces 3 generations of 4 pair carriers, with local $V_4$ symmetry and 7 configuration orbits, but explicitly *fails* to intrinsically select a $3+1$ particle partition.
  - `summary.md`: Identifies geometry recovery as verified, while identifying the Local Coherence Operator (LCO) as the primary missing dynamical piece.

---

## 3. Directional Dependencies and Anti-Backflow Boundaries

The repository enforces strict one-way authority boundaries:

```text
       [ RCF Invariant Kernel ] (Root Authority)
                 │
                 ▼
       [ RCMT Physics Core ]
                 │
        ┌────────┴────────┐
        ▼                 ▼
 [ SMG v2.1 Bridge ]   [ Operational Phase Space (OPS) ]
        │                 │
        └────────┬────────┘
                 ▼
     [ Lisi-SMG Integration ]
```

### Governing Authority Rules:
1. **RCF $\to$ RCMT**: RCF provides the pre-theoretical constraint scaffold. RCMT compiles downstream. No RCMT document may alter or weaken RCF invariants (`RCF/README.md#L12-L13`, `RCMT/README.md#L45-L55`).
2. **RCMT $\to$ SMG v2.1**: SMG v2.1 is an opt-in semantic translation bridge. Claims made within SMG (e.g. Dual Closure, Chromatic-12 particle assignments) are SMG-local and do not constitute upstream facts (`RCMT/smg_v2.1/smg_contract.md#L20-L45`).
3. **Simulations $\to$ Theory**: Computational observations in OPS and `lisi-smg-test` provide evidence for admissibility or failure under constraints. Code completion never establishes physical truth or alters upstream definitions.
