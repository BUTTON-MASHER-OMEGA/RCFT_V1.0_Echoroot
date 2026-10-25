# 06 — Outstanding Open Research Questions

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/`  
**Date of Audit:** October 24, 2026  
**Source Commit SHAs:** RCF `c419353214ee83d83376747064912544aee257b9`, RCMT `b7eecd0dfd8080864b34441af86329f819cba65a`

---

## 1. Prioritized Matrix of Open Questions

The following research questions are ranked by architectural and mathematical priority:

| Priority | Question Title | Primary Domain | Significance & Impact on Research Program |
|:---:|:---|:---:|:---|
| **P1** | **Mathematical Specification of the Local Coherence Operator (LCO)** | RCMT / OPS / Geometry | **The central missing link.** The static geometric substrate is verified, but the dynamical operator that performs mismatch pruning, mode selection, and breaks local $V_4$ symmetry into a $3+1$ partition does not yet exist. |
| **P2** | **Comparative Evaluation of Geometric Lane Candidates: $SO(9)$ vs. $SO(8)$** | RCMT Symmetry | The historical $E_8 \to SO(10)$ cascade has been archived; RCMT now poses two active candidate pairs: $Sp(8) \times SO(9)$ and $Sp(8) \times SO(8)$. Determining which geometric candidate admits an admissible projection to $SO(3,1) \times SO(4)$ under discrete adjacency constraints is a major open task. |
| **P3** | **Coupling Between OPS 8-State Metric and 120-Cell Fibers** | OPS $\leftrightarrow$ Lisi-SMG | In SIM16, the hyperbolic metric family $G(t) = \cosh(t)I + \sinh(t)|\Omega|$ is defined on an 8-state carrier $X_8$. How this 8D carrier maps onto the 4-carrier local systems (Phase 6D) or the 24 dodecahedral fibers is unformulated. |
| **P4** | **Operational Observability of the Metric Parameter $t$** | OPS SIM16.2 | SIM16.2 proved mathematical identifiability of $t$ via signed trace, but classified operational observability as explicitly "UNRESOLVED". It is unknown whether an internal observer can measure $t$ without external reference. |
| **P5** | **Orientation Lifting on $PG(2,2)$ and the Non-Associative Bulk** | OPS Carrier / Algebra | The emergent Fano plane in SIM14 is unoriented and associative ($(\mathbb{Z}_2)^3$). Determining whether a non-trivial 3-cocycle or orientation can be naturally induced to recover octonionic multiplication $\mathbb{O}$ remains an open question. |
| **P6** | **SMG Refactor to Match Recovered Geometry** | SMG v2.1 / Geometry | Phase 6 proved that SMG v2.1 does not share a structure-preserving isomorphism with the transported 120-cell. SMG must be refactored to consume the 24-fiber / 12-pair / 3-generation geometry directly. |

---

## 2. In-Depth Technical Analysis of Top Research Questions

### Question 1: The Local Coherence Operator (LCO) Specification (Priority 1)
- **Current State:**
  - In `lisi-smg-test/summary.md`, the audit concluded: *"LCO remains the primary missing component... The geometry itself does not determine the operator."*
  - In `lisi-smg-test/REPORT_PHASE6B.md`, it was proven that the static geometry of the transported 120-cell is too symmetric: the local automorphism group acts *transitively* on the four pair carriers of each generation, meaning static geometry cannot distinguish one lepton from three quarks.
  - In `RCMT/physics_interface/pipeline/overview.md`, LCO is defined abstractly as an operator that prunes local mismatches and enforces coherence, but lacks an exact algorithmic implementation on the polytope configuration space.
- **Specific Technical Problem:**
  Construct an explicit discrete operator $L_{\text{co}}$ acting on the local $2^4 = 16$ configuration space (or the global $2^{24}$ fiber configuration space) such that:
  1. It acts strictly locally on adjacent relational patches (`RCMT/physics_interface/pipeline/overview.md#L30-L40`).
  2. It preserves persistent global topological invariants (Twist, Winding, Holonomy) (`RCMT/global/invariants.md`).
  3. It dynamically breaks the transitive $V_4$ symmetry of the 4 pair carriers, selecting a stable $3+1$ ground state without imposing arbitrary external flags.

---

### Question 2: Comparative Evaluation of Geometric Lane Candidates ($SO(9)$ vs. $SO(8)$) (Priority 2)
- **Current State:**
  - The repository update (`b7eecd0`) formally relegated the prior cascade $E_8 \to SO(10) \to Sp(8) \times SO(9)$ to historical exploratory work, removing the Lie algebra dimension mismatch ($\dim 45 < 72$) from active dependencies (`RCMT/local/constraints/symmetry_chain.md#L90-L100`).
  - Active candidate structures are now explicitly framed as:
    $$\mathbf{Candidate\ A:} \quad Sp(8) \times SO(9)$$
    $$\mathbf{Candidate\ B:} \quad Sp(8) \times SO(8)$$
  - In both candidates, $Sp(8)$ is the unbroken mode-space lane, while $SO(9)$ or $SO(8)$ serves as the geometric lane candidate targeting a low-energy projection to $SO(3,1) \times SO(4)$.
  - Neither $SO(9)$ nor $SO(8)$ is assumed to contain $SO(3,1)$ via compact symmetry breaking; the Lorentzian structure is an explicit **projection-level requirement**.
- **Specific Technical Problem:**
  Determine mathematical and operational criteria to decide between $SO(9)$ and $SO(8)$:
  1. **$SO(8)$ Advantages:** $SO(8)$ possesses the unique $D_4$ triality outer automorphism ($\text{Out}(SO(8)) \cong S_3$), directly matching the $D_4$ triality uncovered in SIM15.2 (`N_{576}/H_{192} \cong C_3$) and the 3-generation block structure of the transported 120-cell (Phase 6B). It branches naturally to $SO(4) \times SO(4) \cong SU(2)^4$.
  2. **$SO(9)$ Features:** $SO(9)$ has dimension 36, exactly matching the 36 dimensions of the real symplectic algebra $\mathfrak{sp}(8,\mathbb{R})$ (producing a symmetric $36 \times 36$ dimensional pairing), and provides a natural vector representation for 9-dimensional boundaries.
  3. **Projection Realization:** Construct the explicit projection map from the chosen geometric candidate down to local Lorentzian frame $SO(3,1)$ and internal compact symmetry $SO(4)$ without postulating an upstream metric or time manifold.

---

### Question 3: Connecting the OPS Hyperbolic Metric to the 120-Cell Configuration Space (Priority 3)
- **Current State:**
  - OPS SIM16 established that on the 8-state carrier $X_8$, enforcing strong $C_8$ graph locality and signed symmetry $N^\pm$ strictly forces the compatible metric into the 1-parameter hyperbolic family:
    $$G(t) = \cosh(t) I_8 + \sinh(t) |\Omega_8|, \quad t \in \mathbb{R}$$
  - Meanwhile, `lisi-smg-test/REPORT_PHASE6D.md` established that each of the three generations in the 120-cell consists of 8 fibers partitioned into 4 antipodal pair carriers with a local $V_4$ group action.
- **Specific Technical Problem:**
  - Does the 8-state carrier $X_8$ in SIM16 represent the 8 dodecahedral fibers of a single generation block?
  - If so, how does the 8-cycle $C_8$ map onto the inter-fiber bridge connections in the 120-cell?
  - Can the hyperbolic parameter $t$ be identified with an operational boost or scale factor across the three generations?

---

### Question 4: Operational Observability of Metric Parameter $t$ (Priority 4)
- **Current State:**
  - In `SIM16.2.py`, Gate B proves that the signed trace $\text{Tr}(J_0 J(t)) = 8 \sinh(t)$ is strictly increasing on $\mathbb{R}$, which provides a one-to-one mathematical recovery of $t$.
  - However, Gate H explicitly marks operational observability as **"UNRESOLVED"**:
    `TRUTH_PACKET_JSON: {"operational_observability": "UNRESOLVED"}`
- **Specific Technical Problem:**
  - In a discrete relational system without an external background observer, every observable must be an internal relation (Invariant 7: Non-Privileged Observers).
  - Can an internal subsystem, interacting solely through discrete adjacency and local boundary projectors, measure $t$?
  - Does $t$ represent a physical hyperbolic rapidity, a thermodynamic temperature/coherence scale, or an unobservable gauge parameter?

---

### Question 5: Orientation, 3-Cocycles, and the Octonionic Bulk (Priority 5)
- **Current State:**
  - `SIM14.1` derived the Fano plane $PG(2,2)$ from the regular torsor action $G \cong (\mathbb{Z}_2)^3$.
  - `zPost_Theorem_Exploration.md` and `Vocabulary_Packet.md` established that this Fano plane has *unoriented* incidence and is strictly associative and commutative.
  - Standard octonions $\mathbb{O}$ require an orientation on each Fano line specifying $e_i e_j = -e_j e_i = e_k$.
- **Specific Technical Problem:**
  - Can the orientation of the 7 lines of $PG(2,2)$ be induced by the symplectic form $\Omega$ or the graph half-turn $h$?
  - In `zPost_Theorem_Exploration.md`, it is conjectured that the $(\mathbb{Z}_2)^3$ torsor is a quaternionic associative projection of a deeper non-associative bulk. Does there exist a formal discrete boundary map that connects a non-associative structure to this associative boundary?

---

### Question 6: SMG Refactoring (Priority 6)
- **Current State:**
  - `lisi-smg-test/REPORT_PHASE6.md` proved that the existing SMG v2.1 specification (`references/smg_v2.1_concat.md`) cannot be mapped isomorphically onto the recovered 120-cell fiber geometry.
  - SMG v2.1 relies on a rigid 36-mode index set, whereas the transported geometry provides a rich hierarchy: 600-cell $\to$ 24 fibers $\to$ 12 pair carriers $\to$ 3 generation blocks $\to$ 4-carrier configuration spaces.
- **Specific Technical Problem:**
  - Draft SMG v3.0, replacing the abstract 36-mode heuristic matrix with the verified mathematical objects from Phase 5 and Phase 6.
  - Define exact translation rules mapping the 24 dodecahedral fibers and 12 antipodal pairs into Standard Model quantum numbers (color, isospin, hypercharge) while respecting the negative result of Phase 6B.
