# 05 — Survivor and Failure Ledger

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/`  
**Date of Audit:** October 24, 2026  
**Source Commit SHAs:** RCF `57af06b5ac7c73b66839f70ebba3293399a1c880`, RCMT `666ed97aaa1d11f82d0a9881e99bb0c9cae2d225`

---

## 1. Research Ledger Overview

RCF is an **eliminative, constraint-first framework** (`RCF/README.md#L38-L47`):
> *"If a structure persists under increasing relational constraint, RCF allows it. If it collapses, RCF lets it go. No rescue clauses. No patching."*

This ledger records what mathematical constructions survived explicit constraint filtering, what failed, why they failed, and the conditions under which each result holds.

---

## 2. Surviving Mathematical Constructions

| Surviving Construction | Discovery Context | Governing Constraints / Invariants | Why It Survived | Precise Boundary Conditions |
|:---|:---|:---|:---|:---|
| **$V_4$ Orbit-Bridge Regularity Criterion** | OPS SIM14.3 / Analytic Proof | Relational Primacy, Finite Closure, Invariant Kernel | Proved analytically: a commuting involution $s$ produces a regular action of $(\mathbb{Z}_2)^3$ on 8 elements if and only if it completely exchanges the two free $V_4$ orbits. Partial exchange is algebraically impossible. | Applies to any 8-element set with a free $V_4$ action decomposed into $4+4$ orbits (`V4_Orbit-Bridge_Regularity_Theorem.md`). |
| **Symplectic Matching $M_2$** | OPS SIM14.0 - SIM14.3 | $C_8$ Adjacency, Symplectic Preservation | $M_2 = (02)(13)(46)(57)$ is the unique $C_8$-contained involution in $S_K$ that completely exchanges $O_0 \leftrightarrow O_1$, inducing an affine $\mathbb{F}_2^3$-torsor. | Requires frozen $C_8$ graph edges and canonical symplectic form $\Omega$ (`Vocabulary_Packet.md#L180-L200`). |
| **Affine Stabilizer $H_{192} = C_2^3 \rtimes S_4$** | OPS SIM14.6 - SIM14.8 | Spectral Dominance, Minimal Free Parameters | $H_{192} \cong \text{Aut}_{\text{aff}}(|S|)$ is the exact automorphism group preserving the unsigned symplectic form. Its conjugation action on translations $C_2^3$ is precisely $\text{Stab}_{\text{GL}(3,2)}(001) \cong S_4$. | Finite group of order 192; centralizer $C_L(T) = T$; center $Z(L) \cong C_2$ (`SIM14.8.py`). |
| **$W(H_4)$ Coxeter Embedding & Triality** | OPS SIM15.0 - SIM15.2 | Adjacency-First, External Symmetry Stress Test | Out of 1,200 $C_2^3$ subgroups in $W(H_4)$, exactly one conjugacy class of 75 subgroups realizes the $H_{192}$ fingerprint. Its normalizer $N_{576}$ exhibits $N/H_{192} \cong C_3$, proving an intrinsic $D_4$ triality symmetry. | Ambient Coxeter group $W(H_4)$ of order 14,400 (`SIM15.2.py`). |
| **Identity of 25 Architectures with Native 24-Cells ($S_{25} == N_{25}$)** | OPS SIM15.4 - SIM15.5 | Finite Structure, Polytope Compatibility | The 25 conjugate architectures $\mathcal{L}_{25}$ are literally identical to the 25 geometric stabilizers of inscribed 24-cells in the 600-cell. Subgroup intersections $\{8, 12\}$ match geometric vertex overlaps $\{0, 6\}$ bijectively. | Exact identity between 600-cell geometry and $W(H_4)$ coset algebra (`SIM15.5.py`). |
| **Hyperbolic Metric Family $G(t) = \cosh(t)I + \sinh(t)|\Omega|$** | OPS SIM16.1 - SIM16.2 | Strong $C_8$ Metric Sparsity, $N^\pm$ Signed Invariance, $G\Omega G = \Omega$ | Imposing $C_8$ graph sparsity collapses the 20-dimensional compatible metric manifold down to a single rigid 1-parameter hyperbolic family with spectrum $\{e^t: 4, e^{-t}: 4\}$. | Parameter $t \in \mathbb{R}$ uniquely determined by signed trace; sign-flip is external anti-symplectic (`SIM16.2.py`). |
| **Transported 120-Cell Fiber Geometry** | Lisi-SMG Phase 5 | Discrete Adjacency, Fiber Bundle Lifting | Lifts the 24-cell into the 120-cell, producing 480 vertices across 24 dodecahedral fibers, 12 antipodal pairs, 120 residual vertices (24 stars), and 96 bridges with an order-96 automorphism group. | Pure graph-theoretic construction from normalized adjacency tables (`REPORT_PHASE5.md`). |
| **Local 4-Carrier $V_4$ Configuration Space** | Lisi-SMG Phase 6D | Relational Mode Space, Finite Power Set | Local power set $2^4 = 16$ on 4 pair carriers in one generation block has $V_4$ symmetry and classifies into exactly 7 orbits, with occupancy-2 splitting into 3 pairs ($L_2, L_3, L_4$). | Local configuration space of an isolated generation block (`REPORT_PHASE6D.md`). |
| **Dual-Lane Candidates: $Sp(8) \times SO(9)$ & $Sp(8) \times SO(8)$** | RCMT Symmetry Revision (`b7eecd0`) | Dual-Lane Invariant, Finite Structure Invariant | Retained as active candidate survivor structures separating the unbroken symplectic mode-space lane from downstream geometric projection ($SO(9)$ or $SO(8)$). Non-compact symplectic lane reconciled under updated Section 10 invariant. | Candidates for downstream projection; selection remains open under comparative testing (`RCMT/local/constraints/symmetry_chain.md#L45-L75`). |
| **Repaired 120-Cell Coordinate Realization** | Lisi-SMG Maintenance (D11) | Native $H_4$-Root Construction, Nearest-Neighbor Invariance | 600 unique 4D rows derived from SIM15.6 native $H_4$ roots with exact discrete edge-set bijection to supplied graph IDs; repairs corruptions at IDs 327, 552, and 578. | Validated in `120cell_vertices.repaired.txt`, `coordinate_repair.json`, and regression suite (`test_coordinate_repair.py`). |

---

## 3. Failed or Abandoned Approaches

| Failed / Abandoned Construction | Elimination Context | Stated Reason for Failure / Rejection | Architectural & Practical Consequence |
|:---|:---|:---|:---|
| **Symplectic Matching $M_1$ as Regular Carrier** | OPS SIM14.2 / SIM14.3 | **Failed Orbit Bridging:** $M_1 = (01)(24)(35)(67)$ preserves the two $K$-orbits $O_0 \mapsto O_0, O_1 \mapsto O_1$. The resulting action $\langle p, h, M_1 \rangle$ is non-transitive, leaving $X_8$ fractured into two disconnected 4-state worlds. | Discarded as a candidate carrier for regular affine phase space (`Vocabulary_Packet.md#L160-L175`). |
| **Static Geometric Selection of $3+1$ Particle Partition** | Lisi-SMG Phase 6B | **Symmetry Too High (Transitive Orbit):** The verified order-96 automorphism group acts *transitively* on the four pair carriers within each generation. The static geometry provides no intrinsic basis to distinguish one carrier as a lepton and three as quarks. | **Rule Established:** A $3+1$ partition cannot be derived from static 120-cell carrier geometry alone; it strictly requires an asymmetric boundary condition or dynamical operator (`REPORT_PHASE6B.md`). |
| **Full Isomorphism: 120-Cell Geometry $\to$ SMG v2.1** | Lisi-SMG Phase 6 | **Structural Non-Equivalence:** The 36-record structure of SMG v2.1 does not match the heterogeneous carrier rank, incidence, and automorphism group of the transported 120-cell. Only an assumption-dependent forgetful projection could be constructed. | The hypothesis that SMG v2.1 is an exact geometric dual of the transported 120-cell was falsified. SMG v2.1 remains a downstream translation bridge (`REPORT_PHASE6.md`). |
| **Premature Octonion Identification ($\mathbb{O}$) with $PG(2,2)$** | OPS Discussions / zPost | **Algebraic Incompatibility:** The emergent Fano plane $PG(2,2)$ in SIM14 arises from an elementary abelian group $(\mathbb{Z}_2)^3$, which is commutative and associative. Octonions $\mathbb{O}$ require an oriented, non-commutative, non-associative multiplication law. | Strict boundary drawn: $PG(2,2) \not\Rightarrow \mathbb{O}$. SIM14 is an associative projection map, not an octonionic bulk (`Vocabulary_Packet.md#L210-L245`, `zPost_Theorem_Exploration.md`). |
| **Unconstrained Continuous Metric Manifold** | OPS SIM16.0 - SIM16.1 | **Violation of Invariant 6 (Minimal Free Parameters):** An unconstrained compatible metric space $\text{Sp}(8,\mathbb{R})/U(4)$ possesses 20 continuous tunable degrees of freedom. | Completely eliminated by enforcing $C_8$ graph locality, which reduces the degrees of freedom to a single parameter $t$ (`SIM16.1.py`). |
| **Historical Cascade: $E_8 \to SO(10) \to Sp(8) \times SO(9)$** | RCMT Symmetry Revision (`b7eecd0`) | **Dimension Incompatibility & Architectural Pruning:** $\dim(\mathfrak{so}(10)) = 45$, while $\dim(\mathfrak{sp}(8,\mathbb{R})) + \dim(\mathfrak{so}(9)) = 36 + 36 = 72$. A 45-dimensional Lie algebra cannot contain a 72-dimensional direct sum. | Formally archived as historical exploratory work rather than an architectural dependency; replaced by candidate pair $Sp(8) \times SO(9)$ and $Sp(8) \times SO(8)$ (`RCMT/local/constraints/symmetry_chain.md#L90-L100`). |
| **Mixed Output Records in Python Scripts (Resolved)** | OPS SIM16 Maintenance (D11) | **Execution Failure / Defect:** Pasting terminal output logs into the bottom of Python source files without `#` comments caused fatal `SyntaxError` on direct execution. | **Fully Resolved:** Separated executable source from historical output records; logs archived under `recorded_output/` and registered in `ARCHIVE_MANIFEST.json`. |

---

## 4. Methodological Lessons from the Ledger

1. **Failure Is Productive:** The elimination of $M_1$ directly forced the discovery of the $V_4$ Orbit-Bridge Regularity Theorem. The failure of the static $3+1$ partition in `lisi-smg-test` clearly isolated the Local Coherence Operator (LCO) as the necessary symmetry-breaking agent.
2. **Rejection of Post-Hoc Patching:** When static geometry failed to distinguish leptons from quarks, the research team resisted introducing arbitrary ad-hoc flags or manual labels, recording the result as an unambiguous negative boundary in `REPORT_PHASE6B.md`.
3. **Algebraic Rigor Over Heuristic Resemblance:** Numerological matches (e.g. 7 points in Fano plane resembling 7 imaginary octonions, or 36 modes matching 36 composite records) were repeatedly denied promotion to established equivalence without explicit structure-preserving homomorphisms.
