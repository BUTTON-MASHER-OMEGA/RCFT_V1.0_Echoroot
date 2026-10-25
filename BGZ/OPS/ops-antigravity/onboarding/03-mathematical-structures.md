# 03 — Mathematical Structures and Evidence Classification

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/`  
**Date of Audit:** October 24, 2026  
**Source Commit SHAs:** RCF `c419353214ee83d83376747064912544aee257b9`, RCMT `b7eecd0dfd8080864b34441af86329f819cba65a`

---

## 1. Epistemic Classification Scheme

In accordance with Section 3 of the onboarding charter, every mathematical structure used in RCF, RCMT, OPS, and Lisi-SMG must be strictly categorized into one of six epistemic tiers:

1. **Tier 1: Established Mathematics** — Standard definitions, proofs, and theorems.
2. **Tier 2: Demonstrated Computational Results** — Behaviors exhaustively verified or reproduced by code.
3. **Tier 3: Framework Definitions** — Project-specific conventions, laboratory baselines, and contracts.
4. **Tier 4: Working Hypotheses** — Proposed conjectures requiring analytical or numerical proof.
5. **Tier 5: Interpretive Mappings** — Downstream physical or phenomenological identifications.
6. **Tier 6: Failed or Abandoned Approaches** — Constructions ruled out by constraints or explicit falsification.

---

## 2. Comprehensive Structure Inventory

| Mathematical Structure | Domain / Context | Epistemic Tier | Precise Formal Definition / Role | Evidence / Verification Status |
|:---|:---|:---:|:---|:---|
| **Symplectic Form $\Omega$** | OPS / RCMT | **Tier 1** | Nondegenerate alternating bilinear form on $\mathbb{R}^8$: $\Omega = \begin{pmatrix} 0 & I_4 \\ -I_4 & 0 \end{pmatrix}, \Omega^T = -\Omega$. Preserved by $Sp(8, \mathbb{R})$. | Standard symplectic geometry; verified in SIM14.0 (`SIM14.0.py`). |
| **Fano Plane $PG(2,2)$** | OPS / SIM14 | **Tier 1** | Unique projective plane over $\mathbb{F}_2$: 7 points (nonzero elements of $\mathbb{F}_2^3$), 7 lines (order-4 $V_4$ subgroups). | Standard projective geometry; derived analytically in `V4_Orbit-Bridge_Regularity_Theorem.md#L100-L135`. |
| **$V_4$ Orbit-Bridge Regularity** | OPS / SIM14 | **Tier 1** | For $K \cong V_4$ acting freely on 8 elements with orbits $O_0, O_1$ and commuting involution $s \notin K$: $G = \langle K, s \rangle \cong (\mathbb{Z}_2)^3$ acts regularly $\iff s(O_0) = O_1$. | Analytically proved in `V4_Orbit-Bridge_Regularity_Theorem.md`. |
| **Coxeter Group $W(H_4)$** | OPS / SIM15 | **Tier 1** | Non-crystallographic reflection group in $\mathbb{R}^4$ of order 14,400. Invariance group of the 600-cell and 120-cell. | Standard Coxeter theory; verified computationally in `SIM15.0.py`. |
| **$W(D_4)$ Triality** | OPS / SIM15 | **Tier 1** | Weyl group $W(D_4)$ (order 192) having outer automorphism group $\text{Out}(W(D_4)) \cong S_3$, containing order-3 cyclic triality permuting the three 8D representations. | Standard Lie algebra / Coxeter theory; verified constructively in `SIM15.2.py`. |
| **Hyperbolic Metric Family $G(t)$** | OPS / SIM16 | **Tier 1 / Tier 2** | Local metric family $G(t) = \cosh(t) I + \sinh(t) |\Omega|$ ($t \in \mathbb{R}$) satisfying $G \Omega G = \Omega, G > 0, J = -\Omega G, J^2 = -I$. | Proved analytically and verified symbolically in SymPy in `SIM16.1.py` and `SIM16.2.py`. |
| **25 Inscribed 24-Cells in 600-Cell** | OPS / Lisi-SMG | **Tier 2** | Maximal clique enumeration in 600-cell compatibility graph yields exactly 25 inscribed 24-cells ($\mathcal{L}_{25}$), identical to the 25 $W(D_4)$-normalizers $N_{576}$. | Computationally proven in `SIM15.4.py`, `SIM15.5.py`, and `lisi-smg-test/REPORT_PHASE3A.md`. |
| **Transported 120-Cell Fiber Graph** | Lisi-SMG / Phase 5 | **Tier 2** | Transport of 24-cell $\to$ 120-cell induces 24 dodecahedral fibers (480 vertices), 12 antipodal pairs, 96 bridge edges, and an order-96 automorphism group. | Independently verified from adjacency tables in `lisi-smg-test/REPORT_PHASE5.md`. |
| **Local 4-Carrier $V_4$ Action** | Lisi-SMG / Phase 6D | **Tier 2** | Local configuration space on 4 antipodal pair carriers has $V_4$ symmetry, generating 7 orbits across 16 configurations ($2^4$). | Exhaustively computed and verified by tests in `lisi-smg-test/REPORT_PHASE6D.md`. |
| **$X_8$ Carrier & Cycle $C_8$** | OPS / SIM14 | **Tier 3** | Discrete 8-state set $X_8 = \{0,\dots,7\}$ and frozen 8-cycle graph $E(C_8) = \{01, 02, 13, 24, 35, 46, 57, 67\}$. | Frozen project laboratory definition (`Vocabulary_Packet.md#L10-L60`). |
| **Projection Partition $\Pi$ and Involution $p$** | OPS / SIM14 | **Tier 3** | Partition $\Pi = \{\{0,1\}, \{2,3\}, \{4,5\}, \{6,7\}\}$ and involution $p = (01)(23)(45)(67)$. | Project definition representing discrete coarse-graining (`Vocabulary_Packet.md#L70-L95`). |
| **Chromatic-12 & 36 Composite Modes** | SMG v2.1 | **Tier 3** | Indexing structure: 12 chromatic indices partitioned into 3 gamma plateaus, yielding a 36-mode composite matrix. | Downstream translation definition (`RCMT/smg_v2.1/smg_contract.md#L15-L35`). |
| **Candidate Dual-Lane Structures: $Sp(8) \times SO(9)$ & $Sp(8) \times SO(8)$** | RCMT Symmetry | **Tier 4** | Competing candidate structures for unbroken mode-space ($Sp(8)$) and downstream geometric projection ($SO(9)$ or $SO(8)$). | Active working hypotheses under comparative constraint testing (`RCMT/local/constraints/symmetry_chain.md#L45-L75`). |
| **Historical Cascade: $E_8 \to SO(10) \to Sp(8) \times SO(9)$** | RCMT Symmetry | **Tier 6** | Prior branching cascade proposing embedding into $E_8$ via $SO(10)$. | **Archived / Non-Dependency**: Relegated to historical exploratory work; Lie dimension mismatch ($\dim SO(10)=45 < 72$) precluded standard subalgebra containment. |
| **Octonion $\mathbb{O}$ Identification with Fano Plane** | OPS / Discussion | **Tier 4 / Tier 5** | Conjecture that the emergent $PG(2,2)$ incidence encodes octonionic multiplication in the non-associative bulk. | **Unearned hypothesis**; explicitly fenced in `Vocabulary_Packet.md#L210-L240` and `zPost_Theorem_Exploration.md#L1-L40`. |
| **SMG Dual Closure & Standard Model Mapping** | SMG v2.1 | **Tier 5** | Mapping Chromatic-12 to $8+3+1$ ($SU(3) \times SU(2) \times U(1)$) and gamma plateaus to 3 fermion generations. | Interpretive mapping; no continuous gauge connection or dynamic derivation exists (`RCMT/smg_v2.1/`). |
| **Static Geometric Selection of $3+1$ Partition** | Lisi-SMG / Phase 6 | **Tier 6** | Attempt to select a $3+1$ color/lepton or spatial/temporal split from static 120-cell carrier geometry. | **Falsified**: local action on the 4 pair carriers in each generation is transitive; requires symmetry-breaking operator (`REPORT_PHASE6B.md`). |
| **Full Structure-Preserving Isomorphism (120-Cell $\to$ SMG)** | Lisi-SMG / Phase 6 | **Tier 6** | Attempt to map the 24-cell transported geometry isomorphically onto SMG v2.1 36-mode structure. | **Falsified**: only an assumption-dependent forgetful projection survives (`REPORT_PHASE6.md`). |
| **M1 Symplectic Matching as Regular Carrier** | OPS / SIM14 | **Tier 6** | Using conjugate matching $M_1 = (01)(24)(35)(67)$ to generate a regular carrier action. | **Falsified**: $M_1$ preserves the $4+4$ orbits, failing transitivity; only $M_2$ survives regularity (`V4_Orbit-Bridge_Regularity_Theorem.md`). |

---

## 3. Detailed Assessment of Key Mathematical Layers

### 3.1 The $V_4 \to (\mathbb{Z}_2)^3 \to PG(2,2)$ Cascade
The discrete carrier program in SIM14 established an airtight chain of standard mathematical results:
1. Two commuting involutions $p$ (projection pairing) and $h$ ($C_8$ half-turn) generate Klein four-group $K = \langle p, h \rangle \cong V_4$.
2. The action of $K$ on the 8-state carrier $X_8$ is fixed-point free and decomposes $X_8$ into two 4-state orbits $O_0 = \{0,1,6,7\}$ and $O_1 = \{2,3,4,5\}$.
3. The commutant class $S_K$ of fixed-point free involutions commuting with $K$ contains 13 involutions.
4. Exactly 4 of these exchange $O_0 \leftrightarrow O_1$ (orbit bridges).
5. The **$V_4$ Orbit-Bridge Regularity Theorem** proves that $G = \langle K, s \rangle \cong (\mathbb{Z}_2)^3$ acts regularly on $X_8$ if and only if $s$ is an orbit bridge.
6. When $s$ is an orbit bridge (e.g. $M_2$), $X_8$ becomes an affine $\mathbb{F}_2^3$-torsor. Projectivization immediately yields the 7 points and 7 lines of the Fano plane $PG(2,2)$.

### 3.2 The $W(H_4)$ Embedding and Native 24-Cell Identification
In SIM15, the project verified that:
1. $W(H_4)$ has order 14,400 and contains 1,200 subgroups isomorphic to $C_2^3$, partitioned into 5 conjugacy classes.
2. Exactly one conjugacy class (orbit size 75) possesses the complete stabilizer fingerprint $H_{192} = C_2^3 \rtimes S_4 \cong \text{Aut}_{\text{aff}}(|\Omega|)$.
3. The normalizer $N = N_{W(H_4)}(H_{192})$ has order 576. The quotient $N / H_{192} \cong C_3$ is the outer $D_4$ triality automorphism.
4. The conjugacy class of $H_{192}$ contains exactly 25 architectures $\mathcal{L}_{25}$ ($75 / 3 = 25$).
5. In SIM15.4 and SIM15.5, these 25 architectures were proven to be **identical** to the 25 geometric stabilizers of the 25 native 24-cells inscribed in the 600-cell ($S_{25} == N_{25}$).

### 3.3 The Metric Rigidity Theorem (SIM16)
In SIM16, the project investigated the space of symmetric positive-definite metrics $G > 0$ on $\mathbb{R}^8$ that are compatible with the canonical symplectic form $\Omega$ ($G \Omega G = \Omega$), yielding an almost complex structure $J = -\Omega G$ with $J^2 = -I$:
1. Unconstrained, the space of compatible metrics $\text{Sp}(8,\mathbb{R}) / U(4)$ has dimension 20.
2. Enforcing invariance under the signed group $N^\pm$ (order 48) collapses the space of symmetric matrices to dimension 4.
3. Imposing **strong $C_8$ metric locality** (requiring $G_{ij} = 0$ for all non-adjacent pairs on the frozen $C_8$ cycle) reduces the admissible metric family to a **unique 1-parameter hyperbolic family**:
   $$G(t) = \cosh(t) I + \sinh(t) |\Omega|, \quad t \in \mathbb{R}$$
4. The spectrum of $G(t)$ is rigidly locked to $\{e^t: 4, e^{-t}: 4\}$, and the parameter $t$ is uniquely recoverable from the signed trace.
