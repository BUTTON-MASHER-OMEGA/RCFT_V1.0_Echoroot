# 04 — Comprehensive Simulation & Test Inventory

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/`  
**Date of Audit:** October 24, 2026  
**Theory Root Path:** `C:\Users\blaur\projects\theory`  
**Git Commit SHAs:** RCF `57af06b5ac7c73b66839f70ebba3293399a1c880`, RCMT `666ed97aaa1d11f82d0a9881e99bb0c9cae2d225`

---

## 1. Test Suite: `RCF/src/python/tests/`

The upstream RCF codebase contains a validated reference implementation of discrete exterior calculus, simplicial geometry, boundary/coboundary operations, and regular 4D polytopes.

| Test File | Modules Tested | Invariants / Properties Verified | Execution Result |
|:---|:---|:---|:---:|
| `test_calculus_and_constraints.py` | `rcf_core.calculus`, `rcf_core.operators` | Boundary/coboundary nilpotency ($\partial^2 = 0, \delta^2 = 0$), discrete Stokes theorem, LCO mismatch pruning. | **PASS** (2/2 tests) |
| `test_geometry.py` | `rcf_core.geometry.simplicial`, `rcf_core.geometry.topology` | Simplicial complex closure, Euler characteristic calculations, orientability. | **PASS** (2/2 tests) |
| `test_polytopes.py` | `rcf_core.geometry.polytopes.regular_4d`, `higher_dim` | $f$-vectors and adjacency graphs of the 5-cell, 8-cell (tesseract), 16-cell, 24-cell, 120-cell, 600-cell. | **PASS** (9/9 tests) |

**Overall Status:** 13 passed in 7.58s under Python 3.12.4 (`pytest-9.0.3`).

---

## 2. Test Suite: `Projects/lisi-smg-test/tests/`

The integration pipeline between the 24-cell, 600-cell, 120-cell, and SMG v2.1 contains an automated regression suite verifying cryptographic hashes and geometric theorems.

| Test Script | Tested Phase | Target Invariant / Gate | Status |
|:---|:---|:---|:---:|
| `test_phase12.py` | Phase 1 & 2 | Vertex/cell dualization of 600-cell and 120-cell; graph isomorphism. | **PASS** |
| `test_phase3a.py` | Phase 3A | Maximal clique search in 600-cell compatibility graph; exactly 25 inscribed 24-cells. | **PASS** |
| `test_lisi_embedding.py` | Phase 3B | Embedding of Lisi 24-cell root system into 600-cell vertices. | **PASS** |
| `test_lisi_dual_transport.py` | Phase 4 | Lift of 24-cell into 120-cell across 24 dodecahedral fibers. | **PASS** |
| `test_phase5_intrinsic.py` | Phase 5 | Intrinsic geometry of 480 fiber vertices, 120 residuals, 96 bridges, order-96 automorphism group. | **PASS** |
| `test_phase6_comparison.py` | Phase 6 | Structure comparison against SMG v2.1; confirms absence of structure-preserving isomorphism. | **PASS** |
| `test_phase6b_refinement.py` | Phase 6B | Proves 3-generation block structure while confirming non-selection of local $3+1$ split. | **PASS** |
| `test_phase6c_configurations.py`| Phase 6C | Enumeration of $2^{24}$ fiber configurations $\to$ 177,976 distinct orbits under order-96 action. | **PASS** |
| `test_phase6d_local_four.py` | Phase 6D | Power set $2^4 = 16$ local configurations on 4 pair carriers; verifies $V_4$ symmetry & 7 orbits. | **PASS** |
| `test_coordinate_repair.py` | Maintenance D11 | Graph-certified coordinate reconstruction, frame isometry on 597 intact rows, repair of IDs 327/552/578. | **PASS** (5/5 tests) |

**Overall Status:** 53 passed tests in 53.90s under Python 3.12.4 with `PYTHONPATH` set to `lisi-smg-test`.

---

## 3. Operational Phase Space: SIM14 Series (Symplectic $C_8$ Carrier)

All scripts reside in `Projects/Operational_Phase_Space/`.

| Script | Primary Question Tested | Key Assumptions & Parameters | Measured / Computed Quantities | Demonstrated Result | Epistemic Status |
|:---|:---|:---|:---|:---|:---:|
| **SIM14.0.py** | Does the frozen 8-state $C_8$ carrier admit a canonical $\text{Sp}(8,\mathbb{R})$ conjugate-pair realization? | $X_8 = \{0,\dots,7\}$, cycle $C_8$, canonical form $\Omega$. | $S_{ij} = v_i^T \Omega v_j$, search for $C_8$-contained matchings. | Discovered two distinct $C_8$-contained symplectic matchings: $M_1$ and $M_2$. | Demonstrated Result |
| **SIM14.1.py** | What finite structure is induced on $X_8$ by $G = \langle p, h, s_{\Omega,2} \rangle$? | Uses matching $M_2 = (02)(13)(46)(57)$, projection $p$, half-turn $h$. | Group closure, orbit sizes, centralizers, stabilizer subgroups. | $G \cong (\mathbb{Z}_2)^3$ acts regularly on $X_8$; $X_8$ is an $\mathbb{F}_2^3$-torsor with 7 nonzero directions and 7 order-4 $V_4$ subgroups realizing $PG(2,2)$ Fano incidence. | Demonstrated Result |
| **SIM14.2.py** | Does matching $M_1 = (01)(24)(35)(67)$ produce the same regular torsor action as $M_2$? | Swaps input matching from $M_2 \to M_1$. | Orbit partition and transitivity of $\langle p, h, M_1 \rangle$. | **Bifurcation discovered:** $\langle p, h, M_1 \rangle \cong (\mathbb{Z}_2)^3$, but acts non-transitively, preserving $4+4$ orbits $O_0 \sqcup O_1$. | Demonstrated Result |
| **SIM14.3.py** | What controls the regularity vs. non-regularity bifurcation across all possible matchings? | Exhaustive census across all 105 perfect matchings on $X_8$. | Orbit transition matrix $B(s)$, commutant class $S_K$ ($|S_K|=13$). | **$V_4$ Orbit-Bridge Regularity Criterion:** $s(O_0) = O_1 \iff \langle K, s \rangle$ acts regularly on $X_8$. 4 complete bridges are regular; 9 non-bridges are non-regular. | Demonstrated Result (Later proved analytically) |
| **SIM14.4.py** | What is the exhaustive static affine geometry of the $M_2$ carrier before dynamics? | Frozen $M_2$ torsor. No target Lie algebra imposed. | 14 affine planes ($P_{14}$), 7 directions ($D_7$), lines ($L_{28}$). | Complete affine coordinate dictionary established without continuous fitting. | Demonstrated Result |
| **SIM14.5.py** | Does affine translation commute with symplectic pairing? | Affine translations vs. symplectic form $\Omega$. | Equivariance and signed parity of affine translations. | Translations act as signed symplectomorphisms; classified signed group order. | Demonstrated Result |
| **SIM14.6.py** | What is the full automorphism group of the unsigned pairing $|S|$? | Stabilizer of unsigned pairing $|S|$ on $X_8$. | $\text{Aut}_{\text{aff}}(|S|)$, order, normalizers, centralizers. | $\text{Aut}_{\text{aff}}(|S|) = H_{192}$ with $|H_{192}| = 192$. Normal subgroup $G \cong C_2^3$, quotient $H_{192}/G \cong S_4$. | Demonstrated Result |
| **SIM14.7.py** | What is the intersection $V_\Omega = G \cap N^+$? | Signed symplectic normalizer $N^+$ (order 24) vs. translations $G$. | Normality, conjugation action of $N^+$ on $V_\Omega$. | $V_\Omega$ is the unique normal $V_4$ in $N^+$; conjugation action gives $N^+ / V_\Omega \cong S_3$. | Demonstrated Result |
| **SIM14.8.py** | What is the exact internal origin of $H_{192}$ in $\text{GL}(3,2)$? | Action of $H_{192}$ by conjugation on $G \cong C_2^3$. | Image in $\text{Aut}(C_2^3) \cong \text{GL}(3,2)$ (order 168). | Conjugation image is exactly $\text{Stab}_{\text{GL}(3,2)}(001) \cong S_4$ (order 24); unique fixed non-zero vector $v \in C_2^3$, $Z(H_{192}) = \langle v \rangle \cong C_2$. | Demonstrated Result |

---

## 4. Operational Phase Space: SIM15 Series (Coxeter $H_4$ Survivability)

All scripts reside in `Projects/Operational_Phase_Space/`.

| Script | Primary Question Tested | Key Assumptions & Parameters | Measured / Computed Quantities | Demonstrated Result | Epistemic Status |
|:---|:---|:---|:---|:---|:---:|
| **SIM15.0.py** | Does the frozen SIM14.8 $H_{192}$ architecture survive inside $W(H_4)$? | Independent construction of Coxeter group $W(H_4)$ (order 14,400). | Census of 1,200 $C_2^3$ subgroups; normalizers; centralizers. | **Direct Survivability: YES.** Exactly one conjugacy class (orbit size 75) realizes $H_{192} = C_2^3 \rtimes S_4$. Conjugacy class of $H_{192}$ has size 25; normalizer $|N| = 576$; quotient $N/H_{192} \cong C_3$. | Demonstrated Result |
| **SIM15.1.py** | What is the geometric action of the threefold normalizer $N_{576}$? | Blind reconstruction of $T \cong C_2^3, L \cong H_{192}, N \cong N_{576}$. | Orbit decomposition on 600-cell vertices (120) and 120-cell vertices (600). | $C_3 \cong N/L$ cyclically permutes three distinguished $C_2^3$ kernels inside $N_{576}$ and acts as an ambient threefold symmetry on geometry. | Demonstrated Result |
| **SIM15.2.py** | Does $L$ identify with $W(D_4)$ and is $C_3$ standard $D_4$ triality? | Independent $D_4$ root system; order-5 element action. | Coxeter generators; outer automorphism $\text{Out}(W(D_4))$; order-5 action on 25 conjugates $\mathcal{L}_{25}$. | **$L \cong W(D_4)$ confirmed.** $C_3$ acts as standard $D_4$ triality permuting the three outer nodes. Order-5 elements act as fixed-point-free $5^5$ cycles on $\mathcal{L}_{25}$. | Demonstrated Result |
| **SIM15.3.py** | Can the rank-3 incidence on $\mathcal{L}_{25}$ be reconstructed from kernel intersections? | 25 conjugate architectures $\mathcal{L}_{25}$. | Pairwise intersection orders $|L_i \cap L_j| \in \{8, 12, 192\}$. | Intersection sizes $\{8, 12\}$ define an invariant rank-3 strongly regular graph on the 25 architectures. | Demonstrated Result |
| **SIM15.4.py** | Does the family of 25 architectures $\mathcal{L}_{25}$ coincide with native 24-cells in the 600-cell? | Blind census of inscribed 24-cells in 600-cell. | Geometric stabilizers $S_{25}$ vs. algebraic normalizers $N_{25}$. | **Literal Equality: $S_{25} == N_{25}$.** Object-for-object identity between native 24-cells and $H_4$ triality normalizers. | Demonstrated Result |
| **SIM15.5.py** | Does native 24-cell intersection geometry ($0$ or $6$ shared vertices) match the algebraic $\{8, 12\}$ intersections? | 25 native 24-cells $C_i$ vs. $L_i$. | Geometric vertex overlap $|C_i \cap C_j|$ vs. algebraic subgroup intersection $|L_i \cap L_j|$. | Exact bijection: overlap of 0 vertices $\iff |L_i \cap L_j| = 8$; overlap of 6 vertices $\iff |L_i \cap L_j| = 12$. | Demonstrated Result |
| **SIM15.6.py** | Can triality and incidence be verified on a single native cell without ambient census? | Single native 24-cell and its $W(D_4)$ stabilizer. | Local triality cycles on tetrahedral cells. | Self-contained local verification completed. | Demonstrated Result |

---

## 5. Operational Phase Space: SIM16 Series (Metric Rigidity & Complex Structures)

All scripts reside in `Projects/Operational_Phase_Space/`.

| Script | Primary Question Tested | Key Assumptions & Parameters | Measured / Computed Quantities | Demonstrated Result | Epistemic Status |
|:---|:---|:---|:---|:---|:---:|
| **SIM16.0.py** | Does $J$-complex structure admit continuous deformations under signed symmetry? | Exact linear algebra; signed permutation group $N^\pm$ (order 48). | Commutant dimension; invariant symmetric and tangent subspaces. | Commutant dimension 8; symmetric invariant subspace dimension 4; tangent dimension 2. | Demonstrated Result |
| **SIM16.1.py** | What is the exact metric family surviving strong $C_8$ graph locality? | Strong $C_8$ metric sparsity ($G_{ij}=0$ for non-edges); $G\Omega G = \Omega, G > 0$. | Quadratic compatibility equations in metric coordinates. | **Strict Rigidity Theorem:** Local metric family is unique 1-parameter hyperbolic family $G(t) = \cosh(t) I + \sinh(t) |\Omega|$ for $t \in \mathbb{R}$. | Demonstrated Result (Exact symbolic proof) |
| **SIM16.2.py** | Is the metric parameter $t$ operationally identifiable and algebraically closed? | Hyperbolic metric family $G(t)$. | Signed trace, spectrum of $G(t)$, stochastic non-isometry witness. | Signed trace uniquely recovers $t$ via strictly increasing $\sinh(t)$; spectrum is $\{e^t: 4, e^{-t}: 4\}$. Sign-flip is external anti-symplectic. | Demonstrated Result |
| **SIM16.3.py** | Are projection pairing $p$ and complex structure $J$ compatible? | Frozen projection $p$, cycle $C_8$, hyperbolic metric. | Commutator $[p, J(t)]$, basin projectors. | Verified exact projector compatibility conditions and Gate H $C_8$ adjacency closure. | Demonstrated Result |
| **SIM16.4.py** | Can local symplectic generators transport the metric family $G(t)$? | Lie algebra $\mathfrak{sp}(8,\mathbb{R})$; local basis elements. | Infinitesimal generators commuting with signed graph symmetries. | Identified exact local symplectic generators that generate the hyperbolic boost parameter $t$. | Demonstrated Result |

---

## 6. Resolution of Script Execution and Syntax Caveat (Decision D11)

During initial reconnaissance, an operational defect was identified in the SIM16 series:
- Terminal simulation logs and `TRUTH_PACKET_JSON` output strings had been pasted un-commented to the bottom of the Python scripts, triggering a fatal `SyntaxError`.
- **Resolution via Decision D11:** Under the owner-authorized maintenance pass, all 21 OPS scripts (`SIM14.0` through `SIM16.4`) were refactored to separate executable code from historical output. Trailing un-commented logs were moved into dedicated text files under `Projects/Operational_Phase_Space/recorded_output/` and indexed in `ARCHIVE_MANIFEST.json`.
- **Verification:** Direct execution of `python SIM16.1.py` now completes cleanly with exit code 0 and passes all exact algebraic classification gates without requiring external harnesses or patching.
