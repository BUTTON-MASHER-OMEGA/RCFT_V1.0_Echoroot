# OPS Research Assignment 001: Technical Report
## $SO(8)$ vs. $SO(9)$: Candidate Discrimination

**Agent:** Antigravity  
**Workspace:** `Projects/ops-antigravity/research/`  
**Date:** October 24, 2026  
**Status:** Authorized Exploratory Research  
**Authority:** Research findings only; no architectural promotion  
**Test Scripts:** 
- `so8_vs_so9_discrimination.py`
- `weyl_triality_audit.py`

---

## A. Existing Evidence

The comparative evaluation of the candidate geometric lanes draws on explicit mathematical results across SIM14–SIM16 and the 120-cell transport study:

1. **The 8-State Carrier and Canonical Symplectic Form (SIM14.0–14.3):**
   - Microscopic carrier $X_8 = \{0,\dots,7\}$, vector space $V = \mathbb{R}^8$.
   - Canonical symplectic form $\Omega = \begin{pmatrix} 0 & I_4 \\ -I_4 & 0 \end{pmatrix}$.
   - Continuous automorphism group preserving $\Omega$ is $\text{Sp}(8, \mathbb{R})$ [**Verified**; `SIM14.0.py`].
   - Commuting involutions $p$ and $h$ generate $K \cong V_4$ acting freely as $4+4$ ($X_8 = O_0 \sqcup O_1$). The orbit-bridge matching $M_2$ extends $K$ to a regular affine torsor $G = \langle p, h, M_2 \rangle \cong (\mathbb{Z}_2)^3$ [**Verified**; `V4_Orbit-Bridge_Regularity_Theorem.md`].

2. **The Operational Stabilizer $H_{192}$ and $W(D_4)$ Triality (SIM14.6–SIM15.2):**
   - The affine stabilizer of the unsigned symplectic pairing $|S|$ on $X_8$ is $H_{192} = C_2^3 \rtimes S_4 \cong \text{Aut}_{\text{aff}}(|S|)$ [**Verified**; `SIM14.6.py`, `SIM14.8.py`].
   - In SIM15.0–15.2, $H_{192}$ was embedded into Coxeter $W(H_4)$ (order 14,400). Exactly one conjugacy class of 75 subgroups realized this fingerprint.
   - Crucial algebraic identification: **$H_{192} \cong W(D_4)$**, the Weyl group of $SO(8)$ [**Verified**; `SIM15.2.py`].
   - Outer normalizer $N_{576} = N_{W(H_4)}(H_{192})$ exhibits $N_{576}/H_{192} \cong C_3$, proved constructively to be the **$D_4$ triality outer automorphism** [**Verified**; `SIM15.2.py`].

3. **Inscribed 24-Cell Identification (SIM15.4–15.5):**
   - The 25 conjugate architectures $\mathcal{L}_{25}$ in $W(H_4)$ are literally identical to the 25 geometric stabilizers $S_{25}$ of native 24-cells inscribed in the 600-cell ($S_{25} == N_{25}$) [**Verified**; `SIM15.5.py`].

4. **Exact Metric Rigidity (SIM16.0–16.2):**
   - Metric space on $\mathbb{R}^8$ satisfying $G\Omega G = \Omega, G > 0, J = -\Omega G, J^2 = -I$.
   - The unconstrained space of compatible metrics is the symmetric space $\text{Sp}(8,\mathbb{R})/U(4)$ of dimension 20 [**Verified**; `SIM16.0.py`].
   - Under signed permutation symmetry $N^\pm$ (order 48) and strong $C_8$ graph locality, the metric manifold collapses to a unique 1-parameter hyperbolic family $G(t) = \cosh(t)I + \sinh(t)|\Omega|$ [**Verified**; `SIM16.1.py`].

5. **RCMT Dual-Lane Contract (`rcmt/local/constraints/symmetry_chain.md`):**
   - Mode-space lane: $Sp(8,\mathbb{R})$ (unbroken).
   - Geometric lane: Candidate A ($SO(9)$) vs. Candidate B ($SO(8)$).
   - Projection target: Low-energy geometric frame $SO(3,1) \times SO(4)$ [**Candidate**; `symmetry_chain.md#L45-L75`].

---

## B. Candidate Definitions

The two candidate constructions under comparison are:

### Candidate A: $Sp(8, \mathbb{R}) \times SO(9)$
- **Mode-Space Factor:** $Sp(8, \mathbb{R})$, split real form of type $C_4$. Dimension 36, rank 4. Acts faithfully on the 8-state carrier $\mathbb{R}^8$.
- **Geometric Factor:** $SO(9)$, compact real form of type $B_4$. Dimension 36, rank 4.
- **Underlying Spaces:**
  - $Sp(8, \mathbb{R})$ acts on $\mathbb{R}^8$.
  - $SO(9)$ has fundamental representations: vector $\mathbf{9}$ (dim 9) and real spinor $\mathbf{16}$ (dim 16).
- **Documented Rationale:** Originally proposed because $\dim \mathfrak{so}(9) = 36$ matches $\dim \mathfrak{sp}(8,\mathbb{R}) = 36$ [**Candidate**].
- **Unresolved Ambiguity:** The carrier space for $SO(9)$ is not documented. Since $SO(9)$ has no 8D representation, it cannot act on $\mathbb{R}^8$ without extending the carrier (e.g. to $\mathbb{R}^9 \cong \mathbb{R}^8 \oplus \mathbb{R}$ or $\mathbb{R}^{16}$) [**Unknown**].

### Candidate B: $Sp(8, \mathbb{R}) \times SO(8)$
- **Mode-Space Factor:** $Sp(8, \mathbb{R})$ (identical to Candidate A). Dimension 36, rank 4. Acts on $\mathbb{R}^8$.
- **Geometric Factor:** $SO(8)$, compact real form of type $D_4$. Dimension 28, rank 4.
- **Underlying Spaces:**
  - $SO(8)$ acts faithfully on $\mathbb{R}^8$ via its vector representation $\mathbf{8}_v$, or semi-spinor representations $\mathbf{8}_s, \mathbf{8}_c$.
  - Both factors share the identical underlying vector space $V = \mathbb{R}^8$.
- **Documented Rationale:** $D_4$ possesses outer triality $\text{Out}(SO(8)) \cong S_3$, matching the SIM15.2 triality result $N/H_{192} \cong C_3$, the Weyl group $H_{192} \cong W(D_4)$, and the 3-generation block decomposition of the 120-cell [**Derived**].
- **Unresolved Ambiguity:** The exact coupling mechanism between $Sp(8,\mathbb{R})$ and $SO(8)$ on $\mathbb{R}^8$ (e.g. shared maximal compact subgroup $U(4)$ vs. product action) [**Candidate**].

---

## C. Identification of Relevant Differences

| Category | Distinction | Candidate A: $Sp(8,\mathbb{R}) \times SO(9)$ | Candidate B: $Sp(8,\mathbb{R}) \times SO(8)$ | Operational Significance |
|:---|:---|:---|:---|:---|
| **Structural** | **Algebra Dimension** | $\dim \mathfrak{so}(9) = 36 = \dim \mathfrak{sp}(8,\mathbb{R})$ | $\dim \mathfrak{so}(8) = 28 < 36$ | Dimensional coincidence ($36=36$) vs. compact rotation algebra ($28$). |
| **Structural** | **Root System** | $B_4$ (non-simply laced; short and long roots) | $D_4$ (simply laced; all roots equal length) | $D_4$ roots match 24-cell vertices directly (24 roots of equal length). $B_4$ has 32 roots. |
| **Structural** | **Outer Automorphisms & Triality** | $\text{Out}(\mathfrak{so}(9)) = 1$ (trivial; no outer triality) | $\text{Out}(\mathfrak{so}(8)) \cong S_3$ (order 6; contains cyclic $C_3$ triality) | **Critical Operational Distinction:** SIM15 established an intrinsic threefold triality $N_{576}/H_{192} \cong C_3$. This exists intrinsically in $SO(8)$ ($D_4$), but is mathematically impossible in $SO(9)$ ($B_4$). |
| **Representational** | **Action on 8-Carrier $\mathbb{R}^8$** | **IMPOSSIBLE:** No non-trivial representation with $\dim < 9$ exists. | **CANONICAL:** Natural faithful vector $\mathbf{8}_v$ and spinor $\mathbf{8}_s, \mathbf{8}_c$ representations. | **Carrier Compatibility:** $SO(8)$ acts directly on the OPS carrier $\mathbb{R}^8$. $SO(9)$ requires an artificial dimension extension ($\mathbb{R}^9$ or $\mathbb{R}^{16}$). |
| **Representational** | **Shared Maximal Compact Subgroup** | No common action on $\mathbb{R}^8$. | $\text{Sp}(8,\mathbb{R}) \cap SO(8) = U(4)$ ($\dim = 16$). | $U(4)$ is the exact stabilizer of the reference complex structure $J_0$ and Euclidean metric $G_0 = I_8$. |
| **Operational** | **Symmetric Coset Spaces** | Not defined on $\mathbb{R}^8$. | $Sp(8,\mathbb{R})/U(4)$ has $\dim = 20$ (metric manifold). $SO(8)/U(4)$ has $\dim = 12$ (orthogonal complex structures). | The tangent space $\mathfrak{m}_+$ of $Sp(8,\mathbb{R})/U(4)$ is precisely the space of symmetric matrices anti-commuting with $J_0$ analyzed in SIM16. |
| **Operational** | **Weyl Group Alignment** | $W(B_4)$ has order 384. Contains $W(D_4)$ as index 2. | $W(D_4)$ has order 192, matching the SIM14/15 operational stabilizer $|H_{192}| = 192$ exactly. | The SIM15.0 survivor search among 1,200 subgroups selected order 192, not order 384. |
| **Operational** | **Projection to $SO(3,1) \times SO(4)$** | $SO(9) \supset SO(5) \times SO(4)$, requiring a secondary $SO(5) \to SO(3,1)$ reduction. | $SO(8) \supset SO(4) \times SO(4) \to SO(3,1) \times SO(4)$ via balanced $(4+4)$ split. | Directly mirrors the free $V_4$ orbit decomposition $X_8 = 4 + 4$ ($O_0 \sqcup O_1$) from SIM14. |
| **Speculative** | **Phenomenological Interpretation** | 9D vector space interpreted as bulk or boundary. | $SO(8)$ vector and spinors mapped to 3 generations of fermions via triality. | $SO(8)$ aligns with triality-based 3-generation mechanisms in the 120-cell. |

---

## D. Proposed Discriminating Test

### Test Specification: The Carrier Faithful Action & Maximal Compact Intersection Gate
1. **Core Mathematical Question:**  
   *Can the candidate geometric factor act faithfully and irreducibly on the frozen OPS carrier space $\mathbb{R}^8$, and does its intersection with the mode-space group $\text{Sp}(8, \mathbb{R})$ reproduce the maximal compact subgroup $U(4)$ that stabilizes the OPS complex structure $J_0$?*
2. **Explicit Assumptions:**
   - The microscopic carrier space is the 8-state space $V = \mathbb{R}^8$ frozen in SIM14.0.
   - The symplectic form is the canonical form $\Omega = \begin{pmatrix} 0 & I_4 \\ -I_4 & 0 \end{pmatrix}$.
   - The reference almost complex structure is $J_0 = -\Omega$, and reference metric is $G_0 = I_8$.
   - Symmetries must act linearly on the carrier without introducing unobserved auxiliary dimensions (Invariant 6: Minimal Free Parameters).
3. **Pass / Fail Criteria:**
   - **PASS Criteria for $G_{\text{geom}}$:**
     1. Admits a faithful, irreducible representation on $\mathbb{R}^8$.
     2. $\text{Lie}(G_{\text{geom}}) \cap \mathfrak{sp}(8,\mathbb{R}) = \mathfrak{u}(4)$ (dimension 16).
     3. Possesses an outer automorphism group containing $C_3$ triality compatible with $N_{576}/H_{192} \cong C_3$.
     4. Admits a direct $(4,4)$ branching to $SO(4) \times SO(4)$ compatible with projection to $SO(3,1) \times SO(4)$.
   - **FAIL Outcome:**
     - Failure of (1) excludes the group from acting natively on the 8-state carrier, requiring an auxiliary dimension assumption.
     - Failure of (3) excludes the group from explaining the SIM15 triality normalizer $N_{576}$.
4. **Estimated Computational Cost:** Negligible (< 5 seconds of exact symbolic linear algebra in SymPy).

---

## E. Verification Execution & Results

The discriminating test was implemented and executed via two self-contained SymPy/Python scripts in `Projects/ops-antigravity/research/`:
1. `so8_vs_so9_discrimination.py`
2. `weyl_triality_audit.py`

### Executed Checks and Findings:

```text
=== SO(8) VS SO(9) DISCRIMINATION AUDIT OUTPUT ===
Lie Algebra Dimensions:
  dim(so(8))   = 28
  dim(so(9))   = 36
  dim(sp(8,R)) = 36
Dimension of sp(8, R) on R^8: 36
Dimension of so(8) on R^8: 28
Dimension of intersection sp(8, R) cap so(8) = u(4): 16
Dimension of centralizer C_{so(8)}(J0): 16
Dimension of centralizer C_{sp(8,R)}(J0): 16
Dimension of anti-commutant m_- in so(8): 12
Dimension of anti-commutant m_+ in sp(8, R): 20
Representation Theorem: No non-trivial representation of so(9) exists in dimension 8.
=== ALL EXACT ALGEBRAIC GATES PASSED SUCCESSFULLY ===

=== PART 2: WEYL GROUP AND TRIALITY AUDIT OUTPUT ===
|W(D4)| = 192
|W(B4)| = 384
Confirmed: SIM15 survivor group order 192 matches |W(D4)| exactly, not |W(B4)|.
Dynkin Diagram Automorphisms:
  Out(D4) = S3 (order 6, triality)
  Out(B4) = 1  (order 1, NO triality)
=== PART 2 PASSED ===
```

### Analysis of Demonstrated Findings:

1. **Carrier Representation Barrier for $SO(9)$ [Verified]:**
   - In representation theory of Lie algebra $B_4 \cong \mathfrak{so}(9)$, the fundamental weights have dimensions:
     - Vector $\mathbf{9}$: $\dim = 9$
     - Spinor $\mathbf{16}$: $\dim = 16$
     - Adjoint $\mathbf{36}$: $\dim = 36$
     - 3-form $\mathbf{84}$: $\dim = 84$
   - Because all non-trivial irreps have dimension $\ge 9$, **$\mathfrak{so}(9)$ cannot act faithfully on $\mathbb{R}^8$**.
   - *Consequence:* Candidate A ($Sp(8,\mathbb{R}) \times SO(9)$) cannot be an intrinsic symmetry of the 8-state carrier $X_8$ without postulating an extra 9th auxiliary state/dimension.

2. **$U(4)$ Intersection and Symmetric Spaces for $SO(8)$ [Verified]:**
   - On $\mathbb{R}^8$, $\mathfrak{sp}(8,\mathbb{R}) \cap \mathfrak{so}(8) = \mathfrak{u}(4)$ (dimension 16).
   - $\mathfrak{sp}(8,\mathbb{R})$ decomposes under $\mathfrak{u}(4)$ as $\mathfrak{u}(4) \oplus \mathfrak{m}_+$ ($16 + 20 = 36$), where $\mathfrak{m}_+$ is the exact 20-dimensional space of symmetric matrices anti-commuting with $J_0$ (the metric tangent space analyzed in SIM16.0).
   - $\mathfrak{so}(8)$ decomposes under $\mathfrak{u}(4)$ as $\mathfrak{u}(4) \oplus \mathfrak{m}_-$ ($16 + 12 = 28$), where $\mathfrak{m}_-$ is the 12-dimensional space of skew-symmetric matrices anti-commuting with $J_0$.

3. **Weyl Group and Triality Selection [Verified]:**
   - The operational stabilizer discovered in SIM14.6–SIM15.0 has order $|H_{192}| = 192$.
   - $|W(D_4)| = 192$, exactly matching $H_{192}$.
   - $|W(B_4)| = 384 = 2 \times 192$.
   - $\text{Out}(D_4) \cong S_3$ provides the order-3 cyclic triality discovered in SIM15.1/15.2 ($N_{576}/H_{192} \cong C_3$).
   - $\text{Out}(B_4) = 1$, completely lacking triality.

---

## F. Recommended Next Step

Based on the demonstrated mathematical evidence:
1. **Candidate A ($Sp(8,\mathbb{R}) \times SO(9)$) Status:**
   - Retaining $SO(9)$ requires a specific, explicit proof obligation: *define the carrier space on which $SO(9)$ acts, specify why an auxiliary 9th dimension is admissible under Invariant 6 (Minimal Free Parameters), and explain how the SIM15 triality is recovered without outer automorphisms of $B_4$.*
   - Unless an explicit 9D carrier is authorized, $SO(9)$ is non-viable as a linear symmetry of the 8-state OPS carrier.
2. **Candidate B ($Sp(8,\mathbb{R}) \times SO(8)$) Status:**
   - $SO(8)$ is mathematically compatible with the 8-state carrier, provides the exact maximal compact intersection $U(4)$ governing the SIM16 metric manifold, natively accounts for the $W(D_4)$ order-192 stabilizer, and uniquely provides the $D_4$ triality observed in SIM15.
3. **Recommended Low-Cost Follow-Up Action:**
   - Execute a bounded algebraic study of the **$SO(8) \supset SO(4) \times SO(4)$ branching on the 8-state carrier**, explicitly constructing the projection map to $SO(3,1) \times SO(4)$ and testing whether the two $SO(4)$ factors correspond to the two $K$-orbits $O_0$ and $O_1$ from SIM14.
