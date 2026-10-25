# OPS Research Assignment 002 — Codex independent report

2026-10-24. Bounded investigation complete. Research findings only.

**Result:** Neither construction tested produces so(8) → so(9). Adding retained equality constraints reduces the constraint symmetry algebra; squeezing the existing OPS metric family changes the representation by conjugacy without changing its algebra. In the equality toy, a particular constraint-response interface fails and later recovers as constraints are added. A same-density control passes that interface, and nonzero projected surviving configurations remain throughout. No projectability exhaustion is established.

Labels: **Verified** = executed exact check or inspected document/record, with the distinction stated; **Derived** = mathematical argument; **Candidate** = toy definition or additional assumption; **Unknown** = not determined. All source paths are relative to `C:/Users/blaur/projects/theory/`; colon numbers denote source lines. [source-manifest.json](source-manifest.json) records hashes and commits. RCF HEAD: `57af06b5ac7c73b66839f70ebba3293399a1c880`; RCMT HEAD: `666ed97aaa1d11f82d0a9881e99bb0c9cae2d225`. OPS files are copied, non-repository research inputs.

## A. Compression definition

### What the sources supply

**Verified document definitions:** `rcf/modules/temporal-interface/local-constraint-density.md:17–33,70–104` defines κ conceptually through narrowing of finite/effectively finite admissible next-state sets. It requires boundedness, locality, monotonicity and geometry neutrality. It supplies no numerical κ→OPS-operator or κ→symmetry map. Its input exclusions at lines 58–64 rule out deriving κ from geometry or LCO mismatch scores.

The historical sweep is a separate exploratory record: `rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:15–17` explicitly disclaims a conversion between historical sweep gamma, κ and bounded memory γ, and records unresolved provenance/formula discrepancies. `rcmt/local/invariants.md:101–114` retains finite representation/adjacency requirements and distinguishes the temporal parameters. None of these definitions supplies a physical compression law for this experiment.

**Verified baseline:** SIM14's finite orbit-bridge action and SIM15's finite W(D4)/triality identification do not establish a continuous geometric-lane action or a density-dependent extension. Their evidence and limitations are reused from [OPS-001, sections A–C](../assignment-001/report.md). OPS-001's ninth-coordinate obstruction remains conditional on its standard-vector linear projection. Here the principal projection is instead the frozen **B:R8→R4**, not OPS-001's P:R9→R8.

RCMT's two-lane architecture and candidate status remain as in `rcmt/local/constraints/symmetry_chain.md:16,54–94`. SO(8) fits the present natural eight-dimensional carrier; this does not prove its full action preserves all frozen OPS data. SO(9) remains open under other representations. The RCMT projection remains schematic (`rcmt/local/constraints/projection.md:16,55–84`).

### Toy 1: nested relational equalities

**Candidate:** retain X8, V=R^X8, Ω, C8, G0=I and B from `projects/Operational_Phase_Space/SIM16.4.py:18–33,64–68`. Let Bx=(x0+x1,x2+x3,x4+x5,x6+x7). At an active C8 edge e=(a,b), impose x_a=x_b. Define

    v_e = e_a − e_b,   H_e = v_e v_eᵀ,
    S_E = {x in R8 : v_eᵀx=0 for every active e},
    A_E = S_E ∩ {0,1}8,   L_E = Σ_(e in E) H_e.

S_E is the real constraint space; A_E is a finite binary audit of mode narrowing. The binary values are a toy alphabet, with no physical meaning. No continuous rotation symmetry of A_E is claimed. Each H_e is retained as an individual constraint tensor, rather than discarded in favor of only their sum.

The bounded sequence is:

| Stage | Active edges added cumulatively | Toy density c |
|---|---|---|
| E0 | none | 0 |
| E1 | (0,1) | 1/8 |
| E3 | add (0,2),(1,3) | 3/8 |
| E4 | add (2,4) | 1/2 |
| E5 | add (3,5) | 5/8 |

**Candidate:** c=1−log2(|A_E|)/8. For these forests, c=|E|/8. This bounded pruning statistic illustrates the conceptual narrowing definition; it is **not an identification with the project's numerical κ**, nor a global density field. Only a fixed eight-label neighborhood is tested. Admissibility means satisfying the stated equalities; transitions are changes in stabilizer or intertwining equations. The sequence deliberately includes p-paired and unpaired edges to test interface sensitivity, not an asserted temporal evolution.

### Toy 2: symplectic squeeze of the existing metric family

SIM16.1 already supplies G(t)=cosh(t)I+sinh(t)K with K=|Ω|, and SIM16.3 supplies symplectic conjugacy and B-descent (`SIM16.1.py:239–249`; `SIM16.3.py:398–416`).

**Candidate interpretation:** q=exp(t)≥1 measures anisotropic squeezing. Write

    G_q = (q+q⁻¹)I/2 + (q−q⁻¹)K/2,
    J_q = −ΩG_q,   S_q = exp(−tK/2).

The eigenvalues of G_q are q and q⁻¹, each fourfold, so its condition number is q² and determinant is 1. This is volume-preserving squeezing, **not volume compression or increasing relational κ**. Ω, C8 and B remain fixed; G and J transform. Check q=1,2,4 exactly. No singular limit is assumed.

## B. Symmetry behavior

For Toy 1 define these distinct algebras:

    g_E = {X : Xᵀ=−X, [X,H_e]=0 for every active e},
    a_E = {X in g_E : XᵀΩ+ΩX=0 and X is C8-sparse},
    b_E = {X in a_E : X(ker B) ⊆ ker B}.

These are symmetries of the retained quadratic constraints; a_E further preserves the frozen form and adopts SIM16.4 generator sparsity. They are **not generators derived from dynamics**. Sparsity means off-diagonal support only on C8 edges, not commutation with the adjacency matrix or graph-automorphism invariance.

**Verified exact results:**

| Stage | Binary survivors | dim S_E | dim g_E | dim a_E | dim b_E |
|---|---:|---:|---:|---:|---:|
| E0 | 256 | 8 | 28 | 4 | 2 |
| E1 | 128 | 7 | 21 | 2 | 1 |
| E3 | 32 | 5 | 10 | 2 | 1 |
| E4 | 16 | 4 | 6 | 1 | 0 |
| E5 | 8 | 3 | 3 | 0 | 0 |

**Derived:** for skew X, [X,vvᵀ]=0 forces Xv=0. Thus g_E acts as so(8−r) on the complement of the r independent edge vectors: so(8), so(7), so(5), so(4), so(3). These centralizer intersections close under brackets. The a_E spaces are subspaces of the four commuting partner-plane rotations verified in SIM16.4 and OPS-001; their brackets were checked to vanish.

More generally, retaining each old constraint while adding new ones gives g_(E∪{e}) = g_E ∩ centralizer(H_e). Its dimension cannot increase. This nesting argument applies to these **individually preserved tensors in a fixed representation**. It is not a theorem that every shrinking configuration set has a smaller symmetry group: unlabelled constraints, aggregate operators, degenerate survivors and changed representations require separate definitions.

For Toy 2, **Verified at q=1,2,4 and Derived for every finite q>0**:

| Algebra or interface | Result at each q |
|---|---|
| Metric-orthogonal algebra so(G_q) | dimension 28, conjugate to so(8) |
| so(G_q) ∩ sp(Ω) | dimension 16, conjugate to u(4) |
| C8-sparse part of that intersection | dimension 4, abelian |
| B-projectable part of that local algebra | dimension 2 |
| J_q and S_q descent through B | passes |

Indeed S_qᵀΩS_q=Ω, S_qᵀG_qS_q=I and J_q=S_qJ0S_q⁻¹. The metric and compatible stabilizers are conjugated, rather than enlarged. The partner-block transport preserves the relevant local algebra's support and commutes with p, so the local and B-compatible dimensions also persist. This does not mean each original fixed matrix remains a symmetry at every q.

The standard orthogonal/symplectic intersection identity is corroborated by Ana Cannas da Silva's [Lectures on Symplectic Geometry, Homework 8, exercise 2, printed p.72](https://math.mit.edu/classes/18.157/2017sp/lecturesonsymplecticgeometryanacannas.pdf#page=80). The frozen calculations and toy definitions are independent of that reference.

## C. SO(9) assessment

**Derived exclusion within both tested constructions:** no eight new generators arise. Toy 1 intersects existing stabilizers and therefore cannot add them. Toy 2 conjugates an existing so(8) action and retains all its brackets and dimension. A faithful so(9) extension cannot lie in the 28-dimensional metric-orthogonal algebra on this same positive-definite eight-dimensional vector representation. This dimension bound has a specified operational premise: every candidate generator must preserve G_q on V. It is not a dimension-only comparison of unrepresented abstract groups.

The Ω and sparsity conditions narrow the tested algebras further. No ninth direction, extra operation or additional generator has been inserted. Therefore the SO(9) enhancement hypothesis is **unsupported here and excluded under these fixed-carrier orthogonal assumptions**. Other carriers/representations, transformed constraint semantics, or an actual density-dependent construction remain **Unknown**. Sp(8,R) remains a candidate realization, not a proven necessity.

## D. Projection assessment

### Operator descent and its precise failure scope

For any linear A on V, B A = A_bar B exists exactly when B A N=0, where columns of N span ker B. If it exists, A_bar=B A Bᵀ/2. This tests a specified interface, without asserting that B is the full RCMT Π.

For the **Candidate aggregate constraint response** A=L_E, the exact results are:

| Stage | B L_E descends? | defect rank | dim B(S_E) | distinct B(A_E) |
|---|---|---:|---:|---:|
| E0 | yes | 0 | 4 | 81 |
| E1 | yes | 0 | 4 | 54 |
| E3 | yes | 0 | 3 | 18 |
| E4 | no | 1 | 3 | 12 |
| E5 | yes | 0 | 2 | 6 |

**Verified witness at E4:** w=e2−e3 satisfies Bw=0 but B L_E4 w=(0,1,−1,0)ᵀ. Thus two vectors with the same B observation can have different observed constraint responses. Adding (3,5), the p-partner of (2,4), restores aggregate descent.

**Verified same-density control:** the alternative four-edge set {(0,1),(0,2),(1,3),(6,7)} has the same c=1/2 and 16 binary survivors as E4, but its aggregate response descends. Hence density alone does not determine this interface. For symmetric L_E, preserving ker B is equivalent to commuting with the frozen pair swap p. Equal weights on a p-invariant edge set ensure that commutation.

This restoration concerns the **sum** L_E. It does not make the individual edge responses projectable, remove their constraints, or reverse the monotone reduction of g_E. For example, the individual (0,2) response need not descend even when its sum with (1,3) does.

**Scope A, conditional:** E4 fails an auxiliary interface that additionally requires the off-constraint linear response L_E to be determined by Bx on all V. L_E is a local quadratic-constraint response, not an imposed physical evolution. This is not a demonstrated failure of B as a configuration map or of the schematic RCMT projection.

In fact L_E vanishes on S_E. If attention is restricted to exactly admissible configurations, its response is zero and trivially descends at every stage. The explicit failure witness is outside S_E4. Nonzero projected survivors persist: the all-ones binary configuration survives every equality and maps to (2,2,2,2). Zero-dimensional b_E means no nontrivial local continuous symmetry survives that test, **not** no configurations or projections survive.

### Symplectic and adjacency checks

**Verified:** Ω and the original V remain unchanged, and every generator in a_E is symplectic and C8-sparse. Each L_E is symmetric and C8-sparse. This does not make L_E itself a symplectic evolution generator. A separately proposed Hamiltonian response −ΩL_E is symplectic infinitesimally and has the same B-descent pattern, but **fails the adopted C8 generator-sparsity test at every nonempty tested E**. It therefore supplies no admissible sparse dynamics or enhancement mechanism here.

The restriction of Ω to S_E has ranks 8,6,4,4,2 while S_E has dimensions 8,7,5,4,3. Thus replacing the mode carrier by S_E would often destroy nondegeneracy. Such a replacement is not made or asserted to satisfy RCMT. The full fixed Ω remains nondegenerate.

For Toy 2, J_q and S_q preserve ker B and intertwine with their induced quotient operators throughout the tested states and, by the p-commutation argument, throughout finite q>0. Direct descent of the full nondegenerate Ω through rank-four B was already obstructed in SIM16.3 (`SIM16.3.py:255–274`); its split-induced quotient form is a different construction. Squeezing creates no new obstruction to that existing induced interface.

**No scope B or C result is claimed:** no class of all RCMT representations/projections was tested, and no search space was exhausted. The apparent loss and restoration above concern one additional response law, not a projectability horizon.

## E. Verification, provenance and limits

Files: [verify.py](verify.py), [results.json](results.json), [source-manifest.json](source-manifest.json). Reproduce from `C:/Users/blaur/projects/theory/`:

```powershell
& C:/Users/blaur/anaconda3/envs/glyphnet/python.exe -X utf8 -B projects/ops-codex/research/assignment-002/verify.py
```

Python 3.12.4, SymPy 1.14.0; exact rational arithmetic and exact radicals. The final verification completed in **1.210 seconds**, with a 20-second worker limit. It checked five nested edge states, one matched-density control and three metric states. Binary narrowing is audited on just 256 assignments per edge state; there is no large enumeration or parameter sweep. Recorded source hashes were unchanged before and after the run. No historical simulation was executed, dependency installed, authoritative contract changed, compressed file handled, or other agent output inspected.

The relevant SIM14–SIM16 evidence was reused through OPS-001 and the prior reproduction ledger, with targeted rereading of SIM16.1/16.3/16.4 and current density/projection contracts. This is not a fresh exhaustive audit or reproduction of that history.

Limitations: the equality schedule, binary alphabet and aggregate response are explicitly toys; q is a squeeze parameter, not κ. No stochastic update, LCO policy, temporal dynamics, physical compression law, Lorentz projection or global optimum is constructed. No selected surviving binary configuration has been assigned a continuous symmetry group by these computations.

## F. Next step and stopping point

The low-cost follow-up is to specify the intended density-dependent object and covariance law: does compression retain individual constraints, change only an aggregate tensor, squeeze a metric, or change the carrier? For a proposed response A(κ), compute the small matrix B A(κ) N before undertaking simulations. At fixed constraint count, test p-balanced versus p-unbalanced relations, since the verified control shows that placement matters independently of narrowing.

Any enhancement proposal must then derive its additional operations from that rule, verify bracket closure, and declare which frozen structures are retained or transported. The present constructions provide counterexamples to automatic symmetry enhancement and automatic projectability exhaustion; they do not rule out all alternative constructions.

The bounded assignment is complete and investigation stops here. Comparison with Antigravity is pending independent report availability; no consensus or comparison is asserted.
