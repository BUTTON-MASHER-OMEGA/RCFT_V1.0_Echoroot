# OPS Research Assignment 001 — Codex independent report

Date: 2026-10-24. Status: bounded investigation complete; candidate selection remains open.

**Finding:** The standard SO(9) vector action cannot descend equivariantly through a fixed linear projection from R9 onto the frozen R8 OPS amplitude carrier. Its axis-fixing SO(8) subgroup can. This supplies a concrete discriminating test **if that carrier and projection are intended**. Neither candidate's geometric representation or coupling is specified sufficiently in the reviewed contracts to impose this test unconditionally.

Evidence labels: **Verified** = checked by the new exact script or directly inspected record; **Derived** = mathematical consequence with argument below; **Candidate** = additional proposed construction or test assumption; **Unknown** = unresolved in the bounded sources. A verified historical record is not a newly reproduced computation.

Source paths below are relative to `C:/Users/blaur/projects/theory/`; `:L` denotes a one-based line. Provenance and SHA-256 hashes are in [source-manifest.json](source-manifest.json). RCF HEAD: `57af06b5ac7c73b66839f70ebba3293399a1c880`; RCMT HEAD: `666ed97aaa1d11f82d0a9881e99bb0c9cae2d225`. OPS is a copied, non-repository research input. RCF already had modified tracked Python cache files when inspected; individual source hashes identify the inputs used here.

## A. Existing evidence

| Evidence inspected | Status and significance |
|---|---|
| `rcf/core/relational-constraints/geometry.md:8,24,36,45` | **Verified document definition:** an optional, noncanonical dual-lane profile permits shared representational structure while preserving distinct roles; no universal necessity theorem. |
| `rcmt/local/constraints/symmetry_chain.md:16,54–94`; `mode_space.md:83–93` | **Verified document definition:** RCMT adopts two lanes, requires representational separation, and retains both candidate group expressions. Sp(8,R) is provisional. The product symbol is not a demonstrated decomposition. Compactness is profile-specific. |
| `rcmt/local/constraints/projection.md:16,55–84` | **Verified document definition:** projection is a state-carrier schema with locality, adjacency compatibility, finite representation and noninvertibility requirements. No actual intertwiner or group homomorphism is supplied. |
| `projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md:5–76`; `recorded_output/SIM14.3.txt:519–521,623–639` | **Derived theorem / verified record:** a commuting extra involution produces a regular eight-point action exactly when it exchanges the two free V4 orbits. M2 passes; M1 does not. This is a finite-action discriminator, not a continuous SO-group selection. |
| `projects/Operational_Phase_Space/recorded_output/SIM15.2.txt:60–83,225–241`; `SIM15.2.py:2018–2093` | **Verified record and inspected implementation:** explicit L ≅ W(D4) and diagram triality modulo an inner adjustment were reported. Not rerun here. Identifying a finite Weyl group does not identify the geometric lane's continuous group or its representation. |
| `projects/Operational_Phase_Space/recorded_output/SIM15.6.txt:35,65–73` | **Verified record:** the frozen incidence predicate selects one kernel–facet matching; abstract C3 equivariance alone leaves three. This cautions against selecting a representation from group fingerprints alone. |
| `projects/Operational_Phase_Space/SIM16.0.py:159–185`; `SIM16.1.py:239–249` | **Verified code/previous reproduction record:** compatible metrics are not unique; the signed, strongly sparse family includes G(t) = cosh(t)I + sinh(t)K. The present Euclidean test chooses t=0, rather than claiming the family selects it. |
| `projects/Operational_Phase_Space/SIM16.4.py:18–33,253–282,365–389,535–589` | **Verified code/previous reproduction record:** frozen R8, Ω, C8; C8-sparse symplectic generator algebra sp(2,R)^4; fixed-J0 stabilizer consists of four partner-plane rotations. These are algebraic constraints, not selected dynamics. |

Prior Codex reproductions of SIM16.0, 16.1 and 16.4 matched their historical truth packets: `projects/ops-codex/implementation/ops-repair-verification.json:71–127`. This assignment reuses those results and executes no historical simulation. Discussion searches were targeted; no exhaustive review of SIM14–SIM16 is claimed.

## B. Candidate definitions and missing data

**Verified definitions used in this test:** Sp(8,R) means real 8×8 matrices S satisfying SᵀΩS = Ω, with nondegenerate alternating Ω; SO(n) means real n×n Q satisfying QᵀQ = I and det Q = 1. Here n=8 or 9, with the standard vector actions. These are declared test representations, not recovered geometric-lane specifications.

The frozen carrier is V = R^X8, the real vector space on the eight OPS labels. Its basis is e0,…,e7. Ω has positive entries at (0,2), (1,3), (4,6), (5,7), negative transpose entries, and Ω² = −I. J0 = −Ω and G(0)=I. The eight-point set, its eight-dimensional linearization, an eight-dimensional manifold, and a geometric-lane state space are different objects.

**Unknown:** the proposed geometric carrier, chosen SO representation, geometric metric/signature after projection, lane coupling, operational independence criterion, and which frozen objects must be preserved versus transformed covariantly. The reviewed sources do not identify an SO(9) vector, spinor, or other action as the intended construction. No spinorial/Clifford identification is assumed or tested; historical D4 triality does not settle it.

An abstract direct product requires commuting factor images in any representation. Distinct lane roles alone do not establish this. Both products have ordinary block-diagonal representations on **separate** spaces R8 ⊕ Rn: diag(S,Q) preserves the alternating form Ω ⊕ 0 and the symmetric form 0 ⊕ I_n. **Derived:** this simple construction demonstrates mathematical coexistence for both n; it supplies neither an RCMT projection nor locality or Lorentz compatibility.

## C. Differences and operational meaning

| Finding | Classification | Consequence and limit |
|---|---|---|
| The standard SO(9) action moves a chosen ninth axis; fixing that axis gives an SO(8) subgroup. | **Derived — representational** | Removing that coordinate loses transformations needed for closure. The proof below uses actual generators, not group dimension alone. |
| The frozen regular C2³ permutation action has determinant +1 on V and embeds into both SO(8) and diag(SO(8),1) ⊂ SO(9). | **Verified — operational negative control** | The orbit-bridge result by itself discriminates neither candidate. This does not assert every bridge permutation preserves Ω or the C8 graph. |
| SO(8) transformations preserving the chosen Ω form U(4); adding SIM16.4 generator sparsity leaves the four partner rotations. Extending Ω to Ω ⊕ 0 in R9 gives the same U(4) stabilizer inside SO(9). | **Verified linear-algebra checks; Derived group identification — representational/operational** | Fixed-form retention removes the apparent extra SO(9) directions. It also prevents treating full SO(8) as already compatible with all frozen data. Covariant or separate-lane actions are outside this fixed-data test. |
| No nonzero skew-symmetric R8 generator commutes with all twelve local sp(2,R)^4 generators. | **Verified — operational negative control** | Neither geometric candidate can supply a nontrivial connected orthogonal factor commuting with the standard full Sp(8,R) action on this same V. Separate carriers or other actions remain possible. |

No speculative physical interpretation is required. Compactness has not been imposed on the symplectic lane.

## D. Discriminating experiment and proof obligation

**Candidate test assumptions:** use the standard SO(9) action on W=R9; let P:W→V drop coordinate 8 and let i=Pᵀ include V. Require a linear projected action satisfying

    P Q = ρ(Q) P

for the whole geometric group, with ρ a representation on V. This is an additional equivariance requirement on a proposed interface; it is not identified with the schematic RCMT mode-to-geometry projection.

**Question:** does the action descend without retaining the ninth coordinate or restricting the group?

**Derived criterion:** a linear action descends through a surjective P exactly when ker P is invariant. Here ker P = span(e8), so the descending SO(9) subgroup preserves this line. Its identity component fixes e8 and is the standard SO(8). The whole line stabilizer can also reverse the line with an orientation-reversing R8 block; this disconnected extension does not restore full SO(9).

An explicit infinitesimal witness isolates the lost information. Define F_ab = E_ab − E_ba. For X=F_08 and Y=F_28:

    P X i = P Y i = 0,
    P [X,Y] i = −F_02 ≠ 0.

Thus compression X↦PXi is not a Lie-algebra representation of so(9). Equivalently, PXe8=e0 while any ρ_*(X)Pe8=0, directly contradicting intertwining. All eight F_i8 violate kernel invariance. The axis-fixing so(8) block descends exactly. F_02 is an existing OPS partner-plane rotation, so the lost commutator is operationally concrete within the frozen construction.

**Pass/fail:** a proposed *full* standard SO(9) action fails this fixed linear quotient test. The standard SO(8) block passes descent only. Neither thereby passes every OPS condition. Choosing a preferred axis and restricting to its connected stabilizer passes but retains SO(8), rather than a faithful full SO(9) action on the quotient.

**Inconclusive cases:** different representations, nonlinear state carriers/maps, an output retaining the extra coordinate, non-equivariant projection, or an action transporting the frozen data. These require explicit definitions and cannot be scored using this witness. Failure excludes this representation/interface combination, not SO(9) itself.

**Unresolved proof obligation:** for each actual candidate specify carriers M and G, representations, Π_RCMT, and the required covariance law; then prove that law together with locality, adjacency compatibility, noninvertibility, and lane separation. If fixed R8 descent is intended, the witness is already decisive. If not, replace the test with the declared interface's equations.

**Cost:** exact integer matrices of size at most 9×9 and small rational nullspaces. No group enumeration beyond closure of eight frozen permutations; no parameter sweep or numerical tolerance. The script enforces a 15-second worker limit.

## E. Executed verification

Artifacts: [verify.py](verify.py), [results.json](results.json), [source-manifest.json](source-manifest.json).

Reproduce from `C:/Users/blaur/projects/theory/` in PowerShell:

```powershell
& C:/Users/blaur/anaconda3/envs/glyphnet/python.exe -X utf8 -B projects/ops-codex/research/assignment-001/verify.py
```

The script reads literal frozen assignments through Python AST; it does not import or execute SIM16.4. Python 3.12.4 / SymPy 1.14.0, exact arithmetic. All assertions passed; elapsed worker time is recorded in results.json (approximately one second).

| Executed check | Result |
|---|---|
| Kernel-preserving subalgebra of standard so(9) for P | dimension 28; eight extra rotations fail descent |
| Compression bracket witness | entries (0,2)=−1 and (2,0)=+1; rank 2 |
| Standard so(8) block intertwining | passes |
| so(8) preserving Ω | dimension 16 |
| so(9) preserving Ω ⊕ 0 | dimension 16 |
| C8-sparse skew generators preserving Ω | dimension 4, abelian; pairs (0,2),(1,3),(4,6),(5,7) |
| Skew R8 generators commuting with local sp(2,R)^4 | dimension 0 |
| Frozen p,h,s closure | eight elements; regular; every determinant +1 |
| Before/after hashes of all recorded inputs | unchanged |

**Derived identification:** for skew X, XᵀΩ+ΩX=0 is equivalent to [X,Ω]=0. Since Ω²=−I, commuting with Ω means complex linearity on C4; real orthogonality is complex unitarity. Hence the group is U(4), rather than merely an unexplained dimension-16 subspace. In R9, preserving Ω⊕0 preserves its radical line; its R8 restriction is unitary and has real determinant +1, forcing the ninth-axis sign +1 inside SO(9). The group stabilizer is again U(4).

The standard intersection identity is also recorded in Ana Cannas da Silva's [Lectures on Symplectic Geometry, Homework 8, exercise 2, printed p.72](https://math.mit.edu/classes/18.157/2017sp/lecturesonsymplecticgeometryanacannas.pdf#page=80) (revised January 2006; consulted 2026-10-24). This reference supports the general identity; the frozen OPS calculations and SO(9) projection witness are established separately above.

**Derived product control:** the full standard Sp(8,R) contains the verified local sp(2,R)^4 algebra. Commuting orthogonal generators for the full group must commute with that subalgebra; its verified zero-dimensional skew centralizer rules out a nontrivial connected orthogonal commuting factor on this same V. This condition is stronger than the already unsuitable collapse of the two RCMT lanes onto one carrier; it is a diagnostic against reading the product notation as simultaneous natural actions on R8.

Generator sparsity here means zero matrix entries off the C8 edges and diagonal, exactly as in SIM16.4. It is not graph-automorphism invariance, a general consequence of locality, or commutation with the adjacency matrix. The four disjoint partner rotations also have sparse finite-time exponentials; the new script verifies their infinitesimal span and abelianness, while the previous SIM16.4 record supplies that finite-time check.

No dependencies were installed, no long simulations were run, and no research sources, compressed documents, or other agent output were modified or inspected.

## F. Next step and stopping point

The justified low-cost follow-up is a **representation/interface declaration**, not another simulation: identify whether the geometric candidate acts on the frozen amplitude space, on a separate vector space, or on another carrier; state which Ω, metric, adjacency and observation structures are fixed, covariant, or unrelated to that action. For a linear interface, evaluate kernel invariance and the generator intertwining equations first. Both are small exact checks once the matrices are supplied.

The present assignment has supplied a verified conditional discriminator and a precise remaining proof obligation. It has not selected a winner, constructed Lorentzian projection, established physical necessity, or promoted either candidate. Investigation stops here. Antigravity's workspace/report was not read; comparison remains pending until independent reports are available and comparison is requested or supplied in the shared context.
