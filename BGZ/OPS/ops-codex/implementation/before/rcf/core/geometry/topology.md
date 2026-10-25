# **Topology**

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: combinatorial topology grammar
- authority: tier-3 geometry substrate contract
- upstream dependencies: `README.md`, `boundary_ops.md`, and `coboundary_ops.md`
- status: active
- purpose: define Euler characteristic, homology, and Betti-number records
- anti-backflow: combinatorial topology does not define embedding, metric geometry, projection, or physical topology

`/core/geometry/topology.md`

---

## **1. Purpose**

Define the minimal topological invariants and structures used in RCF’s discrete geometric substrate.

This module specifies **purely combinatorial** notions of topology derived from simplicial complexes.
No metric, embedding, geometry, or domain semantics are assumed.

---

## **2. Euler Characteristic**

Given a simplicial complex \(K\) with:

- \( |V| \) vertices
- \( |E| \) edges
- \( |F| \) faces
- \( |T| \) tetrahedra
- and higher‑dimensional simplices as applicable

the **Euler characteristic** is:

\[
\chi(K) = \sum_{k=0}^n (-1)^k |K_k|
\]

where \(K_k\) is the set of \(k\)-simplices.

This invariant is purely combinatorial.

---

## **3. Homology Groups (Definition Only)**

Homology groups measure the structure of cycles and boundaries.

Define:

- \(Z_k = \ker \partial_k\) (cycles)
- \(B_k = \mathrm{im}\, \partial_{k+1}\) (boundaries)

The **\(k\)-th homology group** is:

\[
H_k(K) = Z_k / B_k
\]

No geometric interpretation is attached to cycles or holes.

---

## **4. Betti Numbers**

The **Betti number** \(\beta_k\) is the rank of the \(k\)-th homology group:

\[
\beta_k = \dim H_k(K)
\]

Betti numbers count independent \(k\)-dimensional cycles.

Examples (conceptual only):

- \(\beta_0\): connected components
- \(\beta_1\): independent loops
- \(\beta_2\): voids or cavities

These are structural, not physical.

---

## **5. Chain Complex Structure**

Boundary operators form a **chain complex**:

\[
\cdots \xrightarrow{\partial_{k+1}} C_k(K) \xrightarrow{\partial_k} C_{k-1}(K) \xrightarrow{\partial_{k-1}} \cdots
\]

with the identity:

\[
\partial_{k-1} \circ \partial_k = 0
\]

This structure is the foundation of homology.

---

## **6. Cochain Complex Structure**

Coboundary operators form a **cochain complex**:

\[
\cdots \xrightarrow{\delta_{k-1}} C^k(K) \xrightarrow{\delta_k} C^{k+1}(K) \xrightarrow{\delta_{k+1}} \cdots
\]

with:

\[
\delta_{k+1} \circ \delta_k = 0
\]

This is the dual structure to the chain complex.

---

## **7. Topological Equivalence (Definition Only)**

Two simplicial complexes \(K\) and \(K'\) are **topologically equivalent** if they have:

- isomorphic homology groups
- identical Euler characteristics
- compatible chain and cochain structures

No geometric embedding or deformation is assumed.

---

## **8. Structural Role in RCF**

Topology provides:

- global invariants for stability
- cycle structure for discrete calculus
- a substrate for divergence‑free and curl‑free conditions
- a combinatorial notion of “shape” without geometry

It is the **minimal topological layer** of the RCF core.

---

## **9. Minimal Conceptual Examples**

- a loop of edges with \(\beta_1 = 1\)
- a filled triangle with \(\beta_1 = 0\)
- two disconnected components with \(\beta_0 = 2\)
- a tetrahedral shell with \(\beta_2 = 1\)

Examples are illustrative only.
