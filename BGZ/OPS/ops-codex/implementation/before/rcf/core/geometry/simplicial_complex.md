# **Simplicial Complex Structure**

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: simplicial carrier grammar
- authority: tier-3 geometry substrate contract
- upstream dependencies: `README.md` and `../algebra/tensor_ops.md`
- status: active
- purpose: define the simplicial complex used as the minimal discrete carrier
- anti-backflow: simplicial structure does not define survivorship, projection, metric geometry, or physical interpretation

`/core/geometry/simplicial_complex.md`

---

## **1. Purpose**

Define the simplicial complex structure used in RCF as the **minimal discrete geometric substrate** for calculus, adjacency, and topological invariants.

This module specifies **structural definitions only**.
It assumes no physics, semantics, or domain‑specific interpretation.

---

## **2. Objects**

A **simplicial complex** \( K \) consists of:

- a finite set of **vertices** \( V \)
- a set of **simplices** (edges, faces, tetrahedra, …)
- closure under taking faces

Formally:

- A **0‑simplex** is a vertex
- A **1‑simplex** is an unordered pair of vertices
- A **2‑simplex** is a triple of vertices
- A **3‑simplex** is a quadruple of vertices

A simplex exists only if **all of its faces** exist.

No meaning is attached to vertices or simplices.

---

## **3. Orientation**

Each simplex may be assigned an **orientation**, represented by an ordering of its vertices up to even permutation.

Examples:

- Edge orientation: \([v_i, v_j]\)
- Face orientation: \([v_i, v_j, v_k]\)
- Tetrahedron orientation: \([v_i, v_j, v_k, v_l]\)

Orientation is a **sign structure**, not a geometric embedding.

---

## **4. Adjacency Structure**

Two simplices are adjacent if they share a common face.

Adjacency is defined purely combinatorially:

- edges share a vertex
- faces share an edge
- tetrahedra share a face

No metric or embedding is assumed.

---

## **5. Boundary Operator (Definition Only)**

For an oriented \( k \)-simplex:

\[
\partial_k [v_0, \dots, v_k] =
\sum_{i=0}^k (-1)^i [v_0, \dots, \hat{v_i}, \dots, v_k]
\]

This defines the **boundary** of a simplex as the alternating sum of its faces.

Boundary operators satisfy:

\[
\partial_{k-1} \circ \partial_k = 0
\]

This identity is structural, not geometric.

---

## **6. Coboundary Operator (Definition Only)**

The coboundary operator is the dual of the boundary:

\[
\delta_k = \partial_{k+1}^\top
\]

It maps \( k \)-cochains to \( (k+1) \)-cochains.

No metric or Hodge structure is assumed here.

---

## **7. Validity Conditions**

A simplicial complex must satisfy:

- **closure** under taking faces
- **non‑degeneracy** (no repeated vertices in a simplex)
- **orientation consistency** across shared faces
- **boundary‑of‑boundary is zero**

These are structural admissibility constraints.

---

## **8. Topological Invariants (Definition Only)**

Given counts:

- \( |V| \) vertices
- \( |E| \) edges
- \( |F| \) faces
- \( |T| \) tetrahedra

The **Euler characteristic** is:

\[
\chi = |V| - |E| + |F| - |T|
\]

Betti numbers are defined via ranks of boundary maps:

\[
\beta_k = \dim \ker \partial_k - \dim \mathrm{im}\, \partial_{k+1}
\]

These invariants are purely combinatorial.

---

## **9. Structural Role in RCF**

The simplicial complex provides:

- a discrete substrate for calculus
- adjacency structure for flows
- orientation for sign‑consistent operators
- topological invariants for stability checks
- a combinatorial foundation for downstream modules

It is the **minimal geometric carrier** for RCF’s calculus layer.

---

## **10. Minimal Conceptual Examples**

- a single oriented triangle
- two tetrahedra sharing a face
- a closed loop of edges
- a pinched sphere (topology only)

Examples are illustrative only.
