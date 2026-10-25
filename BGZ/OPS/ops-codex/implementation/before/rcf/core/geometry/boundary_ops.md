# **Boundary Operator**

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: chain-boundary operator
- authority: tier-3 geometry substrate contract
- upstream dependencies: `README.md`, `simplicial_complex.md`, and `orientation.md`
- status: active
- purpose: define the signed boundary map on simplicial chains
- anti-backflow: the boundary operator does not define physical boundaries, projection behavior, or domain meaning

`/core/geometry/boundary_ops.md`

---

## **1. Purpose**

Define the boundary operators used in RCF’s discrete geometric substrate.

This module specifies the **formal combinatorial rules** for mapping \(k\)-simplices to their \((k-1)\)-dimensional faces.
No embedding, metric, or domain semantics are assumed.

---

## **2. Objects**

Let \(K\) be a simplicial complex with oriented simplices.

For each dimension \(k \ge 1\), the **boundary operator** is a linear map:

\[
\partial_k : C_k(K) \rightarrow C_{k-1}(K)
\]

where \(C_k(K)\) is the vector space of \(k\)-chains (formal linear combinations of oriented \(k\)-simplices).

---

## **3. Boundary of a Simplex**

For an oriented \(k\)-simplex:

\[
[v_0, v_1, \dots, v_k]
\]

the boundary is defined as:

\[
\partial_k [v_0, \dots, v_k] =
\sum_{i=0}^k (-1)^i [v_0, \dots, \hat{v_i}, \dots, v_k]
\]

where \(\hat{v_i}\) denotes omission of the \(i\)-th vertex.

This formula is purely combinatorial.

---

## **4. Boundary-of-Boundary Identity**

Boundary operators satisfy:

\[
\partial_{k-1} \circ \partial_k = 0
\]

This expresses that:

- the boundary of a boundary is empty
- faces cancel in pairs with opposite orientation
- no contradictory lower‑dimensional structure is produced

This identity is structural, not geometric.

---

## **5. Orientation Dependence**

Boundary signs depend on the orientation of the simplex:

- reversing the orientation of a simplex multiplies its boundary by \(-1\)
- induced orientations on faces follow the alternating‑sign rule

Orientation conventions are defined in `orientation.md`.

---

## **6. Chain-Level Linearity**

For any \(k\)-chains:

\[
\partial_k(\alpha A + \beta B) =
\alpha \partial_k A + \beta \partial_k B
\]

for scalars \(\alpha, \beta \in \mathbb{R}\).

Boundary operators extend linearly from simplices to chains.

---

## **7. Incidence Interpretation (Combinatorial Only)**

Boundary operators encode adjacency:

- edges → differences of vertices
- faces → alternating sums of edges
- tetrahedra → alternating sums of faces

No metric, embedding, or physical interpretation is implied.

---

## **8. Structural Role in RCF**

Boundary operators provide:

- the discrete backbone of calculus
- the foundation for divergence, curl, and gradient (in later files)
- adjacency structure for flows and conservation laws
- the combinatorial substrate for cochains and coboundary operators

They are **core geometric invariants** of the simplicial substrate.

---

## **9. Minimal Conceptual Examples**

- boundary of an oriented edge:
  \(\partial_1 [v_i, v_j] = [v_j] - [v_i]\)

- boundary of an oriented triangle:
  \(\partial_2 [v_0, v_1, v_2] = [v_1, v_2] - [v_0, v_2] + [v_0, v_1]\)

- boundary of a tetrahedron as alternating faces

Examples are illustrative only.
