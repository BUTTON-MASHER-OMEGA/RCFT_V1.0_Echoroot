# **Coboundary Operator**

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: cochain-coboundary operator
- authority: tier-3 geometry substrate contract
- upstream dependencies: `README.md`, `orientation.md`, and `boundary_ops.md`
- status: active
- purpose: define the formal dual coboundary map
- anti-backflow: the coboundary operator does not define metric, dynamics, projection, or physical interpretation

`/core/geometry/coboundary_ops.md`

---

## **1. Purpose**

Define the **coboundary operators** used in RCF’s discrete geometric substrate.

This module specifies the **formal dual** of the boundary operator.
It assumes no metric, embedding, or domain semantics.

---

## **2. Objects**

Let \(K\) be a simplicial complex.

- \(C^k(K)\) is the vector space of **\(k\)-cochains**
  (linear functionals on \(k\)-chains)

- The **coboundary operator** is a linear map:

\[
\delta_k : C^k(K) \rightarrow C^{k+1}(K)
\]

Coboundaries act on cochains, not simplices.

---

## **3. Definition via Duality**

Coboundary is defined as the **transpose** of the boundary operator:

\[
\delta_k = \partial_{k+1}^{\top}
\]

This ensures:

- structural consistency
- sign coherence
- compatibility with orientation

No metric or Hodge structure is assumed.

---

## **4. Action on Cochains**

For a cochain \(\phi \in C^k(K)\) and a \((k+1)\)-simplex \([v_0,\dots,v_{k+1}]\):

\[
(\delta_k \phi)([v_0,\dots,v_{k+1}]) =
\phi(\partial_{k+1}[v_0,\dots,v_{k+1}])
\]

This is the defining relation.

---

## **5. Coboundary-of-Coboundary Identity**

Coboundary operators satisfy:

\[
\delta_{k+1} \circ \delta_k = 0
\]

This follows from:

\[
\partial_{k} \circ \partial_{k+1} = 0
\]

and duality.

This identity is structural, not geometric.

---

## **6. Linearity**

For cochains \(\phi, \psi \in C^k(K)\) and scalars \(\alpha, \beta\):

\[
\delta_k(\alpha \phi + \beta \psi) =
\alpha \delta_k \phi + \beta \delta_k \psi
\]

Coboundary extends linearly from basis cochains.

---

## **7. Incidence Interpretation (Combinatorial Only)**

Coboundary encodes:

- how \(k\)-cochains extend to \((k+1)\)-simplices
- how local evaluations propagate to higher‑dimensional structure
- adjacency relations in cochain space

No physical or semantic interpretation is implied.

---

## **8. Structural Role in RCF**

Coboundary operators provide:

- the discrete backbone of **cochain calculus**
- the dual structure to boundary operators
- the foundation for gradient, curl, and divergence (in later files)
- a combinatorial substrate for discrete differential forms

They are **core geometric invariants** of the simplicial substrate.

---

## **9. Minimal Conceptual Examples**

- coboundary of a vertex‑cochain producing an edge‑cochain
- coboundary of an edge‑cochain producing a face‑cochain
- alternating‑sign propagation through simplex faces

Examples are illustrative only.
