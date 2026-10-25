# Orientation Structure

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: simplex-orientation grammar
- authority: tier-3 geometry substrate contract
- upstream dependencies: `README.md` and `simplicial_complex.md`
- status: active
- purpose: define sign-consistent orientation for chains and cochains
- anti-backflow: orientation signs do not define metric direction, dynamics, projection, or domain semantics

`/core/geometry/orientation.md`

---

## 1. Purpose

Define the orientation structure used in RCF’s discrete geometric substrate.

This module specifies how simplices are assigned signs and how those signs are used to make boundary and coboundary operations consistent. No embedding, metric, or domain semantics are assumed.

---

## 2. Oriented simplices

Given a simplicial complex \(K\) with vertex set \(V\):

- A **0‑simplex** (vertex) has no orientation choice.
- A **1‑simplex** (edge) is an ordered pair \([v_i, v_j]\), with reversal \([v_j, v_i] = -[v_i, v_j]\).
- A **2‑simplex** (face) is an ordered triple \([v_i, v_j, v_k]\), defined up to even permutation.
- A **3‑simplex** (tetrahedron) is an ordered quadruple \([v_i, v_j, v_k, v_l]\), defined up to even permutation.

Orientation is a sign convention on ordered vertex lists, not a geometric direction.

---

## 3. Orientation equivalence

Two orderings of the same simplex are equivalent if they differ by an even permutation.

- Even permutation → same orientation
- Odd permutation → opposite orientation (sign flip)

Formally, for a \(k\)-simplex:

\[
[v_{\sigma(0)}, \dots, v_{\sigma(k)}] =
\mathrm{sgn}(\sigma)\,[v_0, \dots, v_k]
\]

where \(\mathrm{sgn}(\sigma) \in \{+1,-1\}\).

---

## 4. Induced orientation on faces

The orientation of a \(k\)-simplex induces orientations on its \((k-1)\)-faces via the boundary formula:

\[
\partial_k [v_0, \dots, v_k] =
\sum_{i=0}^k (-1)^i [v_0, \dots, \hat{v_i}, \dots, v_k]
\]

Each face inherits a sign from its position in the ordered list.

This ensures consistent orientation across shared faces when assembling complexes.

---

## 5. Orientation consistency

A simplicial complex is **orientation‑consistent** if:

- shared faces between adjacent simplices have opposite induced orientations (for interior faces), and
- global orientation choices do not produce contradictions on cycles.

This is a combinatorial condition; no embedding is required.

---

## 6. Interaction with boundary and coboundary

Orientation is the structure that makes:

\[
\partial_{k-1} \circ \partial_k = 0
\]

hold with correct signs.

The coboundary operator \(\delta_k = \partial_{k+1}^\top\) inherits its sign structure from the same orientation conventions. No additional assumptions are needed.

---

## 7. Structural role in RCF

Orientation provides:

- sign coherence for discrete calculus
- consistent assembly of local pieces into global structures
- a basis for defining flows, curls, and divergences on complexes (in later files)
- a purely combinatorial notion of “direction” without metric or embedding

It is a **core geometric invariant** of the simplicial substrate.

---

## 8. Minimal conceptual examples

- an oriented edge with reversed orientation
- a triangle with cyclic vs reversed vertex order
- two tetrahedra sharing a face with opposite induced face orientations

Examples are illustrative only.
