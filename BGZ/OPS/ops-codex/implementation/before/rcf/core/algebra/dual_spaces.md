# **Dual Spaces and Pairing Structure**

## Document Contract

- owner: RCF algebraic substrate
- type: Contract
- scope: dual-space and pairing grammar
- authority: tier-3 algebraic substrate contract
- upstream dependencies: `README.md`, `tensor_ops.md`, and `coalgebra.md`
- status: active
- purpose: define dual spaces, nondegenerate pairings, and optional adjoint structure
- anti-backflow: pairing structure does not introduce geometry, semantics, implementation, or new core invariants

`/core/algebra/dual_spaces.md`

---

## **1. Purpose**

Define the dual space, pairing relation, and adjoint structure used in RCF’s algebraic substrate.

This module specifies **structural invariants only**.
It assumes no geometry, semantics, or implementation.

---

## **2. Objects**

Let:

- \( V \) be a vector space
- \( V^* = \mathrm{Hom}(V, \mathbb{R}) \) be its dual
- \( \langle \cdot , \cdot \rangle : V^* \times V \to \mathbb{R} \) be a bilinear pairing

No interpretation is attached to elements of \( V \) or \( V^* \).

---

## **3. Dual Space Definition**

The dual space is:

\[
V^* = \{ f : V \to \mathbb{R} \mid f \text{ is linear} \}
\]

For \( f \in V^* \) and \( x \in V \):

\[
\langle f, x \rangle = f(x)
\]

The pairing is bilinear and nondegenerate.

---

## **4. Bilinearity**

For scalars \( \alpha, \beta \in \mathbb{R} \):

\[
\langle \alpha f_1 + \beta f_2, x \rangle = \alpha \langle f_1, x \rangle + \beta \langle f_2, x \rangle
\]

\[
\langle f, \alpha x_1 + \beta x_2 \rangle =
\alpha \langle f, x_1 \rangle +
\beta \langle f, x_2 \rangle
\]

---

## **5. Nondegeneracy**

The pairing satisfies:

- If \( \langle f, x \rangle = 0 \) for all \( x \), then \( f = 0 \)
- If \( \langle f, x \rangle = 0 \) for all \( f \), then \( x = 0 \)

This ensures information is not lost under dual evaluation.

---

## **6. Compatibility with Composition**

For multiplication \( m : V \otimes V \to V \)
and comultiplication \( \Delta : V \to V \otimes V \):

\[
\langle m(a \otimes b), c \rangle =
\langle a \otimes b, \Delta(c) \rangle
\]

This expresses **duality of synthesis and analysis**.

No semantics are implied.

---

## **7. Adjoint Operators**

Given a linear map \( T : V \to V \),
its adjoint \( T^\dagger : V \to V \) satisfies:

\[
\langle Tx, y \rangle = \langle x, T^\dagger y \rangle
\]

Adjoints are defined purely algebraically.

---

## **8. Optional Riesz-Type Isomorphism**

If \( V \) is equipped with a symmetric, nondegenerate bilinear form \( G \):

\[
\mathcal{R}(x)(y) = \langle y, x \rangle_G
\]

This is an isomorphism \( V \to V^* \).

RCF does **not** assume such a metric; this section is structural only.

---

## **9. Structural Role in RCF**

The dual‑space and pairing structure provides:

- reversible relational evaluation
- compatibility between algebraic operators
- the invariant needed for Hopf pairing
- a substrate for later geometric embeddings (without assuming geometry here)

This is a **core algebraic invariant layer**.

---

## **10. Minimal Conceptual Examples**

- evaluation of a linear functional
- adjoint of a permutation operator
- tensor contraction as induced pairing

Examples are illustrative only.
