# Special Commutative Frobenius Algebra

## Document Contract

- owner: RCF algebraic substrate
- type: Contract
- scope: optional algebra-coalgebra compatibility profile
- authority: optional tier-3 substrate specialization
- upstream dependencies: `README.md`, `tensor_ops.md`, and `coalgebra.md`
- status: optional
- purpose: define an SCFA merge-split compatibility pattern for adopting modules
- anti-backflow: SCFA assumptions apply only where explicitly adopted and do not constrain the RCF invariant kernel or all substrate implementations

`/core/algebra/scfa.md`

---

## 1. Purpose

Define the Special Commutative Frobenius Algebra (SCFA) structure used in RCF as a canonical algebra–coalgebra compatibility pattern.

This module specifies **algebraic identities only**.
No geometry, semantics, or implementation is assumed.

---

## 2. Objects

Let:

- \( V \) be a finite‑dimensional vector space
- \( m : V \otimes V \to V \) be multiplication
- \( \Delta : V \to V \otimes V \) be comultiplication
- \( \eta : \mathbb{R} \to V \) be the unit
- \( \epsilon : V \to \mathbb{R} \) be the counit

These form an algebra and a coalgebra on the same space.

---

## 3. Algebra and Coalgebra Laws

### Associative, Unital Algebra

\[
m \circ (id \otimes m) = m \circ (m \otimes id)
\]

\[
m(\eta \otimes id) = id = m(id \otimes \eta)
\]

### Coassociative, Counital Coalgebra

\[
(\Delta \otimes id)\Delta = (id \otimes \Delta)\Delta
\]

\[
(\epsilon \otimes id)\Delta = id = (id \otimes \epsilon)\Delta
\]

---

## 4. Commutativity and Cocommutativity

The algebra is **commutative**:

\[
m \circ \tau = m
\]

The coalgebra is **cocommutative**:

\[
\tau \circ \Delta = \Delta
\]

where \( \tau : V \otimes V \to V \otimes V \) swaps tensor factors.

---

## 5. Frobenius Law

The Frobenius compatibility condition is:

\[
(id \otimes m)\circ(\Delta \otimes id) =
\Delta \circ m =
(m \otimes id)\circ(id \otimes \Delta)
\]

This ties synthesis and analysis into a single coherent structure.

---

## 6. Specialness

The algebra is **special** if:

\[
m \circ \Delta = \alpha \, id
\]

for some nonzero scalar \( \alpha \).

After rescaling, one may take \( \alpha = 1 \):

\[
m \circ \Delta = id
\]

This expresses that comultiplication followed by multiplication is an isomorphism up to scale.

---

## 7. Structural Role in RCF

SCFA provides:

- a canonical pattern for compatible merge/split operations
- a tightly coupled algebra–coalgebra pair
- a reusable grammar for compositional and decompositional structure

It is a **template**, not a commitment to any specific domain.

---

## 8. Minimal Conceptual Examples

- finite‑dimensional commutative algebras with a compatible trace
- copy/delete structures in categorical formulations
- orthonormal basis algebras with diagonal comultiplication

Examples are illustrative only.

---

## 9. Invariant Role

Within RCF’s algebraic substrate, SCFA:

- encodes a maximally compatible merge/split pattern
- supports reversible refinement and aggregation
- provides a canonical algebraic backbone for later geometric or kinematic constructions

It is a **core‑adjacent algebraic pattern**, available where such tight compatibility is required.
