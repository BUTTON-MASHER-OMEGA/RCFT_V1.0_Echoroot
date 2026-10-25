# Jacobian Entropy — Structural Properties

## Document Contract

- owner: RCF kinematics module
- type: Observation
- scope: module-local content of `kinematics/appendix/jacobian-entropy-properties.md`
- authority: tier-8 downstream change-representation layer
- upstream dependencies: `/modules/kinematics/interface.md`
- status: active
- purpose: represent admissible change without defining dynamics, causality, or temporal law
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## Purpose

This appendix formalizes structural properties of the scalar:

\[
H(J) := \log |\det J|
\]

used in the kinematics module as a refinement-stable measure of local structural distortion.

No thermodynamic, probabilistic, or physical interpretation is assumed.

H is treated purely as an invariant-sensitive scalar functional.

---

## 1. Preliminaries

Let:

* \(J\) = admissible local transformation
* admissible = invariant-preserving map within the kinematic module
* composition defined wherever domains match

Assume:

* determinant is defined on admissible transformations
* composition of admissible transformations is admissible
* invariants determine admissibility class

---

## 2. Theorem — Local Sensitivity

### Statement

Small admissible perturbations of \(J\) induce small changes in \(H(J)\).

### Formal Statement

If:

\[
J_\epsilon \rightarrow J
\]

under admissible perturbation, then:

\[
H(J_\epsilon) \rightarrow H(J)
\]

### Proof

determinant is continuous on admissible transformations.

log is continuous on positive reals.

Composition of continuous functions is continuous.

Therefore:

\[
H = \log \circ |\det|
\]

is continuous.

∎

---

## 3. Theorem — Additivity Under Composition

### Statement

For admissible transformations \(J_1, J_2\):

\[
H(J_2 \circ J_1)
\]
================

\[
H(J_2) + H(J_1)
\]

### Proof

By determinant multiplicativity:

\[
\det(J_2 J_1) = \det(J_2)\det(J_1)
\]

Taking absolute value:

\[
|\det(J_2 J_1)|
\]
===============
\[
|\det(J_2)||\det(J_1)|
\]

Taking logarithm:

\[
\log |\det(J_2 J_1)|
\]
====================

\[
\log |\det(J_2)|
+
\log |\det(J_1)|
\]

Thus:

\[
H(J_2 \circ J_1)
\]
================
\[
H(J_2) + H(J_1)
\]

∎

---

## 4. Theorem — Invariant Dependence

### Statement

If two transformations are invariant-equivalent, their entropy matches:

\[
J \sim J'
\Rightarrow
H(J) = H(J')
\]

### Proof Sketch

Invariant-equivalent transformations share admissible structural class.

Jacobian entropy depends only on structural distortion class.

Thus determinant magnitude is invariant-preserving.

Hence:

\[
H(J) = H(J')
\]

This is a structural requirement of admissibility.

∎

---

## 5. Theorem — Refinement Stability

Let refinement operator:

\[
R(J)
\]

preserve admissible invariant class.

### Statement

\[
H(R(J)) = H(J)
\]

### Proof

Refinement preserves invariant class.

By Theorem 4:

entropy depends only on invariant class.

Therefore:

\[
H(R(J)) = H(J)
\]

∎

---

## 6. Corollary — Entropy Descends to Quotient

Let:

\[
\pi : K \rightarrow \bar{K}
\]

be the kinematic quotient.

Then there exists:

\[
\bar{H} : \bar{K} \rightarrow \mathbb{R}
\]

such that:

\[
H = \bar{H} \circ \pi
\]

Entropy is well-defined on equivalence classes.

---

## 7. Structural Consequences

Jacobian entropy satisfies:

* continuity under admissible perturbation
* additivity under composition
* invariant dependence
* refinement stability
* quotient compatibility

Thus entropy is a structural scalar,
not an interpretive quantity.

It measures distortion class,
not energy, probability, or dynamics.

