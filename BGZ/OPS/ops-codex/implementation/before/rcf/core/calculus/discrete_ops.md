# **Discrete Operators**

## Document Contract

- owner: RCF discrete calculus substrate
- type: Contract
- scope: discrete differential-operator grammar
- authority: tier-3 calculus substrate contract
- upstream dependencies: `README.md`, `../geometry/boundary_ops.md`, `../geometry/coboundary_ops.md`, and `../geometry/cochains.md`
- status: active
- purpose: define gradient-like, curl-like, and divergence-like discrete operators
- anti-backflow: operator names do not imply metric derivatives, fields, dynamics, or physical semantics

`/core/calculus/discrete_ops.md`

## Purpose

Define discrete differential operators derived from the chain/cochain structure of the simplicial substrate.

These operators are algebraic consequences of the boundary and coboundary maps. No metric or embedding is assumed.

---

## 1. Operator Source

Let:

* ∂ = boundary operator (chains)
* δ = coboundary operator (cochains)
* Cₖ = k-chains
* Cᵏ = k-cochains

with:

\[
\partial^2 = 0,\quad \delta^2 = 0
\]

---

## 2. Discrete Gradient

The gradient is the coboundary acting on 0-cochains:

\[
\nabla := \delta : C^0 \rightarrow C^1
\]

It measures admissible 1-cochain variation.

No notion of direction or magnitude is assumed.

---

## 3. Discrete Curl

Curl is the coboundary acting on 1-cochains:

\[
\text{curl} := \delta : C^1 \rightarrow C^2
\]

This represents circulation structure induced by adjacency.

---

## 4. Discrete Divergence

Divergence is the dual boundary acting on 1-chains:

\[
\text{div} := -\partial : C_1 \rightarrow C_0
\]

The sign convention ensures compatibility with pairing and orientation.

---

## 5. Structural Constraints

All operators satisfy:

\[
\delta^2 = 0
\]

\[
\partial^2 = 0
\]

which implies:

* curl(grad) = 0
* div(curl) = 0

These identities are structural, not physical.

---

## 6. Scope

These operators:

* depend only on topology
* require no metric
* are invariant under relabeling
* extend linearly

Metric refinements are defined in downstream modules.
