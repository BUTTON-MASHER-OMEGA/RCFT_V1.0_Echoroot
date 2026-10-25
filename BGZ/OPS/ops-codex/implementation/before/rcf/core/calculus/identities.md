# **Identities**

## Document Contract

- owner: RCF discrete calculus substrate
- type: Contract
- scope: chain-cochain operator identities
- authority: tier-3 calculus substrate contract
- upstream dependencies: `README.md`, `../geometry/boundary_ops.md`, and `../geometry/coboundary_ops.md`
- status: active
- purpose: record boundary and coboundary composition identities
- anti-backflow: these mathematical identities do not add to or reinterpret the RCF invariant kernel

`/core/calculus/identities.md`

## Purpose

Formalize operator identities that must hold in any admissible chain/cochain complex.

These identities are invariants of the simplicial substrate.

---

## 1. Boundary Composition

Let B₁, B₂ be matrix representations of boundary maps:

\[
B_2 B_1 = 0
\]

This encodes:

> the boundary of a boundary is zero

---

## 2. Coboundary Composition

Since δ = Bᵀ:

\[
B_1^T B_2^T = 0
\]

or equivalently:

\[
\delta^2 = 0
\]

---

## 3. Dual Compatibility

These identities guarantee:

* no inconsistent loops
* no dangling boundary artifacts
* topological coherence

They are admissibility conditions, not optional constraints.

---

## 4. Structural Role

All higher operators must preserve these identities.

Any derived operator violating them is invalid.
