# 600-cell/spinor-embedding.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/600-cell/spinor-embedding.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose
This document records how the 120 vertices of the 600‑cell embed into a spinor representation compatible with RCF update rules. The goal is to document the structural relationship between the 600‑cell’s vertex set and the unit icosians without interpreting downstream consequences.

---

## overview
The 600‑cell admits a clean embedding into a spinor structure via the **unit icosians**, a finite subset of the quaternions forming the binary icosahedral group. In this module, the embedding is treated as:

- a mapping from 600‑cell vertices to unit quaternions
- a representation that is compatible with adjacency relations under spinor composition
- an algebraic structure that supports closure under recursive updates

Coordinates and explicit quaternion values are optional and documented separately.

---

## unit icosian correspondence
A standard embedding identifies each vertex of the 600‑cell with one of the 120 **unit icosians**:

- a unit icosian is a quaternion \( q = w + xi + yj + zk \) of unit norm
- the 120 icosians form a discrete, finite subgroup of \( SU(2) \)
- the mapping preserves the 600‑cell’s adjacency and symmetry structure

This correspondence provides an algebraic representation of the 600‑cell’s relational modes.

---

## algebraic closure
The unit icosians form the **binary icosahedral group**, which is closed under quaternion multiplication. Under this embedding:

- the product of any two mapped vertices is another mapped vertex
- compositions correspond to valid relational updates
- no new elements are generated under repeated multiplication

Closure is documented here as an algebraic property of the embedding, not as a survival claim.

---

## role in RCF (structural only)
- provides a spinor representation compatible with LCO/GCO update rules
- provides a representation in which relational updates can be expressed as spinor compositions
- supports stable recursion in standard constraint‑density regimes

Interpretations of survival behavior are documented in `survival-stats.md`.

---

## invariants
- the embedding maps all 120 vertices to unit icosians
- the mapped set is closed under quaternion multiplication
- closure is an algebraic property of the embedding, independent of interpretation

