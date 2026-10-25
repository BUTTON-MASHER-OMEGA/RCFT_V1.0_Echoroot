# Types

## Document Contract

- owner: RCF optional group-structure profile
- type: Interpretation
- scope: module-local content of `group_structures/types.md`
- authority: tier-4 optional algebraic profile
- upstream dependencies: `/modules/group_structures/interface.md`
- status: active
- purpose: classify opt-in group structures without making a group family an RCF invariant
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

This file defines the structural labels used by the Abstract Group Structures module.
These labels do not introduce new mechanisms or new invariants. They provide a shared
vocabulary for downstream modules that need to reference symmetry structure without
redefining geometry or projection behavior.

Each type below is a *label*, not an operator or mechanism.

---

## `group-state-label`

A label indexing an admissible symmetry configuration compatible with upstream
discrete or continuous geometry.

**Intuition:**
Think of this as a “name” for a symmetry regime. It does not describe what the
symmetry means in any domain. It simply identifies a structural configuration
that geometry has already made admissible.

**Constraints:**
- Must correspond to a survivor discrete geometry, or
- Must correspond to a continuous group with a valid continuous-limit-map.

---

## `projection-operator-label`

A label indexing an admissible projection transition already defined in
`geometry/projection`.

**Intuition:**
This is not a projection operator. It is a *reference* to a projection transition
that the projection module already defines. It lets downstream modules talk about
projection structure without redefining Π.

**Constraints:**
- Must correspond to an admissible projection rule in `geometry/projection`.
- Cannot introduce new projection behavior or invariants.

---

## `continuous-limit-map`

A label indexing an invertible-up-to-epsilon mapping between:
- a continuous manifold representation, and
- a discrete survivor geometry.

**Intuition:**
This is a label for a mapping that lets a continuous symmetry group approximate
a discrete survivor structure. It does not define the mapping itself; the
projection module defines the rules.

**Constraints:**
- Must preserve adjacency and survivor constraints.
- Must be invertible up to epsilon.

---

## `phase-orientation-band`

A label indexing a relative orientation class of a `group-state-label` with
respect to a baseline.

**Intuition:**
This is a structural orientation tag. It does not represent a physical phase or
a domain-specific angle. It simply identifies how a symmetry configuration is
oriented relative to a reference orientation.

**Constraints:**
- Must index a relative orientation class.
- Must not introduce domain semantics.

---

## Status

These types define the minimal vocabulary for symmetry-related structure in RCF.
They are labels only and introduce no new mechanisms or invariants.

