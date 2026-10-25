# Admissibility Rules

## Document Contract

- owner: RCF optional group-structure profile
- type: Contract
- scope: module-local content of `group_structures/admissibility.md`
- authority: tier-4 optional algebraic profile
- upstream dependencies: `/modules/group_structures/interface.md`
- status: active
- purpose: classify opt-in group structures without making a group family an RCF invariant
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

These rules define when a label in the Abstract Group Structures module is valid.
They ensure that all symmetry-related labels remain compatible with upstream
geometry and projection, without introducing new mechanisms or domain semantics.

Each rule below applies to *labels only*, not operators or transformations.

---

## `group-state-label` Admissibility

A `group-state-label` is admissible only if it corresponds to structure that
already exists in the upstream geometry modules.

A label is admissible when:

- it corresponds to a survivor discrete geometry, **or**
- it corresponds to a continuous symmetry group with a valid
  `continuous-limit-map` into a survivor geometry.

**Intuition:**
A group-state-label must point to something geometry already allows.
It cannot invent new symmetry structure.

---

## `projection-operator-label` Admissibility

A `projection-operator-label` is admissible only if:

- it indexes a projection transition already defined in `geometry/projection`,
- it preserves bounded mismatch,
- it preserves finite-resolution survivor structure.

**Intuition:**
This is a reference to an existing projection rule.
It cannot define new projection behavior or override Π.

---

## `continuous-limit-map` Admissibility

A `continuous-limit-map` is admissible only if:

- it is invertible up to epsilon,
- it preserves adjacency structure,
- it preserves survivor constraints,
- it corresponds to a mapping allowed by `geometry/projection`.

**Intuition:**
This label identifies a mapping that lets a continuous symmetry group approximate
a discrete survivor. It must respect the same structural limits projection uses.

---

## `phase-orientation-band` Admissibility

A `phase-orientation-band` is admissible only if:

- it indexes a relative orientation class of a `group-state-label`,
- it does not introduce domain semantics,
- it does not imply physical phase, angle, or operator meaning.

**Intuition:**
This is a structural orientation tag.
It identifies how a symmetry configuration is oriented relative to a baseline,
not what that orientation means in any domain.

---

## Status

These rules ensure that all labels in this module remain compatible with
upstream geometry and projection, while introducing no new invariants or
mechanisms. The module remains eliminable and domain-neutral.

