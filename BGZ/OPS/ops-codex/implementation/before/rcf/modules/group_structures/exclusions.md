# Exclusions

## Document Contract

- owner: RCF optional group-structure profile
- type: Contract
- scope: module-local content of `group_structures/exclusions.md`
- authority: tier-4 optional algebraic profile
- upstream dependencies: `/modules/group_structures/interface.md`
- status: active
- purpose: classify opt-in group structures without making a group family an RCF invariant
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

This file documents what the Abstract Group Structures module must *not* include.
These exclusions protect the module’s role as a minimal, structural, and
domain‑neutral vocabulary layer.

The items below are excluded because they would introduce new invariants,
duplicate upstream mechanisms, or pull domain semantics into a module that must
remain eliminable.

---

## 1. Infinite Conformal and Scale-Invariant Groups

Groups such as `SO(4,2)` are excluded when they violate:

- bounded mismatch constraints,
- finite-resolution survivor structure,
- or admissibility rules defined in upstream geometry.

**Reason:**
These groups introduce unbounded structure incompatible with discrete survivor
constraints.

---

## 2. New Projection Behavior

This module may not:

- define new projection operators,
- redefine Π,
- introduce new projection invariants,
- or describe projection mechanisms.

**Reason:**
Projection behavior is owned entirely by `geometry/projection`.
This module may reference projection transitions only through labels.

---

## 3. New Geometric or Combinatorial Structure

This module may not introduce:

- new vertices,
- new edges,
- new motif classes,
- new combinatorial rules,
- new survivor criteria.

**Reason:**
All geometric and combinatorial structure is owned by the geometry modules.

---

## 4. Domain-Specific Interpretation

This module may not include:

- physics semantics,
- cosmological narratives,
- cognitive interpretations,
- ML analogies,
- operator or particle meaning,
- Standard Model mappings.

**Reason:**
Meaning belongs downstream.
This module provides structure only.

---

## 5. Narrative Pipelines or Symmetry Stories

This module may not define:

- symmetry degradation chains,
- plenum/airlock narratives,
- E8 → D5 → … pipelines,
- memory-mass or phase-space stories.

**Reason:**
These are domain-level interpretations and belong in downstream modules.

---

## 6. Upstream Reinterpretation

This module may not:

- reinterpret upstream geometry,
- reinterpret projection outputs,
- introduce new invariants into the core,
- modify admissibility rules defined elsewhere.

**Reason:**
The abstract module must remain eliminable and must not alter upstream meaning.

---

## Status

These exclusions ensure that the Abstract Group Structures module remains a
minimal, structural vocabulary layer. It introduces no new mechanisms, no new
invariants, and no domain semantics.

