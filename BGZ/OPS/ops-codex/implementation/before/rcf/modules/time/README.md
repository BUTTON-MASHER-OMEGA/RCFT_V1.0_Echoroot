# Time Module

## Document Contract

- owner: RCF time module
- type: Status
- scope: module-local content of `time/README.md`
- authority: tier-4 temporal ordering layer
- upstream dependencies: `/modules/time/interface.md`
- status: active
- purpose: define event index and refinement order without importing downstream rates, geometry, or dynamics
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Event Index, Refinement Time, and the Core Ontology of Temporal Order*

The Time module defines RCF's foundational event-ordering structure.
It is downstream of the non-temporal Event Axiom and independent of geometric content.

Time in RCF is **emergent**, **event‑indexed**, and **refinement‑driven**.
It is not a background dimension but a lawful ordering relation that begins only with the first discrete event (D₁).

This module provides:

- the **event index** \(n\), the primitive ordering relation;
- declared refinement-order relations;
- the interface contract that downstream modules may use for event indexing.

This module does **not** define:

- constraint density
- γ (memory drag)
- ε (freedom to move)
- local pacing
- time dilation
- relational pruning
- admissible mode counts

Those belong to the **temporal_interface/** layer.

The Time module is axiom-aligned and geometry-independent.
It defines temporal order, not a law of evolution or how hard it is to advance locally. Optional continuous or linear-unitary refinement representations are downstream profiles and are not part of the Time interface.
