### rcmt/global/constraints.md

#### Constraints on RCMT Global Invariants

##### 1. Purpose
This document specifies the admissibility criteria for implementing RCMT’s three global invariants:

- twist,
- winding,
- holonomy.

An implementation is admissible only if these invariants remain globally encoded, locally trivial, and resistant to all purely local operations, while remaining compatible with the RCMT substrate.

---

##### 2. Normative Form
The terms **must**, **must not**, and **may** are used as normative requirements.

- **Must**: required for admissibility.
- **Must not**: forbidden.
- **May**: optional representation detail.

---

##### 3. Adjacency-First Representation Constraint
Global invariants must be representable strictly in terms of adjacency and transport over adjacency.

- They must not require a standalone field.
- They must not require a coordinate system, metric, or geometric primitive.
- Twist must be represented as mismatch in transported orientation over an adjacency loop.
- Winding must be represented as non-contractible adjacency linkage.
- Holonomy must be represented as accumulated relational transformation along an adjacency path.

An implementation is inadmissible if it cannot be evaluated from adjacency structure alone.

---

##### 4. Locality and Operator Invariance Constraint
Local operators such as LCO or GCO may simplify, prune, or regularize a neighborhood, but they must not be able to erase a genuine global invariant through purely local steps.

- A valid implementation must preserve the invariant under all admissible local cleanup procedures.
- The invariant must remain detectable after local repair if it is structurally present.

Any operator that can remove twist, winding, or holonomy by local action alone is out of scope.

---

##### 5. Dual-Lane Compatibility Constraint
Global invariants must remain compatible with RCMT’s dual-lane symmetry structure:

- geometric lane,
- symplectic mode-space lane.

They may be defined in one lane or span both, but they must not:

- break lane separation,
- privilege one lane as fundamental,
- require a non-adjacency-dependent representation.

---

##### 6. Finite and Compact Representation Constraint
All representations of global invariants must use finite, compact, and bounded symmetry structures.

- No infinite towers.
- No non-compact global groups.
- No unbounded mode proliferation.
- No string-like excitation ladders.

The representation must remain compatible with RCMT’s finite-structure posture.

---

##### 7. Stability Under Constraint Density Constraint
In highly constrained regions, global invariants may become discrete or quantized, but they must remain stable under local update.

- Local operators must not continuously tune or drift the invariant value in a locked regime.
- An implementation should support discrete stability as a representational option under high constraint density.

---

##### 8. Local Detectability and Global Non-Localizability Constraint
Global invariants must be detectable through closed-path or closed-cycle evaluation, but they must not be localizable to any single patch.

- A local patch may appear coherent while the full closed structure remains nontrivial.
- The invariant must be encoded in the whole structure, not in a single node, edge, or local neighborhood.

---

##### 9. Upstream Non-Backflow Constraint
Global invariants may constrain downstream interpretation, but they must not redefine or replace upstream RCMT primitives.

They must not:

- redefine adjacency,
- redefine local mismatch,
- redefine local coherence,
- redefine local constraint semantics,
- redefine any upstream structural primitive.

They are downstream relational facts, not new primitives.

> They must not backflow into upstream RCMT local invariants or local constraints. Global invariants may not redefine triangle primitives, mismatch evaluation rules, adjacency semantics, local constraint capsules, or any aspect of LCO/GCO behavior. These upstream structures must remain strictly prior to global invariant interpretation.


---

##### 10. Projection Compatibility Constraint
Global invariants must remain compatible with RCMT’s projection behavior.



- Twist projects as orientation mismatch.
- Winding projects as non-contractible topology.
- Holonomy projects as accumulated transformation.

Projection may be used as an interpretive layer, but it must not be the fundamental definition of the invariant.

---

##### 11. Acceptance Criterion
An implementation is admissible if it satisfies all of the above constraints and preserves the distinction among:

- twist as orientation mismatch,
- winding as non-contractible linkage,
- holonomy as accumulated transport transformation.

These constraints ensure that the three invariants remain structurally admissible within RCMT and cannot be erased or reinterpreted by local operators or downstream modules.
