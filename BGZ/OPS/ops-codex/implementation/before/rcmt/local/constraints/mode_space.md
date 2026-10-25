# RCMT Mode Space

## Document Contract

- owner: RCMT
- type: Contract
- scope: module-local mode-space requirements
- depends_on: `../invariants.md` and `symmetry_chain.md`
- status: required dual-lane role with provisional group realization

This document defines the **mode‑space lane** required by Relational Coherence Mode Theory (RCMT).

It is downstream of the Relational Coherence Framework (RCF) and must remain compatible with the RCMT invariants and the symmetry chain specification.

This file describes:

- the structural requirements of the symplectic mode‑space,
- its relationship to the geometric lane,
- admissible representations,
- and constraints on projection and LCO evaluation.

It does **not** assert ontology; it constrains admissible model structure.

---

## 1. Structural role of the mode‑space lane

RCMT requires a **non‑breaking symplectic mode‑space lane** that:

- provides the structural domain for mode‑structure not present in the geometric lane,
- supports a measure structure suitable for induced probabilistic behavior,
- remains finite and adjacency‑compatible,
- and participates in dual‑lane projection into the geometric lane.

The mode‑space lane is not a physical container or hidden substrate. It is a **representational domain** required by the dual‑lane symmetry architecture.

---

## 2. Symplectic requirement

All admissible RCMT models must represent mode‑space using a **symplectic group** of the form:



\[
Sp(2n)
\]



with the following constraints:

- the group must be **finite in representation**,
- the group must be **adjacency‑compatible**,
- the group must **not break** under admissible dynamics,
- and the group must support a **non‑degenerate bilinear form** suitable for projection.

The current provisional implementation uses:



\[
Sp(8)
\]



but any \(Sp(2n)\) satisfying the above constraints is admissible.

---

## 3. Relationship to the geometric lane

The mode‑space lane and geometric lane must satisfy:

- **independence:**
  neither lane may be representationally reducible to the other,

- **compatibility:**
  projection from mode‑space into the geometric lane must be well‑defined,

- **duality:**
  the geometric lane provides local Lorentz structure;
  the mode‑space lane provides non‑geometric relational structure.

No admissible model may collapse the two lanes into a single representational domain.

---

## 4. Projection constraint

All observable physical quantities must arise from **projection** of mode‑space structure into the geometric lane.

Admissible models must:

- define a projection map from \(Sp(2n)\) into the geometric lane,
- ensure that projected quantities are representational shadows, not primitives,
- and ensure that projection does not introduce non‑local information channels.

Projection must be compatible with:

- the Local Coherence Operator (LCO),
- the Derived Time Invariant,
- and the Maximum Routing Rate constraint.

---

## 5. LCO evaluation constraint

Mode‑space structure must be representable in a form that allows the LCO to:

- evaluate local mismatch,
- determine admissibility of mode‑transitions,
- and enforce local coherence thresholds.

This requirement does **not** redefine or grant agency to the upstream LCO.
It constrains the RCMT representation so that upstream LCO scoring remains consumable.

---

## 6. Unresolved mode‑structure

Admissible RCMT models must permit representations containing **unresolved mode‑structure** in the symplectic lane.

This unresolved structure must:

- remain representationally distinct from geometric quantities,
- be capable of producing probabilistic behavior under projection,
- and remain compatible with adjacency‑local information flow.

No model may treat unresolved mode‑structure as a hidden variable or ontological entity.

---

## 7. Compatibility with RCMT invariants

The mode‑space lane must satisfy:

- **Adjacency‑First Invariant**
  (mode‑structure must be relational)

- **Dual‑Lane Symmetry Invariant**
  (mode‑space must remain symplectic and non‑breaking)

- **Mode‑Space Projection Invariant**
  (all observables arise from projection)

- **Randomness Representation Constraint**
  (probabilities arise from unresolved mode‑structure)

- **Mass Representation Constraint**
  (mass must be reducible to LCO‑evaluated resistance in projected mode‑structure)

- **Local Information Flow Invariant**
  (no non‑local channels may be introduced)

- **Finite Structure Invariant**
  (no infinite towers or non‑compact global groups)

---

## 8. Status and future revisions

- **Status:**
  This is the RCMT v1.x mode‑space specification.

- **Provisional elements:**
  The specific choice of \(Sp(8)\) is a survivor implementation and may be replaced by any \(Sp(2n)\) satisfying all constraints.

- **Intended use:**
  This file serves as the reference for any RCMT module that depends on mode‑space structure (e.g., measurement, decoherence, mass representation, or projection rules).

