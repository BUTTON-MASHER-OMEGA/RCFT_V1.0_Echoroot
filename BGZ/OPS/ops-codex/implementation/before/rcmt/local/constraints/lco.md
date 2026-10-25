# RCMT LCO Consumption Contract

## Document Contract

- owner: RCMT
- type: Contract
- scope: RCMT consumption profile for upstream LCO outputs
- upstream_owner: RCF LCO module
- depends_on: upstream LCO contract and `../invariants.md`
- status: downstream compatibility specification
- anti-backflow: this file does not define, modify, or retune the LCO

This document defines how Relational Coherence Mode Theory (RCMT) **consumes the upstream Local Coherence Operator (LCO) contract**.
Within RCMT, LCO outputs are consumed as local structural admissibility scores.
It is downstream of the Relational Coherence Framework (RCF) and must remain compatible
with all RCMT invariants.

This file specifies:

- the representational role of the LCO,
- admissible evaluation behavior,
- constraints on mismatch and coherence thresholds,
- and its relationship to projection, measurement, and mode‑space.

It does **not** assert ontology or agency.

---

## 1. Structural role of LCO outputs in RCMT

RCMT consumes the LCO as a **local evaluative contract** for whether a proposed relational
configuration is admissible under:

- adjacency relations,
- local boundary conditions,
- and coherence thresholds.

The LCO is not:

- a physical force,
- a dynamical mechanism,
- a causal agent,
- or a global evolution law.

For RCMT, this is a **representational requirement** for locally coherent relational updates.

---

## 2. LCO evaluation domain

The LCO evaluates:

- **projected geometric configurations**,
- **mode‑space configurations**,
- and **their compatibility** under local constraints.

Admissible RCMT models must consume LCO records through an interface that can:

- assess local mismatch,
- determine admissibility of transitions,
- and enforce coherence thresholds.

Evaluation must remain **strictly local** and **adjacency‑compatible**.

---

## 3. Mismatch and coherence thresholds

The LCO evaluates **mismatch** between:

- current relational structure,
- proposed relational updates,
- and imposed boundary conditions.

Admissible models must:

- define mismatch as a local relational quantity,
- specify coherence thresholds that determine admissibility,
- and avoid introducing global mismatch metrics.

Coherence thresholds are representational constraints, not physical potentials.

---

## 4. No global evolution law

The LCO does not define a global evolution rule.

Admissible models must:

- treat apparent dynamics as sequences of locally admissible transitions,
- avoid global update rules or universal time parameters,
- and remain compatible with the Derived Time Invariant.

Local admissibility determines the appearance of evolution.

---

## 5. Relationship to projection

Projection maps mode‑space structure into the geometric lane.
The LCO evaluates the **projected configuration**, not the underlying mode‑structure
directly.

Admissible models must:

- define projection in a form compatible with LCO evaluation,
- ensure that LCO mismatch is computed on projected quantities,
- and avoid treating projection as an ontological collapse.

---

## 6. Relationship to measurement

Measurement is modeled as **boundary‑constraint resolution**.
The LCO evaluates which configurations survive under the imposed boundary.

Admissible models must:

- represent measurement using LCO evaluation under boundary constraints,
- avoid introducing special measurement dynamics,
- and maintain compatibility with adjacency‑local information flow.

---

## 7. Relationship to randomness

Unresolved mode‑structure produces probabilistic outcomes under projection.
RCMT uses upstream LCO scores to classify which projected configurations remain admissible.

Admissible models must:

- treat randomness as induced by unresolved mode‑structure,
- avoid treating randomness as fundamental,
- and represent probabilistic behavior through LCO-compatible projection records.

---

## 8. Relationship to mass

Within RCMT, mass is represented as **resistance** recorded when projected mode-structure
is assessed against upstream LCO mismatch thresholds.

Admissible models must:

- represent mass as reducible to LCO‑evaluated resistance,
- avoid introducing mass as an independent field or intrinsic property,
- and maintain compatibility with the Mass Representation Constraint.

---

## 9. Locality and information flow

The LCO must operate **strictly locally**.

Admissible models must:

- ensure that LCO evaluation does not introduce non‑local dependencies,
- maintain adjacency‑local information flow,
- and avoid global broadcast or instantaneous propagation.

---

## 10. Compatibility with RCMT invariants

The LCO must satisfy:

- **Local Coherence Operator Invariant**
- **Boundary‑Constraint Resolution Invariant**
- **Mode‑Space Projection Invariant**
- **Randomness Representation Constraint**
- **Mass Representation Constraint**
- **Local Information Flow Invariant**
- **Derived Time Invariant**

---

## 11. Status and future revisions

- **Status:**
  This is the RCMT v1.x LCO specification.

- **Intended use:**
  This file serves as the reference for any RCMT module that depends on local coherence
  evaluation (e.g., measurement, projection, mass representation, or decoherence).

