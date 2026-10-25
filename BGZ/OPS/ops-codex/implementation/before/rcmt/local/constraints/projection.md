# RCMT Mode-Space Projection

## Document Contract

- owner: RCMT
- type: Contract
- scope: RCMT mode-space-to-geometric-lane mapping
- depends_on: upstream projection outputs, `../invariants.md`, `symmetry_chain.md`, and `mode_space.md`
- status: module-local projection contract
- anti-backflow: this file does not redefine the upstream discrete-to-continuous projection handoff

This document defines the **RCMT-local projection structure** required between the RCMT mode-space and geometric lanes.

RCMT-local projection is the representational bridge between the symplectic mode-space lane and the geometric lane.

It is downstream of the Relational Coherence Framework (RCF) and must remain compatible with the RCMT invariants, the symmetry chain, and the mode‑space specification.

This file describes:

- the structural role of projection,
- admissible projection maps,
- constraints on representational behavior,
- and requirements for compatibility with LCO evaluation.

It does **not** assert ontology; it constrains admissible projection structures.

---

## 1. Structural role of projection

Projection provides the **representational interface** between:

- the **mode‑space lane** (symplectic, non‑breaking), and
- the **geometric lane** (Lorentz‑compatible, compact internal symmetry).

Admissible RCMT models must treat projection as:

- a **structural mapping**,
- not a physical process,
- not a dynamical mechanism,
- and not an ontological collapse.

For RCMT, this mapping records how mode-space structure becomes representationally visible in the geometric lane.

---

## 2. Projection map requirements

RCMT models must define a projection map:



\[
\Pi : Sp(2n) \;\longrightarrow\; \mathcal{G}
\]



where \(\mathcal{G}\) is the geometric lane (locally \(SO(3,1)\) with compact internal
structure).

The projection map must satisfy:

- **locality:**
  projection must not introduce non‑local information channels,

- **adjacency‑compatibility:**
  projection must preserve adjacency relations defined in the upstream substrate,

- **finite representability:**
  projection must map finite symplectic representations to finite geometric
  representations,

- **non‑invertibility:**
  projection must not be representationally invertible; mode‑space must contain
  structure not recoverable from geometric data.

No admissible model may define a projection map that violates these constraints.

---

## 3. Projection and representational shadows

All observable physical quantities must be representable as **shadows** of mode‑space structure under projection.

Admissible models must:

- treat projected quantities as **derived**,
- avoid introducing geometric primitives not reducible to projection,
- and ensure that projection does not create new degrees of freedom.

Examples of representational shadows include:

- geometric quantities derived from mode‑structure,
- local mismatch appearing as curvature,
- unresolved mode‑structure appearing as probabilistic behavior.

These are representational categories, not ontological claims.

---

## 4. Compatibility with LCO evaluation

RCMT-local projection must expose records that can be consumed by the upstream Local Coherence Operator (LCO) contract to:

- evaluate mismatch between projected structures,
- determine admissibility of local transitions,
- and enforce coherence thresholds.

This requirement constrains the **form** of projection, not the **mechanism**.

Projection must not:

- grant the LCO agency,
- introduce global evolution laws,
- or create non‑local dependencies.

---

## 5. Projection and unresolved mode‑structure

Admissible RCMT models must allow for **unresolved mode‑structure** in the symplectic
lane that remains unprojected.

This unresolved structure must:

- remain representationally distinct from geometric quantities,
- be capable of producing probabilistic behavior under projection,
- and remain compatible with adjacency‑local information flow.

Projection must not:

- treat unresolved structure as hidden variables,
- or introduce ontological randomness.

---

## 6. Projection and geometric compatibility

Projection must produce geometric representations that:

- are compatible with local Lorentz structure,
- admit compact internal symmetry,
- and remain finite and adjacency‑compatible.

No admissible model may introduce:

- global curvature groups,
- non‑compact global symmetries,
- or infinite representational towers.

---

## 7. Relation to RCMT invariants

Projection must satisfy:

- **Mode‑Space Projection Invariant**
  (all observables arise from projection)

- **Adjacency‑First Invariant**
  (projection must preserve adjacency)

- **Randomness Representation Constraint**
  (probabilities arise from unresolved mode‑structure)

- **Mass Representation Constraint**
  (mass must be reducible to LCO‑evaluated resistance in projected mode‑structure)

- **Local Information Flow Invariant**
  (projection must not introduce non‑local channels)

- **Local Curvature Constraint**
  (curvature must be representational, not fundamental)

- **Finite Structure Invariant**
  (projection must map finite structures to finite structures)

---

## 8. Status and future revisions

- **Status:**
  This is the RCMT v1.x projection specification.

- **Provisional elements:**
  The specific form of the projection map may vary across admissible models, provided
  all constraints are satisfied.

- **Intended use:**
  This file serves as the reference for any RCMT module that depends on projection
  structure (e.g., measurement, decoherence, mass representation, or geometric
  interpretation).

