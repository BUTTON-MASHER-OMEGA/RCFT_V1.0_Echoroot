# RCMT Physics Invariants

## Document Contract

- owner: RCMT
- type: Invariant
- scope: RCMT-local; downstream of RCF
- authority: highest within RCMT, subordinate to all upstream RCF contracts
- purpose: constrain the admissible model space for RCMT physics modules
- anti-backflow: these invariants do not redefine RCF invariants, LCO ownership, Time, projection, or geometry

These invariants define the **admissible structural constraints** for physics modules within the Relational Coherence Mode Theory (RCMT).

They do not assert a complete physical model; they constrain the **space of valid models** that may be expressed downstream of the Relational Coherence Framework (RCF).

Any model violating these invariants is out of scope for RCMT.

---

## 1. Adjacency‑First Invariant
All admissible RCMT models must express physical structure strictly in terms of **adjacency relations**.  No global background metric, absolute space, or field‑first ontology is permitted.

---

## 2. Dual‑Lane Symmetry Invariant (Structural Requirement)
RCMT requires a **finite, adjacency‑compatible dual‑lane symmetry structure** consisting of:

- a **geometric lane** with local Lorentz symmetry, and
- a **non‑breaking symplectic mode‑space lane**.

**Current survivor implementation (candidates):**


\[
Sp(8) \times SO(9)
\]

\[
Sp(8) \times SO(8)
\]

Alternative implementations are admissible if they preserve the required dual‑lane structure and remain compatible with all invariants.

---

## 3. Local Coherence Operator (LCO) Invariant
All admissible RCMT models must consume the upstream **Local Coherence Operator (LCO)** contract when representing locally evaluated relational change.

Within RCMT, no global evolution law is introduced; apparent dynamics are represented as local mode transitions that remain admissible under upstream LCO scoring.

---

## 4. Mode‑Space Projection Invariant
RCMT models must represent observable physical quantities as **projections** of symplectic mode‑space structure into the geometric lane.

No model may treat projected quantities as fundamental primitives.

---

## 5. Mass Representation Constraint
All admissible RCMT models must represent mass in a form **reducible to LCO‑evaluated resistance** associated with changes in projected mode‑structure.

Mass may not be introduced as an independent field, intrinsic particle property, or symmetry‑breaking artifact.

---

## 6. Randomness Representation Constraint
RCMT models must not treat randomness as fundamental. Probabilistic behavior must arise from **unresolved mode‑structure under projection**.

---

## 7. Boundary‑Constraint Resolution Invariant
Measurement, entanglement, and decoherence must be represented using **boundary‑constraint resolution structures**.

No admissible model may introduce non‑local collapse or external measurement primitives.

---

## 8. Coherence Basin Constraint
RCMT models may represent large‑scale structure using **coherence basins** anchored by high‑stability attractors.

Any dark‑matter‑like phenomena must be representable as **non‑projectible mode‑structure**.

---

## 9. Local Curvature Invariant
Global curvature groups (e.g., \(SO(4,1)\), \(SO(4,2)\)) are not admissible.

Within RCMT, curvature must be represented as a **local, continuous shadow** compatible with upstream mismatch and geometry-dynamics descriptors.

---

## 10. Finite Structure Invariant

All symmetry structures used in RCMT must admit finite-dimensional, adjacency-compatible representations.

The **geometric lane** is subject to compactness constraints, except for local Lorentz structure introduced at the projection interface.

The **symplectic mode-space lane** is governed by its own preservation and admissibility constraints and is not required to be compact solely by this invariant.

No infinite towers, KK modes, string excitations, or non-compact global geometric groups are admissible.

---

## 11. Derived Physics Time Invariant
RCMT models must represent physics time (\(t\)) as a **macroscopic representation** of the discrete event index (\(n\)), modulated by local constraint density (\(\gamma\)).

Time may not be introduced as a fundamental dimension or global geometric axis.

---

## 12. Energy Representation Constraint
RCMT models must represent energy as a **derived measure of LCO computational load**
(\(\gamma\)-drag).  No admissible model may treat energy as a substance, field, or independent ontological
quantity.

---

## 13. Maximum Routing Rate (\(c\)) Constraint
RCMT models must treat the constant \(c\) as a **maximum local update rate** for adjacency refresh under the LCO.

No admissible model may interpret \(c\) as a geometric property of spacetime.

---

## 14. Local Information Flow Invariant
All information transfer must be **adjacency‑local**.

No global broadcast, instantaneous propagation, or non‑local information channel is permitted.

Long‑range correlations must arise from:
- shared prior structure,
- boundary‑constraint resolution,
- or sequential local propagation.

---

## 15. Structural Identity Constraint
RCMT models must treat physical identity as **structurally contingent**.

Identity persists only while compatible with local LCO evaluation.

No model may assume intrinsic or preserved identity across regions of high mismatch.

---

## Implementation Notes

### A. Dual‑Lane Symmetry (Provisional)
Section 2 outlines the **current survivor candidate(s)**.

Any replacement must preserve the dual‑lane structure and satisfy all invariants.

### B. Eliminability
These invariants are **downstream constraints**.

The RCF upstream core must remain valid without any specific physics interpretation.

## Status

- maturity: RCMT-local invariant set
- current implementation references: provisional unless explicitly identified as invariant
- compression priority: retain before all other `local/` files

---

## Open Questions
- Do gauge/charge structures require a dedicated invariant, or can they be fully absorbed
  into the mode‑space lane?
- Are additional invariants needed to constrain thermodynamic language?

