# LCO Invariants

## Document Contract

- owner: RCF local coherence operator
- type: Contract
- scope: module-local content of `lco/lco-invariants.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Properties the Local Coherence Operator Must Preserve*

This document specifies the **invariants** that the Local Coherence Operator (LCO) is required to maintain. Unlike primitives (which define what the LCO may use), invariants define what the LCO must **guarantee** about its behavior and outputs.

These invariants ensure that the execution layer remains stable, well‑defined, and neutral with respect to downstream geometry.

---

## **1. Unit‑Norm Mode Invariant**

**Invariant:**
All mode vectors handled by the LCO (inputs and outputs) must satisfy:


\[
\|m\| = 1
\]



**Implication:**
- The LCO may normalize candidate modes.
- The mode space remains a unit‑norm comparison space at all times.
- No drift in norm is allowed.

---

## **2. Locality Invariant**

**Invariant:**
The LCO’s evaluation of a candidate update depends **only** on:

- local relation vectors \( r_i \)
- the memory anchor \( H \)
- the candidate mode \( m \)
- the scalar γ

**Implication:**
- No global state, global statistics, or non‑local information may be used.
- The operator is strictly local in its inputs and behavior.

---

## **3. Geometry‑Neutrality Invariant**

**Invariant:**
The LCO must not depend on any specific geometric or topological structure beyond the unit‑norm mode space used for directional comparison.

**Implication:**
- No references to polytopes, packings, lattices, or coordinate systems.
- No assumptions about physical space, dimensionality preferences, or topology.
- The unit mode space is a representational device only, not a geometric ontology.

---

## **4. Determinism Invariant**

**Invariant:**
Given the same inputs \((\{r_i\}, H, m, \gamma)\), the LCO must produce a consistent coherence score.

**Implication:**
- The operator itself is deterministic.
- Any stochasticity belongs to the update policy, not the operator.

---

## **5. Well‑Defined Coherence Score Invariant**

**Invariant:**
For any admissible input configuration, the LCO must produce a finite, well‑defined scalar coherence score.

**Implication:**
- No undefined operations (division by zero, NaNs, etc.).
- No dependence on ill‑posed quantities.
- The score must always be computable from the primitives.

---

## **6. Monotonic Local Descent Capability**

**Invariant:**
The LCO must be designed such that, under admissible update policies, it can support **local coherence descent**:

- If a candidate update is accepted, it should not increase local incoherence relative to the current state.

**Implication:**
- The operator’s scoring function must be compatible with Lyapunov‑like descent at the local level.
- The update policy governs descent behavior, but the operator must not obstruct it.

---

## **7. Independence from Global Geometry and φ/ε Structure**

**Invariant:**
The LCO must not encode:

- φ‑specific behavior
- ε‑band structure
- any global geometric or topological preference

**Implication:**
- Any appearance of φ, ε, or geometric structure must arise downstream (e.g., persistence or representation), not from the LCO’s internal logic.

---

## **8. γ as External Parameter Invariant**

**Invariant:**
The LCO must treat γ as an **external parameter**, not something it selects or optimizes.

**Implication:**
- γ is provided by the temporal-interface module.
- The LCO only applies γ in its memory weighting; it does not infer or adjust it.

---

### **Status**

These invariants define the **behavioral contract** of the LCO.
Any implementation, refinement, or extension of the operator must preserve all of these properties to remain valid within the RCF execution layer.

