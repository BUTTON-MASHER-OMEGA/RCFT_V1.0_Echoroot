# LCO Primitives

## Document Contract

- owner: RCF local coherence operator
- type: Interpretation
- scope: module-local content of `lco/lco-primitives.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Allowed Ingredients for Local Coherence Evaluation*

This document specifies the **primitive objects and operations** the Local Coherence Operator (LCO) is allowed to use. These are the raw ingredients from which the operator and update policies are constructed. They are **not** invariants; they are the *building blocks*.

The goal is to make explicit what the LCO can “see” and manipulate, while remaining fully neutral with respect to any downstream geometry or representation.

---

## **1. Mode Space Primitive**

- **Unit‑norm mode space:**
  The LCO operates on mode vectors that inhabit a normalized space used purely for directional comparison.

- **Representation:**
  - Modes are represented as vectors \( m \) with \( \|m\| = 1 \).
  - This space is a **representational device**, not a claim about physical geometry or ontology.

---

## **2. Local Relation Primitives**

- **Relation vectors \( r_i \):**
  - Each \( r_i \) is a unit‑norm vector representing a local relational direction.
  - The set \( \{r_i\} \) defines the local neighborhood relevant to a candidate update.

- **Properties:**
  - \( \|r_i\| = 1 \) for all \( i \).
  - No assumptions are made about how these relations are embedded or realized physically.

---

## **3. Memory Primitive**

- **Memory anchor \( H \):**
  - A unit‑norm vector representing the local historical direction or prior state.
  - Used to incorporate temporal influence into the coherence evaluation.

- **Scalar memory weight \( \gamma \):**
  - A scalar in \([0,1]\) that weights the contribution of the memory anchor.
  - Provided by the time module, not chosen by the LCO.

---

## **4. Similarity Primitive**

- **Dot product \( r \cdot m \):**
  - The only primitive used to compare directions.
  - Used to define mismatch between relations, memory, and candidate modes.

- **Rationale:**
  - Directional similarity is captured purely through inner products.
  - No explicit coordinate systems, distances, or external geometric structures are assumed.

---

## **5. Mismatch and Aggregation Primitives**

- **Mismatch values:**
  - For each relation \( r_i \) and candidate mode \( m \):


\[
    \ell_i = 1 - r_i \cdot m
    \]


  - For memory:


\[
    \ell_{\text{mem}} = \gamma (1 - H \cdot m)
    \]



- **Aggregation:**
  The LCO may compute:
  - means (e.g., average mismatch)
  - variances (e.g., dispersion of mismatch)
  - simple algebraic combinations of these quantities

- **Constraints:**
  - All aggregation is local and algebraic.
  - No global statistics or global state are used.

---

## **6. Candidate Mode Primitive**

- **Candidate mode \( m \):**
  - A unit‑norm vector proposed as a potential update direction.
  - May be derived from local relations, memory, or other admissible constructions, provided:
    - it remains unit‑norm
    - it is computed from local primitives only

---

## **7. Coherence Score Primitive**

- **Coherence score:**
  - A scalar value derived from mismatch, dispersion, and memory terms.
  - Used by the operator to evaluate the quality of a candidate update.

- **Form:**
  The score must be constructed solely from:
  - mismatch values
  - their aggregates
  - γ‑weighted memory mismatch

---

## **8. Prohibited Primitives**

The LCO **must not** use:

- any explicit geometric object (polytopes, packings, lattices)
- any coordinate system tied to a specific geometry
- any φ‑specific or ε‑specific structure
- any global state or non‑local information

Only the primitives listed above are admissible.

---

### **Status**

These primitives define the **full ingredient set** available to the LCO.
All operator definitions and update policies must be expressible purely in terms of these objects and operations.

