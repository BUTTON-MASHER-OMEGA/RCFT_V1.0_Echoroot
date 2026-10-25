# LCO Mode Space

## Document Contract

- owner: RCF local coherence operator
- type: Interpretation
- scope: module-local content of `lco/lco-mode-space.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Definition of the Unit‑Norm Representational Space*

This document defines the **mode space** used by the Local Coherence Operator (LCO). The mode space is the only representational structure assumed by the LCO: a unit‑norm vector space used purely for directional comparison.

The mode space is a **representational device**, not a geometric ontology. It provides the minimal structure required for the LCO to compare directions using dot‑product similarity, without introducing any assumptions about physical geometry, topology, dimensionality, or downstream structure.

---

## **1. Purpose of the Mode Space**

The LCO requires a way to:

- represent local relations as directions
- represent candidate updates as directions
- compare these directions using the dot‑product primitive

The unit‑norm mode space provides exactly this and nothing more.

It ensures:

- consistent directional comparison
- bounded mismatch values
- compatibility with local aggregation
- neutrality with respect to any geometric interpretation

---

## **2. Definition of the Mode Space**

The mode space consists of vectors \( v \) satisfying:



\[
\|v\| = 1
\]



This constraint applies to:

- relation vectors \( r_i \)
- the memory anchor \( H \)
- candidate modes \( m \)

All vectors used by the LCO must lie in this space.

### **Properties**

- **Unit‑norm:**
  Ensures all directional comparisons are scale‑invariant.

- **Representation‑only:**
  The space is not interpreted as physical or structural geometry.

- **Minimal:**
  No explicit coordinate system, external geometry, or topology is assumed beyond the normalized inner-product structure used for directional comparison.

- **Neutral:**
  The space does not encode dimensional preferences or geometric constraints.

---

## **3. Rationale for Unit‑Norm Representation**

The LCO uses only the dot‑product similarity primitive:



\[
r \cdot m
\]



Unit‑norm vectors ensure that:

- mismatch values remain bounded
- dispersion remains well‑defined
- comparisons are consistent across relations
- the operator remains insensitive to input scale after normalization into unit norm

This is the minimal structure required for the LCO to function.

---

## **4. Admissible Operations in Mode Space**

The LCO may:

- normalize vectors to unit norm
- compute dot products
- form candidate modes from local primitives, provided they are normalized

The LCO may **not**:

- introduce coordinate systems
- assume any embedding or geometry
- apply transformations that introduce additional representational structure not reducible to local normalization and inner-product comparison
- infer dimensionality or topology

All operations must remain algebraic and local.

---

## **5. Invariant Compatibility**

The mode space satisfies all LCO invariants:

- **Unit‑norm mode invariant:**
  The mode space enforces this directly.

- **Locality:**
  The mode space does not introduce non‑local structure.

- **Geometry‑neutrality:**
  No geometric or topological assumptions are encoded.

- **Determinism:**
  Normalization and dot‑product operations are deterministic.

- **Well‑definedness:**
  All mismatch and dispersion terms remain finite and bounded.

- **γ as external:**
  The mode space does not interact with γ beyond its use in memory mismatch.

---

## **6. Role in the LCO**

The mode space underlies all LCO computations:

- mismatch
- memory mismatch
- aggregate mismatch
- dispersion
- coherence scoring

It provides the representational substrate for directional comparison while maintaining strict neutrality and minimalism.

---

## **Status**

The unit‑norm mode space is the only representational assumption made by the LCO.
It is minimal, geometry‑neutral, and fully compatible with the primitives and invariants of the execution layer.

