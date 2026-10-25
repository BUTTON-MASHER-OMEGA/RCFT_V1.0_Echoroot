# **Jacobian Entropy**

## Document Contract

- owner: RCF discrete calculus substrate
- type: Interpretation
- scope: optional transformation diagnostic
- authority: optional tier-3 calculus profile
- upstream dependencies: `README.md` and `../algebra/tensor_ops.md`
- status: provisional
- purpose: define a determinant-based diagnostic for declared finite-dimensional local transformations
- anti-backflow: this diagnostic does not create physical entropy, metric volume, guaranteed refinement stability, or a new RCF invariant

`/core/calculus/jacobian_entropy.md`

---

## **1. Purpose**

Define a candidate refinement-sensitive scalar diagnostic derived from admissible finite-dimensional local transformations.

This module introduces a **purely structural** measure of volumetric change in configuration space.
No geometric, physical, or thermodynamic interpretation is assumed.

---

## **2. Transformation Field**

Let \(J\) be a locally admissible linear transformation acting on a configuration neighborhood.

Requirements:

- \(J\) must preserve algebraic admissibility
- \(J\) must respect chain/cochain structure
- \(J\) must be locally invertible on its active domain

No coordinate system, embedding, or metric is assumed.

---

## **3. Entropy Scalar**

Define the **Jacobian entropy**:

\[
H := \log |\det J|
\]

This scalar measures **local volumetric distortion** induced by \(J\) at the refinement scale.

Interpretation is strictly structural:

- \(H > 0\): local expansion
- \(H < 0\): local contraction
- \(H = 0\): volume‑preserving

These are algebraic statements about determinants, not geometric or physical claims.

---

## **4. Second‑Order Invariant**

Define the second‑order refinement diagnostic:

\[
\Delta H := \Delta \log |\det J|
\]

Here, \(\Delta\) denotes a **formal second‑order operator** acting on scalars.
Its specific realization is defined in downstream modules (e.g., metric‑dependent calculus).

This file introduces only the **symbolic form**, not the operator implementation.

---

## **5. Invariance Constraints**

Jacobian entropy must satisfy:

### **1. Basis-Conjugation Invariance**
\[
H(J) = H(PJP^{-1})
\]
for any admissible change of basis \(P\).

### **2. Composition Additivity**
\[
H(J_2 \circ J_1) = H(J_2) + H(J_1)
\]

### **3. Declared Refinement Compatibility**
Any claimed stability must be verified against an explicit compatible refinement map for the underlying simplicial or algebraic structure.

These constraints ensure that \(H\) is a **structural scalar invariant**, not a geometric or physical quantity.

---

## **6. Scope**

This module:

- assumes only local linear structure
- introduces no metric or embedding
- defines no physical entropy
- provides a diagnostic scalar for refinement analysis
- defers all metric‑dependent operators to downstream modules

It is an optional diagnostic profile for transformation-based refinement.
