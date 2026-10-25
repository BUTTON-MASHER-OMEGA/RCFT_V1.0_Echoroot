# LCO Mismatch

## Document Contract

- owner: RCF local coherence operator
- type: Interpretation
- scope: module-local content of `lco/lco-mismatch.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Definition and Role of the Mismatch Functional*

This document defines the **mismatch functional**, one of the core components used by the Local Coherence Operator (LCO) to evaluate candidate local updates. Mismatch quantifies how well a candidate mode aligns with each local relation and with the memory anchor, using only the directional comparison primitives permitted at the execution layer.

Mismatch is a purely algebraic construct. It does not encode or assume any geometric structure beyond the unit‑norm mode space used for directional comparison.

---

## **1. Purpose of Mismatch**

Mismatch provides a **local, direction‑based measure** of how well a candidate mode \( m \) preserves the relational structure encoded by the local relations \( r_i \) and the memory anchor \( H \).

It is the fundamental ingredient from which:

- aggregate mismatch
- dispersion
- and the final coherence score

are constructed.

Mismatch is designed to be:

- local
- geometry‑neutral
- compatible with the LCO invariants
- expressible solely in terms of the allowed primitives

---

## **2. Relation Mismatch**

For each local relation vector \( r_i \), the mismatch with respect to a candidate mode \( m \) is defined as:



\[
\ell_i = 1 - r_i \cdot m
\]



### **Properties**

- Uses only the dot‑product similarity primitive.
- Produces a scalar in \([0,2]\) for unit‑norm vectors.
- Is minimized when \( m \) aligns with \( r_i \).
- Is maximized when \( m \) is opposite to \( r_i \).
- Is neutral with respect to any embedding or geometric interpretation.

### **Rationale**

The current unit-norm LCO representation admits only the dot product as its directional comparison primitive.
The form \(1 - r_i \cdot m\):

- yields zero mismatch when directions coincide
- increases smoothly as alignment worsens
- remains bounded and well‑defined for all admissible inputs

Within this representation profile, the mismatch functional remains a direct affine transformation of the dot product and does not introduce additional nonlinear or non-local transformations.

---

## **3. Memory Mismatch**

The mismatch between the candidate mode and the memory anchor is defined as:



\[
\ell_{\text{mem}} = \gamma (1 - H \cdot m)
\]



### **Properties**

- Weighted by the externally supplied scalar \( \gamma \in [0,1] \).
- Uses the same mismatch structure as relation mismatch.
- Incorporates historical influence without altering the operator’s neutrality.
- Treats memory as an additional local relation, scaled by γ.

### **Rationale**

Memory mismatch ensures that:

- the influence of prior state is incorporated locally
- the operator remains compatible with temporal coherence
- γ is applied but never chosen or inferred by the LCO

---

## **4. Combined Mismatch Set**

The full set of mismatch terms used by the LCO is:



\[
\{\ell_i\}_{i=1}^N \cup \{\ell_{\text{mem}}\}
\]



These values form the basis for:

- aggregate mismatch
- dispersion
- and the final coherence score

All subsequent quantities in the LCO core are constructed from this set using only local aggregation primitives.

---

## **5. Invariant Compatibility**

Mismatch satisfies all LCO invariants:

- **Unit‑norm mode invariant:**
  Defined only for unit‑norm vectors.

- **Locality invariant:**
  Depends only on \( r_i \), \( H \), \( m \), and γ.

- **Geometry‑neutrality invariant:**
  Uses only dot products; no external geometric or topological structure is assumed.

- **Determinism invariant:**
  Produces consistent values for identical inputs.

- **Well‑defined score invariant:**
  Always finite and bounded for admissible inputs.

- **γ as external parameter:**
  γ appears only as a scalar multiplier.

---

## **Status**

Mismatch is a foundational component of the LCO.
All higher‑level quantities—aggregate mismatch, dispersion, and coherence score—are constructed from this definition using only the primitives and invariants specified in the LCO module.
