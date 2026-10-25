# Coherence Diagnostics

## Document Contract

- owner: RCF core-adjacent terminology profile
- type: Interpretation
- scope: optional structural coherence diagnostics
- authority: tier-4 profile-local diagnostic definitions
- upstream dependencies: `../README.md` and `../../core/S∞.RCF.Invariants.v1.1.md`
- status: active
- purpose: define declared scores for comparing structural-check preservation across representations and refinements
- anti-backflow: coherence scores do not redefine core admissibility, turn partial invariant preservation into compatibility, or impose a metric on upstream structure

`/core-adjacent/terminology/coherence_metrics.md`

---

## **1. Purpose**

Define minimal structural metrics for assessing **coherence** across representations, transformations, or refinement steps.

These metrics are **not geometric**, **not probabilistic**, and **not semantic**.
They measure only structural consistency.

---

## **2. Coherence (Structural Definition)**

Coherence refers to:

- the degree to which a set of representations preserves shared invariants
- the compatibility of transformations with admissible structure
- the absence of contradiction across refinement levels

Coherence is a property of **structure**, not interpretation.

---

## **3. Local Coherence Metric**

A local coherence score \(C_{\text{local}}\) measures consistency within a declared finite, nonempty set of neighborhood checks:

\[
C_{\text{local}} := \frac{\text{declared local relation checks preserved}}{\text{declared local relation checks evaluated}}
\]

This is a combinatorial ratio, not a probability.

---

## **4. Global Coherence Metric**

A global coherence score \(C_{\text{global}}\) measures consistency across a declared finite, nonempty evaluation domain:

\[
C_{\text{global}} := \frac{\text{declared relation checks preserved}}{\text{declared relation checks evaluated}}
\]

This metric detects large‑scale structural drift.

---

## **5. Transformation Coherence**

For a transformation \(T\):

\[
C(T) := \frac{\text{declared structural checks passed by }T}{\text{declared structural checks evaluated for }T}
\]

This diagnoses check preservation. A transformation that violates any required core invariant remains inadmissible regardless of its score.

---

## **6. Refinement Coherence**

Across refinement steps:

\[
C_{\text{refine}} := \text{consistency of invariants across scales}
\]

This score reports whether the declared cross-scale checks detect a contradiction.

---

## **7. Scope**

These metrics:

- require no metric tensor
- require no geometry
- require no semantics
- depend only on invariants defined in the core

The evaluated check set and zero-denominator behavior must be declared by the adopting module. These are profile-local structural diagnostics, not core admissibility judgments.
