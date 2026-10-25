# Triadic Compatibility Profile

## Document Contract

- owner: RCF core-adjacent framework profile
- type: Interpretation
- scope: optional three-configuration compatibility score
- authority: tier-4 profile-local diagnostic definition
- upstream dependencies: `../terminology/coherence_metrics.md` and `../../core/S∞.RCF.Invariants.v1.1.md`
- status: active
- purpose: define a weakest-link score over three declared pairwise compatibility evaluations
- anti-backflow: the triadic score does not introduce a core metric, weaken invariant preservation, or impose triadic organization on downstream models

`/core-adjacent/framework/triadic_metric.md`

---

## **1. Purpose**

Define an optional **triadic compatibility score** for RCF's core-adjacent layer.

This metric is **not geometric**, **not statistical**, and **not semantic**.
It is a **structural relation** among three configurations that measures their mutual compatibility.

---

## **2. Triadic Relation (Structural Definition)**

Given three configurations \(A, B, C\), the **triadic metric** evaluates:

- how well each pair aligns structurally
- how well the triplet forms a coherent relational unit
- whether the three-way interaction preserves admissible invariants

The metric is a **relation**, not a distance.

---

## **3. Pairwise Compatibility**

For a declared finite, nonempty check set, define a pairwise compatibility score:

\[
\kappa(A, B) := \frac{\text{declared pairwise checks passed}}{\text{declared pairwise checks evaluated}}
\]

This is a structural ratio, not a probability or similarity measure.

---

## **4. Triadic Consistency**

The triadic metric \(T(A, B, C)\) is defined as:

\[
T(A, B, C) := \min \big( \kappa(A,B),\ \kappa(B,C),\ \kappa(C,A) \big)
\]

This captures the **weakest link** in the triad.

A triad passes this profile when all pairwise relations pass the same declared required checks. Any failed core invariant still makes the relevant transformation inadmissible regardless of the score.

---

## **5. Alignment Criterion**

A triad \((A, B, C)\) is **aligned** if:

\[
T(A, B, C) = 1
\]

meaning all pairwise relations preserve all admissible invariants.

Alignment is a **structural property**, not a geometric or semantic one.

---

## **6. Structural Consequences**

The triadic metric supports:

- **consistency checking** across three interacting configurations
- **refinement validation** when multiple structures must cohere
- **stability assessment** for attractor‑like relational clusters
- **compatibility filtering** in multi‑agent or multi‑module settings

It acts as a **three‑way coherence gate**.

---

## **7. Relationship to Lyapunov Descent**

If a refinement step reduces the weakest pairwise compatibility in a triad:

\[
T(A', B', C') < T(A, B, C)
\]

then the step fails the **triadic stability condition** and may be rejected by the Lyapunov descent gate.

The triadic metric thus provides a **multi‑configuration stability check**.

---

## **8. Scope**

This module:

- introduces no geometry
- defines no distances or angles
- specifies no semantics
- asserts no causal or dynamical mechanism

It provides only a **minimal structural relation** among three configurations.
