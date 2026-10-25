# Projection Invariants

## Document Contract

- owner: RCF projection module
- type: Contract
- scope: module-local content of `geometry/projection/projection-invariants.md`
- authority: tier-7 discrete-to-continuous handoff
- upstream dependencies: `/modules/geometry/projection/interface.md`
- status: active
- purpose: define what projection acts on, preserves, smooths, and exposes without selecting survivors or defining downstream metrics
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Structural Properties the Projection Operator Must Preserve*

This document specifies the invariants that the projection operator \( \Pi \) must satisfy.

These invariants ensure that projection remains a faithful shadowing mechanism and does not smuggle in structure that was not forced upstream.

---

## **1. Survivor-Only Domain Invariant**

**Invariant:**
\( \Pi \) is defined only on discrete relational geometries that survive the upstream sieve.

**Implication:**

- Projection never rehabilitates rejected candidates.
- Survival logic remains upstream.
- The continuous shadow is always the shadow of an actual survivor, not of an arbitrary candidate.

---

## **2. No New Combinatorics Invariant**

**Invariant:**
\( \Pi \) may smooth or aggregate discrete structure, but it may not introduce new discrete combinatorial content.

**Implication:**

- No new vertices, edges, tiers, or motif classes may be created by projection.
- Projection may blur, merge, or compress structure, but not invent it.

---

## **3. Adjacency Preservation Up to Smoothing**

**Invariant:**
Discrete relational adjacency must survive projection as continuous neighborhood structure, up to smoothing and aggregation.

**Implication:**

- Neighbor relations in the discrete substrate constrain neighborhood relations in the shadow.
- The continuous shadow may soften adjacency boundaries, but may not erase the upstream dependency of neighborhood structure on discrete motifs.

---

## **4. Residual Persistence Invariant**

**Invariant:**
Residual mismatch that survives upstream recursion cannot be arbitrarily erased by \( \Pi \).

**Implication:**

- Projection may smooth mismatch, but not annihilate its structural remainder.
- This remainder appears downstream as slush, anisotropy, gradient-like variation, or related residual structure.

---

## **5. Geometry-Noncreation Invariant**

**Invariant:**
\( \Pi \) does not introduce a primitive metric, manifold, coordinate chart, or physical geometry.

**Implication:**

- Continuous geometry is a downstream interpretation of the shadow, not an object injected by projection.
- Coordinate descriptions may be used representationally, but are not projection invariants.

---

## **6. Dependency Transparency Invariant**

**Invariant:**
Every stable structure produced by projection must remain traceable to the discrete survivor class and the projection process itself.

**Implication:**

- Downstream modules must be able, in principle, to identify what projected feature corresponds to what upstream structure.
- Projection outputs may be compressed, but not structurally opaque by definition.

---

## **7. Operator Non-Dynamical Invariant**

**Invariant:**
\( \Pi \) defines a representation change, not a time-evolution law.

**Implication:**

- Projection does not itself propagate, flow, or equilibrate the shadow.
- Any such behavior belongs to downstream continuous dynamics modules.

---

## **8. Modular Separation Invariant**

**Invariant:**
Projection must remain separate from LCO, GCO, discrete survival, continuous flow, and downstream physics interpretation.

**Implication:**

- \( \Pi \) consumes upstream outputs.
- \( \Pi \) exposes downstream objects.
- \( \Pi \) does not absorb the responsibilities of neighboring modules.

---

## **Status**

These invariants define the behavioral contract of the projection operator.
Any refinement of \( \Pi \) must preserve them to remain valid within the RCF architecture.

