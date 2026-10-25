# Event Filtration D1 Profile

## Document Contract

- owner: RCF core-adjacent framework profile
- type: Contract
- scope: optional non-temporal D1 transition filtration
- authority: tier-4 profile-local filtration contract
- upstream dependencies: `../README.md`, `../../core/calculus/event_axiom.md`, and `../../core/S∞.RCF.Invariants.v1.1.md`
- status: active
- purpose: classify candidate minimal local transitions before downstream stability gates
- anti-backflow: D1 filtration does not define physical events, temporal order, universal locality bounds, or core admissibility beyond its cited upstream constraints

`/core-adjacent/framework/event_filtration_D1.md`

---

## **1. Purpose**

Define the **D1‑level event filtration rule** used in RCF’s conceptual layer.

This filtration determines which transitions qualify as **D1 candidate events** before any stability or descent checks are applied.

It is a **granularity constraint**, not a causal or physical mechanism.

---

## **2. Filtration Context**

The filtration sits between:

- **raw transitions** (arbitrary structural changes)
- **events** (indivisible admissible transitions, per the Event Axiom)

Its job is to **reject transitions that are too coarse, too fine, or structurally invalid** before they enter the event pipeline.

---

## **3. D1 Filtration Rule**

A transition
\[
X \rightarrow X'
\]
passes the **D1 event filter** if and only if:

1. **Minimality**
   The transition cannot be decomposed into smaller admissible transitions.

2. **Locality**
   The transition affects only a bounded neighborhood in the structure.

3. **Invariant Preservation**
   All core invariants (∂² = 0, δ² = 0, admissibility constraints) remain satisfied.

4. **Refinement Compatibility**
   The transition does not violate refinement‑level consistency.

Formally:

\[
\text{D1}(X \rightarrow X') =
\begin{cases}
\text{admissible} & \text{if conditions 1–4 hold} \\
\text{rejected} & \text{otherwise}
\end{cases}
\]

---

## **4. Minimality Criterion**

A transition fails minimality if:

- it can be expressed as a sequence of smaller admissible transitions
- it contains redundant sub-steps
- it introduces unnecessary intermediate states

Minimality makes the transition a **candidate event** relative to the declared decomposition grammar.

---

## **5. Locality Criterion**

A transition is local if:

- its support lies within a bounded simplicial neighborhood
- it does not require global restructuring
- it does not propagate changes beyond its local domain

Locality prevents global jumps from being misclassified as events.

---

## **6. Invariant Preservation**

A transition must preserve:

- chain/cochain structure
- boundary and coboundary identities
- orientation consistency
- topological admissibility

If any invariant is violated, the transition is rejected before event classification.

---

## **7. Refinement Compatibility**

A transition is refinement‑compatible if:

- it does not contradict the refinement hierarchy
- it respects the Event Axiom’s indivisibility requirement
- it preserves the declared refinement relation

This establishes compatibility with the adopting profile's declared refinement pipeline.

---

## **8. Structural Consequences**

The D1 filter:

- prevents invalid transitions from entering the event pipeline
- classifies candidate events as atomic relative to the declared decomposition grammar
- enforces structural consistency before refinement
- supports attractor and Lyapunov‑based stability mechanisms
- maintains coherence across scales

It is the **first gate** in the event‑refinement chain.

---

## **9. Scope**

This module:

- introduces no dynamics
- defines no physical events
- specifies no temporal ordering
- asserts no causal mechanism

It provides only a **minimal filtration rule** for admissible transitions.
