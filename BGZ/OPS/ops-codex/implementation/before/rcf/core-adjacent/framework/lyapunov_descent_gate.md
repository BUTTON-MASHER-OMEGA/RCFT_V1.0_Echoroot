# Lyapunov Descent Gate Profile

## Document Contract

- owner: RCF core-adjacent framework profile
- type: Contract
- scope: optional monotonic refinement gate
- authority: tier-4 profile-local gate contract
- upstream dependencies: `event_filtration_D1.md`, `../terminology/coherence_metrics.md`, and `../../core/S∞.RCF.Invariants.v1.1.md`
- status: active
- purpose: gate declared refinement steps using a declared scalar diagnostic
- anti-backflow: the gate does not define energy, dynamics, time, a canonical scalar, or universal RCF refinement admissibility

`/core-adjacent/framework/lyapunov_descent_gate.md`

---

## **1. Purpose**

Define the **minimal Lyapunov descent gate** used in RCF’s conceptual layer.

This gate is **not dynamical**, **not energetic**, and **not temporal**.
It is a **monotonicity constraint** ensuring that refinement steps do not increase structural instability.

---

## **2. Lyapunov Quantity (Structural Definition)**

A **Lyapunov quantity** \(L\) is any scalar functional that:

- measures structural inconsistency or instability
- is defined purely from admissible invariants
- decreases (or remains constant) under valid refinement

Examples include:
coherence deficits, violation counts, or other structural diagnostics.
No physical interpretation is assumed.

---

## **3. Descent Condition**

A refinement step
\[
X \rightarrow X'
\]
satisfies the **descent condition** if:

\[
L(X') \le L(X)
\]

This ensures refinement does not introduce additional inconsistency.

If strict descent is required:

\[
L(X') < L(X)
\]

but strictness is optional and context‑dependent.

---

## **4. Gate Definition**

A refinement step passes the **Lyapunov descent gate** if:

- the Lyapunov quantity does not increase
- all core invariants remain satisfied
- the step is admissible under the event axiom

Formally:

\[
\text{Gate}(X \rightarrow X') =
\begin{cases}
\text{admissible} & \text{if } L(X') \le L(X) \\
\text{rejected} & \text{otherwise}
\end{cases}
\]

The gate is a **filter**, not a force.

---

## **5. Structural Consequences**

Within an adopting profile, the descent gate provides:

- refinement stability
- monotonic reduction of inconsistency
- rejection of steps that increase the declared diagnostic
- compatibility with attractor identification
- alignment with event filtration

Non-increase alone permits equal-value cycles. Cycle exclusion requires strict descent or an additional well-founded termination condition.

---

## **6. Relationship to Attractors**

If a configuration \(A\) satisfies:

\[
L(\text{Refine}(A)) = L(A)
\]

and remains structurally equivalent under refinement,
then \(A\) is a **Lyapunov‑stable attractor** in the minimal sense.

This is a **classification**, not a dynamical claim.

---

## **7. Scope**

This module:

- introduces no energy
- defines no physical Lyapunov function
- specifies no time evolution
- asserts no causal mechanism

It provides only a **monotonicity constraint** for refinement admissibility.
