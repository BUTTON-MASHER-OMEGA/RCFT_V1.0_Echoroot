# formal_structure.md
Relational Dynamics — Formal Structure

## Purpose
This file defines the mathematical structures permitted in `relational_dynamics`.
It builds on `relational_kinematics` by adding **admissibility** and **propagation**, still under strict adjacency and non‑interpretive constraints.

No metric, no physics‑time, no PDEs, no hidden scheduler.

---

# 1. Base Structures from Relational Kinematics

We reuse:

- graph \(G = (V, E)\)
- neighborhood \(N(v)\)
- state space \(\Sigma\)
- configurations \(s : V \to \Sigma\)
- local update operator \(U\)
- kinematic transition \(T = (s, s')\)
- support of change


\[
  \text{supp}(T) = \{ v \in V \mid s'(v) \neq s(v) \}
  \]



All locality constraints from `relational_kinematics` remain in force.

---

# 2. Local and Global Admissibility

## 2.1 Local Admissibility Predicate

For each node \(v \in V\), define:



\[
A_v : \Sigma \times \Sigma^{N(v)} \times \Sigma \to \{0, 1\}
\]



with



\[
A_v\big(s(v), (s(u))_{u \in N(v)}, s'(v)\big) = 1
\]



meaning:
“the update at \(v\) from \(s(v)\) to \(s'(v)\) is admissible given its neighborhood.”

## 2.2 Global Admissibility

A transition \(T = (s, s')\) is **globally admissible** if:



\[
\forall v \in V,\quad
A_v\big(s(v), (s(u))_{u \in N(v)}, s'(v)\big) = 1
\]



Define:



\[
A(T) =
\begin{cases}
1 & \text{if the above holds for all } v \\
0 & \text{otherwise}
\end{cases}
\]



Only transitions with \(A(T) = 1\) are considered valid dynamic updates.

This is the formal content of `admissible_update`.

---

# 3. Local Propagation

## 3.1 Local Propagation Map

For each node \(v\), define a **local propagation map**:



\[
P_v : \Sigma \times \Sigma^{N(v)} \to \Sigma^{\{v\} \cup N(v)}
\]



Given the local view at \(v\), \(P_v\) returns updated states for \(v\) and possibly some neighbors, but only within \(\{v\} \cup N(v)\).

## 3.2 Active Set

Let \(\mathcal{A} \subseteq V\) be a finite set of **active nodes** for a propagation step.

No global, implicit, or infinite activation is allowed.

---

# 4. Global Propagation Step

Given a configuration \(s\) and an active set \(\mathcal{A}\), define a new configuration \(s^+\) by:



\[
s^+(w) =
\begin{cases}
\big(P_v(s(v), (s(u))_{u \in N(v)})\big)(w)
& \text{if } \exists v \in \mathcal{A} \text{ with } w \in \{v\} \cup N(v) \\
s(w)
& \text{otherwise}
\end{cases}
\]



We require that for every updated node \(w\), the resulting local change is admissible:



\[
\forall w \in V \text{ with } s^+(w) \neq s(w),\quad
A_w\big(s(w), (s(u))_{u \in N(w)}, s^+(w)\big) = 1
\]



This is the formal content of `propagate_local`.

##### Overlap Rule
The active set must be chosen so that no two active nodes update the same target node during a propagation step. Any propagation step violating this condition is invalid.


---

# 5. Propagation Record

A **propagation record** for a step from \(s\) to \(s^+\) is:



\[
R = (s, s^+, \mathcal{A})
\]



with:



\[
\text{supp}(R) = \{ w \in V \mid s^+(w) \neq s(w) \}
\]



and the constraints:

- \(\mathcal{A}\) is finite
- updates are adjacency‑local via \(P_v\)
- all changes are admissible via \(A_v\)

This record is what `relational_field_theory_discrete` will see as “dynamic structure.”

---

# 6. Discrete Field Handoff

To prepare for the discrete field layer, define the **field handoff record**:



\[
H = \big( \text{supp}(R), (s(w))_{w \in \text{supp}(R)}, (s^+(w))_{w \in \text{supp}(R)} \big)
\]



Optionally paired with \(\mathcal{A}\) if needed:



\[
H_{\text{full}} = (H, \mathcal{A})
\]



This is the formal content of `field_handoff`:

- it identifies where changes occurred,
- it encodes the local dynamic effect,
- it remains adjacency‑local,
- it introduces no metric, no time, no PDEs, no hidden scheduler.

---

# 7. Invariant Compliance

All structures in this file satisfy:

- adjacency‑only propagation
- explicit admissibility
- no metric usage
- no physics‑time usage
- no PDEs
- no hidden scheduler
- no interpretive semantics

This file is RCP‑1.1 compliant.
