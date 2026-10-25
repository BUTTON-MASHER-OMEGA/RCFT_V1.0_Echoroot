# formal_structure.md
Relational Field Theory (Discrete) — Formal Structure

## Purpose
This file defines the mathematical structures permitted in
`relational_field_theory_discrete`.

It lifts **admissible dynamic propagation** into **discrete field structure**:

- adjacency‑indexed fields
- explicit field assignments
- explicit field updates
- no metric
- no continuous interpolation
- no amplitudes
- no PDEs

This is the minimal field layer allowed before continuous fields, metric geometry, and quantum amplitudes.

---

# 1. Base Structures from Relational Dynamics

We reuse:

- graph \(G = (V, E)\)
- neighborhood \(N(v)\)
- configurations \(s : V \to \Sigma\)
- admissibility predicate \(A_v\)
- propagation map \(P_v\)
- propagation record \(R = (s, s^+, \mathcal{A})\)

Support of dynamic change:

\[
\text{supp}(R) = \{ w \in V \mid s^+(w) \neq s(w) \}.
\]

All locality constraints remain in force.

---

# 2. Discrete Field Support

A **discrete field** is defined over the same adjacency graph \(G\).

## 2.1 Field Support

\[
\mathcal{F} = V
\]

or, if constraints require:

\[
\mathcal{F} \subseteq V.
\]

Fields live on adjacency‑indexed supports, not coordinates or metric spaces.

---

# 3. Field Values and Assignments

## 3.1 Field Value Space

Let:

\[
\Phi
\]

be the set of allowed field values.
No interpretation is assigned to elements of \(\Phi\).

## 3.2 Field Assignment

A discrete field configuration is a function:

\[
\varphi : \mathcal{F} \to \Phi.
\]

This is the formal content of `assign_field`.

---

# 4. Field Update Under Propagation

Propagation from relational dynamics induces field updates.

## 4.1 Local Field Update Rule

Define a local field update operator:

\[
U^{\text{field}}_v :
\Phi \times \Phi^{N(v)} \times \Sigma \times \Sigma^{N(v)}
\to \Phi.
\]

It may depend on:

- the current field value at \(v\),
- the field values in its neighborhood,
- the current dynamic state \(s(v)\),
- the dynamic state in the neighborhood \((s(u))_{u \in N(v)}\).

It must remain adjacency‑local.

## 4.2 Global Field Update

Given a propagation record \(R = (s, s^+, \mathcal{A})\), define:

\[
\varphi^+(v) =
U^{\text{field}}_v\big(
\varphi(v), (\varphi(u))_{u \in N(v)},
s(v), (s(u))_{u \in N(v)}
\big).
\]

This is the formal content of `update_field`.

---

# 5. Field Update Record

Define the **field‑change support**:

\[
\text{supp}(F) = \{ v \in \mathcal{F} \mid \varphi^+(v) \neq \varphi(v) \}.
\]

A **field update record** is:

\[
F = (\varphi, \varphi^+, \text{supp}(F)).
\]

No equivalence between \(\text{supp}(F)\) and \(\text{supp}(R)\) is assumed unless stated elsewhere.

The record is adjacency‑local and contains no metric or continuous structure.

---

# 6. Discrete Field Handoff to Continuous Layer

Define the **discrete‑to‑continuous field handoff record**:

\[
H_{\text{disc}} = (\mathcal{F}, \varphi, F).
\]

This record:

- identifies the field support,
- provides the current field values,
- provides the update structure,
- contains no interpolation,
- contains no metric,
- contains no amplitudes.

This is the formal content of `field_support` and the handoff to `continuous_field_theory`.
