# formal_structure.md
Relational Quantum Mechanics — Formal Structure

## Purpose
This file defines the mathematical structures permitted in
`relational_quantum_mechanics`.

It introduces **relational amplitudes** for the first time in the RCF stack:
- adjacency‑indexed
- invariant‑aware
- time‑indexed
- explicit
- non‑interpretive

No Hilbert spaces, no wavefunctions, no probability interpretation,
no collapse, no operator algebra, no PDEs, no hidden state.

---

# 1. Inputs from Upstream Modules

We reuse:

- adjacency graph \(G = (V, E)\)
- neighborhood \(N(v)\)
- Physics Time index set \(\mathcal{T}_P\)
- local invariant records \(I_v\) for \(v \in V\)

No amplitude structure exists upstream of this module.

---

# 2. Amplitude Support

## 2.1 Adjacency‑Indexed Support
Amplitudes are defined over adjacency indices:



\[
\mathcal{A} = V
\]



or a subset thereof:



\[
\mathcal{A} \subseteq V
\]



This ensures amplitudes remain **local** and **graph‑indexed**, not coordinate‑indexed.

---

# 3. Amplitude Value Space

Let:



\[
\mathcal{Q}
\]



be the set of allowed amplitude values.

RCF does **not** require:
- complex numbers
- vectors
- spinors
- Hilbert space elements

\(\mathcal{Q}\) is simply a structured set of amplitude‑like values.

---

# 4. Relational Amplitude Assignment

## 4.1 Amplitude Field
A relational amplitude assignment is a function:



\[
\alpha : \mathcal{A} \to \mathcal{Q}
\]



This is the formal content of `assign_amplitude`.

No normalization, no inner product, no probability interpretation.

---

# 5. Time‑Indexed Amplitudes

If amplitudes vary with Physics Time:



\[
\alpha_\tau : \mathcal{A} \to \mathcal{Q}
\quad\text{for each } \tau \in \mathcal{T}_P
\]



This is a **family** of amplitude assignments, not a dynamical law.

No rule is given for how \(\alpha_\tau\) changes with \(\tau\).

---

# 6. Amplitude Update

Define a **local amplitude update operator**:



\[
U^{\text{amp}}_v :
\mathcal{Q} \times \mathcal{Q}^{N(v)} \times \mathcal{I} \times \mathcal{I}^{N(v)}
\to \mathcal{Q}
\]



where \(\mathcal{I}\) is the space of local invariant records.

Given an amplitude assignment \(\alpha\), define:



\[
\alpha'(v) =
U^{\text{amp}}_v\big(
\alpha(v), (\alpha(u))_{u \in N(v)},
I_v, (I_u)_{u \in N(v)}
\big)
\]



This is the formal content of `update_amplitude`.

No PDEs, no Schrödinger equation, no operator algebra.

---

# 7. Amplitude Update Record

A **relational amplitude update record** is:



\[
R_\alpha = (\alpha, \alpha', S)
\]



where:



\[
S = \{ v \in \mathcal{A} \mid \alpha'(v) \neq \alpha(v) \}
\]



This record:
- identifies where amplitude changes occurred
- is adjacency‑local
- contains no probability semantics

---

# 8. Quantum‑Compatibility Boundary

Define a compatibility predicate:



\[
C_\alpha(\alpha, I) \in \{0, 1\}
\]



where \(I\) denotes the collection of invariant records in play.



\[
C_\alpha(\alpha, I) = 1
\]



means:
- amplitude assignments respect invariant constraints
- no reinterpretation of geometry is introduced
- no hidden state is introduced

The **quantum‑compatibility boundary** is:



\[
B_\alpha = (\alpha_{\tau_{\min}},\ \alpha_{\tau_{\max}})
\]



This is the formal content of `quantum_boundary`.

---

# 9. Handoff to Conversion Laws

The conversion layer receives:



\[
H_\alpha = (\mathcal{A}, \alpha, R_\alpha, B_\alpha)
\]



This record:
- provides amplitude values
- provides update structure
- provides compatibility boundaries
- contains no Hilbert space, no probability, no collapse

---

# 10. Invariant Compliance

All structures in this file satisfy:

- adjacency‑indexed amplitudes
- explicit amplitude assignments
- explicit amplitude updates
- invariant‑aware but non‑interpretive
- no Hilbert spaces
- no wavefunctions
- no probability interpretation
- no collapse
- no operator algebra
- no PDEs
- no hidden state

This file is RCP‑1.1 compliant.
