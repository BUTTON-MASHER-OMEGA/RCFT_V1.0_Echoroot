# **Cochains**

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: cochain and dual-chain grammar
- authority: tier-3 geometry substrate contract
- upstream dependencies: `README.md`, `boundary_ops.md`, and `coboundary_ops.md`
- status: active
- purpose: define cochains and their relation to chains and discrete operators
- anti-backflow: cochains do not introduce metric, continuous fields, projection, or domain semantics

`/core/geometry/cochains.md`

---

## **1. Purpose**

Define **cochains** as the dual objects to chains in RCF’s discrete geometric substrate.

This module specifies the formal structure of cochains, their relation to simplices, and their role in discrete calculus.
No metric, embedding, or domain semantics are assumed.

---

## **2. Objects**

Given a simplicial complex \(K\):

- \(C_k(K)\): vector space of **\(k\)-chains**
  (formal linear combinations of oriented \(k\)-simplices)

- \(C^k(K)\): vector space of **\(k\)-cochains**
  (linear functionals on \(C_k(K)\))

Formally:

\[
C^k(K) = \mathrm{Hom}(C_k(K), \mathbb{R})
\]

Cochains assign real values to oriented simplices.

No interpretation is attached to these values.

---

## **3. Basis Cochains**

For each oriented \(k\)-simplex \(\sigma\), define the **indicator cochain**:

\[
\phi_\sigma(\sigma') =
\begin{cases}
1 & \text{if } \sigma' = \sigma \\
0 & \text{otherwise}
\end{cases}
\]

These basis cochains span \(C^k(K)\).

---

## **4. Linearity**

For cochains \(\phi, \psi \in C^k(K)\) and scalars \(\alpha, \beta\):

\[
(\alpha \phi + \beta \psi)(c) =
\alpha \phi(c) + \beta \psi(c)
\]

for any \(k\)-chain \(c\).

Cochains extend linearly from simplices to chains.

---

## **5. Coboundary Interaction**

Cochains interact with the coboundary operator via:

\[
(\delta_k \phi)(\sigma) =
\phi(\partial_{k+1} \sigma)
\]

This defines how cochains extend to higher‑dimensional simplices.

Coboundary is defined in `coboundary_ops.md`.

---

## **6. Orientation Dependence**

Cochain evaluation respects orientation:

\[
\phi([- \sigma]) = -\phi([\sigma])
\]

This ensures sign consistency across the calculus layer.

---

## **7. Cochain Degree**

Cochains are graded by dimension:

- \(0\)-cochains evaluate vertices
- \(1\)-cochains evaluate edges
- \(2\)-cochains evaluate faces
- \(3\)-cochains evaluate tetrahedra

No geometric interpretation is implied.

---

## **8. Incidence Interpretation (Combinatorial Only)**

Cochains encode:

- local evaluations on simplices
- adjacency relations via coboundary
- combinatorial structure of discrete forms

This is purely structural.

---

## **9. Structural Role in RCF**

Cochains provide:

- the discrete substrate for **gradients, curls, and divergences** (in later files)
- a dual representation of chain‑level structure
- a foundation for discrete differential forms
- compatibility with boundary and coboundary operators

They are **core geometric invariants** of the simplicial substrate.

---

## **10. Minimal Conceptual Examples**

- a \(0\)-cochain assigning values to vertices
- a \(1\)-cochain evaluating oriented edges
- a \(2\)-cochain acting on oriented faces
- coboundary of a vertex‑cochain producing an edge‑cochain

Examples are illustrative only.
