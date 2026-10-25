# **Geometry Validators**

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: substrate consistency validation
- authority: tier-3 geometry module-local validation contract
- upstream dependencies: `README.md`, `simplicial_complex.md`, `orientation.md`, `boundary_ops.md`, `coboundary_ops.md`, `cochains.md`, and `topology.md`
- status: active
- purpose: define failure checks for well-formed discrete geometry records
- anti-backflow: validator results reject malformed substrate records but do not add invariants or alter source operators

`/core/geometry/validators.md`

---

## **1. Purpose**

Define the structural validation rules required for a simplicial complex to serve as an admissible geometric substrate in RCF.

These validators ensure that all geometric operations (boundary, coboundary, cochains, topology) are well‑defined and internally consistent.

No metric, embedding, or domain semantics are assumed.

---

## **2. Simplicial Validity**

A simplicial complex \(K\) is valid if:

1. **Closure under faces**
   Every face of every simplex in \(K\) is also in \(K\).

2. **No degenerate simplices**
   No simplex contains repeated vertices.

3. **Finite dimensionality**
   \(K\) contains only finitely many simplices in each dimension.

These are purely combinatorial constraints.

---

## **3. Orientation Consistency**

Orientation must satisfy:

1. **Face‑induced orientation**
   The orientation of each simplex induces orientations on its faces via the boundary formula.

2. **Shared face compatibility**
   Adjacent simplices induce opposite orientations on shared interior faces.

3. **Cycle coherence**
   No orientation contradictions arise when traversing cycles.

Orientation rules are defined in `orientation.md`.

---

## **4. Boundary Operator Validity**

Boundary operators must satisfy:

1. **Alternating‑sign rule**
   Boundaries follow the standard alternating omission formula.

2. **Boundary‑of‑boundary is zero**
   \[
   \partial_{k-1} \circ \partial_k = 0
   \]

3. **Linear extension**
   Boundary maps extend linearly from simplices to chains.

These ensure a valid chain complex.

---

## **5. Coboundary Operator Validity**

Coboundary operators must satisfy:

1. **Duality with boundary**
   \[
   \delta_k = \partial_{k+1}^{\top}
   \]

2. **Coboundary‑of‑coboundary is zero**
   \[
   \delta_{k+1} \circ \delta_k = 0
   \]

3. **Linearity**
   Coboundary maps extend linearly from basis cochains.

These ensure a valid cochain complex.

---

## **6. Topological Consistency**

Topological invariants must be well‑defined:

1. **Euler characteristic computable**
   \[
   \chi = \sum_k (-1)^k |K_k|
   \]

2. **Homology groups well‑defined**
   \[
   H_k = \ker \partial_k / \mathrm{im}\, \partial_{k+1}
   \]

3. **Betti numbers finite**
   \[
   \beta_k = \dim H_k
   \]

These require only combinatorial structure.

---

## **7. Adjacency Validity**

Adjacency relations must satisfy:

1. **Face‑based adjacency**
   Two simplices are adjacent iff they share a face.

2. **No contradictory adjacency**
   No simplex may be adjacent to itself.
   No adjacency may violate face closure.

3. **Dimension‑consistent adjacency**
   Only simplices of the same dimension may be adjacent.

This ensures a coherent combinatorial neighborhood structure.

---

## **8. Structural Role in RCF**

Validators ensure:

- the geometry substrate is internally consistent
- all calculus operators are well‑defined
- topological invariants are stable
- downstream modules can rely on a trusted geometric core

They form the **admissibility layer** for the entire geometry directory.

---

## **9. Minimal Conceptual Examples**

- rejecting a simplex with repeated vertices
- detecting a boundary‑of‑boundary violation
- verifying orientation consistency across shared faces
- confirming Euler characteristic is computable

Examples are illustrative only.
