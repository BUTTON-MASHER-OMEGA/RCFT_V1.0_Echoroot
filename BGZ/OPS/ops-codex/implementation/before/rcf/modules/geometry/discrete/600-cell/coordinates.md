# 600-cell/coordinates.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/600-cell/coordinates.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose
This document records coordinate embeddings of the 600‑cell for reference and optional downstream use. Coordinates are not required for RCF execution, but they provide a convenient way to visualize or verify adjacency relations.

---

## coordinate families
A standard coordinate embedding of the 600‑cell uses three vertex families:

### 1. 16‑cell vertices


\[
(\pm 1, 0, 0, 0) \text{ and permutations}
\]



### 2. 8‑cell vertices


\[
(\pm \tfrac{1}{2}, \pm \tfrac{1}{2}, \pm \tfrac{1}{2}, \pm \tfrac{1}{2})
\]



### 3. 96 golden‑ratio vertices
Even permutations of:


\[
\left(\pm \tfrac{\phi}{2}, \pm \tfrac{1}{2}, \pm \tfrac{1}{2\phi}, 0\right)
\]


where \(\phi = \frac{1 + \sqrt{5}}{2}\).

These coordinates generate all 120 vertices and reproduce the correct adjacency graph.

---

## invariants
- coordinates are optional; adjacency is primary
- φ‑structured coordinates appear in standard embeddings from the 600‑cell’s symmetry group
- coordinate embeddings do not imply a continuous background space

