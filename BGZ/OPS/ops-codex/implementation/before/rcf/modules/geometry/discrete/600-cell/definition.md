# 600-cell/definition.md

## Document Contract

- owner: RCF discrete geometry module
- type: Contract
- scope: module-local content of `geometry/discrete/600-cell/definition.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose
This document defines the 600‑cell as a discrete relational topology within RCF. It records the combinatorial structure and adjacency properties that make the 600‑cell a viable simplicial substrate in standard constraint‑density regimes.

This file treats the 600‑cell as a relational graph, not a continuous manifold or background space.

---

## structural definition
The 600‑cell (hexacosichoron) is a 4D regular simplicial polytope. In this module it is defined purely by its combinatorial structure:

- **Vertices:** 120
- **Edges:** 720
- **Faces:** 1200 (all triangular)
- **Cells:** 600 (all tetrahedral)

Every local neighborhood is a triangle‑compatible motif, and the entire structure is uniformly simplicial.

---

## adjacency structure
The 600‑cell’s adjacency graph is:
- regular
- fully simplicial
- composed entirely of tetrahedral cells
- symmetric under the full icosahedral Coxeter group

Adjacency is the primitive object; coordinate embeddings are optional and documented separately.

---

## role in RCF (structural only)
- provides a dense, uniform relational substrate in standard constraint‑density regimes
- supports stable triangle‑based evaluation under LCO
- forms a discrete structure used by Π in generating the continuous shadow

This file records structure only.
Survival behavior is documented in `survival-stats.md`.

---

## invariants
- the 600‑cell is defined as an adjacency graph, not a metric space
- all cells are tetrahedral, ensuring full triangle‑compatibility
- symmetry and combinatorics are fixed; interpretations are downstream


