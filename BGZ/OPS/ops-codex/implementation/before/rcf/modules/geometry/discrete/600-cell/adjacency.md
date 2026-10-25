# 600-cell/adjacency.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/600-cell/adjacency.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose
This document records the adjacency structure of the 600‑cell as used in RCF. It defines the relational neighborhood pattern that supports triangle‑compatible local motifs and stable LCO evaluation.

---

## overview
The 600‑cell has 120 vertices, each with a fixed adjacency pattern determined by its tetrahedral cell structure. This file documents:
- vertex degree
- local triangle neighborhoods
- adjacency symmetry
- constraints relevant to mismatch and closure

The focus is on relational structure only, not coordinate embeddings.

---

## role in RCF
- provides the local triangle motifs required by LCO
- supports stable recursion in standard‑density regimes
- forms the upstream adjacency pattern for the everyday continuous shadow

Detailed adjacency tables or generators may be added as needed.

