# adjacency smoothing in the 120‑cell shadow

## Document Contract

- owner: RCF continuous geometry module
- type: Interpretation
- scope: module-local content of `geometry/continuous/120-cell/adjacency_smoothing.md`
- authority: tier-7 continuous-shadow representation layer
- upstream dependencies: `/modules/geometry/continuous/interface.md`
- status: active
- purpose: represent the continuous relational shadow without owning metrics, curvature, flow, or physical interpretation
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document describes how discrete adjacency in the 120‑cell is smoothed
into continuous neighborhood structure by the projection operator Π.

The smoothing process is structural and does not assume a metric or
coordinate system.

---

## overview
Adjacency smoothing converts:

- discrete edges → continuous proximity
- discrete triangles → continuous local patches
- discrete tetrahedral neighborhoods → continuity‑like envelopes

The smoothing preserves all relational invariants.

---

## smoothing rules

### 1. edge smoothing
Each edge in the 120‑cell becomes a continuous proximity band.
The width of the band reflects local constraint density.

### 2. triangle smoothing
Triangular faces become small continuous patches that define local
orientation and neighborhood overlap.

### 3. tetrahedral smoothing
Each tetrahedral cell induces a continuity‑like region whose shape
reflects the local packing constraints.

### 4. adjacency preservation
Smoothing does not alter adjacency:
- if two vertices were adjacent, their shadows overlap
- if they were not adjacent, their shadows remain separated

### 5. slush influence
This module records local deviations in smoothed neighborhoods. Geometry-dynamics may evaluate those deviations using a curvature-like profile downstream.

---

## invariants
- adjacency is preserved exactly
- no new adjacency relations appear
- smoothing introduces no new combinatorics
- all continuity‑like behavior originates from discrete structure
