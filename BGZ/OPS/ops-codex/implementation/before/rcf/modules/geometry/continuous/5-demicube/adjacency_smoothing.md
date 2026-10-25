# adjacency smoothing in the 5‑demicube shadow

## Document Contract

- owner: RCF continuous geometry module
- type: Interpretation
- scope: module-local content of `geometry/continuous/5-demicube/adjacency_smoothing.md`
- authority: tier-7 continuous-shadow representation layer
- upstream dependencies: `/modules/geometry/continuous/interface.md`
- status: active
- purpose: represent the continuous relational shadow without owning metrics, curvature, flow, or physical interpretation
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document describes how discrete adjacency in the 5‑demicube is smoothed
into continuous neighborhood structure by the projection operator Π.

The smoothing process is structural and does not assume a metric or
coordinate system.

---

## overview
Adjacency smoothing converts:

- sparse edges → directional proximity bands
- demicube faces → anisotropic local patches
- higher‑dimensional cells → elongated continuity‑like envelopes

All relational invariants are preserved.

---

## smoothing rules

### 1. edge smoothing
Each edge becomes a continuous proximity band whose width depends on
local constraint density. Bands are direction‑dependent.

### 2. face smoothing
Demicube faces produce elongated patches that define directional
neighborhood overlap.

### 3. cell smoothing
Higher‑dimensional demicube cells induce continuity‑like regions
with anisotropic shape.

### 4. adjacency preservation
Smoothing preserves adjacency exactly:
- adjacent vertices produce overlapping shadow regions
- non‑adjacent vertices remain separated

### 5. slush influence
Local slush produces curvature‑like deviations that are more pronounced
along sparsely connected directions.

---

## invariants
- adjacency is preserved exactly
- no new adjacency relations appear
- smoothing introduces no new combinatorics
- continuity‑like behavior is inherited from discrete structure

