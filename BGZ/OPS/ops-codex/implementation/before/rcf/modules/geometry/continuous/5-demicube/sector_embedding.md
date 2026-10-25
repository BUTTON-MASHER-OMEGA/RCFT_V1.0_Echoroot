# embedding of geometric sectors in the 5‑demicube shadow

## Document Contract

- owner: RCF continuous geometry module
- type: Interpretation
- scope: module-local content of `geometry/continuous/5-demicube/sector_embedding.md`
- authority: tier-7 continuous-shadow representation layer
- upstream dependencies: `/modules/geometry/continuous/interface.md`
- status: active
- purpose: represent the continuous relational shadow without owning metrics, curvature, flow, or physical interpretation
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document describes how the geometric sectors defined in
`geometric_sectors.md` embed into the continuous shadow of the 5‑demicube.

The embedding is structural and does not assign physical meaning.

---

## overview
The geometric sectors embed into the 5‑demicube shadow according to:

- adjacency relations
- directional combinatorial structure
- γ‑plateau assignments
- dual‑closure tags

The embedding preserves all upstream invariants.

---

## embedding rules

### 1. sector → region mapping
Each sector corresponds to a continuous region determined by:
- its discrete support
- its directional adjacency footprint
- its local packing constraints

### 2. γ‑plateau → generation band
Sectors with different γ‑plateau assignments occupy distinct but aligned
bands in the shadow.

### 3. eigenvalue ordering → radial‑directional ordering
Lower eigenvalue sectors embed closer to the shadow’s “core,” but the
embedding is anisotropic due to the demicube’s directional structure.

### 4. adjacency → overlap structure
Adjacent sectors produce overlapping shadow regions.
Non‑adjacent sectors remain disjoint.

### 5. dual‑closure → symmetry persistence
Sectors with matching dual‑closure tags embed into symmetry‑related
regions aligned with the demicube’s coordinate axes.

---

## invariants
- no new sectors appear
- no sectors merge
- embedding preserves adjacency, ordering, and plateau structure
- the embedding is fully determined by Π and the discrete 5‑demicube

