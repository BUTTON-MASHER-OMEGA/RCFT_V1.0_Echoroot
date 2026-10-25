# embedding of geometric sectors in the 120‑cell shadow

## Document Contract

- owner: RCF continuous geometry module
- type: Interpretation
- scope: module-local content of `geometry/continuous/120-cell/sector_embedding.md`
- authority: tier-7 continuous-shadow representation layer
- upstream dependencies: `/modules/geometry/continuous/interface.md`
- status: active
- purpose: represent the continuous relational shadow without owning metrics, curvature, flow, or physical interpretation
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document describes how the geometric sectors defined in
`geometric_sectors.md` embed into the continuous shadow of the 120‑cell.

The embedding is structural and does not assign physical meaning.

---

## overview
The 36 geometric sectors (or however many are provided upstream) embed
into the continuous shadow according to:

- adjacency relations
- local combinatorial structure
- γ‑plateau assignments
- dual‑closure tags

The embedding preserves all upstream invariants.

---

## embedding rules

### 1. sector → region mapping
Each sector corresponds to a continuous region in the shadow determined by:
- its discrete support
- its adjacency footprint
- its local packing constraints

### 2. γ‑plateau → generation band
Sectors with different γ‑plateau assignments occupy distinct but aligned
bands in the shadow.

### 3. eigenvalue ordering → radial ordering
Sectors with lower discrete Laplacian eigenvalues embed closer to the
shadow’s “core,” while higher eigenvalue sectors embed further out.
This is an ordering relation only.

### 4. adjacency → overlap structure
If two sectors are adjacent in the 120‑cell, their shadow regions overlap.
If they are not adjacent, their regions remain disjoint.

### 5. dual‑closure → symmetry persistence
Sectors with matching dual‑closure tags embed into symmetry‑related
regions of the shadow.

---

## invariants
- no new sectors appear
- no sectors merge
- embedding preserves adjacency, ordering, and plateau structure
- the embedding is fully determined by Π and the discrete 120‑cell

