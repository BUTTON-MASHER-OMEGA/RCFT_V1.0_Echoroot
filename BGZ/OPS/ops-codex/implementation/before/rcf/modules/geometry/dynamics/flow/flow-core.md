# flow/flow‑core.md

## Document Contract

- owner: RCF geometry-dynamics module
- type: Contract
- scope: module-local content of `geometry/dynamics/flow/flow-core.md`
- authority: tier-8 downstream continuous-variation layer
- upstream dependencies: `/modules/geometry/dynamics/interface.md`
- status: active
- purpose: describe continuous relational variation without introducing forces, fields, or upstream primitives
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document describes flow as a **relational characterization of changes in mismatch gradients** under Π‑consistent smoothing.
Flow introduces **no new primitives** and remains fully determined by:

- mismatch density,
- curvature‑related variation,
- survivor structure,
- and Π’s smoothing rules.

Flow is parameter‑agnostic and does not assume a time coordinate.

---

## upstream origin
Flow is a **continuous shadow** of how mismatch‑related variation appears after projection.

From upstream documentation:

> “The flow does not assume a metric or manifold.
> It operates entirely on relational distances and slush gradients.”

Under Π:

- mismatch gradients are smoothed,
- second‑order variation is redistributed,
- and no coordinate or metric structure is introduced.

Flow reflects this observed redistribution behavior.

---

## definition
Let \(\Phi\) denote the continuous shadow produced by Π.
Flow may be represented using the structural form:

\[
\Delta \Phi = \mathcal{F}[\Phi]
\]

This expression:

- does **not** assume a time parameter,
- does **not** define a differential equation,
- and does **not** introduce a metric or coordinate system.

It is a **relational description** of how Π‑consistent smoothing affects mismatch‑related variation.

Flow is **Π‑determined**: all structure arises from adjacency, mismatch gradients, curvature‑related variation, and survivor‑consistent smoothing.

---

## structural properties

### 1. locality
Flow is locally described.
Each local description references:

- nearby mismatch gradients,
- local curvature‑related variation,
- adjacency relations,
- and slush redistribution.

### 2. Π‑determinism
Flow is described in terms of:

- mismatch,
- relational gradients,
- curvature‑related structure,
- survivor sequence,
- and Π’s smoothing rules.

No external structure contributes.

### 3. coordinate‑free behavior
Flow is described without:

- coordinate charts,
- basis choices,
- affine connections,
- or differential‑geometric assumptions.

All variation is relational.

### 4. metric‑optional behavior
Flow does not require a metric.
Downstream modules may introduce one, but flow is described independently of any metric choice.

### 5. parameter‑agnostic structure
Flow does not assume:

- time,
- evolution parameters,
- or dynamical trajectories.

Any downstream parameterization (τ) arises only in least‑action and is not part of flow‑core.

### 6. relational variation
Flow characterizes:

- how mismatch gradients appear to change under Π‑consistent smoothing,
- how slush redistribution affects local variation,
- and how curvature‑related structure shapes these observations.

This behavior is consistent with observed properties of mismatch and curvature under projection.

---

## flow behavior
Flow is **observed** to:

- correspond to reduced mismatch gradient variation under Π-consistent smoothing,
- correspond to slush redistribution patterns observed in the survivor structure,
- and smooth relational structure without introducing new primitives.

These observations do not imply mechanism, causality, or optimization.

---

## role in geometry‑dynamics
Flow is referenced by downstream components:

- **curvature‑driven flow descriptions** use curvature as a relational input,
- **least‑action** references flow when describing cumulative mismatch behavior.

Flow does not govern evolution and does not introduce temporal structure.

---

## invariants
Flow satisfies the invariants of the geometry‑dynamics module:

- no new primitives,
- coordinate‑free formulation,
- metric‑optional structure,
- Π‑determinism,
- locality,
- relational structure only,
- evolution without time,
- no physical interpretation.

---

## status
The flow description is structurally complete and ready for use by least‑action and downstream metric‑aware modules.

