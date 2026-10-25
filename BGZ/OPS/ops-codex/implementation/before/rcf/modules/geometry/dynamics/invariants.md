# invariants.md

## Document Contract

- owner: RCF geometry-dynamics module
- type: Contract
- scope: module-local content of `geometry/dynamics/invariants.md`
- authority: tier-8 downstream continuous-variation layer
- upstream dependencies: `/modules/geometry/dynamics/interface.md`
- status: active
- purpose: describe continuous relational variation without introducing forces, fields, or upstream primitives
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document records the structural constraints that govern the entire
geometry‑dynamics module.
All operators in this layer—mismatch, curvature, flow, and least‑action—must
satisfy these invariants to remain upstream‑safe, coordinate‑free, and fully
determined by the projection operator Π.

No new primitives are introduced in this module.
All continuous structure is inherited from the discrete layer.

---

## 1. no new primitives
The dynamics layer must not introduce:

- new fields,
- new forces,
- new potentials,
- new geometric objects,
- new constants,
- or new degrees of freedom.

All operators must be derived from mismatch, slush, recursive closure, and Π’s smoothing rules:

- mismatch,
- curvature,
- slush,
- recursive closure,
- and Π’s smoothing rules.

---

## 2. coordinate‑free formulation
All dynamics must be expressible without:

- coordinate charts,
- basis choices,
- affine connections,
- or differential‑geometric assumptions.

Operators must act on the continuous shadow relationally, not through coordinates.

---

## 3. metric‑optional structure
The dynamics layer must not require a metric.

Metrics may be introduced downstream, but:

- mismatch,
- curvature,
- flow,
- and the action functional

must all be definable without one.

This preserves compatibility with any downstream geometric interpretation.

---

## 4. Π‑determinism
Every operator in geometry‑dynamics must be fully determined by:

- the discrete mismatch structure,
- the survivor sequence,
- and the projection operator Π.

No operator may depend on:

- hidden variables,
- external laws,
- injected physics,
- or arbitrary choices.

Π is the sole source of continuous structure.

---

## 5. locality
All dynamics must be local in the shadow:

- mismatch is local,
- curvature is local,
- flow is local,
- action density is local.

Operators must be locally defined, though their cumulative effects may reflect global survivor structure.

This mirrors the locality of the LCO.

---

## 6. relational structure only
All operators must be expressible in terms of:

- adjacency,
- mismatch gradients,
- curvature response,
- and slush redistribution.

No absolute coordinates, scales, or external frames may appear.

---

## 7. evolution without time
Flow must be expressed without assuming a time coordinate.

The correct structural form is:



\[
\Delta \Phi = \mathcal{F}[\Phi]
\]



not:



\[
\frac{d\Phi}{dt}.
\]



A continuous parameter (τ) appears only downstream in least‑action, where Π
induces a parameterization of the shadow.

Flow‑core must remain parameter‑agnostic.

---

## 8. retroactive optimality
The Principle of Stationary Action (δS = 0) must be interpreted as:

- a structural consequence of survivorship,
- not a forward‑looking optimization,
- not a teleological principle.

The action functional summarizes the survivor; it does not govern evolution.

---

## 9. downstream compatibility
The dynamics layer must remain compatible with:

- any metric choice,
- any coordinate choice,
- any downstream physical interpretation.

But it must not depend on any of them.

This ensures modularity and prevents upstream contamination.

---

## status
This document defines the structural contract for the geometry‑dynamics module.
All operators in this layer must satisfy these invariants to maintain architectural
clarity, upstream safety, and compatibility with the projection framework.

