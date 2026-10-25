# least-action/least-action.md

## Document Contract

- owner: RCF geometry-dynamics module
- type: Interpretation
- scope: module-local content of `geometry/dynamics/least-action/leaast-action.md`
- authority: tier-8 downstream continuous-variation layer
- upstream dependencies: `/modules/geometry/dynamics/interface.md`
- status: active
- purpose: describe continuous relational variation without introducing forces, fields, or upstream primitives
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

The least‑action description is structurally complete and ready for use in downstream variational and metric‑aware modules.

## purpose
This document describes the action functional as a **continuous representation of cumulative mismatch** in the geometry‑dynamics module.
The action introduces **no new primitives** and remains fully determined by:

- the continuous mismatch functional \(L\),
- survivor structure,
- and Π’s smoothing rules.

Stationary action (δS = 0) is treated as a **retroactive structural observation**, not a forward‑looking optimization principle.

---

## upstream origin
The action functional is a **continuous shadow** of discrete cumulative mismatch.

From upstream documentation:

> “The discrete summation of mismatch is represented as a continuous action functional.”

and

> “The projected remainder is observed to correspond to stationary points of the action functional minimum of the action landscape.”

Under projection:

- discrete mismatch contributions are accumulated,
- survivor‑consistent structure is retained,
- this accumulation is represented as a continuous functional and corresponds to configurations where first-order variation of the action vanishes in current evaluations.

No new structure is introduced beyond what is already present in mismatch and Π.

---

## definition
Let \(L[\Phi]\) denote the continuous mismatch functional on the shadow produced by Π.
The action functional \(S\) is described as a continuous accumulation of mismatch:

\[
S[\Phi] = \int L[\Phi] \, d\tau
\]

This expression:

- uses \(\tau\) as a **downstream parameterization** induced by Π,
- does **not** assume physical time,
- and does **not** introduce a metric or coordinate system.

The integral denotes ordered accumulation under Π-induced parameterization and serves as a **structural representation** of cumulative mismatch, not a dynamical law.

---

## structural properties

### 1. Π‑determinism
The action functional is fully determined by:

- mismatch \(L\),
- survivor structure,
- and Π’s smoothing and parameterization rules.

No external laws, hidden variables, or injected physics contribute.

### 2. locality of density
The integrand \(L\) is locally described:

- each contribution reflects local mismatch density,
- all dependence is relational and adjacency‑based,
- and no nonlocal operator is assumed.

The action is cumulative, but its density remains local.

### 3. coordinate‑free behavior
The action is described without:

- coordinate charts,
- basis choices,
- affine connections,
- or differential‑geometric assumptions.

Any parameterization \(\tau\) is induced structurally by Π and does not carry physical interpretation.

### 4. metric‑optional structure
The action does not require a metric.
Downstream modules may introduce one, but \(S\) is described independently of any metric choice.

---

## stationary action (δS = 0)

### 1. retroactive interpretation
Stationary action is treated as:

- a **structural summary** of the survivor,
- not a governing principle,
- not a teleological rule,
- and not a forward‑looking optimization.

In current evaluations, survivor‑consistent configurations are **observed** to correspond to stationary points of the action functional in current evaluations.

### 2. survivorship framing
The condition δS = 0 is understood as:

- a description of the projected remainder,
- consistent with the fact that non‑survivor configurations do not appear in the shadow,
- and not as a constraint imposed on evolution.

This framing avoids teleology and keeps stationary action as a **summary property**, not a cause.

---

## role in geometry‑dynamics
The action functional is referenced when:

- describing cumulative mismatch along Π‑induced parameterizations,
- providing a structural description relating local mismatch density to global survivor-consistent structure,
- and expressing stationary action as a compact summary of survivor behavior.

Action does not govern flow, curvature, or mismatch; it **summarizes** their cumulative relational structure.

---

## invariants
The action functional satisfies the invariants of the geometry‑dynamics module:

- no new primitives,
- coordinate‑free formulation,
- metric‑optional structure,
- Π‑determinism,
- locality of density,
- evolution without time (τ is non‑temporal),
- retroactive optimality (δS = 0 as a summary, not a law),
- no physical interpretation.

---

## status
The least‑action description is structurally complete and ready for use in downstream variational and metric‑aware modules.
