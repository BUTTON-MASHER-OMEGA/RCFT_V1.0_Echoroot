# least-action/survivorship-bias.md

## Document Contract

- owner: RCF geometry-dynamics module
- type: Interpretation
- scope: module-local content of `geometry/dynamics/least-action/survivorship-bias.md`
- authority: tier-8 downstream continuous-variation layer
- upstream dependencies: `/modules/geometry/dynamics/interface.md`
- status: active
- purpose: describe continuous relational variation without introducing forces, fields, or upstream primitives
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document describes survivorship bias in the context of the action functional.
The goal is to record how **survivor‑consistent configurations** appear in the continuous shadow and how this appearance relates to stationary action (δS = 0) **without implying optimization, causality, or physical interpretation**.

Survivorship bias introduces **no new primitives** and remains fully determined by:

- the survivor sequence,
- discrete mismatch structure,
- and Π’s smoothing rules.

---

## upstream origin
Survivorship bias arises from the fact that Π projects only the **surviving** discrete configurations into the continuous shadow.

From upstream documentation:

> “The projected remainder is observed to correspond to stationary points of the action functional in current evaluations.”

This statement is interpreted structurally:

- non‑survivor configurations do not appear in the shadow,
- the survivor sequence corresponds to configurations where first‑order variation of cumulative mismatch is observed to vanish,
- and this correspondence is descriptive, not prescriptive.

No teleology or optimization is assumed.

---

## structural description
Survivorship bias is described as the observation that:

- the continuous shadow contains only survivor‑consistent configurations,
- these configurations correspond to stationary points of the action functional in current evaluations,
- and non‑stationary configurations are absent, consistent with their non-survival under projection.

This framing treats δS = 0 as a **summary property** of the survivor, not a governing rule.

---

## relation to the action functional
Let \(S[\Phi]\) denote the action functional defined as cumulative mismatch.

Survivorship bias is reflected in the following observations:

- survivor‑consistent configurations correspond to stationary points of \(S\),
- Configurations with non-vanishing first-order variation are not observed in the projected shadow,
- and the appearance of δS = 0 in the shadow is a consequence of projection, not a constraint imposed upstream.

This maintains the retroactive interpretation of stationary action.

---

## structural properties

### 1. Π‑determinism
Survivorship bias is fully determined by:

- the survivor sequence,
- mismatch structure,
- and Π’s smoothing behavior and survivor-consistent projection.

No external selection principle is introduced.

### 2. no optimization
Survivorship bias does **not** imply:

- minimization,
- optimality,
- efficiency,
- or teleological behavior.

The correspondence between survivors and stationary action is descriptive.

### 3. no dynamics
Survivorship bias does not describe:

- evolution,
- trajectories,
- or dynamical processes.

It records which configurations appear in the shadow, not how they arise.

### 4. coordinate‑free and metric‑optional
The description does not rely on:

- coordinates,
- metrics,
- or geometric interpretation.

All structure is relational.

---

## role in least‑action
Survivorship bias provides the structural context for interpreting δS = 0:

- stationary action is a property of the survivor,
- not a rule governing flow or mismatch,
- and not a principle imposed on the system.

This file ensures that least‑action remains **retroactive**, **structural**, and **upstream‑safe**.

---

## invariants
Survivorship bias satisfies the invariants of the geometry‑dynamics module:

- no new primitives,
- coordinate‑free formulation,
- metric‑optional structure,
- Π‑determinism,
- locality of density (via mismatch),
- retroactive interpretation of δS = 0,
- no physical interpretation.

---

## status
The survivorship‑bias description is structurally complete and ready for integration with least‑action and the geometry‑dynamics README.

