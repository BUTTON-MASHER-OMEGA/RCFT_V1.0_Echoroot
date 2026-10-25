# relational_kinematics

## Purpose
`relational_kinematics` defines the downstream‑safe kinematic layer for the physics stack.
It provides the minimal structural machinery required to describe admissible local transitions without invoking metric structure, physics‑time, or continuous geometry.

This module is the entry point for all downstream physics behavior.

---

## Position in the Stack
`relational_kinematics` is the first physics module loaded after the upstream RCF invariants and the project‑specific `local/` layer.

It depends on:
- geometry‑discrete survivor records
- geometry‑continuous shadow records
- RCD admissible‑change language
- local invariants and constraints

It provides:
- local transition descriptor
- adjacency‑preserving change summary
- downstream‑safe kinematic boundary record

Continuous shadow structures are consumed from upstream geometry-continuous and are not defined in this module.

---

## What This Module Does
- defines how local states may change
- defines how adjacency‑safe transitions are represented
- defines the boundary conditions for downstream dynamics
- ensures all kinematic updates remain explicit and non‑interpretive

---

## What This Module Does *Not* Do
- no metric assignment
- no physics‑time assignment
- no PDEs
- no continuous evolution
- no hidden state
- no non‑local propagation

These prohibitions are structural and enforced by upstream invariants.

---

## Structural Notes
- All transitions must be adjacency‑local.
- All state changes must be explicit.
- All kinematic descriptors must be representable without metric or time.
- This module defines *what can change*, not *how it evolves*.

Evolution belongs to `relational_dynamics`.

---

## Cross‑Module Links
- `relational_dynamics` consumes the transition descriptors defined here.
- `relational_field_theory_discrete` uses the kinematic boundary record to define discrete field support.
- `time_bridge` cannot import from this module until its own dependencies are satisfied.

---

## Files in This Module
- `interface.md` — explicit interface and boundary rules
- `operators.md` — canonical operators for kinematic transitions
- `examples.md` — adjacency‑safe examples of kinematic transitions

---

# RCP‑1.1 Compliance
- No interpretive semantics
- No hidden state
- All transitions explicit
- All invariants preserved
