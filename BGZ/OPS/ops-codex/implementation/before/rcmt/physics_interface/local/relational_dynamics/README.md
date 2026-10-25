# relational_dynamics

## Purpose
`relational_dynamics` defines the downstream‑safe rules for how admissible local transitions propagate across adjacency.
It builds directly on `relational_kinematics` and provides the structural machinery required for discrete, adjacency‑local evolution.

This module does not define continuous evolution, metric‑based dynamics, or physics‑time behavior.
It defines only the discrete propagation rules permitted before `time_bridge`.

---

## Position in the Stack
`relational_dynamics` is loaded immediately after `relational_kinematics`.

It depends on:
- relational_kinematics
- RCD admissibility language
- local invariants and constraints

It provides:
- local update admissibility record
- adjacency‑propagation record
- discrete field handoff record

---

## What This Module Does
- determines whether a local transition is admissible
- defines how transitions propagate across adjacency
- ensures propagation remains local and explicit
- prepares discrete field structures for downstream modules

---

## What This Module Does *Not* Do
- no metric usage
- no physics‑time usage
- no PDE forms
- no continuous evolution
- no hidden scheduler
- no non‑local propagation

These prohibitions are structural and enforced by upstream invariants.

---

## Structural Notes
- propagation is adjacency‑only
- admissibility is explicit and rule‑based
- no global scheduler is permitted
- no implicit propagation is allowed

---

## Cross‑Module Links
- `relational_field_theory_discrete` consumes the propagation records defined here
- `time_bridge` cannot import from this module until its own constraints are satisfied

---

## Files in This Module
- `interface.md` — explicit interface and boundary rules
- `operators.md` — canonical operators for dynamic propagation
- `examples.md` — adjacency‑safe examples of dynamic updates

---

# RCP‑1.1 Compliance
- No interpretive semantics
- No hidden state
- All propagation adjacency‑local
- All invariants preserved
