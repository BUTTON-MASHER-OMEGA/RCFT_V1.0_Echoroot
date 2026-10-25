# Interface: relational_dynamics

## Purpose
Defines the explicit interface for dynamic propagation.
All downstream modules must use this interface when determining admissibility and adjacency‑local evolution.

---

## Inputs

### 1. Kinematic Inputs
Imported from `relational_kinematics`:
- transition descriptor
- boundary record

### 2. Adjacency Structure
- explicit adjacency list
- local neighborhood

### 3. Local Invariants
Imported from:
- `local/invariants.md`
- `local/constraints/adjacency_refinements.md`
- `local/constraints/containment_refinements.md`

---

## Outputs

### 1. Admissibility Record
Indicates whether a proposed transition is allowed under:
- adjacency rules
- containment rules
- local invariants

### 2. Propagation Record
Describes how the transition propagates across adjacency.

### 3. Discrete Field Handoff
Provides the structural handoff for `relational_field_theory_discrete`.

---

## Allowed Transformations
- explicit adjacency‑local propagation
- explicit admissibility checks
- explicit update sequencing

---

## Forbidden Transformations
- no hidden scheduler
- no global propagation
- no metric usage
- no physics‑time usage
- no PDE forms
- no implicit propagation

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden state
- Adjacency‑only propagation
