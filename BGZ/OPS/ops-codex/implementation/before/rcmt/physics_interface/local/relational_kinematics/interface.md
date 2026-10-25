# Interface: relational_kinematics

## Purpose
Defines the explicit interface for kinematic transitions.
All downstream modules must use this interface when describing admissible local changes.

---

## Inputs

### 1. Discrete Geometry Inputs
- adjacency list (explicit)
- survivor record (geometry‑discrete)

### 2. Continuous Shadow Inputs (consumed from geometry-continuous outputs)
- shadow coordinate (geometry‑continuous)
- shadow adjacency (if present)

### 3. Local Invariants
Imported from:
- `local/invariants.md`
- `local/constraints/adjacency_refinements.md`
- `local/constraints/containment_refinements.md`

---

## Outputs

### 1. Local Transition Descriptor
A canonical record describing:
- the initial local configuration
- the admissible change
- the resulting local configuration

### 2. Adjacency‑Preserving Change Summary
A structural summary ensuring:
- adjacency is preserved
- no non‑local propagation occurs
- no hidden state is introduced

### 3. Kinematic Boundary Record
Defines the boundary conditions for downstream dynamics.

---

## Allowed Transformations
- explicit adjacency‑local updates
- explicit containment‑safe updates
- explicit relabeling (non‑interpretive)
- explicit state transitions

---

## Forbidden Transformations
- no metric usage
- no physics‑time usage
- no PDE forms
- no continuous evolution
- no implicit adjacency
- no hidden state

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden state
- Adjacency‑only propagation
