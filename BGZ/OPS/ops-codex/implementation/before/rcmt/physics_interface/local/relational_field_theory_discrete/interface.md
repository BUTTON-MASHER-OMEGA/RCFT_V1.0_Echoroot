# Interface: relational_field_theory_discrete

## Purpose
Defines the explicit interface for constructing and updating discrete fields.
All downstream modules must use this interface when working with adjacency‑indexed field structures.

---

## Inputs

### 1. Propagation Inputs
Imported from `relational_dynamics`:
- propagation record
- discrete field handoff record

### 2. Adjacency Structure
- explicit adjacency list
- local neighborhood

### 3. Local Invariants
Imported from:
- `local/invariants.md`
- `local/constraints/adjacency_refinements.md`

---

## Outputs

### 1. Discrete Field Support
Defines where field values may be assigned.

### 2. Field Assignment Record
Explicit mapping from:
- adjacency index → field value

### 3. Field Update Record
Describes how field values change under admissible propagation.

---

## Allowed Transformations
- explicit field assignment
- explicit adjacency‑indexed updates
- explicit field propagation

---

## Forbidden Transformations
- no continuous interpolation
- no metric usage
- no PDE forms
- no physics‑time usage
- no amplitude structure
- no hidden state

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden state
- Adjacency‑only propagation
