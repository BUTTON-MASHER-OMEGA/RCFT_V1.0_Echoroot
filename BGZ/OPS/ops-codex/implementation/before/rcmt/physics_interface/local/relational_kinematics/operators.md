# Operators: relational_kinematics

## Purpose
Defines the canonical operators used to construct kinematic transitions.
These operators are adjacency‑safe, explicit, and non‑interpretive.

---

## 1. `local_update`
Describes an admissible change to a local configuration.

**Inputs:**
- local state
- adjacency neighborhood
- invariant constraints

**Outputs:**
- updated local state
- updated adjacency summary

**Rules:**
- must preserve adjacency
- must remain explicit
- must not introduce hidden state

---

## 2. `transition_descriptor`
Constructs the canonical transition record.

**Inputs:**
- pre‑state
- post‑state
- update operator

**Outputs:**
- transition descriptor

**Rules:**
- must be explicit
- must not encode metric or time

---

## 3. `boundary_record`
Defines the kinematic boundary for downstream modules.

**Inputs:**
- transition descriptor
- local invariants

**Outputs:**
- boundary record

**Rules:**
- must not introduce new invariants
- must not encode dynamics

---

# RCP‑1.1 Compliance
- All operators canonical
- No interpretive semantics
- No hidden state
- Adjacency‑only propagation
