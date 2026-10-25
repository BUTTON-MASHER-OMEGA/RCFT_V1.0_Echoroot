# Operators: relational_dynamics

## Purpose
Defines the canonical operators used to evaluate admissibility and propagate transitions.
These operators are adjacency‑safe, explicit, and non‑interpretive.

---

## 1. `admissible_update`
Determines whether a proposed transition is allowed.

**Inputs:**
- transition descriptor
- adjacency neighborhood
- local invariants

**Outputs:**
- admissibility record

**Rules:**
- must be explicit
- must not use metric or time
- must not introduce hidden state

---

## 2. `propagate_local`
Propagates an admissible update across adjacency.

**Inputs:**
- admissibility record
- adjacency list
- local state

**Outputs:**
- propagation record

**Rules:**
- propagation must be adjacency‑local
- no global propagation allowed

---

## 3. `field_handoff`
Constructs the discrete field handoff record.

**Inputs:**
- propagation record
- local invariants

**Outputs:**
- discrete field handoff

**Rules:**
- must not introduce new invariants
- must not encode continuous evolution

---

# RCP‑1.1 Compliance
- All operators canonical
- No interpretive semantics
- No hidden state
- Adjacency‑only propagation
