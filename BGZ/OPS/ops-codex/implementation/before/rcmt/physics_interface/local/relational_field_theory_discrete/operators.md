# Operators: relational_field_theory_discrete

## Purpose
Defines the canonical operators used to construct and update discrete fields.
These operators are adjacency‑safe, explicit, and non‑interpretive.

---

## 1. `assign_field`
Assigns a field value to a discrete support location.

**Inputs:**
- adjacency index
- field value
- local invariants

**Outputs:**
- field assignment record

**Rules:**
- must be explicit
- must not use metric or time

---

## 2. `update_field`
Updates a field value under admissible propagation.

**Inputs:**
- propagation record
- current field value
- adjacency neighborhood

**Outputs:**
- updated field value
- field update record

**Rules:**
- propagation must be adjacency‑local
- no continuous interpolation

---

## 3. `field_support`
Constructs the discrete field support structure.

**Inputs:**
- adjacency list
- local invariants

**Outputs:**
- field support

**Rules:**
- must not introduce new invariants
- must not encode continuous geometry

---

# RCP‑1.1 Compliance
- All operators canonical
- No interpretive semantics
- No hidden state
- Adjacency‑only propagation
