# Operators: continuous_field_theory

## Purpose
Defines canonical operators for lifting and updating continuous fields.

---

## 1. `lift_field`
Lifts a discrete field to a continuous support.

**Inputs:**
- discrete field support
- field assignment record

**Outputs:**
- continuous field support \(X\)
- continuous field assignment \(f\):\(X\) → \(V\)
- continuous field assignment record (a record encoding the lifted field \(f\) over\(X\))

**Rules:**
- lifting must be explicit
- no hidden interpolation

---

## 2. `update_continuous_field`
Updates a continuous field under admissible rules.

**Inputs:**
- continuous field assignment
- physics‑time index

**Outputs:**
- continuous field update record

**Rules:**
- no PDEs
- no metric usage

---

# RCP‑1.1 Compliance
- All operators explicit
- No interpretive semantics
- No hidden state
