# Examples: relational_field_theory_discrete

## Purpose
Provides small, adjacency‑safe examples of discrete field behavior.
These examples illustrate structure only and do not introduce interpretation.

---

## Example 1: Field Assignment

**Given:**
- adjacency index: A
- field value: `v0`

**Apply:**
`assign_field(A, v0)`

**Result:**
- field assignment record
- no continuous interpolation

---

## Example 2: Field Update

**Given:**
- propagation record: `P`
- current field value: `v0`

**Apply:**
`update_field(P, v0) → v1`

**Result:**
- updated field value `v1`
- adjacency preserved

---

## Example 3: Field Support

**Given:**
- adjacency list: {A, B, C}

**Apply:**
`field_support({A, B, C})`

**Result:**
- discrete field support structure
- ready for `continuous_field_theory`

---

# RCP‑1.1 Compliance
- All examples adjacency‑safe
- No interpretive semantics
- No hidden state
