# Examples: relational_dynamics

## Purpose
Provides small, adjacency‑safe examples of dynamic propagation.
These examples illustrate structure only and do not introduce interpretation.

---

## Example 1: Admissible Update

**Given:**
- transition descriptor: `T`
- adjacency: {A, B, C}

**Apply:**
`admissible_update(T, {A, B, C}) → allowed`

**Result:**
- explicit admissibility record
- no hidden scheduler

---

## Example 2: Local Propagation

**Given:**
- admissibility: allowed
- adjacency: {A, B}

**Apply:**
`propagate_local(allowed, {A, B}) → P`

**Result:**
- propagation record `P`
- adjacency preserved

---

## Example 3: Field Handoff

**Given:**
- propagation record: `P`

**Apply:**
`field_handoff(P)`

**Result:**
- discrete field handoff
- ready for `relational_field_theory_discrete`

---

# RCP‑1.1 Compliance
- All examples adjacency‑safe
- No interpretive semantics
- No hidden state
