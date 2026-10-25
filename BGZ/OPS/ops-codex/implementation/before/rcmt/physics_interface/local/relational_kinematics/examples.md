# Examples: relational_kinematics

## Purpose
Provides small, adjacency‑safe examples of kinematic transitions.
These examples illustrate structure only and do not introduce interpretation.

---

## Example 1: Local Update

**Given:**
- node A with adjacency {B, C}
- local state: `s0`

**Apply:**
`local_update(s0, {B, C}) → s1`

**Result:**
- new local state: `s1`
- adjacency preserved
- no metric or time used

---

## Example 2: Transition Descriptor

**Given:**
- pre‑state: `s1`
- post‑state: `s2`

**Apply:**
`transition_descriptor(s1 → s2)`

**Result:**
- explicit transition record
- no hidden state
- no non‑local propagation

---

## Example 3: Boundary Record

**Given:**
- transition descriptor: `T`

**Apply:**
`boundary_record(T)`

**Result:**
- kinematic boundary record
- ready for `relational_dynamics`

---

# RCP‑1.1 Compliance
- All examples adjacency‑safe
- No interpretive semantics
- No hidden state
