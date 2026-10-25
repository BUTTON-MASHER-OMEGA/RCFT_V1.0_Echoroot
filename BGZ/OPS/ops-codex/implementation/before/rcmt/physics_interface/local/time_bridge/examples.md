# Examples: time_bridge

## Purpose
Provides small, explicit examples of time‑bridge usage.
These examples illustrate structure only and do not introduce interpretation.

The **Upstream Time Index** is not temporal.
It is a symbolic ordering key supplied by upstream modules.

---

## Example 1: Time Mapping

**Given:**
- Upstream Time Index: `t0`

**Apply:**
`map_time(t0) → τ0`

**Result:**
- physics‑time index τ0
- no continuous interpolation

> This example illustrates the single‑index special case of the general finite‑window mapping defined in the formal structure.

---

## Example 2: Time Handoff

**Given:**
- upstream index: `t1`
- physics‑time: `τ1`

**Apply:**
`time_handoff(t1, τ1)`

**Result:**
- explicit time‑bridge handoff

---

## Example 3: Temporal Boundary

**Given:**
- time‑bridge handoff: `H`

**Apply:**
`temporal_boundary(H)`

**Result:**
- temporal boundary record
- ready for `continuous_field_theory`

---

# RCP‑1.1 Compliance
- All examples explicit
- No interpretive semantics
- No hidden time
