# Operators: time_bridge

## Purpose
Defines the canonical operators used to construct the Physics‑Time index and temporal boundaries.
These operators are explicit, invariant‑safe, and non‑interpretive.

---

## 1. `map_time`
Constructs the mapping from the Upstream Time Index to Physics‑Time.

**Inputs:**
- Upstream Time Index
- local time constraints

**Outputs:**
- physics‑time index
- time‑mapping record (π)

**Rules:**
- mapping must be explicit
- no continuous interpolation
- no metric usage

---

## 2. `time_handoff`
Constructs the time‑bridge handoff record.

**Inputs:**
- physics‑time index
- Upstream Time Index

**Outputs:**
- time‑bridge handoff

**Rules:**
- must not modify the upstream ordering index
- must not introduce hidden time

---

## 3. `temporal_boundary`
Defines the temporal boundary for downstream modules.

**Inputs:**
- time‑bridge handoff
- local invariants

**Outputs:**
- temporal boundary record

**Rules:**
- must not encode dynamics
- must not introduce new invariants

---

# RCP‑1.1 Compliance
- All operators canonical
- No interpretive semantics
- No hidden time
- Upstream ordering preserved
