# Interface: time_bridge

## Purpose
Defines the explicit interface for introducing Physics Time.
All downstream modules must use this interface when referencing temporal structure.

The **Upstream Time Index** is not temporal.
It is a symbolic ordering key.

---

## Inputs

### 1. Upstream Time Index
Imported from upstream:
- upstream ordering index
- upstream ordering relation

### 2. Local Time Constraints
Imported from:
- `local/constraints/time_constraints.md`

---

## Outputs

### 1. Physics‑Time Index
A downstream‑safe temporal index.

### 2. Time‑Bridge Handoff
Explicit mapping from:
- Upstream Time Index → Physics‑Time

### 3. Temporal Boundary Record
Defines the Physics‑Time boundaries for downstream modules.

---

## Allowed Transformations
- explicit time mapping
- explicit temporal indexing
- explicit boundary construction

---

## Forbidden Transformations
- no continuous time
- no metric time
- no time dilation
- no implicit interpolation
- no hidden scheduler
- no reinterpretation of upstream ordering

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden time
- Upstream ordering preserved
