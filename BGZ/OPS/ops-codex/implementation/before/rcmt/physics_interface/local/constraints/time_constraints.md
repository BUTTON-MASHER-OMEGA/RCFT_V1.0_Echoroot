# Time Constraints (local/constraints/time_constraints.md)

## Purpose
Defines project‑specific refinements to RCF time rules.
Time must remain explicit, emergent, and downstream‑owned.

These refinements:
- do not override the Upstream Time Index
- do not introduce alternative upstream ordering indices
- do not introduce interpretive temporal semantics

---

## 1. Time Ownership Invariants

### 1.1 Ownership
- The RCF Time Core is the sole upstream owner of the **Upstream Time Index (UTI)**.
- The RCF Temporal Interface is the sole upstream owner of exposed local ordering parameters.
- The Upstream Time Index is **not temporal**; it is a symbolic ordering key.
- No module may redefine the RCF Time Core ordering ontology.
- No module may redefine or bypass Temporal Interface parameter boundaries.

### 1.2 Allowed Usage
- The Upstream Time Index may be imported by `time_bridge`.
- The Upstream Time Index may not be modified downstream.

---

## 2. Physics‑Time Boundary

### 2.1 Allowed Usage
- Physics‑Time appears only after `time_bridge`.
- Physics‑Time must be explicitly declared.

### 2.2 Forbidden Usage
- no pre‑bridge physics‑time usage
- no hidden temporal interpolation
- no implicit time dilation
- no upstream temporal semantics

---

## 3. Temporal Refinements

### 3.1 Allowed Refinements
- explicit temporal markers
- explicit time‑bridge handoff tokens

### 3.2 Forbidden Refinements
- no continuous‑time assumptions upstream
- no metric‑derived time upstream
- no reinterpretation of the Upstream Time Index as temporal

---

# RCP‑1.1 Compliance
- All constraints explicit
- No interpretive semantics
- No hidden time
- Ready for downstream usage
