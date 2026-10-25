# Geometry Constraints (local/constraints/geometry_constraints.md)

## Purpose
This file defines project‑specific refinements to RCF geometry rules.
Geometry must remain upstream‑owned and invariant‑compatible.

These refinements:
- do not override geometry‑continuous or geometry‑discrete
- do not introduce hidden geometry
- do not introduce interpretive semantics

---

## 1. Geometry Ownership

### 1.1 Upstream Geometry
- geometry‑continuous and geometry‑discrete remain upstream
- no downstream module may redefine geometry ownership

### 1.2 Allowed Usage
- downstream modules may consume declared upstream geometry interfaces
- geometry may not be modified downstream

---

## 2. Metric Boundary

### 2.1 Allowed Usage
- metric structure appears only after `emergent_metric_geometry`
- metric assignments must be explicit

### 2.2 Forbidden Usage
- no pre‑metric geometry claims
- no hidden metric state
- no metric inference from adjacency

---

## 3. Geometry Refinements

### 3.1 Allowed Refinements
- explicit geometry‑compatible labels
- explicit continuous shadow structures

### 3.2 Forbidden Refinements
- no continuous geometry upstream of geometry‑continuous
- no geometry derived from physics modules

---

# RCP‑1.1 Compliance
- All constraints explicit
- No interpretive semantics
- No hidden geometry
- Ready for downstream usage
