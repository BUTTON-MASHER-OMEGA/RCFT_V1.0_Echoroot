# Operators: emergent_metric_geometry

## Purpose
Defines the canonical operators used to assign metric structure and construct metric boundaries.
These operators are explicit, invariant‑safe, and non‑interpretive.

---

## 1. `assign_metric`
Assigns a metric value to a continuous support location.

**Inputs:**
- continuous support
- local geometry constraints
- physics‑time index (optional)

**Outputs:**
- metric assignment record

**Rules:**
- must be explicit
- must not introduce curvature
- must not use PDEs

---

## 2. `metric_boundary`
Constructs the metric boundary record.

**Inputs:**
- metric assignment record
- temporal boundary record

**Outputs:**
- metric boundary record

**Rules:**
- must not encode dynamics
- must not introduce new invariants

---

## 3. `compatibility_check`
Ensures metric assignments respect upstream geometry.

**Inputs:**
- metric assignment record
- geometry constraints

**Outputs:**
- geometry‑metric compatibility record

**Rules:**
- must not reinterpret geometry
- must not introduce hidden geometry

---

# RCP‑1.1 Compliance
- All operators canonical
- No interpretive semantics
- No hidden geometry
- Geometry ownership preserved
