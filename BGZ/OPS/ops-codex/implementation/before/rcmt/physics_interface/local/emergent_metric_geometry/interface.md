# Interface: emergent_metric_geometry

## Purpose
Defines the explicit interface for introducing metric structure.
All downstream modules must use this interface when referencing metric geometry.

---

## Inputs

### 1. Physics‑Time Inputs
Imported from `time_bridge`:
- physics‑time index
- temporal boundary record

### 2. Continuous Field Inputs
Imported from `continuous_field_theory`:
- continuous field support
- continuous field assignment record
- continuous field update record

### 3. Local Geometry Constraints
Imported from:
- `local/constraints/geometry_constraints.md`

---

## Outputs

### 1. Metric Assignment Record
Explicit mapping from:
- continuous support → metric value

### 2. Metric Boundary Record
Defines the metric boundaries for downstream modules.

### 3. Geometry‑Metric Compatibility Record
Ensures metric assignments respect upstream geometry ownership.

---

## Allowed Transformations
- explicit metric assignment
- explicit compatibility checks
- explicit boundary construction

---

## Forbidden Transformations
- no curvature
- no geodesics
- no PDEs
- no metric evolution
- no reinterpretation of geometry
- no hidden metric state

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden geometry
- Geometry ownership preserved
