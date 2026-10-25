# Local → Geometry Adapter (local/adapters/local_to_geometry_adapter.md)

## Purpose
This adapter defines the explicit, canonical pathways through which geometry modules may import project‑specific refinements from the `local/` layer.

This adapter:
- does not override geometry‑continuous or geometry‑discrete
- does not introduce new geometry primitives
- does not introduce interpretive semantics
- ensures geometry remains upstream‑owned

---

## 1. Importable Local Elements

### 1.1 Invariants
Geometry modules may import:
- adjacency refinements
- containment refinements
- geometry constraints

### 1.2 Constraint Files
Geometry modules may import from:
- `constraints/adjacency_refinements.md`
- `constraints/containment_refinements.md`
- `constraints/geometry_constraints.md`

### 1.3 Prohibited Imports
- no time constraints
- no physics‑layer primitives
- no physics‑layer invariants

---

## 2. Adapter Rules

### 2.1 Allowed Transformations
- explicit mapping of local geometry refinements to geometry‑layer structures
- explicit propagation of adjacency‑safe geometry labels

### 2.2 Forbidden Transformations
- no metric inference
- no geometry reinterpretation
- no hidden geometry state
- no downstream geometry ownership claims

---

## 3. Downstream Usage

### 3.1 Geometry Modules
Geometry modules may:
- import only through this adapter
- use only declared refinements
- preserve upstream geometry ownership

### 3.2 Ownership Boundaries
- geometry modules may not modify local invariants
- geometry modules may not modify constraint files
- geometry modules may not introduce new geometry primitives

---

# RCP‑1.1 Compliance
- All mappings explicit
- No interpretive semantics
- No hidden geometry
- Ready for downstream usage
