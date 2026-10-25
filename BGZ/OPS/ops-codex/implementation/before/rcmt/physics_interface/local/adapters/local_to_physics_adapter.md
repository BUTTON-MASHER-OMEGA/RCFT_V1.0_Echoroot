# Local → Physics Adapter (local/adapters/local_to_physics_adapter.md)

## Purpose
This adapter defines the explicit, canonical pathways through which downstream physics modules may import project‑specific invariants, primitives, and constraints from the `local/` layer.

This adapter:
- does not introduce new invariants
- does not introduce interpretive semantics
- does not bypass local constraints
- ensures downstream modules remain adjacency‑safe

---

## 1. Importable Local Elements

### 1.1 Invariants
Physics modules may import:
- adjacency refinements
- containment refinements
- time constraints
- geometry constraints

### 1.2 Primitives
Physics modules may import primitives declared in:
- `local/primitives.md`

### 1.3 Constraint Files
Physics modules may import from:
- `constraints/adjacency_refinements.md`
- `constraints/containment_refinements.md`
- `constraints/time_constraints.md`
- `constraints/geometry_constraints.md`

---

## 2. Adapter Rules

### 2.1 Allowed Transformations
- explicit mapping of local primitives to physics‑layer primitives
- explicit mapping of local invariants to physics‑layer constraints
- explicit propagation of adjacency‑safe structures

### 2.2 Forbidden Transformations
- no implicit primitive creation
- no hidden state introduction
- no interpretive mapping
- no bypassing of local invariants

---

## 3. Downstream Usage

### 3.1 Physics Modules
Physics modules may:
- import only through this adapter
- use only declared primitives
- use only declared invariants
- use only adjacency‑safe propagation

### 3.2 Ownership Boundaries
- physics modules may not modify local invariants
- physics modules may not modify local primitives
- physics modules may not modify constraint files

---

# RCP‑1.1 Compliance
- All mappings explicit
- No interpretive semantics
- No hidden state
- Ready for downstream usage
