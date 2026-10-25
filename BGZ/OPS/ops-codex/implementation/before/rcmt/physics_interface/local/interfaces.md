# Local Interfaces (local/interfaces.md)

## Purpose
This file defines the public interfaces exposed by the project‑specific upstream layer (`local/`).
Downstream modules may access local‑layer content only through this interface and the adapters it exposes.
These interfaces provide stable, explicit access points to project‑specific invariants, primitives, and constraints.

This file:
- does not redefine RCF core interfaces
- does not introduce interpretive semantics
- does not introduce hidden state
- exposes only explicit, adjacency‑safe structures
- prevents downstream modules from accessing upstream internals

---

## 1. Interface Categories

The local layer exposes interfaces in four categories:

1. **Invariant Interfaces**
   - access to project‑specific invariants
   - read‑only
   - no downstream modification permitted

2. **Primitive Interfaces**
   - access to project‑specific primitives declared in `local/primitives.md`
   - explicit, typed, adjacency‑safe

3. **Constraint Interfaces**
   - access to structured constraint files in `local/constraints/`
   - adjacency, containment, time, and geometry refinements

4. **Adapter Interfaces**
   - controlled mappings from local → downstream layers
   - defined in `local/adapters/`

Downstream modules may import only from these categories.

---

## 2. Invariant Interface

### 2.1 Exposed Elements
- project‑specific adjacency refinements
- project‑specific containment refinements
- project‑specific time constraints
- project‑specific geometry constraints

### 2.2 Access Rules
- invariants are read‑only
- invariants may not be shadowed or redefined downstream
- invariants must be loaded before any physics module

---

## 3. Primitive Interface

### 3.1 Exposed Elements
Primitives declared in `local/primitives.md`, including:

- discrete support primitives
- continuous shadow primitives
- temporal primitives
- conversion primitives

### 3.2 Access Rules
- primitives must be explicitly declared
- no implicit primitive creation is allowed
- downstream modules may not introduce new primitives

---

## 4. Constraint Interface

### 4.1 Exposed Constraint Files
Downstream modules may import from:

- `constraints/adjacency_refinements.md`
- `constraints/containment_refinements.md`
- `constraints/time_constraints.md`
- `constraints/geometry_constraints.md`
- `constraints/group_structure_constraints.md`

### 4.2 Access Rules
- constraints refine RCF core behavior
- constraints must remain adjacency‑safe
- constraints may not introduce hidden state
- constraints may not override upstream geometry or time ownership

---

## 5. Adapter Interface

### 5.1 Exposed Adapter Files
- `adapters/local_to_physics_adapter.md`
- `adapters/local_to_geometry_adapter.md`

### 5.2 Purpose
Adapters define:
- how local invariants map into downstream physics modules
- how local geometry refinements map into geometry modules
- how primitives flow across boundaries

### 5.3 Access Rules
- adapters may not introduce new invariants
- adapters may not bypass local constraints
- adapters may not introduce interpretive semantics

---

## 6. Import Boundary

### 6.1 Allowed Imports
Downstream modules may import only from:
- RCF core
- this file (`local/interfaces.md`)
- their declared dependencies

### 6.2 Forbidden Imports
- no sibling module imports
- no direct imports from `local/` internals
- no undeclared upstream imports
- no bypassing the adapter layer

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden state
- No downstream contamination
- Ready for downstream usage
