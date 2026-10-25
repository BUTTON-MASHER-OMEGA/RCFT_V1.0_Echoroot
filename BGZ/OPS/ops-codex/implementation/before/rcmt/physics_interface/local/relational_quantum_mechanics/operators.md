# Operators: relational_quantum_mechanics

## Purpose
Defines the canonical operators used to assign and update relational amplitudes.
These operators are adjacency‑safe, explicit, and non‑interpretive.

---

## 1. `assign_amplitude`
Assigns an amplitude value to an adjacency‑indexed support.

**Inputs:**
- adjacency index
- local invariant record
- physics‑time index (optional)

**Outputs:**
- amplitude assignment record

**Rules:**
- must be explicit
- must not introduce probability semantics
- must not use Hilbert spaces

---

## 2. `update_amplitude`
Updates an amplitude value under admissible propagation.

**Inputs:**
- amplitude assignment
- adjacency neighborhood
- local invariant records (current + neighbors)
- physics-time index (optional)

**Outputs:**
- amplitude update record

**Rules:**
- propagation must be adjacency‑local
- no continuous interpolation


---

## 3. `quantum_boundary`
Constructs the quantum‑compatibility boundary.

**Inputs:**
- amplitude update record
- geometry constraints

**Outputs:**
- quantum‑compatibility boundary

**Rules:**
- must not reinterpret geometry
- must not introduce hidden state

---

# RCP‑1.1 Compliance
- All operators canonical
- No interpretive semantics
- No hidden state
