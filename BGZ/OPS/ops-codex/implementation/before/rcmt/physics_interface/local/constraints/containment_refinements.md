# Containment Refinements (local/constraints/containment_refinements.md)

## Purpose
This file defines project‑specific refinements to RCF containment rules.
Containment describes structural inclusion relationships and must remain explicit and upstream‑compatible.

These refinements:
- do not override RCF core containment axioms
- do not introduce hidden containment
- do not introduce interpretive semantics

---

## 1. Allowed Containment Refinements

### 1.1 Explicit Containment Extensions
- containment may be extended only through explicit declarations
- containment transitions must preserve upstream containment structure

### 1.2 Containment‑Safe Transformations
- containment may be relabeled if labels remain non‑interpretive
- containment may be compressed if no information is lost

---

## 2. Forbidden Containment Forms

### 2.1 Implicit Containment
- no containment inference from adjacency
- no containment inference from geometry
- no containment inference from metric structure

### 2.2 Hidden Containment
- no containment‑based hidden state
- no containment derived from downstream physics

---

## 3. Downstream Usage Rules

### 3.1 Physics Layer
- physics modules may not introduce new containment types
- containment may not encode physical state

### 3.2 Geometry Layer
- geometry modules may not redefine containment ownership

---

# RCP‑1.1 Compliance
- All refinements explicit
- No interpretive semantics
- No hidden containment
- Ready for downstream usage
