# Adjacency Refinements (local/constraints/adjacency_refinements.md)

## Purpose
This file defines project‑specific refinements to RCF adjacency rules.
These refinements constrain how downstream modules may construct, modify, or propagate adjacency relations.

These refinements:
- do not override RCF core adjacency axioms
- do not introduce non‑local adjacency
- do not introduce interpretive semantics
- remain explicit and canonical

---

## 1. Allowed Adjacency Refinements

### 1.1 Local Adjacency Extensions
- adjacency may be extended only through explicit local rules
- adjacency transitions must preserve discrete structure
- adjacency updates must remain bounded by local operators

### 1.2 Adjacency‑Safe Transformations
- adjacency may be reindexed if the transformation is explicit
- adjacency may be relabeled if labels remain non‑interpretive

---

## 2. Forbidden Adjacency Forms

### 2.1 Non‑Local Adjacency
- no global adjacency operators
- no adjacency shortcuts
- no adjacency inference from non‑adjacent data

### 2.2 Hidden Adjacency
- no implicit adjacency creation
- no adjacency derived from continuous geometry
- no adjacency derived from metric structure

---

## 3. Downstream Usage Rules

### 3.1 Physics Layer
- physics modules must use adjacency‑only propagation
- no physics module may introduce new adjacency types

### 3.2 Geometry Layer
- geometry modules may not redefine adjacency ownership
- adjacency must remain upstream‑compatible

---

# RCP‑1.1 Compliance
- All refinements explicit
- No interpretive semantics
- No hidden adjacency
- Ready for downstream usage
