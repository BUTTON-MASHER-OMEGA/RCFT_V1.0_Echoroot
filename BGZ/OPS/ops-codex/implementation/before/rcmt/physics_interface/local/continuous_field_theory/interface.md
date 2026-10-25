# Interface: continuous_field_theory

## Purpose
Defines the explicit interface for constructing and updating continuous fields.

---

## Inputs

### 1. Discrete Field Inputs
- field support
- field assignment record
- field update record

### 2. Temporal Inputs
- physics‑time index

---

## Outputs

### 1. Continuous Field Support
**Continuous support \(X\)** constructed locally by the lift operator.
\(X\) is a bare set with no coordinates, no metric, and no charts.

### 2. Continuous Field Assignment Record
Mapping:

**continuous support element → field value**
(encodes the lifted field \(f : X \to V\))


### 3. Continuous Field Update Record
Describes admissible continuous updates.

---

## Allowed Transformations
- explicit lifting from discrete to continuous
- explicit continuous assignment
- explicit continuous updates

---

## Forbidden Transformations
- no PDEs
- no metric usage
- no hidden interpolation
- no action principles

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden state
