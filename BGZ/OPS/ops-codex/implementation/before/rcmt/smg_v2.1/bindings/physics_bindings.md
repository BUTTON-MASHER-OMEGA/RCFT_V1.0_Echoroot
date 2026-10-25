# SMGv2.1 Physics Interface Bindings

## Purpose

This file defines the read-only upstream structures SMGv2.1 imports from `physics_interface`.

This is a binding file.
It contains no SMGv2.1 thesis claims and no downstream enforcement policy.

---

## 1. Support Binding

- upstream source: `physics_interface/conversion_laws/formal_structure.md`
- imported structure:
  - discrete support set
  - continuous support set
- local binding names:
  - `Discrete_Support`
  - `Continuous_Support`
- owner: `physics_interface`
- directionality: SMGv2.1 consumes support records read-only
- structural constraint: support ownership remains upstream

---

## 2. Temporal Boundary Binding

- upstream source: `physics_interface` final boundary payload
- imported structure: temporal boundary record
- local binding name: `B_T`
- owner: `physics_interface`
- directionality: SMGv2.1 consumes the boundary read-only
- structural constraint: no time ontology is imported beyond the exposed boundary record

---

## 3. Metric Boundary Binding

- upstream source: `physics_interface` final boundary payload
- imported structure: metric boundary record
- local binding name: `B_g`
- owner: `physics_interface`
- directionality: SMGv2.1 consumes the boundary read-only
- structural constraint: no curvature policy or physical interpretation is imported by this binding

---

## 4. Amplitude Boundary Binding

- upstream source: `physics_interface` final boundary payload
- imported structure: amplitude boundary record
- local binding name: `B_alpha`
- owner: `physics_interface`
- directionality: SMGv2.1 consumes the boundary read-only
- structural constraint: no probability interpretation is imported by this binding

---

## 5. Final Boundary Binding

- upstream source: `physics_interface` final conversion boundary
- imported structure: final boundary tuple
- local binding name: `B_final = (B_T, B_g, B_alpha)`
- owner: `physics_interface`
- directionality: SMGv2.1 consumes the final boundary read-only
- structural constraint: no downstream invalidation rule is declared in this binding

---

## Directionality Clause

SMGv2.1 may use these boundary records as inputs.
SMGv2.1 may not alter, widen, narrow, or reinterpret `physics_interface` ownership.

---

## RCP-1.1 Compatibility

- upstream owner explicit
- local binding names explicit
- no policy declarations
- no interpretive physics imported
