# SMGv2.1 GCM Bindings

## Purpose

This file defines the read-only upstream structures SMGv2.1 imports from `gcm_interface`.

This is a binding file.
It contains no SMGv2.1 thesis claims and no downstream policy.

---

## 1. Chromatic-12 Binding

- upstream source: `gcm_interface/survivor_indexing.md`
- imported structure: 12-element base mode index
- local binding name: `BaseMode_Index`
- local binding form: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`
- owner: `gcm_interface`
- directionality: SMGv2.1 consumes this index read-only
- structural constraint: no gauge interpretation is imported by this binding

---

## 2. Gamma-Plateau Binding

- upstream source: `gcm_interface/survivor_indexing.md`
- imported structure: ordered three-region survivorship threshold record
- local binding name: `Gamma_Plateaus`
- local binding form: `[A, B, C]`
- owner: `gcm_interface`
- directionality: SMGv2.1 consumes this plateau record read-only
- structural constraint: no downstream generation semantics are imported by this binding

---

## 3. Survivor-Mode Binding

- upstream source: `gcm_interface/mode_structure.md`
- imported structure: 36-element composite index set
- local binding name: `Survivor_Modes`
- local binding form: `BaseMode_Index x Gamma_Plateaus`
- owner: `gcm_interface`
- directionality: SMGv2.1 consumes this mode set read-only
- structural constraint: no particle labels are imported by this binding

---

## 4. Sector and Adjacency Binding

- upstream source: `gcm_interface/sectorization.md`
- imported structure: sector partition map and non-metric adjacency graph
- local binding names:
  - `Sector_Map`
  - `Adjacency_Graph = G(V,E)`
- owner: `gcm_interface`
- directionality: SMGv2.1 consumes these descriptors read-only
- structural constraint: adjacency remains non-metric and upstream-owned

---

## Directionality Clause

SMGv2.1 may use these bindings as inputs for claims, dictionaries, translations, policies, and exports.
No output from SMGv2.1 may redefine or repair the upstream GCM records.

---

## RCP-1.1 Compatibility

- upstream owner explicit
- local binding names explicit
- no physical interpretation imported
- no downstream policy declared
