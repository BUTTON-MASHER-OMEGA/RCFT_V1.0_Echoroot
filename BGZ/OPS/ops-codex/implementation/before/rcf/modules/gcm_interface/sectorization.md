# gcm_interface/sectorization.md

## Document Contract

- owner: RCF GCM interface
- type: Interpretation
- scope: module-local content of `gcm_interface/sectorization.md`
- authority: tier-8 downstream bridge contract
- upstream dependencies: `/modules/gcm_interface/representation_contract.md`
- status: active
- purpose: expose survivor and shadow descriptors without adding geometry, metrics, dynamics, or physical interpretation
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

### *Sectorization Input from Upstream Evaluations*

## Overview
This document records the **sectorization structure provided by upstream evaluations** of the 600→120 projection under LCO constraints.
No physical interpretation is assumed.
The purpose is to describe the **structural partitions** that downstream modules may reference.

## 1. Starting Object: The 600‑Cell
The upstream geometry stack evaluates the 600‑cell, a 4‑dimensional regular polytope with:

- 120 vertices
- 720 edges
- 1200 triangular faces
- 600 tetrahedral cells

It is used because its local combinatorial structure supports the triangle‑packing constraints required by the LCO.

## 2. Projection to the 120‑Cell
Upstream modules evaluate **one projection route** from the 600‑cell to the 120‑cell that satisfies LCO compatibility.

This projection preserves:

- adjacency relations
- relevant combinatorial structure

Metric or angle preservation is **not** assumed.

The 120-cell exhibits reduced symmetry under this mapping, as observed in upstream evaluations.

## 3. Triangle‑Packing Compatibility
The LCO requires that the projected structure support a consistent triangle‑packing configuration.
This eliminates:

- degenerate projections
- incompatible face adjacencies
- locally incoherent configurations

The projection route used here is simply the one that **remains LCO‑compatible in upstream evaluations**.

## 4. Sectorization (Upstream Input)
Upstream evaluations currently provide a **36‑sector partition** of the projected structure.

These sectors are:

- structural partitions
- stable under LCO pruning
- non‑metric
- non‑dynamical
- not interpreted as fields or modes

This module **does not derive** the number 36; it receives it as **sectorization input**.

## 5. Role in Downstream Modules
Downstream modules may use these sectors as:

- indexing structures
- stable partitions
- organizational references

They are **not** tunable parameters and **not** physical assumptions.
They are simply the **current upstream sectorization input**.

