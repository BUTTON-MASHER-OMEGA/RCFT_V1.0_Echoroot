# gcm_interface/survivor_indexing.md

## Document Contract

- owner: RCF GCM interface
- type: Interpretation
- scope: module-local content of `gcm_interface/survivor_indexing.md`
- authority: tier-8 downstream bridge contract
- upstream dependencies: `/modules/gcm_interface/representation_contract.md`
- status: active
- purpose: expose survivor and shadow descriptors without adding geometry, metrics, dynamics, or physical interpretation
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

### *Indexing Schemes for Survivor‑Level Structure (Upstream‑Provided)*

## Overview
This document records the **indexing structures** that downstream modules may use to reference survivor‑level geometry.
These indexing schemes are **provided by upstream modules** and are not derived or modified here.

All indexing is:

- structural
- non‑metric
- non‑dynamical
- non‑interpretive
- scoped to evaluated survivors

This module does not define geometry; it exposes **stable identifiers** for downstream consumption.

---

## 1. Chromatic‑12 Indexing
Upstream evaluations provide a **12‑element chromatic indexing scheme** that partitions survivor structure into twelve repeating components.

This indexing:

- originates in geometry‑adjacent
- is stable under projection
- repeats across γ‑plateaus
- is used as a base addressing scheme

Downstream modules may treat Chromatic‑12 as a **structural label set**, not as a physical or dynamical object.

---

## 2. γ‑Plateau Indexing
Survivor geometries exhibit **three broad γ‑plateau regions** in upstream evaluations.

This module receives:

- plateau identifiers
- plateau ordering
- plateau‑level structural tags

These plateaus are **organizational descriptors**, not dynamical phases or physical regimes.

---

## 3. Mode Indexing (36‑Mode Structure)
Upstream evaluations currently provide a **36‑mode structure** arising from the repetition of Chromatic‑12 across the γ‑plateaus.

This module receives:

- mode identifiers (1–36)
- mode grouping by plateau
- mode grouping by chromatic component

These mode labels are **indexing tools**, not physical modes or fields.

---

## 4. Sector Identifiers (36‑Sector Partition)
The `sectorization.md` file provides the upstream **36‑sector partition** of the projected geometry.

This file records:

- sector IDs
- sector adjacency relations
- sector‑to‑mode correspondences (if provided upstream)

Sectors are **structural partitions**, not dynamical or physical entities.

---

## 5. Survivor‑Level Tags
Upstream modules may attach additional structural tags to survivors, such as:

- adjacency‑class identifiers
- symmetry‑class identifiers
- projection‑stable labels
- LCO‑stable descriptors

These tags are **opaque identifiers** from the perspective of this module.
Downstream modules may reference them but must not reinterpret them.

---

## Scope
All indexing schemes described here are:

> **provided by upstream evaluations of survivor geometries under the current RCF formalism.**

This module:

- does not define new indexing
- does not modify upstream labels
- does not interpret labels physically
- does not introduce dynamics or metrics
- does not assert universality

It exposes **stable identifiers** for downstream modules to use.

