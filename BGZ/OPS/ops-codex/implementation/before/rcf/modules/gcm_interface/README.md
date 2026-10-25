# gcm_interface

## Document Contract

- owner: RCF GCM interface
- type: Status
- scope: module-local content of `gcm_interface/README.md`
- authority: tier-8 downstream bridge contract
- upstream dependencies: `/modules/gcm_interface/representation_contract.md`
- status: active
- purpose: expose survivor and shadow descriptors without adding geometry, metrics, dynamics, or physical interpretation
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

### *Bridge Layer Between Continuous Geometry and Downstream Modeling*

The **gcm_interface** module defines the **structural contract** that downstream modules use to consume the continuous relational geometry produced by the RCF pipeline.

It does **not** introduce new geometry, dynamics, metrics, or physical interpretation.
Instead, it provides a **stable interface layer** that organizes survivor‑level structure into forms that downstream modules can reference without modifying upstream behavior.

This module is analogous to an API boundary in software engineering:
it exposes **what downstream modules may rely on**, while keeping the geometry stack clean, non‑interpretive, and upstream‑safe.

---

## **Role of the Module**

The gcm_interface layer:

- provides a **representation contract** for survivor geometries
- exposes **sectorization**, **indexing**, and **structural partitions** derived from upstream modules
- defines **downstream‑consumable descriptors** without asserting physical meaning
- organizes continuous geometry into **stable, non‑metric, non‑dynamical structures**
- ensures downstream modules can operate without depending on internal geometry implementation details

This module is **bridging, not generative**:

- it does not define geometry
- it does not modify projection
- it does not introduce metrics or curvature
- it does not define flow or dynamics
- it does not interpret structures physically

It simply **exposes** the structures that downstream modules may use.

---

## **What This Module Contains**

### **1. Sectorization (Structural Partitioning)**
A description of the **current upstream sectorization input**, including:

- the 36‑sector partition observed in evaluated survivors
- the mapping between Chromatic‑12 repetition and sector structure
- non‑metric, non‑dynamical sector boundaries
- survivor‑level structural tags

This is **descriptive**, not canonical.

### **2. Representation Contract**
Downstream‑safe descriptors such as:

- survivor indexing
- plateau/mode indexing
- structural adjacency patterns
- non‑metric relational descriptors

These provide a stable interface for downstream modules without implying physical fields or dynamics.

### **3. Survivor‑Level Indexing Schemes**
Organizational structures derived from upstream modules, including:

- Chromatic‑12 indexing
- γ‑plateau indexing
- mode indexing
- sector identifiers

These are **representational**, not generative.

### **4. Downstream Hooks (Non‑Interpretive)**
Optional structures that downstream modules may reference, such as:

- structural slots
- adjacency‑based grouping
- survivor‑level partitions

These hooks do **not** define meaning or behavior; they simply expose structure.

---

## **What This Module Does *Not* Contain**

This module intentionally excludes:

- metric definitions or distance functions
- curvature, geodesics, or differential structure
- flow laws or dynamical propagation
- physical fields, masses, couplings, or SMGv2 interpretation
- invariant definitions (owned by LCO/core)
- projection rules (owned by geometry/projection)
- geometry generation or modification

All content is **conditional on the current RCF formalism and evaluated survivors**.

---

## **Position in the Architecture**

```text
geometry-discrete
        ↓
projection (Π↑ / Π↓)
        ↓
geometry-continuous
        ↓
gcm_interface   ← (this module)
        ↓
geometry-dynamics / metric / domain-models
```

The gcm_interface module acts as a **boundary layer**, ensuring that downstream modules can consume structural information without depending on internal geometry implementation details.

---

## **Scope**

All statements in this module describe:

> **representational structures derived from evaluated survivor geometries under the current RCF formalism**

They do **not**:

- define new geometry
- assert physical meaning
- introduce dynamics or metrics
- claim universality
- modify upstream contracts

This module provides **a stable interface**, not a generative layer.

