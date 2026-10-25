# RCMT Symmetry Chain

## Document Contract

- owner: RCMT
- type: Contract
- scope: current symmetry implementation
- depends_on: RCF core constraints and `../invariants.md`
- status: provisional survivor implementation
- replacement_rule: any replacement must preserve the required RCMT dual-lane contract

This document specifies the **symmetry structure** assumed by Relational Coherence Mode Theory (RCMT).
It is downstream of the Relational Coherence Framework (RCF) and must remain compatible with the RCMT physics invariants.

This file defines:

- the required **dual‑lane structure**,
- the current **survivor implementation**,
- and the distinction between **requirements** and **replaceable details**.

It does **not** assert ontology; it constrains admissible symmetry representations.

---

## 1. Dual‑lane symmetry requirement

RCMT requires a **finite, adjacency‑compatible dual‑lane symmetry structure** with:

- a **geometric lane**:

  - locally Lorentzian,
  - compatible with \(SO(3,1)\),
  - with compact internal symmetry at low energy,

- a **mode‑space lane**:

  - symplectic (Sp‑type),
  - non‑breaking under all admissible dynamics,
  - serving as the structural domain for mode-structure not present in the geometric lane.

Any admissible symmetry implementation must:

- realize these two lanes,
- keep the mode‑space lane unbroken,
- and remain compatible with all RCMT invariants.

---

## 2. Current survivor implementations (candidates)

The current candidate implementations of the RCMT dual-lane structure are:

\[
Sp(8) \times SO(9)
\]

\[
Sp(8) \times SO(8)
\]

These are **provisional candidate structures**, not established symmetry decompositions, mandatory group embeddings, or core identities of RCMT.

The proposed lane assignments are:

- **Mode-space lane:** \(Sp(8)\), representing the symplectic structure required by the mode-space constraints.
- **Geometric lane:** \(SO(9)\) or \(SO(8)\), representing alternative candidate structures for downstream geometric projection.
- **Low-energy projection target:** A Lorentzian geometric structure compatible with \(SO(3,1)\), together with the required compact internal symmetry, potentially \(SO(4)\).

The two geometric candidates remain subject to comparative constraint testing. Neither is privileged solely by its dimension, subgroup relationships, or compatibility with a previously proposed symmetry chain.

### 2.1 Admissibility requirements

A candidate implementation must:

- Preserve the required separation between the symplectic mode-space lane and the geometric lane.
- Preserve the symplectic structure without requiring upstream symmetry breaking.
- Remain compatible with finite, adjacency-first relational construction.
- Support a well-defined projection interface through which Lorentzian geometry may be represented.
- Permit the required compact internal symmetry at the appropriate downstream interface.
- Respect all upstream RCF invariants and sealed RCMT interfaces.

These requirements are **constraints to be demonstrated**, not properties established merely by naming the candidate groups.

### 2.2 Projection boundary

Neither \(SO(9)\) nor \(SO(8)\) is asserted to contain \(SO(3,1)\) through a conventional compact-group symmetry-breaking chain.

The emergence of Lorentzian structure is a **projection-level requirement** whose mathematical realization remains to be specified and verified.

In particular, the appearance of \(SO(3,1)\) does **not** imply an upstream temporal dimension, background spacetime, or substrate metric.

RCMT consumes the temporal ontology and sealed temporal interfaces documented upstream. This section does not redefine upstream time.

### 2.3 Implementation status

The previously explored chain

\[
E_8 \longrightarrow SO(10) \longrightarrow Sp(8)\times SO(9)
\]

is retained, where documented, as historical exploratory work rather than a current architectural dependency.

Alternative implementations remain admissible if they preserve the required dual-lane structure and satisfy all applicable invariants.

**Selection between candidates remains open pending explicit mathematical and operational verification.**

---

## 3. Required vs. replaceable structure

### 3.1 Required structural features

Any alternative symmetry implementation must preserve:

- **Dual‑lane structure:**
  - one Lorentz‑compatible geometric lane,
  - one non‑breaking symplectic mode‑space lane.

- **Finite, adjacency‑compatible groups:**
  - no non‑compact global curvature groups,
  - no infinite towers or KK‑like extensions.

- **Projection compatibility:**
  - the geometric lane must admit a projection to \(SO(3,1)\) with compact internal symmetry,
  - the mode‑space lane must support a measure structure suitable for induced probabilistic behavior (Born‑like rules).

### 3.2 Replaceable implementation details

The following are **replaceable**, provided all invariants and structural requirements remain satisfied:

- the specific choice of \(E_8\) as the parent group,
- the specific intermediate \(SO(10)\),
- the specific factorization into \(Sp(8) \times SO(9)\),
- the specific choice of \(SO(4)\) as the internal compact group, if an alternative with equivalent properties is found.

Any replacement must:

- preserve dual‑lane structure,
- remain finite and adjacency‑compatible,
- and maintain compatibility with the RCMT invariants file.

---

## 4. Relation to RCMT invariants

This symmetry chain is constrained by, and must remain compatible with, at least:

- **Adjacency‑First Invariant**
  (no background metric; symmetry acts on relational structure)

- **Dual‑Lane Symmetry Invariant**
  (this file refines that invariant)

- **Finite Structure Invariant**
  (no infinite towers, no non‑compact global groups)

- **Mode‑Space Projection Invariant**
  (symplectic lane must support projection into the geometric lane)

- **Local Curvature Constraint**
  (no global curvature groups; curvature must be local and representational)

No symmetry implementation may violate these constraints.

---

## 5. Status and future revisions

- **Status:**
  This symmetry chain is the **current RCMT v1.x implementation**.

- **Provisional nature:**
  If a better‑fit symmetry structure is found that satisfies all RCMT invariants and preserves the dual‑lane architecture, it may replace this chain without modifying the RCF upstream core.

- **Intended use:**
  This file serves as the reference for any RCMT module that depends on symmetry structure (e.g., mode‑space definitions, projection rules, or representation choices).

