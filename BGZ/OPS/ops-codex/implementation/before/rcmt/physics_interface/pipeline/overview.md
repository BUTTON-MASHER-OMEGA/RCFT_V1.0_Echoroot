### `rcmt/pipeline/overview.md`

---

# **RCMT Pipeline Overview (v1.0)**

This document defines the **end‑to‑end pipeline** of RCMT:
how upstream relational structure becomes a coherent state, how local operators act, and how geometric interpretation arises only at the projection layer.

This file does **not** define ontology.
It specifies the **flow of operations** and the **boundaries between layers**.

---

## **1. Pipeline summary**

The RCMT pipeline consists of the following stages:

1. **Upstream substrate**
2. **Local Coherence Operations (LCO)**
3. **Coherent relational state**
4. **Projection layer**
5. **Projected geometric frame**
6. **Global invariants (twist, holonomy, etc.)**

Each stage has strict responsibilities and strict limits.

---

## **2. Stage 1 — Upstream substrate**

The upstream substrate provides:

- units
- directions
- coherence relations
- basin structure
- the **Upstream Time Index** (ordering only)

It explicitly **does not** contain:

- geometry
- metric
- Lorentz structure
- time as duration
- twist
- holonomy
- curvature
- spinorial structure

This stage is **purely relational**.

---

## **3. Stage 2 — Local Coherence Operations (LCO)**

LCO acts **only locally**.

It performs:

- mismatch pruning
- direction alignment
- local smoothing
- basin stabilization
- coherence enforcement

LCO **cannot**:

- create geometry
- create time
- create twist
- remove global invariants
- alter topology
- introduce Lorentz structure

LCO produces a **locally coherent state**, not a geometric one.

---

## **4. Stage 3 — Coherent relational state**

After LCO, the system is:

- locally consistent
- directionally aligned
- basin‑stable
- free of local mismatch

But still:

- geometry‑free
- time‑free
- twist‑free
- Lorentz‑free

This is the **input** to the projection layer.

---

## **5. Stage 4 — Projection layer**

Projection is the **first non‑local, representational operation**.

It introduces:

- geometric interpretation
- metric structure
- Lorentz signature
- causal cones
- time as a geometric coordinate
- the ability to detect global invariants

Projection **does not modify** the upstream state.
It **interprets** it.

This is where the dual‑lane symmetry chain attaches:

- **Mode‑space lane:** \(Sp(8)\)
- **Geometric lane:** \(SO(9) \to SO(5) \times SO(3,1)\)

Time appears **only here**, never upstream.

---

## **6. Stage 5 — Projected geometric frame**

After projection, the system has:

- a Lorentzian geometric lane
- a compact internal symmetry
- a mode‑space lane that remains unbroken
- a metric
- causal structure
- geometric time
- curvature (local only)

This is the first stage where physics‑like behavior becomes meaningful.

---

## **7. Stage 6 — Global invariants**

Global invariants (e.g., twist) become **detectable** only after projection.

They are:

- locally trivial
- globally persistent
- immune to LCO
- encoded in loops, boundaries, or embeddings

Examples:

- **twist**
- **holonomy**
- **winding class**

These invariants belong to the **projected frame**, not the substrate.

---

## **8. Summary**

The RCMT pipeline is:

> **Upstream substrate → LCO → Coherent state → Projection → Geometric frame → Global invariants**

Upstream computes.
Projection interprets.
Geometry is downstream.
Time is downstream.
Twist is downstream.
Global invariants persist because LCO is local.
