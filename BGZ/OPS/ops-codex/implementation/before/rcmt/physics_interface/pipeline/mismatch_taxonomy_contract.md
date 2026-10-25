# **Mismatch Taxonomy Contract (MTC‑1.0)**
*(RCMT → RCF‑compatible, RCP‑typed)*

This contract defines the **canonical mismatch categories** used by RCMT.
These categories are upstream‑safe, projection‑neutral, and required for all downstream interpretation rules.

The taxonomy ensures that mismatch behavior is described **without** implying fields, forces, particles, or mechanisms.

---

## **1. Purpose**

> **Define the canonical categories of mismatch used by RCMT for compression, projection, and downstream interpretation.**

**Type:** Purpose
**Owner:** RCMT upstream mismatch module

---

## **2. Ownership**

The mismatch module owns:

- the classification of mismatch
- the distinction between free, locked, and structured mismatch
- the upstream semantics of mismatch propagation

The mismatch module does **not** own:

- geometric interpretation
- field‑like interpretation
- curvature‑like interpretation
- projection smoothing

**Type:** Invariant

---

## **3. Canonical Categories**

### **free mismatch**
Unbound mismatch that propagates across adjacency without forming a survivor.

- does not bind
- does not stabilize
- travels
- projection may interpret as field‑like or photon‑like

**Type:** Invariant (category definition)

---

### **locked mismatch**
Mismatch that binds into a stable survivor pattern.

- contributes to matter‑like structures
- does not propagate freely
- projection may interpret as mass, charge, spin, etc.

**Type:** Invariant (category definition)

---

### **structured mismatch**
Semi‑bound mismatch patterns that are neither fully free nor fully locked.

- pattern‑bearing
- adjacency‑sensitive
- projection may interpret as interaction‑like or field‑adjacent behavior

**Type:** Invariant (category definition)

---

## **4. Allowed Inputs**

The mismatch module consumes:

- adjacency relations
- basin structure
- coherence relations from LCO

**Type:** Dependency

---

## **5. Allowed Outputs**

The mismatch module exposes:

- mismatch category labels
- propagation patterns (upstream semantics only)
- binding vs non‑binding behavior

These outputs are **not** geometric or field‑like.

**Type:** Interface Output

---

## **6. Prohibitions**

Mismatch classification must not:

- imply fields
- imply forces
- imply continuous manifolds
- imply geometric propagation
- imply downstream interpretation
- imply mechanism or causality

**Type:** Prohibition

---

## **7. Scope Binding**

All mismatch statements must be scoped:

- “within the present formalism”
- “as upstream mismatch behavior”
- “prior to projection”
- “in the mismatch module”

This prevents downstream leakage.

**Type:** Contract

---

## **8. Observations (Current Evaluations)**

> **In current evaluations, free mismatch propagates across adjacency, locked mismatch stabilizes into survivors, and structured mismatch forms intermediate patterns.**

This is observational, not canonical.

**Type:** Observation

---

## **9. Status**

- Contract: **DRAFT PROPOSED**
- Owner: **RCMT mismatch module**
- Applicability: **All RCMT modules**
- Scope: **Upstream mismatch classification**

---

# The one‑liner
**RCMT mismatch comes in three canonical forms — free, locked, and structured — and projection interprets them downstream without redefining them upstream.**
