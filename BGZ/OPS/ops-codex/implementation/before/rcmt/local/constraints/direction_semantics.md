# RCMT Local Direction Semantics

## Document Contract

- owner: RCMT
- type: Contract
- scope: module-local semantic refinement
- depends_on: upstream direction-bearing unit contract
- status: current RCMT interpretation; replaceable without changing upstream

### **Purpose**

This document specifies how RCMT interprets the *local direction* carried by each unit.
Upstream defines only the invariant:

> **Each unit carries a direction.**

RCMT refines this invariant by committing to a **tangent‑vector interpretation**.
This choice affects drift, projection, basin evolution, and all local flow operators.

---

## **1. Direction as Tangent Vector**

RCMT treats the local direction of a unit as a **tangent-vector-like local descriptor** in its current formalism.

Within this RCMT interpretation, the descriptor records:

- the **local direction of admissible change**,
- the **local flow orientation**,
- the **direction of coherence propagation**,
- a **local ordering of structural resistance**.

This replaces earlier perpendicular/normal interpretations, which implied external correction rather than intrinsic flow.

Readers unfamiliar with tangent‑vector intuition may explore:
**tangent vectors**,
**flow fields**,
**directional derivatives**.

---

## **2. Consequences for Local Dynamics**

### **2.1 Drift Operators**
Drift must evolve units **along** the tangent direction, not perpendicular to it.
This ensures drift reflects *natural coherence flow* rather than corrective pressure.

See: **projection drift**.

---

### **2.2 Basin Evolution**
Within the current RCMT implementation, basin flow may be represented by integral curves of the tangent field.
A basin is not a region “pushed into shape” by normals; it is a region **shaped by flow**.

See: **basin flow**.

---

### **2.3 Containment**
RCMT containment representations must preserve **tangent-vector compatibility** across boundaries.
A discontinuity in tangent direction represents a structural break and must be handled explicitly.

See: **coherence propagation**.

---

## **3. Representational Neutrality**

This constraint does **not** prescribe:

- how the tangent vector is visualized,
- how the manifold is parameterized,
- how the direction is stored or encoded.

It only specifies the **semantic role** of the direction within RCMT.

Upstream remains representation‑agnostic.

---

## **4. Rationale**

The tangent‑vector interpretation aligns RCMT with:

- intrinsic flow rather than external correction,
- geodesic‑like evolution rather than force‑based shaping,
- local coherence propagation rather than normal‑vector pressure.

This choice enables consistent behavior across:

- drift modules,
- basin formation,
- projection operators,
- containment logic,
- future geometry‑adjacent extensions.

---

## **5. Summary**

- Upstream: “each unit carries a direction.”
- RCMT: “that direction is represented by a **tangent-vector-like descriptor** for local flow.”
- All local dynamics must treat direction as **intrinsic**, **continuous**, and **flow‑generating**.
