### rcmt/global/README.md

#### Global invariants (RCMT global)

##### 1. Purpose
This document defines **global invariants** in RCMT: properties of configurations that **cannot be altered by any purely local operation**.

Local operators (like LCO) may change representations, smooth mismatch, or prune noise, but **global invariants persist** unless a specifically global transformation acts on them.

This file does **not** enumerate all possible invariants; it defines the category they belong to.

---

##### 2. Local vs global

**Local properties:**

- **Definition:** Depend only on a neighborhood of a unit or small region.
- **Examples:** local direction, local coherence, local mismatch, local curvature.
- **Behavior:** Can be modified by local operators (e.g., LCO, local drift, local projection).

**Global invariants:**

- **Definition:** Depend on the structure of **entire loops, regions, or embeddings**, not any single local patch.
- **Examples (informal):** winding number, holonomy, global twist of a boundary, topological class of a basin.
- **Behavior:** Cannot be removed or altered by any finite sequence of purely local pruning/smoothing operations.

---

##### 3. Persistence under local operations

A property qualifies as a **global invariant** if:

1.
   \[
   \text{All local neighborhoods can be made locally “flat” or “regular” by LCO or similar operators}
   \]
   **and yet**
2.
   \[
   \text{Some global measurement (around a loop / over a region) still detects a nontrivial structure.}
   \]

Intuitively:

- **Locally:** everything can look ordinary.
- **Globally:** something refuses to go away.

Twist on a closed boundary is the canonical example:
no matter how LCO prunes locally, a loop around the boundary still “comes back rotated.”

---

##### 4. Examples (non‑binding, illustrative)

These are **illustrative**, not required by upstream:

- **Boundary twist:**
  A closed boundary whose transported direction fails to return to its original orientation after a full loop.

- **Winding / linking:**
  Configurations whose “entanglement” cannot be removed by local moves without cutting or global reconfiguration.

- **Topological basin class:**
  Basins that differ not by local shape but by global connectivity or wrapping.

These examples motivate the category; they do not constrain future implementations.

---

##### 5. Architectural role

Global invariants:

- **Constrain what local operators can achieve.**
  LCO can smooth, align, and prune—but cannot remove global invariants.

- **Define “locked” structure.**
  Regions with nontrivial global invariants behave as **structurally committed**: they cannot be simplified away by local cleanup.

- **Provide hooks for downstream physics‑like behavior.**
  Black hole boundaries, quantized twist, conserved “charges,” and other robust structures naturally live here.

---

##### 6. Summary

- **Local operators** act on neighborhoods.
- **Global invariants** live on loops, regions, or embeddings.
- If something **persists no matter how hard LCO prunes locally**, it belongs here.
