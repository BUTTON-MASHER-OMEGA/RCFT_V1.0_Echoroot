### rcmt/global/invariants/winding.md

#### Winding as a global invariant

##### 1. Purpose
This document defines **winding** as a **global invariant** in RCMT: a persistent non-contractibility of an adjacency structure that remains after all local simplification or coherence repair has been applied.

Winding is not a property of a single link or local patch. It is a property of the **entire closed adjacency structure**.

---

##### 2. Local simplicity, global entanglement

Winding has the following pattern:

- **Locally:**
  Any small neighborhood may appear simple, flat, or unlinked under local operators.

- **Globally:**
  The full structure cannot be reduced to a trivial loop without global reconfiguration.

That residual non-contractibility is the **winding**.

---

##### 3. Relation to LCO

LCO can simplify local adjacency and remove local irregularity. It cannot remove winding if the linkage is globally entangled.

If a loop or boundary still cannot be reduced to a trivial adjacency cycle after local cleanup, that residual structure is classified as **winding**.

---

##### 4. Distinction from other invariants

Winding is about **non-contractible linkage or topology**.

- **Twist** concerns orientation mismatch.
- **Holonomy** concerns accumulated transformation under transport.

All three are global, LCO-resistant, and not localizable to a single patch.

---

##### 5. Summary

- Winding is a **global invariant**: locally trivial, globally entangled.
- LCO cannot remove winding; it only simplifies local adjacency.
- Winding belongs to the **entire closed adjacency structure**, not to any of its parts.
