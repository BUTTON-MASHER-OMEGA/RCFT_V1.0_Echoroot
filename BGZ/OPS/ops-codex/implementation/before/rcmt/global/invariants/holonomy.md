### rcmt/global/invariants/holonomy.md

#### Holonomy as a global invariant

##### 1. Purpose
This document defines **holonomy** as a **global invariant** in RCMT: a persistent net transformation of a transported frame, mode-vector, or relational direction after completing a closed circuit.

Holonomy is not a property of any single local patch. It is a property of the **entire closed relational path**.

---

##### 2. Local coherence, global accumulation

Holonomy has the following pattern:

- **Locally:**
  Each small neighborhood may appear coherent, regular, or untwisted under local operators.

- **Globally:**
  After transport around a closed path, the final state differs from the initial state by a net transformation that cannot be removed by local cleanup alone.

That accumulated transformation is the **holonomy**.

---

##### 3. Relation to LCO

LCO can enforce local coherence and make each patch appear trivial. It cannot remove holonomy if the accumulated transformation is encoded in the full circuit.

If a transported structure still exhibits a net relational transformation after local pruning, that transformation is classified as **holonomy**.

---

##### 4. Distinction from other invariants

Holonomy is about **accumulated transformation under transport**.

- **Twist** concerns orientation mismatch.
- **Winding** concerns non-contractible linkage or topology.

All three are global, LCO-resistant, and not localizable to a single patch.

---

##### 5. Summary

- Holonomy is a **global invariant**: locally trivial, globally accumulated.
- LCO cannot remove holonomy; it only enforces local coherence.
- Holonomy belongs to the **entire closed relational path**, not to any of its parts.
