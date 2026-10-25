### rcmt/global/invariants/twist.md

#### Twist as a global invariant

##### 1. Purpose
This document defines **twist** as a **global invariant** in RCMT: a persistent mismatch in orientation or frame transport that remains after all local pruning or coherence repair has been applied.

Twist is not a property of a single unit or local patch. It is a property of the **entire loop or boundary**.

---

##### 2. Local flatness, global mismatch

Twist has the following pattern:

- **Locally:**
  Any small neighborhood along the loop can be made to look regular, flat, or untwisted by local operators.

- **Globally:**
  When a direction or frame is transported once around the loop, it fails to return to its original state.

That residual mismatch is the **twist**.

---

##### 3. Relation to LCO

LCO can simplify local irregularity and make each patch look coherent. It cannot remove twist if the mismatch is encoded in the full closed structure.

If a loop still exhibits a net transport mismatch after local cleanup, that mismatch is classified as **twist**.

---

##### 4. Distinction from other invariants

Twist is about **orientation mismatch**.

- **Winding** concerns non-contractible linkage or topology.
- **Holonomy** concerns accumulated transformation under transport.

All three are global, LCO-resistant, and not localizable to a single patch.

---

##### 5. Summary

- Twist is a **global invariant**: locally trivial, globally nontrivial.
- LCO cannot remove twist; it only makes the local picture look regular.
- Twist belongs to the **entire loop or boundary**, not to any of its parts.