### rcmt/global/invariants.md

#### Global invariants (RCMT global)

##### 1. Purpose

This document defines global invariants in RCMT as properties of closed relational structures that persist after all local pruning, smoothing, or coherence repair has been applied.

Local operators such as LCO may make a neighborhood look regular, flat, or coherent, but they cannot remove a genuine global invariant unless a genuinely global transformation acts on the structure.

This file defines the category and its core examples. It does not claim that every possible invariant has been enumerated.

---

##### 2. Local vs. global

Local properties:

- depend only on a neighborhood or small patch,
- can be changed by local operators,
- are usually expressible in terms of local alignment, mismatch, or adjacency.

Global invariants:

- depend on the structure of an entire loop, boundary, region, or embedding,
- remain nontrivial even when every local patch is made to look ordinary,
- cannot be removed by any finite sequence of purely local operations.

In short:

- locally, the structure may appear trivial,
- globally, something remains that cannot be smoothed away.

---

##### 3. Canonical global invariants in RCMT

The three primary examples treated here are:

- **Twist**: a persistent mismatch in transported orientation or frame after a closed circuit.
- **Winding**: a persistent non-contractibility of adjacency structure, linkage, or boundary topology.
- **Holonomy**: a persistent net relational transformation of a transported structure after completing a closed loop.

These are distinct, but they share the same core behavior:
they are globally defined, locally trivial, and resistant to LCO-style cleanup.

---

##### 4. Twist

Twist is a global invariant of a closed structure when a transported direction or frame fails to return to its original state after a full circuit.

Its essential features are:

- locally, each small segment may appear flat or regular,
- globally, the loop as a whole still exhibits a net mismatch,
- this mismatch cannot be removed by local repair alone.

Twist is therefore an invariant of the loop or boundary as a whole, not of any single point or local patch.

---

##### 5. Winding

Winding is a global invariant of a closed adjacency structure when the overall linkage or wrapping cannot be reduced to a trivial loop by local operations.

Its essential features are:

- locally, the structure may look simple or unlinked,
- globally, the full adjacency pattern remains non-contractible,
- the structure resists local simplification because its connectivity is globally entangled.

Winding is therefore an invariant of the relational topology of the whole closed structure, not of any individual link or local section.

---

##### 6. Holonomy

Holonomy is a global invariant of a closed relational path when a transported frame, mode-vector, or relational direction accumulates a net transformation after completion of the circuit.

Its essential features are:

- locally, each patch may appear coherent,
- globally, the transported structure does not return to its initial state,
- the accumulated transformation persists under local cleanup.

Holonomy is therefore an invariant of the accumulated transport behavior around the full loop, not of any single step in the path.

---

##### 7. Distinctions among the three

These invariants are complementary:

- **Twist** concerns orientation mismatch.
- **Winding** concerns non-contractible linkage or topology.
- **Holonomy** concerns accumulated transformation under transport.

They are related by the fact that all three are:

- global,
- non-localizable to a single patch,
- resistant to LCO-style local repair,
- structurally persistent under local cleanup.

They differ in what kind of global structure they preserve.

---

##### 8. Relation to LCO

The Local Coherence Operator can:

- prune local mismatch,
- enforce local regularity,
- simplify local adjacency,
- make small neighborhoods appear ordinary.

But LCO cannot remove a true global invariant if that invariant is encoded in the whole closed structure. If a loop still exhibits nontrivial behavior after local coherence repair, that behavior belongs to the global invariant class.

---

##### 9. Architectural role

Global invariants serve several roles in RCMT:

- they constrain what local operators can achieve,
- they define structurally locked or committed configurations,
- they provide a basis for downstream behavior such as persistent boundary structure, global mismatch, or transport-based accumulation.

They are not merely descriptive labels. They mark the points at which local coherence is insufficient and global structure must be recognized.

---

##### 10. Summary

Global invariants are properties of whole closed structures that survive local repair.

In RCMT, the core examples are:

- **twist**: global orientation mismatch,
- **winding**: global non-contractible linkage,
- **holonomy**: global accumulated transport transformation.

Each is locally trivial but globally nontrivial, and each marks a form of structure that cannot be erased by local coherence operations alone.