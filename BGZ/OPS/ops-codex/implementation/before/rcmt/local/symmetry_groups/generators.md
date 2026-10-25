# RCMT Symmetry Generators

## Document Contract

- owner: RCMT
- type: Contract
- scope: provisional generator interpretation
- depends_on: `../constraints/symmetry_chain.md` and upstream LCO outputs
- status: replaceable downstream implementation
- prohibition: generators do not define upstream admissibility or preserve identity

## Purpose

This file defines **generators** in RCMT.

Generators specify the **infinitesimal transformation directions** permitted by the symmetry structure defined in `symmetry_chain`.

Generators do not introduce new groups or invariants.
They do not preserve identity.
They define **allowed directions of change**.

Admissibility and identity continuity are determined by LCO evaluation.

---

### DEFINITION

A generator in RCMT is:

> **An infinitesimal transformation direction defined by the symmetry structure.**

Generators define how relational structures may change locally.
They do not determine whether a transformed structure remains valid.

---

### WHAT GENERATORS ACT ON

Generators act on **relational structures**, not coordinates.

They may transform:

- **Modes** — internal relational grammars
- **Boundaries** — admissible regions and coherence anchors
- **Identities** — persistent relational configurations
- **Frames** — projection grammars and measurement orientations
- **Shards** — coherent relational fragments

Generators do not act on absolute space or coordinates.

---

### GROUP-ADMISSIBLE VS STRUCTURE-ADMISSIBLE

RCMT distinguishes two levels of admissibility:

**Group-admissible generator**
- A valid infinitesimal direction defined by the symmetry structure
- Independent of any specific structure

**Structure-admissible generator**
- A generator whose application produces a structure that remains admissible under LCO evaluation

A generator may be group-admissible but structure-inadmissible.

---

### IDENTITY AND ALIGNMENT

Identity in RCMT is not defined by exact preservation or memory.

Identity is defined by **alignment within an admissible basin** under LCO evaluation.

Generators do not preserve identity.
They produce transformed structures.

Identity continuity holds when the transformed structure remains within the same admissible basin.

If the structure exits its basin:

- identity continuity may terminate, or
- the structure may relock into a different basin

RCMT therefore treats:

- identity as **basin-relative**
- identity continuity as **LCO-evaluated alignment**
- identity as **trajectory-dependent, not state-dependent**

---

### ADMISSIBILITY CONDITIONS

A generator step is structure-admissible when:

1. The generator is group-admissible
2. The transformed structure remains within LCO admissibility bounds
3. Coherence remains within acceptable limits
4. Mode compatibility is preserved

These are **filters**, not guarantees.

---

### ADMISSIBLE VS INADMISSIBLE STEPS

A generator step is **admissible** when:

- mismatch remains below threshold
- coherence is preserved
- the structure remains within a valid basin

A generator step is **inadmissible** when:

- mismatch exceeds threshold
- coherence collapses
- the structure exits all admissible basins

Inadmissible steps represent directions that cannot be followed by that structure in that context.

---

### GENERATOR TYPES

RCMT uses three primary generator classes:

**Mode Generators**
- Act on internal relational grammars
- Example: SU(2)-type generators

**Boundary Generators**
- Act on boundary-compatible frames
- Example: Lorentz-type generators

**Identity Generators**
- Act on persistent relational configurations
- Example: U(1)-type generators

These correspond to internal, boundary, and phase transformations, expressed relationally.

---

### RELATION TO SYMMETRY_CHAIN

`symmetry_chain` defines:

- admissible group structure
- invariant hierarchy
- Lie algebra directions

`generators.md` defines:

- how those directions act on relational structures
- how admissibility is evaluated
- how identity continuity is determined

This file is interpretive and does not modify the symmetry structure.

---

### DOWNSTREAM USAGE

Generators are used to:

- construct finite transformations
- define conservation constraints
- analyze local transformation behavior
- evaluate admissibility under change
- build representation-level mappings

Generators define the **local transformation space**.
Upstream LCO outcomes are used to classify which transformations remain admissible under RCMT.
