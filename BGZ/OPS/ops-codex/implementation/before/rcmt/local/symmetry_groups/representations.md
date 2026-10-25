# RCMT Symmetry Representations

## Document Contract

- owner: RCMT
- type: Contract
- scope: provisional response representation
- depends_on: `actions.md`, `../constraints/symmetry_chain.md`, and upstream LCO outputs
- status: replaceable downstream implementation
- prohibition: representations do not define new groups, invariants, or identity guarantees

## Purpose

This file defines **representations** in RCMT.

Representations describe how relational structures respond to admissible transformations defined by the symmetry_chain.

They do not introduce new groups or invariants.

---

## DEFINITION

A representation in RCMT is:

> **The observed transformation behavior of a relational structure under admissible symmetry actions.**

Representations do not enforce identity preservation.
They describe how structures transform and whether alignment is maintained.

---

## WHAT REPRESENTATIONS ACT ON

Representations apply to relational structures:

- Modes
- Boundaries
- Identities
- Frames
- Shards

They do not act on spatial coordinates.

---

## ALIGNMENT AND IDENTITY

Identity in RCMT is not defined by exact preservation.

Identity is defined by **alignment within an admissible basin under LCO evaluation**.

A representation is valid when:

- the transformed structure remains within admissible bounds
- coherence is maintained
- mismatch remains below threshold

Identity continuity is an outcome of alignment, not a requirement of the representation itself.

---

## REQUIREMENTS FOR A VALID REPRESENTATION

A representation is valid when:

1. The transformation is group-admissible
2. The resulting structure is structure-admissible under LCO evaluation
3. Coherence remains within acceptable bounds
4. Mode compatibility is preserved

These are evaluation conditions, not guarantees.

---

## STABILIZERS AND ORBITS

Within the current RCMT representation model, a structure may expose:

- a **stabilizer** — transformations under which alignment remains unchanged
- an **orbit** — the set of admissible transformed structures

These are defined by LCO evaluation, not strict algebraic equality.

---

## REPRESENTATION TYPES

### Mode Representations
Transformations of internal relational grammars

### Boundary Representations
Transformations of boundary-compatible frames

### Identity Representations
Transformations of persistent structures under admissible change

These categories describe transformation behavior, not identity guarantees.

---

## ADMISSIBLE VS INADMISSIBLE REPRESENTATIONS

A representation is admissible when:

- the transformation is group-admissible
- the resulting structure remains LCO-admissible
- coherence is maintained
- mismatch remains below threshold

A representation is inadmissible when:

- coherence collapses
- mismatch exceeds threshold
- the structure exits all admissible basins

---

## RELATION TO SYMMETRY_CHAIN

symmetry_chain defines:

- allowed groups
- transformation structure

representations.md defines:

- how structures respond to those transformations
- how admissibility is evaluated

---

## DOWNSTREAM USAGE

Representations are used to:

- analyze transformation behavior
- describe alignment under symmetry
- evaluate admissibility under transformation
- support interaction and measurement modeling

Representations describe **what happens under transformation**, not what must be preserved.
