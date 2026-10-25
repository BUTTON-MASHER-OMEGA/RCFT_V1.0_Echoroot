# RCMT Symmetry Actions

## Document Contract

- owner: RCMT
- type: Contract
- scope: provisional finite-transformation interpretation
- depends_on: `generators.md`, `../constraints/symmetry_chain.md`, and upstream LCO outputs
- status: replaceable downstream implementation
- prohibition: actions do not define upstream admissibility or guarantee identity continuity

## Purpose

This file defines **actions** in RCMT.

Actions are finite transformations constructed from generators defined in `generators.md`.

They describe how relational structures change under transformation.

---

## DEFINITION

An action in RCMT is:

> **A finite transformation constructed from generators and evaluated by the LCO.**

Actions define transformation paths.
They do not guarantee identity continuity.

---

## WHAT ACTIONS OPERATE ON

Actions operate on relational structures:

- Modes
- Boundaries
- Identities
- Frames
- Shards

They do not operate on coordinates.

---

## ADMISSIBILITY

Two levels of admissibility:

### Group-admissible action
Valid transformation defined by symmetry structure

### Structure-admissible action
Transformation that remains admissible under LCO evaluation

---

## IDENTITY UNDER ACTIONS

Identity is not preserved by action.

Identity is determined by **alignment under LCO evaluation**.

After a transformation:

- identity may continue (alignment maintained)
- identity may shift (relocking into new basin)
- identity may terminate (no admissible basin)

Identity is therefore trajectory-dependent.

---

## BASIN TRANSITIONS

Actions may move structures across basins.

LCO evaluation determines:

- continued alignment
- relocking into new basin
- failure to maintain alignment

---

## ADMISSIBILITY CONDITIONS

A structure-admissible action requires:

- group-admissibility
- LCO acceptance
- coherence stability
- mode compatibility

---

## ADMISSIBLE VS INADMISSIBLE ACTIONS

Admissible:
- structure remains within or transitions into an admissible basin

Inadmissible:
- structure exits all admissible basins
- coherence collapses
- mismatch exceeds threshold

---

## ACTION TYPES

### Mode Actions
Finite transformations of internal grammars

### Boundary Actions
Finite transformations of frames

### Identity Actions
Finite transformations of persistent structures

---

## RELATION TO SYMMETRY_CHAIN

symmetry_chain defines allowed transformations.

actions.md defines:

- how those transformations are realized
- how admissibility is evaluated

---

## DOWNSTREAM USAGE

Actions are used to:

- model transformation processes
- evaluate admissibility
- analyze basin transitions
- describe interaction behavior

Actions define **what transformations occur**.
Upstream LCO outcomes are used to classify **which transformations remain admissible under RCMT**.
