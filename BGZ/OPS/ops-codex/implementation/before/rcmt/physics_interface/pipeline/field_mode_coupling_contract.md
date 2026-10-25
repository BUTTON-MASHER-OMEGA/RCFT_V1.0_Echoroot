### `physics_interface/pipeline/field_mode_coupling_contract.md`

# Field–Mode Coupling Contract (FMC‑1.0)
*RCMT → Projection, RCP‑typed*

This contract defines how **modes** and **mismatch propagation** combine upstream to support **effective field‑like behavior** downstream under projection.

It does **not** define fields as ontological objects.
It defines which upstream structures are allowed to be interpreted as field‑like.

---

## 1. Purpose

**Purpose:**
Define the upstream conditions under which:

- mode structure
- mismatch propagation

may be jointly interpreted by projection as **effective fields** or **field excitations**.

**Type:** Purpose
**Owner:** RCMT field–mode coupling module

---

## 2. Ownership

The field–mode coupling module owns:

- the definition of “field‑compatible” mode patterns
- the relationship between free mismatch and modes
- the upstream semantics of “excitations” as structural changes

It does **not** own:

- geometric interpretation
- field equations
- potentials, flux, or sources
- downstream dynamics

**Type:** Invariant

---

## 3. Field‑compatible configuration

A configuration is **field‑compatible** when:

> **free mismatch propagates across adjacency in a way that preserves a mode pattern over a geometry candidate or basin, under coherence and stability constraints.**

Requirements:

- free mismatch present and propagating
- a persistent mode pattern defined on survivors / geometry candidates
- coherence above threshold
- stability within a basin

This is a **structural condition**, not a field definition.

**Type:** Invariant (category definition)

---

## 4. Upstream roles

In field‑compatible configurations:

- **mismatch propagation** supplies variation
- **modes** supply structured patterning
- **geometry candidates / basins** supply organizational context
- **coherence + stability** gate which patterns persist

Upstream, this is just **structured propagation**, not a field.

**Type:** Dependency

---

## 5. Allowed inputs

The field–mode coupling module consumes:

- mismatch categories and propagation traces
- mode structures and mode families
- geometry candidates
- basin structure
- coherence and stability evaluations

**Type:** Dependency

---

## 6. Allowed outputs

The field–mode coupling module exposes:

- field‑compatible structural patterns
- mode‑tagged propagation signatures
- basin‑ and geometry‑aware variation patterns

These outputs:

- **are upstream structures**,
- **may be interpreted by projection as effective fields or excitations**,
- **but are not fields on their own.**

**Type:** Interface Output

---

## 7. Field–mode → projection interface

Projection may interpret field‑compatible patterns as:

- effective fields
- field excitations
- wave‑like or radiation‑like behavior
- source‑like or flux‑like structure

These are **downstream interpretations**, not upstream facts.

**Type:** Downstream Interpretation

---

## 8. Prohibitions

Field–mode coupling must not:

- define field equations
- define potentials or sources ontologically
- introduce metric or manifold structure
- redefine modes, mismatch, or geometry candidates
- widen upstream invariants

**Type:** Prohibition

---

## 9. Scope binding

All field–mode coupling statements must be scoped:

- “within the present formalism”
- “as upstream field‑compatible structure”
- “prior to projection”
- “in the field–mode coupling module”

**Type:** Contract

---

## 10. Status

- Contract: DRAFT PROPOSED
- Owner: RCMT field–mode coupling module
- Applicability: All projection‑level field interpretations
- Scope: Upstream field‑compatible structure

---

**One‑liner:**
Field–mode coupling = structured mismatch + modes in a stable context that projection is allowed to read as effective fields.
