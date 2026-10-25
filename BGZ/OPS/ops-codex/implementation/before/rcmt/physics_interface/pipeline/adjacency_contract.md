# **Adjacency Contract (AC‑1.0)**
*(RCMT → RCF‑compatible, RCP‑typed)*

This contract defines **adjacency** in RCMT — the primitive relational structure on which mismatch, survivors, modes, geometry candidates, propagation, and coherence all depend.

Adjacency is **the substrate relation**.
It is not space, not distance, not a metric, not a graph embedded anywhere.
Everything else is layered on top of it.

---

## 1. Purpose

> **Define adjacency as the primitive relational structure of the substrate and the constraints on how it may be used.**

Adjacency is the base relation for all upstream structure.

**Type:** Purpose
**Owner:** RCMT adjacency module

---

## 2. Ownership

The adjacency module owns:

- the definition of adjacency
- the constraints on adjacency updates
- the admissible adjacency patterns
- the interface for adjacency queries

The adjacency module does **not** own:

- geometry
- distance
- metric structure
- fields
- waves
- causal structure

**Type:** Invariant

---

## 3. Adjacency definition

**Adjacency** is:

> **a primitive, unembedded relation indicating which substrate elements may directly interact or participate in shared structure.**

Adjacency:

- is primitive
- is discrete
- is not embedded
- carries no metric
- carries no geometry
- carries no direction by default

Adjacency is **just** “can directly relate,” nothing more.

**Type:** Invariant (category definition)

---

## 4. Adjacency usage

Adjacency is used to:

- define where mismatch can exist
- define where mismatch can propagate
- define where survivors can form
- define where modes can be supported
- define where geometry candidates can arise
- define the domain of coherence evaluation

All upstream structure is **constrained by adjacency**.

**Type:** Dependency

---

## 5. Allowed inputs

The adjacency module may consume:

- substrate element identifiers
- relational constraints from invariants
- coherence‑driven update permissions

**Type:** Dependency

---

## 6. Allowed outputs

The adjacency module exposes:

- adjacency relations (who can relate to whom)
- admissible adjacency updates
- adjacency‑compatible neighborhoods
- adjacency‑consistent propagation paths

These outputs:

- **are upstream structural relations**,
- **may be interpreted by projection as spatial neighborhoods or connectivity**,
- **but are not geometric or metric on their own.**

**Type:** Interface Output

---

## 7. Adjacency → projection interface

Projection may interpret adjacency outputs as:

- spatial‑like neighborhoods
- connectivity graphs
- local “closeness” structure
- geometric‑lane locality

These are **downstream interpretations**, not upstream facts.

**Type:** Downstream Interpretation

---

## 8. Prohibitions

Adjacency must not:

- define distance
- define metric
- define geometry
- define fields or waves
- introduce coordinates
- encode causal order directly
- widen upstream invariants

**Type:** Prohibition

---

## 9. Scope binding

All adjacency statements must be scoped:

- “within the present formalism”
- “as primitive adjacency structure”
- “prior to projection”
- “in the adjacency module”

This prevents geometry or physics from leaking upstream.

**Type:** Contract

---

## 10. Observations (current evaluations)

> **In current evaluations, adjacency constrains where mismatch can propagate, where survivors can form, where modes can persist, and where geometry candidates can arise.**

This is observational, not canonical.

**Type:** Observation

---

## 11. Status

- Contract: **DRAFT PROPOSED**
- Owner: **RCMT adjacency module**
- Applicability: **All RCMT modules**
- Scope: **Primitive relational structure**

