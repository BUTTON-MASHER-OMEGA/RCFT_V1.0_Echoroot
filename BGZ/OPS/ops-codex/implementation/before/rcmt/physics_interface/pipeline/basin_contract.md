# Basin Contract (BC‑1.0)
*RCMT → RCF‑compatible, RCP‑typed*

This contract defines **basin structure** in RCMT—how basins organize adjacency, constrain propagation, shape coherence, and gate which patterns can stabilize as survivors, modes, and geometry candidates.

Basins are **upstream organizational structure**, not potentials, not energy wells, not fields.

---

## 1. Purpose

**Purpose:**
Define basin structure as the organizational scaffold that constrains mismatch, propagation, coherence, and survivor formation.

Basins explain *where* patterns can live and *how* they tend to persist or transition.

**Type:** Purpose
**Owner:** RCMT basin module

---

## 2. Ownership

The basin module owns:

- **basin definition** (what counts as a basin)
- **basin membership** (what belongs to which basin)
- **basin transitions** (how membership can change)
- **basin constraints** on propagation and coherence

The basin module does **not** own:

- geometry
- fields or potentials
- metric structure
- causal dynamics
- projection‑level interpretation

**Type:** Invariant

---

## 3. Basin definition

A **basin** is:

> **a region of adjacency‑constrained relational structure within which mismatch, survivors, and modes exhibit characteristic coherence and propagation behavior.**

Basins:

- are adjacency‑structured
- are coherence‑shaped
- constrain what can stabilize
- constrain how propagation behaves
- organize where survivors and geometry candidates can exist

Basins are **not** spatial regions or energy wells.

**Type:** Invariant (category definition)

---

## 4. Basin roles

Basins:

- **constrain propagation:** which adjacency paths are effectively available
- **shape coherence:** which patterns are favored or disfavored
- **gate survivors:** which survivor structures can stabilize in a given basin
- **gate modes:** which mode families are admissible in a basin
- **gate geometry candidates:** which tilings can persist there

Basins are the **context** in which upstream structure lives.

**Type:** Dependency

---

## 5. Allowed inputs

The basin module consumes:

- adjacency relations
- mismatch distributions
- survivor structures
- mode structures
- coherence scores
- propagation traces

**Type:** Dependency

---

## 6. Allowed outputs

The basin module exposes:

- basin identity labels
- basin membership assignments
- basin transition possibilities
- basin‑specific constraints on propagation
- basin‑specific constraints on survivor and mode stability

These outputs:

- **are upstream organizational structure**,
- **may be interpreted by projection as regions, domains, or “phases,”**
- **but are not geometric or energetic on their own.**

**Type:** Interface Output

---

## 7. Basin → projection interface

Projection may interpret basin outputs as:

- spatial‑like regions
- potential‑like domains
- phase‑like sectors
- environment‑like contexts for matter and fields

These are **downstream interpretations**, not upstream facts.

**Type:** Downstream Interpretation

---

## 8. Prohibitions

Basins must not:

- define geometry
- define potentials or energy
- define fields or forces
- introduce metric structure
- encode causal dynamics
- redefine coherence, propagation, or mismatch categories
- widen upstream invariants

**Type:** Prohibition

---

## 9. Scope binding

All basin statements must be scoped:

- “within the present formalism”
- “as upstream basin structure”
- “prior to projection”
- “in the basin module”

This prevents physics‑shaped language from leaking upstream.

**Type:** Contract

---

## 10. Observations (current evaluations)

> In current evaluations, basins appear as adjacency‑structured regions where certain survivors, modes, and geometry candidates are favored, and where propagation and coherence exhibit characteristic patterns.

This is observational, not canonical.

**Type:** Observation

---

## 11. Status

- Contract: DRAFT PROPOSED
- Owner: RCMT basin module
- Applicability: All RCMT modules
- Scope: Upstream basin structure

---

**One‑liner:**
Basins = adjacency‑structured, coherence‑shaped regions that organize where and how upstream patterns can persist—without being potentials, fields, or spatial regions.
