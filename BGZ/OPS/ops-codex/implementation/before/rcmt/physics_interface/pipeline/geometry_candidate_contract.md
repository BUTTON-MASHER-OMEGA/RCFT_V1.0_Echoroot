# **Geometry Candidate Contract (GCC‑1.0)**
*(RCMT → RCF‑compatible, RCP‑typed)*

This contract defines what qualifies as a **geometry candidate** in RCMT.
Geometry candidates are **survivor‑derived structures** that projection may interpret as geometric‑lane artifacts (e.g., 120‑cell boundary, spinor‑like orientation, curvature‑like variation).

Geometry candidates are **not** geometric objects upstream.
They are **survivor structures** that *support* geometric interpretation downstream.

---

## **1. Purpose**

> **Define the upstream criteria for geometry candidates and the interface they expose to projection.**

Geometry candidates provide the substrate for the geometric lane.

**Type:** Purpose
**Owner:** RCMT geometry‑candidate module

---

## **2. Ownership**

The geometry‑candidate module owns:

- the classification of geometry‑compatible survivor structures
- the upstream semantics of adjacency‑stable tilings
- the spin‑profile and mode‑profile requirements
- the interface for projection consumption

The geometry‑candidate module does **not** own:

- geometric interpretation
- curvature
- metric structure
- field‑like behavior
- continuous manifolds

**Type:** Invariant

---

## **3. Geometry Candidate Definition**

A **geometry candidate** is:

> **a survivor‑derived, adjacency‑stable structure with spin‑compatible orientation profiles that persists across evaluations and supports projection‑level geometric interpretation.**

Geometry candidates:

- arise from survivors
- maintain stable adjacency patterns
- carry spin‑compatible orientation structure
- support mode‑bearing behavior
- provide boundary‑like tilings (e.g., 120‑cell)

Geometry candidates are **not** geometric objects upstream.

**Type:** Invariant (category definition)

---

## **4. Allowed Inputs**

The geometry‑candidate module consumes:

- survivor structures
- survivor adjacency signatures
- spin‑profile data
- mode‑bearing internal structure
- LCO coherence scoring

**Type:** Dependency

---

## **5. Allowed Outputs**

The geometry‑candidate module exposes:

- adjacency‑stable tilings
- spin‑compatible orientation profiles
- mode‑bearing structural patterns
- boundary‑like survivor arrangements
- geometry‑compatible survivor signatures

These outputs:

- **are upstream structures**,
- **may be interpreted by projection as geometry**,
- **but are not geometric objects on their own.**

**Type:** Interface Output

---

## **6. Geometry Candidate → Projection Interface**

Projection may interpret geometry‑candidate outputs as:

- geometric‑lane coordinates
- curvature‑like variation
- spinor‑like orientation behavior
- effective field‑supporting boundaries
- continuous‑looking geometric structure

These are **downstream interpretations**, not upstream facts.

**Type:** Downstream Interpretation

---

## **7. Prohibitions**

Geometry candidates must not:

- define geometry
- define curvature
- define fields
- imply continuous manifolds
- introduce metric structure
- redefine survivor behavior
- widen upstream invariants

**Type:** Prohibition

---

## **8. Scope Binding**

All geometry‑candidate statements must be scoped:

- “within the present formalism”
- “as upstream survivor‑derived structure”
- “prior to projection”
- “in the geometry‑candidate module”

This prevents downstream leakage.

**Type:** Contract

---

## **9. Observations (Current Evaluations)**

> **In current evaluations, the 120‑cell boundary appears as a stable geometry candidate with spin‑compatible orientation profiles and adjacency‑stable tiling.**

This is observational, not canonical.

**Type:** Observation

---

## **10. Status**

- Contract: **DRAFT PROPOSED**
- Owner: **RCMT geometry‑candidate module**
- Applicability: **All RCMT modules**
- Scope: **Upstream geometry‑candidate classification**

---

# The one‑liner
**Geometry candidates = survivor‑derived structures that projection interprets as geometry, without being geometric objects upstream.**
