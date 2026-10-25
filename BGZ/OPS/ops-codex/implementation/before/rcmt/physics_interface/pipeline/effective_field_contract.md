### `rcmt/pipeline/effective_field_contract.md`

---

# **Effective Field Contract (EFC‑1.0)**
*(RCMT → RCF‑compatible, RCP‑typed)*

This contract defines how **effective field behavior** is exposed by the projection layer.
It does **not** define fields as ontological objects.
It specifies the **interface** by which downstream modules may interpret mismatch propagation.

---

## **1. Purpose**

> **Define the projection‑level interface for interpreting mismatch propagation as effective field‑like behavior.**

This contract ensures that “fields” remain **interpretations**, not upstream structures.

**Type:** Purpose
**Owner:** Projection layer

---

## **2. Ownership**

Projection owns:

- the interpretation of mismatch patterns
- the smoothing rules that produce effective field behavior
- the mapping from discrete geometry → continuous‑looking variation

Projection does **not** own:

- mismatch generation
- adjacency structure
- geometry candidates
- upstream invariants

**Type:** Invariant
**Owner:** Projection layer

---

## **3. Allowed Inputs**

Projection may consume:

- **free mismatch** (unbound, propagating)
- **structured mismatch** (patterned, semi‑bound)
- **spin‑profiled geometry candidates** (e.g., 120‑cell boundary)
- **coherent relational state** from LCO

**Type:** Dependency

---

## **4. Allowed Outputs**

Projection may expose:

- **effective field‑like variation**
- **effective curvature‑like patterns**
- **phase‑like or orientation‑like behavior**
- **boundary perturbation profiles**
- **spinor‑compatible orientation changes**

These outputs are **interpretations**, not upstream objects.

**Type:** Interface Output

---

## **5. Core Rule (Effective Field Rule)**

> **Within the present formalism, projection interprets mismatch propagation across a spin‑profiled tessellation (e.g., 120‑cell boundary) as effective field‑like behavior.**

This is the exact mapping:

- upstream: discrete mismatch
- geometry: spin‑profiled tessellation
- projection: smoothing + interpretation
- downstream: effective field

**Type:** Downstream Interpretation

---

## **6. Prohibitions**

Projection must not:

- generate continuous fields
- introduce manifold structure
- redefine upstream mismatch categories
- widen upstream invariants
- treat effective fields as ontological
- imply mechanism beyond interpretation

**Type:** Prohibition

---

## **7. Scope Binding**

All effective‑field statements must be scoped:

- “within the present formalism”
- “in the geometric lane”
- “as a downstream interpretation”
- “projection exposes X”

This prevents misinterpretation as upstream ontology.

**Type:** Contract

---

## **8. Observations (Current Evaluations)**

> **In current evaluations, free mismatch traveling across the 120‑cell boundary is exposed as smooth, field‑like variation in the geometric lane.**

This is observational, not canonical.

**Type:** Observation

---

## **9. Status**

- Contract: **DRAFT PROPOSED**
- Owner: **Projection layer**
- Applicability: **RCMT geometric lane**
- Scope: **Downstream interpretation only**

---

# The one‑liner
**Effective fields = projection‑level interpretations of mismatch on a spin‑profiled tessellation, not ontological fields.**
