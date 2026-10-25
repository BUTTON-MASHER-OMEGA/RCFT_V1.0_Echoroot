# **Projection Interpretation Rule (PIR‑1.0)**
*(RCMT → RCF‑compatible, RCP‑typed)*

> **Projection does not create new structure.
> Projection exposes downstream‑interpretable structure already present in the coherent relational state.**

This rule defines how projection is allowed to *interpret* upstream adjacency, mismatch, and geometry‑candidate structure without ever redefining upstream meaning.

---

## **1. Ownership**
**Projection owns interpretation, not generation.**

- Projection **defines** the mapping from coherent relational state → geometric lane.
- Projection **does not define** upstream adjacency, mismatch, or basin structure.
- Projection **exposes** geometric‑lane artifacts (metric‑like, field‑like, curvature‑like).
- Projection **does not generate** new upstream invariants.

**Type:** Invariant
**Owner:** Projection layer

---

## **2. Allowed Inputs**
Projection may consume:

- adjacency relations
- mismatch patterns (free, locked, structured)
- spin‑profiled geometry candidates (e.g., 120‑cell boundary)
- coherent relational state from LCO

**Type:** Dependency
**Owner:** Projection layer

---

## **3. Allowed Outputs**
Projection may expose:

- geometric‑lane coordinates
- effective curvature‑like variation
- effective field‑like behavior
- spinor‑like orientation profiles
- boundary perturbation patterns

These outputs are **interpretations**, not upstream objects.

**Type:** Interface Output
**Owner:** Projection layer

---

## **4. Interpretation Rule**
### **Core Rule**
> **Within the present formalism, projection interprets free mismatch propagating across a spin‑profiled tessellation (e.g., 120‑cell boundary) as effective field‑like perturbation.**

This is the current mapping:

- **free mismatch** → upstream object
- **spin‑profiled tessellation** → geometric candidate
- **projection smoothing** → interpretation
- **effective field** → downstream artifact

**Type:** Downstream Interpretation

---

## **5. Prohibitions**
Projection must not:

- redefine upstream adjacency
- introduce continuous fields
- introduce continuous manifolds
- widen upstream invariants
- treat effective fields as ontological objects
- generate new mismatch categories

**Type:** Prohibition
**Owner:** Projection layer

---

## **6. Scope Binding**
All projection‑level interpretations must be explicitly scoped:

- “within the present formalism”
- “in the geometric lane”
- “as a downstream interpretation”
- “projection exposes X”

This prevents field‑like behavior from being misread as upstream ontology.

**Type:** Contract

---

## **7. Load‑Bearing Sentence Test**
Every projection statement must remain true if all interpretation is removed.

Example:

> “Projection exposes field‑like behavior”
passes because removing interpretation yields:
> “Projection exposes behavior.”

But:

> “Projection generates fields”
fails because removing interpretation yields:
> “Projection generates,”
which violates ownership.

---

## **8. Summary (Normal Form)**

**Purpose:**
Define how projection interprets mismatch and geometry candidates.

**Owner:**
Projection layer.

**Dependencies:**
Adjacency, mismatch, coherent relational state, geometry candidates.

**Allowed Outputs:**
Geometric‑lane coordinates, effective curvature, effective fields, spinor‑like profiles.

**Prohibitions:**
No generation of upstream structure, no continuous fields, no ontology inflation.

**Observations:**
In current evaluations, free mismatch on a spin‑profiled tessellation is exposed as field‑like perturbation.

**Downstream Interpretation:**
Effective fields = projection‑level smoothing of mismatch on the 120‑cell boundary.

---

# The one‑liner
**Projection interprets mismatch on a spin‑profiled tessellation as effective field behavior — never as an ontological field.**
