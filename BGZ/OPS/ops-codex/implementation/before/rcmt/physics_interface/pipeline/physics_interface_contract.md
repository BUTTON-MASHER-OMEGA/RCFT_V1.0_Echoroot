# Physics‑Interface Contract (PIC‑1.0)
*RCMT → Projection → Physics modules, RCP‑typed*

This contract defines the **API boundary** between the projection layer and all downstream physics modules.

Physics modules never see upstream RCMT objects directly.
They only see **projection outputs**, and must treat them as **effective artifacts**, not ontology.

---

## 1. Purpose

**Purpose:**
Define what physics modules may:

- **consume** from projection
- **assume** about those artifacts
- **never infer** about upstream structure

This is the hard firewall between RCMT and physics.

**Type:** Purpose
**Owner:** Physics‑interface layer

---

## 2. Ownership

The physics‑interface layer owns:

- the **shape** of projection outputs as physics‑usable objects
- the **naming and typing** of those outputs
- the **constraints** on how physics modules may use them

It does **not** own:

- upstream semantics (RCMT)
- projection’s internal interpretation rules
- any upstream invariants

**Type:** Invariant

---

## 3. Exposed artifact types

Physics modules may receive from projection:

- **Geometric artifacts**
  - coordinates, distances, curvature‑like quantities
- **Field‑like artifacts**
  - effective fields, potentials, fluxes
- **Matter‑like artifacts**
  - particle‑like entities, worldlines, conserved charges
- **Spinor‑like / mode‑like artifacts**
  - spin states, polarization, harmonics
- **Region / domain artifacts**
  - phase‑like regions, domains, boundaries

All of these are **effective artifacts**, not upstream objects.

**Type:** Interface Types

---

## 4. Allowed inputs (to physics modules)

Physics modules may consume:

- geometric‑lane structures (from projection)
- effective fields and potentials
- particle‑like entities and their attributes
- spinor‑like states and mode labels
- region/domain labels and boundaries
- effective conservation laws (as given by projection)

They may **not** consume:

- adjacency
- mismatch categories
- survivors
- modes (upstream)
- geometry candidates
- coherence scores
- basins
- propagation traces

**Type:** Dependency

---

## 5. Allowed assumptions

Physics modules may assume that:

- geometric artifacts behave **as if** they live on a manifold
- field‑like artifacts behave **as if** they satisfy field equations
- particle‑like artifacts behave **as if** they follow trajectories and conservation laws
- spinor‑like artifacts behave **as if** they follow spin/mode rules
- region artifacts behave **as if** they define phases or domains

They may **not** assume:

- that any of this geometry is upstream
- that any field is ontological
- that any manifold exists upstream
- that any causal structure is fundamental

All assumptions are **effective**, not ontological.

**Type:** Assumption Contract

---

## 6. Forbidden inferences

Physics modules must not:

- infer anything about adjacency structure
- infer anything about mismatch, survivors, modes, or basins
- infer upstream coherence rules
- infer upstream propagation rules
- reinterpret projection outputs as fundamental ontology

They must treat projection outputs as:

> “Given effective structures with these behaviors; upstream origin is opaque.”

**Type:** Prohibition

---

## 7. Directionality

Information flow:

- **Upstream → Physics:**
  RCMT → Projection → Physics‑interface → Physics modules

- **Downstream → Upstream:**
  Physics modules may request *more projection outputs* or *different views*,
  but may not modify or constrain upstream RCMT structure.

No physics module may:

- write into RCMT
- alter projection rules
- redefine interface types

**Type:** Flow Constraint

---

## 8. Scope binding

All physics‑interface statements must be scoped:

- “within the present formalism”
- “as an effective physics interface”
- “downstream of projection”
- “upstream structure is opaque here”

**Type:** Contract

---

## 9. Status

- Contract: **DRAFT PROPOSED**
- Owner: **Physics‑interface layer**
- Applicability: **All physics modules**
- Scope: **Projection → Physics boundary**

---

**One‑liner:**
Physics modules see only projection’s effective artifacts—geometry‑like, field‑like, matter‑like—not the RCMT substrate that generated them.
