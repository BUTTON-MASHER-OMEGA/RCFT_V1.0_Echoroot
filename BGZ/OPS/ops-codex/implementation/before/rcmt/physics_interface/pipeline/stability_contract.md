# **Stability Contract (SC‑1.0)**
*RCMT → RCF‑compatible, RCP‑typed*

This contract defines **stability** in RCMT — the upstream criterion that determines whether mismatch patterns, survivors, modes, and geometry candidates persist across evaluations.

Stability is **not** energy minimization, not equilibrium, not a potential well, not a force balance.
It is **coherence‑driven persistence** under adjacency and basin constraints.

---

## **1. Purpose**

> **Define the upstream criteria for stability, how stability is evaluated, and how stable structures are exposed to projection.**

Stability determines what survives long enough to matter downstream.

**Type:** Purpose
**Owner:** RCMT stability module

---

## **2. Ownership**

The stability module owns:

- the definition of stability
- the evaluation of stability under coherence
- the relationship between basins and stability
- the persistence criteria for survivors, modes, and geometry candidates

The stability module does **not** own:

- geometric interpretation
- mass, energy, or potential
- field‑like behavior
- causal dynamics
- metric structure

**Type:** Invariant

---

## **3. Stability Definition**

A structure is **stable** when:

> **it maintains coherence across adjacency updates within a basin and does not dissolve under propagation.**

Stability requires:

- coherence above threshold
- basin‑compatible structure
- adjacency‑consistent propagation
- survivor‑level or mode‑level persistence

Stability is **structural persistence**, not physical equilibrium.

**Type:** Invariant (category definition)

---

## **4. Stability Classes**

Stability applies to:

- **mismatch patterns**
  (whether mismatch arrangements persist)

- **survivors**
  (whether bound structures remain intact)

- **modes**
  (whether internal patterns persist)

- **geometry candidates**
  (whether tilings remain viable)

These are **evaluation classes**, not ontological categories.

**Type:** Observation

---

## **5. Allowed Inputs**

The stability module consumes:

- coherence scores
- basin membership
- adjacency relations
- propagation traces
- survivor structures
- mode structures
- geometry‑candidate structures

**Type:** Dependency

---

## **6. Allowed Outputs**

The stability module exposes:

- stability decisions (stable / unstable)
- persistence durations
- stability signatures
- stability‑compatible propagation paths
- stability‑compatible survivor/mode/geometry‑candidate sets

These outputs:

- **are upstream structural evaluations**,
- **may be interpreted by projection as mass‑like or persistence‑like behavior**,
- **but are not geometric or energetic on their own.**

**Type:** Interface Output

---

## **7. Stability → Projection Interface**

Projection may interpret stability outputs as:

- mass‑like persistence
- particle‑like identity
- field‑like smoothness
- geometric‑lane continuity
- mode‑like quantization

These are **downstream interpretations**, not upstream facts.

**Type:** Downstream Interpretation

---

## **8. Prohibitions**

Stability must not:

- define geometry
- define energy or potential
- define fields or forces
- imply continuous manifolds
- introduce metric structure
- redefine coherence or basins
- widen upstream invariants

**Type:** Prohibition

---

## **9. Scope Binding**

All stability statements must be scoped:

- “within the present formalism”
- “as upstream stability evaluation”
- “prior to projection”
- “in the stability module”

This prevents physics‑shaped language from leaking upstream.

**Type:** Contract

---

## **10. Observations (current evaluations)**

> In current evaluations, stability arises when coherence remains high across adjacency updates within a basin, allowing survivors, modes, and geometry candidates to persist long enough for projection to interpret them.

This is observational, not canonical.

**Type:** Observation

---

## **11. Status**

- Contract: **DRAFT PROPOSED**
- Owner: RCMT stability module
- Applicability: All RCMT modules
- Scope: Upstream stability evaluation

---

# **One‑liner**
**Stability = coherence‑driven persistence of upstream structure within a basin — not energy, not equilibrium, not fields.**
