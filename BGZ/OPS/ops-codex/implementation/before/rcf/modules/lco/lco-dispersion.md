# LCO Dispersion

## Document Contract

- owner: RCF local coherence operator
- type: Interpretation
- scope: module-local content of `lco/lco-dispersion.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Definition and Role of the Dispersion Functional*

This document defines the **dispersion functional**, one of the core components used by the Local Coherence Operator (LCO) to evaluate candidate local updates. Dispersion measures how uniformly the mismatch values are distributed across the relational neighborhood and the memory anchor.

Dispersion is a stabilizing term: it penalizes candidate modes that align well with some relations but poorly with others, encouraging updates that maintain coherence across the entire local neighborhood.

Dispersion is purely algebraic and uses only the primitives permitted at the execution layer.

The dispersion functional must remain a uniform mean squared deviation and must not introduce relation-specific weighting or non-local normalization.

---

## **1. Purpose of Dispersion**

While mismatch measures *how much* a candidate mode deviates from each relation, **dispersion measures how evenly that deviation is distributed**.

Dispersion helps the LCO:

- discourage updates that create sharp local inconsistencies
- encourage balanced alignment across all relations
- stabilize the update process by reducing sensitivity to outlier relations
- support local coherence descent under admissible update policies

It is a local, geometry‑neutral measure of relational consistency.

---

## **2. Definition of Dispersion**

Given the set of mismatch values:



\[
\{\ell_i\}_{i=1}^N \cup \{\ell_{\text{mem}}\}
\]



the LCO computes:

### **Aggregate mismatch**



\[
\bar{\ell} = \text{mean}(\{\ell_i\} \cup \{\ell_{\text{mem}}\})
\]



### **Dispersion**



\[
\text{Disp} = \text{mean}\big((\ell_k - \bar{\ell})^2\big),
\quad k \in \{1,\dots,N,\text{mem}\}
\]



This is the mean squared deviation (population variance) of the mismatch values, computed using only local aggregation primitives.

---

## **3. Interpretation**

Dispersion captures **how spread out** the mismatch values are:

- **Low dispersion**
  - mismatch values are similar
  - the candidate mode aligns consistently across relations
  - the update is locally coherent

- **High dispersion**
  - mismatch values vary widely
  - the candidate mode aligns with some relations but conflicts with others
  - the candidate mode introduces local inconsistency across relations

Dispersion therefore acts as a **uniformity constraint** on the candidate mode.

---

## **4. Why Dispersion Is Needed**

Mismatch alone can be minimized by aligning strongly with a subset of relations while ignoring others. Dispersion prevents this failure mode by:

- penalizing uneven relational alignment
- encouraging updates that respect the entire neighborhood
- reducing the likelihood of oscillatory or unstable updates
- supporting smooth local coherence descent

Dispersion is especially important when the relational neighborhood is heterogeneous.

---

## **5. Invariant Compatibility**

Dispersion satisfies all LCO invariants:

- **Locality:**
  Computed solely from local mismatch values.

- **Geometry‑neutrality:**
  Uses only dot products and local aggregation; no external geometric or topological structure is assumed.

- **Determinism:**
  Produces consistent values for identical inputs.

- **Well‑definedness:**
  Always finite and bounded for admissible inputs.

- **Unit‑norm mode compatibility:**
  Defined entirely in terms of mismatch values derived from unit‑norm vectors.

- **γ as external parameter:**
  Memory mismatch enters only through its γ‑weighted mismatch term.

---

## **6. Role in the Coherence Score**

Dispersion is one of the core ingredients used by the LCO to construct the coherence score:



\[
C = f(\{\ell_i\}, \ell_{\text{mem}})
\]



It works alongside:

- mismatch
- memory mismatch
- aggregate mismatch

to produce a scalar evaluation of candidate updates.

The exact weighting or combination of dispersion within the scoring function is defined in `lco-operator.md`.

---

## **Status**

Dispersion is a stabilizing, geometry‑neutral component of the LCO.
It ensures that candidate updates maintain coherence across the entire relational neighborhood and supports the operator’s ability to guide local coherence descent.

