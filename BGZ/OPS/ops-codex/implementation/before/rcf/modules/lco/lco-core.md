# LCO Core

## Document Contract

- owner: RCF local coherence operator
- type: Contract
- scope: module-local content of `lco/lco-core.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Canonical Definition of the Local Coherence Operator*

This document specifies the **core definition** of the Local Coherence Operator (LCO): its formal role, input–output contract, and the structure of its coherence evaluation. It ties together the primitives and invariants into a single, canonical operator description.

The LCO is an **evaluation operator**: it assigns a scalar coherence score to a candidate local update, given a relational neighborhood, a memory anchor, and an externally supplied memory weight γ. It does **not** choose updates; that is the responsibility of the update policy.

---

## **1. Operator Role**

The LCO implements a mapping of the form:


\[
\mathcal{L} : (\{r_i\}, H, m, \gamma) \mapsto C
\]



Where:

- \(\{r_i\}\) are local relation vectors
- \(H\) is the memory anchor
- \(m\) is the candidate mode
- \(\gamma \in [0,1]\) is the memory weight
- \(C \in \mathbb{R}\) is the coherence score

The coherence score \(C\) quantifies how well the candidate mode \(m\) preserves local relational structure while incorporating the influence of memory.

The mode space is a normalized comparison space only and does not encode any physical or geometric ontology

---

## **2. Inputs**

The LCO accepts the following inputs:

- **Local relations:**
  A finite set of unit‑norm vectors:


\[
  \{r_i\}_{i=1}^N,\quad \|r_i\| = 1
  \]



- **Memory anchor:**
  A unit‑norm vector:


\[
  H,\quad \|H\| = 1
  \]



- **Candidate mode:**
  A unit‑norm vector:


\[
  m,\quad \|m\| = 1
  \]



- **Memory weight:**
  A scalar:


\[
  \gamma \in [0,1]
  \]


  supplied by the temporal-interface module.

All inputs must satisfy the invariants defined in `lco-primitives.md` and `lco-invariants.md`.

---

## **3. Intermediate Quantities**

The LCO may compute the following intermediate quantities:

- **Relation mismatch:**


\[
  \ell_i = 1 - r_i \cdot m
  \]



- **Memory mismatch:**


\[
  \ell_{\text{mem}} = \gamma (1 - H \cdot m)
  \]



- **Aggregate mismatch:**


\[
  \bar{\ell} = \text{mean}(\{\ell_i\} \cup \{\ell_{\text{mem}}\})
  \]



- **Dispersion of mismatch:**


\[
  \text{Disp} = \text{mean}\big((\ell_k - \bar{\ell})^2\big), \quad k \in \{1,\dots,N,\text{mem}\}
  \]


  where \(\ell_k\) ranges over all mismatch terms (relations and memory).

These constructions use only the allowed primitives: dot products, local aggregation, and γ‑weighted memory.

---

## **4. Coherence Score Structure**

The LCO produces a scalar coherence score \(C\) as a function of mismatch, dispersion, and memory terms. A canonical form consistent with the primitives and invariants is:



\[
C = f(\{\ell_i\}, \ell_{\text{mem}})
\]



where \(f\) is constructed from:

- the mean mismatch \(\bar{\ell}\)
- the dispersion \(\text{Disp}\)
- the γ‑weighted memory mismatch \(\ell_{\text{mem}}\)

A simple example consistent with this structure is:


\[
C = \bar{\ell} + \text{Disp}
\]



The exact choice of \(f\) is specified in the operator’s scoring definition, but must:

- be expressible purely in terms of the allowed primitives
- produce a finite, well‑defined scalar for all admissible inputs
- be compatible with local coherence descent under admissible update policies

The aggregation treats relation and memory mismatch terms uniformly unless otherwise specified by the scoring function ( \(f\) ).

Lower values of \(𝐶\) correspond to higher local coherence under this example scoring choice.

---

## **5. Operator Contract**

Given admissible inputs \((\{r_i\}, H, m, \gamma)\), the LCO must:

- **Produce:**
  A scalar coherence score \(C \in \mathbb{R}\).

- **Respect invariants:**
  - Unit‑norm mode space
  - Locality
  - Geometry‑neutrality
  - Determinism
  - Well‑defined score
  - Independence from φ/ε and global geometry
  - γ treated as external

- **Remain policy‑agnostic:**
  The LCO does not decide whether an update is accepted or rejected. It only provides the score that an update policy may use.

The operator must not perform any normalization or transformation that depends on non-local information or global statistics.

---

## **6. Separation from Update Policy**

The LCO core defines **only** the scoring operator:


\[
\mathcal{L}(\{r_i\}, H, m, \gamma) \rightarrow C
\]



Any logic that:

- compares multiple candidate modes
- selects an update
- introduces stochasticity
- enforces thresholds or admissibility rules

belongs to the **update policy**, not to the LCO core.

---

## **Status**

This document defines the canonical form and behavioral contract of the Local Coherence Operator. All implementations of the LCO must conform to this structure and remain expressible solely in terms of the primitives and invariants defined in the corresponding module documents.

