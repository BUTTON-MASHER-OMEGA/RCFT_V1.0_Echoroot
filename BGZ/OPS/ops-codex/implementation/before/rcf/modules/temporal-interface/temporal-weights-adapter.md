# Temporal Weights Adapter

## Document Contract

- owner: RCF temporal interface
- type: Interpretation
- scope: module-local content of `temporal-interface/temporal-weights-adapter.md`
- authority: tier-5 temporal parameter bridge
- upstream dependencies: `/modules/temporal-interface/interface.md`
- status: active
- purpose: expose sealed temporal parameters without inferring geometry or retuning upstream parameters
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Bounding γ, ε, and Local Pacing from Constraint Density*

The Temporal Weights Adapter uses **constraint density** to **bound and contextualize** the admissible ranges of temporal parameters:

- γ (memory drag),
- ε (freedom to move),
- optional local pacing terms.

These parameters are **sealed**: the adapter does **not** compute or generate them.
It **exposes** γ, ε, and pacing terms by **selecting** values from admissible ranges restricted by constraint density.

---

## 1. Purpose

Given constraint density \( \kappa_n \), the adapter answers:

> *What temporal‑weighting regime is admissible for this local relational context?*

It provides lawful **bounding**, not derivation.

---

## 2. Inputs

- constraint density \( \kappa_n \)

It does **not** consume:
- geometry,
- curvature,
- GCO outputs,
- LCO mismatch,
- global coherence,
- attractor/basin identification.

---

## 3. Exposed Temporal Parameters

The adapter **exposes** temporal parameters by selecting them from sealed admissible ranges.

### 3.1 Memory Drag (γ)



\[
\gamma_n \in \Gamma[\kappa_n]
\]



Higher constraint density → γ is selected from a **higher** admissible range.
Lower constraint density → γ is selected from a **lower** admissible range.

γ remains a **sealed temporal parameter**, not a derived quantity.

---

### 3.2 Freedom to Move (ε)



\[
\varepsilon_n \in \mathcal{E}[\kappa_n]
\]



Higher constraint density → ε is selected from a **lower** admissible range.
Lower constraint density → ε is selected from a **higher** admissible range.

ε is likewise **bounded**, not computed.

---

### 3.3 Local Pacing (optional)



\[
\pi_n \in \Pi[\kappa_n]
\]



This term modulates effective refinement difficulty without modifying event index or temporal ontology.

---

## 4. Properties

The bounding mechanism must be:

- **monotone** (more constraints → higher γ ranges, lower ε ranges),
- **bounded**,
- **local**,
- **geometry‑neutral**,
- **operator‑independent**,
- **non‑diagnostic** (does not infer basins or attractors).

---

## 5. Invariant

The adapter defines **local temporal admissibility**, not global time.
Constraint density **restricts** the temporal‑weighting regime but does **not** generate γ or ε.

---

## 6. Status

The Temporal Weights Adapter is the lawful bridge between relational constraint and temporal weighting.
It exposes sealed temporal parameters without deriving them.

