# Local Constraint Density

## Document Contract

- owner: RCF temporal interface
- type: Interpretation
- scope: module-local content of `temporal-interface/local-constraint-density.md`
- authority: tier-5 temporal parameter bridge
- upstream dependencies: `/modules/temporal-interface/interface.md`
- status: active
- purpose: expose sealed temporal parameters without inferring geometry or retuning upstream parameters
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Relational Admissibility and Mode‑Space Narrowing*

Constraint density is a scalar that quantifies how constrained a system is within its local relational neighborhood.
It measures how many admissible modes remain available for the next refinement step.

Constraint density is defined over admissible sets that are finite or effectively finite under relational constraints.

Mode‑Space Narrowing refers to the reduction of the set of admissible modes under relational constraints.

Constraint density is:
- **local**,
- **relation‑dependent**,
- **geometry‑neutral**,
- **operator‑adjacent**,
- **upstream of temporal weighting**,
- **downstream of the Time Core**.

It does not define geometry, basins, attractors, or global structure.

---

## 1. Purpose

Constraint density answers:

> *How hard is it for this system to advance to its next admissible state?*

High constraint density → narrow admissible mode set → slow advancement.
Low constraint density → wide admissible mode set → fast advancement.

---

## 2. Inputs

Constraint density is computed from:
- the local relational neighborhood,
- admissibility rules,
- mode‑space narrowing,
- relational compatibility constraints.

It does **not** depend on:
- geometry,
- curvature,
- GCO outputs,
- LCO mismatch scores,
- global coherence.

---

## 3. Definition (Conceptual)

Let \( \mathcal{A}_n \) be the set of admissible next states at event \( n \).

Define constraint density \( \kappa_n \) as a monotone function of:

- the size of \( \mathcal{A}_n \),
- the degree of relational pruning required,
- the narrowing of admissible modes.

Examples (conceptual, not geometric):

- **Void:**
  \( \mathcal{A}_n \) large → \( \kappa_n \) low

- **Highly constrained region:**
  \( \mathcal{A}_n \) small → \( \kappa_n \) high

- **Transition region:**
  intermediate \( \kappa_n \)

---

## 4. Properties

Constraint density must be:

- **local** (computed per system or region),
- **bounded** (finite range),
- **monotone** (more constraints → higher density),
- **geometry‑neutral**,
- **operator‑independent** (not derived from LCO/GCO outputs).

---

## 5. Outputs

The output is a single scalar:



\[
\kappa_n \in [0, \kappa_{\max}]
\]



Constraint density **does not generate γ or ε**.
Instead, it **bounds and contextualizes** the admissible temporal‑weighting regime used by the Temporal Weights Adapter.

---

## 6. Status

Constraint density is the relational input that constrains temporal weighting.
It quantifies local refinement difficulty without invoking geometry or global structure.

