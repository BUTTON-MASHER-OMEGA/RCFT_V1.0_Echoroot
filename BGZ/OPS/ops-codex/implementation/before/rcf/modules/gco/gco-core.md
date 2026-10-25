# GCO Core

## Document Contract

- owner: RCF global coherence operator
- type: Contract
- scope: module-local content of `gco/gco-core.md`
- authority: tier-6 derived aggregation layer
- upstream dependencies: `/modules/gco/interface.md`
- status: active
- purpose: summarize recursive LCO outcomes without local scoring, control, or temporal ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Aggregation and Multiscale Coherence Summary*

The GCO Core defines **Global Coherence** as an emergent summary of recursively applied Local Coherence Operator (LCO) dynamics across nested systems.

GCO is not a new execution rule.
It is a **multiscale aggregation layer** that evaluates the long‑horizon, cross‑scale consequences of local coherence behavior.

---

## 1. Inputs

GCO consumes:

- **Local coherence outcomes** from LCO:
  - mismatch scores,
  - accepted/rejected updates,
  - local cost profiles.

- **Relational structure**  (as observed through LCO outcomes and cross-scale relationships):
  - subsystem–container relationships,
  - cross‑system links,
  - nested system hierarchy.

- **Temporal context (read‑only)**:
  - event indices,
  - refinement steps,
  - temporal weights (γ, ε) as parameters only.

GCO does **not**:

- modify time,
- modify γ or ε,
- perform local updates,
- define geometry.

---

## 2. Aggregation Role

GCO aggregates:

- local coherence behavior over **time** (many events),
- local coherence behavior over **scale** (subsystems and containers),
- cross‑scale consistency between nested systems.

Conceptually:

- LCO acts locally and frequently.
- GCO acts infrequently and globally, summarizing what LCO has done.

---

## 3. Multiscale Coherence Summary

GCO constructs:

- **global coherence metrics** (see `global-coherence-metrics.md`),
- **persistence signatures** (see `persistence-detection.md`),
- **structural viability states** (see `structural-viability.md`).

These are derived from:

- accumulated mismatch,
- pattern persistence,
- cross‑scale relational consistency.

---

## 4. No New Law

GCO does **not**:

- introduce a new coherence law,
- override LCO decisions,
- define new admissibility rules,
- act as a scheduler or controller.

It is a **descriptive layer**, not a prescriptive one.

---

## 5. Outputs

GCO outputs:

- global coherence scores,
- persistence indicators,
- structural viability classifications.

These outputs are consumed by:

- policy modules,
- control layers,
- geometry/physics modules,

which may interpret or act on them, but that logic is **outside** GCO.

---

## 6. Status

The GCO Core defines global coherence as an emergent, multiscale summary of LCO dynamics.
It is the structural reporting layer for long‑horizon, cross‑scale coherence.

