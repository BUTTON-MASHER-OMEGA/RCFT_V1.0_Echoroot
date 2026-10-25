# Global Coherence Metrics

## Document Contract

- owner: RCF global coherence operator
- type: Interpretation
- scope: module-local content of `gco/global-coherence-metrics.md`
- authority: tier-6 derived aggregation layer
- upstream dependencies: `/modules/gco/interface.md`
- status: active
- purpose: summarize recursive LCO outcomes without local scoring, control, or temporal ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Metrics over Accumulated Local Coherence Behavior*

Global coherence metrics quantify how coherent a system is when viewed over long horizons and across nested scales.

These metrics are computed from:

- accumulated LCO outcomes,
- relational structure across systems and subsystems,
- temporal sequences of local coherence behavior.

They are **relational**, **global**, and **geometry‑neutral**.

---

## 1. Inputs

Metrics are derived from:

- local mismatch scores over time,
- accepted vs rejected updates,
- cost trajectories,
- subsystem–container relationships,
- cross‑system relational links.

They do **not** depend on:

- geometric distances,
- curvature,
- explicit metrics or coordinates,
- direct temporal modification.

---

## 2. Example Metric Families

### 2.1 Long‑Horizon Mismatch Accumulation

A measure of how much local mismatch accumulates over many events:

- persistent high mismatch → global incoherence,
- decaying mismatch → global stabilization.

### 2.2 Cross‑Scale Consistency

A measure of alignment between:

- subsystem coherence patterns, and
- container‑level coherence patterns.

High cross‑scale consistency → structurally aligned.
Low consistency → structural tension.

### 2.3 Structural Drift

A measure of how quickly relational patterns change under continued LCO dynamics:

- slow drift → stable structure,
- rapid drift → unstable or chaotic structure.

---

## 3. Properties

Global coherence metrics must be:

- **global** (integrate over many events and scales),
- **relational** (based on relations, not geometry),
- **slow** (change on longer timescales than LCO),
- **derived** (computed from LCO behavior, not assumed).

---

## 4. Outputs

The metrics produce:

- scalar scores,
- trend indicators (improving / degrading),
- cross‑scale consistency measures.

These are consumed by:

- `persistence_detection.md` (to identify stable patterns),
- `structural_viability.md` (to classify global states).

---

## 5. Status

Global coherence metrics provide the quantitative backbone for GCO.
They summarize long‑horizon, cross‑scale behavior of LCO without introducing new laws.

