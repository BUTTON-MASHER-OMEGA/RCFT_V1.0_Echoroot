# Allowed Outputs

## Document Contract

- owner: RCF metric module
- type: Contract
- scope: module-local content of `metric/interface/allowed_outputs.md`
- authority: tier-8 optional representational metric layer
- upstream dependencies: `/modules/metric/interface/representational_contract.md`
- status: active
- purpose: construct optional representational metrics without upstream influence or claims of physical geometry
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose

This document records what **representational outputs** the metric
module is permitted to expose to downstream consumers.

All outputs are descriptive, Π‑determined, and non‑interpretive.

---

## 1. representational distance assignments

The metric module may expose:

- adjacency‑based distances,
- slush‑based distances,
- plateau‑based distances,
- effective composite distances.

Each is represented as:

\[
d_\star(x, y)
\]

where:

- \(x, y\) are points in the continuous shadow,
- the value reflects a chosen relational structure (adjacency, slush,
  plateau, or composite),
- and no geometric or physical meaning is assumed.

---

## 2. optional coordinate representations

Downstream modules may use coordinate encodings of metric
assignments. The metric module may therefore expose:

- coordinate charts used to encode \(d_\star(x, y)\),
- matrix or tensor encodings of representational metrics in a chosen
  chart.

These representations:

- are optional,
- do not influence upstream modules,
- and do not imply manifold or physical structure.

---

## 3. optional dual and metric‑adjacent structures

The metric module may expose:

- representational dual vectors and covectors,
- representational norms and inner products,
- representational angle assignments.

All such outputs:

- are constructed from allowed metric assignments,
- remain descriptive encodings,
- and carry no physical or geometric interpretation.

---

## 4. composite and effective structures

The metric module may expose:

- effective metrics that combine adjacency, slush, and plateau
  information,
- derived distance functions built from multiple representational
  metrics,
- summary descriptors (e.g., “regions with higher effective distance
  variation”) that remain structural and relational.

No governing laws or dynamics are exposed.

---

## 5. forbidden outputs

The metric module may **not** expose:

- forces, fields, or potentials,
- curvature as a governing object,
- dynamical equations or evolution laws,
- claims about physical distance or experimental observables.

Any such content must live in downstream, non‑canonical lenses and may
not be expressed through this interface.

---

## status

The allowed outputs list is structurally complete for current metric
constructions and downstream hooks. Extensions must preserve
representational neutrality and downstream‑only behavior.

