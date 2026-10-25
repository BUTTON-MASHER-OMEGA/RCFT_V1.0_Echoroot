# Metric Module

## Document Contract

- owner: RCF metric module
- type: Status
- scope: module-local content of `metric/README.md`
- authority: tier-8 optional representational metric layer
- upstream dependencies: `/modules/metric/interface/representational_contract.md`
- status: active
- purpose: construct optional representational metrics without upstream influence or claims of physical geometry
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

### Optional Representational Metrics on the Π‑Shadow

---

## purpose

The metric module defines **optional representational metrics** on the
continuous shadow produced by Π.

It introduces **no new primitives** and does **not** influence upstream
modules. All metric structures are downstream, Π‑determined, and
interpretation‑neutral.

---

## role in the architecture

The metric module sits **downstream** of geometry‑continuous and
geometry‑dynamics:

```text
geometry‑discrete
      ↓
projection (Π)
      ↓
geometry‑continuous
      ↓
geometry‑dynamics (mismatch, curvature, flow, action)
      ↓
metric (this module)
      ↓
metric/interface
      ↓
downstream metric‑aware / domain modules
```

Its role is to provide **representational distance assignments** that
encode relational structure already present in the Π‑shadow.

---

## what the module provides

The module defines several representational metrics:

- `adjacency_metric.md`
  Describes an adjacency‑based metric that reflects Π‑smoothed
  relational proximity.

- `slush_metric.md`
  Describes a slush‑based metric that corresponds to how slush distribution
  and mismatch‑related variation appear in the shadow.

- `plateau_metric.md`
  Describes a plateau‑based metric that reflects γ‑plateau organization
  and grouping.

- `effective_metric.md`
  provides a unified representational assignment over adjacency, slush, and plateau information into a single representational assignment.

Supporting structure:

- `slush_mapping.md`, `slush_effects.md`
  Record how slush‑related structure appears in the shadow and how it
  can be represented by optional metrics.

- `metric_invariants.md`
  Records the invariants that all metric constructions must satisfy:
  downstream‑only, Π‑determined, coordinate‑optional, and
  non‑interpretive.

- `metric_outputs.md`
  Describes what representational outputs (distance assignments,
  optional tensors, etc.) may be exposed to downstream modules.

- `interface/`
  Defines allowed inputs, allowed outputs, downstream hooks, and the
  representational contract for all metric usage.

---

## constraints and non‑claims

The metric module:

- does **not** define curvature, dynamics, or physical distance,
- does **not** assume a manifold or geometric ontology,
- does **not** introduce new objects, forces, or fields,
- does **not** modify or constrain upstream modules.

All metric structures are:

- **Π‑determined** (traceable to adjacency, mismatch, slush, plateau,
  and survivor structure),
- **coordinate‑free at the core**,
- **metric‑optional** (downstream may use them, but RCF does not
  require them),
- **eliminable** without damaging the core framework.

---

## scope

All statements in this module describe:

> **representational metrics on the Π‑shadow of survivor geometries
> under the current RCF formalism**

They:

- encode relational structure already present in upstream modules,
- remain downstream and optional,
- and carry no physical or geometric interpretation within RCF.

---

## status

The metric module is structurally complete for current evaluations and
is available for downstream visualization, analysis, and domain-specific
modules through the metric interface.
