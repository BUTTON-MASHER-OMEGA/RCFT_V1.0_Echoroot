# Metric Interface

## Document Contract

- owner: RCF metric module
- type: Status
- scope: module-local content of `metric/interface/README.md`
- authority: tier-8 optional representational metric layer
- upstream dependencies: `/modules/metric/interface/representational_contract.md`
- status: active
- purpose: construct optional representational metrics without upstream influence or claims of physical geometry
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose

The metric interface defines how downstream modules may **request** and
**consume** representational metric structure.

It does not define new metrics, introduce new primitives, or modify
upstream behavior. It only specifies:

- which inputs the metric module may read,
- which outputs it may expose,
- and how those outputs may be used representationally.

The interface is downstream of:

- geometry‑continuous,
- geometry‑dynamics,
- and slush / plateau structure as documented upstream.

---

## position in the architecture

The interface sits between the metric module and any downstream
metric‑aware or domain‑specific components.

```text
geometry‑discrete
      ↓
projection (Π)
      ↓
geometry‑continuous
      ↓
geometry‑dynamics (mismatch, curvature, flow, action)
      ↓
metric (adjacency / slush / plateau / effective)
      ↓
metric/interface   ← (this module)
      ↓
downstream metric‑aware modules (visualization, analysis, domains)
```
