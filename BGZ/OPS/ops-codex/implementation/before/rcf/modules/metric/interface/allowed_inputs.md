# Allowed Inputs

## Document Contract

- owner: RCF metric module
- type: Contract
- scope: module-local content of `metric/interface/allowed_inputs.md`
- authority: tier-8 optional representational metric layer
- upstream dependencies: `/modules/metric/interface/representational_contract.md`
- status: active
- purpose: construct optional representational metrics without upstream influence or claims of physical geometry
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose

This document records the **upstream structures** that the metric module
is permitted to reference when constructing representational metrics.

No additional inputs are allowed. This prevents hidden dependencies
and upstream contamination.

---

## 1. continuous shadow

The metric module may reference:

- the continuous shadow structure from geometry‑continuous,
- adjacency‑derived neighborhood information,
- and any documented continuous invariants that are Π‑determined
  and coordinate‑free.

The shadow is treated as:

- a relational structure,
- with no assumed manifold or physical geometry.

---

## 2. mismatch and slush structure

From geometry‑dynamics and slush documentation, the metric module may
reference:

- local mismatch density and its relational gradients,
- slush distribution and its documented gradients,
- survivor‑consistent structure related to mismatch and slush.

These are treated as:

- relational densities and variations,
- not as fields with physical interpretation.

---

## 3. plateau and sector structure

From upstream sector and plateau documentation, the metric module may
reference:

- γ‑plateau assignments,
- plateau boundaries and grouping behavior,
- sector adjacency and ordering,
- survivor‑consistent sector / plateau structure.

These inputs are used only to describe representational grouping and
separation.

---

## 4. adjacency structure

From projection and geometry‑discrete:

- adjacency relations in the discrete layer,
- Π‑smoothed adjacency neighborhoods in the continuous shadow,
- relational distance classes derived from hop counts.

The metric module may not introduce new adjacency notions; it only
references the ones provided upstream.

---

## 5. Π and smoothing behavior

The metric module may rely on:

- documented properties of Π’s smoothing behavior,
- how Π maps discrete relations to continuous neighborhoods,
- how Π preserves or smooths invariants.

No extensions to Π are defined here; only documented behavior may
be referenced.

---

## 6. forbidden inputs

The metric module may **not** depend on:

- external physical models,
- domain‑specific laws or constants,
- coordinate charts or chosen frames,
- additional primitives beyond those documented upstream.

If an input is not explicitly listed above, it is treated as disallowed.

---

## status

The allowed inputs list is structurally complete for current metric
constructions. Any extension must remain Π‑determined, coordinate‑free,
and non‑interpretive.

