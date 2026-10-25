# Representational Contract

## Document Contract

- owner: RCF metric module
- type: Contract
- scope: module-local content of `metric/interface/representational_contract.md`
- authority: tier-8 optional representational metric layer
- upstream dependencies: `/modules/geometry/continuous/README.md` and `/modules/geometry/dynamics/README.md`
- status: active
- purpose: construct optional representational metrics without upstream influence or claims of physical geometry
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

### `interface/representational_contract.md`


## purpose

This document defines the **non‑negotiable rules** governing how metric
outputs may be interpreted and used downstream.

The contract ensures that all metric usage remains:

- downstream‑only,
- representational,
- Π‑determined,
- and non‑interpretive.

---

## 1. no ontology claims

Metric outputs:

- do not assert that a manifold exists,
- do not assert that physical space is metric,
- do not define geometric objects as ontological entities.

They are encodings of relational structure in the Π‑shadow.

---

## 2. no dynamics or causality

Metric outputs:

- do not define evolution equations,
- do not specify causes or forces,
- do not encode forward‑looking optimization.

Any appearance of dynamics must come from downstream modules that
explicitly own those assumptions.

---

## 3. no backflow

Downstream interpretations built on metric outputs:

- may not change metric behavior,
- may not alter upstream modules,
- may not introduce new invariants or primitives upstream.

All such interpretations are strictly downstream and eliminable.

---

## 4. Π‑determined structure

All metric outputs are required to remain traceable to:

- adjacency,
- mismatch and slush,
- plateau and sector structure,
- and Π’s smoothing behavior.

If a structure cannot be traced to these inputs, it violates the
representational contract.

---

## 5. coordinate‑optional usage

Coordinates, tensors, and related encodings:

- are optional conveniences,
- do not carry intrinsic meaning,
- and must not be treated as fundamental structure.

Downstream modules must be able to refer back to coordinate‑free
descriptions.

---

## 6. compatibility and comparability

Multiple metric constructions:

- may coexist,
- may be compared descriptively,
- may be used jointly by downstream modules.

No single metric may be declared “the true” metric within RCF through
this interface.

---

## 7. eliminability

If the entire metric module is removed:

- upstream RCF remains structurally intact,
- invariants and dynamics (in the RCF sense) are unaffected,
- only optional representational convenience is lost.

Downstream modules must be written so that their core logic does not
depend on a specific metric interpretation.

---

## status

The representational contract is binding for all metric/interface
consumers. Any extension of the interface must preserve these
constraints and remain eliminable without damaging upstream structure.

