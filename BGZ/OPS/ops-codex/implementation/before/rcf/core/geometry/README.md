# **Discrete Geometry Substrate for RCF**

## Document Contract

- owner: RCF discrete geometry substrate
- type: Contract
- scope: formal simplicial and combinatorial geometry grammar
- authority: tier-3 substrate contract subordinate to the RCF invariant kernel
- upstream dependencies: `../S∞.RCF.Invariants.v1.1.md` and `../algebra/README.md`
- status: active
- purpose: define discrete adjacency, orientation, chain, cochain, and topological structures
- anti-backflow: substrate geometry does not define projected geometry, physical geometry, domain semantics, or new core RCF invariants
`/core/geometry/README.md`

This directory defines the **core discrete geometry substrate** used in RCF.

It specifies the minimal combinatorial structures required for adjacency, orientation, calculus, and topological invariants. The required definitions are **substrate‑independent**, free of interpretation, and contain **no required metric, embedding, or semantic assumptions**.

Here, *invariant* means a preserved mathematical property of the substrate, not an addition to the RCF invariant kernel.

---

## **Purpose**

The geometry substrate provides:

- a **simplicial complex** as the discrete carrier of geometric structure
- **orientation rules** for sign‑consistent assembly
- **boundary and coboundary operators** for discrete calculus
- **cochains** as dual objects to chains
- **topological invariants** such as Euler characteristic and homology
- an optional **metric** for modules that require inner products

Together, these define the **formal geometric core** that downstream calculus and analysis modules may depend on.

---

## **Files in this Directory**

| Filename               | Role                                                             | Notes                                           |
|------------------------|------------------------------------------------------------------|-------------------------------------------------|
| `simplicial_complex.md` | Defines the combinatorial structure of simplicial complexes       | Foundation for all geometry                     |
| `orientation.md`         | Specifies orientation conventions for simplices                   | Required for sign‑coherent calculus             |
| `boundary_ops.md`        | Defines boundary operators on chains                             | Forms the chain complex                         |
| `coboundary_ops.md`      | Defines coboundary operators on cochains                         | Dual to boundary; forms the cochain complex     |
| `cochains.md`            | Defines cochains as linear functionals on chains                 | Substrate for discrete differential forms       |
| `topology.md`            | Defines Euler characteristic, homology, and Betti numbers        | Purely combinatorial invariants                 |
| `metrics.md`             | (Optional) Defines symmetric bilinear forms on chains/cochains    | Required only for metric‑dependent modules      |
| `validators.md`          | Defines admissibility checks for geometric consistency            | Ensures all structures are well‑formed          |

---

## **Geometry Dependencies**

The geometry substrate has the following minimal dependency relationships:

```
simplicial_complex.md
        ↓
orientation.md
        ↓
boundary_ops.md → coboundary_ops.md
        ↓                 ↓
      cochains.md     topology.md
        ↓                 ↓
      metrics.md (optional)
        ↓
    validators.md
```

- `simplicial_complex.md` is foundational.
- `orientation.md` depends only on simplices.
- `boundary_ops.md` and `coboundary_ops.md` depend on orientation.
- `cochains.md` depends on chains and coboundary.
- `topology.md` depends on boundary and coboundary.
- `metrics.md` is optional and independent of topology.
- `validators.md` checks consistency across all modules.

---

## **Invariant Guarantees**

The geometry substrate collectively guarantees:

- **closure** under faces and adjacency
- **orientation consistency** across shared faces
- **valid chain and cochain complexes**
- **boundary‑of‑boundary and coboundary‑of‑coboundary identities**
- **well‑defined topological invariants**
- **optional metric structure without geometric interpretation**

These properties form the **formal geometry substrate** used by downstream calculus and by analysis modules that adopt it.

---

## **Scope and Limitations**

This directory does **not** contain:

- embeddings or coordinates
- metric geometry or curvature
- physical or semantic interpretations
- differential equations
- Hodge star, Laplacian, or metric‑dependent operators
- narrative or domain‑specific meaning

Those appear in downstream modules.

---

## **How to Use This Substrate**

Modules in other subsystems (e.g., `/core/calculus/`, `/modules/`, `/tooling/`) may depend on these geometric invariants by reference. When importing from this directory:

- Refer to the **name of the invariant** (e.g., boundary operator, Betti number).
- Do **not** redefine or reinterpret operators.
- Maintain eliminability: modules should *depend on* these invariants, never embed or modify them.

---

## **Summary**

This directory encodes the **discrete geometric fabric** of RCF.
It is:

- **substrate‑independent**
- **invariant‑preserving**
- **implementation‑agnostic**
- **ready for downstream calculus and analysis**

Use these files as the **trusted geometric substrate** for any module requiring adjacency, orientation, or topological structure.
