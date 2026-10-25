# **Discrete Calculus Substrate for RCF**

## Document Contract

- owner: RCF discrete calculus substrate
- type: Contract
- scope: formal discrete-operator grammar
- authority: tier-3 substrate contract subordinate to the RCF invariant kernel
- upstream dependencies: `../S∞.RCF.Invariants.v1.1.md` and `../geometry/README.md`
- status: active
- purpose: define discrete operators and diagnostics derived from chain-cochain structure
- anti-backflow: calculus operators do not define RCF Time, physical dynamics, metric geometry, or new core invariants
`/core/calculus/README.md`

This directory defines the **core discrete calculus substrate** used in RCF.

It specifies the minimal operator structures derived from the chain/cochain complex of the geometry layer. All definitions are **algebraic and combinatorial**, with **no metric, embedding, or semantic assumptions**.

---

## **Purpose**

The calculus substrate provides:

- **discrete differential operators** (gradient, curl, divergence)
- **operator identities** derived from the chain/cochain complex
- **refinement‑safe event structure**
- **Jacobian‑based scalar invariants** for admissible transformations

Together, these define the **formal calculus core** that downstream analytic modules may depend on.

Here, *invariant* means an operator identity or profile-local scalar property, not an addition to the RCF invariant kernel.

---

## **Files in this Directory**

| Filename                 | Role                                                                 | Notes                                              |
|--------------------------|----------------------------------------------------------------------|----------------------------------------------------|
| `discrete_ops.md`        | Defines gradient, curl, divergence from δ and ∂                      | Purely topological; no metric required             |
| `identities.md`          | Formalizes ∂² = 0 and δ² = 0 and their consequences                  | Ensures operator coherence                         |
| `event_axiom.md`         | Defines minimal refinement‑atom transitions                          | No ontology; granularity constraint only           |
| `jacobian_entropy.md`    | Defines a structural scalar from admissible transformations           | Optional diagnostic; no geometric interpretation   |

---

## **Calculus Dependencies**

The calculus substrate depends directly on the geometry layer:

```
simplicial_complex.md
        ↓
orientation.md
        ↓
boundary_ops.md → coboundary_ops.md
        ↓                 ↓
      cochains.md     topology.md
        ↓                 ↓
   discrete_ops.md   identities.md
        ↓                 ↓
   event_axiom.md   jacobian_entropy.md
```

- `discrete_ops.md` depends on boundary and coboundary operators.
- `identities.md` depends on the chain/cochain complex.
- `event_axiom.md` depends on refinement structure, not geometry.
- `jacobian_entropy.md` depends only on local linear transformations.

No file in this directory introduces geometry or semantics.

---

## **Invariant Guarantees**

The calculus substrate collectively guarantees:

- **grad, curl, div** are well‑defined from δ and ∂
- **curl(grad) = 0** and **div(curl) = 0**
- **operator identities** preserve topological consistency
- **events** provide minimal refinement granularity
- **Jacobian entropy** is coordinate‑invariant and refinement‑stable

These properties form the **formal calculus core** of RCF.

---

## **Scope and Limitations**

This directory does **not** contain:

- metric‑dependent operators (Hodge star, Laplacian)
- embeddings or coordinate systems
- physical or semantic interpretations of operators
- differential equations or flows
- domain‑specific meaning (physics, ML, cognition, etc.)

Those appear in downstream modules.

---

## **How to Use This Substrate**

Modules in other subsystems (e.g., `/core/analysis/`, `/modules/`, `/tooling/`) may depend on these calculus invariants by reference. When importing from this directory:

- Refer to the **operator name** (e.g., gradient, curl, divergence).
- Do **not** redefine or reinterpret ∂, δ, or their identities.
- Maintain eliminability: modules should *depend on* these invariants, never embed or modify them.

---

## **Summary**

This directory encodes the **discrete calculus fabric** of RCF.
It is:

- **substrate‑independent**
- **invariant‑preserving**
- **implementation‑agnostic**
- **ready for downstream analytic modules**

Use these files as the **trusted calculus substrate** for any module requiring discrete differential structure.
