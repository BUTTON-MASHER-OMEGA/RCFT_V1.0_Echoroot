# mismatch/mismatch-core.md

## Document Contract

- owner: RCF geometry-dynamics module
- type: Contract
- scope: module-local content of `geometry/dynamics/mismatch/mismatch-core.md`
- authority: tier-8 downstream continuous-variation layer
- upstream dependencies: `/modules/geometry/dynamics/interface.md`
- status: active
- purpose: describe continuous relational variation without introducing forces, fields, or upstream primitives
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
This document defines the continuous mismatch functional used throughout the geometry‑dynamics module.
Mismatch is the **first‑order structural quantity** inherited from the discrete mismatch functional \(F(G,\gamma)\).
It introduces **no new primitives** and remains fully determined by:

- discrete adjacency,
- survivor structure,
- slush redistribution,
- and Π’s smoothing rules.

Mismatch serves as the local density referenced by curvature, flow, and the action integral.

---

## upstream origin
The continuous mismatch functional is a **shadow** of the discrete mismatch evaluation.

From upstream documentation:

> “The mismatch functional \(F(G,\gamma)\) evaluates the structural penalty of a local geometry.”

Under projection:

- adjacency is smoothed,
- slush redistribution is applied,
- and discrete mismatch values are translated into a continuous density.

No additional structure is introduced during this translation.

---

## definition
Let \(\Phi\) denote the continuous shadow produced by Π.
The continuous mismatch functional \(L[\Phi]\) is defined as:

- a **local**,
- **coordinate‑free**,
- **Π‑determined**
representation of discrete constraint deviation.

Formally:

- \(L\) assigns a mismatch density to each point of the shadow,
- this density reflects the smoothed discrete mismatch of the corresponding survivor region,
- and no metric, coordinate chart, or external structure is required.

The functional is **relational**: it depends only on adjacency relations and their smoothed variation.

---

## structural properties

### 1. locality
Mismatch is locally defined.
Each value of \(L\) depends only on:

- nearby adjacency relations,
- local slush redistribution,
- and the immediate survivor neighborhood.

### 2. Π‑determinism
All mismatch values are determined by:

- the discrete mismatch functional \(F(G,\gamma)\),
- the survivor sequence,
- and Π’s smoothing rules.

No hidden variables or external laws contribute.

### 3. coordinate‑free behavior
Mismatch does not rely on:

- coordinate charts,
- basis choices,
- or affine structure.

All variation is relational.

### 4. metric‑optional behavior
Mismatch does not require a metric.
Downstream modules may introduce one, but \(L\) is defined independently of any metric choice.

### 5. monotonicity under smoothing
Under Π’s smoothing rules:

- In current evaluations, mismatch values are observed not to increase under Π’s smoothing rules.
- local variation is redistributed but not amplified,
- and the continuous density reflects the survivor‑only remainder.

This behavior is consistent with observed properties of discrete mismatch under projection.

---

## mismatch gradients
The relational gradient of mismatch is described without coordinates or metric structure.

It reflects:

- local variation of mismatch density,
- adjacency‑based deviation,
- and slush‑sensitive redistribution.

This gradient is the input to:

- curvature (second‑order response),
- flow is observed to reduce mismatch gradients under Π-consistent evolution,
- and the action density.

No differential‑geometric assumptions are introduced.

---

## role in geometry‑dynamics
Mismatch provides the foundational structure for all downstream operators:

- **curvature** evaluates second‑order deviation of mismatch gradients,
- **flow** smooths mismatch gradients under Π‑consistent evolution,
- **least‑action** integrates mismatch density to form the action functional.

All of these operators remain shadows of mismatch and do not introduce new primitives.

---

## invariants
Mismatch satisfies the invariants of the geometry‑dynamics module:

- no new primitives,
- coordinate‑free formulation,
- metric‑optional structure,
- Π‑determinism,
- locality,
- relational structure only,
- no physical interpretation.

---

## status
The mismatch definition is structurally complete and ready for use by curvature, flow, and least‑action components.

