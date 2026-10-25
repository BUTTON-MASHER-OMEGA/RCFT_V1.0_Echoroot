# update-algebra.md

## Document Contract

- owner: RCF discrete geometry module
- type: Contract
- scope: module-local content of `geometry/discrete/update-algebra.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose
This document records the update algebra used in current discrete‑geometry evaluations. It defines the structural assumptions under which recursive updates are analyzed and provides the algebraic context for spinor embeddings and closure tests.

This file does not interpret the algebra; it documents the formal structure used in current evaluations.

---

## overview
In the current formalism, local relational updates are treated as:

- distance‑preserving transformations on a 4D relational state space
- represented via the continuous group \( SO(4) \)
- evaluated through its spin cover \( Spin(4) \cong SU(2) \times SU(2) \)

Quaternion composition provides a convenient representation of these update steps.

---

## induced discrete update class
When a discrete geometry is embedded into this algebra:

- each vertex corresponds to a discrete relational mode
- sequential updates correspond to compositions in the induced algebra
- closure is evaluated by checking whether compositions remain within the mapped vertex set

Geometries that are not closed under this induced composition show reduced persistence in current tests.

---

## role in RCF (structural only)
- provides the algebraic context for spinor embeddings
- defines the update structure used in survival‑sieve evaluations
- supports the analysis of mismatch accumulation under recursion

This file records the algebraic framework only.
Interpretations are handled downstream.

---

## invariants
- the update algebra is treated as distance‑preserving in current evaluations
- closure is evaluated relative to this algebraic structure
- no physical interpretation is assumed or implied

