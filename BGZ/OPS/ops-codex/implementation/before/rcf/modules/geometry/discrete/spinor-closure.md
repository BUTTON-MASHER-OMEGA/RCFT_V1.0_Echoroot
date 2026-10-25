# spinor-closure.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/spinor-closure.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## **purpose**
This document defines the algebraic closure requirement used by the current spinor-based survivor profile. Where an LCO-evaluated update is represented by a spinor composition, a candidate must remain stable under that declared composition to pass this profile.

Spinor closure acts as a strong filter in the survival sieve.

---

## **scope**
This file covers:

### **1. the update rule**
In current formulations where sequential updates are represented as spinor compositions.
A geometry adopting this profile must support the declared compositions without leaving its candidate set.

### **2. closure requirement**
A candidate geometry must form a multiplicatively closed set under the relevant spinor operations. If repeated compositions produce vertices not present in the geometry’s adjacency structure, closure fails.

### **3. consequences of failure**
When closure fails:
- mismatch accumulates
- local coherence collapses
- recursion destabilizes
- the geometry falls out of survival statistics

### **4. known closures**
Among tested geometries:
- the **600‑cell** exhibits perfect closure via the unit icosians
- the **dual 5‑demicube** remains stable in its density regime
- the **E₈ $4_{21}$** forms a complete and stable spinor structure at extreme density

---

## **invariants**
- Closure is algebraic, not geometric.
- Geometries that fail closure show reduced persistence under recursion in current representations.
- Closure is evaluated independently of projection or continuous interpretation.
- Only forced invariants are recorded; interpretations are downstream.
