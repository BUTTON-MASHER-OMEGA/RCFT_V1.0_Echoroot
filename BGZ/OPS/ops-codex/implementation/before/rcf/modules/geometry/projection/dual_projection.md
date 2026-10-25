# modules/projection/dual_projection.md

## Document Contract

- owner: RCF projection module
- type: Interpretation
- scope: module-local content of `geometry/projection/dual_projection.md`
- authority: tier-7 discrete-to-continuous handoff
- upstream dependencies: `/modules/geometry/projection/interface.md`
- status: active
- purpose: define what projection acts on, preserves, smooths, and exposes without selecting survivors or defining downstream metrics
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

### **purpose**
To define the bidirectional projection operators that relate discrete relational configurations to their continuous shadows. This document formalizes the structural contract required for dual closure within the projection layer.

This file describes projection behavior only.
It does **not** assign physical interpretation or specify downstream spectral consequences.

Π↑ and Π↓ operate on the same discrete substrate; dual projection does not introduce new geometries or depend on downstream structures.

Π↑ reveals continuous relational structure; downstream modules may interpret some of this structure geometrically, but Π↑ itself does not define a metric

---

## **1. the bidirectional operators**

The projection operator consists of two interdependent mappings:

### **Π↑ — Forward Projection**
Maps a discrete relational configuration into a continuous shadow.
This includes:

- smoothing of adjacency structure
- exposes continuous relational structure without defining geometric objects
- extraction of continuous spectral signatures

Π↑ does not introduce new combinatorics; it provides a continuous representation of the existing discrete structure.

### **Π↓ — Back Projection**
maps continuous structural constraints into consistency conditions that must already be satisfied by admissible discrete configurations.
This includes:

- enforcing admissibility conditions derived from the continuous shadow
- rejecting discrete states whose projected behavior violates structural limits
- ensuring compatibility between discrete updates and continuous constraints

Π↓ does not generate new discrete structures; it filters existing ones.

---

## **2. the fixed‑point condition (structural form)**

A discrete configuration is considered **dually closed** when it remains admissible under both mappings:

\[
\text{Admissible}(G) \iff G = \Pi_{\downarrow}(\Pi_{\uparrow}(G))
\]

This fixed‑point condition expresses:

- **Algebraic Closure:** no contradictions introduced by projection
- **Geometric Closure:** no violations of continuous constraints

The projection module does not specify which configurations satisfy this condition; it defines the rule under which they are evaluated.

---

## **3. structural consequences (scope‑limited)**

Within the present formalism:

- Π↑ and Π↓ act as complementary filters
- only configurations that remain stable under both mappings are retained
- the projection layer does **not** determine the number of surviving modes or sectors
- downstream modules may analyze survivor structure, but such analysis is outside the scope of this file

This document records the **projection contract**, not its downstream outcomes.

