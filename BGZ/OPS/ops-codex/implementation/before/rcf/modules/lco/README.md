# Local Coherence Operator (LCO)

## Document Contract

- owner: RCF local coherence operator
- type: Status
- scope: module-local content of `lco/README.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Execution‑Layer Primitive for RCF*

The Local Coherence Operator (LCO) is the foundational execution‑layer rule in the RCF architecture. Its purpose is to score how well a candidate local update preserves coherence within a relational neighborhood. The LCO operates strictly at the **local**, **algebraic**, and **geometry‑neutral** level. It does not encode or assume any geometric structure beyond the minimal requirement that modes inhabit a unit‑norm space.

The unit mode space is a representational device for comparing update directions, not a claim about the structure of reality.

The LCO is responsible for assessing the quality of a proposed update using three ingredients:

- **Local relations**
- **A memory anchor**
- **A coherence score** derived from mismatch, dispersion, and memory weighting

The LCO does **not** determine global structure, persistence, or geometry. Those emerge downstream through repeated application of the operator and evaluation by higher‑level modules.

The LCO is a design choice within this framework. It is the only operator we have identified that satisfies the invariants implied by Relational Primacy under the current constraint set. Different invariants would yield different operators. Within this system, it represents the minimal operator that survives the applied constraints.

---

## **Purpose**

The LCO provides a minimal, invariant‑preserving rule for evaluating local updates in a relational system. It ensures that each update:

- respects local relational structure
- incorporates historical influence through a scalar memory weight
- remains normalized to the unit mode space
- is designed to contribute to local coherence descent under admissible update conditions

The LCO is the *only* mechanism responsible for local update evaluation. It does not select global geometry, enforce packing, or encode any specific coordinate structure.

---

## **Scope**

The LCO governs:

- how local relations \( r_i \) are compared to a candidate mode
- how mismatch and dispersion are computed
- how memory drag is incorporated
- how the coherence score is formed
- how the coherence score is computed for a candidate update

The LCO does **not** govern:

- global stability
- persistence of structures
- geometric envelopes
- attractor selection
- φ‑related behavior
- polytope‑like packings
- any downstream representational geometry

These belong to higher‑level modules that operate on the *outputs* of the LCO, not its internals.

---

## **Design Principles**

The LCO is built on the following principles:

### **1. Locality**
All evaluations depend only on the immediate relational neighborhood and the memory anchor. No global information is used.

### **2. Geometry-Neutrality**
The operator is physical-geometry and topology-neutral while using a minimal normalized mode space for directional comparison.

Modes lie in a unit-norm space used purely for comparing relational directions.

This representation:
- does not imply any underlying spatial or geometric ontology
- is not interpreted as physical or structural geometry

The LCO does not encode:
- coordinates
- packings
- polytopes
- dimensional preferences
- φ or ε structure

### **3. Invariant Preservation**
The LCO maintains:

- unit‑norm mode space
- produces a well-defined coherence score for a given input configuration
- supports local coherence descent under admissible update conditions
- independence from global geometry

### **4. Separation of Concerns**
The LCO defines:

- the **operator** (how coherence is scored)
- the **update policy** (how scores are used to select updates)

These are conceptually distinct and remain modular.

---

## **Inputs and Outputs**

### **Inputs**
- A set of local relation vectors \( r_i \)
- A memory anchor \( H \)
- A candidate mode \( m \)
- A scalar memory weight \( \gamma \in [0,1] \)

### **Output**
- A coherence score indicating how well the candidate update preserves local structure

The LCO does not modify global state; it only evaluates.

---

## **Role in the RCF Architecture**

The LCO is the **first layer** in the execution pipeline:

```
[ Temporal Interface ] → exposes sealed γ
          ↓
[ LCO — Local Coherence Operator ] → evaluates local updates
          ↓
[ GCO — Global Coherence Operator ] → aggregates persistence across indexed recursion
          ↓
[ Geometry Module ] → interprets emergent structure
```

The LCO is intentionally blind to everything downstream.
Its neutrality ensures that any emergent geometry is a **result**, not an assumption.

---

## **Status**

This module defines the execution‑layer contract for RCF.
It is stable, geometry‑neutral, and ready for downstream integration.
