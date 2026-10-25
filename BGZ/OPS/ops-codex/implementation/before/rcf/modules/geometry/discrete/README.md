# Discrete Geometry Module

## Document Contract

- owner: RCF discrete geometry module
- type: Status
- scope: module-local content of `geometry/discrete/README.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## **purpose**
This directory records the discrete relational geometries that survive the current constraint-layer sieve and may serve as inputs to continuous projection. These geometries are not assumed; they are the statistical remainder of current evaluations using local LCO outcomes, GCO summaries, the mismatch function, and the adopted spinor-closure profile.

The goal of this module is to document:
- the admissibility criteria for discrete geometries
- the failure modes of non‑survivors
- the algebraic closure requirements
- the density‑dependent survivors
- and the role these geometries play in downstream projection

No continuous geometry, metric assumptions, or physics‑shaped interpretations appear here. This layer is strictly discrete, relational, and upstream of projection.

All geometric facts about polytopes (adjacency, symmetry, fibrations, etc.) are treated as imported mathematical structure. All statements involving survivorship, mismatch, relational tiers, or update behavior are properties of the current RCF formalism.

---

## **scope**
This module covers:

### **1. the admissibility sieve**
The rules that determine whether a candidate geometry can survive long‑horizon recursion:
- local mismatch behavior
- mode‑space completeness
- spinor embedding
- recursive closure
- density‑regime stability

Within the current sieve profile, these checks are applied uniformly to all declared candidates.

### **2. candidate evaluation**
A record of which 4D simplicial topologies were tested, and why most fail:
- quantization error
- incomplete mode coverage
- algebraic instability
- closure failure under spinor composition

This records why only a small subset of the evaluated geometries remain viable under the current profile.

### **3. the surviving geometries**
The discrete structures that consistently pass the sieve across relevant density regimes:
- the 600‑cell (standard density)
- the dual 5‑demicube (intermediate density)
- E₈ ($4_{21}$) (extreme density)

Each survivor is documented in terms of:
- definition
- adjacency structure
- spinor embedding
- closure properties
- density‑regime behavior

These are the geometries currently observed to remain stable under the declared update and sieve rules. The result does not exclude unevaluated candidates or alternative invariant-compatible profiles.

---

## **relationship to other modules**

### **LCO / GCO**
The discrete geometries documented here are the objects that LCO and GCO are instantiated on in current evaluations.

LCO defines the local update rule; GCO defines long‑horizon coherence.

This directory describes the structures that remain stable under both.

### **time**
Time is treated as a relational ordering of updates, not a global index.
This module does not define time; it only documents which geometries remain stable *under* time‑ordered recursion.

### **projection**
This module is upstream of projection.
The projection operator Π, the φ‑spectrum, and the continuous shadow all depend on the discrete geometries defined here, but are documented in the Projection Operator (Π) module.

---

## **design principles**
This module follows the core RCF principles:

- **no assumed geometry**
  All geometries must survive the sieve; none are taken as background.

- **no continuous assumptions**
  Continuous geometry is downstream and derived, not part of this layer.

- **no physics‑shaped claims**
  This module documents discrete relational structure only.

- **strict modularity**
  Each geometry is documented independently, with no cross‑contamination.

- **invariant safety**
  Only forced invariants are recorded; interpretations are left downstream.

---

## **summary**
The discrete geometry layer supplies survivor records to the current RCF geometry stack.
It identifies which evaluated relational structures survive the declared constraint pressure, recursion, and spinor-closure profile, and therefore which records are presently available to continuous projection.

Downstream modules using this geometry stack depend on the survivor records documented here.
