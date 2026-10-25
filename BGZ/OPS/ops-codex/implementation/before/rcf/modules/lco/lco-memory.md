# LCO Memory

## Document Contract

- owner: RCF local coherence operator
- type: Interpretation
- scope: module-local content of `lco/lco-memory.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Definition and Role of the Memory Mismatch Term*

This document defines the **memory component** used by the Local Coherence Operator (LCO) to evaluate candidate local updates. Memory introduces a controlled influence from a representation of prior local alignment, represented by the memory anchor \( H \), weighted by an externally supplied scalar \( \gamma \in [0,1] \).

The memory term is a local, algebraic construct. It does not introduce any external geometric or topological assumptions and uses only the primitives permitted at the execution layer.

---

## **1. Purpose of the Memory Term**

The memory component ensures that the LCO:

- incorporates historical influence into local update evaluation
- maintains continuity with prior relational structure
- allows the temporal-interface module to modulate the strength of memory
- remains strictly local and geometry‑neutral

Memory acts as an additional relational constraint, scaled by γ, that encourages updates to remain consistent with the recent past.

The memory anchor does not encode full historical state, but a directional summary of prior local coherence.

---

## **2. Memory Mismatch Definition**

The mismatch between the candidate mode \( m \) and the memory anchor \( H \) is defined as:



\[
\ell_{\text{mem}} = \gamma (1 - H \cdot m)
\]



### **Properties**

- Uses the same mismatch structure as relation mismatch.
- Scaled by γ, which determines the strength of memory influence.
- Produces a scalar in \([0, 2\gamma]\) for unit‑norm inputs.
- Treats memory as an additional local relation, weighted externally.

### **Rationale**

The form \(1 - H \cdot m\):

- is consistent with the mismatch primitive
- is bounded and well‑defined
- requires no external geometric or topological assumptions
- ensures that memory influence increases smoothly as alignment worsens

The γ‑scaling allows the temporal-interface module to control the degree of temporal drag without altering the operator’s internal logic.

---

## **3. γ as an External Parameter**

The LCO treats γ as an **external input**, not something it selects or optimizes.

- γ is supplied by the temporal-interface module.
- The LCO applies γ but does not infer, adjust, or reinterpret it.
- The operator remains agnostic to how γ is chosen or what global structure it reflects.

This separation ensures that temporal influence enters the LCO in a controlled, invariant‑compatible way.

The LCO must not apply nonlinear transformations to γ or combine it with other terms beyond scalar multiplication in the memory mismatch definition.

---

## **4. Integration with Relation Mismatch**

The memory mismatch term is included alongside the relation mismatch terms:



\[
\{\ell_i\}_{i=1}^N \cup \{\ell_{\text{mem}}\}
\]



This combined set forms the basis for:

- aggregate mismatch
- dispersion
- and the final coherence score

Memory is therefore treated uniformly with relational mismatch, except for its γ‑weighting.

---

## **5. Invariant Compatibility**

The memory term satisfies all LCO invariants:

- **Locality:**
  Depends only on \( H \), \( m \), and γ.

- **Geometry‑neutrality:**
  Uses only dot products and scalar multiplication.

- **Determinism:**
  Produces consistent values for identical inputs.

- **Well‑definedness:**
  Always finite and bounded for admissible inputs.

- **Unit‑norm mode compatibility:**
  Defined entirely in terms of unit‑norm vectors.

- **γ as external:**
  γ is applied but never chosen or modified by the operator.

---

## **6. Role in the Coherence Score**

Memory mismatch contributes directly to the coherence score:



\[
C = f(\{\ell_i\}, \ell_{\text{mem}})
\]



Its influence is determined entirely by γ and the scoring function defined in `lco-operator.md`.

---

## **Status**

The memory term provides a controlled, local mechanism for incorporating historical influence into the LCO’s evaluation. It is fully compatible with the primitives and invariants of the execution layer and introduces no additional structure beyond what the LCO is permitted to use.

