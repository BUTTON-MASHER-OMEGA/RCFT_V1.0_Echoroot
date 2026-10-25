# Projection Core

## Document Contract

- owner: RCF projection module
- type: Contract
- scope: module-local content of `geometry/projection/projection-core.md`
- authority: tier-7 discrete-to-continuous handoff
- upstream dependencies: `/modules/geometry/projection/interface.md`
- status: active
- purpose: define what projection acts on, preserves, smooths, and exposes without selecting survivors or defining downstream metrics
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Core Role of the Projection Operator \( \Pi \)*

This document defines the core role of the projection operator \( \Pi \) within the RCF architecture.

\( \Pi \) is the operator that maps a surviving discrete relational geometry into a continuous shadow representation.
It is a **structural shadow map**, not an independent geometric or dynamical law.

---

## **1. Operator Role**

The projection operator implements a mapping of the form:

\[
\Pi : \mathcal{G}_{\text{survive}} \rightarrow \mathcal{S}_{\text{cont}}
\]

Where:

- \(\mathcal{G}_{\text{survive}}\) is the class of discrete relational geometries that survive the upstream sieve,
- \(\mathcal{S}_{\text{cont}}\) is the class of continuous shadow representations produced from those survivors.

The purpose of \( \Pi \) is to convert discrete relational persistence into a smooth, aggregated representation usable by downstream modules.

---

## **2. What Projection Does**

Projection:

- smooths discrete relational structure,
- aggregates local motif behavior,
- carries surviving adjacency into neighborhood structure,
- preserves residual mismatch as shadow content,
- produces a continuous representation of an already-selected substrate.

Projection does **not** choose the substrate.
It acts only after survival has already been determined upstream.

---

## **3. What Projection Does Not Do**

\( \Pi \) does **not**:

- select discrete survivors,
- define a new admissibility law,
- introduce new combinatorics,
- define a physical metric,
- define continuous flow,
- define curvature as a primitive,
- define physical space or spacetime.

Projection is therefore **derived**, not primitive.

---

## **4. Domain Restriction**

\( \Pi \) is defined only on discrete relational structures that survive the upstream architecture:

- local coherence filtering,
- recursive persistence evaluation,
- and discrete survival criteria.

Non-surviving candidates do not project in the RCF sense.
They are eliminated before projection becomes meaningful.

---

## **5. Output Character**

The output of \( \Pi \) is a **continuous shadow**, not a fully specified geometric ontology.

This shadow is treated as:

- smooth relative to the discrete substrate,
- neighborhood-structured rather than vertex-enumerated,
- relational rather than coordinate-primitive,
- inherited from discrete survivors rather than freely imposed.

Downstream modules may refine or interpret this shadow, but they may not treat \( \Pi \) as having introduced additional discrete structure or new survival rules.

---

## **6. Structural Dependency**

The dependency direction is one-way:

\[
\text{Discrete survivor structure} \Rightarrow \Pi \Rightarrow \text{continuous shadow structure}
\]

This has two consequences:

- all continuous structure documented downstream must be traceable to projected discrete structure,
- no downstream module may treat projection outputs as independent from the survivor class that generated them.

---

## **7. Status**

The Projection Core defines \( \Pi \) as the canonical discrete-to-continuous shadow operator within the current RCF geometry stack.
Its role is to formalize the architectural handoff, not to supply downstream interpretation.
