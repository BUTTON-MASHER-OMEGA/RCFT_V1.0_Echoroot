# Adjacency Preservation

## Document Contract

- owner: RCF projection module
- type: Contract
- scope: module-local content of `geometry/projection/adjacency-preservation.md`
- authority: tier-7 discrete-to-continuous handoff
- upstream dependencies: `/modules/geometry/projection/interface.md`
- status: active
- purpose: define what projection acts on, preserves, smooths, and exposes without selecting survivors or defining downstream metrics
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *How Discrete Relational Adjacency Survives as Continuous Neighborhood Structure*

This document defines how projection carries discrete relational adjacency into the continuous shadow.

The goal is to make precise the dependency:

\[
\text{discrete motif adjacency} \rightarrow \text{continuous neighborhood structure}
\]

without introducing a background metric or manifold by assumption.

---

## **1. Discrete Starting Point**

In the upstream discrete layer, adjacency is combinatorial and motif-based.

It is expressed through:

- local relation neighborhoods,
- motif compatibility,
- surviving adjacency classes,
- tiered relational proximity within the discrete substrate.

Projection does not replace this structure with an unrelated notion of nearness.
It smooths the same structure into a continuous shadow.

---

## **2. Continuous Analogue**

The continuous analogue of discrete adjacency is a **neighborhood structure** in the projected shadow.

This neighborhood structure is defined by:

- the projected image of surviving local motifs,
- the smoothing of discrete adjacency relations,
- the persistence of relative relational nearness under aggregation.

Thus, points or regions in the shadow count as neighbors because they arise from overlapping or closely related discrete neighborhoods upstream.

---

## **3. Preservation Rule**

Projection preserves adjacency **up to smoothing**.

This means:

- sharp combinatorial boundaries may become soft neighborhood boundaries,
- explicit vertex-to-vertex relations may become region-to-region relations,
- motif overlap may become neighborhood overlap,
- but the neighborhood structure remains constrained by the discrete survivor that generated it.

Projection cannot produce arbitrary new neighborhoods disconnected from upstream adjacency.

---

## **4. Motif Dependence**

Neighborhood structure is inherited from surviving local motifs, not from isolated vertices.

This matters because:

- local motifs carry compatibility information,
- motifs already encode survivable local packing,
- projection acts on aggregated motif structure rather than on disconnected points.

The continuous shadow is therefore better understood as the image of **surviving relational neighborhoods** than as the image of a raw point cloud.

---

## **5. Consequences for Downstream Modules**

Downstream continuous modules may assume:

- neighborhood structure is inherited, not freely chosen,
- local continuity reflects smoothed motif overlap,
- anisotropy in neighborhoods reflects anisotropy upstream,
- neighborhood structure cannot be interpreted independently of survivor class.

Downstream modules may **not** assume:

- an externally supplied metric notion of adjacency,
- manifold charts as part of the projection contract,
- neighborhood relations detached from discrete provenance.

---

## **6. Status**

Adjacency preservation defines one of the central load-bearing constraints on projection:
continuous neighborhood structure must remain the smoothed shadow of discrete relational adjacency.

