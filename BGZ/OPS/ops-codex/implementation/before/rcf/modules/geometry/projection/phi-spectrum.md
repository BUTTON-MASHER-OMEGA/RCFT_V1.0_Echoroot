# Phi Spectrum

## Document Contract

- owner: RCF projection module
- type: Interpretation
- scope: module-local content of `geometry/projection/phi-spectrum.md`
- authority: tier-7 discrete-to-continuous handoff
- upstream dependencies: `/modules/geometry/projection/interface.md`
- status: active
- purpose: define what projection acts on, preserves, smooths, and exposes without selecting survivors or defining downstream metrics
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose
This document defines the φ-spectrum profile: relational descriptor classes observed when the projection operator Π acts on a surviving discrete geometry. The classes are not core invariants or metric distances; they record a statistical pattern reported by current projection evaluations across declared stable regimes.

The φ‑spectrum is a projection‑level organization of survivor structure; it is not a geometric spectrum and does not introduce new combinatorics.

The goal of this module is to record the structure of the spectrum without interpreting it in physical or geometric terms.

---

## overview
The φ‑spectrum is treated as:
- a set of stable relational descriptor tiers in the continuous shadow
- derived entirely from the adjacency and mismatch structure of the discrete substrate
- a consistent outcome in current evaluations of projection and smoothing, not a free parameter
- a geometry‑neutral invariant that appears across multiple survivors

The spectrum reflects how discrete triangle‑based motifs aggregate under Π.

---

## structural properties
The φ‑spectrum exhibits:
- **a small number of stable relational tiers (three in current evaluations)** that persist across standard, intermediate, and extreme density regimes
- **ratio‑like spacing** between tiers that approximates φ‑structured separation in current evaluations
- **density‑dependent sharpening** where higher constraint densities produce clearer tier boundaries
- **geometry‑specific modulation** where 600‑cell, dual 5‑demicube, and E₈ contribute distinct fine structure

These properties are documented without downstream interpretation.

---

## role in RCF
- provides the relational distance structure used by downstream flow and kinematics modules
- acts as the invariant “skeleton” of the continuous shadow
- constrains how curvature‑like variation appears in `slush-mapping.md`
- provides structural input that downstream modules may use to construct relational metrics

This file records the spectrum’s structural behavior only.
Interpretations of the spectrum are handled downstream.

---

## invariants
- the φ‑spectrum is fully determined by Π and the discrete substrate
- no new structure is introduced; the spectrum reflects persistent relational patterns
- the number of stable tiers is a forced outcome in current evaluations
- the spectrum is documented without assuming any physical meaning

Further detail will be added as the projection formalism develops.
