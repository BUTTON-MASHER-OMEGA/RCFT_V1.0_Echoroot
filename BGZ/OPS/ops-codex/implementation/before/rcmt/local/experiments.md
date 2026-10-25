# RCMT Experimental Description Guidelines

## Document Contract

- owner: RCMT
- type: Interpretation
- scope: downstream experimental descriptions
- depends_on: `invariants.md`, `constraints/measurement.md`, and `constraints/projection.md`
- authority: non-canonical usage guidance
- direction: experiments consume RCMT contracts; examples do not define them
- compression_rule: omit from the canonical shard or summarize as usage guidance

## Purpose

This document provides **guidelines** for describing physical experiments in the language of Relational Coherence Mode Theory (RCMT).

It is downstream of the RCMT invariants and does not define constraints.

It offers a consistent representational framework for expressing:

- preparation,
- apparatus structure,
- boundary conditions,
- post‑selection,
- weak measurement,
- entanglement setups,
- and interferometric configurations.

These guidelines are interpretive and do not modify the RCMT constraint layer.

---

## 1. Describing an experiment in RCMT

For RCMT descriptions, an experiment is represented as a **boundary-conditioned region** with:

- a preparation boundary,
- an apparatus‑imposed constraint structure,
- a region of propagation,
- and a post‑selection or readout boundary.

Admissible descriptions must:

- specify boundary conditions explicitly,
- avoid treating apparatus as ontological objects,
- and express all behavior in terms of projection and LCO evaluation.

---

## 2. Preparation boundary

A preparation boundary defines the **initial admissible mode‑space configurations**.

Examples:

- a photon source defines a constraint on polarization mode‑structure,
- a spin‑polarized beam defines a constraint on spin‑mode alignment,
- a thermal source defines a broad admissible region with unresolved structure.

Preparation boundaries must be described as **constraint specifications**, not as physical initial states.

---

## 3. Apparatus as constraint structure

Apparatus elements (beam splitters, magnets, detectors, cavities, etc.) are represented as **local constraint structures** that:

- restrict admissible mode‑patterns,
- shape projection geometry,
- and define which configurations survive LCO evaluation.

Examples:

- a Stern–Gerlach magnet imposes a spatially varying constraint on spin‑projection,
- a beam splitter imposes a mode‑mixing constraint,
- a cavity imposes a resonance constraint.

RCMT descriptions treat apparatus as defining **admissible relational structure**, not as an upstream causal mechanism.

---

## 4. Propagation region

Between boundaries, relational configurations undergo **locally admissible transitions**.

Descriptions must:

- avoid global evolution laws,
- avoid treating propagation as motion through a background,
- and express changes as sequences of LCO‑admissible projections.

In this guidance, propagation language is representational rather than a new dynamics law.

---

## 5. Post‑selection boundary

Post‑selection is represented as a **final boundary constraint** that:

- eliminates incompatible mode‑patterns,
- conditions projection,
- and defines the representational outcome.

Examples:

- selecting a particular detector click,
- conditioning on a specific polarization,
- conditioning on a specific path in an interferometer.

RCMT describes post-selection as **boundary-conditioned projection**, not as an added collapse mechanism.

---

## 6. Weak measurement setups

Weak measurement is represented as:

- a preparation boundary,
- a weak constraint region (low mismatch sensitivity),
- a strong post‑selection boundary.

Interpretively:

- weak measurement trajectories correspond to conditioned projections,
- they are representational shadows, not ontic paths,
- and they depend on the imposed boundary structure.

No preferred global frame is required.

---

## 7. Entanglement experiments

Entanglement corresponds to **shared constraint structure** across regions.

Descriptions must:

- specify the shared constraint explicitly,
- avoid non‑local signaling,
- and express correlations as consequences of boundary‑conditioned admissibility.

Examples:

- Bell‑type setups,
- entangled photon pairs,
- spin‑correlated pairs.

Entanglement is representational, not ontological.

---

## 8. Interferometric setups

Interferometers (Mach–Zehnder, Michelson, etc.) are described as:

- a preparation boundary,
- a sequence of constraint‑defining elements (beam splitters, mirrors),
- a propagation region with unresolved mode‑structure,
- and a post‑selection boundary.

Interference patterns arise from:

- projection geometry,
- unresolved mode‑structure,
- and LCO‑compatible constraint resolution.

No wave/particle duality is required.

---

## 9. Detector modeling

Detectors are represented as **high‑mismatch boundary constraints** that:

- eliminate all but one admissible projected configuration,
- define the representational outcome,
- and do not introduce collapse dynamics.

A “click” corresponds to a **stable projected configuration** under the imposed boundary.

---

## 10. Classical apparatus behavior

Classical apparatus elements correspond to:

- stable mode‑patterns,
- low unresolved structure,
- and consistent projection across regions.

Descriptions must avoid treating classicality as fundamental.

---

## 11. Status

This file provides interpretive guidelines only.
It does not define constraints and does not modify the RCMT invariants or structural
specifications.

