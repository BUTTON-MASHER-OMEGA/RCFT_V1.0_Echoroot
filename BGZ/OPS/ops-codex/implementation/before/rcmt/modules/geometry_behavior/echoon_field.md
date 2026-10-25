# Echoon Pattern Layer

## Document Contract

- owner: RCMT Geometry-Behavior
- type: Interpretation
- scope: candidate local organization of imported echoon records
- depends_on: `geometry_behavior_contract.md` and `../../../abstract/echoon_minimal_closure/local/abstract_interface.md`
- status: provisional
- anti-backflow: this file does not redefine echoon labels, transitions, density bands, closure, reinjection, or decay

## Purpose

This file defines the candidate RCMT pattern layer historically called the **echoon field**. The word *field* is retained only as a downstream organizational analogy. The represented object is a finite, adjacency-indexed collection of imported echoon records.

## Inputs

- `echoon-state-label`;
- `echoon-density-band`;
- `echoon-transition`;
- `closure-event`;
- `reinjection-event`;
- `decay-event`;
- `echoon-origin-label`;
- local adjacency records.

## Candidate Pattern Record

An `echoon-pattern-record` contains:

- a finite set of echoon record identifiers;
- their adjacency relations;
- imported transition labels connecting them;
- their density-band labels;
- a declared coherence criterion;
- and provisional status.

A pattern may be classified as coherent only when every included transition is allowed by the abstract echoon grammar and every propagation step has explicit adjacency support.

## Candidate Formation Descriptions

Geometry-Behavior may associate an echoon pattern with:

- a recorded reinjection event;
- a compatible sequence of echoon transitions;
- or a repeated local arrangement of compatible echoon records.

These associations classify records. They do not establish a cause of echoon formation.

## Candidate Propagation Description

Propagation is represented as an ordered sequence:

```text
echoon-record_i -> allowed transition -> adjacent echoon-record_i+1
```

Every step must be local, explicit, and traceable. The sequence does not imply motion through spacetime.

## Survivor Association

An echoon pattern may be associated with a survivor response only when:

- the echoon and survivor records share an explicit local neighborhood;
- the proposed survivor transition is independently admissible;
- and the association remains labeled as a candidate.

Permitted association labels include offset adjustment, phase-cycle correlation, local mismatch change, and closure-event correlation. These labels do not assert force or agency.

## Density and Gradient Use

Echoon density bands may contribute to a `mismatch-gradient-record` when the density labels and mismatch samples refer to the same adjacent neighborhood. Density alone does not define curvature, attraction, or a metric.

## Dissipation and Reinjection

This module may sequence imported decay and reinjection labels. It must not reinterpret decay as energy loss or reinjection as physical emission.

## Prohibitions

This file does not permit:

- a continuous field primitive;
- propagation without imported transitions;
- hidden coherence variables;
- causes for echoon events;
- particle identity;
- metric or curvature claims;
- or modification of the abstract echoon grammar.

## Output

The only canonical output shape is `echoon-pattern-record`. Terms such as stream, packet, field-like pattern, and wave-like pattern are optional RCMT descriptions of that output and are noncanonical.

## Status

Provisional RCMT interpretation. Ready for compression under `geometry_behavior_contract.md`.
