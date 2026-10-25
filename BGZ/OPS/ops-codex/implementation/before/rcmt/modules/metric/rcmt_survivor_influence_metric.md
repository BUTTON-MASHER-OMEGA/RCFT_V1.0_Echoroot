# RCMT Survivor Influence Summary

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Interpretation
- scope: optional pairwise summary over imported survivor interaction-pattern records
- depends_on: `metric_interface.md`, `../geometry_behavior/survivor_interaction_rules.md`, and `rcmt_effective_metric.md`
- status: provisional profile definition
- anti-backflow: this summary does not create survivor interactions or strengthen candidate interaction claims

## Purpose

This file defines a pairwise summary of records already classified by Geometry-Behavior as candidate interaction patterns.

**Influence** is a profile label for record association. It is not force, causation, or an interaction law.

## Inputs

The profile may consume:

- `interaction-pattern-record`;
- referenced survivor identifiers;
- referenced mismatch- or coherence-gradient records;
- referenced echoon-pattern records;
- referenced closure-reinjection sequences;
- and their explicit adjacency support.

The profile cannot construct an interaction record from survivor proximity alone.

## Feature Map

For an interaction-pattern record \(r_{ab}\), define a declared feature map:

\[
J_P(r_{ab}) =
\left(J_m,J_c,J_e,J_s,J_h\right)
\]

whose optional components encode:

- mismatch ordering;
- coherence ordering;
- echoon-pattern association;
- survivor-transition classification;
- high-density-event association.

Each component must be traceable to the source interaction record. Missing components remain missing.

## Pairwise Summary

A profile may calculate:

\[
I_P(a,b) =
\sum_{k \in K_P} w_k J_k(r_{ab})
\]

or another explicitly declared finite comparison.

The sign, scale, symmetry, and normalization of \(I_P\) are profile choices. No universal interpretation is implied.

## Output Record

A `survivor-influence-summary` contains:

- survivor and interaction-record identifiers;
- local adjacency support;
- feature map;
- weights and normalization;
- pair ordering and symmetry status;
- result;
- unresolved components;
- and provisional status.

## Range

The canonical summary is local. A multi-step influence summary requires:

- an explicit finite adjacency path;
- a composition rule;
- and preservation of every intermediate source record.

It does not authorize action at a distance.

## Optional Continuous Encoding

A downstream approximation may represent \(I_P\) as a scalar map over continuous support only with an explicit support map, interpolation rule, and validity regime.

## Prohibitions

This profile does not permit:

- physical force or coupling strength;
- inferred causation;
- hidden pair state;
- global optimization;
- creation of survivor identity or transition records;
- or promotion of candidate Geometry-Behavior claims.

## Status

Provisional optional profile. Ready for compression.
