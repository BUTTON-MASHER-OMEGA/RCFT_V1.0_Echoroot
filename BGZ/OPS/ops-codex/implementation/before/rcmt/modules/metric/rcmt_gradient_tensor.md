# RCMT First-Order Variation Summary

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Interpretation
- scope: optional first-order finite variation over adjacent profile records
- depends_on: `metric_interface.md` and `rcmt_effective_metric.md`
- status: provisional profile definition
- anti-backflow: this summary does not redefine Geometry-Behavior gradients or create dynamics

## Purpose

This file defines a first-order variation summary. The historical name **gradient tensor** denotes an optional mathematical encoding; the canonical output is a finite `gradient-summary-record`.

## Discrete Canonical Form

For adjacent record bundles \(r_a\) and \(r_b\), with a declared numeric feature encoding \(F_P\), define:

\[
\Delta_P F(a,b) = F_P(r_b) - F_P(r_a)
\]

The operation is valid only when:

- \(a\) and \(b\) have explicit adjacency support;
- both bundles use the same profile;
- feature normalization is shared;
- and missing values are handled explicitly.

For ordered categorical features, the profile may store only direction or rank change rather than a numeric difference.

## Source Features

The summary may include finite variation in:

- mismatch-gradient ordering;
- coherence-gradient ordering;
- echoon density-band encoding;
- interaction-pattern encoding;
- high-density-event encoding;
- or the effective comparison evaluated under one fixed profile.

Mismatch and coherence orderings may not be counted as independent dimensions unless the profile explains the distinction.

## Output Record

A `gradient-summary-record` contains:

- endpoint source identifiers;
- adjacency support;
- profile and feature-map identifiers;
- finite differences or ordered changes;
- normalization and scale, when used;
- unresolved components;
- approximation metadata;
- and provisional status.

## Optional Derivative Form

A downstream approximation may write:

\[
\mathcal{G}_i(x)
\]

or use \(\partial_i F(x)\) only when it supplies:

- continuous support;
- a coordinate chart or basis;
- a local scale;
- a discrete-to-continuous approximation map;
- and a declared error or validity regime.

Derivative notation is not canonical in this profile.

## Optional Tensor Form

The summary may be encoded as a tensor only when the profile declares:

- index domain;
- basis;
- transformation rule;
- and relation to the canonical finite record.

An indexed array without those declarations remains an array, not a tensor.

## Locality

First-order variation is adjacency-local. Multi-step summaries must retain the finite path and may not replace it with an unrecorded nonlocal difference.

## Non-Claims

This summary does not define:

- a force direction;
- a flow law;
- continuous dynamics;
- physical gradients;
- or geometric tangent structure.

## Status

Provisional optional profile. Ready for compression.
