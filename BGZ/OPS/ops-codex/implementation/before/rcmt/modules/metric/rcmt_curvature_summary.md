# RCMT Second-Order Variation Summary

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Interpretation
- scope: optional second-order finite variation over local profile records
- depends_on: `metric_interface.md` and `rcmt_gradient_tensor.md`
- status: provisional profile definition
- anti-backflow: curvature-like output does not redefine geometry or establish physical curvature

## Purpose

This file defines a second-order local variation summary. The historical name **curvature summary** is retained as an analogy; the canonical output is a `curvature-summary-record`.

## Discrete Canonical Form

For a declared adjacent path \(a \sim b \sim c\), and a shared numeric feature encoding \(F_P\), a profile may define:

\[
\Delta_P^2 F(a,b,c)
= F_P(r_c) - 2F_P(r_b) + F_P(r_a)
\]

The operation summarizes change in first-order finite variation. It is not geometric curvature.

For graph neighborhoods without a linear path ordering, the profile must declare its finite stencil and aggregation rule.

## Source Features

Second-order summaries may be calculated over:

- mismatch-gradient features;
- coherence-gradient features;
- echoon density-band encodings;
- interaction-pattern encodings;
- high-density-event encodings;
- or effective-comparison outputs from one fixed profile.

Every numeric encoding and stencil must be declared.

## Output Record

A `curvature-summary-record` contains:

- source and intermediate record identifiers;
- adjacency path or finite stencil;
- profile and feature-map identifiers;
- first- and second-order finite results;
- normalization and local scale, when used;
- approximation metadata;
- unresolved components;
- and provisional status.

## Optional Differential Encoding

A downstream approximation may use Hessian-like or curvature-like notation only when it supplies:

- continuous support;
- coordinates or basis;
- local scale;
- a discrete-to-continuous approximation map;
- and a validity regime.

An expression such as \(\partial_i\partial_j F\) is an optional approximation to the finite summary, not its canonical definition.

## Curvature Gate

This output remains **curvature-like**.

Calling it geometric curvature additionally requires a downstream contract defining the needed metric, connection, and transformation behavior. Calling it physical spacetime curvature requires still further physical interpretation outside this profile.

## Locality

Every second-order summary uses a finite connected neighborhood. Global fitting may be performed downstream but cannot replace local source support.

## Non-Claims

This summary does not establish:

- bending geometry;
- attraction or clumping;
- stabilization or destabilization;
- field equations;
- or dynamical metric evolution.

## Status

Provisional optional profile. Ready for compression.
