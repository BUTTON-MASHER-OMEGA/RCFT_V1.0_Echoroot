# RCMT Effective Comparison

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Interpretation
- scope: optional weighted comparison of local Geometry-Behavior records
- depends_on: `metric_interface.md`
- status: provisional profile definition
- anti-backflow: the comparison does not alter source features or define upstream distance

## Purpose

This file defines the base composite comparison for this metric profile. The historical name **effective metric** is retained as a profile label; the canonical output is an `effective-comparison-record`.

## Feature Vector

For a locally supported record bundle \(r\), define a declared feature map:

\[
F(r) = \left(F_m(r), F_c(r), F_e(r), F_i(r), F_h(r)\right)
\]

where the optional components encode:

- \(F_m\): mismatch-gradient records;
- \(F_c\): coherence-gradient records;
- \(F_e\): echoon-pattern or density-band records;
- \(F_i\): interaction-pattern records;
- \(F_h\): high-density-event records.

Each encoding must be explicit. Missing features remain missing and are not replaced by zero.

Mismatch and coherence gradients are order-dual. A profile using both must declare whether one is redundant, complementary, or omitted.

## Composite Comparison

For locally comparable bundles \(r_a\) and \(r_b\), a profile may define:

\[
D_P(r_a,r_b) =
\sum_{k \in K_P} w_k\,
\delta_k\!\left(F_k(r_a),F_k(r_b)\right)
\]

where:

- \(P\) is the profile declaration;
- \(K_P\) is its declared feature set;
- \(w_k\) are finite declared weights;
- \(\delta_k\) are declared feature-comparison functions.

The expression is a template, not a canonical RCMT equation.

## Output Record

An `effective-comparison-record` contains:

- both source bundle identifiers;
- connected adjacency support;
- profile identifier;
- feature encodings;
- comparison functions;
- weights and normalization;
- result;
- missing-feature handling;
- validity domain;
- and provisional status.

## Metric Qualification

\(D_P\) is a comparison by default.

It may be called a metric only when the profile verifies, over its declared domain:

- non-negativity;
- identity of indiscernibles;
- symmetry;
- triangle inequality.

If only some conditions hold, the output must use the narrower term supported by those conditions.

## Optional Continuous Encoding

A downstream approximation may encode the comparison using coordinates or an array \(g_{ij}\). That encoding requires:

- an explicit continuous support;
- a chart or basis record;
- a transformation rule;
- a metric-assignment authorization;
- and an approximation validity domain.

Without these inputs, \(g_{ij}\), coordinate distance, and differential notation are forbidden.

## Locality

Comparison is permitted only between:

- records sharing one local neighborhood; or
- records connected by a finite adjacency path within the declared profile bound.

The path and aggregation rule must be included in the output.

## Non-Claims

The effective comparison does not define:

- physical distance;
- spacetime interval;
- dynamics;
- force or field structure;
- a preferred RCMT metric;
- or a requirement for sibling metric modules.

## Status

Provisional optional profile. Ready for compression under `metric_interface.md`.
