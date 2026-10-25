# RCMT Clumping Score

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Interpretation
- scope: optional local aggregation score over declared metric-profile summaries
- depends_on: `metric_interface.md`, `rcmt_curvature_summary.md`, `rcmt_echoon_density_metric.md`, and `rcmt_survivor_influence_metric.md`
- status: provisional profile definition
- anti-backflow: this score does not create clumping behavior, attraction, or cosmological dynamics

## Purpose

This file defines a local aggregation-score template. The historical name **clumping potential** is retained as analogy language; the canonical output is a `clumping-score-record`.

## Inputs

The score may consume locally compatible:

- mismatch- or coherence-gradient features;
- `curvature-summary-record`;
- `echoon-density-summary`;
- `survivor-influence-summary`;
- `high-density-event-record`.

All inputs must use compatible support and declared conversion rules.

## Score Template

For a local neighborhood \(N\), a profile may define:

\[
C_P(N) =
\sum_{k \in K_P} w_k\,Q_k(N)
\]

where:

- \(Q_k\) are declared finite feature summaries;
- \(w_k\) are declared finite weights;
- \(K_P\) is the profile feature set.

A possible profile may include mismatch ordering, echoon density-band contrast, second-order variation, survivor association, and a high-density-event indicator. No term has a fixed sign or interpretation outside its profile declaration.

## Output Record

A `clumping-score-record` contains:

- neighborhood and source-record identifiers;
- connected adjacency support;
- profile identifier;
- feature summaries;
- weights, signs, and normalization;
- result;
- threshold classification, when declared;
- validity domain;
- and provisional status.

## Classification

A consuming model may classify a score as:

- below its declared threshold;
- within its declared transition band;
- above its declared threshold;
- or unresolved.

These labels do not establish aggregation, attraction, collapse, or structure formation.

## Optional Potential Form

A downstream approximation may represent the score as \(\Phi_P(x)\) over a supplied continuous support.

The term **potential** is allowed only as an explicitly representational analogy unless a separate downstream theory defines:

- its physical variables;
- units;
- dynamical coupling;
- and governing law.

This module supplies none of those.

## Locality

The score applies to a finite connected neighborhood. Larger summaries require an explicit aggregation rule and cannot erase their local source support.

## Downstream Use

Cosmology, visualization, and analysis modules may consume the score as one optional feature. They must not treat it as evidence that a physical clump forms.

## Prohibitions

This profile does not permit:

- mismatch-driven attraction as an established mechanism;
- automatic aggregation from a high score;
- physical potential or force;
- global cosmological evolution;
- undeclared signs or weights;
- or feedback into Geometry-Behavior event classification.

## Status

Provisional optional profile. Ready for compression.
