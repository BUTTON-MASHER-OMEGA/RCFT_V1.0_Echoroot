# Mismatch Gradients

## Document Contract

- owner: RCMT Geometry-Behavior
- type: Interpretation
- scope: candidate adjacency-local comparison of imported mismatch samples
- depends_on: `geometry_behavior_contract.md`, `../../local/constraints/lco.md`, and `../../local/constraints/geometry.md`
- status: provisional
- anti-backflow: this file does not define mismatch, LCO scoring, geometry, or metric structure

## Purpose

This file defines the loss-oriented RCMT description of ordered variation among mismatch samples attached to adjacent projected configurations. It tracks increasing refinement burden, mismatch accumulation, or coherence loss as represented by the imported scoring profile.

The gain-oriented complement is defined in `coherence_gradients.md`.

## Definition

For adjacent configurations \(a\) and \(b\), let \(m_a\) and \(m_b\) be imported local mismatch samples evaluated under the same declared comparison profile.

A `mismatch-gradient-record` may store:

- source identifiers for \(a\), \(b\), \(m_a\), and \(m_b\);
- the adjacency record connecting \(a\) and \(b\);
- the ordering of \(m_a\) and \(m_b\);
- the declared comparison profile;
- and provisional status.

The record is a finite local comparison. It is not a differential operator or a vector on a manifold.

## Relation to Coherence Gradients

Under one declared comparison profile:

- an ordering toward greater mismatch is an ordering toward lower coherence;
- an ordering toward lower mismatch is an ordering toward greater coherence.

This is an inverse ordering relation. It is not automatically a numeric identity such as \(c = 1 - m\) or \(\Delta c = -\Delta m\). Either identity requires an explicit imported normalization rule.

## Candidate Sources

A consuming RCMT model may test correlations between mismatch-gradient records and:

- survivor offset-regime labels;
- echoon density-band labels;
- closure or reinjection event labels;
- phase-cycle labels;
- or compatible adjacent projection records.

The correlation must name its inputs and does not establish that any input generated the gradient.

## Candidate Survivor Response

A survivor-response candidate may associate a mismatch-gradient record with a later admissible survivor record. Permitted descriptive labels include:

- offset adjustment;
- phase-cycle adjustment;
- local reorientation;
- or transition toward the lower indexed mismatch sample.

The final label is an ordering description only. It does not assert motion, attraction, or force.

## Echoon Association

An echoon-pattern record may be compared with a mismatch-gradient record when both are supported by the same adjacency neighborhood. Allowed descriptions include alignment, concentration, dispersion, and persistence.

No such description may add an echoon transition absent from the abstract grammar.

## Persistence and Relaxation

A gradient may be labeled persistent when the same declared ordering recurs across a stated finite sequence of local records. It may be labeled relaxed when that ordering is no longer present.

Persistence is not self-reinforcement, and relaxation is not energy dissipation unless a separate downstream interface establishes such a mapping.

## Curvature-Like Analogy

Downstream RCMT modules may use persistent mismatch-gradient records as inputs to curvature-like summaries. The analogy remains:

- local;
- representational;
- metric-free at this layer;
- and eliminable without changing upstream records.

## Prohibitions

This file does not permit:

- global mismatch measures;
- comparison across nonadjacent records;
- undeclared thresholds or normalization;
- metric distance;
- force, potential, or energy claims;
- causal language unsupported by a formal rule;
- or geometric primitives created from the comparison.

## Output

The output is a traceable `mismatch-gradient-record` conforming to `geometry_behavior_contract.md`.

## Status

Provisional RCMT interpretation. Ready for compression.
