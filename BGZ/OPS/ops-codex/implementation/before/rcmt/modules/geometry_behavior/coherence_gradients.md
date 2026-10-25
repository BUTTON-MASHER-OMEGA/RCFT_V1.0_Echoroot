# Coherence Gradients

## Document Contract

- owner: RCMT Geometry-Behavior
- type: Interpretation
- scope: gain-oriented reversal of a valid adjacency-local mismatch ordering
- depends_on: `geometry_behavior_contract.md`, `mismatch_gradients.md`, `../../local/constraints/lco.md`, and `../../local/constraints/geometry.md`
- status: provisional
- anti-backflow: this file does not define coherence, LCO scoring, geometry, or metric structure

## Purpose

This file defines the gain-oriented complement to mismatch gradients.

A coherence gradient records ordered variation in retained or recovered coherence across adjacent projected configurations. Where mismatch gradients emphasize refinement burden and loss, coherence gradients emphasize admissible organization and gain.

The current upstream interface exposes mismatch samples, admissibility records, and coherence thresholds. It does not expose an independent coherence-sample channel. This file therefore derives only an inverse ordering from a valid mismatch-gradient record.

## Definition

For adjacent configurations \(a\) and \(b\), let a valid mismatch-gradient record order imported mismatch samples \(m_a\) and \(m_b\) under one declared comparison profile.

A `coherence-gradient-record` may store:

- the source mismatch-gradient record identifier;
- the reversed ordering of \(a\) and \(b\);
- any applicable imported coherence-threshold or admissibility records;
- and provisional status.

If \(m_a < m_b\), the mismatch ordering points from \(a\) toward \(b\), while the coherence ordering points from \(b\) toward \(a\). The coherence-gradient record adds no independent score.

The record is a finite local ordering view. It is not a differential operator or a vector on a manifold.

## Relation to Mismatch Gradients

Under one declared comparison profile:

- increasing coherence corresponds to decreasing mismatch;
- decreasing coherence corresponds to increasing mismatch.

The two gradient descriptions are therefore order-dual and reference the same source comparison.

The module must not assume a numeric relation such as:

\[
c = 1 - m
\]

or:

\[
\Delta c = -\Delta m
\]

unless the imported scoring contract explicitly defines the normalization and conversion.

## Candidate Sources

A consuming RCMT model may test correlations between coherence-gradient records and:

- survivor retention or admissible survivor changes;
- compatible echoon-pattern records;
- closure, reinjection, or decay event labels;
- phase-cycle labels;
- or adjacent projection records.

The correlation must identify its inputs and does not establish that any input generated the coherence variation.

## Candidate Survivor Response

A survivor-response candidate may associate a coherence-gradient record with a later admissible survivor record. Permitted descriptive labels include:

- offset retention;
- phase-cycle retention;
- local alignment;
- or transition toward the higher indexed coherence sample.

These are ordering descriptions. They do not assert motion, attraction, force, or optimization.

## Echoon Association

An echoon-pattern record may be compared with a coherence-gradient record when both are supported by the same adjacency neighborhood. Allowed descriptions include alignment, persistence, disruption, and recovery.

No description may create an echoon transition absent from the abstract grammar.

## Persistence and Loss

A coherence gradient may be labeled persistent when the same declared ordering recurs across a stated finite sequence of local records. It may be labeled weakened or absent when that ordering changes or disappears.

These labels summarize records. They do not define energy gain, self-repair, or a causal coherence mechanism.

## Downstream Use

Metric and Cosmology modules may consume coherence-gradient records as representational inputs. They must preserve:

- source traceability;
- adjacency support;
- the declared comparison profile;
- provisional status;
- and the distinction between order-duality and numeric inversion.

## Prohibitions

This file does not permit:

- global coherence measures;
- comparison across nonadjacent records;
- undeclared thresholds or normalization;
- an independent coherence score;
- numeric inversion of mismatch without an imported conversion rule;
- metric distance;
- force, potential, or energy claims;
- causal language unsupported by a formal rule;
- or geometric primitives created from the comparison.

## Output

The output is a traceable `coherence-gradient-record` conforming to `geometry_behavior_contract.md`.

## Status

Provisional RCMT interpretation. Ready for compression.
