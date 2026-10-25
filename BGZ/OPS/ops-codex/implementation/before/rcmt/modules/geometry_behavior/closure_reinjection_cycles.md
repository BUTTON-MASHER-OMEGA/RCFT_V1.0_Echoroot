# Closure-Reinjection Sequences

## Document Contract

- owner: RCMT Geometry-Behavior
- type: Interpretation
- scope: candidate sequencing of imported closure and reinjection event labels
- depends_on: `geometry_behavior_contract.md` and `../../../abstract/echoon_minimal_closure/local/abstract_interface.md`
- status: provisional
- anti-backflow: this file describes event sequences and does not define event mechanics or causes

## Purpose

This file defines candidate RCMT sequence descriptions for local closure and reinjection records associated with survivor neighborhoods.

## Inputs

- imported `closure-event` labels;
- imported `reinjection-event` labels;
- imported `decay-event` labels where present;
- local survivor record identifiers;
- local mismatch samples;
- explicit adjacency records;
- and local ordering supplied by the upstream record set.

This file does not create temporal order. It consumes only order already present in the input records.

## Canonical Sequence

A `closure-reinjection-sequence` contains:

1. a local pre-event record;
2. an imported closure-event label;
3. a locally ordered post-closure record;
4. an imported reinjection-event or decay-event label, when present;
5. a locally ordered result record;
6. source and adjacency support for every step.

The sequence may be incomplete. Missing events must remain explicit rather than being inferred.

## Candidate Cycle Classification

A sequence may be labeled cyclic only when the same declared event pattern recurs in a finite record set. A cycle label does not imply:

- periodicity;
- harmonic motion;
- an internal clock;
- energy exchange;
- or a universal update law.

Descriptors such as tighten, release, and retighten are optional prose labels for recorded transitions and are not canonical operators.

## Echoon Association

A reinjection event may be associated with a subsequent echoon-pattern record when:

- the records share an explicit local neighborhood;
- their ordering is supplied by the source data;
- and every echoon transition is valid under the abstract grammar.

The association does not establish that reinjection causes echoon formation.

## Survivor Stability Association

A sequence may be correlated with an imported survivor status before and after the sequence. Allowed result labels are:

- retained;
- changed;
- unresolved;
- or absent from the next local record.

Terms such as stabilization, destabilization, and collapse are noncanonical interpretations of these recorded outcomes.

## Multi-Survivor Sequences

Records from multiple survivors may be grouped only when their neighborhoods are connected by explicit adjacency and their event order is declared. Synchronization may be reported only as an observed ordering relation; no shared hidden phase may be inferred.

## Prohibitions

This file does not permit:

- generation of closure, reinjection, or decay events;
- physical emission, radiation, oscillation, or energy language as canonical content;
- unrecorded causal links;
- global sequencing;
- hidden timing variables;
- or redefinition of survivor identity.

## Output

The output is a traceable `closure-reinjection-sequence` conforming to `geometry_behavior_contract.md`.

## Status

Provisional RCMT interpretation. Ready for compression.
