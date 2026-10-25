# RCMT Geometry-Behavior Module

## Document Contract

- owner: RCMT Geometry-Behavior
- type: Contract
- scope: candidate RCMT behavioral mappings over imported local geometry records
- upstream authority: RCF geometry, projection, and LCO contracts
- depends_on: `../../local/README.md`, `../../../abstract/echoon_minimal_closure/README.md`, and `../survivor_matter_structure/README.md`
- downstream consumers: `../metric/` and RCMT cosmology models
- status: compression-preparation
- anti-backflow: this module does not redefine upstream geometry, LCO evaluation, projection, echoon grammar, or survivor identity

## Purpose

This module defines candidate RCMT interpretations of how imported survivor, mismatch, and echoon records may be organized into local behavioral descriptions.

It is a theory-facing layer. It may name conditional relationships among records, but it does not promote those relationships to upstream facts or physical ontology.

## Authority Boundaries

The module consumes the following records read-only:

- RCF adjacency, geometry, projection, and geometry-dynamics records;
- upstream LCO mismatch and admissibility records;
- echoon state, density-band, transition, closure, reinjection, and decay labels;
- RCMT Matter Structure survivor, offset, and phase-cycle labels.

The module owns only:

- candidate mismatch-gradient descriptions;
- candidate coherence-gradient descriptions;
- candidate mappings between echoon records and local behavioral patterns;
- candidate closure-reinjection sequence descriptions;
- candidate survivor-interaction descriptions;
- candidate high-density event classifications.

It does not own the inputs from which those descriptions are derived.

## Core Interpretation

Within this module, **mismatch** denotes an imported local LCO mismatch record. Geometry-Behavior may compare mismatch records across adjacent configurations and describe increasing refinement burden as a mismatch gradient.

A coherence gradient is the gain-oriented complement: it describes increasing retained or recovered coherence across the same kind of adjacent comparison. Mismatch and coherence orderings are inverse when they use one declared comparison profile. Their numeric magnitudes are inverse only when the imported scoring contract explicitly defines that relation.

Echoon records may be grouped into coherent local patterns when the abstract echoon grammar permits the constituent states and transitions. Terms such as *field-like*, *boson-like*, *interaction-like*, and *curvature-like* name downstream RCMT analogies only.

No analogy in this module establishes a force, particle, field, metric, manifold, or spacetime.

## Adjacency Rule

All propagation and influence descriptions are adjacency-local.

A behavioral record may depend only on:

- the current local record;
- records attached to adjacent configurations;
- an explicitly imported allowed transition;
- and an explicitly imported local admissibility result.

The module does not permit global broadcast, action at a distance, or an unrecorded propagation state.

## Load Order

1. `geometry_behavior_contract.md`
2. `echoon_field.md`
3. `mismatch_gradients.md`
4. `coherence_gradients.md`
5. `closure_reinjection_cycles.md`
6. `boson_like_patterns.md`
7. `survivor_interaction_rules.md`
8. `high_density_events.md`

## File Roles

- `geometry_behavior_contract.md`: authoritative module interface and prohibitions
- `echoon_field.md`: candidate organization of echoon records
- `mismatch_gradients.md`: loss-oriented comparison of local mismatch records
- `coherence_gradients.md`: gain-oriented reversal of local mismatch ordering
- `closure_reinjection_cycles.md`: candidate sequencing of imported event labels
- `boson_like_patterns.md`: candidate higher-order pattern classification
- `survivor_interaction_rules.md`: candidate local coupling descriptions
- `high_density_events.md`: candidate classification of locally concentrated records

## Downstream Use

The Metric module may summarize Geometry-Behavior outputs, but it may not reinterpret them as fundamental metric structure. Cosmology models may aggregate local outputs under their own contracts, but they may not backflow global claims into this module.

## Compression Rules

- upstream contracts outrank every statement in this module;
- all model claims remain explicitly RCMT-local and provisional;
- every output must be traceable to declared inputs;
- no hidden state may be introduced;
- no physical or ontological claim may be inferred from analogy language;
- examples and named correspondences are noncanonical.

## Status

This module is a candidate RCMT interpretation prepared for compression. Its contract is normative for module use; its behavioral mappings remain provisional.
