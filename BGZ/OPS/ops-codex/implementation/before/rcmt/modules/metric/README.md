# RCMT Refinement-Summary Metric Profile

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Contract
- scope: optional downstream metric summaries over traceable RCMT records
- upstream authority: `../../../rcf_compressed.md`
- depends_on: `../geometry_behavior/README.md` and `../geometry_behavior/geometry_behavior_contract.md`
- possible downstream adoption boundary: `../../physics_interface/local/emergent_metric_geometry/`
- status: compression-preparation
- anti-backflow: this profile cannot alter geometry, projection, LCO, echoon, survivor, or Geometry-Behavior records
- non-exclusivity: this profile does not own all RCMT metrics and does not constrain sibling metric modules

## Purpose

This directory defines one optional RCMT metric profile. It turns traceable Geometry-Behavior outputs into finite comparison and summary records for downstream analysis.

The profile is:

- downstream-only;
- representational;
- explicitly parameterized;
- locally supported;
- replaceable;
- and eliminable.

Removing it leaves upstream RCMT structure unchanged.

## Architectural Position

```text
RCF geometry and projection
             |
RCMT Geometry-Behavior
             |
this optional metric profile
             |
physics metric interface / cosmology / analysis
```

Parallel metric modules may consume the same upstream records and define different profiles. Adoption of this profile does not authorize its assumptions in a sibling module.

The existing physics metric interface does not automatically import this profile. Adoption there requires an explicit adapter or contract revision owned by that downstream interface.

## Inputs

This profile may consume:

- `mismatch-gradient-record`;
- `coherence-gradient-record`;
- `echoon-pattern-record`;
- `closure-reinjection-sequence`;
- `interaction-pattern-record`;
- `high-density-event-record`;
- documented continuous-shadow or smoothed-adjacency records;
- explicit downstream approximation records.

No `slush`, `plateau`, continuous field, coordinate, tensor, or derivative input is assumed unless an owning interface supplies it explicitly.

## Outputs

This profile may expose:

- `effective-comparison-record`;
- `gradient-summary-record`;
- `curvature-summary-record`;
- `echoon-density-summary`;
- `survivor-influence-summary`;
- `clumping-score-record`.

These outputs are optional summaries. They are not upstream geometry records, physical observables, or universal RCMT metrics.

## Canonical and Optional Forms

The canonical form in this directory is a finite record with:

- source identifiers;
- local support;
- feature encodings;
- normalization and weights;
- transformation rule;
- validity domain;
- and provisional status.

Coordinate, tensor, derivative, scalar-field, and potential forms are optional implementations. They require an explicit approximation profile and remain downstream-owned.

## Load Order

1. `metric_interface.md`
2. `rcmt_effective_metric.md`
3. `rcmt_gradient_tensor.md`
4. `rcmt_curvature_summary.md`
5. `rcmt_echoon_density_metric.md`
6. `rcmt_survivor_influence_metric.md`
7. `rcmt_clumping_potential.md`

## File Roles

- `metric_interface.md`: authoritative profile boundary
- `rcmt_effective_metric.md`: weighted local comparison template
- `rcmt_gradient_tensor.md`: first-order local variation summary
- `rcmt_curvature_summary.md`: second-order local variation summary
- `rcmt_echoon_density_metric.md`: echoon density-band comparison
- `rcmt_survivor_influence_metric.md`: pairwise survivor-record summary
- `rcmt_clumping_potential.md`: local aggregation-score template

## Extension Rules

This profile may be extended by adding a file that:

- depends on `metric_interface.md`;
- declares its inputs and output schema;
- preserves source traceability and adjacency support;
- identifies every free parameter;
- distinguishes canonical records from optional mathematical encodings;
- and does not strengthen upstream claims.

A sibling metric module may define a different interface, feature set, or mathematical family. It must identify itself as a separate profile rather than silently extending this contract.

## Prohibitions

This profile does not:

- define physical distance, duration, force, energy, or dynamics;
- create continuous support or coordinates;
- authorize nonlocal comparison;
- redefine mismatch or coherence;
- convert analogy into ontology;
- or claim priority over future metric modules.

## Status

Optional RCMT metric profile prepared for compression. `metric_interface.md` outranks broader prose in this directory.
