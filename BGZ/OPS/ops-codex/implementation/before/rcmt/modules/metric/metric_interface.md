# Refinement-Summary Metric Interface

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Contract
- scope: allowed inputs, outputs, transformations, and extension points for this profile
- depends_on: `README.md` and `../geometry_behavior/geometry_behavior_contract.md`
- status: authoritative module-local interface
- anti-backflow: profile outputs cannot modify or reinterpret source records
- sibling boundary: this interface governs only this directory

## Required Inputs

Every operation requires:

- one or more source records permitted by `README.md`;
- explicit local or adjacency support;
- a declared feature-extraction rule;
- and a profile identifier.

## Optional Inputs

An operation may also consume:

- a normalization rule;
- finite weights;
- a neighborhood radius expressed as an adjacency path bound;
- a documented continuous-shadow record;
- a coordinate or embedding record;
- or a downstream metric-assignment record.

Optional inputs must identify their owner and validity domain.

## Profile Declaration

Each metric profile instance must declare:

```text
profile-id
source-record-types
support-domain
feature-map
normalization
weights
comparison-rule
validity-domain
approximation-status
```

No default weight, normalization, coordinate chart, topology, or continuum limit is hidden.

## Allowed Outputs

An output may contain:

- source-record identifiers;
- encoded feature values;
- a finite comparison or summary value;
- local support;
- profile parameters;
- approximation metadata;
- unresolved conditions;
- and provisional status.

## Allowed Transformations

This profile may:

- encode declared categorical or ordered labels numerically;
- compare records sharing explicit local support;
- aggregate over a finite declared adjacency neighborhood;
- calculate weighted finite summaries;
- calculate first- or second-order finite differences;
- expose an optional coordinate or tensor encoding when supplied with an explicit approximation record.

## Mathematical Gates

### Distance

A comparison value may be called a **metric** only if the profile declares and verifies its required metric axioms over the stated domain. Otherwise it is a comparison or dissimilarity summary.

### Derivative

A finite difference may be represented by derivative notation only when an explicit continuous or coordinate approximation supplies the required support and scale.

### Tensor

An array may be called a tensor only when its transformation rule is declared for the supplied representation.

### Curvature

A second-order summary remains curvature-like. Physical or geometric curvature requires a separate downstream contract.

### Potential

A scalar score remains an aggregation or clumping score. A physical potential requires a separate downstream theory and is not produced here.

## Forbidden Transformations

This profile must not:

- compare records without connected adjacency support;
- invent missing source values;
- infer a numeric coherence score from a coherence threshold;
- count order-dual mismatch and coherence features as independent evidence without declaring the duplication;
- introduce an undeclared continuum, chart, norm, derivative, tensor law, or metric axiom;
- infer causation from correlation;
- create dynamics from a summary;
- or feed outputs back into upstream admissibility.

## Sibling Compatibility

Other metric modules may:

- consume the same upstream records;
- use different features or mathematical structures;
- reject this profile's weights and encodings;
- and expose separate downstream outputs.

Cross-profile comparison requires an explicit conversion contract. Equal names do not imply equal semantics.

## Failure Conditions

An output is invalid under this interface when:

- source traceability is incomplete;
- local support is absent;
- a free parameter is undeclared;
- an optional approximation is presented as canonical;
- a term is named metric, tensor, derivative, curvature, or potential without satisfying its gate;
- or the output claims upstream authority.

## Status

Authoritative interface for this optional profile.
