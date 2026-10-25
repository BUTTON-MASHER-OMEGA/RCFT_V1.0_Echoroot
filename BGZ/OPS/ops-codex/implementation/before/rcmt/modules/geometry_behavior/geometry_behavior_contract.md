# Geometry-Behavior Contract

## Document Contract

- owner: RCMT Geometry-Behavior
- type: Contract
- scope: inputs, outputs, permissions, and prohibitions for candidate local behavior descriptions
- depends_on: `README.md`
- status: authoritative module-local contract
- anti-backflow: outputs from this module cannot alter imported records or their ownership

## Inputs

The module may consume:

- adjacency and local geometry records from the upstream geometry stack;
- projected local configurations from the upstream projection handoff;
- mismatch, coherence-threshold, and transition-admissibility records from the upstream LCO contract;
- echoon labels and allowed transitions from Echoon Minimal Closure;
- survivor, offset-regime, and phase-cycle labels from RCMT Matter Structure.

Every consumed record must retain its source identifier and owner.

## Local Terms

### Mismatch Sample

A read-only local LCO mismatch record attached to a projected configuration.

### Mismatch Gradient

An RCMT-local ordered comparison of mismatch samples attached to adjacent configurations. It is not a derivative, force, or metric gradient.

### Coherence Gradient

An RCMT-local reversal of a valid mismatch-gradient ordering. It records increasing retained or recovered coherence without introducing an independent coherence score. It is not a derivative, force, or metric gradient.

For one declared comparison profile, mismatch and coherence gradients may be order-dual: increasing mismatch corresponds to decreasing coherence. Numeric inversion is valid only when the imported scoring contract supplies the required normalization or conversion rule.

### Echoon Pattern

An RCMT-local grouping of imported echoon labels whose transitions are permitted by the abstract echoon grammar.

### Closure-Reinjection Cycle

An RCMT-local sequence description composed from imported closure-event and reinjection-event labels. The description does not define either event or its cause.

### Survivor Response

A candidate association between an imported local behavioral record and a subsequent survivor record, provided the transition is locally admissible.

### High-Density Event

A candidate classification of a local neighborhood containing concentrated mismatch samples, echoon density labels, or survivor records. Thresholds must be declared by the consuming model.

## Outputs

The module may emit:

- `mismatch-gradient-record`;
- `coherence-gradient-record`;
- `echoon-pattern-record`;
- `closure-reinjection-sequence`;
- `survivor-response-candidate`;
- `interaction-pattern-record`;
- `high-density-event-record`.

Each output must contain:

- source record identifiers;
- adjacency support;
- the applied classification rule;
- provisional status;
- and any unresolved condition.

Outputs are descriptions, not new primitives.

## Allowed Transformations

The module may:

- compare adjacent mismatch samples;
- reverse a valid mismatch-gradient ordering to produce its coherence-gradient view;
- group compatible echoon transitions;
- sequence explicitly recorded closure and reinjection events;
- correlate adjacent survivor records with local behavioral records;
- classify local record concentrations using declared thresholds;
- pass traceable outputs to downstream Metric or Cosmology modules.

## Forbidden Transformations

The module must not:

- define or retune the LCO;
- generate mismatch values not present in imported records;
- generate independent coherence values or numeric inverse relations not present in imported records;
- redefine echoon states, transitions, density bands, or event labels;
- redefine survivor identity, offset regimes, or phase cycles;
- create nonadjacent propagation;
- treat a candidate sequence as a causal proof;
- introduce Physics Time, metric structure, a manifold, force, energy, field, or particle ontology;
- infer a universal physical law from a structural analogy;
- modify upstream or sibling records.

## Invariant Preservation

Every module operation must preserve:

- adjacency-local information flow;
- finite representability;
- source-record traceability;
- upstream ownership;
- projection directionality;
- and explicit provisional status.

If an operation cannot demonstrate these properties, it is invalid under this module contract.

## Failure Conditions

An output is invalid under Geometry-Behavior when:

- an input has no declared owner;
- adjacency support is absent;
- an echoon transition is not imported as allowed;
- a local threshold is omitted;
- a physical analogy is used as evidence;
- or a downstream metric or cosmological conclusion is fed back as an input.

## Non-Goals

This module does not provide:

- a physical dynamics law;
- a metric or curvature tensor;
- a theory of forces or particles;
- a survivor taxonomy;
- an echoon state machine;
- or a cosmological model.

## Status

This contract is authoritative for Geometry-Behavior. The mappings implemented by the remaining files are candidate RCMT interpretations and do not acquire upstream authority through compression.
