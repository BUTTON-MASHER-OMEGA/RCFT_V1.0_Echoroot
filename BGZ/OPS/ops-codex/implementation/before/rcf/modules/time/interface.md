# Time Interface

## Document Contract

- owner: RCF time module
- type: Contract
- scope: event-index and refinement-order records
- authority: tier-4 temporal ordering interface
- upstream dependencies: `/core/calculus/event_axiom.md` and `/core/S∞.RCF.Invariants.v1.1.md`
- status: active
- purpose: expose non-metric event order without importing downstream rates or dynamics
- anti-backflow: consumers may index records by Time outputs but may not use downstream duration, density, geometry, or dynamics to redefine event order

## Inputs

Time may consume admissible non-temporal candidate-event records and a declared refinement relation.

## Outputs

Time exposes event identifiers, predecessor or successor relations, refinement indices, and order provenance. These outputs establish order, not duration.

## Prohibitions

Time does not define gamma, local constraint density, metric duration, causal laws, geometry, or physical time.

