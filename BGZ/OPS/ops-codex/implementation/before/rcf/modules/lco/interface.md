# LCO Interface

## Document Contract

- owner: RCF local coherence operator
- type: Contract
- scope: local candidate-evaluation boundary
- authority: tier-5 local execution interface
- upstream dependencies: `/core/rcd/README.md`, `/modules/time/interface.md`, and `/modules/temporal-interface/interface.md`
- status: active
- purpose: define explicit local inputs, outcomes, and state-transfer limits for LCO evaluation
- anti-backflow: LCO may consume sealed upstream records but may not retune them, infer global structure, or export local scores as upstream laws

## Inputs

LCO may consume a declared local relational state, an admissible candidate update, its bounded neighborhood, invariant checks, an event index, and sealed temporal-interface parameters.

## Outputs

LCO may expose the candidate identifier, admissibility result, mismatch or coherence diagnostics defined by this module, updated local record, and provenance sufficient for GCO aggregation.

## Prohibitions

LCO may not perform global selection, modify gamma or epsilon, schedule events, define geometry, infer physical meaning, or retain undeclared state between evaluations.

