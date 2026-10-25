# Temporal Interface Contract

## Document Contract

- owner: RCF temporal interface
- type: Contract
- scope: sealed operator-facing temporal parameter exposure
- authority: tier-5 temporal bridge interface
- upstream dependencies: `/modules/time/interface.md`
- status: active
- purpose: map Time-owned ordering and declared constraint-density inputs into sealed downstream-readable weights
- anti-backflow: consumers may read exposed parameters but may not retune, back-derive, or use them to reinterpret Time

## Inputs

The interface may consume Time records, declared local constraint-density records, and explicitly configured admissible parameter ranges.

## Outputs

It may expose immutable parameter records, including their source event index, applicable scope, admissible range, selected value, and provenance.

## Prohibitions

The interface may not infer geometry, basins, attractors, global coherence, physical duration, or operator outcomes.

