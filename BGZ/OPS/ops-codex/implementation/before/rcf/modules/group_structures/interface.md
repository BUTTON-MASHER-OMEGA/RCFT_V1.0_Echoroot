# Group Structures Interface

## Document Contract

- owner: RCF optional group-structure profile
- type: Contract
- scope: opt-in algebraic structure declarations
- authority: tier-4 optional algebraic profile interface
- upstream dependencies: `/core/algebra/README.md` and `/core/S∞.RCF.Invariants.v1.1.md`
- status: active
- purpose: expose declared group actions and admissibility checks without requiring a group family
- anti-backflow: adopted group structure may constrain its consumers but may not become an RCF invariant or exclude other invariant-compatible algebraic profiles

## Inputs

The profile may consume a declared carrier, operation, identity, inverse rule, action domain, and compatibility checks against the core algebraic substrate.

## Outputs

It may expose group identifiers, operation and action records, verified properties, exclusions local to the adopted profile, and validation provenance.

## Prohibitions

The profile may not claim uniqueness, infer physical symmetry, require a two-lane realization, or universalize example-specific exclusions.

