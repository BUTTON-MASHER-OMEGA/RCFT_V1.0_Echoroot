# Geometry Dynamics Interface

## Document Contract

- owner: RCF geometry-dynamics module
- type: Contract
- scope: downstream continuous-variation descriptors
- authority: tier-8 continuous-variation interface
- upstream dependencies: `/modules/geometry/continuous/interface.md`
- status: active
- purpose: expose mismatch, curvature-like variation, flow, and action summaries within declared profiles
- anti-backflow: variation descriptors may not create upstream geometry, become physical forces, or revise projection and survivor records

## Inputs

The module may consume continuous-shadow records, declared mismatch functionals, neighborhood structure, and explicitly selected variation profiles.

## Outputs

It may expose evaluated mismatch, curvature-like second-order variation, flow descriptors, stationary-action results, and approximation provenance.

## Prohibitions

The module may not introduce new primitives, infer physical fields or causes, select discrete survivors, or present profile equations as universal RCF laws.

