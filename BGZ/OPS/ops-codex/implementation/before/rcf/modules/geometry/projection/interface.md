# Projection Interface

## Document Contract

- owner: RCF projection module
- type: Contract
- scope: discrete-to-continuous handoff
- authority: tier-7 projection interface
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: declare projection inputs, preserved relations, smoothing choices, and shadow outputs
- anti-backflow: projection may transform declared survivor records but may not select survivors, rewrite discrete geometry, or import downstream metrics and dynamics

## Inputs

Projection may consume identified discrete survivor records, adjacency, orientation, sector or plateau tags where documented, and a declared projection profile.

## Outputs

It may expose projected identifiers, preserved invariants, smoothed adjacency, shadow relations, approximation metadata, and loss or ambiguity records.

## Prohibitions

Projection does not define metrics, curvature, flow laws, physical meaning, or hidden reconstruction data.

