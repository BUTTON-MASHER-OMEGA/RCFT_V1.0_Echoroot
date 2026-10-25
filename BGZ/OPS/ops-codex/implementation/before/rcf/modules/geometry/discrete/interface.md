# Discrete Geometry Interface

## Document Contract

- owner: RCF discrete geometry module
- type: Contract
- scope: candidate and survivor geometry records
- authority: tier-7 discrete survivor-recording interface
- upstream dependencies: `/modules/gco/interface.md` and `/core/geometry/README.md`
- status: active
- purpose: expose current discrete candidate, sieve, mismatch, closure, and survivor records
- anti-backflow: survivor records may not redefine the execution stack, require a candidate geometry universally, or determine projection and continuous behavior

## Inputs

The module may consume declared discrete candidates, core-valid adjacency and orientation data, spinor-closure profiles where adopted, and read-only execution outcomes.

## Outputs

It may expose candidate identifiers, sieve criteria, mismatch records, closure results, density regimes, survivor status, and evaluation provenance.

## Prohibitions

The module does not own continuous shadows, smoothing, metrics, curvature, flow, physical interpretation, or universal geometry selection.

