# RCMT Local Layer

## Document Contract

- owner: RCMT
- type: Contract
- scope: project-local, downstream of RCF
- purpose: define compression order, authority, and file roles for `rcmt/local`
- upstream authority: `theory/rcf_projects/rcf_compressed.md`
- compression guidance: `theory/rcf_projects/rcp-1.1.md`

## Authority Rule

RCF owns its invariant kernel, temporal ontology, LCO scoring contract, geometry stack, and projection handoff.

RCMT consumes those upstream contracts and defines model-local physics constraints, provisional implementations, downstream interpretations, and experimental description guidance.

No RCMT statement may redefine upstream ownership or convert a model-local interpretation into an RCF fact.

## Load Order

1. `invariants.md`
2. `constraints/direction_semantics.md`
3. `constraints/symmetry_chain.md`
4. `constraints/mode_space.md`
5. `constraints/projection.md`
6. `constraints/lco.md`
7. `constraints/geometry.md`
8. `constraints/measurement.md`
9. `constraints/decoherence.md`
10. `constraints/identity.md`
11. `constraints/cosmology.md`
12. `symmetry_groups/README.md`
13. `symmetry_groups/generators.md`
14. `symmetry_groups/actions.md`
15. `symmetry_groups/representations.md`
16. `symmetry_groups/patterns/*`
17. `interpretations.md`
18. `experiments.md`

## File Roles

### Invariant

- `invariants.md`

### Module-Local Contract

- `constraints/*`

### Provisional Implementation

- `constraints/symmetry_chain.md`
- current group selections in `constraints/mode_space.md`
- `symmetry_groups/generators.md`
- `symmetry_groups/actions.md`
- `symmetry_groups/representations.md`

### Provisional Pattern

- `symmetry_groups/patterns/*`

### Downstream Interpretation

- `interpretations.md`

### Downstream Interpretation and Usage Guidance

- `experiments.md`

## Compression Rules

- upstream constraints outrank RCMT statements
- RCMT invariants constrain RCMT models only
- provisional implementations must not be promoted to invariant status
- patterns, interpretations, and examples are non-canonical
- causal claims require formal support; otherwise use contract or interpretation language
- physical terms remain RCMT-local representations unless explicitly imported from upstream

## Status

- maturity: compression-preparation
- canonical RCMT source: `invariants.md`
- non-canonical material: patterns, interpretations, examples, and experimental guidance
