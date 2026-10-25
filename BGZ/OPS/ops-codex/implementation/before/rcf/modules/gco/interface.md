# GCO Interface

## Document Contract

- owner: RCF global coherence operator
- type: Contract
- scope: read-only aggregation boundary for GCO
- authority: tier-6 derived aggregation interface
- upstream dependencies: `/modules/lco/README.md` and `/core/S∞.RCF.Invariants.v1.1.md`
- status: active
- purpose: define the records GCO may consume and the summaries it may expose
- anti-backflow: GCO outputs may not modify LCO scoring, temporal parameters, local updates, or upstream admissibility

## Inputs

GCO may consume declared batches or recursions of LCO outcomes, their provenance, and the aggregation domain. Inputs must be read-only and must retain failed as well as surviving outcomes.

## Outputs

GCO may expose persistence, dispersion, structural viability, and global coherence summaries traceable to the supplied LCO records.

## Prohibitions

GCO may not score local candidates, select updates, schedule evaluation, retune temporal parameters, infer geometry, or act as a controller.

