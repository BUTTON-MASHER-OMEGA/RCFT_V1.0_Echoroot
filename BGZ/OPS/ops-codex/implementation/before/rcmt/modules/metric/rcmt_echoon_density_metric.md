# RCMT Echoon Density Summary

## Document Contract

- owner: RCMT Refinement-Summary Metric Profile
- type: Interpretation
- scope: optional comparison of imported echoon density-band and pattern records
- depends_on: `metric_interface.md`, `../geometry_behavior/echoon_field.md`, and `rcmt_gradient_tensor.md`
- status: provisional profile definition
- anti-backflow: this summary does not redefine echoon density bands, coherence criteria, or transitions

## Purpose

This file defines a profile for comparing local echoon density-band and echoon-pattern records. The historical name **echoon density metric** does not imply a continuous field or an independent echoon density primitive.

## Inputs

The profile may consume:

- imported `echoon-density-band` labels contained in source records;
- `echoon-pattern-record`;
- local adjacency support;
- declared pattern-coherence criteria;
- optional first-order variation summaries.

It must not infer numeric density directly from a categorical band.

## Density-Band Encoding

A profile may define an explicit finite encoding:

\[
q_P : B_{\mathrm{echoon}} \rightarrow V_P
\]

where:

- \(B_{\mathrm{echoon}}\) is the imported set of density-band labels;
- \(V_P\) is a declared ordered or numeric value set;
- \(q_P\) is local to this metric profile.

The encoding does not modify the imported labels.

## Local Comparison

For adjacent echoon records \(e_a\) and \(e_b\), the profile may calculate:

\[
E_P(a,b) =
\delta_e\!\left(q_P(e_a),q_P(e_b)\right)
\]

where \(\delta_e\) is declared by the profile.

If the source includes a valid echoon-pattern record, the comparison may also include declared pattern features. Coherence is represented by the pattern's explicit criterion, not by a hidden scalar.

## Output Record

An `echoon-density-summary` contains:

- source echoon and adjacency identifiers;
- density-band encoding;
- optional pattern features;
- comparison rule;
- result;
- approximation metadata;
- and provisional status.

## Optional Field-Like Encoding

A downstream approximation may encode density-band summaries over continuous support as \(e(x)\) only when it supplies:

- a support map from source records to \(x\);
- interpolation or aggregation rules;
- normalization;
- and a validity regime.

The scalar-field form is optional and eliminable.

## Prohibitions

This profile does not permit:

- invented echoon transitions;
- causes for echoon behavior;
- an undeclared coherence score;
- field, force, or particle ontology;
- nonadjacent propagation;
- or claims that echoon density stabilizes or destabilizes survivors.

## Status

Provisional optional profile. Ready for compression.
