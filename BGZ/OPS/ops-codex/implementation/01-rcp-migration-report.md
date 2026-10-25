# RCP-1.2 migration coverage and exceptions

## Document Contract

- owner: Codex approved maintenance record
- type: Observation
- scope: owner-approved D1–D11 cleanup and bounded verification, 2026-10-24
- authority: implementation report; no scientific authority over the cited source contracts
- upstream dependencies: approved-owner-register.txt, baseline.json and cited execution/source records
- status: executed with explicit unresolved metadata and skipped reproductions
- purpose: record actual changes, validation and remaining limitations
- anti-backflow: maintenance evidence does not invent upstream invariants, missing sources, mathematical necessity or physical validation

## Coverage and protocol interpretation

All **163 pre-existing RCMT Markdown files** received canonical RCP-1.2 headers. The new `smg_v2.1/REFACTOR_STATUS.md` and `rcmt_compressed.md` also comply with the opening/field structure, bringing the final corpus to **165** Markdown files. No RCMT Markdown file was silently excluded. There was no byte-frozen Markdown artifact inside this migrated corpus; the Lisi frozen concatenation lies outside RCMT and remains unchanged.

The RCF protocol correction is separate: one document H1 followed immediately by the canonical Document Contract, with no undefined extra sentinel. Fields are owner, type, scope, authority, upstream dependencies, status, purpose and anti-backflow in canonical order. Fenced examples are excluded when counting headings/contracts.

## Classification method and preservation

Existing explicit owner/type/scope/status/authority fields were retained. Legacy `depends_on` entries were checked against the actual source and named public owner interfaces before conversion; unresolved declarations are preserved in a migration-provenance section and do not become invented dependencies. Formerly headerless files were classified using their stated module/owner/purpose and existing parent interface requirements. Existing provisional, proposed and exploratory meanings remain scoped; directory placement does not confer new consumer permissions.

Additional document H1s were changed to section headings without removing their text. Pre-title path material and noncanonical header fields were preserved as migration provenance. RCP-1.1 compatibility/compliance prose is retained as legacy notes, not a current validation certificate. Header migration did not implement new projection, scoring or transport algorithms.

The final canonical records and legacy declarations are traceable through `rcmt-migration.json`, the source baseline, current files, and the exact derived registry. Later path/source-input refinements are in `followup-edits.json` and `path-edits.json`.

## Validation against RCP requirements

| Requirement | Check / result |
|---|---|
| Leading H1 and immediate contract | 165/165 pass |
| Single H1 / contract outside fenced examples | 165/165 pass |
| Eight ordered, nonempty canonical fields / allowed type | 165/165 pass |
| Anti-backflow and authority distinct from status | All have explicit boundaries; no active status substituted for authority |
| Canonical header path resolution | All declared resolved paths exist within theory; missing original inputs remain explicitly unresolved metadata |
| Dependency direction | No cycle in the current canonical header dependency graph; no downstream evidence promoted to kernel authority |
| Active inline Markdown file references | No missing or ambiguous references under document-relative lookup and reviewed legacy-root conventions, excluding intentionally preserved legacy-provenance text/fenced examples |
| Scope/body semantic review | Approved ownership, compactness, parameter, scoring, projection and indexing decisions propagated; proposed SMG policies remain provisional; numerical/physical evidence not promoted |
| Full readiness | **Not certified for the entire corpus:** four source documents retain unresolved upstream metadata, and the aggregate registry propagates that limitation |

This is more than field-presence checking: it includes placement, multiplicity, field/type rules, dependency resolution/cycles and an explicit owner/scope/evidence review. It is not an automated proof that every body assertion is correct. In particular, a canonical header does not resolve missing sources, establish a mathematical theorem or assign an ambiguous deployment-stage permission.

## Unresolved source metadata

- [rcmt/local/constraints/direction_semantics.md:1](C:/Users/blaur/projects/theory/rcmt/local/constraints/direction_semantics.md:1): upstream direction-bearing unit contract — exact upstream direction-bearing contract not identified by original header.
- [rcmt/modules/geometry_behavior/closure_reinjection_cycles.md:1](C:/Users/blaur/projects/theory/rcmt/modules/geometry_behavior/closure_reinjection_cycles.md:1): ../../../abstract/echoon_minimal_closure/local/abstract_interface.md — declared dependency does not resolve; source not invented.
- [rcmt/modules/geometry_behavior/echoon_field.md:1](C:/Users/blaur/projects/theory/rcmt/modules/geometry_behavior/echoon_field.md:1): ../../../abstract/echoon_minimal_closure/local/abstract_interface.md — declared dependency does not resolve; source not invented.
- [rcmt/modules/geometry_behavior/README.md:1](C:/Users/blaur/projects/theory/rcmt/modules/geometry_behavior/README.md:1): ../../../abstract/echoon_minimal_closure/README.md — declared dependency does not resolve; source not invented.

The exact direction-bearing upstream contract has not been recovered. The declared Echoon Minimal Closure sources do not resolve in the authorized input layout. No replacement owner, API or implementation was invented. These four files say metadata is unresolved/not compression-ready; the generated RCMT registry also remains not compression-ready. Of 165 final documents, 160 pass the header/dependency readiness screen, four source documents are withheld, and one generated aggregate carries their limitation. This is a readiness screen, not a universal semantic-compliance badge.

## Derived artifacts

`rcf/rcf_compressed.md` indexes 194 final RCF source documents. `rcmt/rcmt_compressed.md` indexes 164 final RCMT source documents, excluding itself. Both carry exact header registries, conservative summaries and reproducible source manifests containing base commit plus working-tree file hashes. They are subordinate indexes, not replacements for source-body mathematics. Source hashes were checked after the final edits. No claim that the base commit alone describes uncommitted changes is made.

Evidence: [projects/ops-codex/implementation/rcp-validation.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/implementation/rcp-validation.json:1), [projects/ops-codex/implementation/body-path-review.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/implementation/body-path-review.json:1), [projects/ops-codex/implementation/final-verification.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/implementation/final-verification.json:1).
