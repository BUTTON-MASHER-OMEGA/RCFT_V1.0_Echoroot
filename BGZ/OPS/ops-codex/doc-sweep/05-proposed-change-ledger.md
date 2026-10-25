# Proposed change ledger

## Document Contract

- owner: Codex documentation audit
- type: Observation
- scope: proposal-only audit of authorized theory inputs, 2026-10-24
- authority: independent assessment; no authority to revise source contracts
- upstream dependencies: cited working-tree sources, captured Git history and prior scoped verification records
- status: planning proposal; source changes not executed
- purpose: propose controlled consistency and evidence-language repairs
- anti-backflow: proposed downstream clarifications cannot redefine upstream invariants or promote hypotheses into source authority

## How to review this ledger

Every entry is a proposal, not an applied edit. “Future implementation authorization” is required for every source mutation because this request authorizes planning only. A separate **owner decision** is identified where wording would choose an unresolved mathematical or architectural commitment. Editorial repairs can later be batch-authorized; they do not justify a new permission question during this planning pass.

Priorities identify research impact, while risks identify the chance of changing meaning. Replacement wording is ready for review; conditional replacements must not be applied before their decision gate is resolved. No row authorizes changing historical failures, inventing evidence, broad source deletion, or adopting a new model.

## C01 — Two-lane rationale and authority

Priority: **Critical decision**.

**File and section:** rcmt/local/invariants.md §2; rcf/core/relational-constraints/geometry.md §§1/6/summary

**Current wording or precise claim:** RCMT requires two lanes; RCF profile calls its architecture “the minimal structure,” while allowing shared representation.

**Identified inconsistency:** RCMT may adopt a stricter profile, but the original necessity proof was not recovered. Do not infer that invariant labeling supplies one.

**Evidence and source references:** [rcmt/local/invariants.md:25](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:25); [rcf/core/relational-constraints/geometry.md:28](C:/Users/blaur/projects/theory/rcf/core/relational-constraints/geometry.md:28); [projects/ops-codex/doc-sweep/evidence/rcf-two-lane-optionalization.diff:1](C:/Users/blaur/projects/theory/projects/ops-codex/doc-sweep/evidence/rcf-two-lane-optionalization.diff:1)

**Proposed replacement wording:** Add: “This is a current RCMT-local architectural requirement, not a necessity theorem derived here from RCF or projection alone. Its historical memory/motion rationale is recorded separately.” In the RCF profile, replace universal-sounding minimality with “This profile adopts these roles to represent the listed functions; no universal minimality proof is supplied here.”

**Change rationale:** Clarification of evidence/authority, preserving the current requirement. Alternatives B/C in report 02 remain unapplied proposals.

**Affected downstream documents:** rcmt/local/constraints/{symmetry_chain,mode_space,projection}.md; rcmt/global/constraints.md; physics_interface/pipeline/overview.md; RCMT README.

**Risk and approval:** High semantic sensitivity. Explicit owner approval of the provenance wording; any relaxation/promotion is a separate decision D1.

## C02 — Lane independence, product and symplectic notation

Priority: **High**.

**File and section:** rcmt/local/constraints/mode_space.md §§2–4; constraints/projection.md §2; symmetry_chain.md §2

**Current wording or precise claim:** “neither lane may be representationally reducible to the other”; “Π:Sp(2n)→G”; “non-degenerate bilinear form”; bare product candidate notation.

**Identified inconsistency:** Carrier, field, alternation, independence criterion and projection type are unspecified. Separate roles do not establish independent commuting group factors.

**Evidence and source references:** [rcmt/local/constraints/mode_space.md:39](C:/Users/blaur/projects/theory/rcmt/local/constraints/mode_space.md:39); [rcmt/local/constraints/mode_space.md:72](C:/Users/blaur/projects/theory/rcmt/local/constraints/mode_space.md:72); [rcmt/local/constraints/projection.md:47](C:/Users/blaur/projects/theory/rcmt/local/constraints/projection.md:47); [rcmt/local/constraints/symmetry_chain.md:49](C:/Users/blaur/projects/theory/rcmt/local/constraints/symmetry_chain.md:49)

**Proposed replacement wording:** Add: “A realization declares its field, state carrier, nondegenerate alternating form Ω, symmetry action and projection domain/codomain. Transformations claimed symplectic satisfy SᵀΩS=Ω. Independence has the explicitly declared model-local meaning; product-group expressions are candidate realizations, not consequences of two roles.” Retain current prohibition pending a decision on its precise meaning; mark the displayed group-to-geometry arrow schematic until typed.

**Change rationale:** Clarify a named mathematical structure without adopting OPS real form or forcing a new realization.

**Affected downstream documents:** RCMT invariants §2; README; measurement/decoherence contracts; symmetry patterns; OPS cross-reference notes.

**Risk and approval:** High. Owner decision D1/D2 required for independence definition, real form or carrier choice; alternation is a definitional clarification.

## C03 — Obsolete projection-stage narrative

Priority: **High**.

**File and section:** rcmt/physics_interface/pipeline/overview.md §§3–5

**Current wording or precise claim:** After LCO: free of mismatch/time; projection is “first non-local” and introduces metric, cones, geometric time; SO(9)→SO(5)×SO(3,1).

**Identified inconsistency:** Unqualified narrative bypasses locality, time bridge/metric staging and current symmetry-candidate status.

**Evidence and source references:** [rcmt/physics_interface/pipeline/overview.md:80](C:/Users/blaur/projects/theory/rcmt/physics_interface/pipeline/overview.md:80); [rcmt/physics_interface/pipeline/overview.md:100](C:/Users/blaur/projects/theory/rcmt/physics_interface/pipeline/overview.md:100); [rcmt/physics_interface/README.md:66](C:/Users/blaur/projects/theory/rcmt/physics_interface/README.md:66); [rcmt/local/constraints/symmetry_chain.md:84](C:/Users/blaur/projects/theory/rcmt/local/constraints/symmetry_chain.md:84)

**Proposed replacement wording:** Replace with: “Local scoring exposes candidate diagnostics; it does not guarantee zero mismatch or basin stability. Upstream event order is already available. RCMT projection is a representational map preserving local information-flow constraints. Physics-time and metric records enter through their named bridges. Lorentz-compatible representation is required but not realized here; the SO candidates are provisional.” Preserve the removed historical arrows in a dated history note.

**Change rationale:** Downgrade unsupported outcomes; clarify stage ownership; archive obsolete arrows rather than repair them into a new cascade.

**Affected downstream documents:** pipeline/physics_interface_contract.md; pipeline/projection_contract.md; local/time_bridge and emergent_metric_geometry overviews; README.

**Risk and approval:** High. Explicit stage/scope decision D3 before applying; replacement does not itself implement Lorentz projection.

## C04 — Physics consumer permissions

Priority: **High**.

**File and section:** rcmt/physics_interface/README.md allowed inputs; pipeline/physics_interface_contract.md §§3–6; local/adapters/local_to_physics_adapter.md

**Current wording or precise claim:** README permits declared local inputs; PIC prohibits adjacency/coherence/upstream modes for “physics modules”; adapter allows local primitives.

**Identified inconsistency:** Same consumer noun has incompatible import permissions unless distinct stages are intended.

**Evidence and source references:** [rcmt/physics_interface/README.md:30](C:/Users/blaur/projects/theory/rcmt/physics_interface/README.md:30); [rcmt/physics_interface/pipeline/physics_interface_contract.md:1](C:/Users/blaur/projects/theory/rcmt/physics_interface/pipeline/physics_interface_contract.md:1); [rcmt/physics_interface/local/adapters/local_to_physics_adapter.md:14](C:/Users/blaur/projects/theory/rcmt/physics_interface/local/adapters/local_to_physics_adapter.md:14)

**Proposed replacement wording:** Conditional on D3: “Construction-facing interface modules consume declared local records through adapters. Effective-output consumers receive only explicitly exported projection/metric/time/field artifacts and cannot inspect raw upstream adjacency or scores. Each module declares which consumer class it implements.” If the owner instead selects one interface family, mark the other historical with an exact successor.

**Change rationale:** Clarification only after scope selection; otherwise this would alter permissions.

**Affected downstream documents:** physics_interface/local/interfaces.md; local module interfaces; pipeline contracts; SMG physics bindings.

**Risk and approval:** High. Explicit owner decision D3; never resolve by unioning allowed imports.

## C05 — Compactness scope propagation

Priority: **High**.

**File and section:** rcmt/global/constraints.md §6; local/constraints/projection.md §6; README Build Expectations

**Current wording or precise claim:** “No non-compact global groups/symmetries” versus invariant 10’s geometric-only blanket restriction and mode-lane exception.

**Identified inconsistency:** A narrower compact representation profile can coexist, but a universal mode-group ban conflicts with the current parent wording.

**Evidence and source references:** [rcmt/global/constraints.md:62](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:62); [rcmt/local/constraints/projection.md:138](C:/Users/blaur/projects/theory/rcmt/local/constraints/projection.md:138); [rcmt/README.md:68](C:/Users/blaur/projects/theory/rcmt/README.md:68); [rcmt/local/invariants.md:93](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:93)

**Proposed replacement wording:** Conditional on D4: “Global geometric symmetry groups are subject to invariant 10’s compactness restrictions. The symplectic lane is governed by its declared preservation/admissibility contract. A global-invariant implementation that additionally requires a compact transport image must declare that narrower profile explicitly.”

**Change rationale:** Clarify current exemption without asserting any particular symplectic real form acceptable.

**Affected downstream documents:** local/constraints/symmetry_chain.md; mode_space.md; global invariants; physics-interface group constraints; README.

**Risk and approval:** High. Explicit decision D4 on whether global §6 is intentionally a stricter representation profile.

## C06 — Parameter identity and ranges

Priority: **High**.

**File and section:** rcmt/local/invariants.md §11; RCF geometry sweep/plateau prose; LCO memory and temporal-adapter cross-links

**Current wording or precise claim:** Gamma is called constraint density in RCMT/table; LCO gamma is in [0,1], while sweep gamma reaches 5.

**Identified inconsistency:** The existing owner defines density κ and sealed gamma ranges; the sweep does not declare a conversion or separate identity.

**Evidence and source references:** [rcmt/local/invariants.md:105](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:105); [rcf/modules/lco/lco-memory.md:18](C:/Users/blaur/projects/theory/rcf/modules/lco/lco-memory.md:18); [rcf/modules/temporal-interface/local-constraint-density.md:65](C:/Users/blaur/projects/theory/rcf/modules/temporal-interface/local-constraint-density.md:65); [rcf/modules/temporal-interface/temporal-weights-adapter.md:53](C:/Users/blaur/projects/theory/rcf/modules/temporal-interface/temporal-weights-adapter.md:53)

**Proposed replacement wording:** Add: “κ denotes local constraint density; γ_LCO denotes the sealed memory weight in its declared admissible range. Historical γ-sweep values retain their original notation until their profile and relation to κ/γ_LCO are identified; no conversion is asserted here.” Only after that decision update invariant §11’s parenthetical symbol and dependent equations.

**Change rationale:** Clarification; no invented rescaling or extension of [0,1].

**Affected downstream documents:** geometry/discrete/mismatch-function.md and density table; adjacent/gamma-plateau-behavior.md; RCMT time constraints; SMG gamma_generations.md.

**Risk and approval:** High. Explicit decision D5 before changing equation symbols or parameter domains.

## C07 — Density formula, table and provenance

Priority: **High**.

**File and section:** rcf/modules/geometry/discrete/mismatch-function.md; candidate-geometries/density-regimes.md; survivor-statistics files

**Current wording or precise claim:** Fixed-coefficient affine F; table has nonzero second difference 0.122 for 600-cell at gamma 0/1/2; claimed observed transitions.

**Identified inconsistency:** Not the same exact affine function at printed precision; measure normalization, coefficients and run provenance absent from these records.

**Evidence and source references:** [rcf/modules/geometry/discrete/mismatch-function.md:24](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/mismatch-function.md:24); [rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:38](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:38); [rcf/modules/geometry/discrete/e8/survival-stats.md:41](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/e8/survival-stats.md:41)

**Proposed replacement wording:** Add above table: “Historical repository-reported values. Their generation record, parameter convention and connection to the displayed fixed-coefficient model have not been established in this audit; transition thresholds remain reported values, not independently reproduced derivations.” Add to formula: “Reproduction requires the declared spherical measure, normalized vertex set and drag coefficient/profile.” Preserve all current numbers until the source is recovered.

**Change rationale:** Downgrade unverified provenance, not falsification or numerical repair. D6 decides whether table or model is authoritative for that experiment.

**Affected downstream documents:** 600-cell/dual-5-demicube/e8 survival-stats; adjacent plateau/mode records; GCM indexing; downstream density interpretations.

**Risk and approval:** High. Owner decision D6; no fabricated run IDs, fits, error bars or recalculated table.

## C08 — Regime-local survival versus all-plateau rule

Priority: **High**.

**File and section:** rcf/modules/geometry/adjacent/gamma-plateau-behavior.md §§1/4

**Current wording or precise claim:** “Survivors must remain stable across all plateaus”; “F≤1.0 at all γ-levels.”

**Identified inconsistency:** Descriptor layer imposes stronger admission than upstream regime-specific sieve and documented high-gamma 600-cell failure.

**Evidence and source references:** [rcf/modules/geometry/adjacent/gamma-plateau-behavior.md:32](C:/Users/blaur/projects/theory/rcf/modules/geometry/adjacent/gamma-plateau-behavior.md:32); [rcf/modules/geometry/adjacent/gamma-plateau-behavior.md:101](C:/Users/blaur/projects/theory/rcf/modules/geometry/adjacent/gamma-plateau-behavior.md:101); [rcf/modules/geometry/discrete/survival-sieve.md:38](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/survival-sieve.md:38)

**Proposed replacement wording:** Replace with: “Survivor status is scoped to the candidate’s declared density regime and evaluation profile. Across-plateau stability, where separately tested, is an additional descriptor, not a universal sieve requirement.”

**Change rationale:** Clarify downstream scope and remove accidental backflow into admission.

**Affected downstream documents:** adjacent/survivor-properties.md and symmetry-alignment.md; GCM plateau descriptors; SMG generation dictionary.

**Risk and approval:** Medium implementation risk, high evidence relevance. Owner confirmation D6 if an intentionally narrower robust-survivor class was meant.

## C09 — Dual-projection consistency typing and ownership

Priority: **High**.

**File and section:** rcf/modules/geometry/projection/dual_projection.md §§1–3

**Current wording or precise claim:** Back projection rejects discrete states; G=Πdown(Πup(G)) while Πdown is described as producing consistency conditions.

**Identified inconsistency:** Return type and equality undefined; downstream filtering appears to select survivors forbidden by the interface.

**Evidence and source references:** [rcf/modules/geometry/projection/dual_projection.md:40](C:/Users/blaur/projects/theory/rcf/modules/geometry/projection/dual_projection.md:40); [rcf/modules/geometry/projection/interface.md:22](C:/Users/blaur/projects/theory/rcf/modules/geometry/projection/interface.md:22); [rcf/modules/geometry/projection/projection-invariants.md:24](C:/Users/blaur/projects/theory/rcf/modules/geometry/projection/projection-invariants.md:24)

**Proposed replacement wording:** Conditional on D7: “Πdown exposes compatibility predicates over an already admitted survivor and its projected record. Failure is a projection-profile diagnostic and does not change upstream survivor status. Dual closure means that the declared compatibility predicate holds.” Retain the old fixed-point formula as historical/schematic until a graph-returning reconstruction map or explicit equivalence is supplied.

**Change rationale:** Clarification or contract change depending on intended feedback semantics; does not assert an inverse for a lossy map.

**Affected downstream documents:** projection invariants/outputs; adjacent closure records; SMG Dual Closure thesis cross-reference; compressed shard.

**Risk and approval:** High. Explicit projection-owner decision D7; do not implement a new predicate or feedback algorithm in cleanup.

## C10 — Scoring, selection and aggregation actors

Priority: **High**.

**File and section:** rcmt/README.md Build Expectations; global/constraints.md §4; physics_interface/pipeline/overview.md §3

**Current wording or precise claim:** All mode selection expressed through LCO; LCO/GCO prune neighborhoods; post-LCO state declared perfectly coherent.

**Identified inconsistency:** LCO scorer does not select; GCO is read-only aggregation, not a cleanup controller.

**Evidence and source references:** [rcmt/README.md:68](C:/Users/blaur/projects/theory/rcmt/README.md:68); [rcmt/global/constraints.md:38](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:38); [rcf/modules/lco/lco-operator.md:181](C:/Users/blaur/projects/theory/rcf/modules/lco/lco-operator.md:181); [rcf/modules/gco/interface.md:14](C:/Users/blaur/projects/theory/rcf/modules/gco/interface.md:14)

**Proposed replacement wording:** Replace actor claims with: “Local candidate scores are supplied by the LCO contract; an explicitly declared update policy selects/adopts updates. GCO aggregates both failed and surviving records without modifying them. Preservation of a global invariant is required of the declared admissible update moves.”

**Change rationale:** Clarification aligning consumers with upstream ownership; no new update policy.

**Affected downstream documents:** RCMT LCO consumption, global invariant examples, pipeline overview, matter/geometry summaries that name operator agency.

**Risk and approval:** Medium. Future implementation authorization; explicit owner review if any existing module intentionally assigns GCO mutation.

## C11 — Descent and stochastic example correctness

Priority: **High**.

**File and section:** rcf/modules/lco/lco-update-policy.md §§3/5/6

**Current wording or precise claim:** Accepted updates do not increase incoherence; minimum-score selection and Boltzmann sampling are called invariant-compatible; identical inputs imply identical decisions.

**Identified inconsistency:** Minimum among candidates may exceed current score; unrestricted finite-temperature sampling allows uphill moves; stochastic determinism is overclaimed.

**Evidence and source references:** [rcf/modules/lco/lco-update-policy.md:77](C:/Users/blaur/projects/theory/rcf/modules/lco/lco-update-policy.md:77); [rcf/modules/lco/lco-update-policy.md:115](C:/Users/blaur/projects/theory/rcf/modules/lco/lco-update-policy.md:115); [rcf/modules/lco/lco-update-policy.md:140](C:/Users/blaur/projects/theory/rcf/modules/lco/lco-update-policy.md:140)

**Proposed replacement wording:** Replace example annotations with: “For an accepted-update non-increase policy, include the current candidate or reject any proposal above Ccurrent. Stochastic examples specify a normalized conditional distribution on the declared admissible set. Equal non-random inputs determine the same conditional law; deterministic replay additionally requires an explicit random trace/seed contract.” If uphill exploration is intended, declare it a separate profile and its descent criterion.

**Change rationale:** Correct mathematical conditions; distinguish score, proposal distribution and acceptance.

**Affected downstream documents:** lco-invariants.md §6; LCO README; optional Lyapunov gate cross-reference; downstream stochastic profiles.

**Risk and approval:** High. Explicit decision D8 about pathwise non-increase versus other criteria; no silent change to the invariant.

## C12 — LCO uniqueness and physical explanation tone

Priority: **Medium**.

**File and section:** rcf/modules/lco/README.md design paragraph; rcmt/README.md What RCMT Is

**Current wording or precise claim:** “only operator we have identified”; RCMT “explains” mass/randomness/etc and “provides mechanisms.”

**Identified inconsistency:** Contract allows a family of scoring functions; local physics requirements are representations, not verified mechanisms.

**Evidence and source references:** [rcf/modules/lco/README.md:30](C:/Users/blaur/projects/theory/rcf/modules/lco/README.md:30); [rcf/modules/lco/lco-operator.md:110](C:/Users/blaur/projects/theory/rcf/modules/lco/lco-operator.md:110); [rcmt/README.md:24](C:/Users/blaur/projects/theory/rcmt/README.md:24); [rcmt/local/constraints/lco.md:27](C:/Users/blaur/projects/theory/rcmt/local/constraints/lco.md:27)

**Proposed replacement wording:** Use: “LCO names the current family of local scoring contracts; a uniqueness theorem for its functional form is not claimed here.” In RCMT: “RCMT explores candidate representations of mass, randomness, measurement and cosmological structure under the listed constraints. These mappings remain under mathematical, computational and empirical evaluation.”

**Change rationale:** Downgrade summary overstatement while keeping named hypotheses and model requirements.

**Affected downstream documents:** RCMT introductory summaries, onboarding/navigation summaries; no wholesale rewrite of already scoped matter contracts.

**Risk and approval:** Medium. Future implementation authorization; owner wording review, no new invariant decision.

## C13 — Chromatic-12 / 36-mode evidence chain

Priority: **High**.

**File and section:** rcf/modules/geometry/adjacent/mode-structure.md §§1–3; GCM survivor_indexing.md

**Current wording or precise claim:** Mode count is “not imposed” and “emergent,” while candidates lacking stable 12-mode base are eliminated.

**Identified inconsistency:** Missing provenance prevents distinguishing independent observation from the adopted filter and bookkeeping product.

**Evidence and source references:** [rcf/modules/geometry/adjacent/mode-structure.md:34](C:/Users/blaur/projects/theory/rcf/modules/geometry/adjacent/mode-structure.md:34); [rcf/modules/geometry/adjacent/mode-structure.md:52](C:/Users/blaur/projects/theory/rcf/modules/geometry/adjacent/mode-structure.md:52); [rcf/modules/gcm_interface/survivor_indexing.md:28](C:/Users/blaur/projects/theory/rcf/modules/gcm_interface/survivor_indexing.md:28)

**Proposed replacement wording:** Use: “Current descriptor records organize a 12-element base across three plateau labels, yielding a 36-index product. Claims of independent emergence require the candidate set, motif test and evaluation provenance; the indexing record alone does not supply that derivation.” Keep any stronger claim as attributed to its future recovered artifact rather than deleting history.

**Change rationale:** Clarify evidence and conditionality; preserve current index contract and SMG mapping.

**Affected downstream documents:** geometry-adjacent README/symmetry alignment; GCM sectorization; SMG bindings, Dual Closure and dictionary.

**Risk and approval:** High. Owner decision D6/D9 before changing imported structural guarantees or evidential status.

## C14 — Triangle compatibility ownership and carrier

Priority: **High**.

**File and section:** rcf/modules/gcm_interface/sectorization.md §§1–3; discrete/survival-sieve.md §6

**Current wording or precise claim:** LCO requires triangle packing and projected structure must support it; 600→120 route said to preserve adjacency.

**Identified inconsistency:** LCO is geometry-neutral; graph adjacency, face incidence and relation motifs are conflated. Fiber-level dual transport is not a singleton adjacency map.

**Evidence and source references:** [rcf/modules/gcm_interface/sectorization.md:21](C:/Users/blaur/projects/theory/rcf/modules/gcm_interface/sectorization.md:21); [rcf/modules/gcm_interface/sectorization.md:42](C:/Users/blaur/projects/theory/rcf/modules/gcm_interface/sectorization.md:42); [rcf/modules/geometry/discrete/survival-sieve.md:41](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/survival-sieve.md:41); [projects/lisi-smg-test/REPORT_PHASE5.md:15](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE5.md:15)

**Proposed replacement wording:** Add: “Triangle compatibility is a requirement of the declared downstream geometric evaluation profile, not a topology imposed by the LCO scorer. Each projection route identifies its source/target carriers and preserved incidence relation. No claim of triangles in the 120-cell edge graph follows from a triangle-motif or face-incidence requirement.”

**Change rationale:** Clarify actor and relation types; do not replace the intended projection with the Lisi construction without a map.

**Affected downstream documents:** candidate-geometries.md; projection adjacency-preservation; GCM sector graph payload; SMG routing translations.

**Risk and approval:** High. Explicit owner decision D7/D9 on motif/graph/incidence meaning; no new carrier forced.

## C15 — Local observations labeled empirical invariants

Priority: **Medium**.

**File and section:** rcf/modules/geometry/discrete/600-cell/survival-stats.md invariants; adjacent φ-spectrum descriptions

**Current wording or precise claim:** φ-structured behavior is “an empirical invariant”; related bodies call reported tiers forced or invariant.

**Identified inconsistency:** Observation headers do not make reported patterns canonical or empirically validated physics.

**Evidence and source references:** [rcf/modules/geometry/discrete/600-cell/survival-stats.md:44](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/600-cell/survival-stats.md:44); [rcf/modules/geometry/projection/phi-spectrum.md:58](C:/Users/blaur/projects/theory/rcf/modules/geometry/projection/phi-spectrum.md:58)

**Proposed replacement wording:** Use: “Repository-reported computational pattern under the named evaluation profile; not a kernel invariant or physical empirical law. Reproduction and uncertainty are as recorded in the cited run artifact.” Where no artifact is located, say so instead of inventing one.

**Change rationale:** Downgrade over-authoritative nouns; retain reported patterns and profile restrictions.

**Affected downstream documents:** discrete survivor records; projection φ-spectrum; adjacent descriptors; GCM annotations.

**Risk and approval:** Medium. Future implementation authorization and evidence-owner review; changes to actual admissibility remain separate.

## C16 — GCM mode-source binding repair

Priority: **Medium**.

**File and section:** rcmt/smg_v2.1/bindings/gcm_bindings.md §3

**Current wording or precise claim:** upstream source gcm_interface/mode_structure.md (absent).

**Identified inconsistency:** Existing candidate sources are GCM survivor_indexing.md §3 and geometry/adjacent/mode-structure.md; neither path should be silently substituted without scope checking.

**Evidence and source references:** [rcmt/smg_v2.1/bindings/gcm_bindings.md:36](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/bindings/gcm_bindings.md:36); [rcf/modules/gcm_interface/survivor_indexing.md:60](C:/Users/blaur/projects/theory/rcf/modules/gcm_interface/survivor_indexing.md:60)

**Proposed replacement wording:** Proposed upstream-source field: “theory/rcf/modules/gcm_interface/survivor_indexing.md, §3 (36-mode indexing), conditional on evaluated survivors.” Keep geometry-adjacent observations as provenance dependencies, not a new direct import of sibling internals.

**Change rationale:** Editorial dependency repair if the existing index contract is the intended import; otherwise mark unresolved.

**Affected downstream documents:** SMG smg_contract/README; frozen reference remains unchanged; regenerated index references only after confirmation.

**Risk and approval:** Low–medium. Confirm binding identity with owner D9; no new upstream file or payload invented.

## C17 — Physics binding path and boundary provenance

Priority: **Medium**.

**File and section:** rcmt/smg_v2.1/bindings/physics_bindings.md §§1–5

**Current wording or precise claim:** physics_interface/conversion_laws/formal_structure.md; other sources named only as final boundary payload.

**Identified inconsistency:** Path omits local/; actual conversion document defines supports and final boundary tuple but no instantiated experiment payload.

**Evidence and source references:** [rcmt/smg_v2.1/bindings/physics_bindings.md:12](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/bindings/physics_bindings.md:12); [rcmt/physics_interface/local/conversion_laws/formal_structure.md:21](C:/Users/blaur/projects/theory/rcmt/physics_interface/local/conversion_laws/formal_structure.md:21); [rcmt/physics_interface/local/conversion_laws/formal_structure.md:181](C:/Users/blaur/projects/theory/rcmt/physics_interface/local/conversion_laws/formal_structure.md:181)

**Proposed replacement wording:** Use exact schema source “theory/rcmt/physics_interface/local/conversion_laws/formal_structure.md, §§1/6.” Add: “This is a boundary schema import; an instantiated payload must separately identify its producer and provenance.”

**Change rationale:** Upgrade navigability and distinguish definition from supplied numerical/physical data.

**Affected downstream documents:** SMG exports/policy; physics-interface local/conversion_laws interface; status registry.

**Risk and approval:** Low for path, medium for permission scope. Future implementation authorization; D3 confirmation if consumer class unresolved.

## C18 — SMG status, proposed routing and missing refactor record

Priority: **Medium**.

**File and section:** rcmt/smg_v2.1/README.md; translations/24-cell-routing.md and 24-cell-tranlsation.md

**Current wording or precise claim:** Requested REFACTOR_STATUS.md absent; routing says “Canon: Proposed” but “accepts ... only when” identities present; filename tranlsation is misspelled.

**Identified inconsistency:** Proposal scope/version/adoption is not bound to the main SMG contract; a missing status file is not proof of an incomplete refactor.

**Evidence and source references:** [rcmt/smg_v2.1/README.md:88](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/README.md:88); [rcmt/smg_v2.1/translations/24-cell-routing.md:3](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/translations/24-cell-routing.md:3); [rcmt/smg_v2.1/translations/24-cell-routing.md:92](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/translations/24-cell-routing.md:92); [rcmt/smg_v2.1/translations/24-cell-tranlsation.md:109](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/translations/24-cell-tranlsation.md:109)

**Proposed replacement wording:** Add registry wording: “Routing identities are a proposed translation extension; normative use requires an explicit adopting model/version. No completed-refactor claim is made by this registry.” Add links to current split files and decisions. Rename misspelled path only with references/compatibility redirect updated together.

**Change rationale:** Clarification of status; do not manufacture a historical refactor completion record or activate proposed requirements.

**Affected downstream documents:** SMG smg_contract/README/exports; downstream matter alignment; future regenerated references.

**Risk and approval:** Medium. Explicit owner decision D9 for adopting routing extension; editorial registry can follow approved facts.

## C19 — RCP sentinel and header migration

Priority: **Medium**.

**File and section:** rcf/rcp-1.2.md placement/validator sections; RCMT document headers

**Current wording or precise claim:** No leading delimiter; validator rejects absent “leading header sentinel”; RCMT headers omit canonical fields or use depends_on.

**Identified inconsistency:** Protocol wording ambiguous; field migration cannot assign authority from folder names or turn every historical file into an active contract.

**Evidence and source references:** [rcf/rcp-1.2.md:53](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:53); [rcf/rcp-1.2.md:236](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:236); [rcmt/local/invariants.md:3](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:3); [projects/ops-codex/doc-sweep/evidence/census-summary.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/doc-sweep/evidence/census-summary.json:1)

**Proposed replacement wording:** Proposed validator bullet: “the document does not begin with its single H1 followed by the canonical Document Contract.” Migrate active participating headers by classifying existing owner/type/scope/dependencies; keep “depends_on” material only after checking equivalence with upstream dependencies. Tag historical/frozen sources in an external manifest rather than rewriting them.

**Change rationale:** Protocol clarification and editorial migration; preserve status/authority distinction and anti-backflow.

**Affected downstream documents:** All participating RCMT/SMG headers; RCP examples/validators; generated compression; optional source manifest.

**Risk and approval:** Medium. Protocol-owner decision D10 and migration scope approval; routine fields then batch-authorized.

## C20 — Navigation, conceptual path aliases and stale tree

Priority: **Low**.

**File and section:** rcmt/README.md tree/build; local/README.md upstream paths; rcf/modules/geometry/projection/README.md file list

**Current wording or precise claim:** RCMT root invariants/symmetry paths absent; theory/rcf_projects references; projection list uses underscore filenames where actual files are hyphenated.

**Identified inconsistency:** Navigation obscures the actual authority sources. Compressed snapshot is not immutable authority over changed sources.

**Evidence and source references:** [rcmt/README.md:50](C:/Users/blaur/projects/theory/rcmt/README.md:50); [rcmt/README.md:68](C:/Users/blaur/projects/theory/rcmt/README.md:68); [rcmt/local/README.md:3](C:/Users/blaur/projects/theory/rcmt/local/README.md:3); [rcf/modules/geometry/projection/README.md:92](C:/Users/blaur/projects/theory/rcf/modules/geometry/projection/README.md:92)

**Proposed replacement wording:** Replace tree with actual local/, physics_interface/, global/, smg_v2.1/, modules/ locations. Use explicit repository-root path convention. Say: “Read upstream source contracts; a compressed shard is a subordinate snapshot and must identify its source revision.” Fix list labels to existing hyphenated names.

**Change rationale:** Straightforward editorial repair; source selection follows explicit existing contracts, not external old compressed trees.

**Affected downstream documents:** README navigation links; module indexes; compression dependency registry.

**Risk and approval:** Low. Future implementation authorization only; flag any alias whose authority target is not established.

## C21 — Mathematical typesetting without formula repair

Priority: **Low**.

**File and section:** rcf/modules/geometry/discrete/mismatch-function.md equations; RCMT math delimiters

**Current wording or precise claim:** Malformed E*{static}, R*{drag}, integral asterisks; some literal escaped display delimiters.

**Identified inconsistency:** Display corruption is separable from unresolved numerical semantics; neither should hide the other.

**Evidence and source references:** [rcf/modules/geometry/discrete/mismatch-function.md:28](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/mismatch-function.md:28); [rcf/modules/geometry/discrete/mismatch-function.md:42](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/mismatch-function.md:42); [rcmt/local/constraints/projection.md:53](C:/Users/blaur/projects/theory/rcmt/local/constraints/projection.md:53)

**Proposed replacement wording:** Render the existing symbols as E_{static}, R_{drag} and the intended sphere-integral notation, leaving coefficients and measure explicitly unspecified pending D6. Normalize display delimiters after preview checks; do not change variables, parameter domains or equations under a formatting label.

**Change rationale:** Editorial repair only; formula/provenance changes remain C07.

**Affected downstream documents:** Affected rendered equations; derived compressed math excerpts rebuilt last.

**Risk and approval:** Low. Future implementation authorization only; no formula correction smuggled into typesetting.

## C22 — OPS executable and result packaging

Priority: **Medium**.

**File and section:** projects/Operational_Phase_Space/SIM14.0.py–SIM16.4.py

**Current wording or precise claim:** Raw files contain code and unquoted archived stdout; five lack opening docstring delimiters.

**Identified inconsistency:** All raw files fail parsing; arbitrary cleanup could detach archived results from their source.

**Evidence and source references:** [projects/ops-codex/research/ops-syntax-audit.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/ops-syntax-audit.json:1); [projects/ops-codex/research/extraction-manifest.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/extraction-manifest.json:1); [projects/ops-codex/research/reproduction-results.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/reproduction-results.json:1)

**Proposed replacement wording:** Proposed archive note: “Original mixed code/result records remain immutable. Executable extracts identify source SHA-256, retained line ranges and delimiter repairs; recorded and newly reproduced stdout are separate artifacts with their own run metadata.” Later publish a manifest/index to those artifacts rather than overwriting originals.

**Change rationale:** Reproducibility upgrade without history loss; no additional run authorized by this proposal.

**Affected downstream documents:** OPS navigation/status; onboarding reproducibility notes; any future computational evidence citations.

**Risk and approval:** Medium. Owner approval D11 for canonical executable location and archival convention; original sources remain unchanged now.

## C23 — Scoped theorem and reproduced algebra upgrades

Priority: **Medium**.

**File and section:** OPS Vocabulary_Packet.md and SIM14_Discussions.md later-status annotations; SIM16 result index

**Current wording or precise claim:** Chronological discussion calls orbit bridging a candidate; algebraic results may be read as only observations.

**Identified inconsistency:** Existing proof and retained rerun logs justify stronger conditional status, without rewriting discovery history.

**Evidence and source references:** [projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md:5](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md:5); [projects/Operational_Phase_Space/SIM14_Discussions.md:1218](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14_Discussions.md:1218); [projects/ops-codex/research/reproduction-results.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/reproduction-results.json:1)

**Proposed replacement wording:** Add dated note: “The orbit-bridge criterion is proved under the stated free-V4/commuting-involution hypotheses. Its frozen finite-carrier application was reproduced; physical selection remains unproved.” Label SIM16 results “conditional exact algebraic classifications, rerun from documented extracts,” retaining each unresolved selection/observability statement.

**Change rationale:** Upgrade proof/reproducibility status at the exact claimed level, not canon or physical validity.

**Affected downstream documents:** OPS vocabulary/guide/status index; later reports citing M1/M2 or metrics; no upstream invariants changed.

**Risk and approval:** Low–medium. Future implementation authorization and review of exact hypotheses/citations; no theory decision.

## C24 — Lisi coordinate provenance quarantine

Priority: **Medium**.

**File and section:** projects/lisi-smg-test/reference_geometry/120cell_vertices.txt usage documentation; PHASE_1_2_REPORT.md

**Current wording or precise claim:** 600 coordinate rows are parseable, but 597 distinct keys and a zero-length listed edge; cause not identified.

**Identified inconsistency:** Coordinates cannot silently serve as an indexed realization for the validated adjacency graph.

**Evidence and source references:** [projects/lisi-smg-test/PHASE_1_2_REPORT.md:43](C:/Users/blaur/projects/theory/projects/lisi-smg-test/PHASE_1_2_REPORT.md:43); [projects/lisi-smg-test/PHASE_1_2_REPORT.md:74](C:/Users/blaur/projects/theory/projects/lisi-smg-test/PHASE_1_2_REPORT.md:74)

**Proposed replacement wording:** Add usage note: “The supplied coordinate artifact is retained as provenance input and is not validated against adjacency IDs. Graph/dual certificates do not rely on it. Coordinate-dependent use requires a separately sourced, validated realization and an explicit ID map.”

**Change rationale:** Clarify quarantine; preserve validated graph results and original coordinates.

**Affected downstream documents:** Lisi README/reference inventory; geometry loader usage; any coordinate plots or future embedding claims.

**Risk and approval:** Medium. Future implementation authorization for note; explicit source-owner decision D11 before coordinate repair/replacement.

## C25 — Lisi summary separates raw and quotient configurations

Priority: **Medium**.

**File and section:** projects/lisi-smg-test/summary.md Configuration Analysis

**Current wording or precise claim:** “Boolean local configuration structure”; structures “emerged ... rather than being designed into it.”

**Identified inconsistency:** Raw subset carrier was an explicit definition; orbit quotient is non-distributive; global quotient is not a lattice.

**Evidence and source references:** [projects/lisi-smg-test/summary.md:99](C:/Users/blaur/projects/theory/projects/lisi-smg-test/summary.md:99); [projects/lisi-smg-test/REPORT_PHASE6D.md:26](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6D.md:26); [projects/lisi-smg-test/REPORT_PHASE6D.md:149](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6D.md:149); [projects/lisi-smg-test/REPORT_PHASE6C.md:1](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6C.md:1)

**Proposed replacement wording:** Use: “By the explicit subset-configuration definition, four carriers give the Boolean raw lattice B4. The recovered V4 action yields seven orbit classes whose quotient lattice is non-distributive. The global 24-fiber orbit poset is a different object and is reportedly not a lattice.”

**Change rationale:** Clarification and downgrade of emergent-language overreach; preserve recovered symmetry result.

**Affected downstream documents:** Lisi summary and configuration navigation; downstream interpretations referencing seven classes.

**Risk and approval:** Low. Future implementation authorization only; retain global-enumeration verification caveat.

## C26 — Lisi historical plan and frozen comparison target

Priority: **Low**.

**File and section:** projects/lisi-smg-test/PLAN.md repository audit; REPORT_PHASE6.md scope

**Current wording or precise claim:** Plan says src/tests empty and later comparison not begun; Phase 6 compares references/smg_v2.1_concat.md.

**Identified inconsistency:** Plan is historical; frozen comparison does not automatically describe all current split SMG files or proposed routing extension.

**Evidence and source references:** [projects/lisi-smg-test/PLAN.md:5](C:/Users/blaur/projects/theory/projects/lisi-smg-test/PLAN.md:5); [projects/lisi-smg-test/REPORT_PHASE6.md:3](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6.md:3)

**Proposed replacement wording:** Prepend status note after a proper historical-record header: “The following inventory is the planning-time snapshot. Current phase status is indexed separately. Phase 6 conclusions apply to the hashed SMG reference consumed by that comparison; the reference is not silently replaced by current module prose.”

**Change rationale:** Editorial history/status clarification; keep original plan and frozen reference bytes.

**Affected downstream documents:** README_FIRST/current status index; summary; future comparison citations.

**Risk and approval:** Low. Future implementation authorization for external/current note; do not rewrite frozen reference artifacts.

## C27 — Generated compression and invariant-count calibration

Priority: **Medium**.

**File and section:** rcf/rcf_compressed.md build record; RCMT references to it; Codex onboarding invariant summary

**Current wording or precise claim:** Compressed shard dated July 25 with aggregate digests; current sources include later candidate constraints. Shard counts ten, onboarding says nine.

**Identified inconsistency:** Compression is subordinate and may be stale; nine substantive clauses plus numbered closure explains ten without an inconsistency.

**Evidence and source references:** [rcf/rcf_compressed.md:14](C:/Users/blaur/projects/theory/rcf/rcf_compressed.md:14); [rcf/rcf_compressed.md:128](C:/Users/blaur/projects/theory/rcf/rcf_compressed.md:128); [rcf/core/S∞.RCF.Invariants.v1.1.md:126](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:126); [projects/ops-codex/onboarding/02-invariants-and-constraints.md:1](C:/Users/blaur/projects/theory/projects/ops-codex/onboarding/02-invariants-and-constraints.md:1)

**Proposed replacement wording:** Use invariant summary: “Sections 1–9 state substantive invariants; section 10 states invariant closure; the compressed index represents all ten sections.” After approved source migration, regenerate shard with exact source commit/file manifest and a reproducible build method; before then label its freshness explicitly unknown against later edits.

**Change rationale:** Clarify own summary and upgrade derived-artifact provenance; do not change root invariant count/content to make a digest match.

**Affected downstream documents:** RCMT local README upstream references; generated indexes/shards; Codex onboarding summaries (separate derived-note edit).

**Risk and approval:** Medium. Future implementation authorization; D10 approval of regeneration inputs/method. No guessed digest algorithm.

## C28 — Global invariants: input records and admissible moves

Priority: **Medium**.

**File and section:** rcmt/global/constraints.md §§3/4/8; global/invariants/{twist,winding,holonomy}.md

**Current wording or precise claim:** Invariants evaluable from “adjacency alone,” but definitions use transported orientations/relational transformations and resist all local cleanup.

**Identified inconsistency:** Graph topology, transport labels and the permitted move class are not distinguished; preserved value depends on the declared implementation.

**Evidence and source references:** [rcmt/global/constraints.md:25](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:25); [rcmt/global/constraints.md:38](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:38); [rcmt/global/invariants/holonomy.md:1](C:/Users/blaur/projects/theory/rcmt/global/invariants/holonomy.md:1)

**Proposed replacement wording:** Use: “The implementation declares adjacency and any transported orientation/transformation records needed for its invariant. It specifies the admissible local moves under which the invariant is preserved. No metric or standalone ontological field is implied by those records.” Keep any stronger topology-only assertion conditional on an explicit encoding.

**Change rationale:** Clarify under-specified contract rather than prove global invariants nonexistent or change their intended protected role.

**Affected downstream documents:** global README/invariant examples; projection interpretation; cleanup-policy consumers.

**Risk and approval:** Medium–high. Owner decision D4/D7 on intended data and move class; cleanup must not invent them.

## C29 — Spinor closure certificates and terminology

Priority: **Medium**.

**File and section:** rcf/modules/geometry/discrete/spinor-closure.md and candidate spinor-embedding documents

**Current wording or precise claim:** 600-cell, dual demicube and E8 described as closed/stable spinor structures under the current update profile.

**Identified inconsistency:** Common vocabulary does not identify the same product, carrier, field, normalization or exhaustive closure certificate across candidates.

**Evidence and source references:** [rcf/modules/geometry/discrete/spinor-closure.md:24](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/spinor-closure.md:24); [rcf/modules/geometry/discrete/spinor-embedding.md:22](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/spinor-embedding.md:22)

**Proposed replacement wording:** Add: “Each closure claim is relative to the declared representation, composition and candidate set. A mathematical closure certificate and a simulation persistence record are separate evidence items; absent a linked certificate/run, the claim remains repository-reported for that profile.” Preserve any known exact polytope facts separately.

**Change rationale:** Clarify evidence scope; do not substitute a standard multiplication or assert all claims false.

**Affected downstream documents:** candidate sieve and survivor stats; geometry-adjacent closure terminology; downstream Dual Closure references.

**Risk and approval:** Medium. Evidence-owner review D6/D9; no new closure test or multiplication introduced in documentation cleanup.

## C30 — Preserve strong negative comparison result

Priority: **Medium**.

**File and section:** projects/lisi-smg-test/summary.md architectural consequences/current interpretation; downstream SMG comparison summaries

**Current wording or precise claim:** Broad recovery narrative may obscure Phase 6 no-equivalence result and non-invariant local 3+1 selection.

**Identified inconsistency:** Reproduced comparison tests justify a precise negative result while not ruling out a different future enriched model.

**Evidence and source references:** [projects/lisi-smg-test/REPORT_PHASE6.md:14](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6.md:14); [projects/lisi-smg-test/REPORT_PHASE6.md:96](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6.md:96); [projects/ops-codex/research/lisi-selected-tests.txt:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/lisi-selected-tests.txt:1)

**Proposed replacement wording:** Add: “For the retained transported action and supplied SMG reference, no full structure-preserving correspondence is established. A local 3+1 singleton is not invariant under the recovered transitive generation stabilizer. The reproducible heterogeneous 36-record projection requires extra choices and discards relations.”

**Change rationale:** Upgrade visibility of supported negative evidence; not universal impossibility or a new SMG requirement.

**Affected downstream documents:** Lisi summary; SMG comparison-status note; downstream carrier-identification claims; historical reference unchanged.

**Risk and approval:** Low–medium. Future implementation authorization; review scope/action qualifiers, no authority change.
