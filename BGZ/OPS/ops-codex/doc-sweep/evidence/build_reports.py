"""Build proposal-only reports; never writes research source documents."""
from pathlib import Path
import re,json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=HERE.parent
reports={
'00-executive-summary.md':('Documentation sweep: executive summary',r'''
## Result and scope

This is a controlled cleanup proposal. No research source was edited, no theory was developed, and no simulation was rerun during this pass. The earlier onboarding's nine OPS reruns and 42 selected Lisi test passes are reused as dated, scoped evidence, not promoted to physical validation. Source commits remain RCF `c419353214ee83d83376747064912544aee257b9` and RCMT `b7eecd0dfd8080864b34441af86329f819cba65a`.

The repository-wide census covers 383 Markdown files across the four authorized inputs, including two generated pytest-cache READMEs. Excluding those cache documents leaves 381. Contract/heading inventory and broad claim searches cover that corpus; detailed semantic inspection targets authority-bearing documents, their dependencies, historical revisions and the reported problem areas. This is not a proof audit of every assertion in every file. @{projects/ops-codex/doc-sweep/evidence/document-census.json#1}, @{projects/ops-codex/doc-sweep/evidence/git-provenance.json#1}

## Findings by priority

| Priority | Finding | Proposed treatment |
|---|---|---|
| Critical decision gate | The two-lane requirement has a recoverable historical rationale, but no recovered proof of projection-driven necessity | Preserve its current RCMT requirement status; add provenance; obtain a human decision before any promotion/demotion or change to independence |
| High | The pipeline overview retains non-local projection, metric/time introduction, old symmetry arrows and guaranteed post-LCO coherence | Replace or archive affected passages against explicit active interfaces, keeping historical text accessible |
| High | Physics consumer permissions and compactness language differ across contract families | Assign consumer/stage scopes and resolve restrictions explicitly; do not silently union permissions |
| High | Density-table provenance and affine formula disagree; gamma is overloaded; descriptor layer imposes all-plateau survival | Preserve numbers as reported history pending provenance; clarify parameter identities and regime scope |
| High | Dual-projection filtering and fixed-point equation exceed or fail to type against the handoff interface | Owner must decide a consistency-check profile versus a genuinely distinct feedback architecture |
| High | Local update-policy examples do not ensure their stated descent/determinism properties | State conditional laws, acceptance filters and baseline comparisons explicitly |
| Medium | Missing/misdirected bindings, proposed routing status, mixed RCP headers, stale navigation and generated summaries | Repair references and metadata after authority decisions; regenerate derived artifacts last |
| Medium | OPS code/results packaging and Lisi coordinate defects limit reproducibility | Keep originals immutable; later separate executable artifacts with provenance, and quarantine coordinates |
| Positive upgrades | Orbit-bridge result has a reviewed proof; selected exact algebra and Lisi transport/negative results were reproduced | Use stronger scoped mathematical/computational wording where current prose remains tentative |

## Two-lane conclusion

The earliest RCF revision available locally (2026-07-23) asserted memory/containment and motion/direction as distinct functions, while allowing shared representation and accepting symplectic structure **or an equivalent informational invariant**. The 2026-07-25 revision explicitly made this profile optional. RCMT's first source commit already requires symplectic and Lorentz-compatible structures; the local record does not recover the reasoning that makes this stronger specialization necessary. Optional in RCF and required in RCMT is a legitimate architectural specialization, not itself a contradiction. The unsupported step would be calling the RCMT specialization a theorem of RCF or projection. @{projects/ops-codex/doc-sweep/evidence/rcf-earliest-geometry.txt#1}, @{projects/ops-codex/doc-sweep/evidence/rcf-two-lane-optionalization.diff#1}, @{projects/ops-codex/doc-sweep/evidence/rcmt-earliest-invariants.txt#1}

## Recommended order

1. Preserve the current snapshots and approve a decision record for authority-sensitive questions.
2. Repair editorial references/status fields without changing requirements.
3. Reconcile owner contracts and stale summaries in dependency order, in reviewable batches.
4. Add evidence records and scoped mathematical upgrades; preserve failed hypotheses and negative controls.
5. Propagate approved interfaces downstream, then rebuild compression/index artifacts and validate links and examples.

The concrete replacement text, affected files, risks and approval gates are in `05-proposed-change-ledger.md`. Execution is a separate future task. This proposal stops here.
'''),
'01-authority-and-evidence-audit.md':('Authority and evidence classification',r'''
## Classification rules

Authority answers who can impose a requirement; evidence answers what supports an assertion. A locally authoritative invariant can still be an adopted design constraint without a necessity proof. Conversely, a proved finite theorem need not become a framework invariant. RCP-1.2 explicitly separates authority, dependencies and status, and limits body prose by the header and upstream contracts. @{rcf/rcp-1.2.md#110}, @{rcf/rcp-1.2.md#167}, @{rcf/rcp-1.2.md#180}

| Class | Significant claim and sources | Assessment / proposed wording direction |
|---|---|---|
| Upstream RCF invariant/closure | Kernel sections 1–9 and section 10 closure | Preserve; describe as project invariant commitments, not discovered physical laws. There are nine substantive invariant headings plus the numbered closure condition, summarized as ten items in the compressed shard. @{rcf/core/S∞.RCF.Invariants.v1.1.md#35}, @{rcf/core/S∞.RCF.Invariants.v1.1.md#126}, @{rcf/rcf_compressed.md#128} |
| Upstream optional contract | Memory/motion geometry profile; group structures | Preserve opt-in status; clarify internal minimality assertion is not a proved universal minimum. @{rcf/core/relational-constraints/geometry.md#3}, @{rcf/modules/group_structures/interface.md#22} |
| Upstream execution contract | LCO local deterministic scoring; policy selects; GCO aggregates | Strong contractual wording warranted for ownership. No inferred uniqueness of scoring formula or guaranteed zero mismatch. @{rcf/modules/lco/lco-operator.md#181}, @{rcf/modules/gco/interface.md#14} |
| RCMT-local invariant | Two-lane requirement; projected observables; mass/randomness representations | Binding on RCMT today, subordinate to RCF; not proof that all physical models require it. Do not demote without approval. @{rcmt/local/invariants.md#25}, @{rcmt/local/invariants.md#53} |
| Provisional implementation | Sp(8)×SO(9), Sp(8)×SO(8) | Already correctly classified as candidates in the symmetry contract. Propagate that status to READMEs/pipeline; avoid indiscriminate downgrades of the current contract. @{rcmt/local/constraints/symmetry_chain.md#49} |
| Historical/rejected construction | E8→SO(10)→Sp(8)×SO(9); L192=cell stabilizer | Keep history and failures. Current symmetry contract explicitly demotes the old chain; SIM15.4 rejects the stabilizer identity. @{rcmt/local/constraints/symmetry_chain.md#94}, @{projects/Operational_Phase_Space/SIM15.4.py#1862} |
| Mathematical theorem under assumptions | V4 orbit-bridge regularity | Upgrade surrounding tentative discussion to refer to the existing theorem and enumerate assumptions. No physical M2-selection theorem follows. @{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#5} |
| Reproduced computation | SIM14.3/14.5/14.8/15.6/16.0–16.4; selected Lisi tests | Say “reproduced from extracted code” or “selected tests passed” with logs and source hashes. Reusing code is not independent implementation replication. @{projects/ops-codex/research/reproduction-results.json#1}, @{projects/ops-codex/research/lisi-geometry-tests.txt#1}, @{projects/ops-codex/research/lisi-selected-tests.txt#1} |
| Repository-reported computation | Full SIM15.0–15.5 census; Phase 6C; density sweep | Retain as attributed, bounded results. Full H4 census and global configurations were not independently rerun during onboarding; density sweep has unresolved provenance and formula mismatch. @{projects/Operational_Phase_Space/SIM15.5.py#1}, @{projects/lisi-smg-test/REPORT_PHASE6C.md#1}, @{rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md#35} |
| Downstream interpretation/hypothesis | SMG Dual Closure, gauge labels, mass/zeta mapping; matter candidate labels | Preserve model-local thesis language and explicit disclaimers. Count equality is not a Lie algebra construction or physical derivation. @{rcmt/smg_v2.1/smg_contract.md#42}, @{rcmt/smg_v2.1/translations/zeta_resonance.md#26}, @{rcmt/modules/survivor_matter_structure/README.md#1} |
| Historical planning state | Lisi PLAN inventory | Label its planning-time state and point to completed phases; do not erase original assumptions. @{projects/lisi-smg-test/PLAN.md#5} |

## Supported upgrades

- **U1 — orbit bridge:** the conditional iff is proved in the repository and its finite application reproduced. Wording such as “candidate discriminator” may remain in preserved chronology, with a later-status note linking the theorem. Assumptions include a free V4 action with two four-state orbits and a commuting involution outside V4. @{projects/Operational_Phase_Space/SIM14.3.py#29}, @{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#31}
- **U2 — SIM16 algebra:** classify exact symbolic metric/projection/generator results as conditional algebraic classifications rather than tentative numerical hints. Retain parameter-selection and observability limits. @{projects/Operational_Phase_Space/SIM16.1.py#251}, @{projects/Operational_Phase_Space/SIM16.2.py#239}, @{projects/Operational_Phase_Space/SIM16.4.py#391}
- **U3 — transport and obstruction:** the selected tests support enriched fiber-level transport and the symmetry obstruction to an invariant singleton among four pair carriers. State both firmly within the specified action, not as universal impossibility of every possible SMG realization. @{projects/lisi-smg-test/REPORT_LISI_120CELL_TRANSPORT.md#68}, @{projects/lisi-smg-test/REPORT_PHASE6.md#74}

## Downgrades and clarifications

“Explains”/“provides mechanisms” in the RCMT README overstates the local representational contracts. “Empirical invariant” in 600-cell survival statistics overstates unlinked computational observations. “Emergent/not imposed” language in Chromatic-12/36-mode descriptions needs a provenance record because the same descriptor document excludes candidates lacking that motif. These are evidence-language corrections, not permission to erase the model commitments. @{rcmt/README.md#24}, @{rcf/modules/geometry/discrete/600-cell/survival-stats.md#44}, @{rcf/modules/geometry/adjacent/mode-structure.md#34}

## Mechanical contract census

196 of the 199 RCF Markdown files contain all eight canonical field labels outside fenced examples; the three exceptions are two pytest-cache READMEs and the historical RCP-1.1 protocol. All 163 RCMT documents lack at least one canonical RCP-1.2 field under this screen. OPS's six and Lisi's fifteen Markdown documents also lack complete canonical field presence. This checks presence, **not** order, header placement, dependency resolution, ownership correctness or whether migration is required for every file. No blanket scientific-validity judgment follows. Historical records and frozen references should not be silently rewritten to pass a new header screen. @{projects/ops-codex/doc-sweep/evidence/census-summary.json#1}

## Calibration of the onboarding notes

The onboarding's “nine framework invariants” wording should be qualified as nine substantive invariants **plus section 10's invariant-closure condition**. The compressed document's ten-item count is therefore not, by itself, a contradiction. Likewise, the gamma investigation now locates an existing density variable κ and a sealed range-selection adapter; the unresolved part is the sweep's use of gamma, not absence of any parameter architecture. Neither earlier report is edited in this planning pass. @{projects/ops-codex/onboarding/02-invariants-and-constraints.md#1}, @{rcf/modules/temporal-interface/local-constraint-density.md#65}
'''),
'02-two-lane-origin-investigation.md':('Origin and status of the two-lane requirement',r'''
## Finding

No original mathematical derivation of the mandatory symplectic/geometric split was recovered. The available record does recover an earlier functional rationale and a later, explicit change of upstream authority. Current RCMT requirements remain in force for this proposal; their status must not be changed merely because the rationale is incomplete.

## Historical chain available locally

| Record | Repository evidence | What it establishes |
|---|---|---|
| RCF initial source, 2026-07-23 08:42:22 Phoenix, `b7a34065d3cafeab91d02f2bfe2ff513445365d9` | Snapshot @{projects/ops-codex/doc-sweep/evidence/rcf-earliest-geometry.txt#1} | Asserts mandatory memory/containment and motion/direction functions, permits shared representation, allows a symplectic form **or conserved measure/equivalent invariant**; gives no proof of functional minimality |
| RCF Time at the same initial revision | @{projects/ops-codex/doc-sweep/evidence/rcf-earliest-time-core.txt#1} | Adds a historical preservation rationale: unitarity said to preserve “symplectic memory,” Higgs direction and mass distinctions; generator compatibility uses an undeclared stabilizer representation. These assertions do not derive distinct lanes or a product group |
| RCMT initial source, 2026-07-23 18:15:50 Phoenix, `e9c51a3d123a42e639f137e38d0286b772d0694b` | @{projects/ops-codex/doc-sweep/evidence/rcmt-earliest-invariants.txt#1}, @{projects/ops-codex/doc-sweep/evidence/rcmt-earliest-symmetry.txt#1} | Already stipulates unbroken symplectic mode lane and Lorentz-compatible geometric lane, with old chain provisional; no preceding deduction from the functional profile supplied |
| RCF optionalization, 2026-07-25 18:38:21 Phoenix, `8c7bd510a22c01097e4959344d9499e9548b07ef` | Commit message expressly says “Downgraded 2-lane requirement to optional”; diff @{projects/ops-codex/doc-sweep/evidence/rcf-two-lane-optionalization.diff#1} | Explicit upstream narrowing, not an inference from a recent date |
| RCF Time-profile revision, 2026-07-25 19:54:22 Phoenix, `d377a73617a7054560122bcbf92d104777bacf8e` | @{projects/ops-codex/doc-sweep/evidence/rcf-time-profile-revision.diff#1} | Makes continuous/unitary refinement optional and removes symplectic/Higgs/mass preservation from Time-owned claims; adopting modules must declare the preserved structure |
| RCF header cleanup, 2026-07-25, `8f0c98c81f63a0cf99d60218e78192af09512244` | @{projects/ops-codex/doc-sweep/evidence/rcf-geometry-history.txt#1} | Later header adjustment; current optional profile remains subordinate |
| RCMT candidate revision, 2026-10-24 12:34:44 Phoenix, `7a172823c9d80170da310d1eda30203f09b26387` | @{projects/ops-codex/doc-sweep/evidence/rcmt-candidate-revision.diff#1} | Replaces old active chain with SO(9)/SO(8) candidates; marks Lorentz realization unresolved; does not remove two-lane requirement |
| RCMT compactness revision, 2026-10-24 12:43:11 Phoenix, `b7eecd0dfd8080864b34441af86329f819cba65a` | @{projects/ops-codex/doc-sweep/evidence/rcmt-compactness-revision.diff#1} | Separates geometric compactness from symplectic admissibility; does not prove any particular noncompact realization acceptable |

These dates describe local Git records, not when the ideas were first conceived. All locally reachable history was enumerated; targeted file histories and initial snapshots were examined. RCMT has a second root `038a59ef7d8428d37cbd686a02a66e87d9ebc45d` containing only LICENSE, dated later that July 23; it supplies no earlier rationale. No remote history was fetched, and excluded repositories/other-agent output were not searched. The authoring discussions or original derivation may predate these imported initial snapshots. @{projects/ops-codex/doc-sweep/evidence/rcmt-history.txt#1}

## Necessity, choice and implementation

| Question | Answer supported by the repository |
|---|---|
| Does RCF projection require these two lanes? | No such requirement appears in the active projection interface. It consumes survivors/adjacency/orientation/profile and emits shadow/loss records, without requiring symplectic geometry. The optional group interface expressly forbids requiring two lanes. This establishes absence of that upstream contract requirement, not a general impossibility theorem. @{rcf/modules/geometry/projection/interface.md#14}, @{rcf/modules/group_structures/interface.md#22} |
| Does a projection have source and target representations? | The documented handoff does. Different source/target roles alone do not establish independent state spaces, distinct symmetry groups, a symplectic form, a direct product, or mutual irreducibility. No argument for those additional conclusions is supplied by these interfaces. |
| Why memory and motion? | Historical motivation is history-sensitive refinement, conserved informational content, constrained direction, drag analogues and boundary behavior. These are profile desiderata, not proved uniqueness/minimality. @{rcf/core/relational-constraints/geometry.md#28} |
| What is required in current RCMT mode structure? | Finite-dimensional adjacency compatibility, a symplectic structure, preservation under declared admissible transformations, unresolved mode content, measure suitable for induced probabilities, and compatibility with local LCO records/projection. “Non-breaking” needs an operational definition and real form. @{rcmt/local/constraints/mode_space.md#26}, @{rcmt/local/constraints/mode_space.md#39} |
| What belongs in geometric structure? | Local Lorentz-compatible descriptions, compact internal symmetry at the intended interface, derived observables and adjacency-local information flow, no global geometric curvature group. These are RCMT requirements, not completed constructions. @{rcmt/local/constraints/symmetry_chain.md#25}, @{rcmt/local/constraints/projection.md#138} |
| Must groups be independent? | The RCMT mode-space contract says neither representation is reducible to the other and prohibits one shared representational domain. The RCF optional profile allows shared representation with distinct functions. This is a stronger downstream choice, not a contradiction unless advertised as upstream necessity. Neither text defines “independent groups” through commuting faithful actions or a universal factorization theorem. @{rcmt/local/constraints/mode_space.md#72}, @{rcf/core/relational-constraints/geometry.md#30} |
| Must the group be a direct product? | The product expressions are explicitly provisional candidates. The current symmetry contract disclaims mandatory embeddings/decompositions. Naming two functional roles does not supply the commuting actions and intersection data needed to establish a product realization. No universal product requirement was recovered. @{rcmt/local/constraints/symmetry_chain.md#49} |
| Are Sp(8,R), SO(8), SO(9) invariants? | No. RCMT uses bare Sp(8) provisionally, permits other Sp(2n), and names SO(8)/SO(9) as candidates. OPS specifically uses real 8×8 symplectic matrices. The OPS real form cannot silently settle RCMT's notation. @{rcmt/local/constraints/mode_space.md#58}, @{projects/Operational_Phase_Space/SIM14.0.py#1} |
| Where does Lorentz enter? | As RCMT's projection-level target requirement. The current contract explicitly rejects presuming conventional compact-group breaking to SO(3,1), and says a mathematical realization remains to be specified. @{rcmt/local/constraints/symmetry_chain.md#84} |

## Preservation vocabulary that needs a decision

For a realization claiming a symplectic action, use a declared vector space over a declared field, a nondegenerate **alternating** bilinear form Ω, and the condition SᵀΩS=Ω for transformations claimed symplectic. The current phrase “non-degenerate bilinear form” alone omits alternation. This clarifies the named structure; it does not show that each stochastic update kernel is a symplectic transformation. SIM16's stochastic obstructions warn against conflating those requirements. @{rcmt/local/constraints/mode_space.md#51}, @{projects/Operational_Phase_Space/SIM16.2.py#224}

“Representation independence” also needs a precise interpretation. Non-invertibility of Π records information loss in one direction; it is not a definition or proof of mutual irreducibility. `Π:Sp(2n)→G` currently names a group as source while discussing projection of mode configurations. The owner must declare whether it acts on states, representations, group elements or an observable algebra before making preservation and loss claims executable. @{rcmt/local/constraints/projection.md#47}

## Alternative formulations for human choice

**A — retain current RCMT requirement; annotate its status (least change).** “RCMT currently adopts a distinct symplectic mode-space structure and Lorentz-compatible geometric representation as model-local architectural requirements. Their necessity is not derived here from the RCF kernel or projection alone. The listed product groups are provisional implementations.” This preserves the present commitment and exposes the proof gap.

**B — require distinct functional roles, permit shared representation.** “RCMT requires separately specified mode-preservation and geometric-observation roles. A realization may share a carrier if it states the preserved structures, projection and information loss explicitly. Independent symmetry factors are not presumed.” This **changes** the present prohibition on a single representational domain and requires explicit architectural approval and propagation.

**C — make the current symplectic/geometric split an opt-in RCMT implementation profile.** “The dual-lane profile adopts an unbroken symplectic structure and Lorentz-compatible geometric representation; other RCMT profiles must state their own preservation and observation contracts.” This **changes** RCMT invariant 2 and related invariants/projection rules. Approval must identify the remaining RCMT-wide requirements and naming/version impact.

No option is selected or applied. A claim that two lanes are a theorem would additionally require a stated theorem, hypotheses, and proof; absent that artifact, do not upgrade A to mathematical necessity. No option authorizes a historical symmetry cascade to return as active architecture.

## Historical preservation rationale is not a surviving Time derivation

The initial Time text is a second relevant origin source, not an active license. It asserts that unitarity preserves symplectic memory and uses `[H,dρ(X)]=0` for `X` in a stabilizer of `h`, but does not supply the representation, form, carrier or a proof connecting these symbols to a necessary two-lane architecture. The current Time file expressly says those physical/symplectic readings belong downstream and requires an adopter to declare the optional generator compatibility data. This history explains why memory-preservation language appeared in the program; it cannot be cited as a current upstream derivation. @{projects/ops-codex/doc-sweep/evidence/rcf-earliest-time-core.txt#1}, @{rcf/modules/time/time-core.md#112}, @{rcf/modules/time/time-core.md#158}, @{rcf/modules/time/time-core.md#169}
'''),
'03-contract-and-interface-conflicts.md':('Contract and interface conflicts',r'''
## Conflict register

“Contradiction” below means the same unqualified claim cannot satisfy the cited contracts simultaneously. “Scope ambiguity” means stage/consumer identity could reconcile the text but is not declared. A downstream profile may be narrower than its parent; it may not silently claim parent authority.

| ID / priority | Sources and exact issue | Classification and proposed boundary |
|---|---|---|
| C01–C02 / Critical decision | Optional RCF memory/motion profile permits shared representation; RCMT requires irreducible symplectic/geometric domains | Legitimate downstream specialization; origin/necessity unsupported, independence undefined. Preserve current RCMT status until decision. @{rcf/core/relational-constraints/geometry.md#30}, @{rcmt/local/constraints/mode_space.md#72} |
| C03 / High | Pipeline projection is “first non-local” operation; introduces metric, causal cones and time; retains SO(9) arrows | Conflicts with RCMT locality if non-local information transfer is meant; conflicts with staged metric/time ownership and current candidate status. Non-local read-only analysis could instead be a separately scoped downstream diagnostic. @{rcmt/physics_interface/pipeline/overview.md#100}, @{rcmt/local/invariants.md#126}, @{rcmt/physics_interface/README.md#66} |
| C04 / High | Physics README permits declared local records; pipeline PIC prohibits adjacency/modes/coherence inputs; adapter permits local primitive imports | Could be separate internal construction modules and effective-output consumers, but both use unqualified “physics modules.” Do not assume supersession from filenames or union permissions. @{rcmt/physics_interface/README.md#30}, @{rcmt/physics_interface/pipeline/physics_interface_contract.md#1}, @{rcmt/physics_interface/local/adapters/local_to_physics_adapter.md#14} |
| C05 / High | Local invariant 10 exempts symplectic lane from automatic compactness; global §6 bans all noncompact global groups; projection §6 also says noncompact global symmetries forbidden | A compact global-invariant **representation** may be a narrower profile; a blanket ban on the mode-lane group conflicts with the exemption. Resolve applicability without forcing Sp(8,R). @{rcmt/local/invariants.md#93}, @{rcmt/global/constraints.md#62}, @{rcmt/local/constraints/projection.md#138} |
| C06 / High | κ is bounded local density; γ is sealed memory weight; table/RCMT invariant 11 call γ density and use values to 5 | Existing architecture distinguishes variables; their identification in other files is unresolved. Select values/ranges under explicit profiles, do not back-derive gamma from mismatch. @{rcf/modules/temporal-interface/local-constraint-density.md#65}, @{rcf/modules/temporal-interface/temporal-weights-adapter.md#53}, @{rcf/modules/lco/lco-primitives.md#53}, @{rcmt/local/invariants.md#105} |
| C08 / High | Optional geometry descriptor requires F≤1 at all gamma levels; density-specific sieve allows restricted regimes and table excludes 600-cell at high gamma | Genuine unqualified survival-scope conflict. Descriptor cannot redefine upstream admission. If a robust-across-all-plateaus subclass is intended, name it separately. @{rcf/modules/geometry/adjacent/gamma-plateau-behavior.md#101}, @{rcf/modules/geometry/discrete/survival-sieve.md#38} |
| C09 / High | Projection interface cannot select survivors; Π↓ prose rejects discrete states and equation equates G to output of a consistency-condition map | Owner/typing conflict. Separate downstream consistency diagnostics from the upstream sieve, or approve a different feedback profile. @{rcf/modules/geometry/projection/interface.md#22}, @{rcf/modules/geometry/projection/dual_projection.md#40} |
| C10 / High | Pipeline declares zero mismatch/basin stability “after LCO”; global constraints let “LCO or GCO” prune neighborhoods | Score/aggregate contracts do not guarantee selection, perfect coherence or mutation. Change actor nouns and outcome qualifications, preserve failed records. @{rcmt/physics_interface/pipeline/overview.md#80}, @{rcmt/global/constraints.md#38}, @{rcf/modules/gco/interface.md#14} |
| C14 / High | Sectorization says LCO requires triangle packing in projected 120-cell | LCO's own interface is geometry-neutral; a triangle **profile** may exist downstream. Specify whether triangles are graph 3-cycles, cells, relation motifs or a separate incidence object; target 120-cell adjacency has no triangles in the studied transport. @{rcf/modules/gcm_interface/sectorization.md#42}, @{rcf/modules/lco/lco-invariants.md#59}, @{projects/lisi-smg-test/REPORT_PHASE5.md#15} |
| C18 / Medium | Proposed SMG routing document uses “requires/accepts only” without adoption/version binding; explicit substrate implementation belongs upstream | A proposed extension is not automatically an active binding for all adopters; keep metadata and core contract synchronized after decision. @{rcmt/smg_v2.1/translations/24-cell-routing.md#3}, @{rcmt/smg_v2.1/smg_contract.md#95} |
| C19 / Medium | RCP placement forbids leading delimiters; validator asks for a leading header sentinel | Protocol-internal ambiguity. Proposed repair interprets the required leading structure as the H1 plus canonical contract, avoiding a new delimiter. Protocol owner must approve. @{rcf/rcp-1.2.md#53}, @{rcf/rcp-1.2.md#236} |
| C28 / Medium | Global invariants allegedly read from “adjacency alone,” but definitions also require transported orientation/transformations | Under-specified input type, not proof that adjacency can never encode them. Name graph **plus declared transport data**, and declare the allowed invariant-preserving local moves. @{rcmt/global/constraints.md#25}, @{rcmt/global/invariants/holonomy.md#1} |

## Apparent conflicts that should not trigger broad edits

- Optional two lanes in RCF and mandatory two lanes in RCMT can coexist. The disputed inference is necessity or upstream inheritance, not the right of RCMT to adopt a stricter architecture.
- `N+ ≅ S4`, unsigned affine H192 and native H4 cell stabilizer N576 concern different groups/actions. No count-based harmonization is appropriate. SIM15.4's failed L192 equality remains a failure even though SIM15.5 tests N576 successfully. @{projects/Operational_Phase_Space/SIM14.6.py#1}, @{projects/Operational_Phase_Space/SIM15.5.py#1}
- RCF's smooth discrete→continuous shadow and RCMT's mode→observable projection are explicitly distinct scopes. The Lisi incidence dual and OPS's finite observation map are further concrete constructions. Distinct domains/codomains can reconcile differing adjacency and invertibility language. @{rcmt/local/constraints/projection.md#3}, @{rcf/modules/geometry/projection/interface.md#3}
- Stable routing IDs do not assert perpetual physical identity; lifetime/state-transition qualifiers can coexist with RCMT contingent identity. Keep these distinct rather than removing identity labels. @{rcmt/smg_v2.1/translations/24-cell-routing.md#58}, @{rcmt/local/invariants.md#138}
- Local B4 raw configurations, a seven-class non-distributive orbit lattice, and the global non-lattice orbit poset are different objects. Their different lattice properties are consistent. @{projects/lisi-smg-test/REPORT_PHASE6D.md#136}, @{projects/lisi-smg-test/REPORT_PHASE6C.md#1}

## Authority-safe resolution rule

Use explicit ownership to prevent backflow while reporting incompatible claims side by side. Do not silently declare a winner between two unranked RCMT contract families. A proposed split into construction-facing interfaces and effective-consumer interfaces is an editorial/schema migration only after the owner identifies which modules occupy each side; otherwise it changes import permissions.
'''),
'04-mathematical-and-numerical-claims.md':('Mathematical and numerical claims',r'''
## Quantitative findings

**High — fixed affine formula versus table (C07).** The formula defines F(G,γ)=Estatic(G)+γ Rdrag(G), with both coefficients geometry-only. The table gives F600 at γ=0,1,2 as 0.472, 0.550, 0.750: increments are 0.078 and 0.200; the affine identity F(2)−2F(1)+F(0) would be zero but equals 0.122. This exceeds rounding at the displayed precision. That is an internal discrepancy, not proof the simulations never occurred. Noise, changing coefficients or different profiles would need an explicit record. Preserve the table, label it repository-reported and unlinked to a reproducible run, and ask the owner which model it describes. Do not fit new coefficients to rescue it. @{rcf/modules/geometry/discrete/mismatch-function.md#24}, @{rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md#38}

The same document calls an angular integral an expectation without declaring normalized spherical measure and gives Rdrag only up to proportionality. Coefficients, vertex normalization, dimension, measure and update profile are needed to reproduce numerical thresholds. Its fewer-tiers→higher-drag rationale also differs from the E8 narrative associating a single tier with flat drag. These may refer to distinct effects; record their domains rather than assert one universal law. @{rcf/modules/geometry/discrete/mismatch-function.md#37}, @{rcf/modules/geometry/discrete/mismatch-function.md#49}, @{rcf/modules/geometry/discrete/e8/survival-stats.md#41}

**High — stochastic policy examples (C11).** A Boltzmann weight exp(−βCj) is positive for each finite candidate score at finite β. Without an admissibility filter, a candidate with Cj>Ccurrent can therefore be selected; it does not enforce accepted-update non-increase. Picking the minimum candidate score also needs comparison with Ccurrent or inclusion of the current state to ensure descent. The existing “identical inputs must yield identical decisions” conflicts with a stochastic policy unless randomness is part of the input/trace; equal conditional distributions are the appropriate unseeded claim. These are elementary checks of the printed examples, not new model proposals. @{rcf/modules/lco/lco-update-policy.md#77}, @{rcf/modules/lco/lco-update-policy.md#115}, @{rcf/modules/lco/lco-update-policy.md#140}

**High — typed projection and symplectic claims (C02/C09).** Specify the state carrier and alternating form; a group name alone does not define a projection of configurations. Π↓ is described as returning conditions but used as if it returns a graph equal to G; choose a predicate, equivalence relation or actual reconstruction map before using the fixed-point equation. No inverse or round-trip identity follows from noninvertible RCMT projection. These are typing/contract questions, not demands for a conventional subgroup-breaking chain. @{rcmt/local/constraints/projection.md#47}, @{rcf/modules/geometry/projection/dual_projection.md#40}

**High — Chromatic-12 and 36 modes (C13).** `mode-structure.md` says the count is not imposed, while excluding geometries that lack a stable 12-mode base. Conditional filtering and observation can coexist, but the claimed derivation needs the actual candidate pool, motif predicate and independent selection stages. 12×3=36 is arithmetic once the indexing choices are supplied; it is not by itself evidence that physical modes or three generations are derived. Preserve GCM's conditional descriptors and SMG's model-local semantic mapping. @{rcf/modules/geometry/adjacent/mode-structure.md#34}, @{rcf/modules/geometry/adjacent/mode-structure.md#52}, @{rcf/modules/gcm_interface/survivor_indexing.md#28}, @{rcmt/smg_v2.1/smg_contract.md#42}

## Strong claims worth retaining or upgrading

| Result | What can be stated firmly | Limit |
|---|---|---|
| Orbit bridge | Under free V4 action with two four-state orbits and commuting involution outside V4, regularity iff orbit exchange | Existing proof; does not choose a physical M2 branch or require C8. @{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#5} |
| SIM16 compatible metrics | Exact classifications under the declared symmetry/sparsity conditions; signed local family remains one-dimensional | Sparsity is stipulated; no selected parameter or physical observability established. @{projects/Operational_Phase_Space/SIM16.1.py#251}, @{projects/Operational_Phase_Space/SIM16.2.py#239} |
| SIM16 generator audit | Local symplectic algebra, fixed-J stabilizers and family-transport direction computed and rerun | Allowable algebraic directions are not a chosen dynamics or update scheduler. @{projects/Operational_Phase_Space/SIM16.4.py#253} |
| Enriched Lisi transport | Named projective relations preserved at explicit fiber/quotient level after carrier change | Bare adjacency loses metadata; original paper's full spinor phases not verified by carrier tests. @{projects/lisi-smg-test/REPORT_LISI_120CELL_TRANSPORT.md#68} |
| No intrinsic local singleton | Retained generation stabilizer acts transitively on four pairs; invariant one-of-four marker is unavailable under that action | Extra marked data or a changed action gives a different object; not universal impossibility of all future realizations. @{projects/lisi-smg-test/REPORT_PHASE6.md#96} |

The prior execution evidence is captured in @{projects/ops-codex/research/reproduction-results.json#1}. The full H4 family census and global configuration enumeration remain attributed reports; no new runs were made in this documentation sweep.

## Mathematical vocabulary requiring precision

- Bare Sp(8) is not interchangeable with an OPS-specific real matrix realization without declaring field, carrier dimension and form. “Finite-dimensional” does not mean finite cardinality or compact. Retain SO(8)/SO(9) as candidates, not invariant group identities. @{rcmt/local/invariants.md#93}, @{rcmt/local/constraints/symmetry_chain.md#49}
- Three-way triality action does not alone choose an equivariant correspondence. SIM15.6's particular facet-stabilizer relation supplies extra information; the erased-incidence control leaves alternatives. Its coordinates use floating tolerances even though resulting group calculations use exact permutations. @{projects/Operational_Phase_Space/SIM15.6.py#1}
- A Lisi graph-level inscribed 24-cell need not be induced by ambient 600-cell edges. Its predicate uses the ambient exact-distance-two relation. Native H4 stabilizer statements use their own ambient action. @{projects/lisi-smg-test/PHASE3A_VALIDITY_PREDICATE.md#7}, @{projects/Operational_Phase_Space/SIM15.4.py#1}
- If triangle compatibility of a target 120-cell is intended, name an incidence or motif relation; do not implicitly call graph adjacency a triangular face relation. The tested target graph has no triangles. @{projects/lisi-smg-test/REPORT_PHASE5.md#15}, @{rcf/modules/gcm_interface/sectorization.md#42}
- Closure claims for the 600-cell, demicube and E8 require the specific product, representation, normalization and certificate; shared “spinor” vocabulary does not supply a common multiplication. No counterexample to every possible profile was established here. @{rcf/modules/geometry/discrete/spinor-closure.md#24}, @{rcf/modules/geometry/discrete/spinor-embedding.md#22}

## Physical interpretation language

The SMG gauge, spinor and zeta translations already label themselves downstream approximations/theses. Preserve those safeguards. Add explicit pointers to proof or numerical artifacts only if they actually exist; do not invent mass predictions or a Riemann Hypothesis proof. This audit does not prove those interpretations false. It identifies missing derivations and overstatement in broader summaries. @{rcmt/smg_v2.1/translations/gauge_translation.md#1}, @{rcmt/smg_v2.1/translations/spinor_mapping.md#1}, @{rcmt/smg_v2.1/translations/zeta_resonance.md#53}

## Source-artifact defects

All 21 OPS raw files fail parsing because of embedded results and, for five files, missing opening docstring delimiters. This is packaging evidence, not numerical falsification. Nine extracted scripts passed in onboarding; those copies and exact extraction ranges must remain separate from originals. @{projects/ops-codex/research/ops-syntax-audit.json#1}, @{projects/ops-codex/research/extraction-manifest.json#1}

Lisi's supplied coordinate table has duplicate rows and graph-ID alignment problems, while adjacency and incidence-dual tests pass independently. The source report identifies candidate explanations but does not determine the correct repair. Quarantine coordinate-dependent claims; retain graph-based certificates. @{projects/lisi-smg-test/PHASE_1_2_REPORT.md#43}
''')
}

changes=[
('C01','Critical decision','Two-lane rationale and authority',
 'rcmt/local/invariants.md §2; rcf/core/relational-constraints/geometry.md §§1/6/summary',
 'RCMT requires two lanes; RCF profile calls its architecture “the minimal structure,” while allowing shared representation.',
 'RCMT may adopt a stricter profile, but the original necessity proof was not recovered. Do not infer that invariant labeling supplies one.',
 '@{rcmt/local/invariants.md#25}; @{rcf/core/relational-constraints/geometry.md#28}; @{projects/ops-codex/doc-sweep/evidence/rcf-two-lane-optionalization.diff#1}',
 'Add: “This is a current RCMT-local architectural requirement, not a necessity theorem derived here from RCF or projection alone. Its historical memory/motion rationale is recorded separately.” In the RCF profile, replace universal-sounding minimality with “This profile adopts these roles to represent the listed functions; no universal minimality proof is supplied here.”',
 'Clarification of evidence/authority, preserving the current requirement. Alternatives B/C in report 02 remain unapplied proposals.',
 'rcmt/local/constraints/{symmetry_chain,mode_space,projection}.md; rcmt/global/constraints.md; physics_interface/pipeline/overview.md; RCMT README.',
 'High semantic sensitivity. Explicit owner approval of the provenance wording; any relaxation/promotion is a separate decision D1.'),
('C02','High','Lane independence, product and symplectic notation',
 'rcmt/local/constraints/mode_space.md §§2–4; constraints/projection.md §2; symmetry_chain.md §2',
 '“neither lane may be representationally reducible to the other”; “Π:Sp(2n)→G”; “non-degenerate bilinear form”; bare product candidate notation.',
 'Carrier, field, alternation, independence criterion and projection type are unspecified. Separate roles do not establish independent commuting group factors.',
 '@{rcmt/local/constraints/mode_space.md#39}; @{rcmt/local/constraints/mode_space.md#72}; @{rcmt/local/constraints/projection.md#47}; @{rcmt/local/constraints/symmetry_chain.md#49}',
 'Add: “A realization declares its field, state carrier, nondegenerate alternating form Ω, symmetry action and projection domain/codomain. Transformations claimed symplectic satisfy SᵀΩS=Ω. Independence has the explicitly declared model-local meaning; product-group expressions are candidate realizations, not consequences of two roles.” Retain current prohibition pending a decision on its precise meaning; mark the displayed group-to-geometry arrow schematic until typed.',
 'Clarify a named mathematical structure without adopting OPS real form or forcing a new realization.',
 'RCMT invariants §2; README; measurement/decoherence contracts; symmetry patterns; OPS cross-reference notes.',
 'High. Owner decision D1/D2 required for independence definition, real form or carrier choice; alternation is a definitional clarification.'),
('C03','High','Obsolete projection-stage narrative',
 'rcmt/physics_interface/pipeline/overview.md §§3–5',
 'After LCO: free of mismatch/time; projection is “first non-local” and introduces metric, cones, geometric time; SO(9)→SO(5)×SO(3,1).',
 'Unqualified narrative bypasses locality, time bridge/metric staging and current symmetry-candidate status.',
 '@{rcmt/physics_interface/pipeline/overview.md#80}; @{rcmt/physics_interface/pipeline/overview.md#100}; @{rcmt/physics_interface/README.md#66}; @{rcmt/local/constraints/symmetry_chain.md#84}',
 'Replace with: “Local scoring exposes candidate diagnostics; it does not guarantee zero mismatch or basin stability. Upstream event order is already available. RCMT projection is a representational map preserving local information-flow constraints. Physics-time and metric records enter through their named bridges. Lorentz-compatible representation is required but not realized here; the SO candidates are provisional.” Preserve the removed historical arrows in a dated history note.',
 'Downgrade unsupported outcomes; clarify stage ownership; archive obsolete arrows rather than repair them into a new cascade.',
 'pipeline/physics_interface_contract.md; pipeline/projection_contract.md; local/time_bridge and emergent_metric_geometry overviews; README.',
 'High. Explicit stage/scope decision D3 before applying; replacement does not itself implement Lorentz projection.'),
('C04','High','Physics consumer permissions',
 'rcmt/physics_interface/README.md allowed inputs; pipeline/physics_interface_contract.md §§3–6; local/adapters/local_to_physics_adapter.md',
 'README permits declared local inputs; PIC prohibits adjacency/coherence/upstream modes for “physics modules”; adapter allows local primitives.',
 'Same consumer noun has incompatible import permissions unless distinct stages are intended.',
 '@{rcmt/physics_interface/README.md#30}; @{rcmt/physics_interface/pipeline/physics_interface_contract.md#1}; @{rcmt/physics_interface/local/adapters/local_to_physics_adapter.md#14}',
 'Conditional on D3: “Construction-facing interface modules consume declared local records through adapters. Effective-output consumers receive only explicitly exported projection/metric/time/field artifacts and cannot inspect raw upstream adjacency or scores. Each module declares which consumer class it implements.” If the owner instead selects one interface family, mark the other historical with an exact successor.',
 'Clarification only after scope selection; otherwise this would alter permissions.',
 'physics_interface/local/interfaces.md; local module interfaces; pipeline contracts; SMG physics bindings.',
 'High. Explicit owner decision D3; never resolve by unioning allowed imports.'),
('C05','High','Compactness scope propagation',
 'rcmt/global/constraints.md §6; local/constraints/projection.md §6; README Build Expectations',
 '“No non-compact global groups/symmetries” versus invariant 10’s geometric-only blanket restriction and mode-lane exception.',
 'A narrower compact representation profile can coexist, but a universal mode-group ban conflicts with the current parent wording.',
 '@{rcmt/global/constraints.md#62}; @{rcmt/local/constraints/projection.md#138}; @{rcmt/README.md#68}; @{rcmt/local/invariants.md#93}',
 'Conditional on D4: “Global geometric symmetry groups are subject to invariant 10’s compactness restrictions. The symplectic lane is governed by its declared preservation/admissibility contract. A global-invariant implementation that additionally requires a compact transport image must declare that narrower profile explicitly.”',
 'Clarify current exemption without asserting any particular symplectic real form acceptable.',
 'local/constraints/symmetry_chain.md; mode_space.md; global invariants; physics-interface group constraints; README.',
 'High. Explicit decision D4 on whether global §6 is intentionally a stricter representation profile.'),
('C06','High','Parameter identity and ranges',
 'rcmt/local/invariants.md §11; RCF geometry sweep/plateau prose; LCO memory and temporal-adapter cross-links',
 'Gamma is called constraint density in RCMT/table; LCO gamma is in [0,1], while sweep gamma reaches 5.',
 'The existing owner defines density κ and sealed gamma ranges; the sweep does not declare a conversion or separate identity.',
 '@{rcmt/local/invariants.md#105}; @{rcf/modules/lco/lco-memory.md#18}; @{rcf/modules/temporal-interface/local-constraint-density.md#65}; @{rcf/modules/temporal-interface/temporal-weights-adapter.md#53}',
 'Add: “κ denotes local constraint density; γ_LCO denotes the sealed memory weight in its declared admissible range. Historical γ-sweep values retain their original notation until their profile and relation to κ/γ_LCO are identified; no conversion is asserted here.” Only after that decision update invariant §11’s parenthetical symbol and dependent equations.',
 'Clarification; no invented rescaling or extension of [0,1].',
 'geometry/discrete/mismatch-function.md and density table; adjacent/gamma-plateau-behavior.md; RCMT time constraints; SMG gamma_generations.md.',
 'High. Explicit decision D5 before changing equation symbols or parameter domains.'),
('C07','High','Density formula, table and provenance',
 'rcf/modules/geometry/discrete/mismatch-function.md; candidate-geometries/density-regimes.md; survivor-statistics files',
 'Fixed-coefficient affine F; table has nonzero second difference 0.122 for 600-cell at gamma 0/1/2; claimed observed transitions.',
 'Not the same exact affine function at printed precision; measure normalization, coefficients and run provenance absent from these records.',
 '@{rcf/modules/geometry/discrete/mismatch-function.md#24}; @{rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md#38}; @{rcf/modules/geometry/discrete/e8/survival-stats.md#41}',
 'Add above table: “Historical repository-reported values. Their generation record, parameter convention and connection to the displayed fixed-coefficient model have not been established in this audit; transition thresholds remain reported values, not independently reproduced derivations.” Add to formula: “Reproduction requires the declared spherical measure, normalized vertex set and drag coefficient/profile.” Preserve all current numbers until the source is recovered.',
 'Downgrade unverified provenance, not falsification or numerical repair. D6 decides whether table or model is authoritative for that experiment.',
 '600-cell/dual-5-demicube/e8 survival-stats; adjacent plateau/mode records; GCM indexing; downstream density interpretations.',
 'High. Owner decision D6; no fabricated run IDs, fits, error bars or recalculated table.'),
('C08','High','Regime-local survival versus all-plateau rule',
 'rcf/modules/geometry/adjacent/gamma-plateau-behavior.md §§1/4',
 '“Survivors must remain stable across all plateaus”; “F≤1.0 at all γ-levels.”',
 'Descriptor layer imposes stronger admission than upstream regime-specific sieve and documented high-gamma 600-cell failure.',
 '@{rcf/modules/geometry/adjacent/gamma-plateau-behavior.md#32}; @{rcf/modules/geometry/adjacent/gamma-plateau-behavior.md#101}; @{rcf/modules/geometry/discrete/survival-sieve.md#38}',
 'Replace with: “Survivor status is scoped to the candidate’s declared density regime and evaluation profile. Across-plateau stability, where separately tested, is an additional descriptor, not a universal sieve requirement.”',
 'Clarify downstream scope and remove accidental backflow into admission.',
 'adjacent/survivor-properties.md and symmetry-alignment.md; GCM plateau descriptors; SMG generation dictionary.',
 'Medium implementation risk, high evidence relevance. Owner confirmation D6 if an intentionally narrower robust-survivor class was meant.'),
('C09','High','Dual-projection consistency typing and ownership',
 'rcf/modules/geometry/projection/dual_projection.md §§1–3',
 'Back projection rejects discrete states; G=Πdown(Πup(G)) while Πdown is described as producing consistency conditions.',
 'Return type and equality undefined; downstream filtering appears to select survivors forbidden by the interface.',
 '@{rcf/modules/geometry/projection/dual_projection.md#40}; @{rcf/modules/geometry/projection/interface.md#22}; @{rcf/modules/geometry/projection/projection-invariants.md#24}',
 'Conditional on D7: “Πdown exposes compatibility predicates over an already admitted survivor and its projected record. Failure is a projection-profile diagnostic and does not change upstream survivor status. Dual closure means that the declared compatibility predicate holds.” Retain the old fixed-point formula as historical/schematic until a graph-returning reconstruction map or explicit equivalence is supplied.',
 'Clarification or contract change depending on intended feedback semantics; does not assert an inverse for a lossy map.',
 'projection invariants/outputs; adjacent closure records; SMG Dual Closure thesis cross-reference; compressed shard.',
 'High. Explicit projection-owner decision D7; do not implement a new predicate or feedback algorithm in cleanup.'),
('C10','High','Scoring, selection and aggregation actors',
 'rcmt/README.md Build Expectations; global/constraints.md §4; physics_interface/pipeline/overview.md §3',
 'All mode selection expressed through LCO; LCO/GCO prune neighborhoods; post-LCO state declared perfectly coherent.',
 'LCO scorer does not select; GCO is read-only aggregation, not a cleanup controller.',
 '@{rcmt/README.md#68}; @{rcmt/global/constraints.md#38}; @{rcf/modules/lco/lco-operator.md#181}; @{rcf/modules/gco/interface.md#14}',
 'Replace actor claims with: “Local candidate scores are supplied by the LCO contract; an explicitly declared update policy selects/adopts updates. GCO aggregates both failed and surviving records without modifying them. Preservation of a global invariant is required of the declared admissible update moves.”',
 'Clarification aligning consumers with upstream ownership; no new update policy.',
 'RCMT LCO consumption, global invariant examples, pipeline overview, matter/geometry summaries that name operator agency.',
 'Medium. Future implementation authorization; explicit owner review if any existing module intentionally assigns GCO mutation.'),
('C11','High','Descent and stochastic example correctness',
 'rcf/modules/lco/lco-update-policy.md §§3/5/6',
 'Accepted updates do not increase incoherence; minimum-score selection and Boltzmann sampling are called invariant-compatible; identical inputs imply identical decisions.',
 'Minimum among candidates may exceed current score; unrestricted finite-temperature sampling allows uphill moves; stochastic determinism is overclaimed.',
 '@{rcf/modules/lco/lco-update-policy.md#77}; @{rcf/modules/lco/lco-update-policy.md#115}; @{rcf/modules/lco/lco-update-policy.md#140}',
 'Replace example annotations with: “For an accepted-update non-increase policy, include the current candidate or reject any proposal above Ccurrent. Stochastic examples specify a normalized conditional distribution on the declared admissible set. Equal non-random inputs determine the same conditional law; deterministic replay additionally requires an explicit random trace/seed contract.” If uphill exploration is intended, declare it a separate profile and its descent criterion.',
 'Correct mathematical conditions; distinguish score, proposal distribution and acceptance.',
 'lco-invariants.md §6; LCO README; optional Lyapunov gate cross-reference; downstream stochastic profiles.',
 'High. Explicit decision D8 about pathwise non-increase versus other criteria; no silent change to the invariant.'),
('C12','Medium','LCO uniqueness and physical explanation tone',
 'rcf/modules/lco/README.md design paragraph; rcmt/README.md What RCMT Is',
 '“only operator we have identified”; RCMT “explains” mass/randomness/etc and “provides mechanisms.”',
 'Contract allows a family of scoring functions; local physics requirements are representations, not verified mechanisms.',
 '@{rcf/modules/lco/README.md#30}; @{rcf/modules/lco/lco-operator.md#110}; @{rcmt/README.md#24}; @{rcmt/local/constraints/lco.md#27}',
 'Use: “LCO names the current family of local scoring contracts; a uniqueness theorem for its functional form is not claimed here.” In RCMT: “RCMT explores candidate representations of mass, randomness, measurement and cosmological structure under the listed constraints. These mappings remain under mathematical, computational and empirical evaluation.”',
 'Downgrade summary overstatement while keeping named hypotheses and model requirements.',
 'RCMT introductory summaries, onboarding/navigation summaries; no wholesale rewrite of already scoped matter contracts.',
 'Medium. Future implementation authorization; owner wording review, no new invariant decision.'),
('C13','High','Chromatic-12 / 36-mode evidence chain',
 'rcf/modules/geometry/adjacent/mode-structure.md §§1–3; GCM survivor_indexing.md',
 'Mode count is “not imposed” and “emergent,” while candidates lacking stable 12-mode base are eliminated.',
 'Missing provenance prevents distinguishing independent observation from the adopted filter and bookkeeping product.',
 '@{rcf/modules/geometry/adjacent/mode-structure.md#34}; @{rcf/modules/geometry/adjacent/mode-structure.md#52}; @{rcf/modules/gcm_interface/survivor_indexing.md#28}',
 'Use: “Current descriptor records organize a 12-element base across three plateau labels, yielding a 36-index product. Claims of independent emergence require the candidate set, motif test and evaluation provenance; the indexing record alone does not supply that derivation.” Keep any stronger claim as attributed to its future recovered artifact rather than deleting history.',
 'Clarify evidence and conditionality; preserve current index contract and SMG mapping.',
 'geometry-adjacent README/symmetry alignment; GCM sectorization; SMG bindings, Dual Closure and dictionary.',
 'High. Owner decision D6/D9 before changing imported structural guarantees or evidential status.'),
('C14','High','Triangle compatibility ownership and carrier',
 'rcf/modules/gcm_interface/sectorization.md §§1–3; discrete/survival-sieve.md §6',
 'LCO requires triangle packing and projected structure must support it; 600→120 route said to preserve adjacency.',
 'LCO is geometry-neutral; graph adjacency, face incidence and relation motifs are conflated. Fiber-level dual transport is not a singleton adjacency map.',
 '@{rcf/modules/gcm_interface/sectorization.md#21}; @{rcf/modules/gcm_interface/sectorization.md#42}; @{rcf/modules/geometry/discrete/survival-sieve.md#41}; @{projects/lisi-smg-test/REPORT_PHASE5.md#15}',
 'Add: “Triangle compatibility is a requirement of the declared downstream geometric evaluation profile, not a topology imposed by the LCO scorer. Each projection route identifies its source/target carriers and preserved incidence relation. No claim of triangles in the 120-cell edge graph follows from a triangle-motif or face-incidence requirement.”',
 'Clarify actor and relation types; do not replace the intended projection with the Lisi construction without a map.',
 'candidate-geometries.md; projection adjacency-preservation; GCM sector graph payload; SMG routing translations.',
 'High. Explicit owner decision D7/D9 on motif/graph/incidence meaning; no new carrier forced.'),
('C15','Medium','Local observations labeled empirical invariants',
 'rcf/modules/geometry/discrete/600-cell/survival-stats.md invariants; adjacent φ-spectrum descriptions',
 'φ-structured behavior is “an empirical invariant”; related bodies call reported tiers forced or invariant.',
 'Observation headers do not make reported patterns canonical or empirically validated physics.',
 '@{rcf/modules/geometry/discrete/600-cell/survival-stats.md#44}; @{rcf/modules/geometry/projection/phi-spectrum.md#58}',
 'Use: “Repository-reported computational pattern under the named evaluation profile; not a kernel invariant or physical empirical law. Reproduction and uncertainty are as recorded in the cited run artifact.” Where no artifact is located, say so instead of inventing one.',
 'Downgrade over-authoritative nouns; retain reported patterns and profile restrictions.',
 'discrete survivor records; projection φ-spectrum; adjacent descriptors; GCM annotations.',
 'Medium. Future implementation authorization and evidence-owner review; changes to actual admissibility remain separate.'),
('C16','Medium','GCM mode-source binding repair',
 'rcmt/smg_v2.1/bindings/gcm_bindings.md §3',
 'upstream source gcm_interface/mode_structure.md (absent).',
 'Existing candidate sources are GCM survivor_indexing.md §3 and geometry/adjacent/mode-structure.md; neither path should be silently substituted without scope checking.',
 '@{rcmt/smg_v2.1/bindings/gcm_bindings.md#36}; @{rcf/modules/gcm_interface/survivor_indexing.md#60}',
 'Proposed upstream-source field: “theory/rcf/modules/gcm_interface/survivor_indexing.md, §3 (36-mode indexing), conditional on evaluated survivors.” Keep geometry-adjacent observations as provenance dependencies, not a new direct import of sibling internals.',
 'Editorial dependency repair if the existing index contract is the intended import; otherwise mark unresolved.',
 'SMG smg_contract/README; frozen reference remains unchanged; regenerated index references only after confirmation.',
 'Low–medium. Confirm binding identity with owner D9; no new upstream file or payload invented.'),
('C17','Medium','Physics binding path and boundary provenance',
 'rcmt/smg_v2.1/bindings/physics_bindings.md §§1–5',
 'physics_interface/conversion_laws/formal_structure.md; other sources named only as final boundary payload.',
 'Path omits local/; actual conversion document defines supports and final boundary tuple but no instantiated experiment payload.',
 '@{rcmt/smg_v2.1/bindings/physics_bindings.md#12}; @{rcmt/physics_interface/local/conversion_laws/formal_structure.md#21}; @{rcmt/physics_interface/local/conversion_laws/formal_structure.md#181}',
 'Use exact schema source “theory/rcmt/physics_interface/local/conversion_laws/formal_structure.md, §§1/6.” Add: “This is a boundary schema import; an instantiated payload must separately identify its producer and provenance.”',
 'Upgrade navigability and distinguish definition from supplied numerical/physical data.',
 'SMG exports/policy; physics-interface local/conversion_laws interface; status registry.',
 'Low for path, medium for permission scope. Future implementation authorization; D3 confirmation if consumer class unresolved.'),
('C18','Medium','SMG status, proposed routing and missing refactor record',
 'rcmt/smg_v2.1/README.md; translations/24-cell-routing.md and 24-cell-tranlsation.md',
 'Requested REFACTOR_STATUS.md absent; routing says “Canon: Proposed” but “accepts ... only when” identities present; filename tranlsation is misspelled.',
 'Proposal scope/version/adoption is not bound to the main SMG contract; a missing status file is not proof of an incomplete refactor.',
 '@{rcmt/smg_v2.1/README.md#88}; @{rcmt/smg_v2.1/translations/24-cell-routing.md#3}; @{rcmt/smg_v2.1/translations/24-cell-routing.md#92}; @{rcmt/smg_v2.1/translations/24-cell-tranlsation.md#109}',
 'Add registry wording: “Routing identities are a proposed translation extension; normative use requires an explicit adopting model/version. No completed-refactor claim is made by this registry.” Add links to current split files and decisions. Rename misspelled path only with references/compatibility redirect updated together.',
 'Clarification of status; do not manufacture a historical refactor completion record or activate proposed requirements.',
 'SMG smg_contract/README/exports; downstream matter alignment; future regenerated references.',
 'Medium. Explicit owner decision D9 for adopting routing extension; editorial registry can follow approved facts.'),
('C19','Medium','RCP sentinel and header migration',
 'rcf/rcp-1.2.md placement/validator sections; RCMT document headers',
 'No leading delimiter; validator rejects absent “leading header sentinel”; RCMT headers omit canonical fields or use depends_on.',
 'Protocol wording ambiguous; field migration cannot assign authority from folder names or turn every historical file into an active contract.',
 '@{rcf/rcp-1.2.md#53}; @{rcf/rcp-1.2.md#236}; @{rcmt/local/invariants.md#3}; @{projects/ops-codex/doc-sweep/evidence/census-summary.json#1}',
 'Proposed validator bullet: “the document does not begin with its single H1 followed by the canonical Document Contract.” Migrate active participating headers by classifying existing owner/type/scope/dependencies; keep “depends_on” material only after checking equivalence with upstream dependencies. Tag historical/frozen sources in an external manifest rather than rewriting them.',
 'Protocol clarification and editorial migration; preserve status/authority distinction and anti-backflow.',
 'All participating RCMT/SMG headers; RCP examples/validators; generated compression; optional source manifest.',
 'Medium. Protocol-owner decision D10 and migration scope approval; routine fields then batch-authorized.'),
('C20','Low','Navigation, conceptual path aliases and stale tree',
 'rcmt/README.md tree/build; local/README.md upstream paths; rcf/modules/geometry/projection/README.md file list',
 'RCMT root invariants/symmetry paths absent; theory/rcf_projects references; projection list uses underscore filenames where actual files are hyphenated.',
 'Navigation obscures the actual authority sources. Compressed snapshot is not immutable authority over changed sources.',
 '@{rcmt/README.md#50}; @{rcmt/README.md#68}; @{rcmt/local/README.md#3}; @{rcf/modules/geometry/projection/README.md#92}',
 'Replace tree with actual local/, physics_interface/, global/, smg_v2.1/, modules/ locations. Use explicit repository-root path convention. Say: “Read upstream source contracts; a compressed shard is a subordinate snapshot and must identify its source revision.” Fix list labels to existing hyphenated names.',
 'Straightforward editorial repair; source selection follows explicit existing contracts, not external old compressed trees.',
 'README navigation links; module indexes; compression dependency registry.',
 'Low. Future implementation authorization only; flag any alias whose authority target is not established.'),
('C21','Low','Mathematical typesetting without formula repair',
 'rcf/modules/geometry/discrete/mismatch-function.md equations; RCMT math delimiters',
 'Malformed E*{static}, R*{drag}, integral asterisks; some literal escaped display delimiters.',
 'Display corruption is separable from unresolved numerical semantics; neither should hide the other.',
 '@{rcf/modules/geometry/discrete/mismatch-function.md#28}; @{rcf/modules/geometry/discrete/mismatch-function.md#42}; @{rcmt/local/constraints/projection.md#53}',
 'Render the existing symbols as E_{static}, R_{drag} and the intended sphere-integral notation, leaving coefficients and measure explicitly unspecified pending D6. Normalize display delimiters after preview checks; do not change variables, parameter domains or equations under a formatting label.',
 'Editorial repair only; formula/provenance changes remain C07.',
 'Affected rendered equations; derived compressed math excerpts rebuilt last.',
 'Low. Future implementation authorization only; no formula correction smuggled into typesetting.'),
('C22','Medium','OPS executable and result packaging',
 'projects/Operational_Phase_Space/SIM14.0.py–SIM16.4.py',
 'Raw files contain code and unquoted archived stdout; five lack opening docstring delimiters.',
 'All raw files fail parsing; arbitrary cleanup could detach archived results from their source.',
 '@{projects/ops-codex/research/ops-syntax-audit.json#1}; @{projects/ops-codex/research/extraction-manifest.json#1}; @{projects/ops-codex/research/reproduction-results.json#1}',
 'Proposed archive note: “Original mixed code/result records remain immutable. Executable extracts identify source SHA-256, retained line ranges and delimiter repairs; recorded and newly reproduced stdout are separate artifacts with their own run metadata.” Later publish a manifest/index to those artifacts rather than overwriting originals.',
 'Reproducibility upgrade without history loss; no additional run authorized by this proposal.',
 'OPS navigation/status; onboarding reproducibility notes; any future computational evidence citations.',
 'Medium. Owner approval D11 for canonical executable location and archival convention; original sources remain unchanged now.'),
('C23','Medium','Scoped theorem and reproduced algebra upgrades',
 'OPS Vocabulary_Packet.md and SIM14_Discussions.md later-status annotations; SIM16 result index',
 'Chronological discussion calls orbit bridging a candidate; algebraic results may be read as only observations.',
 'Existing proof and retained rerun logs justify stronger conditional status, without rewriting discovery history.',
 '@{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#5}; @{projects/Operational_Phase_Space/SIM14_Discussions.md#1218}; @{projects/ops-codex/research/reproduction-results.json#1}',
 'Add dated note: “The orbit-bridge criterion is proved under the stated free-V4/commuting-involution hypotheses. Its frozen finite-carrier application was reproduced; physical selection remains unproved.” Label SIM16 results “conditional exact algebraic classifications, rerun from documented extracts,” retaining each unresolved selection/observability statement.',
 'Upgrade proof/reproducibility status at the exact claimed level, not canon or physical validity.',
 'OPS vocabulary/guide/status index; later reports citing M1/M2 or metrics; no upstream invariants changed.',
 'Low–medium. Future implementation authorization and review of exact hypotheses/citations; no theory decision.'),
('C24','Medium','Lisi coordinate provenance quarantine',
 'projects/lisi-smg-test/reference_geometry/120cell_vertices.txt usage documentation; PHASE_1_2_REPORT.md',
 '600 coordinate rows are parseable, but 597 distinct keys and a zero-length listed edge; cause not identified.',
 'Coordinates cannot silently serve as an indexed realization for the validated adjacency graph.',
 '@{projects/lisi-smg-test/PHASE_1_2_REPORT.md#43}; @{projects/lisi-smg-test/PHASE_1_2_REPORT.md#74}',
 'Add usage note: “The supplied coordinate artifact is retained as provenance input and is not validated against adjacency IDs. Graph/dual certificates do not rely on it. Coordinate-dependent use requires a separately sourced, validated realization and an explicit ID map.”',
 'Clarify quarantine; preserve validated graph results and original coordinates.',
 'Lisi README/reference inventory; geometry loader usage; any coordinate plots or future embedding claims.',
 'Medium. Future implementation authorization for note; explicit source-owner decision D11 before coordinate repair/replacement.'),
('C25','Medium','Lisi summary separates raw and quotient configurations',
 'projects/lisi-smg-test/summary.md Configuration Analysis',
 '“Boolean local configuration structure”; structures “emerged ... rather than being designed into it.”',
 'Raw subset carrier was an explicit definition; orbit quotient is non-distributive; global quotient is not a lattice.',
 '@{projects/lisi-smg-test/summary.md#99}; @{projects/lisi-smg-test/REPORT_PHASE6D.md#26}; @{projects/lisi-smg-test/REPORT_PHASE6D.md#149}; @{projects/lisi-smg-test/REPORT_PHASE6C.md#1}',
 'Use: “By the explicit subset-configuration definition, four carriers give the Boolean raw lattice B4. The recovered V4 action yields seven orbit classes whose quotient lattice is non-distributive. The global 24-fiber orbit poset is a different object and is reportedly not a lattice.”',
 'Clarification and downgrade of emergent-language overreach; preserve recovered symmetry result.',
 'Lisi summary and configuration navigation; downstream interpretations referencing seven classes.',
 'Low. Future implementation authorization only; retain global-enumeration verification caveat.'),
('C26','Low','Lisi historical plan and frozen comparison target',
 'projects/lisi-smg-test/PLAN.md repository audit; REPORT_PHASE6.md scope',
 'Plan says src/tests empty and later comparison not begun; Phase 6 compares references/smg_v2.1_concat.md.',
 'Plan is historical; frozen comparison does not automatically describe all current split SMG files or proposed routing extension.',
 '@{projects/lisi-smg-test/PLAN.md#5}; @{projects/lisi-smg-test/REPORT_PHASE6.md#3}',
 'Prepend status note after a proper historical-record header: “The following inventory is the planning-time snapshot. Current phase status is indexed separately. Phase 6 conclusions apply to the hashed SMG reference consumed by that comparison; the reference is not silently replaced by current module prose.”',
 'Editorial history/status clarification; keep original plan and frozen reference bytes.',
 'README_FIRST/current status index; summary; future comparison citations.',
 'Low. Future implementation authorization for external/current note; do not rewrite frozen reference artifacts.'),
('C27','Medium','Generated compression and invariant-count calibration',
 'rcf/rcf_compressed.md build record; RCMT references to it; Codex onboarding invariant summary',
 'Compressed shard dated July 25 with aggregate digests; current sources include later candidate constraints. Shard counts ten, onboarding says nine.',
 'Compression is subordinate and may be stale; nine substantive clauses plus numbered closure explains ten without an inconsistency.',
 '@{rcf/rcf_compressed.md#14}; @{rcf/rcf_compressed.md#128}; @{rcf/core/S∞.RCF.Invariants.v1.1.md#126}; @{projects/ops-codex/onboarding/02-invariants-and-constraints.md#1}',
 'Use invariant summary: “Sections 1–9 state substantive invariants; section 10 states invariant closure; the compressed index represents all ten sections.” After approved source migration, regenerate shard with exact source commit/file manifest and a reproducible build method; before then label its freshness explicitly unknown against later edits.',
 'Clarify own summary and upgrade derived-artifact provenance; do not change root invariant count/content to make a digest match.',
 'RCMT local README upstream references; generated indexes/shards; Codex onboarding summaries (separate derived-note edit).',
 'Medium. Future implementation authorization; D10 approval of regeneration inputs/method. No guessed digest algorithm.'),
('C28','Medium','Global invariants: input records and admissible moves',
 'rcmt/global/constraints.md §§3/4/8; global/invariants/{twist,winding,holonomy}.md',
 'Invariants evaluable from “adjacency alone,” but definitions use transported orientations/relational transformations and resist all local cleanup.',
 'Graph topology, transport labels and the permitted move class are not distinguished; preserved value depends on the declared implementation.',
 '@{rcmt/global/constraints.md#25}; @{rcmt/global/constraints.md#38}; @{rcmt/global/invariants/holonomy.md#1}',
 'Use: “The implementation declares adjacency and any transported orientation/transformation records needed for its invariant. It specifies the admissible local moves under which the invariant is preserved. No metric or standalone ontological field is implied by those records.” Keep any stronger topology-only assertion conditional on an explicit encoding.',
 'Clarify under-specified contract rather than prove global invariants nonexistent or change their intended protected role.',
 'global README/invariant examples; projection interpretation; cleanup-policy consumers.',
 'Medium–high. Owner decision D4/D7 on intended data and move class; cleanup must not invent them.'),
('C29','Medium','Spinor closure certificates and terminology',
 'rcf/modules/geometry/discrete/spinor-closure.md and candidate spinor-embedding documents',
 '600-cell, dual demicube and E8 described as closed/stable spinor structures under the current update profile.',
 'Common vocabulary does not identify the same product, carrier, field, normalization or exhaustive closure certificate across candidates.',
 '@{rcf/modules/geometry/discrete/spinor-closure.md#24}; @{rcf/modules/geometry/discrete/spinor-embedding.md#22}',
 'Add: “Each closure claim is relative to the declared representation, composition and candidate set. A mathematical closure certificate and a simulation persistence record are separate evidence items; absent a linked certificate/run, the claim remains repository-reported for that profile.” Preserve any known exact polytope facts separately.',
 'Clarify evidence scope; do not substitute a standard multiplication or assert all claims false.',
 'candidate sieve and survivor stats; geometry-adjacent closure terminology; downstream Dual Closure references.',
 'Medium. Evidence-owner review D6/D9; no new closure test or multiplication introduced in documentation cleanup.'),
('C30','Medium','Preserve strong negative comparison result',
 'projects/lisi-smg-test/summary.md architectural consequences/current interpretation; downstream SMG comparison summaries',
 'Broad recovery narrative may obscure Phase 6 no-equivalence result and non-invariant local 3+1 selection.',
 'Reproduced comparison tests justify a precise negative result while not ruling out a different future enriched model.',
 '@{projects/lisi-smg-test/REPORT_PHASE6.md#14}; @{projects/lisi-smg-test/REPORT_PHASE6.md#96}; @{projects/ops-codex/research/lisi-selected-tests.txt#1}',
 'Add: “For the retained transported action and supplied SMG reference, no full structure-preserving correspondence is established. A local 3+1 singleton is not invariant under the recovered transitive generation stabilizer. The reproducible heterogeneous 36-record projection requires extra choices and discards relations.”',
 'Upgrade visibility of supported negative evidence; not universal impossibility or a new SMG requirement.',
 'Lisi summary; SMG comparison-status note; downstream carrier-identification claims; historical reference unchanged.',
 'Low–medium. Future implementation authorization; review scope/action qualifiers, no authority change.'),
]

ledger=['''## How to review this ledger

Every entry is a proposal, not an applied edit. “Future implementation authorization” is required for every source mutation because this request authorizes planning only. A separate **owner decision** is identified where wording would choose an unresolved mathematical or architectural commitment. Editorial repairs can later be batch-authorized; they do not justify a new permission question during this planning pass.

Priorities identify research impact, while risks identify the chance of changing meaning. Replacement wording is ready for review; conditional replacements must not be applied before their decision gate is resolved. No row authorizes changing historical failures, inventing evidence, broad source deletion, or adopting a new model.

''']
labels=['File and section','Current wording or precise claim','Identified inconsistency','Evidence and source references','Proposed replacement wording','Change rationale','Affected downstream documents','Risk and approval']
for row in changes:
    ident,priority,title,*values=row
    ledger.append(f'## {ident} — {title}\n\nPriority: **{priority}**.\n\n')
    for label,value in zip(labels,values):
        ledger.append(f'**{label}:** {value}\n\n')
reports['05-proposed-change-ledger.md']=('Proposed change ledger',''.join(ledger))

reports.update({
'06-dependency-and-propagation-plan.md':('Dependency reconstruction and staged migration plan',r'''
## Authority and propagation direction

```mermaid
flowchart TD
  RCP[RCP document protocol] --> H[Participating document headers]
  K[RCF kernel and closure] --> E[Optional elaboration profiles]
  K --> L[Time and local execution contracts]
  L --> G[Discrete survivor and projection interfaces]
  G --> A[Geometry descriptor and GCM records]
  K --> RI[RCMT local invariants and requirements]
  RI --> RC[RCMT implementation and projection contracts]
  RC --> PI[Physics construction interfaces and consumer boundaries]
  A --> S[SMG bindings]
  PI --> S
  S --> ST[SMG translations and adopter policies]
  ST --> M[Matter and geometry-behavior applications]
  E -. provenance only .-> RI
  O[OPS and Lisi evidence] -. scoped results only .-> RC
  O -. scoped results only .-> ST
```

This is a map for review/propagation, not a derivation of the theory. RCP governs document structure without adding scientific invariants. Evidence arrows do not confer authority to revise parent contracts. The optional RCF memory/motion profile is historical rationale; current RCMT adoption needs its own decision record. @{rcf/rcp-1.2.md#167}, @{rcf/core/relational-constraints/geometry.md#3}, @{rcmt/local/README.md#13}

## File-resolution plan

| Reference family | Current defect | Minimal proposed resolution | Propagation checkpoint |
|---|---|---|---|
| GCM mode binding | `gcm_interface/mode_structure.md` absent | Confirm `rcf/modules/gcm_interface/survivor_indexing.md` §3 is the intended public index import | No new raw adjacency/action payload inferred; conditional scope retained |
| Physics support/boundaries | Binding omits `local/`; boundary sources named generically | Point to `rcmt/physics_interface/local/conversion_laws/formal_structure.md` §§1/6 and distinguish schema from instance | Check consumer class and source ownership |
| RCMT root tree | Root-level conceptual filenames no longer exist | List actual `local/constraints/`, `physics_interface/`, `global/`, `smg_v2.1/`, `modules/` | Every new navigation link resolves; no authority chosen by folder name |
| RCF projection file list | Underscore/hyphen mismatch | Match the existing exact filenames | Link-only diff; no formula changes |
| Compressed source aliases | `rcf_projects/rcf_compressed.md` and predecessor protocol paths outside audited inputs | Use current RCF source contracts plus a versioned subordinate compressed snapshot | Do not inspect/import unrelated trees to force a match |
| SMG refactor/routing status | Status file absent, proposed routing requirements unbound | Create a factual status registry only after adoption facts are approved | Do not invent refactor completion or activate proposed requirements |

Sources: @{rcmt/smg_v2.1/bindings/gcm_bindings.md#36}, @{rcmt/smg_v2.1/bindings/physics_bindings.md#12}, @{rcmt/README.md#50}, @{rcmt/local/README.md#3}, @{rcf/modules/geometry/projection/README.md#92}.

## Stage 0 — preserve the audit boundary

Record current source SHAs, dirty-file status, hash manifests and relevant history. Preserve original OPS mixed records, frozen SMG comparison reference, Lisi coordinates and past failed results. This proposal already captures the source baseline and local history; see `evidence/`. The closing byte comparison is `evidence/source-integrity.json`. A commit SHA alone does not cover pre-existing modified Python caches.

**Exit criterion:** no source changes in this planning pass; every proposed edit maps to a ledger ID, owner and future authorization. No scientific test reruns are needed to approve a path-only repair.

## Stage 1 — approve scope decisions before editing semantics

Resolve D1–D10 as applicable. The critical first gate is whether RCMT retains its current two-domain requirement, adopts functional separation with shared representation, or moves the split into an opt-in profile. The least-change recommendation is to retain the requirement and annotate its provenance, with mathematical realization still open. Define which physics contracts address construction modules versus effective consumers. Clarify whether global compactness is a stricter representation profile.

**Exit criterion:** a concise decision record states retained requirements, rejected alternatives and exactly which files inherit the choice. No cleanup worker infers a decision from a missing proof, commit recency or a successful downstream simulation.

## Stage 2 — one small editorial batch

After implementation authorization, apply only resolved navigation/schema-path corrections, conditional-status labels, typo/formatting repairs and historical-plan annotations (C16/C17/C20/C21/C26 and factual parts of C18). Header migration follows the agreed protocol scope; leave scientific claims untouched in this batch. Preserve frozen-reference bytes with external manifests/notes rather than in-place modernization.

**Checks:** resolve all changed links; preview math; verify no variable, inequality, coefficient, requirement or import permission changed; compare source diffs against the authorized IDs. Do not rename the misspelled SMG filename without updating its incoming references or a compatibility alias.

## Stage 3 — owner contracts in dependency order

1. Resolve RCP sentinel wording and active-header migration classification (C19); do not manufacture owner authority.
2. Clarify upstream optional-profile language, LCO/policy/GCO actors, accepted-update conditions, and projection-consistency typing within approved scopes (C01/C09/C10/C11/C12).
3. Attach regime-scoped numerical/evidence metadata and clarify geometry descriptor permissions (C06/C07/C08/C13/C14/C15/C29). Preserve table values pending source recovery.
4. Apply approved RCMT invariants/constraint/compactness/consumer-boundary decisions (C01–C06/C28), followed immediately by the summaries and interfaces that consume them.

**Atomic review rule:** do not leave an “active” downstream consumer claiming an import that the same batch removes upstream. Either migrate both ends together or mark the consumer explicitly pending/inactive with the previous version named. Unresolved sections may retain their old text with a conflict annotation; no forced implementation is needed to make every page agree.

**Checks:** each changed normative sentence cites its owning contract; each mathematical expression has a defined domain/codomain; preserved meanings and intentional changes are listed separately. A changed group symbol, compactness scope or lane-independence rule is never classified as formatting.

## Stage 4 — downstream propagation and evidence calibration

Update GCM public descriptor imports before SMG bindings. Then update SMG claim/dictionary/translation/policy/export consistency, followed by matter/geometry-behavior consumers that actually depend on a changed guarantee. Preserve already well-scoped gauge/spinor/zeta and candidate-matter language; do not rewrite every downstream document solely for tone. Update only reverse dependencies that quote changed clauses or expose changed payloads.

Add dated status notes for theorem/proof and exact algebra upgrades, the Lisi no-correspondence result and local singleton obstruction (C23/C25/C30). Keep historical SIM15.4 failures alongside SIM15.5's distinct normalizer success. Global Phase 6C and full H4-family status stay report-only unless separately reproduced in a later authorized task.

**Checks:** every upgraded claim carries assumptions, source revision and verification level; physical validity is never inferred from theorem or code success. Verify that no metadata-only change silently adds missing evidence or changes the retained symmetry action.

## Stage 5 — derived artifacts last

Rebuild the compressed shard and indexes only after source batches settle (C27). Record source commit(s), complete file manifest, build method/version, included/excluded classes and reproducible hashes. Preserve frozen Lisi comparison input as a distinct historical artifact; a new comparison against updated SMG is separate research, not documentation migration.

**Checks:** generated authority never outranks sources; kernel summary retains sections 1–9 plus closure §10; optional RCF profile remains optional; RCMT status matches approved decisions. Run structural document validation rather than asserting compliance from field presence alone. Executable examples changed under C11 warrant targeted checks of the actual acceptance contract; unchanged research mathematics does not need broad expensive reruns.

## Review packages and rollback

Prefer small review packages: navigation/status; upstream contract clarifications; RCMT architectural decision propagation; evidence-language notes; derived regeneration. Attach ledger IDs and before/after semantic summaries to each. Retain source-revision snapshots so a rejected wording change can be reverted without altering archived research results. Do not delete failed hypotheses or overwrite coordinates to make a package pass.

This report proposes the migration sequence only. No source batch, publication, new theory investigation or new computational program is initiated.
'''),
'07-human-decisions-required.md':('Human decisions required before implementation',r'''
## Decision boundary

No source-edit approval is requested mid-audit: the requested output is this concrete proposal. Any implementation requires a subsequent instruction. The decisions below identify changes a cleanup pass must not choose on the owner's behalf. Routine editorial repairs can later receive a single batch authorization.

## Decision register

| Decision / priority | Choice to make | Recommended minimum-change disposition | Consequence / blocked ledger entries |
|---|---|---|---|
| D1 / Critical | Keep RCMT's mandatory distinct symplectic/geometric domains; weaken to functional roles allowing shared carrier; or make it a profile? | **Keep the existing requirement**, explicitly label it adopted architecture and record that its necessity proof was not recovered. Do not change invariant authority during cleanup. | C01/C02 and all lane-dependent propagation. Options B/C are commitment changes, not editorial fixes. |
| D2 / High | What does independence/non-breaking mean; which real form/carrier is intended? | Clarify current definitions and mark realization details open. Do not choose Sp(8,R) merely because OPS used it, and do not require a direct product. | C02; mode/projection/measurement contracts need defined domains and preserved structure before an executable realization claim. |
| D3 / High | Are pipeline PIC and local physics interfaces distinct consumer stages, or is one historical/superseded? | Explicitly separate construction-facing interfaces from effective-output consumers **if that matches intended architecture**, then assign each module. | C03/C04/C17; permissions cannot be silently unioned or inherited from a directory name. |
| D4 / High | Does compactness restrict only global geometric symmetry, or also a special compact transport representation of global invariants? | Retain local invariant 10's current exemption; declare a stricter global representation profile only if intended. | C05/C28; no implicit ban or authorization of any symplectic real form. |
| D5 / High | Are sweep gamma, memory gamma and density κ distinct parameters or related by a declared profile? | Preserve table notation as historical; use the existing κ/γ distinction in explanatory prose; recover the actual mapping before changing formulas/domains. | C06/C07; no ad hoc normalization, [0,5] extension or fitted transform. |
| D6 / High | Which model/run produced the density table, tier-drag narrative and crossover thresholds? | Add a provenance-gap note and retain original numbers until source recovery. If no run survives, classify honestly as historical reported/illustrative only after owner review. | C07/C08/C13/C15/C29; do not fabricate computational provenance or silently downgrade to illustration. |
| D7 / High | Is Πdown a downstream consistency predicate, actual reconstruction map, or a feedback process? What are the carriers/relations for triangle and global-invariant evaluation? | Keep upstream survivor selection sealed; express compatibility as a separate declared diagnostic if that is the intended function. | C09/C14/C28; a feedback architecture or new type is substantive and must be approved explicitly. |
| D8 / High | Is accepted-update non-increase mandatory, or may a distinct policy permit uphill moves under another criterion? | Preserve non-increase for the current stated descent profile; restrict/annotate examples accordingly. Name any other policy profile separately. | C11; changing pathwise descent to expected descent would change the commitment and requires its own specification. |
| D9 / Medium–High | Which GCM index source is authoritative; are proposed routing identities adopted; what is the evidential status of 12/36-mode emergence? | Confirm the existing public index binding, leave proposed routing opt-in/unadopted until explicitly chosen, and preserve observation/definition/interpretation distinctions. | C13/C14/C16/C18/C29; no new payload, new SMG generation policy or inherited graph action invented. |
| D10 / Medium | Does RCP's “sentinel” mean the H1/contract structure, and which active documents must migrate? | Remove the ambiguous sentinel term in favor of the explicit existing leading structure; classify active headers and preserve historical/frozen records externally. | C19/C27; derived compression rebuilt last with approved manifest/build method. |
| D11 / Medium | Where do canonical executable extracts and corrected-coordinate provenance belong? | Preserve original research artifacts; use versioned extracts and quarantine invalid coordinate alignment. Decide source correction only after provenance recovery. | C22/C24; no overwritten script history, inferred coordinate repair or new run program. |

## Facts that do not need a theory decision

- The RCF profile became optional by an explicit recorded revision. Current RCMT invariants still require a two-lane architecture. These facts can be recorded without deciding whether the architecture is physically necessary. @{projects/ops-codex/doc-sweep/evidence/rcf-two-lane-optionalization.diff#1}, @{rcmt/local/invariants.md#25}
- The old chain is already historical under the current symmetry contract. Updating a stale summary to match that status does not select SO(8) or SO(9). @{rcmt/local/constraints/symmetry_chain.md#94}
- The printed 600-cell sweep values are not an exact fixed-coefficient affine function at the stated precision. Their cause remains unresolved. @{rcf/modules/geometry/discrete/mismatch-function.md#24}, @{rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md#38}
- The public GCM mode-source path named by SMG does not exist, while the current GCM indexing document supplies a plausible intended index. A path correction still needs source-identity confirmation. @{rcmt/smg_v2.1/bindings/gcm_bindings.md#36}
- An eight-state torsor, four-carrier power set, 24-cell adjacency, fiber quotient and physical particle interpretation are different objects. They need not be rewritten into one common representation to make the documentation consistent.

## Editorial batch candidates after authorization

Exact existing-file navigation, formatting that preserves equations, historical-plan annotations, schema-versus-instance labels with confirmed paths, and the nine-plus-closure count clarification can be reviewed together. A header should report authority already granted by an owner; adding a header never grants new authority. Proposed wording with unresolved D1–D11 dependencies remains out of that batch.

## Remaining evidence limitations

The original pre-repository two-lane derivation was not found in the allowed local history. Remote commits, excluded historical trees and the other agent's output were not consulted. Source files were inspected directly; the prior onboarding served as inventory rather than sole authority. No new external literature validation, simulation run, full H4 census or global configuration enumeration was undertaken. The Lisi PDF's full textual/phase fidelity remains outside the carrier-level verification previously performed.

The census is a broad structural screen with targeted semantic review, not a complete theorem-verification certificate for the entire corpus. Exact source references, captured historical snapshots and unresolved interpretations are preserved so a later owner decision can be made without silently filling those gaps.

Stop point: proposal delivered. Research sources remain unchanged; no cleanup implementation or theoretical extension has begun.
''')
})

def refs(s):
    return re.sub(r'@\{([^#}]+)#(\d+)\}',lambda m:f'[{m[1]}:{m[2]}]({(ROOT/m[1]).as_posix()}:{m[2]})',s)

def write():
    for name,(title,body) in reports.items():
        header=f'''# {title}

## Document Contract

- owner: Codex documentation audit
- type: Observation
- scope: proposal-only audit of authorized theory inputs, 2026-10-24
- authority: independent assessment; no authority to revise source contracts
- upstream dependencies: cited working-tree sources, captured Git history and prior scoped verification records
- status: planning proposal; source changes not executed
- purpose: propose controlled consistency and evidence-language repairs
- anti-backflow: proposed downstream clarifications cannot redefine upstream invariants or promote hypotheses into source authority

'''
        (OUT/name).write_text(header+refs(body.strip())+'\n',encoding='utf-8')

if __name__=='__main__':write()
