"""Render the independent onboarding notes with absolute source links."""
from pathlib import Path
import json, re

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'projects/ops-codex/onboarding'

reports = {
'00-repository-map.md': ('Repository map and provenance', r'''
Audit date: 2026-10-24 (America/Phoenix). Agent: Codex. Scope: repository rehydration and research audit, ending before new theoretical investigations.

## Source identity

| Input | Git identity | Inspection boundary |
|---|---|---|
| `rcf/` | `c419353214ee83d83376747064912544aee257b9` | Read-only working tree; pre-existing tracked Python cache modifications were present |
| `rcmt/` | `b7eecd0dfd8080864b34441af86329f819cba65a` | Read-only working tree; initial Git status clean |
| `projects/Operational_Phase_Space/` | Not a Git repository at this location | Copied, unversioned research input; file hashes recorded |
| `projects/lisi-smg-test/` | Not a Git repository at this location | Copied, unversioned research input; file hashes recorded |

Git ownership checks required command-local `-c safe.directory=<exact repository>` for read-only queries. No global Git configuration changed. A commit SHA does not identify pre-existing working-tree differences: the 569-file SHA-256 baseline in @{projects/ops-codex/research/source-manifest.json#1} identifies the inspected non-Git-metadata inputs more precisely. The baseline was taken before computational reruns, after initial reconnaissance. The closing comparison is @{projects/ops-codex/research/source-integrity-check.json#1}.

The user's pasted onboarding instructions were inspected in the chat. The external desktop file and attachment were not opened. No content from `projects/ops-antigravity/` was inspected. Other top-level research trees were not used as evidence. No external literature was fetched. The Lisi PDF's original text was not independently reviewed; paper attribution below is attributed to the local reconstruction report, with finite reconstruction tested separately.

## Navigation map

| Area | Role and useful entrypoints |
|---|---|
| RCF identity and authority | @{rcf/README.md#3}; @{rcf/rcp-1.2.md#28}; @{rcf/core/S∞.RCF.Invariants.v1.1.md#3} |
| RCF formal expression | `core/rcd/` describes substrate-independent relational admissibility, explicitly non-empirical; `core/calculus/`, `core/algebra/`, `core/geometry/` provide mathematical vocabulary. @{rcf/core/rcd/README.md#3} |
| Time and execution | `modules/time/`, `modules/temporal-interface/`, `modules/lco/`, `modules/gco/`: ordering, sealed parameters, local scoring, aggregation. @{rcf/modules/lco/interface.md#3} |
| Geometry | `modules/geometry/discrete/`, `projection/`, `continuous/`, `dynamics/`, `adjacent/`: candidate records, handoff, shadows, subsequent descriptors. @{rcf/modules/geometry/discrete/README.md#14} |
| Representation boundary | `modules/gcm_interface/`: indexing and sector descriptors, conditional on evaluated survivors. @{rcf/modules/gcm_interface/README.md#3} |
| RCMT local architecture | `local/invariants.md`, then `local/constraints/` and `local/symmetry_groups/`; load order and provisional roles in @{rcmt/local/README.md#13}. |
| RCMT global layer | `global/`: twist, winding, holonomy subject to adjacency transport and preservation requirements. @{rcmt/global/constraints.md#5} |
| RCMT physics interfaces | `physics_interface/local/`: group structures, relational quantum mechanics, kinematics, dynamics, discrete field theory, time bridge, continuous field theory, emergent metric geometry, conversion laws; `pipeline/` is another contract organization requiring reconciliation. |
| SMG | `smg_v2.1/`: read-only bindings → local claims/dictionaries/translations → policies/exports. @{rcmt/smg_v2.1/README.md#54} |
| Downstream matter and geometry behavior | `modules/survivor_matter_structure/`, `modules/geometry_behavior/`, `modules/metric/`; candidate particle labels and derived geometry descriptions. @{rcmt/modules/survivor_matter_structure/README.md#1} |
| OPS | 21 scripts: SIM14.0–14.8, SIM15.0–15.6, SIM16.0–16.4; vocabulary, learner guide, theorem, two discussion records, post-theorem exploration. All are static carrier/algebra investigations in this supplied series. |
| Lisi study | `README_FIRST.md`, `PLAN.md`, phase reports, `src/`, nine test modules, `artifacts/`, `reference_geometry/`, `references/`. Geometry normalization → inscribed cells → enriched Lisi embedding → dual transport → intrinsic analysis → SMG comparison → refinement/configurations. |
| Codex outputs | Eight reports here; `../research/` contains audit scripts, extracted executable copies, manifests, test logs, and rerun stdout/stderr. |

## Discovery discrepancies

Actual directory case is `rcf/`, `rcmt/`, `projects/`; Windows resolves the supplied uppercase names. `rcmt/smg_v2.1/REFACTOR_STATUS.md` is absent. RCMT's top-level README tree points to files such as root `invariants.md`, while the operative local files are under `local/`; consult contracts rather than that tree. @{rcmt/README.md#50}

The Lisi `PLAN.md` retains a planning-time claim that `src/` and `tests/` are empty and comparison phases have not begun, despite the present reports and executable files. It is historical state, not a current inventory. @{projects/lisi-smg-test/PLAN.md#5}
'''),
'01-architecture.md': ('Architecture and dependency reconstruction', r'''
## Dependency direction

RCF defines a domain-neutral constraint framework. RCMT consumes it to constrain physics-oriented models. SMG is a further downstream semantic translation model, not the RCF kernel and not the physics-interface owner. OPS and the Lisi study are research implementations used to examine particular finite constructions; successful execution cannot grant them upstream authority.

```mermaid
flowchart TD
  K[RCF invariant kernel] --> F[RCD and formal vocabulary]
  F --> T[Time and sealed temporal interface]
  T --> L[LCO local candidate evaluation]
  L --> G[GCO aggregation and persistence]
  G --> D[Discrete survivor records]
  D --> P[Projection handoff]
  P --> C[Continuous structural descriptions]
  C --> I[GCM representation interface]
  K --> R[RCMT local invariants and adapters]
  R --> PI[RCMT physics interfaces]
  I --> S[SMG bindings and semantic translations]
  PI --> S
  S --> M[Candidate matter and geometry behavior]
```

This is a reconstruction of ownership and consumption, not an asserted universal chronology or a demonstrated physical emergence chain. RCD is a formal language, not an evolution law. The LCO and GCO contracts separate local evaluation from aggregation; geometry interprets their records downstream. Projection consumes survivors without selecting them or supplying a metric. Sources: @{rcf/core/rcd/README.md#20}, @{rcf/modules/lco/README.md#1}, @{rcf/modules/geometry/projection/README.md#51}, @{rcf/modules/gcm_interface/README.md#1}, @{rcmt/local/README.md#13}.

## Interfaces that matter

| Boundary | Authorized content | Boundary to preserve |
|---|---|---|
| Kernel → modules | Nine framework invariants | No mechanisms, physical meanings, or candidate geometries become kernel axioms |
| Time → Temporal Interface → LCO | Event ordering, declared density input, sealed scoped weights | LCO cannot retune gamma/epsilon or schedule events |
| LCO → GCO | Candidate admissibility and local diagnostics with provenance | Local score is not a global geometry selector |
| Discrete geometry → projection | Survivor structure and preserved relational content | No metric, spacetime, or physical law introduced by the handoff alone |
| GCM → SMG | Chromatic index, plateau labels, sectors/adjacency descriptors | Evaluated-survivor descriptors are conditional; SMG labels do not backflow |
| RCMT local → physics interfaces | Explicit local records and staged handoffs | No physics-time before time bridge; no metric before emergent metric geometry |
| SMG → adopters | `8+3+1`, three-generation semantics, continuous approximations | Opt-in domain obligations, not universal physics exclusions |

Sources: @{rcf/core/S∞.RCF.Invariants.v1.1.md#20}, @{rcf/modules/temporal-interface/interface.md#14}, @{rcf/modules/lco/interface.md#14}, @{rcmt/physics_interface/README.md#1}, @{rcmt/physics_interface/local/time_bridge/README.md#1}, @{rcmt/smg_v2.1/smg_contract.md#15}.

## OPS's role

SIM14 freezes an eight-state graph, projection pairing, and half-turn, then tests alternative symplectic pairings and resulting group actions. SIM15 tests the recovered abstract extension architecture in an H4 environment and measures native incidence. SIM16 studies compatible complex structures, metrics, projection, and local generator spaces on the frozen OPS representation. They do not implement the missing physical dynamics. In particular, the permutation `p`, basin-observation map, and RCF's general projection operator are different objects until an explicit connecting map is supplied. @{projects/Operational_Phase_Space/Vocabulary_Packet.md#64}, @{projects/Operational_Phase_Space/SIM15.0.py#1}, @{projects/Operational_Phase_Space/SIM16.3.py#1}

The Lisi study takes a separate route: supplied adjacency → tetrahedral duality → graph-defined inscribed 24-cells → named projective state actions → 20-vertex target fibers → configurations. Its Phase 6 compares that enriched object with a frozen concatenated SMG reference. It does not demonstrate that this is the same realization as OPS's H4 construction or the current RCMT mode-space candidate. @{projects/lisi-smg-test/REPORT_LISI_120CELL_TRANSPORT.md#17}, @{projects/lisi-smg-test/REPORT_PHASE6.md#14}

## Competing interface descriptions requiring review

The `physics_interface/README.md` permits declared upstream constraints, project-local adapters, and interface records. The `pipeline/physics_interface_contract.md` instead describes a projection-output-only firewall and prohibits consuming adjacency, upstream modes, and coherence scores. SMG explicitly imports structural support through GCM plus physics-interface records. These may be distinct consumer stages, but the reviewed files do not provide an explicit reconciliation or supersession declaration. Preserve the narrower permissions at each named boundary; do not merge the contracts silently. @{rcmt/physics_interface/README.md#1}, @{rcmt/physics_interface/pipeline/physics_interface_contract.md#1}, @{rcmt/smg_v2.1/smg_contract.md#15}

Likewise, RCMT's introductory claim that it supplies mechanisms must be read alongside the local LCO consumption contract's prohibition on treating LCO as a dynamical mechanism or causal agent. The contracts support a representational research architecture; they do not establish those advertised physical explanations. @{rcmt/README.md#16}, @{rcmt/local/constraints/lco.md#27}
'''),
'02-invariants-and-constraints.md': ('Invariants, authority, and exclusions', r'''
## Authority rules

The root RCF shard has explicit invariant authority and no upstream dependency. Its pre-theoretical commitments are project definitions, not experimentally established laws. Optional modules can be eliminated without changing the kernel. RCP-1.2 requires owner/type/scope/authority/dependencies/status/purpose/anti-backflow; a status such as active does not raise authority. Header scope bounds body prose. @{rcf/core/S∞.RCF.Invariants.v1.1.md#3}, @{rcf/README.md#84}, @{rcf/rcp-1.2.md#72}, @{rcf/rcp-1.2.md#180}

## RCF invariant kernel

| Invariant | Operational reading and limit | Source line |
|---|---|---|
| Persistence under constraint | Project meaning is persistence; failures are eliminated | @{rcf/core/S∞.RCF.Invariants.v1.1.md#35} |
| Emergence via elimination | No additive rescue of unstable relations | @{rcf/core/S∞.RCF.Invariants.v1.1.md#48} |
| Relational primacy | Identity/meaning depends on relations | @{rcf/core/S∞.RCF.Invariants.v1.1.md#58} |
| Spectral dominance | Persistent structure is declared spectral | @{rcf/core/S∞.RCF.Invariants.v1.1.md#68} |
| Constraint over dynamics | Admissibility rather than fundamental equations of motion | @{rcf/core/S∞.RCF.Invariants.v1.1.md#78} |
| Minimal free parameters | Structural ratios should survive parameter removal | @{rcf/core/S∞.RCF.Invariants.v1.1.md#87} |
| Non-privileged observers | Observation constrains resolution; relabeling has no privileged observer | @{rcf/core/S∞.RCF.Invariants.v1.1.md#96} |
| Self-applicability | Framework receives no exemption | @{rcf/core/S∞.RCF.Invariants.v1.1.md#106} |
| Epsilon allowance | Residual alternatives remain; no literal single-mode collapse | @{rcf/core/S∞.RCF.Invariants.v1.1.md#115} |

These are not yet quantitative specifications of what counts as increasing constraint, spectral persistence, or sufficient epsilon in every implementation. Those choices require scoped module definitions.

## RCMT-local admissibility

`local/invariants.md` calls itself highest within RCMT and subordinate to all upstream RCF contracts. It supplies 15 requirements. @{rcmt/local/invariants.md#3}

| Requirement family | Meaning | Status |
|---|---|---|
| Adjacency and locality | No global metric/background/field-first ontology; information transfer adjacency-local | Model constraints, sections 1 and 14 |
| Two symmetry lanes | Lorentz-compatible geometric lane plus unbroken symplectic lane | Required role; `Sp(8)×SO(9)` and `Sp(8)×SO(8)` provisional |
| LCO consumption | Local evaluation, no global evolution law | Upstream contract consumption |
| Projection | Observables are projected mode structure | Required representation; explicit physical realization remains missing |
| Mass/randomness/measurement | Resistance, unresolved projection structure, boundary resolution | Required forms, not derived physical laws |
| Basins/dark-matter-like content | Basins allowed; dark-matter-like models must use non-projectible modes | Scoped representational conditions |
| Curvature/finite structure | No global de Sitter/conformal curvature groups, infinite towers, KK/string ladders; finite-dimensional adjacency-compatible representations | Geometric compactness except local Lorentz interface; symplectic lane not automatically compact |
| Time/energy/c/identity | Derived event-index time; energy as load; c as maximum local refresh rate; contingent identity | RCMT commitments, not measurements reproduced here |

Sources: @{rcmt/local/invariants.md#20}, @{rcmt/local/invariants.md#46}, @{rcmt/local/invariants.md#79}, @{rcmt/local/invariants.md#93}, @{rcmt/local/invariants.md#105}.

The old `E8 → SO(10) → Sp(8)×SO(9)` chain is expressly historical exploratory work. Neither compact SO candidate is claimed to contain Lorentz symmetry by conventional compact symmetry breaking. A projection-level realization remains unspecified. Do not infer an embedding from dimensions or names. @{rcmt/local/constraints/symmetry_chain.md#49}, @{rcmt/local/constraints/symmetry_chain.md#84}

## Module-specific restrictions

LCO scores local candidates using bounded neighborhoods, memory, and sealed upstream parameters; it cannot select global geometry, infer physical meaning, or schedule. Its unit-norm comparison space is a representation. @{rcf/modules/lco/interface.md#14}, @{rcf/modules/lco/README.md#1}

The geometry sieve applies mismatch, mode completeness, a declared spinor-composition profile, recursion, density viability, and triangle compatibility. Its candidate file additionally requires H4 compatibility. These restrict the tested profile and candidate pool, so survival is conditional on that pool and filter; it is not independent discovery that H4 is uniquely physically necessary. @{rcf/modules/geometry/discrete/survival-sieve.md#23}, @{rcf/modules/geometry/discrete/candidate-geometries.md#24}

Global twist, winding, holonomy must use adjacency transport, remain distinguishable, and survive allowed local cleanup. The global layer cannot redefine local primitives. Its blanket compactness language is broader than local invariant 10's symplectic exception; applicability to the mode lane requires human clarification. @{rcmt/global/constraints.md#25}, @{rcmt/global/constraints.md#62}, @{rcmt/global/constraints.md#90}, @{rcmt/local/invariants.md#93}

SMG fixes 36 semantic modes and three generations for adopters; a 37th mode or fourth generation needs a model-contract revision. These are model policies. Gauge, spinor, mass, and zeta language remains downstream. @{rcmt/smg_v2.1/claims/dual_closure.md#1}, @{rcmt/smg_v2.1/policies/limitations_and_boundaries.md#1}

## Definition and contract discrepancies

1. RCP-1.2 says no leading delimiter/front matter, but its validation list still requires a “leading header sentinel.” These requirements need clarification before claiming formal header compliance. @{rcf/rcp-1.2.md#53}, @{rcf/rcp-1.2.md#236}
2. Several RCMT local headers use `depends_on`, omit canonical fields, or lack a Document Contract. SMG files advertise RCP-1.1 compatibility. This is mixed protocol migration, not evidence that their scientific bodies are false. @{rcmt/local/invariants.md#3}, @{rcmt/local/constraints/symmetry_chain.md#3}, @{rcmt/smg_v2.1/README.md#118}
3. The mismatch document states `F(G,gamma)=E_static(G)+gamma R_drag(G)`. The density table's 600-cell values 0.472, 0.550, 0.750 at gamma 0,1,2 do not lie on one such line: successive increments are 0.078 and 0.200. A stochastic estimate, gamma-dependent coefficients, or a different model could explain it, but no such explanation was established here. Do not treat the table as verified output of the stated fixed-coefficient formula. @{rcf/modules/geometry/discrete/mismatch-function.md#24}, @{rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md#35}
4. LCO's overview bounds a memory weight gamma to [0,1], whereas the geometry density table uses gamma up to 5. These may be different parameters linked through a sealed conversion, but that identification was not established in the reviewed material. @{rcf/modules/lco/README.md#1}, @{rcf/modules/temporal-interface/interface.md#14}, @{rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md#35}
'''),
'03-mathematical-structures.md': ('Mathematical structures and evidence status', r'''
Evidence tags: **M** = mathematical consequence whose argument was reviewed; **C** = computation rerun in this audit; **R** = repository report or recorded output only; **D** = framework/model definition; **H** = unverified hypothesis; **I** = interpretive mapping. C means an existing implementation was reproduced, not an independently implemented proof or physical validation.

## Finite OPS carrier

The frozen carrier is `X8={0,...,7}`, cycle edges `{01,02,13,24,35,46,57,67}`, projection involution `p=(01)(23)(45)(67)`, half-turn `h=(07)(16)(25)(34)`. **D**. `C8` denotes the cycle graph here, not a cyclic group symbol. A vector representation on R8 does not turn eight finite states into eight spacetime dimensions. @{projects/Operational_Phase_Space/Vocabulary_Packet.md#8}

`K=<p,h>` is V4 with two free four-state orbits. For a commuting involution `s` outside K, `<K,s>` is C2^3. Equivariance sends an entire K-orbit to an entire K-orbit. Orbit exchange gives transitivity; group and carrier both have order 8, so stabilizers are trivial. Conversely, preserving both orbits prevents transitivity. This proves the orbit-bridge regularity theorem under its explicit assumptions. **M,C**. @{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#1}, @{projects/ops-codex/research/SIM14.3.py.stdout.txt#1}

M1=`(01)(24)(35)(67)` preserves the two orbits; M2=`(02)(13)(46)(57)` exchanges them. Both generate abstract C2^3, but only M2 acts regularly on X8. The abstract group supplies Fano incidence in either case; regularity is needed to realize the carrier as an affine F2^3 torsor. An origin and basis choice are additional conventions. No multiplication with octonionic signs or nonassociativity follows. **M,C**. @{projects/Operational_Phase_Space/Theorem_Learners_Guide.md#1905}, @{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#1}

## Symplectic and affine structures

Omega is a chosen real nondegenerate alternating form; its support is four conjugate pairs, not all eight C8 edges. `Sp(8,R)` in OPS means real 8×8 form-preserving matrices. RCMT's bare `Sp(8)` does not by itself specify that real form or the compact convention. A finite permutation stabilizer, a Lie group, and a finite-dimensional representation are different levels of structure. **D,M**. @{projects/Operational_Phase_Space/SIM14.0.py#1}, @{rcmt/local/constraints/mode_space.md#36}

With the M2 torsor, affine planes number 14 in seven parallel classes and `AGL(3,2)` has order 1344 (**R**, also reconstructed in SIM16 **C**). The unsigned partner stabilizer H192 is an extension `C2^3 ⋊ S4`; the quotient action fixes one nonzero binary direction, and the center has order 2. Its complement changes with origin. The signed stabilizers have `N+ ≅ S4`, `N± ≅ S4×C2`. Do not equate `V_Omega = G∩N+` with the earlier K: their intersection has order 2. **C** for SIM14.5/14.8 and SIM16 group reconstructions; finer SIM14.6/14.7 classifications are **R**. @{projects/Operational_Phase_Space/SIM14.6.py#1}, @{projects/Operational_Phase_Space/SIM14.7.py#1}, @{projects/ops-codex/research/SIM14.8.py.stdout.txt#1}

## H4, D4, and polytopes

SIM15 constructs W(H4), order 14400, and native 600-cell/120-cell geometry, then searches for the previously fixed OPS extension fingerprint. Reports give one matching kernel class, normalizer L of order 192, and its normalizer N of order 576. L is constructively compared with W(D4); N/L is an outer order-three triality extension after additional tests. **R** for the full family census; the one-cell N/L and three-kernel/facet-incidence refinement reran successfully in SIM15.6 **C**. That script uses floating coordinates and tolerance 1e-7 for native geometry; exact permutation arithmetic downstream does not remove this numerical reconstruction dependency. @{projects/Operational_Phase_Space/SIM15.0.py#1}, @{projects/Operational_Phase_Space/SIM15.2.py#1}, @{projects/Operational_Phase_Space/SIM15.6.py#1}

There are reported 25 native 24-cells with two pair relations (disjoint or six shared vertices). The native cell stabilizer is N576, not L192. SIM15.4 falsified the latter; SIM15.5 reports literal equality of the native stabilizer family with the normalizer family, plus equivariance. **R**. Equal orders alone were explicitly insufficient. The 25-object disjointness relation has SRG parameters (25,8,3,2); the Lisi study independently tests a rook-graph isomorphism for its recovered cell relation. Do not infer an ambient identification between the two pipelines without a map. @{projects/Operational_Phase_Space/SIM15.4.py#1}, @{projects/Operational_Phase_Space/SIM15.5.py#1}, @{projects/lisi-smg-test/REPORT_PHASE3A.md#56}

## Compatible complex structures and metrics

For frozen Omega, SIM16 uses `J0=-Omega`, `K=abs(Omega)`, and compatible positive metrics. This **K is an 8×8 partner matrix**, not SIM14's subgroup `<p,h>`. Symmetry alone leaves metric families of dimensions 4 (N+) and 2 (N±). Adding strong C8 matrix sparsity leaves dimensions 2 and 1 respectively. In the signed case `G(t)=cosh(t)I+sinh(t)K` and `J(t)=-Omega G(t)` for real t. **C**, with exact symbolic classifications plus numeric regression witnesses. @{projects/Operational_Phase_Space/SIM16.0.py#156}, @{projects/Operational_Phase_Space/SIM16.1.py#251}

SIM16.2 distinguishes exact parameter identifiability from parameter selection and operational observability. Symplectic conjugacy is not an established operational gauge. A primitive stochastic operator with one-dimensional real fixed space cannot commute with a real complex structure: the invariant fixed line cannot support J²=-I. A positive-metric isometry cannot be a mixing primitive stochastic contraction. These are algebraic obstructions to specific simultaneous requirements, not a ban on all stochastic dynamics. **M,C**. @{projects/Operational_Phase_Space/SIM16.2.py#172}, @{projects/Operational_Phase_Space/SIM16.2.py#224}

SIM16.3 splits the p-eigenspaces as 4+4, with induced symplectic/complex structures and complex quotient descent. Direct two-form descent through the observation quotient fails. The intrinsic sector swap fails the frozen observation and adjacency tests (adjacency traces 2 and -2). **C**. SIM16.4 finds a 12-dimensional graph-local symplectic algebra `sp(2,R)^4`, a four-dimensional abelian fixed-J stabilizer, zero common fixed generator for all Jt, and one family-transport direction `-K/2`. These are allowable algebraic generators, not an LCO or physical time evolution. **C**. @{projects/Operational_Phase_Space/SIM16.3.py#1}, @{projects/Operational_Phase_Space/SIM16.4.py#253}

## Lisi transport and configurations

The tested source is a projective 24-weight structure with named C/P/T and triality permutations. Its 24-cell adjacency is preserved by the tested embedding. The 600→120 duality changes each selected source vertex into its 20 incident tetrahedra, producing 24 disjoint dodecahedral fibers covering 480 target vertices, a 24-cell bridge quotient, and 120 residual target vertices arranged as 24 K1,4 stars. **C** for selected geometry/transport suites; fidelity to the original paper's full spinorial phases remains outside that verification. @{projects/lisi-smg-test/REPORT_LISI_24CELL_EMBEDDING.md#1}, @{projects/lisi-smg-test/REPORT_LISI_120CELL_TRANSPORT.md#34}

Phase 6C defines activity as subset membership of 24 fibers: the Boolean space B24 is **D**, and its raw count 2^24 is **M**, not an emergent physical state count. The reported 177,976 order-96 orbits and non-lattice orbit poset were not recomputed here (**R**). Phase 6D independently reconstructs a regular V4 action on four pair carriers, 16 subsets, seven orbits, three occupancy-two classes, and an orbit lattice containing M3 (**C**). The local Boolean lattice and its seven-class quotient are distinct. @{projects/lisi-smg-test/REPORT_PHASE6C.md#1}, @{projects/lisi-smg-test/REPORT_PHASE6D.md#26}, @{projects/lisi-smg-test/REPORT_PHASE6D.md#136}

## Physics-facing translations

Chromatic-12, ordered gamma plateaus and 36 indices are conditional structural records (**D/R**). SMG's Dual Closure thesis, gauge labels SU(3)/SU(2)/U(1), Spin(4)/Dirac language, generation interpretation, and zeta/mass mapping remain **H/I**. Count matches 8/3/1 do not construct Lie brackets, representations, anomaly cancellation, measured couplings, or a mass spectrum. No such derivation was verified in this audit. @{rcf/modules/gcm_interface/survivor_indexing.md#28}, @{rcmt/smg_v2.1/smg_contract.md#42}, @{rcmt/smg_v2.1/translations/gauge_translation.md#1}, @{rcmt/smg_v2.1/translations/spinor_mapping.md#1}, @{rcmt/smg_v2.1/translations/zeta_resonance.md#1}
'''),
}

reports.update({
'04-simulation-inventory.md': ('Simulation and verification inventory', r'''
## Reproducibility conditions

All 21 supplied OPS files fail AST parsing as stored. They mix executable code with pasted results; five also lack the opening triple quote for their introductory prose. This is an archival packaging failure, not a falsification of the mathematics. No source file was repaired. @{projects/ops-codex/research/ops-syntax-audit.json#1}

The audit created executable copies by truncating at the first results delimiter and, where needed, prepending one opening docstring delimiter. SIM14.5 has a separately recorded cut at source line 208. No equations, group definitions, thresholds, or assertions were edited. Exact cut ranges are in @{projects/ops-codex/research/extraction-manifest.json#1}; source hashes are in the baseline manifest. Twenty-one extracted copies parse successfully. Nine selected copies ran with a 60-second per-process limit, all exiting zero. See @{projects/ops-codex/research/reproduce.py#1} and @{projects/ops-codex/research/reproduction-results.json#1}.

Existing Python 3.12.4 from the explicit executable recorded in the Lisi report was used with `-B`; NumPy, SymPy, NetworkX, and pytest were available. No dependency was installed. Outputs, scripts, and logs are confined to `projects/ops-codex/`. Tests disabled pytest's cache provider. Do not execute the Lisi `*_verify.py` entrypoints in the research source: several write into their own `artifacts/` directory.

**C** below means rerun from the extracted copy; **R** means source code and archived result reviewed, not rerun. For every row, the named source file is the evidence location. All inherit the finite, frozen-input scope and introduce no ISP/LCO dynamics.

## SIM14: finite carrier and symmetry

| Script | Objective / supplied assumptions | Measured or classified result | Audit |
|---|---|---|---|
| SIM14.0 | Canonical symplectic basis on R8; frozen C8,p,h; no vector fitting | Conjugate matching can sit in C8; Omega support is not the whole cycle; nonunique pairing | R |
| SIM14.1 | M2 selected from compatible pairings | C2^3 regular action, torsor coordinates, seven-direction/Fano incidence | R; key action independently checked by 14.3 |
| SIM14.2 | Substitute M1, freeze other inputs | Same abstract C2^3; action remains 4+4 and nonregular | R; contrast checked by 14.3 |
| SIM14.3 | Exhaust all 105 perfect matchings; retain those commuting with p,h | 13 retained; four bridge/regular, nine nonbridge/nonregular; C8 containment leaves one regular candidate | C |
| SIM14.4 | Freeze M2; enumerate planes and carrier permutations | 14 affine planes, seven parallel classes; full affine symmetry 1344; enriched lab stabilizer four | R |
| SIM14.5 | Compare affine and signed/unsigned Omega stabilizers | Signed groups 24/48; unsigned pairing group 192; ablation and incidence classifications | C |
| SIM14.6 | Classify H192, N+, N± and representation commutant | N+ is S4; N± is S4×C2; real representation 2 trivial + 2 standard-three copies; commutant dimension eight | R; group orders and commutant checked again in 16.0 |
| SIM14.7 | Distinguish V_Omega from original K | V_Omega normal V4 in N+; quotient S3; explicit split; K∩V_Omega order two | R |
| SIM14.8 | Conjugation of H192 on translation subgroup | C2^3⋊S4 fingerprint; fixed binary line; center order two; origin-dependent complements | C |

Sources: @{projects/Operational_Phase_Space/SIM14.0.py#1}, @{projects/Operational_Phase_Space/SIM14.1.py#1}, @{projects/Operational_Phase_Space/SIM14.2.py#1}, @{projects/Operational_Phase_Space/SIM14.3.py#1}, @{projects/Operational_Phase_Space/SIM14.4.py#1}, @{projects/Operational_Phase_Space/SIM14.5.py#1}, @{projects/Operational_Phase_Space/SIM14.6.py#1}, @{projects/Operational_Phase_Space/SIM14.7.py#1}, @{projects/Operational_Phase_Space/SIM14.8.py#1}.

Regularity is computed from group actions independently of the bridge matrix, so SIM14.3's discriminator is not defined into the endpoint. Fano incidence, by contrast, is a standard consequence once C2^3 has been established. Choosing M2 fixes a branch; it does not demonstrate physical selection of that branch.

## SIM15: external H4 stress test

| Script | Question / controls | Recorded result | Audit |
|---|---|---|---|
| SIM15.0 | Independently construct H4 then search the full OPS extension fingerprint | Five C2^3 classes; one full matching class; 25 conjugate matching architectures | R |
| SIM15.1 | Normalizer closure and native eight-state primitive survival | L192→N576 split C3; antipodal center; no literal induced C8 on vertex torsors; signed Omega not recovered | R |
| SIM15.2 | D4 triality, order-five transport, degree-25 action | Explicit D4/triality tests pass; action rank three, subdegrees 1/8/16, kernel two; primitive action; C5 pentads not blocks | R |
| SIM15.3 | Reconstruct L-pair geometry from T containment/intersections | 75 kernels → 25 parents, three per parent; containment graph 25 stars; kernel intersection matrices fail to distinguish L relation | R |
| SIM15.4 | Blind native 24-cell census; test literal cell stabilizers=L192 | 25 cells recovered, stabilizers order 576; proposed equality fails; native rank-three graph still survives | R |
| SIM15.5 | Test cell stabilizers=N_W(L) rather than L | Archived all gates pass: literal normalizer-family equality, unique equivariance, disjoint/shared-six ↔ L intersection 8/12 | R; full family census not rerun |
| SIM15.6 | One native cell; distinguish three kernel/facet matchings through incidence, plus erased-incidence control | Nontrivial facet-stabilizer relation selects one matching, equivariant under all 576 elements; abstract C3 alone selects none | C |

Sources: @{projects/Operational_Phase_Space/SIM15.0.py#1}, @{projects/Operational_Phase_Space/SIM15.1.py#1}, @{projects/Operational_Phase_Space/SIM15.2.py#1}, @{projects/Operational_Phase_Space/SIM15.3.py#1}, @{projects/Operational_Phase_Space/SIM15.4.py#1}, @{projects/Operational_Phase_Space/SIM15.5.py#1}, @{projects/Operational_Phase_Space/SIM15.6.py#1}.

The supplied SIM15 discussion principally interprets earlier stages; later script truth packets carry later gate outcomes. False gates caused by failed prerequisites must be distinguished from independently tested falsifications. Tolerance-based H4 coordinate recognition also limits claims of purely exact reconstruction.

## SIM16: exact algebra on the frozen representation

| Script | Constraint or question | Result reproduced | Audit |
|---|---|---|---|
| SIM16.0 | Does signed permutation symmetry uniquely fix compatible J/g? | No: compatible metric dimensions 4/2 for N+/N±; nontrivial witnesses survive | C |
| SIM16.1 | Add strong C8 sparsity to metric entries | Dimensions 2/1; signed family exactly cosh(t)I+sinh(t)K; only 2/4 group elements preserve the full graph | C |
| SIM16.2 | Parameter identification, equivalence and stochastic obstructions | t identifiable algebraically; selection/observability unresolved; conjugacy not proven gauge; primitive mixing commutation/isometry no-go | C |
| SIM16.3 | Projection eigensectors, descent and controls | Compatible complex quotient; direct two-form descent fails; sectors intrinsically equivalent but frozen observation/adjacency distinguishes them | C |
| SIM16.4 | Graph-local symplectic generators, fixed-J and family transport | Local dimensions 24 unrestricted/12 symplectic; sp(2,R)^4 closure; fixed-J dimension four; all-J common space zero; family transport one-dimensional | C |

Sources: @{projects/Operational_Phase_Space/SIM16.0.py#156}, @{projects/Operational_Phase_Space/SIM16.1.py#186}, @{projects/Operational_Phase_Space/SIM16.2.py#108}, @{projects/Operational_Phase_Space/SIM16.3.py#1}, @{projects/Operational_Phase_Space/SIM16.4.py#154}. Each rerun has its own `research/SIM*.py.stdout.txt` and `.stderr.txt`; JSON truth packets are retained verbatim.

## Lisi study: executable scope

| Stage / implementation | Objective and assumptions | Supported outcome | Reproduction |
|---|---|---|---|
| `geometry_loader.py`, `dualization.py`; Phase 1–2 | Supplied adjacency is authoritative separately from coordinates | Graph normalization, tetrahedral-cell duality and explicit graph isomorphism; coordinate alignment defect detected | Tests rerun |
| `phase3a_geometry.py` | Explicit independence/antipode/distance-two 24-cell predicate | Exhaustive recovery under that predicate; 25 cells; overlap and incidence structure | Tests rerun |
| `lisi_24cell_embedding.py` | Exact rational projective weights and named permutations reconstructed by repository authors | Enriched 24-state embedding into recovered cell; no full spinor-phase representation claim | Tests rerun; PDF fidelity not re-audited |
| `lisi_dual_transport.py` | Lift source vertices to incident tetrahedra before dual map | 24 dodecahedral fibers, bridge quotient, named action preservation; carrier change is necessary | Tests rerun |
| `intrinsic_phase5.py` | Analyze transported graph plus distinguished fibers and order-96 action | Pair/generation partitions, residual stars, bridge cycles, full fiber-preserving symmetry comparison | Tests rerun |
| `phase6_compare.py` | Compare explicit preserved relations with frozen SMG concatenation | No full correspondence; numerical projection forgets structure and requires choices | Tests rerun |
| `phase6b_refinement.py` | Classify additions invariant under the specified order-96 action | No invariant local singleton; three equally admissible quad refinements | Tests rerun |
| `phase6c_configurations.py` | All subsets of 24 fibers under that action | Reported 177,976 orbits, non-lattice quotient poset; no selected state | Not rerun: complete 2^24 enumeration and large lattice artifacts |
| `phase6d_local_four.py` | All subsets of one generation's four pair carriers | Regular V4; seven orbits; three occupancy-two classes; non-distributive orbit lattice | Tests rerun |

Eight selected test modules passed: 26 geometry/transport/intrinsic tests in 49.68 seconds and 16 comparison/refinement/local tests in 1.03 seconds, total **42 passed**. This is not the complete nine-module suite. Logs: @{projects/ops-codex/research/lisi-geometry-tests.txt#1}, @{projects/ops-codex/research/lisi-selected-tests.txt#1}.

The comparison tests rely on existing artifacts and the frozen SMG concatenation. They do not prove every stored artifact was generated by the claimed algorithm, establish physical correctness, or compare all current RCMT modules. The separate geometry suites reconstruct substantial earlier geometry, but the full global configuration enumeration remains report-only.
'''),
'05-survivor-and-failure-ledger.md': ('Survivor and failure ledger', r'''
Every survival is relative to a named constraint set. Failure of one realization does not falsify all RCMT; later success of a different object does not erase the failed realization.

| Construction or claim | Outcome | Conditions / evidence boundary | Evidence |
|---|---|---|---|
| C8 admits symplectic pair support | Survives nonuniquely | Canonical basis relabeling; not equality of cycle and Omega support | @{projects/Operational_Phase_Space/SIM14.0.py#1}; archived |
| Any compatible pairing gives a regular X8 torsor | Fails | M1 supplies same abstract C2^3 but two four-state orbits | @{projects/Operational_Phase_Space/SIM14.2.py#1}; checked in 14.3 rerun |
| M2 orbit bridge yields regular action | Survives | Commuting independent involution exchanges two free V4 orbits | @{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#1}; proof reviewed, computation rerun |
| Fano incidence establishes octonions/physical sectors | Unsupported promotion | Fano incidence alone gives no multiplication signs, associator or physical map | @{projects/Operational_Phase_Space/SIM14.1.py#39} |
| H192 split affine extension | Survives | Frozen unsigned pairing/affine carrier; complement depends on origin | @{projects/ops-codex/research/SIM14.8.py.stdout.txt#1}; rerun |
| OPS group architecture survives in H4 | Reported survivor | Full extension fingerprint, not order coincidence; one matching class | @{projects/Operational_Phase_Space/SIM15.0.py#1}; census not rerun |
| H4 vertex torsors reproduce frozen C8 and signed Omega | Fails / not forced | No literal induced C8 in tested vertex torsors; unsigned antipode is insufficient for Omega | @{projects/Operational_Phase_Space/SIM15.1.py#1}; archived |
| Kernel intersection data recover architecture pair geometry | Fails | Cross-fiber matrices do not distinguish the two L relations | @{projects/Operational_Phase_Space/SIM15.3.py#1}; archived |
| L192 equals native 24-cell stabilizer | Fails | Native stabilizers have order 576; literal subgroup-family test fails | @{projects/Operational_Phase_Space/SIM15.4.py#1}; archived |
| N_W(L) equals native stabilizer family | Reported survivor | New hypothesis, equality inside same ambient group plus unique equivariance | @{projects/Operational_Phase_Space/SIM15.5.py#1}; full census not rerun |
| Abstract triality C3 selects a facet correspondence | Fails alone | Three equivariant bijections survive after incidence erased | @{projects/Operational_Phase_Space/SIM15.6.py#1}; rerun |
| Facet stabilizer relation selects a correspondence | Survives | Unique relative to the frozen binary incidence predicate; not uniqueness over all conceivable relations | @{projects/ops-codex/research/SIM15.6.py.stdout.txt#1}; rerun |
| Symmetry alone rigidly fixes J/metric | Fails | 4D/2D compatible families survive | @{projects/ops-codex/research/SIM16.0.py.stdout.txt#1}; rerun |
| C8 sparsity fixes one signed metric | Fails | Leaves exact 1D partner family; no selected t | @{projects/Operational_Phase_Space/SIM16.1.py#251}; rerun |
| Symplectic equivalence is operational gauge | Not established | Algebraic conjugacy does not establish observational equivalence | @{projects/Operational_Phase_Space/SIM16.2.py#239}; rerun metadata explicitly source-limited |
| Mixing primitive stochastic operator also commutes with J or is a positive-metric isometry | Fails under stated hypotheses | Fixed-space parity / spectral modulus obstructions | @{projects/Operational_Phase_Space/SIM16.2.py#224}; argument reviewed and witnesses rerun |
| Projection carries the two-form directly | Fails | Observation quotient does not admit direct Omega descent; induced eigensector structure differs from direct descent | @{projects/Operational_Phase_Space/SIM16.3.py#1}; rerun |
| Intrinsic sector equivalence preserves frozen observation and C8 | Fails | Sector swap fails p/B/adjacency; adjacency traces differ | @{projects/ops-codex/research/SIM16.3.py.stdout.txt#1}; rerun |
| Local symplectic generator family | Survives algebraically | sp(2,R)^4; fixed-J versus moving-family transport kept separate | @{projects/Operational_Phase_Space/SIM16.4.py#253}; rerun |
| Supplied 120-cell coordinates align with graph IDs | Fails | Duplicate coordinates, zero-length listed edge, edge-length outliers; do not repair by assuming a new ID map | @{projects/lisi-smg-test/PHASE_1_2_REPORT.md#43}; geometry tests rerun |
| Supplied graph duality | Survives | Tetrahedral face incidence gives graph isomorphism to supplied 120-cell adjacency | @{projects/lisi-smg-test/PHASE_1_2_REPORT.md#74}; tests rerun |
| Lisi state transports as singleton target vertex | Fails | State must lift to a 20-vertex dodecahedral fiber | @{projects/lisi-smg-test/REPORT_LISI_120CELL_TRANSPORT.md#17}; tests rerun |
| Enriched Lisi relations transport to fibers | Survives | Named relations and fiber metadata retained; bare target adjacency loses labels | @{projects/lisi-smg-test/REPORT_LISI_120CELL_TRANSPORT.md#68}; tests rerun |
| Transported geometry equals SMG | Not established | Only assumption-dependent forgetful projection; missing SMG adjacency/action payload | @{projects/lisi-smg-test/REPORT_PHASE6.md#14}; comparison tests rerun |
| Intrinsic local 3+1 partition | Fails under retained action | Generation stabilizer is transitive on four pairs; singleton requires extra datum | @{projects/lisi-smg-test/REPORT_PHASE6.md#1}; tests rerun |
| Occupancy alone classifies local configurations | Fails at occupancy two | Six 2-subsets form three V4 orbits | @{projects/lisi-smg-test/REPORT_PHASE6D.md#70}; tests rerun |
| Orbit quotient retains Boolean lattice | Fails | Local quotient contains M3; global quotient reportedly not a lattice at all | @{projects/lisi-smg-test/REPORT_PHASE6D.md#136}; local rerun; global report-only |
| 600-cell / dual demicube / 4_21 survive density sieve | Repository evaluation claim | Conditional on mismatch and spinor profile; raw sweep provenance not established here | @{rcf/modules/geometry/discrete/README.md#14}; not reproduced |
| Earlier symmetry chain is active dependency | Rejected by explicit current status | Historical exploratory chain only; two geometry candidates remain open | @{rcmt/local/constraints/symmetry_chain.md#94} |

The last density claim deserves separate scrutiny: the repository's fixed-coefficient affine friction formula and tabulated values do not directly agree. Its list of eliminated candidates names failure categories but does not provide a complete candidate-by-candidate reproducible elimination record in the reviewed files. No such per-candidate result is invented here.
'''),
'06-open-questions.md': ('Prioritized open questions and human review', r'''
Priorities reflect load-bearing mathematical or architectural dependencies, not proposed extensions. This report identifies questions; no new theoretical work has begun.

## Priority 1: research claims depend on these

1. **What explicit RCMT projection realizes the two-lane requirements?** The contract requires Lorentz-compatible geometry but says the mathematical realization remains to be specified. Specify the Sp real form, carrier, alternating form, map domain/codomain, preserved relations, and local Lorentz interface before comparing SO(8)/SO(9) candidates. Finite dimension or group-name resemblance cannot answer this. @{rcmt/local/constraints/symmetry_chain.md#71}, @{rcmt/local/constraints/mode_space.md#36}
2. **Where is the operational ISP/LCO connection to the frozen algebra?** The existing exact audits give constraints and admissible generators, not a transition kernel, local candidate score implementation, or observable bridge. SIM16.2 labels observability unresolved; SIM16.4's transport direction does not select a rate or physical trajectory. Any proposed dynamics must address the stated stochastic commutation/isometry obstructions without silently demanding all symmetries at once. @{projects/Operational_Phase_Space/SIM16.2.py#224}, @{projects/Operational_Phase_Space/SIM16.4.py#449}
3. **What fixes M2 and the parameter t, if physical significance is intended?** M2 is singled out only after regularity plus frozen constraints are requested; that is not an independently established physical selection principle. Compatible signed metrics still form a real family. Identifiability of t does not select it or establish an observable. @{projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md#1}, @{projects/Operational_Phase_Space/SIM16.1.py#251}, @{projects/Operational_Phase_Space/SIM16.2.py#172}
4. **Can SMG supply executable structural interfaces?** Its bindings reference a missing `gcm_interface/mode_structure.md`; survivor indexing exists elsewhere, but that does not silently repair the declared dependency. Concrete adjacency/sector payloads and intended action are needed for a stronger comparison. Phase 6's local singleton obstruction must remain visible; semantic `8+3+1` is not recovered as an invariant partition by the present transported action. @{rcmt/smg_v2.1/bindings/gcm_bindings.md#36}, @{projects/lisi-smg-test/REPORT_PHASE6.md#14}
5. **What supports the density-survivor evidence numerically?** Locate the data, seeds, update rules, normalization of the angular integral, drag coefficients, parameter mapping, and run provenance for the named sweep. Explain the nonlinear table increments despite the fixed-coefficient affine formula. Also identify the exact closure product for each candidate rather than inferring closure from a polytope name. @{rcf/modules/geometry/discrete/mismatch-function.md#24}, @{rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md#35}, @{rcf/modules/geometry/discrete/spinor-closure.md#24}

## Priority 2: resolve document and implementation boundaries

| Issue | Competing evidence / consequence | Review needed |
|---|---|---|
| Physics interface scope | README allows declared local/upstream records; pipeline contract allows effective projection outputs only | Identify distinct consumers/stages or explicit supersession |
| Compactness | Local invariant 10 exempts symplectic lane from automatic compactness; global constraints demand compact bounded representations | State whether global rule applies to that lane and which representations qualify |
| Gamma identity | LCO memory weight [0,1], density table to 5 | Define distinct symbols or explicit conversion record |
| RCP migration | Canonical RCP-1.2 header differs from many RCMT/SMG headers; sentinel clause conflicts with no-delimiter placement | Clarify migration scope and validator requirement |
| Missing status document | Requested SMG `REFACTOR_STATUS.md` absent | Supply status/supersession evidence if it exists elsewhere; audit did not search outside allowed inputs |
| Introductory overstatement | RCMT README advertises explanations; local contracts classify representations/provisional candidates | Human review of wording and authority |
| Archival script packaging | All 21 OPS files invalid as stored | Decide whether a future source-maintenance task should separate code and recorded output; current inputs untouched |
| Lisi coordinate integrity | 597 distinct coordinate keys for 600 IDs, duplicate pairs and graph alignment defect | Obtain source provenance before any repair; graph-based results remain separately testable |
| Lisi plan drift | Plan's empty-src/tests and unfinished-phase state contradict present contents | Treat plan inventory as historical; no automatic authority from recency |

Sources: @{rcmt/physics_interface/README.md#1}, @{rcmt/physics_interface/pipeline/physics_interface_contract.md#1}, @{rcmt/global/constraints.md#62}, @{rcmt/local/invariants.md#93}, @{rcf/rcp-1.2.md#53}, @{rcf/rcp-1.2.md#236}, @{projects/lisi-smg-test/PHASE_1_2_REPORT.md#43}, @{projects/lisi-smg-test/PLAN.md#5}.

## Priority 3: limits of existing mathematical comparisons

- SIM15.1 does not recover the frozen signed Omega or literal C8 from its native vertex torsors. A map between OPS primitives and native H4 carriers is still missing despite abstract extension survival. @{projects/Operational_Phase_Space/SIM15.1.py#1}
- SIM15.5's full 25-family equality and triality identification deserve a dedicated reproducibility pass if a later task depends on them. This onboarding reran the one-cell refinement, not that complete census. @{projects/Operational_Phase_Space/SIM15.5.py#1}
- Phase 6 compares a frozen SMG concatenation; scope conclusions to that reference until correspondence with the current split module is established. @{projects/lisi-smg-test/REPORT_PHASE6.md#3}
- The order-96 action's intrinsic partitions need not remain intrinsic under the larger order-576 fiber-preserving action. The retained action is part of the specification, not decoration. @{projects/lisi-smg-test/REPORT_PHASE5.md#1}, @{projects/lisi-smg-test/REPORT_PHASE6B.md#3}
- Global configuration enumeration has not been rerun here; stored 177,976-orbit and non-lattice claims remain attributed to Phase 6C. No selected active state or operator follows from choosing a power-set carrier. @{projects/lisi-smg-test/REPORT_PHASE6C.md#1}
- Global twist/winding/holonomy preservation requires a precise class of allowed local moves and transport data. Adjacency plus the word “holonomy” alone is not an executable invariant implementation. @{rcmt/global/constraints.md#25}

Potential circularity to guard against: filtering candidates for H4 compatibility and then treating survival as evidence that H4 was unconditionally derived; using SMG's fixed semantic counts as targets and later calling numerical agreement a derivation; or choosing M2/metric sparsity to obtain a desired representation and attributing the resulting structure to unspecified dynamics. These are risks exposed by declared assumptions, not findings that every existing calculation is circular.
'''),
'07-agent-readiness.md': ('Agent readiness and audit limits', r'''
## Readiness

Ready for scoped follow-up research that uses the existing contracts and finite constructions. I can explain the RCF→RCMT ownership boundary, locate the main geometry and physics interfaces, distinguish required two-lane roles from provisional groups, and trace the OPS carrier/action/metric investigations and the independent Lisi transport pipeline. This is onboarding completion, not validation of RCMT or a full mathematical proof audit.

The strongest supported material is finite mathematics: the orbit-bridge theorem with reviewed proof, exhaustive matching classification, affine and signed stabilizer calculations, local H4 kernel/facet incidence, exact compatible-metric and projection audits, and graph-based Lisi embedding/dual transport plus symmetry obstructions to a canonical `3+1`. The local tests support these within their supplied definitions. Physical gauge labels, mass/zeta relations, and dynamics remain downstream hypotheses or interpretations.

## What was verified

- Recorded RCF and RCMT HEAD SHAs and initial working-tree status; hashed 569 input files before reruns; compared source bytes again at completion.
- Parsed all 21 raw OPS files and recorded their failures; produced traceable executable copies with only archival text extraction/docstring repair.
- Reran nine extracted OPS scripts: 14.3, 14.5, 14.8, 15.6, 16.0–16.4, all exit zero. Their assertions, counterexamples, machine packets, stdout, and stderr are retained.
- Reran eight Lisi test modules: **42 tests passed**, covering geometry/dualization, inscribed cells, projective embedding, fiber transport, intrinsic geometry, SMG comparison, refinements, and local four-carrier configurations.
- Reviewed the orbit-bridge argument: product decomposition, equivariance on orbit set, transitivity, and orbit-stabilizer regularity. Reviewed the fixed-line complex-structure and stochastic-isometry obstructions at the level of their stated hypotheses.

Evidence: @{projects/ops-codex/research/source-manifest.json#1}, @{projects/ops-codex/research/source-integrity-check.json#1}, @{projects/ops-codex/research/ops-syntax-audit.json#1}, @{projects/ops-codex/research/extraction-manifest.json#1}, @{projects/ops-codex/research/reproduction-results.json#1}, @{projects/ops-codex/research/lisi-geometry-tests.txt#1}, @{projects/ops-codex/research/lisi-selected-tests.txt#1}.

## Reading coverage

| Level | Material |
|---|---|
| Read directly | RCF README, RCP-1.2, kernel; RCD overview; LCO overview/interface; time and temporal interface; discrete geometry overview, sieve, candidate list, closure, mismatch and density table; projection overview; GCM overview/indexing; RCMT README/local authority/invariants, symmetry/mode-space/LCO/direction constraints, global overview/constraints; physics overview/pipeline contract/time bridge; SMG overview/contract/bindings/central claim/gauge-spinor-zeta translations/policy; matter overview |
| Targeted reading | OPS vocabulary and learner guide sections; orbit-bridge theorem in full; SIM14/SIM15 discussions by headings and relevant claims; post-theorem speculative discussion; every SIM14–16 script's objectives, frozen inputs, relevant calculations, gates and archived conclusions; executed selected bodies |
| Lisi coverage | README_FIRST, plan, summary; normalization, predicate/24-cell, enriched embedding, fiber-transport, intrinsic analysis and Phases 6/6B/6C/6D reports; selected implementation and test functions; saved artifacts consumed by tests |
| Inventoried, not comprehensively read | Entire RCF formal library and geometry submodules; all RCMT local physics modules, symmetry patterns, cosmology/metric/matter detail; full contents of every large archived discussion and script; every large JSON artifact |

The requested SMG refactor-status file was absent. The original Lisi PDF was not independently read; successful exact finite reconstruction tests do not establish textual fidelity to every paper convention or validate the paper's physical interpretation. Full linear/spinorial phases were outside the carrier-level study's stated scope. Historical SIM1–13 scripts were not supplied in the inspected OPS folder; statements about that lineage are source commentary, not reproduced history.

## What remains unverified

The full H4 25-family census in SIM15.0–15.5 was not rerun; SIM15.6's smaller local reconstruction was. Most SIM14 archived scripts were read but not executed individually; downstream regressions check some of their outputs. The global 2^24 configuration enumeration and 2.1-million-cover orbit-poset construction were not rerun. Reported density sweep and per-candidate elimination histories lack an established reproduction chain in this audit. No empirically measured physics result, Standard Model representation derivation, Lorentz projection, operational observation model, or implemented ISP/LCO dynamics was verified.

Reusing authors' tests and implementations is reproducibility evidence, not independent algorithmic replication. Source integrity comparison establishes byte preservation from the recorded baseline, not absence of pre-existing source defects or changes before that baseline.

## Human review required before stronger claims

Resolve the missing GCM dependency and SMG status record; conflicting interface permissions and compactness scopes; gamma mapping and mismatch-table inconsistency; mixed protocol headers and RCP sentinel wording; coordinate corruption/alignment; and archival script packaging. Detailed source citations and priorities are in `06-open-questions.md`. These issues should remain visible rather than being repaired through unrecorded interpretive choices.

No other agent's workspace was inspected or used. No research source was intentionally changed, no dependencies installed, and no new theoretical investigation begun. Stop point: onboarding complete with the verification limits above.
'''),
})

def render_refs(body):
    def sub(m):
        p,line=m.group(1),m.group(2)
        return f'[{p}:{line}]({(ROOT/p).as_posix()}:{line})'
    return re.sub(r'@\{([^#}]+)#(\d+)\}',sub,body)

def write_all():
    for filename,(title,body) in reports.items():
        header=f'''# {title}

## Document Contract

- owner: Codex independent onboarding audit
- type: Observation
- scope: inspected theory workspace, 2026-10-24
- authority: audit notes only; no authority over research sources
- upstream dependencies: cited source documents and recorded execution evidence
- status: completed onboarding assessment with explicit verification limits
- purpose: reconstruct existing research and classify its evidence
- anti-backflow: this report does not revise upstream definitions or promote implementations and interpretations to invariants

'''
        (OUT/filename).write_text(header+render_refs(body.strip())+'\n',encoding='utf-8')

if __name__=='__main__':
    write_all()
