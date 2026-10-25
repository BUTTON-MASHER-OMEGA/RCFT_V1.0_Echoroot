# Repository map and provenance

## Document Contract

- owner: Codex independent onboarding audit
- type: Observation
- scope: inspected theory workspace, 2026-10-24
- authority: audit notes only; no authority over research sources
- upstream dependencies: cited source documents and recorded execution evidence
- status: completed onboarding assessment with explicit verification limits
- purpose: reconstruct existing research and classify its evidence
- anti-backflow: this report does not revise upstream definitions or promote implementations and interpretations to invariants

Audit date: 2026-10-24 (America/Phoenix). Agent: Codex. Scope: repository rehydration and research audit, ending before new theoretical investigations.

## Source identity

| Input | Git identity | Inspection boundary |
|---|---|---|
| `rcf/` | `c419353214ee83d83376747064912544aee257b9` | Read-only working tree; pre-existing tracked Python cache modifications were present |
| `rcmt/` | `b7eecd0dfd8080864b34441af86329f819cba65a` | Read-only working tree; initial Git status clean |
| `projects/Operational_Phase_Space/` | Not a Git repository at this location | Copied, unversioned research input; file hashes recorded |
| `projects/lisi-smg-test/` | Not a Git repository at this location | Copied, unversioned research input; file hashes recorded |

Git ownership checks required command-local `-c safe.directory=<exact repository>` for read-only queries. No global Git configuration changed. A commit SHA does not identify pre-existing working-tree differences: the 569-file SHA-256 baseline in [projects/ops-codex/research/source-manifest.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/source-manifest.json:1) identifies the inspected non-Git-metadata inputs more precisely. The baseline was taken before computational reruns, after initial reconnaissance. The closing comparison is [projects/ops-codex/research/source-integrity-check.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/source-integrity-check.json:1).

The user's pasted onboarding instructions were inspected in the chat. The external desktop file and attachment were not opened. No content from `projects/ops-antigravity/` was inspected. Other top-level research trees were not used as evidence. No external literature was fetched. The Lisi PDF's original text was not independently reviewed; paper attribution below is attributed to the local reconstruction report, with finite reconstruction tested separately.

## Navigation map

| Area | Role and useful entrypoints |
|---|---|
| RCF identity and authority | [rcf/README.md:3](C:/Users/blaur/projects/theory/rcf/README.md:3); [rcf/rcp-1.2.md:28](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:28); [rcf/core/S∞.RCF.Invariants.v1.1.md:3](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:3) |
| RCF formal expression | `core/rcd/` describes substrate-independent relational admissibility, explicitly non-empirical; `core/calculus/`, `core/algebra/`, `core/geometry/` provide mathematical vocabulary. [rcf/core/rcd/README.md:3](C:/Users/blaur/projects/theory/rcf/core/rcd/README.md:3) |
| Time and execution | `modules/time/`, `modules/temporal-interface/`, `modules/lco/`, `modules/gco/`: ordering, sealed parameters, local scoring, aggregation. [rcf/modules/lco/interface.md:3](C:/Users/blaur/projects/theory/rcf/modules/lco/interface.md:3) |
| Geometry | `modules/geometry/discrete/`, `projection/`, `continuous/`, `dynamics/`, `adjacent/`: candidate records, handoff, shadows, subsequent descriptors. [rcf/modules/geometry/discrete/README.md:14](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/README.md:14) |
| Representation boundary | `modules/gcm_interface/`: indexing and sector descriptors, conditional on evaluated survivors. [rcf/modules/gcm_interface/README.md:3](C:/Users/blaur/projects/theory/rcf/modules/gcm_interface/README.md:3) |
| RCMT local architecture | `local/invariants.md`, then `local/constraints/` and `local/symmetry_groups/`; load order and provisional roles in [rcmt/local/README.md:13](C:/Users/blaur/projects/theory/rcmt/local/README.md:13). |
| RCMT global layer | `global/`: twist, winding, holonomy subject to adjacency transport and preservation requirements. [rcmt/global/constraints.md:5](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:5) |
| RCMT physics interfaces | `physics_interface/local/`: group structures, relational quantum mechanics, kinematics, dynamics, discrete field theory, time bridge, continuous field theory, emergent metric geometry, conversion laws; `pipeline/` is another contract organization requiring reconciliation. |
| SMG | `smg_v2.1/`: read-only bindings → local claims/dictionaries/translations → policies/exports. [rcmt/smg_v2.1/README.md:54](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/README.md:54) |
| Downstream matter and geometry behavior | `modules/survivor_matter_structure/`, `modules/geometry_behavior/`, `modules/metric/`; candidate particle labels and derived geometry descriptions. [rcmt/modules/survivor_matter_structure/README.md:1](C:/Users/blaur/projects/theory/rcmt/modules/survivor_matter_structure/README.md:1) |
| OPS | 21 scripts: SIM14.0–14.8, SIM15.0–15.6, SIM16.0–16.4; vocabulary, learner guide, theorem, two discussion records, post-theorem exploration. All are static carrier/algebra investigations in this supplied series. |
| Lisi study | `README_FIRST.md`, `PLAN.md`, phase reports, `src/`, nine test modules, `artifacts/`, `reference_geometry/`, `references/`. Geometry normalization → inscribed cells → enriched Lisi embedding → dual transport → intrinsic analysis → SMG comparison → refinement/configurations. |
| Codex outputs | Eight reports here; `../research/` contains audit scripts, extracted executable copies, manifests, test logs, and rerun stdout/stderr. |

## Discovery discrepancies

Actual directory case is `rcf/`, `rcmt/`, `projects/`; Windows resolves the supplied uppercase names. `rcmt/smg_v2.1/REFACTOR_STATUS.md` is absent. RCMT's top-level README tree points to files such as root `invariants.md`, while the operative local files are under `local/`; consult contracts rather than that tree. [rcmt/README.md:50](C:/Users/blaur/projects/theory/rcmt/README.md:50)

The Lisi `PLAN.md` retains a planning-time claim that `src/` and `tests/` are empty and comparison phases have not begun, despite the present reports and executable files. It is historical state, not a current inventory. [projects/lisi-smg-test/PLAN.md:5](C:/Users/blaur/projects/theory/projects/lisi-smg-test/PLAN.md:5)
