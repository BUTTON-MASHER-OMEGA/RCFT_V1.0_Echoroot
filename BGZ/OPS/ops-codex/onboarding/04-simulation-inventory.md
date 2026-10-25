# Simulation and verification inventory

## Document Contract

- owner: Codex independent onboarding audit
- type: Observation
- scope: inspected theory workspace, 2026-10-24
- authority: audit notes only; no authority over research sources
- upstream dependencies: cited source documents and recorded execution evidence
- status: completed onboarding assessment with explicit verification limits
- purpose: reconstruct existing research and classify its evidence
- anti-backflow: this report does not revise upstream definitions or promote implementations and interpretations to invariants

## Reproducibility conditions

All 21 supplied OPS files fail AST parsing as stored. They mix executable code with pasted results; five also lack the opening triple quote for their introductory prose. This is an archival packaging failure, not a falsification of the mathematics. No source file was repaired. [projects/ops-codex/research/ops-syntax-audit.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/ops-syntax-audit.json:1)

The audit created executable copies by truncating at the first results delimiter and, where needed, prepending one opening docstring delimiter. SIM14.5 has a separately recorded cut at source line 208. No equations, group definitions, thresholds, or assertions were edited. Exact cut ranges are in [projects/ops-codex/research/extraction-manifest.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/extraction-manifest.json:1); source hashes are in the baseline manifest. Twenty-one extracted copies parse successfully. Nine selected copies ran with a 60-second per-process limit, all exiting zero. See [projects/ops-codex/research/reproduce.py:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/reproduce.py:1) and [projects/ops-codex/research/reproduction-results.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/reproduction-results.json:1).

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

Sources: [projects/Operational_Phase_Space/SIM14.0.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.0.py:1), [projects/Operational_Phase_Space/SIM14.1.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.1.py:1), [projects/Operational_Phase_Space/SIM14.2.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.2.py:1), [projects/Operational_Phase_Space/SIM14.3.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.3.py:1), [projects/Operational_Phase_Space/SIM14.4.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.4.py:1), [projects/Operational_Phase_Space/SIM14.5.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.5.py:1), [projects/Operational_Phase_Space/SIM14.6.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.6.py:1), [projects/Operational_Phase_Space/SIM14.7.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.7.py:1), [projects/Operational_Phase_Space/SIM14.8.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM14.8.py:1).

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

Sources: [projects/Operational_Phase_Space/SIM15.0.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.0.py:1), [projects/Operational_Phase_Space/SIM15.1.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.1.py:1), [projects/Operational_Phase_Space/SIM15.2.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.2.py:1), [projects/Operational_Phase_Space/SIM15.3.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.3.py:1), [projects/Operational_Phase_Space/SIM15.4.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.4.py:1), [projects/Operational_Phase_Space/SIM15.5.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.5.py:1), [projects/Operational_Phase_Space/SIM15.6.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.6.py:1).

The supplied SIM15 discussion principally interprets earlier stages; later script truth packets carry later gate outcomes. False gates caused by failed prerequisites must be distinguished from independently tested falsifications. Tolerance-based H4 coordinate recognition also limits claims of purely exact reconstruction.

## SIM16: exact algebra on the frozen representation

| Script | Constraint or question | Result reproduced | Audit |
|---|---|---|---|
| SIM16.0 | Does signed permutation symmetry uniquely fix compatible J/g? | No: compatible metric dimensions 4/2 for N+/N±; nontrivial witnesses survive | C |
| SIM16.1 | Add strong C8 sparsity to metric entries | Dimensions 2/1; signed family exactly cosh(t)I+sinh(t)K; only 2/4 group elements preserve the full graph | C |
| SIM16.2 | Parameter identification, equivalence and stochastic obstructions | t identifiable algebraically; selection/observability unresolved; conjugacy not proven gauge; primitive mixing commutation/isometry no-go | C |
| SIM16.3 | Projection eigensectors, descent and controls | Compatible complex quotient; direct two-form descent fails; sectors intrinsically equivalent but frozen observation/adjacency distinguishes them | C |
| SIM16.4 | Graph-local symplectic generators, fixed-J and family transport | Local dimensions 24 unrestricted/12 symplectic; sp(2,R)^4 closure; fixed-J dimension four; all-J common space zero; family transport one-dimensional | C |

Sources: [projects/Operational_Phase_Space/SIM16.0.py:156](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.0.py:156), [projects/Operational_Phase_Space/SIM16.1.py:186](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.1.py:186), [projects/Operational_Phase_Space/SIM16.2.py:108](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.2.py:108), [projects/Operational_Phase_Space/SIM16.3.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.3.py:1), [projects/Operational_Phase_Space/SIM16.4.py:154](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.4.py:154). Each rerun has its own `research/SIM*.py.stdout.txt` and `.stderr.txt`; JSON truth packets are retained verbatim.

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

Eight selected test modules passed: 26 geometry/transport/intrinsic tests in 49.68 seconds and 16 comparison/refinement/local tests in 1.03 seconds, total **42 passed**. This is not the complete nine-module suite. Logs: [projects/ops-codex/research/lisi-geometry-tests.txt:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/lisi-geometry-tests.txt:1), [projects/ops-codex/research/lisi-selected-tests.txt:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/lisi-selected-tests.txt:1).

The comparison tests rely on existing artifacts and the frozen SMG concatenation. They do not prove every stored artifact was generated by the claimed algorithm, establish physical correctness, or compare all current RCMT modules. The separate geometry suites reconstruct substantial earlier geometry, but the full global configuration enumeration remains report-only.
