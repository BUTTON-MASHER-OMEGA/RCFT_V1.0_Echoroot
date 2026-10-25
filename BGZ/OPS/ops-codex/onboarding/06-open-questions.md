# Prioritized open questions and human review

## Document Contract

- owner: Codex independent onboarding audit
- type: Observation
- scope: inspected theory workspace, 2026-10-24
- authority: audit notes only; no authority over research sources
- upstream dependencies: cited source documents and recorded execution evidence
- status: completed onboarding assessment with explicit verification limits
- purpose: reconstruct existing research and classify its evidence
- anti-backflow: this report does not revise upstream definitions or promote implementations and interpretations to invariants

Priorities reflect load-bearing mathematical or architectural dependencies, not proposed extensions. This report identifies questions; no new theoretical work has begun.

## Priority 1: research claims depend on these

1. **What explicit RCMT projection realizes the two-lane requirements?** The contract requires Lorentz-compatible geometry but says the mathematical realization remains to be specified. Specify the Sp real form, carrier, alternating form, map domain/codomain, preserved relations, and local Lorentz interface before comparing SO(8)/SO(9) candidates. Finite dimension or group-name resemblance cannot answer this. [rcmt/local/constraints/symmetry_chain.md:71](C:/Users/blaur/projects/theory/rcmt/local/constraints/symmetry_chain.md:71), [rcmt/local/constraints/mode_space.md:36](C:/Users/blaur/projects/theory/rcmt/local/constraints/mode_space.md:36)
2. **Where is the operational ISP/LCO connection to the frozen algebra?** The existing exact audits give constraints and admissible generators, not a transition kernel, local candidate score implementation, or observable bridge. SIM16.2 labels observability unresolved; SIM16.4's transport direction does not select a rate or physical trajectory. Any proposed dynamics must address the stated stochastic commutation/isometry obstructions without silently demanding all symmetries at once. [projects/Operational_Phase_Space/SIM16.2.py:224](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.2.py:224), [projects/Operational_Phase_Space/SIM16.4.py:449](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.4.py:449)
3. **What fixes M2 and the parameter t, if physical significance is intended?** M2 is singled out only after regularity plus frozen constraints are requested; that is not an independently established physical selection principle. Compatible signed metrics still form a real family. Identifiability of t does not select it or establish an observable. [projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md:1), [projects/Operational_Phase_Space/SIM16.1.py:251](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.1.py:251), [projects/Operational_Phase_Space/SIM16.2.py:172](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM16.2.py:172)
4. **Can SMG supply executable structural interfaces?** Its bindings reference a missing `gcm_interface/mode_structure.md`; survivor indexing exists elsewhere, but that does not silently repair the declared dependency. Concrete adjacency/sector payloads and intended action are needed for a stronger comparison. Phase 6's local singleton obstruction must remain visible; semantic `8+3+1` is not recovered as an invariant partition by the present transported action. [rcmt/smg_v2.1/bindings/gcm_bindings.md:36](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/bindings/gcm_bindings.md:36), [projects/lisi-smg-test/REPORT_PHASE6.md:14](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6.md:14)
5. **What supports the density-survivor evidence numerically?** Locate the data, seeds, update rules, normalization of the angular integral, drag coefficients, parameter mapping, and run provenance for the named sweep. Explain the nonlinear table increments despite the fixed-coefficient affine formula. Also identify the exact closure product for each candidate rather than inferring closure from a polytope name. [rcf/modules/geometry/discrete/mismatch-function.md:24](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/mismatch-function.md:24), [rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:35](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:35), [rcf/modules/geometry/discrete/spinor-closure.md:24](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/spinor-closure.md:24)

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

Sources: [rcmt/physics_interface/README.md:1](C:/Users/blaur/projects/theory/rcmt/physics_interface/README.md:1), [rcmt/physics_interface/pipeline/physics_interface_contract.md:1](C:/Users/blaur/projects/theory/rcmt/physics_interface/pipeline/physics_interface_contract.md:1), [rcmt/global/constraints.md:62](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:62), [rcmt/local/invariants.md:93](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:93), [rcf/rcp-1.2.md:53](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:53), [rcf/rcp-1.2.md:236](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:236), [projects/lisi-smg-test/PHASE_1_2_REPORT.md:43](C:/Users/blaur/projects/theory/projects/lisi-smg-test/PHASE_1_2_REPORT.md:43), [projects/lisi-smg-test/PLAN.md:5](C:/Users/blaur/projects/theory/projects/lisi-smg-test/PLAN.md:5).

## Priority 3: limits of existing mathematical comparisons

- SIM15.1 does not recover the frozen signed Omega or literal C8 from its native vertex torsors. A map between OPS primitives and native H4 carriers is still missing despite abstract extension survival. [projects/Operational_Phase_Space/SIM15.1.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.1.py:1)
- SIM15.5's full 25-family equality and triality identification deserve a dedicated reproducibility pass if a later task depends on them. This onboarding reran the one-cell refinement, not that complete census. [projects/Operational_Phase_Space/SIM15.5.py:1](C:/Users/blaur/projects/theory/projects/Operational_Phase_Space/SIM15.5.py:1)
- Phase 6 compares a frozen SMG concatenation; scope conclusions to that reference until correspondence with the current split module is established. [projects/lisi-smg-test/REPORT_PHASE6.md:3](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6.md:3)
- The order-96 action's intrinsic partitions need not remain intrinsic under the larger order-576 fiber-preserving action. The retained action is part of the specification, not decoration. [projects/lisi-smg-test/REPORT_PHASE5.md:1](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE5.md:1), [projects/lisi-smg-test/REPORT_PHASE6B.md:3](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6B.md:3)
- Global configuration enumeration has not been rerun here; stored 177,976-orbit and non-lattice claims remain attributed to Phase 6C. No selected active state or operator follows from choosing a power-set carrier. [projects/lisi-smg-test/REPORT_PHASE6C.md:1](C:/Users/blaur/projects/theory/projects/lisi-smg-test/REPORT_PHASE6C.md:1)
- Global twist/winding/holonomy preservation requires a precise class of allowed local moves and transport data. Adjacency plus the word “holonomy” alone is not an executable invariant implementation. [rcmt/global/constraints.md:25](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:25)

Potential circularity to guard against: filtering candidates for H4 compatibility and then treating survival as evidence that H4 was unconditionally derived; using SMG's fixed semantic counts as targets and later calling numerical agreement a derivation; or choosing M2/metric sparsity to obtain a desired representation and attributing the resulting structure to unspecified dynamics. These are risks exposed by declared assumptions, not findings that every existing calculation is circular.
