# Invariants, authority, and exclusions

## Document Contract

- owner: Codex independent onboarding audit
- type: Observation
- scope: inspected theory workspace, 2026-10-24
- authority: audit notes only; no authority over research sources
- upstream dependencies: cited source documents and recorded execution evidence
- status: completed onboarding assessment with explicit verification limits
- purpose: reconstruct existing research and classify its evidence
- anti-backflow: this report does not revise upstream definitions or promote implementations and interpretations to invariants

## Authority rules

The root RCF shard has explicit invariant authority and no upstream dependency. Its pre-theoretical commitments are project definitions, not experimentally established laws. Optional modules can be eliminated without changing the kernel. RCP-1.2 requires owner/type/scope/authority/dependencies/status/purpose/anti-backflow; a status such as active does not raise authority. Header scope bounds body prose. [rcf/core/S∞.RCF.Invariants.v1.1.md:3](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:3), [rcf/README.md:84](C:/Users/blaur/projects/theory/rcf/README.md:84), [rcf/rcp-1.2.md:72](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:72), [rcf/rcp-1.2.md:180](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:180)

## RCF invariant kernel

| Invariant | Operational reading and limit | Source line |
|---|---|---|
| Persistence under constraint | Project meaning is persistence; failures are eliminated | [rcf/core/S∞.RCF.Invariants.v1.1.md:35](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:35) |
| Emergence via elimination | No additive rescue of unstable relations | [rcf/core/S∞.RCF.Invariants.v1.1.md:48](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:48) |
| Relational primacy | Identity/meaning depends on relations | [rcf/core/S∞.RCF.Invariants.v1.1.md:58](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:58) |
| Spectral dominance | Persistent structure is declared spectral | [rcf/core/S∞.RCF.Invariants.v1.1.md:68](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:68) |
| Constraint over dynamics | Admissibility rather than fundamental equations of motion | [rcf/core/S∞.RCF.Invariants.v1.1.md:78](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:78) |
| Minimal free parameters | Structural ratios should survive parameter removal | [rcf/core/S∞.RCF.Invariants.v1.1.md:87](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:87) |
| Non-privileged observers | Observation constrains resolution; relabeling has no privileged observer | [rcf/core/S∞.RCF.Invariants.v1.1.md:96](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:96) |
| Self-applicability | Framework receives no exemption | [rcf/core/S∞.RCF.Invariants.v1.1.md:106](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:106) |
| Epsilon allowance | Residual alternatives remain; no literal single-mode collapse | [rcf/core/S∞.RCF.Invariants.v1.1.md:115](C:/Users/blaur/projects/theory/rcf/core/S∞.RCF.Invariants.v1.1.md:115) |

These are not yet quantitative specifications of what counts as increasing constraint, spectral persistence, or sufficient epsilon in every implementation. Those choices require scoped module definitions.

## RCMT-local admissibility

`local/invariants.md` calls itself highest within RCMT and subordinate to all upstream RCF contracts. It supplies 15 requirements. [rcmt/local/invariants.md:3](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:3)

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

Sources: [rcmt/local/invariants.md:20](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:20), [rcmt/local/invariants.md:46](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:46), [rcmt/local/invariants.md:79](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:79), [rcmt/local/invariants.md:93](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:93), [rcmt/local/invariants.md:105](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:105).

The old `E8 → SO(10) → Sp(8)×SO(9)` chain is expressly historical exploratory work. Neither compact SO candidate is claimed to contain Lorentz symmetry by conventional compact symmetry breaking. A projection-level realization remains unspecified. Do not infer an embedding from dimensions or names. [rcmt/local/constraints/symmetry_chain.md:49](C:/Users/blaur/projects/theory/rcmt/local/constraints/symmetry_chain.md:49), [rcmt/local/constraints/symmetry_chain.md:84](C:/Users/blaur/projects/theory/rcmt/local/constraints/symmetry_chain.md:84)

## Module-specific restrictions

LCO scores local candidates using bounded neighborhoods, memory, and sealed upstream parameters; it cannot select global geometry, infer physical meaning, or schedule. Its unit-norm comparison space is a representation. [rcf/modules/lco/interface.md:14](C:/Users/blaur/projects/theory/rcf/modules/lco/interface.md:14), [rcf/modules/lco/README.md:1](C:/Users/blaur/projects/theory/rcf/modules/lco/README.md:1)

The geometry sieve applies mismatch, mode completeness, a declared spinor-composition profile, recursion, density viability, and triangle compatibility. Its candidate file additionally requires H4 compatibility. These restrict the tested profile and candidate pool, so survival is conditional on that pool and filter; it is not independent discovery that H4 is uniquely physically necessary. [rcf/modules/geometry/discrete/survival-sieve.md:23](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/survival-sieve.md:23), [rcf/modules/geometry/discrete/candidate-geometries.md:24](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/candidate-geometries.md:24)

Global twist, winding, holonomy must use adjacency transport, remain distinguishable, and survive allowed local cleanup. The global layer cannot redefine local primitives. Its blanket compactness language is broader than local invariant 10's symplectic exception; applicability to the mode lane requires human clarification. [rcmt/global/constraints.md:25](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:25), [rcmt/global/constraints.md:62](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:62), [rcmt/global/constraints.md:90](C:/Users/blaur/projects/theory/rcmt/global/constraints.md:90), [rcmt/local/invariants.md:93](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:93)

SMG fixes 36 semantic modes and three generations for adopters; a 37th mode or fourth generation needs a model-contract revision. These are model policies. Gauge, spinor, mass, and zeta language remains downstream. [rcmt/smg_v2.1/claims/dual_closure.md:1](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/claims/dual_closure.md:1), [rcmt/smg_v2.1/policies/limitations_and_boundaries.md:1](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/policies/limitations_and_boundaries.md:1)

## Definition and contract discrepancies

1. RCP-1.2 says no leading delimiter/front matter, but its validation list still requires a “leading header sentinel.” These requirements need clarification before claiming formal header compliance. [rcf/rcp-1.2.md:53](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:53), [rcf/rcp-1.2.md:236](C:/Users/blaur/projects/theory/rcf/rcp-1.2.md:236)
2. Several RCMT local headers use `depends_on`, omit canonical fields, or lack a Document Contract. SMG files advertise RCP-1.1 compatibility. This is mixed protocol migration, not evidence that their scientific bodies are false. [rcmt/local/invariants.md:3](C:/Users/blaur/projects/theory/rcmt/local/invariants.md:3), [rcmt/local/constraints/symmetry_chain.md:3](C:/Users/blaur/projects/theory/rcmt/local/constraints/symmetry_chain.md:3), [rcmt/smg_v2.1/README.md:118](C:/Users/blaur/projects/theory/rcmt/smg_v2.1/README.md:118)
3. The mismatch document states `F(G,gamma)=E_static(G)+gamma R_drag(G)`. The density table's 600-cell values 0.472, 0.550, 0.750 at gamma 0,1,2 do not lie on one such line: successive increments are 0.078 and 0.200. A stochastic estimate, gamma-dependent coefficients, or a different model could explain it, but no such explanation was established here. Do not treat the table as verified output of the stated fixed-coefficient formula. [rcf/modules/geometry/discrete/mismatch-function.md:24](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/mismatch-function.md:24), [rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:35](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:35)
4. LCO's overview bounds a memory weight gamma to [0,1], whereas the geometry density table uses gamma up to 5. These may be different parameters linked through a sealed conversion, but that identification was not established in the reviewed material. [rcf/modules/lco/README.md:1](C:/Users/blaur/projects/theory/rcf/modules/lco/README.md:1), [rcf/modules/temporal-interface/interface.md:14](C:/Users/blaur/projects/theory/rcf/modules/temporal-interface/interface.md:14), [rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:35](C:/Users/blaur/projects/theory/rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md:35)
